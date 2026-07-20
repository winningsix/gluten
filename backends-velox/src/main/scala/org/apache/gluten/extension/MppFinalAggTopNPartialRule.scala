/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.gluten.extension

import org.apache.gluten.execution.{HashAggregateExecTransformer, ProjectExecTransformer, TakeOrderedAndProjectExecTransformer, TopNTransformer}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.SortOrder
import org.apache.spark.sql.catalyst.expressions.aggregate.{Complete, Final}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanExec

/**
 * Mimic Presto's FinalAgg + TopNPartial stage for global ORDER BY + LIMIT queries.
 *
 * Presto keeps only the top-K rows per partition immediately after the final aggregate and before
 * the SINGLE gather that feeds the output TopN. Gluten currently ships every final-agg row across
 * the gather and only applies Sort + Limit in the single-driver TakeOrdered fragment (for Q10 SF1K
 * this is ~44M rows vs 20).
 *
 * When TakeOrderedAndProject sits at the query root, this rule walks the producer subtree and
 * inserts a local [[TopNTransformer]] (global=false) on the spine above the first final-mode hash
 * aggregate, or above the projection that names the final aggregate outputs. The downstream
 * TakeOrdered fragment still performs the final merge/limit on at most `limit * numPartitions` rows
 * instead of the full grouped cardinality.
 *
 * Gated by spark.gluten.mpp.finalAggTopNPartial (default: true).
 */
case class MppFinalAggTopNPartialRule() extends Rule[SparkPlan] with Logging {
  private val confKey = "spark.gluten.mpp.finalAggTopNPartial"

  override def apply(plan: SparkPlan): SparkPlan = {
    val enabled = SparkSession.getActiveSession.forall(_.conf.get(confKey, "true").toBoolean)
    if (!enabled) {
      return plan
    }
    rewriteAll(plan)
  }

  /**
   * Rewrite every query block, not just the outer result spine. A TakeOrderedAndProject may be the
   * broadcast side of a later dimension join (TPC-H Q10's fact Top20, for example). Stopping at the
   * first multi-child node leaves that nested TopN behind a SINGLE gather of the full final
   * aggregate output.
   */
  private def rewriteAll(node: SparkPlan): SparkPlan = node match {
    case aqe: AdaptiveSparkPlanExec => aqe
    case topk: TakeOrderedAndProjectExecTransformer if topk.offset == 0 =>
      val childWithNestedTopN = rewriteAll(topk.child)
      val rewrittenChild =
        insertPartialTopNOnSpine(childWithNestedTopN, topk.limit, topk.sortOrder)
      if (rewrittenChild.fastEquals(topk.child)) {
        topk
      } else {
        logInfo(
          s"MppFinalAggTopNPartialRule: inserted partial TopN(limit=${topk.limit}) " +
            s"after final aggregate before SINGLE gather")
        topk.copy(child = rewrittenChild)
      }
    case other if other.children.nonEmpty =>
      val rewrittenChildren = other.children.map(rewriteAll)
      if (
        rewrittenChildren.zip(other.children).forall {
          case (rewritten, original) =>
            rewritten.eq(original)
        }
      ) {
        // Preserve object identity, SparkPlan tags and exchange-reuse bookkeeping for query
        // blocks that contain no matching TopN. Rebuilding an otherwise unchanged join tree can
        // alter later physical exchange recognition even though its printed shape is identical.
        other
      } else {
        other.withNewChildren(rewrittenChildren)
      }
    case other => other
  }

  /** Walk the single-child spine under the TakeOrdered producer and splice partial TopN once. */
  private def insertPartialTopNOnSpine(
      producer: SparkPlan,
      limit: Long,
      sortOrder: Seq[SortOrder]): SparkPlan = {
    producer match {
      case project: ProjectExecTransformer if isFinalHashAggregatePlan(project.child) =>
        buildPartialTopN(limit, sortOrder, project).getOrElse(producer)
      case agg: HashAggregateExecTransformer if isFinalHashAggregate(agg) =>
        buildPartialTopN(limit, sortOrder, agg).getOrElse(producer)
      case other if other.children.size == 1 && isPartialTopNTransparentSpine(other) =>
        val rewritten = insertPartialTopNOnSpine(other.children.head, limit, sortOrder)
        if (rewritten.fastEquals(other.children.head)) {
          other
        } else {
          other.withNewChildren(Seq(rewritten))
        }
      case other => other
    }
  }

  private def isPartialTopNTransparentSpine(plan: SparkPlan): Boolean = {
    val name = plan.getClass.getSimpleName
    name.contains("ColumnarInputAdapter") ||
    name.contains("ColumnarToColumnar") ||
    name.contains("ColumnarToRow") ||
    name.contains("InputIteratorTransformer") ||
    name.contains("RowToColumnar") ||
    name.contains("RowToVeloxColumnar") ||
    name.contains("WholeStageTransformer")
  }

  private def buildPartialTopN(
      limit: Long,
      sortOrder: Seq[SortOrder],
      child: SparkPlan): Option[SparkPlan] = {
    if (limit <= 0) {
      return None
    }
    if (!sortOrder.forall(_.deterministic)) {
      return None
    }
    val missingReferences = sortOrder.flatMap(_.references).filterNot(child.outputSet.contains)
    if (missingReferences.nonEmpty) {
      logWarning(
        s"MppFinalAggTopNPartialRule: sort references ${missingReferences.mkString(",")} " +
          s"are not produced by ${child.nodeName}; leaving plan unchanged")
      return None
    }
    val topN = TopNTransformer(limit, sortOrder, global = false, child, isPartial = true)
    val validation = topN.doValidate()
    if (validation.ok()) {
      Some(topN)
    } else {
      logWarning(
        s"MppFinalAggTopNPartialRule: TopN validation failed ($validation); leaving plan unchanged")
      None
    }
  }

  private def isFinalHashAggregate(agg: HashAggregateExecTransformer): Boolean = {
    agg.aggregateExpressions.nonEmpty && agg.aggregateExpressions.forall {
      ae => ae.mode == Final || ae.mode == Complete
    }
  }

  private def isFinalHashAggregatePlan(plan: SparkPlan): Boolean = plan match {
    case agg: HashAggregateExecTransformer => isFinalHashAggregate(agg)
    case _ => false
  }
}
