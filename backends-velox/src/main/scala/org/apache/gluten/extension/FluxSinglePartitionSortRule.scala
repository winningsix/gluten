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

import org.apache.gluten.execution.SortExecTransformer

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.plans.physical.{RangePartitioning, SinglePartition}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ProjectExec, SortExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.exchange.{ShuffleExchangeExec, ShuffleExchangeLike}

/**
 * Rewrite `Exchange(RangePartitioning) -> Sort(global)` at the plan root into
 * `Exchange(SinglePartition) -> Sort` to match Presto's SINGLE-gather plan shape for ORDER BY
 * queries.
 *
 * Only the top-of-plan global Sort is rewritten. Per-key sorts inside SortMergeJoin use
 * ClusteredDistribution (HashPartitioning), not OrderedDistribution, so this rule never touches
 * them.
 *
 * Gated by spark.gluten.mpp.singlePartitionSort (default: false).
 */
case class FluxSinglePartitionSortRule() extends Rule[SparkPlan] with Logging {
  private val confKey = "spark.gluten.mpp.singlePartitionSort"

  override def apply(plan: SparkPlan): SparkPlan = {
    val sess = org.apache.spark.sql.SparkSession.getActiveSession
    val enabled = sess.exists(_.conf.get(confKey, "false").toBoolean)
    logDebug(
      s"[FluxSinglePartitionSortRule] apply: enabled=$enabled root=${plan.getClass.getSimpleName}")
    if (enabled) {
      logDebug(s"[FluxSinglePartitionSortRule] tree:\n${plan.treeString.take(1500)}")
    }
    if (!enabled) {
      return plan
    }
    rewriteRoot(plan)
  }

  // Recognize the global-sort root: vanilla SortExec(global=true) or
  // gluten's SortExecTransformer(global=true).
  private object GlobalSort {
    def unapply(p: SparkPlan): Option[SparkPlan] = p match {
      case s: SortExec if s.global => Some(s)
      case t: SortExecTransformer if t.global => Some(t)
      case _ => None
    }
  }

  private def rewriteRoot(node: SparkPlan): SparkPlan = node match {
    case aqe: AdaptiveSparkPlanExec => aqe
    case p: ProjectExec => p.withNewChildren(Seq(rewriteRoot(p.child)))
    case GlobalSort(s) => s.withNewChildren(Seq(rewriteSubtree(s.children.head)))
    case other if other.children.size == 1 && isRootSpine(other) =>
      other.withNewChildren(Seq(rewriteRoot(other.children.head)))
    case other => other
  }

  private def isRootSpine(p: SparkPlan): Boolean = p match {
    case _: SortExec | _: SortExecTransformer | _: ProjectExec => true
    case _ =>
      val n = p.getClass.getSimpleName
      n.contains("ColumnarToRow") || n.contains("InputIteratorTransformer") ||
      n.contains("RowToVeloxColumnar") || n.contains("WholeStageTransformer")
  }

  // Walk into the sort's input subtree and rewrite the first RangePartitioning shuffle
  // we encounter. This handles: direct shuffle, AQE-wrapped shuffle, and the
  // gluten-inserted wrapper chain (InputIteratorTransformer, RowToVeloxColumnar,
  // ColumnarToRow, ColumnarToColumnar) between sort and shuffle.
  private def rewriteSubtree(node: SparkPlan): SparkPlan = node match {
    case stage: ShuffleQueryStageExec =>
      stage.plan match {
        case sh: ShuffleExchangeLike if sh.outputPartitioning.isInstanceOf[RangePartitioning] =>
          logInfo("FluxSinglePartitionSortRule: rewriting RANGE -> SINGLE (AQE stage)")
          rewriteShuffle(sh)
        case _ => node
      }
    case sh: ShuffleExchangeLike if sh.outputPartitioning.isInstanceOf[RangePartitioning] =>
      logInfo(
        s"FluxSinglePartitionSortRule: rewriting RANGE -> SINGLE (${sh.getClass.getSimpleName})")
      rewriteShuffle(sh)
    case other if other.children.size == 1 =>
      other.withNewChildren(Seq(rewriteSubtree(other.children.head)))
    case other => other
  }

  // Rewrite the shuffle's partitioning from RangePartitioning to SinglePartition.
  // For ColumnarShuffleExchangeExec (gluten variant) we still emit a vanilla
  // ShuffleExchangeExec; gluten's columnar rule will re-wrap it on a later pass.
  private def rewriteShuffle(sh: ShuffleExchangeLike): SparkPlan = sh match {
    case se: ShuffleExchangeExec =>
      ShuffleExchangeExec(SinglePartition, se.child, se.shuffleOrigin)
    case other =>
      // Fallback: build a vanilla ShuffleExchangeExec around the same child.
      ShuffleExchangeExec(SinglePartition, other.children.head)
  }
}
