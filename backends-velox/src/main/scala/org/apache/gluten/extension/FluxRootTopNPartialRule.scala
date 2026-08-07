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

import org.apache.gluten.execution.{TakeOrderedAndProjectExecTransformer, TopNTransformer}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.SortOrder
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ProjectExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanExec

/**
 * Insert a local TopN below a root TakeOrderedAndProject.
 *
 * Presto rewrites a single TopN into partial TopN followed by final TopN. This keeps only the top-K
 * rows per local stream before the gather and lets the root TopN merge a much smaller input.
 * Spark's TakeOrderedAndProject hides that split inside one root operator, so the FLUX fragment
 * walk
 * otherwise creates a join-heavy producer fragment followed by a SINGLE gather into a single-driver
 * final TopN fragment.
 *
 * Gated by spark.gluten.mpp.rootTopNPartial (default: false).
 */
case class FluxRootTopNPartialRule() extends Rule[SparkPlan] with Logging {
  private val confKey = "spark.gluten.mpp.rootTopNPartial"

  override def apply(plan: SparkPlan): SparkPlan = {
    val enabled = SparkSession.getActiveSession.exists(_.conf.get(confKey, "false").toBoolean)
    if (!enabled) {
      return plan
    }
    rewriteRoot(plan)
  }

  private def rewriteRoot(node: SparkPlan): SparkPlan = node match {
    case aqe: AdaptiveSparkPlanExec => aqe
    case topk: TakeOrderedAndProjectExecTransformer if topk.offset == 0 =>
      buildPartialTopN(topk.limit, topk.sortOrder, topk.child) match {
        case Some(rewrittenChild) =>
          logWarning(
            s"FluxRootTopNPartialRule: inserted partial TopN(limit=${topk.limit}) " +
              "below root TakeOrderedAndProject")
          topk.copy(child = rewrittenChild)
        case None => topk
      }
    case p: ProjectExec => p.withNewChildren(Seq(rewriteRoot(p.child)))
    case other if other.children.size == 1 && isRootSpine(other) =>
      other.withNewChildren(Seq(rewriteRoot(other.children.head)))
    case other => other
  }

  private def isRootSpine(plan: SparkPlan): Boolean = plan match {
    case _: TakeOrderedAndProjectExecTransformer | _: ProjectExec => true
    case _ =>
      val name = plan.getClass.getSimpleName
      name.contains("ColumnarToRow") || name.contains("InputIteratorTransformer") ||
      name.contains("RowToVeloxColumnar") || name.contains("WholeStageTransformer")
  }

  private def buildPartialTopN(
      limit: Long,
      sortOrder: Seq[SortOrder],
      child: SparkPlan): Option[SparkPlan] = {
    if (limit <= 0 || alreadyPartialTopN(child, limit, sortOrder)) {
      return None
    }
    val topN = TopNTransformer(limit, sortOrder, global = false, child, isPartial = true)
    val validation = topN.doValidate()
    if (validation.ok()) {
      Some(topN)
    } else {
      logWarning(
        s"FluxRootTopNPartialRule: TopN validation failed ($validation); leaving plan unchanged")
      None
    }
  }

  private def alreadyPartialTopN(plan: SparkPlan, limit: Long, sortOrder: Seq[SortOrder]): Boolean =
    plan match {
      case topN: TopNTransformer =>
        topN.limit == limit && !topN.global && topN.sortOrder == sortOrder
      case _ => false
    }
}
