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

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.plans.physical.{RangePartitioning, SinglePartition}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ProjectExec, SortExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec

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
case class MppSinglePartitionSortRule() extends Rule[SparkPlan] with Logging {
  private val confKey = "spark.gluten.mpp.singlePartitionSort"

  override def apply(plan: SparkPlan): SparkPlan = {
    val sess = org.apache.spark.sql.SparkSession.getActiveSession
    val enabled = sess.exists(_.conf.get(confKey, "false").toBoolean)
    logWarning(
      s"MppSinglePartitionSortRule.apply: enabled=$enabled plan=${plan.getClass.getSimpleName}")
    if (!enabled) {
      return plan
    }
    rewriteRoot(plan)
  }

  private def rewriteRoot(node: SparkPlan): SparkPlan = node match {
    case aqe: AdaptiveSparkPlanExec => aqe
    case p: ProjectExec => p.withNewChildren(Seq(rewriteRoot(p.child)))
    case s: SortExec if s.global => s.withNewChildren(Seq(rewriteExchange(s.child)))
    case other if other.children.size == 1 && isRootSpine(other) =>
      other.withNewChildren(Seq(rewriteRoot(other.children.head)))
    case other => other
  }

  private def isRootSpine(p: SparkPlan): Boolean = p match {
    case _: SortExec | _: ProjectExec => true
    case _ => p.getClass.getSimpleName.contains("ColumnarToRow")
  }

  private def rewriteExchange(node: SparkPlan): SparkPlan = node match {
    case stage: ShuffleQueryStageExec =>
      stage.plan match {
        case sh: ShuffleExchangeExec if sh.outputPartitioning.isInstanceOf[RangePartitioning] =>
          logWarning(
            s"MppSinglePartitionSortRule: rewriting RANGE -> SINGLE at root sort (AQE stage)")
          ShuffleExchangeExec(SinglePartition, sh.child, sh.shuffleOrigin)
        case _ => node
      }
    case sh: ShuffleExchangeExec if sh.outputPartitioning.isInstanceOf[RangePartitioning] =>
      logWarning(s"MppSinglePartitionSortRule: rewriting RANGE -> SINGLE at root sort")
      ShuffleExchangeExec(SinglePartition, sh.child, sh.shuffleOrigin)
    case _ => node
  }
}
