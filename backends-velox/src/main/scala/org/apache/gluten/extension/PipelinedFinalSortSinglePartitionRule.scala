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
import org.apache.spark.sql.execution.{ColumnarShuffleExchangeExec, GPUColumnarShuffleExchangeExec, ProjectExec, SortExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.exchange.{ShuffleExchangeExec, ShuffleExchangeLike}
import org.apache.spark.sql.internal.SQLConf

/**
 * Avoid a second execution of the full upstream pipeline for final ORDER BY range sampling.
 *
 * Spark satisfies a root global sort with RangePartitioning. RangePartitioner computes bounds by
 * launching a sampling job over the sort input. In a pipelined UCX shuffle plan that input can be
 * the entire join/aggregate pipeline, so the query first runs once for sampling and then runs again
 * for the actual result. For TPC-H-style final ORDER BY outputs, a SinglePartition gather followed
 * by the existing global sort preserves global ordering and avoids the sampling pass.
 *
 * This rule is intentionally narrow: it only rewrites the first RangePartitioning shuffle directly
 * under the root global Sort spine, and only when both this rule and pipelined shuffle are enabled.
 */
case class PipelinedFinalSortSinglePartitionRule() extends Rule[SparkPlan] with Logging {
  private val confKey =
    "spark.gluten.sql.columnar.pipelined.finalSort.singlePartition.enabled"
  private val pipelinedShuffleKey = "spark.sql.shuffle.pipelined.enabled"

  override def apply(plan: SparkPlan): SparkPlan = {
    val enabled = confBoolean(confKey, defaultValue = false)
    val pipelinedShuffleEnabled = confBoolean(pipelinedShuffleKey, defaultValue = false)
    if (!enabled || !pipelinedShuffleEnabled) {
      return plan
    }
    rewriteRoot(plan)
  }

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
    case GlobalSort(sort) => sort.withNewChildren(Seq(rewriteSortInput(sort.children.head)))
    case other if other.children.size == 1 && isRootSpine(other) =>
      other.withNewChildren(Seq(rewriteRoot(other.children.head)))
    case other => other
  }

  private def isRootSpine(plan: SparkPlan): Boolean = plan match {
    case _: SortExec | _: SortExecTransformer | _: ProjectExec => true
    case _ =>
      val name = plan.getClass.getSimpleName
      name.contains("ColumnarToRow") ||
      name.contains("InputIteratorTransformer") ||
      name.contains("RowToVeloxColumnar") ||
      name.contains("WholeStageTransformer")
  }

  private def rewriteSortInput(node: SparkPlan): SparkPlan = node match {
    case stage: ShuffleQueryStageExec =>
      stage.plan match {
        case shuffle: ShuffleExchangeLike
            if shuffle.outputPartitioning.isInstanceOf[RangePartitioning] =>
          rewriteShuffle(shuffle)
        case _ => node
      }
    case shuffle: ShuffleExchangeLike
        if shuffle.outputPartitioning.isInstanceOf[RangePartitioning] =>
      rewriteShuffle(shuffle)
    case other if other.children.size == 1 =>
      other.withNewChildren(Seq(rewriteSortInput(other.children.head)))
    case other => other
  }

  private def rewriteShuffle(shuffle: ShuffleExchangeLike): SparkPlan = {
    logInfo(
      s"PipelinedFinalSortSinglePartitionRule: rewriting final RANGE shuffle " +
        s"to SINGLE (${shuffle.getClass.getSimpleName})")
    shuffle match {
      case columnar: ColumnarShuffleExchangeExec =>
        columnar.copy(outputPartitioning = SinglePartition)
      case gpu: GPUColumnarShuffleExchangeExec =>
        gpu.copy(outputPartitioning = SinglePartition)
      case row: ShuffleExchangeExec =>
        row.copy(outputPartitioning = SinglePartition)
      case other =>
        ShuffleExchangeExec(SinglePartition, other.children.head)
    }
  }

  private def confBoolean(key: String, defaultValue: Boolean): Boolean =
    SQLConf.get.getConfString(key, defaultValue.toString).toBoolean
}
