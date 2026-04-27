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
import org.apache.spark.sql.catalyst.expressions.SortOrder
import org.apache.spark.sql.catalyst.plans.physical.{RoundRobinPartitioning, SinglePartition}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ProjectExec, SortExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.exchange.{ShuffleExchangeExec, ShuffleExchangeLike}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.SparkSession

/**
 * Mimic Presto's parallel-sort split for global ORDER BY queries.
 *
 * Presto's distributed planner inserts an extra fragment between the last hashed
 * FinalAggregate fragment and the SINGLE-gather output fragment:
 * {{{
 *   Frag N+1 [HASH]        : FinalAgg + PartialSort, output ROUND_ROBIN
 *   Frag N   [ROUND_ROBIN] : LocalMerge + PartialSort over RemoteSource, output SINGLE
 *   Frag N-1 [SINGLE]      : Output / RemoteMerge
 * }}}
 * Gluten currently goes straight from FinalAgg+Sort to SINGLE gather, missing the
 * parallel pre-sort merge step. This rule re-introduces it.
 *
 * Pattern recognized at the plan root (after [[MppSinglePartitionSortRule]] has converted
 * `Exchange(RangePartitioning) -> Sort(global)` into `Exchange(SinglePartition) -> Sort`):
 * {{{
 *   (ColumnarToRow? -> Project?) -> Sort(global=true) -> Exchange(SinglePartition) -> producer
 * }}}
 * Rewritten to:
 * {{{
 *   Sort(global=true)
 *   -> Exchange(SinglePartition)
 *      -> Sort(global=false, sortOrder)             (driver-side merge sort)
 *         -> Exchange(RoundRobinPartitioning(N))
 *            -> Sort(global=false, sortOrder)       (parallel partial sort, N drivers)
 *               -> producer
 * }}}
 *
 * The two new sorts are local; gluten's columnar transform will rewrite them to
 * [[SortExecTransformer]] on a later pass. `N` defaults to
 * `spark.sql.shuffle.partitions` (falling back to 4).
 *
 * Gated by spark.gluten.mpp.parallelSortSplit (default: false).
 */
case class MppParallelSortSplitRule() extends Rule[SparkPlan] with Logging {
  private val confKey = "spark.gluten.mpp.parallelSortSplit"

  override def apply(plan: SparkPlan): SparkPlan = {
    val sess = SparkSession.getActiveSession
    val enabled = sess.exists(_.conf.get(confKey, "false").toBoolean)
    logWarning(
      s"[MppParallelSortSplitRule] apply: enabled=$enabled root=${plan.getClass.getSimpleName}")
    if (!enabled) {
      return plan
    }
    logWarning(s"[MppParallelSortSplitRule] tree:\n${plan.treeString.take(1500)}")
    rewriteRoot(plan)
  }

  /** Recognize the global-sort root: vanilla SortExec(global=true) or SortExecTransformer. */
  private object GlobalSort {
    def unapply(p: SparkPlan): Option[(SparkPlan, Seq[SortOrder], SparkPlan)] = p match {
      case s: SortExec if s.global => Some((s, s.sortOrder, s.child))
      case t: SortExecTransformer if t.global => Some((t, t.sortOrder, t.child))
      case _ => None
    }
  }

  /**
   * Walk through AdaptiveSparkPlanExec / Project / unary spine wrappers at the root, looking for
   * the global Sort. Mirrors [[MppSinglePartitionSortRule.isRootSpine]].
   */
  private def rewriteRoot(node: SparkPlan): SparkPlan = node match {
    case aqe: AdaptiveSparkPlanExec => aqe
    case p: ProjectExec => p.withNewChildren(Seq(rewriteRoot(p.child)))
    case GlobalSort(sortNode, sortOrder, sortChild) =>
      // Try to rewrite the subtree below the global sort: insert RR-merge if we find
      // a SinglePartition shuffle directly below.
      val newSortChild = rewriteSortChild(sortChild, sortOrder)
      sortNode.withNewChildren(Seq(newSortChild))
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

  /**
   * Walk into the global Sort's input subtree. When we find the Exchange(SinglePartition) that
   * gathers the producer, splice in:
   * {{{
   *   Exchange(SinglePartition)
   *     -> Sort(global=false, sortOrder)             // local merge sort on the gather side
   *        -> Exchange(RoundRobinPartitioning(N))
   *           -> Sort(global=false, sortOrder)       // parallel partial sort, N drivers
   *              -> originalProducer
   * }}}
   * Stops at the first SinglePartition shuffle along the spine; if none is found, returns the
   * subtree unchanged.
   */
  private def rewriteSortChild(node: SparkPlan, sortOrder: Seq[SortOrder]): SparkPlan = node match {
    case stage: ShuffleQueryStageExec =>
      stage.plan match {
        case sh: ShuffleExchangeLike if sh.outputPartitioning == SinglePartition =>
          logWarning(
            "MppParallelSortSplitRule: splicing parallel-sort + RR merge (AQE stage)")
          spliceParallelSortMerge(sh, sortOrder)
        case _ => node
      }
    case sh: ShuffleExchangeLike if sh.outputPartitioning == SinglePartition =>
      logWarning(
        s"MppParallelSortSplitRule: splicing parallel-sort + RR merge " +
          s"(${sh.getClass.getSimpleName})")
      spliceParallelSortMerge(sh, sortOrder)
    case other if other.children.size == 1 =>
      other.withNewChildren(Seq(rewriteSortChild(other.children.head, sortOrder)))
    case other => other
  }

  /**
   * Build the new spine over the SinglePartition shuffle's producer:
   * SinglePartition( LocalMerge( RoundRobin( PartialSort( producer ) ) ) ).
   *
   * For a [[ColumnarShuffleExchangeExec]] (gluten variant) we still emit a vanilla
   * [[ShuffleExchangeExec]] for both the SinglePartition gather and the new RoundRobin
   * exchange; gluten's columnar pass will re-wrap them later. The two new
   * [[SortExec]]s are local (`global=false`); columnar transform will lift them
   * to [[SortExecTransformer]] on a later pass.
   */
  private def spliceParallelSortMerge(
      gather: ShuffleExchangeLike,
      sortOrder: Seq[SortOrder]): SparkPlan = {
    val producer = gather.children.head
    val n = numShufflePartitions()

    // Parallel partial sort: one local sort per round-robin driver.
    val partialSort = SortExec(sortOrder, global = false, child = producer)

    // Round-robin redistribution forces N partitions.
    val rrExchange: SparkPlan = ShuffleExchangeExec(RoundRobinPartitioning(n), partialSort)

    // Driver-side merge sort over the per-stream sorted RR output.
    val mergeSort = SortExec(sortOrder, global = false, child = rrExchange)

    // Final SinglePartition gather sits on top, feeding the original global Sort.
    gather match {
      case se: ShuffleExchangeExec =>
        ShuffleExchangeExec(SinglePartition, mergeSort, se.shuffleOrigin)
      case _ =>
        ShuffleExchangeExec(SinglePartition, mergeSort)
    }
  }

  private def numShufflePartitions(): Int = {
    val raw =
      try {
        SQLConf.get.getConfString("spark.sql.shuffle.partitions", "4")
      } catch {
        case _: Throwable => "4"
      }
    val parsed = try raw.toInt
    catch { case _: Throwable => 4 }
    math.max(parsed, 1)
  }
}
