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
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.SortOrder
import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.catalyst.plans.physical.{RoundRobinPartitioning, SinglePartition}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ProjectExec, SortExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.exchange.{ShuffleExchangeExec, ShuffleExchangeLike}
import org.apache.spark.sql.internal.SQLConf

/**
 * Mimic Presto's parallel-sort split for global ORDER BY queries.
 *
 * Presto's distributed planner inserts an extra fragment between the last hashed FinalAggregate
 * fragment and the SINGLE-gather output fragment:
 * {{{
 *   Frag N+1 [HASH]        : FinalAgg + PartialSort, output ROUND_ROBIN
 *   Frag N   [ROUND_ROBIN] : LocalMerge + PartialSort over RemoteSource, output SINGLE
 *   Frag N-1 [SINGLE]      : Output / RemoteMerge
 * }}}
 * Gluten currently goes straight from FinalAgg+Sort to SINGLE gather, missing the parallel pre-sort
 * merge step. This rule re-introduces it.
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
 * [[SortExecTransformer]] on a later pass. `N` defaults to `spark.sql.shuffle.partitions` (falling
 * back to 4).
 *
 * Gated by spark.gluten.mpp.parallelSortSplit (default: false).
 */
case class MppParallelSortSplitRule() extends Rule[SparkPlan] with Logging {
  private val confKey = "spark.gluten.mpp.parallelSortSplit"
  private val minRowsConfKey = "spark.gluten.mpp.parallelSortSplit.minRows"
  private val minBytesConfKey = "spark.gluten.mpp.parallelSortSplit.minBytes"
  private val splitAggregateUnknownStatsConfKey =
    "spark.gluten.mpp.parallelSortSplit.aggregateUnknownStats"
  private val defaultMinRows = BigInt(1000000L)
  private val defaultMinBytes = BigInt(256L * 1024L * 1024L)
  private val maxStatsLookupDepth = 4
  private val maxAggregateLookupDepth = 8

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
          if (shouldSplit(sh.children.head)) {
            logWarning("MppParallelSortSplitRule: splicing parallel-sort + RR merge (AQE stage)")
            spliceParallelSortMerge(sh, sortOrder)
          } else {
            node
          }
        case _ => node
      }
    case sh: ShuffleExchangeLike if sh.outputPartitioning == SinglePartition =>
      if (shouldSplit(sh.children.head)) {
        logWarning(
          s"MppParallelSortSplitRule: splicing parallel-sort + RR merge " +
            s"(${sh.getClass.getSimpleName})")
        spliceParallelSortMerge(sh, sortOrder)
      } else {
        node
      }
    case other if other.children.size == 1 =>
      other.withNewChildren(Seq(rewriteSortChild(other.children.head, sortOrder)))
    case other => other
  }

  /**
   * Build the new spine over the SinglePartition shuffle's producer: SinglePartition( LocalMerge(
   * RoundRobin( PartialSort( producer ) ) ) ).
   *
   * For a [[ColumnarShuffleExchangeExec]] (gluten variant) we still emit a vanilla
   * [[ShuffleExchangeExec]] for both the SinglePartition gather and the new RoundRobin exchange;
   * gluten's columnar pass will re-wrap them later. The two new [[SortExec]]s are local
   * (`global=false`); columnar transform will lift them to [[SortExecTransformer]] on a later pass.
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

  private def shouldSplit(producer: SparkPlan): Boolean = {
    logicalStats(producer, depth = 0) match {
      case Some(stats) =>
        val rowThreshold = minRows()
        val byteThreshold = minBytes()
        val rowsOk = stats.rowCount.exists(_ >= rowThreshold)
        val bytesOk = isConfidentSize(stats.sizeInBytes) && stats.sizeInBytes >= byteThreshold
        if (isAggregateProducer(producer) && !rowsOk) {
          logWarning(
            "MppParallelSortSplitRule: skipped; aggregate producer output is below " +
              "split row threshold or lacks reliable row stats " +
              s"rows=${stats.rowCount.getOrElse("unknown")} bytes=${stats.sizeInBytes} " +
              s"minRows=$rowThreshold minBytes=$byteThreshold")
          return false
        }
        val enabled = rowsOk || bytesOk
        if (!enabled) {
          logWarning(
            "MppParallelSortSplitRule: skipped; producer output below split threshold " +
              s"rows=${stats.rowCount.getOrElse("unknown")} bytes=${stats.sizeInBytes} " +
              s"minRows=$rowThreshold minBytes=$byteThreshold")
        }
        enabled
      case None =>
        if (splitAggregateUnknownStats() && containsAggregateProducer(producer)) {
          logWarning(
            "MppParallelSortSplitRule: splicing parallel-sort + RR merge despite missing " +
              "logical stats; producer contains aggregate and " +
              s"$splitAggregateUnknownStatsConfKey=true")
          true
        } else {
          logWarning("MppParallelSortSplitRule: skipped; producer has no reliable logical stats")
          false
        }
    }
  }

  private def isAggregateProducer(plan: SparkPlan): Boolean =
    isAggregateProducer(plan, depth = 0)

  private def isAggregateProducer(plan: SparkPlan, depth: Int): Boolean = {
    if (isAggregateLike(plan)) {
      true
    } else {
      plan match {
        case _: ShuffleExchangeLike | _: ShuffleQueryStageExec => false
        case _ if depth < maxStatsLookupDepth && plan.children.size == 1 =>
          isAggregateProducer(plan.children.head, depth + 1)
        case _ => false
      }
    }
  }

  private def isAggregateLike(plan: SparkPlan): Boolean = {
    val nodeName = plan.nodeName.toLowerCase(java.util.Locale.ROOT)
    val className = plan.getClass.getSimpleName.toLowerCase(java.util.Locale.ROOT)
    nodeName.contains("aggregate") || className.contains("aggregate")
  }

  private def containsAggregateProducer(plan: SparkPlan): Boolean =
    containsAggregateProducer(plan, depth = 0)

  private def containsAggregateProducer(plan: SparkPlan, depth: Int): Boolean = {
    if (isAggregateLike(plan)) {
      true
    } else {
      plan match {
        case _: ShuffleExchangeLike | _: ShuffleQueryStageExec => false
        case _ if depth < maxAggregateLookupDepth =>
          plan.children.exists(child => containsAggregateProducer(child, depth + 1))
        case _ => false
      }
    }
  }

  private def logicalStats(plan: SparkPlan, depth: Int): Option[Statistics] = {
    plan.logicalLink.map(_.stats).filter(hasUsableStats).orElse {
      plan match {
        case _: ShuffleExchangeLike | _: ShuffleQueryStageExec => None
        case _ if depth < maxStatsLookupDepth && plan.children.size == 1 =>
          logicalStats(plan.children.head, depth + 1)
        case _ => None
      }
    }
  }

  private def hasUsableStats(stats: Statistics): Boolean =
    isConfidentSize(stats.sizeInBytes) || stats.rowCount.exists(_ > 0)

  private def isConfidentSize(size: BigInt): Boolean =
    size > 0 && size < BigInt(Long.MaxValue)

  private def numShufflePartitions(): Int = {
    val raw =
      try {
        SQLConf.get.getConfString("spark.sql.shuffle.partitions", "4")
      } catch {
        case _: Throwable => "4"
      }
    val parsed =
      try raw.toInt
      catch { case _: Throwable => 4 }
    math.max(parsed, 1)
  }

  private def minRows(): BigInt =
    bigIntConf(minRowsConfKey, defaultMinRows)

  private def minBytes(): BigInt =
    bigIntConf(minBytesConfKey, defaultMinBytes)

  private def splitAggregateUnknownStats(): Boolean =
    confString(splitAggregateUnknownStatsConfKey, "true").toBoolean

  private def bigIntConf(key: String, defaultValue: BigInt): BigInt =
    try {
      BigInt(confString(key, defaultValue.toString))
    } catch {
      case _: Throwable => defaultValue
    }

  private def confString(key: String, defaultValue: String): String =
    try {
      SQLConf.get.getConfString(key, defaultValue)
    } catch {
      case _: Throwable => defaultValue
    }
}
