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

import org.apache.gluten.execution.ProjectExecTransformer

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.plans.physical.{HashPartitioning, Partitioning, PartitioningCollection}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarShuffleExchangeExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.AQEShuffleReadExec
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec

/**
 * Eliminate ShuffleExchangeExec(HashPartitioning) when the child's outputPartitioning already
 * satisfies the same hash distribution. This mirrors Presto's distributed planner behavior of
 * skipping redundant exchanges, reducing fragment count for join-heavy queries (TPC-H Q5/Q7/Q8/Q9
 * etc.).
 *
 * Conservative: only matches when child is HashPartitioning (or PartitioningCollection containing
 * one) with identical keys and either the same logical partition count or, behind a second opt-in,
 * the same finalized FLUX native partition count. Skips AQEShuffleReadExec (could have applied
 * coalesce/skew-split that breaks key alignment).
 *
 * Gated by spark.gluten.mpp.removeRedundantShuffle (default: false).
 */
case class FluxRemoveRedundantShuffleRule() extends Rule[SparkPlan] with Logging {
  private val confKey = "spark.gluten.mpp.removeRedundantShuffle"
  private val nativeCapAwareConfKey =
    "spark.gluten.mpp.removeRedundantShuffle.nativePartitionCapAware"

  override def apply(plan: SparkPlan): SparkPlan = {
    val sess = org.apache.spark.sql.SparkSession.getActiveSession
    val enabled = sess.exists(_.conf.get(confKey, "false").toBoolean)
    logDebug(
      s"FluxRemoveRedundantShuffleRule.apply: enabled=$enabled plan=${plan.getClass.getSimpleName}")
    if (!enabled) {
      return plan
    }
    val nativePartitioning = sess.flatMap {
      spark =>
        val capAware = spark.conf.get(nativeCapAwareConfKey, "false").toBoolean
        if (capAware) {
          FluxRemoveRedundantShuffleRule.nativeHashPartitioning {
            (key, default) => spark.conf.get(key, default)
          }
        } else {
          None
        }
    }
    plan.transformUp {
      case s @ ShuffleExchangeExec(req: HashPartitioning, child, _, _)
          if !child.isInstanceOf[AQEShuffleReadExec] &&
            hashSatisfies(child.outputPartitioning, req, nativePartitioning) =>
        logEliminated(s.nodeName, child.outputPartitioning, req, nativePartitioning)
        child
      case s @ ColumnarShuffleExchangeExec(req: HashPartitioning, child, _, _, _)
          if !child.isInstanceOf[AQEShuffleReadExec] &&
            hashSatisfies(child.outputPartitioning, req, nativePartitioning) =>
        logEliminated(s.nodeName, child.outputPartitioning, req, nativePartitioning)
        preserveExchangeOutput(s, child)
    }
  }

  /**
   * Velox's HASH shuffle prepends a synthetic `hash_partition_key` to its child and exposes
   * `projectOutputAttributes` as the exchange output. Removing that exchange must retain this
   * projection boundary or the synthetic key leaks into the query result / table schema.
   */
  private def preserveExchangeOutput(exchange: SparkPlan, child: SparkPlan): SparkPlan = {
    val outputUnchanged =
      exchange.output.length == child.output.length &&
        exchange.output.zip(child.output).forall {
          case (left, right) => left.semanticEquals(right)
        }
    if (outputUnchanged) child else ProjectExecTransformer(exchange.output, child)
  }

  private def logEliminated(
      exchangeName: String,
      childPartitioning: Partitioning,
      requestedPartitioning: HashPartitioning,
      nativePartitioning: Option[FluxRemoveRedundantShuffleRule.NativeHashPartitioning]): Unit = {
    logInfo(
      s"FluxRemoveRedundantShuffle: eliminated redundant $exchangeName, " +
        s"child partitioning $childPartitioning already satisfies $requestedPartitioning" +
        nativePartitioning.map(p => s" after ${p.description}").getOrElse(""))
  }

  private def hashSatisfies(
      child: Partitioning,
      req: HashPartitioning,
      nativePartitioning: Option[FluxRemoveRedundantShuffleRule.NativeHashPartitioning]): Boolean =
    child match {
      case h: HashPartitioning =>
        FluxRemoveRedundantShuffleRule.partitionCountsMatch(
          h.numPartitions,
          req.numPartitions,
          nativePartitioning) &&
        h.expressions.length == req.expressions.length &&
        h.expressions.zip(req.expressions).forall { case (a, b) => a.semanticEquals(b) }
      case PartitioningCollection(parts) =>
        parts.exists(p => hashSatisfies(p, req, nativePartitioning))
      case _ => false
    }
}

object FluxRemoveRedundantShuffleRule {
  private val MultiExecutorEnabledKey = "spark.gluten.mpp.multiExecutor.enabled"
  private val MultiExecutorPartitionsKey = "spark.gluten.mpp.multiExecutor.numPartitions"
  private val RemoteDestinationsKey = "spark.gluten.mpp.remoteHashExchangeDestinations"
  private val LegacyRemoteDestinationsKey = "spark.gluten.mpp.localHashExchangeTasks"
  private val DestinationLanesKey =
    "spark.gluten.sql.columnar.backend.velox.flux.keyedFinalDestinationLanes"
  private val LocalDriversKey =
    "spark.gluten.sql.columnar.backend.velox.flux.keyedFinalLocalDrivers"

  private[extension] case class NativeHashPartitioning(cap: Int, exact: Boolean, source: String) {
    def effective(logicalPartitions: Int): Int =
      if (exact) cap else math.min(logicalPartitions, cap)

    def description: String = s"native HASH cap $cap from $source"
  }

  private[extension] def partitionCountsMatch(
      childPartitions: Int,
      requestedPartitions: Int,
      nativePartitioning: Option[NativeHashPartitioning]): Boolean = {
    childPartitions == requestedPartitions || nativePartitioning.exists {
      native => native.effective(childPartitions) == native.effective(requestedPartitions)
    }
  }

  /**
   * Mirrors FluxNativeQueryExec.effectiveRemoteHashExchangeDestinations. Keep this rule opt-in so a
   * Catalyst partition count is ignored only when both exchanges are guaranteed to be finalized to
   * the same native HASH topology later in the same FLUX planning path.
   */
  private[extension] def nativeHashPartitioning(
      getConf: (String, String) => String): Option[NativeHashPartitioning] = {
    def bool(key: String, default: Boolean): Boolean =
      getConf(key, default.toString).trim.toBoolean
    def positiveInt(key: String): Option[Int] = {
      val raw = getConf(key, "").trim
      if (raw.isEmpty) None
      else {
        val parsed = raw.toInt
        if (parsed > 0) Some(parsed) else None
      }
    }

    if (!bool(MultiExecutorEnabledKey, default = false)) {
      return None
    }

    val configured = positiveInt(RemoteDestinationsKey)
    val legacy = positiveInt(LegacyRemoteDestinationsKey)
    if (configured.nonEmpty && legacy.nonEmpty && configured != legacy) {
      return None
    }
    val (cap, source) = configured
      .map(_ -> RemoteDestinationsKey)
      .orElse(legacy.map(_ -> LegacyRemoteDestinationsKey))
      .getOrElse {
        val peers = positiveInt(MultiExecutorPartitionsKey).getOrElse(2)
        val lanes =
          if (bool(DestinationLanesKey, default = false)) {
            math.min(4, positiveInt(LocalDriversKey).getOrElse(1))
          } else {
            1
          }
        Math.multiplyExact(peers, lanes) ->
          (if (lanes > 1) s"$MultiExecutorPartitionsKey*$DestinationLanesKey"
           else MultiExecutorPartitionsKey)
      }
    Some(NativeHashPartitioning(cap, exact = bool(DestinationLanesKey, default = false), source))
  }
}
