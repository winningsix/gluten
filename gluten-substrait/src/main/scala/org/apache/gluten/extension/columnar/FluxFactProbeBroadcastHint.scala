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
package org.apache.gluten.extension.columnar

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.plans.InnerLike
import org.apache.spark.sql.catalyst.plans.logical.{BROADCAST, HintInfo, Join}
import org.apache.spark.sql.catalyst.plans.logical.{JoinHint, LogicalPlan}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.datasources.LogicalRelation

import scala.util.Try

/**
 * Keep a large fact table on the PROBE side of an inner join by forcing the small dimension side to
 * broadcast (build). This fixes the selection UPSTREAM of EnsureRequirements -- it only chooses the
 * right side; EnsureRequirements then places the BroadcastExchange on the (small) build side.
 *
 * Background: with missing column stats the optimizer's join-cardinality estimates explode, and a
 * runtime bloom filter can deflate a fact-table scan estimate below
 * `spark.sql.autoBroadcastJoinThreshold`. Spark's JoinSelection can then broadcast the large input,
 * which the native FLUX runtime replicates per driver and may OOM. A broadcast hint on the smaller
 * input overrides the size-based choice so the large side streams as probe.
 *
 * The REAL on-disk leaf scan size (sum of leaf relation sizeInBytes) is used only to choose which
 * side is the fact/probe side. Spark plan stats (`plan.stats.sizeInBytes`) drive both Spark's
 * normal auto-broadcast threshold and the optional FLUX network-cost extension after filters and
 * projections shrink an intermediate.
 */
case class FluxFactProbeBroadcastHint(spark: SparkSession) extends Rule[LogicalPlan] with Logging {

  private val singleTaskModeKey = "spark.gluten.sql.columnar.backend.velox.flux.singleTaskMode"
  private val maxBuildBytesKey = "spark.gluten.mpp.factProbeBroadcastHint.maxBuildBytes"
  private val minNetworkSavingsRatioKey =
    "spark.gluten.mpp.factProbeBroadcastHint.minNetworkSavingsRatio"
  private val fluxPartitionsKey = "spark.gluten.mpp.multiExecutor.numPartitions"

  private def enabled: Boolean =
    spark.sessionState.conf
      .getConfString("spark.gluten.mpp.factProbeBroadcastHint", "true")
      .toBoolean && !singleTaskModeEnabled

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!enabled || !plan.resolved) {
      return plan
    }
    val maxBroadcastBytes = BigInt(spark.sessionState.conf.autoBroadcastJoinThreshold)
    if (maxBroadcastBytes < 0) {
      return plan
    }
    plan.transform {
      case join @ Join(left, right, _: InnerLike, Some(_), JoinHint.NONE) =>
        val leftScanBytes = realScanBytes(left)
        val rightScanBytes = realScanBytes(right)
        val leftStatsBytes = broadcastBytes(left)
        val rightStatsBytes = broadcastBytes(right)
        val broadcastHint = HintInfo(strategy = Some(BROADCAST))
        // Broadcast (REPLICATE) the SMALLER side so the largest input stays the
        // in-place probe. The hint only applies to unhinted
        // joins; earlier planning rules may choose SHUFFLE_HASH first when a medium
        // intermediate should stay repartitioned before it is later replicated.
        // Use leaf scan bytes to avoid choosing the fact side. Use Spark plan
        // stats for the threshold and network-cost checks so filtered/projected intermediates
        // follow the same estimates as Spark JoinSelection.
        if (
          leftScanBytes > 0 && rightScanBytes > 0 && leftScanBytes < rightScanBytes &&
          shouldBroadcast(leftStatsBytes, rightStatsBytes, maxBroadcastBytes)
        ) {
          logWarning(
            s"FluxFactProbeBroadcastHint: broadcasting smaller LEFT side " +
              s"(leftScanBytes=$leftScanBytes rightScanBytes=$rightScanBytes " +
              s"leftStatsBytes=$leftStatsBytes rightStatsBytes=$rightStatsBytes " +
              s"autoBroadcastJoinThreshold=$maxBroadcastBytes); larger side stays probe")
          join.copy(hint = JoinHint(Some(broadcastHint), None))
        } else if (
          leftScanBytes > 0 && rightScanBytes > 0 && rightScanBytes < leftScanBytes &&
          shouldBroadcast(rightStatsBytes, leftStatsBytes, maxBroadcastBytes)
        ) {
          logWarning(
            s"FluxFactProbeBroadcastHint: broadcasting smaller RIGHT side " +
              s"(leftScanBytes=$leftScanBytes rightScanBytes=$rightScanBytes " +
              s"leftStatsBytes=$leftStatsBytes rightStatsBytes=$rightStatsBytes " +
              s"autoBroadcastJoinThreshold=$maxBroadcastBytes); larger side stays probe")
          join.copy(hint = JoinHint(None, Some(broadcastHint)))
        } else {
          join
        }
    }
  }

  // Sum the real on-disk bytes of every leaf relation under a subtree.
  private def realScanBytes(plan: LogicalPlan): BigInt = {
    var total = BigInt(0)
    plan.foreach {
      case lr: LogicalRelation => total += BigInt(lr.relation.sizeInBytes)
      case _ =>
    }
    total
  }

  private def broadcastBytes(plan: LogicalPlan): BigInt = plan.stats.sizeInBytes

  /**
   * Extend Spark's fixed auto-broadcast threshold with an FLUX network-cost decision, while keeping
   * an independent per-peer build-memory ceiling.
   *
   * A hash join sends both sides through the network. A replicated join sends the build to every
   * other peer, so its first-order network cost is `buildBytes * (peers - 1)`. Above Spark's normal
   * threshold we only add a broadcast hint when that replicated cost is materially smaller than
   * shuffling both inputs and the build remains below the explicit safety ceiling. The default
   * ceiling is `auto`, which preserves Spark's threshold; launchers for large-memory FLUX workers
   * may raise it without globally changing Spark JoinSelection.
   */
  private def shouldBroadcast(
      buildBytes: BigInt,
      probeBytes: BigInt,
      autoBroadcastBytes: BigInt): Boolean = {
    if (buildBytes <= 0 || probeBytes <= 0) {
      return false
    }
    if (buildBytes <= autoBroadcastBytes) {
      return true
    }

    val maxBuildBytes = configuredMaxBuildBytes(autoBroadcastBytes)
    if (buildBytes > maxBuildBytes) {
      return false
    }

    val peers = configuredFluxPartitions
    if (peers <= 1) {
      return false
    }
    val broadcastNetworkBytes = buildBytes * (peers - 1)
    val shuffleNetworkBytes = probeBytes + buildBytes
    val minSavingsRatio = configuredMinNetworkSavingsRatio
    val beneficial =
      BigDecimal(shuffleNetworkBytes) >=
        BigDecimal(broadcastNetworkBytes) * BigDecimal(minSavingsRatio)
    if (beneficial) {
      logWarning(
        s"FluxFactProbeBroadcastHint: cost-gated broadcast above Spark threshold " +
          s"(buildBytes=$buildBytes probeBytes=$probeBytes peers=$peers " +
          s"broadcastNetworkBytes=$broadcastNetworkBytes " +
          s"shuffleNetworkBytes=$shuffleNetworkBytes " +
          s"minNetworkSavingsRatio=$minSavingsRatio maxBuildBytes=$maxBuildBytes " +
          s"autoBroadcastJoinThreshold=$autoBroadcastBytes)")
    }
    beneficial
  }

  private def configuredMaxBuildBytes(autoBroadcastBytes: BigInt): BigInt = {
    val raw = spark.sessionState.conf.getConfString(maxBuildBytesKey, "auto").trim
    if (raw.isEmpty || raw.equalsIgnoreCase("auto")) {
      autoBroadcastBytes
    } else {
      parseBytes(raw).getOrElse {
        logWarning(
          s"FluxFactProbeBroadcastHint: invalid $maxBuildBytesKey=$raw; " +
            s"using autoBroadcastJoinThreshold=$autoBroadcastBytes")
        autoBroadcastBytes
      }
    }
  }

  private def configuredFluxPartitions: Int = {
    val fallback = spark.sessionState.conf.numShufflePartitions
    Try(spark.sessionState.conf.getConfString(fluxPartitionsKey, fallback.toString).toInt).toOption
      .filter(_ > 0)
      .getOrElse(fallback)
  }

  private def configuredMinNetworkSavingsRatio: Double = {
    Try(
      spark.sessionState.conf
        .getConfString(minNetworkSavingsRatioKey, "1.25")
        .trim
        .toDouble).toOption
      .filter(_ >= 1.0)
      .getOrElse(1.25)
  }

  private def parseBytes(raw: String): Option[BigInt] = {
    val normalized = raw.trim.toLowerCase(java.util.Locale.ROOT)
    val BytePattern = "^([0-9]+)([kmgt]?)(i?b)?$".r
    normalized match {
      case BytePattern(value, unit, _) =>
        val shift = unit match {
          case "" => 0
          case "k" => 10
          case "m" => 20
          case "g" => 30
          case "t" => 40
        }
        Some(BigInt(value) << shift)
      case _ => None
    }
  }

  private def singleTaskModeEnabled: Boolean =
    spark.sessionState.conf.getConfString(singleTaskModeKey, "false").toBoolean

}
