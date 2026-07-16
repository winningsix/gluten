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

import org.apache.gluten.execution.{CartesianProductExecTransformer, ColumnarCartesianProductBridge, ProjectExecTransformer, VeloxBroadcastNestedLoopJoinExecTransformer}

import org.apache.spark.internal.Logging
import org.apache.spark.network.util.JavaUtils
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight, BuildSide}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{Join, Statistics}
import org.apache.spark.sql.catalyst.plans.physical.IdentityBroadcastMode
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.execution.{ColumnarBroadcastExchangeExec, SparkPlan}
import org.apache.spark.sql.internal.SQLConf

/**
 * Replaces Spark's partition-pair Cartesian bridge with a native replicated nested-loop join, but
 * only when Catalyst statistics prove that one complete input is bounded by the configured MPP
 * build-side limit.
 *
 * A plain [[CartesianProductExecTransformer]] cannot be fused into an MPP fragment: Spark's
 * Cartesian RDD pairs every left partition with every right partition, while a local native
 * nested-loop join sees only its peer's inputs. This rule makes the native semantics equivalent by
 * retaining the large side as the probe and placing the complete small side behind a native
 * BROADCAST exchange. The resulting Substrait CrossRel is executed by Velox NestedLoopJoin and
 * CudfNestedLoopJoin.
 *
 * Unknown or over-limit inputs are deliberately left unchanged in non-strict mode, so the normal
 * Spark partition-pair implementation remains available. Strict MPP fails during planning with a
 * targeted validation error instead of running an incomplete local Cartesian product.
 */
case class MppReplicatedCartesianRule() extends Rule[SparkPlan] with Logging {

  import MppReplicatedCartesianRule._

  override def apply(plan: SparkPlan): SparkPlan = {
    if (!enabled) {
      return plan
    }

    plan.transformUp {
      case cart @ CartesianProductExecTransformer(
            ColumnarCartesianProductBridge(left),
            ColumnarCartesianProductBridge(right),
            condition) =>
        val (leftStats, rightStats) = sideStats(cart, left, right)
        chooseBuildSide(leftStats, rightStats, maxBuildBytes) match {
          case Some(BuildRight) =>
            logRewrite(cart, BuildRight, rightStats)
            markReplicatedCartesian(
              VeloxBroadcastNestedLoopJoinExecTransformer(
                left = left,
                right = ColumnarBroadcastExchangeExec(IdentityBroadcastMode, right),
                buildSide = BuildRight,
                joinType = Inner,
                condition = condition))

          case Some(BuildLeft) =>
            logRewrite(cart, BuildLeft, leftStats)
            // Keep the native transformer in streamed/build order (build is always on the right)
            // so MPP iterator slots match Substrait source order. Restore the original Spark
            // Cartesian output contract (left columns followed by right columns) explicitly.
            val nestedLoop = markReplicatedCartesian(
              VeloxBroadcastNestedLoopJoinExecTransformer(
                left = right,
                right = ColumnarBroadcastExchangeExec(IdentityBroadcastMode, left),
                buildSide = BuildRight,
                joinType = Inner,
                condition = condition))
            ProjectExecTransformer(cart.output, nestedLoop)

          case None =>
            val reason = rejectionReason(leftStats, rightStats, maxBuildBytes)
            if (strictMpp) {
              throw new IllegalStateException(
                s"MppReplicatedCartesianRule: unsafe replicated Cartesian: $reason. " +
                  s"Set $MAX_BUILD_BYTES_KEY only to a device-safe upper bound; " +
                  "large-large Cartesian requires the partition-pair implementation.")
            }
            logWarning(s"MppReplicatedCartesianRule: keeping Spark Cartesian bridge: $reason")
            cart
        }
    }
  }

  private def enabled: Boolean =
    SQLConf.get.getConfString(ENABLED_KEY, ENABLED_DEFAULT.toString).toBoolean

  private def strictMpp: Boolean =
    SQLConf.get.getConfString("spark.gluten.mpp.failOnFallback", "false").toBoolean

  private def maxBuildBytes: BigInt = {
    val raw = SQLConf.get.getConfString(MAX_BUILD_BYTES_KEY, MAX_BUILD_BYTES_DEFAULT.toString)
    val parsed =
      try {
        BigInt(JavaUtils.byteStringAsBytes(raw))
      } catch {
        case e: NumberFormatException =>
          throw new IllegalArgumentException(
            s"Invalid $MAX_BUILD_BYTES_KEY=$raw: expected a positive byte size",
            e)
      }
    require(parsed > 0, s"$MAX_BUILD_BYTES_KEY must be positive, found $raw")
    parsed
  }

  private def logRewrite(
      cart: CartesianProductExecTransformer,
      side: BuildSide,
      stats: Option[SideStats]): Unit = {
    val selected = stats.map(describe).getOrElse("unknown")
    logInfo(
      s"MppReplicatedCartesianRule: replacing ${cart.nodeName} with native replicated " +
        s"nested-loop join; buildSide=$side buildStats=$selected " +
        s"maxBuildBytes=$maxBuildBytes")
  }

  private def markReplicatedCartesian(join: VeloxBroadcastNestedLoopJoinExecTransformer)
      : VeloxBroadcastNestedLoopJoinExecTransformer = {
    // Carry the exact, already canonicalized byte count with this physical query. MPP forwards the
    // tag as an explicit uint64 query argument; native code must not re-parse the user's Spark byte
    // string (for example, "256m").
    require(maxBuildBytes.isValidLong, s"$MAX_BUILD_BYTES_KEY exceeds signed 64-bit byte range")
    join.setTagValue(REPLICATED_CARTESIAN_MAX_BUILD_BYTES_TAG, maxBuildBytes.longValue)
    join
  }
}

object MppReplicatedCartesianRule {
  val ENABLED_KEY = "spark.gluten.mpp.replicatedCartesian.enabled"
  val ENABLED_DEFAULT = true
  val MAX_BUILD_BYTES_KEY = "spark.gluten.mpp.replicatedCartesian.maxBuildBytes"
  val MAX_BUILD_BYTES_DEFAULT: Long = 256L << 20
  val REPLICATED_CARTESIAN_MAX_BUILD_BYTES_TAG: TreeNodeTag[Long] =
    TreeNodeTag[Long]("gluten.mpp.replicatedCartesian.maxBuildBytes")

  private[extension] case class SideStats(
      sizeInBytes: BigInt,
      rowCount: Option[BigInt],
      source: String)

  private[extension] def chooseBuildSide(
      left: Option[SideStats],
      right: Option[SideStats],
      maxBuildBytes: BigInt): Option[BuildSide] = {
    val leftSafe = left.exists(isSafe(_, maxBuildBytes))
    val rightSafe = right.exists(isSafe(_, maxBuildBytes))
    (leftSafe, rightSafe) match {
      case (true, true) =>
        // Prefer the right side on a tie because it preserves the original output ordering
        // without an additional projection.
        if (left.get.sizeInBytes < right.get.sizeInBytes) Some(BuildLeft) else Some(BuildRight)
      case (true, false) => Some(BuildLeft)
      case (false, true) => Some(BuildRight)
      case (false, false) => None
    }
  }

  private def sideStats(
      cart: CartesianProductExecTransformer,
      left: SparkPlan,
      right: SparkPlan): (Option[SideStats], Option[SideStats]) = {
    val fromLogicalJoin = cart.logicalLink.collect {
      case join: Join =>
        (
          fromStatistics(join.left.stats, "logical join left"),
          fromStatistics(join.right.stats, "logical join right"))
    }
    fromLogicalJoin.getOrElse((planStats(left), planStats(right)))
  }

  private def planStats(plan: SparkPlan): Option[SideStats] = {
    plan.logicalLink
      .flatMap(logical => fromStatistics(logical.stats, "physical subtree logical link"))
    // Do not inherit a child's statistics merely because this node is unary. Generate, projection,
    // and other unary operators can increase row count or row width, so the child's size is not an
    // upper bound for the complete physical subtree. A convention wrapper that wants to participate
    // must retain the logical link for the subtree it transparently represents.
  }

  private def fromStatistics(stats: Statistics, source: String): Option[SideStats] = {
    val candidate = SideStats(stats.sizeInBytes, stats.rowCount, source)
    if (isKnown(candidate)) Some(candidate) else None
  }

  private def isKnown(stats: SideStats): Boolean =
    (stats.sizeInBytes > 0 && stats.sizeInBytes < BigInt(Long.MaxValue)) ||
      (stats.sizeInBytes == 0 && stats.rowCount.contains(BigInt(0)))

  private def isSafe(stats: SideStats, maxBuildBytes: BigInt): Boolean =
    isKnown(stats) && stats.sizeInBytes <= maxBuildBytes

  private def rejectionReason(
      left: Option[SideStats],
      right: Option[SideStats],
      maxBuildBytes: BigInt): String = {
    s"neither complete input is stats-proven below maxBuildBytes=$maxBuildBytes " +
      s"(left=${left.map(describe).getOrElse("unknown")}, " +
      s"right=${right.map(describe).getOrElse("unknown")})"
  }

  private def describe(stats: SideStats): String =
    s"{bytes=${stats.sizeInBytes},rows=${stats.rowCount.getOrElse("unknown")}," +
      s"source=${stats.source}}"
}
