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

/**
 * Keep a large fact table on the PROBE side of an inner join by forcing the small dimension side to
 * broadcast (build). This fixes the selection UPSTREAM of EnsureRequirements -- it only chooses the
 * right side; EnsureRequirements then places the BroadcastExchange on the (small) build side.
 *
 * Background: with missing column stats the optimizer's join-cardinality estimates explode, and a
 * runtime bloom filter can deflate a fact-table scan estimate below
 * `spark.sql.autoBroadcastJoinThreshold`. Spark's JoinSelection can then broadcast the large input,
 * which the native MPP runtime replicates per driver and may OOM. A broadcast hint on the smaller
 * input overrides the size-based choice so the large side streams as probe.
 *
 * The REAL on-disk leaf scan size (sum of leaf relation sizeInBytes) is used only to choose which
 * side is the fact/probe side. Spark plan stats (`plan.stats.sizeInBytes`) are used for the
 * auto-broadcast threshold check, matching Spark JoinSelection behavior after filters and
 * projections shrink an intermediate.
 */
case class MppFactProbeBroadcastHint(spark: SparkSession) extends Rule[LogicalPlan] with Logging {

  private val singleTaskModeKey = "spark.gluten.sql.columnar.backend.velox.mpp.singleTaskMode"

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
        // stats for the threshold check so filtered/projected intermediates follow
        // the same auto-broadcast contract as Spark JoinSelection.
        if (
          leftScanBytes > 0 && rightScanBytes > 0 && leftScanBytes < rightScanBytes &&
          leftStatsBytes > 0 && leftStatsBytes <= maxBroadcastBytes
        ) {
          logWarning(
            s"MppFactProbeBroadcastHint: broadcasting smaller LEFT side " +
              s"(leftScanBytes=$leftScanBytes rightScanBytes=$rightScanBytes " +
              s"leftStatsBytes=$leftStatsBytes rightStatsBytes=$rightStatsBytes " +
              s"autoBroadcastJoinThreshold=$maxBroadcastBytes); larger side stays probe")
          join.copy(hint = JoinHint(Some(broadcastHint), None))
        } else if (
          leftScanBytes > 0 && rightScanBytes > 0 && rightScanBytes < leftScanBytes &&
          rightStatsBytes > 0 && rightStatsBytes <= maxBroadcastBytes
        ) {
          logWarning(
            s"MppFactProbeBroadcastHint: broadcasting smaller RIGHT side " +
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

  private def singleTaskModeEnabled: Boolean =
    spark.sessionState.conf.getConfString(singleTaskModeKey, "false").toBoolean

}
