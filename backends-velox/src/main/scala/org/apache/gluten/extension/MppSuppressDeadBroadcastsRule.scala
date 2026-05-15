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

import org.apache.gluten.execution.MppNativeQueryExec

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarBroadcastExchangeExec, SparkPlan}

/**
 * Mark every ColumnarBroadcastExchangeExec inside an MppNativeQueryExec subtree as superseded by
 * the enclosing native fragment.
 *
 * Why: Plan C MPP single-task merge inlines BROADCAST-source plans directly into the consumer
 * fragment (MppJniWrapper "single-task merge BROADCAST, no LocalPartition wrap"), so the
 * driver-side ColumnarBroadcastExchangeExec.relationFuture collect is dead work. Without this rule
 * the framework's SparkPlan.prepare() still walks the entire child subtree and triggers
 * relationFuture, which materializes ColumnarBatchSerializeResult arrays in driver heap. Across
 * iterations the arrays accumulate (~5 GB / iter on Q14 SF1000 with autoBroadcastJoinThreshold=2GB)
 * and OOM the 20 GB driver heap by iter 3, then race against a Spark shutdown hook that resets
 * VeloxBackend's globalMemoryManager and crash with SIGSEGV at
 * gluten::defaultLeafVeloxMemoryPool.
 *
 * The marker pattern (transient var on the exchange) is used instead of replacing the node so that
 * Catalyst tree walks, schema introspection, and the BSP fallback path that does need
 * executeBroadcast are unaffected.
 *
 * Gated by spark.gluten.mpp.suppressDeadBroadcast (default: true). Set to false only to retain the
 * legacy driver-collect path for diagnostic comparison.
 */
case class MppSuppressDeadBroadcastsRule() extends Rule[SparkPlan] with Logging {
  private val confKey = "spark.gluten.mpp.suppressDeadBroadcast"
  private val confDefault = "true"

  override def apply(plan: SparkPlan): SparkPlan = {
    val sess = org.apache.spark.sql.SparkSession.getActiveSession
    val enabled = sess.exists(_.conf.get(confKey, confDefault).toBoolean)
    if (!enabled) {
      return plan
    }

    var markedCount = 0
    plan.foreach {
      case mpp: MppNativeQueryExec =>
        mpp.child.foreach {
          case bex: ColumnarBroadcastExchangeExec if !bex.isMppSuppressed =>
            bex.setTagValue(ColumnarBroadcastExchangeExec.MppSuppressedTag, true)
            markedCount += 1
          case _ =>
        }
      case _ =>
    }

    if (markedCount > 0) {
      logDebug(
        s"MppSuppressDeadBroadcastsRule: marked $markedCount ColumnarBroadcastExchangeExec " +
          s"node(s) as dead under MppNativeQueryExec; driver-side relationFuture collect " +
          s"will be skipped")
    }
    plan
  }
}
