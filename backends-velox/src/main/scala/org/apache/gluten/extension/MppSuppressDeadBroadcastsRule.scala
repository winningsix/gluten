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
import org.apache.spark.sql.execution.SparkPlan

/**
 * Defer dead-broadcast suppression until native MPP execution is committed.
 *
 * This is intentionally a non-mutating final rule. RANGE preparation executes a bounded Spark
 * sample of the exchange producer and that producer may contain broadcast exchanges. Suppression is
 * irreversible: it fails the relation promise and may cancel an already-started broadcast job. A
 * final planning rule cannot reliably decide that an exchange is dead because the same exchange can
 * be reached through reused, shared, adaptive, or subquery plan topology.
 *
 * [[MppNativeQueryExec]] owns the commit point instead. It first completes RANGE preparation,
 * Substrait generation, and every validation path that may still delegate to BSP; only then does it
 * call `suppressDeadBroadcastsForNativeMpp`. Thus BSP fallback and RANGE sampling retain live
 * broadcasts, while committed native MPP still avoids the dead driver-side collect.
 *
 * Keep this rule in the final-rule chain as an explicit lifecycle boundary and compatibility point
 * for deployments that already refer to its class name.
 */
case class MppSuppressDeadBroadcastsRule() extends Rule[SparkPlan] with Logging {
  override def apply(plan: SparkPlan): SparkPlan = {
    if (plan.exists(_.isInstanceOf[MppNativeQueryExec])) {
      logDebug(
        "MppSuppressDeadBroadcastsRule: deferring irreversible broadcast suppression " +
          "until MppNativeQueryExec completes fallback-capable validation")
    }
    plan
  }
}
