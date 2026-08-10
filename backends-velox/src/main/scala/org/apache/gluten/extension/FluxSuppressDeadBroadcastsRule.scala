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

import org.apache.gluten.execution.{FluxNativeQueryExec, FluxPreparedChildExec}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarBroadcastExchangeExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{BroadcastQueryStageExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.exchange.ReusedExchangeExec
import org.apache.spark.sql.internal.SQLConf

import java.util.{Collections, IdentityHashMap}

private[gluten] object FluxBroadcastLifecycle {

  /**
   * Mark every broadcast reachable from an FLUX child as prepare-deferred.
   *
   * This helper is deliberately shared by the final planning rule and FluxNativeQueryExec's
   * execution entry. The latter is required for plans introduced or rebuilt after the final rule
   * (for example the fact-probe broadcast created while Q15's cross-cut rules are applied).
   */
  def deferBroadcastPreparation(plan: SparkPlan): Int = {
    var marked = 0
    val visited = Collections.newSetFromMap(new IdentityHashMap[SparkPlan, java.lang.Boolean]())

    def visit(node: SparkPlan): Unit = {
      if (!visited.add(node)) {
        return
      }
      node match {
        case prepared: FluxPreparedChildExec => visit(prepared.hiddenPlan)
        case stage: BroadcastQueryStageExec => visit(stage.plan)
        case stage: ShuffleQueryStageExec => visit(stage.plan)
        case reused: ReusedExchangeExec => visit(reused.child)
        case broadcast: ColumnarBroadcastExchangeExec =>
          if (broadcast.deferPrepareForFluxNativeExecution()) {
            marked += 1
          }
          visit(broadcast.child)
        case other => other.children.foreach(visit)
      }
      // SparkPlan.prepare() also prepares physical subqueries referenced only from expressions.
      // They are not regular children and include Q22's otherwise-dead broadcast producer.
      node.subqueries.foreach(visit)
    }

    visit(plan)
    marked
  }
}

/**
 * Defer eager broadcast preparation until native FLUX execution is committed.
 *
 * Spark calls `prepare()` before `FluxNativeQueryExec` reaches its runtime validation point.
 * Leaving a broadcast untouched here starts its relation future eagerly, so a native FLUX query
 * runs an unused Spark collect in parallel and may send tens of GiB to the driver. Marking the
 * exchange as prepare-deferred prevents that eager launch without failing its relation promise.
 *
 * [[FluxNativeQueryExec]] still owns irreversible suppression. RANGE sampling and BSP fallback can
 * call `doExecuteBroadcast`, which starts a deferred exchange lazily. Once all fallback-capable
 * validation succeeds, native FLUX calls `suppressDeadBroadcastsForNativeFlux` and permanently
 * fails any dead relation promise.
 *
 * Keep this rule in the final-rule chain as an explicit lifecycle boundary and compatibility point
 * for deployments that already refer to its class name.
 */
case class FluxSuppressDeadBroadcastsRule() extends Rule[SparkPlan] with Logging {
  private val confKey = "spark.gluten.mpp.suppressDeadBroadcast"
  private val confDefault = "true"

  override def apply(plan: SparkPlan): SparkPlan = {
    val enabled = SQLConf.get.getConfString(confKey, confDefault).toBoolean
    if (!enabled) {
      return plan
    }

    var markedCount = 0
    plan.foreach {
      case flux: FluxNativeQueryExec =>
        markedCount += FluxBroadcastLifecycle.deferBroadcastPreparation(flux.child)
      case _ =>
    }

    if (markedCount > 0) {
      logDebug(
        s"FluxSuppressDeadBroadcastsRule: deferred eager prepare for $markedCount " +
          s"ColumnarBroadcastExchangeExec node(s); RANGE/BSP may still execute them lazily")
    }
    plan
  }
}
