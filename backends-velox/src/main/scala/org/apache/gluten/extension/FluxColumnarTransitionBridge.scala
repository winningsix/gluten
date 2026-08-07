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

import org.apache.gluten.execution.{ColumnarToRowExecBase, RowToColumnarExecBase}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarShuffleExchangeExecBase, ColumnarToRowExec, RowToColumnarExec, SparkPlan}

/**
 * Removes a redundant row round trip introduced only to reconcile Spark plan conventions.
 *
 * Spark can insert `RowToColumnar(ColumnarToRow(columnarChild))` around a shuffle or V2 write even
 * when FLUX absorbs that boundary and keeps the data columnar end to end. Executing those adapters
 * would materialize every native batch as JVM rows and immediately rebuild it as a native batch.
 * This rule bridges only an adjacent, schema-identical pair by returning `columnarChild` directly.
 * It also removes an identity C2R directly below a Gluten columnar shuffle because FLUX absorbs the
 * shuffle itself; neither node executes in that path. A real row operator between transitions, or
 * an isolated transition outside that absorbed boundary, is intentionally not matched.
 */
case class FluxColumnarTransitionBridge() extends Rule[SparkPlan] with Logging {

  override def apply(plan: SparkPlan): SparkPlan = plan.transformUp {
    case r2c: RowToColumnarExecBase =>
      r2c.children.headOption match {
        case Some(c2r: ColumnarToRowExec) => bridge(r2c, c2r, c2r.child)
        case Some(c2r: ColumnarToRowExecBase) =>
          bridge(r2c, c2r, c2r.children.head)
        case _ => r2c
      }

    case r2c: RowToColumnarExec =>
      r2c.child match {
        case c2r: ColumnarToRowExec => bridge(r2c, c2r, c2r.child)
        case c2r: ColumnarToRowExecBase =>
          bridge(r2c, c2r, c2r.children.head)
        case _ => r2c
      }

    case exchange: ColumnarShuffleExchangeExecBase =>
      exchange.child match {
        case c2r: ColumnarToRowExec =>
          bridgeExchangeInput(exchange, c2r, c2r.child)
        case c2r: ColumnarToRowExecBase =>
          bridgeExchangeInput(exchange, c2r, c2r.children.head)
        case _ => exchange
      }
  }

  private def bridge(r2c: SparkPlan, c2r: SparkPlan, columnarChild: SparkPlan): SparkPlan = {
    if (
      columnarChild.supportsColumnar &&
      sameOutput(r2c.output, c2r.output) &&
      sameOutput(c2r.output, columnarChild.output)
    ) {
      logInfo(
        s"FluxColumnarTransitionBridge: removed redundant " +
          s"${r2c.getClass.getSimpleName}(${c2r.getClass.getSimpleName}) pair; " +
          "preserving the original ColumnarBatch stream")
      columnarChild
    } else {
      r2c
    }
  }

  private def bridgeExchangeInput(
      exchange: ColumnarShuffleExchangeExecBase,
      c2r: SparkPlan,
      columnarChild: SparkPlan): SparkPlan = {
    if (
      exchange.supportsColumnar &&
      columnarChild.supportsColumnar &&
      sameOutput(c2r.output, columnarChild.output)
    ) {
      logInfo(
        s"FluxColumnarTransitionBridge: removed identity ${c2r.getClass.getSimpleName} " +
          s"below absorbed ${exchange.getClass.getSimpleName}; preserving the original " +
          "ColumnarBatch stream")
      exchange.withNewChildren(Seq(columnarChild))
    } else {
      exchange
    }
  }

  private def sameOutput(left: Seq[Attribute], right: Seq[Attribute]): Boolean =
    left.length == right.length && left.zip(right).forall {
      case (l, r) =>
        l.exprId == r.exprId && l.dataType == r.dataType && l.nullable == r.nullable
    }
}
