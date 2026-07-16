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

import org.apache.gluten.execution.{RowToVeloxColumnarExec, VeloxColumnarToRowExec}

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, AttributeReference}
import org.apache.spark.sql.catalyst.plans.physical.HashPartitioning
import org.apache.spark.sql.execution.{ColumnarShuffleExchangeExec, ColumnarToRowExec, LeafExecNode, ProjectExec, RowToColumnarExec, SparkPlan}
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.vectorized.ColumnarBatch

class MppColumnarTransitionBridgeSuite extends SharedSparkSession {

  private case class ColumnarLeaf(output: Seq[Attribute]) extends LeafExecNode {
    override def supportsColumnar: Boolean = true
    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("planning-only test")
    override protected def doExecuteColumnar(): RDD[ColumnarBatch] =
      throw new UnsupportedOperationException("planning-only test")
  }

  private def leaf(): ColumnarLeaf =
    ColumnarLeaf(Seq(AttributeReference("a", IntegerType, nullable = true)()))

  test("bridges Gluten R2C over Spark C2R without executing either transition") {
    val child = leaf()
    val plan = RowToVeloxColumnarExec(ColumnarToRowExec(child))

    assert(MppColumnarTransitionBridge()(plan) eq child)
  }

  test("bridges Spark R2C over Gluten C2R without executing either transition") {
    val child = leaf()
    val plan = RowToColumnarExec(VeloxColumnarToRowExec(child))

    assert(MppColumnarTransitionBridge()(plan) eq child)
  }

  test("bridges same-family transition pairs") {
    val child = leaf()

    assert(MppColumnarTransitionBridge()(RowToColumnarExec(ColumnarToRowExec(child))) eq child)
    assert(
      MppColumnarTransitionBridge()(RowToVeloxColumnarExec(VeloxColumnarToRowExec(child))) eq
        child)
  }

  test("bridges an identity C2R below an MPP-absorbed columnar shuffle") {
    val child = leaf()
    val exchange = ColumnarShuffleExchangeExec(
      outputPartitioning = HashPartitioning(Seq(child.output.head), 4),
      child = ColumnarToRowExec(child),
      projectOutputAttributes = child.output)
    val rewritten = MppColumnarTransitionBridge()(exchange)

    assert(rewritten.isInstanceOf[ColumnarShuffleExchangeExec])
    assert(rewritten.children.head eq child)
    assert(rewritten.find(_.isInstanceOf[ColumnarToRowExec]).isEmpty)
  }

  test("does not bridge across a real row operator") {
    val child = leaf()
    val c2r = ColumnarToRowExec(child)
    val projected = ProjectExec(Seq(Alias(c2r.output.head, "renamed")()), c2r)
    val plan = RowToVeloxColumnarExec(projected)
    val rewritten = MppColumnarTransitionBridge()(plan)

    assert(rewritten.isInstanceOf[RowToVeloxColumnarExec])
    assert(rewritten.find(_.isInstanceOf[ProjectExec]).isDefined)
    assert(rewritten.find(_.isInstanceOf[ColumnarToRowExec]).isDefined)
  }

  test("does not treat an isolated transition as a columnar bridge") {
    val child = leaf()
    val c2r: SparkPlan = ColumnarToRowExec(child)
    val r2c: SparkPlan = RowToColumnarExec(ProjectExec(child.output, c2r))

    assert(MppColumnarTransitionBridge()(c2r).isInstanceOf[ColumnarToRowExec])
    assert(MppColumnarTransitionBridge()(r2c).isInstanceOf[RowToColumnarExec])
  }
}
