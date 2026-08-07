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
package org.apache.gluten.execution

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, Attribute, AttributeReference, Literal, Rand, SortOrder}
import org.apache.spark.sql.execution.{LeafExecNode, ProjectExec}
import org.apache.spark.sql.types.{MapType, StringType}

import org.scalatest.funsuite.AnyFunSuite

class FluxHashJoinInputSortRewriteSuite extends AnyFunSuite {
  private case class TestLeaf(override val output: Seq[Attribute]) extends LeafExecNode {
    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("planner-only test leaf")
  }

  private val key = AttributeReference("join_key", StringType)()
  private val payload =
    AttributeReference("map_payload", MapType(StringType, StringType, valueContainsNull = true))()
  private val leaf = TestLeaf(Seq(key, payload))

  test("a local sort is removed through deterministic native and Spark projects") {
    val sort = SortExecTransformer(
      Seq(SortOrder(key, Ascending)),
      global = false,
      leaf,
      testSpillFrequency = 0)
    val inner =
      ProjectExecTransformer(Seq(key, payload, Alias(Literal(1), "computed_sort_key")()), sort)
    val outer = ProjectExec(Seq(inner.output(0), inner.output(1)), inner)
    val expectedOutputExprIds = outer.output.map(_.exprId)

    val rewritten = FluxHashJoinInputSortRewrite.strip(outer)

    assert(rewritten.strippedSorts == 1)
    assert(rewritten.plan.output.map(_.exprId) == expectedOutputExprIds)
    assert(rewritten.plan.collect { case _: SortExecTransformer => 1 }.isEmpty)
    assert(rewritten.plan.collect { case _: ProjectExecTransformer => 1 }.size == 1)
    assert(rewritten.plan.collect { case _: ProjectExec => 1 }.size == 1)
  }

  test("a nondeterministic project is a fail-closed boundary") {
    val sort = SortExecTransformer(
      Seq(SortOrder(key, Ascending)),
      global = false,
      leaf,
      testSpillFrequency = 0)
    val project = ProjectExec(Seq(key, payload, Alias(Rand(17L), "random_value")()), sort)

    val rewritten = FluxHashJoinInputSortRewrite.strip(project)

    assert(rewritten.strippedSorts == 0)
    assert(rewritten.plan eq project)
    assert(rewritten.plan.collect { case _: SortExecTransformer => 1 }.size == 1)
  }

  test("a global sort is retained") {
    val sort = SortExecTransformer(
      Seq(SortOrder(key, Ascending)),
      global = true,
      leaf,
      testSpillFrequency = 0)

    val rewritten = FluxHashJoinInputSortRewrite.strip(sort)

    assert(rewritten.strippedSorts == 0)
    assert(rewritten.plan eq sort)
  }
}
