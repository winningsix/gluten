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
import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, Attribute, AttributeReference, CurrentRow, Descending, Expression, Rank, RowFrame, RowNumber, SortOrder, SpecifiedWindowFrame, UnboundedPreceding, WindowExpression, WindowSpecDefinition}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Complete, Sum}
import org.apache.spark.sql.execution.{LeafExecNode, SparkPlan}
import org.apache.spark.sql.types.{IntegerType, LongType}

import org.scalatest.funsuite.AnyFunSuite

class MppWindowInputOrderingSuite extends AnyFunSuite {
  private val partition = AttributeReference("partition", LongType)()
  private val order = AttributeReference("order", LongType)()
  private val value = AttributeReference("value", IntegerType)()
  private val leaf = TestLeaf(Seq(partition, order, value))

  private case class TestLeaf(override val output: Seq[Attribute]) extends LeafExecNode {
    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("planner-only test leaf")
  }

  test("marks partitioned row_number and rank with a complete local sort") {
    Seq(RowNumber(), Rank(Seq(order))).foreach {
      function =>
        val window = makeWindow(
          Seq(function),
          Seq(partition),
          Seq(SortOrder(order, Ascending)),
          SpecifiedWindowFrame(RowFrame, UnboundedPreceding, CurrentRow))

        assertMarked(window)
    }
  }

  test("requires a partitioned Window and the exact direct local Sort") {
    val function = RowNumber()
    val orderSpec = Seq(SortOrder(order, Ascending))
    val frame = SpecifiedWindowFrame(RowFrame, UnboundedPreceding, CurrentRow)

    assertNotMarked(makeWindow(Seq(function), Seq.empty, orderSpec, frame))
    assertNotMarked(
      makeWindow(Seq(function), Seq(partition), orderSpec, frame, includeSort = false))
    assertNotMarked(makeWindow(Seq(function), Seq(partition), orderSpec, frame, globalSort = true))
    assertNotMarked(
      makeWindow(Seq(function), Seq(partition), orderSpec, frame, sortOrder = Some(orderSpec)))
    assertNotMarked(
      makeWindow(
        Seq(function),
        Seq(partition),
        orderSpec,
        frame,
        sortOrder = Some(
          Seq(
            SortOrder(partition, Ascending),
            SortOrder(order, Ascending),
            SortOrder(value, Ascending)))
      ))
    assertNotMarked(
      makeWindow(
        Seq(function),
        Seq(partition),
        orderSpec,
        frame,
        sortOrder = Some(Seq(SortOrder(partition, Ascending), SortOrder(order, Descending)))))
  }

  test("does not mark a mixed rank and aggregate Window") {
    val sum = AggregateExpression(Sum(value), Complete, isDistinct = false, filter = None)
    val window = makeWindow(
      Seq(RowNumber(), sum),
      Seq(partition),
      Seq(SortOrder(order, Ascending)),
      SpecifiedWindowFrame(RowFrame, UnboundedPreceding, CurrentRow))

    assertNotMarked(window)
  }

  private def makeWindow(
      functions: Seq[Expression],
      partitionSpec: Seq[Expression],
      orderSpec: Seq[SortOrder],
      frame: SpecifiedWindowFrame,
      sortOrder: Option[Seq[SortOrder]] = None,
      includeSort: Boolean = true,
      globalSort: Boolean = false): WindowExecTransformer = {
    val spec = WindowSpecDefinition(partitionSpec, orderSpec, frame)
    val expressions = functions.zipWithIndex.map {
      case (function, index) =>
        Alias(WindowExpression(function, spec), s"window_$index")()
    }
    val child: SparkPlan =
      if (includeSort) {
        val required =
          partitionSpec.map(SortOrder(_, Ascending)) ++ orderSpec
        SortExecTransformer(sortOrder.getOrElse(required), globalSort, leaf)
      } else {
        leaf
      }
    WindowExecTransformer(expressions, partitionSpec, orderSpec, child)
  }

  private def assertMarked(window: WindowExecTransformer): Unit = {
    val (rewritten, stats) = MppWindowInputOrdering(window)
    assert(rewritten eq window)
    assert(stats.markedWindows == 1)
    assert(WindowExecTransformer.inputsSorted(window))
  }

  private def assertNotMarked(window: WindowExecTransformer): Unit = {
    val (rewritten, stats) = MppWindowInputOrdering(window)
    assert(rewritten eq window)
    assert(stats.markedWindows == 0)
    assert(!WindowExecTransformer.inputsSorted(window))
  }
}
