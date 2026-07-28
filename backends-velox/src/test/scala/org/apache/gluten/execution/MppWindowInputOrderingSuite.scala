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
import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, Attribute, AttributeReference, CurrentRow, Descending, Expression, Literal, RangeFrame, Rank, RowFrame, RowNumber, SortOrder, SpecifiedWindowFrame, UnboundedFollowing, UnboundedPreceding, WindowExpression, WindowSpecDefinition}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Complete, Count, Sum}
import org.apache.spark.sql.execution.{LeafExecNode, SparkPlan}
import org.apache.spark.sql.types.{DecimalType, DoubleType, IntegerType, LongType}

import org.scalatest.funsuite.AnyFunSuite

class MppWindowInputOrderingSuite extends AnyFunSuite {
  private val partition = AttributeReference("partition", LongType)()
  private val order = AttributeReference("order", LongType)()
  private val secondOrder = AttributeReference("second_order", LongType)()
  private val value = AttributeReference("value", IntegerType)()
  private val doubleValue = AttributeReference("double_value", DoubleType)()
  private val decimalValue = AttributeReference("decimal_value", DecimalType(12, 2))()
  private val leaf =
    TestLeaf(Seq(partition, order, secondOrder, value, doubleValue, decimalValue))

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

  test("marks ordering preserved through a deterministic projection") {
    val projectedPartition = Alias(partition, "projected_partition")()
    val projectedOrder = Alias(order, "projected_order")()
    val sort = SortExecTransformer(
      Seq(SortOrder(partition, Ascending), SortOrder(order, Ascending)),
      global = false,
      leaf)
    val project = ProjectExecTransformer(Seq(projectedPartition, projectedOrder, value), sort)
    val frame = SpecifiedWindowFrame(RowFrame, UnboundedPreceding, CurrentRow)
    val spec = WindowSpecDefinition(
      Seq(projectedPartition.toAttribute),
      Seq(SortOrder(projectedOrder.toAttribute, Ascending)),
      frame)
    val window = WindowExecTransformer(
      Seq(Alias(WindowExpression(Rank(Seq(projectedOrder.toAttribute)), spec), "rank")()),
      spec.partitionSpec,
      spec.orderSpec,
      project)

    assertMarked(window)
  }

  test("resolves equivalent projected aliases in the Sort contract") {
    val actualOrder = Alias(order, "actual_order")()
    val requiredOrder = Alias(order, "required_order")()
    val preSortProject =
      ProjectExecTransformer(Seq(partition, actualOrder, requiredOrder, value), leaf)
    val sort = SortExecTransformer(
      Seq(SortOrder(partition, Ascending), SortOrder(actualOrder.toAttribute, Ascending)),
      global = false,
      preSortProject)
    val postSortProject =
      ProjectExecTransformer(Seq(partition, requiredOrder.toAttribute, value), sort)
    val frame = SpecifiedWindowFrame(RowFrame, UnboundedPreceding, CurrentRow)
    val spec = WindowSpecDefinition(
      Seq(partition),
      Seq(SortOrder(requiredOrder.toAttribute, Ascending)),
      frame)
    val window = WindowExecTransformer(
      Seq(Alias(WindowExpression(Rank(Seq(requiredOrder.toAttribute)), spec), "rank")()),
      spec.partitionSpec,
      spec.orderSpec,
      postSortProject)

    assertMarked(window)
  }

  test("uses the physical Window contract after projection rewrites") {
    val embeddedOrder = Alias(order, "embedded_order")()
    val physicalOrder = Alias(order, "physical_order")()
    val preSortProject =
      ProjectExecTransformer(Seq(partition, embeddedOrder, physicalOrder, value), leaf)
    val sort = SortExecTransformer(
      Seq(SortOrder(partition, Ascending), SortOrder(embeddedOrder.toAttribute, Ascending)),
      global = false,
      preSortProject)
    val postSortProject =
      ProjectExecTransformer(Seq(partition, physicalOrder.toAttribute, value), sort)
    val frame = SpecifiedWindowFrame(RowFrame, UnboundedPreceding, CurrentRow)
    val embeddedSpec = WindowSpecDefinition(
      Seq(partition),
      Seq(SortOrder(embeddedOrder.toAttribute, Ascending)),
      frame)
    val physicalOrderSpec = Seq(SortOrder(physicalOrder.toAttribute, Ascending))
    val window = WindowExecTransformer(
      Seq(Alias(WindowExpression(Rank(Seq(embeddedOrder.toAttribute)), embeddedSpec), "rank")()),
      Seq(partition),
      physicalOrderSpec,
      postSortProject)

    assertMarked(window)
  }

  test("marks full-partition COUNT of a non-null literal") {
    val count =
      AggregateExpression(Count(Seq(Literal(1))), Complete, isDistinct = false, filter = None)
    val window = makeWindow(
      Seq(count),
      Seq(partition),
      Seq.empty,
      SpecifiedWindowFrame(RowFrame, UnboundedPreceding, UnboundedFollowing))

    assertMarked(window)
  }

  test("marks supported multi-column peer-aware RANGE SUM") {
    Seq(value, doubleValue).foreach {
      input =>
        val sum = AggregateExpression(Sum(input), Complete, isDistinct = false, filter = None)
        val window = makeWindow(
          Seq(sum),
          Seq(partition),
          Seq(SortOrder(order, Ascending), SortOrder(secondOrder, Descending)),
          SpecifiedWindowFrame(RangeFrame, UnboundedPreceding, CurrentRow)
        )

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

  test("rejects unsupported COUNT and RANGE SUM contracts") {
    val countValue =
      AggregateExpression(Count(Seq(value)), Complete, isDistinct = false, filter = None)
    val countNull = AggregateExpression(
      Count(Seq(Literal.create(null, IntegerType))),
      Complete,
      isDistinct = false,
      filter = None)
    val fullRowFrame =
      SpecifiedWindowFrame(RowFrame, UnboundedPreceding, UnboundedFollowing)
    assertNotMarked(makeWindow(Seq(countValue), Seq(partition), Seq.empty, fullRowFrame))
    assertNotMarked(makeWindow(Seq(countNull), Seq(partition), Seq.empty, fullRowFrame))

    val decimalSum =
      AggregateExpression(Sum(decimalValue), Complete, isDistinct = false, filter = None)
    val integerSum = AggregateExpression(Sum(value), Complete, isDistinct = false, filter = None)
    val orderSpec =
      Seq(SortOrder(order, Ascending), SortOrder(secondOrder, Descending))
    val rangeFrame =
      SpecifiedWindowFrame(RangeFrame, UnboundedPreceding, CurrentRow)
    assertNotMarked(makeWindow(Seq(decimalSum), Seq(partition), orderSpec, rangeFrame))
    assertNotMarked(
      makeWindow(
        Seq(integerSum),
        Seq(partition),
        orderSpec,
        SpecifiedWindowFrame(RowFrame, UnboundedPreceding, CurrentRow)))
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
