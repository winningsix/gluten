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

import org.apache.gluten.config.GlutenConfig
import org.apache.gluten.execution.{HashAggregateExecBaseTransformer, LocalTableScanExecTransformer, ProjectExecTransformer, RowToVeloxColumnarExec, SortExecTransformer, VeloxColumnarToRowExec}
import org.apache.gluten.extension.columnar.FallbackTags

import org.apache.spark.SparkFunSuite
import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, AttributeReference, Cast, EqualTo, If, Literal, Multiply, NamedExpression, NullsFirst, SortOrder}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Partial, Sum}
import org.apache.spark.sql.catalyst.plans.physical.HashPartitioning
import org.apache.spark.sql.execution.{ColumnarShuffleExchangeExec, ColumnarToRowExec, ProjectExec, SparkPlan}
import org.apache.spark.sql.execution.aggregate.{BaseAggregateExec, HashAggregateExec}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StringType}

import org.mockito.Mockito.{mock, when, withSettings}

class MppCollapseRuleRowBoundarySuite extends SparkFunSuite {

  private def nativeLeaf(): LocalTableScanExecTransformer =
    LocalTableScanExecTransformer(
      Seq(AttributeReference("a", IntegerType, nullable = true)()),
      Seq.empty)

  private def columnarExchange() = {
    val child = nativeLeaf()
    ColumnarShuffleExchangeExec(
      outputPartitioning = HashPartitioning(Seq(child.output.head), 4),
      child = child,
      projectOutputAttributes = child.output)
  }

  test("V2 write normalization strips Spark C2R directly over Gluten ColumnarExchange") {
    val exchange = columnarExchange()
    val adapter = ColumnarToRowExec(exchange)
    val result = MppCollapseRule.normalizeV2WriteExchangeAdapters(adapter)

    assert(result.strippedAdapters == 1)
    assert(result.plan eq exchange)
  }

  test("V2 write normalization strips Gluten C2R directly over Gluten ColumnarExchange") {
    val exchange = columnarExchange()
    val adapter = VeloxColumnarToRowExec(exchange)
    val result = MppCollapseRule.normalizeV2WriteExchangeAdapters(adapter)

    assert(result.strippedAdapters == 1)
    assert(result.plan eq exchange)
  }

  test("V2 write normalization finds only exact exchange adapters inside a larger query") {
    val exchange = columnarExchange()
    val adapter = ColumnarToRowExec(exchange)
    val parent = ProjectExecTransformer(adapter.output, adapter)
    val result = MppCollapseRule.normalizeV2WriteExchangeAdapters(parent)

    assert(result.strippedAdapters == 1)
    assert(result.plan.children.head eq exchange)
  }

  test("V2 write normalization strips its top-level C2R over a native leaf") {
    val child = nativeLeaf()
    val adapter = ColumnarToRowExec(child)
    val result = MppCollapseRule.normalizeV2WriteExchangeAdapters(adapter)

    assert(result.strippedAdapters == 1)
    assert(result.plan eq child)
  }

  test("V2 write normalization strips its top-level C2R over a columnar query") {
    val exchange = columnarExchange()
    val project = ProjectExecTransformer(exchange.output, exchange)
    val adapter = ColumnarToRowExec(project)
    val result = MppCollapseRule.normalizeV2WriteExchangeAdapters(adapter)

    assert(result.strippedAdapters == 1)
    assert(result.plan eq project)
  }

  test("V2 write normalization does not strip a nested C2R over a non-exchange operator") {
    val exchange = columnarExchange()
    val project = ProjectExecTransformer(exchange.output, exchange)
    val nestedAdapter = ColumnarToRowExec(project)
    val parent = ProjectExecTransformer(nestedAdapter.output, nestedAdapter)
    val result = MppCollapseRule.normalizeV2WriteExchangeAdapters(parent)

    assert(result.strippedAdapters == 0)
    assert(result.plan eq parent)
  }

  test("strict native validation rejects every row transition left after normalization") {
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))
    val child = nativeLeaf()

    assert(!rule.isFullyNativeSupported(ColumnarToRowExec(child)))
    assert(!rule.isFullyNativeSupported(VeloxColumnarToRowExec(child)))
    assert(!rule.isFullyNativeSupported(RowToVeloxColumnarExec(child)))
  }

  test("strict row-boundary diagnostic reports the row child and its fallback reason") {
    val child = nativeLeaf()
    val rowChild = ProjectExec(child.output, child)
    FallbackTags.add(rowChild, "unsupported expression nf_example")
    val boundary = RowToVeloxColumnarExec(rowChild)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val diagnostic = rule.describeExecutionBoundary(boundary, "row-to-native")

    assert(diagnostic.contains("RowToVeloxColumnarExec: row-to-native execution boundary"))
    assert(diagnostic.contains("child=ProjectExec"))
    assert(diagnostic.contains("firstFallback=ProjectExec: unsupported expression nf_example"))
  }

  test("strict normalization materializes computed sort keys and removes stale transitions") {
    val child = nativeLeaf()
    val rowInput = ColumnarToRowExec(child)
    val computedOrder =
      SortOrder(Cast(rowInput.output.head, LongType), Ascending, NullsFirst, Seq.empty)
    val rowSort = org.apache.spark.sql.execution
      .SortExec(Seq(computedOrder), global = false, rowInput, testSpillFrequency = 0)
    val stalePlan = RowToVeloxColumnarExec(rowSort)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val normalized = rule.normalizeMppNativeOperators(stalePlan)

    assert(normalized.find(_.isInstanceOf[RowToVeloxColumnarExec]).isEmpty)
    assert(normalized.find(_.isInstanceOf[ColumnarToRowExec]).isEmpty)
    val nativeSort = normalized
      .find(_.isInstanceOf[SortExecTransformer])
      .get
      .asInstanceOf[SortExecTransformer]
    assert(nativeSort.sortOrder.forall(_.child.isInstanceOf[AttributeReference]))
    assert(normalized.find(_.isInstanceOf[ProjectExecTransformer]).isDefined)
    assert(rule.isFullyNativeSupported(normalized))
  }

  test("strict normalization materializes aggregate grouping and function expressions") {
    val metric = AttributeReference("metric_name", StringType, nullable = true)()
    val value = AttributeReference("metric_value", DoubleType, nullable = true)()
    val child = LocalTableScanExecTransformer(Seq(metric, value), Seq.empty)
    val rowInput = ColumnarToRowExec(child)
    val grouping = Alias(Literal("Global"), "global")()
    val conditionalValue = If(
      EqualTo(metric, Literal("view_hours_1d")),
      Multiply(value, Literal(3600.0d)),
      Literal.create(null, DoubleType))
    val aggregate = AggregateExpression(
      Sum(conditionalValue),
      Partial,
      isDistinct = false,
      filter = None,
      resultId = NamedExpression.newExprId)
    val rowAggregate = HashAggregateExec(
      requiredChildDistributionExpressions = None,
      isStreaming = false,
      numShufflePartitions = None,
      groupingExpressions = Seq(grouping),
      aggregateExpressions = Seq(aggregate),
      aggregateAttributes = Seq(aggregate.resultAttribute),
      initialInputBufferOffset = 0,
      resultExpressions = Seq(grouping.toAttribute, aggregate.resultAttribute),
      child = rowInput
    )
    val stalePlan = RowToVeloxColumnarExec(rowAggregate)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val normalized = rule.normalizeMppNativeOperators(stalePlan)

    assert(normalized.find(_.isInstanceOf[RowToVeloxColumnarExec]).isEmpty)
    assert(normalized.find(_.isInstanceOf[ColumnarToRowExec]).isEmpty)
    val nativeAggregate = normalized
      .find(_.isInstanceOf[HashAggregateExecBaseTransformer])
      .get
      .asInstanceOf[HashAggregateExecBaseTransformer]
    assert(nativeAggregate.groupingExpressions.forall(_.isInstanceOf[AttributeReference]))
    assert(
      nativeAggregate.aggregateExpressions.forall(
        _.aggregateFunction.children.forall(_.isInstanceOf[AttributeReference])))
    assert(normalized.find(_.isInstanceOf[ProjectExecTransformer]).isDefined)
    assert(rule.isFullyNativeSupported(normalized))
  }

  test("bounded scalar result recognizes only a global aggregate without grouping") {
    def aggregatePlan(): (SparkPlan, BaseAggregateExec) = {
      val plan =
        mock(classOf[SparkPlan], withSettings().extraInterfaces(classOf[BaseAggregateExec]))
      (plan, plan.asInstanceOf[BaseAggregateExec])
    }

    val (scalarPlan, scalarAggregate) = aggregatePlan()
    when(scalarAggregate.groupingExpressions).thenReturn(Seq.empty)
    val (groupedPlan, groupedAggregate) = aggregatePlan()
    when(groupedAggregate.groupingExpressions)
      .thenReturn(Seq(AttributeReference("key", IntegerType, nullable = false)()))

    assert(MppCollapseRule.isBoundedScalarResult(scalarPlan))
    assert(!MppCollapseRule.isBoundedScalarResult(groupedPlan))
    assert(!MppCollapseRule.isBoundedScalarResult(nativeLeaf()))
  }
}
