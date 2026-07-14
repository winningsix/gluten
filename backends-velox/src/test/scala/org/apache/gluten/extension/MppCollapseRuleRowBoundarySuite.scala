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
import org.apache.gluten.execution.{FlushableHashAggregateExecTransformer, HashAggregateExecBaseTransformer, LocalTableScanExecTransformer, MppExistingRddStreamInput, MppNativeQueryExec, ProjectExecTransformer, RegularHashAggregateExecTransformer, RowToVeloxColumnarExec, SortExecTransformer, VeloxColumnarToRowExec}
import org.apache.gluten.expression.aggregate.VeloxCollectList
import org.apache.gluten.extension.columnar.FallbackTags

import org.apache.spark.SparkFunSuite
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, AttributeReference, Cast, CreateNamedStruct, EqualTo, If, Literal, Multiply, NamedExpression, NullsFirst, SortOrder}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Partial, Sum}
import org.apache.spark.sql.catalyst.plans.physical.HashPartitioning
import org.apache.spark.sql.execution.{ColumnarInputAdapter, ColumnarShuffleExchangeExec, ColumnarToRowExec, ProjectExec, RDDScanExec, SparkPlan}
import org.apache.spark.sql.execution.aggregate.{BaseAggregateExec, HashAggregateExec}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StringType}

import org.mockito.Mockito.{mock, when, withSettings}

class MppCollapseRuleRowBoundarySuite extends SparkFunSuite {

  private def nativeLeaf(): LocalTableScanExecTransformer =
    LocalTableScanExecTransformer(
      Seq(AttributeReference("a", IntegerType, nullable = true)()),
      Seq.empty)

  private def existingRddScan(attr: AttributeReference): RDDScanExec =
    RDDScanExec(Seq(attr), mock(classOf[RDD[InternalRow]]), "ExistingRDD")

  private def columnarExchange() = {
    val child = nativeLeaf()
    ColumnarShuffleExchangeExec(
      outputPartitioning = HashPartitioning(Seq(child.output.head), 4),
      child = child,
      projectOutputAttributes = child.output)
  }

  private def collectStructAggregate(flushable: Boolean): HashAggregateExecBaseTransformer = {
    val subject = AttributeReference("subject", StringType, nullable = true)()
    val predicate = AttributeReference("predicate", StringType, nullable = true)()
    val value = AttributeReference("value", StringType, nullable = true)()
    val child = LocalTableScanExecTransformer(Seq(subject, predicate, value), Seq.empty)
    val struct = CreateNamedStruct(
      Seq(Literal("subject"), subject, Literal("predicate"), predicate, Literal("value"), value))
    val aggregate = AggregateExpression(
      VeloxCollectList(struct),
      Partial,
      isDistinct = false,
      filter = None,
      resultId = NamedExpression.newExprId)
    if (flushable) {
      FlushableHashAggregateExecTransformer(
        requiredChildDistributionExpressions = None,
        groupingExpressions = Seq(subject),
        aggregateExpressions = Seq(aggregate),
        aggregateAttributes = Seq(aggregate.resultAttribute),
        initialInputBufferOffset = 0,
        resultExpressions = Seq(subject, aggregate.resultAttribute),
        child = child
      )
    } else {
      RegularHashAggregateExecTransformer(
        requiredChildDistributionExpressions = None,
        groupingExpressions = Seq(subject),
        aggregateExpressions = Seq(aggregate),
        aggregateAttributes = Seq(aggregate.resultAttribute),
        initialInputBufferOffset = 0,
        resultExpressions = Seq(subject, aggregate.resultAttribute),
        child = child
      )
    }
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

  test("root native-to-row output is an intentional MPP egress, not a native row island") {
    val child = nativeLeaf()
    val boundary = ColumnarToRowExec(child)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(boundary)

    assert(collapsed.isInstanceOf[ColumnarToRowExec])
    assert(collapsed.children.head.isInstanceOf[MppNativeQueryExec])
    assert(collapsed.children.head.asInstanceOf[MppNativeQueryExec].output == child.output)
    // The strict native validator itself remains fail-closed for C2R. Only the root egress path
    // may retain this adapter outside MppNativeQueryExec.
    assert(!rule.isFullyNativeSupported(boundary))
  }

  test("Gluten root native-to-row output is also an intentional MPP egress") {
    val child = nativeLeaf()
    val boundary = VeloxColumnarToRowExec(child)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(boundary)

    assert(collapsed.isInstanceOf[VeloxColumnarToRowExec])
    assert(collapsed.children.head.isInstanceOf[MppNativeQueryExec])
    assert(!rule.isFullyNativeSupported(boundary))
  }

  test("root RDD egress connects the exact ExistingRDD ingress hybrid to MPP") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val ingress = RowToVeloxColumnarExec(scan)
    val nativeSuffix = ProjectExecTransformer(ingress.output, ingress)
    val boundary = ColumnarToRowExec(nativeSuffix)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(boundary)

    assert(collapsed.isInstanceOf[ColumnarToRowExec])
    val mpp = collapsed.children.head.asInstanceOf[MppNativeQueryExec]
    assert(
      mpp.child.find(node => MppExistingRddStreamInput.scan(node).contains(scan)).isDefined)
  }

  test("a nested native-to-row boundary cannot activate MPP") {
    val boundary = VeloxColumnarToRowExec(nativeLeaf())
    val parent = ProjectExecTransformer(boundary.output, boundary)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val result = rule(parent)

    assert(result.find(_.isInstanceOf[MppNativeQueryExec]).isEmpty)
  }

  test("ExistingRDD hybrid validation accepts only an exact RDD ingress below a native suffix") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val ingress = RowToVeloxColumnarExec(scan)
    val nativeSuffix = ProjectExecTransformer(ingress.output, ingress)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(!rule.isFullyNativeSupported(nativeSuffix))
    assert(rule.isSupportedExistingRddHybridPlan(nativeSuffix))
    assert(MppExistingRddStreamInput.scan(ingress).contains(scan))
    assert(
      MppExistingRddStreamInput
        .scan(ColumnarInputAdapter(ingress))
        .contains(scan))
  }

  test("ExistingRDD hybrid validation rejects row operators hidden below R2C") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val rowProject = ProjectExec(Seq(attr), scan)
    val ingress = RowToVeloxColumnarExec(rowProject)
    val nativeSuffix = ProjectExecTransformer(ingress.output, ingress)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(!rule.isSupportedExistingRddHybridPlan(nativeSuffix))
    assert(MppExistingRddStreamInput.scan(ingress).isEmpty)
  }

  test("ExistingRDD matcher rejects a native-to-row round trip") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val roundTrip = RowToVeloxColumnarExec(ColumnarToRowExec(RowToVeloxColumnarExec(scan)))
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(MppExistingRddStreamInput.scan(roundTrip).isEmpty)
    assert(
      !rule.isSupportedExistingRddHybridPlan(
        ProjectExecTransformer(roundTrip.output, roundTrip)))
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

  Seq(false, true).foreach {
    flushable =>
      val implementation = if (flushable) "flushable" else "regular"
      test(s"strict normalization pre-projects an already native $implementation aggregate") {
        val original = collectStructAggregate(flushable)
        val originalOutputIds = original.output.map(_.exprId)
        val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

        val normalized = rule.normalizeMppNativeOperators(original)
        val rebuilt = normalized
          .find(_.isInstanceOf[HashAggregateExecBaseTransformer])
          .get
          .asInstanceOf[HashAggregateExecBaseTransformer]

        assert(rebuilt.output.map(_.exprId) == originalOutputIds)
        assert(
          rebuilt.aggregateExpressions.forall(
            _.aggregateFunction.children.forall(_.isInstanceOf[AttributeReference])))
        assert(rebuilt.child.isInstanceOf[ProjectExecTransformer])
        val preProject = rebuilt.child.asInstanceOf[ProjectExecTransformer]
        assert(preProject.projectList.exists(_.exists(_.isInstanceOf[CreateNamedStruct])))
        assert(
          rebuilt.aggregateExpressions.map(_.mode) == original.aggregateExpressions.map(_.mode))
        if (flushable) {
          assert(rebuilt.isInstanceOf[FlushableHashAggregateExecTransformer])
        } else {
          assert(rebuilt.isInstanceOf[RegularHashAggregateExecTransformer])
        }
        assert(rule.isFullyNativeSupported(normalized))
      }
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
