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
import org.apache.gluten.exception.GlutenException
import org.apache.gluten.execution.{ColumnarUnionExec, FlushableHashAggregateExecTransformer, GenerateExecTransformer, HashAggregateExecBaseTransformer, LocalTableScanExecTransformer, MppExistingRddStreamInput, MppNativeQueryExec, ProjectExecTransformer, RegularHashAggregateExecTransformer, RowToVeloxColumnarExec, SortExecTransformer, UnionExecTransformer, VeloxColumnarToRowExec}
import org.apache.gluten.expression.aggregate.VeloxCollectList
import org.apache.gluten.extension.columnar.FallbackTags

import org.apache.spark.SparkFunSuite
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, Attribute, AttributeReference, Cast, CreateArray, CreateNamedStruct, EqualTo, Explode, If, IsNotNull, Literal, Multiply, NamedExpression, NullsFirst, SortOrder}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Partial, Sum}
import org.apache.spark.sql.catalyst.expressions.objects.CreateExternalRow
import org.apache.spark.sql.catalyst.plans.physical.{HashPartitioning, RangePartitioning, UnknownPartitioning}
import org.apache.spark.sql.execution.{ColumnarInputAdapter, ColumnarShuffleExchangeExec, ColumnarToRowExec, DeserializeToObjectExec, ExternalRDDScanExec, FilterExec, GenerateExec, InputAdapter, MapPartitionsExec, ProjectExec, RDDScanExec, SerializeFromObjectExec, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.aggregate.{BaseAggregateExec, HashAggregateExec, SortAggregateExec}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, ObjectType, StringType, StructField, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch

import org.mockito.Mockito.{mock, when, withSettings}

class MppCollapseRuleRowBoundarySuite extends SparkFunSuite {

  private case class NonNativePythonExec(child: SparkPlan) extends UnaryExecNode {
    override def output: Seq[Attribute] = child.output

    override def supportsColumnar: Boolean = true

    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("planning-only Python fixture")

    override protected def doExecuteColumnar(): RDD[ColumnarBatch] =
      throw new UnsupportedOperationException("planning-only Python fixture")

    override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan =
      copy(child = newChild)
  }

  private def nativeLeaf(): LocalTableScanExecTransformer =
    LocalTableScanExecTransformer(
      Seq(AttributeReference("a", IntegerType, nullable = true)()),
      Seq.empty)

  private def existingRddScan(attr: AttributeReference): RDDScanExec =
    RDDScanExec(Seq(attr), mock(classOf[RDD[InternalRow]]), "ExistingRDD")

  private def deserializeRows(child: SparkPlan): DeserializeToObjectExec = {
    val schema = StructType(
      child.output.map(attr => StructField(attr.name, attr.dataType, attr.nullable, attr.metadata)))
    val deserializer = CreateExternalRow(child.output, schema)
    val outputObject =
      AttributeReference("obj", ObjectType(classOf[Row]), nullable = false)()
    DeserializeToObjectExec(deserializer, outputObject, child)
  }

  private def assertStrictRejection(
      rule: MppCollapseRule,
      plan: SparkPlan,
      expectedMessageParts: String*): Unit = {
    val error = intercept[RuntimeException] {
      rule(plan)
    }
    assert(error.isInstanceOf[GlutenException] || error.isInstanceOf[IllegalStateException])
    expectedMessageParts.foreach(part => assert(error.getMessage.contains(part)))
  }

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

  test("terminal root Spark C2R is removed before MPP row egress") {
    val child = nativeLeaf()
    val boundary = ColumnarToRowExec(child)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(boundary)

    assert(collapsed.isInstanceOf[MppNativeQueryExec])
    assert(collapsed.output == child.output)
    assert(collapsed.find(_.isInstanceOf[ColumnarToRowExec]).isEmpty)
    // The strict native validator itself remains fail-closed for C2R. Only the root egress path
    // may recognize and safely remove this standard Spark adapter.
    assert(!rule.isFullyNativeSupported(boundary))
  }

  test("Gluten terminal root native-to-row output is also an intentional MPP egress") {
    val child = nativeLeaf()
    val boundary = VeloxColumnarToRowExec(child)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(boundary)

    assert(collapsed.isInstanceOf[VeloxColumnarToRowExec])
    assert(collapsed.children.head.isInstanceOf[MppNativeQueryExec])
    assert(!rule.isFullyNativeSupported(boundary))
  }

  test("root DeserializeToObject without a direct C2R remains rejected") {
    val child = nativeLeaf()
    val boundary = deserializeRows(child)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val result = rule(boundary)

    assert(result.find(_.isInstanceOf[MppNativeQueryExec]).isEmpty)
    assert(!rule.isFullyNativeSupported(boundary))
  }

  test("root DeserializeToObject removes Spark C2R before MPP row egress") {
    val child = nativeLeaf()
    val boundary = deserializeRows(ColumnarToRowExec(child))
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(boundary)

    assert(collapsed.isInstanceOf[DeserializeToObjectExec])
    assert(collapsed.children.head.isInstanceOf[MppNativeQueryExec])
    assert(collapsed.find(_.isInstanceOf[ColumnarToRowExec]).isEmpty)
    assert(collapsed.children.head.output == child.output)
  }

  test("root DeserializeToObject preserves its one direct Gluten C2R outside MPP") {
    val child = nativeLeaf()
    val boundary = deserializeRows(VeloxColumnarToRowExec(child))
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(boundary)

    assert(collapsed.isInstanceOf[DeserializeToObjectExec])
    val c2r = collapsed.children.head
    assert(c2r.isInstanceOf[VeloxColumnarToRowExec])
    assert(c2r.children.head.isInstanceOf[MppNativeQueryExec])
    assert(c2r.children.head.output == child.output)
  }

  test("root DeserializeToObject composes Spark C2R with the exact ExistingRDD ingress hybrid") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val ingress = RowToVeloxColumnarExec(scan)
    val nativeSuffix = ProjectExecTransformer(ingress.output, ingress)
    val boundary = deserializeRows(ColumnarToRowExec(nativeSuffix))
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(boundary)

    assert(collapsed.isInstanceOf[DeserializeToObjectExec])
    assert(collapsed.children.head.isInstanceOf[MppNativeQueryExec])
    assert(collapsed.find(_.isInstanceOf[ColumnarToRowExec]).isEmpty)
    assert(collapsed.find(node => MppExistingRddStreamInput.scan(node).contains(scan)).isDefined)
  }

  test("root DeserializeToObject composes Gluten C2R with the exact ExistingRDD ingress hybrid") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val ingress = RowToVeloxColumnarExec(scan)
    val nativeSuffix = ProjectExecTransformer(ingress.output, ingress)
    val boundary = deserializeRows(VeloxColumnarToRowExec(nativeSuffix))
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(boundary)

    assert(collapsed.isInstanceOf[DeserializeToObjectExec])
    val c2r = collapsed.children.head
    assert(c2r.isInstanceOf[VeloxColumnarToRowExec])
    assert(c2r.children.head.isInstanceOf[MppNativeQueryExec])
    assert(c2r.find(node => MppExistingRddStreamInput.scan(node).contains(scan)).isDefined)
  }

  test("root DeserializeToObject repairs a sortable row shell over ExistingRDD ingress") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val ingress = RowToVeloxColumnarExec(scan)
    val rowInput = VeloxColumnarToRowExec(ingress)
    val rowSort = org.apache.spark.sql.execution.SortExec(
      Seq(SortOrder(attr, Ascending, NullsFirst, Seq.empty)),
      global = false,
      rowInput,
      testSpillFrequency = 0)
    val stalePlan = RowToVeloxColumnarExec(rowSort)
    val boundary = deserializeRows(VeloxColumnarToRowExec(stalePlan))
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(boundary)

    val c2r = collapsed.children.head
    assert(c2r.isInstanceOf[VeloxColumnarToRowExec])
    val mpp = c2r.children.head
    assert(mpp.isInstanceOf[MppNativeQueryExec])
    assert(mpp.find(_.isInstanceOf[SortExecTransformer]).isDefined)
    assert(mpp.find(_.isInstanceOf[VeloxColumnarToRowExec]).isEmpty)
    assert(mpp.find(node => MppExistingRddStreamInput.scan(node).contains(scan)).isDefined)
  }

  test("nested DeserializeToObject remains a strict MPP rejection") {
    val objectBoundary = deserializeRows(ColumnarToRowExec(nativeLeaf()))
    val parent = ProjectExecTransformer(objectBoundary.output, objectBoundary)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assertStrictRejection(rule, parent, "No viable transition found", "DeserializeToObject")
  }

  test("root object operators other than DeserializeToObject remain rejected") {
    val child = nativeLeaf()
    val serialized = SerializeFromObjectExec(Seq(Alias(Literal(1), "value")()), child)
    val mappedObject = AttributeReference("mapped", ObjectType(classOf[Row]), nullable = false)()
    val mapped = MapPartitionsExec((rows: Iterator[Any]) => rows, mappedObject, child)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assertStrictRejection(rule, serialized, "No viable transition found", "SerializeFromObject")
    assertStrictRejection(rule, mapped, "No viable transition found", "MapPartitions")
  }

  test("root DeserializeToObject does not admit a Python producer") {
    val python = NonNativePythonExec(nativeLeaf())
    val boundary = deserializeRows(ColumnarToRowExec(python))
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assertStrictRejection(rule, boundary, "No viable transition found", "NonNativePython")
  }

  test("terminal root egress connects the exact ExistingRDD ingress hybrid to MPP") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val ingress = RowToVeloxColumnarExec(scan)
    val nativeSuffix = ProjectExecTransformer(ingress.output, ingress)
    val boundary = ColumnarToRowExec(nativeSuffix)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(boundary)

    assert(collapsed.isInstanceOf[MppNativeQueryExec])
    val mpp = collapsed.asInstanceOf[MppNativeQueryExec]
    assert(collapsed.find(_.isInstanceOf[ColumnarToRowExec]).isEmpty)
    assert(mpp.child.find(node => MppExistingRddStreamInput.scan(node).contains(scan)).isDefined)
  }

  test("an ingress-only ExistingRDD plan gets an identity native fragment anchor") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val ingress = RowToVeloxColumnarExec(scan)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(ingress)

    assert(collapsed.isInstanceOf[MppNativeQueryExec])
    val mpp = collapsed.asInstanceOf[MppNativeQueryExec]
    assert(mpp.child.isInstanceOf[ProjectExecTransformer])
    assert(mpp.child.output.map(_.exprId) == ingress.output.map(_.exprId))
    assert(mpp.child.find(node => MppExistingRddStreamInput.scan(node).contains(scan)).isDefined)
  }

  test("a nested native-to-row boundary cannot activate MPP") {
    val boundary = VeloxColumnarToRowExec(nativeLeaf())
    val parent = ProjectExecTransformer(boundary.output, boundary)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(rule.findUnpairedNestedRowOutput(parent).contains(boundary))

    val result = rule(parent)

    assert(result.find(_.isInstanceOf[MppNativeQueryExec]).isEmpty)
  }

  test("closed Generate row island is repaired without admitting its nested C2R") {
    val accountId = AttributeReference("account_id", LongType, nullable = true)()
    val id = AttributeReference("id", StringType, nullable = true)()
    val nestedJson = AttributeReference("sub_nested_json", StringType, nullable = true)()
    val nativeInput = LocalTableScanExecTransformer(Seq(accountId, id, nestedJson), Seq.empty)
    val nativeProject = ProjectExecTransformer(nativeInput.output, nativeInput)
    val boundary = VeloxColumnarToRowExec(nativeProject)
    val generatedJson = AttributeReference("consumption_json", StringType, nullable = true)()
    val generate = GenerateExec(
      Explode(CreateArray(Seq(nestedJson))),
      requiredChildOutput = Seq(accountId, id),
      outer = false,
      generatorOutput = Seq(generatedJson),
      child = boundary)
    val rowFilter = FilterExec(IsNotNull(generatedJson), generate)
    val closedIsland = RowToVeloxColumnarExec(rowFilter)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(rule.findUnpairedNestedRowOutput(closedIsland).contains(boundary))

    val collapsed = rule(closedIsland)

    assert(collapsed.isInstanceOf[MppNativeQueryExec])
    assert(collapsed.find(_.isInstanceOf[GenerateExecTransformer]).isDefined)
    assert(collapsed.find(_.isInstanceOf[ProjectExecTransformer]).isDefined)
    assert(collapsed.find(_.isInstanceOf[VeloxColumnarToRowExec]).isEmpty)
    assert(collapsed.find(_.isInstanceOf[RowToVeloxColumnarExec]).isEmpty)
  }

  test("Generate repair rejects an arbitrary row operator below Generate") {
    val accountId = AttributeReference("account_id", LongType, nullable = true)()
    val id = AttributeReference("id", StringType, nullable = true)()
    val nestedJson = AttributeReference("sub_nested_json", StringType, nullable = true)()
    val nativeInput = LocalTableScanExecTransformer(Seq(accountId, id, nestedJson), Seq.empty)
    val boundary = VeloxColumnarToRowExec(nativeInput)
    val arbitraryRowProject = ProjectExec(boundary.output, boundary)
    val generatedJson = AttributeReference("consumption_json", StringType, nullable = true)()
    val generate = GenerateExec(
      Explode(CreateArray(Seq(nestedJson))),
      requiredChildOutput = Seq(accountId, id),
      outer = false,
      generatorOutput = Seq(generatedJson),
      child = arbitraryRowProject)
    val closedButArbitrary = RowToVeloxColumnarExec(FilterExec(IsNotNull(generatedJson), generate))
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assertStrictRejection(
      rule,
      closedButArbitrary,
      "No viable transition found",
      "Generate explode")
  }

  test("unpaired row-output diagnostic preserves an adjacent round-trip adapter") {
    val boundary = VeloxColumnarToRowExec(nativeLeaf())
    val roundTrip = RowToVeloxColumnarExec(boundary)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(rule.findUnpairedNestedRowOutput(roundTrip).isEmpty)
  }

  test("unpaired row-output diagnostic reports the root-to-boundary child path") {
    val boundary = VeloxColumnarToRowExec(nativeLeaf())
    val rowProject = ProjectExec(boundary.output, boundary)
    val rowInput = RowToVeloxColumnarExec(rowProject)
    val root = ProjectExecTransformer(rowInput.output, rowInput)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(rule.findUnpairedNestedRowOutput(root).contains(boundary))
    assert(
      rule.describeAncestorPath(root, boundary) ==
        "root=ProjectExecTransformer -> child[0]=RowToVeloxColumnarExec -> " +
        "child[0]=ProjectExec -> child[0]=VeloxColumnarToRowExec")
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

  test("ExistingRDD hybrid captures the exact SerializeFromObject ExternalRDD ingress") {
    val objectAttr =
      AttributeReference("obj", ObjectType(classOf[Row]), nullable = true)()
    val external = ExternalRDDScanExec(objectAttr, mock(classOf[RDD[Row]]))
    val encodedValue = If(IsNotNull(objectAttr), Literal(1), Literal(0))
    val serializer = SerializeFromObjectExec(Seq(Alias(encodedValue, "a")()), external)
    val ingress = RowToVeloxColumnarExec(serializer)
    val nativeSuffix = ProjectExecTransformer(ingress.output, ingress)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(!rule.isFullyNativeSupported(nativeSuffix))
    assert(rule.isSupportedExistingRddHybridPlan(nativeSuffix))
    assert(MppExistingRddStreamInput.rowInput(ingress).contains(serializer))
    assert(
      MppExistingRddStreamInput
        .rowInput(ColumnarInputAdapter(ingress))
        .contains(serializer))

    val collapsed = rule(nativeSuffix)
    assert(collapsed.isInstanceOf[MppNativeQueryExec])
    assert(
      collapsed
        .find(node => MppExistingRddStreamInput.rowInput(node).contains(serializer))
        .isDefined)

    val guardOutcomes = MppExistingRddStreamInput.objectIngressGuardOutcomes(nativeSuffix)
    assert(guardOutcomes.size == 1)
    val guardOutcome = guardOutcomes.head
    assert(guardOutcome.contains("childShape=ExternalRDDScanExec"))
    assert(guardOutcome.contains("directExternalObjectScan=true"))
    assert(guardOutcome.contains("externalOutputArity=1"))
    assert(guardOutcome.contains("exactlyOneObjectInput=true"))
    assert(guardOutcome.contains("serializerReferencesWithinInput=true"))
    assert(guardOutcome.contains("outputContainsNoObjectType=true"))
    assert(guardOutcome.contains("accepted=true"))
  }

  test("ExistingRDD hybrid captures Spark's codegen InputAdapter object ingress") {
    val objectAttr =
      AttributeReference("obj", ObjectType(classOf[Row]), nullable = true)()
    val external = ExternalRDDScanExec(objectAttr, mock(classOf[RDD[Row]]))
    val encodedValue = If(IsNotNull(objectAttr), Literal(1), Literal(0))
    val serializer =
      SerializeFromObjectExec(Seq(Alias(encodedValue, "a")()), InputAdapter(external))
    val ingress = RowToVeloxColumnarExec(serializer)
    val nativeSuffix = ProjectExecTransformer(ingress.output, ingress)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(!rule.isFullyNativeSupported(nativeSuffix))
    assert(rule.isSupportedExistingRddHybridPlan(nativeSuffix))
    assert(MppExistingRddStreamInput.rowInput(ingress).contains(serializer))

    val collapsed = rule(nativeSuffix)
    assert(collapsed.isInstanceOf[MppNativeQueryExec])
    assert(
      collapsed
        .find(node => MppExistingRddStreamInput.rowInput(node).contains(serializer))
        .isDefined)
  }

  test("ExistingRDD object ingress rejects non-direct and non-ExternalRDD serializers") {
    val objectAttr =
      AttributeReference("obj", ObjectType(classOf[Row]), nullable = true)()
    val external = ExternalRDDScanExec(objectAttr, mock(classOf[RDD[Row]]))
    val nested = MapPartitionsExec((rows: Iterator[Any]) => rows, objectAttr, external)
    val nestedSerializer = SerializeFromObjectExec(Seq(Alias(Literal(1), "a")()), nested)
    val relationalScan = existingRddScan(AttributeReference("a", IntegerType, nullable = true)())
    val relationalSerializer =
      SerializeFromObjectExec(Seq(Alias(Literal(1), "a")()), relationalScan)
    val directExternalIngress = RowToVeloxColumnarExec(external)
    val objectSerializer =
      SerializeFromObjectExec(Seq(Alias(objectAttr, "still_object")()), external)
    val nestedObjectSerializer =
      SerializeFromObjectExec(Seq(Alias(CreateArray(Seq(objectAttr)), "objects")()), external)
    val foreignAttr = AttributeReference("foreign", IntegerType, nullable = true)()
    val foreignReferencedSerializer =
      SerializeFromObjectExec(Seq(Alias(foreignAttr, "foreign")()), external)
    val doubleInputAdapterSerializer =
      SerializeFromObjectExec(Seq(Alias(Literal(1), "a")()), InputAdapter(InputAdapter(external)))
    val adapterOverNestedSerializer =
      SerializeFromObjectExec(Seq(Alias(Literal(1), "a")()), InputAdapter(nested))
    val adapterOverRelationalSerializer =
      SerializeFromObjectExec(Seq(Alias(Literal(1), "a")()), InputAdapter(relationalScan))
    val columnarAdapterSerializer =
      SerializeFromObjectExec(Seq(Alias(Literal(1), "a")()), ColumnarInputAdapter(external))
    val adapterObjectSerializer =
      SerializeFromObjectExec(Seq(Alias(objectAttr, "still_object")()), InputAdapter(external))
    val adapterForeignReferencedSerializer =
      SerializeFromObjectExec(Seq(Alias(foreignAttr, "foreign")()), InputAdapter(external))
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(MppExistingRddStreamInput.rowInput(directExternalIngress).isEmpty)
    assert(
      !rule.isSupportedExistingRddHybridPlan(
        ProjectExecTransformer(directExternalIngress.output, directExternalIngress)))

    Seq(
      nestedSerializer,
      relationalSerializer,
      objectSerializer,
      nestedObjectSerializer,
      foreignReferencedSerializer,
      doubleInputAdapterSerializer,
      adapterOverNestedSerializer,
      adapterOverRelationalSerializer,
      columnarAdapterSerializer,
      adapterObjectSerializer,
      adapterForeignReferencedSerializer
    ).foreach {
      serializer =>
        val ingress = RowToVeloxColumnarExec(serializer)
        assert(MppExistingRddStreamInput.rowInput(ingress).isEmpty)
        assert(
          !rule.isSupportedExistingRddHybridPlan(ProjectExecTransformer(ingress.output, ingress)))
    }

    val adapterGuards =
      MppExistingRddStreamInput.objectIngressGuardOutcomes(doubleInputAdapterSerializer)
    assert(adapterGuards.size == 1)
    val adapterGuard = adapterGuards.head
    assert(adapterGuard.contains("childShape=InputAdapter->InputAdapter"))
    assert(adapterGuard.contains("directExternalObjectScan=false"))
    assert(adapterGuard.contains("accepted=false"))

    val referenceGuards =
      MppExistingRddStreamInput.objectIngressGuardOutcomes(foreignReferencedSerializer)
    assert(referenceGuards.size == 1)
    val referenceGuard = referenceGuards.head
    assert(referenceGuard.contains("childShape=ExternalRDDScanExec"))
    assert(referenceGuard.contains("serializerReferencesWithinInput=false"))
    assert(referenceGuard.contains("accepted=false"))
  }

  test("ExistingRDD hybrid validation absorbs a shuffle directly over the exact ingress") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val ingress = RowToVeloxColumnarExec(scan)
    val exchange = ColumnarShuffleExchangeExec(
      outputPartitioning =
        RangePartitioning(Seq(SortOrder(attr, Ascending, NullsFirst, Seq.empty)), 4),
      child = ingress,
      projectOutputAttributes = ingress.output)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(!rule.isFullyNativeSupported(exchange))
    assert(rule.isSupportedExistingRddHybridPlan(exchange))
    val collapsed = rule(exchange)
    assert(collapsed.isInstanceOf[MppNativeQueryExec])
    val nativePlan = collapsed.asInstanceOf[MppNativeQueryExec].child
    val anchoredExchange = nativePlan
      .find(_.isInstanceOf[ColumnarShuffleExchangeExec])
      .get
      .asInstanceOf[ColumnarShuffleExchangeExec]
    assert(anchoredExchange.child.isInstanceOf[ProjectExecTransformer])
    assert(anchoredExchange.child.output.map(_.exprId) == ingress.output.map(_.exprId))
    assert(nativePlan.find(node => MppExistingRddStreamInput.scan(node).contains(scan)).isDefined)
  }

  test("ExistingRDD hybrid repairs a computed collect-list aggregate before strict validation") {
    val subject = AttributeReference("subject", StringType, nullable = true)()
    val predicate = AttributeReference("predicate", StringType, nullable = true)()
    val value = AttributeReference("value", StringType, nullable = true)()
    val scan =
      RDDScanExec(Seq(subject, predicate, value), mock(classOf[RDD[InternalRow]]), "ExistingRDD")
    val ingress = RowToVeloxColumnarExec(scan)
    val rowInput = ColumnarToRowExec(ingress)
    val struct = CreateNamedStruct(
      Seq(Literal("subject"), subject, Literal("predicate"), predicate, Literal("value"), value))
    val aggregate = AggregateExpression(
      VeloxCollectList(struct),
      Partial,
      isDistinct = false,
      filter = None,
      resultId = NamedExpression.newExprId)
    val rowAggregate = SortAggregateExec(
      requiredChildDistributionExpressions = None,
      isStreaming = false,
      numShufflePartitions = None,
      groupingExpressions = Seq(subject),
      aggregateExpressions = Seq(aggregate),
      aggregateAttributes = Seq(aggregate.resultAttribute),
      initialInputBufferOffset = 0,
      resultExpressions = Seq(subject, aggregate.resultAttribute),
      child = rowInput
    )
    val stalePlan = RowToVeloxColumnarExec(rowAggregate)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(stalePlan)

    assert(collapsed.isInstanceOf[MppNativeQueryExec])
    val nativePlan = collapsed.asInstanceOf[MppNativeQueryExec].child
    val nativeAggregate = nativePlan
      .find(_.isInstanceOf[HashAggregateExecBaseTransformer])
      .get
      .asInstanceOf[HashAggregateExecBaseTransformer]
    assert(
      nativeAggregate.aggregateExpressions.forall(
        _.aggregateFunction.children.forall(_.isInstanceOf[AttributeReference])))
    assert(nativeAggregate.child.isInstanceOf[ProjectExecTransformer])
    assert(nativePlan.find(node => MppExistingRddStreamInput.scan(node).contains(scan)).isDefined)
    assert(nativePlan.find(_.isInstanceOf[ColumnarToRowExec]).isEmpty)
  }

  test("ExistingRDD hybrid bridges a native sort below a computed collect-list aggregate") {
    val subject = AttributeReference("subject", StringType, nullable = true)()
    val predicate = AttributeReference("predicate", StringType, nullable = true)()
    val value = AttributeReference("value", StringType, nullable = true)()
    val scan =
      RDDScanExec(Seq(subject, predicate, value), mock(classOf[RDD[InternalRow]]), "ExistingRDD")
    val ingress = RowToVeloxColumnarExec(scan)
    val nativeSort = SortExecTransformer(
      Seq(SortOrder(subject, Ascending, NullsFirst, Seq.empty)),
      global = false,
      ingress,
      testSpillFrequency = 0)
    val rowInput = VeloxColumnarToRowExec(nativeSort)
    val struct = CreateNamedStruct(
      Seq(Literal("subject"), subject, Literal("predicate"), predicate, Literal("value"), value))
    val aggregate = AggregateExpression(
      VeloxCollectList(struct),
      Partial,
      isDistinct = false,
      filter = None,
      resultId = NamedExpression.newExprId)
    val rowAggregate = SortAggregateExec(
      requiredChildDistributionExpressions = None,
      isStreaming = false,
      numShufflePartitions = None,
      groupingExpressions = Seq(subject),
      aggregateExpressions = Seq(aggregate),
      aggregateAttributes = Seq(aggregate.resultAttribute),
      initialInputBufferOffset = 0,
      resultExpressions = Seq(subject, aggregate.resultAttribute),
      child = rowInput
    )
    val stalePlan = RowToVeloxColumnarExec(rowAggregate)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(stalePlan)

    assert(collapsed.isInstanceOf[MppNativeQueryExec])
    val nativePlan = collapsed.asInstanceOf[MppNativeQueryExec].child
    val nativeAggregate = nativePlan
      .find(_.isInstanceOf[HashAggregateExecBaseTransformer])
      .get
      .asInstanceOf[HashAggregateExecBaseTransformer]
    assert(
      nativeAggregate.aggregateExpressions.forall(
        _.aggregateFunction.children.forall(_.isInstanceOf[AttributeReference])))
    assert(nativePlan.find(_.isInstanceOf[SortExecTransformer]).isDefined)
    assert(nativePlan.find(_.isInstanceOf[ColumnarToRowExec]).isEmpty)
    assert(nativePlan.find(_.isInstanceOf[VeloxColumnarToRowExec]).isEmpty)
    assert(nativePlan.find(node => MppExistingRddStreamInput.scan(node).contains(scan)).isDefined)
  }

  test("ExistingRDD hybrid materializes computed keys on an already-native sort") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val ingress = RowToVeloxColumnarExec(scan)
    val computedOrder = SortOrder(Cast(attr, LongType), Ascending, NullsFirst, Seq.empty)
    val nativeSort =
      SortExecTransformer(Seq(computedOrder), global = false, ingress, testSpillFrequency = 0)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(nativeSort)

    assert(collapsed.isInstanceOf[MppNativeQueryExec])
    val nativePlan = collapsed.asInstanceOf[MppNativeQueryExec].child
    val rewrittenSort = nativePlan
      .find(_.isInstanceOf[SortExecTransformer])
      .get
      .asInstanceOf[SortExecTransformer]
    assert(rewrittenSort.sortOrder.forall(_.child.isInstanceOf[AttributeReference]))
    assert(nativePlan.collect { case _: ProjectExecTransformer => 1 }.sum >= 2)
    assert(nativePlan.output == ingress.output)
    assert(nativePlan.find(node => MppExistingRddStreamInput.scan(node).contains(scan)).isDefined)
  }

  test("ExistingRDD hybrid normalizes a collect-list row shell across a native union") {
    val subject = AttributeReference("subject", StringType, nullable = true)()
    val predicate = AttributeReference("predicate", StringType, nullable = true)()
    val value = AttributeReference("value", StringType, nullable = true)()
    val scan =
      RDDScanExec(Seq(subject, predicate, value), mock(classOf[RDD[InternalRow]]), "ExistingRDD")
    val ingress = RowToVeloxColumnarExec(scan)
    val nativeSibling =
      LocalTableScanExecTransformer(Seq(subject, predicate, value), Seq.empty)
    val union = ColumnarUnionExec(Seq(ingress, nativeSibling), UnknownPartitioning(0))
    val nativeSort = SortExecTransformer(
      Seq(SortOrder(subject, Ascending, NullsFirst, Seq.empty)),
      global = false,
      union,
      testSpillFrequency = 0)
    val rowInput = VeloxColumnarToRowExec(nativeSort)
    val struct = CreateNamedStruct(
      Seq(Literal("subject"), subject, Literal("predicate"), predicate, Literal("value"), value))
    val aggregate = AggregateExpression(
      VeloxCollectList(struct),
      Partial,
      isDistinct = false,
      filter = None,
      resultId = NamedExpression.newExprId)
    val rowAggregate = SortAggregateExec(
      requiredChildDistributionExpressions = None,
      isStreaming = true,
      numShufflePartitions = None,
      groupingExpressions = Seq(subject),
      aggregateExpressions = Seq(aggregate),
      aggregateAttributes = Seq(aggregate.resultAttribute),
      initialInputBufferOffset = 0,
      resultExpressions = Seq(subject, aggregate.resultAttribute),
      child = rowInput
    )
    val stalePlan = RowToVeloxColumnarExec(rowAggregate)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val normalized = rule.normalizeMppNativeOperators(
      stalePlan,
      preserveExistingRddIngress = true,
      rewriteNativeUnion = _.transformUp {
        case union: ColumnarUnionExec => UnionExecTransformer(union.children)
      })

    assert(normalized.find(_.isInstanceOf[UnionExecTransformer]).isDefined)
    assert(normalized.find(_.isInstanceOf[ColumnarUnionExec]).isEmpty)
    assert(normalized.find(_.isInstanceOf[RowToVeloxColumnarExec]).size == 1)
    assert(normalized.find(_.isInstanceOf[VeloxColumnarToRowExec]).isEmpty)
    val nativeAggregate = normalized
      .find(_.isInstanceOf[HashAggregateExecBaseTransformer])
      .get
      .asInstanceOf[HashAggregateExecBaseTransformer]
    assert(
      nativeAggregate.aggregateExpressions.forall(
        _.aggregateFunction.children.forall(_.isInstanceOf[AttributeReference])))
    assert(nativeAggregate.child.isInstanceOf[ProjectExecTransformer])
    assert(normalized.find(node => MppExistingRddStreamInput.scan(node).contains(scan)).isDefined)
  }

  test("ExistingRDD matcher rejects non-Existing RDDScan names") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = RDDScanExec(Seq(attr), mock(classOf[RDD[InternalRow]]), "OneRowRelation")
    val ingress = RowToVeloxColumnarExec(scan)
    val nativeSuffix = ProjectExecTransformer(ingress.output, ingress)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(MppExistingRddStreamInput.scan(ingress).isEmpty)
    assert(!rule.isSupportedExistingRddHybridPlan(nativeSuffix))
  }

  test("ExistingRDD matcher rejects Spark 4 streaming RDDScan") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val batchScan = existingRddScan(attr)
    val spark4Constructor = classOf[RDDScanExec].getConstructors.find(_.getParameterCount == 6)

    spark4Constructor match {
      case Some(constructor) =>
        // Spark 4's sixth constructor argument is Option[SparkDataStream]. The element type is
        // erased, and the matcher only needs to prove that a non-empty stream is rejected.
        val streamingScan = constructor
          .newInstance(
            batchScan.output.asInstanceOf[AnyRef],
            batchScan.rdd,
            batchScan.name,
            batchScan.outputPartitioning,
            batchScan.outputOrdering.asInstanceOf[AnyRef],
            Some(new Object()).asInstanceOf[AnyRef]
          )
          .asInstanceOf[RDDScanExec]
        val ingress = RowToVeloxColumnarExec(streamingScan)
        val nativeSuffix = ProjectExecTransformer(ingress.output, ingress)
        val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

        assert(MppExistingRddStreamInput.scan(ingress).isEmpty)
        assert(!rule.isSupportedExistingRddHybridPlan(nativeSuffix))
      case None =>
        // Spark 3.x has no streaming RDDScanExec form; keep this cross-version suite meaningful
        // there by asserting that the accessor itself is absent.
        assert(!classOf[RDDScanExec].getMethods.exists(_.getName == "stream"))
    }
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

    val exchange = ColumnarShuffleExchangeExec(
      outputPartitioning = HashPartitioning(Seq(attr), 4),
      child = ingress,
      projectOutputAttributes = ingress.output)
    assert(!rule.isSupportedExistingRddHybridPlan(exchange))
  }

  test("ExistingRDD matcher rejects a native-to-row round trip") {
    val attr = AttributeReference("a", IntegerType, nullable = true)()
    val scan = existingRddScan(attr)
    val roundTrip = RowToVeloxColumnarExec(ColumnarToRowExec(RowToVeloxColumnarExec(scan)))
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    assert(MppExistingRddStreamInput.scan(roundTrip).isEmpty)
    assert(
      !rule.isSupportedExistingRddHybridPlan(ProjectExecTransformer(roundTrip.output, roundTrip)))
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

  test("strict normalization materializes computed keys on an already-native sort") {
    val child = nativeLeaf()
    val computedOrder =
      SortOrder(Cast(child.output.head, LongType), Ascending, NullsFirst, Seq.empty)
    val nativeSort =
      SortExecTransformer(Seq(computedOrder), global = false, child, testSpillFrequency = 0)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val normalized = rule.normalizeMppNativeOperators(nativeSort)

    val rewrittenSort = normalized
      .find(_.isInstanceOf[SortExecTransformer])
      .get
      .asInstanceOf[SortExecTransformer]
    assert(rewrittenSort.sortOrder.forall(_.child.isInstanceOf[AttributeReference]))
    assert(normalized.collect { case _: ProjectExecTransformer => 1 }.sum >= 2)
    assert(normalized.output == child.output)
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

  test("strict collapse normalizes a stale collect-list struct aggregate before row rejection") {
    val subject = AttributeReference("subject", StringType, nullable = true)()
    val predicate = AttributeReference("predicate", StringType, nullable = true)()
    val value = AttributeReference("value", StringType, nullable = true)()
    val child = LocalTableScanExecTransformer(Seq(subject, predicate, value), Seq.empty)
    val rowInput = ColumnarToRowExec(child)
    val struct = CreateNamedStruct(
      Seq(Literal("subject"), subject, Literal("predicate"), predicate, Literal("value"), value))
    val aggregate = AggregateExpression(
      VeloxCollectList(struct),
      Partial,
      isDistinct = false,
      filter = None,
      resultId = NamedExpression.newExprId)
    val rowAggregate = SortAggregateExec(
      requiredChildDistributionExpressions = None,
      isStreaming = false,
      numShufflePartitions = None,
      groupingExpressions = Seq(subject),
      aggregateExpressions = Seq(aggregate),
      aggregateAttributes = Seq(aggregate.resultAttribute),
      initialInputBufferOffset = 0,
      resultExpressions = Seq(subject, aggregate.resultAttribute),
      child = rowInput
    )
    val stalePlan = RowToVeloxColumnarExec(rowAggregate)
    val rule = MppCollapseRule(new GlutenConfig(SQLConf.get))

    val collapsed = rule(stalePlan)

    assert(collapsed.isInstanceOf[MppNativeQueryExec])
    val nativePlan = collapsed.asInstanceOf[MppNativeQueryExec].child
    assert(nativePlan.find(_.isInstanceOf[RowToVeloxColumnarExec]).isEmpty)
    assert(nativePlan.find(_.isInstanceOf[ColumnarToRowExec]).isEmpty)
    val nativeAggregate = nativePlan
      .find(_.isInstanceOf[HashAggregateExecBaseTransformer])
      .get
      .asInstanceOf[HashAggregateExecBaseTransformer]
    assert(
      nativeAggregate.aggregateExpressions.forall(
        _.aggregateFunction.children.forall(_.isInstanceOf[AttributeReference])))
    assert(nativeAggregate.child.isInstanceOf[ProjectExecTransformer])
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
