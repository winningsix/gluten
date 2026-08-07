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

import org.apache.gluten.extension.NativeFragment

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference}
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight, BuildSide}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftOuter, RightOuter}
import org.apache.spark.sql.catalyst.plans.logical.{Join, JoinHint, LeafNode, Statistics}
import org.apache.spark.sql.connector.catalog.{Table => CatalogTable}
import org.apache.spark.sql.connector.read.{Statistics => SourceStatistics, SupportsReportStatistics}
import org.apache.spark.sql.execution.{LeafExecNode, ProjectExec, SparkPlan}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{LongType, StructType}

import org.mockito.Mockito.mock
import org.scalatest.funsuite.AnyFunSuite

import java.util.OptionalLong

class FluxOuterJoinBuildSideSuite extends AnyFunSuite {
  private val NormalizeJoinKey = "spark.gluten.mpp.normalizeJoinBuildSide"
  private val NormalizeOuterJoinKey = "spark.gluten.mpp.normalizeOuterJoinBuildSide"
  private val ForceOuterJoinKey = "spark.gluten.mpp.forceOuterJoinPreservedBuildSide"
  private val SmallScanBytesKey = "spark.gluten.mpp.outerJoinSmallScanMaxEstimatedBytes"
  private val SmallScanRowsKey = "spark.gluten.mpp.outerJoinSmallScanMaxEstimatedRows"

  test("outer join defaults to a clearly smaller trusted build side") {
    withNormalizationConfig(outerJoinValue = "null") {
      val left = statsLeaf("left_key", sizeInBytes = 160, rowCount = Some(BigInt(10)))
      val right = statsLeaf("right_key", sizeInBytes = 16000, rowCount = Some(BigInt(1000)))
      val normalized = normalize(leftOuterJoin(left, right, BuildRight))

      assert(normalized.buildSide == BuildLeft)
      assert(normalized.buildPlan eq left)
      assert(normalized.streamedPlan eq right)
      assert(normalized.output.map(_.exprId) == (left.output ++ right.output).map(_.exprId))
    }
  }

  test("outer join normalization can be explicitly disabled") {
    withNormalizationConfig(outerJoinValue = "false") {
      val left = statsLeaf("left_key", sizeInBytes = 160, rowCount = Some(BigInt(10)))
      val right = statsLeaf("right_key", sizeInBytes = 16000, rowCount = Some(BigInt(1000)))

      assert(normalize(leftOuterJoin(left, right, BuildRight)).buildSide == BuildRight)
    }
  }

  test("unknown derived statistics do not override Spark's outer build side") {
    withNormalizationConfig(outerJoinValue = "null") {
      val derivedLeft =
        derivedJoin("derived_left", BigInt("46020280547441875"), rowCount = None)
      val right = statsLeaf("right_key", BigInt("977369399721542706"), rowCount = None)

      assert(normalize(leftOuterJoin(derivedLeft, right, BuildRight)).buildSide == BuildRight)
    }
  }

  test("a small preserved V2 scan can replace an untrusted derived outer build") {
    withNormalizationConfig(outerJoinValue = "null") {
      val leftScan = reportingV2Scan("small_left_key", sourceRows = Some(85670L))
      val smallLeft = ProjectExec(Seq(leftScan.output.head), leftScan)
      val derivedRight =
        derivedJoin("derived_right", BigInt("104029216484"), rowCount = None)
      val join = withLogicalJoinStats(
        leftOuterJoin(smallLeft, derivedRight, BuildRight),
        leftSize = BigInt("16105960"),
        rightSize = BigInt("104029216484"))

      assert(normalize(join).buildSide == BuildLeft)
    }
  }

  test("small preserved scan selection is symmetric for right outer joins") {
    withNormalizationConfig(outerJoinValue = "null") {
      val derivedLeft =
        derivedJoin("derived_left", BigInt("104029216484"), rowCount = None)
      val rightScan = reportingV2Scan("small_right_key", sourceRows = Some(85670L))
      val smallRight = ProjectExec(Seq(rightScan.output.head), rightScan)
      val join = withLogicalJoinStats(
        rightOuterJoin(derivedLeft, smallRight, BuildLeft),
        leftSize = BigInt("104029216484"),
        rightSize = BigInt("16105960"))

      assert(normalize(join).buildSide == BuildRight)
    }
  }

  test("small preserved scan selection enforces row evidence and capacity bounds") {
    withNormalizationConfig(
      outerJoinValue = "null",
      smallScanBytes = "16m",
      smallScanRows = "1000") {
      val derivedRight =
        derivedJoin("derived_right", BigInt("104029216484"), rowCount = None)

      val noRows = withLogicalJoinStats(
        leftOuterJoin(reportingV2Scan("no_rows", None), derivedRight, BuildRight),
        leftSize = BigInt("16105960"),
        rightSize = BigInt("104029216484"))
      assert(normalize(noRows).buildSide == BuildRight)

      val tooManyRows = withLogicalJoinStats(
        leftOuterJoin(reportingV2Scan("too_many_rows", Some(1001L)), derivedRight, BuildRight),
        leftSize = BigInt("16105960"),
        rightSize = BigInt("104029216484"))
      assert(normalize(tooManyRows).buildSide == BuildRight)

      val tooWide = withLogicalJoinStats(
        leftOuterJoin(reportingV2Scan("too_wide", Some(10L)), derivedRight, BuildRight),
        leftSize = BigInt("16777217"),
        rightSize = BigInt("104029216484"))
      assert(normalize(tooWide).buildSide == BuildRight)
    }
  }

  private def normalize(join: ShuffledHashJoinExecTransformer): ShuffledHashJoinExecTransformer = {
    FluxNativeQueryExec(join, Seq.empty[NativeFragment], Seq.empty)
      .normalizeFluxJoinBuildSide(join)
      .asInstanceOf[ShuffledHashJoinExecTransformer]
  }

  private def leftOuterJoin(
      left: SparkPlan,
      right: SparkPlan,
      buildSide: BuildSide): ShuffledHashJoinExecTransformer = {
    ShuffledHashJoinExecTransformer(
      leftKeys = Seq(left.output.head),
      rightKeys = Seq(right.output.head),
      joinType = LeftOuter,
      buildSide = buildSide,
      condition = None,
      left = left,
      right = right,
      isSkewJoin = false
    )
  }

  private def rightOuterJoin(
      left: SparkPlan,
      right: SparkPlan,
      buildSide: BuildSide): ShuffledHashJoinExecTransformer = {
    ShuffledHashJoinExecTransformer(
      leftKeys = Seq(left.output.head),
      rightKeys = Seq(right.output.head),
      joinType = RightOuter,
      buildSide = buildSide,
      condition = None,
      left = left,
      right = right,
      isSkewJoin = false
    )
  }

  private def statsLeaf(name: String, sizeInBytes: BigInt, rowCount: Option[BigInt]): TestLeaf = {
    val attribute = AttributeReference(name, LongType, nullable = false)()
    val scan = TestLeaf(Seq(attribute))
    scan.setLogicalLink(StatsLeaf(Seq(attribute), sizeInBytes, rowCount))
    scan
  }

  private def reportingV2Scan(name: String, sourceRows: Option[Long]): BatchScanExecTransformer = {
    val attribute = AttributeReference(name, LongType, nullable = false)()
    BatchScanExecTransformer(
      output = Seq(attribute),
      scan = new TestReportingScan(new StructType().add(name, LongType, false), sourceRows),
      runtimeFilters = Seq.empty,
      table = mock(classOf[CatalogTable])
    )
  }

  private def derivedJoin(
      name: String,
      sizeInBytes: BigInt,
      rowCount: Option[BigInt]): ShuffledHashJoinExecTransformer = {
    val left = statsLeaf(s"${name}_left_key", 128, Some(BigInt(8)))
    val right = statsLeaf(s"${name}_right_key", 128, Some(BigInt(8)))
    val join = ShuffledHashJoinExecTransformer(
      leftKeys = Seq(left.output.head),
      rightKeys = Seq(right.output.head),
      joinType = Inner,
      buildSide = BuildRight,
      condition = None,
      left = left,
      right = right,
      isSkewJoin = false
    )
    join.setLogicalLink(StatsLeaf(join.output, sizeInBytes, rowCount))
    join
  }

  private def withLogicalJoinStats(
      join: ShuffledHashJoinExecTransformer,
      leftSize: BigInt,
      rightSize: BigInt): ShuffledHashJoinExecTransformer = {
    join.setLogicalLink(
      Join(
        StatsLeaf(join.left.output, leftSize, rowCount = None),
        StatsLeaf(join.right.output, rightSize, rowCount = None),
        join.joinType,
        condition = None,
        JoinHint.NONE))
    join
  }

  private def withNormalizationConfig[T](
      outerJoinValue: String,
      smallScanBytes: String = "64m",
      smallScanRows: String = "1000000")(body: => T): T = {
    val conf = SQLConf.get
    val values = Seq(
      NormalizeJoinKey -> "null",
      NormalizeOuterJoinKey -> outerJoinValue,
      ForceOuterJoinKey -> "false",
      SmallScanBytesKey -> smallScanBytes,
      SmallScanRowsKey -> smallScanRows
    )
    val previous = values.map { case (key, _) => key -> conf.getAllConfs.get(key) }
    values.foreach { case (key, value) => conf.setConfString(key, value) }
    try {
      body
    } finally {
      previous.foreach {
        case (key, Some(value)) => conf.setConfString(key, value)
        case (key, None) => conf.unsetConf(key)
      }
    }
  }

  private class TestReportingScan(schema: StructType, rows: Option[Long])
    extends SupportsReportStatistics {
    override def readSchema(): StructType = schema

    override def estimateStatistics(): SourceStatistics = new SourceStatistics {
      override def sizeInBytes(): OptionalLong = OptionalLong.empty()
      override def numRows(): OptionalLong = rows match {
        case Some(value) => OptionalLong.of(value)
        case None => OptionalLong.empty()
      }
    }
  }

  private case class StatsLeaf(
      attributes: Seq[Attribute],
      sizeInBytes: BigInt,
      rowCount: Option[BigInt])
    extends LeafNode {
    override def output: Seq[Attribute] = attributes
    override def computeStats(): Statistics = Statistics(sizeInBytes, rowCount)
  }

  private case class TestLeaf(override val output: Seq[Attribute]) extends LeafExecNode {
    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("planning-only test node")
  }
}
