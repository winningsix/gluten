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

import org.apache.gluten.execution.{CartesianProductExecTransformer, ColumnarCartesianProductBridge, ProjectExecTransformer, VeloxBroadcastNestedLoopJoinExecTransformer}
import org.apache.gluten.extension.FluxReplicatedCartesianRule.{MAX_BUILD_BYTES_KEY, REPLICATED_CARTESIAN_MAX_BUILD_BYTES_TAG, SideStats}
import org.apache.gluten.utils.LocalTableScanExecCompat

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, GreaterThan, Literal}
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight}
import org.apache.spark.sql.catalyst.plans.logical.{LeafNode, Statistics}
import org.apache.spark.sql.execution.{ColumnarBroadcastExchangeExec, FilterExec, LocalTableScanExec, SparkPlan}
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.IntegerType

class FluxReplicatedCartesianRuleSuite extends QueryTest with SharedSparkSession {

  private case class SizedLeaf(output: Seq[Attribute], sizeInBytes: BigInt, rows: Option[BigInt])
    extends LeafNode {
    override def computeStats(): Statistics = Statistics(sizeInBytes, rows)
  }

  private def scan(name: String, sizeInBytes: BigInt, rows: Option[BigInt]): LocalTableScanExec = {
    val attr = AttributeReference(name, IntegerType, nullable = true)()
    val physical = LocalTableScanExecCompat(Seq(attr), Seq.empty[InternalRow])
    physical.setLogicalLink(SizedLeaf(Seq(attr), sizeInBytes, rows))
    physical
  }

  private def cartesian(left: SparkPlan, right: SparkPlan): CartesianProductExecTransformer =
    CartesianProductExecTransformer(
      ColumnarCartesianProductBridge(left),
      ColumnarCartesianProductBridge(right),
      condition = None)

  test("chooseBuildSide selects only a stats-bounded complete input") {
    val small = Some(SideStats(128, Some(8), "test"))
    val smaller = Some(SideStats(64, Some(4), "test"))
    val large = Some(SideStats(4096, Some(256), "test"))
    val max = BigInt(1024)

    assert(FluxReplicatedCartesianRule.chooseBuildSide(small, large, max).contains(BuildLeft))
    assert(FluxReplicatedCartesianRule.chooseBuildSide(large, small, max).contains(BuildRight))
    assert(FluxReplicatedCartesianRule.chooseBuildSide(small, smaller, max).contains(BuildRight))
    assert(FluxReplicatedCartesianRule.chooseBuildSide(None, large, max).isEmpty)
    assert(FluxReplicatedCartesianRule.chooseBuildSide(large, large, max).isEmpty)
  }

  test("right small side becomes a native broadcast nested-loop build") {
    withSQLConf(MAX_BUILD_BYTES_KEY -> "1kb", "spark.gluten.mpp.failOnFallback" -> "true") {
      val left = scan("l", 4096, Some(256))
      val right = scan("r", 128, Some(8))
      val rewritten = FluxReplicatedCartesianRule()(cartesian(left, right))

      val join = rewritten.asInstanceOf[VeloxBroadcastNestedLoopJoinExecTransformer]
      assert(join.left eq left)
      assert(join.right.isInstanceOf[ColumnarBroadcastExchangeExec])
      assert(join.right.asInstanceOf[ColumnarBroadcastExchangeExec].child eq right)
      assert(join.joinBuildSide == BuildRight)
      assert(join.getTagValue(REPLICATED_CARTESIAN_MAX_BUILD_BYTES_TAG).contains(1024L))
      assert(rewritten.output.map(_.exprId) == (left.output ++ right.output).map(_.exprId))
    }
  }

  test("left small side is broadcast and original schema order is restored") {
    withSQLConf(MAX_BUILD_BYTES_KEY -> "1kb", "spark.gluten.mpp.failOnFallback" -> "true") {
      val left = scan("l", 128, Some(8))
      val right = scan("r", 4096, Some(256))
      val original = cartesian(left, right)
      val rewritten = FluxReplicatedCartesianRule()(original)

      val project = rewritten.asInstanceOf[ProjectExecTransformer]
      val join = project.child.asInstanceOf[VeloxBroadcastNestedLoopJoinExecTransformer]
      assert(join.left eq right)
      assert(join.right.asInstanceOf[ColumnarBroadcastExchangeExec].child eq left)
      assert(join.joinBuildSide == BuildRight)
      assert(join.getTagValue(REPLICATED_CARTESIAN_MAX_BUILD_BYTES_TAG).contains(1024L))
      assert(project.output.map(_.exprId) == original.output.map(_.exprId))
    }
  }

  test("conditional Cartesian keeps its condition after build-side reordering") {
    withSQLConf(MAX_BUILD_BYTES_KEY -> "1kb", "spark.gluten.mpp.failOnFallback" -> "true") {
      val left = scan("l", 128, Some(8))
      val right = scan("r", 4096, Some(256))
      val condition = GreaterThan(left.output.head, right.output.head)
      val original = cartesian(left, right).copy(condition = Some(condition))
      val rewritten = FluxReplicatedCartesianRule()(original)
      val join = rewritten
        .asInstanceOf[ProjectExecTransformer]
        .child
        .asInstanceOf[VeloxBroadcastNestedLoopJoinExecTransformer]

      assert(join.condition.contains(condition))
      assert(rewritten.output.map(_.exprId) == original.output.map(_.exprId))
    }
  }

  test("strict FLUX rejects unknown and over-limit builds during planning") {
    withSQLConf(MAX_BUILD_BYTES_KEY -> "1kb", "spark.gluten.mpp.failOnFallback" -> "true") {
      val overLimit = cartesian(scan("l1", 2048, Some(128)), scan("r1", 4096, Some(256)))
      val overLimitError = intercept[IllegalStateException] {
        FluxReplicatedCartesianRule()(overLimit)
      }
      assert(overLimitError.getMessage.contains("unsafe replicated Cartesian"))
      assert(overLimitError.getMessage.contains("maxBuildBytes=1024"))

      val unknownLeft = LocalTableScanExecCompat(
        Seq(AttributeReference("l2", IntegerType, nullable = true)()),
        Seq.empty[InternalRow])
      val unknownRight = LocalTableScanExecCompat(
        Seq(AttributeReference("r2", IntegerType, nullable = true)()),
        Seq.empty[InternalRow])
      val unknownError = intercept[IllegalStateException] {
        FluxReplicatedCartesianRule()(cartesian(unknownLeft, unknownRight))
      }
      assert(unknownError.getMessage.contains("left=unknown"))
      assert(unknownError.getMessage.contains("right=unknown"))
    }
  }

  test("non-strict mode retains Spark partition-pair Cartesian when unsafe") {
    withSQLConf(MAX_BUILD_BYTES_KEY -> "1kb", "spark.gluten.mpp.failOnFallback" -> "false") {
      val original = cartesian(scan("l", 2048, Some(128)), scan("r", 4096, Some(256)))
      assert(FluxReplicatedCartesianRule()(original) eq original)
    }
  }

  test("unlinked unary operators do not inherit child statistics") {
    withSQLConf(MAX_BUILD_BYTES_KEY -> "1kb", "spark.gluten.mpp.failOnFallback" -> "false") {
      val wrappedSmall = FilterExec(Literal.TrueLiteral, scan("l", 128, Some(8)))
      val original = cartesian(wrappedSmall, scan("r", 4096, Some(256)))

      // The manually constructed unary node has no logical link. Its child statistics are not a
      // valid upper bound for every unary operator (for example Generate can expand rows), so the
      // rule must fail closed instead of authorizing a replicated build from the child's size.
      assert(FluxReplicatedCartesianRule()(original) eq original)
    }
  }
}
