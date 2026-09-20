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

import org.apache.gluten.execution.ProjectExecTransformer
import org.apache.gluten.extension.FluxRemoveRedundantShuffleRule.NativeHashPartitioning
import org.apache.gluten.utils.LocalTableScanExecCompat

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.plans.physical.HashPartitioning
import org.apache.spark.sql.execution.ColumnarShuffleExchangeExec
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.IntegerType

class FluxRemoveRedundantShuffleRuleSuite extends QueryTest with SharedSparkSession {

  override protected def sparkConf: org.apache.spark.SparkConf = {
    val conf = super.sparkConf
    conf.getAll.collect { case (key, "null") => key }.foreach(conf.remove)
    conf
      .set("spark.master", "local[2]")
      .set("spark.executor.cores", "2")
      .set("spark.executor.memory", "1g")
      .set("spark.executor.memoryOverhead", "512m")
      .set("spark.sql.adaptive.enabled", "false")
  }

  private def nestedHashExchanges(innerPartitions: Int, outerPartitions: Int) = {
    val key = AttributeReference("destination_key", IntegerType, nullable = false)()
    val scan = LocalTableScanExecCompat(Seq(key), Seq.empty[InternalRow])
    val inner = ColumnarShuffleExchangeExec(
      outputPartitioning = HashPartitioning(Seq(key), innerPartitions),
      child = scan,
      projectOutputAttributes = scan.output)
    val outer =
      ShuffleExchangeExec(HashPartitioning(Seq(inner.output.head), outerPartitions), inner)
    (inner, outer)
  }

  private def nestedColumnarHashExchanges(innerPartitions: Int, outerPartitions: Int) = {
    val key = AttributeReference("destination_key", IntegerType, nullable = false)()
    val scan = LocalTableScanExecCompat(Seq(key), Seq.empty[InternalRow])
    val inner = ColumnarShuffleExchangeExec(
      outputPartitioning = HashPartitioning(Seq(key), innerPartitions),
      child = scan,
      projectOutputAttributes = scan.output)
    val outer = ColumnarShuffleExchangeExec(
      outputPartitioning = HashPartitioning(Seq(inner.output.head), outerPartitions),
      child = inner,
      projectOutputAttributes = inner.output)
    (inner, outer)
  }

  test("keeps unequal logical partition counts without native-cap awareness") {
    val (inner, outer) = nestedHashExchanges(innerPartitions = 16, outerPartitions = 640)
    withSQLConf(
      "spark.gluten.mpp.removeRedundantShuffle" -> "true",
      "spark.gluten.mpp.multiExecutor.enabled" -> "true",
      "spark.gluten.mpp.multiExecutor.numPartitions" -> "4",
      "spark.gluten.mpp.remoteHashExchangeDestinations" -> "4"
    ) {
      val rewritten = FluxRemoveRedundantShuffleRule()(outer)
      assert(rewritten eq outer)
      assert(rewritten ne inner)
    }
  }

  test("removes unequal logical HASH exchanges finalized to one native cap") {
    val (inner, outer) = nestedHashExchanges(innerPartitions = 16, outerPartitions = 640)
    withSQLConf(
      "spark.gluten.mpp.removeRedundantShuffle" -> "true",
      "spark.gluten.mpp.removeRedundantShuffle.nativePartitionCapAware" -> "true",
      "spark.gluten.mpp.multiExecutor.enabled" -> "true",
      "spark.gluten.mpp.multiExecutor.numPartitions" -> "4",
      "spark.gluten.mpp.remoteHashExchangeDestinations" -> "4"
    ) {
      assert(FluxRemoveRedundantShuffleRule()(outer) eq inner)
    }
  }

  test("removes the columnar HASH exchange seen by FLUX cross-cut planning") {
    val (inner, outer) = nestedColumnarHashExchanges(innerPartitions = 16, outerPartitions = 640)
    withSQLConf(
      "spark.gluten.mpp.removeRedundantShuffle" -> "true",
      "spark.gluten.mpp.removeRedundantShuffle.nativePartitionCapAware" -> "true",
      "spark.gluten.mpp.multiExecutor.enabled" -> "true",
      "spark.gluten.mpp.multiExecutor.numPartitions" -> "4",
      "spark.gluten.mpp.remoteHashExchangeDestinations" -> "4"
    ) {
      assert(FluxRemoveRedundantShuffleRule()(outer) eq inner)
    }
  }

  test("preserves the columnar exchange output projection when removing a synthetic hash key") {
    val syntheticHash = AttributeReference("hash_partition_key", IntegerType, nullable = false)()
    val destinationKey = AttributeReference("destination_key", IntegerType, nullable = false)()
    val scan =
      LocalTableScanExecCompat(Seq(syntheticHash, destinationKey), Seq.empty[InternalRow])
    val inner = ColumnarShuffleExchangeExec(
      outputPartitioning = HashPartitioning(Seq(destinationKey), 16),
      child = scan,
      projectOutputAttributes = scan.output)
    val outer = ColumnarShuffleExchangeExec(
      outputPartitioning = HashPartitioning(Seq(destinationKey), 640),
      child = inner,
      projectOutputAttributes = Seq(destinationKey))

    withSQLConf(
      "spark.gluten.mpp.removeRedundantShuffle" -> "true",
      "spark.gluten.mpp.removeRedundantShuffle.nativePartitionCapAware" -> "true",
      "spark.gluten.mpp.multiExecutor.enabled" -> "true",
      "spark.gluten.mpp.multiExecutor.numPartitions" -> "4",
      "spark.gluten.mpp.remoteHashExchangeDestinations" -> "4"
    ) {
      val rewritten = FluxRemoveRedundantShuffleRule()(outer)
      assert(rewritten.isInstanceOf[ProjectExecTransformer])
      assert(rewritten.output == Seq(destinationKey))
      assert(rewritten.children == Seq(inner))
      assert(!rewritten.output.exists(_.name == "hash_partition_key"))
    }
  }

  test("does not equate partition counts on different sides of the native cap") {
    val capped = NativeHashPartitioning(cap = 4, exact = false, source = "test")
    assert(FluxRemoveRedundantShuffleRule.partitionCountsMatch(16, 640, Some(capped)))
    assert(!FluxRemoveRedundantShuffleRule.partitionCountsMatch(2, 640, Some(capped)))
    assert(FluxRemoveRedundantShuffleRule.partitionCountsMatch(2, 2, None))
  }

  test("derives the same peer-times-lane cap as native topology planning") {
    val conf = Map(
      "spark.gluten.mpp.multiExecutor.enabled" -> "true",
      "spark.gluten.mpp.multiExecutor.numPartitions" -> "4",
      "spark.gluten.sql.columnar.backend.velox.flux.keyedFinalDestinationLanes" -> "true",
      "spark.gluten.sql.columnar.backend.velox.flux.keyedFinalLocalDrivers" -> "3"
    )
    val native = FluxRemoveRedundantShuffleRule
      .nativeHashPartitioning((key, default) => conf.getOrElse(key, default))
      .get
    assert(native.cap == 12)
    assert(native.exact)
  }
}
