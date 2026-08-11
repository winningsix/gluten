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

import org.apache.gluten.extension.ExchangeSpec

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference}
import org.apache.spark.sql.catalyst.optimizer.BuildRight
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.physical.{HashPartitioning, Partitioning, SinglePartition}
import org.apache.spark.sql.execution.{ColumnarInputAdapter, ColumnarShuffleExchangeExec, InputIteratorTransformer, LeafExecNode}
import org.apache.spark.sql.execution.exchange.ENSURE_REQUIREMENTS
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.LongType

import org.scalatest.funsuite.AnyFunSuite

class FluxExchangeTopologySuite extends AnyFunSuite {
  private def exchange(
      id: Int,
      consumerId: Int,
      exchangeType: String,
      numPartitions: Int): ExchangeSpec = {
    ExchangeSpec(
      id = id,
      producerFragmentId = id + 10,
      consumerFragmentId = consumerId,
      exchangeType = exchangeType,
      numPartitions = numPartitions,
      partitionKeys = Seq.empty)
  }

  test("accepts aligned non-broadcast inputs and ignores broadcast fan-in") {
    val exchanges = Seq(
      exchange(1, consumerId = 7, exchangeType = "HASH", numPartitions = 4),
      exchange(2, consumerId = 7, exchangeType = "RANGE", numPartitions = 4),
      exchange(3, consumerId = 7, exchangeType = "BROADCAST", numPartitions = 1),
      exchange(4, consumerId = 8, exchangeType = "HASH", numPartitions = 2),
      exchange(5, consumerId = 8, exchangeType = "ROUND_ROBIN", numPartitions = 2)
    )

    assert(FluxExchangeTopology.inconsistentInboundPartitionCountReason(exchanges).isEmpty)
  }

  test("rejects asymmetric native inputs to the same consumer") {
    val exchanges = Seq(
      exchange(2, consumerId = 7, exchangeType = "HASH", numPartitions = 4),
      exchange(1, consumerId = 7, exchangeType = "HASH", numPartitions = 200))

    assert(
      FluxExchangeTopology
        .inconsistentInboundPartitionCountReason(exchanges)
        .contains("fragment 7 has inconsistent non-broadcast inbound partition counts " +
          "[E1:HASH=200, E2:HASH=4]"))
  }

  test("resolves one finalized HASH count for a consumer") {
    val exchanges = Seq(
      exchange(1, consumerId = 7, exchangeType = "HASH", numPartitions = 4),
      exchange(2, consumerId = 7, exchangeType = "HASH", numPartitions = 4),
      exchange(3, consumerId = 7, exchangeType = "BROADCAST", numPartitions = 1)
    )

    assert(FluxExchangeTopology.finalizedHashInboundPartitionCount(exchanges, 7) === Right(4))
  }

  test("does not resolve inconsistent finalized HASH counts") {
    val exchanges = Seq(
      exchange(1, consumerId = 7, exchangeType = "HASH", numPartitions = 1),
      exchange(2, consumerId = 7, exchangeType = "HASH", numPartitions = 4))

    assert(
      FluxExchangeTopology.finalizedHashInboundPartitionCount(exchanges, 7) ===
        Left("fragment 7 has inconsistent inbound HASH partition counts [E1=1, E2=4]"))
  }

  test("requires a positive finalized HASH input for a genuine shuffled-join fragment") {
    val noHash = Seq(exchange(1, consumerId = 7, exchangeType = "SINGLE", numPartitions = 1))
    val nonPositiveHash =
      Seq(exchange(2, consumerId = 7, exchangeType = "HASH", numPartitions = 0))

    assert(
      FluxExchangeTopology.finalizedHashInboundPartitionCount(noHash, 7) ===
        Left("fragment 7 has no inbound HASH exchange"))
    assert(
      FluxExchangeTopology.finalizedHashInboundPartitionCount(nonPositiveHash, 7) ===
        Left("fragment 7 has non-positive inbound HASH partition counts [E2=0]"))
  }

  test("rejects a non-positive native partition count") {
    val exchanges = Seq(exchange(5, consumerId = 9, exchangeType = "HASH", numPartitions = 0))

    assert(
      FluxExchangeTopology
        .inconsistentInboundPartitionCountReason(exchanges)
        .contains("fragment 9 has non-positive inbound partition counts [E5=0]"))
  }

  test("uses finalized topology without reading asymmetric Catalyst metadata") {
    val left = PartitionedLeaf(
      Seq(AttributeReference("left_key", LongType, nullable = false)()),
      numPartitions = 200)
    val right = PartitionedLeaf(
      Seq(AttributeReference("right_key", LongType, nullable = false)()),
      numPartitions = 4)
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
    val exec = FluxNativeQueryExec(join, fragments = Seq.empty, exchanges = Seq.empty)

    assert(exec.fragmentDriverPartitionCountForTests(join, Some(4)) === 4)
    intercept[IllegalArgumentException](join.outputPartitioning)
  }

  test("caps 200/4 HASH fan-in before topology-first driver planning") {
    withSQLConf("spark.gluten.mpp.localHashExchangeTasks" -> "4") {
      val exec = FluxNativeQueryExec(PartitionedLeaf(Seq.empty, 1), Seq.empty, Seq.empty)
      val capped = exec.capLocalHashExchangeTasksForTests(
        Seq(
          exchange(1, consumerId = 7, exchangeType = "HASH", numPartitions = 200),
          exchange(2, consumerId = 7, exchangeType = "HASH", numPartitions = 4)))

      assert(capped.map(_.numPartitions) === Seq(4, 4))
      assert(FluxExchangeTopology.finalizedHashInboundPartitionCount(capped, 7) === Right(4))
      assert(FluxExchangeTopology.inconsistentInboundPartitionCountReason(capped).isEmpty)
    }
  }

  test("fails closed when the HASH cap leaves 1/4 fan-in") {
    withSQLConf("spark.gluten.mpp.localHashExchangeTasks" -> "4") {
      val exec = FluxNativeQueryExec(PartitionedLeaf(Seq.empty, 1), Seq.empty, Seq.empty)
      val capped = exec.capLocalHashExchangeTasksForTests(
        Seq(
          exchange(1, consumerId = 7, exchangeType = "HASH", numPartitions = 1),
          exchange(2, consumerId = 7, exchangeType = "HASH", numPartitions = 4)))

      assert(capped.map(_.numPartitions) === Seq(1, 4))
      assert(FluxExchangeTopology.finalizedHashInboundPartitionCount(capped, 7).isLeft)
      assert(FluxExchangeTopology.inconsistentInboundPartitionCountReason(capped).nonEmpty)
    }
  }

  test("preserves Catalyst inference for a source fragment without inbound topology") {
    val plan = PartitionedLeaf(
      Seq(AttributeReference("key", LongType, nullable = false)()),
      numPartitions = 7)
    val exec = FluxNativeQueryExec(plan, fragments = Seq.empty, exchanges = Seq.empty)

    assert(exec.fragmentDriverPartitionCountForTests(plan) === 7)
  }

  test("does not suppress unrelated Catalyst partitioning failures") {
    val plan = PartitioningFailureLeaf(
      Seq(AttributeReference("key", LongType, nullable = false)()),
      "unrelated partitioning failure")
    val exec = FluxNativeQueryExec(plan, fragments = Seq.empty, exchanges = Seq.empty)

    val error = intercept[IllegalArgumentException] {
      exec.fragmentDriverPartitionCountForTests(plan)
    }
    assert(error.getMessage === "unrelated partitioning failure")
  }

  test("does not inherit a producer shuffled join across a synthetic SINGLE boundary") {
    val join = shuffledJoin(leftPartitions = 4, rightPartitions = 4)
    val producer = WholeStageTransformer(join)(transformStageId = 11)
    val single = ColumnarShuffleExchangeExec(
      SinglePartition,
      producer,
      ENSURE_REQUIREMENTS,
      producer.output,
      None)
    val parent = WholeStageTransformer(
      InputIteratorTransformer(ColumnarInputAdapter(single)))(transformStageId = 12)
    val exec = FluxNativeQueryExec(parent, fragments = Seq.empty, exchanges = Seq.empty)

    assert(
      parent.find(_.isInstanceOf[ColumnarShuffledJoin]).isDefined,
      "SparkPlan.find must reproduce the old cross-boundary false positive")
    assert(!exec.containsFragmentLocalShuffledJoinForTests(parent))
    withSQLConf(
      "spark.executor.cores" -> "16",
      "spark.gluten.mpp.joinDriversPerFragment" -> "3") {
      assert(exec.inferParallelismForTests(parent) === 1)
    }
  }

  test("still finds a shuffled join in a truly local native iterator subtree") {
    val join = shuffledJoin(leftPartitions = 4, rightPartitions = 4)
    val parent = WholeStageTransformer(
      InputIteratorTransformer(ColumnarInputAdapter(join)))(transformStageId = 13)
    val exec = FluxNativeQueryExec(parent, fragments = Seq.empty, exchanges = Seq.empty)

    assert(exec.containsFragmentLocalShuffledJoinForTests(parent))
    withSQLConf(
      "spark.executor.cores" -> "16",
      "spark.gluten.mpp.joinDriversPerFragment" -> "3") {
      assert(exec.inferParallelismForTests(parent) === 3)
    }
  }

  private def withSQLConf(values: (String, String)*)(body: => Unit): Unit = {
    val conf = SQLConf.get
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

  private def shuffledJoin(
      leftPartitions: Int,
      rightPartitions: Int): ShuffledHashJoinExecTransformer = {
    val left = PartitionedLeaf(
      Seq(AttributeReference("left_key", LongType, nullable = false)()),
      numPartitions = leftPartitions)
    val right = PartitionedLeaf(
      Seq(AttributeReference("right_key", LongType, nullable = false)()),
      numPartitions = rightPartitions)
    ShuffledHashJoinExecTransformer(
      leftKeys = Seq(left.output.head),
      rightKeys = Seq(right.output.head),
      joinType = Inner,
      buildSide = BuildRight,
      condition = None,
      left = left,
      right = right,
      isSkewJoin = false)
  }

  private case class PartitioningFailureLeaf(override val output: Seq[Attribute], message: String)
    extends LeafExecNode {
    override def outputPartitioning: Partitioning = throw new IllegalArgumentException(message)

    override protected def doExecute(): RDD[InternalRow] = {
      throw new UnsupportedOperationException("PartitioningFailureLeaf is planning-only")
    }
  }

  private case class PartitionedLeaf(override val output: Seq[Attribute], numPartitions: Int)
    extends LeafExecNode {
    override def outputPartitioning: Partitioning = {
      HashPartitioning(Seq(output.head), numPartitions)
    }

    override protected def doExecute(): RDD[InternalRow] = {
      throw new UnsupportedOperationException("PartitionedLeaf is planning-only")
    }
  }
}
