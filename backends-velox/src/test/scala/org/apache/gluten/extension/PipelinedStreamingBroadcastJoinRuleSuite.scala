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

import org.apache.gluten.execution.{BroadcastHashJoinExecTransformer, ReplicatedPartitioning, ShuffledHashJoinExecTransformer, VeloxBroadcastNestedLoopJoinExecTransformer, VeloxReplicatedNestedLoopJoinExecTransformer}

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.optimizer.BuildRight
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.physical.{IdentityBroadcastMode, SinglePartition}
import org.apache.spark.sql.execution.{CoalesceExec, ColumnarBroadcastExchangeExec, ColumnarShuffleExchangeExec, LocalTableScanExec}
import org.apache.spark.sql.execution.exchange.{ENSURE_REQUIREMENTS, ReusedExchangeExec}
import org.apache.spark.sql.execution.joins.HashedRelationBroadcastMode
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.LongType

class PipelinedStreamingBroadcastJoinRuleSuite extends QueryTest with SharedSparkSession {

  test("broadcast hash join becomes a replicated pipelined shuffle") {
    withSQLConf(
      "spark.sql.shuffle.pipelined.enabled" -> "true",
      "spark.gluten.sql.columnar.pipelined.streamingBroadcast.enabled" -> "true") {
      val probeKey = AttributeReference("probe_key", LongType, nullable = false)()
      val buildKey = AttributeReference("build_key", LongType, nullable = false)()
      val probe = CoalesceExec(
        4,
        LocalTableScanExec(Seq(probeKey), Seq.empty[InternalRow], None))
      val build = LocalTableScanExec(Seq(buildKey), Seq.empty[InternalRow], None)
      val broadcast = ColumnarBroadcastExchangeExec(
        HashedRelationBroadcastMode(Seq(buildKey)),
        build)
      val original = BroadcastHashJoinExecTransformer(
        Seq(probeKey),
        Seq(buildKey),
        Inner,
        BuildRight,
        condition = None,
        probe,
        broadcast,
        isNullAwareAntiJoin = false)

      val rewritten = PipelinedStreamingBroadcastJoinRule()(original)
        .asInstanceOf[ShuffledHashJoinExecTransformer]
      val exchange = rewritten.right.asInstanceOf[ColumnarShuffleExchangeExec]

      assert(exchange.outputPartitioning == ReplicatedPartitioning(4))
      assert(exchange.child eq build)
      assert(rewritten.left eq probe)
      assert(rewritten.outputPartitioning == probe.outputPartitioning)
      assert(rewritten.output.map(_.exprId) == original.output.map(_.exprId))
    }
  }

  test("rule is disabled unless both streaming broadcast and pipelined shuffle are enabled") {
    val probeKey = AttributeReference("probe_key", LongType, nullable = false)()
    val buildKey = AttributeReference("build_key", LongType, nullable = false)()
    val probe = CoalesceExec(
      2,
      LocalTableScanExec(Seq(probeKey), Seq.empty[InternalRow], None))
    val build = LocalTableScanExec(Seq(buildKey), Seq.empty[InternalRow], None)
    val original = BroadcastHashJoinExecTransformer(
      Seq(probeKey),
      Seq(buildKey),
      Inner,
      BuildRight,
      condition = None,
      probe,
      ColumnarBroadcastExchangeExec(HashedRelationBroadcastMode(Seq(buildKey)), build),
      isNullAwareAntiJoin = false)

    withSQLConf(
      "spark.sql.shuffle.pipelined.enabled" -> "true",
      "spark.gluten.sql.columnar.pipelined.streamingBroadcast.enabled" -> "false") {
      assert(PipelinedStreamingBroadcastJoinRule()(original) eq original)
    }
  }

  test("broadcast nested-loop join becomes a replicated pipelined shuffle") {
    withSQLConf(
      "spark.sql.shuffle.pipelined.enabled" -> "true",
      "spark.gluten.sql.columnar.pipelined.streamingBroadcast.enabled" -> "true") {
      val probeKey = AttributeReference("probe_key", LongType, nullable = false)()
      val scalar = AttributeReference("scalar", LongType, nullable = false)()
      val probe = CoalesceExec(
        4,
        LocalTableScanExec(Seq(probeKey), Seq.empty[InternalRow], None))
      val build = LocalTableScanExec(Seq(scalar), Seq.empty[InternalRow], None)
      val original = VeloxBroadcastNestedLoopJoinExecTransformer(
        left = probe,
        right = ColumnarBroadcastExchangeExec(IdentityBroadcastMode, build),
        buildSide = BuildRight,
        joinType = Inner,
        condition = None)

      val rewritten = PipelinedStreamingBroadcastJoinRule()(original)
        .asInstanceOf[VeloxReplicatedNestedLoopJoinExecTransformer]
      val exchange = rewritten.right.asInstanceOf[ColumnarShuffleExchangeExec]

      assert(exchange.outputPartitioning == ReplicatedPartitioning(4))
      assert(exchange.child eq build)
      assert(rewritten.left eq probe)
      assert(rewritten.output.map(_.exprId) == original.output.map(_.exprId))
    }
  }

  test("reused broadcast gets an independent replicated exchange") {
    withSQLConf(
      "spark.sql.shuffle.pipelined.enabled" -> "true",
      "spark.gluten.sql.columnar.pipelined.streamingBroadcast.enabled" -> "true") {
      val probeKey = AttributeReference("probe_key", LongType, nullable = false)()
      val buildKey = AttributeReference("build_key", LongType, nullable = false)()
      val reusedBuildKey = buildKey.newInstance()
      val probe = CoalesceExec(
        4,
        LocalTableScanExec(Seq(probeKey), Seq.empty[InternalRow], None))
      val build = LocalTableScanExec(Seq(buildKey), Seq.empty[InternalRow], None)
      val broadcast = ColumnarBroadcastExchangeExec(
        HashedRelationBroadcastMode(Seq(buildKey)),
        build)
      val reused = ReusedExchangeExec(Seq(reusedBuildKey), broadcast)
      val original = BroadcastHashJoinExecTransformer(
        Seq(probeKey),
        Seq(reusedBuildKey),
        Inner,
        BuildRight,
        condition = None,
        probe,
        reused,
        isNullAwareAntiJoin = false)

      val rewritten = PipelinedStreamingBroadcastJoinRule()(original)
        .asInstanceOf[ShuffledHashJoinExecTransformer]
      val exchange = rewritten.right.asInstanceOf[ColumnarShuffleExchangeExec]

      assert(exchange.outputPartitioning == ReplicatedPartitioning(4))
      assert(exchange.child eq build)
      assert(exchange.output.map(_.exprId) == reused.output.map(_.exprId))
      assert(rewritten.output.map(_.exprId) == original.output.map(_.exprId))
    }
  }

  test("shared shuffle below broadcast is duplicated for each replicated consumer") {
    withSQLConf(
      "spark.sql.shuffle.pipelined.enabled" -> "true",
      "spark.gluten.sql.columnar.pipelined.streamingBroadcast.enabled" -> "true") {
      val probeKey = AttributeReference("probe_key", LongType, nullable = false)()
      val scalar = AttributeReference("scalar", LongType, nullable = false)()
      val probe = CoalesceExec(
        4,
        LocalTableScanExec(Seq(probeKey), Seq.empty[InternalRow], None))
      val build = LocalTableScanExec(Seq(scalar), Seq.empty[InternalRow], None)
      val sharedSingle = ColumnarShuffleExchangeExec(
        SinglePartition,
        build,
        ENSURE_REQUIREMENTS,
        build.output,
        advisoryPartitionSize = None)

      def rewrite(): VeloxReplicatedNestedLoopJoinExecTransformer = {
        val original = VeloxBroadcastNestedLoopJoinExecTransformer(
          left = probe,
          right = ColumnarBroadcastExchangeExec(IdentityBroadcastMode, sharedSingle),
          buildSide = BuildRight,
          joinType = Inner,
          condition = None)
        PipelinedStreamingBroadcastJoinRule()(original)
          .asInstanceOf[VeloxReplicatedNestedLoopJoinExecTransformer]
      }

      val firstNested = rewrite().right
        .asInstanceOf[ColumnarShuffleExchangeExec]
        .child
        .asInstanceOf[ColumnarShuffleExchangeExec]
      val secondNested = rewrite().right
        .asInstanceOf[ColumnarShuffleExchangeExec]
        .child
        .asInstanceOf[ColumnarShuffleExchangeExec]

      assert(!(firstNested eq sharedSingle))
      assert(!(secondNested eq sharedSingle))
      assert(!(firstNested eq secondNested))
      assert(firstNested.outputPartitioning == SinglePartition)
      assert(secondNested.outputPartitioning == SinglePartition)
    }
  }
}
