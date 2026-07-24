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

import org.apache.gluten.execution.{BroadcastHashJoinExecTransformer, FileSourceScanExecTransformerBase, ReplicatedPartitioning, ShuffledHashJoinExecTransformer, VeloxBroadcastNestedLoopJoinExecTransformer, VeloxReplicatedNestedLoopJoinExecTransformer}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarBroadcastExchangeExec, ColumnarShuffleExchangeExec, SparkPlan}
import org.apache.spark.sql.execution.exchange.{ENSURE_REQUIREMENTS, ReusedExchangeExec}
import org.apache.spark.sql.internal.SQLConf

/**
 * Replaces Spark driver broadcasts with a Spark-native, streaming UCX broadcast exchange.
 *
 * The streamed side keeps its current partitioning. The build side becomes a pipelined shuffle
 * with one replicated output per streamed partition, so every native hash-join task receives the
 * complete build input incrementally. This deliberately reuses the normal shuffled-hash-join
 * transformer: only the exchange data movement is broadcast-specific.
 */
case class PipelinedStreamingBroadcastJoinRule() extends Rule[SparkPlan] with Logging {
  private val enabledKey =
    "spark.gluten.sql.columnar.pipelined.streamingBroadcast.enabled"
  private val pipelinedShuffleKey = "spark.sql.shuffle.pipelined.enabled"

  override def apply(plan: SparkPlan): SparkPlan = {
    if (!confBoolean(enabledKey, defaultValue = false) ||
        !confBoolean(pipelinedShuffleKey, defaultValue = false)) {
      return plan
    }

    val rewritten = plan.transformUp {
      case join: BroadcastHashJoinExecTransformer if !join.isNullAwareAntiJoin =>
        rewrite(join).getOrElse(join)
      case join: VeloxBroadcastNestedLoopJoinExecTransformer =>
        rewriteNestedLoop(join).getOrElse(join)
    }
    // A UCX destination queue is destructive and has one sequence space. Until the transport
    // supports consumer-aware fan-out, give every reused consumer an independent exchange.
    rewritten.transformUp {
      case reused @ ReusedExchangeExec(output, exchange: ColumnarShuffleExchangeExec)
          if isReplicated(exchange) =>
        logInfo(
          s"PipelinedStreamingBroadcastJoinRule: expanding reused streaming exchange " +
            s"${exchange.id} for consumer ${reused.id}")
        newReplicatedExchange(exchange.child, output, exchange.outputPartitioning.numPartitions)
    }
  }

  private def rewrite(join: BroadcastHashJoinExecTransformer): Option[SparkPlan] = {
    val numConsumers = resolveNumConsumers(join.streamedPlan, join.id)
    if (numConsumers <= 0) {
      logWarning(
        s"PipelinedStreamingBroadcastJoinRule: keeping Spark broadcast for join ${join.id}; " +
          s"invalid streamed partition count $numConsumers")
      return None
    }

    replaceBroadcastBoundary(join.buildPlan, numConsumers).map {
      replicatedBuild =>
        val (newLeft, newRight) = join.buildSide match {
          case BuildLeft => (replicatedBuild, join.right)
          case BuildRight => (join.left, replicatedBuild)
        }
        logInfo(
          s"PipelinedStreamingBroadcastJoinRule: rewriting join ${join.id} to streaming UCX " +
            s"broadcast with $numConsumers consumers buildSide=${join.buildSide}")
        ShuffledHashJoinExecTransformer(
          join.leftKeys,
          join.rightKeys,
          join.joinType,
          join.buildSide,
          join.condition,
          newLeft,
          newRight,
          isSkewJoin = false)
    }
  }

  private def rewriteNestedLoop(
      join: VeloxBroadcastNestedLoopJoinExecTransformer): Option[SparkPlan] = {
    val (buildPlan, streamedPlan) = join.joinBuildSide match {
      case BuildLeft => (join.left, join.right)
      case BuildRight => (join.right, join.left)
    }
    val numConsumers = resolveNumConsumers(streamedPlan, join.id)
    if (numConsumers <= 0) {
      logWarning(
        s"PipelinedStreamingBroadcastJoinRule: keeping Spark nested-loop broadcast for join " +
          s"${join.id}; invalid streamed partition count $numConsumers")
      return None
    }

    replaceBroadcastBoundary(buildPlan, numConsumers).map {
      replicatedBuild =>
        val (newLeft, newRight) = join.joinBuildSide match {
          case BuildLeft => (replicatedBuild, join.right)
          case BuildRight => (join.left, replicatedBuild)
        }
        logInfo(
          s"PipelinedStreamingBroadcastJoinRule: rewriting nested-loop join ${join.id} to " +
            s"streaming UCX broadcast with $numConsumers consumers " +
            s"buildSide=${join.joinBuildSide}")
        VeloxReplicatedNestedLoopJoinExecTransformer(
          left = newLeft,
          right = newRight,
          buildSide = join.joinBuildSide,
          joinType = join.joinType,
          condition = join.condition)
    }
  }

  private def resolveNumConsumers(streamedPlan: SparkPlan, joinId: Int): Int = {
    val planned = streamedPlan.outputPartitioning.numPartitions
    if (planned > 0) {
      return planned
    }

    val scanPartitionCounts = streamedPlan.collect {
      case scan: FileSourceScanExecTransformerBase => scan.getPartitions.size
    }.filter(_ > 0)

    scanPartitionCounts match {
      case Seq(resolved) =>
        logInfo(
          s"PipelinedStreamingBroadcastJoinRule: resolved join $joinId consumer count " +
            s"from its file scan: $resolved (output partitioning reported $planned)")
        resolved
      case _ =>
        logWarning(
          s"PipelinedStreamingBroadcastJoinRule: cannot safely resolve consumer count for join " +
            s"$joinId; file scan partition counts were " +
            scanPartitionCounts.mkString("[", ",", "]"))
        planned
    }
  }

  private def replaceBroadcastBoundary(
      buildPlan: SparkPlan,
      numConsumers: Int): Option[SparkPlan] = {
    case class Boundary(node: SparkPlan, child: SparkPlan, output: Seq[Attribute])

    val boundaries = buildPlan.collect {
      case exchange: ColumnarBroadcastExchangeExec =>
        Boundary(exchange, exchange.child, exchange.output)
      case reused @ ReusedExchangeExec(output, exchange: ColumnarBroadcastExchangeExec) =>
        Boundary(reused, exchange.child, output)
      case reused @ ReusedExchangeExec(output, exchange: ColumnarShuffleExchangeExec)
          if isReplicated(exchange) =>
        Boundary(reused, exchange.child, output)
    }
    if (boundaries.size != 1) {
      logWarning(
        s"PipelinedStreamingBroadcastJoinRule: expected one ColumnarBroadcastExchange under " +
          s"build ${buildPlan.id}, found ${boundaries.size}; keeping Spark broadcast")
      return None
    }

    val target = boundaries.head
    val replacement = newReplicatedExchange(target.child, target.output, numConsumers)
    Some(
      buildPlan.transformDown {
        case node if node eq target.node => replacement
      })
  }

  private def isReplicated(exchange: ColumnarShuffleExchangeExec): Boolean =
    exchange.outputPartitioning.isInstanceOf[ReplicatedPartitioning]

  private def newReplicatedExchange(
      child: SparkPlan,
      output: Seq[Attribute],
      numConsumers: Int): ColumnarShuffleExchangeExec =
    ColumnarShuffleExchangeExec(
      ReplicatedPartitioning(numConsumers),
      duplicateNestedShuffleBoundaries(child),
      ENSURE_REQUIREMENTS,
      output,
      advisoryPartitionSize = None)

  /**
   * A native UCX destination queue has one destructive sequence space. Spark can share the same
   * exchange object below multiple broadcast consumers, which would register multiple readers for
   * one `(shuffleId, reduceId)` and let the last registration steal the stream. Recreate every
   * nested shuffle boundary after Spark's exchange-reuse rule has run so each replicated build owns
   * its complete UCX path.
   */
  private def duplicateNestedShuffleBoundaries(plan: SparkPlan): SparkPlan =
    plan.transformUp {
      case exchange: ColumnarShuffleExchangeExec => exchange.copy()
    }

  private def confBoolean(key: String, defaultValue: Boolean): Boolean =
    SQLConf.get.getConfString(key, defaultValue.toString).toBoolean
}
