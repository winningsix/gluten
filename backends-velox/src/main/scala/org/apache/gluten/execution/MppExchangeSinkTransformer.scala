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

import org.apache.gluten.metrics.MetricsUpdater
import org.apache.gluten.substrait.SubstraitContext

import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}

/**
 * Wraps a fragment's root operator to add an exchange sink (GpuPartitionedOutput) at the boundary
 * between two MPP fragments.
 *
 * In MPP mode, each fragment that feeds into a downstream fragment via a streaming GPU exchange
 * needs a sink operator that partitions and buffers the output for consumption by the downstream
 * fragment's [[MppExchangeSourceTransformer]].
 *
 * On the native (C++) side, this maps to `GpuPartitionedOutput` which:
 *   - Partitions rows by the specified partitioning keys
 *   - Writes partitioned data to `OutputBufferManager`
 *   - Enables zero-copy streaming between fragments on the same GPU
 *
 * @param child
 *   The root operator of the producing fragment's subtree.
 * @param exchangeId
 *   The ID of the exchange this sink writes to.
 * @param numPartitions
 *   The number of output partitions.
 * @param partitionKeys
 *   The attributes used for hash partitioning (empty for BROADCAST/SINGLE).
 */
case class MppExchangeSinkTransformer(
    child: SparkPlan,
    exchangeId: Int,
    numPartitions: Int,
    partitionKeys: Seq[Attribute]
) extends UnaryTransformSupport {

  override def output: Seq[Attribute] = child.output

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

  override def nodeName: String =
    s"MppExchangeSink(exchange=$exchangeId, partitions=$numPartitions)"

  override def simpleString(maxFields: Int): String = {
    val keysStr = if (partitionKeys.nonEmpty) {
      s", keys=${partitionKeys.map(_.name).mkString("[", ", ", "]")}"
    } else ""
    s"MppExchangeSink(exchange=$exchangeId, partitions=$numPartitions$keysStr)"
  }

  @transient
  override lazy val metrics: Map[String, SQLMetric] = Map(
    "outputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "outputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches"),
    "sinkTimeMs" -> SQLMetrics.createTimingMetric(sparkContext, "time spent in sink (ms)")
  )

  override protected def doTransform(context: SubstraitContext): TransformContext = {
    val childCtx = child.asInstanceOf[TransformSupport].transform(context)
    // For now, pass through the child's Substrait plan unchanged.
    // The exchange sink operator will be added at the native side during
    // MPP plan compilation, where it wraps the fragment's plan root with
    // GpuPartitionedOutput based on the ExchangeSpec metadata.
    // TODO (M4): Add an ExchangeSinkRel wrapper around childCtx.root
    //            that carries exchangeId, numPartitions, and partitionKeys
    //            so the native side can instantiate GpuPartitionedOutput directly.
    TransformContext(output, childCtx.root)
  }

  override def metricsUpdater(): MetricsUpdater = MetricsUpdater.None

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = {
    copy(child = newChild)
  }
}
