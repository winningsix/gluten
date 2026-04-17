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

import org.apache.gluten.backendsapi.BackendsApiManager
import org.apache.gluten.iterator.Iterators
import org.apache.gluten.runtime.Runtimes
import org.apache.gluten.vectorized.MppQueryJniWrapper

import org.apache.spark.{Partition, SparkContext, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * RDD that executes an MPP query plan via JNI.
 *
 * This is a single-partition RDD: the native MPP coordinator handles all parallelism internally.
 * Substrait plans are pre-generated on the driver side (by [[MppNativeQueryExec]]) and passed as
 * serialized byte arrays. This avoids accessing SparkPlan.sparkContext on executor nodes (which
 * would cause NPE).
 *
 * The native side creates an MppQueryCoordinator that:
 *   - Launches all fragment pipelines concurrently
 *   - Connects them via streaming GPU exchanges (OutputBufferManager / GpuExchange)
 *   - Returns the final fragment's output as an iterator of ColumnarBatch
 *
 * @param sc
 *   The SparkContext.
 * @param fragmentPlans
 *   Pre-serialized Substrait plan bytes for each fragment (generated on driver).
 * @param numDriversPerFragment
 *   The parallelism for each fragment.
 * @param exchangeSpecsJson
 *   JSON-serialized exchange specifications (generated on driver).
 * @param pipelineTime
 *   Metric for tracking total pipeline execution time.
 * @param outputRows
 *   Metric for tracking number of output rows.
 * @param outputBatches
 *   Metric for tracking number of output batches.
 */
class MppNativeQueryRDD(
    @transient sc: SparkContext,
    fragmentPlans: Array[Array[Byte]],
    numDriversPerFragment: Array[Int],
    exchangeSpecsJson: String,
    fragmentSplitInfos: Array[Array[Array[Byte]]],
    pipelineTime: SQLMetric,
    outputRows: SQLMetric,
    outputBatches: SQLMetric
) extends RDD[ColumnarBatch](sc, Nil)
  with Logging {

  override protected def getPartitions: Array[Partition] = {
    // Single partition -- the MPP coordinator handles parallelism internally.
    Array(MppPartition(0))
  }

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] = {
    val planBuildStart = System.nanoTime()

    // Substrait plans and exchange specs were pre-generated on the driver side
    // (by MppNativeQueryExec.doExecuteColumnar) to avoid NPE from accessing
    // SparkPlan.sparkContext on executors.

    logInfo(
      s"MppNativeQueryRDD: *** LAUNCHING MPP EXECUTION *** " +
        s"with ${fragmentPlans.length} fragments")
    logDebug(s"MppNativeQueryRDD: exchange specs JSON: $exchangeSpecsJson")

    // Submit all fragment plans to MppQueryCoordinator via JNI.
    // This launches ALL fragments concurrently (MPP all-stages-up)
    // and wires them together via OutputBufferManager streaming exchange.
    val runtime = Runtimes.contextInstance(BackendsApiManager.getBackendName, "MppQuery")
    val jniWrapper = MppQueryJniWrapper.create(runtime)
    val mppHandle = jniWrapper.nativeCreateMppQuery(
      fragmentPlans,
      numDriversPerFragment,
      exchangeSpecsJson.getBytes("UTF-8"),
      fragmentSplitInfos)
    jniWrapper.nativeStartMppQuery(mppHandle)

    logInfo("MppNativeQueryRDD: all MPP fragments started, streaming exchange active")

    val tracker = org.apache.gluten.metrics.TaskWallTimeTracker.get()
    tracker.planBuildNanos += (System.nanoTime() - planBuildStart)

    // 4. Create iterator that pulls batches from MppQueryCoordinator via JNI.
    val mppIter = new Iterator[ColumnarBatch] {
      private var nextHandle: Long = -1L
      private var finished = false

      override def hasNext: Boolean = {
        if (finished) return false
        if (nextHandle != -1L) return true
        nextHandle = jniWrapper.nativeGetMppOutput(mppHandle)
        if (nextHandle == 0L) {
          finished = true
          jniWrapper.nativeCloseMppQuery(mppHandle)
          logInfo("MppNativeQueryRDD: MPP execution complete, all fragments finished")
          false
        } else {
          true
        }
      }

      override def next(): ColumnarBatch = {
        if (!hasNext) throw new NoSuchElementException("MPP iterator exhausted")
        val handle = nextHandle
        nextHandle = -1L
        // Convert native handle to ColumnarBatch using Gluten's standard mechanism
        val batch = org.apache.gluten.columnarbatch.ColumnarBatches.create(handle)
        outputRows += batch.numRows()
        outputBatches += 1
        batch
      }
    }

    // Wrap with lifecycle management
    Iterators
      .wrap(mppIter)
      .protectInvocationFlow()
      .recyclePayload(batch => batch.close())
      .collectLifeMillis(millis => pipelineTime += millis)
      .asInterruptible(context)
      .create()
  }

}

/** Simple partition for the single-partition MppNativeQueryRDD. */
private[execution] case class MppPartition(override val index: Int) extends Partition
