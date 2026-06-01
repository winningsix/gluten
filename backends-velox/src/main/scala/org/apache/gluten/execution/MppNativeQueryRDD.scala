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
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.ColumnarBatch

import java.util.{Iterator => JIterator}

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
    fusedBroadcastsByConsumer: Map[Int, Seq[FusedBroadcast]],
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
    val mppPartition = split match {
      case p: MppPartition => p
      case other =>
        throw new IllegalArgumentException(
          s"MppNativeQueryRDD expected MppPartition, got ${other.getClass.getName}")
    }
    val planBuildStart = System.nanoTime()
    val runtimeTimingProbeEnabled = MppNativeQueryRDD.runtimeTimingProbeEnabled
    val localPeerId = Option(org.apache.spark.SparkEnv.get).map(_.executorId).getOrElse("unknown")
    val mppQueryId =
      s"${Option(sc.applicationId).getOrElse("unknown-app")}-${context.stageId()}-" +
        s"${context.partitionId()}-${context.taskAttemptId()}"

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
    val tCreateStart = System.nanoTime()

    // For each consumer fragment that has fused broadcasts, materialize the
    // build side as Iterator[ColumnarBatch] (one iterator per fused broadcast)
    // and pack into parallel int[] / Object[] arrays the JNI side can iterate.
    // numFragments is implied by fragmentPlans.length; we emit ragged arrays
    // (entries are null for fragments with no fused broadcasts).
    val numFragments = fragmentPlans.length
    val broadcastSlotIndicesPerFrag = new Array[Array[Int]](numFragments)
    val broadcastIteratorsPerFrag = new Array[Array[JIterator[ColumnarBatch]]](numFragments)
    fusedBroadcastsByConsumer.foreach {
      case (consumerId, fusedList) =>
        if (consumerId >= 0 && consumerId < numFragments) {
          val sortedBySlot = fusedList.sortBy(_.slotIdx)
          broadcastSlotIndicesPerFrag(consumerId) = sortedBySlot.map(_.slotIdx).toArray
          broadcastIteratorsPerFrag(consumerId) = sortedBySlot
            .map(MppNativeQueryRDD.materializeFusedBroadcastIteratorImpl)
            .toArray
          logInfo(
            s"MppNativeQueryRDD: fragment $consumerId has " +
              s"${sortedBySlot.size} fused broadcast(s) at slots " +
              sortedBySlot.map(_.slotIdx).mkString("[", ",", "]"))
        }
    }

    val mppHandle = jniWrapper.nativeCreateMppQuery(
      fragmentPlans,
      numDriversPerFragment,
      exchangeSpecsJson.getBytes("UTF-8"),
      fragmentSplitInfos,
      broadcastSlotIndicesPerFrag,
      broadcastIteratorsPerFrag.asInstanceOf[Array[Array[Object]]]
    )
    val tCreateDoneStartBegin = System.nanoTime()
    jniWrapper.nativeStartMppQuery(mppHandle)
    val tStartDone = System.nanoTime()
    logWarning(
      f"MppNativeQueryRDD: TIMING " +
        f"nativeCreateMppQuery=${(tCreateDoneStartBegin - tCreateStart) / 1e6}%.1fms " +
        f"nativeStartMppQuery=${(tStartDone - tCreateDoneStartBegin) / 1e6}%.1fms")
    if (runtimeTimingProbeEnabled) {
      logWarning(
        MppNativeQueryRDD.runtimeTimingProbeLine(
          "native_start",
          Seq(
            "queryId" -> MppNativeQueryRDD.quotedJson(mppQueryId),
            "sparkPartition" -> mppPartition.index.toString,
            "sparkPartitions" -> "1",
            "fragments" -> fragmentPlans.length.toString,
            "localPeerId" -> MppNativeQueryRDD.quotedJson(localPeerId),
            "nativeCreateMppQueryNanos" -> (tCreateDoneStartBegin - tCreateStart).toString,
            "nativeStartMppQueryNanos" -> (tStartDone - tCreateDoneStartBegin).toString)))
    }

    logInfo("MppNativeQueryRDD: all MPP fragments started, streaming exchange active")

    val tracker = org.apache.gluten.metrics.TaskWallTimeTracker.get()
    tracker.planBuildNanos += (System.nanoTime() - planBuildStart)

    // 4. Create iterator that pulls batches from MppQueryCoordinator via JNI.
    val mppIter = new Iterator[ColumnarBatch] {
      private var nextHandle: Long = -1L
      private var finished = false
      // Per-batch timing accumulators. We dump aggregate at end-of-stream so
      // the per-batch logWarning doesn't drown the log file (240 batches!).
      private var totalGetOutputNanos: Long = 0L
      private var totalGetOutputCalls: Long = 0L
      private var maxGetOutputNanos: Long = 0L
      private var minGetOutputNanos: Long = Long.MaxValue
      private val firstBatchStart: Long = System.nanoTime()
      private var firstBatchNanos: Long = -1L
      private var totalOutputRows: Long = 0L
      private var totalOutputBatches: Long = 0L

      override def hasNext: Boolean = {
        if (finished) return false
        if (nextHandle != -1L) return true
        val tStart = System.nanoTime()
        nextHandle = jniWrapper.nativeGetMppOutput(mppHandle)
        val elapsed = System.nanoTime() - tStart
        totalGetOutputNanos += elapsed
        totalGetOutputCalls += 1
        if (elapsed > maxGetOutputNanos) maxGetOutputNanos = elapsed
        if (elapsed < minGetOutputNanos) minGetOutputNanos = elapsed
        if (firstBatchNanos < 0 && nextHandle != 0L) {
          firstBatchNanos = System.nanoTime() - firstBatchStart
        }
        if (nextHandle == 0L) {
          finished = true
          val tCloseStart = System.nanoTime()
          jniWrapper.nativeCloseMppQuery(mppHandle)
          val closeNanos = System.nanoTime() - tCloseStart
          val avgMs =
            totalGetOutputNanos.toDouble / math.max(1L, totalGetOutputCalls) / 1e6
          logWarning(
            f"MppNativeQueryRDD: TIMING " +
              f"nativeGetMppOutput totalCalls=$totalGetOutputCalls " +
              f"sum=${totalGetOutputNanos / 1e6}%.1fms " +
              f"avg=$avgMs%.2fms " +
              f"min=${minGetOutputNanos / 1e6}%.2fms " +
              f"max=${maxGetOutputNanos / 1e6}%.2fms " +
              f"timeToFirstBatch=${firstBatchNanos / 1e6}%.1fms " +
              f"nativeCloseMppQuery=${closeNanos / 1e6}%.1fms")
          if (runtimeTimingProbeEnabled) {
            val safeMinGetOutputNanos =
              if (totalGetOutputCalls == 0) 0L else minGetOutputNanos
            val safeFirstBatchNanos = if (firstBatchNanos < 0) 0L else firstBatchNanos
            logWarning(
              MppNativeQueryRDD.runtimeTimingProbeLine(
                "native_output_complete",
                Seq(
                  "queryId" -> MppNativeQueryRDD.quotedJson(mppQueryId),
                  "sparkPartition" -> mppPartition.index.toString,
                  "sparkPartitions" -> "1",
                  "fragments" -> fragmentPlans.length.toString,
                  "nativeGetMppOutputCalls" -> totalGetOutputCalls.toString,
                  "nativeGetMppOutputSumNanos" -> totalGetOutputNanos.toString,
                  "nativeGetMppOutputMinNanos" -> safeMinGetOutputNanos.toString,
                  "nativeGetMppOutputMaxNanos" -> maxGetOutputNanos.toString,
                  "timeToFirstBatchNanos" -> safeFirstBatchNanos.toString,
                  "nativeCloseMppQueryNanos" -> closeNanos.toString,
                  "outputRows" -> totalOutputRows.toString,
                  "outputBatches" -> totalOutputBatches.toString)))
          }
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
        totalOutputRows += batch.numRows()
        totalOutputBatches += 1L
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

private[execution] object MppNativeQueryRDD {
  val runtimeTimingProbeKey: String = "spark.gluten.mpp.runtimeTimingProbe"

  def runtimeTimingProbeEnabled: Boolean =
    SQLConf.get.getConfString(runtimeTimingProbeKey, "false").toBoolean

  private[execution] def runtimeTimingProbeLine(
      event: String,
      fields: Seq[(String, String)]): String = {
    val fieldJson = (Seq("event" -> quotedJson(event)) ++ fields)
      .map { case (key, value) => quotedJson(key) + ":" + value }
      .mkString(",")
    s"[MPP_RUNTIME_PROBE] {$fieldJson}"
  }

  private[execution] def quotedJson(value: String): String = s"\"${jsonEscape(value)}\""

  private[execution] def jsonEscape(value: String): String = {
    value.flatMap {
      case '"' => "\\\""
      case '\\' => "\\\\"
      case '\b' => "\\b"
      case '\f' => "\\f"
      case '\n' => "\\n"
      case '\r' => "\\r"
      case '\t' => "\\t"
      case c if c < ' ' => f"\\u${c.toInt}%04x"
      case c => c.toString
    }
  }

  /**
   * Materialize the broadcasted BuildSideRelation into a Java Iterator[ColumnarBatch] suitable for
   * hand-off to the C++ side via `makeJniColumnarBatchIterator`. Mirrors
   * [[VeloxBroadcastBuildSideRDD.genBroadcastBuildSideIterator]] but lives here to keep the MPP RDD
   * self-contained.
   */
  def materializeFusedBroadcastIteratorImpl(fb: FusedBroadcast): JIterator[ColumnarBatch] = {
    val start = System.nanoTime()
    val relation = fb.broadcast.value.asReadOnlyCopy()
    val scalaIter = relation.deserialized
    val tracker =
      org.apache.gluten.metrics.TaskWallTimeTracker.get()
    tracker.broadcastBuildNanos += (System.nanoTime() - start)
    new JIterator[ColumnarBatch] {
      override def hasNext: Boolean = scalaIter.hasNext
      override def next(): ColumnarBatch = scalaIter.next()
    }
  }
}
