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
import org.apache.gluten.mpp.control.{GlutenMppExecutorService, MppPeerState, MppQueryRunId}
import org.apache.gluten.runtime.Runtimes
import org.apache.gluten.vectorized.{ColumnarBatchInIterator, MppQueryJniWrapper}

import org.apache.spark.{NarrowDependency, Partition, SparkContext, SparkEnv, TaskContext, TaskKilledException}
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.{Namespace, SparkDirectoryUtil}

import org.apache.commons.io.FileUtils

import java.io.File
import java.util.{HashMap => JHashMap, Iterator => JIterator}
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.control.NonFatal

/**
 * RDD that executes an MPP query plan via JNI.
 *
 * By default this is a single-partition RDD: the native MPP coordinator handles all parallelism
 * internally. When multi-executor MPP is enabled, each Spark partition is pinned to one executor
 * peer and starts its own native coordinator with a sharded set of scan splits. Substrait plans are
 * pre-generated on the driver side (by [[MppNativeQueryExec]]) and passed as serialized byte
 * arrays. This avoids accessing SparkPlan.sparkContext on executor nodes (which would cause NPE).
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
 * @param mppQueryId
 *   Query id shared by all native peers for this MPP query attempt.
 * @param configuredPeerEndpointsJson
 *   JSON array of producer peer endpoint records from discovered peers or
 *   spark.gluten.mpp.peerEndpoints.
 * @param peerInfos
 *   Optional Spark executor peer metadata. Non-empty means one Spark partition per peer.
 * @param sparkPartitionCount
 *   Number of Spark tasks to expose for this native MPP query.
 * @param broadcastProducerFragmentIds
 *   Fragments that produce BROADCAST exchanges. Only peer 0 scans these fragments to avoid
 *   duplicating build-side rows across peers.
 * @param replicatedCartesianMaxBuildBytes
 *   Canonical build-size cap for a query containing a replicated Cartesian, or zero when no native
 *   nested-loop join cap should be installed.
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
    mppQueryId: String,
    configuredPeerEndpointsJson: String,
    peerInfos: Array[MppPeerInfo],
    fragmentSplitInfos: Array[Array[Array[Byte]]],
    fusedBroadcastsByConsumer: Map[Int, Seq[FusedBroadcast]],
    localStreamSlots: Seq[MppLocalStreamSlot],
    var localInputRDDs: ColumnarInputRDDsWrapper,
    sparkPartitionCount: Int,
    broadcastProducerFragmentIds: Set[Int],
    keepDeviceOutput: Boolean,
    replicatedCartesianMaxBuildBytes: Long,
    pipelineTime: SQLMetric,
    outputRows: SQLMetric,
    outputBatches: SQLMetric
) extends RDD[ColumnarBatch](sc, localInputRDDs.getDependencies)
  with Logging {

  require(
    replicatedCartesianMaxBuildBytes >= 0,
    s"replicated Cartesian max build bytes must be non-negative, found " +
      replicatedCartesianMaxBuildBytes)

  private val numSparkPartitions: Int =
    if (peerInfos.nonEmpty) peerInfos.length else math.max(1, sparkPartitionCount)

  override protected def getPartitions: Array[Partition] = {
    Array.tabulate(numSparkPartitions) {
      partitionIndex =>
        MppPartition(
          index = partitionIndex,
          totalPartitions = numSparkPartitions,
          peerInfo = peerInfos.lift(partitionIndex),
          localInputPartitions =
            if (localStreamSlots.nonEmpty) localInputRDDs.getPartitions(partitionIndex)
            else Seq.empty
        )
    }
  }

  override protected def getPreferredLocations(split: Partition): Seq[String] = {
    split match {
      case partition: MppPartition =>
        partition.peerInfo.map(_.preferredLocation).toSeq
      case _ =>
        Nil
    }
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

    // Substrait plans and exchange specs were pre-generated on the driver side
    // (by MppNativeQueryExec.doExecuteColumnar) to avoid NPE from accessing
    // SparkPlan.sparkContext on executors.

    logInfo(
      s"MppNativeQueryRDD: *** LAUNCHING MPP EXECUTION *** " +
        s"with ${fragmentPlans.length} fragments " +
        s"(Spark partition ${mppPartition.index}/${mppPartition.totalPartitions})")
    logDebug(s"MppNativeQueryRDD: exchange specs JSON: $exchangeSpecsJson")

    // Submit all fragment plans to MppQueryCoordinator via JNI.
    // This launches ALL fragments concurrently (MPP all-stages-up)
    // and wires them together via OutputBufferManager streaming exchange.
    val runtimeExtraConf = new JHashMap[String, String]()
    if (MppNativeQueryRDD.largeParquetScanChunksEnabled) {
      runtimeExtraConf.put(MppNativeQueryRDD.largeParquetScanChunksKey, "true")
    }
    if (keepDeviceOutput) {
      runtimeExtraConf.put("spark.gluten.sql.columnar.cudf.skipOutputToVelox", "true")
    }
    val runtime =
      Runtimes.contextInstance(BackendsApiManager.getBackendName, "MppQuery", runtimeExtraConf)
    val jniWrapper = MppQueryJniWrapper.create(runtime)
    val actualExecutorId = Option(SparkEnv.get).map(_.executorId).getOrElse("unknown")
    val localPeerId = mppPartition.peerInfo.map(_.peerId).getOrElse(actualExecutorId)
    val nativeMppQueryId =
      if (mppPartition.peerInfo.isEmpty && mppPartition.totalPartitions == 1) {
        s"$mppQueryId-t${context.taskAttemptId()}"
      } else {
        mppQueryId
      }
    mppPartition.peerInfo.foreach {
      expected =>
        if (expected.peerId != actualExecutorId) {
          throw new IllegalStateException(
            s"MppNativeQueryRDD partition ${mppPartition.index} expected executor " +
              s"${expected.peerId}, but Spark scheduled it on $actualExecutorId")
        }
    }
    val mppPeerSpecJson =
      MppNativeQueryRDD.buildPeerSpecJson(
        nativeMppQueryId,
        localPeerId,
        configuredPeerEndpointsJson,
        mppPartition.index,
        mppPartition.totalPartitions)
    logInfo(s"MppNativeQueryRDD: peer spec JSON: $mppPeerSpecJson")
    val tCreateStart = System.nanoTime()
    val localFragmentSplitInfos = MppNativeQueryRDD.partitionSplitInfos(
      fragmentSplitInfos,
      mppPartition.index,
      mppPartition.totalPartitions,
      broadcastProducerFragmentIds)

    // Pack every JVM-backed ValueStream input into parallel slot/iterator arrays. Fused broadcasts
    // materialize from their broadcast relation; local streams read the aligned parent RDD
    // partition owned by this Spark task.
    val numFragments = fragmentPlans.length
    val jvmStreamsByConsumer =
      mutable.HashMap[Int, mutable.ArrayBuffer[(Int, JIterator[ColumnarBatch])]]()
    fusedBroadcastsByConsumer.foreach {
      case (consumerId, fusedList) =>
        if (consumerId >= 0 && consumerId < numFragments) {
          val sortedBySlot = fusedList.sortBy(_.slotIdx)
          val streams = jvmStreamsByConsumer.getOrElseUpdate(consumerId, mutable.ArrayBuffer.empty)
          sortedBySlot.foreach {
            fused =>
              streams += fused.slotIdx ->
                MppNativeQueryRDD.materializeFusedBroadcastIteratorImpl(fused)
          }
          logInfo(
            s"MppNativeQueryRDD: fragment $consumerId has " +
              s"${sortedBySlot.size} fused broadcast(s) at slots " +
              sortedBySlot.map(_.slotIdx).mkString("[", ",", "]"))
        }
    }

    if (localStreamSlots.nonEmpty) {
      val localIterators =
        localInputRDDs.getIterators(mppPartition.localInputPartitions, context)
      require(
        localIterators.size == localStreamSlots.size,
        s"MPP local stream iterator count ${localIterators.size} did not match slot count " +
          s"${localStreamSlots.size}"
      )
      localStreamSlots.zip(localIterators).foreach {
        case (slot, iterator) =>
          val streams =
            jvmStreamsByConsumer.getOrElseUpdate(slot.fragmentId, mutable.ArrayBuffer.empty)
          streams += slot.slotIdx -> iterator.asJava
      }
    }

    val jvmStreamSlotIndicesPerFrag = new Array[Array[Int]](numFragments)
    val jvmStreamIteratorsPerFrag = new Array[Array[Object]](numFragments)
    jvmStreamsByConsumer.foreach {
      case (consumerId, streams) =>
        val sortedBySlot = streams.sortBy(_._1)
        val duplicateSlots = sortedBySlot.groupBy(_._1).collect {
          case (slot, occurrences) if occurrences.size > 1 => slot
        }
        require(
          duplicateSlots.isEmpty,
          s"MPP fragment $consumerId has duplicate JVM stream slot(s): " +
            duplicateSlots.toSeq.sorted.mkString("[", ",", "]"))
        jvmStreamSlotIndicesPerFrag(consumerId) = sortedBySlot.map(_._1).toArray
        jvmStreamIteratorsPerFrag(consumerId) = sortedBySlot.map {
          case (_, iterator) =>
            new ColumnarBatchInIterator(BackendsApiManager.getBackendName, iterator): Object
        }.toArray
        logInfo(
          s"MppNativeQueryRDD: fragment $consumerId has ${sortedBySlot.size} JVM stream(s) " +
            s"at slots ${sortedBySlot.map(_._1).mkString("[", ",", "]")}")
    }

    // MPP creates multiple Velox Tasks behind this one Spark task. Allocate the query-level root
    // only after all JVM-side validation and stream materialization has succeeded. The lease keeps
    // ownership until nativeCreateMppQuery returns; a native parse/plan-conversion failure
    // therefore deletes the root instead of leaking it for the lifetime of the executor.
    val spillRootLease =
      MppNativeQueryRDD.createSpillRootLease(SparkDirectoryUtil.get().namespace("gluten-spill"))
    val mppHandle = spillRootLease.handoffAfterCreate {
      spillRootPath =>
        jniWrapper.nativeCreateMppQuery(
          fragmentPlans,
          numDriversPerFragment,
          exchangeSpecsJson.getBytes("UTF-8"),
          mppPeerSpecJson.getBytes("UTF-8"),
          localFragmentSplitInfos,
          jvmStreamSlotIndicesPerFrag,
          jvmStreamIteratorsPerFrag,
          replicatedCartesianMaxBuildBytes,
          spillRootPath
        )
    }
    @volatile var mppClosed = false
    def closeMppHandle(): Long = this.synchronized {
      if (mppClosed) {
        0L
      } else {
        val tCloseStart = System.nanoTime()
        try {
          jniWrapper.nativeCloseMppQuery(mppHandle)
        } finally {
          mppClosed = true
        }
        System.nanoTime() - tCloseStart
      }
    }

    def closeMppHandleAtTaskCompletion(): Unit = {
      if (!mppClosed) {
        try {
          closeMppHandle()
        } catch {
          case NonFatal(e) =>
            logWarning("MppNativeQueryRDD: failed to close MPP query at task completion", e)
        }
      }
    }

    def closeMppHandleAfterSetupFailure(setupFailure: Throwable): Nothing =
      MppNativeQueryRDD.rethrowAfterCleanup(setupFailure, () => closeMppHandle())

    // Install the native-handle owner immediately after creation. In particular, no control-plane
    // registration failure may leave a successfully-created coordinator without a completion hook.
    try {
      context.addTaskCompletionListener[Unit](_ => closeMppHandleAtTaskCompletion())
    } catch {
      case listenerFailure: Throwable => closeMppHandleAfterSetupFailure(listenerFailure)
    }

    val runId = MppQueryRunId(mppQueryId, context.stageId(), context.stageAttemptNumber())
    val activeQuery =
      try {
        GlutenMppExecutorService.registerQuery(
          runId,
          mppPartition.index,
          mppPartition.totalPartitions,
          context,
          () => jniWrapper.nativeAbortMppQuery(mppHandle))
      } catch {
        case registrationFailure: Throwable =>
          closeMppHandleAfterSetupFailure(registrationFailure)
      }

    def reportTerminal(state: String): Unit =
      activeQuery.foreach(GlutenMppExecutorService.reportTerminal(_, state))

    try {
      context.addTaskCompletionListener[Unit] {
        taskContext =>
          // This listener is registered after the close-only safety hook and therefore runs first
          // under Spark's LIFO listener ordering. Keep the old close-before-terminal-report
          // contract; the earlier hook remains an idempotent backstop for registration races.
          closeMppHandleAtTaskCompletion()
          val terminalState =
            if (taskContext.isInterrupted() || activeQuery.exists(_.isAbortRequested)) {
              MppPeerState.Aborted
            } else {
              MppPeerState.Succeeded
            }
          reportTerminal(terminalState)
      }
    } catch {
      case listenerFailure: Throwable =>
        MppNativeQueryRDD.rethrowAfterCleanup(
          listenerFailure,
          () =>
            activeQuery.foreach(
              query => GlutenMppExecutorService.reportFailure(query, listenerFailure)),
          () => closeMppHandle(),
          () => reportTerminal(MppPeerState.Failed)
        )
    }

    val tCreateDoneStartBegin = System.nanoTime()
    if (activeQuery.exists(query => !query.beginNativeStart())) {
      try {
        closeMppHandle()
      } finally {
        reportTerminal(MppPeerState.Aborted)
      }
      throw new TaskKilledException("MPP task interrupted before native query start")
    }
    try {
      jniWrapper.nativeStartMppQuery(mppHandle)
      activeQuery.foreach(_.finishNativeStart(succeeded = true))
    } catch {
      case NonFatal(e) =>
        activeQuery.foreach {
          query =>
            query.finishNativeStart(succeeded = false)
            GlutenMppExecutorService.reportFailure(query, e)
        }
        try {
          closeMppHandle()
        } finally {
          reportTerminal(MppPeerState.Failed)
        }
        throw e
    }
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
            "queryId" -> MppNativeQueryRDD.quotedJson(nativeMppQueryId),
            "sparkPartition" -> mppPartition.index.toString,
            "sparkPartitions" -> mppPartition.totalPartitions.toString,
            "fragments" -> fragmentPlans.length.toString,
            "localPeerId" -> MppNativeQueryRDD.quotedJson(localPeerId),
            "nativeCreateMppQueryNanos" -> (tCreateDoneStartBegin - tCreateStart).toString,
            "nativeStartMppQueryNanos" -> (tStartDone - tCreateDoneStartBegin).toString
          )
        ))
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
        try {
          nextHandle = jniWrapper.nativeGetMppOutput(mppHandle)
        } catch {
          case NonFatal(e) =>
            finished = true
            activeQuery.filterNot(_.isAbortRequested).foreach {
              query => GlutenMppExecutorService.reportFailure(query, e)
            }
            try {
              closeMppHandle()
            } catch {
              case NonFatal(closeError) =>
                logWarning(
                  "MppNativeQueryRDD: failed to close MPP query after native error",
                  closeError)
            } finally {
              val terminalState =
                if (activeQuery.exists(_.isAbortRequested)) MppPeerState.Aborted
                else MppPeerState.Failed
              reportTerminal(terminalState)
            }
            throw e
        }
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
          // A peer can exhaust its local root before remote peers have consumed all of the
          // producers hosted by this executor. Publish output EOS through the MPP control plane
          // and keep every native coordinator alive until the driver has observed EOS from all
          // peers. A peer failure releases this wait through the existing abort path.
          if (mppPartition.totalPartitions > 1) {
            val completionStart = System.nanoTime()
            val query = activeQuery.getOrElse {
              val error = new IllegalStateException(
                "Multi-peer MPP completion requires the MPP query control plane")
              try {
                closeMppHandle()
              } catch {
                case NonFatal(closeError) => error.addSuppressed(closeError)
              }
              throw error
            }
            if (!GlutenMppExecutorService.awaitPeerCompletion(query)) {
              val closeNanos = closeMppHandle()
              reportTerminal(MppPeerState.Aborted)
              logWarning(
                f"MppNativeQueryRDD: peer completion was aborted; " +
                  f"nativeCloseMppQuery=${closeNanos / 1e6}%.1fms")
              throw new TaskKilledException(
                "MPP peer completion aborted before every peer reached output EOS")
            }
            logInfo(
              f"MppNativeQueryRDD: all-peer output EOS acknowledged in " +
                f"${(System.nanoTime() - completionStart) / 1e6}%.1fms")
          }
          // A downstream consumer may retain a zero-copy view of the final native batch after
          // exhausting this iterator. Iceberg's Parquet writer does this until commit(), so keep
          // the coordinator and its output pool alive through the Spark task completion listener.
          // Error and abort paths still close eagerly above.
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
              f"nativeCloseMppQuery=deferred")
          if (runtimeTimingProbeEnabled) {
            val safeMinGetOutputNanos =
              if (totalGetOutputCalls == 0) 0L else minGetOutputNanos
            val safeFirstBatchNanos = if (firstBatchNanos < 0) 0L else firstBatchNanos
            logWarning(
              MppNativeQueryRDD.runtimeTimingProbeLine(
                "native_output_complete",
                Seq(
                  "queryId" -> MppNativeQueryRDD.quotedJson(nativeMppQueryId),
                  "sparkPartition" -> mppPartition.index.toString,
                  "sparkPartitions" -> mppPartition.totalPartitions.toString,
                  "fragments" -> fragmentPlans.length.toString,
                  "nativeGetMppOutputCalls" -> totalGetOutputCalls.toString,
                  "nativeGetMppOutputSumNanos" -> totalGetOutputNanos.toString,
                  "nativeGetMppOutputMinNanos" -> safeMinGetOutputNanos.toString,
                  "nativeGetMppOutputMaxNanos" -> maxGetOutputNanos.toString,
                  "timeToFirstBatchNanos" -> safeFirstBatchNanos.toString,
                  "nativeCloseMppQueryDeferred" -> "true",
                  "outputRows" -> totalOutputRows.toString,
                  "outputBatches" -> totalOutputBatches.toString
                )
              ))
          }
          reportTerminal(MppPeerState.Succeeded)
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

  override protected def clearDependencies(): Unit = {
    super.clearDependencies()
    localInputRDDs = null
  }

}

/** Spark partition descriptor for one native MPP query. */
private[execution] case class MppPartition(
    override val index: Int,
    totalPartitions: Int,
    peerInfo: Option[MppPeerInfo],
    localInputPartitions: Seq[Partition])
  extends Partition

private[execution] case class MppAlignedInputPartition(
    override val index: Int,
    parentPartitions: Array[Partition])
  extends Partition

private[execution] class MppAlignedInputRDD(
    var parentRDD: RDD[ColumnarBatch],
    parentGroups: Array[Array[Int]])
  extends RDD[ColumnarBatch](
    parentRDD.sparkContext,
    Seq(new NarrowDependency[ColumnarBatch](parentRDD) {
      override def getParents(partitionId: Int): Seq[Int] = parentGroups(partitionId).toSeq
    })
  ) {

  override protected def getPartitions: Array[Partition] = {
    val parentPartitions = parentRDD.partitions
    parentGroups.zipWithIndex.map {
      case (parentIndices, index) =>
        MppAlignedInputPartition(index, parentIndices.map(parentPartitions))
    }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] = {
    split
      .asInstanceOf[MppAlignedInputPartition]
      .parentPartitions
      .iterator
      // Creating an RDD iterator may start its runtime (for example, a Python worker). Keep the
      // aligned parents sequential so a coalesced partition does not start every parent at once.
      .flatMap(parentRDD.iterator(_, context))
  }

  override protected def getPreferredLocations(split: Partition): Seq[String] = {
    split
      .asInstanceOf[MppAlignedInputPartition]
      .parentPartitions
      .flatMap(parentRDD.preferredLocations)
      .distinct
  }

  override protected def clearDependencies(): Unit = {
    super.clearDependencies()
    parentRDD = null
  }
}

private[gluten] case class MppPeerInfo(
    peerId: String,
    host: String,
    port: Int,
    preferredLocation: String)

/**
 * Owns a newly-created MPP spill root until the native coordinator has accepted it.
 *
 * [[handoffAfterCreate]] transfers ownership only after its callback returns. Its `finally` closes
 * the lease in both cases: before handoff that recursively deletes the root, while after handoff it
 * is an idempotent no-op because native MppQueryCoordinator owns cleanup.
 */
final private[execution] class MppSpillRootLease(val root: File, deleteRoot: File => Unit)
  extends AutoCloseable {
  private val ownsRoot = new AtomicBoolean(true)

  def path: String = root.getAbsolutePath

  def handoffAfterCreate[T](create: String => T): T = {
    try {
      val result = create(path)
      ownsRoot.set(false)
      result
    } finally {
      close()
    }
  }

  override def close(): Unit = {
    if (ownsRoot.compareAndSet(true, false)) {
      deleteRoot(root)
    }
  }
}

private[execution] object MppNativeQueryRDD extends Logging {

  private[execution] def rethrowAfterCleanup(
      primaryFailure: Throwable,
      cleanupActions: (() => Unit)*): Nothing = {
    cleanupActions.foreach {
      cleanup =>
        try {
          cleanup()
        } catch {
          case cleanupFailure: Throwable if cleanupFailure ne primaryFailure =>
            primaryFailure.addSuppressed(cleanupFailure)
          case _: Throwable =>
        }
    }
    throw primaryFailure
  }

  private[execution] def createSpillRootLease(namespace: Namespace): MppSpillRootLease = {
    val root = namespace.mkChildDirRoundRobin(s"mpp-${UUID.randomUUID()}")
    new MppSpillRootLease(
      root,
      file =>
        if (!FileUtils.deleteQuietly(file) && file.exists()) {
          logWarning(s"MppNativeQueryRDD: failed to delete unowned MPP spill root $file")
        })
  }

  def alignInputRDD(rdd: RDD[ColumnarBatch], targetPartitions: Int): RDD[ColumnarBatch] = {
    require(targetPartitions > 0, s"targetPartitions must be positive, got $targetPartitions")
    val sourcePartitions = rdd.partitions.length
    if (sourcePartitions == targetPartitions) {
      return rdd
    }

    val parentGroups =
      if (sourcePartitions >= targetPartitions) {
        Array.tabulate(targetPartitions) {
          targetIndex =>
            val start = (targetIndex.toLong * sourcePartitions / targetPartitions).toInt
            val end = ((targetIndex + 1L) * sourcePartitions / targetPartitions).toInt
            (start until end).toArray
        }
      } else {
        Array.tabulate(targetPartitions) {
          targetIndex =>
            if (targetIndex < sourcePartitions) Array(targetIndex) else Array.emptyIntArray
        }
      }
    new MppAlignedInputRDD(rdd, parentGroups)
  }

  /** Opt-in: raise cuDF parquet chunk/pass read limits for MPP table scans. */
  val largeParquetScanChunksKey: String = "spark.gluten.mpp.largeParquetScanChunks"

  val runtimeTimingProbeKey: String = "spark.gluten.mpp.runtimeTimingProbe"

  def largeParquetScanChunksEnabled: Boolean =
    SQLConf.get.getConfString(largeParquetScanChunksKey, "false").toBoolean

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

  private[execution] def quotedJson(value: String): String = "\"" + jsonEscape(value) + "\""

  def buildPeerSpecJson(
      queryId: String,
      localPeerId: String,
      configuredPeerEndpointsJson: String,
      peerIndex: Int,
      peerCount: Int): String = {
    val peersJson = Option(configuredPeerEndpointsJson)
      .map(_.trim)
      .filter(_.nonEmpty)
      .getOrElse("[]")
    s"""{
       |  "queryId": "${jsonEscape(queryId)}",
       |  "localPeerId": "${jsonEscape(localPeerId)}",
       |  "peerIndex": $peerIndex,
       |  "peerCount": $peerCount,
       |  "peers": $peersJson
       |}""".stripMargin
  }

  def partitionSplitInfos(
      allSplits: Array[Array[Array[Byte]]],
      peerIndex: Int,
      peerCount: Int,
      broadcastProducerFragmentIds: Set[Int]): Array[Array[Array[Byte]]] = {
    // Each byte array is one scan-node SplitInfo blob, not one file split.
    // File-level sharding happens after JNI parses SplitInfo on the native side.
    allSplits
  }

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
