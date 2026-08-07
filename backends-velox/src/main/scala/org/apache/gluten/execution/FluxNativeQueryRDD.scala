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
import org.apache.gluten.flux.control.{GlutenFluxExecutorService, FluxPeerState, FluxQueryRunId}
import org.apache.gluten.runtime.Runtimes
import org.apache.gluten.vectorized.{ColumnarBatchInIterator, FluxQueryJniWrapper}

import org.apache.spark.{
  NarrowDependency,
  Partition,
  SparkContext,
  SparkEnv,
  TaskContext,
  TaskKilledException}
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.task.TaskResources
import org.apache.spark.util.{Namespace, SparkDirectoryUtil}

import org.apache.commons.io.FileUtils

import java.io.File
import java.util.{HashMap => JHashMap, Iterator => JIterator}
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.control.NonFatal

/**
 * RDD that executes an FLUX query plan via JNI.
 *
 * By default this is a single-partition RDD: the native FLUX coordinator handles all parallelism
 * internally. When multi-executor FLUX is enabled, each Spark partition is pinned to one executor
 * peer and starts its own native coordinator with a sharded set of scan splits. Substrait plans are
 * pre-generated on the driver side (by [[FluxNativeQueryExec]]) and passed as serialized byte
 * arrays. This avoids accessing SparkPlan.sparkContext on executor nodes (which would cause NPE).
 *
 * The native side creates an FluxQueryCoordinator that:
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
 * @param fluxQueryId
 *   Query id shared by all native peers for this FLUX query attempt.
 * @param configuredPeerEndpointsJson
 *   JSON array of producer peer endpoint records from discovered peers or
 *   spark.gluten.mpp.peerEndpoints.
 * @param peerInfos
 *   Optional Spark executor peer metadata. Non-empty means one Spark partition per peer.
 * @param sparkPartitionCount
 *   Number of Spark tasks to expose for this native FLUX query.
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
class FluxNativeQueryRDD(
    @transient sc: SparkContext,
    fragmentPlans: Array[Array[Byte]],
    numDriversPerFragment: Array[Int],
    exchangeSpecsJson: String,
    fluxQueryId: String,
    configuredPeerEndpointsJson: String,
    peerInfos: Array[FluxPeerInfo],
    fragmentSplitInfos: Array[Array[Array[Byte]]],
    fusedBroadcastsByConsumer: Map[Int, Seq[FusedBroadcast]],
    localStreamSlots: Seq[FluxLocalStreamSlot],
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

  // FluxNativeQueryRDD instances are constructed on the driver. Capture query-scoped SQLConf here
  // so the executor receives the values that were active for this query after RDD serialization.
  private val serializedRuntimeExtraConf = FluxNativeQueryRDD.queryScopedNativeConfSnapshot

  private val runtimeNumDriversPerFragment =
    FluxNativeQueryRDD.singleGatherRootDriverCounts(numDriversPerFragment, exchangeSpecsJson)

  private val numSparkPartitions: Int =
    if (peerInfos.nonEmpty) peerInfos.length else math.max(1, sparkPartitionCount)

  override protected def getPartitions: Array[Partition] = {
    Array.tabulate(numSparkPartitions) {
      partitionIndex =>
        FluxPartition(
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
      case partition: FluxPartition =>
        partition.peerInfo.map(_.preferredLocation).toSeq
      case _ =>
        Nil
    }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] = {
    val fluxPartition = split match {
      case p: FluxPartition => p
      case other =>
        throw new IllegalArgumentException(
          s"FluxNativeQueryRDD expected FluxPartition, got ${other.getClass.getName}")
    }

    val planBuildStart = System.nanoTime()
    val runtimeTimingProbeEnabled = FluxNativeQueryRDD.runtimeTimingProbeEnabled

    // Substrait plans and exchange specs were pre-generated on the driver side
    // (by FluxNativeQueryExec.doExecuteColumnar) to avoid NPE from accessing
    // SparkPlan.sparkContext on executors.

    logInfo(
      s"FluxNativeQueryRDD: *** LAUNCHING FLUX EXECUTION *** " +
        s"with ${fragmentPlans.length} fragments " +
        s"(Spark partition ${fluxPartition.index}/${fluxPartition.totalPartitions})")
    logDebug(s"FluxNativeQueryRDD: exchange specs JSON: $exchangeSpecsJson")

    // Submit all fragment plans to FluxQueryCoordinator via JNI.
    // This launches ALL fragments concurrently (FLUX all-stages-up)
    // and wires them together via OutputBufferManager streaming exchange.
    val runtimeExtraConf = new JHashMap[String, String]()
    serializedRuntimeExtraConf.foreach { case (key, value) => runtimeExtraConf.put(key, value) }
    if (FluxNativeQueryRDD.largeParquetScanChunksEnabled) {
      runtimeExtraConf.put(FluxNativeQueryRDD.largeParquetScanChunksKey, "true")
    }
    if (keepDeviceOutput) {
      runtimeExtraConf.put("spark.gluten.sql.columnar.cudf.skipOutputToVelox", "true")
    }
    val runtime =
      Runtimes.contextInstance(BackendsApiManager.getBackendName, "FluxQuery", runtimeExtraConf)
    val jniWrapper = FluxQueryJniWrapper.create(runtime)
    val actualExecutorId = Option(SparkEnv.get).map(_.executorId).getOrElse("unknown")
    val localPeerId = fluxPartition.peerInfo.map(_.peerId).getOrElse(actualExecutorId)
    // One physical RDD can appear in more than one JVM stream slot. Spark then invokes compute()
    // repeatedly for that RDD inside the same task while the earlier native coordinator is still
    // live. Give those invocations distinct IDs; otherwise both the executor control registry and
    // native UCX routing see two simultaneous peers with the same identity. The per-task ordinal is
    // deterministic because every FLUX peer traverses the aligned stream slots in the same order.
    val invocationFluxQueryId = FluxNativeQueryRDD.nextInvocationQueryId(fluxQueryId, context)
    val nativeFluxQueryId =
      if (fluxPartition.peerInfo.isEmpty && fluxPartition.totalPartitions == 1) {
        s"$invocationFluxQueryId-t${context.taskAttemptId()}"
      } else {
        invocationFluxQueryId
      }
    fluxPartition.peerInfo.foreach {
      expected =>
        if (expected.peerId != actualExecutorId) {
          throw new IllegalStateException(
            s"FluxNativeQueryRDD partition ${fluxPartition.index} expected executor " +
              s"${expected.peerId}, but Spark scheduled it on $actualExecutorId")
        }
    }
    val effectivePeerEndpointsJson =
      if (
        fluxPartition.totalPartitions == 1 &&
        Option(configuredPeerEndpointsJson).forall(_.trim.isEmpty)
      ) {
        GlutenFluxExecutorService.localPeerEndpointsJson(actualExecutorId).getOrElse {
          logWarning(
            s"FluxNativeQueryRDD: no registered local UCX endpoint for one-peer query " +
              s"$nativeFluxQueryId on executor $actualExecutorId; native fallback will apply")
          configuredPeerEndpointsJson
        }
      } else {
        configuredPeerEndpointsJson
      }
    val fluxPeerSpecJson =
      FluxNativeQueryRDD.buildPeerSpecJson(
        nativeFluxQueryId,
        localPeerId,
        effectivePeerEndpointsJson,
        fluxPartition.index,
        fluxPartition.totalPartitions)
    logInfo(s"FluxNativeQueryRDD: peer spec JSON: $fluxPeerSpecJson")
    val tCreateStart = System.nanoTime()
    val localFragmentSplitInfos = FluxNativeQueryRDD.partitionSplitInfos(
      fragmentSplitInfos,
      fluxPartition.index,
      fluxPartition.totalPartitions,
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
                FluxNativeQueryRDD.materializeFusedBroadcastIteratorImpl(fused)
          }
          logInfo(
            s"FluxNativeQueryRDD: fragment $consumerId has " +
              s"${sortedBySlot.size} fused broadcast(s) at slots " +
              sortedBySlot.map(_.slotIdx).mkString("[", ",", "]"))
        }
    }

    if (localStreamSlots.nonEmpty) {
      val localIterators =
        localInputRDDs.getIterators(fluxPartition.localInputPartitions, context)
      require(
        localIterators.size == localStreamSlots.size,
        s"FLUX local stream iterator count ${localIterators.size} did not match slot count " +
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
          s"FLUX fragment $consumerId has duplicate JVM stream slot(s): " +
            duplicateSlots.toSeq.sorted.mkString("[", ",", "]"))
        jvmStreamSlotIndicesPerFrag(consumerId) = sortedBySlot.map(_._1).toArray
        jvmStreamIteratorsPerFrag(consumerId) = sortedBySlot.map {
          case (_, iterator) =>
            FluxNativeQueryRDD.createTaskContextAwareInputIterator(
              BackendsApiManager.getBackendName,
              iterator,
              context): Object
        }.toArray
        logInfo(
          s"FluxNativeQueryRDD: fragment $consumerId has ${sortedBySlot.size} JVM stream(s) " +
            s"at slots ${sortedBySlot.map(_._1).mkString("[", ",", "]")}")
    }

    // FLUX creates multiple Velox Tasks behind this one Spark task. Allocate the query-level root
    // only after all JVM-side validation and stream materialization has succeeded. The lease keeps
    // ownership until nativeCreateFluxQuery returns; a native parse/plan-conversion failure
    // therefore deletes the root instead of leaking it for the lifetime of the executor.
    val spillRootLease =
      FluxNativeQueryRDD.createSpillRootLease(SparkDirectoryUtil.get().namespace("gluten-spill"))
    val fluxHandle = spillRootLease.handoffAfterCreate {
      spillRootPath =>
        jniWrapper.nativeCreateFluxQuery(
          fragmentPlans,
          runtimeNumDriversPerFragment,
          exchangeSpecsJson.getBytes("UTF-8"),
          fluxPeerSpecJson.getBytes("UTF-8"),
          localFragmentSplitInfos,
          jvmStreamSlotIndicesPerFrag,
          jvmStreamIteratorsPerFrag,
          replicatedCartesianMaxBuildBytes,
          spillRootPath
        )
    }
    @volatile var fluxClosed = false
    def closeFluxHandle(): Long = this.synchronized {
      if (fluxClosed) {
        0L
      } else {
        val tCloseStart = System.nanoTime()
        FluxNativeQueryRDD.closeAfterHoldingMemory(
          () => runtime.memoryManager().hold(),
          () => jniWrapper.nativeCloseFluxQuery(fluxHandle),
          () => fluxClosed = true)
        System.nanoTime() - tCloseStart
      }
    }

    def closeFluxHandleAtTaskCompletion(): Unit = {
      if (!fluxClosed) {
        try {
          closeFluxHandle()
        } catch {
          case NonFatal(e) =>
            logWarning("FluxNativeQueryRDD: failed to close FLUX query at task completion", e)
        }
      }
    }

    def closeFluxHandleAfterSetupFailure(setupFailure: Throwable): Nothing =
      FluxNativeQueryRDD.rethrowAfterCleanup(setupFailure, () => closeFluxHandle())

    // Install the native-handle owner immediately after creation. In particular, no control-plane
    // registration failure may leave a successfully-created coordinator without a completion hook.
    try {
      context.addTaskCompletionListener[Unit](_ => closeFluxHandleAtTaskCompletion())
    } catch {
      case listenerFailure: Throwable => closeFluxHandleAfterSetupFailure(listenerFailure)
    }

    val runId =
      FluxQueryRunId(invocationFluxQueryId, context.stageId(), context.stageAttemptNumber())
    val activeQuery =
      try {
        GlutenFluxExecutorService.registerQuery(
          runId,
          fluxPartition.index,
          fluxPartition.totalPartitions,
          context,
          () => jniWrapper.nativeAbortFluxQuery(fluxHandle))
      } catch {
        case registrationFailure: Throwable =>
          closeFluxHandleAfterSetupFailure(registrationFailure)
      }

    def reportTerminal(state: String): Unit =
      activeQuery.foreach(GlutenFluxExecutorService.reportTerminal(_, state))

    try {
      context.addTaskCompletionListener[Unit] {
        taskContext =>
          // This listener is registered after the close-only safety hook and therefore runs first
          // under Spark's LIFO listener ordering. Keep the old close-before-terminal-report
          // contract; the earlier hook remains an idempotent backstop for registration races.
          closeFluxHandleAtTaskCompletion()
          val terminalState =
            if (taskContext.isInterrupted() || activeQuery.exists(_.isAbortRequested)) {
              FluxPeerState.Aborted
            } else {
              FluxPeerState.Succeeded
            }
          reportTerminal(terminalState)
      }
    } catch {
      case listenerFailure: Throwable =>
        FluxNativeQueryRDD.rethrowAfterCleanup(
          listenerFailure,
          () =>
            activeQuery.foreach(
              query => GlutenFluxExecutorService.reportFailure(query, listenerFailure)),
          () => closeFluxHandle(),
          () => reportTerminal(FluxPeerState.Failed)
        )
    }

    val tCreateDoneStartBegin = System.nanoTime()
    if (activeQuery.exists(query => !query.beginNativeStart())) {
      try {
        closeFluxHandle()
      } finally {
        reportTerminal(FluxPeerState.Aborted)
      }
      throw new TaskKilledException("FLUX task interrupted before native query start")
    }
    try {
      jniWrapper.nativeStartFluxQuery(fluxHandle)
      activeQuery.foreach(_.finishNativeStart(succeeded = true))
    } catch {
      case NonFatal(e) =>
        activeQuery.foreach {
          query =>
            query.finishNativeStart(succeeded = false)
            GlutenFluxExecutorService.reportFailure(query, e)
        }
        try {
          closeFluxHandle()
        } finally {
          reportTerminal(FluxPeerState.Failed)
        }
        throw e
    }
    val tStartDone = System.nanoTime()
    logWarning(
      f"FluxNativeQueryRDD: TIMING " +
        f"nativeCreateFluxQuery=${(tCreateDoneStartBegin - tCreateStart) / 1e6}%.1fms " +
        f"nativeStartFluxQuery=${(tStartDone - tCreateDoneStartBegin) / 1e6}%.1fms")
    if (runtimeTimingProbeEnabled) {
      logWarning(
        FluxNativeQueryRDD.runtimeTimingProbeLine(
          "native_start",
          Seq(
            "queryId" -> FluxNativeQueryRDD.quotedJson(nativeFluxQueryId),
            "sparkPartition" -> fluxPartition.index.toString,
            "sparkPartitions" -> fluxPartition.totalPartitions.toString,
            "fragments" -> fragmentPlans.length.toString,
            "localPeerId" -> FluxNativeQueryRDD.quotedJson(localPeerId),
            "nativeCreateFluxQueryNanos" -> (tCreateDoneStartBegin - tCreateStart).toString,
            "nativeStartFluxQueryNanos" -> (tStartDone - tCreateDoneStartBegin).toString
          )
        ))
    }

    logInfo("FluxNativeQueryRDD: all FLUX fragments started, streaming exchange active")

    val tracker = org.apache.gluten.metrics.TaskWallTimeTracker.get()
    tracker.planBuildNanos += (System.nanoTime() - planBuildStart)

    // 4. Create iterator that pulls batches from FluxQueryCoordinator via JNI.
    val fluxIter = new Iterator[ColumnarBatch] {
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
          nextHandle = jniWrapper.nativeGetFluxOutput(fluxHandle)
        } catch {
          case NonFatal(e) =>
            finished = true
            activeQuery.filterNot(_.isAbortRequested).foreach {
              query => GlutenFluxExecutorService.reportFailure(query, e)
            }
            try {
              closeFluxHandle()
            } catch {
              case NonFatal(closeError) =>
                logWarning(
                  "FluxNativeQueryRDD: failed to close FLUX query after native error",
                  closeError)
            } finally {
              val terminalState =
                if (activeQuery.exists(_.isAbortRequested)) FluxPeerState.Aborted
                else FluxPeerState.Failed
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
          // producers hosted by this executor. Publish output EOS through the FLUX control plane
          // and keep every native coordinator alive until the driver has observed EOS from all
          // peers. A peer failure releases this wait through the existing abort path.
          if (fluxPartition.totalPartitions > 1) {
            val completionStart = System.nanoTime()
            val query = activeQuery.getOrElse {
              val error = new IllegalStateException(
                "Multi-peer FLUX completion requires the FLUX query control plane")
              try {
                closeFluxHandle()
              } catch {
                case NonFatal(closeError) => error.addSuppressed(closeError)
              }
              throw error
            }
            if (!GlutenFluxExecutorService.awaitPeerCompletion(query)) {
              val closeNanos = closeFluxHandle()
              reportTerminal(FluxPeerState.Aborted)
              logWarning(
                f"FluxNativeQueryRDD: peer completion was aborted; " +
                  f"nativeCloseFluxQuery=${closeNanos / 1e6}%.1fms")
              throw new TaskKilledException(
                "FLUX peer completion aborted before every peer reached output EOS")
            }
            logInfo(
              f"FluxNativeQueryRDD: all-peer output EOS acknowledged in " +
                f"${(System.nanoTime() - completionStart) / 1e6}%.1fms")
          }
          // A downstream consumer may retain a zero-copy view of the final native batch after
          // exhausting this iterator. Iceberg's Parquet writer does this until commit(), so keep
          // the coordinator and its output pool alive through the Spark task completion listener.
          // Error and abort paths still close eagerly above.
          val avgMs =
            totalGetOutputNanos.toDouble / math.max(1L, totalGetOutputCalls) / 1e6
          logWarning(
            f"FluxNativeQueryRDD: TIMING " +
              f"nativeGetFluxOutput totalCalls=$totalGetOutputCalls " +
              f"sum=${totalGetOutputNanos / 1e6}%.1fms " +
              f"avg=$avgMs%.2fms " +
              f"min=${minGetOutputNanos / 1e6}%.2fms " +
              f"max=${maxGetOutputNanos / 1e6}%.2fms " +
              f"timeToFirstBatch=${firstBatchNanos / 1e6}%.1fms " +
              f"nativeCloseFluxQuery=deferred")
          if (runtimeTimingProbeEnabled) {
            val safeMinGetOutputNanos =
              if (totalGetOutputCalls == 0) 0L else minGetOutputNanos
            val safeFirstBatchNanos = if (firstBatchNanos < 0) 0L else firstBatchNanos
            logWarning(
              FluxNativeQueryRDD.runtimeTimingProbeLine(
                "native_output_complete",
                Seq(
                  "queryId" -> FluxNativeQueryRDD.quotedJson(nativeFluxQueryId),
                  "sparkPartition" -> fluxPartition.index.toString,
                  "sparkPartitions" -> fluxPartition.totalPartitions.toString,
                  "fragments" -> fragmentPlans.length.toString,
                  "nativeGetFluxOutputCalls" -> totalGetOutputCalls.toString,
                  "nativeGetFluxOutputSumNanos" -> totalGetOutputNanos.toString,
                  "nativeGetFluxOutputMinNanos" -> safeMinGetOutputNanos.toString,
                  "nativeGetFluxOutputMaxNanos" -> maxGetOutputNanos.toString,
                  "timeToFirstBatchNanos" -> safeFirstBatchNanos.toString,
                  "nativeCloseFluxQueryDeferred" -> "true",
                  "outputRows" -> totalOutputRows.toString,
                  "outputBatches" -> totalOutputBatches.toString
                )
              ))
          }
          reportTerminal(FluxPeerState.Succeeded)
          logInfo("FluxNativeQueryRDD: FLUX execution complete, all fragments finished")
          false
        } else {
          true
        }
      }

      override def next(): ColumnarBatch = {
        if (!hasNext) throw new NoSuchElementException("FLUX iterator exhausted")
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
      .wrap(fluxIter)
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

/** Spark partition descriptor for one native FLUX query. */
private[execution] case class FluxPartition(
    override val index: Int,
    totalPartitions: Int,
    peerInfo: Option[FluxPeerInfo],
    localInputPartitions: Seq[Partition])
  extends Partition

private[execution] case class FluxAlignedInputPartition(
    override val index: Int,
    parentPartitions: Array[Partition])
  extends Partition

private[execution] class FluxAlignedInputRDD(
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
        FluxAlignedInputPartition(index, parentIndices.map(parentPartitions))
    }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] = {
    split
      .asInstanceOf[FluxAlignedInputPartition]
      .parentPartitions
      .iterator
      // Creating an RDD iterator may start its runtime (for example, a Python worker). Keep the
      // aligned parents sequential so a coalesced partition does not start every parent at once.
      .flatMap(parentRDD.iterator(_, context))
  }

  override protected def getPreferredLocations(split: Partition): Seq[String] = {
    split
      .asInstanceOf[FluxAlignedInputPartition]
      .parentPartitions
      .flatMap(parentRDD.preferredLocations)
      .distinct
  }

  override protected def clearDependencies(): Unit = {
    super.clearDependencies()
    parentRDD = null
  }
}

private[gluten] case class FluxPeerInfo(
    peerId: String,
    host: String,
    port: Int,
    preferredLocation: String)

/**
 * Owns a newly-created FLUX spill root until the native coordinator has accepted it.
 *
 * [[handoffAfterCreate]] transfers ownership only after its callback returns. Its `finally` closes
 * the lease in both cases: before handoff that recursively deletes the root, while after handoff it
 * is an idempotent no-op because native FluxQueryCoordinator owns cleanup.
 */
final private[execution] class FluxSpillRootLease(val root: File, deleteRoot: File => Unit)
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

private[execution] object FluxNativeQueryRDD extends Logging {

  private val exchangeEndpointPattern =
    ("""(?s)"producerFragmentId"\s*:\s*(-?\d+)\s*,\s*""" +
      """"consumerFragmentId"\s*:\s*(-?\d+)\s*,\s*"exchangeType"\s*:\s*"([^"]+)""").r

  private[execution] def singleGatherRootDriverCounts(
      driverCounts: Array[Int],
      exchangeSpecsJson: String): Array[Int] = {
    val exchanges = exchangeEndpointPattern
      .findAllMatchIn(Option(exchangeSpecsJson).getOrElse(""))
      .map(m => (m.group(1).toInt, m.group(2).toInt, m.group(3)))
      .toSeq
    val producerIds = exchanges.iterator.map(_._1).toSet
    val rootIds = exchanges.iterator.map(_._2).filterNot(producerIds.contains).toSet
    val singleGatherRootIds = rootIds.filter {
      rootId =>
        val inbound = exchanges.filter(_._2 == rootId)
        inbound.nonEmpty && inbound.forall(_._3 == "SINGLE")
    }
    val adjusted = driverCounts.clone()
    singleGatherRootIds.foreach {
      rootId =>
        if (rootId >= 0 && rootId < adjusted.length) {
          adjusted(rootId) = 1
        }
    }
    adjusted
  }

  final private case class InvocationKey(
      queryId: String,
      stageId: Int,
      stageAttemptNumber: Int,
      taskAttemptId: Long)

  // RDD references normally retain object identity when Spark deserializes a task, but using a
  // process-wide key also covers independently-deserialized references to the same physical FLUX
  // RDD. The task-attempt component prevents concurrent Spark retries from sharing an ordinal.
  private val invocationOrdinals = new ConcurrentHashMap[InvocationKey, AtomicInteger]()

  private[execution] def nextInvocationQueryId(queryId: String, context: TaskContext): String = {
    val key = invocationKey(queryId, context)
    val proposed = new AtomicInteger(0)
    val existing = invocationOrdinals.putIfAbsent(key, proposed)
    val ordinal =
      if (existing == null) {
        try {
          context.addTaskCompletionListener[Unit](_ => invocationOrdinals.remove(key, proposed))
        } catch {
          case listenerFailure: Throwable =>
            invocationOrdinals.remove(key, proposed)
            throw listenerFailure
        }
        proposed.getAndIncrement()
      } else {
        existing.getAndIncrement()
      }
    if (ordinal == 0) queryId else s"$queryId-invocation-$ordinal"
  }

  private[execution] def clearInvocationQueryIds(queryId: String, context: TaskContext): Unit = {
    invocationOrdinals.remove(invocationKey(queryId, context))
  }

  private def invocationKey(queryId: String, context: TaskContext): InvocationKey =
    InvocationKey(queryId, context.stageId(), context.stageAttemptNumber(), context.taskAttemptId())

  /**
   * Keep native memory pools alive before destroying an FLUX coordinator. Returned zero-copy
   * batches
   * can outlive the coordinator on both success and error paths; closing first would leave their
   * buffers pointing at an already-destroyed MemoryPool.
   */
  private[execution] def closeAfterHoldingMemory(
      holdMemory: () => Unit,
      closeNativeQuery: () => Unit,
      markClosed: () => Unit): Unit = {
    holdMemory()
    try {
      closeNativeQuery()
    } finally {
      markClosed()
    }
  }

  /**
   * Native FLUX drivers pull JVM stream inputs from native worker threads, outside Spark's task
   * thread. Bind the owning task context and its context classloader around the complete bridge
   * call so the delegated iterator can load task classes and ColumnarBatch handle conversion uses
   * the task's live resource registry.
   */
  private[execution] def createTaskContextAwareInputIterator(
      backendName: String,
      delegated: JIterator[ColumnarBatch],
      taskContext: TaskContext): ColumnarBatchInIterator = {
    val ownerContextClassLoader = Thread.currentThread().getContextClassLoader
    require(
      ownerContextClassLoader != null,
      "FLUX input callback bridge requires a non-null owner thread context classloader")
    new ColumnarBatchInIterator(backendName, delegated) {
      private def runWithOwnerThreadContext[T](body: => T): T = {
        val callbackThread = Thread.currentThread()
        val previousContextClassLoader = callbackThread.getContextClassLoader
        callbackThread.setContextClassLoader(ownerContextClassLoader)
        try {
          TaskResources.runWithTaskContext(taskContext)(body)
        } finally {
          callbackThread.setContextClassLoader(previousContextClassLoader)
        }
      }

      override def hasNext(): Boolean = {
        runWithOwnerThreadContext {
          super.hasNext()
        }
      }

      override def next(): Long = {
        runWithOwnerThreadContext {
          super.next()
        }
      }
    }
  }

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

  private[execution] def createSpillRootLease(namespace: Namespace): FluxSpillRootLease = {
    val root = namespace.mkChildDirRoundRobin(s"flux-${UUID.randomUUID()}")
    new FluxSpillRootLease(
      root,
      file =>
        if (!FileUtils.deleteQuietly(file) && file.exists()) {
          logWarning(s"FluxNativeQueryRDD: failed to delete unowned FLUX spill root $file")
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
    new FluxAlignedInputRDD(rdd, parentGroups)
  }

  /** Opt-in: raise cuDF parquet chunk/pass read limits for FLUX table scans. */
  val largeParquetScanChunksKey: String = "spark.gluten.mpp.largeParquetScanChunks"

  val runtimeTimingProbeKey: String = "spark.gluten.mpp.runtimeTimingProbe"

  // Runtime resources are task-scoped and keyed by extraConf. Forward settings that may be
  // changed with SET between benchmark queries instead of freezing their SparkConf startup value.
  private[execution] val queryScopedNativeConf: Seq[(String, String)] = Seq(
    "spark.gluten.sql.columnar.backend.velox.cudf.groupbyStreamingMaxDistinctKeys" -> "0",
    "spark.gluten.sql.columnar.backend.velox.cudf.partialIdentityAggregation" -> "false"
  )

  private[execution] def queryScopedNativeConfSnapshot: Map[String, String] = {
    val conf = SQLConf.get
    queryScopedNativeConf.map {
      case (key, defaultValue) => key -> conf.getConfString(key, defaultValue)
    }.toMap
  }

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
    s"[FLUX_RUNTIME_PROBE] {$fieldJson}"
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
   * [[VeloxBroadcastBuildSideRDD.genBroadcastBuildSideIterator]] but lives here to keep the FLUX
   * RDD
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
