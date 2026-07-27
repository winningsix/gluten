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
package org.apache.spark.shuffle

import org.apache.gluten.metrics.TaskWallTimeTracker
import org.apache.gluten.shuffle.SupportsColumnarShuffle

import org.apache.spark.{ShuffleDependency, SparkConf, SparkEnv, SparkException, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler.MapStatus
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.Utils

import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong, AtomicReference}

import scala.util.control.NonFatal

class UcxColumnarShuffleManager(conf: SparkConf, isDriver: Boolean)
  extends ShuffleManager
  with PipelinedShuffleSchedulingProvider
  with PipelinedShuffleControlPlane
  with SupportsColumnarShuffle
  with Logging {

  private lazy val coordinator: UcxShuffleCoordinator =
    UcxShuffleCoordinator.getOrCreate(conf, isDriver)
  private lazy val nativeBridge: UcxShuffleNativeBridge =
    UcxShuffleNativeBridge.getOrCreate(conf)

  logInfo(
    s"Using ${UcxColumnarShuffleManager.ClassName} " +
      s"(isDriver=$isDriver, dataPlane=${UcxColumnarShuffleManager.DataPlaneStatus})")

  override def registerShuffle[K, V, C](
      shuffleId: Int,
      dependency: ShuffleDependency[K, V, C]): ShuffleHandle = {
    dependency match {
      case pipelined: PipelinedColumnarShuffleDependency[_, _, _] =>
        val typed = pipelined.asInstanceOf[PipelinedColumnarShuffleDependency[K, V, C]]
        logInfo(
          s"Registering UCX incremental columnar shuffle shuffleId=$shuffleId " +
            s"partitions=${typed.partitioner.numPartitions} " +
            s"partitioning=${typed.nativePartitioning.getShortName}")
        coordinator.registerShuffle(
          shuffleId,
          typed.rdd.partitions.length,
          typed.partitioner.numPartitions)
        new UcxColumnarShuffleHandle[K, V, C](shuffleId, typed)
      case other =>
        throw new SparkException(
          s"${UcxColumnarShuffleManager.ClassName} only supports " +
            s"${classOf[PipelinedColumnarShuffleDependency[_, _, _]].getName}; got " +
            s"${other.getClass.getName}. Use spark.shuffle.manager for regular shuffles and " +
            s"spark.shuffle.manager.incremental=${UcxColumnarShuffleManager.ClassName} only " +
            s"with Gluten columnar pipelined shuffle.")
    }
  }

  override def getWriter[K, V](
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext,
      metrics: ShuffleWriteMetricsReporter): ShuffleWriter[K, V] = {
    handle match {
      case ucx: UcxColumnarShuffleHandle[_, _, _] =>
        new UcxColumnarShuffleWriter[K, V](
          ucx.asInstanceOf[UcxColumnarShuffleHandle[K, V, Any]],
          mapId,
          context,
          metrics,
          coordinator,
          nativeBridge,
          conf)
      case other =>
        throw new SparkException(
          s"${UcxColumnarShuffleManager.ClassName} received non-UCX shuffle handle " +
            s"${other.getClass.getName} for writer shuffleId=${other.shuffleId}")
    }
  }

  override def wrapShuffleMapTaskInput(
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext)(
      createInput: => Iterator[_]): Iterator[_] = {
    handle match {
      case ucx: UcxColumnarShuffleHandle[_, _, _] =>
        NativeUcxShuffleExecution.wrapShuffleMapTaskInput(
          ucx, mapId, context)(createInput)
      case other =>
        throw new SparkException(
          s"${UcxColumnarShuffleManager.ClassName} received non-UCX shuffle handle " +
            s"${other.getClass.getName} for map input shuffleId=${other.shuffleId}")
    }
  }

  override def getReader[K, C](
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] = {
    handle match {
      case ucx: UcxColumnarShuffleHandle[_, _, _] =>
        new UcxColumnarShuffleReader[K, C](
          ucx.asInstanceOf[UcxColumnarShuffleHandle[K, Any, C]],
          startMapIndex,
          endMapIndex,
          startPartition,
          endPartition,
          context,
          metrics,
          coordinator,
          nativeBridge)
      case other =>
        throw new SparkException(
          s"${UcxColumnarShuffleManager.ClassName} received non-UCX shuffle handle " +
            s"${other.getClass.getName} for reader shuffleId=${other.shuffleId}")
    }
  }

  override def unregisterShuffle(shuffleId: Int): Boolean = {
    coordinator.unregisterShuffle(shuffleId)
    true
  }

  override def schedulingRequirements(
      group: PipelinedShuffleGroupMetadata): PipelinedGroupSchedulingRequirements = {
    PipelinedGroupSchedulingRequirements(
      residencyPolicy = ReaderResidencyWithElasticProducers(
        minProducerTasksPerStage =
          conf.getInt(UcxColumnarShuffleManager.ProducerMinRunningTasksPerStageConf, 1),
        maxProducerTasksPerStage =
          conf.getOption(UcxColumnarShuffleManager.ProducerMaxRunningTasksPerStageConf)
            .map(_.toInt)
            .filter(_ > 0)))
  }

  override def registerPipelinedShuffleGroup(group: PipelinedShuffleGroupMetadata): Unit = {
    coordinator.registerPipelinedShuffleGroup(group)
  }

  override def admitPipelinedShuffleGroup(groupAttemptId: String): Unit = {
    coordinator.admitPipelinedShuffleGroup(groupAttemptId)
  }

  override def completePipelinedShuffleGroup(groupAttemptId: String): Unit = {
    coordinator.completePipelinedShuffleGroup(groupAttemptId)
  }

  override def abortPipelinedShuffleGroup(groupAttemptId: String, reason: String): Unit = {
    coordinator.abortPipelinedShuffleGroup(groupAttemptId, reason)
  }

  override def shuffleBlockResolver: ShuffleBlockResolver = {
    throw new UnsupportedOperationException(
      s"${UcxColumnarShuffleManager.ClassName} is incremental and does not expose " +
        "Spark shuffle blocks")
  }

  override def stop(): Unit = {
    UcxShuffleCoordinator.stop()
    UcxShuffleNativeBridge.stop()
  }
}

private[spark] object UcxColumnarShuffleManager {
  val ClassName: String = "org.apache.spark.shuffle.UcxColumnarShuffleManager"
  val DataPlaneStatus: String = "velox-native-ucx-exchange"
  val ProducerMinRunningTasksPerStageConf: String =
    "spark.scheduler.pipelined.group.producer.minRunningTasksPerStage"
  val ProducerMaxRunningTasksPerStageConf: String =
    "spark.scheduler.pipelined.group.producer.maxRunningTasksPerStage"
  val ReaderEndpointWaitMsConf: String = "spark.gluten.ucx.shuffle.reader.endpointWaitMs"
  val ReaderEndpointPollMsConf: String = "spark.gluten.ucx.shuffle.reader.endpointPollMs"
  val ReaderStatePollMsConf: String = "spark.gluten.ucx.shuffle.reader.statePollMs"
  val ReaderDataWaitAfterWritersFinishedMsConf: String =
    "spark.gluten.ucx.shuffle.reader.dataWaitAfterWritersFinishedMs"
  val ReaderSlowNextBatchLogMsConf: String =
    "spark.gluten.ucx.shuffle.reader.slowNextBatchLogMs"
  val WriterWaitForReadersReadyConf: String =
    "spark.gluten.ucx.shuffle.writer.waitForReadersReady"
  val WriterWaitForGroupReadersReadyConf: String =
    "spark.gluten.ucx.shuffle.writer.waitForGroupReadersReady"
  val WriterReadersReadyWaitMsConf: String =
    "spark.gluten.ucx.shuffle.writer.readersReadyWaitMs"
  val WriterReadersReadyPollMsConf: String =
    "spark.gluten.ucx.shuffle.writer.readersReadyPollMs"
  val WriterMaxActiveTasksPerShuffleConf: String =
    "spark.gluten.ucx.shuffle.writer.maxActiveTasksPerShuffle"
  val WriterCreditWaitMsConf: String =
    "spark.gluten.ucx.shuffle.writer.creditWaitMs"
  val WriterCreditPollMsConf: String =
    "spark.gluten.ucx.shuffle.writer.creditPollMs"
  val NativeWriterNoMoreDataPollMsConf: String =
    "spark.gluten.ucx.shuffle.native.writer.noMoreDataPollMs"
  val NativeWriterNoMoreDataWaitMsConf: String =
    "spark.gluten.ucx.shuffle.native.writer.noMoreDataWaitMs"
  val NativeWriterStateReportMsConf: String =
    "spark.gluten.ucx.shuffle.native.writer.stateReportMs"

  def dataPlaneNotImplemented(operation: String, handle: ShuffleHandle): SparkException = {
    new SparkException(
      s"Gluten UCX incremental shuffle reached $operation for shuffle ${handle.shuffleId}, " +
        s"but the active native data plane does not support that operation.")
  }
}

private[spark] class UcxColumnarShuffleHandle[K, V, C](
    shuffleId: Int,
    dependency: PipelinedColumnarShuffleDependency[K, V, C])
  extends BaseShuffleHandle[K, V, C](shuffleId, dependency)

private[spark] class UcxColumnarShuffleWriter[K, V](
    handle: UcxColumnarShuffleHandle[K, V, _],
    mapId: Long,
    context: TaskContext,
    metrics: ShuffleWriteMetricsReporter,
    coordinator: UcxShuffleCoordinator,
    nativeBridge: UcxShuffleNativeBridge,
    conf: SparkConf)
  extends ShuffleWriter[K, V]
  with Logging {

  private val columnarDependency =
    handle.dependency.asInstanceOf[ColumnarShuffleDependencyLike]
  private val numPartitions = handle.dependency.partitioner.numPartitions
  private val ucxMapId = context.partitionId().toLong
  private val attemptId = context.taskAttemptId()
  @volatile private var endpointRegistered = false
  @volatile private var nativeWriterHandle = -1L
  @volatile private var writerCreditAcquired = false
  @volatile private var writerFinishedMarked = false
  @volatile private var nativeWriterMetricsReported = false
  @volatile private var nativeWriterFinalDrainStateReported = false
  @volatile private var nativeWriterNoMoreDataPoller: Thread = _
  @volatile private var nativeWriterNoMoreDataPollerStopRequested = false
  @volatile private var stopped = false

  context.addTaskCompletionListener[Unit] {
    taskContext =>
      if (taskContext.isFailed() || taskContext.isInterrupted()) {
        stopNativeWriterNoMoreDataPoller()
        releaseWriterCreditIfNeeded()
        coordinator.abortShuffle(
          handle.shuffleId,
          s"UCX shuffle writer task ended unsuccessfully: shuffleId=${handle.shuffleId} " +
            s"ucxMapId=$ucxMapId sparkMapId=$mapId attemptId=$attemptId " +
            s"failed=${taskContext.isFailed()} " +
            s"interrupted=${taskContext.isInterrupted()}"
        )
      }
  }

  override def write(records: Iterator[Product2[K, V]]): Unit = {
    val tracker = TaskWallTimeTracker.get()
    if (NativeUcxShuffleExecution.enabled(conf)) {
      writeNativeExchange(records, tracker)
      return
    }
    acquireWriterCreditIfNeeded(tracker)
    val endpoint = registerEndpointIfNeeded()
    val openStartNs = System.nanoTime()
    try {
      nativeWriterHandle = nativeBridge.openWriter(
        endpoint,
        numPartitions,
        columnarDependency.nativePartitioning.getShortName,
        GlutenShuffleUtils.getStartPartitionId(
          columnarDependency.nativePartitioning,
          context.partitionId())
      )
    } finally {
      tracker.ucxWriterOpenNanos += System.nanoTime() - openStartNs
    }
    waitForReadersReadyIfNeeded(tracker)
    records.foreach {
      record =>
        val writeStartNs = System.nanoTime()
        try {
          nativeBridge.writeBatch(
            nativeWriterHandle,
            partitionId(record._1),
            columnarBatch(record._2))
        } finally {
          tracker.ucxWriterWriteBatchNanos += System.nanoTime() - writeStartNs
          tracker.ucxWriterWriteBatchCalls += 1
        }
    }
  }

  private def writeNativeExchange(
      records: Iterator[Product2[K, V]],
      tracker: TaskWallTimeTracker): Unit = {
    acquireWriterCreditIfNeeded(tracker)
    val endpoint = registerEndpointIfNeeded()
    nativeWriterNoMoreDataPoller = startNativeWriterNoMoreDataPoller(endpoint)
    var nativeExchangeCompleted = false
    try {
      waitForReadersReadyIfNeeded(tracker)
      val partitioning = columnarDependency.nativePartitioning.getShortName
      val writerContext = NativeUcxShuffleWriterContext(
        shuffleId = handle.shuffleId,
        mapId = ucxMapId,
        attemptId = attemptId,
        nativeTaskId = endpoint.nativeTaskId,
        numPartitions = numPartitions,
        partitioning = partitioning,
        startPartitionId = GlutenShuffleUtils.getStartPartitionId(
          columnarDependency.nativePartitioning,
          context.partitionId()),
        dropFirstColumn =
          partitioning == GlutenShuffleUtils.HashPartitioningShortName &&
            columnarDependency.nativePartitioning.getKeyIndices == null,
        partitionKeyIndices =
          Option(columnarDependency.nativePartitioning.getKeyIndices)
            .map(_.toSeq)
            .getOrElse(Seq.empty)
      )
      logInfo(
        s"Driving Velox native UCX shuffle producer shuffleId=${handle.shuffleId} " +
          s"ucxMapId=$ucxMapId sparkMapId=$mapId attemptId=$attemptId " +
          s"nativeTaskId=${endpoint.nativeTaskId} " +
          s"partitions=$numPartitions partitioning=$partitioning")
      NativeUcxShuffleExecution.withWriterContext(writerContext) {
        records.foreach(_ => tracker.ucxWriterWriteBatchCalls += 1)
      }
      nativeExchangeCompleted = true
    } finally {
      reportNativeWriterMetrics(endpoint)
      reportFinalNativeWriterNoMoreDataIfReady(endpoint)
      // The native producer can remain detached after the Spark task returns while UCX drains
      // its output queue. Keep its telemetry alive until the transport reports that drain.
      if (!nativeExchangeCompleted) {
        stopNativeWriterNoMoreDataPoller()
      }
    }
  }

  private def reportNativeWriterMetrics(endpoint: UcxShuffleEndpoint): Unit = synchronized {
    if (nativeWriterMetricsReported) {
      return
    }
    nativeBridge.writerRuntimeStats(endpoint.nativeTaskId) match {
      case Some(stats) =>
        def addExchangeMetric(name: String, value: Long): Unit = {
          if (value >= 0) {
            columnarDependency.metrics.get(name).foreach(_.add(value))
          }
        }

        addExchangeMetric("dataSize", stats.totalBytesSent)
        addExchangeMetric("numInputRows", stats.totalRowsSent)
        addExchangeMetric("inputBatches", stats.totalPagesSent)
        if (stats.totalBytesSent >= 0) {
          metrics.incBytesWritten(stats.totalBytesSent)
        }
        if (stats.totalRowsSent >= 0) {
          metrics.incRecordsWritten(stats.totalRowsSent)
        }
        nativeWriterMetricsReported = true
        logInfo(
          s"Reported native UCX shuffle writer SQL metrics " +
            s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
            s"sparkMapId=$mapId attemptId=$attemptId " +
            s"nativeTaskId=${endpoint.nativeTaskId} " +
            s"bytes=${stats.totalBytesSent} rows=${stats.totalRowsSent} " +
            s"pages=${stats.totalPagesSent}")
      case None =>
        logWarning(
          s"Native UCX shuffle writer runtime stats unavailable for SQL metrics " +
            s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
            s"sparkMapId=$mapId attemptId=$attemptId " +
            s"nativeTaskId=${endpoint.nativeTaskId}")
    }
  }

  override def stop(success: Boolean): Option[MapStatus] = {
    if (!stopped) {
      stopped = true
      if (nativeWriterHandle != -1L) {
        val closeStartNs = System.nanoTime()
        try {
          nativeBridge.closeWriter(nativeWriterHandle, success)
        } finally {
          TaskWallTimeTracker.get().ucxWriterCloseNanos += System.nanoTime() - closeStartNs
          nativeWriterHandle = -1L
        }
      }
      releaseWriterCreditIfNeeded()
      if (success) {
        markWriterFinishedIfNeeded("shuffle writer stop")
      } else {
        stopNativeWriterNoMoreDataPoller()
        coordinator.abortShuffle(
          handle.shuffleId,
          s"UCX shuffle writer stopped unsuccessfully: shuffleId=${handle.shuffleId} " +
            s"ucxMapId=$ucxMapId sparkMapId=$mapId attemptId=$attemptId")
      }
    }
    Some(MapStatus(SparkEnv.get.blockManager.shuffleServerId, getPartitionLengths(), mapId))
  }

  override def getPartitionLengths(): Array[Long] = Array.fill(numPartitions)(0L)

  private def acquireWriterCreditIfNeeded(tracker: TaskWallTimeTracker): Unit = {
    if (!writerCreditEnabled || writerCreditAcquired) {
      return
    }
    synchronized {
      if (writerCreditAcquired) {
        return
      }
      val timeoutMs =
        conf.getLong(UcxColumnarShuffleManager.WriterCreditWaitMsConf, 300000L)
      val pollMs =
        math.max(1L, conf.getLong(UcxColumnarShuffleManager.WriterCreditPollMsConf, 10L))
      val waitStartNs = System.nanoTime()
      try {
        val credit =
          coordinator.waitUntilWriterCreditAvailable(
            handle.shuffleId,
            ucxMapId,
            attemptId,
            context.stageId(),
            context.stageAttemptNumber(),
            timeoutMs,
            pollMs)
        writerCreditAcquired = true
        val elapsedMs = (System.nanoTime() - waitStartNs) / 1000000L
        logInfo(
            s"UCX shuffle writer credit gate opened shuffleId=${handle.shuffleId} " +
            s"ucxMapId=$ucxMapId sparkMapId=$mapId attemptId=$attemptId " +
            s"stage=${context.stageId()}.${context.stageAttemptNumber()} " +
            s"elapsedMs=$elapsedMs " +
            s"groupId=${credit.groupId} " +
            s"shuffleActive=${credit.activeShuffleWriters}/${credit.maxShuffleWriters}")
      } finally {
        tracker.ucxWriterCreditWaitNanos += System.nanoTime() - waitStartNs
      }
    }
  }

  private def releaseWriterCreditIfNeeded(): Unit = {
    if (!writerCreditAcquired) {
      return
    }
    synchronized {
      if (!writerCreditAcquired) {
        return
      }
      try {
        coordinator.releaseWriterCredit(handle.shuffleId, ucxMapId, attemptId)
      } catch {
        case NonFatal(e) =>
          logWarning(
            s"Failed to release UCX shuffle writer credit shuffleId=${handle.shuffleId} " +
              s"ucxMapId=$ucxMapId sparkMapId=$mapId attemptId=$attemptId",
            e)
      } finally {
        writerCreditAcquired = false
      }
    }
  }

  private def writerCreditEnabled: Boolean = {
    conf.getInt(UcxColumnarShuffleManager.WriterMaxActiveTasksPerShuffleConf, 0) > 0
  }

  private def markWriterFinishedIfNeeded(reason: String): Unit = synchronized {
    if (writerFinishedMarked) {
      return
    }
    val marked = coordinator.markWriterFinished(handle.shuffleId, ucxMapId, attemptId)
    writerFinishedMarked = true
    logInfo(
      s"Marked UCX shuffle writer finished shuffleId=${handle.shuffleId} " +
        s"ucxMapId=$ucxMapId sparkMapId=$mapId attemptId=$attemptId " +
        s"reason=$reason marked=$marked")
  }

  private def reportNativeWriterState(
      endpoint: UcxShuffleEndpoint,
      noMoreData: Boolean,
      stats: Option[UcxShuffleNativeWriterRuntimeStats] = None): UcxNativeWriterStateResponse =
    synchronized {
      val runtimeStats = stats.getOrElse(
        UcxShuffleNativeWriterRuntimeStats(
          noMoreData = noMoreData,
          finished = false,
          queuedBytes = -1L,
          queuedPages = -1L,
          totalBytesSent = -1L,
          totalRowsSent = -1L,
          totalPagesSent = -1L,
          averageBufferTimeMs = -1L,
          blocked = false))
      val response =
        coordinator.reportNativeWriterState(
          UcxNativeWriterState(
            shuffleId = handle.shuffleId,
            mapId = ucxMapId,
            attemptId = attemptId,
            nativeTaskId = endpoint.nativeTaskId,
            noMoreData = noMoreData || runtimeStats.noMoreData,
            queuedBytes = runtimeStats.queuedBytes,
            blocked = runtimeStats.blocked,
            timestampMs = System.currentTimeMillis(),
            finished = runtimeStats.finished,
            queuedPages = runtimeStats.queuedPages,
            totalBytesSent = runtimeStats.totalBytesSent,
            totalRowsSent = runtimeStats.totalRowsSent,
            totalPagesSent = runtimeStats.totalPagesSent,
            averageBufferTimeMs = runtimeStats.averageBufferTimeMs))
      if (response.writerFinishedMarked) {
        writerFinishedMarked = true
      }
      if (response.writerCreditReleased) {
        writerCreditAcquired = false
      }
      if (
        response.accepted &&
        (noMoreData || runtimeStats.noMoreData) &&
        runtimeStats.finished
      ) {
        nativeWriterFinalDrainStateReported = true
      }
      response
    }

  private def startNativeWriterNoMoreDataPoller(endpoint: UcxShuffleEndpoint): Thread = {
    val pollMs = math.max(
      1L,
      conf.getLong(
        UcxColumnarShuffleManager.NativeWriterNoMoreDataPollMsConf,
        conf.getLong(UcxColumnarShuffleManager.WriterCreditPollMsConf, 10L)))
    val timeoutMs =
      conf.getLong(
        UcxColumnarShuffleManager.NativeWriterNoMoreDataWaitMsConf,
        conf.getLong(UcxColumnarShuffleManager.WriterCreditWaitMsConf, 300000L))
    val stateReportMs =
      conf.getLong(UcxColumnarShuffleManager.NativeWriterStateReportMsConf, 1000L)
    val detachedStatePollMs = math.max(1L, if (stateReportMs > 0) stateReportMs else 1000L)
    nativeWriterNoMoreDataPollerStopRequested = false
    val poller = new Thread(
      new Runnable {
        override def run(): Unit = {
          val startNs = System.nanoTime()
          var nextLogMs = 5000L
          var nextStateReportMs = stateReportMs
          var observedNoMoreData = false
          var lastStats: Option[UcxShuffleNativeWriterRuntimeStats] = None
          try {
            while (!nativeWriterNoMoreDataPollerStopRequested &&
                !Thread.currentThread().isInterrupted) {
              val stats = nativeBridge.writerRuntimeStats(endpoint.nativeTaskId)
              if (stats.nonEmpty) {
                lastStats = stats
              }
              val noMoreData =
                stats
                  .map(_.noMoreData)
                  .getOrElse(nativeBridge.writerNoMoreData(endpoint.nativeTaskId))
              val elapsedMs = (System.nanoTime() - startNs) / 1000000L
              if (noMoreData && nativeWriterFinalDrainStateReported) {
                return
              }
              if (noMoreData && !observedNoMoreData) {
                val response = reportNativeWriterState(endpoint, noMoreData = true, stats)
                observedNoMoreData = true
                logInfo(
                  s"Observed native UCX shuffle writer noMoreData " +
                    s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
                    s"sparkMapId=$mapId attemptId=$attemptId " +
                    s"nativeTaskId=${endpoint.nativeTaskId} elapsedMs=$elapsedMs " +
                    s"queuedBytes=${stats.map(_.queuedBytes)} " +
                    s"finished=${stats.map(_.finished)} " +
                    s"blocked=${stats.map(_.blocked)} " +
                    s"accepted=${response.accepted} groupId=${response.groupId} " +
                    s"groupState=${response.groupState} " +
                    s"writerFinishedMarked=${response.writerFinishedMarked} " +
                    s"writerCreditReleased=${response.writerCreditReleased} " +
                    s"reason=${response.reason}")
                if (!response.accepted || stats.exists(_.finished)) {
                  return
                }
                nextStateReportMs = elapsedMs + detachedStatePollMs
              } else if (observedNoMoreData && stats.isEmpty) {
                val drainedStats =
                  lastStats
                    .map(
                      _.copy(
                        noMoreData = true,
                        finished = true,
                        queuedBytes = 0L,
                        queuedPages = 0L,
                        blocked = false))
                    .orElse(
                      Some(
                        UcxShuffleNativeWriterRuntimeStats(
                          noMoreData = true,
                          finished = true,
                          queuedBytes = 0L,
                          queuedPages = 0L,
                          totalBytesSent = -1L,
                          totalRowsSent = -1L,
                          totalPagesSent = -1L,
                          averageBufferTimeMs = -1L,
                          blocked = false)))
                val response =
                  reportNativeWriterState(endpoint, noMoreData = true, drainedStats)
                logInfo(
                  s"Observed drained detached native UCX shuffle writer " +
                    s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
                    s"sparkMapId=$mapId attemptId=$attemptId " +
                    s"nativeTaskId=${endpoint.nativeTaskId} elapsedMs=$elapsedMs " +
                    s"accepted=${response.accepted} groupId=${response.groupId} " +
                    s"groupState=${response.groupState} reason=${response.reason}")
                return
              } else if (elapsedMs >= nextStateReportMs &&
                  (stateReportMs > 0 || observedNoMoreData)) {
                val response = reportNativeWriterState(endpoint, noMoreData = false, stats)
                if (!response.accepted) {
                  logWarning(
                    s"Native UCX shuffle writer state rejected by transport coordinator " +
                      s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
                      s"sparkMapId=$mapId attemptId=$attemptId " +
                      s"nativeTaskId=${endpoint.nativeTaskId} " +
                      s"groupId=${response.groupId} groupState=${response.groupState} " +
                      s"reason=${response.reason}")
                  return
                }
                if (observedNoMoreData && stats.exists(_.finished)) {
                  logInfo(
                    s"Observed finished detached native UCX shuffle writer " +
                      s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
                      s"sparkMapId=$mapId attemptId=$attemptId " +
                      s"nativeTaskId=${endpoint.nativeTaskId} elapsedMs=$elapsedMs " +
                      s"queuedBytes=${stats.map(_.queuedBytes)}")
                  return
                }
                nextStateReportMs = elapsedMs +
                  (if (observedNoMoreData) detachedStatePollMs else stateReportMs)
              }
              if (!observedNoMoreData && elapsedMs >= nextLogMs) {
                logInfo(
                  s"Waiting for native UCX shuffle writer noMoreData " +
                    s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
                    s"sparkMapId=$mapId attemptId=$attemptId " +
                    s"nativeTaskId=${endpoint.nativeTaskId} elapsedMs=$elapsedMs")
                nextLogMs += 5000L
              }
              if (timeoutMs > 0 && elapsedMs >= timeoutMs) {
                logWarning(
                  s"Timed out after ${timeoutMs}ms waiting for native UCX shuffle writer " +
                    s"noMoreData shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
                    s"sparkMapId=$mapId attemptId=$attemptId " +
                    s"nativeTaskId=${endpoint.nativeTaskId}")
                return
              }
              Thread.sleep(if (observedNoMoreData) detachedStatePollMs else pollMs)
            }
          } catch {
            case _: InterruptedException =>
              Thread.currentThread().interrupt()
            case NonFatal(e) =>
              logWarning(
                s"Native UCX shuffle writer noMoreData poller failed " +
                  s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
                  s"sparkMapId=$mapId attemptId=$attemptId " +
                  s"nativeTaskId=${endpoint.nativeTaskId}",
                e)
          }
        }
      },
      s"ucx-shuffle-writer-no-more-data-${handle.shuffleId}-$ucxMapId"
    )
    poller.setDaemon(true)
    poller.start()
    poller
  }

  private def reportFinalNativeWriterNoMoreDataIfReady(endpoint: UcxShuffleEndpoint): Unit = {
    if (writerFinishedMarked && !writerCreditAcquired) {
      return
    }
    try {
      val stats = nativeBridge.writerRuntimeStats(endpoint.nativeTaskId)
      val noMoreData =
        stats.map(_.noMoreData).getOrElse(nativeBridge.writerNoMoreData(endpoint.nativeTaskId))
      if (noMoreData) {
        val response = reportNativeWriterState(endpoint, noMoreData = true, stats)
        logInfo(
          s"Reported final native UCX shuffle writer noMoreData " +
            s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
            s"sparkMapId=$mapId attemptId=$attemptId " +
            s"nativeTaskId=${endpoint.nativeTaskId} accepted=${response.accepted} " +
            s"queuedBytes=${stats.map(_.queuedBytes)} " +
            s"finished=${stats.map(_.finished)} " +
            s"blocked=${stats.map(_.blocked)} " +
            s"groupId=${response.groupId} groupState=${response.groupState} " +
            s"writerFinishedMarked=${response.writerFinishedMarked} " +
            s"writerCreditReleased=${response.writerCreditReleased} " +
            s"reason=${response.reason}")
      }
    } catch {
      case NonFatal(e) =>
        logWarning(
          s"Failed to report final native UCX shuffle writer state " +
            s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
            s"sparkMapId=$mapId attemptId=$attemptId " +
            s"nativeTaskId=${endpoint.nativeTaskId}",
          e)
    }
  }

  private def stopNativeWriterNoMoreDataPoller(): Unit = {
    val poller = nativeWriterNoMoreDataPoller
    nativeWriterNoMoreDataPoller = null
    nativeWriterNoMoreDataPollerStopRequested = true
    if (poller != null && poller.isAlive) {
      poller.interrupt()
      try {
        poller.join(500L)
      } catch {
        case _: InterruptedException =>
          Thread.currentThread().interrupt()
      }
      if (poller.isAlive) {
        logWarning(
          s"Native UCX shuffle writer noMoreData poller is still stopping " +
            s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId " +
            s"sparkMapId=$mapId attemptId=$attemptId")
      }
    }
  }

  private def waitForReadersReadyIfNeeded(tracker: TaskWallTimeTracker): Unit = {
    if (!conf.getBoolean(UcxColumnarShuffleManager.WriterWaitForReadersReadyConf, true)) {
      logWarning(
        s"Skipping UCX pipelined writer readers-ready wait for shuffleId=${handle.shuffleId} " +
          s"mapId=$mapId because " +
          s"${UcxColumnarShuffleManager.WriterWaitForReadersReadyConf}=false")
      return
    }
    val timeoutMs =
      conf.getLong(UcxColumnarShuffleManager.WriterReadersReadyWaitMsConf, 300000L)
    val pollMs =
      math.max(1L, conf.getLong(UcxColumnarShuffleManager.WriterReadersReadyPollMsConf, 10L))
    val requireGroupReadersReady =
      conf.getBoolean(UcxColumnarShuffleManager.WriterWaitForGroupReadersReadyConf, false)
    val waitStartNs = System.nanoTime()
    try {
      val state =
        coordinator.waitUntilPipelinedReadersReady(
          handle.shuffleId,
          timeoutMs,
          pollMs,
          requireGroupReadersReady)
      val elapsedMs = (System.nanoTime() - waitStartNs) / 1000000L
      logInfo(
        s"UCX pipelined writer readers-ready gate opened shuffleId=${handle.shuffleId} " +
          s"ucxMapId=$ucxMapId sparkMapId=$mapId attemptId=$attemptId elapsedMs=$elapsedMs " +
          s"groupId=${state.groupId} requireGroupReadersReady=$requireGroupReadersReady " +
          s"coverage=${state.readerCoverage}")
    } finally {
      tracker.ucxWriterReadersReadyWaitNanos += System.nanoTime() - waitStartNs
    }
  }

  private def registerEndpointIfNeeded(): UcxShuffleEndpoint = {
    if (endpointRegistered) {
      return endpoint
    }
    synchronized {
      if (endpointRegistered) {
        return endpoint
      }
      endpoint =
        UcxColumnarShuffleWriter.buildEndpoint(
          conf,
          nativeBridge,
          handle.shuffleId,
          ucxMapId,
          context)
      val registered = coordinator.registerWriter(handle.shuffleId, ucxMapId, endpoint)
      if (!registered) {
        throw new SparkException(
          s"Failed to register UCX shuffle writer endpoint for shuffleId=${handle.shuffleId} " +
            s"ucxMapId=$ucxMapId sparkMapId=$mapId attemptId=$attemptId")
      }
      endpointRegistered = true
      logInfo(
        s"Registered UCX shuffle writer endpoint shuffleId=${handle.shuffleId} " +
          s"ucxMapId=$ucxMapId sparkMapId=$mapId attemptId=$attemptId endpoint=$endpoint")
      endpoint
    }
  }

  @volatile private var endpoint: UcxShuffleEndpoint = _

  private def partitionId(key: K): Int = key match {
    case id: Int => id
    case other =>
      throw new SparkException(
        s"UCX columnar shuffle writer expected Int partition id, got " +
          s"${Option(other).map(_.getClass.getName).getOrElse("null")}")
  }

  private def columnarBatch(value: V): ColumnarBatch = value match {
    case batch: ColumnarBatch => batch
    case other =>
      throw new SparkException(
        s"UCX columnar shuffle writer expected ColumnarBatch value, got " +
          s"${Option(other).map(_.getClass.getName).getOrElse("null")}")
  }
}

private[spark] object UcxColumnarShuffleWriter {
  private val UcxPortConf = "spark.gluten.ucx.shuffle.port"

  def buildEndpoint(
      conf: SparkConf,
      nativeBridge: UcxShuffleNativeBridge,
      shuffleId: Int,
      mapId: Long,
      context: TaskContext): UcxShuffleEndpoint = {
    val env = SparkEnv.get
    val host = Option(env.rpcEnv.address).map(_.host).getOrElse(Utils.localCanonicalHostName())
    val attemptId = context.taskAttemptId()
    val ucxPort = conf.getInt(UcxPortConf, -1) match {
      case configured if configured > 0 => configured
      case _ => nativeBridge.localShuffleEndpointPort()
    }
    if (ucxPort <= 0) {
      throw new SparkException(
        s"Unable to determine local UCX shuffle endpoint port for shuffleId=$shuffleId " +
          s"mapId=$mapId attemptId=$attemptId. Set $UcxPortConf or use a native bridge that " +
          "exposes the Velox UCX exchange listener.")
    }
    val nativeTaskId = NativeUcxShuffleExecution.writerTaskId(shuffleId, mapId, attemptId)
    UcxShuffleEndpoint(
      executorId = env.executorId,
      host = host,
      ucxPort = ucxPort,
      shuffleId = shuffleId,
      mapId = mapId,
      attemptId = attemptId,
      stageId = context.stageId(),
      stageAttemptNumber = context.stageAttemptNumber(),
      nativeTaskId = nativeTaskId,
      deviceId = gpuDeviceId(context),
      epoch = attemptId
    )
  }

  private def gpuDeviceId(context: TaskContext): Int = {
    context
      .resources()
      .get("gpu")
      .flatMap {
        resource =>
          resource.addresses.headOption.flatMap {
            address =>
              try {
                Some(address.toInt)
              } catch {
                case _: NumberFormatException => None
              }
          }
      }
      .getOrElse(-1)
  }
}

private[spark] object UcxColumnarShuffleReader {
  def buildEndpoint(
      shuffleId: Int,
      reducePartitionId: Int,
      context: TaskContext): UcxShuffleReaderEndpoint = {
    val env = SparkEnv.get
    val host = Option(env.rpcEnv.address).map(_.host).getOrElse(Utils.localCanonicalHostName())
    val attemptId = context.taskAttemptId()
    UcxShuffleReaderEndpoint(
      executorId = env.executorId,
      host = host,
      shuffleId = shuffleId,
      reducePartitionId = reducePartitionId,
      taskAttemptId = attemptId,
      stageId = context.stageId(),
      stageAttemptNumber = context.stageAttemptNumber(),
      nativeReaderId = s"ucx-shuffle-$shuffleId-reduce-$reducePartitionId-attempt-$attemptId",
      epoch = attemptId
    )
  }
}

private[spark] class UcxColumnarShuffleReader[K, C](
    handle: UcxColumnarShuffleHandle[K, _, C],
    startMapIndex: Int,
    endMapIndex: Int,
    startPartition: Int,
    endPartition: Int,
    context: TaskContext,
    metrics: ShuffleReadMetricsReporter,
    coordinator: UcxShuffleCoordinator,
    nativeBridge: UcxShuffleNativeBridge)
  extends ShuffleReader[K, C]
  with Logging {

  @volatile private var nativeReaderHandle = -1L
  @volatile private var registeredReaderEndpoint: UcxShuffleReaderEndpoint = _
  @volatile private var registeredReaderExpectedMaps = -1
  @volatile private var registeredReaderSeenMaps = -1
  @volatile private var registeredReaderCompletedMaps = -1
  @volatile private var registeredReaderNoMoreSplits = false
  private val columnarDependency =
    handle.dependency.asInstanceOf[ColumnarShuffleDependencyLike]

  context.addTaskCompletionListener[Unit] {
    taskContext =>
      if (taskContext.isFailed() || taskContext.isInterrupted()) {
        coordinator.abortShuffle(
          handle.shuffleId,
          s"UCX shuffle reader task ended unsuccessfully: shuffleId=${handle.shuffleId} " +
            s"partitionRange=[$startPartition,$endPartition) " +
            s"attemptId=${taskContext.taskAttemptId()} failed=${taskContext.isFailed()} " +
            s"interrupted=${taskContext.isInterrupted()}"
        )
      } else {
        val endpoint = registeredReaderEndpoint
        if (endpoint != null) {
          val completedMaps =
            if (registeredReaderExpectedMaps >= 0 &&
                registeredReaderCompletedMaps < registeredReaderExpectedMaps) {
              registeredReaderExpectedMaps
            } else {
              registeredReaderCompletedMaps
            }
          reportNativeReaderState(
            endpoint,
            seenMaps = math.max(registeredReaderSeenMaps, completedMaps),
            completedMaps = completedMaps,
            expectedMaps = registeredReaderExpectedMaps,
            noMoreSplits = true,
            finished = true)
        }
      }
  }

  override def read(): Iterator[Product2[K, C]] = {
    if (NativeUcxShuffleExecution.enabled(SparkEnv.get.conf)) {
      return readNativeExchange()
    }
    val reducePartitionId = singleReducePartitionId()
    val readerEndpoint =
      UcxColumnarShuffleReader.buildEndpoint(handle.shuffleId, reducePartitionId, context)
    val readerRegistered =
      coordinator.registerReader(handle.shuffleId, reducePartitionId, readerEndpoint)
    if (!readerRegistered) {
      throw new SparkException(
        s"Failed to register UCX shuffle reader endpoint for shuffleId=${handle.shuffleId} " +
          s"reducePartitionId=$reducePartitionId attemptId=${context.taskAttemptId()}")
    }
    registeredReaderEndpoint = readerEndpoint
    val initialResponse = coordinator.getAvailableWriters(handle.shuffleId).getOrElse {
      throw new SparkException(
        s"UCX shuffle ${handle.shuffleId} is not registered or has been aborted")
    }
    val expectedMaps = expectedMapCount(initialResponse)
    val initialEndpoints = availableEndpoints(initialResponse)
    val initialSeen = new java.util.HashSet[Long]()
    initialEndpoints.foreach(endpoint => initialSeen.add(endpoint.mapId))
    registeredReaderExpectedMaps = expectedMaps
    registeredReaderSeenMaps = initialSeen.size
    registeredReaderCompletedMaps = completedMapCount(initialResponse, initialSeen)
    logInfo(
      s"Opening UCX streaming shuffle reader with ${initialEndpoints.size}/$expectedMaps " +
        s"initial writer endpoint(s) for shuffleId=${handle.shuffleId} " +
        s"mapRange=[$startMapIndex,$endMapIndex) " +
        s"partitionRange=[$startPartition,$endPartition)")
    val tracker = TaskWallTimeTracker.get()
    val openStartNs = System.nanoTime()
    try {
      nativeReaderHandle = nativeBridge.openReader(
        handle.shuffleId,
        reducePartitionId,
        columnarDependency.outputSchema,
        initialEndpoints)
    } finally {
      tracker.ucxReaderOpenNanos += System.nanoTime() - openStartNs
    }
    val readerClosed = new AtomicBoolean(false)
    val pollerFailure = new AtomicReference[Throwable](null)
    val lastReaderProgressNs = new AtomicLong(System.nanoTime())

    def releaseNativeReader(): Unit = {
      val handleToClose = nativeReaderHandle
      nativeReaderHandle = -1L
      if (handleToClose != -1L) {
        val closeStartNs = System.nanoTime()
        try {
          nativeBridge.closeReader(handleToClose)
        } finally {
          TaskWallTimeTracker.get().ucxReaderCloseNanos += System.nanoTime() - closeStartNs
        }
      }
    }

    def closeFromPoller(): Unit = {
      if (readerClosed.compareAndSet(false, true)) {
        releaseNativeReader()
      }
    }

    val endpointPoller = startEndpointPoller(
      readerEndpoint,
      nativeReaderHandle,
      initialEndpoints,
      expectedMaps,
      readerClosed,
      pollerFailure,
      lastReaderProgressNs,
      closeFromPoller _)

    new Iterator[Product2[K, C]] {
      private var nextValue: Option[ColumnarBatch] = fetchNext()

      private def throwIfPollerFailed(): Unit = {
        val error = pollerFailure.get()
        if (error != null) {
          val message =
            "UCX streaming shuffle reader endpoint poller failed for " +
              s"shuffleId=${handle.shuffleId} " +
              s"mapRange=[$startMapIndex,$endMapIndex) " +
              s"partitionRange=[$startPartition,$endPartition)"
          throw new SparkException(message, error)
        }
      }

      private def joinEndpointPoller(): Unit = {
        if (Thread.currentThread() != endpointPoller) {
          try {
            endpointPoller.join(1000L)
          } catch {
            case e: InterruptedException =>
              Thread.currentThread().interrupt()
              pollerFailure.compareAndSet(null, e)
          }
        }
      }

      private def closeFromIterator(): Unit = {
        if (readerClosed.compareAndSet(false, true)) {
          endpointPoller.interrupt()
          joinEndpointPoller()
          releaseNativeReader()
        }
      }

      private def fetchNext(): Option[ColumnarBatch] = {
        throwIfPollerFailed()
        if (readerClosed.get()) {
          return None
        }
        try {
          val nextStartNs = System.nanoTime()
          val batch =
            try {
              nativeBridge.nextBatch(nativeReaderHandle)
            } finally {
              val elapsedNs = System.nanoTime() - nextStartNs
              tracker.ucxReaderNextBatchNanos += elapsedNs
              tracker.ucxReaderNextBatchCalls += 1
              logSlowNextBatch(elapsedNs)
            }
          if (batch.nonEmpty) {
            lastReaderProgressNs.set(System.nanoTime())
          }
          if (batch.isEmpty) {
            closeFromIterator()
            throwIfPollerFailed()
          }
          batch
        } catch {
          case NonFatal(e) =>
            closeFromIterator()
            throw e
        }
      }

      override def hasNext: Boolean = {
        throwIfPollerFailed()
        if (nextValue.isEmpty) {
          closeFromIterator()
        }
        nextValue.isDefined
      }

      override def next(): Product2[K, C] = {
        if (!hasNext) {
          Iterator.empty.next()
        }
        val batch = nextValue.get
        nextValue = fetchNext()
        (reducePartitionId.asInstanceOf[K], batch.asInstanceOf[C])
      }
    }
  }

  private def readNativeExchange(): Iterator[Product2[K, C]] = {
    val reducePartitionId = singleReducePartitionId()
    val readerEndpoint =
      UcxColumnarShuffleReader.buildEndpoint(handle.shuffleId, reducePartitionId, context)
    val readerRegistered =
      coordinator.registerReader(handle.shuffleId, reducePartitionId, readerEndpoint)
    if (!readerRegistered) {
      throw new SparkException(
        s"Failed to register native UCX shuffle reader endpoint " +
          s"for shuffleId=${handle.shuffleId} " +
          s"reducePartitionId=$reducePartitionId attemptId=${context.taskAttemptId()}")
    }
    registeredReaderEndpoint = readerEndpoint
    val initialResponse = coordinator.getAvailableWriters(handle.shuffleId).getOrElse {
      throw new SparkException(
        s"UCX shuffle ${handle.shuffleId} is not registered or has been aborted")
    }
    val expectedMaps = expectedMapCount(initialResponse)
    val initialEndpoints = availableEndpoints(initialResponse)
    val initialSeen = new java.util.HashSet[Long]()
    initialEndpoints.foreach(endpoint => initialSeen.add(endpoint.mapId))
    registeredReaderExpectedMaps = expectedMaps
    registeredReaderSeenMaps = initialSeen.size
    registeredReaderCompletedMaps = completedMapCount(initialResponse, initialSeen)
    logInfo(
      s"Registered Velox native UCX Exchange reader with ${initialEndpoints.size}/$expectedMaps " +
        s"initial producer endpoint(s) for shuffleId=${handle.shuffleId} " +
        s"mapRange=[$startMapIndex,$endMapIndex) " +
        s"partitionRange=[$startPartition,$endPartition)")
    val spec = NativeUcxShuffleReadSpec(
      shuffleId = handle.shuffleId,
      reducePartitionId = reducePartitionId,
      startMapIndex = startMapIndex,
      endMapIndex = endMapIndex,
      expectedMaps = expectedMaps,
      initialEndpoints = initialEndpoints,
      taskAttemptId = context.taskAttemptId(),
      nativeReaderId = readerEndpoint.nativeReaderId,
      replicated =
        columnarDependency.nativePartitioning.getShortName ==
          GlutenShuffleUtils.BroadcastPartitioningShortName
    )
    NativeUcxShuffleExecution.captureReadSpec(spec)
    new NativeUcxShuffleReadProductIterator[K, C](spec)
  }

  private def reportNativeReaderState(
      endpoint: UcxShuffleReaderEndpoint,
      seenMaps: Int,
      completedMaps: Int,
      expectedMaps: Int,
      noMoreSplits: Boolean,
      finished: Boolean): Unit = {
    try {
      val response =
        coordinator.reportNativeReaderState(
          UcxNativeReaderState(
            shuffleId = handle.shuffleId,
            reducePartitionId = endpoint.reducePartitionId,
            taskAttemptId = endpoint.taskAttemptId,
            nativeReaderId = endpoint.nativeReaderId,
            seenMaps = seenMaps,
            completedMaps = completedMaps,
            expectedMaps = expectedMaps,
            noMoreSplits = noMoreSplits,
            finished = finished,
            timestampMs = System.currentTimeMillis()))
      if (!response.accepted) {
        logWarning(
          s"UCX native reader state rejected by transport coordinator " +
            s"shuffleId=${handle.shuffleId} reduce=${endpoint.reducePartitionId} " +
            s"taskAttemptId=${endpoint.taskAttemptId} groupId=${response.groupId} " +
            s"groupState=${response.groupState} " +
            s"reason=${response.reason}")
      }
    } catch {
      case NonFatal(e) =>
        logWarning(
          s"Failed to report UCX native reader state shuffleId=${handle.shuffleId} " +
            s"reduce=${endpoint.reducePartitionId} taskAttemptId=${endpoint.taskAttemptId} " +
            s"noMoreSplits=$noMoreSplits finished=$finished",
          e)
    }
  }

  private def logSlowNextBatch(elapsedNs: Long): Unit = {
    val thresholdMs =
      SparkEnv.get.conf.getLong(UcxColumnarShuffleManager.ReaderSlowNextBatchLogMsConf, 1000L)
    if (thresholdMs > 0 && elapsedNs / 1000000L >= thresholdMs) {
      logInfo(
        s"UCX streaming shuffle reader nextBatch waited ${elapsedNs / 1000000L}ms " +
          s"shuffleId=${handle.shuffleId} " +
          s"mapRange=[$startMapIndex,$endMapIndex) " +
          s"partitionRange=[$startPartition,$endPartition)")
    }
  }

  private def singleReducePartitionId(): Int = {
    if (endPartition != startPartition + 1) {
      throw new SparkException(
        s"UCX columnar shuffle reader expects one reduce partition per Spark task, got " +
          s"partitionRange=[$startPartition,$endPartition)")
    }
    startPartition
  }

  private def startEndpointPoller(
      readerEndpoint: UcxShuffleReaderEndpoint,
      readerHandle: Long,
      initialEndpoints: Seq[UcxShuffleEndpoint],
      expectedMaps: Int,
      readerClosed: AtomicBoolean,
      pollerFailure: AtomicReference[Throwable],
      lastReaderProgressNs: AtomicLong,
      closeReader: () => Unit): Thread = {
    val timeoutMs =
      SparkEnv.get.conf.getLong(UcxColumnarShuffleManager.ReaderEndpointWaitMsConf, 300000L)
    val pollMs = math.max(
      1L,
      SparkEnv.get.conf.getLong(UcxColumnarShuffleManager.ReaderEndpointPollMsConf, 10L))
    val statePollMs = math.max(
      pollMs,
      SparkEnv.get.conf.getLong(UcxColumnarShuffleManager.ReaderStatePollMsConf, 1000L))
    val dataWaitAfterWritersFinishedMs =
      SparkEnv.get.conf
        .getLong(UcxColumnarShuffleManager.ReaderDataWaitAfterWritersFinishedMsConf, 300000L)
    val seen = new java.util.HashSet[Long]()
    initialEndpoints.foreach(endpoint => seen.add(endpoint.mapId))
    val poller = new Thread(
      new Runnable {
        override def run(): Unit = {
          val startNs = System.nanoTime()
          var lastSeen = -1
          var lastFinished = -1
          var noMoreEndpointsSent = false
          try {
            while (!readerClosed.get()) {
              val response = coordinator.getAvailableWriters(handle.shuffleId).getOrElse {
                throw new SparkException(
                  s"UCX shuffle ${handle.shuffleId} is not registered or has been aborted")
              }
              val endpoints = availableEndpoints(response)
              val newEndpoints = endpoints.filter(endpoint => seen.add(endpoint.mapId))
              if (newEndpoints.nonEmpty) {
                registeredReaderSeenMaps = seen.size
                nativeBridge.addReaderEndpoints(readerHandle, newEndpoints)
                logInfo(
                  s"Added ${newEndpoints.size} UCX writer endpoint(s) to streaming reader " +
                    s"shuffleId=${handle.shuffleId} " +
                    s"mapRange=[$startMapIndex,$endMapIndex) " +
                    s"seen=${seen.size}/$expectedMaps")
              }
              val completedMaps = completedMapCount(response, seen)
              registeredReaderCompletedMaps = completedMaps
              if (!noMoreEndpointsSent && completedMaps >= expectedMaps) {
                nativeBridge.noMoreReaderEndpoints(readerHandle)
                noMoreEndpointsSent = true
                registeredReaderNoMoreSplits = true
                reportNativeReaderState(
                  readerEndpoint,
                  seenMaps = seen.size,
                  completedMaps = completedMaps,
                  expectedMaps = expectedMaps,
                  noMoreSplits = true,
                  finished = false)
                logInfo(
                  s"UCX streaming shuffle reader reached noMoreEndpoints " +
                    s"shuffleId=${handle.shuffleId} mapRange=[$startMapIndex,$endMapIndex) " +
                    s"seen=${seen.size}/$expectedMaps completed=$completedMaps/$expectedMaps")
              }
              if (seen.size != lastSeen) {
                lastSeen = seen.size
                logInfo(
                  s"Polling UCX shuffle writer endpoints shuffleId=${handle.shuffleId} " +
                    s"mapRange=[$startMapIndex,$endMapIndex) seen=${seen.size}/$expectedMaps")
              }
              val finished = finishedMapCount(response)
              if (finished != lastFinished) {
                lastFinished = finished
                logInfo(
                  s"UCX streaming shuffle writer finished coverage shuffleId=${handle.shuffleId} " +
                    s"mapRange=[$startMapIndex,$endMapIndex) finished=$finished/$expectedMaps")
              }
              val elapsedMs = (System.nanoTime() - startNs) / 1000000L
              if (!noMoreEndpointsSent && elapsedMs > timeoutMs) {
                throw new SparkException(
                  s"Timed out after ${timeoutMs}ms polling UCX shuffle writer endpoints " +
                    s"shuffleId=${handle.shuffleId} mapRange=[$startMapIndex,$endMapIndex) " +
                    s"seen=${seen.size}/$expectedMaps")
              }
              if (noMoreEndpointsSent && finished >= expectedMaps) {
                val idleMs = (System.nanoTime() - lastReaderProgressNs.get()) / 1000000L
                if (
                  dataWaitAfterWritersFinishedMs > 0 &&
                  idleMs > dataWaitAfterWritersFinishedMs
                ) {
                  throw new SparkException(
                    s"Timed out after ${dataWaitAfterWritersFinishedMs}ms waiting for UCX " +
                      s"shuffle reader data/EOS after all writers finished " +
                      s"shuffleId=${handle.shuffleId} mapRange=[$startMapIndex,$endMapIndex) " +
                      s"partitionRange=[$startPartition,$endPartition) " +
                      s"seen=${seen.size}/$expectedMaps finished=$finished/$expectedMaps")
                }
              }
              Thread.sleep(if (noMoreEndpointsSent) statePollMs else pollMs)
            }
          } catch {
            case e: InterruptedException =>
              Thread.currentThread().interrupt()
              if (!readerClosed.get()) {
                pollerFailure.compareAndSet(null, e)
                closeReader()
              }
            case NonFatal(e) =>
              if (!readerClosed.get()) {
                pollerFailure.compareAndSet(null, e)
                closeReader()
              }
          }
        }
      },
      s"ucx-shuffle-reader-endpoint-poller-${handle.shuffleId}-$startPartition"
    )
    poller.setDaemon(true)
    poller.start()
    poller
  }

  private def availableEndpoints(response: UcxShuffleEndpointResponse): Seq[UcxShuffleEndpoint] = {
    response.endpoints.iterator
      .filter { case (mapId, _) => inMapRange(mapId) }
      .map(_._2)
      .toSeq
      .sortBy(_.mapId)
  }

  private def finishedMapCount(response: UcxShuffleEndpointResponse): Int = {
    response.finishedMapIds.count(inMapRange)
  }

  private def completedMapCount(
      response: UcxShuffleEndpointResponse,
      seen: java.util.HashSet[Long]): Int = {
    val finishedWithoutEndpoint =
      response.finishedMapIds.count(mapId => inMapRange(mapId) && !seen.contains(mapId))
    seen.size + finishedWithoutEndpoint
  }

  private def expectedMapCount(response: UcxShuffleEndpointResponse): Int = {
    val actualEndMapIndex =
      if (endMapIndex == Int.MaxValue) response.numMaps else endMapIndex
    math.max(0, actualEndMapIndex - startMapIndex)
  }

  private def inMapRange(mapId: Long): Boolean = {
    if (endMapIndex == Int.MaxValue) {
      mapId >= startMapIndex
    } else {
      mapId >= startMapIndex && mapId < endMapIndex
    }
  }
}
