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

  override def registerPipelinedShuffleGroup(group: PipelinedShuffleGroupMetadata): Unit = {
    coordinator.registerPipelinedShuffleGroup(group)
  }

  override def requiresAllPipelinedShuffleReadersResident(
      group: PipelinedShuffleGroupMetadata): Boolean = true

  override def admitPipelinedShuffleGroup(groupId: String): Unit = {
    coordinator.admitPipelinedShuffleGroup(groupId)
  }

  override def completePipelinedShuffleGroup(groupId: String): Unit = {
    coordinator.completePipelinedShuffleGroup(groupId)
  }

  override def abortPipelinedShuffleGroup(groupId: String, reason: String): Unit = {
    coordinator.abortPipelinedShuffleGroup(groupId, reason)
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
  val ReaderEndpointWaitMsConf: String = "spark.gluten.ucx.shuffle.reader.endpointWaitMs"
  val ReaderEndpointPollMsConf: String = "spark.gluten.ucx.shuffle.reader.endpointPollMs"
  val ReaderStatePollMsConf: String = "spark.gluten.ucx.shuffle.reader.statePollMs"
  val ReaderDataWaitAfterWritersFinishedMsConf: String =
    "spark.gluten.ucx.shuffle.reader.dataWaitAfterWritersFinishedMs"
  val ReaderSlowNextBatchLogMsConf: String =
    "spark.gluten.ucx.shuffle.reader.slowNextBatchLogMs"
  val WriterWaitForReadersReadyConf: String =
    "spark.gluten.ucx.shuffle.writer.waitForReadersReady"
  val WriterReadersReadyWaitMsConf: String =
    "spark.gluten.ucx.shuffle.writer.readersReadyWaitMs"
  val WriterReadersReadyPollMsConf: String =
    "spark.gluten.ucx.shuffle.writer.readersReadyPollMs"

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
  private val attemptId = context.taskAttemptId()
  @volatile private var endpointRegistered = false
  @volatile private var nativeWriterHandle = -1L
  @volatile private var stopped = false

  context.addTaskCompletionListener[Unit] {
    taskContext =>
      if (taskContext.isFailed() || taskContext.isInterrupted()) {
        coordinator.abortShuffle(
          handle.shuffleId,
          s"UCX shuffle writer task ended unsuccessfully: shuffleId=${handle.shuffleId} " +
            s"mapId=$mapId attemptId=$attemptId failed=${taskContext.isFailed()} " +
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
    val endpoint = registerEndpointIfNeeded()
    waitForReadersReadyIfNeeded(tracker)
    val partitioning = columnarDependency.nativePartitioning.getShortName
    val writerContext = NativeUcxShuffleWriterContext(
      shuffleId = handle.shuffleId,
      mapId = mapId,
      attemptId = attemptId,
      nativeTaskId = endpoint.nativeTaskId,
      numPartitions = numPartitions,
      partitioning = partitioning,
      startPartitionId = GlutenShuffleUtils.getStartPartitionId(
        columnarDependency.nativePartitioning,
        context.partitionId()),
      dropFirstColumn = partitioning == GlutenShuffleUtils.HashPartitioningShortName)
    logInfo(
      s"Driving Velox native UCX shuffle producer shuffleId=${handle.shuffleId} " +
        s"mapId=$mapId attemptId=$attemptId nativeTaskId=${endpoint.nativeTaskId} " +
        s"partitions=$numPartitions partitioning=$partitioning")
    NativeUcxShuffleExecution.withWriterContext(writerContext) {
      records.foreach {
        _ =>
          tracker.ucxWriterWriteBatchCalls += 1
      }
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
      if (success) {
        coordinator.markWriterFinished(handle.shuffleId, mapId, attemptId)
      } else {
        coordinator.abortShuffle(
          handle.shuffleId,
          s"UCX shuffle writer stopped unsuccessfully: shuffleId=${handle.shuffleId} " +
            s"mapId=$mapId attemptId=$attemptId")
      }
    }
    Some(MapStatus(SparkEnv.get.blockManager.shuffleServerId, getPartitionLengths(), mapId))
  }

  override def getPartitionLengths(): Array[Long] = Array.fill(numPartitions)(0L)

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
    val waitStartNs = System.nanoTime()
    try {
      val state =
        coordinator.waitUntilPipelinedReadersReady(handle.shuffleId, timeoutMs, pollMs)
      val elapsedMs = (System.nanoTime() - waitStartNs) / 1000000L
      logInfo(
        s"UCX pipelined writer readers-ready gate opened shuffleId=${handle.shuffleId} " +
          s"mapId=$mapId attemptId=$attemptId elapsedMs=$elapsedMs " +
          s"groupId=${state.groupId} coverage=${state.readerCoverage}")
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
        UcxColumnarShuffleWriter.buildEndpoint(conf, nativeBridge, handle.shuffleId, mapId, context)
      val registered = coordinator.registerWriter(handle.shuffleId, mapId, endpoint)
      if (!registered) {
        throw new SparkException(
          s"Failed to register UCX shuffle writer endpoint for shuffleId=${handle.shuffleId} " +
            s"mapId=$mapId attemptId=$attemptId")
      }
      endpointRegistered = true
      logInfo(
        s"Registered UCX shuffle writer endpoint shuffleId=${handle.shuffleId} mapId=$mapId " +
          s"attemptId=$attemptId endpoint=$endpoint")
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
    val initialResponse = coordinator.getAvailableWriters(handle.shuffleId).getOrElse {
      throw new SparkException(
        s"UCX shuffle ${handle.shuffleId} is not registered or has been aborted")
    }
    val expectedMaps = expectedMapCount(initialResponse)
    val initialEndpoints = availableEndpoints(initialResponse)
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
    val initialResponse = coordinator.getAvailableWriters(handle.shuffleId).getOrElse {
      throw new SparkException(
        s"UCX shuffle ${handle.shuffleId} is not registered or has been aborted")
    }
    val expectedMaps = expectedMapCount(initialResponse)
    val initialEndpoints = availableEndpoints(initialResponse)
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
      initialEndpoints = initialEndpoints)
    NativeUcxShuffleExecution.captureReadSpec(spec)
    new NativeUcxShuffleReadProductIterator[K, C](spec)
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
      SparkEnv.get.conf.getLong(
        UcxColumnarShuffleManager.ReaderDataWaitAfterWritersFinishedMsConf,
        300000L)
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
                nativeBridge.addReaderEndpoints(readerHandle, newEndpoints)
                logInfo(
                  s"Added ${newEndpoints.size} UCX writer endpoint(s) to streaming reader " +
                    s"shuffleId=${handle.shuffleId} " +
                    s"mapRange=[$startMapIndex,$endMapIndex) " +
                    s"seen=${seen.size}/$expectedMaps")
              }
              if (!noMoreEndpointsSent && seen.size >= expectedMaps) {
                nativeBridge.noMoreReaderEndpoints(readerHandle)
                noMoreEndpointsSent = true
                logInfo(
                  s"UCX streaming shuffle reader reached noMoreEndpoints " +
                    s"shuffleId=${handle.shuffleId} mapRange=[$startMapIndex,$endMapIndex) " +
                    s"seen=${seen.size}/$expectedMaps")
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
                if (dataWaitAfterWritersFinishedMs > 0 &&
                    idleMs > dataWaitAfterWritersFinishedMs) {
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
