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

import org.apache.spark.{SparkConf, SparkEnv, SparkException}
import org.apache.spark.internal.Logging
import org.apache.spark.rpc.{RpcCallContext, RpcEndpoint, RpcEndpointRef, RpcEnv}
import org.apache.spark.util.{RpcUtils, ThreadUtils}

import java.util.concurrent.{ConcurrentHashMap, LinkedBlockingQueue, ThreadPoolExecutor}

import scala.jdk.CollectionConverters._
import scala.reflect.ClassTag
import scala.util.control.NonFatal

case class UcxShuffleEndpoint(
    executorId: String,
    host: String,
    ucxPort: Int,
    shuffleId: Int,
    mapId: Long,
    attemptId: Long,
    stageId: Int,
    stageAttemptNumber: Int,
    nativeTaskId: String,
    deviceId: Int,
    epoch: Long)

private[spark] case class UcxShuffleReaderEndpoint(
    executorId: String,
    host: String,
    shuffleId: Int,
    reducePartitionId: Int,
    taskAttemptId: Long,
    stageId: Int,
    stageAttemptNumber: Int,
    nativeReaderId: String,
    epoch: Long)

private[spark] case class UcxShuffleEndpointResponse(
    endpoints: Map[Long, UcxShuffleEndpoint],
    numMaps: Int,
    finishedMapIds: Set[Long])

private[spark] case class UcxShuffleReaderCoverageResponse(
    readers: Map[Int, UcxShuffleReaderEndpoint],
    numReduces: Int)

private[spark] case class UcxShuffleReaderCoverage(registeredReaders: Int, expectedReaders: Int)

private[spark] case class UcxPipelinedShuffleStateResponse(
    shuffleId: Int,
    groupId: Option[String],
    groupState: String,
    readersReady: Boolean,
    readerCoverage: Map[Int, UcxShuffleReaderCoverage],
    abortedReason: Option[String])

private[spark] case class UcxShuffleWriterCreditResponse(
    granted: Boolean,
    shuffleId: Int,
    groupId: Option[String],
    activeShuffleWriters: Int,
    maxShuffleWriters: Int,
    reason: Option[String])

private[spark] case class UcxNativeWriterState(
    shuffleId: Int,
    mapId: Long,
    attemptId: Long,
    nativeTaskId: String,
    noMoreData: Boolean,
    queuedBytes: Long,
    blocked: Boolean,
    timestampMs: Long,
    finished: Boolean = false,
    queuedPages: Long = -1L,
    totalBytesSent: Long = -1L,
    totalRowsSent: Long = -1L,
    totalPagesSent: Long = -1L,
    averageBufferTimeMs: Long = -1L)

private[spark] case class UcxNativeWriterStateResponse(
    accepted: Boolean,
    shuffleId: Int,
    groupId: Option[String],
    groupState: String,
    writerFinishedMarked: Boolean,
    writerCreditReleased: Boolean,
    reason: Option[String])

private[spark] case class UcxNativeReaderState(
    shuffleId: Int,
    reducePartitionId: Int,
    taskAttemptId: Long,
    nativeReaderId: String,
    seenMaps: Int,
    completedMaps: Int,
    expectedMaps: Int,
    noMoreSplits: Boolean,
    finished: Boolean,
    timestampMs: Long)

private[spark] case class UcxNativeReaderStateResponse(
    accepted: Boolean,
    shuffleId: Int,
    groupId: Option[String],
    groupState: String,
    reason: Option[String])

sealed private[spark] trait UcxShuffleCoordinatorMessage

private[spark] case class RegisterUcxShuffle(shuffleId: Int, numMaps: Int, numReduces: Int)
  extends UcxShuffleCoordinatorMessage

private[spark] case class RegisterUcxShuffleWriter(
    shuffleId: Int,
    mapId: Long,
    endpoint: UcxShuffleEndpoint)
  extends UcxShuffleCoordinatorMessage

private[spark] case class RegisterUcxShuffleReader(
    shuffleId: Int,
    reducePartitionId: Int,
    endpoint: UcxShuffleReaderEndpoint)
  extends UcxShuffleCoordinatorMessage

private[spark] case class GetAvailableUcxShuffleWriters(shuffleId: Int)
  extends UcxShuffleCoordinatorMessage

private[spark] case class GetUcxShuffleReaderCoverage(shuffleId: Int)
  extends UcxShuffleCoordinatorMessage

private[spark] case class GetUcxPipelinedShuffleState(shuffleId: Int)
  extends UcxShuffleCoordinatorMessage

private[spark] case class MarkUcxShuffleWriterFinished(shuffleId: Int, mapId: Long, attemptId: Long)
  extends UcxShuffleCoordinatorMessage

private[spark] case class TryAcquireUcxShuffleWriterCredit(
    shuffleId: Int,
    mapId: Long,
    attemptId: Long,
    stageId: Int,
    stageAttemptNumber: Int)
  extends UcxShuffleCoordinatorMessage

private[spark] case class ReleaseUcxShuffleWriterCredit(
    shuffleId: Int,
    mapId: Long,
    attemptId: Long)
  extends UcxShuffleCoordinatorMessage

private[spark] case class ReportUcxNativeWriterState(state: UcxNativeWriterState)
  extends UcxShuffleCoordinatorMessage

private[spark] case class ReportUcxNativeReaderState(state: UcxNativeReaderState)
  extends UcxShuffleCoordinatorMessage

private[spark] case class AbortUcxShuffle(shuffleId: Int, reason: String)
  extends UcxShuffleCoordinatorMessage

private[spark] case class RegisterUcxPipelinedShuffleGroup(group: PipelinedShuffleGroupMetadata)
  extends UcxShuffleCoordinatorMessage

private[spark] case class AdmitUcxPipelinedShuffleGroup(groupId: String)
  extends UcxShuffleCoordinatorMessage

private[spark] case class CompleteUcxPipelinedShuffleGroup(groupId: String)
  extends UcxShuffleCoordinatorMessage

private[spark] case class AbortUcxPipelinedShuffleGroup(groupId: String, reason: String)
  extends UcxShuffleCoordinatorMessage

private[spark] case object StopUcxShuffleCoordinator extends UcxShuffleCoordinatorMessage

sealed private[spark] trait UcxShuffleCoordinatorMasterMessage

private[spark] case class RegisterUcxShuffleMasterMessage(
    shuffleId: Int,
    numMaps: Int,
    numReduces: Int,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class RegisterUcxShuffleWriterMasterMessage(
    shuffleId: Int,
    mapId: Long,
    endpoint: UcxShuffleEndpoint,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class RegisterUcxShuffleReaderMasterMessage(
    shuffleId: Int,
    reducePartitionId: Int,
    endpoint: UcxShuffleReaderEndpoint,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class GetAvailableUcxShuffleWritersMasterMessage(
    shuffleId: Int,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class GetUcxShuffleReaderCoverageMasterMessage(
    shuffleId: Int,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class GetUcxPipelinedShuffleStateMasterMessage(
    shuffleId: Int,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class MarkUcxShuffleWriterFinishedMasterMessage(
    shuffleId: Int,
    mapId: Long,
    attemptId: Long,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class TryAcquireUcxShuffleWriterCreditMasterMessage(
    shuffleId: Int,
    mapId: Long,
    attemptId: Long,
    stageId: Int,
    stageAttemptNumber: Int,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class ReleaseUcxShuffleWriterCreditMasterMessage(
    shuffleId: Int,
    mapId: Long,
    attemptId: Long,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class ReportUcxNativeWriterStateMasterMessage(
    state: UcxNativeWriterState,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class ReportUcxNativeReaderStateMasterMessage(
    state: UcxNativeReaderState,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class AbortUcxShuffleMasterMessage(
    shuffleId: Int,
    reason: String,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class RegisterUcxPipelinedShuffleGroupMasterMessage(
    group: PipelinedShuffleGroupMetadata,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class AdmitUcxPipelinedShuffleGroupMasterMessage(
    groupId: String,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class CompleteUcxPipelinedShuffleGroupMasterMessage(
    groupId: String,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class AbortUcxPipelinedShuffleGroupMasterMessage(
    groupId: String,
    reason: String,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class UcxShuffleInfo(numMaps: Int, numReduces: Int)

private[spark] object UcxPipelinedShuffleGroupState {
  val NoGroup: String = "NO_GROUP"
  val Registered: String = "REGISTERED"
  val Admitted: String = "ADMITTED"
  val ReadersReady: String = "READERS_READY"
  val Draining: String = "DRAINING"
  val Finished: String = "FINISHED"
  val Aborted: String = "ABORTED"
}

private[spark] case class UcxPipelinedShuffleGeneration(
    writerStageId: Int,
    writerStageAttempt: Int)

private[spark] class UcxPipelinedTransportGroupState(
    val groupAttemptId: String) {

  @volatile var state: String = UcxPipelinedShuffleGroupState.Registered
  @volatile var abortedReason: Option[String] = None

  private val metadataByGroup =
    new ConcurrentHashMap[String, PipelinedShuffleGroupMetadata]()
  private val stateByGroup = new ConcurrentHashMap[String, String]()
  private val shuffleIdSet = ConcurrentHashMap.newKeySet[Int]()

  val nativeWriterStates = new ConcurrentHashMap[String, UcxNativeWriterState]()
  val nativeReaderStates = new ConcurrentHashMap[String, UcxNativeReaderState]()

  def registerGroup(group: PipelinedShuffleGroupMetadata, shuffleIds: Seq[Int]): Unit =
    synchronized {
      val groupAttemptId = group.groupAttemptId
      val previous = Option(metadataByGroup.put(groupAttemptId, group))
      if (previous.forall(groupGeneration(_) != groupGeneration(group))) {
        stateByGroup.put(groupAttemptId, UcxPipelinedShuffleGroupState.Registered)
      }
      shuffleIdSet.clear()
      metadataByGroup.values().asScala
        .flatMap(_.stages.flatMap(_.shuffleId))
        .foreach(shuffleIdSet.add)
    }

  def admitGroup(groupId: String): Unit = {
    stateByGroup.put(groupId, UcxPipelinedShuffleGroupState.Admitted)
    if (
      state != UcxPipelinedShuffleGroupState.Aborted &&
      state != UcxPipelinedShuffleGroupState.Finished
    ) {
      state = UcxPipelinedShuffleGroupState.Admitted
    }
  }

  def markGroupReadersReady(groupId: String): Unit = {
    stateByGroup.put(groupId, UcxPipelinedShuffleGroupState.ReadersReady)
    if (
      state != UcxPipelinedShuffleGroupState.Aborted &&
      state != UcxPipelinedShuffleGroupState.Finished &&
      activeGroupIds.forall(
        group => groupState(group) == UcxPipelinedShuffleGroupState.ReadersReady)
    ) {
      state = UcxPipelinedShuffleGroupState.ReadersReady
    }
  }

  def completeGroup(groupId: String): Unit = {
    stateByGroup.put(groupId, UcxPipelinedShuffleGroupState.Draining)
    state = UcxPipelinedShuffleGroupState.Draining
  }

  def finishGroup(groupId: String): Unit = {
    stateByGroup.put(groupId, UcxPipelinedShuffleGroupState.Finished)
    state = UcxPipelinedShuffleGroupState.Finished
  }

  def abortAllGroups(): Unit = {
    stateByGroup.keySet().asScala.foreach {
      groupId => stateByGroup.put(groupId, UcxPipelinedShuffleGroupState.Aborted)
    }
  }

  def groupIds: Set[String] = metadataByGroup.keySet().asScala.toSet

  def activeGroupIds: Set[String] =
    groupIds.filterNot { groupId =>
      val groupState = stateByGroup.get(groupId)
      groupState == UcxPipelinedShuffleGroupState.Finished ||
      groupState == UcxPipelinedShuffleGroupState.Aborted
    }

  def groupState(groupId: String): String =
    Option(stateByGroup.get(groupId)).getOrElse(UcxPipelinedShuffleGroupState.Registered)

  def groupMetadata(groupId: String): Option[PipelinedShuffleGroupMetadata] =
    Option(metadataByGroup.get(groupId))

  def groupIdsForShuffle(shuffleId: Int): Set[String] =
    metadataByGroup.asScala.collect {
      case (groupId, metadata)
          if metadata.stages.exists(_.shuffleId.contains(shuffleId)) =>
        groupId
    }.toSet

  def writerStage(shuffleId: Int): Option[PipelinedShuffleStageMetadata] =
    metadataByGroup.values().asScala
      .flatMap(_.stages)
      .filter(_.shuffleId.contains(shuffleId))
      .toSeq
      .sortBy(stage => (stage.stageId, stage.attemptId))
      .lastOption

  def readerStage(stageId: Int): Option[PipelinedShuffleStageMetadata] =
    metadataByGroup.values().asScala
      .flatMap(_.stages)
      .filter(_.stageId == stageId)
      .toSeq
      .sortBy(_.attemptId)
      .lastOption

  def shuffleIds: Set[Int] = shuffleIdSet.asScala.toSet

  def groupId: String = groupIds.toSeq.sorted.mkString("[", ",", "]")

  private def groupGeneration(
      group: PipelinedShuffleGroupMetadata): Seq[(Int, Int, Option[Int], Seq[Int])] = {
    group.stages
      .map {
        stage =>
          (
            stage.stageId,
            stage.attemptId,
            stage.shuffleId,
            stage.pipelinedParentShuffleIds.sorted)
      }
      .sortBy { case (stageId, attemptId, shuffleId, _) =>
        (stageId, attemptId, shuffleId.getOrElse(-1))
      }
  }

}

abstract private[spark] class UcxShuffleCoordinator(conf: SparkConf) extends Logging {

  @volatile var coordinatorEndpoint: RpcEndpointRef = _

  protected def askCoordinator[T: ClassTag](message: Any): T = {
    try {
      coordinatorEndpoint.askSync[T](message)
    } catch {
      case e: Exception =>
        logError(log"Error communicating with UcxShuffleCoordinator", e)
        throw new SparkException("Error communicating with UcxShuffleCoordinator", e)
    }
  }

  protected def sendCoordinator(message: Any): Unit = {
    val response = askCoordinator[Boolean](message)
    if (response != true) {
      throw new SparkException(
        "Error reply received from UcxShuffleCoordinator. Expecting true, got " + response)
    }
  }

  def registerShuffle(shuffleId: Int, numMaps: Int, numReduces: Int): Unit

  def registerWriter(shuffleId: Int, mapId: Long, endpoint: UcxShuffleEndpoint): Boolean

  def registerReader(
      shuffleId: Int,
      reducePartitionId: Int,
      endpoint: UcxShuffleReaderEndpoint): Boolean

  def getAvailableWriters(shuffleId: Int): Option[UcxShuffleEndpointResponse]

  def getReaderCoverage(shuffleId: Int): Option[UcxShuffleReaderCoverageResponse]

  def getPipelinedShuffleState(shuffleId: Int): UcxPipelinedShuffleStateResponse

  def waitUntilPipelinedReadersReady(
      shuffleId: Int,
      timeoutMs: Long,
      pollMs: Long,
      requireGroupReadersReady: Boolean = false): UcxPipelinedShuffleStateResponse = {
    val startNs = System.nanoTime()
    val effectivePollMs = math.max(1L, pollMs)
    var lastLoggedState: String = ""
    while (true) {
      val state = getPipelinedShuffleState(shuffleId)
      state.abortedReason.foreach {
        reason =>
          throw new SparkException(
            s"UCX pipelined shuffle $shuffleId was aborted while waiting for readers: $reason")
      }
      if (pipelinedShuffleReadersReady(shuffleId, state, requireGroupReadersReady)) {
        return state
      }
      val elapsedMs = (System.nanoTime() - startNs) / 1000000L
      if (elapsedMs > timeoutMs) {
        throw new SparkException(
          s"Timed out after ${timeoutMs}ms waiting for UCX pipelined shuffle readers " +
            s"shuffleId=$shuffleId state=${state.groupState} groupId=${state.groupId} " +
            s"readerCoverage=${state.readerCoverage}")
      }
      val stateForLog =
          s"groupId=${state.groupId} state=${state.groupState} " +
          s"currentShuffleCoverage=${state.readerCoverage.get(shuffleId)} " +
          s"groupCoverage=${state.readerCoverage} " +
          s"requireGroupReadersReady=$requireGroupReadersReady"
      if (stateForLog != lastLoggedState) {
        logInfo(s"Waiting for UCX pipelined shuffle readers shuffleId=$shuffleId $stateForLog")
        lastLoggedState = stateForLog
      }
      try {
        Thread.sleep(effectivePollMs)
      } catch {
        case e: InterruptedException =>
          Thread.currentThread().interrupt()
          throw new SparkException(
            s"Interrupted waiting for UCX pipelined shuffle readers shuffleId=$shuffleId",
            e)
      }
    }
    throw new IllegalStateException("unreachable")
  }

  private def pipelinedShuffleReadersReady(
      shuffleId: Int,
      state: UcxPipelinedShuffleStateResponse,
      requireGroupReadersReady: Boolean): Boolean = {
    state.groupId match {
      case None =>
        state.readersReady
      case Some(_) =>
        if (state.readersReady) {
          return true
        }
        if (requireGroupReadersReady) {
          state.readerCoverage.nonEmpty && state.readerCoverage.values.forall {
            coverage =>
              coverage.expectedReaders > 0 &&
              coverage.registeredReaders >= coverage.expectedReaders
          }
        } else {
          state.readerCoverage.get(shuffleId).exists {
            coverage =>
              coverage.expectedReaders > 0 &&
              coverage.registeredReaders >= coverage.expectedReaders
          }
        }
    }
  }

  def markWriterFinished(shuffleId: Int, mapId: Long, attemptId: Long): Boolean

  def tryAcquireWriterCredit(
      shuffleId: Int,
      mapId: Long,
      attemptId: Long,
      stageId: Int,
      stageAttemptNumber: Int): UcxShuffleWriterCreditResponse

  def waitUntilWriterCreditAvailable(
      shuffleId: Int,
      mapId: Long,
      attemptId: Long,
      stageId: Int,
      stageAttemptNumber: Int,
      timeoutMs: Long,
      pollMs: Long): UcxShuffleWriterCreditResponse = {
    val startNs = System.nanoTime()
    val effectivePollMs = math.max(1L, pollMs)
    var lastLoggedState: String = ""
    while (true) {
      val credit =
        tryAcquireWriterCredit(
          shuffleId,
          mapId,
          attemptId,
          stageId,
          stageAttemptNumber)
      if (credit.granted) {
        return credit
      }
      credit.reason.foreach {
        reason =>
          throw new SparkException(
            s"UCX shuffle writer credit was denied for shuffleId=$shuffleId mapId=$mapId " +
              s"attemptId=$attemptId stage=$stageId.$stageAttemptNumber: $reason")
      }
      val elapsedMs = (System.nanoTime() - startNs) / 1000000L
      if (elapsedMs > timeoutMs) {
        throw new SparkException(
          s"Timed out after ${timeoutMs}ms waiting for UCX shuffle writer credit " +
            s"shuffleId=$shuffleId mapId=$mapId attemptId=$attemptId " +
            s"stage=$stageId.$stageAttemptNumber lastState=$credit")
      }
      val stateForLog =
        s"groupId=${credit.groupId} " +
          s"shuffleActive=${credit.activeShuffleWriters}/${credit.maxShuffleWriters}"
      if (stateForLog != lastLoggedState) {
        logInfo(
          s"Waiting for UCX shuffle writer credit shuffleId=$shuffleId " +
            s"mapId=$mapId attemptId=$attemptId $stateForLog")
        lastLoggedState = stateForLog
      }
      try {
        Thread.sleep(effectivePollMs)
      } catch {
        case e: InterruptedException =>
          Thread.currentThread().interrupt()
          throw new SparkException(
            s"Interrupted waiting for UCX shuffle writer credit shuffleId=$shuffleId " +
              s"mapId=$mapId attemptId=$attemptId",
            e)
      }
    }
    throw new IllegalStateException("unreachable")
  }

  def releaseWriterCredit(shuffleId: Int, mapId: Long, attemptId: Long): Boolean

  def reportNativeWriterState(state: UcxNativeWriterState): UcxNativeWriterStateResponse

  def reportNativeReaderState(state: UcxNativeReaderState): UcxNativeReaderStateResponse

  def abortShuffle(shuffleId: Int, reason: String): Boolean

  def registerPipelinedShuffleGroup(group: PipelinedShuffleGroupMetadata): Unit

  def admitPipelinedShuffleGroup(groupAttemptId: String): Boolean

  def completePipelinedShuffleGroup(groupAttemptId: String): Boolean

  def abortPipelinedShuffleGroup(groupAttemptId: String, reason: String): Boolean

  def unregisterShuffle(shuffleId: Int): Unit = {}

  def stop(): Unit = {}
}

private[spark] class UcxShuffleCoordinatorMaster(conf: SparkConf)
  extends UcxShuffleCoordinator(conf) {

  private val shuffleInfos = new ConcurrentHashMap[Int, UcxShuffleInfo]()
  private val endpoints =
    new ConcurrentHashMap[Int, ConcurrentHashMap[Long, UcxShuffleEndpoint]]()
  private val readerEndpoints =
    new ConcurrentHashMap[Int, ConcurrentHashMap[Int, UcxShuffleReaderEndpoint]]()
  private val finishedAttempts =
    new ConcurrentHashMap[Int, ConcurrentHashMap[Long, java.lang.Long]]()
  private val abortedShuffles = new ConcurrentHashMap[Int, String]()
  private val pipelinedGroups =
    new ConcurrentHashMap[String, UcxPipelinedTransportGroupState]()
  private val abortedPipelinedGroups = new ConcurrentHashMap[String, String]()
  private val shuffleIdToPipelinedGroupId = new ConcurrentHashMap[Int, String]()
  private val shuffleGenerationById =
    new ConcurrentHashMap[Int, UcxPipelinedShuffleGeneration]()
  private val pendingShuffleUnregistrations = ConcurrentHashMap.newKeySet[Int]()
  private val writerCreditsByShuffle =
    new ConcurrentHashMap[Int, ConcurrentHashMap[Long, java.lang.Long]]()
  private val writerCreditLock = new Object()
  private val pipelinedRegistrationLock = new Object()

  private val maxActiveWriterTasksPerShuffle =
    math.max(0, conf.getInt(UcxColumnarShuffleManager.WriterMaxActiveTasksPerShuffleConf, 0))

  private val coordinatorMessages = new LinkedBlockingQueue[UcxShuffleCoordinatorMasterMessage]

  private val threadpool: ThreadPoolExecutor = {
    val pool = ThreadUtils.newDaemonFixedThreadPool(
      math.max(1, conf.getInt("spark.gluten.ucx.shuffle.coordinator.threads", 4)),
      "ucx-shuffle-coordinator-dispatcher")
    for (_ <- 0 until pool.getCorePoolSize) {
      pool.execute(new MessageLoop)
    }
    pool
  }

  override def registerShuffle(shuffleId: Int, numMaps: Int, numReduces: Int): Unit = {
    logInfo(s"Registering UCX shuffle shuffleId=$shuffleId numMaps=$numMaps numReduces=$numReduces")
    pendingShuffleUnregistrations.remove(shuffleId)
    val existing = shuffleInfos.putIfAbsent(shuffleId, UcxShuffleInfo(numMaps, numReduces))
    if (existing != null && existing != UcxShuffleInfo(numMaps, numReduces)) {
      throw new IllegalArgumentException(
        s"UCX shuffle $shuffleId registered twice with different shape: " +
          s"existing=$existing new=($numMaps,$numReduces)")
    }
  }

  override def registerWriter(
      shuffleId: Int,
      mapId: Long,
      endpoint: UcxShuffleEndpoint): Boolean = {
    if (
      !shuffleInfos.containsKey(shuffleId) ||
      abortedShuffles.containsKey(shuffleId) ||
      isPipelinedGroupAborted(shuffleId)
    ) {
      logWarning(
        s"Ignoring UCX writer registration for inactive shuffleId=$shuffleId " +
          s"mapId=$mapId endpoint=$endpoint")
      return false
    }
    writerStageInactiveReason(
      shuffleId,
      endpoint.stageId,
      endpoint.stageAttemptNumber).foreach {
      reason =>
        logWarning(
          s"Ignoring stale-generation UCX writer registration shuffleId=$shuffleId " +
            s"mapId=$mapId attempt=${endpoint.attemptId} " +
            s"stage=${endpoint.stageId}.${endpoint.stageAttemptNumber} reason=$reason")
        return false
    }
    val registered =
      endpoints.computeIfAbsent(shuffleId, _ => new ConcurrentHashMap[Long, UcxShuffleEndpoint]())
    val current = registered.compute(
      mapId,
      (_, previous) => {
        if (previous == null || endpoint.epoch >= previous.epoch) endpoint else previous
      })
    if (current.attemptId != endpoint.attemptId) {
      logWarning(
        s"Ignoring stale UCX writer registration shuffleId=$shuffleId mapId=$mapId " +
          s"attempt=${endpoint.attemptId} currentAttempt=${current.attemptId}")
      return false
    }
    Option(finishedAttempts.get(shuffleId)).foreach {
      finished =>
        Option(finished.get(mapId)).foreach {
          finishedAttempt =>
            if (finishedAttempt.longValue() != endpoint.attemptId) {
              finished.remove(mapId, finishedAttempt)
            }
        }
    }
    transportGroupForShuffle(shuffleId).foreach {
      transportGroup =>
        val key = writerCreditKey(shuffleId, mapId)
        Option(transportGroup.nativeWriterStates.get(key)).foreach {
          previous =>
            if (previous.attemptId != endpoint.attemptId) {
              transportGroup.nativeWriterStates.remove(key, previous)
            }
        }
    }
    logInfo(
      s"Registered UCX writer endpoint shuffleId=$shuffleId mapId=$mapId " +
        s"attempt=${endpoint.attemptId} executor=${endpoint.executorId} " +
        s"host=${endpoint.host} port=${endpoint.ucxPort} nativeTaskId=${endpoint.nativeTaskId}")
    true
  }

  override def registerReader(
      shuffleId: Int,
      reducePartitionId: Int,
      endpoint: UcxShuffleReaderEndpoint): Boolean = {
    val info = shuffleInfos.get(shuffleId)
    if (
      info == null ||
      abortedShuffles.containsKey(shuffleId) ||
      isPipelinedGroupAborted(shuffleId) ||
      reducePartitionId < 0 ||
      reducePartitionId >= info.numReduces
    ) {
      logWarning(
        s"Ignoring UCX reader registration for inactive shuffleId=$shuffleId " +
          s"reducePartitionId=$reducePartitionId endpoint=$endpoint")
      return false
    }
    readerStageInactiveReason(
      shuffleId,
      endpoint.stageId,
      endpoint.stageAttemptNumber).foreach {
      reason =>
        logWarning(
          s"Ignoring stale-generation UCX reader registration shuffleId=$shuffleId " +
            s"reducePartitionId=$reducePartitionId attempt=${endpoint.taskAttemptId} " +
            s"stage=${endpoint.stageId}.${endpoint.stageAttemptNumber} reason=$reason")
        return false
    }
    val registered = readerEndpoints.computeIfAbsent(
      shuffleId,
      _ => new ConcurrentHashMap[Int, UcxShuffleReaderEndpoint]())
    val current = registered.compute(
      reducePartitionId,
      (_, previous) => {
        if (previous == null || endpoint.epoch >= previous.epoch) endpoint else previous
      })
    if (current.taskAttemptId != endpoint.taskAttemptId) {
      logWarning(
        s"Ignoring stale UCX reader registration shuffleId=$shuffleId " +
          s"reducePartitionId=$reducePartitionId attempt=${endpoint.taskAttemptId} " +
          s"currentAttempt=${current.taskAttemptId}")
      return false
    }
    transportGroupForShuffle(shuffleId).foreach {
      transportGroup =>
        val key = readerStateKey(shuffleId, reducePartitionId)
        Option(transportGroup.nativeReaderStates.get(key)).foreach {
          previous =>
            if (previous.taskAttemptId != endpoint.taskAttemptId) {
              transportGroup.nativeReaderStates.remove(key, previous)
            }
        }
    }
    logInfo(
      s"Registered UCX reader endpoint shuffleId=$shuffleId reducePartitionId=$reducePartitionId " +
        s"attempt=${endpoint.taskAttemptId} executor=${endpoint.executorId} " +
        s"host=${endpoint.host} " +
        s"stage=${endpoint.stageId}.${endpoint.stageAttemptNumber} " +
        s"nativeReaderId=${endpoint.nativeReaderId} coverage=${registered.size}/${info.numReduces}")
    maybeMarkReadersReady(shuffleId)
    true
  }

  override def getAvailableWriters(shuffleId: Int): Option[UcxShuffleEndpointResponse] = {
    val info = shuffleInfos.get(shuffleId)
    if (
      info == null || abortedShuffles.containsKey(shuffleId) ||
      isPipelinedGroupAborted(shuffleId)
    ) {
      return None
    }
    val registered = Option(endpoints.get(shuffleId)).map(_.asScala.toMap).getOrElse(Map.empty)
    val finished =
      Option(finishedAttempts.get(shuffleId))
        .map {
          attempts =>
            attempts.asScala.collect {
              case (mapId, attemptId)
                  if registered.get(mapId).exists(
                    _.attemptId == attemptId.longValue()) =>
                mapId.toLong
            }.toSet
        }
        .getOrElse(Set.empty[Long])
    Some(UcxShuffleEndpointResponse(registered, info.numMaps, finished))
  }

  override def getReaderCoverage(shuffleId: Int): Option[UcxShuffleReaderCoverageResponse] = {
    val info = shuffleInfos.get(shuffleId)
    if (
      info == null || abortedShuffles.containsKey(shuffleId) ||
      isPipelinedGroupAborted(shuffleId)
    ) {
      return None
    }
    val readers = Option(readerEndpoints.get(shuffleId)).map(_.asScala.toMap).getOrElse(Map.empty)
    Some(UcxShuffleReaderCoverageResponse(readers, info.numReduces))
  }

  override def getPipelinedShuffleState(shuffleId: Int): UcxPipelinedShuffleStateResponse = {
    val shuffleAbort = Option(abortedShuffles.get(shuffleId))
    val groupId = Option(shuffleIdToPipelinedGroupId.get(shuffleId))
    groupId match {
      case None =>
        UcxPipelinedShuffleStateResponse(
          shuffleId = shuffleId,
          groupId = None,
          groupState = UcxPipelinedShuffleGroupState.NoGroup,
          readersReady = shuffleAbort.isEmpty,
          readerCoverage = readerCoverageForShuffles(Seq(shuffleId)),
          abortedReason = shuffleAbort
        )
      case Some(id) =>
        val transportGroup = transportGroupForAttempt(id)
        val groupState =
          transportGroup.map(_.groupState(id)).getOrElse {
            Option(abortedPipelinedGroups.get(id))
              .map(_ => UcxPipelinedShuffleGroupState.Aborted)
              .getOrElse(UcxPipelinedShuffleGroupState.Registered)
          }
        val shuffleIds =
          transportGroup
            .flatMap(_.groupMetadata(id))
            .map(pipelinedShuffleIds)
            .getOrElse(shuffleIdsForGroup(id))
        val groupAbort =
          shuffleIds.iterator.flatMap(s => Option(abortedShuffles.get(s))).toSeq.headOption
        val abortedReason =
          shuffleAbort.orElse(transportGroup.flatMap(_.abortedReason)).orElse(
            Option(abortedPipelinedGroups.get(id))).orElse(groupAbort)
        UcxPipelinedShuffleStateResponse(
          shuffleId = shuffleId,
          groupId = Some(id),
          groupState = groupState,
          readersReady =
            abortedReason.isEmpty && readersReadyState(groupState),
          readerCoverage = readerCoverageForShuffles(shuffleIds),
          abortedReason = abortedReason
        )
    }
  }

  override def markWriterFinished(shuffleId: Int, mapId: Long, attemptId: Long): Boolean = {
    if (!shuffleInfos.containsKey(shuffleId)) {
      return false
    }
    writerAttemptInactiveReason(shuffleId, mapId, attemptId).foreach {
      reason =>
        logWarning(
          s"Ignoring stale UCX writer completion shuffleId=$shuffleId mapId=$mapId " +
            s"attemptId=$attemptId reason=$reason")
        return false
    }
    val finished = finishedAttempts.computeIfAbsent(
      shuffleId,
      _ => new ConcurrentHashMap[Long, java.lang.Long]())
    finished.put(mapId, attemptId)
    logInfo(
      s"Marked UCX writer finished shuffleId=$shuffleId mapId=$mapId attemptId=$attemptId " +
        s"finished=${finished.size}/${shuffleInfos.get(shuffleId).numMaps}")
    true
  }

  override def tryAcquireWriterCredit(
      shuffleId: Int,
      mapId: Long,
      attemptId: Long,
      stageId: Int,
      stageAttemptNumber: Int): UcxShuffleWriterCreditResponse = {
    val groupId = groupIdForShuffle(shuffleId)
    if (maxActiveWriterTasksPerShuffle <= 0) {
      return writerCreditResponse(
        granted = true,
        shuffleId = shuffleId,
        groupId = groupId,
        reason = None)
    }
    val inactiveReason =
      writerCreditInactiveReason(shuffleId).orElse(
        writerStageInactiveReason(shuffleId, stageId, stageAttemptNumber))
    if (inactiveReason.isDefined) {
      return writerCreditResponse(
        granted = false,
        shuffleId = shuffleId,
        groupId = groupId,
        reason = inactiveReason)
    }

    writerCreditLock.synchronized {
      val shuffleCredits =
        writerCreditsByShuffle.computeIfAbsent(
          shuffleId,
          _ => new ConcurrentHashMap[Long, java.lang.Long]())
      val existing = Option(shuffleCredits.get(mapId)).map(_.longValue())
      if (existing.contains(attemptId)) {
        return writerCreditResponse(
          granted = true,
          shuffleId = shuffleId,
          groupId = groupId,
          reason = None)
      }

      val granted = shuffleCredits.size() < maxActiveWriterTasksPerShuffle
      if (granted) {
        shuffleCredits.put(mapId, attemptId)
        logInfo(
          s"Granted UCX shuffle writer credit shuffleId=$shuffleId mapId=$mapId " +
            s"attemptId=$attemptId " +
            s"shuffleActive=${shuffleCredits.size()}/$maxActiveWriterTasksPerShuffle")
      }
      writerCreditResponse(
        granted = granted,
        shuffleId = shuffleId,
        groupId = groupId,
        reason = None)
    }
  }

  override def releaseWriterCredit(shuffleId: Int, mapId: Long, attemptId: Long): Boolean = {
    releaseWriterCreditInternal(shuffleId, mapId, attemptId)
    true
  }

  override def reportNativeWriterState(
      state: UcxNativeWriterState): UcxNativeWriterStateResponse = {
    val groupId = groupIdForShuffle(state.shuffleId)
    val transportGroup = transportGroupForShuffle(state.shuffleId)
    val inactiveReason =
      writerCreditInactiveReason(state.shuffleId).orElse(
        writerAttemptInactiveReason(state.shuffleId, state.mapId, state.attemptId))
    if (inactiveReason.isEmpty) {
      transportGroup.foreach {
        group =>
          group.nativeWriterStates.put(writerCreditKey(state.shuffleId, state.mapId), state)
      }
    }
    val writerDone = state.noMoreData || state.finished
    val writerFinishedMarked =
      writerDone && inactiveReason.isEmpty &&
        markWriterFinished(state.shuffleId, state.mapId, state.attemptId)
    val writerCreditReleased =
      writerDone && releaseWriterCreditInternal(state.shuffleId, state.mapId, state.attemptId)
    val response = UcxNativeWriterStateResponse(
      accepted = inactiveReason.isEmpty,
      shuffleId = state.shuffleId,
      groupId = groupId,
      groupState = transportGroup.map(_.state).getOrElse(UcxPipelinedShuffleGroupState.NoGroup),
      writerFinishedMarked = writerFinishedMarked,
      writerCreditReleased = writerCreditReleased,
      reason = inactiveReason)
    if (inactiveReason.isEmpty && state.finished) {
      transportGroup.foreach(maybeFinishDrainingTransportGroup)
    }
    response
  }

  override def reportNativeReaderState(
      state: UcxNativeReaderState): UcxNativeReaderStateResponse = {
    val groupId = groupIdForShuffle(state.shuffleId)
    val transportGroup = transportGroupForShuffle(state.shuffleId)
    val inactiveReason =
      readerStateInactiveReason(state.shuffleId).orElse(
        readerAttemptInactiveReason(
          state.shuffleId,
          state.reducePartitionId,
          state.taskAttemptId,
          state.nativeReaderId))
    if (inactiveReason.isEmpty) {
      transportGroup.foreach {
        group =>
          group.nativeReaderStates.put(
            readerStateKey(state.shuffleId, state.reducePartitionId),
            state)
      }
      if (state.noMoreSplits || state.finished) {
        logInfo(
          s"Reported UCX native reader state shuffleId=${state.shuffleId} " +
            s"reduce=${state.reducePartitionId} taskAttemptId=${state.taskAttemptId} " +
            s"noMoreSplits=${state.noMoreSplits} finished=${state.finished} " +
            s"seenMaps=${state.seenMaps}/${state.expectedMaps} " +
          s"completedMaps=${state.completedMaps}/${state.expectedMaps}")
      }
    }
    UcxNativeReaderStateResponse(
      accepted = inactiveReason.isEmpty,
      shuffleId = state.shuffleId,
      groupId = groupId,
      groupState = transportGroup.map(_.state).getOrElse(UcxPipelinedShuffleGroupState.NoGroup),
      reason = inactiveReason)
  }

  private def releaseWriterCreditInternal(shuffleId: Int, mapId: Long, attemptId: Long): Boolean = {
    writerCreditLock.synchronized {
      val removed =
        Option(writerCreditsByShuffle.get(shuffleId)).exists {
          credits => removeCreditIfSameAttempt(credits, mapId, attemptId)
        }
      if (removed) {
        logInfo(
          s"Released UCX shuffle writer credit shuffleId=$shuffleId " +
            s"mapId=$mapId attemptId=$attemptId")
      }
      removed
    }
  }

  override def abortShuffle(shuffleId: Int, reason: String): Boolean = {
    val groupId = shuffleIdToPipelinedGroupId.get(shuffleId)
    if (groupId != null) {
      return abortPipelinedShuffleGroup(groupId, s"shuffleId=$shuffleId aborted: $reason")
    }
    abortShuffleInternal(shuffleId, reason)
  }

  override def registerPipelinedShuffleGroup(group: PipelinedShuffleGroupMetadata): Unit = {
    pipelinedRegistrationLock.synchronized {
      val groupAttemptId = group.groupAttemptId
      val shuffleIds = pipelinedShuffleIds(group)
      val created = new UcxPipelinedTransportGroupState(groupAttemptId)
      val existing = pipelinedGroups.putIfAbsent(groupAttemptId, created)
      val transportGroup = Option(existing).getOrElse(created)

      writerGenerations(group).foreach {
        case (shuffleId, generation) =>
          Option(shuffleGenerationById.get(shuffleId)).foreach {
            previous =>
              if (previous != generation) {
                resetPipelinedShuffleRuntime(shuffleId)
                logInfo(
                  s"Reset UCX shuffle runtime for new writer generation shuffleId=$shuffleId " +
                    s"previous=${previous.writerStageId}.${previous.writerStageAttempt} " +
                    s"current=${generation.writerStageId}.${generation.writerStageAttempt} " +
                    s"groupAttemptId=$groupAttemptId")
              }
          }
          shuffleGenerationById.put(shuffleId, generation)
      }

      transportGroup.registerGroup(group, shuffleIds)
      val replacedGroups = shuffleIds.flatMap {
        shuffleId =>
          Option(shuffleIdToPipelinedGroupId.put(shuffleId, groupAttemptId))
            .filter(_ != groupAttemptId)
      }.distinct
      replacedGroups.foreach(cleanupDetachedPipelinedGroup)
      Option(abortedPipelinedGroups.get(groupAttemptId)).foreach {
        reason =>
          transportGroup.state = UcxPipelinedShuffleGroupState.Aborted
          transportGroup.abortedReason = Some(reason)
      }
      logInfo(
        s"Registered UCX pipelined transport group logicalGroupId=${group.groupId} " +
          s"groupAttemptId=$groupAttemptId " +
          s"stages=${group.stages.map(_.stageId)} shuffles=$shuffleIds " +
          s"state=${transportGroup.state}")
      if (transportGroup.state == UcxPipelinedShuffleGroupState.Aborted) {
        abortPipelinedShuffleGroup(
          groupAttemptId,
          "group attempt was already aborted before registration")
      }
    }
  }

  override def admitPipelinedShuffleGroup(groupAttemptId: String): Boolean = {
    val transportGroup = transportGroupForAttempt(groupAttemptId).getOrElse {
      logWarning(
        s"Rejected UCX pipelined transport admission for unknown " +
          s"groupAttemptId=$groupAttemptId")
      return false
    }
    if (transportGroup.state == UcxPipelinedShuffleGroupState.Aborted) {
      return false
    }
    transportGroup.admitGroup(groupAttemptId)
    logInfo(
      s"Admitted UCX pipelined transport group groupAttemptId=$groupAttemptId")
    transportGroup
      .groupMetadata(groupAttemptId)
      .map(pipelinedShuffleIds)
      .getOrElse(shuffleIdsForGroup(groupAttemptId))
      .foreach(maybeMarkReadersReady)
    true
  }

  override def completePipelinedShuffleGroup(groupAttemptId: String): Boolean = {
    transportGroupForAttempt(groupAttemptId) match {
      case Some(transportGroup) =>
        transportGroup.completeGroup(groupAttemptId)
        val shuffleIds =
          transportGroup
            .groupMetadata(groupAttemptId)
            .map(pipelinedShuffleIds)
            .getOrElse(shuffleIdsForGroup(groupAttemptId))
        logInfo(
          s"Spark completed UCX pipelined transport group groupAttemptId=$groupAttemptId " +
            s"shuffles=$shuffleIds; waiting for detached native writers to drain")
        maybeFinishDrainingTransportGroup(transportGroup)
      case None =>
        cleanupLegacyPipelinedGroup(groupAttemptId)
    }
    true
  }

  override def abortPipelinedShuffleGroup(
      groupAttemptId: String,
      reason: String): Boolean = {
    abortedPipelinedGroups.put(groupAttemptId, reason)
    val transportGroup = transportGroupForAttempt(groupAttemptId)
    val shuffleIds =
      transportGroup.map(_.shuffleIds.toSeq).getOrElse(shuffleIdsForGroup(groupAttemptId))
    transportGroup.foreach { group =>
      group.state = UcxPipelinedShuffleGroupState.Aborted
      group.abortedReason = Some(reason)
      group.abortAllGroups()
    }
    shuffleIds.foreach {
      shuffleId =>
        abortShuffleInternal(
          shuffleId,
          s"pipelined group $groupAttemptId aborted: $reason")
    }
    logWarning(
      s"Aborted UCX pipelined transport group groupAttemptId=$groupAttemptId " +
        s"shuffles=$shuffleIds reason=$reason")
    true
  }

  private def cleanupDetachedPipelinedGroup(groupAttemptId: String): Unit = {
    if (!shuffleIdToPipelinedGroupId.containsValue(groupAttemptId)) {
      Option(pipelinedGroups.remove(groupAttemptId)).foreach {
        group =>
          group.nativeWriterStates.clear()
          group.nativeReaderStates.clear()
      }
      abortedPipelinedGroups.remove(groupAttemptId)
    }
  }

  private def cleanupLegacyPipelinedGroup(groupId: String): Unit = {
    val shuffleIds = shuffleIdsForGroup(groupId)
    shuffleIds.foreach(clearPipelinedShuffleState)
    abortedPipelinedGroups.remove(groupId)
    logInfo(s"Completed legacy UCX pipelined group groupId=$groupId shuffles=$shuffleIds")
  }

  private def cleanupPipelinedTransportGroup(
      transportGroup: UcxPipelinedTransportGroupState): Unit = {
    val shuffleIds = transportGroup.shuffleIds.toSeq
    transportGroup.nativeWriterStates.clear()
    transportGroup.nativeReaderStates.clear()
    shuffleIds.foreach(clearPipelinedShuffleState)
    transportGroup.groupIds.foreach {
      groupId =>
        abortedPipelinedGroups.remove(groupId)
    }
    pipelinedGroups.remove(transportGroup.groupAttemptId, transportGroup)
    logInfo(
      s"Cleaned UCX pipelined transport group " +
        s"groupAttemptId=${transportGroup.groupAttemptId} shuffles=$shuffleIds")
  }

  private def maybeFinishDrainingTransportGroup(
      transportGroup: UcxPipelinedTransportGroupState): Unit = {
    pipelinedRegistrationLock.synchronized {
      if (
        transportGroup.state != UcxPipelinedShuffleGroupState.Draining ||
        !allNativeWritersDrained(transportGroup)
      ) {
        return
      }

      transportGroup.groupIds.foreach(transportGroup.finishGroup)
      val shuffleIds = transportGroup.shuffleIds.toSeq
      shuffleIds.foreach(resetPipelinedShuffleRuntime)
      logInfo(
        s"Finished draining UCX pipelined transport group " +
          s"groupAttemptId=${transportGroup.groupAttemptId} shuffles=$shuffleIds")

      if (shuffleIds.forall(pendingShuffleUnregistrations.contains)) {
        cleanupPipelinedTransportGroup(transportGroup)
      }
    }
  }

  private def allNativeWritersDrained(
      transportGroup: UcxPipelinedTransportGroupState): Boolean = {
    transportGroup.shuffleIds.forall {
      shuffleId =>
        Option(endpoints.get(shuffleId)).forall {
          registered =>
            registered.asScala.forall {
              case (mapId, endpoint) =>
                Option(
                  transportGroup.nativeWriterStates.get(writerCreditKey(shuffleId, mapId)))
                  .exists(
                    state =>
                      state.attemptId == endpoint.attemptId &&
                        state.noMoreData &&
                        state.finished)
            }
        }
    }
  }

  private def clearPipelinedShuffleState(shuffleId: Int): Unit = {
    pendingShuffleUnregistrations.remove(shuffleId)
    shuffleInfos.remove(shuffleId)
    endpoints.remove(shuffleId)
    readerEndpoints.remove(shuffleId)
    finishedAttempts.remove(shuffleId)
    abortedShuffles.remove(shuffleId)
    clearWriterCreditsForShuffle(shuffleId)
    clearNativeStatesForShuffle(shuffleId)
    shuffleIdToPipelinedGroupId.remove(shuffleId)
    shuffleGenerationById.remove(shuffleId)
  }

  private def resetPipelinedShuffleRuntime(shuffleId: Int): Unit = {
    endpoints.remove(shuffleId)
    readerEndpoints.remove(shuffleId)
    finishedAttempts.remove(shuffleId)
    abortedShuffles.remove(shuffleId)
    clearWriterCreditsForShuffle(shuffleId)
    clearNativeStatesForShuffle(shuffleId)
  }

  private def abortShuffleInternal(shuffleId: Int, reason: String): Boolean = {
    abortedShuffles.put(shuffleId, reason)
    endpoints.remove(shuffleId)
    readerEndpoints.remove(shuffleId)
    finishedAttempts.remove(shuffleId)
    clearWriterCreditsForShuffle(shuffleId)
    clearNativeStatesForShuffle(shuffleId)
    logWarning(s"Aborted UCX shuffle shuffleId=$shuffleId reason=$reason")
    true
  }

  override def unregisterShuffle(shuffleId: Int): Unit = {
    val activeGroup = transportGroupForShuffle(shuffleId)
    if (activeGroup.exists(_.state == UcxPipelinedShuffleGroupState.Draining)) {
      pendingShuffleUnregistrations.add(shuffleId)
      logInfo(
        s"Deferred UCX shuffle unregistration until native drain shuffleId=$shuffleId " +
          s"groupAttemptId=${activeGroup.get.groupAttemptId}")
      return
    }

    shuffleInfos.remove(shuffleId)
    endpoints.remove(shuffleId)
    readerEndpoints.remove(shuffleId)
    finishedAttempts.remove(shuffleId)
    abortedShuffles.remove(shuffleId)
    clearWriterCreditsForShuffle(shuffleId)
    clearNativeStatesForShuffle(shuffleId)
    pendingShuffleUnregistrations.remove(shuffleId)
    val groupId = shuffleIdToPipelinedGroupId.remove(shuffleId)
    shuffleGenerationById.remove(shuffleId)
    if (groupId != null && !shuffleIdToPipelinedGroupId.containsValue(groupId)) {
      transportGroupForAttempt(groupId).foreach {
        transportGroup =>
          if (
            transportGroup.state == UcxPipelinedShuffleGroupState.Aborted ||
            transportGroup.state == UcxPipelinedShuffleGroupState.Finished
          ) {
            cleanupPipelinedTransportGroup(transportGroup)
          }
      }
    }
    logInfo(s"Unregistered UCX shuffle shuffleId=$shuffleId")
  }

  private def isPipelinedGroupAborted(shuffleId: Int): Boolean = {
    transportGroupForShuffle(shuffleId)
      .exists(_.state == UcxPipelinedShuffleGroupState.Aborted) ||
    groupIdForShuffle(shuffleId).exists(abortedPipelinedGroups.containsKey)
  }

  private def pipelinedShuffleIds(group: PipelinedShuffleGroupMetadata): Seq[Int] = {
    group.stages.flatMap(_.shuffleId).distinct
  }

  private def writerGenerations(
      group: PipelinedShuffleGroupMetadata): Seq[(Int, UcxPipelinedShuffleGeneration)] = {
    group.stages.flatMap {
      stage =>
        stage.shuffleId.map {
          shuffleId =>
            shuffleId -> UcxPipelinedShuffleGeneration(stage.stageId, stage.attemptId)
        }
    }
  }

  private def shuffleIdsForGroup(groupId: String): Seq[Int] = {
    shuffleIdToPipelinedGroupId.asScala.collect {
      case (shuffleId, mappedGroupId) if mappedGroupId == groupId => shuffleId
    }.toSeq
  }

  private def maybeMarkReadersReady(shuffleId: Int): Unit = {
    val transportGroup = transportGroupForShuffle(shuffleId).orNull
    val groupId = groupIdForShuffle(shuffleId).orNull
    if (
      transportGroup == null || groupId == null ||
      transportGroup.groupState(groupId) != UcxPipelinedShuffleGroupState.Admitted
    ) {
      return
    }
    val shuffleIds =
      transportGroup
        .groupMetadata(groupId)
        .map(pipelinedShuffleIds)
        .getOrElse(shuffleIdsForGroup(groupId))
    val allReadersReady = shuffleIds.nonEmpty && shuffleIds.forall {
      id =>
        val info = shuffleInfos.get(id)
        val readers = readerEndpoints.get(id)
        info != null && readers != null && readers.size >= info.numReduces
    }
    if (allReadersReady) {
      transportGroup.markGroupReadersReady(groupId)
      logInfo(
        s"UCX pipelined transport group groupAttemptId=$groupId " +
          s"reached READERS_READY " +
          s"readerCoverage=${readerCoverageSummary(shuffleIds)}")
    }
  }

  private def readersReadyState(state: String): Boolean = {
    state == UcxPipelinedShuffleGroupState.ReadersReady ||
    state == UcxPipelinedShuffleGroupState.Draining ||
    state == UcxPipelinedShuffleGroupState.Finished
  }

  private def readerCoverageSummary(shuffleIds: Seq[Int]): String = {
    shuffleIds.sorted
      .map {
        shuffleId =>
          val expected = Option(shuffleInfos.get(shuffleId)).map(_.numReduces).getOrElse(0)
          val actual = Option(readerEndpoints.get(shuffleId)).map(_.size).getOrElse(0)
          s"$shuffleId=$actual/$expected"
      }
      .mkString("[", ",", "]")
  }

  private def readerCoverageForShuffles(
      shuffleIds: Seq[Int]): Map[Int, UcxShuffleReaderCoverage] = {
    shuffleIds.map {
      shuffleId =>
        val expected = Option(shuffleInfos.get(shuffleId)).map(_.numReduces).getOrElse(0)
        val actual = Option(readerEndpoints.get(shuffleId)).map(_.size).getOrElse(0)
        shuffleId -> UcxShuffleReaderCoverage(actual, expected)
    }.toMap
  }

  private def writerStageInactiveReason(
      shuffleId: Int,
      stageId: Int,
      stageAttemptNumber: Int): Option[String] = {
    transportGroupForShuffle(shuffleId).flatMap {
      transportGroup =>
        transportGroup.writerStage(shuffleId) match {
          case Some(stage)
              if stage.stageId == stageId && stage.attemptId == stageAttemptNumber =>
            None
          case Some(stage) =>
            Some(
              s"writer stage=$stageId.$stageAttemptNumber does not match current " +
                s"stage=${stage.stageId}.${stage.attemptId} " +
                s"for groupId=${transportGroup.groupId}")
          case None =>
            Some(
              s"shuffleId=$shuffleId has no writer stage in current " +
                s"groupId=${transportGroup.groupId}")
        }
    }
  }

  private def readerStageInactiveReason(
      shuffleId: Int,
      stageId: Int,
      stageAttemptNumber: Int): Option[String] = {
    transportGroupForShuffle(shuffleId).flatMap {
      transportGroup =>
        transportGroup.readerStage(stageId) match {
          case Some(stage)
              if stage.attemptId == stageAttemptNumber &&
                stage.pipelinedParentShuffleIds.contains(shuffleId) =>
            None
          case Some(stage) if stage.attemptId != stageAttemptNumber =>
            Some(
              s"reader stage=$stageId.$stageAttemptNumber does not match current " +
                s"stage=${stage.stageId}.${stage.attemptId} " +
                s"for groupId=${transportGroup.groupId}")
          case Some(_) =>
            Some(
              s"reader stage=$stageId.$stageAttemptNumber does not consume shuffleId=$shuffleId " +
                s"in current groupId=${transportGroup.groupId}")
          case None =>
            Some(
              s"reader stage=$stageId.$stageAttemptNumber is not in current " +
                s"groupId=${transportGroup.groupId}")
        }
    }
  }

  private def writerAttemptInactiveReason(
      shuffleId: Int,
      mapId: Long,
      attemptId: Long): Option[String] = {
    Option(endpoints.get(shuffleId))
      .flatMap(endpoints => Option(endpoints.get(mapId))) match {
      case Some(endpoint) if endpoint.attemptId == attemptId =>
        None
      case Some(endpoint) =>
        Some(
          s"stale writer attemptId=$attemptId; current attemptId=${endpoint.attemptId} " +
            s"for shuffleId=$shuffleId mapId=$mapId")
      case None =>
        Some(
          s"writer endpoint is not registered for shuffleId=$shuffleId mapId=$mapId " +
            s"attemptId=$attemptId")
    }
  }

  private def readerAttemptInactiveReason(
      shuffleId: Int,
      reducePartitionId: Int,
      taskAttemptId: Long,
      nativeReaderId: String): Option[String] = {
    Option(readerEndpoints.get(shuffleId))
      .flatMap(readers => Option(readers.get(reducePartitionId))) match {
      case Some(endpoint)
          if endpoint.taskAttemptId == taskAttemptId &&
            endpoint.nativeReaderId == nativeReaderId =>
        None
      case Some(endpoint) =>
        Some(
          s"stale reader taskAttemptId=$taskAttemptId nativeReaderId=$nativeReaderId; " +
            s"current taskAttemptId=${endpoint.taskAttemptId} " +
            s"nativeReaderId=${endpoint.nativeReaderId} for shuffleId=$shuffleId " +
            s"reducePartitionId=$reducePartitionId")
      case None =>
        Some(
          s"reader endpoint is not registered for shuffleId=$shuffleId " +
            s"reducePartitionId=$reducePartitionId taskAttemptId=$taskAttemptId")
    }
  }

  private def writerCreditInactiveReason(shuffleId: Int): Option[String] = {
    if (!shuffleInfos.containsKey(shuffleId)) {
      Some(s"shuffleId=$shuffleId is not registered")
    } else {
      Option(abortedShuffles.get(shuffleId)).orElse {
        transportGroupForShuffle(shuffleId)
          .flatMap {
            transportGroup =>
              transportGroup.abortedReason.orElse {
                transportGroup.state match {
                  case UcxPipelinedShuffleGroupState.Aborted =>
                    Some(s"pipelined transport group ${transportGroup.groupId} is aborted")
                  case UcxPipelinedShuffleGroupState.Finished =>
                    Some(s"pipelined transport group ${transportGroup.groupId} is finished")
                  case _ =>
                    None
                }
              }
          }
          .orElse {
            groupIdForShuffle(shuffleId).flatMap {
              id => Option(abortedPipelinedGroups.get(id))
            }
          }
      }
    }
  }

  private def readerStateInactiveReason(shuffleId: Int): Option[String] = {
    if (!shuffleInfos.containsKey(shuffleId)) {
      Some(s"shuffleId=$shuffleId is not registered")
    } else {
      Option(abortedShuffles.get(shuffleId)).orElse {
        transportGroupForShuffle(shuffleId)
          .flatMap {
            transportGroup =>
              transportGroup.abortedReason.orElse {
                if (transportGroup.state == UcxPipelinedShuffleGroupState.Aborted) {
                  Some(s"pipelined transport group ${transportGroup.groupId} is aborted")
                } else {
                  None
                }
              }
          }
          .orElse {
            groupIdForShuffle(shuffleId).flatMap {
              id => Option(abortedPipelinedGroups.get(id))
            }
          }
      }
    }
  }

  private def writerCreditResponse(
      granted: Boolean,
      shuffleId: Int,
      groupId: Option[String],
      reason: Option[String]): UcxShuffleWriterCreditResponse = {
    val shuffleActive =
      Option(writerCreditsByShuffle.get(shuffleId)).map(_.size()).getOrElse(0)
    UcxShuffleWriterCreditResponse(
      granted = granted,
      shuffleId = shuffleId,
      groupId = groupId,
      activeShuffleWriters = shuffleActive,
      maxShuffleWriters = maxActiveWriterTasksPerShuffle,
      reason = reason)
  }

  private def groupIdForShuffle(shuffleId: Int): Option[String] = {
    Option(shuffleIdToPipelinedGroupId.get(shuffleId))
  }

  private def transportGroupForAttempt(
      groupId: String): Option[UcxPipelinedTransportGroupState] = {
    Option(pipelinedGroups.get(groupId))
  }

  private def transportGroupForShuffle(
      shuffleId: Int): Option[UcxPipelinedTransportGroupState] = {
    groupIdForShuffle(shuffleId).flatMap(transportGroupForAttempt)
  }

  private def writerCreditKey(shuffleId: Int, mapId: Long): String = {
    s"$shuffleId:$mapId"
  }

  private def readerStateKey(shuffleId: Int, reducePartitionId: Int): String = {
    s"$shuffleId:$reducePartitionId"
  }

  private def removeCreditIfSameAttempt[K](
      credits: ConcurrentHashMap[K, java.lang.Long],
      key: K,
      attemptId: Long): Boolean = {
    Option(credits.get(key)).exists {
      current =>
        current.longValue() == attemptId && {
          credits.remove(key, current)
          true
        }
    }
  }

  private def clearWriterCreditsForShuffle(shuffleId: Int): Unit = {
    writerCreditLock.synchronized {
      writerCreditsByShuffle.remove(shuffleId)
    }
  }

  private def clearNativeStatesForShuffle(shuffleId: Int): Unit = {
    val prefix = s"$shuffleId:"
    pipelinedGroups.asScala.values.foreach {
      transportGroup =>
        transportGroup.nativeWriterStates
          .keySet()
          .asScala
          .filter(_.startsWith(prefix))
          .foreach(key => transportGroup.nativeWriterStates.remove(key))
        transportGroup.nativeReaderStates
          .keySet()
          .asScala
          .filter(_.startsWith(prefix))
          .foreach(key => transportGroup.nativeReaderStates.remove(key))
    }
  }

  def post(message: UcxShuffleCoordinatorMasterMessage): Unit = {
    coordinatorMessages.offer(message)
  }

  override def stop(): Unit = {
    coordinatorMessages.offer(UcxShuffleCoordinatorMaster.PoisonPill)
    threadpool.shutdown()
    if (coordinatorEndpoint != null) {
      try {
        sendCoordinator(StopUcxShuffleCoordinator)
      } catch {
        case e: SparkException =>
          logError(log"Could not tell UCX shuffle coordinator we are stopping.", e)
      }
      coordinatorEndpoint = null
    }
  }

  private class MessageLoop extends Runnable {
    override def run(): Unit = {
      try {
        while (true) {
          try {
            coordinatorMessages.take() match {
              case UcxShuffleCoordinatorMaster.PoisonPill =>
                coordinatorMessages.offer(UcxShuffleCoordinatorMaster.PoisonPill)
                return
              case RegisterUcxShuffleMasterMessage(shuffleId, numMaps, numReduces, context) =>
                registerShuffle(shuffleId, numMaps, numReduces)
                context.reply(true)
              case RegisterUcxShuffleWriterMasterMessage(shuffleId, mapId, endpoint, context) =>
                context.reply(registerWriter(shuffleId, mapId, endpoint))
              case RegisterUcxShuffleReaderMasterMessage(
                    shuffleId,
                    reducePartitionId,
                    endpoint,
                    context) =>
                context.reply(registerReader(shuffleId, reducePartitionId, endpoint))
              case GetAvailableUcxShuffleWritersMasterMessage(shuffleId, context) =>
                context.reply(getAvailableWriters(shuffleId))
              case GetUcxShuffleReaderCoverageMasterMessage(shuffleId, context) =>
                context.reply(getReaderCoverage(shuffleId))
              case GetUcxPipelinedShuffleStateMasterMessage(shuffleId, context) =>
                context.reply(getPipelinedShuffleState(shuffleId))
              case MarkUcxShuffleWriterFinishedMasterMessage(
                    shuffleId,
                    mapId,
                    attemptId,
                    context) =>
                context.reply(markWriterFinished(shuffleId, mapId, attemptId))
              case TryAcquireUcxShuffleWriterCreditMasterMessage(
                    shuffleId,
                    mapId,
                    attemptId,
                    stageId,
                    stageAttemptNumber,
                    context) =>
                context.reply(
                  tryAcquireWriterCredit(
                    shuffleId,
                    mapId,
                    attemptId,
                    stageId,
                    stageAttemptNumber))
              case ReleaseUcxShuffleWriterCreditMasterMessage(
                    shuffleId,
                    mapId,
                    attemptId,
                    context) =>
                context.reply(releaseWriterCredit(shuffleId, mapId, attemptId))
              case ReportUcxNativeWriterStateMasterMessage(state, context) =>
                context.reply(reportNativeWriterState(state))
              case ReportUcxNativeReaderStateMasterMessage(state, context) =>
                context.reply(reportNativeReaderState(state))
              case AbortUcxShuffleMasterMessage(shuffleId, reason, context) =>
                context.reply(abortShuffle(shuffleId, reason))
              case RegisterUcxPipelinedShuffleGroupMasterMessage(group, context) =>
                registerPipelinedShuffleGroup(group)
                context.reply(true)
              case AdmitUcxPipelinedShuffleGroupMasterMessage(groupId, context) =>
                context.reply(admitPipelinedShuffleGroup(groupId))
              case CompleteUcxPipelinedShuffleGroupMasterMessage(groupId, context) =>
                context.reply(completePipelinedShuffleGroup(groupId))
              case AbortUcxPipelinedShuffleGroupMasterMessage(groupId, reason, context) =>
                context.reply(abortPipelinedShuffleGroup(groupId, reason))
            }
          } catch {
            case NonFatal(e) => logError(s"Error in UCX shuffle coordinator message loop", e)
          }
        }
      } catch {
        case _: InterruptedException =>
      }
    }
  }
}

private[spark] object UcxShuffleCoordinatorMaster {
  val PoisonPill: UcxShuffleCoordinatorMasterMessage =
    GetAvailableUcxShuffleWritersMasterMessage(-99, null)
}

private[spark] class UcxShuffleCoordinatorWorker(conf: SparkConf)
  extends UcxShuffleCoordinator(conf) {

  override def registerShuffle(shuffleId: Int, numMaps: Int, numReduces: Int): Unit = {
    sendCoordinator(RegisterUcxShuffle(shuffleId, numMaps, numReduces))
  }

  override def registerWriter(
      shuffleId: Int,
      mapId: Long,
      endpoint: UcxShuffleEndpoint): Boolean = {
    askCoordinator[Boolean](RegisterUcxShuffleWriter(shuffleId, mapId, endpoint))
  }

  override def registerReader(
      shuffleId: Int,
      reducePartitionId: Int,
      endpoint: UcxShuffleReaderEndpoint): Boolean = {
    askCoordinator[Boolean](RegisterUcxShuffleReader(shuffleId, reducePartitionId, endpoint))
  }

  override def getAvailableWriters(shuffleId: Int): Option[UcxShuffleEndpointResponse] = {
    askCoordinator[Option[UcxShuffleEndpointResponse]](GetAvailableUcxShuffleWriters(shuffleId))
  }

  override def getReaderCoverage(shuffleId: Int): Option[UcxShuffleReaderCoverageResponse] = {
    askCoordinator[Option[UcxShuffleReaderCoverageResponse]](GetUcxShuffleReaderCoverage(shuffleId))
  }

  override def getPipelinedShuffleState(shuffleId: Int): UcxPipelinedShuffleStateResponse = {
    askCoordinator[UcxPipelinedShuffleStateResponse](GetUcxPipelinedShuffleState(shuffleId))
  }

  override def markWriterFinished(shuffleId: Int, mapId: Long, attemptId: Long): Boolean = {
    askCoordinator[Boolean](MarkUcxShuffleWriterFinished(shuffleId, mapId, attemptId))
  }

  override def tryAcquireWriterCredit(
      shuffleId: Int,
      mapId: Long,
      attemptId: Long,
      stageId: Int,
      stageAttemptNumber: Int): UcxShuffleWriterCreditResponse = {
    askCoordinator[UcxShuffleWriterCreditResponse](
      TryAcquireUcxShuffleWriterCredit(
        shuffleId,
        mapId,
        attemptId,
        stageId,
        stageAttemptNumber))
  }

  override def releaseWriterCredit(shuffleId: Int, mapId: Long, attemptId: Long): Boolean = {
    askCoordinator[Boolean](ReleaseUcxShuffleWriterCredit(shuffleId, mapId, attemptId))
  }

  override def reportNativeWriterState(
      state: UcxNativeWriterState): UcxNativeWriterStateResponse = {
    askCoordinator[UcxNativeWriterStateResponse](ReportUcxNativeWriterState(state))
  }

  override def reportNativeReaderState(
      state: UcxNativeReaderState): UcxNativeReaderStateResponse = {
    askCoordinator[UcxNativeReaderStateResponse](ReportUcxNativeReaderState(state))
  }

  override def abortShuffle(shuffleId: Int, reason: String): Boolean = {
    askCoordinator[Boolean](AbortUcxShuffle(shuffleId, reason))
  }

  override def registerPipelinedShuffleGroup(group: PipelinedShuffleGroupMetadata): Unit = {
    sendCoordinator(RegisterUcxPipelinedShuffleGroup(group))
  }

  override def admitPipelinedShuffleGroup(groupAttemptId: String): Boolean = {
    askCoordinator[Boolean](AdmitUcxPipelinedShuffleGroup(groupAttemptId))
  }

  override def completePipelinedShuffleGroup(groupAttemptId: String): Boolean = {
    askCoordinator[Boolean](CompleteUcxPipelinedShuffleGroup(groupAttemptId))
  }

  override def abortPipelinedShuffleGroup(
      groupAttemptId: String,
      reason: String): Boolean = {
    askCoordinator[Boolean](AbortUcxPipelinedShuffleGroup(groupAttemptId, reason))
  }

}

private[spark] class UcxShuffleCoordinatorEndpoint(
    override val rpcEnv: RpcEnv,
    coordinator: UcxShuffleCoordinatorMaster,
    conf: SparkConf)
  extends RpcEndpoint
  with Logging {

  override def receiveAndReply(context: RpcCallContext): PartialFunction[Any, Unit] = {
    case RegisterUcxShuffle(shuffleId, numMaps, numReduces) =>
      coordinator.post(RegisterUcxShuffleMasterMessage(shuffleId, numMaps, numReduces, context))

    case RegisterUcxShuffleWriter(shuffleId, mapId, endpoint) =>
      coordinator.post(RegisterUcxShuffleWriterMasterMessage(shuffleId, mapId, endpoint, context))

    case RegisterUcxShuffleReader(shuffleId, reducePartitionId, endpoint) =>
      coordinator.post(
        RegisterUcxShuffleReaderMasterMessage(shuffleId, reducePartitionId, endpoint, context))

    case GetAvailableUcxShuffleWriters(shuffleId) =>
      coordinator.post(GetAvailableUcxShuffleWritersMasterMessage(shuffleId, context))

    case GetUcxShuffleReaderCoverage(shuffleId) =>
      coordinator.post(GetUcxShuffleReaderCoverageMasterMessage(shuffleId, context))

    case GetUcxPipelinedShuffleState(shuffleId) =>
      coordinator.post(GetUcxPipelinedShuffleStateMasterMessage(shuffleId, context))

    case MarkUcxShuffleWriterFinished(shuffleId, mapId, attemptId) =>
      coordinator.post(
        MarkUcxShuffleWriterFinishedMasterMessage(shuffleId, mapId, attemptId, context))

    case TryAcquireUcxShuffleWriterCredit(
          shuffleId,
          mapId,
          attemptId,
          stageId,
          stageAttemptNumber) =>
      coordinator.post(
        TryAcquireUcxShuffleWriterCreditMasterMessage(
          shuffleId,
          mapId,
          attemptId,
          stageId,
          stageAttemptNumber,
          context))

    case ReleaseUcxShuffleWriterCredit(shuffleId, mapId, attemptId) =>
      coordinator.post(
        ReleaseUcxShuffleWriterCreditMasterMessage(shuffleId, mapId, attemptId, context))

    case ReportUcxNativeWriterState(state) =>
      coordinator.post(ReportUcxNativeWriterStateMasterMessage(state, context))

    case ReportUcxNativeReaderState(state) =>
      coordinator.post(ReportUcxNativeReaderStateMasterMessage(state, context))

    case AbortUcxShuffle(shuffleId, reason) =>
      coordinator.post(AbortUcxShuffleMasterMessage(shuffleId, reason, context))

    case RegisterUcxPipelinedShuffleGroup(group) =>
      coordinator.post(RegisterUcxPipelinedShuffleGroupMasterMessage(group, context))

    case AdmitUcxPipelinedShuffleGroup(groupId) =>
      coordinator.post(AdmitUcxPipelinedShuffleGroupMasterMessage(groupId, context))

    case CompleteUcxPipelinedShuffleGroup(groupId) =>
      coordinator.post(CompleteUcxPipelinedShuffleGroupMasterMessage(groupId, context))

    case AbortUcxPipelinedShuffleGroup(groupId, reason) =>
      coordinator.post(AbortUcxPipelinedShuffleGroupMasterMessage(groupId, reason, context))

    case StopUcxShuffleCoordinator =>
      logInfo(log"UCX shuffle coordinator endpoint stopped.")
      stop()
      context.reply(true)
  }
}

private[spark] object UcxShuffleCoordinator extends Logging {
  val EndpointName: String = "UcxShuffleCoordinator"

  @volatile private var coordinator: Option[UcxShuffleCoordinator] = None

  def getOrCreate(conf: SparkConf, isDriver: Boolean): UcxShuffleCoordinator = synchronized {
    coordinator.getOrElse {
      val env = SparkEnv.get
      val created = if (isDriver) {
        val master = new UcxShuffleCoordinatorMaster(conf)
        master.coordinatorEndpoint = env.rpcEnv.setupEndpoint(
          EndpointName,
          new UcxShuffleCoordinatorEndpoint(env.rpcEnv, master, conf))
        master
      } else {
        val worker = new UcxShuffleCoordinatorWorker(conf)
        worker.coordinatorEndpoint = RpcUtils.makeDriverRef(EndpointName, conf, env.rpcEnv)
        worker
      }
      coordinator = Some(created)
      logInfo(s"Initialized UCX shuffle coordinator (isDriver=$isDriver)")
      created
    }
  }

  def stop(): Unit = synchronized {
    coordinator.foreach(_.stop())
    coordinator = None
  }
}
