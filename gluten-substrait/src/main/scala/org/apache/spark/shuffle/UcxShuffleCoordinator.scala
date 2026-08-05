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

private[spark] case class UcxPipelinedShuffleStageMetadata(
    stageId: Int,
    attemptId: Int,
    numTasks: Int,
    shuffleId: Option[Int],
    pipelinedParentShuffleIds: Seq[Int] = Seq.empty)

private[spark] case class UcxPipelinedShuffleGroupMetadata(
    groupId: String,
    jobId: Int,
    queryExecutionId: Option[Long],
    stages: Seq[UcxPipelinedShuffleStageMetadata])

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

private[spark] case class UcxPipelinedQueryRuntimeSummary(
    queryId: String,
    activeWriterCredits: Int,
    nativeWriterStates: Int,
    finishedWriters: Int,
    expectedWriters: Int,
    queuedBytes: Long,
    blockedWriters: Int,
    registeredReaders: Int,
    finishedReaders: Int,
    expectedReaders: Int,
    backpressured: Boolean)

private[spark] case class UcxPipelinedShuffleStateResponse(
    shuffleId: Int,
    groupId: Option[String],
    groupState: String,
    readersReady: Boolean,
    readerCoverage: Map[Int, UcxShuffleReaderCoverage],
    abortedReason: Option[String],
    querySummary: Option[UcxPipelinedQueryRuntimeSummary] = None)

private[spark] case class UcxShuffleWriterCreditResponse(
    granted: Boolean,
    shuffleId: Int,
    groupId: Option[String],
    activeShuffleWriters: Int,
    maxShuffleWriters: Int,
    activeGroupWriters: Int,
    maxGroupWriters: Int,
    reason: Option[String],
    queryId: Option[String] = None,
    queryState: Option[String] = None,
    queryQueuedBytes: Long = 0L,
    queryBlockedWriters: Int = 0,
    queryBackpressured: Boolean = false)

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
    queryId: Option[String],
    queryState: String,
    writerFinishedMarked: Boolean,
    writerCreditReleased: Boolean,
    queryQueuedBytes: Long,
    queryBlockedWriters: Int,
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
    queryId: Option[String],
    queryState: String,
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

private[spark] case class RegisterUcxPipelinedShuffleGroup(group: UcxPipelinedShuffleGroupMetadata)
  extends UcxShuffleCoordinatorMessage

private[spark] case class AdmitUcxPipelinedShuffleGroup(groupId: String)
  extends UcxShuffleCoordinatorMessage

private[spark] case class CompleteUcxPipelinedShuffleGroup(groupId: String)
  extends UcxShuffleCoordinatorMessage

private[spark] case class AbortUcxPipelinedShuffleGroup(groupId: String, reason: String)
  extends UcxShuffleCoordinatorMessage

private[spark] case class CompleteUcxPipelinedQuery(queryExecutionId: Long)
  extends UcxShuffleCoordinatorMessage

private[spark] case class AbortUcxPipelinedQuery(queryExecutionId: Long, reason: String)
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
    group: UcxPipelinedShuffleGroupMetadata,
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

private[spark] case class CompleteUcxPipelinedQueryMasterMessage(
    queryExecutionId: Long,
    context: RpcCallContext)
  extends UcxShuffleCoordinatorMasterMessage

private[spark] case class AbortUcxPipelinedQueryMasterMessage(
    queryExecutionId: Long,
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

private[spark] class UcxPipelinedQueryState(
    val queryKey: String,
    val queryExecutionId: Option[Long]) {

  @volatile var state: String = UcxPipelinedShuffleGroupState.Registered
  @volatile var abortedReason: Option[String] = None

  private val metadataByGroup =
    new ConcurrentHashMap[String, UcxPipelinedShuffleGroupMetadata]()
  private val stateByGroup = new ConcurrentHashMap[String, String]()
  private val shuffleIdSet = ConcurrentHashMap.newKeySet[Int]()
  @volatile private var currentShuffleDepths = Map.empty[Int, Int]

  val activeWriterCredits = new ConcurrentHashMap[String, java.lang.Long]()
  val nativeWriterStates = new ConcurrentHashMap[String, UcxNativeWriterState]()
  val nativeReaderStates = new ConcurrentHashMap[String, UcxNativeReaderState]()
  @volatile var backpressured: Boolean = false
  @volatile var producerLaunchPaused: Boolean = false

  def registerGroup(group: UcxPipelinedShuffleGroupMetadata, shuffleIds: Seq[Int]): Unit =
    synchronized {
      val previous = Option(metadataByGroup.put(group.groupId, group))
      if (previous.forall(groupGeneration(_) != groupGeneration(group))) {
        stateByGroup.put(group.groupId, UcxPipelinedShuffleGroupState.Registered)
      }
      shuffleIdSet.clear()
      metadataByGroup.values().asScala
        .flatMap(_.stages.flatMap(_.shuffleId))
        .foreach(shuffleIdSet.add)
      currentShuffleDepths = computeShuffleDepths()
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
    stateByGroup.put(groupId, UcxPipelinedShuffleGroupState.Finished)
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

  def groupMetadata(groupId: String): Option[UcxPipelinedShuffleGroupMetadata] =
    Option(metadataByGroup.get(groupId))

  def groupIdsForShuffle(shuffleId: Int): Set[String] =
    metadataByGroup.asScala.collect {
      case (groupId, metadata)
          if metadata.stages.exists(_.shuffleId.contains(shuffleId)) =>
        groupId
    }.toSet

  def writerStage(shuffleId: Int): Option[UcxPipelinedShuffleStageMetadata] =
    metadataByGroup.values().asScala
      .flatMap(_.stages)
      .filter(_.shuffleId.contains(shuffleId))
      .toSeq
      .sortBy(stage => (stage.stageId, stage.attemptId))
      .lastOption

  def readerStage(stageId: Int): Option[UcxPipelinedShuffleStageMetadata] =
    metadataByGroup.values().asScala
      .flatMap(_.stages)
      .filter(_.stageId == stageId)
      .toSeq
      .sortBy(_.attemptId)
      .lastOption

  def shuffleIds: Set[Int] = shuffleIdSet.asScala.toSet

  def activeShuffleIds: Set[Int] =
    shuffleIds.filter { shuffleId =>
      groupIdsForShuffle(shuffleId).exists { groupId =>
        val currentState = groupState(groupId)
        currentState != UcxPipelinedShuffleGroupState.Finished &&
        currentState != UcxPipelinedShuffleGroupState.Aborted
      }
    }

  def groupId: String = groupIds.toSeq.sorted.mkString("[", ",", "]")

  def queryId: String = queryKey

  def shuffleDepths: Map[Int, Int] = currentShuffleDepths

  def shuffleDepth(shuffleId: Int): Int = currentShuffleDepths.getOrElse(shuffleId, 0)

  private def groupGeneration(
      group: UcxPipelinedShuffleGroupMetadata): Seq[(Int, Int, Option[Int], Seq[Int])] = {
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

  private def computeShuffleDepths(): Map[Int, Int] = {
    val stagesByShuffleId =
      metadataByGroup.values().asScala
        .flatMap(_.stages)
        .flatMap(stage => stage.shuffleId.map(_ -> stage))
        .toMap
    val memo = scala.collection.mutable.HashMap[Int, Int]()

    def depthOf(shuffleId: Int, visiting: Set[Int]): Int = {
      memo.getOrElseUpdate(
        shuffleId, {
          if (visiting.contains(shuffleId)) {
            0
          } else {
            stagesByShuffleId.get(shuffleId) match {
              case Some(stage) =>
                val parentDepths =
                  stage.pipelinedParentShuffleIds
                    .filter(stagesByShuffleId.contains)
                    .map(parent => depthOf(parent, visiting + shuffleId))
                if (parentDepths.isEmpty) {
                  0
                } else {
                  parentDepths.max + 1
                }
              case None =>
                0
            }
          }
        })
    }

    shuffleIds.map(shuffleId => shuffleId -> depthOf(shuffleId, Set.empty)).toMap
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
          s"querySummary=${state.querySummary} " +
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
          s"queryId=${credit.queryId} queryState=${credit.queryState} " +
          s"shuffleActive=${credit.activeShuffleWriters}/${credit.maxShuffleWriters} " +
          s"queryActive=${credit.activeGroupWriters}/${credit.maxGroupWriters} " +
          s"queryQueuedBytes=${credit.queryQueuedBytes} " +
          s"queryBlockedWriters=${credit.queryBlockedWriters} " +
          s"queryBackpressured=${credit.queryBackpressured}"
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

  def pipelinedProducerLaunchCap(groupId: String, configuredCap: Int): Int = configuredCap

  def abortShuffle(shuffleId: Int, reason: String): Boolean

  def registerPipelinedShuffleGroup(group: UcxPipelinedShuffleGroupMetadata): Unit

  def admitPipelinedShuffleGroup(groupId: String): Boolean

  def completePipelinedShuffleGroup(groupId: String): Boolean

  def abortPipelinedShuffleGroup(groupId: String, reason: String): Boolean

  def completePipelinedQuery(queryExecutionId: Long): Boolean

  def abortPipelinedQuery(queryExecutionId: Long, reason: String): Boolean

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
  private val pipelinedQueries = new ConcurrentHashMap[String, UcxPipelinedQueryState]()
  private val abortedPipelinedQueries = new ConcurrentHashMap[String, String]()
  private val groupIdToPipelinedQueryKey = new ConcurrentHashMap[String, String]()
  private val shuffleIdToPipelinedGroupId = new ConcurrentHashMap[Int, String]()
  private val shuffleGenerationById =
    new ConcurrentHashMap[Int, UcxPipelinedShuffleGeneration]()
  private val writerCreditsByShuffle =
    new ConcurrentHashMap[Int, ConcurrentHashMap[Long, java.lang.Long]]()
  private val writerCreditLock = new Object()
  private val pipelinedRegistrationLock = new Object()

  private val maxActiveWriterTasksPerShuffle =
    math.max(0, conf.getInt(UcxColumnarShuffleManager.WriterMaxActiveTasksPerShuffleConf, 0))
  private val maxActiveWriterTasksPerQuery =
    math.max(
      0,
      conf.getInt(
        UcxColumnarShuffleManager.WriterMaxActiveTasksPerQueryConf,
        conf.getInt(UcxColumnarShuffleManager.WriterMaxActiveTasksPerGroupConf, 0)))
  private val minFrontierWriterTasksPerQuery =
    math.max(
      0,
      conf.getInt(UcxColumnarShuffleManager.WriterMinFrontierTasksPerQueryConf, 0))
  private val maxQueuedBytesPerQuery =
    math.max(0L, conf.getLong(UcxColumnarShuffleManager.QueryMaxQueuedBytesConf, 0L))
  private val resumeQueuedBytesPerQuery =
    math.max(
      0L,
      conf.getLong(
        UcxColumnarShuffleManager.QueryResumeQueuedBytesConf,
        if (maxQueuedBytesPerQuery > 0) maxQueuedBytesPerQuery / 2 else 0L))
  private val backpressuredLaunchWriters =
    math.max(
      0,
      conf.getInt(UcxColumnarShuffleManager.QueryBackpressuredLaunchWritersConf, 0))

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
    queryForShuffle(shuffleId).foreach {
      query =>
        val key = writerCreditKey(shuffleId, mapId)
        Option(query.nativeWriterStates.get(key)).foreach {
          previous =>
            if (previous.attemptId != endpoint.attemptId) {
              query.nativeWriterStates.remove(key, previous)
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
    queryForShuffle(shuffleId).foreach {
      query =>
        val key = readerStateKey(shuffleId, reducePartitionId)
        Option(query.nativeReaderStates.get(key)).foreach {
          previous =>
            if (previous.taskAttemptId != endpoint.taskAttemptId) {
              query.nativeReaderStates.remove(key, previous)
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
        val coverage = readerCoverageForShuffles(Seq(shuffleId))
        val readersReady = coverage.get(shuffleId).exists {
          current =>
            current.expectedReaders > 0 &&
              current.registeredReaders >= current.expectedReaders
        }
        UcxPipelinedShuffleStateResponse(
          shuffleId = shuffleId,
          groupId = None,
          groupState =
            if (readersReady) UcxPipelinedShuffleGroupState.ReadersReady
            else UcxPipelinedShuffleGroupState.NoGroup,
          readersReady = shuffleAbort.isEmpty && readersReady,
          readerCoverage = coverage,
          abortedReason = shuffleAbort,
          querySummary = None
        )
      case Some(id) =>
        val query = queryForGroup(id)
        val groupState =
          query.map(_.groupState(id)).getOrElse {
            Option(abortedPipelinedQueries.get(id))
              .map(_ => UcxPipelinedShuffleGroupState.Aborted)
              .getOrElse(UcxPipelinedShuffleGroupState.Registered)
          }
        val shuffleIds =
          query
            .flatMap(_.groupMetadata(id))
            .map(pipelinedShuffleIds)
            .getOrElse(shuffleIdsForGroup(id))
        val groupAbort =
          shuffleIds.iterator.flatMap(s => Option(abortedShuffles.get(s))).toSeq.headOption
        val abortedReason =
          shuffleAbort.orElse(query.flatMap(_.abortedReason)).orElse(
            Option(abortedPipelinedQueries.get(id))).orElse(groupAbort)
        UcxPipelinedShuffleStateResponse(
          shuffleId = shuffleId,
          groupId = Some(id),
          groupState = groupState,
          readersReady =
            abortedReason.isEmpty && readersReadyState(groupState),
          readerCoverage = readerCoverageForShuffles(shuffleIds),
          abortedReason = abortedReason,
          querySummary = query.map(queryRuntimeSummary)
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
    val query = queryForShuffle(shuffleId)
    val queryCreditEnabled = query.isDefined && maxActiveWriterTasksPerQuery > 0
    val shuffleCreditEnabled = query.isEmpty && maxActiveWriterTasksPerShuffle > 0
    if (!queryCreditEnabled && !shuffleCreditEnabled) {
      return writerCreditResponse(
        granted = true,
        shuffleId = shuffleId,
        groupId = groupId,
        query = query,
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
        query = query,
        reason = inactiveReason)
    }

    writerCreditLock.synchronized {
      query match {
        case Some(queryState) =>
          val queryCreditKey = writerCreditKey(shuffleId, mapId)
          val existing =
            Option(queryState.activeWriterCredits.get(queryCreditKey)).map(_.longValue())
          if (existing.contains(attemptId)) {
            return writerCreditResponse(
              granted = true,
              shuffleId = shuffleId,
              groupId = groupId,
              query = query,
              reason = None)
          }

          val decision = queryWriterCreditDecision(queryState, shuffleId)
          if (decision.granted) {
            queryState.activeWriterCredits.put(queryCreditKey, attemptId)
            val postFrontierDepth = queryFrontierDepth(queryState)
            val postAdmissionActive = activeAdmissionWriterCount(queryState, postFrontierDepth)
            logInfo(
              s"Granted UCX query writer credit queryId=${queryState.queryId} " +
                s"groupId=${queryState.groupId} shuffleId=$shuffleId mapId=$mapId " +
                s"attemptId=$attemptId " +
                s"queryActive=$postAdmissionActive/$maxActiveWriterTasksPerQuery " +
                s"queryTotalActive=${queryState.activeWriterCredits.size()} " +
                s"shuffleActive=${decision.activeShuffleWriters}/" +
                s"${decision.shuffleFairShare} " +
                s"activeFrontier=${decision.activeFrontierWriters}/" +
                s"${decision.frontierReserve} " +
                s"writerDepth=${decision.writerDepth} " +
                s"frontierDepth=${decision.frontierDepth} " +
                s"frontierReserve=${decision.frontierReserve} " +
                s"downstreamDrainBypassesQueryCap=" +
                s"${decision.downstreamDrainBypassesQueryCap} " +
                s"frontierProgressBypassesQueryCap=" +
                s"${decision.frontierProgressBypassesQueryCap} " +
                s"holdingFrontierForDrain=${decision.holdingFrontierForDrain} " +
                s"underfilledShuffles=${decision.underfilledShuffles} " +
                s"belowFairShuffles=${decision.belowFairShuffles} " +
                s"queuedBytes=${decision.queuedBytes} " +
                s"blockedWriters=${decision.blockedWriters} " +
                s"backpressured=${decision.backpressured}")
            writerCreditResponse(
              granted = true,
              shuffleId = shuffleId,
              groupId = groupId,
              query = query,
              reason = None)
          } else {
            if (decision.shouldLogBlocked) {
              logInfo(
                s"Deferring UCX query writer credit queryId=${queryState.queryId} " +
                  s"groupId=${queryState.groupId} shuffleId=$shuffleId mapId=$mapId " +
                  s"attemptId=$attemptId " +
                  s"queryActive=${decision.activeWriters}/$maxActiveWriterTasksPerQuery " +
                  s"queryTotalActive=${decision.totalActiveWriters} " +
                  s"shuffleActive=${decision.activeShuffleWriters}/" +
                  s"${decision.shuffleFairShare} " +
                  s"activeFrontier=${decision.activeFrontierWriters}/" +
                  s"${decision.frontierReserve} " +
                  s"writerDepth=${decision.writerDepth} " +
                  s"frontierDepth=${decision.frontierDepth} " +
                  s"frontierReserve=${decision.frontierReserve} " +
                  s"downstreamDrainBypassesQueryCap=" +
                  s"${decision.downstreamDrainBypassesQueryCap} " +
                  s"frontierProgressBypassesQueryCap=" +
                  s"${decision.frontierProgressBypassesQueryCap} " +
                  s"holdingFrontierForDrain=${decision.holdingFrontierForDrain} " +
                  s"underfilledShuffles=${decision.underfilledShuffles} " +
                  s"belowFairShuffles=${decision.belowFairShuffles} " +
                  s"queuedBytes=${decision.queuedBytes} " +
                  s"blockedWriters=${decision.blockedWriters} " +
                  s"backpressured=${decision.backpressured}")
            }
            writerCreditResponse(
              granted = false,
              shuffleId = shuffleId,
              groupId = groupId,
              query = query,
              reason = None)
          }

        case None =>
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
              query = query,
              reason = None)
          }

          val shuffleAllowed =
            maxActiveWriterTasksPerShuffle <= 0 ||
              shuffleCredits.size() < maxActiveWriterTasksPerShuffle
          if (shuffleAllowed) {
            shuffleCredits.put(mapId, attemptId)
            logInfo(
              s"Granted UCX shuffle writer credit shuffleId=$shuffleId mapId=$mapId " +
                s"attemptId=$attemptId " +
                s"shuffleActive=${shuffleCredits.size()}/$maxActiveWriterTasksPerShuffle")
            writerCreditResponse(
              granted = true,
              shuffleId = shuffleId,
              groupId = groupId,
              query = query,
              reason = None)
          } else {
            writerCreditResponse(
              granted = false,
              shuffleId = shuffleId,
              groupId = groupId,
              query = query,
              reason = None)
          }
      }
    }
  }

  override def releaseWriterCredit(shuffleId: Int, mapId: Long, attemptId: Long): Boolean = {
    releaseWriterCreditInternal(shuffleId, mapId, attemptId)
    true
  }

  override def reportNativeWriterState(
      state: UcxNativeWriterState): UcxNativeWriterStateResponse = {
    val groupId = groupIdForShuffle(state.shuffleId)
    val query = queryForShuffle(state.shuffleId)
    val inactiveReason =
      writerCreditInactiveReason(state.shuffleId).orElse(
        writerAttemptInactiveReason(state.shuffleId, state.mapId, state.attemptId))
    if (inactiveReason.isEmpty) {
      query.foreach {
        queryState =>
          queryState.nativeWriterStates.put(writerCreditKey(state.shuffleId, state.mapId), state)
      }
    }
    val writerDone = state.noMoreData || state.finished
    val writerFinishedMarked =
      writerDone && inactiveReason.isEmpty &&
        markWriterFinished(state.shuffleId, state.mapId, state.attemptId)
    val writerCreditReleased =
      writerDone && releaseWriterCreditInternal(state.shuffleId, state.mapId, state.attemptId)
    query.foreach {
      queryState =>
        writerCreditLock.synchronized {
          updateQueryBackpressure(queryState, queryQueuedBytes(queryState))
        }
        maybeAdvanceQueryState(queryState)
    }
    UcxNativeWriterStateResponse(
      accepted = inactiveReason.isEmpty,
      shuffleId = state.shuffleId,
      groupId = groupId,
      queryId = query.map(_.queryId),
      queryState = query.map(_.state).getOrElse(UcxPipelinedShuffleGroupState.NoGroup),
      writerFinishedMarked = writerFinishedMarked,
      writerCreditReleased = writerCreditReleased,
      queryQueuedBytes = query.map(queryQueuedBytes).getOrElse(0L),
      queryBlockedWriters = query.map(queryBlockedWriters).getOrElse(0),
      reason = inactiveReason)
  }

  override def pipelinedProducerLaunchCap(groupId: String, configuredCap: Int): Int = {
    val effectiveCap = math.max(0, configuredCap)
    writerCreditLock.synchronized {
      queryForGroup(groupId).map {
        query =>
          val queuedBytes = queryQueuedBytes(query)
          val backpressured = updateQueryBackpressure(query, queuedBytes)
          val activeAdmissionWriters =
            activeAdmissionWriterCount(query, queryFrontierDepth(query))
          val backpressuredCap = math.min(effectiveCap, backpressuredLaunchWriters)
          val throttleLaunch = backpressured && backpressuredCap < effectiveCap
          if (query.producerLaunchPaused != throttleLaunch) {
            query.producerLaunchPaused = throttleLaunch
            if (throttleLaunch) {
              logInfo(
                s"Throttled UCX pipelined pure-producer launch queryId=${query.queryId} " +
                  s"groupId=$groupId queuedBytes=$queuedBytes " +
                  s"activeAdmissionWriters=$activeAdmissionWriters " +
                  s"backpressuredCap=$backpressuredCap configuredCap=$effectiveCap")
            } else {
              logInfo(
                s"Resumed UCX pipelined pure-producer launch queryId=${query.queryId} " +
                  s"groupId=$groupId queuedBytes=$queuedBytes " +
                  s"activeAdmissionWriters=$activeAdmissionWriters " +
                  s"configuredCap=$effectiveCap")
            }
          }
          if (backpressured) backpressuredCap else effectiveCap
      }.getOrElse(effectiveCap)
    }
  }

  override def reportNativeReaderState(
      state: UcxNativeReaderState): UcxNativeReaderStateResponse = {
    val groupId = groupIdForShuffle(state.shuffleId)
    val query = queryForShuffle(state.shuffleId)
    val inactiveReason =
      readerStateInactiveReason(state.shuffleId).orElse(
        readerAttemptInactiveReason(
          state.shuffleId,
          state.reducePartitionId,
          state.taskAttemptId,
          state.nativeReaderId))
    if (inactiveReason.isEmpty) {
      query.foreach {
        queryState =>
          queryState.nativeReaderStates.put(
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
      query.foreach(maybeAdvanceQueryState)
    }
    UcxNativeReaderStateResponse(
      accepted = inactiveReason.isEmpty,
      shuffleId = state.shuffleId,
      groupId = groupId,
      queryId = query.map(_.queryId),
      queryState = query.map(_.state).getOrElse(UcxPipelinedShuffleGroupState.NoGroup),
      reason = inactiveReason)
  }

  private def releaseWriterCreditInternal(shuffleId: Int, mapId: Long, attemptId: Long): Boolean = {
    writerCreditLock.synchronized {
      val removedFromShuffle =
        Option(writerCreditsByShuffle.get(shuffleId)).exists {
          credits => removeCreditIfSameAttempt(credits, mapId, attemptId)
        }
      val queryCreditKey = writerCreditKey(shuffleId, mapId)
      var removedFromQuery = false
      queryForShuffle(shuffleId).foreach {
        query =>
          removedFromQuery =
            removeCreditIfSameAttempt(query.activeWriterCredits, queryCreditKey, attemptId)
      }
      if (removedFromShuffle || removedFromQuery) {
        logInfo(
          s"Released UCX shuffle writer credit shuffleId=$shuffleId " +
            s"mapId=$mapId attemptId=$attemptId")
      }
      removedFromShuffle || removedFromQuery
    }
  }

  override def abortShuffle(shuffleId: Int, reason: String): Boolean = {
    val groupId = shuffleIdToPipelinedGroupId.get(shuffleId)
    if (groupId != null) {
      return abortPipelinedShuffleGroup(groupId, s"shuffleId=$shuffleId aborted: $reason")
    }
    abortShuffleInternal(shuffleId, reason)
  }

  override def registerPipelinedShuffleGroup(group: UcxPipelinedShuffleGroupMetadata): Unit = {
    pipelinedRegistrationLock.synchronized {
      val shuffleIds = pipelinedShuffleIds(group)
      val queryKey = pipelinedQueryKey(group)
      val created = new UcxPipelinedQueryState(queryKey, group.queryExecutionId)
      val existing = pipelinedQueries.putIfAbsent(queryKey, created)
      val query = Option(existing).getOrElse(created)

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
                    s"queryId=${query.queryId} groupId=${group.groupId}")
              }
          }
          shuffleGenerationById.put(shuffleId, generation)
      }

      query.registerGroup(group, shuffleIds)
      groupIdToPipelinedQueryKey.put(group.groupId, queryKey)
      shuffleIds.foreach(shuffleIdToPipelinedGroupId.put(_, group.groupId))
      Option(abortedPipelinedQueries.get(group.groupId)).foreach {
        reason =>
          query.state = UcxPipelinedShuffleGroupState.Aborted
          query.abortedReason = Some(reason)
      }
      logInfo(
        s"Registered UCX pipelined query queryId=${query.queryId} groupId=${group.groupId} " +
          s"jobId=${group.jobId} queryExecutionId=${group.queryExecutionId} " +
          s"stages=${group.stages.map(_.stageId)} shuffles=$shuffleIds " +
          s"queryGroups=${query.groupIds.toSeq.sorted} " +
          s"shuffleDepths=${query.shuffleDepths.toSeq.sortBy(_._1)} " +
          s"state=${query.state}")
      if (query.state == UcxPipelinedShuffleGroupState.Aborted) {
        abortPipelinedShuffleGroup(group.groupId, "query was already aborted before registration")
      }
    }
  }

  override def admitPipelinedShuffleGroup(groupId: String): Boolean = {
    val query = queryForGroup(groupId).getOrElse {
      val queryState = new UcxPipelinedQueryState(groupId, None)
      queryState.registerGroup(
        UcxPipelinedShuffleGroupMetadata(groupId, -1, None, Seq.empty),
        shuffleIdsForGroup(groupId))
      pipelinedQueries.putIfAbsent(groupId, queryState)
      groupIdToPipelinedQueryKey.putIfAbsent(groupId, groupId)
      Option(pipelinedQueries.get(groupId)).getOrElse(queryState)
    }
    if (query.state == UcxPipelinedShuffleGroupState.Aborted) {
      return false
    }
    query.admitGroup(groupId)
    logInfo(s"Admitted UCX pipelined query queryId=${query.queryId} groupId=$groupId")
    query
      .groupMetadata(groupId)
      .map(pipelinedShuffleIds)
      .getOrElse(shuffleIdsForGroup(groupId))
      .foreach(maybeMarkReadersReady)
    true
  }

  override def completePipelinedShuffleGroup(groupId: String): Boolean = {
    queryForGroup(groupId) match {
      case Some(query) if query.queryExecutionId.isDefined =>
        query.completeGroup(groupId)
        query
          .groupMetadata(groupId)
          .map(pipelinedShuffleIds)
          .getOrElse(shuffleIdsForGroup(groupId))
          .filterNot(query.activeShuffleIds.contains)
          .foreach(clearWriterCreditsForShuffle)
        maybeMarkQueryDraining(query)
        logInfo(
          s"Completed UCX pipelined group queryId=${query.queryId} groupId=$groupId; " +
            s"retaining query control state until SQL execution completion")
      case Some(query) =>
        query.completeGroup(groupId)
        cleanupPipelinedQuery(query)
      case None =>
        cleanupLegacyPipelinedGroup(groupId)
    }
    true
  }

  override def abortPipelinedShuffleGroup(groupId: String, reason: String): Boolean = {
    abortedPipelinedQueries.put(groupId, reason)
    val query = queryForGroup(groupId)
    val shuffleIds = query.map(_.shuffleIds.toSeq).getOrElse(shuffleIdsForGroup(groupId))
    query.foreach { queryState =>
      queryState.state = UcxPipelinedShuffleGroupState.Aborted
      queryState.abortedReason = Some(reason)
      queryState.abortAllGroups()
      queryState.groupIds.foreach(abortedPipelinedQueries.put(_, reason))
    }
    shuffleIds.foreach {
      shuffleId =>
        abortShuffleInternal(
          shuffleId,
          s"pipelined query ${query.map(_.queryId).getOrElse(groupId)} aborted: $reason")
    }
    logWarning(
      s"Aborted UCX pipelined query queryId=${query.map(_.queryId)} groupId=$groupId " +
        s"shuffles=$shuffleIds reason=$reason")
    true
  }

  override def completePipelinedQuery(queryExecutionId: Long): Boolean = {
    val queryKey = pipelinedQueryKey(queryExecutionId)
    Option(pipelinedQueries.get(queryKey)).foreach {
      query =>
        if (query.state != UcxPipelinedShuffleGroupState.Aborted) {
          query.state = UcxPipelinedShuffleGroupState.Finished
          logInfo(
            s"Completed UCX pipelined SQL query queryId=${query.queryId} " +
              s"groups=${query.groupIds.toSeq.sorted} ${queryRuntimeSummary(query)}")
          cleanupPipelinedQuery(query)
        }
    }
    true
  }

  override def abortPipelinedQuery(queryExecutionId: Long, reason: String): Boolean = {
    val queryKey = pipelinedQueryKey(queryExecutionId)
    Option(pipelinedQueries.get(queryKey)).foreach {
      query =>
        query.groupIds.headOption match {
          case Some(groupId) => abortPipelinedShuffleGroup(groupId, reason)
          case None =>
            query.state = UcxPipelinedShuffleGroupState.Aborted
            query.abortedReason = Some(reason)
        }
    }
    true
  }

  private def cleanupLegacyPipelinedGroup(groupId: String): Unit = {
    val shuffleIds = shuffleIdsForGroup(groupId)
    shuffleIds.foreach(clearPipelinedShuffleState)
    abortedPipelinedQueries.remove(groupId)
    groupIdToPipelinedQueryKey.remove(groupId)
    logInfo(s"Completed legacy UCX pipelined group groupId=$groupId shuffles=$shuffleIds")
  }

  private def cleanupPipelinedQuery(query: UcxPipelinedQueryState): Unit = {
    val shuffleIds = query.shuffleIds.toSeq
    query.activeWriterCredits.clear()
    query.nativeWriterStates.clear()
    query.nativeReaderStates.clear()
    shuffleIds.foreach(clearPipelinedShuffleState)
    query.groupIds.foreach {
      groupId =>
        abortedPipelinedQueries.remove(groupId)
        groupIdToPipelinedQueryKey.remove(groupId, query.queryKey)
    }
    pipelinedQueries.remove(query.queryKey, query)
    logInfo(
      s"Cleaned UCX pipelined query queryId=${query.queryId} " +
        s"groups=${query.groupIds.toSeq.sorted} shuffles=$shuffleIds")
  }

  private def clearPipelinedShuffleState(shuffleId: Int): Unit = {
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
    shuffleInfos.remove(shuffleId)
    endpoints.remove(shuffleId)
    readerEndpoints.remove(shuffleId)
    finishedAttempts.remove(shuffleId)
    abortedShuffles.remove(shuffleId)
    clearWriterCreditsForShuffle(shuffleId)
    clearNativeStatesForShuffle(shuffleId)
    val groupId = shuffleIdToPipelinedGroupId.remove(shuffleId)
    shuffleGenerationById.remove(shuffleId)
    if (groupId != null && !shuffleIdToPipelinedGroupId.containsValue(groupId)) {
      queryForGroup(groupId).foreach {
        query =>
          if (
            query.state == UcxPipelinedShuffleGroupState.Aborted ||
            query.state == UcxPipelinedShuffleGroupState.Finished
          ) {
            cleanupPipelinedQuery(query)
          }
      }
    }
    logInfo(s"Unregistered UCX shuffle shuffleId=$shuffleId")
  }

  private def isPipelinedGroupAborted(shuffleId: Int): Boolean = {
    queryForShuffle(shuffleId).exists(_.state == UcxPipelinedShuffleGroupState.Aborted) ||
    groupIdForShuffle(shuffleId).exists(abortedPipelinedQueries.containsKey)
  }

  private def pipelinedShuffleIds(group: UcxPipelinedShuffleGroupMetadata): Seq[Int] = {
    group.stages.flatMap(_.shuffleId).distinct
  }

  private def writerGenerations(
      group: UcxPipelinedShuffleGroupMetadata): Seq[(Int, UcxPipelinedShuffleGeneration)] = {
    group.stages.flatMap {
      stage =>
        stage.shuffleId.map {
          shuffleId =>
            shuffleId -> UcxPipelinedShuffleGeneration(stage.stageId, stage.attemptId)
        }
    }
  }

  private def pipelinedQueryKey(group: UcxPipelinedShuffleGroupMetadata): String = {
    group.queryExecutionId.map(pipelinedQueryKey).getOrElse(group.groupId)
  }

  private def pipelinedQueryKey(queryExecutionId: Long): String = {
    s"queryExecution-$queryExecutionId"
  }

  private def shuffleIdsForGroup(groupId: String): Seq[Int] = {
    shuffleIdToPipelinedGroupId.asScala.collect {
      case (shuffleId, mappedGroupId) if mappedGroupId == groupId => shuffleId
    }.toSeq
  }

  private def maybeMarkReadersReady(shuffleId: Int): Unit = {
    val query = queryForShuffle(shuffleId).orNull
    val groupId = groupIdForShuffle(shuffleId).orNull
    if (
      query == null || groupId == null ||
      query.groupState(groupId) != UcxPipelinedShuffleGroupState.Admitted
    ) {
      return
    }
    val shuffleIds =
      query.groupMetadata(groupId).map(pipelinedShuffleIds).getOrElse(shuffleIdsForGroup(groupId))
    val allReadersReady = shuffleIds.nonEmpty && shuffleIds.forall {
      id =>
        val info = shuffleInfos.get(id)
        val readers = readerEndpoints.get(id)
        info != null && readers != null && readers.size >= info.numReduces
    }
    if (allReadersReady) {
      query.markGroupReadersReady(groupId)
      logInfo(
        s"UCX pipelined query queryId=${query.queryId} groupId=$groupId " +
          s"reached READERS_READY " +
          s"readerCoverage=${readerCoverageSummary(shuffleIds)}")
    }
  }

  private def readersReadyState(state: String): Boolean = {
    state == UcxPipelinedShuffleGroupState.ReadersReady ||
    state == UcxPipelinedShuffleGroupState.Draining ||
    state == UcxPipelinedShuffleGroupState.Finished
  }

  private def maybeMarkQueryDraining(query: UcxPipelinedQueryState): Unit = {
    if (
      query.state == UcxPipelinedShuffleGroupState.Aborted ||
      query.state == UcxPipelinedShuffleGroupState.Finished
    ) {
      return
    }
    val activeShuffleIds = query.activeShuffleIds
    val allWritersFinished = query.shuffleIds.nonEmpty &&
      (activeShuffleIds.isEmpty || activeShuffleIds.forall {
      shuffleId => shuffleFullyFinished(shuffleId)
    })
    if (allWritersFinished && query.state != UcxPipelinedShuffleGroupState.Draining) {
      query.state = UcxPipelinedShuffleGroupState.Draining
      logInfo(
        s"UCX pipelined query queryId=${query.queryId} groupId=${query.groupId} " +
          s"entered DRAINING activeCredits=${query.activeWriterCredits.size()} " +
          s"nativeWriters=${query.nativeWriterStates.size()} " +
          s"shuffles=${query.shuffleIds.toSeq.sorted}")
    }
  }

  private def maybeAdvanceQueryState(query: UcxPipelinedQueryState): Unit = {
    if (
      query.state == UcxPipelinedShuffleGroupState.Aborted ||
      query.state == UcxPipelinedShuffleGroupState.Finished
    ) {
      return
    }
    maybeMarkQueryDraining(query)
  }

  private def queryRuntimeSummary(
      query: UcxPipelinedQueryState): UcxPipelinedQueryRuntimeSummary = {
    UcxPipelinedQueryRuntimeSummary(
      queryId = query.queryId,
      activeWriterCredits = query.activeWriterCredits.size(),
      nativeWriterStates = query.nativeWriterStates.size(),
      finishedWriters = finishedWriterCount(query),
      expectedWriters = expectedWriterCount(query),
      queuedBytes = queryQueuedBytes(query),
      blockedWriters = queryBlockedWriters(query),
      registeredReaders = registeredReaderCount(query),
      finishedReaders = finishedReaderCount(query),
      expectedReaders = expectedReaderCount(query),
      backpressured = query.backpressured)
  }

  private def expectedWriterCount(query: UcxPipelinedQueryState): Int = {
    query.shuffleIds.toSeq
      .flatMap(shuffleId => Option(shuffleInfos.get(shuffleId)))
      .map(_.numMaps)
      .sum
  }

  private def finishedWriterCount(query: UcxPipelinedQueryState): Int = {
    query.shuffleIds.toSeq.map {
      shuffleId => Option(finishedAttempts.get(shuffleId)).map(_.size()).getOrElse(0)
    }.sum
  }

  private def expectedReaderCount(query: UcxPipelinedQueryState): Int = {
    query.shuffleIds.toSeq
      .flatMap(shuffleId => Option(shuffleInfos.get(shuffleId)))
      .map(_.numReduces)
      .sum
  }

  private def registeredReaderCount(query: UcxPipelinedQueryState): Int = {
    query.shuffleIds.toSeq.map {
      shuffleId => Option(readerEndpoints.get(shuffleId)).map(_.size()).getOrElse(0)
    }.sum
  }

  private def finishedReaderCount(query: UcxPipelinedQueryState): Int = {
    query.nativeReaderStates.asScala.values.count(_.finished)
  }

  private def queryQueuedBytes(query: UcxPipelinedQueryState): Long = {
    val activeShuffleIds = query.activeShuffleIds
    query.nativeWriterStates.asScala.values
      .filter(state => activeShuffleIds.contains(state.shuffleId))
      .map(state => math.max(0L, state.queuedBytes))
      .sum
  }

  private def queryBlockedWriters(query: UcxPipelinedQueryState): Int = {
    val activeShuffleIds = query.activeShuffleIds
    query.nativeWriterStates.asScala.values.count {
      state => activeShuffleIds.contains(state.shuffleId) && state.blocked
    }
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

  private case class QueryWriterCreditDecision(
      granted: Boolean,
      activeWriters: Int,
      totalActiveWriters: Int,
      activeShuffleWriters: Int,
      activeFrontierWriters: Int,
      writerDepth: Int,
      frontierDepth: Option[Int],
      frontierReserve: Int,
      shuffleFairShare: Int,
      underfilledShuffles: Seq[Int],
      belowFairShuffles: Seq[Int],
      downstreamDrainBypassesQueryCap: Boolean,
      frontierProgressBypassesQueryCap: Boolean,
      holdingFrontierForDrain: Boolean,
      queuedBytes: Long,
      blockedWriters: Int,
      backpressured: Boolean,
      shouldLogBlocked: Boolean = false)

  private def queryWriterCreditDecision(
      query: UcxPipelinedQueryState,
      shuffleId: Int): QueryWriterCreditDecision = {
    val writerDepth = query.shuffleDepth(shuffleId)
    val frontierDepth = queryFrontierDepth(query)
    val totalActiveWriters = query.activeWriterCredits.size()
    val activeWriters = activeAdmissionWriterCount(query, frontierDepth)
    val frontierReserve =
      frontierDepth.map(depth => queryFrontierWriterReserve(query, depth)).getOrElse(0)
    val unfinishedShuffles =
      query.activeShuffleIds.toSeq.filterNot(shuffleFullyFinished).sorted
    val admissionControlledShuffles =
      frontierDepth
        .map(depth => unfinishedShuffles.filter(id => query.shuffleDepth(id) == depth))
        .getOrElse(unfinishedShuffles)
    val activeByShuffle = activeWriterCountsByShuffle(query)
    val activeShuffleWriters = activeByShuffle.getOrElse(shuffleId, 0)
    val activeFrontierWriters =
      frontierDepth
        .map {
          depth =>
            activeByShuffle.iterator
              .filter { case (activeShuffleId, _) =>
                query.shuffleDepth(activeShuffleId) == depth
              }
              .map(_._2)
              .sum
        }
        .getOrElse(0)
    val shuffleFairShare =
      if (maxActiveWriterTasksPerQuery > 0 && admissionControlledShuffles.nonEmpty) {
        math.max(1, maxActiveWriterTasksPerQuery / admissionControlledShuffles.size)
      } else {
        0
      }
    val underfilledShuffles =
      if (maxActiveWriterTasksPerQuery >= admissionControlledShuffles.size) {
        admissionControlledShuffles.filter(id => activeByShuffle.getOrElse(id, 0) == 0)
      } else {
        Seq.empty[Int]
      }
    val belowFairShuffles =
      if (shuffleFairShare > 0) {
        admissionControlledShuffles.filter(
          id => activeByShuffle.getOrElse(id, 0) < shuffleFairShare)
      } else {
        Seq.empty[Int]
      }
    val queuedBytes = queryQueuedBytes(query)
    val blockedWriters = queryBlockedWriters(query)
    val backpressured = updateQueryBackpressure(query, queuedBytes)

    // Start the full downstream drain chain before releasing bulk producers. A writer below the
    // frontier is also a reader of an upstream exchange. Blocking any deeper level can leave most
    // exchange partitions undrained and turn streaming execution into a stage fence.
    val downstreamDrain =
      frontierDepth.exists(depth => writerDepth > depth)
    val hasUnfinishedDirectDownstream =
      frontierDepth.exists { depth =>
        unfinishedShuffles.exists(id => query.shuffleDepth(id) == depth + 1)
      }
    val downstreamDrainBypassesQueryCap = downstreamDrain
    val frontierProgressBypassesQueryCap =
      frontierDepth.contains(writerDepth) &&
        activeFrontierWriters < frontierReserve
    val holdingFrontierForDrain =
      backpressured &&
        frontierDepth.contains(writerDepth) &&
        hasUnfinishedDirectDownstream &&
        activeFrontierWriters >= frontierReserve
    val waitingForOtherUnderfilledShuffle =
      !downstreamDrain &&
        underfilledShuffles.exists(_ != shuffleId) &&
        !underfilledShuffles.contains(shuffleId)
    val waitingForOtherBelowFairShuffle =
      !downstreamDrain &&
        shuffleFairShare > 0 &&
        activeShuffleWriters >= shuffleFairShare &&
        belowFairShuffles.exists(_ != shuffleId)
    val depthAllowed =
      if (backpressured) {
        frontierDepth.forall(depth => writerDepth >= depth)
      } else {
        true
      }
    val granted =
      if (maxActiveWriterTasksPerQuery <= 0) {
        true
      } else if (
        activeWriters >= maxActiveWriterTasksPerQuery &&
        !downstreamDrainBypassesQueryCap &&
        !frontierProgressBypassesQueryCap
      ) {
        false
      } else if (holdingFrontierForDrain) {
        false
      } else if (!depthAllowed) {
        false
      } else if (waitingForOtherUnderfilledShuffle) {
        false
      } else if (waitingForOtherBelowFairShuffle) {
        false
      } else {
        true
      }

    QueryWriterCreditDecision(
      granted = granted,
      activeWriters = activeWriters,
      totalActiveWriters = totalActiveWriters,
      activeShuffleWriters = activeShuffleWriters,
      activeFrontierWriters = activeFrontierWriters,
      writerDepth = writerDepth,
      frontierDepth = frontierDepth,
      frontierReserve = frontierReserve,
      shuffleFairShare = shuffleFairShare,
      underfilledShuffles = underfilledShuffles,
      belowFairShuffles = belowFairShuffles,
      downstreamDrainBypassesQueryCap = downstreamDrainBypassesQueryCap,
      frontierProgressBypassesQueryCap = frontierProgressBypassesQueryCap,
      holdingFrontierForDrain = holdingFrontierForDrain,
      queuedBytes = queuedBytes,
      blockedWriters = blockedWriters,
      backpressured = backpressured)
  }

  private def updateQueryBackpressure(query: UcxPipelinedQueryState, queuedBytes: Long): Boolean = {
    if (maxQueuedBytesPerQuery <= 0) {
      query.backpressured = false
      return false
    }
    if (query.backpressured) {
      if (queuedBytes <= resumeQueuedBytesPerQuery) {
        query.backpressured = false
        logInfo(
          s"UCX pipelined query queryId=${query.queryId} groupId=${query.groupId} " +
            s"left backpressure queuedBytes=$queuedBytes " +
            s"resumeQueuedBytes=$resumeQueuedBytesPerQuery")
      }
    } else if (queuedBytes >= maxQueuedBytesPerQuery) {
      query.backpressured = true
      logInfo(
        s"UCX pipelined query queryId=${query.queryId} groupId=${query.groupId} " +
          s"entered backpressure queuedBytes=$queuedBytes " +
          s"maxQueuedBytes=$maxQueuedBytesPerQuery")
    }
    query.backpressured
  }

  private def queryFrontierDepth(query: UcxPipelinedQueryState): Option[Int] = {
    val unfinishedDepths =
      query.activeShuffleIds.toSeq
        .filterNot(shuffleFullyFinished)
        .map(query.shuffleDepth)
    if (unfinishedDepths.isEmpty) {
      None
    } else {
      Some(unfinishedDepths.min)
    }
  }

  private def queryFrontierWriterReserve(
      query: UcxPipelinedQueryState,
      frontierDepth: Int): Int = {
    if (maxActiveWriterTasksPerQuery <= 0) {
      return 0
    }
    // Writer credit is currently acquired after Spark launches the task. Shrinking the frontier
    // below the query cap can therefore consume every executor slot with credit waiters while
    // downstream tasks wait for those producers. Keep the full admission window until the
    // coordinator can pause native producers without occupying Spark task slots. Explicit
    // configuration remains available for scheduler-integrated experiments.
    val computedReserve = maxActiveWriterTasksPerQuery
    val reserve =
      if (minFrontierWriterTasksPerQuery > 0) {
        minFrontierWriterTasksPerQuery
      } else {
        computedReserve
      }
    math.min(maxActiveWriterTasksPerQuery, reserve)
  }

  private def activeWriterCountsByShuffle(query: UcxPipelinedQueryState): Map[Int, Int] = {
    query.activeWriterCredits
      .keySet()
      .asScala
      .iterator
      .flatMap {
        key =>
          val delimiter = key.indexOf(':')
          if (delimiter <= 0) {
            None
          } else {
            try {
              Some(key.substring(0, delimiter).toInt)
            } catch {
              case _: NumberFormatException => None
            }
          }
      }
      .toSeq
      .groupBy(identity)
      .view
      .mapValues(_.size)
      .toMap
  }

  private def activeAdmissionWriterCount(
      query: UcxPipelinedQueryState,
      frontierDepth: Option[Int]): Int = {
    frontierDepth match {
      case Some(frontier) =>
        activeWriterCountsByShuffle(query).iterator
          .filterNot { case (shuffleId, _) =>
            isDownstreamDrainShuffle(query, shuffleId, frontier)
          }
          .map(_._2)
          .sum
      case None =>
        query.activeWriterCredits.size()
    }
  }

  private def isDownstreamDrainShuffle(
      query: UcxPipelinedQueryState,
      shuffleId: Int,
      frontierDepth: Int): Boolean = {
    query.shuffleDepth(shuffleId) > frontierDepth &&
      query.activeShuffleIds.exists { candidate =>
        query.shuffleDepth(candidate) == frontierDepth && !shuffleFullyFinished(candidate)
      }
  }

  private def shuffleFullyFinished(shuffleId: Int): Boolean = {
    val info = shuffleInfos.get(shuffleId)
    val finished = finishedAttempts.get(shuffleId)
    info != null && finished != null && finished.size >= info.numMaps
  }

  private def writerStageInactiveReason(
      shuffleId: Int,
      stageId: Int,
      stageAttemptNumber: Int): Option[String] = {
    queryForShuffle(shuffleId).flatMap {
      query =>
        query.writerStage(shuffleId) match {
          case Some(stage)
              if stage.stageId == stageId && stage.attemptId == stageAttemptNumber =>
            None
          case Some(stage) =>
            Some(
              s"writer stage=$stageId.$stageAttemptNumber does not match current " +
                s"stage=${stage.stageId}.${stage.attemptId} for groupId=${query.groupId}")
          case None =>
            Some(
              s"shuffleId=$shuffleId has no writer stage in current groupId=${query.groupId}")
        }
    }
  }

  private def readerStageInactiveReason(
      shuffleId: Int,
      stageId: Int,
      stageAttemptNumber: Int): Option[String] = {
    queryForShuffle(shuffleId).flatMap {
      query =>
        query.readerStage(stageId) match {
          case Some(stage)
              if stage.attemptId == stageAttemptNumber &&
                stage.pipelinedParentShuffleIds.contains(shuffleId) =>
            None
          case Some(stage) if stage.attemptId != stageAttemptNumber =>
            Some(
              s"reader stage=$stageId.$stageAttemptNumber does not match current " +
                s"stage=${stage.stageId}.${stage.attemptId} for groupId=${query.groupId}")
          case Some(_) =>
            Some(
              s"reader stage=$stageId.$stageAttemptNumber does not consume shuffleId=$shuffleId " +
                s"in current groupId=${query.groupId}")
          case None =>
            Some(
              s"reader stage=$stageId.$stageAttemptNumber is not in current " +
                s"groupId=${query.groupId}")
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
        queryForShuffle(shuffleId)
          .flatMap {
            query =>
              query.abortedReason.orElse {
                query.state match {
                  case UcxPipelinedShuffleGroupState.Aborted =>
                    Some(s"pipelined query ${query.groupId} is aborted")
                  case UcxPipelinedShuffleGroupState.Finished =>
                    Some(s"pipelined query ${query.groupId} is finished")
                  case _ =>
                    None
                }
              }
          }
          .orElse {
            groupIdForShuffle(shuffleId).flatMap {
              id => Option(abortedPipelinedQueries.get(id))
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
        queryForShuffle(shuffleId)
          .flatMap {
            query =>
              query.abortedReason.orElse {
                if (query.state == UcxPipelinedShuffleGroupState.Aborted) {
                  Some(s"pipelined query ${query.groupId} is aborted")
                } else {
                  None
                }
              }
          }
          .orElse {
            groupIdForShuffle(shuffleId).flatMap {
              id => Option(abortedPipelinedQueries.get(id))
            }
          }
      }
    }
  }

  private def writerCreditResponse(
      granted: Boolean,
      shuffleId: Int,
      groupId: Option[String],
      query: Option[UcxPipelinedQueryState],
      reason: Option[String]): UcxShuffleWriterCreditResponse = {
    val shuffleActive =
      query
        .map(activeWriterCountsByShuffle(_).getOrElse(shuffleId, 0))
        .getOrElse(Option(writerCreditsByShuffle.get(shuffleId)).map(_.size()).getOrElse(0))
    val queryActive =
      query
        .map { queryState =>
          activeAdmissionWriterCount(queryState, queryFrontierDepth(queryState))
        }
        .getOrElse(0)
    UcxShuffleWriterCreditResponse(
      granted = granted,
      shuffleId = shuffleId,
      groupId = groupId,
      activeShuffleWriters = shuffleActive,
      maxShuffleWriters = maxActiveWriterTasksPerShuffle,
      activeGroupWriters = queryActive,
      maxGroupWriters = maxActiveWriterTasksPerQuery,
      reason = reason,
      queryId = query.map(_.queryId),
      queryState = query.map(_.state),
      queryQueuedBytes = query.map(queryQueuedBytes).getOrElse(0L),
      queryBlockedWriters = query.map(queryBlockedWriters).getOrElse(0),
      queryBackpressured = query.exists(_.backpressured)
    )
  }

  private def groupIdForShuffle(shuffleId: Int): Option[String] = {
    Option(shuffleIdToPipelinedGroupId.get(shuffleId))
  }

  private def queryForGroup(groupId: String): Option[UcxPipelinedQueryState] = {
    Option(groupIdToPipelinedQueryKey.get(groupId))
      .flatMap(queryKey => Option(pipelinedQueries.get(queryKey)))
  }

  private def queryForShuffle(shuffleId: Int): Option[UcxPipelinedQueryState] = {
    groupIdForShuffle(shuffleId).flatMap(queryForGroup)
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
      val prefix = s"$shuffleId:"
      pipelinedQueries.asScala.values.foreach {
        query =>
          query.activeWriterCredits
            .keySet()
            .asScala
            .filter(_.startsWith(prefix))
            .foreach(key => query.activeWriterCredits.remove(key))
      }
    }
  }

  private def clearNativeStatesForShuffle(shuffleId: Int): Unit = {
    val prefix = s"$shuffleId:"
    pipelinedQueries.asScala.values.foreach {
      query =>
        query.nativeWriterStates
          .keySet()
          .asScala
          .filter(_.startsWith(prefix))
          .foreach(key => query.nativeWriterStates.remove(key))
        query.nativeReaderStates
          .keySet()
          .asScala
          .filter(_.startsWith(prefix))
          .foreach(key => query.nativeReaderStates.remove(key))
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
              case CompleteUcxPipelinedQueryMasterMessage(queryExecutionId, context) =>
                context.reply(completePipelinedQuery(queryExecutionId))
              case AbortUcxPipelinedQueryMasterMessage(queryExecutionId, reason, context) =>
                context.reply(abortPipelinedQuery(queryExecutionId, reason))
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

  override def registerPipelinedShuffleGroup(group: UcxPipelinedShuffleGroupMetadata): Unit = {
    sendCoordinator(RegisterUcxPipelinedShuffleGroup(group))
  }

  override def admitPipelinedShuffleGroup(groupId: String): Boolean = {
    askCoordinator[Boolean](AdmitUcxPipelinedShuffleGroup(groupId))
  }

  override def completePipelinedShuffleGroup(groupId: String): Boolean = {
    askCoordinator[Boolean](CompleteUcxPipelinedShuffleGroup(groupId))
  }

  override def abortPipelinedShuffleGroup(groupId: String, reason: String): Boolean = {
    askCoordinator[Boolean](AbortUcxPipelinedShuffleGroup(groupId, reason))
  }

  override def completePipelinedQuery(queryExecutionId: Long): Boolean = {
    askCoordinator[Boolean](CompleteUcxPipelinedQuery(queryExecutionId))
  }

  override def abortPipelinedQuery(queryExecutionId: Long, reason: String): Boolean = {
    askCoordinator[Boolean](AbortUcxPipelinedQuery(queryExecutionId, reason))
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

    case CompleteUcxPipelinedQuery(queryExecutionId) =>
      coordinator.post(CompleteUcxPipelinedQueryMasterMessage(queryExecutionId, context))

    case AbortUcxPipelinedQuery(queryExecutionId, reason) =>
      coordinator.post(AbortUcxPipelinedQueryMasterMessage(queryExecutionId, reason, context))

    case StopUcxShuffleCoordinator =>
      logInfo(log"UCX shuffle coordinator endpoint stopped.")
      stop()
      context.reply(true)
  }
}

private[spark] object UcxShuffleCoordinator extends Logging {
  val EndpointName: String = "UcxShuffleCoordinator"

  @volatile private var coordinator: Option[UcxShuffleCoordinator] = None

  def pipelinedProducerLaunchCap(groupId: String, configuredCap: Int): Int = {
    coordinator
      .map(_.pipelinedProducerLaunchCap(groupId, configuredCap))
      .getOrElse(math.max(0, configuredCap))
  }

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
