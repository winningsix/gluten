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

private[spark] case class UcxShuffleReaderCoverage(
    registeredReaders: Int,
    expectedReaders: Int)

private[spark] case class UcxPipelinedShuffleStateResponse(
    shuffleId: Int,
    groupId: Option[String],
    groupState: String,
    readersReady: Boolean,
    readerCoverage: Map[Int, UcxShuffleReaderCoverage],
    abortedReason: Option[String])

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
  val Aborted: String = "ABORTED"
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
      pollMs: Long): UcxPipelinedShuffleStateResponse = {
    val startNs = System.nanoTime()
    val effectivePollMs = math.max(1L, pollMs)
    var lastLoggedState: String = ""
    while (true) {
      val state = getPipelinedShuffleState(shuffleId)
      state.abortedReason.foreach { reason =>
        throw new SparkException(
          s"UCX pipelined shuffle $shuffleId was aborted while waiting for readers: $reason")
      }
      if (pipelinedShuffleReadersReady(shuffleId, state)) {
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
          s"groupCoverage=${state.readerCoverage}"
      if (stateForLog != lastLoggedState) {
        logInfo(
          s"Waiting for UCX pipelined shuffle readers shuffleId=$shuffleId $stateForLog")
        lastLoggedState = stateForLog
      }
      try {
        Thread.sleep(effectivePollMs)
      } catch {
        case e: InterruptedException =>
          Thread.currentThread().interrupt()
          throw new SparkException(
            s"Interrupted waiting for UCX pipelined shuffle readers shuffleId=$shuffleId", e)
      }
    }
    throw new IllegalStateException("unreachable")
  }

  private def pipelinedShuffleReadersReady(
      shuffleId: Int,
      state: UcxPipelinedShuffleStateResponse): Boolean = {
    state.groupId match {
      case None =>
        state.readersReady
      case Some(_) =>
        state.readerCoverage.get(shuffleId).exists {
          coverage =>
            coverage.expectedReaders > 0 &&
              coverage.registeredReaders >= coverage.expectedReaders
        }
    }
  }

  def markWriterFinished(shuffleId: Int, mapId: Long, attemptId: Long): Boolean

  def abortShuffle(shuffleId: Int, reason: String): Boolean

  def registerPipelinedShuffleGroup(group: PipelinedShuffleGroupMetadata): Unit

  def admitPipelinedShuffleGroup(groupId: String): Boolean

  def completePipelinedShuffleGroup(groupId: String): Boolean

  def abortPipelinedShuffleGroup(groupId: String, reason: String): Boolean

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
  private val pipelinedGroups = new ConcurrentHashMap[String, PipelinedShuffleGroupMetadata]()
  private val pipelinedGroupStates = new ConcurrentHashMap[String, String]()
  private val shuffleIdToPipelinedGroupId = new ConcurrentHashMap[Int, String]()

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
    val registered =
      endpoints.computeIfAbsent(shuffleId, _ => new ConcurrentHashMap[Long, UcxShuffleEndpoint]())
    registered.compute(
      mapId,
      (_, previous) => {
        if (previous == null || endpoint.epoch >= previous.epoch) endpoint else previous
      })
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
    val registered = readerEndpoints.computeIfAbsent(
      shuffleId,
      _ => new ConcurrentHashMap[Int, UcxShuffleReaderEndpoint]())
    registered.compute(
      reducePartitionId,
      (_, previous) => {
        if (previous == null || endpoint.epoch >= previous.epoch) endpoint else previous
      })
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
      Option(finishedAttempts.get(shuffleId)).map(_.keySet().asScala.map(_.toLong).toSet)
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
          abortedReason = shuffleAbort)
      case Some(id) =>
        val groupState =
          Option(pipelinedGroupStates.get(id)).getOrElse(UcxPipelinedShuffleGroupState.Registered)
        val shuffleIds = shuffleIdsForGroup(id)
        val groupAbort =
          shuffleIds.iterator.flatMap(s => Option(abortedShuffles.get(s))).toSeq.headOption
        val abortedReason = shuffleAbort.orElse(groupAbort)
        UcxPipelinedShuffleStateResponse(
          shuffleId = shuffleId,
          groupId = Some(id),
          groupState = groupState,
          readersReady =
            abortedReason.isEmpty && groupState == UcxPipelinedShuffleGroupState.ReadersReady,
          readerCoverage = readerCoverageForShuffles(shuffleIds),
          abortedReason = abortedReason)
    }
  }

  override def markWriterFinished(shuffleId: Int, mapId: Long, attemptId: Long): Boolean = {
    if (!shuffleInfos.containsKey(shuffleId)) {
      return false
    }
    val finished = finishedAttempts.computeIfAbsent(
      shuffleId,
      _ => new ConcurrentHashMap[Long, java.lang.Long]())
    finished.put(mapId, attemptId)
    true
  }

  override def abortShuffle(shuffleId: Int, reason: String): Boolean = {
    val groupId = shuffleIdToPipelinedGroupId.get(shuffleId)
    if (groupId != null) {
      return abortPipelinedShuffleGroup(groupId, s"shuffleId=$shuffleId aborted: $reason")
    }
    abortShuffleInternal(shuffleId, reason)
  }

  override def registerPipelinedShuffleGroup(group: PipelinedShuffleGroupMetadata): Unit = {
    pipelinedGroups.put(group.groupId, group)
    pipelinedGroupStates.putIfAbsent(group.groupId, UcxPipelinedShuffleGroupState.Registered)
    val shuffleIds = pipelinedShuffleIds(group)
    shuffleIds.foreach(shuffleIdToPipelinedGroupId.put(_, group.groupId))
    logInfo(
      s"Registered UCX pipelined shuffle group groupId=${group.groupId} " +
        s"jobId=${group.jobId} queryExecutionId=${group.queryExecutionId} " +
        s"stages=${group.stages.map(_.stageId)} shuffles=$shuffleIds " +
        s"state=${pipelinedGroupStates.get(group.groupId)}")
    if (pipelinedGroupStates.get(group.groupId) == UcxPipelinedShuffleGroupState.Aborted) {
      abortPipelinedShuffleGroup(group.groupId, "group was already aborted before registration")
    }
  }

  override def admitPipelinedShuffleGroup(groupId: String): Boolean = {
    if (pipelinedGroupStates.get(groupId) == UcxPipelinedShuffleGroupState.Aborted) {
      return false
    }
    pipelinedGroupStates.put(groupId, UcxPipelinedShuffleGroupState.Admitted)
    logInfo(s"Admitted UCX pipelined shuffle group groupId=$groupId")
    shuffleIdsForGroup(groupId).foreach(maybeMarkReadersReady)
    true
  }

  override def completePipelinedShuffleGroup(groupId: String): Boolean = {
    val group = pipelinedGroups.remove(groupId)
    val shuffleIds = if (group == null) {
      shuffleIdsForGroup(groupId)
    } else {
      pipelinedShuffleIds(group)
    }
    shuffleIds.foreach {
      shuffleId =>
        endpoints.remove(shuffleId)
        readerEndpoints.remove(shuffleId)
        finishedAttempts.remove(shuffleId)
        abortedShuffles.remove(shuffleId)
        shuffleIdToPipelinedGroupId.remove(shuffleId)
    }
    pipelinedGroupStates.remove(groupId)
    logInfo(s"Completed UCX pipelined shuffle group groupId=$groupId shuffles=$shuffleIds")
    true
  }

  override def abortPipelinedShuffleGroup(groupId: String, reason: String): Boolean = {
    pipelinedGroupStates.put(groupId, UcxPipelinedShuffleGroupState.Aborted)
    val group = pipelinedGroups.get(groupId)
    val shuffleIds = if (group == null) {
      shuffleIdsForGroup(groupId)
    } else {
      pipelinedShuffleIds(group)
    }
    shuffleIds.foreach {
      shuffleId => abortShuffleInternal(shuffleId, s"pipelined group $groupId aborted: $reason")
    }
    logWarning(
      s"Aborted UCX pipelined shuffle group groupId=$groupId " +
        s"shuffles=$shuffleIds reason=$reason")
    true
  }

  private def abortShuffleInternal(shuffleId: Int, reason: String): Boolean = {
    abortedShuffles.put(shuffleId, reason)
    endpoints.remove(shuffleId)
    readerEndpoints.remove(shuffleId)
    finishedAttempts.remove(shuffleId)
    logWarning(s"Aborted UCX shuffle shuffleId=$shuffleId reason=$reason")
    true
  }

  override def unregisterShuffle(shuffleId: Int): Unit = {
    shuffleInfos.remove(shuffleId)
    endpoints.remove(shuffleId)
    readerEndpoints.remove(shuffleId)
    finishedAttempts.remove(shuffleId)
    abortedShuffles.remove(shuffleId)
    val groupId = shuffleIdToPipelinedGroupId.remove(shuffleId)
    if (groupId != null && !shuffleIdToPipelinedGroupId.containsValue(groupId)) {
      pipelinedGroups.remove(groupId)
      pipelinedGroupStates.remove(groupId)
    }
    logInfo(s"Unregistered UCX shuffle shuffleId=$shuffleId")
  }

  private def isPipelinedGroupAborted(shuffleId: Int): Boolean = {
    val groupId = shuffleIdToPipelinedGroupId.get(shuffleId)
    groupId != null &&
    pipelinedGroupStates.get(groupId) == UcxPipelinedShuffleGroupState.Aborted
  }

  private def pipelinedShuffleIds(group: PipelinedShuffleGroupMetadata): Seq[Int] = {
    group.stages.flatMap(_.shuffleId).distinct
  }

  private def shuffleIdsForGroup(groupId: String): Seq[Int] = {
    shuffleIdToPipelinedGroupId.asScala.collect {
      case (shuffleId, mappedGroupId) if mappedGroupId == groupId => shuffleId
    }.toSeq
  }

  private def maybeMarkReadersReady(shuffleId: Int): Unit = {
    val groupId = shuffleIdToPipelinedGroupId.get(shuffleId)
    if (
      groupId == null ||
      pipelinedGroupStates.get(groupId) != UcxPipelinedShuffleGroupState.Admitted
    ) {
      return
    }
    val shuffleIds = shuffleIdsForGroup(groupId)
    val allReadersReady = shuffleIds.nonEmpty && shuffleIds.forall {
      id =>
        val info = shuffleInfos.get(id)
        val readers = readerEndpoints.get(id)
        info != null && readers != null && readers.size >= info.numReduces
    }
    if (allReadersReady) {
      pipelinedGroupStates.put(groupId, UcxPipelinedShuffleGroupState.ReadersReady)
      logInfo(
        s"UCX pipelined shuffle group groupId=$groupId reached READERS_READY " +
          s"readerCoverage=${readerCoverageSummary(shuffleIds)}")
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

  override def abortShuffle(shuffleId: Int, reason: String): Boolean = {
    askCoordinator[Boolean](AbortUcxShuffle(shuffleId, reason))
  }

  override def registerPipelinedShuffleGroup(group: PipelinedShuffleGroupMetadata): Unit = {
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
