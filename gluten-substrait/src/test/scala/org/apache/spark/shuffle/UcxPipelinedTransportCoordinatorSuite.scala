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

import org.apache.spark.SparkConf

import org.scalatest.funsuite.AnyFunSuite

class UcxPipelinedTransportCoordinatorSuite extends AnyFunSuite {

  test("leave group producer admission to Spark") {
    val legacyQueryPolicyConf = new SparkConf(false)
      .set("spark.gluten.ucx.shuffle.query.maxActiveWriters", "1")
      .set("spark.gluten.ucx.shuffle.query.maxQueuedBytes", "1")
      .set("spark.gluten.ucx.shuffle.query.backpressuredLaunchWriters", "0")

    val metadata = group("group-a", 10, 1, 2)
    Seq(true, false).foreach {
      isDriver =>
        assert(
          new UcxColumnarShuffleManager(legacyQueryPolicyConf, isDriver)
            .requiresAllPipelinedShuffleReadersResident(metadata))
    }
  }

  test("isolate transport state between groups") {
    withCoordinator() {
      coordinator =>
        coordinator.registerShuffle(shuffleId = 10, numMaps = 2, numReduces = 2)
        coordinator.registerShuffle(shuffleId = 20, numMaps = 3, numReduces = 3)
        coordinator.registerPipelinedShuffleGroup(group("group-a", 10, 1, 2))
        coordinator.registerPipelinedShuffleGroup(group("group-b", 20, 3, 4))
        assert(coordinator.admitPipelinedShuffleGroup("group-a"))
        assert(coordinator.admitPipelinedShuffleGroup("group-b"))

        val groupA = coordinator.getPipelinedShuffleState(10)
        val groupB = coordinator.getPipelinedShuffleState(20)
        assert(groupA.groupId.contains("group-a"))
        assert(groupB.groupId.contains("group-b"))
        assert(groupA.readerCoverage.get(10).exists(_.expectedReaders == 2))
        assert(groupB.readerCoverage.get(20).exists(_.expectedReaders == 3))

        assert(coordinator.completePipelinedShuffleGroup("group-a"))
        assert(
          coordinator.getPipelinedShuffleState(10).groupState ==
            UcxPipelinedShuffleGroupState.Finished)
        assert(
          coordinator.getPipelinedShuffleState(20).groupState ==
            UcxPipelinedShuffleGroupState.Admitted)

        assert(coordinator.abortPipelinedShuffleGroup("group-b", "test failure"))
        assert(
          coordinator.getPipelinedShuffleState(10).groupState ==
            UcxPipelinedShuffleGroupState.Finished)
        val aborted = coordinator.getPipelinedShuffleState(20)
        assert(aborted.groupState == UcxPipelinedShuffleGroupState.Aborted)
        assert(aborted.abortedReason.exists(_.contains("test failure")))
        assert(coordinator.getAvailableWriters(20).isEmpty)
    }
  }

  test("key transport lifecycle by group attempt instead of logical group") {
    withCoordinator() {
      coordinator =>
        val metadata =
          group("logical-group", 60, 50, 51)
            .copy(groupAttemptId = "logical-group-attempt-0")
        coordinator.registerShuffle(shuffleId = 60, numMaps = 2, numReduces = 2)
        coordinator.registerPipelinedShuffleGroup(metadata)
        assert(coordinator.admitPipelinedShuffleGroup(metadata.groupAttemptId))

        val state = coordinator.getPipelinedShuffleState(60)
        assert(state.groupId.contains(metadata.groupAttemptId))
        assert(state.groupState == UcxPipelinedShuffleGroupState.Admitted)
        assert(coordinator.completePipelinedShuffleGroup(metadata.groupAttemptId))
    }
  }

  test("group completion defers runtime cleanup until detached native writers drain") {
    withCoordinator() {
      coordinator =>
        coordinator.registerShuffle(shuffleId = 30, numMaps = 1, numReduces = 1)
        coordinator.registerPipelinedShuffleGroup(group("group-c", 30, 5, 6))
        assert(coordinator.admitPipelinedShuffleGroup("group-c"))

        val writer = writerEndpoint(shuffleId = 30, attemptId = 100L, stageId = 5)
        val reader = readerEndpoint(shuffleId = 30, attemptId = 101L, stageId = 6)
        assert(coordinator.registerWriter(30, 0L, writer))
        assert(coordinator.registerReader(30, 0, reader))
        assert(coordinator.markWriterFinished(30, 0L, writer.attemptId))

        assert(coordinator.completePipelinedShuffleGroup("group-c"))
        val draining = coordinator.getPipelinedShuffleState(30)
        assert(draining.groupState == UcxPipelinedShuffleGroupState.Draining)
        assert(coordinator.getAvailableWriters(30).exists(_.endpoints.nonEmpty))

        val nativeFinishedBeforeNoMoreData = coordinator.reportNativeWriterState(
          UcxNativeWriterState(
            shuffleId = 30,
            mapId = 0L,
            attemptId = writer.attemptId,
            nativeTaskId = writer.nativeTaskId,
            noMoreData = false,
            queuedBytes = 0L,
            blocked = false,
            timestampMs = System.currentTimeMillis(),
            finished = true))
        assert(nativeFinishedBeforeNoMoreData.accepted)
        assert(
          coordinator.getPipelinedShuffleState(30).groupState ==
            UcxPipelinedShuffleGroupState.Draining)
        assert(coordinator.getAvailableWriters(30).exists(_.endpoints.nonEmpty))

        val finalState = coordinator.reportNativeWriterState(
          UcxNativeWriterState(
            shuffleId = 30,
            mapId = 0L,
            attemptId = writer.attemptId,
            nativeTaskId = writer.nativeTaskId,
            noMoreData = true,
            queuedBytes = 0L,
            blocked = false,
            timestampMs = System.currentTimeMillis(),
            finished = true))
        assert(finalState.accepted)
        assert(
          coordinator.getPipelinedShuffleState(30).groupState ==
            UcxPipelinedShuffleGroupState.Finished)
        assert(coordinator.getAvailableWriters(30).exists(_.endpoints.isEmpty))
        assert(coordinator.getReaderCoverage(30).exists(_.readers.isEmpty))
    }
  }

  test("shuffle unregistration waits for detached native writer drain") {
    withCoordinator() {
      coordinator =>
        coordinator.registerShuffle(shuffleId = 31, numMaps = 1, numReduces = 1)
        coordinator.registerPipelinedShuffleGroup(group("group-d", 31, 7, 8))
        assert(coordinator.admitPipelinedShuffleGroup("group-d"))

        val writer = writerEndpoint(shuffleId = 31, attemptId = 110L, stageId = 7)
        assert(coordinator.registerWriter(31, 0L, writer))
        assert(coordinator.completePipelinedShuffleGroup("group-d"))
        coordinator.unregisterShuffle(31)
        assert(coordinator.getAvailableWriters(31).exists(_.endpoints.nonEmpty))

        val finalState = coordinator.reportNativeWriterState(
          UcxNativeWriterState(
            shuffleId = 31,
            mapId = 0L,
            attemptId = writer.attemptId,
            nativeTaskId = writer.nativeTaskId,
            noMoreData = true,
            queuedBytes = 0L,
            blocked = false,
            timestampMs = System.currentTimeMillis(),
            finished = true))
        assert(finalState.accepted)
        assert(coordinator.getAvailableWriters(31).isEmpty)
      }
  }

  test("reset transport runtime when a later group reuses a shuffle") {
    withCoordinator() {
      coordinator =>
        coordinator.registerShuffle(shuffleId = 40, numMaps = 1, numReduces = 1)
        val firstGroup = group("group-first", 40, 10, 11)
        coordinator.registerPipelinedShuffleGroup(firstGroup)
        assert(coordinator.admitPipelinedShuffleGroup("group-first"))

        val firstWriter = writerEndpoint(shuffleId = 40, attemptId = 100L, stageId = 10)
        val firstReader = readerEndpoint(shuffleId = 40, attemptId = 101L, stageId = 11)
        assert(coordinator.registerWriter(40, 0L, firstWriter))
        assert(coordinator.registerReader(40, 0, firstReader))
        assert(coordinator.markWriterFinished(40, 0L, 100L))
        assert(coordinator.completePipelinedShuffleGroup("group-first"))

        val secondGroup = group("group-second", 40, 20, 21)
        coordinator.registerPipelinedShuffleGroup(secondGroup)
        assert(coordinator.getAvailableWriters(40).exists(_.endpoints.isEmpty))
        assert(coordinator.getReaderCoverage(40).exists(_.readers.isEmpty))
        assert(coordinator.admitPipelinedShuffleGroup("group-second"))

        assert(!coordinator.registerWriter(40, 0L, firstWriter))
        val secondWriter = writerEndpoint(shuffleId = 40, attemptId = 200L, stageId = 20)
        assert(coordinator.registerWriter(40, 0L, secondWriter))
        assert(
          coordinator.getAvailableWriters(40).exists {
            response =>
              response.endpoints.get(0L).exists(_.attemptId == 200L) &&
                response.finishedMapIds.isEmpty
          })
    }
  }

  test("retain only per-shuffle writer credit admission") {
    val conf = coordinatorConf()
      .set(UcxColumnarShuffleManager.WriterMaxActiveTasksPerShuffleConf, "1")
    withCoordinator(conf) {
      coordinator =>
        coordinator.registerShuffle(shuffleId = 50, numMaps = 2, numReduces = 1)
        coordinator.registerPipelinedShuffleGroup(group("group-credit", 50, 30, 31))
        assert(coordinator.admitPipelinedShuffleGroup("group-credit"))

        val first =
          coordinator.tryAcquireWriterCredit(50, 0L, 100L, 30, 0)
        val second =
          coordinator.tryAcquireWriterCredit(50, 1L, 101L, 30, 0)
        assert(first.granted)
        assert(!second.granted)
        assert(first.maxShuffleWriters == 1)
        assert(first.groupId.contains("group-credit"))

        assert(coordinator.releaseWriterCredit(50, 0L, 100L))
        assert(coordinator.tryAcquireWriterCredit(50, 1L, 101L, 30, 0).granted)
    }
  }

  test("abort fences the old group attempt and allows a new generation") {
    withCoordinator() {
      coordinator =>
        coordinator.registerShuffle(shuffleId = 70, numMaps = 1, numReduces = 1)
        coordinator.registerPipelinedShuffleGroup(group("group-old", 70, 40, 41))
        assert(coordinator.admitPipelinedShuffleGroup("group-old"))
        val oldWriter = writerEndpoint(shuffleId = 70, attemptId = 100L, stageId = 40)
        assert(coordinator.registerWriter(70, 0L, oldWriter))

        assert(coordinator.abortPipelinedShuffleGroup("group-old", "injected failure"))
        assert(!coordinator.registerWriter(70, 0L, oldWriter))

        coordinator.registerPipelinedShuffleGroup(group("group-new", 70, 50, 51))
        assert(coordinator.admitPipelinedShuffleGroup("group-new"))
        val newWriter = writerEndpoint(shuffleId = 70, attemptId = 200L, stageId = 50)
        assert(coordinator.registerWriter(70, 0L, newWriter))
    }
  }

  private def withCoordinator(
      conf: SparkConf = coordinatorConf())(
      body: UcxShuffleCoordinatorMaster => Unit): Unit = {
    val coordinator = new UcxShuffleCoordinatorMaster(conf)
    try {
      body(coordinator)
    } finally {
      coordinator.stop()
    }
  }

  private def coordinatorConf(): SparkConf = {
    new SparkConf(false)
      .set("spark.gluten.ucx.shuffle.coordinator.threads", "1")
  }

  private def writerEndpoint(
      shuffleId: Int,
      attemptId: Long,
      stageId: Int): UcxShuffleEndpoint = {
    UcxShuffleEndpoint(
      executorId = "executor-1",
      host = "localhost",
      ucxPort = 10000,
      shuffleId = shuffleId,
      mapId = 0L,
      attemptId = attemptId,
      stageId = stageId,
      stageAttemptNumber = 0,
      nativeTaskId = s"writer-$attemptId",
      deviceId = 0,
      epoch = attemptId)
  }

  private def readerEndpoint(
      shuffleId: Int,
      attemptId: Long,
      stageId: Int): UcxShuffleReaderEndpoint = {
    UcxShuffleReaderEndpoint(
      executorId = "executor-1",
      host = "localhost",
      shuffleId = shuffleId,
      reducePartitionId = 0,
      taskAttemptId = attemptId,
      stageId = stageId,
      stageAttemptNumber = 0,
      nativeReaderId = s"reader-$attemptId",
      epoch = attemptId)
  }

  private def group(
      groupId: String,
      shuffleId: Int,
      writerStageId: Int,
      readerStageId: Int): PipelinedShuffleGroupMetadata = {
    PipelinedShuffleGroupMetadata(
      groupId = groupId,
      groupAttemptId = groupId,
      stages = Seq(
        PipelinedShuffleStageMetadata(
          stageId = writerStageId,
          attemptId = 0,
          numTasks = 2,
          shuffleId = Some(shuffleId)),
        PipelinedShuffleStageMetadata(
          stageId = readerStageId,
          attemptId = 0,
          numTasks = 2,
          shuffleId = None,
          pipelinedParentShuffleIds = Seq(shuffleId))))
  }
}
