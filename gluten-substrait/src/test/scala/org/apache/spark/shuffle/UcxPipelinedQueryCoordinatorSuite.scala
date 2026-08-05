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
import org.apache.spark.shuffle.streaming.StreamingShuffleManager

import org.scalatest.funsuite.AnyFunSuite

class UcxPipelinedQueryCoordinatorSuite extends AnyFunSuite {

  test("use the community streaming shuffle manager contract") {
    assert(
      new UcxColumnarShuffleManager(new SparkConf(false), isDriver = true)
        .isInstanceOf[StreamingShuffleManager])
  }

  test("discover full reader residency without Spark group callbacks") {
    val coordinator = new UcxShuffleCoordinatorMaster(
      new SparkConf(false)
        .set("spark.gluten.ucx.shuffle.coordinator.threads", "1"))

    try {
      coordinator.registerShuffle(shuffleId = 5, numMaps = 1, numReduces = 2)
      assert(!coordinator.getPipelinedShuffleState(5).readersReady)

      assert(coordinator.registerReader(5, 0, readerEndpoint(5, 100L, 10, 0)))
      assert(!coordinator.getPipelinedShuffleState(5).readersReady)

      assert(coordinator.registerReader(5, 1, readerEndpoint(5, 101L, 10, 1)))
      val ready = coordinator.getPipelinedShuffleState(5)
      assert(ready.readersReady)
      assert(ready.groupState == UcxPipelinedShuffleGroupState.ReadersReady)
      assert(
        ready.readerCoverage.get(5).contains(UcxShuffleReaderCoverage(2, 2)))
    } finally {
      coordinator.stop()
    }
  }

  test("coordinate all pipelined shuffle groups in one SQL execution") {
    val coordinator = new UcxShuffleCoordinatorMaster(
      new SparkConf(false)
        .set("spark.gluten.ucx.shuffle.coordinator.threads", "1"))

    try {
      coordinator.registerShuffle(shuffleId = 10, numMaps = 2, numReduces = 2)
      coordinator.registerShuffle(shuffleId = 20, numMaps = 3, numReduces = 3)
      coordinator.registerPipelinedShuffleGroup(group("group-a", 1, 42L, 10, 1, 2))
      coordinator.registerPipelinedShuffleGroup(group("group-b", 2, 42L, 20, 3, 4))
      assert(coordinator.admitPipelinedShuffleGroup("group-a"))
      assert(coordinator.admitPipelinedShuffleGroup("group-b"))

      val groupA = coordinator.getPipelinedShuffleState(10)
      val groupB = coordinator.getPipelinedShuffleState(20)
      assert(groupA.querySummary.map(_.queryId).contains("queryExecution-42"))
      assert(groupB.querySummary.map(_.queryId).contains("queryExecution-42"))
      assert(groupA.querySummary.map(_.expectedWriters).contains(5))
      assert(groupB.querySummary.map(_.expectedReaders).contains(5))

      assert(coordinator.completePipelinedShuffleGroup("group-a"))
      assert(
        coordinator.getPipelinedShuffleState(10).groupState ==
          UcxPipelinedShuffleGroupState.Finished)
      assert(
        coordinator.getPipelinedShuffleState(20).groupState ==
          UcxPipelinedShuffleGroupState.Admitted)
      assert(coordinator.getPipelinedShuffleState(10).querySummary.nonEmpty)

      assert(coordinator.abortPipelinedShuffleGroup("group-b", "test failure"))
      Seq(10, 20).foreach {
        shuffleId =>
          val state = coordinator.getPipelinedShuffleState(shuffleId)
          assert(state.groupState == UcxPipelinedShuffleGroupState.Aborted)
          assert(state.abortedReason.exists(_.contains("test failure")))
          assert(coordinator.getAvailableWriters(shuffleId).isEmpty)
      }
    } finally {
      coordinator.stop()
    }
  }

  test("retain exchange state until SQL execution completion") {
    val coordinator = new UcxShuffleCoordinatorMaster(
      new SparkConf(false)
        .set("spark.gluten.ucx.shuffle.coordinator.threads", "1"))

    try {
      coordinator.registerShuffle(shuffleId = 30, numMaps = 2, numReduces = 2)
      coordinator.registerPipelinedShuffleGroup(group("group-c", 3, 84L, 30, 5, 6))
      assert(coordinator.admitPipelinedShuffleGroup("group-c"))
      assert(coordinator.completePipelinedShuffleGroup("group-c"))
      assert(coordinator.getPipelinedShuffleState(30).querySummary.nonEmpty)

      assert(coordinator.completePipelinedQuery(84L))
      val cleaned = coordinator.getPipelinedShuffleState(30)
      assert(cleaned.groupId.isEmpty)
      assert(cleaned.groupState == UcxPipelinedShuffleGroupState.NoGroup)
      assert(cleaned.querySummary.isEmpty)
      assert(coordinator.getAvailableWriters(30).isEmpty)
    } finally {
      coordinator.stop()
    }
  }

  test("reset exchange runtime when a later job reuses a streaming shuffle") {
    val coordinator = new UcxShuffleCoordinatorMaster(
      new SparkConf(false)
        .set("spark.gluten.ucx.shuffle.coordinator.threads", "1"))

    try {
      coordinator.registerShuffle(shuffleId = 40, numMaps = 1, numReduces = 1)
      val firstGroup = group("group-first", 4, 126L, 40, 10, 11)
      coordinator.registerPipelinedShuffleGroup(firstGroup)
      assert(coordinator.admitPipelinedShuffleGroup("group-first"))

      val firstWriter = writerEndpoint(shuffleId = 40, attemptId = 100L, stageId = 10)
      val firstReader = readerEndpoint(shuffleId = 40, attemptId = 101L, stageId = 11)
      assert(coordinator.registerWriter(40, 0L, firstWriter))
      assert(coordinator.registerReader(40, 0, firstReader))
      assert(coordinator.markWriterFinished(40, 0L, 100L))
      assert(coordinator.getAvailableWriters(40).exists(_.finishedMapIds == Set(0L)))
      assert(coordinator.getReaderCoverage(40).exists(_.readers.keySet == Set(0)))
      assert(coordinator.completePipelinedShuffleGroup("group-first"))

      val secondGroup = group("group-second", 5, 126L, 40, 20, 21)
      coordinator.registerPipelinedShuffleGroup(secondGroup)
      val resetWriters = coordinator.getAvailableWriters(40).get
      assert(resetWriters.endpoints.isEmpty)
      assert(resetWriters.finishedMapIds.isEmpty)
      assert(coordinator.getReaderCoverage(40).exists(_.readers.isEmpty))

      assert(coordinator.admitPipelinedShuffleGroup("group-second"))
      coordinator.registerPipelinedShuffleGroup(secondGroup)
      assert(
        coordinator.getPipelinedShuffleState(40).groupState ==
          UcxPipelinedShuffleGroupState.Admitted)

      assert(!coordinator.registerWriter(40, 0L, firstWriter))
      val secondWriter = writerEndpoint(shuffleId = 40, attemptId = 200L, stageId = 20)
      assert(coordinator.registerWriter(40, 0L, secondWriter))
      assert(
        coordinator.getAvailableWriters(40).exists {
          response =>
            response.endpoints.get(0L).exists(_.attemptId == 200L) &&
              response.finishedMapIds.isEmpty
        })
    } finally {
      coordinator.stop()
    }
  }

  test("admit the complete downstream drain chain under query backpressure") {
    val coordinator = new UcxShuffleCoordinatorMaster(
      new SparkConf(false)
        .set("spark.gluten.ucx.shuffle.coordinator.threads", "1")
        .set("spark.gluten.ucx.shuffle.query.maxActiveWriters", "1")
        .set("spark.gluten.ucx.shuffle.query.maxQueuedBytes", "1")
        .set("spark.gluten.ucx.shuffle.query.resumeQueuedBytes", "0"))

    try {
      Seq(10, 20, 30).foreach {
        shuffleId => coordinator.registerShuffle(shuffleId, numMaps = 2, numReduces = 2)
      }
      coordinator.registerPipelinedShuffleGroup(chainedGroup())
      assert(coordinator.admitPipelinedShuffleGroup("group-chain"))

      assert(
        coordinator
          .tryAcquireWriterCredit(10, 0L, 100L, stageId = 10, stageAttemptNumber = 0)
          .granted)
      val frontierWriter = writerEndpoint(shuffleId = 10, attemptId = 100L, stageId = 10)
      assert(coordinator.registerWriter(10, 0L, frontierWriter))
      assert(
        coordinator
          .reportNativeWriterState(
            UcxNativeWriterState(
              shuffleId = 10,
              mapId = 0L,
              attemptId = 100L,
              nativeTaskId = frontierWriter.nativeTaskId,
              noMoreData = false,
              queuedBytes = 1024L,
              blocked = true,
              timestampMs = System.currentTimeMillis()))
          .accepted)

      val blockedFrontier =
        coordinator.tryAcquireWriterCredit(
          10,
          1L,
          101L,
          stageId = 10,
          stageAttemptNumber = 0)
      assert(!blockedFrontier.granted)
      assert(blockedFrontier.queryBackpressured)

      val directDrain =
        coordinator.tryAcquireWriterCredit(
          20,
          0L,
          200L,
          stageId = 20,
          stageAttemptNumber = 0)
      val deepDrain0 =
        coordinator.tryAcquireWriterCredit(
          30,
          0L,
          300L,
          stageId = 30,
          stageAttemptNumber = 0)
      val deepDrain1 =
        coordinator.tryAcquireWriterCredit(
          30,
          1L,
          301L,
          stageId = 30,
          stageAttemptNumber = 0)

      assert(directDrain.granted)
      assert(deepDrain0.granted)
      assert(deepDrain1.granted)
      assert(deepDrain1.queryBackpressured)
    } finally {
      coordinator.stop()
    }
  }

  test("admit the downstream drain chain before query backpressure is enabled") {
    val coordinator = new UcxShuffleCoordinatorMaster(
      new SparkConf(false)
        .set("spark.gluten.ucx.shuffle.coordinator.threads", "1")
        .set("spark.gluten.ucx.shuffle.query.maxActiveWriters", "1"))

    try {
      Seq(10, 20, 30).foreach {
        shuffleId => coordinator.registerShuffle(shuffleId, numMaps = 2, numReduces = 2)
      }
      coordinator.registerPipelinedShuffleGroup(chainedGroup())
      assert(coordinator.admitPipelinedShuffleGroup("group-chain"))

      assert(
        coordinator
          .tryAcquireWriterCredit(10, 0L, 100L, stageId = 10, stageAttemptNumber = 0)
          .granted)

      val directDrain =
        coordinator.tryAcquireWriterCredit(
          20,
          0L,
          200L,
          stageId = 20,
          stageAttemptNumber = 0)
      val deepDrain =
        coordinator.tryAcquireWriterCredit(
          30,
          0L,
          300L,
          stageId = 30,
          stageAttemptNumber = 0)

      assert(directDrain.granted)
      assert(deepDrain.granted)
      assert(!directDrain.queryBackpressured)
      assert(!deepDrain.queryBackpressured)
    } finally {
      coordinator.stop()
    }
  }

  test("do not shrink launched frontier below query writer cap under backpressure") {
    val coordinator = new UcxShuffleCoordinatorMaster(
      new SparkConf(false)
        .set("spark.gluten.ucx.shuffle.coordinator.threads", "1")
        .set("spark.gluten.ucx.shuffle.query.maxActiveWriters", "8")
        .set("spark.gluten.ucx.shuffle.query.maxQueuedBytes", "1")
        .set("spark.gluten.ucx.shuffle.query.resumeQueuedBytes", "0"))

    try {
      coordinator.registerShuffle(shuffleId = 10, numMaps = 9, numReduces = 2)
      coordinator.registerShuffle(shuffleId = 20, numMaps = 2, numReduces = 2)
      coordinator.registerPipelinedShuffleGroup(wideFrontierGroup())
      assert(coordinator.admitPipelinedShuffleGroup("group-wide-frontier"))

      assert(
        coordinator
          .tryAcquireWriterCredit(10, 0L, 100L, stageId = 10, stageAttemptNumber = 0)
          .granted)
      val frontierWriter = writerEndpoint(shuffleId = 10, attemptId = 100L, stageId = 10)
      assert(coordinator.registerWriter(10, 0L, frontierWriter))
      assert(
        coordinator
          .reportNativeWriterState(
            UcxNativeWriterState(
              shuffleId = 10,
              mapId = 0L,
              attemptId = 100L,
              nativeTaskId = frontierWriter.nativeTaskId,
              noMoreData = false,
              queuedBytes = 1024L,
              blocked = true,
              timestampMs = System.currentTimeMillis()))
          .accepted)

      (1L to 7L).foreach {
        mapId =>
          assert(
            coordinator
              .tryAcquireWriterCredit(
                10,
                mapId,
                100L + mapId,
                stageId = 10,
                stageAttemptNumber = 0)
              .granted)
      }
      val overWindow =
        coordinator.tryAcquireWriterCredit(
          10,
          8L,
          108L,
          stageId = 10,
          stageAttemptNumber = 0)
      assert(!overWindow.granted)
      assert(overWindow.queryBackpressured)

      assert(
        coordinator
          .tryAcquireWriterCredit(20, 0L, 200L, stageId = 20, stageAttemptNumber = 0)
          .granted)
    } finally {
      coordinator.stop()
    }
  }

  test("throttle pure-producer launch until queued bytes reach the resume watermark") {
    val coordinator = new UcxShuffleCoordinatorMaster(
      new SparkConf(false)
        .set("spark.gluten.ucx.shuffle.coordinator.threads", "1")
        .set("spark.gluten.ucx.shuffle.query.maxActiveWriters", "8")
        .set("spark.gluten.ucx.shuffle.query.maxQueuedBytes", "1")
        .set("spark.gluten.ucx.shuffle.query.resumeQueuedBytes", "0")
        .set("spark.gluten.ucx.shuffle.query.backpressuredLaunchWriters", "2"))

    try {
      coordinator.registerShuffle(shuffleId = 10, numMaps = 2, numReduces = 2)
      coordinator.registerPipelinedShuffleGroup(
        group("group-prelaunch-backpressure", 8, 252L, 10, 10, 11))
      assert(coordinator.admitPipelinedShuffleGroup("group-prelaunch-backpressure"))
      assert(coordinator.pipelinedProducerLaunchCap("unknown-group", 8) == 8)
      assert(coordinator.pipelinedProducerLaunchCap("group-prelaunch-backpressure", 8) == 8)

      assert(
        coordinator
          .tryAcquireWriterCredit(10, 0L, 100L, stageId = 10, stageAttemptNumber = 0)
          .granted)
      val frontierWriter = writerEndpoint(shuffleId = 10, attemptId = 100L, stageId = 10)
      assert(coordinator.registerWriter(10, 0L, frontierWriter))
      val noMoreData =
        coordinator.reportNativeWriterState(
          UcxNativeWriterState(
            shuffleId = 10,
            mapId = 0L,
            attemptId = 100L,
            nativeTaskId = frontierWriter.nativeTaskId,
            noMoreData = true,
            queuedBytes = 1024L,
            blocked = true,
            timestampMs = System.currentTimeMillis()))
      assert(noMoreData.accepted)
      assert(noMoreData.writerCreditReleased)

      // Native noMoreData releases the execution credit, but its detached queue is still live.
      // Keep only the bounded refill window until that queue reaches the resume watermark.
      assert(coordinator.pipelinedProducerLaunchCap("group-prelaunch-backpressure", 8) == 2)

      assert(
        coordinator
          .reportNativeWriterState(
            UcxNativeWriterState(
              shuffleId = 10,
              mapId = 0L,
              attemptId = 100L,
              nativeTaskId = frontierWriter.nativeTaskId,
              noMoreData = true,
              queuedBytes = 0L,
              blocked = false,
              timestampMs = System.currentTimeMillis(),
              finished = true,
              queuedPages = 0L))
          .accepted)
      assert(coordinator.pipelinedProducerLaunchCap("group-prelaunch-backpressure", 8) == 8)
    } finally {
      coordinator.stop()
    }
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
      stageId: Int,
      reducePartitionId: Int = 0): UcxShuffleReaderEndpoint = {
    UcxShuffleReaderEndpoint(
      executorId = "executor-1",
      host = "localhost",
      shuffleId = shuffleId,
      reducePartitionId = reducePartitionId,
      taskAttemptId = attemptId,
      stageId = stageId,
      stageAttemptNumber = 0,
      nativeReaderId = s"reader-$attemptId",
      epoch = attemptId)
  }

  private def group(
      groupId: String,
      jobId: Int,
      queryExecutionId: Long,
      shuffleId: Int,
      writerStageId: Int,
      readerStageId: Int): UcxPipelinedShuffleGroupMetadata = {
    UcxPipelinedShuffleGroupMetadata(
      groupId = groupId,
      jobId = jobId,
      queryExecutionId = Some(queryExecutionId),
      stages = Seq(
        UcxPipelinedShuffleStageMetadata(
          stageId = writerStageId,
          attemptId = 0,
          numTasks = 2,
          shuffleId = Some(shuffleId)),
        UcxPipelinedShuffleStageMetadata(
          stageId = readerStageId,
          attemptId = 0,
          numTasks = 2,
          shuffleId = None,
          pipelinedParentShuffleIds = Seq(shuffleId))))
  }

  private def chainedGroup(): UcxPipelinedShuffleGroupMetadata = {
    UcxPipelinedShuffleGroupMetadata(
      groupId = "group-chain",
      jobId = 6,
      queryExecutionId = Some(168L),
      stages = Seq(
        UcxPipelinedShuffleStageMetadata(
          stageId = 10,
          attemptId = 0,
          numTasks = 2,
          shuffleId = Some(10)),
        UcxPipelinedShuffleStageMetadata(
          stageId = 20,
          attemptId = 0,
          numTasks = 2,
          shuffleId = Some(20),
          pipelinedParentShuffleIds = Seq(10)),
        UcxPipelinedShuffleStageMetadata(
          stageId = 30,
          attemptId = 0,
          numTasks = 2,
          shuffleId = Some(30),
          pipelinedParentShuffleIds = Seq(20)),
        UcxPipelinedShuffleStageMetadata(
          stageId = 40,
          attemptId = 0,
          numTasks = 2,
          shuffleId = None,
          pipelinedParentShuffleIds = Seq(30))))
  }

  private def wideFrontierGroup(): UcxPipelinedShuffleGroupMetadata = {
    UcxPipelinedShuffleGroupMetadata(
      groupId = "group-wide-frontier",
      jobId = 7,
      queryExecutionId = Some(210L),
      stages = Seq(
        UcxPipelinedShuffleStageMetadata(
          stageId = 10,
          attemptId = 0,
          numTasks = 9,
          shuffleId = Some(10)),
        UcxPipelinedShuffleStageMetadata(
          stageId = 20,
          attemptId = 0,
          numTasks = 2,
          shuffleId = Some(20),
          pipelinedParentShuffleIds = Seq(10)),
        UcxPipelinedShuffleStageMetadata(
          stageId = 30,
          attemptId = 0,
          numTasks = 2,
          shuffleId = None,
          pipelinedParentShuffleIds = Seq(20))))
  }
}
