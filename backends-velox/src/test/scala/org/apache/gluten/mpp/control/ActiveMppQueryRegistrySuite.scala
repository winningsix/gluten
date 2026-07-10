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
package org.apache.gluten.mpp.control

import org.apache.spark.TaskContext

import org.mockito.Mockito.{mock, when}
import org.scalatest.funsuite.AnyFunSuite

import java.io.{ByteArrayOutputStream, ObjectOutputStream}
import java.util.concurrent.{CountDownLatch, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger, AtomicReference}

class ActiveMppQueryRegistrySuite extends AnyFunSuite {
  private val run = MppQueryRunId("q", 3, 0)

  private def taskContext(
      attemptId: Long,
      interrupted: AtomicBoolean = new AtomicBoolean(false)): TaskContext = {
    val context = mock(classOf[TaskContext])
    when(context.taskAttemptId()).thenReturn(attemptId)
    when(context.isInterrupted()).thenAnswer(_ => interrupted.get())
    context
  }

  test("abort requested before native start is latched and native abort is skipped") {
    val interrupted = new AtomicBoolean(true)
    val nativeAbortCalls = new AtomicInteger(0)
    val agent = new GlutenMppExecutorControlAgent(
      "e0",
      "s0",
      _ => MppQueryHeartbeatAck(Nil, Nil, 0L),
      interruptPollMs = 50L,
      heartbeatMs = 250L,
      startThreads = false)
    try {
      val query = agent.register(
        run,
        peerIndex = 0,
        expectedPeerCount = 1,
        taskContext(10L, interrupted),
        () => nativeAbortCalls.incrementAndGet())

      agent.scanInterrupts()
      assert(!query.beginNativeStart())
      assert(query.isAbortRequested)
      assert(nativeAbortCalls.get() == 0)
      assert(agent.pendingFailureCount == 1)
    } finally {
      agent.shutdown()
    }
  }

  test("remote abort sequences are accepted but invoke native abort exactly once") {
    val nativeAbort = new CountDownLatch(1)
    val nativeAbortCalls = new AtomicInteger(0)
    val response = new AtomicReference[AnyRef]()
    val agent = new GlutenMppExecutorControlAgent(
      "e0",
      "s0",
      _ => response.get(),
      interruptPollMs = 50L,
      heartbeatMs = 250L,
      startThreads = false)
    try {
      val query = agent.register(
        run,
        peerIndex = 0,
        expectedPeerCount = 1,
        taskContext(10L),
        () => {
          nativeAbortCalls.incrementAndGet()
          nativeAbort.countDown()
        })
      assert(query.beginNativeStart())
      query.finishNativeStart(succeeded = true)

      def command(sequence: Long): MppAbortQuery =
        MppAbortQuery(run, 0, 10L, "e0", "s0", sequence, "peer failed", 0L)

      response.set(MppQueryHeartbeatAck(Seq(command(1L), command(2L)), Nil, 0L))
      agent.sendHeartbeat()
      assert(nativeAbort.await(5L, TimeUnit.SECONDS))
      assert(nativeAbortCalls.get() == 1)

      response.set(MppQueryHeartbeatAck(Nil, Nil, 0L))
      agent.sendHeartbeat()
      assert(agent.activeQueries.head.snapshot().acceptedAbortSequence == 2L)
    } finally {
      agent.shutdown()
    }
  }

  test("a blocked heartbeat cannot prevent local interruption from queueing abort") {
    val interrupted = new AtomicBoolean(false)
    val askEntered = new CountDownLatch(1)
    val releaseAsk = new CountDownLatch(1)
    val nativeAbort = new CountDownLatch(1)
    val agent = new GlutenMppExecutorControlAgent(
      "e0",
      "s0",
      _ => {
        askEntered.countDown()
        releaseAsk.await(5L, TimeUnit.SECONDS)
        MppQueryHeartbeatAck(Nil, Nil, 0L)
      },
      interruptPollMs = 50L,
      heartbeatMs = 250L,
      startThreads = false
    )
    try {
      val query = agent.register(
        run,
        peerIndex = 0,
        expectedPeerCount = 1,
        taskContext(10L, interrupted),
        () => nativeAbort.countDown())
      assert(query.beginNativeStart())
      query.finishNativeStart(succeeded = true)

      val heartbeatThread = new Thread(() => agent.sendHeartbeat())
      heartbeatThread.start()
      assert(askEntered.await(5L, TimeUnit.SECONDS))

      interrupted.set(true)
      agent.scanInterrupts()
      assert(nativeAbort.await(5L, TimeUnit.SECONDS))

      releaseAsk.countDown()
      heartbeatThread.join(5000L)
    } finally {
      releaseAsk.countDown()
      agent.shutdown()
    }
  }

  test("terminal event is retained through heartbeat ack and first terminal state wins") {
    val response = new AtomicReference[AnyRef](MppQueryHeartbeatAck(Nil, Nil, 0L))
    val agent = new GlutenMppExecutorControlAgent(
      "e0",
      "s0",
      _ => response.get(),
      interruptPollMs = 50L,
      heartbeatMs = 250L,
      startThreads = false)
    try {
      val query =
        agent.register(run, peerIndex = 0, expectedPeerCount = 1, taskContext(10L), () => ())
      assert(query.beginNativeStart())
      query.finishNativeStart(succeeded = true)

      agent.reportTerminal(query, MppPeerState.Failed)
      agent.reportTerminal(query, MppPeerState.Succeeded)
      assert(query.snapshot().state == MppPeerState.Failed)
      assert(agent.pendingTerminalCount == 1)

      val eventId = s"${run.logId}/0/10/terminal"
      response.set(MppQueryHeartbeatAck(Nil, Seq(eventId), 0L))
      agent.sendHeartbeat()
      assert(agent.pendingTerminalCount == 0)
      assert(agent.activeQueries.isEmpty)
    } finally {
      agent.shutdown()
    }
  }

  test("heartbeat serializes strict peer snapshots without retaining active queries") {
    val serialized = new AtomicBoolean(false)
    val agent = new GlutenMppExecutorControlAgent(
      "e0",
      "s0",
      message => {
        val bytes = new ByteArrayOutputStream()
        val output = new ObjectOutputStream(bytes)
        try {
          output.writeObject(message)
          output.flush()
          serialized.set(bytes.size() > 0)
        } finally {
          output.close()
        }
        MppQueryHeartbeatAck(Nil, Nil, 0L)
      },
      interruptPollMs = 50L,
      heartbeatMs = 250L,
      startThreads = false
    )
    try {
      val query =
        agent.register(run, peerIndex = 0, expectedPeerCount = 2, taskContext(10L), () => ())
      assert(query.beginNativeStart())
      query.finishNativeStart(succeeded = true)

      agent.sendHeartbeat()
      assert(serialized.get())
    } finally {
      agent.shutdown()
    }
  }

  test("output EOS waits for matching driver completion authorization") {
    val outputCompleteHeartbeat = new CountDownLatch(1)
    val response = new AtomicReference[AnyRef](MppQueryHeartbeatAck(Nil, Nil, 0L))
    val agent = new GlutenMppExecutorControlAgent(
      "e0",
      "s0",
      message => {
        message match {
          case heartbeat: MppQueryHeartbeat
              if heartbeat.peers.exists(_.state == MppPeerState.OutputComplete) =>
            outputCompleteHeartbeat.countDown()
          case _ =>
        }
        response.get()
      },
      interruptPollMs = 50L,
      heartbeatMs = 250L,
      startThreads = false
    )
    try {
      val query =
        agent.register(run, peerIndex = 0, expectedPeerCount = 2, taskContext(10L), () => ())
      assert(query.beginNativeStart())
      query.finishNativeStart(succeeded = true)

      val completed = new AtomicBoolean(false)
      val waiter = new Thread(() => completed.set(agent.awaitPeerCompletion(query)))
      waiter.start()
      assert(outputCompleteHeartbeat.await(5L, TimeUnit.SECONDS))
      assert(query.snapshot().state == MppPeerState.OutputComplete)
      assert(waiter.isAlive)

      val wrongAttempt = MppPeerCompletion(run, 0, 999L, "e0", "s0")
      response.set(MppQueryHeartbeatAck(Nil, Nil, 0L, Seq(wrongAttempt)))
      agent.sendHeartbeat()
      assert(waiter.isAlive)

      val matchingAttempt = MppPeerCompletion(run, 0, 10L, "e0", "s0")
      response.set(MppQueryHeartbeatAck(Nil, Nil, 0L, Seq(matchingAttempt)))
      agent.sendHeartbeat()
      waiter.join(5000L)
      assert(!waiter.isAlive)
      assert(completed.get())
    } finally {
      agent.shutdown()
    }
  }

  test("output EOS fast-polls completion without waiting for the periodic heartbeat") {
    val outputCompleteHeartbeats = new AtomicInteger(0)
    val completion = MppPeerCompletion(run, 0, 10L, "e0", "s0")
    val agent = new GlutenMppExecutorControlAgent(
      "e0",
      "s0",
      message => {
        val count = message match {
          case heartbeat: MppQueryHeartbeat
              if heartbeat.peers.exists(_.state == MppPeerState.OutputComplete) =>
            outputCompleteHeartbeats.incrementAndGet()
          case _ =>
            outputCompleteHeartbeats.get()
        }
        MppQueryHeartbeatAck(Nil, Nil, 0L, if (count >= 2) Seq(completion) else Nil)
      },
      interruptPollMs = 50L,
      heartbeatMs = 250L,
      startThreads = false
    )
    try {
      val query =
        agent.register(run, peerIndex = 0, expectedPeerCount = 2, taskContext(10L), () => ())
      assert(query.beginNativeStart())
      query.finishNativeStart(succeeded = true)

      val completed = new AtomicBoolean(false)
      val waiter = new Thread(() => completed.set(agent.awaitPeerCompletion(query)))
      waiter.start()
      waiter.join(5000L)
      assert(!waiter.isAlive)
      assert(completed.get())
      assert(outputCompleteHeartbeats.get() >= 2)
    } finally {
      agent.shutdown()
    }
  }

  test("peer abort releases output EOS wait without authorizing close") {
    val outputCompleteHeartbeat = new CountDownLatch(1)
    val response = new AtomicReference[AnyRef](MppQueryHeartbeatAck(Nil, Nil, 0L))
    val nativeAbort = new CountDownLatch(1)
    val agent = new GlutenMppExecutorControlAgent(
      "e0",
      "s0",
      message => {
        message match {
          case heartbeat: MppQueryHeartbeat
              if heartbeat.peers.exists(_.state == MppPeerState.OutputComplete) =>
            outputCompleteHeartbeat.countDown()
          case _ =>
        }
        response.get()
      },
      interruptPollMs = 50L,
      heartbeatMs = 250L,
      startThreads = false
    )
    try {
      val query = agent.register(
        run,
        peerIndex = 0,
        expectedPeerCount = 2,
        taskContext(10L),
        () => nativeAbort.countDown())
      assert(query.beginNativeStart())
      query.finishNativeStart(succeeded = true)

      val completed = new AtomicBoolean(true)
      val waiter = new Thread(() => completed.set(agent.awaitPeerCompletion(query)))
      waiter.start()
      assert(outputCompleteHeartbeat.await(5L, TimeUnit.SECONDS))

      val abort = MppAbortQuery(run, 0, 10L, "e0", "s0", 1L, "peer failed", 0L)
      response.set(MppQueryHeartbeatAck(Seq(abort), Nil, 0L))
      agent.sendHeartbeat()
      waiter.join(5000L)
      assert(!waiter.isAlive)
      assert(!completed.get())
      assert(nativeAbort.await(5L, TimeUnit.SECONDS))
    } finally {
      agent.shutdown()
    }
  }

  test("idle agent does not contact driver") {
    val asks = new AtomicInteger(0)
    val agent = new GlutenMppExecutorControlAgent(
      "e0",
      "s0",
      _ => {
        asks.incrementAndGet()
        MppQueryHeartbeatAck(Nil, Nil, 0L)
      },
      interruptPollMs = 50L,
      heartbeatMs = 250L,
      startThreads = false)
    try {
      agent.sendHeartbeat()
      assert(asks.get() == 0)
    } finally {
      agent.shutdown()
    }
  }
}
