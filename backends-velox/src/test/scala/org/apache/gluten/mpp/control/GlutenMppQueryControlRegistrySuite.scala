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

import org.scalatest.funsuite.AnyFunSuite

class GlutenMppQueryControlRegistrySuite extends AnyFunSuite {
  private val run = MppQueryRunId("q", 7, 0)

  private def peer(
      index: Int,
      attempt: Long = -1L,
      executor: String = "",
      session: String = "",
      state: String = MppPeerState.Running,
      accepted: Long = 0L): MppPeerSnapshot = {
    val resolvedAttempt = if (attempt < 0L) index + 100L else attempt
    val resolvedExecutor = if (executor.isEmpty) s"e$index" else executor
    val resolvedSession = if (session.isEmpty) s"s$index" else session
    MppPeerSnapshot(
      run,
      index,
      2,
      resolvedAttempt,
      resolvedExecutor,
      resolvedSession,
      state,
      accepted)
  }

  private def heartbeat(
      snapshot: MppPeerSnapshot,
      failures: Seq[MppQueryFailure] = Nil,
      terminals: Seq[MppTerminalEvent] = Nil): MppQueryHeartbeat =
    MppQueryHeartbeat(
      snapshot.executorId,
      snapshot.executorSessionId,
      0L,
      Seq(snapshot),
      failures,
      terminals)

  private def failure(snapshot: MppPeerSnapshot, id: String, message: String): MppQueryFailure =
    MppQueryFailure(
      id,
      snapshot.runId,
      snapshot.peerIndex,
      snapshot.expectedPeerCount,
      snapshot.taskAttemptId,
      snapshot.executorId,
      snapshot.executorSessionId,
      "java.lang.RuntimeException",
      message,
      "stack",
      0L
    )

  test("run enters RUNNING only after every expected peer joins") {
    var now = 0L
    val registry = new GlutenMppQueryControlRegistry(5000L, clock = () => now)

    registry.processHeartbeat(heartbeat(peer(0)))
    assert(registry.snapshot(run).get.state == MppQueryRunState.Starting)

    registry.processHeartbeat(heartbeat(peer(1)))
    val snapshot = registry.snapshot(run).get
    assert(snapshot.state == MppQueryRunState.Running)
    assert(snapshot.peers.map(_.peerIndex) == Seq(0, 1))
  }

  test("first peer failure is preserved and abort commands replay until accepted") {
    var now = 100L
    val registry = new GlutenMppQueryControlRegistry(5000L, clock = () => now)
    val p0 = peer(0)
    val p1 = peer(1)
    registry.processHeartbeat(heartbeat(p0))
    registry.processHeartbeat(heartbeat(p1))

    val firstAck =
      registry.processHeartbeat(heartbeat(p0, failures = Seq(failure(p0, "failure-1", "CUDA OOM"))))
    assert(firstAck.abortCommands.map(_.peerIndex) == Seq(0))
    val peerOneAck = registry.processHeartbeat(heartbeat(p1))
    assert(peerOneAck.abortCommands.map(_.peerIndex) == Seq(1))
    val peerOneCommand = peerOneAck.abortCommands.head

    val replay = registry.processHeartbeat(heartbeat(p1))
    assert(replay.abortCommands.map(_.sequence) == Seq(peerOneCommand.sequence))

    val accepted =
      registry.processHeartbeat(heartbeat(p1.copy(acceptedAbortSequence = peerOneCommand.sequence)))
    assert(accepted.abortCommands.isEmpty)

    registry.processHeartbeat(heartbeat(p1, failures = Seq(failure(p1, "failure-2", "secondary"))))
    val snapshot = registry.snapshot(run).get
    assert(snapshot.state == MppQueryRunState.Aborting)
    assert(snapshot.firstFailure.exists(_.contains("CUDA OOM")))
  }

  test("duplicate task attempt aborts both the original and duplicate attempts") {
    val registry = new GlutenMppQueryControlRegistry(5000L)
    val original = peer(0, attempt = 100L, executor = "e0", session = "s0")
    val duplicate = peer(0, attempt = 200L, executor = "e1", session = "s1")
    registry.processHeartbeat(heartbeat(original))

    val duplicateAck = registry.processHeartbeat(heartbeat(duplicate))
    assert(duplicateAck.abortCommands.exists(_.taskAttemptId == 200L))

    val originalAck = registry.processHeartbeat(heartbeat(original))
    assert(originalAck.abortCommands.exists(_.taskAttemptId == 100L))
    assert(registry.snapshot(run).get.firstFailure.exists(_.contains("Duplicate attempt")))
  }

  test("peer lease starts only after RUNNING and aborts surviving peer") {
    var now = 0L
    val registry = new GlutenMppQueryControlRegistry(5000L, clock = () => now)
    val p0 = peer(0)
    val p1 = peer(1)
    registry.processHeartbeat(heartbeat(p0))

    now = 6000L
    registry.expirePeers()
    assert(registry.snapshot(run).get.state == MppQueryRunState.Starting)

    registry.processHeartbeat(heartbeat(p1))
    now = 9000L
    registry.processHeartbeat(heartbeat(p1))
    now = 11001L
    registry.expirePeers()

    val snapshot = registry.snapshot(run).get
    assert(snapshot.state == MppQueryRunState.Aborting)
    assert(snapshot.peers.find(_.peerIndex == 0).exists(_.terminal))
    assert(snapshot.peers.find(_.peerIndex == 1).exists(peer => !peer.terminal))
    assert(registry.processHeartbeat(heartbeat(p1)).abortCommands.nonEmpty)
  }

  test("driver authorizes close only after every peer reaches output EOS") {
    val registry = new GlutenMppQueryControlRegistry(5000L)
    val p0 = peer(0)
    val p1 = peer(1)
    registry.processHeartbeat(heartbeat(p0))
    registry.processHeartbeat(heartbeat(p1))

    val first = registry.processHeartbeat(heartbeat(p0.copy(state = MppPeerState.OutputComplete)))
    assert(first.peerCompletions.isEmpty)
    assert(registry.snapshot(run).exists(_.state == MppQueryRunState.Running))

    val last = registry.processHeartbeat(heartbeat(p1.copy(state = MppPeerState.OutputComplete)))
    assert(last.peerCompletions.map(_.peerIndex) == Seq(1))
    val completing = registry.snapshot(run).get
    assert(completing.state == MppQueryRunState.Completing)
    assert(completing.peers.forall(_.outputComplete))
    assert(completing.peers.forall(peer => !peer.terminal))

    val retry = registry.processHeartbeat(heartbeat(p0.copy(state = MppPeerState.OutputComplete)))
    assert(retry.peerCompletions.map(_.peerIndex) == Seq(0))
  }

  test("non-success peer terminal before output EOS aborts the remaining peers") {
    val registry = new GlutenMppQueryControlRegistry(5000L)
    val p0 = peer(0)
    val p1 = peer(1)
    registry.processHeartbeat(heartbeat(p0))
    registry.processHeartbeat(heartbeat(p1))

    registry.processHeartbeat(heartbeat(p0.copy(state = MppPeerState.Aborted)))
    val snapshot = registry.snapshot(run).get
    assert(snapshot.state == MppQueryRunState.Aborting)
    assert(snapshot.firstFailure.exists(_.contains("before output EOS")))
    assert(registry.processHeartbeat(heartbeat(p1)).abortCommands.map(_.peerIndex) == Seq(1))
  }

  test("successful Spark task end is an implicit output EOS") {
    val registry = new GlutenMppQueryControlRegistry(5000L)
    val p0 = peer(0)
    val p1 = peer(1)
    registry.processHeartbeat(heartbeat(p0))
    registry.processHeartbeat(heartbeat(p1))

    registry.onTaskEnd(p0.taskAttemptId, p0.executorId, failed = false, "success")
    val oneFinished = registry.snapshot(run).get
    assert(oneFinished.state == MppQueryRunState.Running)
    assert(oneFinished.firstFailure.isEmpty)
    assert(oneFinished.peers.find(_.peerIndex == 0).exists(_.outputComplete))

    registry.onTaskEnd(p1.taskAttemptId, p1.executorId, failed = false, "success")
    val finished = registry.snapshot(run).get
    assert(finished.state == MppQueryRunState.Terminal)
    assert(finished.firstFailure.isEmpty)
    assert(finished.peers.forall(peer => peer.outputComplete && peer.terminal))
  }

  test("terminal tombstone rejects late traffic and expires after retention") {
    var now = 0L
    val registry = new GlutenMppQueryControlRegistry(
      peerTimeoutMs = 5000L,
      terminalRetentionMs = 1000L,
      clock = () => now)
    val p0 = peer(0)
    val p1 = peer(1)
    registry.processHeartbeat(heartbeat(p0))
    registry.processHeartbeat(heartbeat(p1))
    registry.processHeartbeat(heartbeat(p0.copy(state = MppPeerState.OutputComplete)))
    registry.processHeartbeat(heartbeat(p1.copy(state = MppPeerState.OutputComplete)))

    def terminal(snapshot: MppPeerSnapshot): MppTerminalEvent =
      MppTerminalEvent(
        s"terminal-${snapshot.peerIndex}",
        snapshot.runId,
        snapshot.peerIndex,
        snapshot.expectedPeerCount,
        snapshot.taskAttemptId,
        snapshot.executorId,
        snapshot.executorSessionId,
        MppPeerState.Succeeded,
        now
      )

    registry.processHeartbeat(heartbeat(p0, terminals = Seq(terminal(p0))))
    registry.processHeartbeat(heartbeat(p1, terminals = Seq(terminal(p1))))
    assert(registry.snapshot(run).exists(_.state == MppQueryRunState.Terminal))

    val stale = p0.copy(taskAttemptId = 999L, executorSessionId = "new")
    registry.processHeartbeat(heartbeat(stale))
    assert(registry.snapshot(run).get.peers.map(_.taskAttemptId) == Seq(100L, 101L))

    now = 1001L
    registry.expirePeers()
    assert(registry.snapshot(run).isEmpty)
  }
}
