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
package org.apache.gluten.flux.control

import org.apache.spark.internal.Logging

import scala.collection.mutable

/**
 * Driver-owned aggregate lifecycle for all peers in one [[FluxQueryRunId]].
 *
 * This must not be confused with [[FluxPeerState]], which is reported independently by each
 * executor. A run stays `Starting` until all expected peer indices join, moves to `Completing` once
 * every peer reports output EOS, moves to `Aborting` on the first failure, and becomes `Terminal`
 * after every known peer is terminal.
 */
private[control] object FluxQueryRunState {
  val Starting: String = "STARTING"
  val Running: String = "RUNNING"
  val Completing: String = "COMPLETING"
  val Aborting: String = "ABORTING"
  val Terminal: String = "TERMINAL"
}

/** Immutable test/debug view of one peer binding held by the driver registry. */
final private[control] case class FluxPeerControlSnapshot(
    peerIndex: Int,
    taskAttemptId: Long,
    executorId: String,
    executorSessionId: String,
    state: String,
    lastHeartbeatMs: Long,
    acceptedAbortSequence: Long,
    terminal: Boolean,
    outputComplete: Boolean)

/** Immutable test/debug view of one driver-owned FLUX run record. */
final private[control] case class FluxQueryRunSnapshot(
    runId: FluxQueryRunId,
    expectedPeerCount: Int,
    state: String,
    peers: Seq[FluxPeerControlSnapshot],
    firstFailure: Option[String],
    abortSequence: Long,
    terminalAtMs: Option[Long])

/**
 * Driver-side state machine for the peers participating in Spark FLUX stage attempts.
 *
 * Heartbeats register retry-safe peer bindings and refresh leases. The first explicit failure,
 * executor removal, Spark task failure, or expired peer lease moves the run to `Aborting` and
 * allocates attempt-bound abort commands for all non-terminal peers. Commands are replayed until a
 * matching peer reports the sequence as accepted.
 *
 * All mutation is serialized on this registry instance. Terminal records are retained as tombstones
 * so late heartbeats cannot recreate a completed run or deliver stale commands to a retry.
 *
 * @param peerTimeoutMs
 *   Maximum heartbeat silence for a peer after its run starts.
 * @param terminalRetentionMs
 *   Duration to retain terminal tombstones before purging them.
 * @param clock
 *   Time source injected for deterministic lease and retention tests.
 */
class GlutenFluxQueryControlRegistry(
    peerTimeoutMs: Long,
    terminalRetentionMs: Long = GlutenFluxControlPlaneConfig.TerminalRetentionMsDefault,
    clock: () => Long = () => System.currentTimeMillis())
  extends Logging {

  final private class PeerBinding(
      val peerIndex: Int,
      val taskAttemptId: Long,
      val executorId: String,
      val executorSessionId: String,
      var state: String,
      var lastHeartbeatMs: Long,
      var acceptedAbortSequence: Long,
      var terminal: Boolean,
      var outputComplete: Boolean)

  final private class RunRecord(
      val runId: FluxQueryRunId,
      val expectedPeerCount: Int,
      var state: String,
      val peers: mutable.Map[Int, PeerBinding],
      var firstFailure: Option[String],
      var abortSequence: Long,
      val commands: mutable.ArrayBuffer[FluxAbortQuery],
      var terminalAtMs: Option[Long])

  private val runs = mutable.HashMap.empty[FluxQueryRunId, RunRecord]

  def processHeartbeat(heartbeat: FluxQueryHeartbeat): FluxQueryHeartbeatAck = synchronized {
    val now = clock()
    val acknowledgedEvents = mutable.ArrayBuffer.empty[String]

    heartbeat.peers.foreach(registerPeerLocked(_, heartbeat, now))
    heartbeat.failures.foreach {
      failure =>
        acknowledgedEvents += failure.eventId
        withMatchingPeerLocked(
          failure.runId,
          failure.peerIndex,
          failure.taskAttemptId,
          failure.executorId,
          failure.executorSessionId)(run => failRunLocked(run, failure.reason, now))
    }
    heartbeat.terminals.foreach {
      terminal =>
        acknowledgedEvents += terminal.eventId
        withMatchingPeerLocked(
          terminal.runId,
          terminal.peerIndex,
          terminal.taskAttemptId,
          terminal.executorId,
          terminal.executorSessionId) {
          run =>
            val peer = run.peers(terminal.peerIndex)
            peer.state = terminal.terminalState
            peer.terminal = true
            peer.lastHeartbeatMs = now
            maybeFinishRunLocked(run, now)
        }
    }
    // Refresh the sender before evaluating leases so a heartbeat arriving on the timeout boundary
    // cannot expire its own peer using the previous heartbeat timestamp.
    expirePeersLocked(now)

    val commands = heartbeat.peers.flatMap {
      snapshot =>
        runs.get(snapshot.runId).toSeq.flatMap {
          run =>
            run.commands.filter {
              command =>
                command.peerIndex == snapshot.peerIndex &&
                command.taskAttemptId == snapshot.taskAttemptId &&
                command.executorId == snapshot.executorId &&
                command.executorSessionId == snapshot.executorSessionId &&
                snapshot.acceptedAbortSequence < command.sequence
            }
        }
    }.distinct

    val peerCompletions = heartbeat.peers.flatMap {
      snapshot =>
        runs.get(snapshot.runId).toSeq.flatMap {
          run =>
            run.peers
              .get(snapshot.peerIndex)
              .filter {
                peer =>
                  run.state == FluxQueryRunState.Completing &&
                  !peer.terminal &&
                  peer.taskAttemptId == snapshot.taskAttemptId &&
                  peer.executorId == snapshot.executorId &&
                  peer.executorSessionId == snapshot.executorSessionId
              }
              .map {
                peer =>
                  FluxPeerCompletion(
                    run.runId,
                    peer.peerIndex,
                    peer.taskAttemptId,
                    peer.executorId,
                    peer.executorSessionId)
              }
        }
    }.distinct

    FluxQueryHeartbeatAck(commands, acknowledgedEvents.distinct.toSeq, now, peerCompletions)
  }

  def onExecutorRemoved(executorId: String, reason: String): Unit = synchronized {
    val now = clock()
    runs.values.foreach {
      run =>
        val removedPeers = run.peers.values.filter(p => p.executorId == executorId && !p.terminal)
        if (removedPeers.nonEmpty) {
          failRunLocked(run, s"Executor $executorId removed: $reason", now)
          removedPeers.foreach {
            peer =>
              peer.state = FluxPeerState.Failed
              peer.terminal = true
          }
          maybeFinishRunLocked(run, now)
        }
    }
  }

  def onTaskEnd(taskAttemptId: Long, executorId: String, failed: Boolean, reason: String): Unit =
    synchronized {
      val now = clock()
      runs.values.foreach {
        run =>
          run.peers.values
            .find(
              p => p.taskAttemptId == taskAttemptId && p.executorId == executorId && !p.terminal)
            .foreach {
              peer =>
                if (failed) {
                  failRunLocked(run, s"Spark task $taskAttemptId failed: $reason", now)
                  peer.state = FluxPeerState.Failed
                } else {
                  // A successful Spark task has consumed (or deliberately accepted an empty)
                  // root output.  Treat task success as an implicit output-EOS acknowledgement.
                  // This is important for distributed file writes: FileFormatWriter may finish an
                  // empty/sparse output partition without another iterator hasNext() call, so the
                  // executor's explicit OutputComplete heartbeat can race with
                  // SparkListenerTaskEnd.
                  peer.outputComplete = true
                  peer.state = FluxPeerState.Succeeded
                }
                peer.terminal = true
                maybeFinishRunLocked(run, now)
            }
      }
    }

  /** Invoked by the service sweeper and exposed for deterministic tests. */
  def expirePeers(): Unit = synchronized {
    expirePeersLocked(clock())
  }

  private[control] def snapshot(
      runId: FluxQueryRunId): Option[FluxQueryRunSnapshot] = synchronized {
    runs.get(runId).map {
      run =>
        FluxQueryRunSnapshot(
          run.runId,
          run.expectedPeerCount,
          run.state,
          run.peers.values.toSeq.sortBy(_.peerIndex).map {
            peer =>
              FluxPeerControlSnapshot(
                peer.peerIndex,
                peer.taskAttemptId,
                peer.executorId,
                peer.executorSessionId,
                peer.state,
                peer.lastHeartbeatMs,
                peer.acceptedAbortSequence,
                peer.terminal,
                peer.outputComplete
              )
          },
          run.firstFailure,
          run.abortSequence,
          run.terminalAtMs
        )
    }
  }

  private[control] def size: Int = synchronized(runs.size)

  private def registerPeerLocked(
      snapshot: FluxPeerSnapshot,
      heartbeat: FluxQueryHeartbeat,
      now: Long): Unit = {
    if (
      snapshot.executorId != heartbeat.executorId ||
      snapshot.executorSessionId != heartbeat.executorSessionId ||
      snapshot.peerIndex < 0 || snapshot.peerIndex >= snapshot.expectedPeerCount ||
      snapshot.expectedPeerCount <= 0
    ) {
      logWarning(s"Ignoring invalid FLUX peer snapshot for run=${snapshot.runId.logId}")
      return
    }

    val run = runs.get(snapshot.runId) match {
      case Some(existing) =>
        if (existing.expectedPeerCount != snapshot.expectedPeerCount) {
          failRunLocked(
            existing,
            s"Peer-count mismatch: expected=${existing.expectedPeerCount}, " +
              s"reported=${snapshot.expectedPeerCount}",
            now)
        }
        existing
      case None =>
        val created = new RunRecord(
          snapshot.runId,
          snapshot.expectedPeerCount,
          FluxQueryRunState.Starting,
          mutable.HashMap.empty,
          None,
          0L,
          mutable.ArrayBuffer.empty,
          None)
        runs.put(snapshot.runId, created)
        created
    }

    if (run.state == FluxQueryRunState.Terminal) {
      return
    }

    run.peers.get(snapshot.peerIndex) match {
      case Some(peer)
          if peer.taskAttemptId == snapshot.taskAttemptId &&
            peer.executorId == snapshot.executorId &&
            peer.executorSessionId == snapshot.executorSessionId =>
        peer.state = snapshot.state
        peer.lastHeartbeatMs = now
        peer.acceptedAbortSequence =
          math.max(peer.acceptedAbortSequence, snapshot.acceptedAbortSequence)
        // Succeeded is emitted only after the owning Spark task has accepted all of its output.
        // Preserve that fact even if the intermediate OutputComplete heartbeat lost the race with
        // the terminal heartbeat/task-end event.
        peer.outputComplete = peer.outputComplete ||
          snapshot.state == FluxPeerState.OutputComplete ||
          snapshot.state == FluxPeerState.Succeeded
        peer.terminal = FluxPeerState.isTerminal(snapshot.state)
      case Some(peer) =>
        failRunLocked(
          run,
          s"Duplicate attempt for peer ${snapshot.peerIndex}: existing task=" +
            s"${peer.taskAttemptId}@${peer.executorId}/${peer.executorSessionId}, new task=" +
            s"${snapshot.taskAttemptId}@${snapshot.executorId}/${snapshot.executorSessionId}",
          now
        )
        addAbortCommandLocked(
          run,
          snapshot.peerIndex,
          snapshot.taskAttemptId,
          snapshot.executorId,
          snapshot.executorSessionId,
          now)
      case None =>
        run.peers.put(
          snapshot.peerIndex,
          new PeerBinding(
            snapshot.peerIndex,
            snapshot.taskAttemptId,
            snapshot.executorId,
            snapshot.executorSessionId,
            snapshot.state,
            now,
            snapshot.acceptedAbortSequence,
            FluxPeerState.isTerminal(snapshot.state),
            snapshot.state == FluxPeerState.OutputComplete ||
              snapshot.state == FluxPeerState.Succeeded
          )
        )
    }

    val peer = run.peers(snapshot.peerIndex)
    if (
      run.expectedPeerCount > 1 && peer.terminal &&
      (!peer.outputComplete || peer.state != FluxPeerState.Succeeded)
    ) {
      val reason =
        if (!peer.outputComplete) {
          s"Peer ${peer.peerIndex} terminated before output EOS"
        } else {
          s"Peer ${peer.peerIndex} terminated with state ${peer.state}"
        }
      failRunLocked(run, reason, now)
    }

    if (
      run.state == FluxQueryRunState.Starting &&
      run.peers.keySet == (0 until run.expectedPeerCount).toSet
    ) {
      run.state = FluxQueryRunState.Running
      logInfo(s"FLUX query run ${run.runId.logId} entered RUNNING")
    }
    maybeAuthorizeCompletionLocked(run)
    if (run.state == FluxQueryRunState.Aborting) {
      ensureAbortCommandsLocked(run, now)
    }
    maybeFinishRunLocked(run, now)
  }

  private def withMatchingPeerLocked(
      runId: FluxQueryRunId,
      peerIndex: Int,
      taskAttemptId: Long,
      executorId: String,
      executorSessionId: String)(f: RunRecord => Unit): Unit = {
    runs.get(runId).foreach {
      run =>
        run.peers
          .get(peerIndex)
          .filter {
            peer =>
              peer.taskAttemptId == taskAttemptId &&
              peer.executorId == executorId &&
              peer.executorSessionId == executorSessionId
          }
          .foreach(_ => f(run))
    }
  }

  private def failRunLocked(run: RunRecord, reason: String, now: Long): Unit = {
    if (run.state == FluxQueryRunState.Terminal) {
      return
    }
    if (run.firstFailure.isEmpty) {
      run.firstFailure = Some(reason)
      logWarning(s"FLUX query run ${run.runId.logId} failed: $reason")
    }
    run.state = FluxQueryRunState.Aborting
    ensureAbortCommandsLocked(run, now)
  }

  private def ensureAbortCommandsLocked(run: RunRecord, now: Long): Unit = {
    run.peers.values.filterNot(_.terminal).foreach {
      peer =>
        addAbortCommandLocked(
          run,
          peer.peerIndex,
          peer.taskAttemptId,
          peer.executorId,
          peer.executorSessionId,
          now)
    }
  }

  private def addAbortCommandLocked(
      run: RunRecord,
      peerIndex: Int,
      taskAttemptId: Long,
      executorId: String,
      executorSessionId: String,
      now: Long): Unit = {
    if (
      !run.commands.exists {
        command =>
          command.peerIndex == peerIndex &&
          command.taskAttemptId == taskAttemptId &&
          command.executorId == executorId &&
          command.executorSessionId == executorSessionId
      }
    ) {
      run.abortSequence += 1L
      run.commands += FluxAbortQuery(
        run.runId,
        peerIndex,
        taskAttemptId,
        executorId,
        executorSessionId,
        run.abortSequence,
        run.firstFailure.getOrElse("FLUX peer failed"),
        now)
    }
  }

  private def expirePeersLocked(now: Long): Unit = {
    runs.values.toSeq.foreach {
      run =>
        if (
          run.state == FluxQueryRunState.Running ||
          run.state == FluxQueryRunState.Completing ||
          run.state == FluxQueryRunState.Aborting
        ) {
          val expired = run.peers.values.filter {
            peer => !peer.terminal && now - peer.lastHeartbeatMs >= peerTimeoutMs
          }.toSeq
          expired.foreach {
            peer =>
              failRunLocked(
                run,
                s"Peer ${peer.peerIndex} on executor ${peer.executorId} missed heartbeat for " +
                  s"${now - peer.lastHeartbeatMs} ms",
                now)
              peer.state = FluxPeerState.Failed
              peer.terminal = true
          }
          maybeFinishRunLocked(run, now)
        }
    }
    runs.retain {
      case (_, run) =>
        run.terminalAtMs.forall(terminalAt => now - terminalAt < terminalRetentionMs)
    }
  }

  private def maybeAuthorizeCompletionLocked(run: RunRecord): Unit = {
    if (
      run.state == FluxQueryRunState.Running &&
      run.peers.size == run.expectedPeerCount &&
      run.peers.values.forall(peer => peer.outputComplete && !peer.terminal)
    ) {
      run.state = FluxQueryRunState.Completing
      logInfo(s"FLUX query run ${run.runId.logId} authorized all peers to close")
    }
  }

  private def maybeFinishRunLocked(run: RunRecord, now: Long): Unit = {
    val allExpectedFinished =
      run.peers.size == run.expectedPeerCount && run.peers.values.forall(_.terminal)
    val allKnownFinishedAfterAbort =
      run.state == FluxQueryRunState.Aborting &&
        run.peers.nonEmpty &&
        run.peers.values.forall(_.terminal)
    if (allExpectedFinished || allKnownFinishedAfterAbort) {
      run.state = FluxQueryRunState.Terminal
      if (run.terminalAtMs.isEmpty) {
        run.terminalAtMs = Some(now)
        logInfo(s"FLUX query run ${run.runId.logId} entered TERMINAL")
      }
    }
  }
}
