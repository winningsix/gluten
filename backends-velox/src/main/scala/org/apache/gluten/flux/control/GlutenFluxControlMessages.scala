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

/**
 * Serializable protocol exchanged by [[GlutenFluxExecutorService]] on each executor and
 * [[GlutenFluxDriverService]] on the Spark driver.
 *
 * Endpoint messages retain the existing executor-to-UCX-endpoint registry. Query messages carry
 * active peer snapshots and retryable failure or terminal events to the driver; heartbeat replies
 * carry attempt-bound abort commands back to executors. The protocol uses Spark plugin
 * `PluginContext.ask` and intentionally contains only serializable values.
 */
sealed trait GlutenFluxControlMessage extends Serializable

/** Executor to Driver: announce / refresh this executor's UCX-capable endpoint. */
final case class RegisterEndpoint(record: FluxExecutorEndpointRecord)
  extends GlutenFluxControlMessage

/** Driver to Executor: ack of a registration, with the registry generation after insert. */
final case class RegisterEndpointAck(generation: Long) extends GlutenFluxControlMessage

/** Executor to Driver: graceful deregistration prior to executor shutdown. */
final case class DeregisterEndpoint(executorId: String) extends GlutenFluxControlMessage

/** Driver to Executor: ack of a deregistration, with the registry generation after remove. */
final case class DeregisterEndpointAck(generation: Long) extends GlutenFluxControlMessage

/**
 * Identifies one Spark stage attempt executing a native FLUX query.
 *
 * `queryId` alone is not retry-safe: a failed stage can be re-run with the same query ID. Including
 * Spark's stage ID and stage-attempt number keeps abort commands and terminal events isolated from
 * a retried stage.
 *
 * @param queryId
 *   Logical FLUX query ID shared by all peers.
 * @param stageId
 *   Spark stage ID that owns the peer tasks.
 * @param stageAttemptNumber
 *   Spark attempt number for that stage.
 */
final case class FluxQueryRunId(queryId: String, stageId: Int, stageAttemptNumber: Int)
  extends Serializable {
  def logId: String = s"$queryId/$stageId/$stageAttemptNumber"
}

/**
 * Executor-reported lifecycle of one native FLUX peer.
 *
 * This is distinct from `FluxQueryRunState`, which is the driver-owned aggregate state for all
 * peers
 * in a run. A peer reaches `AbortRequested` when an abort is accepted locally and reaches a
 * terminal state only after the Spark task thread reports cleanup.
 */
object FluxPeerState {
  val Created: String = "CREATED"
  val Starting: String = "STARTING"
  val Running: String = "RUNNING"
  val OutputComplete: String = "OUTPUT_COMPLETE"
  val AbortRequested: String = "ABORT_REQUESTED"
  val Succeeded: String = "SUCCEEDED"
  val Failed: String = "FAILED"
  val Aborted: String = "ABORTED"

  def isTerminal(state: String): Boolean =
    state == Succeeded || state == Failed || state == Aborted
}

/**
 * Executor-to-driver snapshot of one active peer.
 *
 * The task-attempt and executor-session fields bind the snapshot to the process that owns the
 * native handle. `acceptedAbortSequence` acknowledges that a command was validated and queued; it
 * does not claim that native cleanup has finished.
 */
final case class FluxPeerSnapshot(
    runId: FluxQueryRunId,
    peerIndex: Int,
    expectedPeerCount: Int,
    taskAttemptId: Long,
    executorId: String,
    executorSessionId: String,
    state: String,
    acceptedAbortSequence: Long)
  extends Serializable

/**
 * Retryable executor-to-driver report of a peer failure.
 *
 * The executor retains the event until its `eventId` is acknowledged in a heartbeat response. The
 * driver preserves the first failure as the run's root cause.
 */
final case class FluxQueryFailure(
    eventId: String,
    runId: FluxQueryRunId,
    peerIndex: Int,
    expectedPeerCount: Int,
    taskAttemptId: Long,
    executorId: String,
    executorSessionId: String,
    exceptionClass: String,
    message: String,
    stackTrace: String,
    observedAtMs: Long)
  extends Serializable {
  def reason: String = {
    val detail = Option(message).filter(_.nonEmpty).map(m => s": $m").getOrElse("")
    s"$exceptionClass$detail"
  }
}

/**
 * Retryable notification that the Spark task thread reached terminal cleanup.
 *
 * This event is separate from abort acceptance so the driver can distinguish propagation latency
 * from the time required for native unwind and exactly-once handle close.
 */
final case class FluxTerminalEvent(
    eventId: String,
    runId: FluxQueryRunId,
    peerIndex: Int,
    expectedPeerCount: Int,
    taskAttemptId: Long,
    executorId: String,
    executorSessionId: String,
    terminalState: String,
    observedAtMs: Long)
  extends Serializable

/**
 * Periodic executor-to-driver control request.
 *
 * A heartbeat contains all active peer snapshots and any unacknowledged failure or terminal events
 * owned by this executor session.
 */
final case class FluxQueryHeartbeat(
    executorId: String,
    executorSessionId: String,
    sentAtMs: Long,
    peers: Seq[FluxPeerSnapshot],
    failures: Seq[FluxQueryFailure],
    terminals: Seq[FluxTerminalEvent])
  extends GlutenFluxControlMessage

/**
 * Driver command to abort one exact native FLUX peer.
 *
 * Every identity field must match the active holder before the executor accepts `sequence`. The
 * driver replays the command until that sequence appears in the peer snapshot.
 */
final case class FluxAbortQuery(
    runId: FluxQueryRunId,
    peerIndex: Int,
    taskAttemptId: Long,
    executorId: String,
    executorSessionId: String,
    sequence: Long,
    reason: String,
    issuedAtMs: Long)
  extends Serializable

/**
 * Driver authorization for one peer to close its native coordinator after global end-of-stream.
 *
 * Every expected peer must first report [[FluxPeerState.OutputComplete]]. The driver then returns
 * this attempt-bound authorization on heartbeats until the task closes and reports terminal state.
 * Keeping the identity fields aligned with [[FluxAbortQuery]] prevents a late authorization from a
 * previous task or executor incarnation from releasing a retried peer.
 */
final case class FluxPeerCompletion(
    runId: FluxQueryRunId,
    peerIndex: Int,
    taskAttemptId: Long,
    executorId: String,
    executorSessionId: String)
  extends Serializable

/**
 * Driver response containing pending abort commands and acknowledged event IDs.
 *
 * Acknowledging a failure or terminal event allows the executor to remove it from its retry buffer.
 * Abort commands remain in driver state until the matching peer reports their sequence as accepted.
 */
final case class FluxQueryHeartbeatAck(
    abortCommands: Seq[FluxAbortQuery],
    acknowledgedEventIds: Seq[String],
    driverTimeMs: Long,
    peerCompletions: Seq[FluxPeerCompletion] = Nil)
  extends GlutenFluxControlMessage
