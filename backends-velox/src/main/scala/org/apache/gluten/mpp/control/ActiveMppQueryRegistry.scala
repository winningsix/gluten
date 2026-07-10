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
import org.apache.spark.internal.Logging

import java.util.UUID
import java.util.concurrent.{ConcurrentHashMap, CountDownLatch, Executors, ScheduledExecutorService, ThreadFactory, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong, AtomicReference}

import scala.collection.JavaConverters._
import scala.util.control.NonFatal

/** Executor-local key for the Spark task attempt that owns one native MPP peer handle. */
final private[control] case class ActiveMppQueryKey(
    runId: MppQueryRunId,
    peerIndex: Int,
    taskAttemptId: Long)

/**
 * Executor-local lifecycle holder for one native MPP peer.
 *
 * The Spark task thread registers this holder after native create and before native start. The
 * holder coordinates a start/abort race with `startFinished`: an early abort is latched, while an
 * abort that overlaps native start waits for the task thread to publish the start result.
 *
 * `abortNative` requests cooperative native abort only. The holder deliberately does not own close
 * or the native handle; `MppNativeQueryRDD` remains the sole exactly-once close owner.
 */
final private[gluten] class ActiveMppQuery private[control] (
    val runId: MppQueryRunId,
    val peerIndex: Int,
    val expectedPeerCount: Int,
    val taskAttemptId: Long,
    val executorId: String,
    val executorSessionId: String,
    val taskContext: TaskContext,
    abortNative: () => Unit) {

  private val currentState = new AtomicReference[String](MppPeerState.Created)
  private val abortRequestedFlag = new AtomicBoolean(false)
  private val abortQueued = new AtomicBoolean(false)
  private val acceptedAbortSeq = new AtomicLong(0L)
  private val startFinished = new CountDownLatch(1)
  private val peerCompletion = new CountDownLatch(1)
  private val startSucceeded = new AtomicBoolean(false)
  private val peerCompletionAuthorized = new AtomicBoolean(false)
  private val failureReported = new AtomicBoolean(false)
  private val terminalReported = new AtomicBoolean(false)

  @volatile private var firstAbortReason: String = ""

  private[control] val key = ActiveMppQueryKey(runId, peerIndex, taskAttemptId)

  /** Task-thread gate called immediately before nativeStartMppQuery. */
  def beginNativeStart(): Boolean = synchronized {
    if (abortRequestedFlag.get() || taskContext.isInterrupted()) {
      abortRequestedFlag.set(true)
      currentState.set(MppPeerState.AbortRequested)
      startFinished.countDown()
      false
    } else {
      currentState.set(MppPeerState.Starting)
      true
    }
  }

  /** Task-thread notification that releases an abort worker racing with native start. */
  def finishNativeStart(succeeded: Boolean): Unit = synchronized {
    startSucceeded.set(succeeded)
    if (succeeded && !abortRequestedFlag.get()) {
      currentState.set(MppPeerState.Running)
    } else if (abortRequestedFlag.get()) {
      currentState.set(MppPeerState.AbortRequested)
    }
    startFinished.countDown()
  }

  /**
   * Claim an abort request. Returns true only for the caller responsible for enqueueing native
   * abort. Later command sequences are still acknowledged even when an abort is already queued.
   */
  private[control] def requestAbort(sequence: Long, reason: String): Boolean = {
    if (sequence > 0L) {
      var current = acceptedAbortSeq.get()
      while (sequence > current && !acceptedAbortSeq.compareAndSet(current, sequence)) {
        current = acceptedAbortSeq.get()
      }
    }
    if (firstAbortReason.isEmpty) {
      firstAbortReason = reason
    }
    abortRequestedFlag.set(true)
    peerCompletion.countDown()
    if (!isTerminal) {
      currentState.set(MppPeerState.AbortRequested)
      abortQueued.compareAndSet(false, true)
    } else {
      false
    }
  }

  /** Dedicated abort-worker call. It never closes or releases the native handle. */
  private[control] def runNativeAbort(): Unit = {
    startFinished.await()
    if (startSucceeded.get() && !isTerminal) {
      abortNative()
    }
  }

  private[control] def markFailureReported(): Boolean = failureReported.compareAndSet(false, true)

  /** Publish local root EOS without closing the native coordinator. */
  private[control] def markOutputComplete(): Boolean = synchronized {
    if (abortRequestedFlag.get() || isTerminal || taskContext.isInterrupted()) {
      false
    } else {
      currentState.set(MppPeerState.OutputComplete)
      true
    }
  }

  /** Release the task only when the driver authorizes this exact peer attempt to close. */
  private[control] def authorizePeerCompletion(): Unit = synchronized {
    if (!abortRequestedFlag.get() && !isTerminal) {
      peerCompletionAuthorized.set(true)
      peerCompletion.countDown()
    }
  }

  /**
   * Wait for completion authorization or abort without depending on the periodic heartbeat phase. A
   * timeout leaves the latch intact so the caller can publish another output-complete heartbeat.
   */
  @throws[InterruptedException]
  private[control] def awaitPeerCompletion(timeoutMs: Long): Option[Boolean] = {
    if (peerCompletion.await(timeoutMs, TimeUnit.MILLISECONDS)) {
      Some(
        peerCompletionAuthorized.get() && !abortRequestedFlag.get() &&
          !taskContext.isInterrupted())
    } else {
      None
    }
  }

  private[control] def markTerminal(state: String): Boolean = {
    if (terminalReported.compareAndSet(false, true)) {
      currentState.set(state)
      startFinished.countDown()
      peerCompletion.countDown()
      true
    } else {
      false
    }
  }

  private[control] def snapshot(): MppPeerSnapshot =
    MppPeerSnapshot(
      runId,
      peerIndex,
      expectedPeerCount,
      taskAttemptId,
      executorId,
      executorSessionId,
      currentState.get(),
      acceptedAbortSeq.get())

  def isAbortRequested: Boolean = abortRequestedFlag.get()

  def isTerminal: Boolean = MppPeerState.isTerminal(currentState.get())

  private[control] def abortReason: String = firstAbortReason
}

/**
 * Concurrent executor-local index of active peer holders.
 *
 * Remote commands first resolve the run, peer index, and task attempt key, then validate executor
 * and executor-session identity before they can claim an abort sequence.
 */
private[control] class ActiveMppQueryRegistry {
  private val entries = new ConcurrentHashMap[ActiveMppQueryKey, ActiveMppQuery]()

  def register(query: ActiveMppQuery): Unit = {
    val previous = entries.putIfAbsent(query.key, query)
    if (previous != null) {
      throw new IllegalStateException(
        s"MPP query already registered: run=${query.runId.logId} peer=${query.peerIndex} " +
          s"taskAttempt=${query.taskAttemptId}")
    }
  }

  def find(command: MppAbortQuery): Option[ActiveMppQuery] =
    Option(entries.get(ActiveMppQueryKey(command.runId, command.peerIndex, command.taskAttemptId)))
      .filter {
        query =>
          query.executorId == command.executorId &&
          query.executorSessionId == command.executorSessionId
      }

  def find(completion: MppPeerCompletion): Option[ActiveMppQuery] =
    Option(
      entries.get(
        ActiveMppQueryKey(completion.runId, completion.peerIndex, completion.taskAttemptId)))
      .filter {
        query =>
          query.executorId == completion.executorId &&
          query.executorSessionId == completion.executorSessionId
      }

  // `Iterable.toSeq` is a lazy Stream in Scala 2.12. Keep this snapshot strict: heartbeat mapping
  // must not serialize a Stream closure that still captures the non-serializable ActiveMppQuery.
  def values: Seq[ActiveMppQuery] = entries.values().asScala.iterator.toVector

  def remove(query: ActiveMppQuery): Unit = entries.remove(query.key, query)

  def nonEmpty: Boolean = !entries.isEmpty
}

/**
 * Executor-side orchestration for heartbeat, interruption detection, and native abort.
 *
 * Three independent daemon executors isolate control work:
 *   - the interrupt watcher scans `TaskContext.isInterrupted`;
 *   - the heartbeat poller performs potentially blocking driver RPC;
 *   - the abort worker serializes calls to the registered native-abort closure.
 *
 * Local interruption and remote commands share the same sequence-deduplicated abort path. Failure
 * and terminal events remain buffered until the driver acknowledges their stable event IDs.
 *
 * @param executorId
 *   Spark executor that owns every holder in this agent.
 * @param executorSessionId
 *   Process-unique identity used to reject commands from a previous executor incarnation.
 * @param askDriver
 *   Spark plugin request-response function used for heartbeats.
 * @param interruptPollMs
 *   Interval between local TaskContext interruption scans.
 * @param heartbeatMs
 *   Interval between control heartbeats while active or retryable state exists.
 * @param clock
 *   Time source injected for deterministic tests.
 * @param startThreads
 *   Whether to create background workers; false in deterministic unit tests.
 */
private[control] class GlutenMppExecutorControlAgent(
    val executorId: String,
    val executorSessionId: String,
    askDriver: Any => AnyRef,
    interruptPollMs: Long,
    heartbeatMs: Long,
    clock: () => Long = () => System.currentTimeMillis(),
    startThreads: Boolean = true)
  extends Logging {

  private val MaxFailureMessageChars = 2048
  private val MaxFailureStackChars = 6144
  private val registry = new ActiveMppQueryRegistry
  private val pendingFailures = new ConcurrentHashMap[String, MppQueryFailure]()
  private val pendingTerminals = new ConcurrentHashMap[String, MppTerminalEvent]()
  private val closed = new AtomicBoolean(false)
  private val heartbeatFailureLogged = new AtomicBoolean(false)

  // Spark's plugin control channel is pull-only, so a peer that reaches EOS first cannot receive
  // the authorization produced by the last peer's heartbeat. Poll briefly from the waiting task
  // to avoid quantizing normal peer skew by the 250 ms periodic heartbeat. After one configured
  // heartbeat interval, return to the regular cadence to bound failure-path RPC load.
  private val outputCompletionFastPollMs = math.max(1L, math.min(5L, heartbeatMs))

  private var interruptWatcher: ScheduledExecutorService = _
  private var heartbeatPoller: ScheduledExecutorService = _
  private val abortWorker =
    Executors.newSingleThreadExecutor(threadFactory("gluten-mpp-abort-worker"))

  if (startThreads) {
    interruptWatcher =
      Executors.newSingleThreadScheduledExecutor(threadFactory("gluten-mpp-interrupt-watcher"))
    heartbeatPoller =
      Executors.newSingleThreadScheduledExecutor(threadFactory("gluten-mpp-heartbeat-poller"))
    interruptWatcher.scheduleWithFixedDelay(
      new Runnable {
        override def run(): Unit = safely("interrupt scan")(scanInterrupts())
      },
      interruptPollMs,
      interruptPollMs,
      TimeUnit.MILLISECONDS)
    heartbeatPoller.scheduleWithFixedDelay(
      new Runnable {
        override def run(): Unit = safely("heartbeat")(sendHeartbeat())
      },
      heartbeatMs,
      heartbeatMs,
      TimeUnit.MILLISECONDS)
  }

  def register(
      runId: MppQueryRunId,
      peerIndex: Int,
      expectedPeerCount: Int,
      taskContext: TaskContext,
      abortNative: () => Unit): ActiveMppQuery = {
    val query = new ActiveMppQuery(
      runId,
      peerIndex,
      expectedPeerCount,
      taskContext.taskAttemptId(),
      executorId,
      executorSessionId,
      taskContext,
      abortNative)
    registry.register(query)
    query
  }

  def reportFailure(query: ActiveMppQuery, error: Throwable): Unit = {
    if (query.markFailureReported()) {
      val message = truncate(Option(error.getMessage).getOrElse(""), MaxFailureMessageChars)
      val stack = truncate(error.getStackTrace.mkString("\n"), MaxFailureStackChars)
      val event = MppQueryFailure(
        failureEventId(query),
        query.runId,
        query.peerIndex,
        query.expectedPeerCount,
        query.taskAttemptId,
        query.executorId,
        query.executorSessionId,
        error.getClass.getName,
        message,
        stack,
        clock()
      )
      pendingFailures.put(event.eventId, event)
    }
  }

  def reportTerminal(query: ActiveMppQuery, state: String): Unit = {
    if (query.markTerminal(state)) {
      val event = MppTerminalEvent(
        terminalEventId(query),
        query.runId,
        query.peerIndex,
        query.expectedPeerCount,
        query.taskAttemptId,
        query.executorId,
        query.executorSessionId,
        state,
        clock())
      pendingTerminals.put(event.eventId, event)
    }
  }

  /**
   * Publish local root EOS and wait for the driver's all-peer completion authorization. Native
   * abort remains independent and releases the waiter immediately on peer failure or interruption.
   */
  def awaitPeerCompletion(query: ActiveMppQuery): Boolean = {
    if (!query.markOutputComplete()) {
      return false
    }
    // Do not wait for the next scheduled tick to publish EOS. Periodic heartbeats continue to
    // retry if this request fails or another peer has not reached EOS yet.
    try {
      safely("output-complete heartbeat")(sendHeartbeat())
      val fastPollStart = System.nanoTime()
      val fastPollWindowNanos = TimeUnit.MILLISECONDS.toNanos(heartbeatMs)
      var result = query.awaitPeerCompletion(outputCompletionFastPollMs)
      while (result.isEmpty) {
        safely("output-complete heartbeat")(sendHeartbeat())
        val pollMs =
          if (System.nanoTime() - fastPollStart < fastPollWindowNanos) {
            outputCompletionFastPollMs
          } else {
            heartbeatMs
          }
        result = query.awaitPeerCompletion(pollMs)
      }
      result.get
    } catch {
      case _: InterruptedException =>
        val reason = "MPP peer completion wait interrupted"
        Thread.currentThread().interrupt()
        reportFailure(query, new InterruptedException(reason))
        enqueueAbort(query, sequence = 0L, reason)
        false
    }
  }

  private[control] def scanInterrupts(): Unit = {
    if (closed.get()) return
    registry.values
      .filter(query => !query.isTerminal && query.taskContext.isInterrupted())
      .foreach {
        query =>
          val reason = "Spark TaskContext interrupted"
          reportFailure(query, new InterruptedException(reason))
          enqueueAbort(query, sequence = 0L, reason)
      }
  }

  // The periodic worker and an EOS waiter can both request a heartbeat. Keep snapshot/ask/ack
  // processing single-flight so an older response cannot be applied after a newer terminal state.
  private[control] def sendHeartbeat(): Unit = synchronized {
    if (closed.get()) return
    if (!registry.nonEmpty && pendingFailures.isEmpty && pendingTerminals.isEmpty) return

    val heartbeat = MppQueryHeartbeat(
      executorId,
      executorSessionId,
      clock(),
      registry.values.map(_.snapshot()),
      pendingFailures.values().asScala.iterator.toVector,
      pendingTerminals.values().asScala.iterator.toVector
    )
    askDriver(heartbeat) match {
      case ack: MppQueryHeartbeatAck =>
        ack.acknowledgedEventIds.foreach {
          eventId =>
            pendingFailures.remove(eventId)
            val terminal = pendingTerminals.remove(eventId)
            if (terminal != null) {
              registry.values
                .find(query => terminalEventId(query) == eventId)
                .foreach(registry.remove)
            }
        }
        ack.abortCommands.foreach {
          command =>
            registry
              .find(command)
              .foreach(enqueueAbort(_, command.sequence, command.reason))
        }
        ack.peerCompletions.foreach {
          completion => registry.find(completion).foreach(_.authorizePeerCompletion())
        }
      case other =>
        logWarning(s"Unexpected MPP query heartbeat response: $other")
    }
  }

  private def enqueueAbort(query: ActiveMppQuery, sequence: Long, reason: String): Unit = {
    if (query.requestAbort(sequence, reason)) {
      logWarning(
        s"Queueing MPP abort run=${query.runId.logId} peer=${query.peerIndex} " +
          s"taskAttempt=${query.taskAttemptId} sequence=$sequence reason=$reason")
      abortWorker.execute(new Runnable {
        override def run(): Unit = {
          try {
            logWarning(
              s"Starting native MPP abort run=${query.runId.logId} peer=${query.peerIndex}")
            query.runNativeAbort()
            logWarning(
              s"Native MPP abort returned run=${query.runId.logId} peer=${query.peerIndex}")
          } catch {
            case NonFatal(e) =>
              logWarning(
                s"Native MPP abort failed run=${query.runId.logId} peer=${query.peerIndex}",
                e)
          }
        }
      })
    }
  }

  def shutdown(): Unit = {
    if (!closed.compareAndSet(false, true)) return
    registry.values.filterNot(_.isTerminal).foreach {
      query => enqueueAbort(query, sequence = 0L, "Executor plugin shutting down")
    }
    if (interruptWatcher != null) interruptWatcher.shutdownNow()
    if (heartbeatPoller != null) heartbeatPoller.shutdownNow()
    abortWorker.shutdown()
  }

  private[control] def activeQueries: Seq[ActiveMppQuery] = registry.values

  private[control] def pendingFailureCount: Int = pendingFailures.size()

  private[control] def pendingTerminalCount: Int = pendingTerminals.size()

  private def failureEventId(query: ActiveMppQuery): String =
    s"${query.runId.logId}/${query.peerIndex}/${query.taskAttemptId}/failure"

  private def terminalEventId(query: ActiveMppQuery): String =
    s"${query.runId.logId}/${query.peerIndex}/${query.taskAttemptId}/terminal"

  private def truncate(value: String, maxChars: Int): String =
    if (value.length <= maxChars) value else value.substring(0, maxChars)

  private def safely(label: String)(body: => Unit): Unit = {
    val isHeartbeat = label.endsWith("heartbeat")
    try {
      body
      if (isHeartbeat) {
        heartbeatFailureLogged.set(false)
      }
    } catch {
      case _: InterruptedException if closed.get() =>
      case NonFatal(e) if isHeartbeat && heartbeatFailureLogged.compareAndSet(false, true) =>
        logWarning(s"MPP query-control $label failed; retries will continue", e)
      case NonFatal(e) => logDebug(s"MPP query-control $label failed", e)
    }
  }

  private def threadFactory(name: String): ThreadFactory = new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val thread = new Thread(r, name)
      thread.setDaemon(true)
      thread
    }
  }
}

private[control] object GlutenMppExecutorControlAgent {
  def apply(
      executorId: String,
      askDriver: Any => AnyRef,
      interruptPollMs: Long,
      heartbeatMs: Long): GlutenMppExecutorControlAgent =
    new GlutenMppExecutorControlAgent(
      executorId,
      UUID.randomUUID().toString,
      askDriver,
      interruptPollMs,
      heartbeatMs)
}
