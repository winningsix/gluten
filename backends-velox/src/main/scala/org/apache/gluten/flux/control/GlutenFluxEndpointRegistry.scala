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

import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantLock

import scala.collection.concurrent.TrieMap

/**
 * In-memory driver-side registry of executor UCX endpoints (Phase 1 POC).
 *
 * Phase 1 stores records purely in JVM memory; there is no RPC, persistence, or cross-driver
 * coordination. C3 will wire `RegisterEndpoint` messages from each executor to update this
 * registry; C5 will let `FluxNativeQueryExec` consult it instead of probing.
 *
 * Threading: registration / deregistration is thread-safe; `snapshot()` returns a sorted immutable
 * view; `awaitMinExecutors` blocks until the LIVE count meets `min` or the timeout elapses.
 *
 * The registry assigns a monotonically increasing `generation` to every mutation so callers can
 * detect updates without subscribing.
 */
class GlutenFluxEndpointRegistry {

  // executorId -> latest record. TrieMap is lock-free for reads and atomically updated.
  private val records = TrieMap.empty[String, FluxExecutorEndpointRecord]

  private val generation = new AtomicLong(0L)

  // Lock + condition variable used only by awaitMinExecutors / notifyChange.
  private val updateLock = new ReentrantLock()
  private val updateCond = updateLock.newCondition()

  /**
   * Insert or replace the record for the given executor and return the registry generation after
   * the change. The record stored in the registry has its `generation` field rewritten to match.
   */
  def register(record: FluxExecutorEndpointRecord): Long = {
    val g = generation.incrementAndGet()
    val stamped = record.copy(generation = g)
    records.put(record.executorId, stamped)
    notifyChange()
    g
  }

  /** Remove the record (if any) for the given executor and return the registry generation. */
  def deregister(executorId: String): Long = {
    val g = generation.incrementAndGet()
    records.remove(executorId)
    notifyChange()
    g
  }

  /** Immutable, deterministic snapshot of LIVE records sorted by executorId. */
  def snapshot(): Seq[FluxExecutorEndpointRecord] =
    records.values.toSeq.filter(_.isLive).sortBy(_.executorId)

  /** Current monotonic generation counter. Useful for change detection in resolver layers. */
  def currentGeneration(): Long = generation.get()

  /**
   * Block until the registry contains at least `min` LIVE records or `timeoutMs` elapses. Returns
   * true on threshold met, false on timeout. Returns true immediately when the threshold is already
   * satisfied.
   */
  def awaitMinExecutors(min: Int, timeoutMs: Long): Boolean = {
    if (min <= 0) return true
    if (snapshot().size >= min) return true
    val deadlineNs = System.nanoTime() + math.max(0L, timeoutMs) * 1000000L
    updateLock.lock()
    try {
      while (snapshot().size < min) {
        val remainingNs = deadlineNs - System.nanoTime()
        if (remainingNs <= 0L) return false
        updateCond.awaitNanos(remainingNs)
      }
      true
    } finally {
      updateLock.unlock()
    }
  }

  private def notifyChange(): Unit = {
    updateLock.lock()
    try { updateCond.signalAll() }
    finally { updateLock.unlock() }
  }
}

object GlutenFluxEndpointRegistry {
  def apply(): GlutenFluxEndpointRegistry = new GlutenFluxEndpointRegistry()
}
