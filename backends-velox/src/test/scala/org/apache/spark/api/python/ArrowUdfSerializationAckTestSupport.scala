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
package org.apache.spark.api.python

import java.util.concurrent.{CountDownLatch, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong, AtomicReference}

object ArrowUdfSerializationAckTestSupport {
  private val AckDelayMillis = 75L

  private val deferredAck = new AtomicReference[() => Unit]()
  private val ackWasDeferred = new AtomicBoolean(false)
  private val outputWaitWasObserved = new AtomicBoolean(false)
  private val queueWaitWasObserved = new AtomicBoolean(false)
  private val ackRanOnDifferentThread = new AtomicBoolean(false)
  private val outputWaitStartedAtNanos = new AtomicLong(0L)
  private val ackDelayNanos = new AtomicLong(0L)
  private val outputThread = new AtomicReference[Thread]()
  private val workerFailure = new AtomicReference[Throwable]()
  @volatile private var outputWaitStarted = new CountDownLatch(1)
  @volatile private var ackWorker: Thread = null

  private object Hook extends ArrowUdfSerializationAckHook {
    override def acknowledge(ack: () => Unit): Unit = {
      if (deferredAck.compareAndSet(null, ack)) {
        ackWasDeferred.set(true)
      } else {
        ack()
      }
    }

    override def outputWaitingForAck(): Unit = {
      outputWaitWasObserved.set(true)
      if (deferredAck.get() == null) {
        throw new IllegalStateException("Output waited before the test hook deferred an ACK")
      }
      outputThread.set(Thread.currentThread())
      outputWaitStartedAtNanos.set(System.nanoTime())
      outputWaitStarted.countDown()
    }
  }

  def install(): Unit = {
    deferredAck.set(null)
    ackWasDeferred.set(false)
    outputWaitWasObserved.set(false)
    queueWaitWasObserved.set(false)
    ackRanOnDifferentThread.set(false)
    outputWaitStartedAtNanos.set(0L)
    ackDelayNanos.set(0L)
    outputThread.set(null)
    workerFailure.set(null)
    outputWaitStarted = new CountDownLatch(1)
    ArrowUdfSerializationAckHookRegistry.installForTesting(Hook)

    ackWorker = new Thread(
      () => {
        try {
          if (!outputWaitStarted.await(5, TimeUnit.SECONDS)) {
            throw new IllegalStateException(
              "Timed out waiting for Arrow UDF output ACK coordination")
          }
          Thread.sleep(AckDelayMillis)
          val taskThread = outputThread.get()
          val waitDeadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
          while (
            taskThread.getState != Thread.State.TIMED_WAITING &&
            System.nanoTime() < waitDeadlineNanos
          ) {
            Thread.sleep(1L)
          }
          if (taskThread.getState != Thread.State.TIMED_WAITING) {
            throw new IllegalStateException(
              "Output task thread did not enter the serialization ACK timed wait")
          }
          queueWaitWasObserved.set(true)
          ackRanOnDifferentThread.set(Thread.currentThread() ne taskThread)
          ackDelayNanos.set(System.nanoTime() - outputWaitStartedAtNanos.get())
          val ack = deferredAck.getAndSet(null)
          if (ack == null) {
            throw new IllegalStateException("Deferred Arrow UDF serialization ACK was missing")
          }
          ack()
        } catch {
          case interrupted: InterruptedException =>
            Thread.currentThread().interrupt()
            workerFailure.compareAndSet(null, interrupted)
          case t: Throwable => workerFailure.compareAndSet(null, t)
        }
      },
      "arrow-udf-serialization-ack-test-worker"
    )
    ackWorker.setDaemon(false)
    ackWorker.start()
  }

  def clear(): Unit = {
    try {
      val worker = ackWorker
      if (worker != null && worker.isAlive) {
        worker.interrupt()
        worker.join(TimeUnit.SECONDS.toMillis(5))
        if (worker.isAlive) {
          throw new IllegalStateException("Arrow UDF serialization ACK test worker did not stop")
        }
      }
      val ack = deferredAck.getAndSet(null)
      if (ack != null) {
        ack()
      }
    } finally {
      ArrowUdfSerializationAckHookRegistry.clearForTesting()
    }
  }

  def ackDeferred: Boolean = ackWasDeferred.get()

  def outputWaitObserved: Boolean = outputWaitWasObserved.get()

  def actualQueueWaitObserved: Boolean = queueWaitWasObserved.get()

  def ackFromDifferentThread: Boolean = ackRanOnDifferentThread.get()

  def observedAckDelayMillis: Long = TimeUnit.NANOSECONDS.toMillis(ackDelayNanos.get())

  def asyncFailure: Throwable = workerFailure.get()

  def workerAlive: Boolean = ackWorker != null && ackWorker.isAlive
}
