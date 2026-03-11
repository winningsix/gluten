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

import org.apache.gluten.vectorized.ShufflePayloadCatalogJniWrapper

import org.apache.spark.SparkContext
import org.apache.spark.api.plugin._
import org.apache.spark.internal.Logging

import java.util
import java.util.concurrent._
import java.util.concurrent.atomic.AtomicBoolean

// -- Messages (must be Java Serializable) --

case class CatalogCleanupPollMsg(executorId: String)
case class CatalogCleanupResponseMsg(shuffleIds: Array[Int])

/**
 * Spark Plugin for proactive catalog cleanup. Handles both driver side (track pending shuffles,
 * respond to executor polls) and executor side (poll driver, clean local catalog).
 *
 * Add to spark config: spark.plugins=...,org.apache.spark.shuffle.CatalogCleanupPlugin
 *
 * Or registered automatically by ColumnarShuffleManager when skipMerge is enabled.
 */
class CatalogCleanupPlugin extends SparkPlugin {
  override def driverPlugin(): DriverPlugin =
    new CatalogCleanupDriverPlugin()
  override def executorPlugin(): ExecutorPlugin =
    new CatalogCleanupExecutorPlugin()
}

/**
 * Driver side: listens for SQL execution end, stores pending shuffle IDs, responds to executor
 * polls.
 */
class CatalogCleanupDriverPlugin extends DriverPlugin with Logging {

  override def init(sc: SparkContext, ctx: PluginContext): util.Map[String, String] = {
    sc.addSparkListener(new CatalogCleanupListener() {
      override protected def onCleanup(shuffleId: Int): Unit = {
        CatalogCleanupDriverPlugin
          .addPendingCleanup(shuffleId)
        // In local mode, also clean directly
        if (sc.isLocal) {
          try {
            ShufflePayloadCatalogJniWrapper
              .unregisterShuffle(shuffleId)
          } catch {
            case _: Exception =>
          }
        }
      }
    })
    logInfo("CatalogCleanupDriverPlugin initialized")
    java.util.Collections.emptyMap()
  }

  override def receive(msg: Any): AnyRef = {
    msg match {
      case CatalogCleanupPollMsg(executorId) =>
        val ids =
          CatalogCleanupDriverPlugin.drainPending()
        CatalogCleanupResponseMsg(ids)
      case _ => null
    }
  }
}

object CatalogCleanupDriverPlugin {
  private val pending =
    new ConcurrentLinkedQueue[Integer]()

  def addPendingCleanup(shuffleId: Int): Unit = {
    pending.add(shuffleId)
  }

  def drainPending(): Array[Int] = {
    val result =
      new java.util.ArrayList[Int]()
    var id: Integer = pending.poll()
    while (id != null) {
      result.add(id)
      id = pending.poll()
    }
    val arr = new Array[Int](result.size())
    var i = 0
    val iter = result.iterator()
    while (iter.hasNext) {
      arr(i) = iter.next()
      i += 1
    }
    arr
  }
}

/** Executor side: polls driver every second for shuffle IDs to clean from the local catalog. */
class CatalogCleanupExecutorPlugin extends ExecutorPlugin with Logging {

  private var scheduler: ScheduledExecutorService = _
  private val closed = new AtomicBoolean(false)

  override def init(ctx: PluginContext, extraConf: util.Map[String, String]): Unit = {
    val executorId = ctx.executorID()
    scheduler = Executors.newSingleThreadScheduledExecutor(new ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = new Thread(r, "catalog-cleanup-poller")
        t.setDaemon(true)
        t
      }
    })

    scheduler.scheduleWithFixedDelay(
      new Runnable {
        override def run(): Unit = {
          if (closed.get()) return
          try {
            val resp = ctx.ask(CatalogCleanupPollMsg(executorId))
            resp match {
              case CatalogCleanupResponseMsg(ids) if ids.nonEmpty =>
                logInfo(
                  s"Cleaning ${ids.length} shuffle(s)" +
                    s" from catalog: " +
                    s"${ids.mkString(", ")}")
                ids.foreach {
                  id =>
                    try {
                      ShufflePayloadCatalogJniWrapper
                        .unregisterShuffle(id)
                    } catch {
                      case e: Exception =>
                        logWarning(s"Failed to clean shuffle $id", e)
                    }
                }
              case _ =>
            }
          } catch {
            case _: Exception if closed.get() =>
            case e: Exception =>
              logDebug("Poll failed: " + e.getMessage)
          }
        }
      },
      2,
      1,
      TimeUnit.SECONDS
    )

    logInfo(
      s"CatalogCleanupExecutorPlugin started " +
        s"on executor $executorId")
  }

  override def shutdown(): Unit = {
    closed.set(true)
    if (scheduler != null) {
      scheduler.shutdown()
    }
  }
}
