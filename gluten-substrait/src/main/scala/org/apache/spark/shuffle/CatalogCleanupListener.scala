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

import org.apache.spark.SparkEnv
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler._
import org.apache.spark.sql.execution.SQLExecution
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd
import org.apache.spark.util.ShutdownHookManager

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * SparkListener that tracks shuffle-to-execution mapping and triggers cleanup when SQL executions
 * complete. Subclass and override onCleanup() to customize the cleanup action.
 */
class CatalogCleanupListener extends SparkListener with Logging {

  private case class ExecutionShuffleState(
      shuffles: mutable.Set[Int] = mutable.Set.empty,
      activeJobs: mutable.Set[Int] = mutable.Set.empty,
      var sqlEnded: Boolean = false)

  private val executionStates =
    new ConcurrentHashMap[Long, ExecutionShuffleState]()
  private val jobExecutions =
    new ConcurrentHashMap[Int, Long]()
  private val waitMonitor = new Object
  @volatile private var shutdownHookRegistered = false

  private def stateFor(executionId: Long): ExecutionShuffleState =
    executionStates.computeIfAbsent(executionId, _ => ExecutionShuffleState())

  private def cleanupOnSqlEnd: Boolean =
    Option(SparkEnv.get).exists(
      _.conf.getBoolean("spark.gluten.sql.columnar.shuffle.catalog.cleanupOnSqlEnd", false))

  private def waitForActiveJobsOnShutdown: Boolean =
    Option(SparkEnv.get).forall(
      _.conf
        .getBoolean("spark.gluten.sql.columnar.shuffle.catalog.waitForActiveJobsOnShutdown", true))

  private def shutdownWaitTimeoutMs: Long =
    Option(SparkEnv.get)
      .map(
        _.conf.getLong("spark.gluten.sql.columnar.shuffle.catalog.shutdownWaitTimeoutMs", 30000L))
      .getOrElse(30000L)
      .max(0L)

  private def ensureShutdownHookRegistered(): Unit = {
    if (!waitForActiveJobsOnShutdown || shutdownHookRegistered) {
      return
    }
    synchronized {
      if (!waitForActiveJobsOnShutdown || shutdownHookRegistered) {
        return
      }
      ShutdownHookManager.addShutdownHook(() => waitForActiveJobsBeforeShutdown())
      shutdownHookRegistered = true
    }
  }

  private def notifyStateChanged(): Unit =
    waitMonitor.synchronized {
      waitMonitor.notifyAll()
    }

  private def activeEndedSqlExecutions: Seq[(Long, Seq[Int], Seq[Int])] =
    executionStates.entrySet().asScala.toSeq.flatMap {
      entry =>
        val state = entry.getValue
        state.synchronized {
          if (state.sqlEnded && state.activeJobs.nonEmpty && state.shuffles.nonEmpty) {
            Some((entry.getKey, state.activeJobs.toSeq.sorted, state.shuffles.toSeq.sorted))
          } else {
            None
          }
        }
    }

  private def describeActiveExecutions(active: Seq[(Long, Seq[Int], Seq[Int])]): String =
    active
      .map {
        case (executionId, jobs, shuffles) =>
          s"$executionId[jobs=${jobs.mkString(",")}, shuffles=${shuffles.mkString(",")}]"
      }
      .mkString("; ")

  private def waitForActiveJobsBeforeShutdown(): Unit = {
    if (!waitForActiveJobsOnShutdown) {
      return
    }

    val timeoutMs = shutdownWaitTimeoutMs
    if (timeoutMs <= 0L) {
      return
    }

    waitMonitor.synchronized {
      var active = activeEndedSqlExecutions
      if (active.isEmpty) {
        return
      }

      val deadlineNs = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs)
      logInfo(
        s"JVM shutdown waiting up to ${timeoutMs}ms for active SQL shuffle job(s) to finish: " +
          describeActiveExecutions(active))

      var remainingNs = deadlineNs - System.nanoTime()
      while (active.nonEmpty && remainingNs > 0L) {
        val waitMs = math.max(1L, math.min(1000L, TimeUnit.NANOSECONDS.toMillis(remainingNs)))
        waitMonitor.wait(waitMs)
        active = activeEndedSqlExecutions
        remainingNs = deadlineNs - System.nanoTime()
      }

      if (active.nonEmpty) {
        logWarning(
          s"Timed out waiting for active SQL shuffle job(s) before JVM shutdown: " +
            describeActiveExecutions(active))
      } else {
        logInfo("Active SQL shuffle job(s) finished before JVM shutdown")
      }
    }
  }

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    val executionIdOpt = Option(jobStart.properties)
      .flatMap(p => Option(p.getProperty(SQLExecution.EXECUTION_ID_KEY)))
      .flatMap(s => scala.util.Try(s.toLong).toOption)

    executionIdOpt.foreach {
      executionId =>
        ensureShutdownHookRegistered()
        val shuffleIds = jobStart.stageInfos
          .flatMap(_.shuffleDepId)
          .toSet
        val state = stateFor(executionId)
        state.synchronized {
          state.activeJobs += jobStart.jobId
          state.shuffles ++= shuffleIds
        }
        jobExecutions.put(jobStart.jobId, executionId)
        notifyStateChanged()
    }
  }

  override def onJobEnd(jobEnd: SparkListenerJobEnd): Unit = {
    Option(jobExecutions.remove(jobEnd.jobId)).foreach {
      executionId =>
        Option(executionStates.get(executionId)).foreach {
          state =>
            state.synchronized {
              state.activeJobs -= jobEnd.jobId
            }
        }
        cleanupIfReady(executionId)
        notifyStateChanged()
    }
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case e: SparkListenerSQLExecutionEnd =>
        onSQLExecutionEnd(e)
      case _ =>
    }
  }

  private def onSQLExecutionEnd(event: SparkListenerSQLExecutionEnd): Unit = {
    ensureShutdownHookRegistered()
    val state = stateFor(event.executionId)
    state.synchronized {
      state.sqlEnded = true
    }
    cleanupIfReady(event.executionId)
    notifyStateChanged()
  }

  private def cleanupIfReady(executionId: Long): Unit = {
    val state = executionStates.get(executionId)
    if (state == null) {
      return
    }

    val ids = state.synchronized {
      if (state.sqlEnded && state.activeJobs.isEmpty) {
        if (executionStates.remove(executionId, state)) {
          state.shuffles.toSeq
        } else {
          Seq.empty[Int]
        }
      } else {
        if (state.sqlEnded && state.activeJobs.nonEmpty) {
          logInfo(
            s"SQL execution $executionId ended, deferring shuffle cleanup until " +
              s"${state.activeJobs.size} active job(s) finish: ${state.activeJobs.mkString(", ")}")
        }
        Seq.empty[Int]
      }
    }

    if (ids.nonEmpty && cleanupOnSqlEnd) {
      logInfo(
        s"SQL execution $executionId ended, cleaning ${ids.size} shuffle(s): ${ids.mkString(", ")}")
      ids.foreach {
        shuffleId =>
          try {
            onCleanup(shuffleId)
          } catch {
            case e: Exception =>
              logWarning(s"Failed to clean shuffle $shuffleId", e)
          }
      }
    } else if (ids.nonEmpty) {
      logInfo(
        s"SQL execution $executionId ended, keeping ${ids.size} shuffle(s) until " +
          s"shuffle manager or application cleanup: ${ids.mkString(", ")}")
    }
  }

  /** Override to customize cleanup action. */
  protected def onCleanup(shuffleId: Int): Unit = {
    val env = SparkEnv.get
    if (env != null) {
      try {
        env.blockManager.master
          .removeShuffle(shuffleId, false)
      } catch {
        case e: Exception =>
          logWarning(
            s"removeShuffle via BlockManagerMaster " +
              s"failed for $shuffleId, " +
              s"falling back to direct JNI",
            e)
          ShufflePayloadCatalogJniWrapper
            .unregisterShuffle(shuffleId)
      }
    } else {
      ShufflePayloadCatalogJniWrapper
        .unregisterShuffle(shuffleId)
    }
  }
}
