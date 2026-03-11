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

import org.apache.spark.internal.Logging
import org.apache.spark.scheduler._
import org.apache.spark.sql.execution.SQLExecution
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd

import java.util.concurrent.ConcurrentHashMap

import scala.collection.mutable

/**
 * SparkListener that tracks shuffle-to-execution mapping and triggers cleanup when SQL executions
 * complete. Subclass and override onCleanup() to customize the cleanup action.
 */
class CatalogCleanupListener extends SparkListener with Logging {

  private val executionShuffles =
    new ConcurrentHashMap[Long, mutable.Set[Int]]()

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    val executionIdOpt = Option(jobStart.properties)
      .flatMap(p => Option(p.getProperty(SQLExecution.EXECUTION_ID_KEY)))
      .flatMap(s => scala.util.Try(s.toLong).toOption)

    executionIdOpt.foreach {
      executionId =>
        val shuffleIds = jobStart.stageInfos
          .flatMap(_.shuffleDepId)
          .toSet

        if (shuffleIds.nonEmpty) {
          executionShuffles.compute(
            executionId,
            (_, existing) => {
              val set =
                if (existing == null) mutable.Set[Int]()
                else existing
              set ++= shuffleIds
              set
            })
        }
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
    Option(executionShuffles.remove(event.executionId)).foreach {
      ids =>
        if (ids.nonEmpty) {
          logInfo(
            s"SQL execution ${event.executionId} " +
              s"ended, cleaning ${ids.size} " +
              s"shuffle(s): ${ids.mkString(", ")}")
          ids.foreach {
            shuffleId =>
              try {
                onCleanup(shuffleId)
              } catch {
                case e: Exception =>
                  logWarning(
                    s"Failed to clean shuffle " +
                      s"$shuffleId",
                    e)
              }
          }
        }
    }
  }

  /** Override to customize cleanup action. */
  protected def onCleanup(shuffleId: Int): Unit = {
    // Default: directly call JNI (local mode).
    // In standalone mode, the CatalogCleanupPlugin
    // overrides this to use the polling mechanism.
    org.apache.gluten.vectorized.ShufflePayloadCatalogJniWrapper
      .unregisterShuffle(shuffleId)
  }
}
