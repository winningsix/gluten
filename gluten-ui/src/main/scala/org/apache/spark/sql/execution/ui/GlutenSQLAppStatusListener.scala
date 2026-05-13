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
package org.apache.spark.sql.execution.ui

import org.apache.gluten.events.{GlutenBuildInfoEvent, GlutenMppPlanEvent, GlutenPlanFallbackEvent}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler.{SparkListener, SparkListenerEvent}
import org.apache.spark.sql.internal.StaticSQLConf.UI_RETAINED_EXECUTIONS
import org.apache.spark.status.{ElementTrackingStore, KVUtils}

import scala.collection.mutable

private class GlutenSQLAppStatusListener(conf: SparkConf, kvstore: ElementTrackingStore)
  extends SparkListener
  with Logging {
  private val executionIdToDescription = new mutable.HashMap[Long, String]
  private val executionIdToFallbackEvent = new mutable.HashMap[Long, GlutenPlanFallbackEvent]
  private val executionIdToMppPlanEvent = new mutable.HashMap[Long, GlutenMppPlanEvent]

  kvstore.addTrigger(classOf[GlutenSQLExecutionUIData], conf.get[Int](UI_RETAINED_EXECUTIONS)) {
    count => cleanupExecutions(count)
  }

  private def onGlutenBuildInfo(event: GlutenBuildInfoEvent): Unit = {
    val uiData = new GlutenBuildInfoUIData(event.info.toSeq.sortBy(_._1))
    kvstore.write(uiData)
  }

  private def onGlutenPlanFallback(event: GlutenPlanFallbackEvent): Unit = {
    val description = executionIdToDescription.get(event.executionId)
    if (description.isDefined) {
      writeExecution(event.executionId, description.get, Some(event), None)
    } else {
      // the first stage applies rule before post `SparkListenerSQLExecutionStart`,
      // so we should wait `SparkListenerSQLExecutionStart` then write to store.
      executionIdToFallbackEvent.put(event.executionId, event.copy())
    }
  }

  private def onGlutenMppPlan(event: GlutenMppPlanEvent): Unit = {
    val description = executionIdToDescription.get(event.executionId)
    if (description.isDefined) {
      writeExecution(event.executionId, description.get, None, Some(event))
    } else {
      executionIdToMppPlanEvent.put(event.executionId, event.copy())
    }
  }

  private def onSQLExecutionStart(event: SparkListenerSQLExecutionStart): Unit = {
    val fallbackEvent = executionIdToFallbackEvent.get(event.executionId)
    val mppPlanEvent = executionIdToMppPlanEvent.get(event.executionId)
    if (fallbackEvent.isDefined || mppPlanEvent.isDefined) {
      writeExecution(event.executionId, event.description, fallbackEvent, mppPlanEvent)
      fallbackEvent.foreach(_ => executionIdToFallbackEvent.remove(event.executionId))
      mppPlanEvent.foreach(_ => executionIdToMppPlanEvent.remove(event.executionId))
    }
    executionIdToDescription.put(event.executionId, event.description)
  }

  private def onSQLExecutionEnd(event: SparkListenerSQLExecutionEnd): Unit = {
    executionIdToDescription.remove(event.executionId)
    executionIdToFallbackEvent.remove(event.executionId)
    executionIdToMppPlanEvent.remove(event.executionId)
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = event match {
    case e: SparkListenerSQLExecutionStart => onSQLExecutionStart(e)
    case e: SparkListenerSQLExecutionEnd => onSQLExecutionEnd(e)
    case e: GlutenBuildInfoEvent => onGlutenBuildInfo(e)
    case e: GlutenPlanFallbackEvent => onGlutenPlanFallback(e)
    case e: GlutenMppPlanEvent => onGlutenMppPlan(e)
    case _ => // Ignore
  }

  private def writeExecution(
      executionId: Long,
      description: String,
      fallbackEvent: Option[GlutenPlanFallbackEvent],
      mppPlanEvent: Option[GlutenMppPlanEvent]): Unit = {
    val existing = readExecution(executionId)
    val fallback = fallbackEvent
      .map(ExistingFallbackData.from)
      .orElse(existing.map(ExistingFallbackData.from))
    val mppPlan = mppPlanEvent.map(toMppPlanUIData).orElse(existing.map(_.mppPlan).filter(_ != null))

    val uiData = new GlutenSQLExecutionUIData(
      executionId,
      description,
      fallback.map(_.numGlutenNodes).getOrElse(0),
      fallback.map(_.numFallbackNodes).getOrElse(0),
      fallback.map(_.physicalPlanDescription).getOrElse(""),
      fallback.map(_.fallbackNodeToReason.toSeq.sortBy(_._1)).getOrElse(Seq.empty),
      mppPlan.orNull
    )
    kvstore.write(uiData, checkTriggers = true)
  }

  private def readExecution(executionId: Long): Option[GlutenSQLExecutionUIData] = {
    try {
      Some(kvstore.read(classOf[GlutenSQLExecutionUIData], executionId))
    } catch {
      case _: NoSuchElementException => None
    }
  }

  private def toMppPlanUIData(event: GlutenMppPlanEvent): GlutenMppPlanUIData = {
    new GlutenMppPlanUIData(
      event.queryId,
      event.numFragments,
      event.numExchanges,
      event.dumpPath,
      event.totalOriginalCharCount,
      event.planSha256,
      event.truncated,
      event.captureEnabled,
      event.captureError,
      event.fragments
        .map {
          fragment =>
            new GlutenMppPlanFragmentUIData(
              fragment.fragmentId,
              fragment.plan,
              fragment.originalCharCount,
              fragment.sha256,
              fragment.truncated)
        }
        .sortBy(_.fragmentId)
    )
  }

  private def cleanupExecutions(count: Long): Unit = {
    val countToDelete = count - conf.get(UI_RETAINED_EXECUTIONS)
    if (countToDelete <= 0) {
      return
    }

    val view = kvstore.view(classOf[GlutenSQLExecutionUIData]).first(0L)
    val toDelete = KVUtils.viewToSeq(view, countToDelete.toInt)(_ => true)
    toDelete.foreach(e => kvstore.delete(e.getClass, e.executionId))
  }

  private case class ExistingFallbackData(
      executionId: Long,
      numGlutenNodes: Int,
      numFallbackNodes: Int,
      physicalPlanDescription: String,
      fallbackNodeToReason: Map[String, String])

  private object ExistingFallbackData {
    def from(event: GlutenPlanFallbackEvent): ExistingFallbackData = {
      ExistingFallbackData(
        event.executionId,
        event.numGlutenNodes,
        event.numFallbackNodes,
        event.physicalPlanDescription,
        event.fallbackNodeToReason)
    }

    def from(data: GlutenSQLExecutionUIData): ExistingFallbackData = {
      ExistingFallbackData(
        data.executionId,
        data.numGlutenNodes,
        data.numFallbackNodes,
        data.fallbackDescription,
        data.fallbackNodeToReason.toMap)
    }
  }
}

object GlutenSQLAppStatusListener {
  def register(sc: SparkContext): Unit = {
    val kvStore = sc.statusStore.store.asInstanceOf[ElementTrackingStore]
    val listener = new GlutenSQLAppStatusListener(sc.conf, kvStore)
    sc.listenerBus.addToStatusQueue(listener)
  }
}
