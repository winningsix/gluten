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

import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler.{SparkListener, SparkListenerExecutorRemoved, SparkListenerTaskEnd}

import java.util.concurrent.{Executors, ScheduledExecutorService, ThreadFactory, TimeUnit}

import scala.util.control.NonFatal

/**
 * Driver-owned resources that exist only when MPP query cancellation is enabled.
 *
 * Keeping these values together makes optional query cancellation independent from endpoint
 * discovery. Production starts the lease sweeper; tests can reuse the state machine without
 * creating a background thread.
 *
 * @param registry
 *   Run-scoped peer state and abort-command registry.
 * @param heartbeatMs
 *   Interval used by the driver lease sweeper.
 * @param startSweeper
 *   Whether this service instance owns a background lease-sweeper thread.
 */
final private[control] case class QueryControlRuntime(
    registry: GlutenMppQueryControlRegistry,
    heartbeatMs: Long,
    startSweeper: Boolean)

/**
 * Driver-side dispatcher for the two MPP control-plane responsibilities.
 *
 * Endpoint registration is executor-scoped and populates [[GlutenMppEndpointRegistry]]. Query
 * cancellation is run-scoped and, when enabled, delegates heartbeats and scheduler failure signals
 * to [[GlutenMppQueryControlRegistry]]. Keeping the two paths independent allows either feature to
 * be disabled without changing the other's lifetime.
 *
 * The service also owns the query lease sweeper. Executor RPC remains pull-based: abort commands
 * are returned from [[receive]] in response to executor heartbeats.
 *
 * @param endpointRegistryRef
 *   Registry used for UCX endpoint discovery.
 * @param endpointRegistryEnabled
 *   Whether endpoint register and deregister messages are accepted.
 * @param queryControl
 *   Optional query-cancellation state and sweeper configuration.
 */
class GlutenMppDriverService private[control] (
    private val endpointRegistryRef: GlutenMppEndpointRegistry,
    private val endpointRegistryEnabled: Boolean,
    private val queryControl: Option[QueryControlRuntime])
  extends Logging {

  private var sweeper: ScheduledExecutorService = _

  queryControl.filter(_.startSweeper).foreach {
    control =>
      sweeper = Executors.newSingleThreadScheduledExecutor(new ThreadFactory {
        override def newThread(r: Runnable): Thread = {
          val thread = new Thread(r, "gluten-mpp-query-control-sweeper")
          thread.setDaemon(true)
          thread
        }
      })
      sweeper.scheduleWithFixedDelay(
        new Runnable {
          override def run(): Unit = {
            try {
              control.registry.expirePeers()
            } catch {
              case NonFatal(e) => logWarning("MPP query-control lease sweep failed", e)
            }
          }
        },
        control.heartbeatMs,
        control.heartbeatMs,
        TimeUnit.MILLISECONDS
      )
  }

  def endpointRegistry: GlutenMppEndpointRegistry = endpointRegistryRef

  private[control] def queryControlRegistry: Option[GlutenMppQueryControlRegistry] =
    queryControl.map(_.registry)

  def receive(msg: Any): AnyRef = {
    msg match {
      case RegisterEndpoint(record) if endpointRegistryEnabled =>
        val generation = endpointRegistryRef.register(record)
        logInfo(
          s"GlutenMppDriverService: registered endpoint executorId=${record.executorId} " +
            s"generation=$generation")
        RegisterEndpointAck(generation)
      case DeregisterEndpoint(executorId) if endpointRegistryEnabled =>
        val generation = endpointRegistryRef.deregister(executorId)
        logInfo(
          s"GlutenMppDriverService: deregistered endpoint executorId=$executorId " +
            s"generation=$generation")
        DeregisterEndpointAck(generation)
      case heartbeat: MppQueryHeartbeat =>
        queryControl.fold[AnyRef](null)(_.registry.processHeartbeat(heartbeat))
      case _ =>
        null
    }
  }

  def onExecutorRemoved(executorId: String, reason: String): Unit = {
    if (endpointRegistryEnabled) {
      endpointRegistryRef.deregister(executorId)
    }
    queryControl.foreach(_.registry.onExecutorRemoved(executorId, reason))
  }

  def onTaskEnd(taskAttemptId: Long, executorId: String, failed: Boolean, reason: String): Unit = {
    queryControl.foreach(_.registry.onTaskEnd(taskAttemptId, executorId, failed, reason))
  }

  private[control] def close(): Unit = {
    if (sweeper != null) {
      sweeper.shutdownNow()
      sweeper = null
    }
  }
}

/** Process-wide lifecycle for the driver service created by the Spark driver plugin. */
object GlutenMppDriverService extends Logging {
  @volatile private var instance: Option[GlutenMppDriverService] = None

  def init(conf: SparkConf): Unit = synchronized {
    val endpointEnabled = GlutenMppControlPlaneConfig.endpointRegistryEnabled(conf)
    val queryEnabled = GlutenMppControlPlaneConfig.queryCancellationEnabled(conf)
    if (!endpointEnabled && !queryEnabled) {
      instance.foreach(_.close())
      instance = None
      logInfo("GlutenMppDriverService: MPP control plane disabled")
      return
    }
    instance.foreach(_.close())
    val queryControl =
      if (queryEnabled) {
        Some(
          QueryControlRuntime(
            new GlutenMppQueryControlRegistry(GlutenMppControlPlaneConfig.peerTimeoutMs(conf)),
            GlutenMppControlPlaneConfig.heartbeatMs(conf),
            startSweeper = true))
      } else {
        None
      }
    instance =
      Some(new GlutenMppDriverService(GlutenMppEndpointRegistry(), endpointEnabled, queryControl))
    logInfo(
      s"GlutenMppDriverService initialized: endpointRegistry=$endpointEnabled " +
        s"queryCancellation=$queryEnabled")
  }

  def get(): Option[GlutenMppDriverService] = instance

  def shutdown(): Unit = synchronized {
    instance.foreach(_.close())
    instance = None
  }

  private[control] def createForTests(
      registry: GlutenMppEndpointRegistry,
      enabled: Boolean): GlutenMppDriverService =
    new GlutenMppDriverService(
      registry,
      enabled,
      if (enabled) {
        Some(QueryControlRuntime(new GlutenMppQueryControlRegistry(5000L), 250L, false))
      } else {
        None
      })

  private[control] def createForTests(
      registry: GlutenMppEndpointRegistry,
      queryRegistry: GlutenMppQueryControlRegistry,
      endpointEnabled: Boolean,
      queryEnabled: Boolean): GlutenMppDriverService =
    new GlutenMppDriverService(
      registry,
      endpointEnabled,
      if (queryEnabled) {
        Some(QueryControlRuntime(queryRegistry, 250L, false))
      } else {
        None
      })
}

/**
 * Spark scheduler fallbacks for failures that may arrive before or without an executor heartbeat.
 *
 * Executor removal immediately fails affected runs. Task completion is a secondary signal for peers
 * that cannot report their own terminal event.
 */
class GlutenMppQueryControlListener(service: GlutenMppDriverService) extends SparkListener {
  override def onExecutorRemoved(event: SparkListenerExecutorRemoved): Unit = {
    service.onExecutorRemoved(event.executorId, event.reason)
  }

  override def onTaskEnd(event: SparkListenerTaskEnd): Unit = {
    service.onTaskEnd(
      event.taskInfo.taskId,
      event.taskInfo.executorId,
      event.reason != org.apache.spark.Success,
      event.reason.toString)
  }
}
