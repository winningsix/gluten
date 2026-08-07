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

import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler.{SparkListener, SparkListenerExecutorRemoved, SparkListenerTaskEnd}

import java.util.concurrent.{Executors, ScheduledExecutorService, ThreadFactory, TimeUnit}

import scala.util.control.NonFatal

/**
 * Driver-owned resources that exist only when FLUX query cancellation is enabled.
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
    registry: GlutenFluxQueryControlRegistry,
    heartbeatMs: Long,
    startSweeper: Boolean)

/**
 * Driver-side dispatcher for the two FLUX control-plane responsibilities.
 *
 * Endpoint registration is executor-scoped and populates [[GlutenFluxEndpointRegistry]]. Query
 * cancellation is run-scoped and, when enabled, delegates heartbeats and scheduler failure signals
 * to [[GlutenFluxQueryControlRegistry]]. Keeping the two paths independent allows either feature to
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
class GlutenFluxDriverService private[control] (
    private val endpointRegistryRef: GlutenFluxEndpointRegistry,
    private val endpointRegistryEnabled: Boolean,
    private val queryControl: Option[QueryControlRuntime])
  extends Logging {

  private var sweeper: ScheduledExecutorService = _

  queryControl.filter(_.startSweeper).foreach {
    control =>
      sweeper = Executors.newSingleThreadScheduledExecutor(new ThreadFactory {
        override def newThread(r: Runnable): Thread = {
          val thread = new Thread(r, "gluten-flux-query-control-sweeper")
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
              case NonFatal(e) => logWarning("FLUX query-control lease sweep failed", e)
            }
          }
        },
        control.heartbeatMs,
        control.heartbeatMs,
        TimeUnit.MILLISECONDS
      )
  }

  def endpointRegistry: GlutenFluxEndpointRegistry = endpointRegistryRef

  private[control] def queryControlRegistry: Option[GlutenFluxQueryControlRegistry] =
    queryControl.map(_.registry)

  def receive(msg: Any): AnyRef = {
    msg match {
      case RegisterEndpoint(record) if endpointRegistryEnabled =>
        val generation = endpointRegistryRef.register(record)
        logInfo(
          s"GlutenFluxDriverService: registered endpoint executorId=${record.executorId} " +
            s"generation=$generation")
        RegisterEndpointAck(generation)
      case DeregisterEndpoint(executorId) if endpointRegistryEnabled =>
        val generation = endpointRegistryRef.deregister(executorId)
        logInfo(
          s"GlutenFluxDriverService: deregistered endpoint executorId=$executorId " +
            s"generation=$generation")
        DeregisterEndpointAck(generation)
      case heartbeat: FluxQueryHeartbeat =>
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
object GlutenFluxDriverService extends Logging {
  @volatile private var instance: Option[GlutenFluxDriverService] = None

  def init(conf: SparkConf): Unit = synchronized {
    val endpointEnabled = GlutenFluxControlPlaneConfig.endpointRegistryEnabled(conf)
    val queryEnabled = GlutenFluxControlPlaneConfig.queryCancellationEnabled(conf)
    if (!endpointEnabled && !queryEnabled) {
      instance.foreach(_.close())
      instance = None
      logInfo("GlutenFluxDriverService: FLUX control plane disabled")
      return
    }
    instance.foreach(_.close())
    val queryControl =
      if (queryEnabled) {
        Some(
          QueryControlRuntime(
            new GlutenFluxQueryControlRegistry(GlutenFluxControlPlaneConfig.peerTimeoutMs(conf)),
            GlutenFluxControlPlaneConfig.heartbeatMs(conf),
            startSweeper = true))
      } else {
        None
      }
    instance =
      Some(new GlutenFluxDriverService(GlutenFluxEndpointRegistry(), endpointEnabled, queryControl))
    logInfo(
      s"GlutenFluxDriverService initialized: endpointRegistry=$endpointEnabled " +
        s"queryCancellation=$queryEnabled")
  }

  def get(): Option[GlutenFluxDriverService] = instance

  def shutdown(): Unit = synchronized {
    instance.foreach(_.close())
    instance = None
  }

  private[control] def createForTests(
      registry: GlutenFluxEndpointRegistry,
      enabled: Boolean): GlutenFluxDriverService =
    new GlutenFluxDriverService(
      registry,
      enabled,
      if (enabled) {
        Some(QueryControlRuntime(new GlutenFluxQueryControlRegistry(5000L), 250L, false))
      } else {
        None
      })

  private[control] def createForTests(
      registry: GlutenFluxEndpointRegistry,
      queryRegistry: GlutenFluxQueryControlRegistry,
      endpointEnabled: Boolean,
      queryEnabled: Boolean): GlutenFluxDriverService =
    new GlutenFluxDriverService(
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
class GlutenFluxQueryControlListener(service: GlutenFluxDriverService) extends SparkListener {
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
