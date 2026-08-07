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

import org.apache.gluten.backendsapi.BackendsApiManager
import org.apache.gluten.init.NativeBackendInitializer

import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.internal.Logging

import java.net.{Inet4Address, InetAddress, NetworkInterface, URI}

import scala.collection.JavaConverters._
import scala.util.control.NonFatal

/**
 * Executor-plugin entry point for endpoint discovery and FLUX query cancellation.
 *
 * Endpoint registration and query cancellation are initialized independently. In particular, a
 * missing or failed UCX endpoint registration must not disable the local interruption watcher or
 * peer-failure propagation. Query lifecycle calls delegate to the process-wide
 * [[GlutenFluxExecutorControlAgent]] created for the current executor session.
 */
object GlutenFluxExecutorService extends Logging {
  @volatile private var registeredContext: Option[PluginContext] = None
  @volatile private var registeredEndpoint: Option[FluxExecutorEndpointRecord] = None
  @volatile private var controlAgent: Option[GlutenFluxExecutorControlAgent] = None

  def onExecutorStart(ctx: PluginContext): Unit = {
    controlAgent.foreach(_.shutdown())
    controlAgent = None
    registeredContext = None
    registeredEndpoint = None
    if (GlutenFluxControlPlaneConfig.queryCancellationEnabled(ctx.conf())) {
      val agent = GlutenFluxExecutorControlAgent(
        executorIdOf(ctx),
        msg => ctx.ask(msg),
        GlutenFluxControlPlaneConfig.interruptPollMs(ctx.conf()),
        GlutenFluxControlPlaneConfig.heartbeatMs(ctx.conf()))
      controlAgent = Some(agent)
      logInfo(
        s"GlutenFluxExecutorService: query cancellation initialized " +
          s"executorId=${executorIdOf(ctx)} session=${agent.executorSessionId}")
    }
    if (!GlutenFluxControlPlaneConfig.endpointRegistryEnabled(ctx.conf())) {
      logInfo("GlutenFluxExecutorService: endpoint registry disabled; skip register")
      return
    }
    try {
      buildEndpointRecord(ctx) match {
        case Some(record) =>
          ctx.ask(RegisterEndpoint(record)) match {
            case ack: RegisterEndpointAck =>
              registeredContext = Some(ctx)
              registeredEndpoint = Some(record)
              logInfo(
                s"GlutenFluxExecutorService: registered endpoint executorId=${record.executorId} " +
                  s"generation=${ack.generation} endpoint=${record.ucxEndpoint}")
            case other =>
              logWarning(s"GlutenFluxExecutorService: unexpected registration ack: $other")
          }
        case None =>
          logWarning("GlutenFluxExecutorService: UCX listener endpoint unavailable; skip register")
      }
    } catch {
      case NonFatal(e) =>
        logWarning("GlutenFluxExecutorService: failed to register UCX endpoint", e)
    }
  }

  def onExecutorShutdown(): Unit = {
    val agent = controlAgent
    controlAgent = None
    agent.foreach(_.shutdown())
    val context = registeredContext
    registeredContext = None
    registeredEndpoint = None
    context.foreach {
      ctx =>
        try {
          val executorId = executorIdOf(ctx)
          ctx.ask(DeregisterEndpoint(executorId)) match {
            case ack: DeregisterEndpointAck =>
              logInfo(
                s"GlutenFluxExecutorService: deregistered endpoint executorId=$executorId " +
                  s"generation=${ack.generation}")
            case other =>
              logWarning(s"GlutenFluxExecutorService: unexpected deregistration ack: $other")
          }
        } catch {
          case NonFatal(e) =>
            logDebug("GlutenFluxExecutorService: failed to deregister UCX endpoint", e)
        }
    }
  }

  /** Register a peer immediately after native create and before native start. */
  def registerQuery(
      runId: FluxQueryRunId,
      peerIndex: Int,
      expectedPeerCount: Int,
      taskContext: TaskContext,
      abortNative: () => Unit): Option[ActiveFluxQuery] =
    controlAgent.map(_.register(runId, peerIndex, expectedPeerCount, taskContext, abortNative))

  /** Retain the first local native failure until the driver acknowledges its event ID. */
  def reportFailure(query: ActiveFluxQuery, error: Throwable): Unit =
    controlAgent.foreach(_.reportFailure(query, error))

  /** Report task-thread cleanup separately from abort-command acceptance. */
  def reportTerminal(query: ActiveFluxQuery, state: String): Unit =
    controlAgent.foreach(_.reportTerminal(query, state))

  /** Keep the native coordinator alive until every peer reports output EOS. */
  def awaitPeerCompletion(query: ActiveFluxQuery): Boolean =
    controlAgent.exists(_.awaitPeerCompletion(query))

  /**
   * Return this executor's routable UCX endpoint for an executor-local, one-peer FLUX query.
   *
   * Such queries do not use the driver-side multi-peer resolver, but their native fragments can
   * still contain an exchange (for example Spark runtime-bloom partial -> final aggregation). An
   * empty peer list makes native code fall back to 127.0.0.1 even though the UCX listener is bound
   * to the advertised interface. Reuse the endpoint that this executor already registered.
   */
  def localPeerEndpointsJson(executorId: String): Option[String] =
    registeredEndpoint.flatMap(localPeerEndpointsJson(_, executorId))

  private[control] def localPeerEndpointsJson(
      record: FluxExecutorEndpointRecord,
      executorId: String): Option[String] = {
    Option(executorId)
      .filter(_.nonEmpty)
      .filter(_ == record.executorId)
      .map(
        _ =>
          GlutenFluxPeerMapper.toPeerEndpointsJson(
            GlutenFluxPeerMapper.fromEndpointRecords(Seq(record), requestedCount = 1)))
  }

  private[control] def buildEndpointRecord(
      ctx: PluginContext): Option[FluxExecutorEndpointRecord] = {
    val blockManagerId =
      Option(SparkEnv.get)
        .flatMap(e => Option(e.blockManager))
        .flatMap(bm => Option(bm.blockManagerId))
    val blockManagerHost = blockManagerId
      .flatMap(id => Option(id.host))
      .filter(_.nonEmpty)
      .getOrElse(localHostName())
    val blockManagerPort = blockManagerId.map(_.port).getOrElse(-1)
    val advertisedHost = localHostName(blockManagerHost)
    val initializer = NativeBackendInitializer.forBackend(BackendsApiManager.getBackendName)
    val endpoint = Option(initializer.getUcxListenerEndpoint(advertisedHost)).filter(_.nonEmpty)
    endpoint.map {
      rawEndpoint =>
        buildEndpointRecord(
          executorId = executorIdOf(ctx),
          host = advertisedHost,
          blockManagerHost = blockManagerHost,
          blockManagerPort = blockManagerPort,
          gpuResourceAddresses = gpuResourceAddresses(ctx),
          nativeUcxListenerEndpoint = rawEndpoint,
          registeredAtMs = System.currentTimeMillis()
        )
    }
  }

  private[control] def buildEndpointRecord(
      executorId: String,
      host: String,
      blockManagerHost: String,
      blockManagerPort: Int,
      gpuResourceAddresses: Seq[String],
      nativeUcxListenerEndpoint: String,
      registeredAtMs: Long): FluxExecutorEndpointRecord = {
    val endpoint = new URI(nativeUcxListenerEndpoint)
    val listenerPort = endpoint.getPort
    if (listenerPort <= 3) {
      throw new IllegalStateException(
        s"Invalid UCX listener endpoint for executor $executorId: $endpoint")
    }
    FluxExecutorEndpointRecord(
      executorId = executorId,
      host = host,
      blockManagerHost = blockManagerHost,
      blockManagerPort = blockManagerPort,
      ucxListenerPort = listenerPort,
      remoteConnectorPort = listenerPort - 3,
      ucxEndpoint = nativeUcxListenerEndpoint,
      gpuResourceAddresses = gpuResourceAddresses,
      registeredAtMs = registeredAtMs,
      generation = 0L,
      state = FluxExecutorEndpointRecord.StateLive
    )
  }

  private def executorIdOf(ctx: PluginContext): String =
    Option(ctx.executorID()).filter(_.nonEmpty).getOrElse("unknown")

  private def gpuResourceAddresses(ctx: PluginContext): Seq[String] =
    Option(ctx.resources())
      .flatMap(resources => Option(resources.get("gpu")))
      .map(_.addresses.toSeq)
      .getOrElse(Seq.empty)

  private def localHostName(fallback: String = "unknown"): String = {
    def nonLoopbackIPv4: Option[String] = {
      NetworkInterface.getNetworkInterfaces.asScala
        .filter {
          iface =>
            val name = iface.getName
            iface.isUp &&
            !iface.isLoopback &&
            !name.startsWith("docker") &&
            !name.startsWith("br-") &&
            !name.startsWith("veth")
        }
        .flatMap(_.getInetAddresses.asScala)
        .collectFirst {
          case address: Inet4Address if !address.isLoopbackAddress => address.getHostAddress
        }
    }

    try {
      nonLoopbackIPv4
        .orElse(Option(InetAddress.getLocalHost.getHostAddress).filter(_.nonEmpty))
        .getOrElse(fallback)
    } catch {
      case NonFatal(_) => fallback
    }
  }
}
