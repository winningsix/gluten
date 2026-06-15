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

import org.apache.gluten.backendsapi.BackendsApiManager
import org.apache.gluten.init.NativeBackendInitializer

import org.apache.spark.SparkEnv
import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.internal.Logging

import java.net.{Inet4Address, InetAddress, NetworkInterface, URI}

import scala.collection.JavaConverters._
import scala.util.control.NonFatal

object GlutenMppExecutorService extends Logging {
  @volatile private var registeredContext: Option[PluginContext] = None

  def onExecutorStart(ctx: PluginContext): Unit = {
    if (!GlutenMppControlPlaneConfig.endpointRegistryEnabled(ctx.conf())) {
      logInfo("GlutenMppExecutorService: endpoint registry disabled; skip register")
      return
    }
    try {
      buildEndpointRecord(ctx) match {
        case Some(record) =>
          ctx.ask(RegisterEndpoint(record)) match {
            case ack: RegisterEndpointAck =>
              registeredContext = Some(ctx)
              logInfo(
                s"GlutenMppExecutorService: registered endpoint executorId=${record.executorId} " +
                  s"generation=${ack.generation} endpoint=${record.ucxEndpoint}")
            case other =>
              logWarning(s"GlutenMppExecutorService: unexpected registration ack: $other")
          }
        case None =>
          logWarning("GlutenMppExecutorService: UCX listener endpoint unavailable; skip register")
      }
    } catch {
      case NonFatal(e) =>
        logWarning("GlutenMppExecutorService: failed to register UCX endpoint", e)
    }
  }

  def onExecutorShutdown(): Unit = {
    val context = registeredContext
    registeredContext = None
    context.foreach {
      ctx =>
        try {
          val executorId = executorIdOf(ctx)
          ctx.ask(DeregisterEndpoint(executorId)) match {
            case ack: DeregisterEndpointAck =>
              logInfo(
                s"GlutenMppExecutorService: deregistered endpoint executorId=$executorId " +
                  s"generation=${ack.generation}")
            case other =>
              logWarning(s"GlutenMppExecutorService: unexpected deregistration ack: $other")
          }
        } catch {
          case NonFatal(e) =>
            logDebug("GlutenMppExecutorService: failed to deregister UCX endpoint", e)
        }
    }
  }

  private[control] def buildEndpointRecord(
      ctx: PluginContext): Option[MppExecutorEndpointRecord] = {
    val blockManagerId =
      Option(SparkEnv.get).flatMap(e => Option(e.blockManager)).map(_.blockManagerId)
    val blockManagerHost = blockManagerId.map(_.host).getOrElse(localHostName())
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
      registeredAtMs: Long): MppExecutorEndpointRecord = {
    val endpoint = new URI(nativeUcxListenerEndpoint)
    val listenerPort = endpoint.getPort
    if (listenerPort <= 3) {
      throw new IllegalStateException(
        s"Invalid UCX listener endpoint for executor $executorId: $endpoint")
    }
    MppExecutorEndpointRecord(
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
      state = MppExecutorEndpointRecord.StateLive
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
