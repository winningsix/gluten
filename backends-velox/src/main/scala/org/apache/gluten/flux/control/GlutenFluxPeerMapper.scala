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

import org.apache.gluten.execution.{FluxPeerInfo, UcxEndpointInfo}

import com.google.common.net.InetAddresses

import java.net.URI

/**
 * Shared `UcxEndpointInfo` -> `FluxPeerInfo` mapping.
 *
 * Behaviour MUST match the existing inline mapping in `FluxNativeQueryExec.fluxPeerInfos()` (lines
 * 1909-1927 as of velox HEAD a482411a0b) byte-for-byte:
 *
 *   - take(requestedCount) of sortBy(executorId)-ordered endpoints
 *   - parse `nativeUcxListenerEndpoint` URI to extract listenerPort
 *   - reject listenerPort <= 3
 *   - UCX connection host selection: a non-loopback literal blockManagerHost takes precedence over
 *     the listener URI host, which then falls through to blockManagerHost and `info.host`
 *   - `port = listenerPort - 3` (this is the "remote connector port" native convention)
 *   - Spark placement remains `executor_<blockManagerHost>_<executorId>` so executor affinity is
 *     independent of the network address advertised to UCX
 *
 * JSON serialisation (toPeerEndpointsJson) produces the contract consumed by the native JNI side:
 *
 * [{"peerId":"<ESCAPED>","host":"<ESCAPED>","port":N,"peerIndex":I}, ...]
 */
object GlutenFluxPeerMapper {

  /**
   * Map a sorted prefix of discovered endpoints into `FluxPeerInfo` records. The caller is expected
   * to sort `infos` by `executorId` and pass `infos.length` (i.e. no truncation) or a smaller count
   * for `requestedCount` to match the original truncation semantics.
   */
  def toFluxPeerInfos(infos: Seq[UcxEndpointInfo], requestedCount: Int): Seq[FluxPeerInfo] = {
    require(requestedCount >= 0, s"requestedCount=$requestedCount must be non-negative")
    val selectedInfos = infos.take(requestedCount)
    // Spark standalone workers started on loopback can host multiple executors in separate JVMs.
    // Their UCX listeners advertise the same physical NIC address with different ports, even
    // though every listener accepts loopback connections. Keep that traffic on loopback: using
    // the physical address forces a fresh set of UCX/TCP wire-up connections through the NIC for
    // each query. A one-peer query is executor-local by construction, so it must use the same
    // loopback address too; that lets scalar/broadcast stages reuse the endpoint established by a
    // preceding multi-peer stage. Do not apply this to records with different listener hosts,
    // since those may be executors on different machines whose loopback addresses are not mutually
    // reachable.
    val useSharedLoopback = selectedInfos.nonEmpty &&
      selectedInfos.forall(info => Option(info.blockManagerHost).exists(isLoopbackIpLiteral)) &&
      selectedInfos
        .flatMap(_.nativeUcxListenerEndpoint)
        .map(endpoint => Option(new URI(endpoint).getHost).getOrElse(""))
        .filter(_.nonEmpty)
        .distinct
        .size == 1

    selectedInfos.map {
      info =>
        val rawEndpoint = info.nativeUcxListenerEndpoint
          .getOrElse(
            throw new IllegalStateException(
              s"Executor ${info.executorId} has no nativeUcxListenerEndpoint"))
        val endpoint = new URI(rawEndpoint)
        val listenerPort = endpoint.getPort
        if (listenerPort <= 3) {
          throw new IllegalStateException(
            s"Invalid UCX listener endpoint for executor ${info.executorId}: $endpoint")
        }
        val endpointHost = Option(endpoint.getHost).filter(_.nonEmpty)
        val blockManagerHost = Option(info.blockManagerHost)
          .filter(_.nonEmpty)
        // A literal BlockManager address identifies the interface selected by Spark and
        // UCX_NET_DEVICES. Prefer it when Java's interface enumeration advertised a different NIC.
        // For hostnames and loopback placement, retain the native listener address to avoid UCXX
        // hostname parsing failures under concurrent TableWrite startup.
        val blockManagerIp = blockManagerHost.filter(isNonLoopbackIpLiteral)
        val connectionHost =
          if (useSharedLoopback) {
            blockManagerHost.get
          } else {
            blockManagerIp.orElse(endpointHost).getOrElse(blockManagerHost.getOrElse(info.host))
          }
        val placementHost = blockManagerHost.getOrElse(info.host)
        FluxPeerInfo(
          peerId = info.executorId,
          host = connectionHost,
          port = listenerPort - 3,
          preferredLocation = s"executor_${placementHost}_${info.executorId}")
    }
  }

  private def isNonLoopbackIpLiteral(host: String): Boolean = {
    if (!InetAddresses.isInetAddress(host)) {
      false
    } else {
      val address = InetAddresses.forString(host)
      !address.isAnyLocalAddress && !address.isLoopbackAddress
    }
  }

  private def isLoopbackIpLiteral(host: String): Boolean =
    InetAddresses.isInetAddress(host) && InetAddresses.forString(host).isLoopbackAddress

  /** Map registry endpoint records through the same contract as probe-discovered endpoint info. */
  def fromEndpointRecords(
      records: Seq[FluxExecutorEndpointRecord],
      requestedCount: Int): Seq[FluxPeerInfo] = {
    toFluxPeerInfos(
      records.map {
        record =>
          UcxEndpointInfo(
            executorId = record.executorId,
            host = record.host,
            blockManagerHost = record.blockManagerHost,
            blockManagerPort = record.blockManagerPort,
            gpuResourceAddresses = record.gpuResourceAddresses,
            nativeUcxListenerEndpoint = Some(record.ucxEndpoint)
          )
      },
      requestedCount
    )
  }

  /**
   * Build the JSON consumed by the native JNI side. Field names and ordering are the canonical
   * cross-language contract.
   */
  def toPeerEndpointsJson(peers: Seq[FluxPeerInfo]): String =
    peers.zipWithIndex
      .map {
        case (peer, peerIndex) =>
          val peerId = jsonEscape(peer.peerId)
          val host = jsonEscape(peer.host)
          s"""{"peerId":"$peerId","host":"$host","port":${peer.port},"peerIndex":$peerIndex}"""
      }
      .mkString("[", ",", "]")

  /**
   * JSON-escape minimal characters. Mirrors the helper used inside `FluxNativeQueryRDD`
   * (`jsonEscape`); we keep a private copy here because the original is `private[execution]`. If
   * the original ever escapes a wider character set, this method must follow.
   */
  private[control] def jsonEscape(value: String): String = {
    val sb = new StringBuilder(value.length + 8)
    var i = 0
    while (i < value.length) {
      val c = value.charAt(i)
      c match {
        case '"' => sb.append("\\\"")
        case '\\' => sb.append("\\\\")
        case '\b' => sb.append("\\b")
        case '\f' => sb.append("\\f")
        case '\n' => sb.append("\\n")
        case '\r' => sb.append("\\r")
        case '\t' => sb.append("\\t")
        case c if c < 0x20 =>
          sb.append("\\u%04x".format(c.toInt))
        case c => sb.append(c)
      }
      i += 1
    }
    sb.toString
  }
}
