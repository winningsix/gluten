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

import org.apache.gluten.execution.{MppPeerInfo, UcxEndpointInfo}

import java.net.URI

/**
 * Shared `UcxEndpointInfo` -> `MppPeerInfo` mapping.
 *
 * Behaviour MUST match the existing inline mapping in `MppNativeQueryExec.mppPeerInfos()` (lines
 * 1909-1927 as of velox HEAD a482411a0b) byte-for-byte:
 *
 *   - take(requestedCount) of sortBy(executorId)-ordered endpoints
 *   - parse `nativeUcxListenerEndpoint` URI to extract listenerPort
 *   - reject listenerPort <= 3
 *   - host selection: blockManagerHost (when non-empty) falls through to the URI host, then to
 *     `info.host`
 *   - `port = listenerPort - 3` (this is the "remote connector port" native convention)
 *   - `preferredLocation = executor_<host>_<executorId>`
 *
 * JSON serialisation (toPeerEndpointsJson) produces the contract consumed by the native JNI side:
 *
 * [{"peerId":"<ESCAPED>","host":"<ESCAPED>","port":N,"peerIndex":I}, ...]
 */
object GlutenMppPeerMapper {

  /**
   * Map a sorted prefix of discovered endpoints into `MppPeerInfo` records. The caller is expected
   * to sort `infos` by `executorId` and pass `infos.length` (i.e. no truncation) or a smaller count
   * for `requestedCount` to match the original truncation semantics.
   */
  def toMppPeerInfos(infos: Seq[UcxEndpointInfo], requestedCount: Int): Seq[MppPeerInfo] = {
    require(requestedCount >= 0, s"requestedCount=$requestedCount must be non-negative")
    infos.take(requestedCount).map {
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
        val host = blockManagerHost.getOrElse(endpointHost.getOrElse(info.host))
        MppPeerInfo(
          peerId = info.executorId,
          host = host,
          port = listenerPort - 3,
          preferredLocation = s"executor_${host}_${info.executorId}")
    }
  }

  /** Map registry endpoint records through the same contract as probe-discovered endpoint info. */
  def fromEndpointRecords(
      records: Seq[MppExecutorEndpointRecord],
      requestedCount: Int): Seq[MppPeerInfo] = {
    toMppPeerInfos(
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
  def toPeerEndpointsJson(peers: Seq[MppPeerInfo]): String =
    peers.zipWithIndex
      .map {
        case (peer, peerIndex) =>
          val peerId = jsonEscape(peer.peerId)
          val host = jsonEscape(peer.host)
          s"""{"peerId":"$peerId","host":"$host","port":${peer.port},"peerIndex":$peerIndex}"""
      }
      .mkString("[", ",", "]")

  /**
   * JSON-escape minimal characters. Mirrors the helper used inside `MppNativeQueryRDD`
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
