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

/**
 * Driver-side record describing one executor's UCX-capable endpoint.
 *
 * Phase 1 POC data model. The driver-side endpoint registry stores instances of this case class
 * indexed by `executorId`. Records are the sole endpoint-discovery source: the per-query
 * `UcxEndpointProbeRDD` has been removed and the registry is always active.
 *
 * Invariants:
 *   - `remoteConnectorPort == ucxListenerPort - 3` matches the existing native convention used by
 *     `MppNativeQueryExec.mppPeerInfos()` (do not break this without a coordinated native change;
 *     native JNI consumers depend on it).
 */
case class MppExecutorEndpointRecord(
    executorId: String,
    host: String,
    blockManagerHost: String,
    blockManagerPort: Int,
    ucxListenerPort: Int,
    remoteConnectorPort: Int,
    ucxEndpoint: String,
    gpuResourceAddresses: Seq[String],
    registeredAtMs: Long,
    generation: Long,
    state: String) {

  require(
    remoteConnectorPort == ucxListenerPort - 3,
    s"remoteConnectorPort=$remoteConnectorPort must equal " +
      s"ucxListenerPort-3=${ucxListenerPort - 3} " +
      s"for executor $executorId"
  )

  def isLive: Boolean = state == MppExecutorEndpointRecord.StateLive
}

object MppExecutorEndpointRecord {

  /** Phase 1 only emits LIVE records; later phases may add LEAVING / STOPPED states. */
  val StateLive: String = "LIVE"
}
