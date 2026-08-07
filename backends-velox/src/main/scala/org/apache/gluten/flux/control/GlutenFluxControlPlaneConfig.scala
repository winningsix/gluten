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

/**
 * Spark configuration keys and defaults for endpoint discovery and query cancellation.
 *
 * UcxEndpointProbeRDD has been removed: the driver endpoint registry is the sole endpoint-discovery
 * path when native FLUX is enabled. Driver-only sessions and non-FLUX runs must not initialize or
 * register UCX endpoints.
 */
object GlutenFluxControlPlaneConfig {

  val FluxEnabledKey: String = "spark.gluten.mpp.enabled"
  val FluxEnabledDefault: Boolean = false

  val EndpointRegistryEnabledKey: String =
    "spark.gluten.mpp.controlPlane.endpointRegistry.enabled"
  val EndpointRegistryEnabledDefault: Boolean = true

  // ---- Endpoint discovery ----

  /** When true, the resolver blocks until at least `requestedCount` endpoints are LIVE. */
  val AwaitMinExecutorsKey: String =
    "spark.gluten.mpp.controlPlane.endpointRegistry.awaitMinExecutors"
  val AwaitMinExecutorsDefault: Boolean = true

  /** Maximum time (ms) the resolver waits for the registry to populate. */
  val AwaitTimeoutMsKey: String =
    "spark.gluten.mpp.controlPlane.endpointRegistry.awaitTimeoutMs"
  // Cold standalone executors may spend more than ten seconds fetching the Gluten bundle and
  // loading the native Velox/UCX runtime before they can publish a listener endpoint. Keep peer
  // discovery bounded, but cover that normal initialization window.
  val AwaitTimeoutMsDefault: Long = 60000L

  // ---- Query cancellation ----

  /**
   * Enables executor interruption watching and driver-mediated peer abort while FLUX is enabled.
   */
  val QueryCancellationEnabledKey: String =
    "spark.gluten.mpp.controlPlane.queryCancellation.enabled"
  val QueryCancellationEnabledDefault: Boolean = true

  /** Interval for scanning active Spark TaskContexts for local interruption. */
  val InterruptPollMsKey: String =
    "spark.gluten.mpp.controlPlane.queryCancellation.interruptPollMs"
  val InterruptPollMsDefault: Long = 50L

  /** Interval for sending active-peer state and retryable events to the driver. */
  val HeartbeatMsKey: String =
    "spark.gluten.mpp.controlPlane.queryCancellation.heartbeatMs"
  val HeartbeatMsDefault: Long = 250L

  /** Maximum peer heartbeat silence after a run reaches RUNNING. */
  val PeerTimeoutMsKey: String =
    "spark.gluten.mpp.controlPlane.queryCancellation.peerTimeoutMs"
  val PeerTimeoutMsDefault: Long = 5000L

  /** Duration for retaining terminal run tombstones against late heartbeat recreation. */
  val TerminalRetentionMsDefault: Long = 60000L

  // ---- Accessors ----

  def endpointRegistryEnabled(conf: SparkConf): Boolean =
    conf.getBoolean(FluxEnabledKey, FluxEnabledDefault) &&
      conf.getBoolean(EndpointRegistryEnabledKey, EndpointRegistryEnabledDefault)

  def awaitMinExecutors(conf: SparkConf): Boolean =
    conf.getBoolean(AwaitMinExecutorsKey, AwaitMinExecutorsDefault)

  def awaitTimeoutMs(conf: SparkConf): Long =
    conf.getLong(AwaitTimeoutMsKey, AwaitTimeoutMsDefault)

  def queryCancellationEnabled(conf: SparkConf): Boolean =
    conf.getBoolean(FluxEnabledKey, FluxEnabledDefault) &&
      conf.getBoolean(QueryCancellationEnabledKey, QueryCancellationEnabledDefault)

  def interruptPollMs(conf: SparkConf): Long =
    positive(conf.getLong(InterruptPollMsKey, InterruptPollMsDefault), InterruptPollMsDefault)

  def heartbeatMs(conf: SparkConf): Long =
    positive(conf.getLong(HeartbeatMsKey, HeartbeatMsDefault), HeartbeatMsDefault)

  def peerTimeoutMs(conf: SparkConf): Long =
    positive(conf.getLong(PeerTimeoutMsKey, PeerTimeoutMsDefault), PeerTimeoutMsDefault)

  private def positive(value: Long, defaultValue: Long): Long =
    if (value > 0L) value else defaultValue
}
