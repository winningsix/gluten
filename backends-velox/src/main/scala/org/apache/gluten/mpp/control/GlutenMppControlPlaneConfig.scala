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

/**
 * Spark configuration keys + defaults for the driver-side MPP control plane.
 *
 * UcxEndpointProbeRDD has been removed: the driver endpoint registry is the sole endpoint-discovery
 * path and is always active (executors register on native init; the resolver consults the
 * registry). There is therefore no endpoint-registry enable flag and no probe fallback knob.
 */
object GlutenMppControlPlaneConfig {

  // ---- Phase 1 (active) ----

  /** When true, the resolver blocks until at least `requestedCount` endpoints are LIVE. */
  val AwaitMinExecutorsKey: String =
    "spark.gluten.mpp.controlPlane.endpointRegistry.awaitMinExecutors"
  val AwaitMinExecutorsDefault: Boolean = true

  /** Maximum time (ms) the resolver waits for the registry to populate. */
  val AwaitTimeoutMsKey: String =
    "spark.gluten.mpp.controlPlane.endpointRegistry.awaitTimeoutMs"
  val AwaitTimeoutMsDefault: Long = 10000L

  // ---- Accessors ----

  def awaitMinExecutors(conf: SparkConf): Boolean =
    conf.getBoolean(AwaitMinExecutorsKey, AwaitMinExecutorsDefault)

  def awaitTimeoutMs(conf: SparkConf): Long =
    conf.getLong(AwaitTimeoutMsKey, AwaitTimeoutMsDefault)
}
