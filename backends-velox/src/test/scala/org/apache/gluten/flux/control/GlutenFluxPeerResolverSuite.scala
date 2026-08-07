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

import org.scalatest.funsuite.AnyFunSuite

class GlutenFluxPeerResolverSuite extends AnyFunSuite {

  // UcxEndpointProbeRDD removed: the registry is the sole discovery path (no enable flag). Disable
  // the blocking await so the tests resolve against the snapshot synchronously.
  private def conf(): SparkConf =
    new SparkConf(loadDefaults = false)
      .set(GlutenFluxControlPlaneConfig.AwaitMinExecutorsKey, "false")

  private def record(
      executorId: String,
      host: String,
      listenerPort: Int,
      gpu: Seq[String] = Seq("0")): FluxExecutorEndpointRecord =
    FluxExecutorEndpointRecord(
      executorId = executorId,
      host = host,
      blockManagerHost = host,
      blockManagerPort = 7079,
      ucxListenerPort = listenerPort,
      remoteConnectorPort = listenerPort - 3,
      ucxEndpoint = s"ucx://$host:$listenerPort",
      gpuResourceAddresses = gpu,
      registeredAtMs = 0L,
      generation = 0L,
      state = FluxExecutorEndpointRecord.StateLive
    )

  private def serviceWith(records: FluxExecutorEndpointRecord*): GlutenFluxDriverService = {
    val registry = GlutenFluxEndpointRegistry()
    records.foreach(registry.register)
    GlutenFluxDriverService.createForTests(registry, enabled = true)
  }

  test("manual peerEndpoints JSON wins and passes through verbatim") {
    val manual = "  [{\"peerId\":\"manual\",\"host\":\"h\",\"port\":1,\"peerIndex\":0}]  "
    val resolved = GlutenFluxPeerResolver.resolve(
      conf(),
      requestedCount = 2,
      manualPeerEndpointsJson = manual,
      driverService = Some(serviceWith())
    )
    assert(resolved.peerInfos.isEmpty)
    assert(resolved.peerEndpointsJson == manual)
  }

  test("registry with enough endpoints resolves peers (no probe)") {
    val resolved = GlutenFluxPeerResolver.resolve(
      conf(),
      requestedCount = 2,
      manualPeerEndpointsJson = "",
      driverService =
        Some(serviceWith(record("e2", "host-b", 50200), record("e1", "host-a", 50100)))
    )
    assert(resolved.peerInfos.map(_.peerId).toSeq == Seq("e1", "e2"))
    assert(
      resolved.peerEndpointsJson ==
        """[{"peerId":"e1","host":"host-a","port":50097,"peerIndex":0},""" +
        """{"peerId":"e2","host":"host-b","port":50197,"peerIndex":1}]""")
  }

  test("resolver waits for delayed endpoint registrations within the bounded deadline") {
    val registry = GlutenFluxEndpointRegistry()
    val service = GlutenFluxDriverService.createForTests(registry, enabled = true)
    val waitingConf = new SparkConf(loadDefaults = false)
      .set(GlutenFluxControlPlaneConfig.AwaitMinExecutorsKey, "true")
      .set(GlutenFluxControlPlaneConfig.AwaitTimeoutMsKey, "5000")
    val registration = new Thread(
      () => {
        Thread.sleep(50L)
        registry.register(record("e1", "host-a", 50100))
        registry.register(record("e2", "host-b", 50200))
      })
    registration.start()

    val resolved = GlutenFluxPeerResolver.resolve(
      waitingConf,
      requestedCount = 2,
      manualPeerEndpointsJson = "",
      driverService = Some(service))
    registration.join()

    assert(resolved.peerInfos.map(_.peerId).toSeq == Seq("e1", "e2"))
  }

  test("registry with insufficient endpoints fails fast (no probe fallback)") {
    val error = intercept[IllegalStateException] {
      GlutenFluxPeerResolver.resolve(
        conf(),
        requestedCount = 2,
        manualPeerEndpointsJson = "",
        driverService = Some(serviceWith(record("e1", "host-a", 50100)))
      )
    }
    assert(error.getMessage.contains("UcxEndpointProbeRDD fallback has been removed"))
  }
}
