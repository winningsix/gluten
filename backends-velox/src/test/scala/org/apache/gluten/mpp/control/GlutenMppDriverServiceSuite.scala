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

import org.scalatest.funsuite.AnyFunSuite

class GlutenMppDriverServiceSuite extends AnyFunSuite {

  private def rec(
      executorId: String,
      host: String = "host",
      listenerPort: Int = 12345): MppExecutorEndpointRecord =
    MppExecutorEndpointRecord(
      executorId = executorId,
      host = host,
      blockManagerHost = host,
      blockManagerPort = 7079,
      ucxListenerPort = listenerPort,
      remoteConnectorPort = listenerPort - 3,
      ucxEndpoint = s"ucx://$host:$listenerPort",
      gpuResourceAddresses = Seq("0"),
      registeredAtMs = 0L,
      generation = 0L,
      state = MppExecutorEndpointRecord.StateLive
    )

  test("register endpoint updates registry and returns ack") {
    val registry = GlutenMppEndpointRegistry()
    val service = GlutenMppDriverService.createForTests(registry, enabled = true)

    val ack = service.receive(RegisterEndpoint(rec("e1"))).asInstanceOf[RegisterEndpointAck]

    assert(ack.generation == 1L)
    assert(registry.snapshot().map(_.executorId) == Seq("e1"))
  }

  test("deregister endpoint removes registry entry and returns ack") {
    val registry = GlutenMppEndpointRegistry()
    val service = GlutenMppDriverService.createForTests(registry, enabled = true)
    service.receive(RegisterEndpoint(rec("e1")))

    val ack = service.receive(DeregisterEndpoint("e1")).asInstanceOf[DeregisterEndpointAck]

    assert(ack.generation == 2L)
    assert(registry.snapshot().isEmpty)
  }

  test("re-registering an executor replaces the previous record") {
    val registry = GlutenMppEndpointRegistry()
    val service = GlutenMppDriverService.createForTests(registry, enabled = true)

    service.receive(RegisterEndpoint(rec("e1", host = "old-host", listenerPort = 10000)))
    val ack = service
      .receive(RegisterEndpoint(rec("e1", host = "new-host", listenerPort = 20000)))
      .asInstanceOf[RegisterEndpointAck]

    val snapshot = registry.snapshot()
    assert(ack.generation == 2L)
    assert(snapshot.size == 1)
    assert(snapshot.head.host == "new-host")
    assert(snapshot.head.ucxListenerPort == 20000)
  }

  test("disabled service ignores registration messages") {
    val registry = GlutenMppEndpointRegistry()
    val service = GlutenMppDriverService.createForTests(registry, enabled = false)

    assert(service.receive(RegisterEndpoint(rec("e1"))) == null)
    assert(registry.snapshot().isEmpty)
  }

  test("unknown messages are ignored") {
    val registry = GlutenMppEndpointRegistry()
    val service = GlutenMppDriverService.createForTests(registry, enabled = true)

    assert(service.receive("unknown") == null)
    assert(registry.snapshot().isEmpty)
  }

  test("init creates the always-on driver service") {
    GlutenMppDriverService.shutdown()
    try {
      GlutenMppDriverService.init(new SparkConf(loadDefaults = false))
      assert(GlutenMppDriverService.get().isDefined)
    } finally {
      GlutenMppDriverService.shutdown()
    }
  }
}
