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

import org.scalatest.funsuite.AnyFunSuite

class GlutenMppEndpointRegistrySuite extends AnyFunSuite {

  private def rec(
      executorId: String,
      host: String = "host",
      blockManagerHost: String = "bmHost",
      blockManagerPort: Int = 7079,
      ucxListenerPort: Int = 12345,
      ucxEndpoint: Option[String] = None,
      gpu: Seq[String] = Seq("0"),
      state: String = MppExecutorEndpointRecord.StateLive): MppExecutorEndpointRecord =
    MppExecutorEndpointRecord(
      executorId = executorId,
      host = host,
      blockManagerHost = blockManagerHost,
      blockManagerPort = blockManagerPort,
      ucxListenerPort = ucxListenerPort,
      remoteConnectorPort = ucxListenerPort - 3,
      ucxEndpoint = ucxEndpoint.getOrElse(s"ucx://$host:$ucxListenerPort"),
      gpuResourceAddresses = gpu,
      registeredAtMs = 0L,
      generation = 0L,
      state = state
    )

  test("remoteConnectorPort invariant rejects mismatch") {
    intercept[IllegalArgumentException] {
      MppExecutorEndpointRecord(
        executorId = "e0",
        host = "h",
        blockManagerHost = "h",
        blockManagerPort = 1,
        ucxListenerPort = 100,
        remoteConnectorPort = 99, // should be 97
        ucxEndpoint = "ucx://h:100",
        gpuResourceAddresses = Nil,
        registeredAtMs = 0L,
        generation = 0L,
        state = MppExecutorEndpointRecord.StateLive
      )
    }
  }

  test("empty snapshot is Nil and generation starts at zero") {
    val r = GlutenMppEndpointRegistry()
    assert(r.snapshot() == Nil)
    assert(r.currentGeneration() == 0L)
  }

  test("register populates snapshot with the assigned generation") {
    val r = GlutenMppEndpointRegistry()
    val g1 = r.register(rec("e1"))
    assert(g1 == 1L)
    val s = r.snapshot()
    assert(s.size == 1)
    assert(s.head.executorId == "e1")
    assert(s.head.generation == 1L)
  }

  test("deregister removes a record and bumps generation") {
    val r = GlutenMppEndpointRegistry()
    r.register(rec("e1"))
    r.register(rec("e2"))
    val gBefore = r.currentGeneration()
    val gAfter = r.deregister("e1")
    assert(gAfter > gBefore)
    assert(r.snapshot().map(_.executorId) == Seq("e2"))
  }

  test("snapshot is sorted by executorId and filters non-LIVE") {
    val r = GlutenMppEndpointRegistry()
    r.register(rec("e3"))
    r.register(rec("e1"))
    r.register(rec("e2", state = "STOPPED"))
    assert(r.snapshot().map(_.executorId) == Seq("e1", "e3"))
  }

  test("generation is monotonic across many registers") {
    val r = GlutenMppEndpointRegistry()
    val gs = (1 to 25).map(i => r.register(rec(s"e$i")))
    assert(gs == gs.sorted)
    assert(gs.distinct.size == gs.size)
  }

  test("awaitMinExecutors returns true when threshold already met") {
    val r = GlutenMppEndpointRegistry()
    r.register(rec("e1"))
    r.register(rec("e2"))
    assert(r.awaitMinExecutors(2, 100L))
  }

  test("awaitMinExecutors wakes up when async register satisfies threshold") {
    val r = GlutenMppEndpointRegistry()
    val asyncT = new Thread(new Runnable {
      override def run(): Unit = {
        Thread.sleep(50)
        r.register(rec("e1"))
        r.register(rec("e2"))
      }
    })
    asyncT.start()
    val ok = r.awaitMinExecutors(2, 5000L)
    asyncT.join()
    assert(ok)
  }

  test("awaitMinExecutors returns false on timeout") {
    val r = GlutenMppEndpointRegistry()
    r.register(rec("e1"))
    val start = System.currentTimeMillis()
    val ok = r.awaitMinExecutors(5, 100L)
    val elapsed = System.currentTimeMillis() - start
    assert(!ok, "should not meet 5-executor threshold with 1 registered")
    assert(elapsed >= 90L, s"should block ~100ms (got ${elapsed}ms)")
  }

  test("awaitMinExecutors with min <= 0 returns immediately") {
    val r = GlutenMppEndpointRegistry()
    assert(r.awaitMinExecutors(0, 100L))
    assert(r.awaitMinExecutors(-1, 100L))
  }
}
