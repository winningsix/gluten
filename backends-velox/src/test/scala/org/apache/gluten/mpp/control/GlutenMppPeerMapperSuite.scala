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

import org.scalatest.funsuite.AnyFunSuite

class GlutenMppPeerMapperSuite extends AnyFunSuite {

  private def info(
      executorId: String,
      host: String,
      blockManagerHost: String,
      blockManagerPort: Int = 7079,
      ucxListenerPort: Int = 12345,
      ucxHost: String = "",
      gpu: Seq[String] = Seq("0")): UcxEndpointInfo = {
    val effectiveUcxHost = if (ucxHost.nonEmpty) ucxHost else host
    UcxEndpointInfo(
      executorId = executorId,
      host = host,
      blockManagerHost = blockManagerHost,
      blockManagerPort = blockManagerPort,
      gpuResourceAddresses = gpu,
      nativeUcxListenerEndpoint = Some(s"ucx://$effectiveUcxHost:$ucxListenerPort")
    )
  }

  // ---- Mapping ----

  test("empty input -> empty output") {
    val out = GlutenMppPeerMapper.toMppPeerInfos(Nil, 0)
    assert(out == Nil)
  }

  test("requestedCount=0 -> empty output even with inputs") {
    val out = GlutenMppPeerMapper.toMppPeerInfos(Seq(info("e1", "h1", "h1")), 0)
    assert(out == Nil)
  }

  test("requestedCount negative -> require fails") {
    intercept[IllegalArgumentException] {
      GlutenMppPeerMapper.toMppPeerInfos(Nil, -1)
    }
  }

  test("port = listenerPort - 3 invariant") {
    val out =
      GlutenMppPeerMapper.toMppPeerInfos(Seq(info("e1", "h1", "h1", ucxListenerPort = 20000)), 1)
    assert(out.head.port == 19997)
  }

  test("rejects listenerPort <= 3") {
    intercept[IllegalStateException] {
      GlutenMppPeerMapper.toMppPeerInfos(Seq(info("e1", "h1", "h1", ucxListenerPort = 2)), 1)
    }
  }

  test("rejects missing nativeUcxListenerEndpoint") {
    val bad = UcxEndpointInfo(
      executorId = "e1",
      host = "h1",
      blockManagerHost = "h1",
      blockManagerPort = 0,
      gpuResourceAddresses = Nil,
      nativeUcxListenerEndpoint = None)
    intercept[IllegalStateException] {
      GlutenMppPeerMapper.toMppPeerInfos(Seq(bad), 1)
    }
  }

  test("UCX connection prefers listener IP while Spark placement uses blockManagerHost") {
    val infos = Seq(info("e1", host = "infoHost", blockManagerHost = "bmHost", ucxHost = "ucxHost"))
    val out = GlutenMppPeerMapper.toMppPeerInfos(infos, 1)
    assert(out.head.host == "ucxHost")
    assert(out.head.preferredLocation == "executor_bmHost_e1")
  }

  test("UCX connection prefers a non-loopback BlockManager IP over another listener NIC") {
    val infos = Seq(
      info(
        "e1",
        host = "10.87.131.11",
        blockManagerHost = "10.87.131.11",
        ucxHost = "10.87.131.15"))
    val out = GlutenMppPeerMapper.toMppPeerInfos(infos, 1)
    assert(out.head.host == "10.87.131.11")
    assert(out.head.preferredLocation == "executor_10.87.131.11_e1")
  }

  test("UCX connection does not replace listener host with localhost placement host") {
    val infos =
      Seq(info("e1", host = "infoHost", blockManagerHost = "localhost", ucxHost = "ucxHost"))
    val out = GlutenMppPeerMapper.toMppPeerInfos(infos, 1)
    assert(out.head.host == "ucxHost")
    assert(out.head.preferredLocation == "executor_localhost_e1")
  }

  test("UCX connection does not replace listener host with loopback placement host") {
    val infos =
      Seq(info("e1", host = "infoHost", blockManagerHost = "127.0.0.1", ucxHost = "ucxHost"))
    val out = GlutenMppPeerMapper.toMppPeerInfos(infos, 1)
    assert(out.head.host == "ucxHost")
    assert(out.head.preferredLocation == "executor_127.0.0.1_e1")
  }

  test("host selection falls through to URI host when blockManagerHost empty") {
    val infos = Seq(info("e1", host = "infoHost", blockManagerHost = "", ucxHost = "ucxHost"))
    val out = GlutenMppPeerMapper.toMppPeerInfos(infos, 1)
    assert(out.head.host == "ucxHost")
    assert(out.head.preferredLocation == "executor_infoHost_e1")
  }

  test("requestedCount truncation matches take()") {
    val infos = (1 to 5).map(i => info(s"e$i", s"h$i", s"h$i"))
    val out = GlutenMppPeerMapper.toMppPeerInfos(infos, 2)
    assert(out.size == 2)
    assert(out.map(_.peerId) == Seq("e1", "e2"))
  }

  // ---- JSON parity ----

  test("toPeerEndpointsJson empty -> []") {
    assert(GlutenMppPeerMapper.toPeerEndpointsJson(Nil) == "[]")
  }

  test("toPeerEndpointsJson golden: 2 peers, byte-exact field order") {
    val peers = GlutenMppPeerMapper.toMppPeerInfos(
      Seq(
        info("e1", host = "host-a", blockManagerHost = "host-a", ucxListenerPort = 50100),
        info("e2", host = "host-b", blockManagerHost = "host-b", ucxListenerPort = 50200)),
      2
    )
    val json = GlutenMppPeerMapper.toPeerEndpointsJson(peers)
    val expected =
      """[{"peerId":"e1","host":"host-a","port":50097,"peerIndex":0},""" +
        """{"peerId":"e2","host":"host-b","port":50197,"peerIndex":1}]"""
    assert(json == expected, s"\n  actual:   $json\n  expected: $expected")
  }

  test("one-peer executor-local JSON uses the registered routable endpoint") {
    val record = MppExecutorEndpointRecord(
      executorId = "12",
      host = "10.87.140.44",
      blockManagerHost = "presto-gb200-gcn-09",
      blockManagerPort = 40200,
      ucxListenerPort = 48366,
      remoteConnectorPort = 48363,
      ucxEndpoint = "ucx://10.87.140.44:48366",
      gpuResourceAddresses = Seq("1"),
      registeredAtMs = 1L,
      generation = 0L,
      state = MppExecutorEndpointRecord.StateLive
    )

    assert(
      GlutenMppExecutorService
        .localPeerEndpointsJson(record, "12")
        .contains("""[{"peerId":"12","host":"10.87.140.44","port":48363,"peerIndex":0}]"""))
    assert(GlutenMppExecutorService.localPeerEndpointsJson(record, "13").isEmpty)
  }

  test("toPeerEndpointsJson escapes quote / backslash / control chars in peerId and host") {
    val peers = Seq(MppPeerInfo("""e"1\x""", "h\ta\nb", 59997, "unused"))
    val json = GlutenMppPeerMapper.toPeerEndpointsJson(peers)
    // Must escape: " -> \", \ -> \\, \t -> \t, \n -> \n
    val expected =
      """[{"peerId":"e\"1\\x","host":"h\ta\nb","port":59997,"peerIndex":0}]"""
    assert(json == expected, s"\n  actual:   $json\n  expected: $expected")
  }

  test("jsonEscape low control chars use \\u04x format") {
    // 0x01 -> 
    val escaped = GlutenMppPeerMapper.jsonEscape("ab")
    assert(escaped == "a\\u0001b")
  }

  test("jsonEscape leaves printable ASCII untouched") {
    val s = "abc 0123 !@#"
    assert(GlutenMppPeerMapper.jsonEscape(s) == s)
  }
}
