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

import org.apache.gluten.execution.MppPeerInfo

import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging

private[gluten] case class GlutenMppPeerResolution(
    peerInfos: Array[MppPeerInfo],
    peerEndpointsJson: String)

object GlutenMppPeerResolver extends Logging {

  /**
   * Resolve MPP peers: manual `spark.gluten.mpp.peerEndpoints` JSON wins; otherwise the driver
   * endpoint registry is the sole discovery source (UcxEndpointProbeRDD has been removed). If the
   * registry cannot satisfy `requestedCount`, this throws rather than falling back to a probe.
   */
  def resolve(
      conf: SparkConf,
      requestedCount: Int,
      manualPeerEndpointsJson: String,
      driverService: Option[GlutenMppDriverService] = GlutenMppDriverService.get())
      : GlutenMppPeerResolution = {
    val manual = Option(manualPeerEndpointsJson).filter(_.trim.nonEmpty)
    if (manual.isDefined) {
      return GlutenMppPeerResolution(Array.empty, manual.get)
    }

    if (requestedCount <= 0) {
      return GlutenMppPeerResolution(Array.empty, "")
    }

    val attempt = registryPeerInfos(conf, requestedCount, driverService)
    if (attempt.peerInfos.length >= requestedCount) {
      val peers = attempt.peerInfos.take(requestedCount).toArray
      return GlutenMppPeerResolution(peers, GlutenMppPeerMapper.toPeerEndpointsJson(peers))
    }
    throw insufficientRegistryEndpoints(requestedCount, attempt)
  }

  private case class RegistryAttempt(
      peerInfos: Seq[MppPeerInfo],
      availableCount: Int,
      detail: String)

  private def registryPeerInfos(
      conf: SparkConf,
      requestedCount: Int,
      driverService: Option[GlutenMppDriverService]): RegistryAttempt = {
    driverService match {
      case Some(service) =>
        val registry = service.endpointRegistry
        val awaitEnabled = GlutenMppControlPlaneConfig.awaitMinExecutors(conf)
        val timeoutMs = GlutenMppControlPlaneConfig.awaitTimeoutMs(conf)
        val startedNs = System.nanoTime()
        val thresholdMet =
          if (awaitEnabled) {
            logInfo(
              s"GlutenMppPeerResolver: waiting for $requestedCount executor UCX endpoints " +
                s"(timeoutMs=$timeoutMs, current=${registry.snapshot().length})")
            registry.awaitMinExecutors(requestedCount, timeoutMs)
          } else {
            false
          }
        val snapshot = registry.snapshot()
        if (awaitEnabled) {
          val elapsedMs = (System.nanoTime() - startedNs) / 1000000L
          logInfo(
            s"GlutenMppPeerResolver: endpoint wait finished thresholdMet=$thresholdMet " +
              s"elapsedMs=$elapsedMs current=${snapshot.length} requested=$requestedCount")
        }
        RegistryAttempt(
          GlutenMppPeerMapper.fromEndpointRecords(snapshot, requestedCount),
          snapshot.length,
          snapshot
            .map(
              record =>
                s"${record.executorId}@${record.blockManagerHost} " +
                  s"gpu=${record.gpuResourceAddresses.mkString("[", ",", "]")}")
            .mkString("[", ", ", "]")
        )
      case None =>
        RegistryAttempt(Nil, 0, "[driverService=unavailable]")
    }
  }

  private def insufficientRegistryEndpoints(
      requestedCount: Int,
      attempt: RegistryAttempt): IllegalStateException = {
    new IllegalStateException(
      s"spark.gluten.mpp.multiExecutor.numPartitions=$requestedCount but only " +
        s"${attempt.availableCount} executor UCX endpoint(s) were resolved from the driver " +
        s"endpoint registry (UcxEndpointProbeRDD fallback has been removed): " +
        attempt.detail)
  }
}
