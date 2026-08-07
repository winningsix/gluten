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

import org.apache.gluten.execution.FluxPeerInfo

import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging

private[gluten] case class GlutenFluxPeerResolution(
    peerInfos: Array[FluxPeerInfo],
    peerEndpointsJson: String)

object GlutenFluxPeerResolver extends Logging {

  /**
   * Resolve FLUX peers: manual `spark.gluten.mpp.peerEndpoints` JSON wins; otherwise the driver
   * endpoint registry is the sole discovery source (UcxEndpointProbeRDD has been removed). If the
   * registry cannot satisfy `requestedCount`, this throws rather than falling back to a probe.
   */
  def resolve(
      conf: SparkConf,
      requestedCount: Int,
      manualPeerEndpointsJson: String,
      driverService: Option[GlutenFluxDriverService] = GlutenFluxDriverService.get())
      : GlutenFluxPeerResolution = {
    val manual = Option(manualPeerEndpointsJson).filter(_.trim.nonEmpty)
    if (manual.isDefined) {
      return GlutenFluxPeerResolution(Array.empty, manual.get)
    }

    if (requestedCount <= 0) {
      return GlutenFluxPeerResolution(Array.empty, "")
    }

    val attempt = registryPeerInfos(conf, requestedCount, driverService)
    if (attempt.peerInfos.length >= requestedCount) {
      val peers = attempt.peerInfos.take(requestedCount).toArray
      return GlutenFluxPeerResolution(peers, GlutenFluxPeerMapper.toPeerEndpointsJson(peers))
    }
    throw insufficientRegistryEndpoints(requestedCount, attempt)
  }

  private case class RegistryAttempt(
      peerInfos: Seq[FluxPeerInfo],
      availableCount: Int,
      detail: String)

  private def registryPeerInfos(
      conf: SparkConf,
      requestedCount: Int,
      driverService: Option[GlutenFluxDriverService]): RegistryAttempt = {
    driverService match {
      case Some(service) =>
        val registry = service.endpointRegistry
        val awaitEnabled = GlutenFluxControlPlaneConfig.awaitMinExecutors(conf)
        val timeoutMs = GlutenFluxControlPlaneConfig.awaitTimeoutMs(conf)
        val startedNs = System.nanoTime()
        val thresholdMet =
          if (awaitEnabled) {
            logInfo(
              s"GlutenFluxPeerResolver: waiting for $requestedCount executor UCX endpoints " +
                s"(timeoutMs=$timeoutMs, current=${registry.snapshot().length})")
            registry.awaitMinExecutors(requestedCount, timeoutMs)
          } else {
            false
          }
        val snapshot = registry.snapshot()
        if (awaitEnabled) {
          val elapsedMs = (System.nanoTime() - startedNs) / 1000000L
          logInfo(
            s"GlutenFluxPeerResolver: endpoint wait finished thresholdMet=$thresholdMet " +
              s"elapsedMs=$elapsedMs current=${snapshot.length} requested=$requestedCount")
        }
        RegistryAttempt(
          GlutenFluxPeerMapper.fromEndpointRecords(snapshot, requestedCount),
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
