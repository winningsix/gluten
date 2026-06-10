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
import org.apache.spark.internal.Logging

class GlutenMppDriverService private[control] (
    private val registry: GlutenMppEndpointRegistry,
    private val enabled: Boolean)
  extends Logging {

  def endpointRegistry: GlutenMppEndpointRegistry = registry

  def receive(msg: Any): AnyRef = {
    if (!enabled) {
      return null
    }
    msg match {
      case RegisterEndpoint(record) =>
        val generation = registry.register(record)
        logInfo(
          s"GlutenMppDriverService: registered endpoint executorId=${record.executorId} " +
            s"generation=$generation")
        RegisterEndpointAck(generation)
      case DeregisterEndpoint(executorId) =>
        val generation = registry.deregister(executorId)
        logInfo(
          s"GlutenMppDriverService: deregistered endpoint executorId=$executorId " +
            s"generation=$generation")
        DeregisterEndpointAck(generation)
      case _ =>
        null
    }
  }
}

object GlutenMppDriverService extends Logging {
  @volatile private var instance: Option[GlutenMppDriverService] = None

  def init(conf: SparkConf): Unit = synchronized {
    // UcxEndpointProbeRDD removed: the endpoint registry is the sole discovery path, always on.
    instance = Some(new GlutenMppDriverService(GlutenMppEndpointRegistry(), enabled = true))
    logInfo("GlutenMppDriverService: endpoint registry initialized")
  }

  def get(): Option[GlutenMppDriverService] = instance

  def shutdown(): Unit = synchronized {
    instance = None
  }

  private[control] def createForTests(
      registry: GlutenMppEndpointRegistry,
      enabled: Boolean): GlutenMppDriverService =
    new GlutenMppDriverService(registry, enabled)
}
