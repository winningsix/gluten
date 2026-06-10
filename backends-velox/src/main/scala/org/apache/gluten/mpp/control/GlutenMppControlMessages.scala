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
 * Control-plane messages exchanged between `GlutenMppExecutorService` (on each executor) and
 * `GlutenMppDriverService` (on the Spark driver).
 *
 * Phase 1 defines the message *shapes* but does not wire any RPC channel; C3 will add that via the
 * Spark plugin's `DriverPlugin.receive` mechanism. These are intentionally plain `Serializable`
 * case classes so they can travel over any Spark-supplied transport.
 */
sealed trait GlutenMppControlMessage extends Serializable

/** Executor to Driver: announce / refresh this executor's UCX-capable endpoint. */
final case class RegisterEndpoint(record: MppExecutorEndpointRecord) extends GlutenMppControlMessage

/** Driver to Executor: ack of a registration, with the registry generation after insert. */
final case class RegisterEndpointAck(generation: Long) extends GlutenMppControlMessage

/** Executor to Driver: graceful deregistration prior to executor shutdown. */
final case class DeregisterEndpoint(executorId: String) extends GlutenMppControlMessage

/** Driver to Executor: ack of a deregistration, with the registry generation after remove. */
final case class DeregisterEndpointAck(generation: Long) extends GlutenMppControlMessage
