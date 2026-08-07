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
package org.apache.gluten.execution

import org.apache.gluten.metrics.MetricsUpdater
import org.apache.gluten.substrait.SubstraitContext
import org.apache.gluten.substrait.rel.{RelBuilder, SplitInfo}
import org.apache.gluten.substrait.rel.LocalFilesNode.ReadFileFormat

import org.apache.spark.Partition
import org.apache.spark.sql.catalyst.expressions.Attribute

import scala.collection.JavaConverters._

/**
 * Replaces [[InputIteratorTransformer]] at FLUX fragment boundaries.
 *
 * In BSP mode, [[InputIteratorTransformer]] generates a `ReadRel` that maps to a `ValueStreamNode`
 * on the native side, consuming data from a JNI callback iterator.
 *
 * In FLUX mode, this transformer generates a `ReadRel` with a special tag indicating
 * "flux-exchange-source". The C++ side will create a `GpuExchange` operator that reads from the
 * `OutputBufferManager` of the producing fragment, enabling streaming data flow between fragments
 * without JNI round-trips.
 *
 * @param exchangeId
 *   The ID of the exchange this source reads from.
 * @param outputAttributes
 *   The output schema attributes.
 */
case class FluxExchangeSourceTransformer(
    exchangeId: Int,
    outputAttributes: Seq[Attribute]
) extends LeafTransformSupport {

  override def output: Seq[Attribute] = outputAttributes

  override def nodeName: String = s"FluxExchangeSource(exchange=$exchangeId)"

  override def simpleString(maxFields: Int): String = {
    s"FluxExchangeSource(exchange=$exchangeId, " +
      s"output=${outputAttributes.map(_.name).mkString("[", ", ", "]")})"
  }

  override protected def doTransform(context: SubstraitContext): TransformContext = {
    val operatorId = context.nextOperatorId(nodeName)
    // Generate a ReadRel for this exchange source.
    // The native side identifies this as an FLUX exchange source via the exchangeId
    // property carried in the Substrait plan extension.
    // TODO: Add advanced extension tag/property to the ReadRel to mark as
    //       "flux-exchange-source" with exchangeId, so the C++ side creates
    //       GpuExchange instead of ValueStreamNode when it sees this tag.
    val readRel =
      RelBuilder.makeReadRelForInputIterator(outputAttributes.asJava, context, operatorId)
    TransformContext(outputAttributes, readRel)
  }

  override def metricsUpdater(): MetricsUpdater = MetricsUpdater.Terminate

  override def getSplitInfos: Seq[SplitInfo] = Seq.empty

  override def getPartitions: Seq[Partition] = Seq.empty

  override def getPartitionWithReadFileFormats: Seq[(Partition, ReadFileFormat)] = Seq.empty
}
