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
package org.apache.spark.shuffle

import org.apache.spark.{SparkConf, SparkException}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.Utils

private[spark] case class UcxShuffleNativeWriterRuntimeStats(
    noMoreData: Boolean,
    finished: Boolean,
    queuedBytes: Long,
    queuedPages: Long,
    totalBytesSent: Long,
    totalRowsSent: Long,
    totalPagesSent: Long,
    averageBufferTimeMs: Long,
    blocked: Boolean)

private[spark] trait UcxShuffleNativeBridge {
  def localShuffleEndpointPort(): Int = -1

  def openWriter(
      endpoint: UcxShuffleEndpoint,
      numPartitions: Int,
      partitioningName: String,
      startPartitionId: Int): Long

  def writeBatch(writerHandle: Long, partitionId: Int, batch: ColumnarBatch): Unit

  def closeWriter(writerHandle: Long, success: Boolean): Unit

  def writerNoMoreData(nativeTaskId: String): Boolean = false

  def writerRuntimeStats(nativeTaskId: String): Option[UcxShuffleNativeWriterRuntimeStats] = None

  def openReader(
      shuffleId: Int,
      reducePartitionId: Int,
      outputSchema: org.apache.spark.sql.types.StructType,
      endpoints: Seq[UcxShuffleEndpoint]): Long

  def addReaderEndpoints(readerHandle: Long, endpoints: Seq[UcxShuffleEndpoint]): Unit

  def noMoreReaderEndpoints(readerHandle: Long): Unit

  def nextBatch(readerHandle: Long): Option[ColumnarBatch]

  def closeReader(readerHandle: Long): Unit
}

private[spark] object UcxShuffleNativeBridge extends Logging {
  val BridgeClassConf: String = "spark.gluten.ucx.shuffle.nativeBridge.class"
  val DefaultBridgeClass: String = "org.apache.spark.shuffle.VeloxUcxShuffleNativeBridge"

  @volatile private var instance: UcxShuffleNativeBridge = _

  def getOrCreate(conf: SparkConf): UcxShuffleNativeBridge = {
    var bridge = instance
    if (bridge != null) {
      return bridge
    }
    synchronized {
      bridge = instance
      if (bridge == null) {
        bridge = load(conf)
        instance = bridge
      }
      bridge
    }
  }

  def stop(): Unit = synchronized {
    instance match {
      case closeable: AutoCloseable =>
        closeable.close()
      case _ =>
    }
    instance = null
  }

  private def load(conf: SparkConf): UcxShuffleNativeBridge = {
    val className = conf.get(BridgeClassConf, DefaultBridgeClass)
    try {
      val klass = Utils.classForName[AnyRef](className)
      val bridge: AnyRef =
        try {
          val ctor = klass.getDeclaredConstructor(classOf[SparkConf])
          ctor.setAccessible(true)
          ctor.newInstance(conf).asInstanceOf[AnyRef]
        } catch {
          case _: NoSuchMethodException =>
            val ctor = klass.getDeclaredConstructor()
            ctor.setAccessible(true)
            ctor.newInstance().asInstanceOf[AnyRef]
        }
      bridge match {
        case typed: UcxShuffleNativeBridge =>
          logInfo(s"Loaded Gluten UCX shuffle native bridge $className")
          typed
        case other =>
          throw new SparkException(
            s"$className does not implement ${classOf[UcxShuffleNativeBridge].getName}; " +
              s"got ${Option(other).map(_.getClass.getName).getOrElse("null")}")
      }
    } catch {
      case e: SparkException =>
        throw e
      case e: ClassNotFoundException =>
        throw new SparkException(
          s"Unable to load Gluten UCX shuffle native bridge $className. Ensure the Velox " +
            "backend jar is on both driver and executor classpaths, or set " +
            s"$BridgeClassConf to a custom implementation.",
          e
        )
      case e: Throwable =>
        throw new SparkException(
          s"Unable to instantiate Gluten UCX shuffle native bridge $className",
          e)
    }
  }
}
