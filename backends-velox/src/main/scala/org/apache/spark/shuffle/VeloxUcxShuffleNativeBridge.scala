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

import org.apache.gluten.backendsapi.BackendsApiManager
import org.apache.gluten.columnarbatch.ColumnarBatches
import org.apache.gluten.memory.arrow.alloc.ArrowBufferAllocators
import org.apache.gluten.runtime.Runtimes
import org.apache.gluten.utils.ArrowAbiUtil
import org.apache.gluten.vectorized.UcxShuffleJniWrapper

import org.apache.spark.{SparkConf, SparkException}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.utils.SparkSchemaUtil
import org.apache.spark.sql.vectorized.ColumnarBatch

import com.fasterxml.jackson.databind.ObjectMapper
import org.apache.arrow.c.ArrowSchema

import java.nio.charset.StandardCharsets
import java.util.concurrent.ConcurrentHashMap

private[spark] class VeloxUcxShuffleNativeBridge(conf: SparkConf)
  extends UcxShuffleNativeBridge
  with Logging {

  def this() = this(new SparkConf())

  private val mapper = new ObjectMapper()
  private val readerJnis = new ConcurrentHashMap[Long, UcxShuffleJniWrapper]()

  private def jni(): UcxShuffleJniWrapper = {
    UcxShuffleJniWrapper.create(
      Runtimes.contextInstance(BackendsApiManager.getBackendName, "VeloxUcxShuffle"))
  }

  private def readerJni(readerHandle: Long): UcxShuffleJniWrapper = {
    val wrapper = readerJnis.get(readerHandle)
    if (wrapper == null) {
      throw new SparkException(s"Unknown UCX shuffle reader handle $readerHandle")
    }
    wrapper
  }

  override def localShuffleEndpointPort(): Int = {
    val listenerPort = jni().nativeGetListenerPort()
    if (listenerPort > 3) {
      listenerPort - 3
    } else {
      listenerPort
    }
  }

  override def openWriter(
      endpoint: UcxShuffleEndpoint,
      numPartitions: Int,
      partitioningName: String,
      startPartitionId: Int): Long = {
    val writerHandle = jni().nativeOpenWriter(
      endpoint.executorId,
      endpoint.host,
      endpoint.ucxPort,
      endpoint.shuffleId,
      endpoint.mapId,
      endpoint.attemptId,
      endpoint.nativeTaskId,
      endpoint.deviceId,
      endpoint.epoch,
      numPartitions,
      partitioningName,
      startPartitionId
    )
    logInfo(
      s"Opened UCX incremental shuffle writer handle=$writerHandle " +
        s"shuffleId=${endpoint.shuffleId} mapId=${endpoint.mapId} " +
        s"attemptId=${endpoint.attemptId} partitions=$numPartitions " +
        s"partitioning=$partitioningName startPartitionId=$startPartitionId " +
        s"endpoint=$endpoint")
    writerHandle
  }

  override def writeBatch(writerHandle: Long, partitionId: Int, batch: ColumnarBatch): Unit = {
    val batchHandle = ColumnarBatches.getNativeHandle(BackendsApiManager.getBackendName, batch)
    jni().nativeWrite(writerHandle, partitionId, batchHandle, batch.numRows())
  }

  override def closeWriter(writerHandle: Long, success: Boolean): Unit = {
    jni().nativeCloseWriter(writerHandle, success)
  }

  override def openReader(
      shuffleId: Int,
      reducePartitionId: Int,
      outputSchema: StructType,
      endpoints: Seq[UcxShuffleEndpoint]): Long = {
    val allocator = ArrowBufferAllocators.contextInstance()
    val cSchema = ArrowSchema.allocateNew(allocator)
    try {
      ArrowAbiUtil.exportSchema(
        allocator,
        SparkSchemaUtil.toArrowSchema(outputSchema, SQLConf.get.sessionLocalTimeZone),
        cSchema)
      val wrapper = jni()
      val readerHandle = wrapper.nativeOpenReader(
        shuffleId,
        reducePartitionId,
        encodeEndpoints(endpoints),
        cSchema.memoryAddress())
      readerJnis.put(readerHandle, wrapper)
      logInfo(
        s"Opened UCX incremental shuffle reader handle=$readerHandle shuffleId=$shuffleId " +
          s"reducePartition=$reducePartitionId initialEndpoints=${endpoints.size}")
      readerHandle
    } finally {
      cSchema.close()
    }
  }

  override def addReaderEndpoints(readerHandle: Long, endpoints: Seq[UcxShuffleEndpoint]): Unit = {
    if (endpoints.nonEmpty) {
      readerJni(readerHandle).nativeAddReaderEndpoints(readerHandle, encodeEndpoints(endpoints))
    }
  }

  override def noMoreReaderEndpoints(readerHandle: Long): Unit = {
    readerJni(readerHandle).nativeNoMoreReaderEndpoints(readerHandle)
  }

  override def nextBatch(readerHandle: Long): Option[ColumnarBatch] = {
    val batchHandle = readerJni(readerHandle).nativeNextBatch(readerHandle)
    if (batchHandle == 0L) {
      None
    } else {
      Some(ColumnarBatches.create(batchHandle))
    }
  }

  override def closeReader(readerHandle: Long): Unit = {
    val wrapper = readerJnis.remove(readerHandle)
    if (wrapper != null) {
      wrapper.nativeCloseReader(readerHandle)
    }
  }

  private def encodeEndpoints(endpoints: Seq[UcxShuffleEndpoint]): Array[Byte] = {
    val root = mapper.createObjectNode()
    root.put("version", 1)
    val array = root.putArray("endpoints")
    endpoints.sortBy(e => (e.mapId, e.attemptId)).foreach {
      endpoint =>
        val node = array.addObject()
        node.put("executorId", endpoint.executorId)
        node.put("host", endpoint.host)
        node.put("ucxPort", endpoint.ucxPort)
        node.put("shuffleId", endpoint.shuffleId)
        node.put("mapId", endpoint.mapId)
        node.put("attemptId", endpoint.attemptId)
        node.put("nativeTaskId", endpoint.nativeTaskId)
        node.put("deviceId", endpoint.deviceId)
        node.put("epoch", endpoint.epoch)
    }
    mapper.writeValueAsString(root).getBytes(StandardCharsets.UTF_8)
  }
}
