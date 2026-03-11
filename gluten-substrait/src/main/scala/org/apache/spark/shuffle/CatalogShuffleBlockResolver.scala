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

import org.apache.gluten.vectorized.ShufflePayloadCatalogJniWrapper

import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.network.buffer.{ManagedBuffer, NettyManagedBuffer, NioManagedBuffer}
import org.apache.spark.storage.{BlockId, ShuffleBlockId}

import io.netty.buffer.Unpooled

import java.nio.ByteBuffer

class CatalogShuffleBlockResolver(conf: SparkConf)
  extends IndexShuffleBlockResolver(conf)
  with Logging {

  private val emptyBuf: ManagedBuffer =
    new NioManagedBuffer(ByteBuffer.allocate(0))

  override def getBlockData(blockId: BlockId, dirs: Option[Array[String]]): ManagedBuffer = {
    blockId match {
      case sid: ShuffleBlockId =>
        val segments =
          ShufflePayloadCatalogJniWrapper
            .resolveBlockDirect(sid.shuffleId, sid.mapId, sid.reduceId)
        if (segments == null) {
          return super.getBlockData(blockId, dirs)
        }
        if (segments.isEmpty) {
          return emptyBuf
        }
        // Wrap each DirectByteBuffer as a Netty ByteBuf
        // and compose them. Zero copy.
        val bufs = segments.map(Unpooled.wrappedBuffer(_))
        val composite =
          Unpooled.wrappedBuffer(bufs: _*)
        new NettyManagedBuffer(composite)
      case _ =>
        super.getBlockData(blockId, dirs)
    }
  }

  def unregisterCatalogShuffle(shuffleId: Int): Unit = {
    try {
      ShufflePayloadCatalogJniWrapper
        .unregisterShuffle(shuffleId)
    } catch {
      case e: Exception =>
        logWarning(
          s"Failed to unregister catalog shuffle " +
            s"$shuffleId",
          e)
    }
  }
}
