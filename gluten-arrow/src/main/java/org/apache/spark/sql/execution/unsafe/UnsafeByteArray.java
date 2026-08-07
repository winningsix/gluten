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
package org.apache.spark.sql.execution.unsafe;

import org.apache.gluten.memory.arrow.alloc.ArrowBufferAllocators;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoSerializable;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.apache.arrow.memory.ArrowBuf;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.io.BufferedOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/** A serializable unsafe byte array. */
public class UnsafeByteArray implements Externalizable, KryoSerializable {
  // Large cache pages are streamed to and from DiskStore.  The previous 8 KiB
  // bounce buffer made hundreds of thousands of ArrowBuf/JVM calls per page
  // and dominated materialize time.  One MiB remains bounded while reaching
  // practical host-memory and NVMe throughput.
  private static final int STREAM_CHUNK_SIZE = 1024 * 1024;
  private static final long CUDF_CACHE_MAGIC = 0x314548434143464EL;
  private static final int CUDF_CACHE_PARQUET_VERSION = 6;
  private static final int CUDF_CACHE_HEADER_SIZE = 32;
  private ArrowBuf buffer;
  private long size;
  // DiskStore consumes a CachedColumnarBatch exactly once. Marking that
  // staging page lets Java serialization return its Arrow allocation as soon
  // as ObjectOutputStream has copied the last byte. Keep the default false:
  // broadcast/build-side instances can be serialized more than once.
  private transient boolean releaseAfterExternalWrite;

  UnsafeByteArray(ArrowBuf buffer, long size) {
    this.buffer = buffer;
    this.buffer.getReferenceManager().retain();
    this.size = size;
  }

  public UnsafeByteArray() {}

  public long address() {
    return buffer.memoryAddress();
  }

  public long size() {
    return size;
  }

  public UnsafeByteArray releaseAfterExternalWrite() {
    releaseAfterExternalWrite = true;
    return this;
  }

  public void release() {
    if (buffer != null) {
      buffer.close();
      buffer = null;
      size = 0;
    }
  }

  /** Write a bounded range to a standalone file without creating a JVM-sized byte array. */
  public void writeRangeToFile(String file, long offset, long length) throws IOException {
    if (offset < 0 || length < 0 || offset + length > size) {
      throw new IllegalArgumentException(
          "Invalid UnsafeByteArray file range: offset=" + offset + ", length=" + length
              + ", size=" + size);
    }
    Path path = Paths.get(file);
    Files.createDirectories(path.getParent());
    try (BufferedOutputStream out = new BufferedOutputStream(
        Files.newOutputStream(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE),
        STREAM_CHUNK_SIZE)) {
      byte[] tmp = new byte[STREAM_CHUNK_SIZE];
      long remaining = length;
      long sourceOffset = offset;
      while (remaining > 0) {
        int chunk = (int) Math.min(STREAM_CHUNK_SIZE, remaining);
        buffer.getBytes(sourceOffset, tmp, 0, chunk);
        out.write(tmp, 0, chunk);
        sourceOffset += chunk;
        remaining -= chunk;
      }
    }
  }

  /**
   * Return the offset of an ordinary Parquet payload inside the native cache envelope, or -1 when
   * this buffer uses the Velox/packed fallback format.
   */
  public long parquetPayloadOffset() {
    if (buffer == null || size < CUDF_CACHE_HEADER_SIZE) {
      return -1;
    }
    long magic = readLittleEndianLong(0);
    int version = readLittleEndianInt(8);
    long metadataSize = readLittleEndianLong(16);
    long parquetSize = readLittleEndianLong(24);
    long offset = CUDF_CACHE_HEADER_SIZE + metadataSize;
    if (magic != CUDF_CACHE_MAGIC
        || version != CUDF_CACHE_PARQUET_VERSION
        || metadataSize < 0
        || parquetSize < 8
        || offset < CUDF_CACHE_HEADER_SIZE
        || offset > size
        || parquetSize > size - offset) {
      return -1;
    }
    return hasParquetMagic(offset) && hasParquetMagic(offset + parquetSize - 4) ? offset : -1;
  }

  public long parquetPayloadSize() {
    return parquetPayloadOffset() < 0 ? -1 : readLittleEndianLong(24);
  }

  private boolean hasParquetMagic(long offset) {
    return buffer.getByte(offset) == 'P'
        && buffer.getByte(offset + 1) == 'A'
        && buffer.getByte(offset + 2) == 'R'
        && buffer.getByte(offset + 3) == '1';
  }

  private int readLittleEndianInt(long offset) {
    int value = 0;
    for (int index = 0; index < Integer.BYTES; ++index) {
      value |= (buffer.getByte(offset + index) & 0xff) << (index * Byte.SIZE);
    }
    return value;
  }

  private long readLittleEndianLong(long offset) {
    long value = 0;
    for (int index = 0; index < Long.BYTES; ++index) {
      value |= ((long) buffer.getByte(offset + index) & 0xffL) << (index * Byte.SIZE);
    }
    return value;
  }

  // ------------ KryoSerializable ------------

  @Override
  public void write(Kryo kryo, Output output) {
    // write length first
    output.writeLong(size);

    // stream bytes out of ArrowBuf
    final int chunkSize = STREAM_CHUNK_SIZE;
    byte[] tmp = new byte[chunkSize];

    long remaining = size;
    int index = 0;
    while (remaining > 0) {
      int chunk = (int) Math.min(chunkSize, remaining);
      buffer.getBytes(index, tmp, 0, chunk);
      output.write(tmp, 0, chunk);
      index += chunk;
      remaining -= chunk;
    }
  }

  @Override
  public void read(Kryo kryo, Input input) {
    // read length
    this.size = input.readLong();

    if (size > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("UnsafeByteArray size too large: " + size);
    }

    // allocate ArrowBuf
    this.buffer = ArrowBufferAllocators.globalInstance().buffer((int) size);

    // stream bytes into ArrowBuf
    final int chunkSize = STREAM_CHUNK_SIZE;
    byte[] tmp = new byte[chunkSize];

    long remaining = size;
    int index = 0;
    while (remaining > 0) {
      int chunk = (int) Math.min(chunkSize, remaining);
      input.readBytes(tmp, 0, chunk);
      buffer.setBytes(index, tmp, 0, chunk);
      index += chunk;
      remaining -= chunk;
    }
  }

  // ------------ Externalizable ------------

  @Override
  public void writeExternal(ObjectOutput out) throws IOException {
    try {
      // write length first
      out.writeLong(size);

      final int chunkSize = STREAM_CHUNK_SIZE;
      byte[] tmp = new byte[chunkSize];

      long remaining = size;
      int index = 0;
      while (remaining > 0) {
        int chunk = (int) Math.min(chunkSize, remaining);
        buffer.getBytes(index, tmp, 0, chunk);
        out.write(tmp, 0, chunk);
        index += chunk;
        remaining -= chunk;
      }
    } finally {
      if (releaseAfterExternalWrite) {
        release();
      }
    }
  }

  @Override
  public void readExternal(ObjectInput in) throws IOException {
    this.size = in.readLong();

    if (size > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("UnsafeByteArray size too large: " + size);
    }

    this.buffer = ArrowBufferAllocators.globalInstance().buffer((int) size);

    final int chunkSize = STREAM_CHUNK_SIZE;
    byte[] tmp = new byte[chunkSize];

    long remaining = size;
    int index = 0;
    while (remaining > 0) {
      int chunk = (int) Math.min(chunkSize, remaining);
      // ObjectInput extends DataInput, so we can use readFully
      in.readFully(tmp, 0, chunk);
      buffer.setBytes(index, tmp, 0, chunk);
      index += chunk;
      remaining -= chunk;
    }
  }

  /**
   * It's needed once the broadcast variable is garbage collected. Since now, we don't have an
   * elegant way to free the underlying memory in off-heap.
   *
   * <p>Since: https://github.com/apache/incubator-gluten/pull/8127.
   */
  public void finalize() throws Throwable {
    release();
    super.finalize();
  }
}
