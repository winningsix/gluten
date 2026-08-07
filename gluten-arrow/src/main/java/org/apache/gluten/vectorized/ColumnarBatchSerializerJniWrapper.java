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
package org.apache.gluten.vectorized;

import org.apache.gluten.runtime.Runtime;
import org.apache.gluten.runtime.RuntimeAware;

import org.apache.spark.TaskContext;
import org.apache.spark.sql.execution.unsafe.JniUnsafeByteBuffer;
import org.apache.spark.util.TaskCompletionListener;

public class ColumnarBatchSerializerJniWrapper implements RuntimeAware {
  private final Runtime runtime;

  private ColumnarBatchSerializerJniWrapper(Runtime runtime) {
    this.runtime = runtime;
    TaskContext taskContext = TaskContext.get();
    if (taskContext != null
        && "parquet".equals(System.getenv("GLUTEN_CACHE_FORMAT"))
        && "true".equals(System.getenv("GLUTEN_FILE_BACKED_TABLE_CACHE"))) {
      // Standalone Parquet descriptors may be published only after all
      // deferred file writes are durable. Register at the task boundary so
      // file N can drain while file N+1 is encoded without requiring an
      // EMR-version-specific replacement of the Scala cache serializer.
      taskContext.addTaskCompletionListener(
          (TaskCompletionListener)
              context -> {
                long startNanos = System.nanoTime();
                try {
                  drainParquetFileWriters();
                } finally {
                  double elapsedMs = (System.nanoTime() - startNanos) / 1000000.0;
                  System.err.printf(
                      "[CACHE_PARQUET_TASK_DRAIN] partition=%d taskAttemptId=%d wallMs=%.3f%n",
                      context.partitionId(), context.taskAttemptId(), elapsedMs);
                }
              });
    }
  }

  public static ColumnarBatchSerializerJniWrapper create(Runtime runtime) {
    return new ColumnarBatchSerializerJniWrapper(runtime);
  }

  @Override
  public long rtHandle() {
    return runtime.getHandle();
  }

  public native JniUnsafeByteBuffer serialize(long handle);

  /** Serialize several adjacent native batches as one cache page. */
  public native JniUnsafeByteBuffer serializeMany(long[] handles);

  /** Start one stateful, independently readable Parquet cache page. */
  public native long initParquetFileWriter(String path);

  /** Append one bounded native batch and return the current on-disk bytes. */
  public native long appendParquetFileWriter(long writerHandle, long batchHandle);

  /** Concatenate one bounded native micro-batch, append it, and return the current file bytes. */
  public native long appendParquetFileWriterMany(long writerHandle, long[] batchHandles);

  /** Write the footer, close the page, and return its final on-disk bytes. */
  public native long finishParquetFileWriter(long writerHandle);

  /** Wait for all asynchronously finalized Parquet cache pages in this executor. */
  public native void drainParquetFileWriters();

  // Return the native ColumnarBatchSerializer handle
  public native long init(long cSchema);

  public native long deserialize(long serializerHandle, byte[] data);

  // Return the native ColumnarBatch handle using memory address and length
  public native long deserializeDirect(long serializerHandle, long offset, int len);

  // Deserialize only the requested top-level columns. Formats such as
  // Parquet apply this projection during decode rather than after restoring
  // the full cached batch.
  public native long deserializeDirectSelected(
      long serializerHandle, long offset, int len, int[] columnIndices);

  /** Decode a standalone Parquet cache page directly from its file path. */
  public native long deserializeParquetFile(long serializerHandle, String path);

  /** Decode selected top-level columns directly from a standalone Parquet cache page. */
  public native long deserializeParquetFileSelected(
      long serializerHandle, String path, int[] columnIndices);

  public native void close(long serializerHandle);
}
