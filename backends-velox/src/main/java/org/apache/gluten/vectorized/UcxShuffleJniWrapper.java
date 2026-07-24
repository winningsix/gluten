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

public class UcxShuffleJniWrapper implements RuntimeAware {
  private final Runtime runtime;

  private UcxShuffleJniWrapper(Runtime runtime) {
    this.runtime = runtime;
  }

  public static UcxShuffleJniWrapper create(Runtime runtime) {
    return new UcxShuffleJniWrapper(runtime);
  }

  @Override
  public long rtHandle() {
    return runtime.getHandle();
  }

  public native int nativeGetListenerPort();

  public native long nativeOpenWriter(
      String executorId,
      String host,
      int ucxPort,
      int shuffleId,
      long mapId,
      long attemptId,
      String nativeTaskId,
      int deviceId,
      long epoch,
      int numPartitions,
      String partitioningName,
      int startPartitionId);

  public native void nativeWrite(
      long writerHandle, int partitionId, long columnarBatchHandle, int numRows);

  public native void nativeCloseWriter(long writerHandle, boolean success);

  public native boolean nativeWriterNoMoreData(String nativeTaskId);

  public native long[] nativeWriterStats(String nativeTaskId);

  public native long nativeOpenReader(
      int shuffleId, int reducePartitionId, byte[] endpointsJson, long cSchema);

  public native void nativeAddReaderEndpoints(long readerHandle, byte[] endpointsJson);

  public native void nativeNoMoreReaderEndpoints(long readerHandle);

  public native long nativeNextBatch(long readerHandle);

  public native void nativeCloseReader(long readerHandle);
}
