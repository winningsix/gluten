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
package org.apache.gluten.metrics;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Per-task wall-time tracker using ThreadLocal. Accumulates timing across all instrumentation
 * points within a single Spark task, then logs at task completion.
 *
 * <p>Native call nesting: Velox plan execution (outer ColumnarBatchOutIterator) may callback to
 * Java for shuffle read, which itself calls an inner ColumnarBatchOutIterator. To avoid
 * double-counting, only the outermost native call accumulates wall time; nested calls are already
 * included in the outer measurement. {@link #nativeNestingDepth} tracks the current depth.
 */
public class TaskWallTimeTracker {
  private static final Logger LOG = LoggerFactory.getLogger(TaskWallTimeTracker.class);

  private static final ThreadLocal<TaskWallTimeTracker> INSTANCE =
      ThreadLocal.withInitial(TaskWallTimeTracker::new);

  // Envelope: set once at task start (TaskResources.onTaskStart)
  public long taskStartNanos;
  // genFirstStageIterator / genFinalStageIterator total
  public long planBuildNanos;
  // outermost nativeHasNext wall time (excludes nested calls)
  public long nativeHasNextNanos;
  // outermost nativeNext wall time (excludes nested calls)
  public long nativeNextNanos;
  // ColumnarBatches.load (Arrow C Data import)
  public long arrowImportNanos;
  // ColumnarShuffleReader.read() setup
  public long shuffleReadInitNanos;
  // genBroadcastBuildSideIterator
  public long broadcastBuildNanos;
  // native shuffle writer lazy initialization (first batch)
  public long shuffleWriterInitNanos;
  // shuffleWriterJniWrapper.write() per-batch calls
  public long shuffleWriteJniNanos;
  // shuffleWriterJniWrapper.stop() finalization
  public long shuffleWriteStopNanos;
  // writeMetadataFileAndCommit
  public long shuffleWriteMetaNanos;
  // UCX incremental shuffle writer setup.
  public long ucxWriterOpenNanos;
  // Driver-side control-plane wait before a UCX writer starts pushing data.
  public long ucxWriterReadersReadyWaitNanos;
  // UCX incremental shuffle writer native write calls.
  public long ucxWriterWriteBatchNanos;
  // UCX incremental shuffle writer close.
  public long ucxWriterCloseNanos;
  // UCX incremental shuffle reader setup.
  public long ucxReaderOpenNanos;
  // UCX incremental shuffle reader native nextBatch calls, including data wait.
  public long ucxReaderNextBatchNanos;
  // UCX incremental shuffle reader close.
  public long ucxReaderCloseNanos;
  // task completion listener cleanup
  public long taskCleanupNanos;

  public int nativeHasNextCalls;
  public int nativeNextCalls;
  public int arrowImportCalls;
  public int ucxWriterWriteBatchCalls;
  public int ucxReaderNextBatchCalls;

  // Nesting depth for native JNI calls. Only depth-0 calls
  // accumulate wall time to avoid double-counting.
  public int nativeNestingDepth;

  public static TaskWallTimeTracker get() {
    return INSTANCE.get();
  }

  public static void reset() {
    INSTANCE.remove();
  }

  public void logAndReset(int stageId, long taskAttemptId) {
    long taskWallNanos = (taskStartNanos > 0) ? System.nanoTime() - taskStartNanos : 0;
    LOG.warn(
        "[TASK_TIMING] stageId={} taskAttemptId={}"
            + " taskWallNanos={}"
            + " planBuildNanos={}"
            + " nativeHasNextNanos={}"
            + " nativeNextNanos={}"
            + " arrowImportNanos={}"
            + " shuffleReadInitNanos={}"
            + " broadcastBuildNanos={}"
            + " shuffleWriterInitNanos={}"
            + " shuffleWriteJniNanos={}"
            + " shuffleWriteStopNanos={}"
            + " shuffleWriteMetaNanos={}"
            + " ucxWriterOpenNanos={}"
            + " ucxWriterReadersReadyWaitNanos={}"
            + " ucxWriterWriteBatchNanos={}"
            + " ucxWriterCloseNanos={}"
            + " ucxReaderOpenNanos={}"
            + " ucxReaderNextBatchNanos={}"
            + " ucxReaderCloseNanos={}"
            + " taskCleanupNanos={}"
            + " nativeHasNextCalls={}"
            + " nativeNextCalls={}"
            + " arrowImportCalls={}"
            + " ucxWriterWriteBatchCalls={}"
            + " ucxReaderNextBatchCalls={}",
        stageId,
        taskAttemptId,
        taskWallNanos,
        planBuildNanos,
        nativeHasNextNanos,
        nativeNextNanos,
        arrowImportNanos,
        shuffleReadInitNanos,
        broadcastBuildNanos,
        shuffleWriterInitNanos,
        shuffleWriteJniNanos,
        shuffleWriteStopNanos,
        shuffleWriteMetaNanos,
        ucxWriterOpenNanos,
        ucxWriterReadersReadyWaitNanos,
        ucxWriterWriteBatchNanos,
        ucxWriterCloseNanos,
        ucxReaderOpenNanos,
        ucxReaderNextBatchNanos,
        ucxReaderCloseNanos,
        taskCleanupNanos,
        nativeHasNextCalls,
        nativeNextCalls,
        arrowImportCalls,
        ucxWriterWriteBatchCalls,
        ucxReaderNextBatchCalls);
    reset();
  }
}
