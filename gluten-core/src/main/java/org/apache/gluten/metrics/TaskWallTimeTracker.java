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
  // task completion listener cleanup
  public long taskCleanupNanos;

  public int nativeHasNextCalls;
  public int nativeNextCalls;
  public int arrowImportCalls;

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
    // Driver-side validation uses synthetic contexts with stage/task id -1. They are not Spark
    // tasks and can number in the thousands for a large MPP plan, so logging them adds noise and
    // measurable planning overhead. Keep full timing for real executor tasks.
    if (stageId >= 0 && taskAttemptId >= 0) {
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
              + " taskCleanupNanos={}"
              + " nativeHasNextCalls={}"
              + " nativeNextCalls={}"
              + " arrowImportCalls={}",
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
          taskCleanupNanos,
          nativeHasNextCalls,
          nativeNextCalls,
          arrowImportCalls);
    }
    reset();
  }
}
