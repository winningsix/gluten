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
 */
public class TaskWallTimeTracker {
  private static final Logger LOG = LoggerFactory.getLogger(TaskWallTimeTracker.class);

  private static final ThreadLocal<TaskWallTimeTracker> INSTANCE =
      ThreadLocal.withInitial(TaskWallTimeTracker::new);

  // genFirstStageIterator / genFinalStageIterator total
  public long planBuildNanos;
  // nativeHasNext total across all iterators
  public long nativeHasNextNanos;
  // nativeNext total across all iterators
  public long nativeNextNanos;
  // ColumnarBatches.load (Arrow C Data import)
  public long arrowImportNanos;
  // ColumnarShuffleReader.read() setup
  public long shuffleReadInitNanos;
  // genBroadcastBuildSideIterator
  public long broadcastBuildNanos;
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

  public static TaskWallTimeTracker get() {
    return INSTANCE.get();
  }

  public static void reset() {
    INSTANCE.remove();
  }

  public void logAndReset(int stageId, long taskAttemptId) {
    LOG.warn(
        "[TASK_TIMING] stageId={} taskAttemptId={}"
            + " planBuildNanos={}"
            + " nativeHasNextNanos={}"
            + " nativeNextNanos={}"
            + " arrowImportNanos={}"
            + " shuffleReadInitNanos={}"
            + " broadcastBuildNanos={}"
            + " shuffleWriteJniNanos={}"
            + " shuffleWriteStopNanos={}"
            + " shuffleWriteMetaNanos={}"
            + " taskCleanupNanos={}"
            + " nativeHasNextCalls={}"
            + " nativeNextCalls={}"
            + " arrowImportCalls={}",
        stageId,
        taskAttemptId,
        planBuildNanos,
        nativeHasNextNanos,
        nativeNextNanos,
        arrowImportNanos,
        shuffleReadInitNanos,
        broadcastBuildNanos,
        shuffleWriteJniNanos,
        shuffleWriteStopNanos,
        shuffleWriteMetaNanos,
        taskCleanupNanos,
        nativeHasNextCalls,
        nativeNextCalls,
        arrowImportCalls);
    reset();
  }
}
