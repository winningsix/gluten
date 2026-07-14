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

/**
 * JNI bridge to the C++ MppQueryCoordinator for multi-fragment plan execution with streaming
 * exchange.
 *
 * <p>This enables Velox-native multi-stage query execution (MPP) within a single process. Multiple
 * plan fragments are wired together via Velox's OutputBufferManager / ExchangeClient mechanism.
 *
 * <p>Lifecycle:
 *
 * <ol>
 *   <li>{@link #nativeCreateMppQuery} — parse Substrait plans, build fragments, create coordinator
 *   <li>{@link #nativeStartMppQuery} — launch all fragments concurrently (all-stages-up)
 *   <li>{@link #nativeGetMppOutput} — pull output batches from root fragment (blocking)
 *   <li>{@link #nativeAbortMppQuery} — optionally request cooperative abort without releasing
 *       handle
 *   <li>{@link #nativeCloseMppQuery} — abort remaining tasks and release resources
 * </ol>
 *
 * <p>The JNI methods follow the same pattern as {@link PlanEvaluatorJniWrapper}: the wrapper
 * implements {@link RuntimeAware} so the C++ side can resolve the Velox runtime context via {@code
 * getRuntime(env, wrapper)}.
 */
public class MppQueryJniWrapper implements RuntimeAware {
  private final Runtime runtime;

  private MppQueryJniWrapper(Runtime runtime) {
    this.runtime = runtime;
  }

  public static MppQueryJniWrapper create(Runtime runtime) {
    return new MppQueryJniWrapper(runtime);
  }

  @Override
  public long rtHandle() {
    return runtime.getHandle();
  }

  /**
   * Create an MPP query coordinator with multiple plan fragments and exchanges.
   *
   * @param substraitPlans array of Substrait plan bytes, one per fragment (index = fragment id).
   *     Fragment 0 is the root whose output is returned by {@link #nativeGetMppOutput}.
   * @param numDriversPerFragment number of parallel driver threads per fragment. Length must match
   *     {@code substraitPlans.length}.
   * @param exchangeSpecsJson UTF-8 JSON array describing exchanges, e.g.: {@code
   *     [{"producerFragmentId":1,"consumerFragmentId":0,"exchangeNodeId":"n3"}]}
   * @param mppPeerSpecJson UTF-8 JSON object with the native query id, local peer id, and optional
   *     producer peer endpoints. Empty {@code peers} preserves the single-process local path.
   * @param splitInfosPerFragment serialized scan split info per fragment.
   * @param broadcastSlotIndicesPerFragment for each consumer fragment, the iterator slot indices
   *     occupied by fused broadcasts (sorted ascending). {@code null} or empty when no broadcasts
   *     are fused into that fragment. Outer length equals {@code substraitPlans.length}.
   * @param broadcastIteratorsPerFragment parallel to {@code broadcastSlotIndicesPerFragment}; each
   *     entry is a {@link ColumnarBatchInIterator} over the JVM-backed batches that the C++ side
   *     wraps via {@code makeJniColumnarBatchIterator} and pushes into {@code
   *     placeholderIters[slotIdx]}. Without this the substrait plan's {@code ReadRel(iterator:N)}
   *     for fused broadcasts has no backing iterator and triggers {@code streamIdx N vs size N} OOB
   *     at {@code constructCudfValueStreamNode}.
   * @param replicatedCartesianMaxBuildBytes canonical build-size cap for a query containing a
   *     replicated Cartesian, or zero to leave native nested-loop joins unbounded.
   * @return native handle (opaque jlong) for use with the other methods.
   * @throws RuntimeException on Substrait parse failure or invalid specs.
   */
  public native long nativeCreateMppQuery(
      byte[][] substraitPlans,
      int[] numDriversPerFragment,
      byte[] exchangeSpecsJson,
      byte[] mppPeerSpecJson,
      byte[][][] splitInfosPerFragment,
      int[][] broadcastSlotIndicesPerFragment,
      Object[][] broadcastIteratorsPerFragment,
      long replicatedCartesianMaxBuildBytes);

  /**
   * Build the MPP fragments and return the final Velox plan text for each fragment without starting
   * execution.
   *
   * <p>The returned plans are captured after the same MPP rewrites used by {@link
   * #nativeCreateMppQuery}: ValueStream inputs are replaced with Exchange nodes and every fragment
   * is wrapped with a PartitionedOutput node.
   */
  public native String[] nativeExplainMppQuery(
      byte[][] substraitPlans,
      int[] numDriversPerFragment,
      byte[] exchangeSpecsJson,
      byte[][][] splitInfosPerFragment,
      int[][] broadcastSlotIndicesPerFragment,
      Object[][] broadcastIteratorsPerFragment);

  /**
   * Start all fragments concurrently (all-stages-up scheduling). Must be called exactly once after
   * {@link #nativeCreateMppQuery}.
   *
   * @param handle native handle returned by {@link #nativeCreateMppQuery}.
   * @throws RuntimeException if already started or if a fragment fails to launch.
   */
  public native void nativeStartMppQuery(long handle);

  /**
   * Pull the next output batch from the root fragment.
   *
   * <p>This call may block until data is available or all fragments complete.
   *
   * @param handle native handle returned by {@link #nativeCreateMppQuery}.
   * @return a columnar batch handle (for use with {@code ColumnarBatches.create()}), or 0 when no
   *     more data is available (all fragments finished).
   * @throws RuntimeException if a fragment has failed.
   */
  public native long nativeGetMppOutput(long handle);

  /**
   * Request cooperative abort of every Velox task owned by this MPP coordinator. This does not
   * release {@code handle}; the Spark task thread must still call {@link #nativeCloseMppQuery}.
   *
   * @param handle native handle returned by {@link #nativeCreateMppQuery}.
   */
  public native void nativeAbortMppQuery(long handle);

  /**
   * Close the MPP query: abort any running fragments and release native resources. Must be called
   * exactly once. The caller is responsible for guarding against double-close (same pattern as
   * {@link ColumnarBatchOutIterator}).
   *
   * @param handle native handle returned by {@link #nativeCreateMppQuery}.
   */
  public native void nativeCloseMppQuery(long handle);
}
