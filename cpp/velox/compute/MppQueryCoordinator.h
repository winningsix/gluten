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

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "substrait/SubstraitToVeloxPlan.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/core/PlanFragment.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/SerializedPage.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/Task.h"

namespace gluten {

/// Describes a single plan fragment to be executed as a Velox Task.
/// The root fragment is the one whose id never appears as a
/// producerFragmentId in any MppExchangeSpec. Its output is consumed by
/// the coordinator via OutputBufferManager::getData().
struct MppFragmentSpec {
  /// Fragment identifier. Fragments are numbered contiguously starting
  /// at 0; the root is NOT necessarily id 0 (Scala-side emission typically
  /// puts leaf/producer fragments first and the root last).
  int32_t id;

  /// The Velox plan fragment (plan tree + execution strategy).
  facebook::velox::core::PlanFragment planFragment;

  /// Number of driver threads for this fragment.
  /// For replicated consumers (replicas > 1) this is interpreted per-replica;
  /// today we override to 1/replica when replicated. TODO: scale with N/cores.
  int32_t numDrivers{1};

  /// Scan split information for table scan nodes in this fragment.
  /// Only populated for scan-containing (leaf) fragments.
  std::vector<std::shared_ptr<SplitInfo>> scanInfos;
  std::vector<facebook::velox::core::PlanNodeId> scanNodeIds;
  /// Connector IDs for each scan node (e.g., "test-hive" or "cudf-hive").
  /// Must match the TableScanNode's connector ID in the plan.
  std::vector<std::string> scanConnectorIds;
};

/// Describes a data exchange between two fragments.
/// The consumer fragment has an ExchangeNode whose input comes from
/// the producer fragment's PartitionedOutput.
struct MppExchangeSpec {
  /// Exchange identifier.
  int32_t id;

  /// Index into fragmentSpecs_ for the producing fragment.
  int32_t producerFragmentId;

  /// Index into fragmentSpecs_ for the consuming fragment.
  int32_t consumerFragmentId;

  /// PlanNodeId of the ExchangeNode in the consumer fragment's plan tree.
  /// Used as the split target when wiring RemoteConnectorSplits.
  std::string exchangeNodeId;

  /// Number of output partitions for this exchange.
  int32_t numPartitions{1};

  /// Partitioning type, mirrored from Scala ExchangeSpec.exchangeType.
  /// One of "HASH", "ROUND_ROBIN", "SINGLE", "RANGE", "BROADCAST".
  /// Drives which PartitionFunctionSpec the producer's PartitionedOutputNode
  /// uses (see MppJniWrapper.cc). Default "ROUND_ROBIN" for backward compat.
  std::string partitionType{"ROUND_ROBIN"};

  /// For partitioned exchanges (HASH, RANGE), the column indices of the
  /// partitioning keys in the producer's output schema. Used directly as
  /// Velox keyChannels for HashPartitionFunctionSpec.
  std::vector<int32_t> partitionKeyIndices;
};

/// Coordinates execution of multiple Velox Task fragments within a single
/// process, wiring them together via OutputBufferManager for streaming
/// data exchange.
///
/// This is architecturally equivalent to Presto's AllAtOnceExecutionSchedule:
/// all fragments are started concurrently, and exchange data flows through
/// the existing OutputBufferManager + ExchangeClient mechanism.
///
/// Usage:
///   auto coord = MppQueryCoordinator::create(queryId, fragments, exchanges,
///       queryCtx, executor);
///   coord->start();
///   while (auto batch = coord->next()) {
///     // process batch
///   }
///   coord->waitForCompletion();
class MppQueryCoordinator {
 public:
  /// Factory method.
  static std::shared_ptr<MppQueryCoordinator> create(
      const std::string& queryId,
      std::vector<MppFragmentSpec> fragments,
      std::vector<MppExchangeSpec> exchanges,
      std::shared_ptr<facebook::velox::core::QueryCtx> queryCtx,
      folly::Executor* executor);

  /// Launch all fragments concurrently (all-stages-up scheduling).
  /// Each fragment becomes a Velox Task running in parallel mode.
  /// Exchange wiring is done via RemoteConnectorSplits after all tasks start.
  void start();

  /// Get the next batch of output from the root fragment (the one with
  /// no downstream consumer in the exchange graph). The root fragment
  /// must have a PartitionedOutput node with 1 destination.
  /// Returns nullptr when no more data is available.
  facebook::velox::RowVectorPtr next();

  /// Check if all fragments have reached a terminal state
  /// (finished, failed, aborted, or canceled).
  bool isFinished() const;

  /// Abort all running fragments and wait (bounded) for each one to reach
  /// a terminal state. Calls Task::requestAbort() on every tracked Task that
  /// is not already terminal, then blocks on each Task's
  /// taskCompletionFuture() up to `perTaskTimeout`. If a task fails to reach
  /// terminal within the timeout we log loudly and continue — abort() never
  /// blocks the caller indefinitely. This is the path that releases all
  /// per-task MemoryPool reservations; without it, JVM shutdown can hit
  /// MemoryManager::removePool() VELOX_CHECK(reservedBytes==0) and abort.
  ///
  /// Idempotent: safe to call multiple times. Safe to call before start()
  /// (no-op).
  void abort(
      std::chrono::milliseconds perTaskTimeout =
          std::chrono::milliseconds(10000));

  /// Block until all fragments have reached a terminal state.
  /// Throws if any fragment ended with an error.
  void waitForCompletion();

  ~MppQueryCoordinator();

 private:
  MppQueryCoordinator(
      std::string queryId,
      std::vector<MppFragmentSpec> fragments,
      std::vector<MppExchangeSpec> exchanges,
      std::shared_ptr<facebook::velox::core::QueryCtx> queryCtx,
      folly::Executor* executor);

  /// Build the task ID string for a given fragment + replica index.
  /// Every Task ID carries a replica suffix, including single-replica
  /// fragments (replicaIdx=0), so exchange wiring treats all fragments
  /// uniformly (no special case for replicated vs non-replicated).
  std::string makeTaskId(int32_t fragmentId, int32_t replicaIdx) const;

  /// Check if a task state is terminal.
  static bool isTerminalState(facebook::velox::exec::TaskState state);

  /// Fetch the next page of serialized data from the root task's output buffer.
  /// Returns true if data was fetched, false if at end-of-stream.
  /// Populates `pages` with the received SerializedPageBase objects. Wraps
  /// OutputBufferManager::getData() (IOBuf-callback in IBM-baseline) by
  /// constructing PrestoSerializedPage around each IOBuf so the consumer
  /// path (prepareStreamForDeserialize -> VectorStreamGroup::read) is
  /// unchanged.
  bool fetchNextOutputPage(
      std::vector<std::unique_ptr<facebook::velox::exec::SerializedPageBase>>&
          pages);

  std::string queryId_;
  std::vector<MppFragmentSpec> fragmentSpecs_;
  std::vector<MppExchangeSpec> exchangeSpecs_;
  std::shared_ptr<facebook::velox::core::QueryCtx> queryCtx_;
  folly::Executor* executor_;

  /// Id of the fragment whose output is the final query result (the one
  /// that is never a producer in any exchange). Computed in the
  /// constructor from exchangeSpecs_.
  int32_t rootFragmentId_{-1};

  /// Physical Tasks per fragment, indexed as fragmentTasks_[fragId][replicaIdx].
  /// Inner size = fragmentReplicaCount_[fragId]. For fragments consuming an
  /// N-partition exchange inner size is N; for leaf/non-consumer fragments
  /// inner size is 1. Each replica i carries destination=i at Task::create.
  std::vector<std::vector<std::shared_ptr<facebook::velox::exec::Task>>>
      fragmentTasks_;

  /// Per-fragment replica count. Derived at start() from inbound exchanges'
  /// numPartitions. Fragments with no inbound exchange have count = 1.
  std::vector<int32_t> fragmentReplicaCount_;

  bool started_{false};
  bool noMoreData_{false};

  /// Guards the abort path so concurrent abort() callers (e.g. JNI explicit
  /// close racing with destructor) don't double-issue requestAbort or stomp
  /// on each other's wait loops. fragmentTasks_ itself is only mutated in
  /// start() (single-threaded, before any abort path can fire) so it does not
  /// need the mutex on the read side.
  mutable std::mutex abortMutex_;
  bool aborted_{false};

  /// Output buffer reading state for the root fragment. When root is
  /// replicated we track per-replica sequence + atEnd. Drain strategy is
  /// selected at start() time based on root's inbound exchange type.
  std::shared_ptr<facebook::velox::exec::OutputBufferManager> bufferManager_;
  std::vector<int64_t> rootOutputSequence_;
  std::vector<bool> rootReplicaAtEnd_;
  int32_t rootFetchCursor_{0};
  /// True for RANGE (order-preserving) drain; false for round-robin.
  bool rootDrainSequential_{false};

  /// Watchdog thread periodically (every 5s after start) logs state of
  /// every (fragId, replicaIdx) Task so we can diagnose where the pipeline
  /// is stalling. Runs until destructor.
  std::thread watchdogThread_;
  std::atomic<bool> watchdogStop_{false};

  /// Leaf memory pool for deserializing pages in next(). Velox requires
  /// allocations to happen on leaf pools, not the aggregate root returned
  /// by queryCtx_->pool(). Created lazily on first next() call.
  std::shared_ptr<facebook::velox::memory::MemoryPool> deserializePool_;
};

} // namespace gluten
