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

#include <memory>
#include <string>
#include <vector>

#include "substrait/SubstraitToVeloxPlan.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/core/PlanFragment.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Exchange.h"
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

  /// Destination partition index for PartitionedOutput.
  /// Typically 0 for the root fragment.
  int32_t destination{0};

  /// Number of driver threads for this fragment.
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

  /// Abort all running fragments. Non-blocking; use waitForCompletion()
  /// to block until all tasks have actually stopped.
  void abort();

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

  /// Build the task ID string for a given fragment.
  std::string makeTaskId(int32_t fragmentId) const;

  /// Check if a task state is terminal.
  static bool isTerminalState(facebook::velox::exec::TaskState state);

  /// Fetch the next page of serialized data from the root task's output buffer.
  /// Returns true if data was fetched, false if at end-of-stream.
  /// Populates `iobufs` with the received pages.
  bool fetchNextOutputPage(
      std::vector<std::unique_ptr<folly::IOBuf>>& iobufs);

  std::string queryId_;
  std::vector<MppFragmentSpec> fragmentSpecs_;
  std::vector<MppExchangeSpec> exchangeSpecs_;
  std::shared_ptr<facebook::velox::core::QueryCtx> queryCtx_;
  folly::Executor* executor_;

  /// Id of the fragment whose output is the final query result (the one
  /// that is never a producer in any exchange). Computed in the
  /// constructor from exchangeSpecs_.
  int32_t rootFragmentId_{-1};

  /// One Velox Task per fragment, indexed by fragment id.
  std::vector<std::shared_ptr<facebook::velox::exec::Task>> tasks_;
  bool started_{false};
  bool noMoreData_{false};

  /// Output buffer reading state for the root fragment.
  std::shared_ptr<facebook::velox::exec::OutputBufferManager> bufferManager_;
  int64_t outputSequence_{0};

  /// Leaf memory pool for deserializing pages in next(). Velox requires
  /// allocations to happen on leaf pools, not the aggregate root returned
  /// by queryCtx_->pool(). Created lazily on first next() call.
  std::shared_ptr<facebook::velox::memory::MemoryPool> deserializePool_;
};

} // namespace gluten
