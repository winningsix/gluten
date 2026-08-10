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

#include "FluxQueryCoordinator.h"

#include "compute/FluxOperatorMetrics.h"

#include <fmt/format.h>
#include <folly/dynamic.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/InlineExecutor.h>
#include <folly/io/IOBuf.h>
#include <folly/json.h>
#include <nvtx3/nvtx3.hpp>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <numeric>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#ifdef GLUTEN_ENABLE_GPU
// Must be included before any header that (transitively) pulls in
// CudfHiveConnectorSplit.h, because that header only forward-declares
// cudf::io::source_info. When this TU instantiates the destructor of
// std::shared_ptr<CudfHiveConnectorSplit>, the compiler needs the full
// definition of cudf::io::source_info to emit unique_ptr's deleter.
#include <cudf/io/types.hpp>
#endif

#include "config/VeloxConfig.h"
#include "cudf/GpuMemoryTracker.h"
#include "iceberg/IcebergPlanConverter.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/OutputBuffer.h"
#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/SerializedPage.h"
#include "velox/exec/Task.h"
#ifdef GLUTEN_ENABLE_GPU
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/ExecutorSplitPrefetch.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#endif
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/vector/VectorStream.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace gluten {

namespace {

/// Task ID prefix for all fragments. Must NOT contain "://" or any other
/// character that breaks URL-path parsing: the producer task's taskId is
/// embedded as a single path component in the inter-fragment exchange URL
/// (`/v1/task/<taskId>/results/<dest>`), and UcxExchangeSource extracts the
/// component via folly::split('/', ...). With "gpu-local://" the slashes
/// shift the components and the producer's queue-manager key
/// (UcxPartitionedOutput uses `this->taskId()` directly) no longer matches
/// what the Acceptor parses out of the URL, so the consumer hangs waiting
/// for data that's already enqueued under a different key.
constexpr const char* kTaskIdPrefix = "gpu-local-";

/// Bound each CPU root-output fetch well below Velox's output-buffer limit.
///
/// OutputBuffer::getData() implicitly acknowledges the previous sequence, but
/// FluxQueryCoordinator::next() returns only one page to Spark at a time.  An
/// unbounded fetch can therefore move an entire 1 GiB output buffer (roughly a
/// thousand pages for wide rows) into pendingRootPages_.  The producer then
/// refills to the buffer limit while Spark drains that oversized local backlog,
/// leaving both sides in prolonged backpressure with no acknowledgement
/// cadence.  Keep the local backlog small enough that getData() advances and
/// releases producer memory regularly.  A single oversized page is still
/// returned by Velox, so this is a batching limit rather than a row-size limit.
constexpr uint64_t kRootOutputFetchMaxBytes = 64ULL << 20;

int envIntOrDefault(const char* name, int defaultValue) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return defaultValue;
  }
  char* end = nullptr;
  auto parsed = std::strtol(value, &end, 10);
  if (end == value) {
    LOG(WARNING) << "Ignoring invalid " << name << "='" << value << "'";
    return defaultValue;
  }
  return static_cast<int>(parsed);
}

void removeTaskOutputState(
    const std::shared_ptr<Task>& task,
    const std::shared_ptr<DefaultOutputBufferManager>& bufferManager,
    std::string_view queryId,
    std::string_view reason) {
  if (task == nullptr) {
    return;
  }
  const auto tid = task->taskId();
  LOG(WARNING) << "FluxQueryCoordinator[" << queryId << "]: removeTask(" << tid << ") reason=" << reason
               << " state=" << static_cast<int>(task->state());
  if (bufferManager != nullptr) {
    try {
      bufferManager->removeTask(tid);
    } catch (const std::exception& e) {
      LOG(ERROR) << "FluxQueryCoordinator[" << queryId << "]: DefaultOutputBufferManager::removeTask(" << tid
                 << ") threw: " << e.what();
    } catch (...) {
      LOG(ERROR) << "FluxQueryCoordinator[" << queryId << "]: DefaultOutputBufferManager::removeTask(" << tid
                 << ") threw unknown exception";
    }
  }
}

bool isSafeTaskIdComponent(const std::string& value) {
  return !value.empty() && value.find('/') == std::string::npos && value.find("://") == std::string::npos;
}

bool fluxOperatorMetricsEnabled() {
  const char* value = std::getenv("GLUTEN_FLUX_OPERATOR_METRICS_ENABLED");
  return value != nullptr && std::string(value) == "1";
}

bool fluxLifecycleLogEnabled() {
  const char* value = std::getenv("GLUTEN_FLUX_LIFECYCLE_LOG_ENABLED");
  return value != nullptr && std::string(value) == "1";
}

#ifdef GLUTEN_ENABLE_GPU
std::string cleanedCudfPath(std::string path) {
  constexpr std::string_view kFilePrefix = "file:";
  constexpr std::string_view kS3APrefix = "s3a:";
  if (path.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
    return path.substr(kFilePrefix.size());
  }
  if (path.compare(0, kS3APrefix.size(), kS3APrefix) == 0) {
    path.erase(kS3APrefix.size() - 2, 1);
  }
  return path;
}
#endif

// NVTX domain for gluten FLUX. Same name as the one in FluxJniWrapper.cc so
// both files emit ranges into the same nsys lane.
struct GlutenFluxDomain {
  static constexpr char const* name{"gluten-flux"};
};

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

FluxQueryCoordinator::FluxQueryCoordinator(
    std::string queryId,
    std::vector<FluxFragmentSpec> fragments,
    std::vector<FluxExchangeSpec> exchanges,
    std::shared_ptr<core::QueryCtx> queryCtx,
    folly::Executor* executor,
    std::optional<common::SpillDiskOptions> spillDiskOpts,
    std::string localPeerId,
    int32_t peerIndex,
    int32_t peerCount)
    : queryId_(std::move(queryId)),
      fragmentSpecs_(std::move(fragments)),
      exchangeSpecs_(std::move(exchanges)),
      queryCtx_(std::move(queryCtx)),
      executor_(executor),
      spillDiskOpts_(std::move(spillDiskOpts)),
      localPeerId_(std::move(localPeerId)),
      peerIndex_(peerIndex),
      peerCount_(peerCount),
      bufferManager_(DefaultOutputBufferManager::getInstanceRef()) {
  VELOX_CHECK(!fragmentSpecs_.empty(), "At least one fragment is required");
  VELOX_CHECK(queryCtx_ != nullptr, "QueryCtx must not be null");
  VELOX_CHECK(executor_ != nullptr, "Executor must not be null");
  VELOX_CHECK(
      isSafeTaskIdComponent(localPeerId_),
      "FLUX local peer id must be non-empty and must not contain '/' or '://', got '{}'",
      localPeerId_);
  VELOX_CHECK_GE(peerIndex_, 0, "FLUX peer index must be non-negative");
  VELOX_CHECK_GT(peerCount_, 0, "FLUX peer count must be positive");
  VELOX_CHECK_LT(peerIndex_, peerCount_, "FLUX peer index {} must be less than peer count {}", peerIndex_, peerCount_);
  if (spillDiskOpts_.has_value()) {
    auto& opts = spillDiskOpts_.value();
    if (!opts.spillDirCreated) {
      VELOX_CHECK_NOT_NULL(
          opts.spillDirCreateCb,
          "FLUX spill root must either exist or provide a create callback");
      opts.spillDirPath = opts.spillDirCreateCb();
      opts.spillDirCreated = true;
      opts.spillDirCreateCb = nullptr;
    }
    VELOX_CHECK(
        !opts.spillDirPath.empty(), "FLUX spill root path must not be empty");
    std::filesystem::create_directories(opts.spillDirPath);
  }

  // Validate that fragment ids form a contiguous 0-based sequence so we can
  // use them as vector indices.
  for (size_t i = 0; i < fragmentSpecs_.size(); ++i) {
    VELOX_CHECK_EQ(
        fragmentSpecs_[i].id,
        static_cast<int32_t>(i),
        "Fragment ids must be contiguous starting from 0, got {} at index {}",
        fragmentSpecs_[i].id,
        i);
  }

  // Compute the root fragment: the one whose id never appears as a
  // producerFragmentId in any exchange. For a single-fragment plan that's
  // trivially fragment 0. For a multi-fragment DAG exactly one fragment
  // has no downstream consumer; any other shape is a planner bug.
  std::vector<bool> isProducer(fragmentSpecs_.size(), false);
  for (auto& exchange : exchangeSpecs_) {
    VELOX_CHECK_GE(exchange.producerFragmentId, 0);
    VELOX_CHECK_LT(static_cast<size_t>(exchange.producerFragmentId), fragmentSpecs_.size());
    isProducer[exchange.producerFragmentId] = true;
  }
  for (size_t i = 0; i < fragmentSpecs_.size(); ++i) {
    if (!isProducer[i]) {
      VELOX_CHECK_EQ(
          rootFragmentId_,
          -1,
          "Multiple root fragments (no downstream consumer): {} and {}",
          rootFragmentId_,
          fragmentSpecs_[i].id);
      rootFragmentId_ = fragmentSpecs_[i].id;
    }
  }
  VELOX_CHECK_NE(rootFragmentId_, -1, "No root fragment (every fragment is a producer; exchange DAG has a cycle)");
#ifdef GLUTEN_ENABLE_GPU
  deviceRootOutput_ = queryCtx_->queryConfig().get<bool>(
      facebook::velox::cudf_velox::CudfConfig::kCudfSkipOutputToVelox,
      false);
#endif
}

std::shared_ptr<FluxQueryCoordinator> FluxQueryCoordinator::create(
    const std::string& queryId,
    std::vector<FluxFragmentSpec> fragments,
    std::vector<FluxExchangeSpec> exchanges,
    std::shared_ptr<core::QueryCtx> queryCtx,
    folly::Executor* executor,
    std::optional<common::SpillDiskOptions> spillDiskOpts,
    std::string localPeerId,
    int32_t peerIndex,
    int32_t peerCount) {
  // Using new + shared_ptr because the constructor is private.
  return std::shared_ptr<FluxQueryCoordinator>(new FluxQueryCoordinator(
      queryId,
      std::move(fragments),
      std::move(exchanges),
      std::move(queryCtx),
      executor,
      std::move(spillDiskOpts),
      std::move(localPeerId),
      peerIndex,
      peerCount));
}

FluxQueryCoordinator::~FluxQueryCoordinator() {
  nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{"coordinator::~destructor"};
  // Stop the watchdog first so it doesn't touch half-destructed state.
  // notify_all wakes the watchdog from its cv.wait_for so join() returns
  // promptly instead of blocking up to 5s for the next tick.
  {
    std::lock_guard<std::mutex> lock(watchdogMutex_);
    watchdogStop_.store(true, std::memory_order_release);
  }
  watchdogCv_.notify_all();
  {
    nvtx3::scoped_range_in<GlutenFluxDomain> r{"coordinator::~destructor:watchdog.join"};
    if (watchdogThread_.joinable()) {
      watchdogThread_.join();
    }
  }
  size_t totalTasks = 0;
  for (auto& replicas : fragmentTasks_) {
    totalTasks += replicas.size();
  }
  LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: destructor entry"
               << " started=" << started_ << " fragments=" << fragmentTasks_.size() << " totalTasks=" << totalTasks
               << " rootFragmentId=" << rootFragmentId_;
  // Per-operator cpu/blocked/peak-mem stats for the surviving tasks, so we
  // can 1:1 compare against pv-cli's plan-with-stats dump. Only print for
  // non-trivial tasks (skip the marker-row tasks with pipelines=2). Gated by
  // VLOG(1) so production runs don't pay the per-task plan-string cost;
  // enable via GLOG_v=1.
  if (started_ && VLOG_IS_ON(1)) {
    for (size_t f = 0; f < fragmentTasks_.size(); ++f) {
      for (auto& task : fragmentTasks_[f]) {
        if (!task)
          continue;
        try {
          const auto ts = task->taskStats();
          if (ts.pipelineStats.size() < 5) {
            continue;
          }
          VLOG(1) << "FluxQueryCoordinator[" << queryId_ << "]: plan-with-stats for fragment " << f
                  << " taskId=" << task->taskId() << " (begin):";
          // Split per-line so glog single-line buffer doesn't truncate
          // the tree below ~64KB; the merged Q8 plan exceeds that.
          const auto planStr = task->printPlanWithStats(true);
          size_t pos = 0;
          while (pos <= planStr.size()) {
            const auto eol = planStr.find('\n', pos);
            const auto lineEnd = eol == std::string::npos ? planStr.size() : eol;
            VLOG(1) << "  STATS[" << task->taskId() << "] " << planStr.substr(pos, lineEnd - pos);
            if (eol == std::string::npos) {
              break;
            }
            pos = eol + 1;
          }
          VLOG(1) << "FluxQueryCoordinator[" << queryId_ << "]: plan-with-stats for fragment " << f
                  << " taskId=" << task->taskId() << " (end)";
        } catch (const std::exception& e) {
          VLOG(1) << "  printPlanWithStats failed: " << e.what();
        }
      }
    }
  }
  if (started_) {
    // Abort any still-running tasks (with bounded wait) and then remove
    // their OutputBuffer entries from the global DefaultOutputBufferManager.
    // Failing to call removeTask() leaks the producer-side buffer (and its
    // pages) into the process-wide manager, which causes subsequent FLUX
    // queries in the same JVM to hang because ExchangeClient state there
    // is not cleanly reset. abort() is idempotent: if the JVM-side close
    // already drove abort, the inner aborted_ guard short-circuits.
    {
      nvtx3::scoped_range_in<GlutenFluxDomain> r{"coordinator::~destructor:abort_call"};
      try {
        abort();
      } catch (...) {
      }
    }
    {
      nvtx3::scoped_range_in<GlutenFluxDomain> removeTaskRange{"coordinator::~destructor:removeTaskFromOutputBuffer"};
      for (auto& replicas : fragmentTasks_) {
        for (auto& task : replicas) {
          removeTaskOutputState(task, bufferManager_, queryId_, "coordinator-destructor");
        }
      }
    }
  }
  // Explicit teardown with NVTX so we can attribute close-phase time to
  // specific resources. Without these, the cost falls into implicit
  // class-member destruction (reverse declaration order) after this NVTX
  // range, invisible in profiles. Order matters: drop the Task vector
  // first (which joins driver threads + frees per-task memory pools),
  // then DefaultOutputBufferManager handle, then per-query deserialize pool,
  // then QueryCtx (which holds the root memory pool).
  {
    nvtx3::scoped_range_in<GlutenFluxDomain> r{"coordinator::~destructor:fragmentTasks.clear"};
    fragmentTasks_.clear();
  }
#ifdef GLUTEN_ENABLE_GPU
  // All split readers are gone after fragmentTasks_.clear(). Release the
  // query-scoped prefetch queue and join its scheduler workers now instead
  // of accumulating one 16-thread scheduler per query until executor exit.
  cudf_velox::connector::hive::ExecutorSplitPrefetch::eraseQuery(
      executor_, queryCtx_->queryId());
#endif
  if (spillDiskOpts_.has_value()) {
    std::error_code error;
    std::filesystem::remove_all(spillDiskOpts_->spillDirPath, error);
    if (error) {
      LOG(ERROR) << "FluxQueryCoordinator[" << queryId_
                 << "]: failed to remove spill root '"
                 << spillDiskOpts_->spillDirPath << "': " << error.message();
    }
  }
  {
    nvtx3::scoped_range_in<GlutenFluxDomain> r{"coordinator::~destructor:bufferManager.reset"};
    bufferManager_.reset();
  }
  {
    nvtx3::scoped_range_in<GlutenFluxDomain> r{"coordinator::~destructor:deserializePool.reset"};
    deserializePool_.reset();
  }
#ifdef GLUTEN_ENABLE_GPU
  // Preserve the query-scoped trim policy before releasing QueryCtx. -1
  // explicitly suppresses the executor environment fallback; -2 keeps it for
  // callers that do not provide the newer QueryConfig key.
  const auto asyncQueryEndTrimBytes = queryCtx_
      ? queryCtx_->queryConfig().cudfAsyncQueryEndTrimBytes()
      : int64_t{-2};
#endif
  {
    nvtx3::scoped_range_in<GlutenFluxDomain> r{"coordinator::~destructor:queryCtx.reset"};
    queryCtx_.reset();
  }
#ifdef GLUTEN_ENABLE_GPU
  // cuda_async_memory_resource deliberately caches its high-water mark. Once
  // all tasks and the QueryCtx are gone, trim unused blocks at the FLUX query
  // boundary so a later query (or UCX's independent cudaMalloc pool) is not
  // starved by memory that has no live RMM owner.
  {
    nvtx3::scoped_range_in<GlutenFluxDomain> r{
        "coordinator::~destructor:trimAsyncMemoryPools"};
    if (asyncQueryEndTrimBytes >= 0) {
      (void)facebook::velox::cudf_velox::trimAsyncMemoryPoolsAtQueryEnd(
          static_cast<std::size_t>(asyncQueryEndTrimBytes));
    } else if (asyncQueryEndTrimBytes == -2) {
      (void)facebook::velox::cudf_velox::trimAsyncMemoryPoolsAtQueryEnd();
    }
  }
#endif
  LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: destructor exit";
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string FluxQueryCoordinator::makeTaskId(int32_t fragmentId, int32_t replicaIdx) const {
  return makeTaskIdForPeer(localPeerId_, fragmentId, replicaIdx);
}

std::string FluxQueryCoordinator::makeTaskIdForPeer(const std::string& peerId, int32_t fragmentId, int32_t replicaIdx)
    const {
  // Root-fragment output is consumed by this coordinator via
  // DefaultOutputBufferManager::getData() (kHttp PartitionedOutput).
  // Non-root fragment edges flow through IBM's UCX / IntraNodeTransfer
  // path; they share the same task-id prefix because routing is decided
  // by adapters, not by the prefix.
  VELOX_CHECK(
      isSafeTaskIdComponent(peerId),
      "FLUX peer id must be non-empty and must not contain '/' or '://', got '{}'",
      peerId);
  return fmt::format("{}{}-{}-{}-p{}", kTaskIdPrefix, queryId_, peerId, fragmentId, replicaIdx);
}

bool FluxQueryCoordinator::isTerminalState(TaskState state) {
  return state == TaskState::kFinished || state == TaskState::kCanceled || state == TaskState::kAborted ||
      state == TaskState::kFailed;
}

// ---------------------------------------------------------------------------
// start()
// ---------------------------------------------------------------------------

void FluxQueryCoordinator::start() {
  nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{"coordinator::start"};
  VELOX_CHECK(!started_, "FluxQueryCoordinator already started");
  started_ = true;
  lifecycleStartTime_ = std::chrono::steady_clock::now();
  const bool lifecycleLogEnabled = fluxLifecycleLogEnabled();
  const auto lifecycleElapsedMs = [this]() -> int64_t {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - lifecycleStartTime_)
        .count();
  };
  if (lifecycleLogEnabled) {
    LOG(WARNING) << "[FLUX_LIFECYCLE] event=query_start elapsedMs=0"
                 << " queryId=" << queryId_ << " peer=" << peerIndex_ << "/" << peerCount_
                 << " fragments=" << fragmentSpecs_.size() << " rootFragment=" << rootFragmentId_;
  }
  LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: start() " << fragmentSpecs_.size()
               << " fragments, rootFragmentId=" << rootFragmentId_ << " peer=" << peerIndex_ << "/" << peerCount_
               << " localPeerId=" << localPeerId_;

  // Write-in-FLUX: detect whether the root fragment is a distributed TableWrite.
  // A write root runs exactly one TableWrite per peer over all HASH/RANGE
  // destinations that peer owns and drains its local commit batch. Non-write
  // RANGE roots use the same peer ownership, but retain sequential local drain
  // so RANGE destinations stay ordered within each Spark output partition.
  const bool rootIsWrite = [&]() {
    std::function<bool(const core::PlanNodePtr&)> hasTableWrite = [&](const core::PlanNodePtr& node) -> bool {
      if (node == nullptr) {
        return false;
      }
      if (std::dynamic_pointer_cast<const core::TableWriteNode>(node) != nullptr) {
        return true;
      }
      for (const auto& source : node->sources()) {
        if (hasTableWrite(source)) {
          return true;
        }
      }
      return false;
    };
    return hasTableWrite(fragmentSpecs_[rootFragmentId_].planFragment.planNode);
  }();
  if (rootIsWrite) {
    LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: root fragment " << rootFragmentId_
                 << " is a distributed TableWrite (1 writer/peer, fan "
                 << "destinations across peers, all peers emit commit)";
  }

  // Phase 1: Create and start a Velox Task for each fragment.
  //
  // Task::create() + Task::start() internally handles:
  //   - initializePartitionOutput() ->
  //     DefaultOutputBufferManager::initializeTask()
  //     (for fragments whose plan ends with PartitionedOutputNode)
  //   - createExchangeClientLocked() -> creates ExchangeClient for any
  //     ExchangeNode in the plan
  //
  // Create all tasks first, inject scan and exchange splits, then start tasks
  // in fragment order (producers precede their consumers). Starting a complex
  // multi-input consumer before its exchange splits are attached can leave all
  // of its pipelines permanently dormant: Q17/Q21 showed six created drivers
  // with zero operator activity while every upstream PartitionedOutput was
  // blocked waiting for a consumer. Producer-first startup still registers
  // each output buffer before the corresponding consumer can fetch it.
  //
  // --- Derive per-fragment driver count from inbound exchanges (Presto-style) ---
  // Presto-aligned task model: 1 task per fragment per worker (here single-node
  // = 1 task per fragment). Within that task, drivers handle the per-partition
  // parallelism: for a HASH/RANGE/ROUND_ROBIN inbound exchange with N partitions,
  // the consumer task has N drivers each pulling one producer-destination split.
  // This replaces the old N-replica model where each replica pinned destination=i.
  //
  // Memory benefit: 1 task with shared memory pool across drivers, not N tasks
  // each with their own pool. Q1 SF1K's 16 replicas × 512MB FilterProject was
  // 16x the budget; now drivers share.
  // BROADCAST inbound exchanges carry numPartitions=1 (one payload replicated
  // to all consumers) and don't drive driver count.
  fragmentReplicaCount_.assign(fragmentSpecs_.size(), 1);
  std::vector<int32_t> consumerInboundPartitions(fragmentSpecs_.size(), 0);
  // Per-fragment flags: HASH can keep all logical destinations while bounding
  // local consumer tasks on one GPU; RANGE keeps one task per destination for
  // ordered drain semantics.
  std::vector<bool> isHashConsumer(fragmentSpecs_.size(), false);
  std::vector<bool> isRangeConsumer(fragmentSpecs_.size(), false);
  for (auto& exchange : exchangeSpecs_) {
    if (exchange.partitionType == "BROADCAST") {
      continue;
    }
    const auto consumer = exchange.consumerFragmentId;
    const auto n = std::max(1, exchange.numPartitions);
    auto& slot = consumerInboundPartitions[consumer];
    if (slot == 0) {
      slot = n;
    } else {
      VELOX_CHECK_EQ(
          slot,
          n,
          "Fragment {} has inbound non-broadcast exchanges with inconsistent "
          "numPartitions ({} vs {}). All HASH/RANGE/ROUND_ROBIN inbound "
          "exchanges must agree.",
          consumer,
          slot,
          n);
    }
    if (exchange.partitionType == "HASH") {
      isHashConsumer[consumer] = true;
    } else if (exchange.partitionType == "RANGE") {
      isRangeConsumer[consumer] = true;
    }
  }
  // Apply consumer fan-out. HASH keeps the logical destination count on the
  // producer buffers, but bounds local tasks by the fragment driver budget so
  // one GPU does not build N replicated hash states. RANGE still uses one task
  // per destination.
  for (size_t i = 0; i < fragmentSpecs_.size(); ++i) {
    if (isRangeConsumer[i]) {
      fragmentReplicaCount_[i] = std::max(1, consumerInboundPartitions[i]);
    } else if (isHashConsumer[i]) {
      const auto logicalDestinations = std::max(1, consumerInboundPartitions[i]);
      const auto localTaskBudget = std::max(1, fragmentSpecs_[i].numDrivers);
      fragmentReplicaCount_[i] = std::min(logicalDestinations, localTaskBudget);
      LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: HASH fragment " << i
                   << " logicalDestinations=" << logicalDestinations
                   << " localConsumerTasks=" << fragmentReplicaCount_[i] << " taskBudget=" << localTaskBudget;
    }
  }

  // Scans cannot be replicated: the scan-split wiring below VELOX_CHECKs that a
  // scan-bearing fragment has exactly 1 task. A fragment that both bears a scan
  // and consumes a HASH/RANGE exchange would otherwise be assigned
  // min(logicalDestinations, numDrivers) > 1 replicas (Presto keeps scan/leaf
  // fragments single-replica). Clamp such fragments back to one replica; the
  // HASH producer keeps its N destination buffers and the single local task
  // drains them (identical to the numDrivers=1 budget case handled above).
  for (size_t i = 0; i < fragmentSpecs_.size(); ++i) {
    if (!fragmentSpecs_[i].scanNodeIds.empty() && fragmentReplicaCount_[i] != 1) {
      LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: scan-bearing fragment " << i << " replicaCount "
                   << fragmentReplicaCount_[i] << " -> 1 (scans wired to single-replica tasks)";
      fragmentReplicaCount_[i] = 1;
    }
  }

  // Distributed write root: exactly one native TableWrite task per peer (one
  // writer per Spark task attempt dir). >1 replica would share the single
  // per-task injected write filename and collide ("File exists"). With one
  // consumer task the RANGE/HASH split wiring (dest % 1 == 0) routes every
  // destination this peer owns to that single task, so it gathers the peer's
  // whole slice into one output file.
  if (rootIsWrite && fragmentReplicaCount_[rootFragmentId_] != 1) {
    LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: write root fragment " << rootFragmentId_
                 << " replicaCount " << fragmentReplicaCount_[rootFragmentId_] << " -> 1 (single writer per peer)";
    fragmentReplicaCount_[rootFragmentId_] = 1;
  }

  const auto baseFragmentReplicaCount = fragmentReplicaCount_;

  std::vector<bool> isBroadcastProducer(fragmentSpecs_.size(), false);
  for (const auto& exchange : exchangeSpecs_) {
    if (exchange.partitionType == "BROADCAST") {
      const auto producer = exchange.producerFragmentId;
      if (producer >= 0 && static_cast<size_t>(producer) < isBroadcastProducer.size()) {
        isBroadcastProducer[producer] = true;
      }
    }
  }

  const auto peerSlots = std::max(1, peerCount_);
  std::vector<double> peerWeights(peerSlots, 1.0);
  if (const auto* weightsEnv = std::getenv("GLUTEN_FLUX_PEER_WEIGHTS")) {
    std::stringstream weightsStream(weightsEnv);
    std::string token;
    size_t index = 0;
    while (std::getline(weightsStream, token, ',') && index < peerWeights.size()) {
      try {
        peerWeights[index] = std::max(0.01, std::stod(token));
      } catch (...) {
        VELOX_FAIL("Invalid GLUTEN_FLUX_PEER_WEIGHTS value '{}' at index {}", token, index);
      }
      ++index;
    }
    VELOX_CHECK_EQ(
        index,
        peerWeights.size(),
        "GLUTEN_FLUX_PEER_WEIGHTS must contain exactly {} comma-separated weights",
        peerWeights.size());
  }
  const auto destinationOwner = [&](int32_t destination, int32_t total) {
    VELOX_CHECK_GT(total, 0);
    const auto totalWeight = std::accumulate(peerWeights.begin(), peerWeights.end(), 0.0);
    const auto target = (static_cast<double>(destination) + 0.5) * totalWeight / total;
    double cumulative = 0;
    for (int32_t peer = 0; peer < peerSlots; ++peer) {
      cumulative += peerWeights[peer];
      if (target < cumulative) {
        return peer;
      }
    }
    return peerSlots - 1;
  };
  // Assign scan splits by estimated bytes, not by file ordinal.  Iceberg
  // tables commonly contain files with very different sizes; j % peerCount
  // gives every peer the same number of files but can leave one GPU processing
  // minutes after the others have gone idle.  Build the same deterministic
  // longest-processing-time schedule in every peer coordinator, then use it
  // both for locality decisions and for actual split wiring below.
  std::unordered_map<const SplitInfo*, std::vector<int32_t>> scanSplitOwners;
  for (const auto& spec : fragmentSpecs_) {
    for (const auto& scanInfo : spec.scanInfos) {
      std::vector<int32_t> owners(scanInfo->paths.size(), 0);
      if (peerCount_ > 1 && !owners.empty()) {
        std::vector<size_t> order(owners.size());
        std::iota(order.begin(), order.end(), 0);
        const auto splitBytes = [&](size_t index) -> uint64_t {
          return index < scanInfo->lengths.size() ? std::max<uint64_t>(1, scanInfo->lengths[index]) : 1;
        };
        std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
          return splitBytes(left) > splitBytes(right);
        });
        std::vector<uint64_t> assignedBytes(peerSlots, 0);
        for (const auto index : order) {
          int32_t leastLoaded = 0;
          for (int32_t peer = 1; peer < peerSlots; ++peer) {
            if (assignedBytes[peer] / peerWeights[peer] < assignedBytes[leastLoaded] / peerWeights[leastLoaded]) {
              leastLoaded = peer;
            }
          }
          owners[index] = leastLoaded;
          assignedBytes[leastLoaded] += splitBytes(index);
        }
      }
      scanSplitOwners.emplace(scanInfo.get(), std::move(owners));
    }
  }
  std::vector<std::vector<size_t>> scanSplitsByPeer(fragmentSpecs_.size(), std::vector<size_t>(peerSlots, 0));
  std::vector<size_t> totalScanSplits(fragmentSpecs_.size(), 0);
  for (const auto& spec : fragmentSpecs_) {
    if (spec.scanInfos.empty()) {
      continue;
    }
    for (const auto& scanInfo : spec.scanInfos) {
      const auto& owners = scanSplitOwners.at(scanInfo.get());
      for (size_t j = 0; j < scanInfo->paths.size(); ++j) {
        const auto splitPeer = owners[j];
        scanSplitsByPeer[spec.id][splitPeer]++;
        totalScanSplits[spec.id]++;
      }
    }
    if (isBroadcastProducer[spec.id]) {
      LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: broadcast scan producer fragment " << spec.id
                   << " split ownership total=" << totalScanSplits[spec.id] << " peer=" << peerIndex_ << "/"
                   << peerCount_ << " localSplits=" << scanSplitsByPeer[spec.id][peerIndex_];
    }
  }

  std::vector<bool> bootstrapBroadcastProducer(fragmentSpecs_.size(), false);
  for (const auto& spec : fragmentSpecs_) {
    bootstrapBroadcastProducer[spec.id] = isBroadcastProducer[spec.id] && !spec.scanInfos.empty() &&
        totalScanSplits[spec.id] <= static_cast<size_t>(peerSlots);
    if (bootstrapBroadcastProducer[spec.id]) {
      LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: bootstrap broadcast scan producer fragment " << spec.id
                   << " before bulk scans (totalSplits=" << totalScanSplits[spec.id] << ", peer=" << peerIndex_ << "/"
                   << peerCount_ << ", localSplits=" << scanSplitsByPeer[spec.id][peerIndex_] << ")";
    }
  }

  const auto peerOwnsExchangeDestination = [&](const FluxExchangeSpec& exchange, int32_t peerIndex) {
    if (peerCount_ <= 1 || exchange.partitionType == "BROADCAST") {
      return true;
    }
    if (exchange.partitionType == "HASH" || exchange.partitionType == "RANGE") {
      for (int dest = 0; dest < std::max(1, exchange.numPartitions); ++dest) {
        if (destinationOwner(dest, std::max(1, exchange.numPartitions)) == peerIndex) {
          return true;
        }
      }
      return false;
    }
    // SINGLE / ROUND_ROBIN consumers are global singletons in this
    // multi-peer coordinator. Only peer0 owns them; downstream exchanges
    // must fetch that producer from peer0 only.
    return peerIndex == 0;
  };
  const auto peerHasFragment = [&](int32_t fragmentId, int32_t peerIndex) {
    if (peerCount_ <= 1 || peerIndex < 0) {
      return true;
    }
    if (fragmentId >= 0 && static_cast<size_t>(fragmentId) < bootstrapBroadcastProducer.size() &&
        bootstrapBroadcastProducer[fragmentId]) {
      // A broadcast build over an empty Iceberg table still needs one producer
      // task to publish EOS.  Dropping the zero-split fragment from every peer
      // leaves its consumer with no exchange endpoint.  Run the empty producer
      // exactly once on peer 0; non-empty tiny producers remain striped only to
      // peers that own an input split.
      if (totalScanSplits[fragmentId] == 0) {
        return peerIndex == 0;
      }
      const auto peerSlot = static_cast<size_t>(peerIndex);
      if (peerSlot >= scanSplitsByPeer[fragmentId].size() || scanSplitsByPeer[fragmentId][peerSlot] == 0) {
        return false;
      }
    }
    // K-way distributed broadcast production: a scan-bearing BROADCAST producer
    // runs on every peer that owns at least one stripe of the leaf (see the
    // Phase 2 split striping below) and broadcasts ONLY its partial. Tiny
    // bootstrap producers with no local split are not materialized; they cannot
    // contribute rows, and starting them only makes downstream consumers wait
    // for empty EOS. Each consumer replica fans in all active producer partials
    // (the Phase 3 BROADCAST wiring already loops over producer endpoints) and
    // its HashBuild merges the k partials into the full build table -- the
    // k-way merge. This replaces the old peer-0-only model where peer 0 alone
    // scanned + built + broadcast the entire build side (serial; ~20x slower on
    // Q5's all-REPLICATE plan). Correctness: every output row of the producer's
    // join carries exactly one row from the striped scan, so the scan stripe
    // cleanly partitions the producer output with no duplication or loss,
    // regardless of the build/probe role of the broadcasts it consumes. This
    // change only affects scan-bearing broadcast producers (the ones the old
    // gate pinned to peer 0); scan-less producers are unaffected -- they always
    // fell through to the inbound-exchange check below and remain gated by
    // whatever non-broadcast (HASH/RANGE/SINGLE) exchange they consume.
    for (const auto& exchange : exchangeSpecs_) {
      if (exchange.consumerFragmentId != fragmentId || exchange.partitionType == "BROADCAST") {
        continue;
      }
      if (!peerOwnsExchangeDestination(exchange, peerIndex)) {
        return false;
      }
    }
    return true;
  };

  // In multi-peer mode the logical HASH/RANGE destinations are already
  // striped across peers.  Do not instantiate the full logical replica set on
  // every peer: with 32 destinations and 32 peers that created 32 tasks per
  // executor (1024 producer endpoints), although each executor owned only one
  // destination.  Besides wasting tasks, downstream broadcast/exchange wiring
  // waited on hundreds of empty endpoints and could deadlock behind output
  // backpressure on complex plans (Q17/Q21).
  const auto replicaCountForPeer =
      [&](int32_t fragmentId, int32_t peerIndex) -> int32_t {
    if (!peerHasFragment(fragmentId, peerIndex)) {
      return 0;
    }
    const auto base = baseFragmentReplicaCount[fragmentId];
    if (peerCount_ <= 1 || base <= 1) {
      return base;
    }
    for (const auto& exchange : exchangeSpecs_) {
      if (exchange.consumerFragmentId != fragmentId ||
          exchange.partitionType == "BROADCAST") {
        continue;
      }
      if (exchange.partitionType != "HASH" &&
          exchange.partitionType != "RANGE") {
        return peerIndex == 0 ? base : 0;
      }
      int32_t ownedDestinations = 0;
      const auto destinations = std::max(1, exchange.numPartitions);
      for (int32_t dest = 0; dest < destinations; ++dest) {
        ownedDestinations += destinationOwner(dest, destinations) == peerIndex ? 1 : 0;
      }
      return std::min(base, ownedDestinations);
    }
    return base;
  };
  if (peerCount_ > 1) {
    for (auto& spec : fragmentSpecs_) {
      const auto localReplicas = replicaCountForPeer(spec.id, peerIndex_);
      if (localReplicas != fragmentReplicaCount_[spec.id]) {
        const auto oldReplicas = fragmentReplicaCount_[spec.id];
        fragmentReplicaCount_[spec.id] = localReplicas;
        LOG(WARNING) << "FluxQueryCoordinator[" << queryId_
                     << "]: fragment " << spec.id
                     << " replicaCount " << oldReplicas << " -> "
                     << localReplicas << " for peer=" << peerIndex_ << "/"
                     << peerCount_;
      }
    }
  }

  // Decide root drain strategy: sequential for RANGE (order-preserving);
  // round-robin otherwise. If the root has no inbound exchange (single
  // fragment query), replica count is 1 and strategy is moot.
  rootDrainSequential_ = false;
  rootProducesOutput_ = fragmentReplicaCount_[rootFragmentId_] > 0;
  for (auto& exchange : exchangeSpecs_) {
    if (exchange.consumerFragmentId == rootFragmentId_) {
      if (exchange.partitionType == "RANGE") {
        rootDrainSequential_ = true;
      }
      if (peerCount_ > 1) {
        if (exchange.partitionType == "HASH" || exchange.partitionType == "RANGE") {
          // FluxNativeQueryRDD partition index is the native peer index. The
          // weighted owner function assigns contiguous destination ranges in
          // peer order, so RANGE remains globally ordered across Spark output
          // partitions while each peer drains its local replicas sequentially.
          rootProducesOutput_ = fragmentReplicaCount_[rootFragmentId_] > 0;
        } else if (exchange.partitionType != "BROADCAST") {
          rootProducesOutput_ = peerIndex_ == 0;
        }
      }
      break; // root consumes at most one exchange
    }
  }
  if (rootIsWrite) {
    // Distributed write: a peer writes (and drains its own commit batch back to
    // VeloxColumnarWriteFilesRDD) iff it owns >=1 destination of the root's
    // inbound exchange. peerHasFragment() above already zeroed replicaCount for
    // peers that own none: a SINGLE/ROUND_ROBIN gather (global top-N / global
    // aggregation / scalar subquery) is peer0-only, while HASH/RANGE is fanned
    // across peers. Mirror that here. Forcing every peer to produce breaks
    // SINGLE-gather writes -- peer!=0 has no slice, so its TableWrite's inbound
    // exchange source blocks forever in WaitingForMetadata (no producer ever
    // targets that destination), hanging the query. HASH/RANGE writes still
    // distribute because each peer owns a fanned destination (replicaCount>0).
    rootProducesOutput_ = fragmentReplicaCount_[rootFragmentId_] > 0;
  }
  if (!rootProducesOutput_) {
    fragmentReplicaCount_[rootFragmentId_] = 0;
  }
  LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: rootFragmentId=" << rootFragmentId_
               << " replicas=" << fragmentReplicaCount_[rootFragmentId_] << " producesOutput=" << rootProducesOutput_
               << " drain=" << (rootDrainSequential_ ? "sequential" : "roundRobin");

  // --- Precompute: is fragment `i` a BROADCAST producer, and if so, what is
  // its consumer fragment's replica count? For those, we call
  // updateOutputBuffers(N, /*noMoreBuffers=*/true) BEFORE task->start() so
  // Velox's kBroadcast OutputBuffer has all N destination slots ready before
  // any driver runs. Otherwise producer enqueues to the default 1-slot
  // buffer and replicas 1..N-1 of the consumer fragment never see data
  // (build-side table is null -> CudfHashJoinBuild fails with
  // "tbl != nullptr"). See plan/issue-broadcast-fanout.md.
  std::vector<int32_t> broadcastFanout(fragmentSpecs_.size(), 0);
  for (const auto& exchange : exchangeSpecs_) {
    if (exchange.partitionType != "BROADCAST") {
      continue;
    }
    const auto producer = exchange.producerFragmentId;
    const auto consumer = exchange.consumerFragmentId;
    if (producer >= 0 && static_cast<size_t>(producer) < broadcastFanout.size()) {
      int32_t consumerTasks = 0;
      if (peerCount_ <= 1) {
        consumerTasks = fragmentReplicaCount_[consumer];
      } else {
        for (int32_t peer = 0; peer < peerCount_; ++peer) {
          consumerTasks += replicaCountForPeer(consumer, peer);
        }
      }
      broadcastFanout[producer] = std::max(1, consumerTasks);
    }
  }

  // --- Create Tasks: one per (fragment, replica) ---
  // Create all Task objects first so exchange splits can be attached to
  // downstream consumers before those consumers start. We start tiny bootstrap
  // broadcast producers ahead of bulk scans below; this keeps their zero-row
  // EOS path from waiting behind large scan/probe drivers.
  fragmentTasks_.assign(fragmentSpecs_.size(), {});
  std::vector<std::vector<int32_t>> fragmentTaskDrivers(fragmentSpecs_.size());
  std::vector<std::vector<int32_t>> fragmentTaskBroadcastFanout(fragmentSpecs_.size());
  std::vector<std::vector<bool>> fragmentTaskStarted(fragmentSpecs_.size());
  for (auto& spec : fragmentSpecs_) {
    const auto replicas = fragmentReplicaCount_[spec.id];
    fragmentTasks_[spec.id].reserve(replicas);
    fragmentTaskDrivers[spec.id].reserve(replicas);
    fragmentTaskBroadcastFanout[spec.id].reserve(replicas);
    fragmentTaskStarted[spec.id].reserve(replicas);
    // TODO: scale per-replica driver count with N and core budget. Default
    // to 1/replica for replicated fragments (matches GpuMultiFragmentTest);
    // preserve Scala-supplied numDrivers for non-replicated fragments.
    // Pin broadcast producers to 1 driver so the kBroadcast
    // OutputBuffer's end-marker fires deterministically on the one
    // noMoreData() call (stock Velox end-marker only fires when ALL drivers
    // have called noMoreData; multi-driver broadcast hangs when any driver
    // receives zero splits or blocks for any other reason).
    // Presto-style: 1 task per fragment, drivers = max(spec.numDrivers,
    // inbound numPartitions). Broadcast producers stay at 1 driver because
    // the kBroadcast OutputBuffer end-marker requires a deterministic single
    // noMoreData() call.
    const auto bcastN = broadcastFanout[spec.id];
    const auto inboundN = consumerInboundPartitions[spec.id];
    // HASH/RANGE consumers route data by task destination, so each replica
    // consumes one bucket. SINGLE/ROUND_ROBIN consumers receive all producer
    // destinations as splits in a single task; keep their local driver count
    // bounded by the Scala-supplied fragment budget instead of inflating it
    // back to the producer destination count.
    const auto perReplicaDrivers = (bcastN > 0) ? 1
        : (rootIsWrite && spec.id == rootFragmentId_) ? 1
        : isRangeConsumer[spec.id] ? 1
        : isHashConsumer[spec.id]
            ? (spec.keyedFinalLocalRepartition
                   ? std::max(1, spec.numDrivers)
                   : (replicas == 1 &&
                              (spec.rightSemiProjectMultiDriverSafe ||
                               spec.innerJoinMultiDriverSafe)
                          ? std::min(2, std::max(1, spec.numDrivers))
                          : 1))
        : std::max(1, spec.numDrivers);
    for (int32_t i = 0; i < replicas; ++i) {
      auto taskId = makeTaskId(spec.id, i);
      std::optional<common::SpillDiskOptions> taskSpillDiskOpts;
      if (spillDiskOpts_.has_value()) {
        const auto taskSpillDir =
            std::filesystem::path(spillDiskOpts_->spillDirPath) /
            fmt::format("fragment-{}-replica-{}", spec.id, i);
        std::filesystem::create_directories(taskSpillDir);
        taskSpillDiskOpts = common::SpillDiskOptions{
            .spillDirPath = taskSpillDir.string(),
            .spillDirCreated = true,
            .spillDirCreateCb = nullptr};
      }
      auto task = Task::create(
          taskId,
          spec.planFragment,
          // For HASH consumer fragments we pin each task to its bucket so
          // the producer's PartitionedOutputBuffer routes the correct
          // partition's data here. For non-HASH single-task fragments
          // destination is moot; 0 is fine.
          /*destination=*/i,
          queryCtx_,
          Task::ExecutionMode::kParallel,
          /*consumer=*/Consumer{},
          /*memoryArbitrationPriority=*/0,
          std::move(taskSpillDiskOpts));
      fragmentTasks_[spec.id].push_back(std::move(task));
      fragmentTaskDrivers[spec.id].push_back(perReplicaDrivers);
      fragmentTaskBroadcastFanout[spec.id].push_back(bcastN);
      fragmentTaskStarted[spec.id].push_back(false);
      LOG(WARNING) << "FluxQueryCoordinator[" << queryId_
                   << "]: created fragment " << spec.id << " replica " << i
                   << "/" << replicas << " taskId=" << taskId
                   << " drivers=" << perReplicaDrivers
                   << " inboundN=" << inboundN
                   << " keyedFinalLocalRepartition="
                   << spec.keyedFinalLocalRepartition
                   << " rightSemiProjectMultiDriverSafe="
                   << spec.rightSemiProjectMultiDriverSafe
                   << " innerJoinMultiDriverSafe="
                   << spec.innerJoinMultiDriverSafe
                   << (bcastN > 0 ? fmt::format(" bcastFanout={}", bcastN)
                                  : std::string{});
    }
  }

  const auto startFragmentTask = [&](int32_t fragmentId, int32_t replica) {
    auto& task = fragmentTasks_[fragmentId][replica];
    if (!task || fragmentTaskStarted[fragmentId][replica]) {
      return;
    }
    const auto perReplicaDrivers = fragmentTaskDrivers[fragmentId][replica];
    const auto bcastN = fragmentTaskBroadcastFanout[fragmentId][replica];
    const auto inboundN = consumerInboundPartitions[fragmentId];
    LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: starting fragment " << fragmentId << " replica "
                 << replica << "/" << fragmentTasks_[fragmentId].size() << " taskId=" << task->taskId()
                 << " drivers=" << perReplicaDrivers << " inboundN=" << inboundN
                 << (bcastN > 0 ? fmt::format(" bcastFanout={}", bcastN) : std::string{});
    // Current Velox initializes the UCX output queue from Task::start(), after
    // LocalPlanner has resolved the actual output-driver count. Do not prime it
    // here as well: the coordinator's earlier workaround predates the
    // Task-owned lifecycle and can publish queue metadata before the task has
    // finalized its pipelines.
    task->start(perReplicaDrivers);
    fragmentTaskStarted[fragmentId][replica] = true;
    if (lifecycleLogEnabled) {
      LOG(WARNING) << "[FLUX_LIFECYCLE] event=task_start"
                   << " elapsedMs=" << lifecycleElapsedMs() << " queryId=" << queryId_ << " peer=" << peerIndex_ << "/"
                   << peerCount_ << " fragment=" << fragmentId << " replica=" << replica
                   << " replicas=" << fragmentTasks_[fragmentId].size() << " drivers=" << perReplicaDrivers
                   << " inboundN=" << inboundN << " bcastFanout=" << bcastN << " taskId=" << task->taskId();
    }
    // Pipeline structure after Velox LocalPlanner splits the merged plan
    // at LocalPartitionNode boundaries. Used to compare batch
    // fragmentation against pv-cli's single-stage layout. Per-pipeline
    // driver count is filled in post-completion via operatorStats; here
    // we capture the overall topology immediately after start.
    // Gated by VLOG(1) so production runs don't drown in topology logs;
    // enable via GLOG_v=1.
    if (VLOG_IS_ON(1)) {
      const auto ts = task->taskStats();
      VLOG(1) << "FluxQueryCoordinator[" << queryId_ << "]: task " << task->taskId()
              << " pipelines=" << ts.pipelineStats.size() << " numTotalDrivers=" << task->numTotalDrivers();
      for (size_t pid = 0; pid < ts.pipelineStats.size(); ++pid) {
        const auto& ps = ts.pipelineStats[pid];
        VLOG(1) << "  pipeline[" << pid << "] input=" << ps.inputPipeline << " output=" << ps.outputPipeline;
      }
    }
    if (bcastN > 0) {
      // Task::start() has now initializePartitionOutput() registered the
      // kBroadcast OutputBuffer with numBuffers=1 (the plan's placeholder).
      // Expand to N destination buffers AND stamp noMoreBuffers=true so
      // enqueueBroadcastOutputLocked replicates every page to all N
      // consumer replicas and isFinishedLocked() can eventually return.
      // Bootstrap producers are started before their splits are added, so this
      // still happens before any enqueue. Velox's registered partitioned-output
      // manager forwards this update to the UCX queue as part of Task's generic
      // output lifecycle.
      task->updateOutputBuffers(bcastN, /*noMoreBuffers=*/true);
    }
  };

  // Phase 2/3: Add file scan splits and wire exchanges.
  //
  // Small scan-bearing BROADCAST producers (for example region/nation in
  // TPC-H Q5) need to publish EOS before large scans occupy the driver pool.
  // Otherwise downstream build operators wait for 0-row producer endpoints,
  // which shows up as seconds of UCX/control-plane blocked time even though no
  // data is being transferred. Bootstrap those small broadcast chains first,
  // then release the bulk scans. This preserves all producer endpoints and
  // partition ownership; it only changes split delivery order.
  std::vector<bool> scanSplitsWired(fragmentSpecs_.size(), false);
  std::vector<bool> exchangeWired(exchangeSpecs_.size(), false);

  const auto addScanSplitsForFragment = [&](FluxFragmentSpec& spec) {
    if (scanSplitsWired[spec.id] || spec.scanNodeIds.empty()) {
      return;
    }
    scanSplitsWired[spec.id] = true;
    if (fragmentTasks_[spec.id].empty()) {
      LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: skipping scan split wiring for non-local fragment "
                   << spec.id << " on peer " << peerIndex_ << "/" << peerCount_;
      return;
    }
    VELOX_CHECK_EQ(
        fragmentTasks_[spec.id].size(),
        1u,
        "Scan-bearing fragment {} has {} replicas; scans are only wired to "
        "single-replica tasks.",
        spec.id,
        fragmentTasks_[spec.id].size());
    auto& task = fragmentTasks_[spec.id][0];
    const bool fragmentIsBroadcastProducer = isBroadcastProducer[spec.id];
    VELOX_CHECK_EQ(
        spec.scanNodeIds.size(),
        spec.scanInfos.size(),
        "Fragment {} has {} scan node IDs but {} scan infos",
        spec.id,
        spec.scanNodeIds.size(),
        spec.scanInfos.size());

    for (size_t i = 0; i < spec.scanNodeIds.size(); i++) {
      const auto& scanInfo = spec.scanInfos[i];
      const auto& scanNodeId = spec.scanNodeIds[i];
      std::vector<bool> scanSplitConsumed(scanInfo->paths.size(), false);
      // Use the connector ID from the plan's TableScanNode.
      // This is critical: "test-hive" -> Velox Hive connector,
      // "cudf-hive" -> cuDF GPU connector (handles type casting).
      const auto& connectorId = (i < spec.scanConnectorIds.size() && !spec.scanConnectorIds[i].empty())
          ? spec.scanConnectorIds[i]
          : kHiveConnectorId;

      size_t addedSplits = 0;
      for (size_t j = 0; j < scanInfo->paths.size(); j++) {
        if (scanSplitConsumed[j]) {
          continue;
        }
        // K-way distributed broadcast production: every scan-bearing fragment --
        // INCLUDING BROADCAST producers -- strides its files 1/peerCount so each
        // peer scans a distinct slice. For a BROADCAST producer this means each
        // peer broadcasts only its partial; consumers fan in all peers' partials
        // and the HashBuild merges them (see peerHasFragment + Phase 3 BROADCAST
        // wiring). The old model pinned the entire build-side scan to peer 0,
        // which then serially scanned + built + broadcast it all.
        if (scanSplitOwners.at(scanInfo.get())[j] != peerIndex_) {
          continue;
        }
        std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
        if (!scanInfo->partitionColumns.empty() && j < scanInfo->partitionColumns.size()) {
          for (const auto& [key, value] : scanInfo->partitionColumns[j]) {
            partitionKeys[key] = value;
          }
        }

        std::shared_ptr<connector::ConnectorSplit> connectorSplit;
        if (auto icebergSplitInfo = std::dynamic_pointer_cast<IcebergSplitInfo>(scanInfo)) {
          std::unordered_map<std::string, std::string> metadataColumn;
          if (j < scanInfo->metadataColumns.size()) {
            metadataColumn = scanInfo->metadataColumns[j];
          }
          std::vector<connector::hive::iceberg::IcebergDeleteFile> deleteFiles;
          if (j < icebergSplitInfo->deleteFilesVec.size()) {
            const auto& splitDeleteFiles = icebergSplitInfo->deleteFilesVec[j];
            deleteFiles.reserve(splitDeleteFiles.size());
            for (const auto& deleteFile : splitDeleteFiles) {
              deleteFiles.emplace_back(deleteFile);
            }
          }
          std::unordered_map<std::string, std::string> customSplitInfo{{"table_format", "hive-iceberg"}};
          std::vector<connector::hive::iceberg::IcebergCoalescedFile>
              coalescedFiles;
#ifdef GLUTEN_ENABLE_GPU
          const auto configuredMultiFileTarget =
              queryCtx_->queryConfig().get<uint64_t>(
                  kCudfIcebergMultiFileTargetBytes,
                  kCudfIcebergMultiFileTargetBytesDefault);
          const auto targetBytes = configuredMultiFileTarget > 0
              ? configuredMultiFileTarget
              : queryCtx_->queryConfig().get<uint64_t>(
                    kCudfGpuTargetBatchBytes,
                    std::stoull(kCudfGpuTargetBatchBytesDefault));
          const auto maxFiles = queryCtx_->queryConfig().get<int32_t>(
              kCudfIcebergMultiFileMaxFiles,
              kCudfIcebergMultiFileMaxFilesDefault);
          const auto maxFileBytes = queryCtx_->queryConfig().get<uint64_t>(
              kCudfIcebergMultiFileMaxFileBytes,
              kCudfIcebergMultiFileMaxFileBytesDefault);
          const bool useExperimentalReader =
              queryCtx_->queryConfig().get<bool>(
                  kCudfHiveUseExperimentalReader, false);
          // A whole-file Iceberg Parquet task covers [4, fileSize), excluding
          // its PAR1 header. Require the range to reach EOF so genuine
          // row-group splits are never coalesced.
          const auto isWholeFile = [&](size_t index) {
            return index < scanInfo->starts.size() &&
                index < scanInfo->lengths.size() &&
                index < scanInfo->properties.size() &&
                scanInfo->starts[index] <= 4 &&
                scanInfo->properties[index].has_value() &&
                scanInfo->properties[index]->fileSize.has_value() &&
                scanInfo->starts[index] <= static_cast<uint64_t>(
                    *scanInfo->properties[index]->fileSize) &&
                scanInfo->lengths[index] >= static_cast<uint64_t>(
                    *scanInfo->properties[index]->fileSize) -
                    scanInfo->starts[index];
          };
          const bool useCudfIceberg = connectorId == kCudfIcebergConnectorId;
          const bool primaryCanCoalesce = useCudfIceberg &&
              !useExperimentalReader &&
              targetBytes > 0 && maxFiles > 1 && deleteFiles.empty() &&
              isWholeFile(j) &&
              static_cast<uint64_t>(*scanInfo->properties[j]->fileSize) <=
                  targetBytes &&
              static_cast<uint64_t>(*scanInfo->properties[j]->fileSize) <=
                  maxFileBytes;
          uint64_t accumulatedBytes = primaryCanCoalesce
              ? static_cast<uint64_t>(*scanInfo->properties[j]->fileSize)
              : 0;
          if (primaryCanCoalesce) {
            for (size_t next = j + 1;
                 next < scanInfo->paths.size() &&
                 coalescedFiles.size() + 1 < static_cast<size_t>(maxFiles) &&
                 accumulatedBytes < targetBytes;
                 ++next) {
              if (scanSplitConsumed[next] ||
                  scanSplitOwners.at(scanInfo.get())[next] != peerIndex_) {
                continue;
              }
              const bool hasDeletes =
                  next < icebergSplitInfo->deleteFilesVec.size() &&
                  !icebergSplitInfo->deleteFilesVec[next].empty();
              const bool samePartition =
                  scanInfo->partitionColumns.empty() ||
                  scanInfo->partitionColumns[next] ==
                      scanInfo->partitionColumns[j];
              const bool sameMetadata =
                  scanInfo->metadataColumns[next] ==
                  scanInfo->metadataColumns[j];
              if (hasDeletes || !isWholeFile(next) || !samePartition ||
                  !sameMetadata ||
                  static_cast<uint64_t>(
                      *scanInfo->properties[next]->fileSize) > maxFileBytes) {
                continue;
              }
              const auto fileSize = static_cast<uint64_t>(
                  *scanInfo->properties[next]->fileSize);
              if (accumulatedBytes >= targetBytes ||
                  fileSize > targetBytes - accumulatedBytes) {
                continue;
              }
              coalescedFiles.push_back(
                  {scanInfo->paths[next], fileSize});
              accumulatedBytes += fileSize;
              scanSplitConsumed[next] = true;
            }
          }
#endif
          connectorSplit = std::make_shared<connector::hive::iceberg::HiveIcebergSplit>(
              connectorId,
              scanInfo->paths[j],
              scanInfo->format,
              scanInfo->starts[j],
              scanInfo->lengths[j],
              partitionKeys,
              std::nullopt,
              customSplitInfo,
              nullptr,
              true,
              std::move(deleteFiles),
              metadataColumn,
              j < scanInfo->properties.size() ? scanInfo->properties[j] : std::nullopt,
              /*dataSequenceNumber=*/0,
              std::move(coalescedFiles));
#ifdef GLUTEN_ENABLE_GPU
          const auto prefetchPrimaryPath =
              cleanedCudfPath(scanInfo->paths[j]);
          if (connectorId == kCudfIcebergConnectorId &&
              queryCtx_->executor() != nullptr &&
              prefetchPrimaryPath.starts_with("s3://")) {
            const auto icebergConnectorSplit =
                std::dynamic_pointer_cast<
                    connector::hive::iceberg::HiveIcebergSplit>(
                    connectorSplit);
            VELOX_CHECK_NOT_NULL(icebergConnectorSplit);
            std::vector<
                cudf_velox::connector::hive::SplitPrefetchFile>
                prefetchFiles;
            std::optional<uint64_t> primaryFileSize;
            if (icebergConnectorSplit->properties.has_value() &&
                icebergConnectorSplit->properties->fileSize.has_value()) {
              primaryFileSize = static_cast<uint64_t>(
                  *icebergConnectorSplit->properties->fileSize);
            }
            if (primaryFileSize.has_value()) {
              prefetchFiles.reserve(
                  1 + icebergConnectorSplit->coalescedFiles.size());
              prefetchFiles.push_back(
                  {prefetchPrimaryPath,
                   *primaryFileSize});
              for (const auto& file :
                   icebergConnectorSplit->coalescedFiles) {
                prefetchFiles.push_back(
                    {cleanedCudfPath(file.filePath), file.length});
              }
              cudf_velox::connector::hive::ExecutorSplitPrefetch::
                  registerSplit(
                      queryCtx_->executor(),
                      queryCtx_->queryId(),
                      prefetchPrimaryPath,
                      std::move(prefetchFiles));
            }
          }
#endif
        } else
#ifdef GLUTEN_ENABLE_GPU
            if (connectorId == kCudfHiveConnectorId && scanInfo->canUseCudfConnector()) {
          std::unordered_map<std::string, std::string> metadataColumn;
          if (j < scanInfo->metadataColumns.size()) {
            metadataColumn = scanInfo->metadataColumns[j];
          }
          connectorSplit = std::make_shared<cudf_velox::connector::hive::CudfHiveConnectorSplit>(
              kCudfHiveConnectorId,
              cleanedCudfPath(scanInfo->paths[j]),
              scanInfo->starts[j],
              scanInfo->lengths[j],
              /*splitWeight=*/0,
              metadataColumn);
        } else
#endif
        {
          connectorSplit = std::make_shared<connector::hive::HiveConnectorSplit>(
              connectorId,
              scanInfo->paths[j],
              scanInfo->format,
              scanInfo->starts[j],
              scanInfo->lengths[j],
              partitionKeys);
        }

        task->addSplit(scanNodeId, Split(std::move(connectorSplit)));
        ++addedSplits;
      }
      task->noMoreSplits(scanNodeId);

      LOG(WARNING) << "FluxQueryCoordinator: added " << addedSplits << "/" << scanInfo->paths.size()
                   << " scan splits (connector='" << connectorId << "') to fragment " << spec.id << " scan node "
                   << scanNodeId << " peer=" << peerIndex_ << "/" << peerCount_
                   << (fragmentIsBroadcastProducer ? " broadcastProducer" : "");
      if (lifecycleLogEnabled) {
        LOG(WARNING) << "[FLUX_LIFECYCLE] event=scan_splits"
                     << " elapsedMs=" << lifecycleElapsedMs() << " queryId=" << queryId_ << " peer=" << peerIndex_
                     << "/" << peerCount_ << " fragment=" << spec.id << " scanNode=" << scanNodeId
                     << " added=" << addedSplits << " total=" << scanInfo->paths.size() << " connector=" << connectorId
                     << " broadcastProducer=" << fragmentIsBroadcastProducer;
      }
    }
  };

  const auto wireExchange = [&](size_t exchangeIndex) {
    if (exchangeWired[exchangeIndex]) {
      return;
    }
    exchangeWired[exchangeIndex] = true;
    auto& exchange = exchangeSpecs_[exchangeIndex];
    auto& producerReplicas = fragmentTasks_[exchange.producerFragmentId];
    auto& consumerReplicas = fragmentTasks_[exchange.consumerFragmentId];
    if (consumerReplicas.empty()) {
      LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: skipping exchange " << exchange.id
                   << " on peer=" << peerIndex_ << "/" << peerCount_ << " because consumer fragment "
                   << exchange.consumerFragmentId << " is not local to this peer";
      return;
    }

    struct ProducerEndpointForSplit {
      std::string taskId;
      std::string host;
      int32_t urlPort;
    };
    auto comm = facebook::velox::ucx_exchange::Communicator::getInstance();
    VELOX_CHECK_NOT_NULL(comm, "UCX Communicator must be initialized before FLUX exchange wiring");
    const int urlPort = static_cast<int>(comm->getListenerPort()) - 3;
    const bool isBroadcast = exchange.partitionType == "BROADCAST";
    std::vector<ProducerEndpointForSplit> producerEndpoints;
    const auto appendProducerEndpoints =
        [&](const std::string& peerId, const std::string& host, int32_t port, int32_t replicaCount) {
          for (int32_t j = 0; j < replicaCount; ++j) {
            producerEndpoints.push_back(
                ProducerEndpointForSplit{
                    peerId == localPeerId_ ? makeTaskId(exchange.producerFragmentId, j)
                                           : makeTaskIdForPeer(peerId, exchange.producerFragmentId, j),
                    host,
                    port});
          }
        };

    if (peerHasFragment(exchange.producerFragmentId, peerIndex_)) {
      // The handshake still goes through UCX before workerId-based
      // same-process bypass is negotiated.  Do not force loopback here:
      // production runs constrain UCX_NET_DEVICES to the routable interface,
      // and newer UCX versions correctly reject 127.0.0.1 in that setup.
      // Prefer the endpoint this executor registered with the driver so the
      // handshake uses an address supported by the configured UCX device.
      const auto localEndpoint = std::find_if(
          exchange.producerEndpoints.begin(),
          exchange.producerEndpoints.end(),
          [&](const auto& peer) { return peer.peerId == localPeerId_; });
      if (localEndpoint != exchange.producerEndpoints.end()) {
        VELOX_CHECK(
            !localEndpoint->host.empty(),
            "Local FLUX peer endpoint {} has empty host",
            localPeerId_);
        VELOX_CHECK_GT(
            localEndpoint->port,
            0,
            "Local FLUX peer endpoint {} has invalid RemoteConnectorSplit URL port {}",
            localPeerId_,
            localEndpoint->port);
        VELOX_CHECK_EQ(
            localEndpoint->port,
            urlPort,
            "Local FLUX peer endpoint {} port {} does not match communicator URL port {}",
            localPeerId_,
            localEndpoint->port,
            urlPort);
        appendProducerEndpoints(
            localPeerId_,
            localEndpoint->host,
            localEndpoint->port,
            static_cast<int32_t>(producerReplicas.size()));
      } else {
        // Preserve standalone/single-peer compatibility when no registry
        // endpoint was supplied.
        appendProducerEndpoints(
            localPeerId_,
            "127.0.0.1",
            urlPort,
            static_cast<int32_t>(producerReplicas.size()));
      }
    }
    for (const auto& peer : exchange.producerEndpoints) {
      if (peer.peerId == localPeerId_) {
        continue;
      }
      if (!peerHasFragment(exchange.producerFragmentId, peer.peerIndex)) {
        continue;
      }
      VELOX_CHECK(
          isSafeTaskIdComponent(peer.peerId),
          "FLUX peer id must be non-empty and must not contain '/' or '://', got '{}'",
          peer.peerId);
      VELOX_CHECK_GT(
          peer.port,
          0,
          "FLUX peer endpoint {} has invalid RemoteConnectorSplit URL port {}",
          peer.peerId,
          peer.port);
      VELOX_CHECK(
          !peer.host.empty(),
          "FLUX peer endpoint {} has empty host",
          peer.peerId);
      appendProducerEndpoints(
          peer.peerId,
          peer.host,
          peer.port,
          replicaCountForPeer(exchange.producerFragmentId, peer.peerIndex));
    }
    VELOX_CHECK(
        !producerEndpoints.empty(),
        "Exchange {} producer fragment {} has no local or remote producer endpoints on peer {}/{}",
        exchange.id,
        exchange.producerFragmentId,
        peerIndex_,
        peerCount_);

    LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: wiring exchange " << exchange.id
                 << " type=" << exchange.partitionType << " producerF=" << exchange.producerFragmentId << " ("
                 << producerReplicas.size() << " replicas)"
                 << " -> consumerF=" << exchange.consumerFragmentId << " (" << consumerReplicas.size() << " replicas)"
                 << " exchangeNode=" << exchange.exchangeNodeId;

    // Build the IBM ucx-exchange URL format expected by
    // UcxExchangeSource::extractTaskAndDestinationId:
    //   http://127.0.0.1:<port-3>/v1/task/<taskId>/results/<dest>
    const bool isHash = exchange.partitionType == "HASH";
    const bool isRange = exchange.partitionType == "RANGE";
    const auto totalDestinations = std::max(1, exchange.numPartitions);
    const auto peerOwnsDestination = [&](int dest) {
      if (peerCount_ <= 1) {
        return true;
      }
      if (isBroadcast) {
        return true;
      }
      if (isHash || isRange) {
        return destinationOwner(dest, totalDestinations) == peerIndex_;
      }
      return peerIndex_ == 0;
    };
    int32_t splitCount = 0;
    for (size_t cIdx = 0; cIdx < consumerReplicas.size(); ++cIdx) {
      auto& consumerTask = consumerReplicas[cIdx];
      if (isBroadcast) {
        int32_t destination = static_cast<int32_t>(cIdx);
        for (int32_t peer = 0; peer < peerIndex_; ++peer) {
          destination +=
              replicaCountForPeer(exchange.consumerFragmentId, peer);
        }
        for (const auto& producerEndpoint : producerEndpoints) {
          const auto url = fmt::format(
              "http://{}:{}/v1/task/{}/results/{}",
              producerEndpoint.host,
              producerEndpoint.urlPort,
              producerEndpoint.taskId,
              destination);
          consumerTask->addSplit(exchange.exchangeNodeId, Split(std::make_shared<RemoteConnectorSplit>(url)));
          ++splitCount;
        }
        consumerTask->noMoreSplits(exchange.exchangeNodeId);
        continue;
      }

      int32_t ownedOrdinal = 0;
      for (int dest = 0; dest < totalDestinations; ++dest) {
        if (!peerOwnsDestination(dest)) {
          continue;
        }
        const bool assigned = isRange
            ? ownedOrdinal % static_cast<int>(consumerReplicas.size()) ==
                  static_cast<int>(cIdx)
            : isHash
                ? (consumerReplicas.size() == 1 ||
                   ownedOrdinal % static_cast<int>(consumerReplicas.size()) ==
                       static_cast<int>(cIdx))
                : true;
        ++ownedOrdinal;
        if (!assigned) {
          continue;
        }
        for (const auto& producerEndpoint : producerEndpoints) {
          const auto url = fmt::format(
              "http://{}:{}/v1/task/{}/results/{}",
              producerEndpoint.host,
              producerEndpoint.urlPort,
              producerEndpoint.taskId,
              dest);
          consumerTask->addSplit(exchange.exchangeNodeId, Split(std::make_shared<RemoteConnectorSplit>(url)));
          ++splitCount;
        }
      }
      consumerTask->noMoreSplits(exchange.exchangeNodeId);
    }
    LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: exchange " << exchange.id << " wired (" << splitCount
                 << " splits across " << consumerReplicas.size() << " consumer task(s), " << producerEndpoints.size()
                 << " producer endpoint(s), " << totalDestinations << " producer destinations, "
                 << (isHash        ? "HASH-striped"
                         : isRange ? "RANGE-fanout"
                                   : "single-consumer")
                 << ", peer=" << peerIndex_ << "/" << peerCount_ << ")";
    if (lifecycleLogEnabled) {
      LOG(WARNING) << "[FLUX_LIFECYCLE] event=exchange_wired"
                   << " elapsedMs=" << lifecycleElapsedMs() << " queryId=" << queryId_ << " peer=" << peerIndex_ << "/"
                   << peerCount_ << " exchange=" << exchange.id << " type=" << exchange.partitionType
                   << " producerF=" << exchange.producerFragmentId << " consumerF=" << exchange.consumerFragmentId
                   << " consumerTasks=" << consumerReplicas.size() << " producerEndpoints=" << producerEndpoints.size()
                   << " prunedEmptyProducerEndpoints=0"
                   << " splits=" << splitCount << " totalDestinations=" << totalDestinations;
    }
  };

  for (const auto& spec : fragmentSpecs_) {
    if (!bootstrapBroadcastProducer[spec.id]) {
      continue;
    }
    for (int32_t i = 0; i < static_cast<int32_t>(fragmentTasks_[spec.id].size()); ++i) {
      startFragmentTask(spec.id, i);
    }
  }

  for (auto& spec : fragmentSpecs_) {
    if (bootstrapBroadcastProducer[spec.id]) {
      addScanSplitsForFragment(spec);
    }
  }
  for (size_t i = 0; i < exchangeSpecs_.size(); ++i) {
    const auto& exchange = exchangeSpecs_[i];
    if (exchange.partitionType == "BROADCAST" && bootstrapBroadcastProducer[exchange.producerFragmentId]) {
      wireExchange(i);
    }
  }

  for (auto& spec : fragmentSpecs_) {
    addScanSplitsForFragment(spec);
  }
  for (size_t i = 0; i < exchangeSpecs_.size(); ++i) {
    wireExchange(i);
  }

  // Fragment ids are emitted in producer-before-consumer order.  At this
  // point every consumer already has all RemoteConnectorSplits and every scan
  // already has its Hive splits, so no task starts in an incomplete split
  // state.  Starting in this order registers producer output buffers before
  // downstream ExchangeSources begin fetching.
  for (const auto& spec : fragmentSpecs_) {
    for (int32_t i = 0; i < static_cast<int32_t>(fragmentTasks_[spec.id].size()); ++i) {
      startFragmentTask(spec.id, i);
    }
  }

  // Phase 3.5: broadcast producer output buffer fan-out is set inside
  // startFragmentTask() immediately after Task::start(); nothing else to do.

  // Diagnostic watchdog: every 5s dump the state of every (fragId, replicaIdx)
  // Task so we can see where a hang is happening. Cheap: N_tasks log lines
  // per 5s. Stops on destructor.
  watchdogThread_ = std::thread([this]() {
    int tick = 0;
    const int gpuDiagnosticsIntervalTicks = envIntOrDefault("GLUTEN_GPU_MEMORY_DIAGNOSTICS_WATCHDOG_INTERVAL_TICKS", 0);
    const int watchdogIntervalMs = std::max(1, envIntOrDefault("GLUTEN_FLUX_WATCHDOG_INTERVAL_MS", 5000));
    const int planStatsIntervalTicks = envIntOrDefault("GLUTEN_FLUX_WATCHDOG_PLAN_STATS_INTERVAL_TICKS", 0);
    const int planStatsSampleTasks = std::max(1, envIntOrDefault("GLUTEN_FLUX_WATCHDOG_PLAN_STATS_SAMPLE_TASKS", 4));
    const bool lifecycleLogEnabled = fluxLifecycleLogEnabled();
    // Track which failed taskIds we've already logged to avoid repeating the
    // same error message every tick.
    std::unordered_set<std::string> reportedFailures;
    while (!watchdogStop_.load(std::memory_order_acquire)) {
      // wait_for returns true if the predicate is met (stop requested),
      // false if it timed out — in either case the next iteration's loop
      // condition checks watchdogStop_, so the thread exits within microseconds
      // of notify_all from the destructor instead of waiting up to 5s.
      std::unique_lock<std::mutex> lock(watchdogMutex_);
      if (watchdogCv_.wait_for(lock, std::chrono::milliseconds(watchdogIntervalMs), [this]() {
            return watchdogStop_.load(std::memory_order_acquire);
          })) {
        break;
      }
      lock.unlock();
      ++tick;
      const auto elapsedMs =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - lifecycleStartTime_)
              .count();
      // Summarize per-fragment state counts (more scannable than per-task).
      for (size_t f = 0; f < fragmentTasks_.size(); ++f) {
        int counts[5] = {0, 0, 0, 0, 0}; // running,finished,canceled,aborted,failed
        for (auto& task : fragmentTasks_[f]) {
          if (!task)
            continue;
          auto s = static_cast<int>(task->state());
          if (s >= 0 && s < 5) {
            counts[s]++;
          }
        }
        LOG(WARNING) << "FluxWatchdog[" << queryId_ << "] tick=" << tick << " frag=" << f
                     << " replicas=" << fragmentTasks_[f].size() << " running=" << counts[0]
                     << " finished=" << counts[1] << " canceled=" << counts[2] << " aborted=" << counts[3]
                     << " failed=" << counts[4];
      }
      // Temporary (LOCAL-01 Q3 OOM diagnosis): dump per-operator memory
      // reservation once per tick so we can see which operator is holding
      // 30 GB on the GPU path.
      if (tick <= 6 || tick % 5 == 0) {
        std::function<void(memory::MemoryPool*, int)> dumpPool = [&](memory::MemoryPool* pool, int depth) {
          if (!pool)
            return;
          const auto reserved = pool->reservedBytes();
          const auto peak = pool->peakBytes();
          if (reserved > 0 || peak > 0) {
            LOG(WARNING) << "MemPool[" << queryId_ << "] tick=" << tick << " " << std::string(depth * 2, ' ')
                         << pool->name() << " reserved=" << reserved << " peak=" << peak;
          }
          pool->visitChildren([&](memory::MemoryPool* child) {
            dumpPool(child, depth + 1);
            return true;
          });
        };
        for (size_t f = 0; f < fragmentTasks_.size(); ++f) {
          for (auto& task : fragmentTasks_[f]) {
            if (!task)
              continue;
            if (task->pool() != nullptr) {
              dumpPool(task->pool(), 0);
            }
          }
        }
      }

      if (gpuDiagnosticsIntervalTicks > 0 && tick % gpuDiagnosticsIntervalTicks == 0) {
        GpuMemoryTracker::dumpDiagnosticsToLog(
            "FluxWatchdog[" + queryId_ + "] tick=" + std::to_string(tick) + " periodic");
      }

      // Dump error message for any newly-failed task. Velox Task::setError is
      // completely silent (only stashes exception_), so without this the only
      // visible signal is the watchdog's failed-count going up.
      for (size_t f = 0; f < fragmentTasks_.size(); ++f) {
        for (auto& task : fragmentTasks_[f]) {
          if (!task)
            continue;
          if (task->state() != TaskState::kFailed)
            continue;
          const auto& tid = task->taskId();
          if (reportedFailures.insert(tid).second) {
            LOG(ERROR) << "FluxWatchdog[" << queryId_ << "] tick=" << tick << " FAILED-TASK taskId=" << tid
                       << " frag=" << f << " errorMessage={" << task->errorMessage() << "}";
            GpuMemoryTracker::dumpDiagnosticsToLog("FluxWatchdog[" + queryId_ + "] taskId=" + tid);
            LOG(ERROR) << "FluxWatchdog[" << queryId_ << "] taskId=" << tid << " planWithStats:\n"
                       << task->printPlanWithStats(/*includeCustomStats=*/true);
          }
        }
      }
      // Sample a few non-terminal tasks with full taskId for deeper debugging.
      int sampled = 0;
      for (size_t f = 0; f < fragmentTasks_.size() && sampled < 6; ++f) {
        for (auto& task : fragmentTasks_[f]) {
          if (!task)
            continue;
          if (!isTerminalState(task->state())) {
            const auto liveStats = task->taskStats();
            std::string blockedReasons;
            for (const auto& [reason, count] : liveStats.numBlockedDrivers) {
              if (!blockedReasons.empty()) {
                blockedReasons.append(",");
              }
              blockedReasons.append(fmt::format("{}={}", reason, count));
            }
            LOG(WARNING) << "FluxWatchdog[" << queryId_ << "] tick=" << tick
                         << " non-terminal taskId=" << task->taskId()
                         << " state=" << static_cast<int>(task->state())
                         << " numDrivers=" << task->numTotalDrivers()
                         << " numFinishedDrivers="
                         << task->numFinishedDrivers()
                         << " queuedDrivers=" << liveStats.numQueuedDrivers
                         << " runningDrivers=" << liveStats.numRunningDrivers
                         << " blockedDrivers={" << blockedReasons << "}";
            if (++sampled >= 6) break;
          }
        }
      }

      if (lifecycleLogEnabled) {
        for (size_t f = 0; f < fragmentTasks_.size(); ++f) {
          for (size_t replica = 0; replica < fragmentTasks_[f].size(); ++replica) {
            const auto& task = fragmentTasks_[f][replica];
            if (!task) {
              continue;
            }
            const auto taskStats = task->taskStats();
            LOG(WARNING) << "[FLUX_LIFECYCLE] event=task_snapshot"
                         << " elapsedMs=" << elapsedMs << " tick=" << tick << " queryId=" << queryId_
                         << " peer=" << peerIndex_ << "/" << peerCount_ << " fragment=" << f << " replica=" << replica
                         << " state=" << static_cast<int>(task->state()) << " drivers=" << task->numTotalDrivers()
                         << " finishedDrivers=" << task->numFinishedDrivers()
                         << " pipelines=" << taskStats.pipelineStats.size() << " taskId=" << task->taskId();
            for (size_t pipelineIdx = 0; pipelineIdx < taskStats.pipelineStats.size(); ++pipelineIdx) {
              const auto& pipelineStats = taskStats.pipelineStats[pipelineIdx];
              for (const auto& opStats : pipelineStats.operatorStats) {
                const auto wallNanos = opStats.addInputTiming.wallNanos + opStats.getOutputTiming.wallNanos +
                    opStats.finishTiming.wallNanos;
                const bool interesting = opStats.blockedWallNanos > 0 || wallNanos > 0 || opStats.inputPositions > 0 ||
                    opStats.outputPositions > 0 || opStats.rawInputPositions > 0 ||
                    opStats.planNodeId.find("flux_") != std::string::npos ||
                    opStats.operatorType.find("Exchange") != std::string::npos ||
                    opStats.operatorType.find("PartitionedOutput") != std::string::npos ||
                    opStats.operatorType.find("TableScan") != std::string::npos ||
                    opStats.operatorType.find("HashJoin") != std::string::npos;
                if (!interesting) {
                  continue;
                }
                LOG(WARNING) << "[FLUX_LIFECYCLE] event=operator_snapshot"
                             << " elapsedMs=" << elapsedMs << " tick=" << tick << " queryId=" << queryId_
                             << " peer=" << peerIndex_ << "/" << peerCount_ << " fragment=" << f
                             << " replica=" << replica << " pipeline=" << pipelineIdx
                             << " planNode=" << opStats.planNodeId << " operator=" << opStats.operatorType
                             << " drivers=" << opStats.numDrivers << " splits=" << opStats.numSplits
                             << " rawInRows=" << opStats.rawInputPositions << " inRows=" << opStats.inputPositions
                             << " outRows=" << opStats.outputPositions
                             << " blockedMs=" << static_cast<int64_t>(opStats.blockedWallNanos / 1000000)
                             << " wallMs=" << static_cast<int64_t>(wallNanos / 1000000);
              }
            }
          }
        }
      }

      if (planStatsIntervalTicks > 0 && tick % planStatsIntervalTicks == 0) {
        int statsSampled = 0;
        for (size_t f = 0; f < fragmentTasks_.size() && statsSampled < planStatsSampleTasks; ++f) {
          for (auto& task : fragmentTasks_[f]) {
            if (!task)
              continue;
            if (isTerminalState(task->state()))
              continue;
            LOG(ERROR) << "FluxWatchdog[" << queryId_ << "] tick=" << tick
                       << " non-terminal planWithStats taskId=" << task->taskId() << " frag=" << f
                       << " state=" << static_cast<int>(task->state()) << " numDrivers=" << task->numTotalDrivers()
                       << " numFinishedDrivers=" << task->numFinishedDrivers() << "\n"
                       << task->printPlanWithStats(/*includeCustomStats=*/true);
            if (++statsSampled >= planStatsSampleTasks)
              break;
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// next() - fetch output from the root fragment
// ---------------------------------------------------------------------------

bool FluxQueryCoordinator::fetchNextOutputPage(std::vector<std::unique_ptr<SerializedPageBase>>& pages) {
  nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{"coordinator::fetchNextOutputPage"};
  const auto rootReplicas = fragmentReplicaCount_[rootFragmentId_];
  constexpr int32_t kDestination = 0; // each root Task gathers to dest 0

  if (rootOutputSequence_.empty()) {
    rootOutputSequence_.assign(rootReplicas, 0);
    rootReplicaAtEnd_.assign(rootReplicas, false);
  }

  // Drain policy:
  //   - Sequential: finish replica 0 fully before moving to 1, etc. Used
  //     for RANGE to preserve global ORDER BY.
  //   - Round-robin: fair across replicas for HASH/ROUND_ROBIN/BROADCAST
  //     where no global order contract exists.
  auto pickNext = [&]() -> int32_t {
    if (rootDrainSequential_) {
      for (int32_t i = 0; i < rootReplicas; ++i) {
        if (!rootReplicaAtEnd_[i]) {
          return i;
        }
      }
      return -1;
    }
    for (int32_t attempted = 0; attempted < rootReplicas; ++attempted) {
      int32_t idx = rootFetchCursor_ % rootReplicas;
      rootFetchCursor_ = (rootFetchCursor_ + 1) % rootReplicas;
      if (!rootReplicaAtEnd_[idx]) {
        return idx;
      }
    }
    return -1;
  };

  for (int32_t attempts = 0; attempts < rootReplicas; ++attempts) {
    int32_t idx = pickNext();
    if (idx < 0) {
      noMoreData_ = true;
      return false;
    }
    auto rootTaskId = makeTaskId(rootFragmentId_, idx);
    // Lifetime-safe fetch state owned by the getData callback (shared_ptr by
    // value). A sibling/intermediate task failure can make
    // rethrowFirstTaskError() throw out of the poll loop below while a local
    // root producer is still live and later flushes during abort; capturing
    // stack locals by reference then caused setValue() on a destroyed promise
    // -> "pure virtual method called" -> SIGABRT (the whole executor died,
    // poisoning the continuous session). Owning the state keeps it alive past
    // the unwind so a late firing lands on an orphaned-but-valid object.
    struct FetchState {
      ContinuePromise promise{"FluxQueryCoordinator::fetchNextOutputPage"};
      std::atomic<bool> fulfilled{false};
      bool complete{false};
      int64_t inSequence{0};
      std::vector<std::unique_ptr<SerializedPageBase>> pages;
    };
    auto requestedSeq = rootOutputSequence_[idx];
    auto state = std::make_shared<FetchState>();
    state->inSequence = requestedSeq;

    LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: fetchNextOutputPage rootTask=" << rootTaskId
                 << " seq=" << requestedSeq
                 << " rootState=" << static_cast<int>(fragmentTasks_[rootFragmentId_][idx]->state());

    // IBM-baseline: DefaultOutputBufferManager exposes getData (IOBuf-vector
    // callback) instead of getPages (SerializedPageBase-vector callback).
    // Wrap each IOBuf in a PrestoSerializedPage so the downstream
    // prepareStreamForDeserialize() call site keeps working unchanged.
    auto ok = bufferManager_->getData(
        rootTaskId,
        kDestination,
        kRootOutputFetchMaxBytes,
        requestedSeq,
        [state, idx, qid = queryId_](
            std::vector<std::unique_ptr<folly::IOBuf>> gotPages,
            int64_t inSequence,
            std::vector<int64_t> /*remainingBytes*/) {
          LOG(WARNING) << "FluxQueryCoordinator[" << qid << "]: getData callback fired"
                       << " replica=" << idx << " pages=" << gotPages.size() << " inSeq=" << inSequence;
          for (auto& iobuf : gotPages) {
            if (iobuf != nullptr) {
              ++inSequence;
              state->pages.push_back(std::make_unique<PrestoSerializedPage>(std::move(iobuf)));
            } else {
              state->complete = true;
            }
          }
          state->inSequence = inSequence;
          // Only the first firing fulfills the promise (folly setValue throws
          // if already satisfied); a late abort-time firing is a no-op.
          bool expected = false;
          if (state->fulfilled.compare_exchange_strong(expected, true)) {
            state->promise.setValue();
          }
        });

    if (!ok) {
      LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: getData returned ok=false for replica=" << idx
                   << " (task not registered?)";
      rootReplicaAtEnd_[idx] = true;
      continue;
    }

    // Poll the data future. The getData callback fulfills dataPromise the
    // moment a batch is ready; a short 5ms poll keeps first-batch latency low
    // (the old 500ms poll quantized timeToFirstBatch to 500ms multiples — the
    // dominant per-query coordination latency) while still checking task
    // failure each tick (a failed producer/intermediate task does not always
    // drive a root OutputBuffer callback, so we must surface its error instead
    // of blocking forever).
    auto dataFuture = state->promise.getSemiFuture();
    while (!dataFuture.isReady()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (dataFuture.isReady()) {
        break;
      }
      // Keep the root-specific log line for the common direct-root failure
      // case.
      const auto& rootTask = fragmentTasks_[rootFragmentId_][idx];
      if (rootTask != nullptr &&
          (rootTask->state() == TaskState::kFailed || rootTask->state() == TaskState::kAborted ||
           rootTask->error() != nullptr)) {
        LOG(ERROR) << "FluxQueryCoordinator[" << queryId_ << "]: root task became non-OK while waiting for output"
                   << " rootTask=" << rootTaskId << " state=" << static_cast<int>(rootTask->state());
        rethrowFirstTaskError();
      }
      rethrowFirstTaskError();
    }

    // Harvest the owned state into the caller's outputs (the callback wrote
    // into `state`, which outlives any unwind of this frame).
    rootOutputSequence_[idx] = state->inSequence;
    if (state->complete) {
      rootReplicaAtEnd_[idx] = true;
      bufferManager_->acknowledge(rootTaskId, kDestination, rootOutputSequence_[idx]);
      bufferManager_->deleteResults(rootTaskId, kDestination);
    }
    for (auto& page : state->pages) {
      pages.push_back(std::move(page));
    }

    if (!pages.empty()) {
      return true;
    }
    // Empty poll (timeout or end-of-stream with no data): loop to try
    // another replica. For sequential drain, pickNext returns the same
    // replica again until it reaches end; for round-robin, advance.
  }

  noMoreData_ = std::all_of(rootReplicaAtEnd_.begin(), rootReplicaAtEnd_.end(), [](bool b) { return b; });
  return false;
}

#ifdef GLUTEN_ENABLE_GPU
RowVectorPtr FluxQueryCoordinator::fetchNextDeviceOutput() {
  nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{
      "coordinator::fetchNextDeviceOutput"};
  const auto rootReplicas = fragmentReplicaCount_[rootFragmentId_];
  constexpr int32_t kDestination = 0;
  constexpr uint64_t kMaxBytes = std::numeric_limits<uint64_t>::max();

  if (rootOutputSequence_.empty()) {
    rootOutputSequence_.assign(rootReplicas, 0);
    rootReplicaAtEnd_.assign(rootReplicas, false);
  }

  const auto pickNext = [&]() -> int32_t {
    if (rootDrainSequential_) {
      for (int32_t i = 0; i < rootReplicas; ++i) {
        if (!rootReplicaAtEnd_[i]) {
          return i;
        }
      }
      return -1;
    }
    for (int32_t attempted = 0; attempted < rootReplicas; ++attempted) {
      const int32_t idx = rootFetchCursor_ % rootReplicas;
      rootFetchCursor_ = (rootFetchCursor_ + 1) % rootReplicas;
      if (!rootReplicaAtEnd_[idx]) {
        return idx;
      }
    }
    return -1;
  };

  auto queueManager =
      facebook::velox::ucx_exchange::UcxOutputQueueManager::getInstanceRef();
  while (!noMoreData_) {
    const int32_t idx = pickNext();
    if (idx < 0) {
      noMoreData_ = true;
      rethrowFirstTaskError();
      logOperatorMetrics();
      return nullptr;
    }

    struct DeviceFetchState {
      ContinuePromise promise{
          "FluxQueryCoordinator::fetchNextDeviceOutput"};
      std::atomic<bool> fulfilled{false};
      std::shared_ptr<cudf::packed_columns> data;
      int64_t sequence{0};
    };

    const auto rootTaskId = makeTaskId(rootFragmentId_, idx);
    const auto requestedSequence = rootOutputSequence_[idx];
    auto state = std::make_shared<DeviceFetchState>();
    state->sequence = requestedSequence;

    VLOG(2) << "FluxQueryCoordinator[" << queryId_
            << "]: fetchNextDeviceOutput rootTask=" << rootTaskId
            << " seq=" << requestedSequence
            << " rootState="
            << static_cast<int>(
                   fragmentTasks_[rootFragmentId_][idx]->state());

    queueManager->getData(
        rootTaskId,
        kDestination,
        kMaxBytes,
        requestedSequence,
        [state, idx, qid = queryId_](
            std::shared_ptr<cudf::packed_columns> data,
            int64_t sequence,
            std::vector<int64_t> /*remainingBytes*/) {
          VLOG(2) << "FluxQueryCoordinator[" << qid
                  << "]: device getData callback fired replica=" << idx
                  << " sequence=" << sequence
                  << " hasData=" << (data != nullptr);
          state->data = std::move(data);
          state->sequence = sequence;
          bool expected = false;
          if (state->fulfilled.compare_exchange_strong(expected, true)) {
            state->promise.setValue();
          }
        });

    auto dataFuture = state->promise.getSemiFuture();
    while (!dataFuture.isReady()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (!dataFuture.isReady()) {
        rethrowFirstTaskError();
      }
    }

    rethrowFirstTaskError();
    VELOX_CHECK_EQ(
        state->sequence,
        requestedSequence,
        "Unexpected UCX root output sequence for task {}",
        rootTaskId);

    if (state->data == nullptr) {
      rootReplicaAtEnd_[idx] = true;
      queueManager->deleteResults(rootTaskId, kDestination);
      noMoreData_ = std::all_of(
          rootReplicaAtEnd_.begin(),
          rootReplicaAtEnd_.end(),
          [](bool atEnd) { return atEnd; });
      continue;
    }

    rootOutputSequence_[idx] = state->sequence + 1;
    auto packed = std::move(state->data);
    VELOX_CHECK_EQ(
        packed.use_count(),
        1,
        "GPU sink root output must have unique ownership");
    cudf::packed_columns packedColumns(
        std::move(packed->metadata), std::move(packed->gpu_data));
    packed.reset();

    auto tableView = cudf::unpack(packedColumns);
    auto packedTable = std::make_unique<cudf::packed_table>(
        cudf::packed_table{tableView, std::move(packedColumns)});
    auto outputType = std::dynamic_pointer_cast<const RowType>(
        fragmentSpecs_[rootFragmentId_]
            .planFragment.planNode->outputType());
    VELOX_CHECK_NOT_NULL(outputType, "Root fragment must have RowType output");
    if (deserializePool_ == nullptr) {
      deserializePool_ =
          queryCtx_->pool()->addLeafChild("flux_device_output");
    }
    return std::make_shared<facebook::velox::cudf_velox::CudfVector>(
        deserializePool_.get(),
        outputType,
        tableView.num_rows(),
        std::move(packedTable),
        deviceRootOutputStream_.view());
  }

  rethrowFirstTaskError();
  logOperatorMetrics();
  return nullptr;
}
#endif

void FluxQueryCoordinator::rethrowFirstTaskError() const {
  // Scan every tracked Task. If any has FAILED/ABORTED, rethrow the first
  // captured exception so the JNI translates it to a Java RuntimeException
  // instead of returning nullptr (= end-of-stream = false PASS in tests).
  // A clean kFinished task with task->error()==nullptr is left alone, so
  // legitimate count=0 results (Q11 empty bloom filter, restrictive
  // predicates) keep flowing through as nullptr.
  std::exception_ptr firstError;
  std::string firstFailedTaskId;
  TaskState firstFailedState = TaskState::kRunning;
  std::shared_ptr<Task> firstFailedTask;
  for (auto& replicas : fragmentTasks_) {
    for (auto& task : replicas) {
      if (task == nullptr) {
        continue;
      }
      const auto state = task->state();
      const bool nonOkTerminal = state == TaskState::kFailed || state == TaskState::kAborted;
      auto err = task->error();
      if (nonOkTerminal || err) {
        if (!firstError) {
          firstError = err;
          firstFailedTaskId = task->taskId();
          firstFailedState = state;
          firstFailedTask = task;
        }
      }
    }
  }
  if (firstError) {
    LOG(ERROR) << "FluxQueryCoordinator[" << queryId_ << "]: rethrowFirstTaskError taskId=" << firstFailedTaskId
               << " state=" << static_cast<int>(firstFailedState);
    logOperatorMetrics();
    GpuMemoryTracker::dumpDiagnosticsToLog("FluxQueryCoordinator[" + queryId_ + "] taskId=" + firstFailedTaskId);
    if (firstFailedTask != nullptr) {
      try {
        LOG(ERROR) << "FluxQueryCoordinator[" << queryId_ << "]: failed task planWithStats taskId=" << firstFailedTaskId
                   << "\n"
                   << firstFailedTask->printPlanWithStats(/*includeCustomStats=*/true);
      } catch (const std::exception& e) {
        LOG(ERROR) << "FluxQueryCoordinator[" << queryId_
                   << "]: failed to print planWithStats for taskId=" << firstFailedTaskId << ": " << e.what();
      }
    }
    std::rethrow_exception(firstError);
  }
  // No captured std::exception_ptr but some task is in kFailed/kAborted
  // (Velox can transition without a stored exception in odd cases, e.g.
  // requestAbort() from outside). Still surface the failure rather than
  // pretending the query succeeded.
  for (auto& replicas : fragmentTasks_) {
    for (auto& task : replicas) {
      if (task == nullptr) {
        continue;
      }
      const auto state = task->state();
      if (state == TaskState::kFailed || state == TaskState::kAborted) {
        logOperatorMetrics();
        VELOX_FAIL(
            "FluxQueryCoordinator[{}]: task {} ended in non-OK terminal "
            "state {} with no captured error; failing the query rather "
            "than returning empty result",
            queryId_,
            task->taskId(),
            static_cast<int>(state));
      }
    }
  }
}

RowVectorPtr FluxQueryCoordinator::next() {
  nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{"coordinator::next"};
  VELOX_CHECK(started_, "Must call start() before next()");

  if (!rootProducesOutput_) {
    if (!noMoreData_) {
      waitForCompletion();
      noMoreData_ = true;
      rethrowFirstTaskError();
      logOperatorMetrics();
    }
    return nullptr;
  }

#ifdef GLUTEN_ENABLE_GPU
  if (deviceRootOutput_) {
    if (noMoreData_) {
      rethrowFirstTaskError();
      return nullptr;
    }
    return fetchNextDeviceOutput();
  }
#endif

  const auto deserializePage = [&](std::unique_ptr<SerializedPageBase>& page) -> RowVectorPtr {
    // Deserialize the CPU page via Presto serde into a RowVector.
    auto inputStream = page->prepareStreamForDeserialize();
    auto outputType =
        std::dynamic_pointer_cast<const RowType>(fragmentSpecs_[rootFragmentId_].planFragment.planNode->outputType());
    VELOX_CHECK(outputType != nullptr, "Root fragment must have RowType output");
    if (deserializePool_ == nullptr) {
      deserializePool_ = queryCtx_->pool()->addLeafChild("flux_deserialize");
    }
    RowVectorPtr result;
    VectorStreamGroup::read(
        inputStream.get(),
        deserializePool_.get(),
        outputType,
        getVectorSerde(),
        &result,
        /*options=*/nullptr);
    return result;
  };

  if (noMoreData_) {
    rethrowFirstTaskError();
    return nullptr;
  }

  // Keep fetching until we get data pages or hit end-of-stream.
  while (!noMoreData_) {
    while (!pendingRootPages_.empty()) {
      auto page = std::move(pendingRootPages_.front());
      pendingRootPages_.pop_front();
      if (!page) {
        continue;
      }
      rethrowFirstTaskError();
      return deserializePage(page);
    }

    std::vector<std::unique_ptr<SerializedPageBase>> pages;
    bool gotData = fetchNextOutputPage(pages);

    if (!gotData) {
      // EOS: distinguish real "all tasks finished cleanly" from
      // "some task failed and produced no pages" before returning nullptr.
      rethrowFirstTaskError();
      logOperatorMetrics();
      return nullptr;
    }

    // A failed producer/consumer can race with the root aggregate producing a
    // page (for global aggregates this may be a single NULL row). Surface the
    // native failure before handing that page to Spark, otherwise a partial
    // result can masquerade as a successful one-row answer.
    rethrowFirstTaskError();

    // Queue all non-null pages returned by this getData call. The root
    // fragment's kHttp PartitionedOutput produces CPU-serialized pages
    // (PrestoVectorSerde / IOBuf-backed); inter-fragment GPU edges flow
    // through IBM's UCX / IntraNodeTransfer path and never reach this
    // coordinator, so there is no GpuSerializedPage shape to handle here.
    for (auto& page : pages) {
      if (!page) {
        continue;
      }
      pendingRootPages_.push_back(std::move(page));
    }
  }

  // Loop fell through without producing data: same EOS surface as the
  // !gotData path above; consult task error state before reporting empty.
  rethrowFirstTaskError();
  logOperatorMetrics();
  return nullptr;
}

// ---------------------------------------------------------------------------
// isFinished()
// ---------------------------------------------------------------------------

bool FluxQueryCoordinator::isFinished() const {
  if (!started_) {
    return false;
  }
  for (auto& replicas : fragmentTasks_) {
    for (auto& task : replicas) {
      if (!isTerminalState(task->state())) {
        return false;
      }
    }
  }
  return true;
}

void FluxQueryCoordinator::logOperatorMetrics() const {
  if (!fluxOperatorMetricsEnabled()) {
    operatorMetricsLogged_.store(true, std::memory_order_release);
    return;
  }
  if (operatorMetricsLogged_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  size_t emitted = 0;
  for (size_t fragmentId = 0; fragmentId < fragmentTasks_.size(); ++fragmentId) {
    const auto& replicas = fragmentTasks_[fragmentId];
    for (size_t replicaId = 0; replicaId < replicas.size(); ++replicaId) {
      const auto& task = replicas[replicaId];
      if (task == nullptr) {
        continue;
      }

      const auto taskStats = task->taskStats();
      for (size_t pipelineIdx = 0; pipelineIdx < taskStats.pipelineStats.size(); ++pipelineIdx) {
        const auto& pipelineStats = taskStats.pipelineStats[pipelineIdx];
        for (const auto& opStats : pipelineStats.operatorStats) {
          const auto wallNanos =
              opStats.addInputTiming.wallNanos + opStats.getOutputTiming.wallNanos + opStats.finishTiming.wallNanos;
          const auto cpuNanos =
              opStats.addInputTiming.cpuNanos + opStats.getOutputTiming.cpuNanos + opStats.finishTiming.cpuNanos;

          folly::dynamic row = folly::dynamic::object;
          row["queryId"] = queryId_;
          row["fragmentId"] = static_cast<int64_t>(fragmentId);
          row["replicaId"] = static_cast<int64_t>(replicaId);
          row["taskId"] = task->taskId();
          row["taskState"] = static_cast<int64_t>(task->state());
          row["pipelineId"] = static_cast<int64_t>(pipelineIdx);
          row["operatorId"] = opStats.operatorId;
          row["operatorPipelineId"] = opStats.pipelineId;
          row["planNodeId"] = opStats.planNodeId;
          row["operatorType"] = opStats.operatorType;
          row["numDrivers"] = opStats.numDrivers;
          row["numSplits"] = opStats.numSplits;
          row["rawInputRows"] = static_cast<int64_t>(opStats.rawInputPositions);
          row["rawInputBytes"] = static_cast<int64_t>(opStats.rawInputBytes);
          row["inputRows"] = static_cast<int64_t>(opStats.inputPositions);
          row["inputBytes"] = static_cast<int64_t>(opStats.inputBytes);
          row["inputVectors"] = static_cast<int64_t>(opStats.inputVectors);
          row["outputRows"] = static_cast<int64_t>(opStats.outputPositions);
          row["outputBytes"] = static_cast<int64_t>(opStats.outputBytes);
          row["outputVectors"] = static_cast<int64_t>(opStats.outputVectors);
          row["blockedWallNanos"] = static_cast<int64_t>(opStats.blockedWallNanos);
          row["addInputWallNanos"] = static_cast<int64_t>(opStats.addInputTiming.wallNanos);
          row["getOutputWallNanos"] = static_cast<int64_t>(opStats.getOutputTiming.wallNanos);
          row["finishWallNanos"] = static_cast<int64_t>(opStats.finishTiming.wallNanos);
          row["wallNanos"] = static_cast<int64_t>(wallNanos);
          row["addInputCpuNanos"] = static_cast<int64_t>(opStats.addInputTiming.cpuNanos);
          row["getOutputCpuNanos"] = static_cast<int64_t>(opStats.getOutputTiming.cpuNanos);
          row["finishCpuNanos"] = static_cast<int64_t>(opStats.finishTiming.cpuNanos);
          row["cpuNanos"] = static_cast<int64_t>(cpuNanos);
          row["peakTotalMemoryBytes"] = static_cast<int64_t>(opStats.memoryStats.peakTotalMemoryReservation);
          row["numMemoryAllocations"] = static_cast<int64_t>(opStats.memoryStats.numMemoryAllocations);
          row["spilledBytes"] = static_cast<int64_t>(opStats.spilledBytes);
          row["spilledRows"] = static_cast<int64_t>(opStats.spilledRows);
          row["customStats"] = detail::serializeFluxOperatorRuntimeStats(opStats.runtimeStats);
          LOG(WARNING) << "[FLUX_OPERATOR_METRICS] " << folly::toJson(row);
          ++emitted;
        }
      }
    }
  }
  LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: emitted " << emitted << " FLUX operator metric row(s)";
}

// ---------------------------------------------------------------------------
// abort()
// ---------------------------------------------------------------------------

void FluxQueryCoordinator::abort(std::chrono::milliseconds perTaskTimeout) {
  nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{"coordinator::abort"};
  // Serialize abort callers. The same coordinator can be aborted by two
  // paths (an explicit JNI nativeAbortFluxQuery from the JVM-side close, and
  // the destructor when ~FluxQueryHandle drops the shared_ptr); without this
  // gate they would race on requestAbort() + parallel taskCompletionFuture()
  // waits.
  std::lock_guard<std::mutex> lk(abortMutex_);
  if (!started_) {
    LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: abort() called before start(), skipping";
    return;
  }
  if (aborted_) {
    LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: abort() already completed, skipping";
    return;
  }
  LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: abort() begin perTaskTimeoutMs=" << perTaskTimeout.count();

  // Phase 1: fire requestAbort() on every non-terminal task. This is
  // non-blocking; each call returns a future. We don't need those futures
  // because we'll wait on taskCompletionFuture() in Phase 2 — that's
  // realized whenever the task is no longer running, regardless of which
  // call drove it to terminal.
  nvtx3::mark_in<GlutenFluxDomain>("abort:Phase1-requestAbort-begin");
  size_t firedCount = 0;
  size_t alreadyTerminalCount = 0;
  for (auto& replicas : fragmentTasks_) {
    for (auto& task : replicas) {
      if (task == nullptr) {
        continue;
      }
      const auto state = task->state();
      if (!isTerminalState(state)) {
        LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: requestAbort(" << task->taskId()
                     << ") state=" << static_cast<int>(state);
        try {
          task->requestAbort();
        } catch (const std::exception& e) {
          LOG(ERROR) << "FluxQueryCoordinator[" << queryId_ << "]: requestAbort threw for " << task->taskId() << ": "
                     << e.what();
        } catch (...) {
          LOG(ERROR) << "FluxQueryCoordinator[" << queryId_ << "]: requestAbort threw unknown exception for "
                     << task->taskId();
        }
        ++firedCount;
      } else {
        ++alreadyTerminalCount;
      }
    }
  }
  LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: abort() requested " << firedCount << " task(s), "
               << alreadyTerminalCount << " already terminal";

  // Phase 2: bounded wait for each task to reach a terminal state. We use
  // taskCompletionFuture() which is realized when the task is no longer
  // running. Cap each wait so a stuck task can't hold up shutdown forever —
  // a leak warning at process exit is strictly better than an infinite hang
  // or a removePool() VELOX_CHECK abort.
  nvtx3::scoped_range_in<GlutenFluxDomain> phase2Range{"coordinator::abort:Phase2-waitTaskTerminal"};
  size_t terminalAfterWait = 0;
  size_t timedOut = 0;
  for (auto& replicas : fragmentTasks_) {
    for (auto& task : replicas) {
      if (task == nullptr) {
        continue;
      }
      if (isTerminalState(task->state())) {
        ++terminalAfterWait;
        continue;
      }
      try {
        nvtx3::scoped_range_in<GlutenFluxDomain> waitRange{"coordinator::abort:taskCompletionFuture.wait"};
        // Wait blocks up to perTaskTimeout. After the wait, just check the
        // Task's own state; we don't depend on wait()'s return value to
        // sidestep folly Future API drift between Velox versions.
        task->taskCompletionFuture().wait(perTaskTimeout);
        if (isTerminalState(task->state())) {
          ++terminalAfterWait;
        } else {
          ++timedOut;
          LOG(ERROR) << "FluxQueryCoordinator[" << queryId_ << "]: abort() TIMEOUT waiting for taskId=" << task->taskId()
                     << " state=" << static_cast<int>(task->state()) << " numDrivers=" << task->numTotalDrivers()
                     << " numFinishedDrivers=" << task->numFinishedDrivers()
                     << " — task may still hold MemoryPool reservations";
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << "FluxQueryCoordinator[" << queryId_ << "]: taskCompletionFuture wait threw for " << task->taskId()
                   << ": " << e.what();
      } catch (...) {
        LOG(ERROR) << "FluxQueryCoordinator[" << queryId_ << "]: taskCompletionFuture wait threw unknown for "
                   << task->taskId();
      }
    }
  }

  aborted_ = true;
  LOG(WARNING) << "FluxQueryCoordinator[" << queryId_ << "]: abort() end"
               << " terminal=" << terminalAfterWait << " timedOut=" << timedOut;
}

// ---------------------------------------------------------------------------
// waitForCompletion()
// ---------------------------------------------------------------------------

void FluxQueryCoordinator::waitForCompletion() {
  VELOX_CHECK(started_, "Must call start() before waitForCompletion()");

  struct CompletionWaitState {
    std::mutex mutex;
    std::condition_variable cv;
    size_t completed{0};
  };

  // Surface a task that failed before watcher registration immediately. Its
  // completion future can remain pending until all task threads have exited.
  rethrowFirstTaskError();

  auto waitState = std::make_shared<CompletionWaitState>();
  std::vector<folly::Future<folly::Unit>> completionWatchers;
  for (auto& replicas : fragmentTasks_) {
    for (auto& task : replicas) {
      if (task == nullptr) {
        continue;
      }
      completionWatchers.emplace_back(
          std::move(task->taskCompletionFuture())
              .via(&folly::InlineExecutor::instance())
              .thenTry([waitState](folly::Try<folly::Unit>&&) -> folly::Unit {
                std::lock_guard<std::mutex> lock(waitState->mutex);
                ++waitState->completed;
                waitState->cv.notify_all();
                return folly::Unit{};
              }));
    }
  }

  // Close the race where a task fails after the first scan but before its
  // completion watcher is registered.
  rethrowFirstTaskError();

  size_t observed = 0;
  while (observed < completionWatchers.size()) {
    {
      std::unique_lock<std::mutex> lock(waitState->mutex);
      waitState->cv.wait(lock, [&]() { return waitState->completed > observed; });
      observed = waitState->completed;
    }
    // Do not block on tasks in fragment order. Any task reaching a terminal
    // state wakes this waiter so a failed intermediate fragment is surfaced
    // immediately even if an earlier producer is still running.
    rethrowFirstTaskError();
  }

  rethrowFirstTaskError();

  // Clean up output buffer entries for all tasks.
  for (auto& replicas : fragmentTasks_) {
    for (auto& task : replicas) {
      removeTaskOutputState(task, bufferManager_, queryId_, "waitForCompletion");
    }
  }
}

} // namespace gluten
