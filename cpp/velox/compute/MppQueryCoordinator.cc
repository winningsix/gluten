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

#include "MppQueryCoordinator.h"

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
/// MppQueryCoordinator::next() returns only one page to Spark at a time.  An
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
  LOG(WARNING) << "MppQueryCoordinator[" << queryId << "]: removeTask(" << tid << ") reason=" << reason
               << " state=" << static_cast<int>(task->state());
  if (bufferManager != nullptr) {
    try {
      bufferManager->removeTask(tid);
    } catch (const std::exception& e) {
      LOG(ERROR) << "MppQueryCoordinator[" << queryId << "]: OutputBufferManager::removeTask(" << tid
                 << ") threw: " << e.what();
    } catch (...) {
      LOG(ERROR) << "MppQueryCoordinator[" << queryId << "]: OutputBufferManager::removeTask(" << tid
                 << ") threw unknown exception";
    }
  }
#ifdef GLUTEN_ENABLE_GPU
  try {
    auto queueMgr = facebook::velox::ucx_exchange::UcxOutputQueueManager::getInstanceRef();
    queueMgr->removeTask(tid);
  } catch (const std::exception& e) {
    LOG(ERROR) << "MppQueryCoordinator[" << queryId << "]: UcxOutputQueueManager::removeTask(" << tid
               << ") threw: " << e.what();
  } catch (...) {
    LOG(ERROR) << "MppQueryCoordinator[" << queryId << "]: UcxOutputQueueManager::removeTask(" << tid
               << ") threw unknown exception";
  }
#endif
}

bool isSafeTaskIdComponent(const std::string& value) {
  return !value.empty() && value.find('/') == std::string::npos && value.find("://") == std::string::npos;
}

bool mppOperatorMetricsEnabled() {
  const char* value = std::getenv("GLUTEN_MPP_OPERATOR_METRICS_ENABLED");
  return value != nullptr && std::string(value) == "1";
}

bool mppLifecycleLogEnabled() {
  const char* value = std::getenv("GLUTEN_MPP_LIFECYCLE_LOG_ENABLED");
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

// NVTX domain for gluten MPP. Same name as the one in MppJniWrapper.cc so
// both files emit ranges into the same nsys lane.
struct GlutenMppDomain {
  static constexpr char const* name{"gluten-mpp"};
};

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MppQueryCoordinator::MppQueryCoordinator(
    std::string queryId,
    std::vector<MppFragmentSpec> fragments,
    std::vector<MppExchangeSpec> exchanges,
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
      "MPP local peer id must be non-empty and must not contain '/' or '://', got '{}'",
      localPeerId_);
  VELOX_CHECK_GE(peerIndex_, 0, "MPP peer index must be non-negative");
  VELOX_CHECK_GT(peerCount_, 0, "MPP peer count must be positive");
  VELOX_CHECK_LT(peerIndex_, peerCount_, "MPP peer index {} must be less than peer count {}", peerIndex_, peerCount_);
  if (spillDiskOpts_.has_value()) {
    auto& opts = spillDiskOpts_.value();
    if (!opts.spillDirCreated) {
      VELOX_CHECK_NOT_NULL(
          opts.spillDirCreateCb,
          "MPP spill root must either exist or provide a create callback");
      opts.spillDirPath = opts.spillDirCreateCb();
      opts.spillDirCreated = true;
      opts.spillDirCreateCb = nullptr;
    }
    VELOX_CHECK(
        !opts.spillDirPath.empty(), "MPP spill root path must not be empty");
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

std::shared_ptr<MppQueryCoordinator> MppQueryCoordinator::create(
    const std::string& queryId,
    std::vector<MppFragmentSpec> fragments,
    std::vector<MppExchangeSpec> exchanges,
    std::shared_ptr<core::QueryCtx> queryCtx,
    folly::Executor* executor,
    std::optional<common::SpillDiskOptions> spillDiskOpts,
    std::string localPeerId,
    int32_t peerIndex,
    int32_t peerCount) {
  // Using new + shared_ptr because the constructor is private.
  return std::shared_ptr<MppQueryCoordinator>(new MppQueryCoordinator(
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

MppQueryCoordinator::~MppQueryCoordinator() {
  nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{"coordinator::~destructor"};
  // Stop the watchdog first so it doesn't touch half-destructed state.
  // notify_all wakes the watchdog from its cv.wait_for so join() returns
  // promptly instead of blocking up to 5s for the next tick.
  {
    std::lock_guard<std::mutex> lock(watchdogMutex_);
    watchdogStop_.store(true, std::memory_order_release);
  }
  watchdogCv_.notify_all();
  {
    nvtx3::scoped_range_in<GlutenMppDomain> r{"coordinator::~destructor:watchdog.join"};
    if (watchdogThread_.joinable()) {
      watchdogThread_.join();
    }
  }
  size_t totalTasks = 0;
  for (auto& replicas : fragmentTasks_) {
    totalTasks += replicas.size();
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: destructor entry"
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
          VLOG(1) << "MppQueryCoordinator[" << queryId_ << "]: plan-with-stats for fragment " << f
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
          VLOG(1) << "MppQueryCoordinator[" << queryId_ << "]: plan-with-stats for fragment " << f
                  << " taskId=" << task->taskId() << " (end)";
        } catch (const std::exception& e) {
          VLOG(1) << "  printPlanWithStats failed: " << e.what();
        }
      }
    }
  }
  if (started_) {
    // Abort any still-running tasks (with bounded wait) and then remove
    // their OutputBuffer entries from the global OutputBufferManager.
    // Failing to call removeTask() leaks the producer-side buffer (and its
    // pages) into the process-wide manager, which causes subsequent MPP
    // queries in the same JVM to hang because ExchangeClient state there
    // is not cleanly reset. abort() is idempotent: if the JVM-side close
    // already drove abort, the inner aborted_ guard short-circuits.
    {
      nvtx3::scoped_range_in<GlutenMppDomain> r{"coordinator::~destructor:abort_call"};
      try {
        abort();
      } catch (...) {
      }
    }
    {
      nvtx3::scoped_range_in<GlutenMppDomain> removeTaskRange{"coordinator::~destructor:removeTaskFromOutputBuffer"};
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
  // then OutputBufferManager handle, then per-query deserialize pool,
  // then QueryCtx (which holds the root memory pool).
  {
    nvtx3::scoped_range_in<GlutenMppDomain> r{"coordinator::~destructor:fragmentTasks.clear"};
    fragmentTasks_.clear();
  }
  if (spillDiskOpts_.has_value()) {
    std::error_code error;
    std::filesystem::remove_all(spillDiskOpts_->spillDirPath, error);
    if (error) {
      LOG(ERROR) << "MppQueryCoordinator[" << queryId_
                 << "]: failed to remove spill root '"
                 << spillDiskOpts_->spillDirPath << "': " << error.message();
    }
  }
  {
    nvtx3::scoped_range_in<GlutenMppDomain> r{"coordinator::~destructor:bufferManager.reset"};
    bufferManager_.reset();
  }
  {
    nvtx3::scoped_range_in<GlutenMppDomain> r{"coordinator::~destructor:deserializePool.reset"};
    deserializePool_.reset();
  }
  {
    nvtx3::scoped_range_in<GlutenMppDomain> r{"coordinator::~destructor:queryCtx.reset"};
    queryCtx_.reset();
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: destructor exit";
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string MppQueryCoordinator::makeTaskId(int32_t fragmentId, int32_t replicaIdx) const {
  return makeTaskIdForPeer(localPeerId_, fragmentId, replicaIdx);
}

std::string MppQueryCoordinator::makeTaskIdForPeer(const std::string& peerId, int32_t fragmentId, int32_t replicaIdx)
    const {
  // Root-fragment output is consumed by this coordinator via
  // OutputBufferManager::getData() (kHttp PartitionedOutput).
  // Non-root fragment edges flow through IBM's UCX / IntraNodeTransfer
  // path; they share the same task-id prefix because routing is decided
  // by adapters, not by the prefix.
  VELOX_CHECK(
      isSafeTaskIdComponent(peerId),
      "MPP peer id must be non-empty and must not contain '/' or '://', got '{}'",
      peerId);
  return fmt::format("{}{}-{}-{}-p{}", kTaskIdPrefix, queryId_, peerId, fragmentId, replicaIdx);
}

bool MppQueryCoordinator::isTerminalState(TaskState state) {
  return state == TaskState::kFinished || state == TaskState::kCanceled || state == TaskState::kAborted ||
      state == TaskState::kFailed;
}

// ---------------------------------------------------------------------------
// start()
// ---------------------------------------------------------------------------

void MppQueryCoordinator::start() {
  nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{"coordinator::start"};
  VELOX_CHECK(!started_, "MppQueryCoordinator already started");
  started_ = true;
  lifecycleStartTime_ = std::chrono::steady_clock::now();
  const bool lifecycleLogEnabled = mppLifecycleLogEnabled();
  const auto lifecycleElapsedMs = [this]() -> int64_t {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - lifecycleStartTime_)
        .count();
  };
  if (lifecycleLogEnabled) {
    LOG(WARNING) << "[MPP_LIFECYCLE] event=query_start elapsedMs=0"
                 << " queryId=" << queryId_ << " peer=" << peerIndex_ << "/" << peerCount_
                 << " fragments=" << fragmentSpecs_.size() << " rootFragment=" << rootFragmentId_;
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: start() " << fragmentSpecs_.size()
               << " fragments, rootFragmentId=" << rootFragmentId_ << " peer=" << peerIndex_ << "/" << peerCount_
               << " localPeerId=" << localPeerId_;

  // Write-in-MPP: detect whether the root fragment is a distributed TableWrite.
  // When it is, the root must NOT use the SELECT ORDER-BY drain model (one
  // ordered stream on peer0). Instead every peer runs one TableWrite over the
  // slice it owns and drains its own commit batch. This single flag flips the
  // three places that otherwise special-case a root RANGE exchange for ordered
  // SELECT: per-peer replica count (1 writer/peer), destination ownership (fan
  // across peers, not peer0-only), and rootProducesOutput_ (all peers emit).
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
    LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: root fragment " << rootFragmentId_
                 << " is a distributed TableWrite (1 writer/peer, fan "
                 << "destinations across peers, all peers emit commit)";
  }

  // Phase 1: Create and start a Velox Task for each fragment.
  //
  // Task::create() + Task::start() internally handles:
  //   - initializePartitionOutput() -> OutputBufferManager::initializeTask()
  //     (for fragments whose plan ends with PartitionedOutputNode)
  //   - createExchangeClientLocked() -> creates ExchangeClient for any
  //     ExchangeNode in the plan
  //
  // We start ALL fragments before wiring exchanges so that producer tasks'
  // output buffers are already registered in OutputBufferManager when
  // consumers try to fetch data.
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
      LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: HASH fragment " << i
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
      LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: scan-bearing fragment " << i << " replicaCount "
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
    LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: write root fragment " << rootFragmentId_
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
  if (const auto* weightsEnv = std::getenv("GLUTEN_MPP_PEER_WEIGHTS")) {
    std::stringstream weightsStream(weightsEnv);
    std::string token;
    size_t index = 0;
    while (std::getline(weightsStream, token, ',') && index < peerWeights.size()) {
      try {
        peerWeights[index] = std::max(0.01, std::stod(token));
      } catch (...) {
        VELOX_FAIL("Invalid GLUTEN_MPP_PEER_WEIGHTS value '{}' at index {}", token, index);
      }
      ++index;
    }
    VELOX_CHECK_EQ(
        index,
        peerWeights.size(),
        "GLUTEN_MPP_PEER_WEIGHTS must contain exactly {} comma-separated weights",
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
      LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: broadcast scan producer fragment " << spec.id
                   << " split ownership total=" << totalScanSplits[spec.id] << " peer=" << peerIndex_ << "/"
                   << peerCount_ << " localSplits=" << scanSplitsByPeer[spec.id][peerIndex_];
    }
  }

  std::vector<bool> bootstrapBroadcastProducer(fragmentSpecs_.size(), false);
  for (const auto& spec : fragmentSpecs_) {
    bootstrapBroadcastProducer[spec.id] = isBroadcastProducer[spec.id] && !spec.scanInfos.empty() &&
        totalScanSplits[spec.id] <= static_cast<size_t>(peerSlots);
    if (bootstrapBroadcastProducer[spec.id]) {
      LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: bootstrap broadcast scan producer fragment " << spec.id
                   << " before bulk scans (totalSplits=" << totalScanSplits[spec.id] << ", peer=" << peerIndex_ << "/"
                   << peerCount_ << ", localSplits=" << scanSplitsByPeer[spec.id][peerIndex_] << ")";
    }
  }

  const auto peerOwnsExchangeDestination = [&](const MppExchangeSpec& exchange, int32_t peerIndex) {
    if (peerCount_ <= 1 || exchange.partitionType == "BROADCAST") {
      return true;
    }
    if (exchange.consumerFragmentId == rootFragmentId_ && exchange.partitionType == "RANGE" && !rootIsWrite) {
      // Preserve final ORDER BY by draining the root RANGE exchange on
      // peer0 until we implement a k-way merge across Spark partitions.
      // A distributed write root instead fans destinations across peers
      // (below), so each peer writes the slice it owns.
      return peerIndex == 0;
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
  if (peerCount_ > 1) {
    for (auto& spec : fragmentSpecs_) {
      if (!peerHasFragment(spec.id, peerIndex_)) {
        fragmentReplicaCount_[spec.id] = 0;
        LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: fragment " << spec.id
                     << " is not local to peer=" << peerIndex_ << "/" << peerCount_;
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
        if (exchange.partitionType == "RANGE") {
          // Final ORDER BY uses RANGE partitions. Keep the final ordered drain
          // on one Spark output partition until a k-way merge output iterator
          // exists across peer coordinators.
          rootProducesOutput_ = peerIndex_ == 0;
        } else if (exchange.partitionType == "HASH") {
          rootProducesOutput_ = false;
          for (int dest = 0; dest < std::max(1, exchange.numPartitions); ++dest) {
            if (destinationOwner(dest, std::max(1, exchange.numPartitions)) == peerIndex_) {
              rootProducesOutput_ = true;
              break;
            }
          }
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
    // dest % peerCount. Mirror that here. Forcing every peer to produce breaks
    // SINGLE-gather writes -- peer!=0 has no slice, so its TableWrite's inbound
    // exchange source blocks forever in WaitingForMetadata (no producer ever
    // targets that destination), hanging the query. HASH/RANGE writes still
    // distribute because each peer owns a fanned destination (replicaCount>0).
    rootProducesOutput_ = fragmentReplicaCount_[rootFragmentId_] > 0;
  }
  if (!rootProducesOutput_) {
    fragmentReplicaCount_[rootFragmentId_] = 0;
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: rootFragmentId=" << rootFragmentId_
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
          if (peerHasFragment(consumer, peer)) {
            consumerTasks += baseFragmentReplicaCount[consumer];
          }
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
    const auto perReplicaDrivers = (bcastN > 0)                 ? 1
        : (rootIsWrite && spec.id == rootFragmentId_)           ? 1
        : (isHashConsumer[spec.id] || isRangeConsumer[spec.id]) ? 1
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
      LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: created fragment " << spec.id << " replica " << i << "/"
                   << replicas << " taskId=" << taskId << " drivers=" << perReplicaDrivers << " inboundN=" << inboundN
                   << (bcastN > 0 ? fmt::format(" bcastFanout={}", bcastN) : std::string{});
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
    LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: starting fragment " << fragmentId << " replica "
                 << replica << "/" << fragmentTasks_[fragmentId].size() << " taskId=" << task->taskId()
                 << " drivers=" << perReplicaDrivers << " inboundN=" << inboundN
                 << (bcastN > 0 ? fmt::format(" bcastFanout={}", bcastN) : std::string{});
#ifdef GLUTEN_ENABLE_GPU
    // Velox's generic Task lifecycle is built without the cuDF/UCX integration definition in the
    // current library layering, while the cuDF adapter still replaces a kUcx output operator at
    // runtime. Register its strict output queue here, before any producer driver can enqueue. The
    // coordinator already owns UCX fanout updates and teardown, and initializeTask is idempotent
    // with a future Task-side lifecycle hook.
    const auto outputNode = std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
        fragmentSpecs_[fragmentId].planFragment.planNode);
    if (outputNode != nullptr &&
        outputNode->transportType() == core::PartitionedOutputNode::TransportType::kUcx) {
      facebook::velox::ucx_exchange::UcxOutputQueueManager::getInstanceRef()->initializeTask(
          task,
          outputNode->kind(),
          outputNode->numPartitions(),
          perReplicaDrivers);
      LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: initialized UCX output queue before task start"
                   << " taskId=" << task->taskId() << " destinations=" << outputNode->numPartitions()
                   << " drivers=" << perReplicaDrivers;
    }
#endif
    task->start(perReplicaDrivers);
    fragmentTaskStarted[fragmentId][replica] = true;
    if (lifecycleLogEnabled) {
      LOG(WARNING) << "[MPP_LIFECYCLE] event=task_start"
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
      VLOG(1) << "MppQueryCoordinator[" << queryId_ << "]: task " << task->taskId()
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
      // still happens before any enqueue.
      task->updateOutputBuffers(bcastN, /*noMoreBuffers=*/true);
#ifdef GLUTEN_ENABLE_GPU
      const auto ucxQueueUpdated =
          facebook::velox::ucx_exchange::UcxOutputQueueManager::getInstanceRef()->updateOutputBuffersIfExists(
              task->taskId(), bcastN, /*noMoreBuffers=*/true);
      VLOG(1) << "MppQueryCoordinator[" << queryId_ << "]: broadcast fanout taskId=" << task->taskId()
              << " destinations=" << bcastN << " ucxQueueUpdated=" << ucxQueueUpdated;
#endif
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

  const auto addScanSplitsForFragment = [&](MppFragmentSpec& spec) {
    if (scanSplitsWired[spec.id] || spec.scanNodeIds.empty()) {
      return;
    }
    scanSplitsWired[spec.id] = true;
    if (fragmentTasks_[spec.id].empty()) {
      LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: skipping scan split wiring for non-local fragment "
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
      // Use the connector ID from the plan's TableScanNode.
      // This is critical: "test-hive" -> Velox Hive connector,
      // "cudf-hive" -> cuDF GPU connector (handles type casting).
      const auto& connectorId = (i < spec.scanConnectorIds.size() && !spec.scanConnectorIds[i].empty())
          ? spec.scanConnectorIds[i]
          : kHiveConnectorId;

      size_t addedSplits = 0;
      for (size_t j = 0; j < scanInfo->paths.size(); j++) {
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
              j < scanInfo->properties.size() ? scanInfo->properties[j] : std::nullopt);
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

      LOG(WARNING) << "MppQueryCoordinator: added " << addedSplits << "/" << scanInfo->paths.size()
                   << " scan splits (connector='" << connectorId << "') to fragment " << spec.id << " scan node "
                   << scanNodeId << " peer=" << peerIndex_ << "/" << peerCount_
                   << (fragmentIsBroadcastProducer ? " broadcastProducer" : "");
      if (lifecycleLogEnabled) {
        LOG(WARNING) << "[MPP_LIFECYCLE] event=scan_splits"
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
      LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: skipping exchange " << exchange.id
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
    VELOX_CHECK_NOT_NULL(comm, "UCX Communicator must be initialized before MPP exchange wiring");
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
      appendProducerEndpoints(localPeerId_, "127.0.0.1", urlPort, static_cast<int32_t>(producerReplicas.size()));
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
          "MPP peer id must be non-empty and must not contain '/' or '://', got '{}'",
          peer.peerId);
      VELOX_CHECK_GT(
          peer.port, 0, "MPP peer endpoint {} has invalid RemoteConnectorSplit URL port {}", peer.peerId, peer.port);
      VELOX_CHECK(!peer.host.empty(), "MPP peer endpoint {} has empty host", peer.peerId);
      appendProducerEndpoints(peer.peerId, peer.host, peer.port, baseFragmentReplicaCount[exchange.producerFragmentId]);
    }
    VELOX_CHECK(
        !producerEndpoints.empty(),
        "Exchange {} producer fragment {} has no local or remote producer endpoints on peer {}/{}",
        exchange.id,
        exchange.producerFragmentId,
        peerIndex_,
        peerCount_);

    LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: wiring exchange " << exchange.id
                 << " type=" << exchange.partitionType << " producerF=" << exchange.producerFragmentId << " ("
                 << producerReplicas.size() << " replicas)"
                 << " -> consumerF=" << exchange.consumerFragmentId << " (" << consumerReplicas.size() << " replicas)"
                 << " exchangeNode=" << exchange.exchangeNodeId;

    // Build the IBM ucx-exchange URL format expected by
    // UcxExchangeSource::extractTaskAndDestinationId:
    //   http://127.0.0.1:<port-3>/v1/task/<taskId>/results/<dest>
    const bool isHash = exchange.partitionType == "HASH";
    const bool isRange = exchange.partitionType == "RANGE";
    const bool isRootConsumer = exchange.consumerFragmentId == rootFragmentId_;
    const auto totalDestinations = std::max(1, exchange.numPartitions);
    const auto peerOwnsDestination = [&](int dest) {
      if (peerCount_ <= 1) {
        return true;
      }
      if (isBroadcast) {
        return true;
      }
      if (isRootConsumer && isRange && !rootIsWrite) {
        return peerIndex_ == 0;
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
        const auto destination =
            peerIndex_ * static_cast<int32_t>(consumerReplicas.size()) + static_cast<int32_t>(cIdx);
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

      for (int dest = 0; dest < totalDestinations; ++dest) {
        if (!peerOwnsDestination(dest)) {
          continue;
        }
        const bool assigned = isRange ? dest % static_cast<int>(consumerReplicas.size()) == static_cast<int>(cIdx)
            : isHash                  ? (consumerReplicas.size() == 1 ||
                        dest % static_cast<int>(consumerReplicas.size()) == static_cast<int>(cIdx))
                                      : true;
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
    LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: exchange " << exchange.id << " wired (" << splitCount
                 << " splits across " << consumerReplicas.size() << " consumer task(s), " << producerEndpoints.size()
                 << " producer endpoint(s), " << totalDestinations << " producer destinations, "
                 << (isHash        ? "HASH-striped"
                         : isRange ? "RANGE-fanout"
                                   : "single-consumer")
                 << ", peer=" << peerIndex_ << "/" << peerCount_ << ")";
    if (lifecycleLogEnabled) {
      LOG(WARNING) << "[MPP_LIFECYCLE] event=exchange_wired"
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

  for (const auto& spec : fragmentSpecs_) {
    for (int32_t i = 0; i < static_cast<int32_t>(fragmentTasks_[spec.id].size()); ++i) {
      startFragmentTask(spec.id, i);
    }
  }

  for (auto& spec : fragmentSpecs_) {
    addScanSplitsForFragment(spec);
  }
  for (size_t i = 0; i < exchangeSpecs_.size(); ++i) {
    wireExchange(i);
  }

  // Phase 3.5: broadcast producer output buffer fan-out is set inside
  // startFragmentTask() immediately after Task::start(); nothing else to do.

  // Diagnostic watchdog: every 5s dump the state of every (fragId, replicaIdx)
  // Task so we can see where a hang is happening. Cheap: N_tasks log lines
  // per 5s. Stops on destructor.
  watchdogThread_ = std::thread([this]() {
    int tick = 0;
    const int gpuDiagnosticsIntervalTicks = envIntOrDefault("GLUTEN_GPU_MEMORY_DIAGNOSTICS_WATCHDOG_INTERVAL_TICKS", 0);
    const int watchdogIntervalMs = std::max(1, envIntOrDefault("GLUTEN_MPP_WATCHDOG_INTERVAL_MS", 5000));
    const int planStatsIntervalTicks = envIntOrDefault("GLUTEN_MPP_WATCHDOG_PLAN_STATS_INTERVAL_TICKS", 0);
    const int planStatsSampleTasks = std::max(1, envIntOrDefault("GLUTEN_MPP_WATCHDOG_PLAN_STATS_SAMPLE_TASKS", 4));
    const bool lifecycleLogEnabled = mppLifecycleLogEnabled();
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
        LOG(WARNING) << "MppWatchdog[" << queryId_ << "] tick=" << tick << " frag=" << f
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
            "MppWatchdog[" + queryId_ + "] tick=" + std::to_string(tick) + " periodic");
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
            LOG(ERROR) << "MppWatchdog[" << queryId_ << "] tick=" << tick << " FAILED-TASK taskId=" << tid
                       << " frag=" << f << " errorMessage={" << task->errorMessage() << "}";
            GpuMemoryTracker::dumpDiagnosticsToLog("MppWatchdog[" + queryId_ + "] taskId=" + tid);
            LOG(ERROR) << "MppWatchdog[" << queryId_ << "] taskId=" << tid << " planWithStats:\n"
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
            LOG(WARNING) << "MppWatchdog[" << queryId_ << "] tick=" << tick << " non-terminal taskId=" << task->taskId()
                         << " state=" << static_cast<int>(task->state()) << " numDrivers=" << task->numTotalDrivers()
                         << " numFinishedDrivers=" << task->numFinishedDrivers();
            if (++sampled >= 6)
              break;
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
            LOG(WARNING) << "[MPP_LIFECYCLE] event=task_snapshot"
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
                    opStats.planNodeId.find("mpp_") != std::string::npos ||
                    opStats.operatorType.find("Exchange") != std::string::npos ||
                    opStats.operatorType.find("PartitionedOutput") != std::string::npos ||
                    opStats.operatorType.find("TableScan") != std::string::npos ||
                    opStats.operatorType.find("HashJoin") != std::string::npos;
                if (!interesting) {
                  continue;
                }
                LOG(WARNING) << "[MPP_LIFECYCLE] event=operator_snapshot"
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
            LOG(ERROR) << "MppWatchdog[" << queryId_ << "] tick=" << tick
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

bool MppQueryCoordinator::fetchNextOutputPage(std::vector<std::unique_ptr<SerializedPageBase>>& pages) {
  nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{"coordinator::fetchNextOutputPage"};
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
      ContinuePromise promise{"MppQueryCoordinator::fetchNextOutputPage"};
      std::atomic<bool> fulfilled{false};
      bool complete{false};
      int64_t inSequence{0};
      std::vector<std::unique_ptr<SerializedPageBase>> pages;
    };
    auto requestedSeq = rootOutputSequence_[idx];
    auto state = std::make_shared<FetchState>();
    state->inSequence = requestedSeq;

    LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: fetchNextOutputPage rootTask=" << rootTaskId
                 << " seq=" << requestedSeq
                 << " rootState=" << static_cast<int>(fragmentTasks_[rootFragmentId_][idx]->state());

    // IBM-baseline: OutputBufferManager exposes getData (IOBuf-vector
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
          LOG(WARNING) << "MppQueryCoordinator[" << qid << "]: getData callback fired"
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
      LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: getData returned ok=false for replica=" << idx
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
        LOG(ERROR) << "MppQueryCoordinator[" << queryId_ << "]: root task became non-OK while waiting for output"
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
RowVectorPtr MppQueryCoordinator::fetchNextDeviceOutput() {
  nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{
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
          "MppQueryCoordinator::fetchNextDeviceOutput"};
      std::atomic<bool> fulfilled{false};
      std::shared_ptr<cudf::packed_columns> data;
      int64_t sequence{0};
    };

    const auto rootTaskId = makeTaskId(rootFragmentId_, idx);
    const auto requestedSequence = rootOutputSequence_[idx];
    auto state = std::make_shared<DeviceFetchState>();
    state->sequence = requestedSequence;

    LOG(WARNING) << "MppQueryCoordinator[" << queryId_
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
          LOG(WARNING) << "MppQueryCoordinator[" << qid
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
          queryCtx_->pool()->addLeafChild("mpp_device_output");
    }
    return std::make_shared<facebook::velox::cudf_velox::CudfVector>(
        deserializePool_.get(),
        outputType,
        tableView.num_rows(),
        std::move(packedTable),
        rmm::cuda_stream_default);
  }

  rethrowFirstTaskError();
  logOperatorMetrics();
  return nullptr;
}
#endif

void MppQueryCoordinator::rethrowFirstTaskError() const {
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
    LOG(ERROR) << "MppQueryCoordinator[" << queryId_ << "]: rethrowFirstTaskError taskId=" << firstFailedTaskId
               << " state=" << static_cast<int>(firstFailedState);
    logOperatorMetrics();
    GpuMemoryTracker::dumpDiagnosticsToLog("MppQueryCoordinator[" + queryId_ + "] taskId=" + firstFailedTaskId);
    if (firstFailedTask != nullptr) {
      try {
        LOG(ERROR) << "MppQueryCoordinator[" << queryId_ << "]: failed task planWithStats taskId=" << firstFailedTaskId
                   << "\n"
                   << firstFailedTask->printPlanWithStats(/*includeCustomStats=*/true);
      } catch (const std::exception& e) {
        LOG(ERROR) << "MppQueryCoordinator[" << queryId_
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
            "MppQueryCoordinator[{}]: task {} ended in non-OK terminal "
            "state {} with no captured error; failing the query rather "
            "than returning empty result",
            queryId_,
            task->taskId(),
            static_cast<int>(state));
      }
    }
  }
}

RowVectorPtr MppQueryCoordinator::next() {
  nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{"coordinator::next"};
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
      deserializePool_ = queryCtx_->pool()->addLeafChild("mpp_deserialize");
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

bool MppQueryCoordinator::isFinished() const {
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

void MppQueryCoordinator::logOperatorMetrics() const {
  if (!mppOperatorMetricsEnabled()) {
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

          LOG(WARNING) << "[MPP_OPERATOR_METRICS] " << folly::toJson(row);
          ++emitted;
        }
      }
    }
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: emitted " << emitted << " MPP operator metric row(s)";
}

// ---------------------------------------------------------------------------
// abort()
// ---------------------------------------------------------------------------

void MppQueryCoordinator::abort(std::chrono::milliseconds perTaskTimeout) {
  nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{"coordinator::abort"};
  // Serialize abort callers. The same coordinator can be aborted by two
  // paths (an explicit JNI nativeAbortMppQuery from the JVM-side close, and
  // the destructor when ~MppQueryHandle drops the shared_ptr); without this
  // gate they would race on requestAbort() + parallel taskCompletionFuture()
  // waits.
  std::lock_guard<std::mutex> lk(abortMutex_);
  if (!started_) {
    LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: abort() called before start(), skipping";
    return;
  }
  if (aborted_) {
    LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: abort() already completed, skipping";
    return;
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: abort() begin perTaskTimeoutMs=" << perTaskTimeout.count();

  // Phase 1: fire requestAbort() on every non-terminal task. This is
  // non-blocking; each call returns a future. We don't need those futures
  // because we'll wait on taskCompletionFuture() in Phase 2 — that's
  // realized whenever the task is no longer running, regardless of which
  // call drove it to terminal.
  nvtx3::mark_in<GlutenMppDomain>("abort:Phase1-requestAbort-begin");
  size_t firedCount = 0;
  size_t alreadyTerminalCount = 0;
  for (auto& replicas : fragmentTasks_) {
    for (auto& task : replicas) {
      if (task == nullptr) {
        continue;
      }
      const auto state = task->state();
      if (!isTerminalState(state)) {
        LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: requestAbort(" << task->taskId()
                     << ") state=" << static_cast<int>(state);
        try {
          task->requestAbort();
        } catch (const std::exception& e) {
          LOG(ERROR) << "MppQueryCoordinator[" << queryId_ << "]: requestAbort threw for " << task->taskId() << ": "
                     << e.what();
        } catch (...) {
          LOG(ERROR) << "MppQueryCoordinator[" << queryId_ << "]: requestAbort threw unknown exception for "
                     << task->taskId();
        }
        ++firedCount;
      } else {
        ++alreadyTerminalCount;
      }
    }
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: abort() requested " << firedCount << " task(s), "
               << alreadyTerminalCount << " already terminal";

  // Phase 2: bounded wait for each task to reach a terminal state. We use
  // taskCompletionFuture() which is realized when the task is no longer
  // running. Cap each wait so a stuck task can't hold up shutdown forever —
  // a leak warning at process exit is strictly better than an infinite hang
  // or a removePool() VELOX_CHECK abort.
  nvtx3::scoped_range_in<GlutenMppDomain> phase2Range{"coordinator::abort:Phase2-waitTaskTerminal"};
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
        nvtx3::scoped_range_in<GlutenMppDomain> waitRange{"coordinator::abort:taskCompletionFuture.wait"};
        // Wait blocks up to perTaskTimeout. After the wait, just check the
        // Task's own state; we don't depend on wait()'s return value to
        // sidestep folly Future API drift between Velox versions.
        task->taskCompletionFuture().wait(perTaskTimeout);
        if (isTerminalState(task->state())) {
          ++terminalAfterWait;
        } else {
          ++timedOut;
          LOG(ERROR) << "MppQueryCoordinator[" << queryId_ << "]: abort() TIMEOUT waiting for taskId=" << task->taskId()
                     << " state=" << static_cast<int>(task->state()) << " numDrivers=" << task->numTotalDrivers()
                     << " numFinishedDrivers=" << task->numFinishedDrivers()
                     << " — task may still hold MemoryPool reservations";
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << "MppQueryCoordinator[" << queryId_ << "]: taskCompletionFuture wait threw for " << task->taskId()
                   << ": " << e.what();
      } catch (...) {
        LOG(ERROR) << "MppQueryCoordinator[" << queryId_ << "]: taskCompletionFuture wait threw unknown for "
                   << task->taskId();
      }
    }
  }

  aborted_ = true;
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: abort() end"
               << " terminal=" << terminalAfterWait << " timedOut=" << timedOut;
}

// ---------------------------------------------------------------------------
// waitForCompletion()
// ---------------------------------------------------------------------------

void MppQueryCoordinator::waitForCompletion() {
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
