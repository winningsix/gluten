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
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/io/IOBuf.h>
#include <unordered_set>

// Must be included before any header that (transitively) pulls in
// CudfHiveConnectorSplit.h, because that header only forward-declares
// cudf::io::source_info. When this TU instantiates the destructor of
// std::shared_ptr<CudfHiveConnectorSplit> (via make_shared in inline code
// such as CudfHiveConnectorSplitBuilder::build()), the compiler needs the
// full definition of cudf::io::source_info to emit unique_ptr's deleter.
#include <cudf/io/types.hpp>

#include "config/VeloxConfig.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/OutputBuffer.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/SerializedPage.h"
#include "velox/exec/Task.h"
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

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MppQueryCoordinator::MppQueryCoordinator(
    std::string queryId,
    std::vector<MppFragmentSpec> fragments,
    std::vector<MppExchangeSpec> exchanges,
    std::shared_ptr<core::QueryCtx> queryCtx,
    folly::Executor* executor)
    : queryId_(std::move(queryId)),
      fragmentSpecs_(std::move(fragments)),
      exchangeSpecs_(std::move(exchanges)),
      queryCtx_(std::move(queryCtx)),
      executor_(executor),
      bufferManager_(OutputBufferManager::getInstanceRef()) {
  VELOX_CHECK(!fragmentSpecs_.empty(), "At least one fragment is required");
  VELOX_CHECK(queryCtx_ != nullptr, "QueryCtx must not be null");
  VELOX_CHECK(executor_ != nullptr, "Executor must not be null");

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
    VELOX_CHECK_LT(
        static_cast<size_t>(exchange.producerFragmentId),
        fragmentSpecs_.size());
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
  VELOX_CHECK_NE(
      rootFragmentId_,
      -1,
      "No root fragment (every fragment is a producer; exchange DAG has a cycle)");
}

std::shared_ptr<MppQueryCoordinator> MppQueryCoordinator::create(
    const std::string& queryId,
    std::vector<MppFragmentSpec> fragments,
    std::vector<MppExchangeSpec> exchanges,
    std::shared_ptr<core::QueryCtx> queryCtx,
    folly::Executor* executor) {
  // Using new + shared_ptr because the constructor is private.
  return std::shared_ptr<MppQueryCoordinator>(new MppQueryCoordinator(
      queryId,
      std::move(fragments),
      std::move(exchanges),
      std::move(queryCtx),
      executor));
}

MppQueryCoordinator::~MppQueryCoordinator() {
  // Stop the watchdog first so it doesn't touch half-destructed state.
  watchdogStop_ = true;
  if (watchdogThread_.joinable()) {
    watchdogThread_.join();
  }
  size_t totalTasks = 0;
  for (auto& replicas : fragmentTasks_) {
    totalTasks += replicas.size();
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: destructor entry"
               << " started=" << started_
               << " fragments=" << fragmentTasks_.size()
               << " totalTasks=" << totalTasks
               << " rootFragmentId=" << rootFragmentId_;
  if (started_) {
    // Abort any still-running tasks (with bounded wait) and then remove
    // their OutputBuffer entries from the global OutputBufferManager.
    // Failing to call removeTask() leaks the producer-side buffer (and its
    // pages) into the process-wide manager, which causes subsequent MPP
    // queries in the same JVM to hang because ExchangeClient state there
    // is not cleanly reset. abort() is idempotent: if the JVM-side close
    // already drove abort, the inner aborted_ guard short-circuits.
    try {
      abort();
    } catch (...) {
    }
    for (auto& replicas : fragmentTasks_) {
      for (auto& task : replicas) {
        if (task == nullptr) {
          continue;
        }
        const auto& tid = task->taskId();
        LOG(WARNING) << "MppQueryCoordinator[" << queryId_
                     << "]: removeTask(" << tid
                     << ") state=" << static_cast<int>(task->state());
        try {
          bufferManager_->removeTask(tid);
        } catch (...) {
        }
      }
    }
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: destructor exit";
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string MppQueryCoordinator::makeTaskId(
    int32_t fragmentId,
    int32_t replicaIdx) const {
  // Root-fragment output is consumed by this coordinator via
  // OutputBufferManager::getData() (kHttp PartitionedOutput).
  // Non-root fragment edges flow through IBM's UCX / IntraNodeTransfer
  // path; they share the same task-id prefix because routing is decided
  // by adapters, not by the prefix.
  return fmt::format(
      "{}{}-{}-p{}", kTaskIdPrefix, queryId_, fragmentId, replicaIdx);
}

bool MppQueryCoordinator::isTerminalState(TaskState state) {
  return state == TaskState::kFinished || state == TaskState::kCanceled ||
      state == TaskState::kAborted || state == TaskState::kFailed;
}

// ---------------------------------------------------------------------------
// start()
// ---------------------------------------------------------------------------

void MppQueryCoordinator::start() {
  VELOX_CHECK(!started_, "MppQueryCoordinator already started");
  started_ = true;
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: start() "
               << fragmentSpecs_.size() << " fragments, rootFragmentId="
               << rootFragmentId_;

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
  }

  // Decide root drain strategy: sequential for RANGE (order-preserving);
  // round-robin otherwise. If the root has no inbound exchange (single
  // fragment query), replica count is 1 and strategy is moot.
  rootDrainSequential_ = false;
  for (auto& exchange : exchangeSpecs_) {
    if (exchange.consumerFragmentId == rootFragmentId_) {
      if (exchange.partitionType == "RANGE") {
        rootDrainSequential_ = true;
      }
      break; // root consumes at most one exchange
    }
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_
               << "]: rootFragmentId=" << rootFragmentId_
               << " replicas=" << fragmentReplicaCount_[rootFragmentId_]
               << " drain="
               << (rootDrainSequential_ ? "sequential" : "roundRobin");

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
    if (producer >= 0 && static_cast<size_t>(producer) < broadcastFanout.size()) {
      broadcastFanout[producer] =
          std::max(1, fragmentReplicaCount_[exchange.consumerFragmentId]);
    }
  }

  // --- Create Tasks: one per (fragment, replica) ---
  fragmentTasks_.assign(fragmentSpecs_.size(), {});
  for (auto& spec : fragmentSpecs_) {
    const auto replicas = fragmentReplicaCount_[spec.id];
    fragmentTasks_[spec.id].reserve(replicas);
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
    const auto perReplicaDrivers = (bcastN > 0) ? 1
        : std::max({1, spec.numDrivers, inboundN});
    for (int32_t i = 0; i < replicas; ++i) {
      auto taskId = makeTaskId(spec.id, i);
      auto task = Task::create(
          taskId,
          spec.planFragment,
          /*destination=*/0,
          queryCtx_,
          Task::ExecutionMode::kParallel);
      LOG(WARNING) << "MppQueryCoordinator[" << queryId_
                   << "]: starting fragment " << spec.id << " replica " << i
                   << "/" << replicas << " taskId=" << taskId
                   << " drivers=" << perReplicaDrivers
                   << " inboundN=" << inboundN
                   << (bcastN > 0 ? fmt::format(" bcastFanout={}", bcastN)
                                  : std::string{});
      task->start(perReplicaDrivers);
      if (bcastN > 0) {
        // Task::start() has now initializePartitionOutput() registered the
        // kBroadcast OutputBuffer with numBuffers=1 (the plan's placeholder).
        // Expand to N destination buffers AND stamp noMoreBuffers=true so
        // enqueueBroadcastOutputLocked replicates every page to all N
        // consumer replicas and isFinishedLocked() can eventually return.
        // Drivers have been scheduled but scan-leaf producers block on
        // splits (added in Phase 3), so this happens before any enqueue.
        task->updateOutputBuffers(bcastN, /*noMoreBuffers=*/true);
      }
      fragmentTasks_[spec.id].push_back(std::move(task));
    }
  }

  // Phase 2: Wire exchanges via RemoteConnectorSplits (cartesian product).
  //
  // For each exchange E(producer P, consumer C) with M producer replicas and
  // N consumer replicas: each of the N consumer replicas adds M splits (one
  // per producer replica) to its ExchangeNode. Consumer replica i is pinned
  // to destination=i (set at Task::create), so it fetches bucket-i from every
  // producer replica. This mirrors Presto's BasePlanFragmenter pattern: one
  // RemoteSourceNode per consumer, listing all upstream Task IDs; Velox
  // exchange operator fans them in at runtime.
  //
  // Task count stays O(k*N) across the query; split count is O(N*M) per
  // exchange (but splits are lightweight Velox objects, not Task lifecycles).
  for (auto& exchange : exchangeSpecs_) {
    auto& producerReplicas = fragmentTasks_[exchange.producerFragmentId];
    auto& consumerReplicas = fragmentTasks_[exchange.consumerFragmentId];
    VELOX_CHECK(!producerReplicas.empty() && !consumerReplicas.empty());

    std::vector<std::string> producerTaskIds;
    producerTaskIds.reserve(producerReplicas.size());
    for (size_t j = 0; j < producerReplicas.size(); ++j) {
      producerTaskIds.push_back(
          makeTaskId(exchange.producerFragmentId, static_cast<int32_t>(j)));
    }

    LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: wiring exchange "
                 << exchange.id << " type=" << exchange.partitionType
                 << " producerF=" << exchange.producerFragmentId << " ("
                 << producerReplicas.size() << " replicas)"
                 << " -> consumerF=" << exchange.consumerFragmentId << " ("
                 << consumerReplicas.size() << " replicas)"
                 << " exchangeNode=" << exchange.exchangeNodeId;

    // Build the IBM ucx-exchange URL format expected by
    // UcxExchangeSource::extractTaskAndDestinationId:
    //   http://127.0.0.1:<port-3>/v1/task/<taskId>/results/<dest>
    // The "+3" port hack is documented in UcxExchangeSource::create
    // (host port = uri.port() + 3). The taskId is embedded verbatim as a
    // single path component, so kTaskIdPrefix MUST NOT contain "://" or
    // "/" (see kTaskIdPrefix definition). The producer publishes via the
    // same per-process Communicator, so loopback + the taskId suffices;
    // UcxExchangeServer/Source detect same-Communicator and bypass the
    // wire via IntraNodeTransferRegistry.
    // Presto-style: 1 consumer task, add ALL N producer-destination splits to
    // it. Drivers within the consumer task pick up splits dynamically (one
    // driver per producer destination is the typical pattern with
    // perReplicaDrivers = numPartitions).
    auto comm = facebook::velox::ucx_exchange::Communicator::getInstance();
    const int urlPort = static_cast<int>(comm->getListenerPort()) - 3;
    VELOX_CHECK_EQ(
        consumerReplicas.size(),
        1u,
        "Presto-style task model: consumer fragment must have exactly 1 task; "
        "got {}",
        consumerReplicas.size());
    auto& consumerTask = consumerReplicas[0];
    const auto numDestinations =
        exchange.partitionType == "BROADCAST" ? 1 : std::max(1, exchange.numPartitions);
    int32_t splitCount = 0;
    for (int dest = 0; dest < numDestinations; ++dest) {
      for (const auto& prodId : producerTaskIds) {
        const auto url = fmt::format(
            "http://127.0.0.1:{}/v1/task/{}/results/{}", urlPort, prodId, dest);
        consumerTask->addSplit(
            exchange.exchangeNodeId,
            Split(std::make_shared<RemoteConnectorSplit>(url)));
        ++splitCount;
      }
    }
    consumerTask->noMoreSplits(exchange.exchangeNodeId);
    LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: exchange "
                 << exchange.id << " wired (" << splitCount
                 << " splits total, " << numDestinations << " destinations)";
  }

  // Phase 2.5: broadcast producer output buffer fan-out already set in
  // Phase 1 (before task->start()); nothing to do here. See the
  // broadcastFanout precompute above for the ordering rationale.

  // Phase 3: Add file scan splits to scan-containing fragments.
  //
  // Leaf fragments that read from files (e.g., Parquet scans) need their
  // file paths injected as HiveConnectorSplits. This mirrors the pattern
  // in WholeStageResultIterator::noMoreSplits().
  for (auto& spec : fragmentSpecs_) {
    if (spec.scanNodeIds.empty()) {
      continue;
    }
    // Scan-bearing fragments are always leaves (no inbound exchange) and
    // therefore have exactly one replica.
    VELOX_CHECK_EQ(
        fragmentTasks_[spec.id].size(),
        1u,
        "Scan-bearing fragment {} has {} replicas; scans are only wired to "
        "leaf fragments which must be single-replica.",
        spec.id,
        fragmentTasks_[spec.id].size());
    auto& task = fragmentTasks_[spec.id][0];
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
      // This is critical: "test-hive" → Velox Hive connector,
      // "cudf-hive" → cuDF GPU connector (handles type casting).
      const auto& connectorId = (i < spec.scanConnectorIds.size() &&
                                  !spec.scanConnectorIds[i].empty())
          ? spec.scanConnectorIds[i]
          : kHiveConnectorId;

      for (size_t j = 0; j < scanInfo->paths.size(); j++) {
        std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
        if (!scanInfo->partitionColumns.empty() &&
            j < scanInfo->partitionColumns.size()) {
          for (const auto& [key, value] : scanInfo->partitionColumns[j]) {
            partitionKeys[key] = value;
          }
        }

        // Use HiveConnectorSplit with the appropriate connector ID.
        // When connectorId is "cudf-hive", Velox routes to the cuDF connector
        // which handles type coercion (BIGINT→DOUBLE) via libcudf.
        auto connectorSplit =
            std::make_shared<connector::hive::HiveConnectorSplit>(
                connectorId,
                scanInfo->paths[j],
                scanInfo->format,
                scanInfo->starts[j],
                scanInfo->lengths[j],
                partitionKeys);

        task->addSplit(scanNodeId, Split(std::move(connectorSplit)));
      }
      task->noMoreSplits(scanNodeId);

      LOG(WARNING) << "MppQueryCoordinator: added " << scanInfo->paths.size()
                   << " scan splits (connector='" << connectorId
                   << "') to fragment " << spec.id
                   << " scan node " << scanNodeId;
    }
  }

  // Diagnostic watchdog: every 5s dump the state of every (fragId, replicaIdx)
  // Task so we can see where a hang is happening. Cheap: N_tasks log lines
  // per 5s. Stops on destructor.
  watchdogThread_ = std::thread([this]() {
    int tick = 0;
    // Track which failed taskIds we've already logged to avoid repeating the
    // same error message every tick.
    std::unordered_set<std::string> reportedFailures;
    while (!watchdogStop_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
      if (watchdogStop_.load(std::memory_order_acquire)) {
        break;
      }
      ++tick;
      // Summarize per-fragment state counts (more scannable than per-task).
      for (size_t f = 0; f < fragmentTasks_.size(); ++f) {
        int counts[5] = {0, 0, 0, 0, 0}; // running,finished,canceled,aborted,failed
        for (auto& task : fragmentTasks_[f]) {
          if (!task) continue;
          auto s = static_cast<int>(task->state());
          if (s >= 0 && s < 5) {
            counts[s]++;
          }
        }
        LOG(WARNING) << "MppWatchdog[" << queryId_ << "] tick=" << tick
                     << " frag=" << f
                     << " replicas=" << fragmentTasks_[f].size()
                     << " running=" << counts[0]
                     << " finished=" << counts[1]
                     << " canceled=" << counts[2]
                     << " aborted=" << counts[3]
                     << " failed=" << counts[4];
      }
      // Temporary (LOCAL-01 Q3 OOM diagnosis): dump per-operator memory
      // reservation once per tick so we can see which operator is holding
      // 30 GB on the GPU path.
      if (tick <= 6 || tick % 5 == 0) {
        std::function<void(memory::MemoryPool*, int)> dumpPool =
            [&](memory::MemoryPool* pool, int depth) {
              if (!pool) return;
              const auto reserved = pool->reservedBytes();
              const auto peak = pool->peakBytes();
              if (reserved > 0 || peak > 0) {
                LOG(WARNING) << "MemPool[" << queryId_ << "] tick=" << tick
                             << " " << std::string(depth * 2, ' ')
                             << pool->name()
                             << " reserved=" << reserved
                             << " peak=" << peak;
              }
              pool->visitChildren([&](memory::MemoryPool* child) {
                dumpPool(child, depth + 1);
                return true;
              });
            };
        for (size_t f = 0; f < fragmentTasks_.size(); ++f) {
          for (auto& task : fragmentTasks_[f]) {
            if (!task) continue;
            if (task->pool() != nullptr) {
              dumpPool(task->pool(), 0);
            }
          }
        }
      }

      // Dump error message for any newly-failed task. Velox Task::setError is
      // completely silent (only stashes exception_), so without this the only
      // visible signal is the watchdog's failed-count going up.
      for (size_t f = 0; f < fragmentTasks_.size(); ++f) {
        for (auto& task : fragmentTasks_[f]) {
          if (!task) continue;
          if (task->state() != TaskState::kFailed) continue;
          const auto& tid = task->taskId();
          if (reportedFailures.insert(tid).second) {
            LOG(ERROR) << "MppWatchdog[" << queryId_ << "] tick=" << tick
                       << " FAILED-TASK taskId=" << tid
                       << " frag=" << f
                       << " errorMessage={" << task->errorMessage() << "}";
          }
        }
      }
      // Sample a few non-terminal tasks with full taskId for deeper debugging.
      int sampled = 0;
      for (size_t f = 0; f < fragmentTasks_.size() && sampled < 6; ++f) {
        for (auto& task : fragmentTasks_[f]) {
          if (!task) continue;
          if (!isTerminalState(task->state())) {
            LOG(WARNING) << "MppWatchdog[" << queryId_ << "] tick=" << tick
                         << " non-terminal taskId=" << task->taskId()
                         << " state=" << static_cast<int>(task->state())
                         << " numDrivers=" << task->numTotalDrivers()
                         << " numFinishedDrivers="
                         << task->numFinishedDrivers();
            if (++sampled >= 6) break;
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// next() - fetch output from the root fragment
// ---------------------------------------------------------------------------

bool MppQueryCoordinator::fetchNextOutputPage(
    std::vector<std::unique_ptr<SerializedPageBase>>& pages) {
  const auto rootReplicas = fragmentReplicaCount_[rootFragmentId_];
  constexpr int32_t kDestination = 0; // each root Task gathers to dest 0
  constexpr uint64_t kMaxBytes = std::numeric_limits<uint64_t>::max();

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
    bool complete = false;
    auto dataPromise =
        ContinuePromise("MppQueryCoordinator::fetchNextOutputPage");
    auto requestedSeq = rootOutputSequence_[idx];

    LOG(WARNING) << "MppQueryCoordinator[" << queryId_
                 << "]: fetchNextOutputPage rootTask=" << rootTaskId
                 << " seq=" << requestedSeq
                 << " rootState="
                 << static_cast<int>(fragmentTasks_[rootFragmentId_][idx]->state());

    // IBM-baseline: OutputBufferManager exposes getData (IOBuf-vector
    // callback) instead of getPages (SerializedPageBase-vector callback).
    // Wrap each IOBuf in a PrestoSerializedPage so the downstream
    // prepareStreamForDeserialize() call site keeps working unchanged.
    auto ok = bufferManager_->getData(
        rootTaskId,
        kDestination,
        kMaxBytes,
        requestedSeq,
        [&](std::vector<std::unique_ptr<folly::IOBuf>> gotPages,
            int64_t inSequence,
            std::vector<int64_t> /*remainingBytes*/) {
          LOG(WARNING) << "MppQueryCoordinator[" << queryId_
                       << "]: getData callback fired"
                       << " replica=" << idx
                       << " pages=" << gotPages.size()
                       << " inSeq=" << inSequence;
          for (auto& iobuf : gotPages) {
            if (iobuf != nullptr) {
              ++inSequence;
              pages.push_back(
                  std::make_unique<PrestoSerializedPage>(
                      std::move(iobuf)));
            } else {
              complete = true;
            }
          }
          rootOutputSequence_[idx] = inSequence;
          dataPromise.setValue();
        });

    if (!ok) {
      LOG(WARNING) << "MppQueryCoordinator[" << queryId_
                   << "]: getData returned ok=false for replica=" << idx
                   << " (task not registered?)";
      rootReplicaAtEnd_[idx] = true;
      continue;
    }

    dataPromise.getSemiFuture().wait();

    if (complete) {
      rootReplicaAtEnd_[idx] = true;
      bufferManager_->acknowledge(
          rootTaskId, kDestination, rootOutputSequence_[idx]);
      bufferManager_->deleteResults(rootTaskId, kDestination);
    }

    if (!pages.empty()) {
      return true;
    }
    // Empty poll (timeout or end-of-stream with no data): loop to try
    // another replica. For sequential drain, pickNext returns the same
    // replica again until it reaches end; for round-robin, advance.
  }

  noMoreData_ = std::all_of(
      rootReplicaAtEnd_.begin(),
      rootReplicaAtEnd_.end(),
      [](bool b) { return b; });
  return false;
}

RowVectorPtr MppQueryCoordinator::next() {
  VELOX_CHECK(started_, "Must call start() before next()");

  if (noMoreData_) {
    return nullptr;
  }

  // Keep fetching until we get data pages or hit end-of-stream.
  while (!noMoreData_) {
    std::vector<std::unique_ptr<SerializedPageBase>> pages;
    bool gotData = fetchNextOutputPage(pages);

    if (!gotData) {
      return nullptr;
    }

    // Unwrap and return the first non-null page. The root fragment's output
    // is a kHttp PartitionedOutput, which produces CPU-serialized pages
    // (PrestoVectorSerde / IOBuf-backed). Inter-fragment GPU edges flow
    // through IBM's UCX / IntraNodeTransfer path and never reach this
    // coordinator, so there is no GpuSerializedPage shape to handle here.
    // Remaining pages in the batch are discarded here; they will be re-fetched
    // on subsequent next() calls. Sequence ack already advanced inside
    // fetchNextOutputPage() so this is safe.
    for (auto& page : pages) {
      if (!page) {
        continue;
      }

      // Deserialize the CPU page via Presto serde into a RowVector.
      auto inputStream = page->prepareStreamForDeserialize();
      auto outputType = std::dynamic_pointer_cast<const RowType>(
          fragmentSpecs_[rootFragmentId_].planFragment.planNode->outputType());
      VELOX_CHECK(
          outputType != nullptr, "Root fragment must have RowType output");
      if (deserializePool_ == nullptr) {
        deserializePool_ =
            queryCtx_->pool()->addLeafChild("mpp_deserialize");
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
    }
  }

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

// ---------------------------------------------------------------------------
// abort()
// ---------------------------------------------------------------------------

void MppQueryCoordinator::abort(std::chrono::milliseconds perTaskTimeout) {
  // Serialize abort callers. The same coordinator can be aborted by two
  // paths (an explicit JNI nativeAbortMppQuery from the JVM-side close, and
  // the destructor when ~MppQueryHandle drops the shared_ptr); without this
  // gate they would race on requestAbort() + parallel taskCompletionFuture()
  // waits.
  std::lock_guard<std::mutex> lk(abortMutex_);
  if (!started_) {
    LOG(WARNING) << "MppQueryCoordinator[" << queryId_
                 << "]: abort() called before start(), skipping";
    return;
  }
  if (aborted_) {
    LOG(WARNING) << "MppQueryCoordinator[" << queryId_
                 << "]: abort() already completed, skipping";
    return;
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_
               << "]: abort() begin perTaskTimeoutMs="
               << perTaskTimeout.count();

  // Phase 1: fire requestAbort() on every non-terminal task. This is
  // non-blocking; each call returns a future. We don't need those futures
  // because we'll wait on taskCompletionFuture() in Phase 2 — that's
  // realized whenever the task is no longer running, regardless of which
  // call drove it to terminal.
  size_t firedCount = 0;
  size_t alreadyTerminalCount = 0;
  for (auto& replicas : fragmentTasks_) {
    for (auto& task : replicas) {
      if (task == nullptr) {
        continue;
      }
      const auto state = task->state();
      if (!isTerminalState(state)) {
        LOG(WARNING) << "MppQueryCoordinator[" << queryId_
                     << "]: requestAbort(" << task->taskId()
                     << ") state=" << static_cast<int>(state);
        try {
          task->requestAbort();
        } catch (const std::exception& e) {
          LOG(ERROR) << "MppQueryCoordinator[" << queryId_
                     << "]: requestAbort threw for " << task->taskId()
                     << ": " << e.what();
        } catch (...) {
          LOG(ERROR) << "MppQueryCoordinator[" << queryId_
                     << "]: requestAbort threw unknown exception for "
                     << task->taskId();
        }
        ++firedCount;
      } else {
        ++alreadyTerminalCount;
      }
    }
  }
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_
               << "]: abort() requested " << firedCount
               << " task(s), " << alreadyTerminalCount << " already terminal";

  // Phase 2: bounded wait for each task to reach a terminal state. We use
  // taskCompletionFuture() which is realized when the task is no longer
  // running. Cap each wait so a stuck task can't hold up shutdown forever —
  // a leak warning at process exit is strictly better than an infinite hang
  // or a removePool() VELOX_CHECK abort.
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
        // folly SemiFuture::wait(timeout) blocks up to the timeout and
        // returns the (rvalue-reference) future itself; check .isReady()
        // afterwards to see whether the timeout fired or the task became
        // terminal.
        auto waited = task->taskCompletionFuture().wait(perTaskTimeout);
        if (waited.isReady() || isTerminalState(task->state())) {
          ++terminalAfterWait;
        } else {
          ++timedOut;
          LOG(ERROR) << "MppQueryCoordinator[" << queryId_
                     << "]: abort() TIMEOUT waiting for taskId="
                     << task->taskId()
                     << " state=" << static_cast<int>(task->state())
                     << " numDrivers=" << task->numTotalDrivers()
                     << " numFinishedDrivers="
                     << task->numFinishedDrivers()
                     << " — task may still hold MemoryPool reservations";
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << "MppQueryCoordinator[" << queryId_
                   << "]: taskCompletionFuture wait threw for "
                   << task->taskId() << ": " << e.what();
      } catch (...) {
        LOG(ERROR) << "MppQueryCoordinator[" << queryId_
                   << "]: taskCompletionFuture wait threw unknown for "
                   << task->taskId();
      }
    }
  }

  aborted_ = true;
  LOG(WARNING) << "MppQueryCoordinator[" << queryId_ << "]: abort() end"
               << " terminal=" << terminalAfterWait
               << " timedOut=" << timedOut;
}

// ---------------------------------------------------------------------------
// waitForCompletion()
// ---------------------------------------------------------------------------

void MppQueryCoordinator::waitForCompletion() {
  VELOX_CHECK(started_, "Must call start() before waitForCompletion()");

  std::exception_ptr firstError;

  for (auto& replicas : fragmentTasks_) {
    for (auto& task : replicas) {
      if (isTerminalState(task->state())) {
        // Already done; check for error.
        if (auto error = task->error()) {
          if (!firstError) {
            firstError = error;
          }
        }
        continue;
      }

      // Block until this task reaches a terminal state.
      auto future = task->taskCompletionFuture();
      std::move(future).wait();

      if (auto error = task->error()) {
        if (!firstError) {
          firstError = error;
        }
      }
    }
  }

  // Clean up output buffer entries for all tasks.
  for (auto& replicas : fragmentTasks_) {
    for (auto& task : replicas) {
      bufferManager_->removeTask(task->taskId());
    }
  }

  if (firstError) {
    std::rethrow_exception(firstError);
  }
}

} // namespace gluten
