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
#include "WholeStageResultIterator.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include "VeloxBackend.h"
#include "VeloxPlanConverter.h"
#include "VeloxRuntime.h"
#include "config/VeloxConfig.h"
#include "utils/ConfigExtractor.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/exec/BlockingReason.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PlanNodeStats.h"
#ifdef GLUTEN_ENABLE_GPU
#include <cudf/io/types.hpp>
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#endif
#include "operators/plannodes/RowVectorStream.h"


using namespace facebook;

namespace gluten {

namespace {

// metrics
const std::string kDynamicFiltersProduced = "dynamicFiltersProduced";
const std::string kDynamicFiltersAccepted = "dynamicFiltersAccepted";
const std::string kReplacedWithDynamicFilterRows = "replacedWithDynamicFilterRows";
const std::string kFlushRowCount = "flushRowCount";
const std::string kNumCoalescedBatches = "numCoalescedBatches";
const std::string kLoadedToValueHook = "loadedToValueHook";
const std::string kBloomFilterBlocksByteSize = "bloomFilterSize";
const std::string kTotalScanTime = "totalScanTime";
const std::string kSkippedSplits = "skippedSplits";
const std::string kProcessedSplits = "processedSplits";
const std::string kSkippedStrides = "skippedStrides";
const std::string kProcessedStrides = "processedStrides";
const std::string kRemainingFilterTime = "totalRemainingFilterTime";
const std::string kIoWaitTime = "ioWaitWallNanos";
const std::string kStorageReadBytes = "storageReadBytes";
const std::string kLocalReadBytes = "localReadBytes";
const std::string kRamReadBytes = "ramReadBytes";
const std::string kPreloadSplits = "readyPreloadedSplits";
const std::string kPageLoadTime = "pageLoadTimeNs";
const std::string kDataSourceAddSplitWallNanos = "dataSourceAddSplitWallNanos";
const std::string kWaitForPreloadSplitNanos = "waitForPreloadSplitNanos";
const std::string kDataSourceReadWallNanos = "dataSourceReadWallNanos";
const std::string kNumWrittenFiles = "numWrittenFiles";
const std::string kWriteIOTime = "writeIOWallNanos";
const std::string kPinnedAllocBytes = "pinnedAllocBytes";
const std::string kPageableAllocBytes = "pageableAllocBytes";
const std::string kGpuComputeNanos = "gpuComputeNanos";

// others
const std::string kHiveDefaultPartition = "__HIVE_DEFAULT_PARTITION__";
constexpr uint64_t kParallelResultQueueMaxBytes = 64UL << 20;
constexpr uint32_t kParallelTaskMaxDrivers = 1;

bool planRequiresParallelExecution(
    const std::shared_ptr<const velox::core::PlanNode>& planNode) {
  if (std::dynamic_pointer_cast<const velox::core::ExchangeNode>(planNode) !=
          nullptr ||
      std::dynamic_pointer_cast<const velox::core::PartitionedOutputNode>(
          planNode) != nullptr) {
    return true;
  }
  for (const auto& source : planNode->sources()) {
    if (planRequiresParallelExecution(source)) {
      return true;
    }
  }
  return false;
}

bool planRootIsPartitionedOutput(
    const std::shared_ptr<const velox::core::PlanNode>& planNode) {
  return std::dynamic_pointer_cast<const velox::core::PartitionedOutputNode>(
             planNode) != nullptr;
}

std::string describeUcxOutputQueue(
    const std::shared_ptr<velox::exec::Task>& task) {
#ifdef GLUTEN_ENABLE_GPU
  if (task == nullptr) {
    return "task=null";
  }
  try {
    auto queueMgr =
        facebook::velox::ucx_exchange::UcxOutputQueueManager::getInstanceRef();
    auto stats = queueMgr->stats(task->taskId());
    if (!stats.has_value()) {
      return "ucxQueue=missing";
    }
    return fmt::format(
        "ucxQueue=noMoreData:{} finished:{} bufferedBytes:{} "
        "bufferedPages:{} totalBytesSent:{} totalRowsSent:{} "
        "totalPagesSent:{} buffers:{}",
        stats->noMoreData,
        stats->finished,
        stats->bufferedBytes,
        stats->bufferedPages,
        stats->totalBytesSent,
        stats->totalRowsSent,
        stats->totalPagesSent,
        stats->buffersStats.size());
  } catch (const std::exception& e) {
    return fmt::format("ucxQueueStatsError={}", e.what());
  }
#else
  return "ucxQueue=disabled";
#endif
}

#ifdef GLUTEN_ENABLE_GPU
bool ucxOutputQueueNoMoreData(
    const std::shared_ptr<velox::exec::Task>& task) {
  if (task == nullptr) {
    return false;
  }
  try {
    auto queueMgr =
        facebook::velox::ucx_exchange::UcxOutputQueueManager::getInstanceRef();
    auto stats = queueMgr->stats(task->taskId());
    return stats.has_value() && stats->noMoreData;
  } catch (const std::exception& e) {
    VLOG(1) << "Failed to inspect UCX output queue noMoreData taskId="
            << task->taskId() << " error=" << e.what();
    return false;
  }
}

int64_t detachedUcxTaskTimeoutMs() {
  if (const char* value =
          std::getenv("GLUTEN_UCX_SHUFFLE_DETACHED_TASK_TIMEOUT_MS")) {
    try {
      return std::max<int64_t>(0, std::stoll(value));
    } catch (...) {
      LOG(WARNING) << "Invalid GLUTEN_UCX_SHUFFLE_DETACHED_TASK_TIMEOUT_MS="
                   << value << ", using default 300000";
    }
  }
  return 300000;
}

class DetachedUcxProducerTaskRegistry {
 public:
  void retain(
      std::shared_ptr<velox::exec::Task> task,
      std::shared_ptr<folly::Executor> taskExecutor,
      std::shared_ptr<folly::Executor> spillExecutor,
      MemoryManager* memoryManager,
      const std::string& reason) {
    if (task == nullptr) {
      return;
    }
    startReaperIfNeeded();
    const auto taskId = task->taskId();
    MemoryManager::retainForAsyncTask(
        memoryManager,
        fmt::format("detached UCX producer taskId={} reason={}", taskId, reason));
    const auto now = std::chrono::steady_clock::now();
    DetachedTask entry{
        std::move(task),
        std::move(taskExecutor),
        std::move(spillExecutor),
        memoryManager,
        now,
        now + std::chrono::milliseconds(detachedUcxTaskTimeoutMs()),
        reason};
    auto logTask = entry.task;
    {
      std::lock_guard<std::mutex> l(mutex_);
      tasks_[taskId] = std::move(entry);
    }
    LOG(WARNING) << "[UCX-DETACH] retained root partitioned output task"
                 << " taskId=" << taskId << " reason=" << reason
                 << " queueStats=" << describeUcxOutputQueue(logTask);
  }

 private:
  struct DetachedTask {
    std::shared_ptr<velox::exec::Task> task;
    std::shared_ptr<folly::Executor> taskExecutor;
    std::shared_ptr<folly::Executor> spillExecutor;
    MemoryManager* memoryManager;
    std::chrono::steady_clock::time_point retainedAt;
    std::chrono::steady_clock::time_point deadline;
    std::string reason;
  };

  void startReaperIfNeeded() {
    bool expected = false;
    if (!reaperStarted_.compare_exchange_strong(expected, true)) {
      return;
    }
    std::thread([this]() { reaperLoop(); }).detach();
  }

  void reaperLoop() {
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      reapOnce();
    }
  }

  void reapOnce() {
    std::vector<std::pair<std::string, DetachedTask>> finished;
    std::vector<std::pair<std::string, DetachedTask>> expired;
    const auto now = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> l(mutex_);
      for (auto it = tasks_.begin(); it != tasks_.end();) {
        const auto& taskId = it->first;
        auto& entry = it->second;
        if (isReadyToRelease(taskId, entry)) {
          finished.emplace_back(taskId, std::move(entry));
          it = tasks_.erase(it);
        } else if (now >= entry.deadline) {
          expired.emplace_back(taskId, std::move(entry));
          it = tasks_.erase(it);
        } else {
          ++it;
        }
      }
    }
    for (auto& pair : finished) {
      releaseFinished(pair.first, pair.second);
    }
    for (auto& pair : expired) {
      cancelExpired(pair.first, pair.second);
    }
  }

  bool isReadyToRelease(const std::string& taskId, DetachedTask& entry) {
    try {
      auto future = entry.task->taskCompletionFuture();
      if (future.valid() && future.isReady()) {
        return true;
      }
      auto queueMgr =
          facebook::velox::ucx_exchange::UcxOutputQueueManager::getInstanceRef();
      auto stats = queueMgr->stats(taskId);
      if (stats.has_value()) {
        return stats->finished;
      }
      return !entry.task->isRunning();
    } catch (const std::exception& e) {
      LOG(WARNING) << "[UCX-DETACH] failed to inspect detached task"
                   << " taskId=" << taskId << " error=" << e.what()
                   << "; keeping task retained";
      return false;
    }
  }

  void releaseFinished(const std::string& taskId, DetachedTask& entry) {
    try {
      auto future = entry.task->taskCompletionFuture();
      if (future.valid() && future.isReady()) {
        future.wait();
      } else if (entry.task->isRunning()) {
        auto cancelFuture = entry.task->requestCancel();
        if (cancelFuture.valid()) {
          cancelFuture.wait();
        }
      } else if (future.valid()) {
        future.wait();
      }
      LOG(WARNING) << "[UCX-DETACH] releasing drained detached task"
                   << " taskId=" << taskId
                   << " retainedMs="
                   << std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - entry.retainedAt)
                          .count()
                   << " queueStats=" << describeUcxOutputQueue(entry.task);
    } catch (const std::exception& e) {
      LOG(WARNING) << "[UCX-DETACH] releasing detached task after error"
                   << " taskId=" << taskId << " error=" << e.what();
    }
    releaseRetainedResources(taskId, entry, "drained");
  }

  void cancelExpired(const std::string& taskId, DetachedTask& entry) {
    try {
      LOG(WARNING) << "[UCX-DETACH] cancelling expired detached task"
                   << " taskId=" << taskId << " reason=" << entry.reason
                   << " queueStats=" << describeUcxOutputQueue(entry.task);
      auto future = entry.task->requestCancel();
      if (future.valid()) {
        future.wait();
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "[UCX-DETACH] failed to cancel expired detached task"
                   << " taskId=" << taskId << " error=" << e.what();
    }
    releaseRetainedResources(taskId, entry, "expired");
  }

  void releaseRetainedResources(
      const std::string& taskId,
      DetachedTask& entry,
      const std::string& reason) {
    if (entry.task != nullptr) {
      auto deletionFuture = entry.task->taskDeletionFuture();
      entry.task.reset();
      if (deletionFuture.valid()) {
        std::move(deletionFuture).wait(std::chrono::seconds(30));
      }
    }
    entry.taskExecutor.reset();
    entry.spillExecutor.reset();
    if (entry.memoryManager != nullptr) {
      MemoryManager::releaseAsyncTaskRetain(
          entry.memoryManager,
          fmt::format("detached UCX producer taskId={} {}", taskId, reason));
      entry.memoryManager = nullptr;
    }
  }

  std::mutex mutex_;
  std::unordered_map<std::string, DetachedTask> tasks_;
  std::atomic<bool> reaperStarted_{false};
};

DetachedUcxProducerTaskRegistry& detachedUcxProducerTaskRegistry() {
  static auto* registry = new DetachedUcxProducerTaskRegistry();
  return *registry;
}
#else
bool ucxOutputQueueNoMoreData(
    const std::shared_ptr<velox::exec::Task>& task) {
  (void)task;
  return false;
}
#endif

} // namespace

class WholeStageParallelResultQueue {
 public:
  explicit WholeStageParallelResultQueue(uint64_t maxBytes)
      : maxBytes_(maxBytes) {}

  velox::exec::BlockingReason enqueue(
      velox::RowVectorPtr vector,
      bool drained,
      velox::ContinueFuture* future) {
    if (vector == nullptr) {
      std::lock_guard<std::mutex> l(mutex_);
      if (drained) {
        ++drainedProducers_;
      } else {
        ++finishedProducers_;
      }
      if (consumerBlocked_) {
        consumerBlocked_ = false;
        consumerPromise_.setValue();
      }
      return velox::exec::BlockingReason::kNotBlocked;
    }

    const auto bytes = vector->retainedSize();
    std::lock_guard<std::mutex> l(mutex_);
    if (closed_) {
      throw std::runtime_error("WholeStage parallel result queue is closed");
    }

    queue_.push_back({std::move(vector), bytes});
    totalBytes_ += bytes;
    if (consumerBlocked_) {
      consumerBlocked_ = false;
      consumerPromise_.setValue();
    }

    if (totalBytes_ > maxBytes_) {
      auto [promise, producerFuture] =
          velox::makeVeloxContinuePromiseContract(
              "WholeStageParallelResultQueue::enqueue");
      producerUnblockPromises_.emplace_back(std::move(promise));
      *future = std::move(producerFuture);
      return velox::exec::BlockingReason::kWaitForConsumer;
    }
    return velox::exec::BlockingReason::kNotBlocked;
  }

  velox::RowVectorPtr dequeue() {
    for (;;) {
      velox::RowVectorPtr vector;
      std::vector<velox::ContinuePromise> mayContinue;
      {
        std::lock_guard<std::mutex> l(mutex_);
        if (closed_) {
          return nullptr;
        }

        if (!queue_.empty()) {
          auto result = std::move(queue_.front());
          queue_.pop_front();
          totalBytes_ -= result.bytes;
          vector = std::move(result.vector);
          if (totalBytes_ < maxBytes_ / 2) {
            mayContinue = std::move(producerUnblockPromises_);
          }
        } else if (
            numProducers_.has_value() &&
            finishedProducers_ >= numProducers_.value()) {
          return nullptr;
        } else if (
            numProducers_.has_value() &&
            drainedProducers_ >= numProducers_.value()) {
          return nullptr;
        }

        if (vector == nullptr) {
          consumerBlocked_ = true;
          consumerPromise_ =
              velox::ContinuePromise("WholeStageParallelResultQueue::dequeue");
          consumerFuture_ = consumerPromise_.getFuture();
        }
      }

      for (auto& promise : mayContinue) {
        promise.setValue();
      }
      if (vector != nullptr) {
        return vector;
      }
      consumerFuture_.wait();
    }
  }

  void setNumProducers(int32_t n) {
    std::lock_guard<std::mutex> l(mutex_);
    numProducers_ = n;
  }

  void close() {
    std::lock_guard<std::mutex> l(mutex_);
    closed_ = true;
    for (auto& promise : producerUnblockPromises_) {
      promise.setValue();
    }
    producerUnblockPromises_.clear();
    if (consumerBlocked_) {
      consumerBlocked_ = false;
      consumerPromise_.setValue();
    }
  }

 private:
  struct Entry {
    velox::RowVectorPtr vector;
    uint64_t bytes;
  };

  std::deque<Entry> queue_;
  uint64_t totalBytes_ = 0;
  const uint64_t maxBytes_;
  std::optional<int32_t> numProducers_;
  int32_t finishedProducers_ = 0;
  int32_t drainedProducers_ = 0;
  std::mutex mutex_;
  std::vector<velox::ContinuePromise> producerUnblockPromises_;
  bool consumerBlocked_ = false;
  velox::ContinuePromise consumerPromise_{
      velox::ContinuePromise::makeEmpty()};
  velox::ContinueFuture consumerFuture_;
  bool closed_ = false;
};

WholeStageResultIterator::WholeStageResultIterator(
    VeloxMemoryManager* memoryManager,
    const std::shared_ptr<const facebook::velox::core::PlanNode>& planNode,
    const std::vector<facebook::velox::core::PlanNodeId>& scanNodeIds,
    const std::vector<std::shared_ptr<SplitInfo>>& scanInfos,
    const std::vector<facebook::velox::core::PlanNodeId>& streamIds,
    const std::string spillDir,
    const std::shared_ptr<facebook::velox::config::ConfigBase>& veloxCfg,
    const SparkTaskInfo& taskInfo,
    const std::string& taskIdOverride)
    : memoryManager_(memoryManager),
      veloxCfg_(veloxCfg),
#ifdef GLUTEN_ENABLE_GPU
      enableCudf_(veloxCfg_->get<bool>(kCudfEnabled, kCudfEnabledDefault)),
#endif
      taskInfo_(taskInfo),
      veloxPlan_(planNode),
      scanNodeIds_(scanNodeIds),
      scanInfos_(scanInfos),
      streamIds_(streamIds) {
  spillStrategy_ = veloxCfg_->get<std::string>(kSpillStrategy, kSpillStrategyDefaultValue);
  auto spillThreadNum = veloxCfg_->get<uint32_t>(kSpillThreadNum, kSpillThreadNumDefaultValue);
  if (spillThreadNum > 0) {
    spillExecutor_ = std::make_shared<folly::CPUThreadPoolExecutor>(spillThreadNum);
  }
  getOrderedNodeIds(veloxPlan_, orderedNodeIds_);
  requiresParallelExecution_ = planRequiresParallelExecution(veloxPlan_);
  parallelTaskProducesOutput_ =
      requiresParallelExecution_ && !planRootIsPartitionedOutput(veloxPlan_);
  if (requiresParallelExecution_) {
    taskExecutor_ = std::make_shared<folly::CPUThreadPoolExecutor>(
        kParallelTaskMaxDrivers);
    if (parallelTaskProducesOutput_) {
      parallelResultQueue_ =
          std::make_shared<WholeStageParallelResultQueue>(
              kParallelResultQueueMaxBytes);
    }
  }

  auto fileSystem = velox::filesystems::getFileSystem(spillDir, nullptr);
  GLUTEN_CHECK(fileSystem != nullptr, "File System for spilling is null!");
  fileSystem->mkdir(spillDir);
  velox::common::SpillDiskOptions spillOpts{
      .spillDirPath = spillDir, .spillDirCreated = true, .spillDirCreateCb = nullptr};

  // Create task instance.
  std::unordered_set<velox::core::PlanNodeId> emptySet;
  velox::core::PlanFragment planFragment{planNode, velox::core::ExecutionStrategy::kUngrouped, 1, emptySet};
  std::shared_ptr<velox::core::QueryCtx> queryCtx = createNewVeloxQueryCtx();
  const auto veloxTaskId = taskIdOverride.empty()
      ? fmt::format(
            "Gluten_Stage_{}_TID_{}_VTID_{}",
            std::to_string(taskInfo_.stageId),
            std::to_string(taskInfo_.taskId),
            std::to_string(taskInfo.vId))
      : taskIdOverride;
  velox::exec::Consumer consumer{};
  std::function<void(std::exception_ptr)> onError;
  if (parallelResultQueue_ != nullptr) {
    auto queueHolder = std::weak_ptr<WholeStageParallelResultQueue>(
        parallelResultQueue_);
    consumer =
        [queueHolder, veloxTaskId](
            velox::RowVectorPtr vector,
            bool drained,
            velox::ContinueFuture* future) {
          auto queue = queueHolder.lock();
          if (queue == nullptr) {
            LOG(ERROR) << "WholeStage parallel result queue destroyed, taskId="
                       << veloxTaskId;
            return velox::exec::BlockingReason::kNotBlocked;
          }
          return queue->enqueue(std::move(vector), drained, future);
        };
    onError = [queueHolder, veloxTaskId](std::exception_ptr) {
      auto queue = queueHolder.lock();
      if (queue == nullptr) {
        LOG(ERROR) << "WholeStage parallel result queue destroyed, taskId="
                   << veloxTaskId;
        return;
      }
      queue->close();
    };
  }

  task_ = velox::exec::Task::create(
      veloxTaskId,
      std::move(planFragment),
      0,
      std::move(queryCtx),
      requiresParallelExecution_
          ? velox::exec::Task::ExecutionMode::kParallel
          : velox::exec::Task::ExecutionMode::kSerial,
      /*consumer=*/std::move(consumer),
      /*memoryArbitrationPriority=*/0,
      /*spillDiskOpts=*/spillOpts,
      /*onError=*/std::move(onError));
  LOG(INFO) << "[WS-ITER] created taskId=" << veloxTaskId
            << " rootNode=" << veloxPlan_->id()
            << " rootPlan=" << veloxPlan_->toString()
            << " requiresParallelExecution=" << requiresParallelExecution_
            << " parallelTaskProducesOutput=" << parallelTaskProducesOutput_
            << " scanNodes=" << scanNodeIds_.size()
            << " streamNodes=" << streamIds_.size();
  if (!requiresParallelExecution_ && !task_->supportSerialExecutionMode()) {
    throw std::runtime_error("Task doesn't support single threaded execution: " + planNode->toString());
  }

  // Generate splits for all scan nodes.
  splits_.reserve(scanInfos.size());
  if (scanNodeIds.size() != scanInfos.size()) {
    throw std::runtime_error("Invalid scan information.");
  }

  for (const auto& scanInfo : scanInfos) {
    // Get the information for TableScan.
    // Partition index in scan info is not used.
    const auto& paths = scanInfo->paths;
    const auto& starts = scanInfo->starts;
    const auto& lengths = scanInfo->lengths;
    const auto& properties = scanInfo->properties;
    const auto& format = scanInfo->format;
    const auto& partitionColumns = scanInfo->partitionColumns;
    const auto& metadataColumns = scanInfo->metadataColumns;
#ifdef GLUTEN_ENABLE_GPU
    // Under the pre-condition that all the split infos has same partition column and format.
    const auto canUseCudfConnector = scanInfo->canUseCudfConnector();
#endif
    std::vector<std::shared_ptr<velox::connector::ConnectorSplit>> connectorSplits;
    connectorSplits.reserve(paths.size());
#ifdef GLUTEN_ENABLE_GPU
    struct CudfFileInfo {
      std::string path;
      uint64_t start;
      uint64_t length;
      std::unordered_map<std::string, std::string> infoColumns;
    };
    std::vector<CudfFileInfo> cudfFileInfos;
    cudfFileInfos.reserve(paths.size());
#endif
    for (int idx = 0; idx < paths.size(); idx++) {
      auto metadataColumn = metadataColumns[idx];
      std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
      if (!partitionColumns.empty()) {
        auto partitionColumn = partitionColumns[idx];
        constructPartitionColumns(partitionKeys, partitionColumn);
      }

      std::shared_ptr<velox::connector::ConnectorSplit> split;
      if (auto icebergSplitInfo = std::dynamic_pointer_cast<IcebergSplitInfo>(scanInfo)) {
        // Set Iceberg split (never coalesced).
        std::unordered_map<std::string, std::string> customSplitInfo{{"table_format", "hive-iceberg"}};
        std::vector<velox::connector::hive::iceberg::IcebergDeleteFile> deleteFiles;
        if (idx < icebergSplitInfo->deleteFilesVec.size()) {
          const auto& splitDeleteFiles = icebergSplitInfo->deleteFilesVec[idx];
          deleteFiles.reserve(splitDeleteFiles.size());
          for (const auto& deleteFile : splitDeleteFiles) {
            deleteFiles.emplace_back(deleteFile);
          }
        }
        auto connectorId = kHiveConnectorId;
#ifdef GLUTEN_ENABLE_GPU
        if (canUseCudfConnector && enableCudf_ &&
            veloxCfg_->get<bool>(kCudfEnableTableScan, kCudfEnableTableScanDefault)) {
          connectorId = kCudfIcebergConnectorId;
        }
#endif
        split = std::make_shared<velox::connector::hive::iceberg::HiveIcebergSplit>(
            connectorId,
            paths[idx],
            format,
            starts[idx],
            lengths[idx],
            partitionKeys,
            std::nullopt,
            customSplitInfo,
            nullptr,
            true,
            std::move(deleteFiles),
            std::unordered_map<std::string, std::string>(),
            properties[idx]);
        connectorSplits.emplace_back(split);
      } else {
#ifdef GLUTEN_ENABLE_GPU
        const bool useCudf = canUseCudfConnector && enableCudf_ &&
            veloxCfg_->get<bool>(kCudfEnableTableScan, kCudfEnableTableScanDefault);
        if (useCudf) {
          // For CUDF connector: collect file info for coalesced split building.
          // Actual CudfHiveConnectorSplit creation happens below after the loop.
          // Strip URI scheme prefixes so downstream std::filesystem / std::ifstream
          // calls receive native OS paths (e.g., "file:///data/..." → "///data/...").
          std::string cleanedPath = paths[idx];
          constexpr std::string_view kFilePrefix = "file:";
          constexpr std::string_view kS3APrefix = "s3a:";
          if (cleanedPath.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
            cleanedPath = cleanedPath.substr(kFilePrefix.size());
          } else if (cleanedPath.compare(0, kS3APrefix.size(), kS3APrefix) == 0) {
            cleanedPath.erase(kS3APrefix.size() - 2, 1);
          }
          cudfFileInfos.push_back(
              {std::move(cleanedPath), starts[idx], lengths[idx], metadataColumn});
        } else {
#endif
          split = std::make_shared<velox::connector::hive::HiveConnectorSplit>(
              kHiveConnectorId,
              paths[idx],
              format,
              starts[idx],
              lengths[idx],
              partitionKeys,
              std::nullopt /*tableBucketName*/,
              std::unordered_map<std::string, std::string>(),
              nullptr,
              std::unordered_map<std::string, std::string>(),
              0,
              true,
              metadataColumn,
              properties[idx]);
          connectorSplits.emplace_back(split);
#ifdef GLUTEN_ENABLE_GPU
        }
#endif
      }
    }

#ifdef GLUTEN_ENABLE_GPU
    // Coalesce CUDF file splits: group small files into coalesced splits
    // to reduce per-file overhead and produce larger GPU batches.
    // Split-file ranges (start != 0) are NOT coalesced because the pre-read
    // buffer path creates a HOST_BUFFER datasource from [start, start+length)
    // which the cuDF Parquet reader would treat as a complete file, failing
    // the header/footer magic check.
    if (!cudfFileInfos.empty()) {
      // IBM baseline drops CoalescedFileRange + the 7-arg ctor. Per ferd:
      // IO coalescing optimization not needed; one split per file is fine.
      for (const auto& f : cudfFileInfos) {
        auto cudfSplit = std::make_shared<
            velox::cudf_velox::connector::hive::CudfHiveConnectorSplit>(
            kCudfHiveConnectorId,
            f.path,
            f.start,
            f.length,
            /*splitWeight=*/0,
            f.infoColumns);
        connectorSplits.emplace_back(std::move(cudfSplit));
      }
      VLOG(1) << "Built " << connectorSplits.size() << " CUDF splits (one per file)";
    }
#endif

    std::vector<velox::exec::Split> scanSplits;
    scanSplits.reserve(connectorSplits.size());
    for (const auto& connectorSplit : connectorSplits) {
      // Bucketed group id (-1 means 'none').
      int32_t groupId = -1;
      scanSplits.emplace_back(velox::exec::Split(folly::copy(connectorSplit), groupId));
    }
    splits_.emplace_back(scanSplits);
  }
}

WholeStageResultIterator::~WholeStageResultIterator() {
  closeVeloxTask();
  // GPU thread-region tracking dropped in IBM-baseline switch (GpuGuard /
  // endGpuRegion removed). The diagnostic semaphore-permit cleanup is no
  // longer needed; cuDF stream-pool + RMM thread-safety handle concurrency.
}

void WholeStageResultIterator::closeVeloxTask() {
  if (parallelResultQueue_ != nullptr) {
    parallelResultQueue_->close();
  }
  if (task_ == nullptr) {
    parallelResultQueue_.reset();
    taskExecutor_.reset();
    return;
  }

  const auto taskId = task_->taskId();
  if (detachedUcxPartitionedOutputTask_) {
    LOG(WARNING) << "Releasing detached UCX partitioned output task from "
                 << "WholeStageResultIterator without cancellation, taskId="
                 << taskId << ", state=" << static_cast<int>(task_->state())
                 << ", running=" << task_->isRunning()
                 << ", queueStats=" << describeUcxOutputQueue(task_);
    task_.reset();
    parallelResultQueue_.reset();
    taskExecutor_.reset();
    spillExecutor_.reset();
    return;
  }
  std::shared_ptr<velox::memory::MemoryPool> taskPool;
  if (task_->pool() != nullptr) {
    taskPool = task_->pool()->shared_from_this();
  }
  try {
    LOG(INFO) << "Closing Velox task from WholeStageResultIterator, taskId="
              << taskId << ", state=" << static_cast<int>(task_->state())
              << ", running=" << task_->isRunning()
              << ", taskPoolUseCount="
              << (taskPool == nullptr ? 0 : taskPool.use_count());
    if (task_->isRunning()) {
      auto future = task_->requestCancel();
      if (future.valid()) {
        future.wait();
      }
    } else {
      auto future = task_->taskCompletionFuture();
      if (future.valid()) {
        future.wait();
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed while closing Velox task from WholeStageResultIterator, taskId="
               << taskId << ", error=" << e.what();
  } catch (...) {
    LOG(ERROR) << "Failed while closing Velox task from WholeStageResultIterator, taskId="
               << taskId << ", unknown error";
  }

  auto deletionFuture = task_->taskDeletionFuture();
  task_.reset();
  if (deletionFuture.valid()) {
    std::move(deletionFuture).wait(std::chrono::seconds(30));
  }
  LOG(INFO) << "Released Velox task from WholeStageResultIterator, taskId="
            << taskId << ", taskPoolUseCountAfterTaskReset="
            << (taskPool == nullptr ? 0 : taskPool.use_count());
  taskPool.reset();
  parallelResultQueue_.reset();
  taskExecutor_.reset();
}

void WholeStageResultIterator::detachUcxPartitionedOutputTask(
    const std::string& reason) {
#ifdef GLUTEN_ENABLE_GPU
  if (detachedUcxPartitionedOutputTask_ || task_ == nullptr) {
    return;
  }
  detachedUcxProducerTaskRegistry().retain(
      task_, taskExecutor_, spillExecutor_, memoryManager_, reason);
  detachedUcxPartitionedOutputTask_ = true;
  LOG(WARNING) << "[UCX-DETACH] detached root partitioned output task from "
               << "Spark task lifecycle taskId=" << task_->taskId()
               << " reason=" << reason
               << " queueStats=" << describeUcxOutputQueue(task_);
#else
  (void)reason;
#endif
}

std::shared_ptr<velox::core::QueryCtx> WholeStageResultIterator::createNewVeloxQueryCtx() {
  std::unordered_map<std::string, std::shared_ptr<velox::config::ConfigBase>> connectorConfigs;
  auto hiveConnectorSessionConfig = createHiveConnectorSessionConfig(veloxCfg_);
  connectorConfigs[kHiveConnectorId] = hiveConnectorSessionConfig;
#ifdef GLUTEN_ENABLE_GPU
  connectorConfigs[kCudfHiveConnectorId] = hiveConnectorSessionConfig;
  connectorConfigs[kCudfIcebergConnectorId] = hiveConnectorSessionConfig;
#endif
  std::shared_ptr<velox::core::QueryCtx> ctx = velox::core::QueryCtx::create(
      taskExecutor_.get(),
      facebook::velox::core::QueryConfig{getQueryContextConf()},
      connectorConfigs,
      gluten::VeloxBackend::get()->getAsyncDataCache(),
      memoryManager_->getAggregateMemoryPool(),
      spillExecutor_.get(),
      fmt::format(
          "Gluten_Stage_{}_TID_{}_VTID_{}",
          std::to_string(taskInfo_.stageId),
          std::to_string(taskInfo_.taskId),
          std::to_string(taskInfo_.vId)));
  return ctx;
}

void WholeStageResultIterator::startParallelTaskIfNeeded() {
  if (parallelTaskStarted_) {
    return;
  }
  parallelTaskStarted_ = true;
  try {
    LOG(INFO) << "[WS-ITER] starting parallel task taskId=" << task_->taskId()
              << " requestedDrivers=" << kParallelTaskMaxDrivers
              << " producesOutput=" << parallelTaskProducesOutput_;
    task_->start(kParallelTaskMaxDrivers);
    LOG(INFO) << "[WS-ITER] started parallel task taskId=" << task_->taskId()
              << " numOutputDrivers=" << task_->numOutputDrivers()
              << " numTotalDrivers=" << task_->numTotalDrivers()
              << " queueStats=" << describeUcxOutputQueue(task_);
    if (parallelResultQueue_ != nullptr) {
      parallelResultQueue_->setNumProducers(task_->numOutputDrivers());
    }
  } catch (...) {
    if (parallelResultQueue_ != nullptr) {
      parallelResultQueue_->close();
    }
    throw;
  }
}

void WholeStageResultIterator::checkTaskError() {
  if (task_ == nullptr) {
    return;
  }
  auto error = task_->error();
  if (error == nullptr) {
    return;
  }
  if (parallelResultQueue_ != nullptr) {
    parallelResultQueue_->close();
  }
  std::rethrow_exception(error);
}

std::shared_ptr<ColumnarBatch> WholeStageResultIterator::toColumnarBatch(
    const velox::RowVectorPtr& vector) {
  if (vector == nullptr || vector->size() == 0) {
    return nullptr;
  }

#ifdef GLUTEN_ENABLE_GPU
  if (auto cudfVec = std::dynamic_pointer_cast<
          velox::cudf_velox::CudfVector>(vector)) {
    auto numCols = cudfVec->getTableView().num_columns();
    return std::make_shared<VeloxColumnarBatch>(vector, numCols);
  }
#endif

  {
    ScopedTimer timer(&loadLazyVectorTime_);
    for (auto& child : vector->children()) {
      child->loadedVector();
    }
  }

  return std::make_shared<VeloxColumnarBatch>(vector);
}

std::shared_ptr<ColumnarBatch> WholeStageResultIterator::next() {
  return requiresParallelExecution_ ? nextParallel() : nextSerial();
}

std::shared_ptr<ColumnarBatch> WholeStageResultIterator::nextSerial() {
  auto nextStart = std::chrono::steady_clock::now();
  ++nextCallCount_;

  if (task_->isFinished()) {
    auto nextEnd = std::chrono::steady_clock::now();
    totalNextNanos_ += std::chrono::duration_cast<
        std::chrono::nanoseconds>(nextEnd - nextStart).count();
    return nullptr;
  }
  velox::RowVectorPtr vector;

  // GPU locking is NOT applied at this level. The pipeline (task_->next())
  // contains both GPU work (cuDF operators, D2H) and CPU work (Parquet I/O,
  // operator scheduling). Serializing the entire pipeline with maxConcurrent=1
  // blocks multi-task CPU parallelism and causes 12-21% regression.
  //
  // Instead, each GPU-touching component manages its own lock:
  //   - CudfHiveDataSource: GpuGuard around scan/H2D
  //   - VeloxGpuColumnarBatchSerializer: GpuLockGuard around deserialize H2D
  //   - GpuBufferBatchResizer / CudfVectorStream: GpuLockGuard around H2D
  // cuDF operators are stream-safe (per-op streams) and RMM is thread-safe.

  while (true) {
    auto future = velox::ContinueFuture::makeEmpty();
    auto veloxStart = std::chrono::steady_clock::now();
    auto out = task_->next(&future);
    auto veloxEnd = std::chrono::steady_clock::now();
    totalVeloxNextNanos_ += std::chrono::duration_cast<
        std::chrono::nanoseconds>(veloxEnd - veloxStart).count();
    if (!future.valid()) {
      vector = std::move(out);
      break;
    }
    GLUTEN_CHECK(
        out == nullptr,
        "Expected to wait but still got non-null output");
    VLOG(2) << "Velox task " << task_->taskId()
            << " is busy when ::next() is called. "
            << "Will wait and try again. Task state: "
            << taskStateString(task_->state());
    // GPU thread-region tracking dropped in IBM-baseline switch
    // (endGpuRegion removed). cuDF operators are stream-safe and RMM is
    // thread-safe, so blocking on the future without releasing a permit
    // does not cause deadlock in this configuration.
    future.wait();
  }

  auto recordAndReturn =
      [&](std::shared_ptr<ColumnarBatch> result)
      -> std::shared_ptr<ColumnarBatch> {
    auto nextEnd = std::chrono::steady_clock::now();
    totalNextNanos_ += std::chrono::duration_cast<
        std::chrono::nanoseconds>(nextEnd - nextStart).count();
    return result;
  };

  if (vector == nullptr) {
    return recordAndReturn(nullptr);
  }
  return recordAndReturn(toColumnarBatch(vector));
}

std::shared_ptr<ColumnarBatch> WholeStageResultIterator::nextParallel() {
  auto nextStart = std::chrono::steady_clock::now();
  ++nextCallCount_;

  auto recordAndReturn =
      [&](std::shared_ptr<ColumnarBatch> result)
      -> std::shared_ptr<ColumnarBatch> {
    auto nextEnd = std::chrono::steady_clock::now();
    totalNextNanos_ += std::chrono::duration_cast<
        std::chrono::nanoseconds>(nextEnd - nextStart).count();
    return result;
  };

  if (parallelTaskFinished_) {
    return recordAndReturn(nullptr);
  }

  startParallelTaskIfNeeded();
  checkTaskError();

  if (!parallelTaskProducesOutput_) {
    auto veloxStart = std::chrono::steady_clock::now();
    auto future = task_->taskCompletionFuture();
    if (future.valid()) {
      int64_t waitedMs = 0;
      int64_t nextLogMs = 5000;
      while (!future.isReady()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        waitedMs += 10;
        if (ucxOutputQueueNoMoreData(task_)) {
          LOG(WARNING) << "[WS-ITER] detaching root partitioned output task "
                       << "after UCX noMoreData taskId=" << task_->taskId()
                       << " waitedMs=" << waitedMs
                       << " state=" << static_cast<int>(task_->state())
                       << " running=" << task_->isRunning()
                       << " numFinishedDrivers="
                       << task_->numFinishedDrivers()
                       << "/" << task_->numTotalDrivers()
                       << " queueStats=" << describeUcxOutputQueue(task_);
          detachUcxPartitionedOutputTask("ucx output noMoreData");
          parallelTaskFinished_ = true;
          checkTaskError();
          return recordAndReturn(nullptr);
        }
        if (waitedMs >= nextLogMs) {
          LOG(WARNING) << "[WS-ITER] waiting for root partitioned output task"
                       << " taskId=" << task_->taskId()
                       << " waitedMs=" << waitedMs
                       << " state=" << static_cast<int>(task_->state())
                       << " running=" << task_->isRunning()
                       << " numFinishedDrivers="
                       << task_->numFinishedDrivers()
                       << "/" << task_->numTotalDrivers()
                       << " queueStats=" << describeUcxOutputQueue(task_);
          nextLogMs += 5000;
        }
        checkTaskError();
      }
      future.wait();
    }
    auto veloxEnd = std::chrono::steady_clock::now();
    totalVeloxNextNanos_ += std::chrono::duration_cast<
        std::chrono::nanoseconds>(veloxEnd - veloxStart).count();
    parallelTaskFinished_ = true;
    checkTaskError();
    return recordAndReturn(nullptr);
  }

  auto veloxStart = std::chrono::steady_clock::now();
  auto vector = parallelResultQueue_->dequeue();
  auto veloxEnd = std::chrono::steady_clock::now();
  totalVeloxNextNanos_ += std::chrono::duration_cast<
      std::chrono::nanoseconds>(veloxEnd - veloxStart).count();
  checkTaskError();

  if (vector == nullptr) {
    auto future = task_->taskCompletionFuture();
    if (future.valid()) {
      future.wait();
    }
    parallelTaskFinished_ = true;
    checkTaskError();
    return recordAndReturn(nullptr);
  }
  return recordAndReturn(toColumnarBatch(vector));
}

int64_t WholeStageResultIterator::spillFixedSize(int64_t size) {
  auto pool = memoryManager_->getAggregateMemoryPool();
  std::string poolName{pool->root()->name() + "/" + pool->name()};
  std::string logPrefix{"Spill[" + poolName + "]: "};
  int64_t shrunken = memoryManager_->shrink(size);
  if (spillStrategy_ == "auto") {
    int64_t remaining = size - shrunken;
    LOG(INFO) << fmt::format("{} trying to request spill for {}.", logPrefix, velox::succinctBytes(remaining));
    auto mm = memoryManager_->getMemoryManager();
    uint64_t spilledOut = mm->arbitrator()->shrinkCapacity(remaining); // this conducts spill
    uint64_t total = shrunken + spilledOut;
    LOG(INFO) << fmt::format(
        "{} successfully reclaimed total {} with shrunken {} and spilled {}.",
        logPrefix,
        velox::succinctBytes(total),
        velox::succinctBytes(shrunken),
        velox::succinctBytes(spilledOut));
    return total;
  }
  LOG(WARNING) << "Spill-to-disk was disabled since " << kSpillStrategy << " was not configured.";
  VLOG(2) << logPrefix << "Successfully reclaimed total " << shrunken << " bytes.";
  return shrunken;
}

void WholeStageResultIterator::getOrderedNodeIds(
    const std::shared_ptr<const velox::core::PlanNode>& planNode,
    std::vector<velox::core::PlanNodeId>& nodeIds) {
  bool isProjectNode = (std::dynamic_pointer_cast<const velox::core::ProjectNode>(planNode) != nullptr);
  bool isLocalExchangeNode = (std::dynamic_pointer_cast<const velox::core::LocalPartitionNode>(planNode) != nullptr);
  bool isUnionNode = isLocalExchangeNode &&
      std::dynamic_pointer_cast<const velox::core::LocalPartitionNode>(planNode)->type() ==
          velox::core::LocalPartitionNode::Type::kGather;
  const auto& sourceNodes = planNode->sources();
  if (isProjectNode) {
    GLUTEN_CHECK(sourceNodes.size() == 1, "Illegal state");
    const auto sourceNode = sourceNodes.at(0);
    // Filter over Project are mapped into FilterProject operator in Velox.
    // Metrics are all applied on Project node, and the metrics for Filter node
    // do not exist.
    if (std::dynamic_pointer_cast<const velox::core::FilterNode>(sourceNode)) {
      omittedNodeIds_.insert(sourceNode->id());
    }
    getOrderedNodeIds(sourceNode, nodeIds);
    nodeIds.emplace_back(planNode->id());
    return;
  }

  if (isUnionNode) {
    // FIXME: The whole metrics system in gluten-substrait is magic. Passing metrics trees through JNI with a trivial
    //  array is possible but requires for a solid design. Apparently we haven't had it. All the code requires complete
    //  rework.
    // Union was interpreted as LocalPartition + LocalExchange + 2 fake projects as children in Velox. So we only fetch
    // metrics from the root node.
    std::vector<std::shared_ptr<const velox::core::PlanNode>> unionChildren{};
    for (const auto& source : planNode->sources()) {
      const auto projectedChild = std::dynamic_pointer_cast<const velox::core::ProjectNode>(source);
      GLUTEN_CHECK(projectedChild != nullptr, "Illegal state");
      const auto projectSources = projectedChild->sources();
      GLUTEN_CHECK(projectSources.size() == 1, "Illegal state");
      const auto projectSource = projectSources.at(0);
      getOrderedNodeIds(projectSource, nodeIds);
    }
    nodeIds.emplace_back(planNode->id());
    return;
  }

  for (const auto& sourceNode : sourceNodes) {
    // Post-order traversal.
    getOrderedNodeIds(sourceNode, nodeIds);
  }
  nodeIds.emplace_back(planNode->id());
}

void WholeStageResultIterator::constructPartitionColumns(
    std::unordered_map<std::string, std::optional<std::string>>& partitionKeys,
    const std::unordered_map<std::string, std::string>& map) {
  for (const auto& partitionColumn : map) {
    auto key = partitionColumn.first;
    const auto value = partitionColumn.second;
    if (!veloxCfg_->get<bool>(kCaseSensitive, false)) {
      folly::toLowerAscii(key);
    }
    if (value == kHiveDefaultPartition) {
      partitionKeys[key] = std::nullopt;
    } else {
      partitionKeys[key] = value;
    }
  }
}

void WholeStageResultIterator::addIteratorSplits(const std::vector<std::shared_ptr<ResultIterator>>& inputIterators) {
  GLUTEN_CHECK(!allSplitsAdded_, "Method addIteratorSplits should not be called since all splits has been added to the Velox task.");
  // Create IteratorConnectorSplit for each iterator
  for (size_t i = 0; i < streamIds_.size() && i < inputIterators.size(); ++i) {
    if (inputIterators[i] == nullptr) {
      continue;
    }
    auto connectorSplit = std::make_shared<IteratorConnectorSplit>(
        kIteratorConnectorId, inputIterators[i]);
    exec::Split split(folly::copy(connectorSplit), -1);
    task_->addSplit(streamIds_[i], std::move(split));
  }
}

void WholeStageResultIterator::addRemoteExchangeSplits(
    const velox::core::PlanNodeId& exchangeNodeId,
    const std::vector<std::string>& remoteTaskIds) {
  VELOX_CHECK_NOT_NULL(task_, "Cannot add UCX exchange splits before Velox task is created");
  for (const auto& remoteTaskId : remoteTaskIds) {
    exec::Split split(
        std::make_shared<velox::exec::RemoteConnectorSplit>(remoteTaskId),
        -1);
    task_->addSplit(exchangeNodeId, std::move(split));
  }
}

void WholeStageResultIterator::noMoreRemoteExchangeSplits(
    const velox::core::PlanNodeId& exchangeNodeId) {
  VELOX_CHECK_NOT_NULL(task_, "Cannot finish UCX exchange splits before Velox task is created");
  task_->noMoreSplits(exchangeNodeId);
}

void WholeStageResultIterator::noMoreSplits() {
  if (allSplitsAdded_) {
    return;
  }
  // Mark no more splits for all scan nodes
  for (int idx = 0; idx < scanNodeIds_.size(); idx++) {
    for (auto& split : splits_[idx]) {
      task_->addSplit(scanNodeIds_[idx], std::move(split));
    }
  }

  for (const auto& scanNodeId : scanNodeIds_) {
    task_->noMoreSplits(scanNodeId);
  }
  
  // Mark no more splits for all stream nodes
  for (const auto& streamId : streamIds_) {
    task_->noMoreSplits(streamId);
  }
  allSplitsAdded_ = true;
}

void WholeStageResultIterator::collectMetrics() {
  if (metrics_) {
    return;
  }

  LOG(WARNING) << "collectMetrics() called, task state="
               << static_cast<int>(task_->state());

  LOG(WARNING) << "[TIMING] " << taskInfo_
               << " totalNextNanos=" << totalNextNanos_
               << " veloxNextNanos=" << totalVeloxNextNanos_
               << " wrapperNanos="
               << (totalNextNanos_ - totalVeloxNextNanos_)
               << " nextCalls=" << nextCallCount_;

  const auto& taskStats = task_->taskStats();
  if (taskStats.executionStartTimeMs == 0) {
    LOG(INFO) << "Skip collect task metrics since "
              << "task did not call next().";
    return;
  }

  // Save and print the plan with stats if debug mode is enabled or showTaskMetricsWhenFinished is true.
  if (veloxCfg_->get<bool>(kDebugModeEnabled, false) ||
      veloxCfg_->get<bool>(kShowTaskMetricsWhenFinished, kShowTaskMetricsWhenFinishedDefault)) {
    auto planWithStats = velox::exec::printPlanWithStats(*veloxPlan_.get(), taskStats, true);
    std::ostringstream oss;
    oss << "Native Plan with stats for: " << taskInfo_ << "\n";
    oss << "TaskStats: totalTime: " << taskStats.executionEndTimeMs - taskStats.executionStartTimeMs
        << "; startTime: " << taskStats.executionStartTimeMs << "; endTime: " << taskStats.executionEndTimeMs;
    oss << "\n" << planWithStats << std::endl;
    LOG(WARNING) << oss.str();
  }

  auto planStats = velox::exec::toPlanStats(taskStats);
  // Calculate the total number of metrics.
  int statsNum = 0;
  for (int idx = 0; idx < orderedNodeIds_.size(); idx++) {
    const auto& nodeId = orderedNodeIds_[idx];
    if (planStats.find(nodeId) == planStats.end()) {
      if (omittedNodeIds_.find(nodeId) == omittedNodeIds_.end()) {
        LOG(WARNING) << "Not found node id: " << nodeId;
        LOG(WARNING) << "Plan Node: " << std::endl << veloxPlan_->toString(true, true);
        throw std::runtime_error("Node id cannot be found in plan status.");
      }
      // Special handing for Filter over Project case. Filter metrics are
      // omitted.
      statsNum += 1;
      continue;
    }
    statsNum += planStats.at(nodeId).operatorStats.size();
  }

  metrics_ = std::make_unique<Metrics>(statsNum);

  int metricIndex = 0;
  for (int idx = 0; idx < orderedNodeIds_.size(); idx++) {
    metrics_->get(Metrics::kLoadLazyVectorTime)[metricIndex] = 0;

    const auto& nodeId = orderedNodeIds_[idx];
    if (planStats.find(nodeId) == planStats.end()) {
      // Special handing for Filter over Project case. Filter metrics are
      // omitted.
      metrics_->get(Metrics::kOutputRows)[metricIndex] = 0;
      metrics_->get(Metrics::kOutputVectors)[metricIndex] = 0;
      metrics_->get(Metrics::kOutputBytes)[metricIndex] = 0;
      metrics_->get(Metrics::kCpuCount)[metricIndex] = 0;
      metrics_->get(Metrics::kWallNanos)[metricIndex] = 0;
      metrics_->get(Metrics::kPeakMemoryBytes)[metricIndex] = 0;
      metrics_->get(Metrics::kNumMemoryAllocations)[metricIndex] = 0;
      metricIndex += 1;
      continue;
    }

    const auto& stats = planStats.at(nodeId);
    // Add each operator stats into metrics.
    for (const auto& entry : stats.operatorStats) {
      const auto& second = entry.second;
      metrics_->get(Metrics::kInputRows)[metricIndex] = second->inputRows;
      metrics_->get(Metrics::kInputVectors)[metricIndex] = second->inputVectors;
      metrics_->get(Metrics::kInputBytes)[metricIndex] = second->inputBytes;
      metrics_->get(Metrics::kRawInputRows)[metricIndex] = second->rawInputRows;
      metrics_->get(Metrics::kRawInputBytes)[metricIndex] = second->rawInputBytes;
      metrics_->get(Metrics::kOutputRows)[metricIndex] = second->outputRows;
      metrics_->get(Metrics::kOutputVectors)[metricIndex] = second->outputVectors;
      metrics_->get(Metrics::kOutputBytes)[metricIndex] = second->outputBytes;
      metrics_->get(Metrics::kCpuCount)[metricIndex] = second->cpuWallTiming.count;
      metrics_->get(Metrics::kWallNanos)[metricIndex] = second->cpuWallTiming.wallNanos;
      metrics_->get(Metrics::kPeakMemoryBytes)[metricIndex] = second->peakMemoryBytes;
      metrics_->get(Metrics::kNumMemoryAllocations)[metricIndex] = second->numMemoryAllocations;
      metrics_->get(Metrics::kSpilledInputBytes)[metricIndex] = second->spilledInputBytes;
      metrics_->get(Metrics::kSpilledBytes)[metricIndex] = second->spilledBytes;
      metrics_->get(Metrics::kSpilledRows)[metricIndex] = second->spilledRows;
      metrics_->get(Metrics::kSpilledPartitions)[metricIndex] = second->spilledPartitions;
      metrics_->get(Metrics::kSpilledFiles)[metricIndex] = second->spilledFiles;
      metrics_->get(Metrics::kNumDynamicFiltersProduced)[metricIndex] =
          runtimeMetric("sum", second->customStats, kDynamicFiltersProduced);
      metrics_->get(Metrics::kNumDynamicFiltersAccepted)[metricIndex] =
          runtimeMetric("sum", second->customStats, kDynamicFiltersAccepted);
      metrics_->get(Metrics::kNumReplacedWithDynamicFilterRows)[metricIndex] =
          runtimeMetric("sum", second->customStats, kReplacedWithDynamicFilterRows);
      metrics_->get(Metrics::kFlushRowCount)[metricIndex] = runtimeMetric("sum", second->customStats, kFlushRowCount);
      metrics_->get(Metrics::kLoadedToValueHook)[metricIndex] =
          runtimeMetric("sum", second->customStats, kLoadedToValueHook);
      metrics_->get(Metrics::kBloomFilterBlocksByteSize)[metricIndex] =
          runtimeMetric("sum", second->customStats, kBloomFilterBlocksByteSize);
      metrics_->get(Metrics::kScanTime)[metricIndex] = runtimeMetric("sum", second->customStats, kTotalScanTime);
      metrics_->get(Metrics::kSkippedSplits)[metricIndex] = runtimeMetric("sum", second->customStats, kSkippedSplits);
      metrics_->get(Metrics::kProcessedSplits)[metricIndex] =
          runtimeMetric("sum", second->customStats, kProcessedSplits);
      metrics_->get(Metrics::kSkippedStrides)[metricIndex] = runtimeMetric("sum", second->customStats, kSkippedStrides);
      metrics_->get(Metrics::kProcessedStrides)[metricIndex] =
          runtimeMetric("sum", second->customStats, kProcessedStrides);
      metrics_->get(Metrics::kRemainingFilterTime)[metricIndex] =
          runtimeMetric("sum", second->customStats, kRemainingFilterTime);
      metrics_->get(Metrics::kIoWaitTime)[metricIndex] = runtimeMetric("sum", second->customStats, kIoWaitTime);
      metrics_->get(Metrics::kStorageReadBytes)[metricIndex] =
          runtimeMetric("sum", second->customStats, kStorageReadBytes);
      metrics_->get(Metrics::kLocalReadBytes)[metricIndex] = runtimeMetric("sum", second->customStats, kLocalReadBytes);
      metrics_->get(Metrics::kRamReadBytes)[metricIndex] = runtimeMetric("sum", second->customStats, kRamReadBytes);
      metrics_->get(Metrics::kPreloadSplits)[metricIndex] =
          runtimeMetric("sum", entry.second->customStats, kPreloadSplits);
      metrics_->get(Metrics::kPageLoadTime)[metricIndex] = runtimeMetric("sum", second->customStats, kPageLoadTime);
      metrics_->get(Metrics::kDataSourceAddSplitWallNanos)[metricIndex] =
          runtimeMetric("sum", second->customStats, kDataSourceAddSplitWallNanos) +
          runtimeMetric("sum", second->customStats, kWaitForPreloadSplitNanos);
      metrics_->get(Metrics::kDataSourceReadWallNanos)[metricIndex] =
          runtimeMetric("sum", second->customStats, kDataSourceReadWallNanos);
      metrics_->get(Metrics::kNumWrittenFiles)[metricIndex] =
          runtimeMetric("sum", entry.second->customStats, kNumWrittenFiles);
      metrics_->get(Metrics::kPhysicalWrittenBytes)[metricIndex] = second->physicalWrittenBytes;
      metrics_->get(Metrics::kWriteIOTime)[metricIndex] = runtimeMetric("sum", second->customStats, kWriteIOTime);
      metrics_->get(Metrics::kNumCoalescedBatches)[metricIndex] =
          runtimeMetric("sum", second->customStats, kNumCoalescedBatches);
      metrics_->get(Metrics::kPinnedAllocBytes)[metricIndex] =
          runtimeMetric("sum", second->customStats, kPinnedAllocBytes);
      metrics_->get(Metrics::kPageableAllocBytes)[metricIndex] =
          runtimeMetric("sum", second->customStats, kPageableAllocBytes);
      auto gpuVal =
          runtimeMetric("sum", second->customStats, kGpuComputeNanos);
      metrics_->get(Metrics::kGpuComputeTime)[metricIndex] = gpuVal;
      if (gpuVal > 0 || second->customStats.count(kGpuComputeNanos)) {
        LOG(WARNING) << "collectMetrics opType="
                     << entry.first
                     << " gpuComputeNanos=" << gpuVal
                     << " present="
                     << second->customStats.count(kGpuComputeNanos);
      }

      metricIndex += 1;
    }
  }

  // Put the loadLazyVector time into the metrics of the last operator.
  metrics_->get(Metrics::kLoadLazyVectorTime)[orderedNodeIds_.size() - 1] = loadLazyVectorTime_;

  // Populate the metrics with task stats for long running tasks.
  if (const int64_t collectTaskStatsThreshold =
          veloxCfg_->get<int64_t>(kTaskMetricsToEventLogThreshold, kTaskMetricsToEventLogThresholdDefault);
      collectTaskStatsThreshold >= 0 &&
      static_cast<int64_t>(taskStats.terminationTimeMs - taskStats.executionStartTimeMs) >
          collectTaskStatsThreshold * 1'000) {
    auto jsonStats = velox::exec::toPlanStatsJson(taskStats);
    metrics_->stats = folly::toJson(jsonStats);
  }
}

int64_t WholeStageResultIterator::runtimeMetric(
    const std::string& type,
    const std::unordered_map<std::string, velox::RuntimeMetric>& runtimeStats,
    const std::string& metricId) {
  if (runtimeStats.find(metricId) == runtimeStats.end()) {
    return 0;
  }

  if (type == "sum") {
    return runtimeStats.at(metricId).sum;
  } else if (type == "count") {
    return runtimeStats.at(metricId).count;
  } else if (type == "min") {
    return runtimeStats.at(metricId).min;
  } else if (type == "max") {
    return runtimeStats.at(metricId).max;
  } else {
    return 0;
  }
}

std::unordered_map<std::string, std::string> WholeStageResultIterator::getQueryContextConf() {
  std::unordered_map<std::string, std::string> configs = {};
  // Find batch size from Spark confs. If found, set the preferred and max batch size.
  configs[velox::core::QueryConfig::kPreferredOutputBatchRows] =
      std::to_string(veloxCfg_->get<uint32_t>(kSparkBatchSize, 4096));
  configs[velox::core::QueryConfig::kMaxOutputBatchRows] =
      std::to_string(veloxCfg_->get<uint32_t>(kSparkBatchSize, 4096));
  configs[velox::core::QueryConfig::kPreferredOutputBatchBytes] =
      std::to_string(veloxCfg_->get<uint64_t>(kVeloxPreferredBatchBytes, 10L << 20));
  try {
    configs[velox::core::QueryConfig::kSparkAnsiEnabled] = veloxCfg_->get<std::string>(kAnsiEnabled, "false");
    configs[velox::core::QueryConfig::kSessionTimezone] = veloxCfg_->get<std::string>(kSessionTimezone, "");
    // Adjust timestamp according to the above configured session timezone.
    configs[velox::core::QueryConfig::kAdjustTimestampToTimezone] = "true";

    {
      // Find offheap size from Spark confs. If found, set the max memory usage of partial aggregation.
      // Partial aggregation memory configurations.
      // TODO: Move the calculations to Java side.
      auto offHeapMemory = veloxCfg_->get<int64_t>(kSparkTaskOffHeapMemory, facebook::velox::memory::kMaxMemory);
      auto maxPartialAggregationMemory = std::max<int64_t>(
          1 << 24,
          veloxCfg_->get<int64_t>(kMaxPartialAggregationMemory).has_value()
              ? veloxCfg_->get<int64_t>(kMaxPartialAggregationMemory).value()
              : static_cast<int64_t>(veloxCfg_->get<double>(kMaxPartialAggregationMemoryRatio, 0.1) * offHeapMemory));
      auto maxExtendedPartialAggregationMemory = std::max<int64_t>(
          1 << 26,
          veloxCfg_->get<int64_t>(kMaxExtendedPartialAggregationMemory).has_value()
              ? veloxCfg_->get<int64_t>(kMaxExtendedPartialAggregationMemory).value()
              : static_cast<int64_t>(
                    veloxCfg_->get<double>(kMaxExtendedPartialAggregationMemoryRatio, 0.15) * offHeapMemory));
      configs[velox::core::QueryConfig::kMaxPartialAggregationMemory] = std::to_string(maxPartialAggregationMemory);
      configs[velox::core::QueryConfig::kMaxExtendedPartialAggregationMemory] =
          std::to_string(maxExtendedPartialAggregationMemory);
      configs[velox::core::QueryConfig::kAbandonPartialAggregationMinPct] =
          std::to_string(veloxCfg_->get<int32_t>(kAbandonPartialAggregationMinPct, 90));
      configs[velox::core::QueryConfig::kAbandonPartialAggregationMinRows] =
          std::to_string(veloxCfg_->get<int32_t>(kAbandonPartialAggregationMinRows, 100000));
    }
    // Spill configs
    if (spillStrategy_ == "none") {
      configs[velox::core::QueryConfig::kSpillEnabled] = "false";
    } else {
      configs[velox::core::QueryConfig::kSpillEnabled] = "true";
    }
    configs[velox::core::QueryConfig::kAggregationSpillEnabled] =
        std::to_string(veloxCfg_->get<bool>(kAggregationSpillEnabled, true));
    configs[velox::core::QueryConfig::kJoinSpillEnabled] =
        std::to_string(veloxCfg_->get<bool>(kJoinSpillEnabled, true));
    configs[velox::core::QueryConfig::kOrderBySpillEnabled] =
        std::to_string(veloxCfg_->get<bool>(kOrderBySpillEnabled, true));
    configs[velox::core::QueryConfig::kWindowSpillEnabled] =
        std::to_string(veloxCfg_->get<bool>(kWindowSpillEnabled, true));
    configs[velox::core::QueryConfig::kMaxSpillLevel] = std::to_string(veloxCfg_->get<int32_t>(kMaxSpillLevel, 4));
    configs[velox::core::QueryConfig::kMaxSpillFileSize] =
        std::to_string(veloxCfg_->get<uint64_t>(kMaxSpillFileSize, 1L * 1024 * 1024 * 1024));
    configs[velox::core::QueryConfig::kMaxSpillRunRows] =
        std::to_string(veloxCfg_->get<uint64_t>(kMaxSpillRunRows, 3L * 1024 * 1024));
    configs[velox::core::QueryConfig::kMaxSpillBytes] =
        std::to_string(veloxCfg_->get<uint64_t>(kMaxSpillBytes, 107374182400LL));
    configs[velox::core::QueryConfig::kSpillWriteBufferSize] =
        std::to_string(veloxCfg_->get<uint64_t>(kShuffleSpillDiskWriteBufferSize, 1L * 1024 * 1024));
    configs[velox::core::QueryConfig::kSpillReadBufferSize] =
        std::to_string(veloxCfg_->get<int32_t>(kSpillReadBufferSize, 1L * 1024 * 1024));
    configs[velox::core::QueryConfig::kSpillStartPartitionBit] =
        std::to_string(veloxCfg_->get<uint8_t>(kSpillStartPartitionBit, 48));
    configs[velox::core::QueryConfig::kSpillNumPartitionBits] =
        std::to_string(veloxCfg_->get<uint8_t>(kSpillPartitionBits, 3));
    configs[velox::core::QueryConfig::kSpillableReservationGrowthPct] =
        std::to_string(veloxCfg_->get<uint8_t>(kSpillableReservationGrowthPct, 25));
    configs[velox::core::QueryConfig::kSpillPrefixSortEnabled] =
        veloxCfg_->get<std::string>(kSpillPrefixSortEnabled, "false");
    if (veloxCfg_->get<bool>(kSparkShuffleSpillCompress, true)) {
      configs[velox::core::QueryConfig::kSpillCompressionKind] =
          veloxCfg_->get<std::string>(kSpillCompressionKind, veloxCfg_->get<std::string>(kCompressionKind, "lz4"));
    } else {
      configs[velox::core::QueryConfig::kSpillCompressionKind] = "none";
    }
    configs[velox::core::QueryConfig::kSparkBloomFilterExpectedNumItems] =
        std::to_string(veloxCfg_->get<int64_t>(kBloomFilterExpectedNumItems, 1000000));
    configs[velox::core::QueryConfig::kSparkBloomFilterNumBits] =
        std::to_string(veloxCfg_->get<int64_t>(kBloomFilterNumBits, 8388608));
    configs[velox::core::QueryConfig::kSparkBloomFilterMaxNumBits] =
        std::to_string(veloxCfg_->get<int64_t>(kBloomFilterMaxNumBits, 4194304));

    configs[velox::core::QueryConfig::kHashProbeDynamicFilterPushdownEnabled] =
        std::to_string(veloxCfg_->get<bool>(kHashProbeDynamicFilterPushdownEnabled, true));
    configs[velox::core::QueryConfig::kHashProbeBloomFilterPushdownMaxSize] =
        std::to_string(veloxCfg_->get<uint64_t>(kHashProbeBloomFilterPushdownMaxSize, 0));
    // spark.gluten.sql.columnar.backend.velox.SplitPreloadPerDriver takes no effect if
    // spark.gluten.sql.columnar.backend.velox.IOThreads is set to 0
    configs[velox::core::QueryConfig::kMaxSplitPreloadPerDriver] =
        std::to_string(veloxCfg_->get<int32_t>(kVeloxSplitPreloadPerDriver, 2));

    // hashtable build optimizations
    configs[velox::core::QueryConfig::kAbandonDedupHashMapMinRows] =
        std::to_string(veloxCfg_->get<int32_t>(kAbandonDedupHashMapMinRows, 100000));
    configs[velox::core::QueryConfig::kAbandonDedupHashMapMinPct] =
        std::to_string(veloxCfg_->get<int32_t>(kAbandonDedupHashMapMinPct, 0));

    // Disable driver cpu time slicing.
    configs[velox::core::QueryConfig::kDriverCpuTimeSliceLimitMs] = "0";

    configs[velox::core::QueryConfig::kSparkPartitionId] = std::to_string(taskInfo_.partitionId);

    // Enable Spark legacy date formatter if spark.sql.legacy.timeParserPolicy is set to 'LEGACY'
    // or 'legacy'
    if (veloxCfg_->get<std::string>(kSparkLegacyTimeParserPolicy, "") == "LEGACY") {
      configs[velox::core::QueryConfig::kSparkLegacyDateFormatter] = "true";
    } else {
      configs[velox::core::QueryConfig::kSparkLegacyDateFormatter] = "false";
    }

    if (veloxCfg_->get<std::string>(kSparkMapKeyDedupPolicy, "") == "EXCEPTION") {
      configs[velox::core::QueryConfig::kThrowExceptionOnDuplicateMapKeys] = "true";
    } else {
      configs[velox::core::QueryConfig::kThrowExceptionOnDuplicateMapKeys] = "false";
    }

    configs[velox::core::QueryConfig::kSparkLegacyStatisticalAggregate] =
        std::to_string(veloxCfg_->get<bool>(kSparkLegacyStatisticalAggregate, false));

    configs[velox::core::QueryConfig::kSparkJsonIgnoreNullFields] =
        std::to_string(veloxCfg_->get<bool>(kSparkJsonIgnoreNullFields, true));

    configs[velox::core::QueryConfig::kExprMaxCompiledRegexes] =
        std::to_string(veloxCfg_->get<int32_t>(kExprMaxCompiledRegexes, 100));

#ifdef GLUTEN_ENABLE_GPU
    configs[velox::cudf_velox::CudfConfig::kCudfEnabled] =
        std::to_string(veloxCfg_->get<bool>(kCudfEnabled, false));
    // IBM baseline removed kCudfSkipOutputToVelox. Output-to-Velox is
    // unconditional now; gluten consumers always materialize to RowVector.
#endif

    const auto setIfExists = [&](const std::string& glutenKey, const std::string& veloxKey) {
      const auto valueOptional = veloxCfg_->get<std::string>(glutenKey);
      if (valueOptional.has_value()) {
        configs[veloxKey] = valueOptional.value();
      }
    };
    setIfExists(kQueryTraceEnabled, velox::core::QueryConfig::kQueryTraceEnabled);
    setIfExists(kQueryTraceDir, velox::core::QueryConfig::kQueryTraceDir);
    setIfExists(kQueryTraceMaxBytes, velox::core::QueryConfig::kQueryTraceMaxBytes);
    setIfExists(kQueryTraceTaskRegExp, velox::core::QueryConfig::kQueryTraceTaskRegExp);
    setIfExists(kOpTraceDirectoryCreateConfig, velox::core::QueryConfig::kOpTraceDirectoryCreateConfig);

    overwriteVeloxConf(veloxCfg_.get(), configs, kDynamicBackendConfPrefix);
  } catch (const std::invalid_argument& err) {
    std::string errDetails = err.what();
    throw std::runtime_error("Invalid conf arg: " + errDetails);
  }
  return configs;
}

} // namespace gluten
