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

#include <jni.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <folly/dynamic.h>
#include <folly/json.h>

#include <jni/JniCommon.h>
#include <jni/JniError.h>

#include "compute/Runtime.h"
#include "memory/ColumnarBatch.h"
#include "memory/VeloxColumnarBatch.h"
#include "memory/VeloxMemoryManager.h"
#include "shuffle/Partitioning.h"
#include "utils/Exception.h"
#include "utils/ObjectStore.h"

#ifdef GLUTEN_ENABLE_GPU
#include "cudf/GpuLock.h"

#include "velox/core/PlanNode.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/vector/arrow/Bridge.h"

#include <cudf/binaryop.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>
#endif

using gluten::GlutenException;

namespace {

#ifndef GLUTEN_ENABLE_GPU
[[noreturn]] void failGpuRequired(const std::string& operation) {
  throw GlutenException(
      "Gluten UCX incremental shuffle native " + operation +
      " requires a GPU-enabled Velox backend.");
}
#else

using facebook::velox::ContinueFuture;
using facebook::velox::RowTypePtr;
using facebook::velox::asRowType;
using facebook::velox::cudf_velox::CudfVector;
using facebook::velox::memory::MemoryPool;
using facebook::velox::ucx_exchange::PackedTableWithStreamPtr;
using facebook::velox::ucx_exchange::UcxExchangeClient;
using facebook::velox::ucx_exchange::UcxOutputQueueManager;

struct RemoteEndpoint {
  std::string executorId;
  std::string host;
  int32_t ucxPort{-1};
  int32_t shuffleId{-1};
  int64_t mapId{-1};
  int64_t attemptId{-1};
  std::string nativeTaskId;
  int32_t deviceId{-1};
  int64_t epoch{0};
};

uint64_t outputBufferBytes() {
  if (const char* value = std::getenv("GLUTEN_UCX_SHUFFLE_OUTPUT_BUFFER_BYTES")) {
    try {
      const auto parsed = static_cast<int64_t>(std::stoll(value));
      if (parsed > 0) {
        return static_cast<uint64_t>(parsed);
      }
    } catch (...) {
    }
  }
  return 512ULL << 20;
}

int64_t targetRowsPerChunk() {
  if (const char* value = std::getenv("GLUTEN_UCX_PARTITIONED_OUTPUT_BATCH_ROWS")) {
    try {
      const auto parsed = static_cast<int64_t>(std::stoll(value));
      if (parsed > 0) {
        return parsed;
      }
    } catch (...) {
    }
  }
  return 1048576;
}

bool backpressureEnabled() {
  const char* value = std::getenv("GLUTEN_UCX_SHUFFLE_BACKPRESSURE_ENABLED");
  if (value == nullptr) {
    return true;
  }
  std::string normalized(value);
  std::transform(
      normalized.begin(),
      normalized.end(),
      normalized.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized != "0" && normalized != "false" &&
      normalized != "no" && normalized != "off";
}

int64_t writerDrainWaitMs() {
  if (const char* value = std::getenv("GLUTEN_UCX_SHUFFLE_WRITER_DRAIN_WAIT_MS")) {
    try {
      const auto parsed = static_cast<int64_t>(std::stoll(value));
      if (parsed >= -1) {
        return parsed;
      }
    } catch (...) {
    }
  }
  return 300000;
}

std::vector<RemoteEndpoint> parseEndpointsJson(const uint8_t* data, int32_t size) {
  if (data == nullptr || size <= 0) {
    return {};
  }
  std::string json(reinterpret_cast<const char*>(data), size);
  auto parsed = folly::parseJson(json);
  VELOX_CHECK(parsed.isObject(), "UCX shuffle endpoints payload must be a JSON object");
  VELOX_CHECK(parsed.count("endpoints") > 0, "UCX shuffle endpoints payload missing endpoints");
  const auto& endpoints = parsed["endpoints"];
  VELOX_CHECK(endpoints.isArray(), "UCX shuffle endpoints must be an array");

  std::vector<RemoteEndpoint> result;
  result.reserve(endpoints.size());
  for (const auto& item : endpoints) {
    RemoteEndpoint endpoint;
    endpoint.executorId = item["executorId"].asString();
    endpoint.host = item["host"].asString();
    endpoint.ucxPort = static_cast<int32_t>(item["ucxPort"].asInt());
    endpoint.shuffleId = static_cast<int32_t>(item["shuffleId"].asInt());
    endpoint.mapId = item["mapId"].asInt();
    endpoint.attemptId = item["attemptId"].asInt();
    endpoint.nativeTaskId = item["nativeTaskId"].asString();
    endpoint.deviceId = static_cast<int32_t>(item["deviceId"].asInt());
    endpoint.epoch = item["epoch"].asInt();
    result.push_back(std::move(endpoint));
  }
  return result;
}

std::string remoteTaskUrl(const RemoteEndpoint& endpoint, int32_t destination) {
  VELOX_CHECK_GT(
      endpoint.ucxPort,
      0,
      "UCX shuffle endpoint for task {} has invalid base port {}. Set spark.gluten.ucx.shuffle.port.",
      endpoint.nativeTaskId,
      endpoint.ucxPort);
  return fmt::format(
      "http://{}:{}/v1/task/{}/results/{}",
      endpoint.host,
      endpoint.ucxPort,
      endpoint.nativeTaskId,
      destination);
}

RowTypePtr importRowTypeFromArrowSchema(jlong cSchema) {
  auto* arrowSchema = reinterpret_cast<struct ArrowSchema*>(cSchema);
  VELOX_CHECK_NOT_NULL(arrowSchema, "UCX shuffle reader received null ArrowSchema");
  auto rowType = asRowType(facebook::velox::importFromArrow(*arrowSchema));
  VELOX_CHECK_NOT_NULL(rowType, "UCX shuffle reader output schema must be a RowType");
  return rowType;
}

MemoryPool* leafPool(gluten::Runtime* runtime) {
  auto* memoryManager =
      dynamic_cast<gluten::VeloxMemoryManager*>(runtime->memoryManager());
  VELOX_CHECK_NOT_NULL(
      memoryManager,
      "UCX shuffle requires VeloxMemoryManager, got a non-Velox memory manager");
  return memoryManager->getLeafMemoryPool().get();
}

std::vector<cudf::column_view> columnsFrom(
    cudf::table_view table,
    cudf::size_type firstColumn) {
  std::vector<cudf::column_view> columns;
  columns.reserve(table.num_columns() - firstColumn);
  for (cudf::size_type i = firstColumn; i < table.num_columns(); ++i) {
    columns.push_back(table.column(i));
  }
  return columns;
}

void waitForFuture(ContinueFuture future) {
  if (!future.isReady()) {
    std::move(future).wait();
  }
}

int64_t monotonicNanos() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

class NativeUcxShuffleWriter {
 public:
  NativeUcxShuffleWriter(
      MemoryPool* pool,
      std::string taskId,
      int32_t numPartitions,
      gluten::Partitioning partitioning,
      int32_t startPartitionId)
      : pool_(pool),
        taskId_(std::move(taskId)),
        numPartitions_(numPartitions),
        partitioning_(partitioning),
        startPartitionId_(startPartitionId),
        queueManager_(UcxOutputQueueManager::getInstanceRef()),
        outputBufferBytes_(outputBufferBytes()),
        targetRowsPerChunk_(targetRowsPerChunk()),
        backpressureEnabled_(backpressureEnabled()),
        writerDrainWaitMs_(writerDrainWaitMs()) {
    VELOX_CHECK_NOT_NULL(pool_, "UCX shuffle writer requires a Velox memory pool");
    VELOX_CHECK_GT(numPartitions_, 0, "UCX shuffle writer requires at least one partition");
    queueManager_->initializeStandaloneTask(
        taskId_,
        facebook::velox::core::PartitionedOutputNode::Kind::kPartitioned,
        numPartitions_,
        1,
        outputBufferBytes_);
    VLOG(1) << "Opened UCX incremental shuffle writer task=" << taskId_
            << " partitions=" << numPartitions_
            << " partitioning=" << partitioning_
            << " startPartitionId=" << startPartitionId_
            << " outputBufferBytes=" << outputBufferBytes_
            << " targetRowsPerChunk=" << targetRowsPerChunk_
            << " backpressureEnabled=" << backpressureEnabled_;
  }

  ~NativeUcxShuffleWriter() {
    if (!closed_) {
      close(false);
    }
  }

  void write(int32_t partitionId, const std::shared_ptr<gluten::ColumnarBatch>& batch) {
    VELOX_CHECK(!closed_, "Cannot write to closed UCX shuffle writer task {}", taskId_);
    VELOX_CHECK_NOT_NULL(batch, "UCX shuffle writer received null ColumnarBatch");
    if (batch->numRows() == 0 || batch->numColumns() == 0) {
      return;
    }
    ++batchesWritten_;
    rowsWritten_ += batch->numRows();

    gluten::GpuLockGuard gpuLock;
    auto veloxBatch = gluten::VeloxColumnarBatch::from(pool_, batch);
    auto rowVector = veloxBatch->getRowVector();
    auto cudfVector = std::dynamic_pointer_cast<CudfVector>(rowVector);

    auto writeTable =
        [&](cudf::table_view tableView, rmm::cuda_stream_view stream) {
          switch (partitioning_) {
            case gluten::Partitioning::kSingle:
              enqueueSingle(tableView, stream);
              break;
            case gluten::Partitioning::kHash:
            case gluten::Partitioning::kRange:
              partitionByFirstColumn(tableView, stream);
              break;
            case gluten::Partitioning::kRoundRobin:
              equalPartition(tableView, stream);
              break;
            case gluten::Partitioning::kRandom:
              enqueuePrePartitioned(partitionId, tableView, stream);
              break;
          }
        };

    if (cudfVector != nullptr) {
      writeTable(cudfVector->getTableView(), cudfVector->stream());
    } else {
      VELOX_CHECK_NOT_NULL(
          rowVector,
          "UCX shuffle writer failed to materialize a Velox RowVector from ColumnarBatch type {}",
          batch->getType());
      auto stream = facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
      auto convertedTable = facebook::velox::cudf_velox::with_arrow::toCudfTable(
          rowVector,
          pool_,
          stream,
          facebook::velox::cudf_velox::get_output_mr());
      VELOX_CHECK_NOT_NULL(
          convertedTable,
          "UCX shuffle writer failed to convert Velox RowVector input to a cuDF table");
      auto tableView = convertedTable->view();
      VLOG(2) << "UCX shuffle writer converted Velox RowVector input to cuDF table for task "
              << taskId_ << " rows=" << tableView.num_rows()
              << " columns=" << tableView.num_columns();
      writeTable(tableView, stream);
    }

    waitForBackpressureIfNeeded();
  }

  void close(bool success) {
    if (closed_) {
      return;
    }
    closed_ = true;
    if (success) {
      queueManager_->noMoreData(taskId_);
      scheduleSuccessfulCleanup();
    } else {
      queueManager_->removeTask(taskId_);
    }
    LOG(WARNING) << "[UCX_SHUFFLE_WRITER_STATS] task=" << taskId_
                 << " success=" << success
                 << " batches=" << batchesWritten_
                 << " rows=" << rowsWritten_
                 << " packedPayloads=" << packedColumnsEnqueued_
                 << " packedColumns=" << packedColumnsEnqueued_
                 << " bytes=" << bytesEnqueued_
                 << " backpressureWaitCount=" << backpressureWaitCount_
                 << " backpressureWaitMs=" << (backpressureWaitNanos_ / 1000000);
  }

 private:
  void enqueuePacked(
      int32_t destination,
      cudf::table_view table,
      rmm::cuda_stream_view stream) {
    if (table.num_rows() == 0) {
      return;
    }
    waitForBackpressureIfNeeded();
    auto packed = cudf::pack(
        table, stream, cudf::get_current_device_resource_ref());
    stream.synchronize();
    auto packedPtr = std::make_unique<cudf::packed_columns>(
        std::move(packed.metadata), std::move(packed.gpu_data));
    recordPackedColumns(packedPtr.get());
    queueManager_->enqueue(taskId_, destination, std::move(packedPtr), table.num_rows());
    waitForBackpressureIfNeeded();
  }

  void enqueueSingle(cudf::table_view table, rmm::cuda_stream_view stream) {
    if (table.num_rows() == 0) {
      return;
    }
    const auto rowsPerChunk = std::max<cudf::size_type>(
        1,
        targetRowsPerChunk_ > 0
            ? std::min<cudf::size_type>(
                  table.num_rows(),
                  static_cast<cudf::size_type>(targetRowsPerChunk_))
            : table.num_rows());
    for (cudf::size_type start = 0; start < table.num_rows(); start += rowsPerChunk) {
      const auto end = std::min<cudf::size_type>(table.num_rows(), start + rowsPerChunk);
      auto slices = cudf::slice(table, {start, end});
      VELOX_CHECK_EQ(slices.size(), 1);
      enqueuePacked(0, slices[0], stream);
    }
  }

  void enqueuePrePartitioned(
      int32_t partitionId,
      cudf::table_view table,
      rmm::cuda_stream_view stream) {
    VELOX_CHECK_GE(partitionId, 0);
    VELOX_CHECK_LT(partitionId, numPartitions_);
    enqueueRowsInChunks(partitionId, table, stream);
  }

  void partitionByFirstColumn(cudf::table_view table, rmm::cuda_stream_view stream) {
    VELOX_CHECK_GE(
        table.num_columns(),
        2,
        "Hash/range UCX shuffle input must include partition column plus data columns");
    const auto firstCol = table.column(0);
    auto dataColumns = columnsFrom(table, 1);
    cudf::table_view dataTable(dataColumns);

    std::unique_ptr<cudf::column> pidColumn;
    cudf::column_view pidView;
    if (partitioning_ == gluten::Partitioning::kHash) {
      auto numPartitionsScalar = cudf::numeric_scalar<int32_t>(
          numPartitions_, true, stream);
      pidColumn = cudf::binary_operation(
          firstCol,
          numPartitionsScalar,
          cudf::binary_operator::PYMOD,
          cudf::data_type{cudf::type_id::INT32},
          stream);
      pidView = pidColumn->view();
    } else {
      pidView = firstCol;
    }

    auto [partitioned, offsets] = cudf::partition(
        dataTable,
        pidView,
        static_cast<cudf::size_type>(numPartitions_),
        stream);
    VELOX_CHECK_EQ(offsets.size(), numPartitions_ + 1);

    std::vector<cudf::size_type> splitOffsets;
    splitOffsets.reserve(numPartitions_ > 0 ? numPartitions_ - 1 : 0);
    for (int32_t i = 1; i < numPartitions_; ++i) {
      splitOffsets.push_back(offsets[i]);
    }
    splitAndEnqueue(partitioned->view(), splitOffsets, stream);
  }

  void equalPartition(cudf::table_view table, rmm::cuda_stream_view stream) {
    std::vector<cudf::size_type> offsets;
    offsets.reserve(numPartitions_ > 0 ? numPartitions_ - 1 : 0);
    const auto rows = table.num_rows();
    for (int32_t i = 1; i < numPartitions_; ++i) {
      offsets.push_back(rows * i / numPartitions_);
    }
    splitAndEnqueue(table, offsets, stream);
  }

  void splitAndEnqueue(
      cudf::table_view table,
      const std::vector<cudf::size_type>& offsets,
      rmm::cuda_stream_view stream) {
    auto splits = cudf::contiguous_split(
        table, offsets, stream, cudf::get_current_device_resource_ref());
    stream.synchronize();
    VELOX_CHECK_EQ(splits.size(), numPartitions_);
    for (int32_t partition = 0; partition < numPartitions_; ++partition) {
      const auto rows = splits[partition].table.num_rows();
      if (rows == 0) {
        continue;
      }
      if (targetRowsPerChunk_ > 0 && rows > targetRowsPerChunk_) {
        VLOG(2) << "UCX shuffle writer chunking task=" << taskId_
                << " destination=" << partition << " rows=" << rows
                << " targetRowsPerChunk=" << targetRowsPerChunk_;
        enqueueRowsInChunks(partition, splits[partition].table, stream);
        continue;
      }
      auto packedPtr = std::make_unique<cudf::packed_columns>(
          std::move(splits[partition].data.metadata),
          std::move(splits[partition].data.gpu_data));
      recordPackedColumns(packedPtr.get());
      waitForBackpressureIfNeeded();
      queueManager_->enqueue(taskId_, partition, std::move(packedPtr), rows);
      waitForBackpressureIfNeeded();
    }
  }

  void enqueueRowsInChunks(
      int32_t destination,
      cudf::table_view table,
      rmm::cuda_stream_view stream) {
    if (table.num_rows() == 0) {
      return;
    }
    const auto rowsPerChunk = std::max<cudf::size_type>(
        1,
        targetRowsPerChunk_ > 0
            ? std::min<cudf::size_type>(
                  table.num_rows(),
                  static_cast<cudf::size_type>(targetRowsPerChunk_))
            : table.num_rows());
    for (cudf::size_type start = 0; start < table.num_rows(); start += rowsPerChunk) {
      const auto end = std::min<cudf::size_type>(table.num_rows(), start + rowsPerChunk);
      auto slices = cudf::slice(table, {start, end});
      VELOX_CHECK_EQ(slices.size(), 1);
      enqueuePacked(destination, slices[0], stream);
    }
  }

  void recordPackedColumns(const cudf::packed_columns* packed) {
    if (packed == nullptr || packed->gpu_data == nullptr) {
      return;
    }
    ++packedColumnsEnqueued_;
    bytesEnqueued_ += packed->gpu_data->size();
  }

  void waitForBackpressureIfNeeded() {
    if (!backpressureEnabled_) {
      return;
    }
    ContinueFuture future = ContinueFuture::makeEmpty();
    if (queueManager_->checkBlocked(taskId_, &future)) {
      const auto startNs = monotonicNanos();
      waitForFuture(std::move(future));
      const auto elapsedNs = monotonicNanos() - startNs;
      ++backpressureWaitCount_;
      backpressureWaitNanos_ += elapsedNs;
      if (elapsedNs >= 1000000000LL) {
        LOG(WARNING) << "[UCX_SHUFFLE_BACKPRESSURE_WAIT] task=" << taskId_
                     << " waitMs=" << (elapsedNs / 1000000)
                     << " totalWaitMs=" << (backpressureWaitNanos_ / 1000000)
                     << " count=" << backpressureWaitCount_;
      }
    }
  }

  void scheduleSuccessfulCleanup() {
    if (cleanupScheduled_ || writerDrainWaitMs_ < 0) {
      return;
    }
    cleanupScheduled_ = true;
    const auto taskId = taskId_;
    const auto queueManager = queueManager_;
    const auto deadlineMs = writerDrainWaitMs_;
    std::thread([taskId, queueManager, deadlineMs]() {
      const auto startNs = monotonicNanos();
      while (true) {
        try {
          if (queueManager->isFinished(taskId)) {
            queueManager->removeTask(taskId);
            const auto elapsedMs = (monotonicNanos() - startNs) / 1000000;
            LOG(WARNING) << "[UCX_SHUFFLE_WRITER_DRAINED] task=" << taskId
                         << " elapsedMs=" << elapsedMs;
            return;
          }
        } catch (const std::exception& e) {
          VLOG(1) << "[UCX_SHUFFLE_WRITER_DRAIN_CLEANUP_SKIP] task=" << taskId
                  << " error=" << e.what();
          return;
        } catch (...) {
          VLOG(1) << "[UCX_SHUFFLE_WRITER_DRAIN_CLEANUP_SKIP] task=" << taskId
                  << " unknown error";
          return;
        }
        const auto elapsedMs = (monotonicNanos() - startNs) / 1000000;
        if (deadlineMs == 0 || elapsedMs >= deadlineMs) {
          LOG(WARNING) << "[UCX_SHUFFLE_WRITER_DRAIN_TIMEOUT] task=" << taskId
                       << " elapsedMs=" << elapsedMs
                       << " timeoutMs=" << deadlineMs;
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }).detach();
  }

  MemoryPool* const pool_;
  const std::string taskId_;
  const int32_t numPartitions_;
  const gluten::Partitioning partitioning_;
  const int32_t startPartitionId_;
  const std::shared_ptr<UcxOutputQueueManager> queueManager_;
  const uint64_t outputBufferBytes_;
  const int64_t targetRowsPerChunk_;
  const bool backpressureEnabled_;
  const int64_t writerDrainWaitMs_;
  int64_t batchesWritten_{0};
  int64_t rowsWritten_{0};
  int64_t packedColumnsEnqueued_{0};
  int64_t bytesEnqueued_{0};
  int64_t backpressureWaitCount_{0};
  int64_t backpressureWaitNanos_{0};
  bool cleanupScheduled_{false};
  bool closed_{false};
};

class NativeUcxShuffleReader {
 public:
  NativeUcxShuffleReader(
      MemoryPool* pool,
      int32_t shuffleId,
      int32_t reducePartitionId,
      RowTypePtr outputType,
      std::vector<RemoteEndpoint> endpoints)
      : pool_(pool),
        shuffleId_(shuffleId),
        reducePartitionId_(reducePartitionId),
        outputType_(std::move(outputType)),
        client_(std::make_shared<UcxExchangeClient>(
            fmt::format("ucx-shuffle-{}-reduce-{}", shuffleId_, reducePartitionId_),
            reducePartitionId_,
            1)) {
    VELOX_CHECK_NOT_NULL(pool_, "UCX shuffle reader requires a Velox memory pool");
    VELOX_CHECK_NOT_NULL(outputType_, "UCX shuffle reader requires output type");
    const auto initialEndpointCount = endpoints.size();
    addEndpoints(std::move(endpoints));
    VLOG(1) << "Opened UCX incremental shuffle reader shuffleId=" << shuffleId_
            << " reducePartition=" << reducePartitionId_
            << " endpoints=" << initialEndpointCount;
  }

  ~NativeUcxShuffleReader() {
    close();
  }

  void addEndpoints(std::vector<RemoteEndpoint> endpoints) {
    if (closed_.load()) {
      return;
    }
    VELOX_CHECK(
        !noMoreEndpoints_.load(),
        "Cannot add UCX shuffle reader endpoints after noMoreEndpoints shuffleId={} reducePartition={}",
        shuffleId_,
        reducePartitionId_);
    for (const auto& endpoint : endpoints) {
      client_->addRemoteTaskId(remoteTaskUrl(endpoint, reducePartitionId_));
    }
    if (!endpoints.empty()) {
      VLOG(1) << "Added UCX incremental shuffle reader endpoints shuffleId=" << shuffleId_
              << " reducePartition=" << reducePartitionId_
              << " endpoints=" << endpoints.size();
    }
  }

  void noMoreEndpoints() {
    if (closed_.load()) {
      return;
    }
    if (!noMoreEndpoints_.exchange(true)) {
      client_->noMoreRemoteTasks();
      VLOG(1) << "UCX incremental shuffle reader noMoreEndpoints shuffleId=" << shuffleId_
              << " reducePartition=" << reducePartitionId_;
    }
  }

  std::shared_ptr<gluten::ColumnarBatch> next() {
    ++nextCalls_;
    while (!closed_.load()) {
      bool atEnd = false;
      ContinueFuture future = ContinueFuture::makeEmpty();
      PackedTableWithStreamPtr data = client_->next(0, &atEnd, &future);
      if (data != nullptr) {
        auto rows = data->packedTable->table.num_rows();
        ++batchesRead_;
        rowsRead_ += rows;
        auto vector = std::make_shared<CudfVector>(
            pool_,
            outputType_,
            rows,
            std::move(data->packedTable),
            data->stream);
        return std::make_shared<gluten::VeloxColumnarBatch>(
            vector, outputType_->size());
      }
      if (atEnd) {
        close();
        return nullptr;
      }
      const auto startNs = monotonicNanos();
      waitForFuture(std::move(future));
      const auto elapsedNs = monotonicNanos() - startNs;
      ++futureWaitCount_;
      futureWaitNanos_ += elapsedNs;
      if (elapsedNs >= 1000000000LL) {
        LOG(WARNING) << "[UCX_SHUFFLE_READER_WAIT] shuffleId=" << shuffleId_
                     << " reducePartition=" << reducePartitionId_
                     << " waitMs=" << (elapsedNs / 1000000)
                     << " totalWaitMs=" << (futureWaitNanos_ / 1000000)
                     << " count=" << futureWaitCount_;
      }
    }
    return nullptr;
  }

  void close() {
    if (!closed_.exchange(true)) {
      client_->close();
      LOG(WARNING) << "[UCX_SHUFFLE_READER_STATS] shuffleId=" << shuffleId_
                   << " reducePartition=" << reducePartitionId_
                   << " nextCalls=" << nextCalls_
                   << " batches=" << batchesRead_
                   << " rows=" << rowsRead_
                   << " futureWaitCount=" << futureWaitCount_
                   << " futureWaitMs=" << (futureWaitNanos_ / 1000000);
    }
  }

 private:
  MemoryPool* const pool_;
  const int32_t shuffleId_;
  const int32_t reducePartitionId_;
  const RowTypePtr outputType_;
  const std::shared_ptr<UcxExchangeClient> client_;
  int64_t nextCalls_{0};
  int64_t batchesRead_{0};
  int64_t rowsRead_{0};
  int64_t futureWaitCount_{0};
  int64_t futureWaitNanos_{0};
  std::atomic<bool> closed_{false};
  std::atomic<bool> noMoreEndpoints_{false};
};

#endif

} // namespace

extern "C" {

JNIEXPORT jint JNICALL
Java_org_apache_gluten_vectorized_UcxShuffleJniWrapper_nativeGetListenerPort( // NOLINT
    JNIEnv* env,
    jobject wrapper) {
  JNI_METHOD_START
#ifndef GLUTEN_ENABLE_GPU
  failGpuRequired("getListenerPort");
#else
  (void)env;
  (void)wrapper;
  auto comm = facebook::velox::ucx_exchange::Communicator::getInstance();
  VELOX_CHECK_NOT_NULL(comm, "UCX Communicator must be initialized before opening UCX shuffle");
  return static_cast<jint>(comm->getListenerPort());
#endif
  JNI_METHOD_END(0)
}

JNIEXPORT jlong JNICALL
Java_org_apache_gluten_vectorized_UcxShuffleJniWrapper_nativeOpenWriter( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jstring executorId,
    jstring host,
    jint ucxPort,
    jint shuffleId,
    jlong mapId,
    jlong attemptId,
    jstring nativeTaskId,
    jint deviceId,
    jlong epoch,
    jint numPartitions,
    jstring partitioningName,
    jint startPartitionId) {
  JNI_METHOD_START
#ifndef GLUTEN_ENABLE_GPU
  failGpuRequired("openWriter");
#else
  const auto ctx = gluten::getRuntime(env, wrapper);
  (void)executorId;
  (void)host;
  (void)ucxPort;
  (void)shuffleId;
  (void)mapId;
  (void)attemptId;
  (void)deviceId;
  (void)epoch;
  const auto taskId = jStringToCString(env, nativeTaskId);
  const auto partitioning = gluten::toPartitioning(jStringToCString(env, partitioningName));
  auto writer = std::make_shared<NativeUcxShuffleWriter>(
      leafPool(ctx),
      taskId,
      numPartitions,
      partitioning,
      startPartitionId);
  return ctx->saveObject(writer);
#endif
  JNI_METHOD_END(0)
}

JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_UcxShuffleJniWrapper_nativeWrite( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong writerHandle,
    jint partitionId,
    jlong columnarBatchHandle,
    jint numRows) {
  JNI_METHOD_START
#ifndef GLUTEN_ENABLE_GPU
  failGpuRequired("write");
#else
  (void)wrapper;
  (void)numRows;
  auto writer = gluten::ObjectStore::retrieve<NativeUcxShuffleWriter>(writerHandle);
  VELOX_CHECK_NOT_NULL(writer, "Invalid UCX shuffle writer handle {}", writerHandle);
  auto batch = gluten::ObjectStore::retrieve<gluten::ColumnarBatch>(columnarBatchHandle);
  VELOX_CHECK_NOT_NULL(batch, "Invalid UCX shuffle ColumnarBatch handle {}", columnarBatchHandle);
  writer->write(partitionId, batch);
#endif
  JNI_METHOD_END()
}

JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_UcxShuffleJniWrapper_nativeCloseWriter( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong writerHandle,
    jboolean success) {
  JNI_METHOD_START
#ifndef GLUTEN_ENABLE_GPU
  failGpuRequired("closeWriter");
#else
  (void)env;
  (void)wrapper;
  auto writer = gluten::ObjectStore::retrieve<NativeUcxShuffleWriter>(writerHandle);
  VELOX_CHECK_NOT_NULL(writer, "Invalid UCX shuffle writer handle {}", writerHandle);
  writer->close(success == JNI_TRUE);
  gluten::ObjectStore::release(writerHandle);
#endif
  JNI_METHOD_END()
}

JNIEXPORT jlong JNICALL
Java_org_apache_gluten_vectorized_UcxShuffleJniWrapper_nativeOpenReader( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jint shuffleId,
    jint reducePartitionId,
    jbyteArray endpointsJson,
    jlong cSchema) {
  JNI_METHOD_START
#ifndef GLUTEN_ENABLE_GPU
  failGpuRequired("openReader");
#else
  const auto ctx = gluten::getRuntime(env, wrapper);
  auto safeArray = gluten::getByteArrayElementsSafe(env, endpointsJson);
  auto endpoints = parseEndpointsJson(
      reinterpret_cast<const uint8_t*>(safeArray.elems()),
      safeArray.length());
  auto reader = std::make_shared<NativeUcxShuffleReader>(
      leafPool(ctx),
      shuffleId,
      reducePartitionId,
      importRowTypeFromArrowSchema(cSchema),
      std::move(endpoints));
  return ctx->saveObject(reader);
#endif
  JNI_METHOD_END(0)
}

JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_UcxShuffleJniWrapper_nativeAddReaderEndpoints( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong readerHandle,
    jbyteArray endpointsJson) {
  JNI_METHOD_START
#ifndef GLUTEN_ENABLE_GPU
  failGpuRequired("addReaderEndpoints");
#else
  (void)wrapper;
  auto safeArray = gluten::getByteArrayElementsSafe(env, endpointsJson);
  auto endpoints = parseEndpointsJson(
      reinterpret_cast<const uint8_t*>(safeArray.elems()),
      safeArray.length());
  auto reader = gluten::ObjectStore::retrieve<NativeUcxShuffleReader>(readerHandle);
  VELOX_CHECK_NOT_NULL(reader, "Invalid UCX shuffle reader handle {}", readerHandle);
  reader->addEndpoints(std::move(endpoints));
#endif
  JNI_METHOD_END()
}

JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_UcxShuffleJniWrapper_nativeNoMoreReaderEndpoints( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong readerHandle) {
  JNI_METHOD_START
#ifndef GLUTEN_ENABLE_GPU
  failGpuRequired("noMoreReaderEndpoints");
#else
  (void)env;
  (void)wrapper;
  auto reader = gluten::ObjectStore::retrieve<NativeUcxShuffleReader>(readerHandle);
  VELOX_CHECK_NOT_NULL(reader, "Invalid UCX shuffle reader handle {}", readerHandle);
  reader->noMoreEndpoints();
#endif
  JNI_METHOD_END()
}

JNIEXPORT jlong JNICALL
Java_org_apache_gluten_vectorized_UcxShuffleJniWrapper_nativeNextBatch( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong readerHandle) {
  JNI_METHOD_START
#ifndef GLUTEN_ENABLE_GPU
  failGpuRequired("nextBatch");
#else
  const auto ctx = gluten::getRuntime(env, wrapper);
  auto reader = gluten::ObjectStore::retrieve<NativeUcxShuffleReader>(readerHandle);
  VELOX_CHECK_NOT_NULL(reader, "Invalid UCX shuffle reader handle {}", readerHandle);
  auto batch = reader->next();
  if (batch == nullptr) {
    return 0;
  }
  return ctx->saveObject(batch);
#endif
  JNI_METHOD_END(0)
}

JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_UcxShuffleJniWrapper_nativeCloseReader( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong readerHandle) {
  JNI_METHOD_START
#ifndef GLUTEN_ENABLE_GPU
  failGpuRequired("closeReader");
#else
  (void)env;
  (void)wrapper;
  auto reader = gluten::ObjectStore::retrieve<NativeUcxShuffleReader>(readerHandle);
  if (reader != nullptr) {
    reader->close();
    gluten::ObjectStore::release(readerHandle);
  }
#endif
  JNI_METHOD_END()
}

} // extern "C"
