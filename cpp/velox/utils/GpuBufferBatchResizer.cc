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

#include "GpuBufferBatchResizer.h"
#include "cudf/GpuLock.h"
#include "memory/GpuBufferColumnarBatch.h"
#include "utils/Timer.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/PinnedHostMemory.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/vector/FlatVector.h"
#include <nvtx3/nvtx3.hpp>

#include <arrow/buffer.h>
#include <arrow/memory_pool.h>

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

using namespace facebook::velox;

namespace gluten {

namespace {

/// arrow::MemoryPool backed by PreferredPinnedPool (pinned pool → pageable
/// malloc).  Composed buffers allocated here can be DMA'd directly by
/// cudaMemcpyAsync when pinned; CUDA transparently handles pageable buffers
/// via internal staging.
///
/// Global singleton — Arrow buffers may be freed on a different thread.
/// The pinnedFlags_ map is mutex-protected for cross-thread safety.
class PinnedArrowMemoryPool : public arrow::MemoryPool {
 public:
  using Pool = facebook::velox::cudf_velox::PreferredPinnedPool;

  static PinnedArrowMemoryPool& instance() {
    static PinnedArrowMemoryPool pool;
    return pool;
  }

  arrow::Status Allocate(int64_t size, int64_t /*alignment*/, uint8_t** out) override {
    auto [ptr, pinned] = Pool::instance().allocate(static_cast<size_t>(size));
    *out = static_cast<uint8_t*>(ptr);
    {
      std::lock_guard<std::mutex> lock(mu_);
      pinnedFlags_[*out] = pinned;
    }
    bytesAllocated_.fetch_add(size, std::memory_order_relaxed);
    return arrow::Status::OK();
  }

  arrow::Status Reallocate(int64_t oldSize, int64_t newSize, int64_t alignment, uint8_t** ptr) override {
    uint8_t* newBuf = nullptr;
    ARROW_RETURN_NOT_OK(Allocate(newSize, alignment, &newBuf));
    if (oldSize > 0 && *ptr) {
      std::memcpy(newBuf, *ptr, std::min(oldSize, newSize));
      Free(*ptr, oldSize, alignment);
    }
    *ptr = newBuf;
    return arrow::Status::OK();
  }

  void Free(uint8_t* buffer, int64_t size, int64_t /*alignment*/) override {
    if (!buffer) return;
    bool pinned = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = pinnedFlags_.find(buffer);
      if (it != pinnedFlags_.end()) {
        pinned = it->second;
        pinnedFlags_.erase(it);
      }
    }
    Pool::instance().deallocate(buffer, static_cast<size_t>(size), pinned);
    bytesAllocated_.fetch_sub(size, std::memory_order_relaxed);
  }

  int64_t bytes_allocated() const override {
    return bytesAllocated_.load(std::memory_order_relaxed);
  }

  int64_t max_memory() const override { return -1; }
  int64_t total_bytes_allocated() const override { return -1; }
  int64_t num_allocations() const override { return -1; }
  std::string backend_name() const override { return "preferred_pinned"; }

 private:
  PinnedArrowMemoryPool() = default;
  std::atomic<int64_t> bytesAllocated_{0};
  std::mutex mu_;
  std::unordered_map<uint8_t*, bool> pinnedFlags_;
};

/// Count null values from a CPU-resident Arrow validity bitmask (bit SET =
/// valid).  Returns 0 when the mask is absent.  Avoids the implicit GPU sync
/// that cudf::null_count() would cause.
cudf::size_type cpuNullCount(const uint8_t* mask, int32_t numRows) {
  if (!mask || numRows <= 0) {
    return 0;
  }
  int64_t setBits = 0;
  const int64_t fullBytes = numRows / 8;
  for (int64_t i = 0; i < fullBytes; ++i) {
    setBits += __builtin_popcount(mask[i]);
  }
  const int rem = numRows % 8;
  if (rem > 0) {
    setBits += __builtin_popcount(mask[fullBytes] & ((1 << rem) - 1));
  }
  return static_cast<cudf::size_type>(numRows - setBits);
}

struct DispatchColumn {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
  const std::vector<std::shared_ptr<arrow::Buffer>>& buffers;
  const int32_t numRows;
  int32_t bufferIdx = 0;

  std::unique_ptr<rmm::device_buffer> getMaskBuffer(const std::shared_ptr<arrow::Buffer>& buffer) {
    if (buffer == nullptr || buffer->size() == 0) {
      return std::make_unique<rmm::device_buffer>(0, stream, mr);
    }

    auto mask = std::make_unique<rmm::device_buffer>(buffer->size(), stream, mr);
    CUDF_CUDA_TRY(
        cudaMemcpyAsync(mask->data(), buffer->data(), buffer->size(), cudaMemcpyHostToDevice, stream.value()));
    return mask;
  }

  template <TypeKind Kind, typename T = typename TypeTraits<Kind>::NativeType>
  std::unique_ptr<cudf::column> readFlatColumn(cudf::type_id typeId) {
    auto nulls = buffers[bufferIdx++];
    auto values = buffers[bufferIdx++];

    rmm::device_buffer dataBuf(values->size(), stream, mr);
    CUDF_CUDA_TRY(
        cudaMemcpyAsync(dataBuf.data(), values->data(), values->size(), cudaMemcpyHostToDevice, stream.value()));

    auto nullBuf = getMaskBuffer(nulls);

    cudf::data_type cudfType{typeId};
    cudf::size_type nullCount = (nulls == nullptr || nulls->size() == 0)
        ? 0
        : cpuNullCount(nulls->data(), numRows);
    return std::make_unique<cudf::column>(cudfType, numRows, std::move(dataBuf), std::move(*nullBuf), nullCount);
  }

  std::unique_ptr<cudf::column> getOffsetsColumn(const std::shared_ptr<arrow::Buffer>& offsets) {
    VELOX_CHECK_GT(numRows, 0);
    rmm::device_buffer offsetBuf(offsets->size(), stream, mr);
    CUDF_CUDA_TRY(
        cudaMemcpyAsync(offsetBuf.data(), offsets->data(), offsets->size(), cudaMemcpyHostToDevice, stream.value()));

    rmm::device_buffer nullBuf(0, stream, mr);

    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(numRows + 1),
        std::move(offsetBuf),
        std::move(nullBuf),
        0);
  }

  std::unique_ptr<cudf::column> readFlatColumnStringView(cudf::type_id /*typeId*/) {
    auto nulls = buffers[bufferIdx++];
    auto offsets = buffers[bufferIdx++];
    auto valueBuffer = buffers[bufferIdx++];

    if (numRows == 0) {
      return make_empty_column(cudf::type_id::STRING);
    }

    auto mask = getMaskBuffer(nulls);

    cudf::size_type nullCount = (nulls == nullptr || nulls->size() == 0)
        ? 0
        : cpuNullCount(nulls->data(), numRows);

    VELOX_CHECK_EQ(
        offsets->size(),
        static_cast<int64_t>((numRows + 1) * sizeof(int32_t)),
        "String offsets buffer size mismatch: expected {} got {}",
        (numRows + 1) * sizeof(int32_t),
        offsets->size());

    auto offsetColumn = getOffsetsColumn(offsets);

    rmm::device_buffer chars(valueBuffer->size(), stream, mr);
    if (valueBuffer->size() > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          chars.data(), valueBuffer->data_as<uint8_t>(), chars.size(),
          cudaMemcpyHostToDevice, stream.value()));
    }
    return cudf::make_strings_column(
        numRows, std::move(offsetColumn), std::move(chars), nullCount, std::move(*mask));
  }
};

template <>
std::unique_ptr<cudf::column> DispatchColumn::readFlatColumn<TypeKind::VARCHAR>(cudf::type_id typeId) {
  return readFlatColumnStringView(typeId);
}

template <>
std::unique_ptr<cudf::column> DispatchColumn::readFlatColumn<TypeKind::VARBINARY>(cudf::type_id typeId) {
  return readFlatColumnStringView(typeId);
}

} // namespace

std::shared_ptr<VeloxColumnarBatch> gpuBuffersToCudfVector(
    RowTypePtr type,
    int32_t numRows,
    const std::vector<std::shared_ptr<arrow::Buffer>>& buffers,
    memory::MemoryPool* pool) {
  std::vector<std::unique_ptr<cudf::column>> cudfColumns;
  cudfColumns.reserve(type->size());

  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  DispatchColumn dispatch{stream, cudf::get_current_device_resource_ref(), buffers, numRows};
  for (const auto& colType : type->children()) {
    auto res = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        dispatch.readFlatColumn, colType->kind(), cudf_velox::veloxToCudfTypeId(colType));
    cudfColumns.emplace_back(std::move(res));
  }
  auto cudfTable = std::make_unique<cudf::table>(std::move(cudfColumns));
  stream.synchronize();
  CUDF_CUDA_TRY(cudaGetLastError());
  return std::make_shared<VeloxColumnarBatch>(
      std::make_shared<cudf_velox::CudfVector>(pool, type, numRows, std::move(cudfTable), stream), type->size());
}

arrow::MemoryPool* getPinnedArrowMemoryPool() {
  return &PinnedArrowMemoryPool::instance();
}

GpuBufferBatchResizer::GpuBufferBatchResizer(
    arrow::MemoryPool* arrowPool,
    facebook::velox::memory::MemoryPool* pool,
    int32_t minOutputBatchSize,
    int64_t minOutputBatchSizeInBytes,
    std::unique_ptr<ColumnarBatchIterator> in)
    : arrowPool_(arrowPool),
      pool_(pool),
      minOutputBatchSize_(minOutputBatchSize),
      minOutputBatchSizeInBytes_(minOutputBatchSizeInBytes),
      in_(std::move(in)) {
  VELOX_CHECK_GT(minOutputBatchSize_, 0, "minOutputBatchSize should be larger than 0");
  VELOX_CHECK_GT(minOutputBatchSizeInBytes_, 0, "minOutputBatchSizeInBytes should be larger than 0");
}

GpuBufferBatchResizer::~GpuBufferBatchResizer() {
  if (batches_ > 0) {
    LOG(INFO) << "GpuBufferBatchResizer summary: batches=" << batches_
              << " rows=" << totalRows_
              << " composeMs=" << (composeNs_ / 1'000'000)
              << " h2dUploadMs=" << (h2dUploadNs_ / 1'000'000);
  }
}

std::shared_ptr<ColumnarBatch> GpuBufferBatchResizer::next() {
  std::vector<std::shared_ptr<GpuBufferColumnarBatch>> cachedBatches;
  int32_t cachedRows = 0;
  int64_t cachedBytes = 0;
  while (cachedRows < minOutputBatchSize_ && cachedBytes < minOutputBatchSizeInBytes_) {
    auto nextCb = in_->next();
    if (!nextCb) {
      break;
    }

    auto nextBatch = std::dynamic_pointer_cast<GpuBufferColumnarBatch>(nextCb);
    VELOX_CHECK_NOT_NULL(nextBatch);
    if (nextBatch->numRows() == 0) {
        continue;
    }

    cachedRows += nextBatch->numRows();
    cachedBytes += nextBatch->numBytes();
    cachedBatches.push_back(std::move(nextBatch));
  }
  if (cachedRows == 0) {
    return nullptr;
  }

  ++batches_;
  totalRows_ += cachedRows;

  using VD = facebook::velox::cudf_velox::VeloxDomain;

  std::shared_ptr<GpuBufferColumnarBatch> batch;
  {
    nvtx3::scoped_range_in<VD> compRange(
        nvtx3::event_attributes{
            "ShuffleRead::compose",
            nvtx3::rgb{50, 205, 50}});
    ScopedTimer composeTimer(&composeNs_);
    batch = GpuBufferColumnarBatch::compose(
        getPinnedArrowMemoryPool(),
        cachedBatches, cachedRows);
  }

  GpuLockGuard gpuLock;
  {
    nvtx3::scoped_range_in<VD> h2dRange(
        nvtx3::event_attributes{
            "ShuffleRead::H2D",
            nvtx3::rgb{220, 20, 60}});
    ScopedTimer h2dTimer(&h2dUploadNs_);
    return gpuBuffersToCudfVector(
        batch->getRowType(), batch->numRows(),
        batch->buffers(), pool_);
  }
}

int64_t GpuBufferBatchResizer::spillFixedSize(int64_t size) {
  return in_->spillFixedSize(size);
}

} // namespace gluten
