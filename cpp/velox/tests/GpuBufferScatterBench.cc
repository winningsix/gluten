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

#include <gtest/gtest.h>

#include <cstdint>

#include "cudf/GpuLock.h"
#include "memory/GpuBufferColumnarBatch.h"
#include "utils/GpuBufferBatchResizer.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/type/Type.h"

#include <arrow/buffer.h>
#include <arrow/memory_pool.h>

#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <iostream>
#include <numeric>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <vector>

using namespace facebook::velox;

namespace {

cudf::size_type cpuNullCount(const uint8_t* mask, int32_t numRows) {
  if (!mask || numRows <= 0)
    return 0;
  int64_t setBits = 0;
  const int64_t fullBytes = numRows / 8;
  for (int64_t i = 0; i < fullBytes; ++i)
    setBits += __builtin_popcount(mask[i]);
  const int rem = numRows % 8;
  if (rem > 0)
    setBits += __builtin_popcount(mask[fullBytes] & ((1 << rem) - 1));
  return static_cast<cudf::size_type>(numRows - setBits);
}

/// Build a cudf::table from composed Arrow buffers using the ORIGINAL code path
/// (cudf::null_count on CPU bitmask pointer — triggers implicit GPU sync per
/// column with nulls).
std::unique_ptr<cudf::table> makeCudfTableOriginal(
    const RowTypePtr& type,
    int32_t numRows,
    const std::vector<std::shared_ptr<arrow::Buffer>>& buffers,
    rmm::cuda_stream_view stream) {
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> cols;
  int idx = 0;
  for (const auto& colType : type->children()) {
    auto nulls = buffers[idx++];
    if (colType->kind() == TypeKind::VARCHAR ||
        colType->kind() == TypeKind::VARBINARY) {
      auto offsets = buffers[idx++];
      auto values = buffers[idx++];
      if (numRows == 0) {
        cols.push_back(cudf::make_empty_column(cudf::type_id::STRING));
        continue;
      }
      rmm::device_buffer maskDev(0, stream, mr);
      size_t nullCount = 0;
      if (nulls && nulls->size() > 0) {
        maskDev = rmm::device_buffer(nulls->size(), stream, mr);
        cudaMemcpyAsync(
            maskDev.data(), nulls->data(), nulls->size(),
            cudaMemcpyHostToDevice, stream.value());
        // Must sync before null_count can read the device buffer.
        stream.synchronize();
        nullCount = cudf::null_count(
            reinterpret_cast<const cudf::bitmask_type*>(maskDev.data()),
            0, numRows, stream);
      }
      rmm::device_buffer offsetDev(offsets->size(), stream, mr);
      cudaMemcpyAsync(
          offsetDev.data(), offsets->data(), offsets->size(),
          cudaMemcpyHostToDevice, stream.value());
      auto offsetCol = std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT32},
          static_cast<cudf::size_type>(numRows + 1),
          std::move(offsetDev), rmm::device_buffer(0, stream, mr), 0);
      rmm::device_buffer charsDev(values->size(), stream, mr);
      cudaMemcpyAsync(
          charsDev.data(), values->data(), values->size(),
          cudaMemcpyDefault, stream.value());
      cols.push_back(cudf::make_strings_column(
          numRows, std::move(offsetCol), std::move(charsDev),
          nullCount, std::move(maskDev)));
    } else {
      auto values = buffers[idx++];
      rmm::device_buffer dataDev(values->size(), stream, mr);
      cudaMemcpyAsync(
          dataDev.data(), values->data(), values->size(),
          cudaMemcpyHostToDevice, stream.value());
      rmm::device_buffer maskDev(0, stream, mr);
      size_t nullCount = 0;
      if (nulls && nulls->size() > 0) {
        maskDev = rmm::device_buffer(nulls->size(), stream, mr);
        cudaMemcpyAsync(
            maskDev.data(), nulls->data(), nulls->size(),
            cudaMemcpyHostToDevice, stream.value());
        stream.synchronize();
        nullCount = cudf::null_count(
            reinterpret_cast<const cudf::bitmask_type*>(maskDev.data()),
            0, numRows, stream);
      }
      cudf::data_type dt{cudf_velox::veloxToCudfTypeId(colType)};
      cols.push_back(std::make_unique<cudf::column>(
          dt, numRows, std::move(dataDev), std::move(maskDev), nullCount));
    }
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Build a cudf::table using OPTIMIZED code path (cpuNullCount — no implicit
/// GPU sync).
std::unique_ptr<cudf::table> makeCudfTableOptimized(
    const RowTypePtr& type,
    int32_t numRows,
    const std::vector<std::shared_ptr<arrow::Buffer>>& buffers,
    rmm::cuda_stream_view stream) {
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> cols;
  int idx = 0;
  for (const auto& colType : type->children()) {
    auto nulls = buffers[idx++];
    if (colType->kind() == TypeKind::VARCHAR ||
        colType->kind() == TypeKind::VARBINARY) {
      auto offsets = buffers[idx++];
      auto values = buffers[idx++];
      if (numRows == 0) {
        cols.push_back(cudf::make_empty_column(cudf::type_id::STRING));
        continue;
      }
      rmm::device_buffer maskDev(0, stream, mr);
      cudf::size_type nullCount = 0;
      if (nulls && nulls->size() > 0) {
        maskDev = rmm::device_buffer(nulls->size(), stream, mr);
        cudaMemcpyAsync(
            maskDev.data(), nulls->data(), nulls->size(),
            cudaMemcpyHostToDevice, stream.value());
        nullCount = cpuNullCount(nulls->data(), numRows);
      }
      rmm::device_buffer offsetDev(offsets->size(), stream, mr);
      cudaMemcpyAsync(
          offsetDev.data(), offsets->data(), offsets->size(),
          cudaMemcpyHostToDevice, stream.value());
      auto offsetCol = std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT32},
          static_cast<cudf::size_type>(numRows + 1),
          std::move(offsetDev), rmm::device_buffer(0, stream, mr), 0);
      rmm::device_buffer charsDev(values->size(), stream, mr);
      cudaMemcpyAsync(
          charsDev.data(), values->data(), values->size(),
          cudaMemcpyDefault, stream.value());
      cols.push_back(cudf::make_strings_column(
          numRows, std::move(offsetCol), std::move(charsDev),
          nullCount, std::move(maskDev)));
    } else {
      auto values = buffers[idx++];
      rmm::device_buffer dataDev(values->size(), stream, mr);
      cudaMemcpyAsync(
          dataDev.data(), values->data(), values->size(),
          cudaMemcpyHostToDevice, stream.value());
      rmm::device_buffer maskDev(0, stream, mr);
      cudf::size_type nullCount = 0;
      if (nulls && nulls->size() > 0) {
        maskDev = rmm::device_buffer(nulls->size(), stream, mr);
        cudaMemcpyAsync(
            maskDev.data(), nulls->data(), nulls->size(),
            cudaMemcpyHostToDevice, stream.value());
        nullCount = cpuNullCount(nulls->data(), numRows);
      }
      cudf::data_type dt{cudf_velox::veloxToCudfTypeId(colType)};
      cols.push_back(std::make_unique<cudf::column>(
          dt, numRows, std::move(dataDev), std::move(maskDev), nullCount));
    }
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

class GpuBufferScatterBench : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::SharedArbitrator::registerFactory();
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool("ScatterBench");
  }

  /// Create a GpuBufferColumnarBatch with ~10% null values.
  std::shared_ptr<gluten::GpuBufferColumnarBatch> makeBatch(
      const RowTypePtr& type,
      int32_t numRows,
      bool withNulls) {
    std::vector<std::shared_ptr<arrow::Buffer>> buffers;

    for (const auto& colType : type->children()) {
      if (!withNulls) {
        buffers.push_back(nullptr);
      } else {
        int64_t maskBytes = (numRows + 7) / 8;
        std::shared_ptr<arrow::Buffer> maskBuf;
        GLUTEN_ASSIGN_OR_THROW(
            maskBuf,
            arrow::AllocateBuffer(maskBytes, arrow::default_memory_pool()));
        auto* mask = const_cast<uint8_t*>(maskBuf->data());
        std::memset(mask, 0xFF, maskBytes);
        // ~10% nulls: clear every 10th bit.
        for (int32_t r = 0; r < numRows; r += 10) {
          mask[r / 8] &= ~(1 << (r % 8));
        }
        buffers.push_back(std::move(maskBuf));
      }

      if (colType->isFixedWidth()) {
        int32_t elemSize = 0;
        switch (colType->kind()) {
          case TypeKind::BIGINT: elemSize = 8; break;
          case TypeKind::INTEGER: elemSize = 4; break;
          case TypeKind::DOUBLE: elemSize = 8; break;
          default: elemSize = 8;
        }
        int64_t dataBytes = static_cast<int64_t>(numRows) * elemSize;
        std::shared_ptr<arrow::Buffer> buf;
        GLUTEN_ASSIGN_OR_THROW(
            buf,
            arrow::AllocateBuffer(dataBytes, arrow::default_memory_pool()));
        std::memset(const_cast<uint8_t*>(buf->data()), 0x42, dataBytes);
        buffers.push_back(std::move(buf));
      } else {
        const int32_t avgLen = 8;
        int64_t lenBytes = static_cast<int64_t>(numRows) * sizeof(int32_t);
        std::shared_ptr<arrow::Buffer> lenBuf;
        GLUTEN_ASSIGN_OR_THROW(
            lenBuf,
            arrow::AllocateBuffer(lenBytes, arrow::default_memory_pool()));
        auto* lengths = reinterpret_cast<int32_t*>(
            const_cast<uint8_t*>(lenBuf->data()));
        int64_t totalChars = 0;
        for (int32_t i = 0; i < numRows; ++i) {
          lengths[i] = avgLen;
          totalChars += avgLen;
        }
        std::shared_ptr<arrow::Buffer> valBuf;
        GLUTEN_ASSIGN_OR_THROW(
            valBuf,
            arrow::AllocateBuffer(totalChars, arrow::default_memory_pool()));
        std::memset(const_cast<uint8_t*>(valBuf->data()), 'A', totalChars);
        buffers.push_back(std::move(lenBuf));
        buffers.push_back(std::move(valBuf));
      }
    }

    return std::make_shared<gluten::GpuBufferColumnarBatch>(
        type, std::move(buffers), numRows);
  }

  double computeDataMB(
      const std::vector<std::shared_ptr<gluten::GpuBufferColumnarBatch>>& bs) {
    double bytes = 0;
    for (const auto& b : bs)
      for (const auto& buf : b->buffers())
        if (buf) bytes += buf->size();
    return bytes / (1024.0 * 1024.0);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

class ReplayIterator : public gluten::ColumnarBatchIterator {
 public:
  explicit ReplayIterator(
      std::vector<std::shared_ptr<gluten::GpuBufferColumnarBatch>> batches)
      : batches_(std::move(batches)), idx_(0) {}
  std::shared_ptr<gluten::ColumnarBatch> next() override {
    if (idx_ >= batches_.size()) return nullptr;
    return batches_[idx_++];
  }
  int64_t spillFixedSize(int64_t) override { return 0; }
 private:
  std::vector<std::shared_ptr<gluten::GpuBufferColumnarBatch>> batches_;
  size_t idx_;
};

} // namespace

/// A/B comparison: original makeCudfTable (cudf::null_count) vs optimized
/// (cpuNullCount), both operating on the same composed batch WITH nulls.
TEST_F(GpuBufferScatterBench, originalVsOptimized) {
  auto type = ROW(
      {"c0", "c1", "c2", "c3", "c4"},
      {BIGINT(), INTEGER(), DOUBLE(), BIGINT(), VARCHAR()});

  constexpr int kBatchRows = 4000;
  constexpr int kNumBatches = 5000;
  constexpr int kTargetRows = kBatchRows * kNumBatches;

  std::cout << "\n=== A/B: original vs optimized makeCudfTable (WITH nulls) ==="
            << "\n" << kNumBatches << " batches x " << kBatchRows
            << " rows = " << kTargetRows << " rows, "
            << type->size() << " cols, ~10% nulls\n";

  std::vector<std::shared_ptr<gluten::GpuBufferColumnarBatch>> batches;
  batches.reserve(kNumBatches);
  for (int i = 0; i < kNumBatches; ++i) {
    batches.push_back(makeBatch(type, kBatchRows, true));
  }

  double dataMB = computeDataMB(batches);
  std::cout << "Data volume: " << dataMB << " MB\n\n";

  // compose once (shared by both).
  auto composed = gluten::GpuBufferColumnarBatch::compose(
      arrow::default_memory_pool(), batches, kTargetRows);

  gluten::GpuLockGuard lock;
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();

  // Warmup.
  {
    auto t = makeCudfTableOptimized(
        type, composed->numRows(), composed->buffers(), stream);
    stream.synchronize();
  }

  // ORIGINAL (cudf::null_count).
  constexpr int kRuns = 3;
  double origMs = 0;
  for (int r = 0; r < kRuns; ++r) {
    auto s = std::chrono::high_resolution_clock::now();
    auto t = makeCudfTableOriginal(
        type, composed->numRows(), composed->buffers(), stream);
    stream.synchronize();
    auto e = std::chrono::high_resolution_clock::now();
    origMs += std::chrono::duration_cast<std::chrono::microseconds>(e - s)
                  .count() / 1000.0;
  }
  origMs /= kRuns;

  // OPTIMIZED (cpuNullCount).
  double optMs = 0;
  for (int r = 0; r < kRuns; ++r) {
    auto s = std::chrono::high_resolution_clock::now();
    auto t = makeCudfTableOptimized(
        type, composed->numRows(), composed->buffers(), stream);
    stream.synchronize();
    auto e = std::chrono::high_resolution_clock::now();
    optMs += std::chrono::duration_cast<std::chrono::microseconds>(e - s)
                 .count() / 1000.0;
  }
  optMs /= kRuns;

  std::cout << "[ORIGINAL] makeCudfTable (cudf::null_count): "
            << origMs << " ms  (" << (dataMB / (origMs / 1000.0))
            << " MB/s)\n";
  std::cout << "[OPTIMIZED] makeCudfTable (cpuNullCount):    "
            << optMs << " ms  (" << (dataMB / (optMs / 1000.0))
            << " MB/s)\n";
  double speedup = origMs / optMs;
  std::cout << "Speedup: " << speedup << "x\n";
}

/// End-to-end benchmark through GpuBufferBatchResizer (with nulls).
TEST_F(GpuBufferScatterBench, endToEndWithNulls) {
  auto type = ROW(
      {"c0", "c1", "c2", "c3", "c4"},
      {BIGINT(), INTEGER(), DOUBLE(), BIGINT(), VARCHAR()});

  constexpr int kBatchRows = 4000;
  constexpr int kNumBatches = 5000;
  constexpr int kTargetRows = kBatchRows * kNumBatches;

  std::cout << "\n=== End-to-end GpuBufferBatchResizer (WITH nulls) ===\n"
            << kNumBatches << " batches x " << kBatchRows
            << " rows = " << kTargetRows << " rows\n";

  std::vector<std::shared_ptr<gluten::GpuBufferColumnarBatch>> batches;
  batches.reserve(kNumBatches);
  for (int i = 0; i < kNumBatches; ++i) {
    batches.push_back(makeBatch(type, kBatchRows, true));
  }

  double dataMB = computeDataMB(batches);
  std::cout << "Data volume: " << dataMB << " MB\n\n";

  // Warmup.
  {
    auto wb = std::vector<std::shared_ptr<gluten::GpuBufferColumnarBatch>>(
        batches.begin(), batches.begin() + 100);
    auto it = std::make_unique<ReplayIterator>(wb);
    gluten::GpuBufferBatchResizer resizer(
        arrow::default_memory_pool(), pool_.get(), 100000, INT64_MAX, std::move(it));
    while (auto cb = resizer.next()) {}
  }

  // Measure compose alone.
  auto cs = std::chrono::high_resolution_clock::now();
  auto composed = gluten::GpuBufferColumnarBatch::compose(
      arrow::default_memory_pool(), batches, kTargetRows);
  auto ce = std::chrono::high_resolution_clock::now();
  double composeMs =
      std::chrono::duration_cast<std::chrono::microseconds>(ce - cs)
          .count() / 1000.0;

  // End-to-end.
  auto s = std::chrono::high_resolution_clock::now();
  {
    auto it = std::make_unique<ReplayIterator>(batches);
    gluten::GpuBufferBatchResizer resizer(
        arrow::default_memory_pool(), pool_.get(), kTargetRows, INT64_MAX, std::move(it));
    auto result = resizer.next();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->numRows(), kTargetRows);
  }
  auto e = std::chrono::high_resolution_clock::now();
  double e2eMs =
      std::chrono::duration_cast<std::chrono::microseconds>(e - s)
          .count() / 1000.0;
  double transferMs = e2eMs - composeMs;

  std::cout << "compose (CPU, no GPU lock): " << composeMs << " ms\n";
  std::cout << "makeCudfTable (GPU locked):  ~" << transferMs << " ms\n";
  std::cout << "end-to-end total:           " << e2eMs << " ms  ("
            << (dataMB / (e2eMs / 1000.0)) << " MB/s)\n";
}

/// Varying batch sizes (no nulls, fixed-width only).
TEST_F(GpuBufferScatterBench, varyingBatchCount) {
  auto type = ROW(
      {"c0", "c1", "c2", "c3"},
      {BIGINT(), INTEGER(), DOUBLE(), BIGINT()});

  constexpr int kTotalRows = 5000000;
  std::vector<int> batchSizes = {100, 500, 2000, 5000, 10000};

  std::cout << "\n=== Varying Batch Size (total " << kTotalRows
            << " rows, 4 fixed-width cols, no nulls) ===\n";
  std::cout << "BatchSize | NumBatches | Time(ms) | Throughput(MB/s)\n";
  std::cout << "----------|------------|----------|----------------\n";

  for (int batchSize : batchSizes) {
    int numBatches = kTotalRows / batchSize;

    std::vector<std::shared_ptr<gluten::GpuBufferColumnarBatch>> batches;
    batches.reserve(numBatches);
    for (int i = 0; i < numBatches; ++i) {
      batches.push_back(makeBatch(type, batchSize, false));
    }

    double dataMB = computeDataMB(batches);

    auto start = std::chrono::high_resolution_clock::now();
    {
      auto it = std::make_unique<ReplayIterator>(batches);
      gluten::GpuBufferBatchResizer resizer(
          arrow::default_memory_pool(), pool_.get(), kTotalRows, INT64_MAX, std::move(it));
      auto result = resizer.next();
      ASSERT_NE(result, nullptr);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() / 1000.0;

    printf(
        "%9d | %10d | %8.1f | %15.0f\n",
        batchSize, numBatches, ms, dataMB / (ms / 1000.0));
  }
}
