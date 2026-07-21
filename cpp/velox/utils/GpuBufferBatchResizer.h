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

#include "memory/ColumnarBatchIterator.h"
#include "memory/VeloxColumnarBatch.h"
#include "utils/Exception.h"
#include "velox/common/memory/MemoryPool.h"

#include <arrow/buffer.h>
#include <arrow/memory_pool.h>

namespace gluten {

/// Convert CPU-resident Arrow buffers (from GpuBufferColumnarBatch) directly to
/// a CudfVector on the GPU, bypassing the Velox RowVector intermediate format.
std::shared_ptr<VeloxColumnarBatch> gpuBuffersToCudfVector(
    facebook::velox::RowTypePtr type,
    int32_t numRows,
    const std::vector<std::shared_ptr<arrow::Buffer>>& buffers,
    facebook::velox::memory::MemoryPool* pool);

/// arrow::MemoryPool backed by cudf's pinned-memory pool, enabling
/// DMA-friendly allocations for composed shuffle buffers.
arrow::MemoryPool* getPinnedArrowMemoryPool();

class GpuBufferBatchResizer : public ColumnarBatchIterator {
 public:
  GpuBufferBatchResizer(
      arrow::MemoryPool* arrowPool,
      facebook::velox::memory::MemoryPool* pool,
      int32_t minOutputBatchSize,
      int64_t minOutputBatchSizeInBytes,
      std::unique_ptr<ColumnarBatchIterator> in);

  ~GpuBufferBatchResizer() override;

  std::shared_ptr<ColumnarBatch> next() override;

  int64_t spillFixedSize(int64_t size) override;

 private:
  arrow::MemoryPool* arrowPool_;
  facebook::velox::memory::MemoryPool* pool_;
  const int32_t minOutputBatchSize_;
  const int64_t minOutputBatchSizeInBytes_;
  std::unique_ptr<ColumnarBatchIterator> in_;
  std::shared_ptr<ColumnarBatch> stashedBatch_;

  int64_t batches_{0};
  int64_t totalRows_{0};
  int64_t composeNs_{0};
  int64_t h2dUploadNs_{0};
};

} // namespace gluten
