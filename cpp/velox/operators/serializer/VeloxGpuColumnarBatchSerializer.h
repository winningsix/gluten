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

#include <arrow/c/abi.h>
#include <cudf/packed_types.hpp>
#include <vector>

#include "VeloxColumnarBatchSerializer.h"

#include "memory/ColumnarBatch.h"
#include "operators/serializer/ColumnarBatchSerializer.h"
#include "velox/serializers/PrestoSerializer.h"

namespace gluten {

class VeloxGpuColumnarBatchSerializer final : public VeloxColumnarBatchSerializer {
 public:
  VeloxGpuColumnarBatchSerializer(
      arrow::MemoryPool* arrowPool,
      std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
      struct ArrowSchema* cSchema);

  ~VeloxGpuColumnarBatchSerializer() override;

  void append(const std::shared_ptr<ColumnarBatch>& batch) override;

  void appendMany(
      const std::vector<std::shared_ptr<ColumnarBatch>>& batches) override;

  int64_t maxSerializedSize() override;

  void serializeTo(uint8_t* address, int64_t size) override;

  // Deserialize to cudf table, then the Cudf pipeline accepts CudfVector, we can remove CudfFromveloc operator from the
  // velox pipeline input.
  std::shared_ptr<ColumnarBatch> deserialize(uint8_t* data, int32_t size) override;

  std::shared_ptr<ColumnarBatch> deserializeSelected(
      uint8_t* data,
      int32_t size,
      const std::vector<int32_t>& columnIndices) override;

  std::shared_ptr<ColumnarBatch> deserializeParquetFile(
      const std::string& path) override;

  std::shared_ptr<ColumnarBatch> deserializeParquetFileSelected(
      const std::string& path,
      const std::vector<int32_t>& columnIndices) override;

  void beginParquetFile(const std::string& path) override;

  int64_t appendParquetFile(
      const std::shared_ptr<ColumnarBatch>& batch) override;

  int64_t appendParquetFileMany(
      const std::vector<std::shared_ptr<ColumnarBatch>>& batches) override;

  int64_t finishParquetFile() override;

  void drainParquetFileWrites() override;

 private:
  struct ParquetFileWriterState;
  std::unique_ptr<ParquetFileWriterState> parquetFileWriterState_;
  std::unique_ptr<cudf::packed_columns> packedColumns_;
  std::vector<uint8_t> hostGpuData_;
  uint32_t uncompressedGpuDataSize_{0};
  uint32_t cacheVersion_{1};
  bool cudfCache_{false};
};

} // namespace gluten
