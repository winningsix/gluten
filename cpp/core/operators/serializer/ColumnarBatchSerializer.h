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
#include <stdexcept>
#include <string>
#include <vector>

#include "memory/ColumnarBatch.h"

namespace gluten {

class ColumnarBatchSerializer {
 public:
  ColumnarBatchSerializer(arrow::MemoryPool* arrowPool) : arrowPool_(arrowPool) {}

  virtual ~ColumnarBatchSerializer() = default;

  virtual void append(const std::shared_ptr<ColumnarBatch>& batch) = 0;

  virtual void appendMany(
      const std::vector<std::shared_ptr<ColumnarBatch>>& batches) {
    for (const auto& batch : batches) {
      append(batch);
    }
  }

  virtual int64_t maxSerializedSize() = 0;

  virtual void serializeTo(uint8_t* address, int64_t size) = 0;

  virtual std::shared_ptr<ColumnarBatch> deserialize(uint8_t* data, int32_t size) = 0;

  // Deserialize only the requested top-level columns. Implementations whose
  // storage format supports projection (for example Parquet) can avoid
  // materializing unrequested payload. Other implementations must still
  // return a batch containing exactly columnIndices in the requested order.
  virtual std::shared_ptr<ColumnarBatch> deserializeSelected(
      uint8_t* data,
      int32_t size,
      const std::vector<int32_t>& columnIndices) = 0;

  virtual std::shared_ptr<ColumnarBatch> deserializeParquetFile(
      const std::string& path) {
    throw std::runtime_error("Standalone Parquet cache pages are unsupported");
  }

  virtual std::shared_ptr<ColumnarBatch> deserializeParquetFileSelected(
      const std::string& path,
      const std::vector<int32_t>& columnIndices) {
    throw std::runtime_error("Standalone selected Parquet cache pages are unsupported");
  }

  // Stateful file-backed cache writer. The caller appends one bounded batch
  // at a time and rolls the file when the returned on-disk size reaches its
  // page target. This avoids retaining hundreds of device batches merely to
  // produce one reasonably sized Parquet file.
  virtual void beginParquetFile(const std::string& path) {
    throw std::runtime_error("Stateful Parquet cache writing is unsupported");
  }

  virtual int64_t appendParquetFile(
      const std::shared_ptr<ColumnarBatch>& batch) {
    throw std::runtime_error("Stateful Parquet cache writing is unsupported");
  }

  virtual int64_t appendParquetFileMany(
      const std::vector<std::shared_ptr<ColumnarBatch>>& batches) {
    throw std::runtime_error("Stateful grouped Parquet cache writing is unsupported");
  }

  virtual int64_t finishParquetFile() {
    throw std::runtime_error("Stateful Parquet cache writing is unsupported");
  }

  // Wait for file-backed cache pages whose footer has been produced but whose
  // asynchronous sink is still draining. Implementations without deferred
  // file completion have nothing to do.
  virtual void drainParquetFileWrites() {}

 protected:
  arrow::MemoryPool* arrowPool_;
};

} // namespace gluten
