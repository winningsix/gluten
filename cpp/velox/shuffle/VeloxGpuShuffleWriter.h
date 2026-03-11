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

#include "VeloxHashShuffleWriter.h"

namespace facebook::velox::cudf_velox {
class CudfVector;
}

namespace gluten {

class VeloxGpuHashShuffleWriter : public VeloxHashShuffleWriter {
 public:
  static arrow::Result<std::shared_ptr<VeloxShuffleWriter>> create(
      uint32_t numPartitions,
      const std::shared_ptr<PartitionWriter>& partitionWriter,
      const std::shared_ptr<ShuffleWriterOptions>& options,
      MemoryManager* memoryManager);

  VeloxGpuHashShuffleWriter(
      uint32_t numPartitions,
      const std::shared_ptr<PartitionWriter>& partitionWriter,
      const std::shared_ptr<GpuHashShuffleWriterOptions>& options,
      MemoryManager* memoryManager)
      : VeloxHashShuffleWriter(numPartitions, partitionWriter, options, memoryManager),
        gpuPartitionEnabled_(options->gpuPartition) {}

  arrow::Status write(std::shared_ptr<ColumnarBatch> cb, int64_t memLimit) override;

  arrow::Status stop() override {
    if (gpuWriteBatches_ > 0) {
      LOG(INFO) << "GpuShuffleWriter summary: batches=" << gpuWriteBatches_
                << " gpuPartitionMs=" << (gpuPartitionNs_ / 1'000'000)
                << " d2hMs=" << (d2hNs_ / 1'000'000)
                << " extractMs=" << (extractBufferNs_ / 1'000'000)
                << " evictMs=" << (evictNs_ / 1'000'000)
                << " cpuFallbackBatches=" << cpuFallbackBatches_;
    }
    return VeloxHashShuffleWriter::stop();
  }

 private:
  void splitBoolValueType(const uint8_t* srcAddr, const std::vector<uint8_t*>& dstAddrs) override;

  uint64_t valueBufferSizeForBool(uint32_t newSize) override {
    return newSize;
  }

  bool boolIsBit() override {
    return false;
  }

  arrow::Status splitTimestamp(const uint8_t* srcAddr, const std::vector<uint8_t*>& dstAddrs) override;

  uint64_t valueBufferSizeForTimestamp(uint32_t newSize) override {
    return sizeof(int64_t) * newSize;
  }

  arrow::Status gpuPartitionAndEvict(
      std::shared_ptr<facebook::velox::cudf_velox::CudfVector> cudfVec);

  // Fast path for data pre-partitioned by CudfShufflePartition in the pipeline.
  // The RowVector's first column contains sorted PIDs; scan for boundaries and
  // use extractBuffersFromRowVector per partition (sequential memcpy, no scatter).
  arrow::Status prePartitionedEvict(const facebook::velox::RowVectorPtr& rv);

  // Extract flat buffer list from a range [start, start+numRows) of a Velox RowVector
  // in the format expected by InMemoryPayload. Works on the full (non-sliced) RowVector
  // to avoid offset complications. Handles bool bit→byte, timestamp int128→int64, etc.
  arrow::Status extractBuffersFromRowVector(
      const facebook::velox::RowVector& rv,
      int64_t start,
      uint32_t numRows,
      std::vector<std::shared_ptr<arrow::Buffer>>& outBuffers);

  bool gpuPartitionEnabled_{false};
  bool gpuSchemaInitialized_{false};
  int64_t gpuWriteBatches_{0};
  int64_t cpuFallbackBatches_{0};
  int64_t gpuPartitionNs_{0};
  int64_t d2hNs_{0};
  int64_t extractBufferNs_{0};
  int64_t evictNs_{0};
};
} // namespace gluten
