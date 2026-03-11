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

#include "shuffle/Payload.h"

#include <arrow/buffer.h>
#include <arrow/memory_pool.h>
#include <arrow/util/compression.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace gluten;

// ~70% compression ratio (similar to production shuffle data)
static std::shared_ptr<arrow::Buffer> makeRealisticBuffer(
    int64_t size, uint32_t seed) {
  auto pool = arrow::default_memory_pool();
  auto buf = *arrow::AllocateResizableBuffer(size, pool);
  std::mt19937 gen(seed);
  auto* data = buf->mutable_data();
  for (int64_t i = 0; i < size; i += 8) {
    uint64_t val = gen();
    auto len = std::min<int64_t>(8, size - i);
    memcpy(data + i, &val, len);
  }
  return std::move(buf);
}

struct BenchConfig {
  std::string label;
  int numBuffers;
  int64_t bufferSize;
  int numBatches;
};

static void runBench(const BenchConfig& cfg) {
  auto pool = arrow::default_memory_pool();
  auto codec = *arrow::util::Codec::Create(
      arrow::Compression::LZ4_FRAME);
  std::vector<bool> isValidity(cfg.numBuffers, false);

  std::vector<std::shared_ptr<arrow::Buffer>> templates;
  for (int j = 0; j < cfg.numBuffers; j++) {
    templates.push_back(
        makeRealisticBuffer(cfg.bufferSize, j));
  }

  // Warm up
  for (int i = 0; i < 3; i++) {
    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    for (int j = 0; j < cfg.numBuffers; j++) {
      auto copy = *arrow::AllocateResizableBuffer(
          cfg.bufferSize, pool);
      memcpy(
          copy->mutable_data(), templates[j]->data(),
          cfg.bufferSize);
      buffers.push_back(std::move(copy));
    }
    auto r = BlockPayload::fromBuffers(
        Payload::kCompressed, 100, std::move(buffers),
        &isValidity, pool, codec.get());
    (void)r;
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int b = 0; b < cfg.numBatches; b++) {
    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    for (int j = 0; j < cfg.numBuffers; j++) {
      auto copy = *arrow::AllocateResizableBuffer(
          cfg.bufferSize, pool);
      memcpy(
          copy->mutable_data(), templates[j]->data(),
          cfg.bufferSize);
      buffers.push_back(std::move(copy));
    }
    auto r = BlockPayload::fromBuffers(
        Payload::kCompressed, 100, std::move(buffers),
        &isValidity, pool, codec.get());
    (void)r;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<
                 std::chrono::microseconds>(end - start)
                .count() /
      1000.0;

  std::cout << "  " << ms << " ms  ("
            << (ms / cfg.numBatches) << " ms/batch)"
            << std::endl;
}

int main() {
  std::vector<BenchConfig> configs = {
      {"Q9-like (8 bufs x 1MB)", 8, 1024 * 1024, 50},
      {"Q3/Q7-like (16 bufs x 512KB)", 16, 512 * 1024, 50},
      {"Production-like (8 bufs x 8MB)",
       8, 8 * 1024 * 1024, 20},
      {"Large batch (16 bufs x 8MB)",
       16, 8 * 1024 * 1024, 10},
  };

  for (const auto& cfg : configs) {
    int64_t totalMB = (int64_t)cfg.numBuffers *
        cfg.bufferSize / (1024 * 1024);
    std::cout << "\n=== " << cfg.label << " ("
              << totalMB << " MB/batch, "
              << cfg.numBatches << " batches) ==="
              << std::endl;
    runBench(cfg);
  }
  return 0;
}
