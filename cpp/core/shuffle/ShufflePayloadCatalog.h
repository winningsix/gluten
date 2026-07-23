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

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace gluten {

struct PartitionEntry {
  std::vector<std::shared_ptr<arrow::Buffer>> serializedBlocks;
  // Prevent the pool from being destroyed while buffers
  // allocated from it are still alive in the catalog.
  std::shared_ptr<arrow::MemoryPool> poolRef;
};

class ShufflePayloadCatalog {
 public:
  static ShufflePayloadCatalog& instance();

  void registerPartition(int32_t shuffleId, int64_t mapId, int32_t partitionId, PartitionEntry entry);

  struct ResolveResult {
    bool found{false};
    // Segment pointers/sizes. Empty if not found.
    std::vector<std::pair<const uint8_t*, int64_t>> segments;
  };

  // Combined has+resolve in one call with shared_lock.
  ResolveResult resolveBlockDirect(int32_t shuffleId, int64_t mapId, int32_t partitionId);

  void unregisterShuffle(int32_t shuffleId);

  void unregisterMapOutput(int32_t shuffleId, int64_t mapId);

 private:
  ShufflePayloadCatalog() = default;

  struct BlockKey {
    int32_t shuffleId;
    int64_t mapId;
    int32_t partitionId;

    bool operator==(const BlockKey& o) const {
      return shuffleId == o.shuffleId && mapId == o.mapId && partitionId == o.partitionId;
    }
  };

  struct BlockKeyHash {
    size_t operator()(const BlockKey& k) const {
      size_t h = std::hash<int32_t>()(k.shuffleId);
      h ^= std::hash<int64_t>()(k.mapId) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<int32_t>()(k.partitionId) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  struct MapKey {
    int32_t shuffleId;
    int64_t mapId;

    bool operator==(const MapKey& o) const {
      return shuffleId == o.shuffleId && mapId == o.mapId;
    }
  };

  struct MapKeyHash {
    size_t operator()(const MapKey& k) const {
      size_t h = std::hash<int32_t>()(k.shuffleId);
      h ^= std::hash<int64_t>()(k.mapId) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  // Read-write lock: readers (resolve) use shared_lock,
  // writers (register/unregister) use unique_lock.
  mutable std::shared_mutex mutex_;

  std::unordered_map<BlockKey, PartitionEntry, BlockKeyHash> catalog_;

  std::unordered_map<MapKey, std::vector<BlockKey>, MapKeyHash> mapIndex_;

  std::unordered_map<int32_t, std::vector<MapKey>> shuffleIndex_;
};

// An InputStream that reads from a sequence of Buffers
// without copying them into contiguous memory.
class MultiSegmentInputStream : public arrow::io::InputStream {
 public:
  explicit MultiSegmentInputStream(std::vector<std::shared_ptr<arrow::Buffer>> segments);

  arrow::Status Close() override;
  bool closed() const override;
  arrow::Result<int64_t> Tell() const override;
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override;

 private:
  std::vector<std::shared_ptr<arrow::Buffer>> segments_;
  size_t currentSegment_{0};
  int64_t posInSegment_{0};
  int64_t totalPos_{0};
  bool closed_{false};
};

} // namespace gluten
