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

#include "shuffle/ShufflePayloadCatalog.h"

#include <glog/logging.h>
#include <mutex>

namespace gluten {

ShufflePayloadCatalog& ShufflePayloadCatalog::instance() {
  static auto* inst = new ShufflePayloadCatalog();
  return *inst;
}

void ShufflePayloadCatalog::registerPartition(
    int32_t shuffleId,
    int64_t mapId,
    int32_t partitionId,
    PartitionEntry entry) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  BlockKey bk{shuffleId, mapId, partitionId};
  MapKey mk{shuffleId, mapId};

  catalog_[bk] = std::move(entry);
  mapIndex_[mk].push_back(bk);

  auto& maps = shuffleIndex_[shuffleId];
  bool found = false;
  for (const auto& m : maps) {
    if (m == mk) {
      found = true;
      break;
    }
  }
  if (!found) {
    maps.push_back(mk);
  }
}

ShufflePayloadCatalog::ResolveResult
ShufflePayloadCatalog::resolveBlockDirect(
    int32_t shuffleId,
    int64_t mapId,
    int32_t partitionId) {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  BlockKey bk{shuffleId, mapId, partitionId};
  auto it = catalog_.find(bk);
  if (it == catalog_.end()) {
    return {false, {}};
  }

  ResolveResult result;
  result.found = true;
  for (const auto& b : it->second.serializedBlocks) {
    if (b && b->size() > 0) {
      result.segments.emplace_back(
          b->data(), b->size());
    }
  }
  return result;
}

void ShufflePayloadCatalog::unregisterShuffle(
    int32_t shuffleId) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  auto sit = shuffleIndex_.find(shuffleId);
  if (sit == shuffleIndex_.end()) {
    return;
  }

  for (const auto& mk : sit->second) {
    auto mit = mapIndex_.find(mk);
    if (mit != mapIndex_.end()) {
      for (const auto& bk : mit->second) {
        catalog_.erase(bk);
      }
      mapIndex_.erase(mit);
    }
  }
  shuffleIndex_.erase(sit);
}

void ShufflePayloadCatalog::unregisterMapOutput(
    int32_t shuffleId, int64_t mapId) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  MapKey mk{shuffleId, mapId};
  auto mit = mapIndex_.find(mk);
  if (mit == mapIndex_.end()) {
    return;
  }

  for (const auto& bk : mit->second) {
    catalog_.erase(bk);
  }
  mapIndex_.erase(mit);

  auto sit = shuffleIndex_.find(shuffleId);
  if (sit != shuffleIndex_.end()) {
    auto& maps = sit->second;
    maps.erase(
        std::remove(maps.begin(), maps.end(), mk),
        maps.end());
    if (maps.empty()) {
      shuffleIndex_.erase(sit);
    }
  }
}

// -- MultiSegmentInputStream --

MultiSegmentInputStream::MultiSegmentInputStream(
    std::vector<std::shared_ptr<arrow::Buffer>> segments)
    : segments_(std::move(segments)) {}

arrow::Status MultiSegmentInputStream::Close() {
  closed_ = true;
  segments_.clear();
  return arrow::Status::OK();
}

bool MultiSegmentInputStream::closed() const {
  return closed_;
}

arrow::Result<int64_t>
MultiSegmentInputStream::Tell() const {
  return totalPos_;
}

arrow::Result<int64_t> MultiSegmentInputStream::Read(
    int64_t nbytes, void* out) {
  if (closed_) {
    return arrow::Status::Invalid("Stream is closed");
  }

  auto* dst = static_cast<uint8_t*>(out);
  int64_t totalRead = 0;

  while (totalRead < nbytes &&
         currentSegment_ < segments_.size()) {
    const auto& seg = segments_[currentSegment_];
    int64_t available = seg->size() - posInSegment_;
    int64_t toRead =
        std::min(nbytes - totalRead, available);

    memcpy(
        dst + totalRead,
        seg->data() + posInSegment_, toRead);
    totalRead += toRead;
    posInSegment_ += toRead;

    if (posInSegment_ >= seg->size()) {
      currentSegment_++;
      posInSegment_ = 0;
    }
  }

  totalPos_ += totalRead;
  return totalRead;
}

arrow::Result<std::shared_ptr<arrow::Buffer>>
MultiSegmentInputStream::Read(int64_t nbytes) {
  ARROW_ASSIGN_OR_RAISE(
      auto buf,
      arrow::AllocateResizableBuffer(
          nbytes, arrow::default_memory_pool()));
  ARROW_ASSIGN_OR_RAISE(
      auto bytesRead,
      Read(nbytes, buf->mutable_data()));
  RETURN_NOT_OK(buf->Resize(bytesRead));
  return std::shared_ptr<arrow::Buffer>(std::move(buf));
}

} // namespace gluten
