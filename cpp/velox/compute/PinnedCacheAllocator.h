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

#ifdef GLUTEN_ENABLE_GPU

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "velox/common/memory/MmapAllocator.h"

namespace gluten {

// MemoryAllocator adapter used only as AsyncDataCache backing. Byte
// allocations are CUDA-owned pinned host memory, so cache hits can be copied
// directly to a device without cudaHostRegister or a pageable bounce copy.
// AsyncDataCache remains responsible for admission, eviction, and retry.
class PinnedCacheAllocator final : public facebook::velox::memory::MemoryAllocator {
 public:
  struct HostAllocationHooks {
    std::function<cudaError_t(void**, size_t, unsigned int)> allocate;
    std::function<cudaError_t(void*)> free;
  };

  static constexpr uint64_t kDefaultPreallocationSlabBytes = 256ULL << 20;

  explicit PinnedCacheAllocator(
      const Options& options,
      uint64_t preallocateBytes = 0,
      uint64_t preallocationSlabBytes = kDefaultPreallocationSlabBytes)
      : PinnedCacheAllocator(
            options,
            preallocateBytes,
            preallocationSlabBytes,
            HostAllocationHooks{
                [](void** pointer, size_t bytes, unsigned int flags) { return cudaHostAlloc(pointer, bytes, flags); },
                [](void* pointer) { return cudaFreeHost(pointer); }}) {}

  PinnedCacheAllocator(
      const Options& options,
      uint64_t preallocateBytes,
      uint64_t preallocationSlabBytes,
      HostAllocationHooks hooks)
      : MemoryAllocator(options.largestSizeClass),
        capacity_(options.capacity),
        fallback_(options),
        hooks_(std::move(hooks)) {
    preallocate(std::min<uint64_t>(preallocateBytes, capacity_), preallocationSlabBytes);
    if (preallocateBytes > capacity_) {
      LOG(WARNING) << "PinnedCacheAllocator preallocation limited to capacity: requestedBytes=" << preallocateBytes
                   << " capacity=" << capacity_;
    }
  }

  ~PinnedCacheAllocator() override {
    if (usedBytes_.load(std::memory_order_relaxed) != 0) {
      LOG(ERROR) << "PinnedCacheAllocator destroyed with " << usedBytes_.load(std::memory_order_relaxed)
                 << " live bytes";
    }
    std::lock_guard<std::mutex> lock(arenaMutex_);
    if (!arenaAllocations_.empty()) {
      LOG(ERROR) << "PinnedCacheAllocator destroyed with " << arenaAllocations_.size() << " live arena allocations";
    }
    for (const auto& slab : slabs_) {
      const auto status = hooks_.free(slab.base);
      if (status == cudaErrorCudartUnloading) {
        // The process-wide CUDA runtime is already gone. Remaining slabs are
        // reclaimed by process exit; avoid one misleading error per slab.
        cudaGetLastError();
        break;
      }
      if (status != cudaSuccess) {
        LOG(ERROR) << "PinnedCacheAllocator failed to release a preallocated slab: " << cudaGetErrorString(status);
        cudaGetLastError();
      }
    }
  }

  Kind kind() const override {
    return Kind::kMmap;
  }

  void registerCache(const std::shared_ptr<facebook::velox::memory::Cache>& cache) override {
    VELOX_CHECK_NULL(cache_);
    VELOX_CHECK_NOT_NULL(cache);
    VELOX_CHECK(cache->allocator() == this);
    cache_ = cache;
  }

  size_t capacity() const override {
    return capacity_;
  }

  int64_t freeNonContiguous(facebook::velox::memory::Allocation& allocation) override {
    return fallback_.freeNonContiguous(allocation);
  }

  void freeContiguous(facebook::velox::memory::ContiguousAllocation& allocation) override {
    fallback_.freeContiguous(allocation);
  }

  void freeBytes(void* pointer, uint64_t bytes) noexcept override {
    if (pointer == nullptr) {
      return;
    }
    bool belongsToArena = false;
    uint64_t allocatedBytes = 0;
    {
      std::lock_guard<std::mutex> lock(arenaMutex_);
      const auto allocationIt = arenaAllocations_.find(pointer);
      if (allocationIt != arenaAllocations_.end()) {
        const auto allocation = allocationIt->second;
        allocatedBytes = allocation.bytes;
        releaseArenaBlock(allocation);
        arenaAllocations_.erase(allocationIt);
        arenaUsedBytes_.fetch_sub(allocatedBytes, std::memory_order_relaxed);
        belongsToArena = true;
      } else {
        const auto address = reinterpret_cast<uintptr_t>(pointer);
        belongsToArena = std::any_of(slabs_.begin(), slabs_.end(), [address](const auto& slab) {
          const auto begin = reinterpret_cast<uintptr_t>(slab.base);
          return address >= begin && address < begin + slab.bytes;
        });
      }
    }

    if (belongsToArena) {
      if (allocatedBytes == 0) {
        LOG(ERROR) << "PinnedCacheAllocator received an invalid or duplicate arena free at " << pointer;
        return;
      }
      if (allocatedBytes != bytes) {
        LOG(ERROR) << "PinnedCacheAllocator arena free size mismatch: allocatedBytes=" << allocatedBytes
                   << " freeBytes=" << bytes;
      }
      usedBytes_.fetch_sub(allocatedBytes, std::memory_order_relaxed);
      allocations_.fetch_sub(1, std::memory_order_relaxed);
      return;
    }

    const auto status = hooks_.free(pointer);
    if (status != cudaSuccess) {
      LOG(ERROR) << "cudaFreeHost failed for Velox cache backing: " << cudaGetErrorString(status);
      cudaGetLastError();
    }
    usedBytes_.fetch_sub(bytes, std::memory_order_relaxed);
    allocations_.fetch_sub(1, std::memory_order_relaxed);
  }

  facebook::velox::memory::MachinePageCount unmap(facebook::velox::memory::MachinePageCount targetPages) override {
    return fallback_.unmap(targetPages);
  }

  bool checkConsistency() const override {
    return fallback_.checkConsistency();
  }

  size_t totalUsedBytes() const override {
    return usedBytes_.load(std::memory_order_relaxed) + fallback_.totalUsedBytes();
  }

  facebook::velox::memory::MachinePageCount numAllocated() const override {
    return facebook::velox::memory::AllocationTraits::numPages(usedBytes_.load(std::memory_order_relaxed)) +
        fallback_.numAllocated();
  }

  facebook::velox::memory::MachinePageCount numMapped() const override {
    return facebook::velox::memory::AllocationTraits::numPages(usedBytes_.load(std::memory_order_relaxed)) +
        fallback_.numMapped();
  }

  facebook::velox::memory::MachinePageCount numExternalMapped() const override {
    return fallback_.numExternalMapped();
  }

  std::string toString() const override {
    std::ostringstream out;
    out << "PinnedCacheAllocator[capacity=" << capacity_ << " usedBytes=" << usedBytes_.load(std::memory_order_relaxed)
        << " allocations=" << allocations_.load(std::memory_order_relaxed)
        << " preallocatedBytes=" << preallocatedBytes_
        << " arenaUsedBytes=" << arenaUsedBytes_.load(std::memory_order_relaxed) << "]";
    return out.str();
  }

 private:
  bool allocateNonContiguousWithoutRetry(const SizeMix& sizeMix, facebook::velox::memory::Allocation& out) override {
    return fallback_.allocateNonContiguous(sizeMix.totalPages, out);
  }

  bool allocateContiguousWithoutRetry(
      facebook::velox::memory::MachinePageCount numPages,
      facebook::velox::memory::Allocation* collateral,
      facebook::velox::memory::ContiguousAllocation& allocation,
      facebook::velox::memory::MachinePageCount maxPages = 0) override {
    return fallback_.allocateContiguous(numPages, collateral, allocation, nullptr, maxPages);
  }

  bool growContiguousWithoutRetry(
      facebook::velox::memory::MachinePageCount increment,
      facebook::velox::memory::ContiguousAllocation& allocation) override {
    return fallback_.growContiguous(increment, allocation);
  }

  void* allocateBytesWithoutRetry(uint64_t bytes, uint16_t alignment) override {
    auto current = usedBytes_.load(std::memory_order_relaxed);
    do {
      if (bytes > capacity_ || current > capacity_ - bytes) {
        return nullptr;
      }
    } while (!usedBytes_.compare_exchange_weak(
        current, current + bytes, std::memory_order_acq_rel, std::memory_order_relaxed));

    if (auto* pointer = allocateFromArena(bytes, alignment)) {
      allocations_.fetch_add(1, std::memory_order_relaxed);
      return pointer;
    }

    void* pointer = nullptr;
    const auto status = hooks_.allocate(&pointer, bytes, cudaHostAllocPortable);
    if (status != cudaSuccess) {
      usedBytes_.fetch_sub(bytes, std::memory_order_relaxed);
      LOG(WARNING) << "cudaHostAlloc failed for " << bytes << " Velox cache bytes: " << cudaGetErrorString(status);
      cudaGetLastError();
      return nullptr;
    }
    allocations_.fetch_add(1, std::memory_order_relaxed);
    return pointer;
  }

  void* allocateZeroFilledWithoutRetry(uint64_t bytes) override {
    auto* pointer = allocateBytesWithoutRetry(bytes, kMinAlignment);
    if (pointer != nullptr) {
      std::memset(pointer, 0, bytes);
    }
    return pointer;
  }

  facebook::velox::memory::Cache* cache() const override {
    return cache_.get();
  }

  struct Slab {
    uint8_t* base;
    uint64_t bytes;
    std::map<uint64_t, uint64_t> freeBlocks;
  };

  struct ArenaAllocation {
    size_t slabIndex;
    uint64_t offset;
    uint64_t bytes;
  };

  void preallocate(uint64_t targetBytes, uint64_t slabBytes) {
    if (targetBytes == 0) {
      return;
    }
    VELOX_USER_CHECK_GT(slabBytes, 0, "Pinned cache preallocation slab size must be positive");
    while (preallocatedBytes_ < targetBytes) {
      const auto bytes = std::min(slabBytes, targetBytes - preallocatedBytes_);
      void* pointer = nullptr;
      const auto status = hooks_.allocate(&pointer, bytes, cudaHostAllocPortable);
      if (status != cudaSuccess) {
        LOG(WARNING) << "PinnedCacheAllocator preallocation stopped after " << preallocatedBytes_ << " of "
                     << targetBytes << " bytes: " << cudaGetErrorString(status);
        cudaGetLastError();
        break;
      }
      slabs_.push_back(Slab{
          .base = static_cast<uint8_t*>(pointer),
          .bytes = bytes,
          .freeBlocks = {{0, bytes}},
      });
      preallocatedBytes_ += bytes;
    }
    LOG(INFO) << "PinnedCacheAllocator preallocated " << preallocatedBytes_ << " of " << targetBytes << " bytes in "
              << slabs_.size() << " slabs";
  }

  void* allocateFromArena(uint64_t bytes, uint16_t alignment) {
    if (slabs_.empty()) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(arenaMutex_);
    for (size_t slabIndex = 0; slabIndex < slabs_.size(); ++slabIndex) {
      auto& slab = slabs_[slabIndex];
      for (auto blockIt = slab.freeBlocks.begin(); blockIt != slab.freeBlocks.end(); ++blockIt) {
        const auto blockOffset = blockIt->first;
        const auto blockBytes = blockIt->second;
        const auto blockAddress = reinterpret_cast<uintptr_t>(slab.base) + blockOffset;
        const auto alignedAddress = (blockAddress + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
        const auto alignmentPadding = alignedAddress - blockAddress;
        if (alignmentPadding > blockBytes || bytes > blockBytes - alignmentPadding) {
          continue;
        }

        const auto alignedOffset = blockOffset + alignmentPadding;
        const auto suffixOffset = alignedOffset + bytes;
        const auto suffixBytes = blockBytes - alignmentPadding - bytes;
        slab.freeBlocks.erase(blockIt);
        if (alignmentPadding > 0) {
          slab.freeBlocks.emplace(blockOffset, alignmentPadding);
        }
        if (suffixBytes > 0) {
          slab.freeBlocks.emplace(suffixOffset, suffixBytes);
        }

        auto* pointer = slab.base + alignedOffset;
        arenaAllocations_.emplace(pointer, ArenaAllocation{slabIndex, alignedOffset, bytes});
        arenaUsedBytes_.fetch_add(bytes, std::memory_order_relaxed);
        return pointer;
      }
    }
    return nullptr;
  }

  void releaseArenaBlock(const ArenaAllocation& allocation) {
    auto& freeBlocks = slabs_[allocation.slabIndex].freeBlocks;
    auto offset = allocation.offset;
    auto bytes = allocation.bytes;
    auto next = freeBlocks.lower_bound(offset);
    if (next != freeBlocks.begin()) {
      auto previous = std::prev(next);
      if (previous->first + previous->second == offset) {
        offset = previous->first;
        bytes += previous->second;
        freeBlocks.erase(previous);
      }
    }
    next = freeBlocks.lower_bound(offset);
    if (next != freeBlocks.end() && offset + bytes == next->first) {
      bytes += next->second;
      freeBlocks.erase(next);
    }
    freeBlocks.emplace(offset, bytes);
  }

  const size_t capacity_;
  facebook::velox::memory::MmapAllocator fallback_;
  HostAllocationHooks hooks_;
  std::shared_ptr<facebook::velox::memory::Cache> cache_;
  mutable std::mutex arenaMutex_;
  std::vector<Slab> slabs_;
  std::unordered_map<void*, ArenaAllocation> arenaAllocations_;
  uint64_t preallocatedBytes_{0};
  std::atomic<uint64_t> usedBytes_{0};
  std::atomic<uint64_t> allocations_{0};
  std::atomic<uint64_t> arenaUsedBytes_{0};
};

} // namespace gluten

#endif
