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

#include <atomic>
#include <cstring>
#include <memory>
#include <sstream>

#include "velox/common/memory/MmapAllocator.h"

namespace gluten {

// MemoryAllocator adapter used only as AsyncDataCache backing. Byte
// allocations are CUDA-owned pinned host memory, so cache hits can be copied
// directly to a device without cudaHostRegister or a pageable bounce copy.
// AsyncDataCache remains responsible for admission, eviction, and retry.
class PinnedCacheAllocator final : public facebook::velox::memory::MemoryAllocator {
 public:
  explicit PinnedCacheAllocator(const Options& options)
      : MemoryAllocator(options.largestSizeClass), capacity_(options.capacity), fallback_(options) {}

  ~PinnedCacheAllocator() override {
    if (usedBytes_.load(std::memory_order_relaxed) != 0) {
      LOG(ERROR) << "PinnedCacheAllocator destroyed with " << usedBytes_.load(std::memory_order_relaxed)
                 << " live bytes";
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
    const auto status = cudaFreeHost(pointer);
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
        << " allocations=" << allocations_.load(std::memory_order_relaxed) << "]";
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

  void* allocateBytesWithoutRetry(uint64_t bytes, uint16_t /* alignment */) override {
    auto current = usedBytes_.load(std::memory_order_relaxed);
    do {
      if (bytes > capacity_ || current > capacity_ - bytes) {
        return nullptr;
      }
    } while (!usedBytes_.compare_exchange_weak(
        current, current + bytes, std::memory_order_acq_rel, std::memory_order_relaxed));

    void* pointer = nullptr;
    const auto status = cudaHostAlloc(&pointer, bytes, cudaHostAllocPortable);
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

  const size_t capacity_;
  facebook::velox::memory::MmapAllocator fallback_;
  std::shared_ptr<facebook::velox::memory::Cache> cache_;
  std::atomic<uint64_t> usedBytes_{0};
  std::atomic<uint64_t> allocations_{0};
};

} // namespace gluten

#endif
