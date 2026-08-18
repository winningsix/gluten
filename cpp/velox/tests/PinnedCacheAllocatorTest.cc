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

#include "compute/PinnedCacheAllocator.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <memory>

namespace gluten {
namespace {

struct HookStats {
  std::atomic<uint64_t> allocations{0};
  std::atomic<uint64_t> frees{0};
};

TEST(PinnedCacheAllocatorTest, suballocatesAndCoalescesPreallocatedSlabs) {
  auto stats = std::make_shared<HookStats>();
  PinnedCacheAllocator::HostAllocationHooks hooks;
  hooks.allocate = [stats](void** pointer, size_t bytes, unsigned int) {
    ++stats->allocations;
    return posix_memalign(pointer, 4096, bytes) == 0 ? cudaSuccess : cudaErrorMemoryAllocation;
  };
  hooks.free = [stats](void* pointer) {
    ++stats->frees;
    std::free(pointer);
    return cudaSuccess;
  };

  facebook::velox::memory::MemoryAllocator::Options options;
  options.capacity = 2048;
  {
    PinnedCacheAllocator allocator(options, 1024, 512, std::move(hooks));
    EXPECT_EQ(stats->allocations.load(), 2);
    EXPECT_NE(allocator.toString().find("preallocatedBytes=1024"), std::string::npos);

    auto* first = allocator.allocateBytes(128, 64);
    auto* middle = allocator.allocateBytes(256, 64);
    auto* last = allocator.allocateBytes(128, 64);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(middle, nullptr);
    ASSERT_NE(last, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(first) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(middle) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(last) % 64, 0);

    allocator.freeBytes(middle, 256);
    allocator.freeBytes(first, 128);
    allocator.freeBytes(last, 128);

    auto* coalesced = allocator.allocateBytes(512, 64);
    auto* secondSlab = allocator.allocateBytes(512, 64);
    auto* fallback = allocator.allocateBytes(512, 64);
    ASSERT_NE(coalesced, nullptr);
    ASSERT_NE(secondSlab, nullptr);
    ASSERT_NE(fallback, nullptr);
    EXPECT_EQ(stats->allocations.load(), 3);
    EXPECT_EQ(allocator.totalUsedBytes(), 1536);
    EXPECT_NE(allocator.toString().find("arenaUsedBytes=1024"), std::string::npos);

    allocator.freeBytes(secondSlab, 512);
    allocator.freeBytes(fallback, 512);
    allocator.freeBytes(coalesced, 512);
    EXPECT_EQ(allocator.totalUsedBytes(), 0);
    EXPECT_EQ(stats->frees.load(), 1);
  }
  EXPECT_EQ(stats->frees.load(), 3);
}

} // namespace
} // namespace gluten
