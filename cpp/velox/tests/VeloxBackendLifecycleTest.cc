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

#include "compute/VeloxBackend.h"

#include <gtest/gtest.h>
#include <cstdlib>

#include "memory/VeloxMemoryManager.h"
#include "utils/Exception.h"
#include "velox/common/caching/FileIds.h"

namespace gluten {

TEST(VeloxBackendLifecycleTest, MemoryOnlyCacheReleasesPagesWithoutSsdDirectory) {
  // Backend factories are process-global and can only be registered once.
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        const std::string prefix = "spark.gluten.sql.columnar.backend.velox.";
        VeloxBackend::create(
            AllocationListener::noop(),
            {{prefix + "cacheEnabled", "true"},
             {prefix + "memCacheSize", "67108864"},
             {prefix + "ssdCacheSize", "0"},
             {prefix + "cacheContiguousEntries", "true"},
             {prefix + "cachePinnedBytes", "0"},
             {prefix + "cachePinnedPrewarmBytes", "0"}});
        auto* cache = VeloxBackend::get()->getAsyncDataCache();
        ASSERT_NE(cache, nullptr);
        auto* allocator = cache->allocator();
        facebook::velox::StringIdLease file(facebook::velox::fileIds(), "memory-only-cache-teardown");
        for (const auto bytes : {16 << 10, 2 << 20}) {
          auto pin = cache->findOrCreate(
              facebook::velox::cache::RawFileCacheKey{file.id(), static_cast<uint64_t>(bytes)}, bytes, true);
          pin.checkedEntry()->setExclusiveToShared();
          pin.clear();
        }
        ASSERT_GT(allocator->numAllocated(), 0);
        ASSERT_GT(allocator->numExternalMapped(), 0);

        VeloxBackend::get()->tearDown();

        EXPECT_EQ(allocator->numAllocated(), 0);
        EXPECT_EQ(allocator->numExternalMapped(), 0);
        EXPECT_NO_THROW(VeloxBackend::get()->tearDown());
        std::exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST(VeloxBackendLifecycleTest, RejectsMemoryManagerAccessAfterTerminalTearDown) {
  VeloxBackend::create(AllocationListener::noop(), {});
  ASSERT_NE(VeloxBackend::get()->getGlobalMemoryManager(), nullptr);

  VeloxBackend::get()->tearDown();

  EXPECT_THROW(VeloxBackend::get()->getGlobalMemoryManager(), GlutenException);
  EXPECT_THROW(defaultLeafVeloxMemoryPool(), GlutenException);
  EXPECT_NO_THROW(VeloxBackend::get()->tearDown());
}

} // namespace gluten
