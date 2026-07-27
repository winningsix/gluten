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

#include "cudf/GpuLock.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace gluten {
namespace {

TEST(GpuTaskLockTest, BoundsConcurrentTasks) {
  setMaxConcurrentGpuTasks(2);
  std::atomic<int> active{0};
  std::atomic<int> peak{0};
  std::vector<std::thread> threads;

  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      GpuTaskLockGuard guard;
      const auto current = ++active;
      auto observed = peak.load();
      while (current > observed &&
             !peak.compare_exchange_weak(observed, current)) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      --active;
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(active, 0);
  EXPECT_EQ(peak, 2);
}

TEST(GpuTaskLockTest, ReleasesOnlyOutermostGuard) {
  setMaxConcurrentGpuTasks(1);
  std::atomic<bool> contenderStarted{false};
  std::atomic<bool> contenderAcquired{false};

  lockGpuTask();
  lockGpuTask();
  std::thread contender([&] {
    contenderStarted = true;
    GpuTaskLockGuard guard;
    contenderAcquired = true;
  });

  while (!contenderStarted) {
    std::this_thread::yield();
  }
  unlockGpuTask();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(contenderAcquired);

  unlockGpuTask();
  contender.join();
  EXPECT_TRUE(contenderAcquired);
}

} // namespace
} // namespace gluten
