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

#include "GpuLock.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <glog/logging.h>
#include <nvtx3/nvtx3.hpp>
#include "velox/experimental/cudf/exec/NvtxHelper.h"

// Keep the old MPP-diagnosis bypass as the default. Streaming shuffle POC can
// opt back into the counting semaphore with GLUTEN_GPULOCK_ENABLED=true.
// Stderr markers are compiled out to keep executor logs quiet; flip
// GLUTEN_GPULOCK_TRACE to 1 to re-enable.
#ifndef GLUTEN_GPULOCK_TRACE
#define GLUTEN_GPULOCK_TRACE 0
#endif

namespace gluten {

namespace {

struct GpuLockState {
  std::mutex mutex;
  std::condition_variable cv;
  int maxConcurrent{1};
  int activeCount{0};
  std::atomic<int> enabledScopes{0};
};

GpuLockState& getState() {
  static GpuLockState state;
  return state;
}

bool gpuLockEnabledByEnvironment() {
  static const bool enabled = [] {
    const auto* value = std::getenv("GLUTEN_GPULOCK_ENABLED");
    if (value == nullptr) {
      return false;
    }
    std::string normalized{value};
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !normalized.empty() && normalized != "0" &&
        normalized != "false" && normalized != "off" && normalized != "no";
  }();
  return enabled;
}

bool gpuLockEnabled() {
  return gpuLockEnabledByEnvironment() ||
      getState().enabledScopes.load(std::memory_order_relaxed) > 0;
}

thread_local int tLocalRefCount = 0;

} // namespace

void setMaxConcurrentGpuTasks(int n) {
  auto& s = getState();
  std::unique_lock<std::mutex> lock(s.mutex);
  int prev = s.maxConcurrent;
  s.maxConcurrent = std::max(1, n);
  LOG(INFO) << "GPU concurrency"
            << (gpuLockEnabled() ? "" : " (bypassed)") << ": " << prev
            << " -> " << s.maxConcurrent;
  if (s.maxConcurrent > prev) {
    s.cv.notify_all();
  }
}

int getMaxConcurrentGpuTasks() {
  auto& s = getState();
  std::unique_lock<std::mutex> lock(s.mutex);
  return s.maxConcurrent;
}

void acquireGpuLockEnableScope() {
  auto& s = getState();
  const auto previous =
      s.enabledScopes.fetch_add(1, std::memory_order_acq_rel);
  if (previous == 0 && !gpuLockEnabledByEnvironment()) {
    LOG(INFO) << "GPU concurrency semaphore enabled by query-scoped runtime";
  }
}

void releaseGpuLockEnableScope() {
  auto& s = getState();
  const auto previous =
      s.enabledScopes.fetch_sub(1, std::memory_order_acq_rel);
  if (previous <= 0) {
    s.enabledScopes.store(0, std::memory_order_release);
    LOG(ERROR) << "Unbalanced query-scoped GPU concurrency semaphore release";
    return;
  }
  if (previous == 1 && !gpuLockEnabledByEnvironment()) {
    LOG(INFO) << "GPU concurrency semaphore disabled after query-scoped runtime";
  }
}

void lockGpu() {
  if (!gpuLockEnabled()) {
#if GLUTEN_GPULOCK_TRACE
    std::cerr << "GPU_LOCK [lockGpu-bypass] tid="
              << std::this_thread::get_id() << std::endl;
#endif
    return;
  }
  if (tLocalRefCount > 0) {
    ++tLocalRefCount;
    VLOG(2) << "GPU_LOCK [lockGpu-reentrant] tid="
            << std::this_thread::get_id()
            << " refCount=" << tLocalRefCount;
    return;
  }
  auto& s = getState();
  std::unique_lock<std::mutex> lock(s.mutex);
  VLOG(2) << "GPU_LOCK [lockGpu-wait] tid=" << std::this_thread::get_id()
          << " activeCount=" << s.activeCount
          << " maxConcurrent=" << s.maxConcurrent;
  {
    nvtx3::scoped_range_in<
        facebook::velox::cudf_velox::VeloxDomain>
        waitRange(nvtx3::event_attributes{
            "GpuLock::wait",
            nvtx3::rgb{255, 69, 0}});
    s.cv.wait(lock, [&] {
      return s.activeCount < s.maxConcurrent;
    });
  }
  ++s.activeCount;
  tLocalRefCount = 1;
  VLOG(2) << "GPU_LOCK [lockGpu-acquired] tid=" << std::this_thread::get_id()
          << " activeCount=" << s.activeCount
          << " refCount=" << tLocalRefCount;
}

void unlockGpu() {
  // A dynamic query scope can end while an unrelated task is finishing a
  // guard that it acquired during the overlap. Always release an acquired
  // permit even if the semaphore is no longer enabled for new calls.
  if (tLocalRefCount <= 0 && !gpuLockEnabled()) {
#if GLUTEN_GPULOCK_TRACE
    std::cerr << "GPU_LOCK [unlockGpu-bypass] tid="
              << std::this_thread::get_id() << std::endl;
#endif
    return;
  }
  if (tLocalRefCount <= 0) {
    VLOG(2) << "GPU_LOCK [unlockGpu-noop] tid=" << std::this_thread::get_id()
            << " refCount=" << tLocalRefCount;
    return;
  }
  --tLocalRefCount;
  if (tLocalRefCount > 0) {
    VLOG(2) << "GPU_LOCK [unlockGpu-reentrant] tid="
            << std::this_thread::get_id()
            << " refCount=" << tLocalRefCount;
    return;
  }
  auto& s = getState();
  std::unique_lock<std::mutex> lock(s.mutex);
  --s.activeCount;
  VLOG(2) << "GPU_LOCK [unlockGpu-released] tid="
          << std::this_thread::get_id()
          << " activeCount=" << s.activeCount;
  lock.unlock();
  s.cv.notify_one();
}

} // namespace gluten
