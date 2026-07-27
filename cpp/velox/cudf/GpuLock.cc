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
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <glog/logging.h>

// Fine-grained GPU locking remains bypassed for MPP execution. BSP task
// admission uses the shared concurrency state below.
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
};

GpuLockState& getState() {
  static GpuLockState state;
  return state;
}

thread_local int taskLockRefCount = 0;

} // namespace

void setMaxConcurrentGpuTasks(int n) {
  auto& s = getState();
  std::unique_lock<std::mutex> lock(s.mutex);
  int prev = s.maxConcurrent;
  s.maxConcurrent = std::max(1, n);
  LOG(INFO) << "GPU task concurrency: " << prev << " -> " << s.maxConcurrent;
  if (s.maxConcurrent > prev) {
    s.cv.notify_all();
  }
}

int getMaxConcurrentGpuTasks() {
  auto& s = getState();
  std::unique_lock<std::mutex> lock(s.mutex);
  return s.maxConcurrent;
}

void lockGpu() {
  // MPP livelock diagnosis: no-op. No futex/cv wait.
#if GLUTEN_GPULOCK_TRACE
  std::cerr << "GPU_LOCK [lockGpu-bypass] tid="
            << std::this_thread::get_id() << std::endl;
#endif
}

void unlockGpu() {
  // MPP livelock diagnosis: no-op. Paired with the bypassed lockGpu.
#if GLUTEN_GPULOCK_TRACE
  std::cerr << "GPU_LOCK [unlockGpu-bypass] tid="
            << std::this_thread::get_id() << std::endl;
#endif
}

void lockGpuTask() {
  if (taskLockRefCount > 0) {
    ++taskLockRefCount;
    return;
  }
  auto& state = getState();
  std::unique_lock<std::mutex> lock(state.mutex);
  state.cv.wait(
      lock, [&] { return state.activeCount < state.maxConcurrent; });
  ++state.activeCount;
  taskLockRefCount = 1;
}

void unlockGpuTask() {
  if (taskLockRefCount <= 0) {
    return;
  }
  if (--taskLockRefCount > 0) {
    return;
  }
  auto& state = getState();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    --state.activeCount;
  }
  state.cv.notify_one();
}

} // namespace gluten
