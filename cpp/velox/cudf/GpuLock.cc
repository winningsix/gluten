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

// FLUX livelock diagnosis: GpuLock is bypassed entirely. lockGpu/unlockGpu
// early-return with no futex/cv wait so we can prove the lock is a symptom,
// not the root cause of the FluxNativeQueryExec hang on Q1.
// Symbols are kept exported so other TUs that reference them still link.
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
};

GpuLockState& getState() {
  static GpuLockState state;
  return state;
}

} // namespace

void setMaxConcurrentGpuTasks(int n) {
  auto& s = getState();
  std::unique_lock<std::mutex> lock(s.mutex);
  int prev = s.maxConcurrent;
  s.maxConcurrent = std::max(1, n);
  LOG(INFO) << "GPU concurrency (bypassed): " << prev << " -> "
            << s.maxConcurrent;
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
  // FLUX livelock diagnosis: no-op. No futex/cv wait.
#if GLUTEN_GPULOCK_TRACE
  std::cerr << "GPU_LOCK [lockGpu-bypass] tid="
            << std::this_thread::get_id() << std::endl;
#endif
}

void unlockGpu() {
  // FLUX livelock diagnosis: no-op. Paired with the bypassed lockGpu.
#if GLUTEN_GPULOCK_TRACE
  std::cerr << "GPU_LOCK [unlockGpu-bypass] tid="
            << std::this_thread::get_id() << std::endl;
#endif
}

} // namespace gluten
