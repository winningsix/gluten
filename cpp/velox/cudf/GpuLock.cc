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
#include <mutex>
#include <glog/logging.h>
#include <nvtx3/nvtx3.hpp>
#include "velox/experimental/cudf/exec/NvtxHelper.h"

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

thread_local int tLocalRefCount = 0;

} // namespace

void setMaxConcurrentGpuTasks(int n) {
  auto& s = getState();
  std::unique_lock<std::mutex> lock(s.mutex);
  int prev = s.maxConcurrent;
  s.maxConcurrent = std::max(1, n);
  LOG(INFO) << "GPU concurrency: " << prev << " -> " << s.maxConcurrent;
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
  if (tLocalRefCount > 0) {
    ++tLocalRefCount;
    return;
  }
  auto& s = getState();
  std::unique_lock<std::mutex> lock(s.mutex);
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
}

void unlockGpu() {
  if (tLocalRefCount <= 0) {
    return;
  }
  --tLocalRefCount;
  if (tLocalRefCount > 0) {
    return;
  }
  auto& s = getState();
  std::unique_lock<std::mutex> lock(s.mutex);
  --s.activeCount;
  lock.unlock();
  s.cv.notify_one();
}

} // namespace gluten
