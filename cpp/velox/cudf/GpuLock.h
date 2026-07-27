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

#include <thread>

namespace gluten {

/// Set the maximum number of concurrent GPU operations.
/// Default is 1 (serial, same as the original mutex behavior).
/// Values > 1 allow multiple threads to use the GPU concurrently.
void setMaxConcurrentGpuTasks(int n);

/// Get the current max concurrent GPU tasks setting.
int getMaxConcurrentGpuTasks();

/// Enable the GPU semaphore while at least one query-scoped runtime needs it.
/// The process-wide environment switch remains supported; scoped users add a
/// safe dynamic override without changing unrelated sequential queries.
void acquireGpuLockEnableScope();

/// Release one query-scoped GPU semaphore user.
void releaseGpuLockEnableScope();

/// Acquire GPU access. Blocks if the concurrency limit is reached.
/// Reentrant: multiple calls from the same thread are ref-counted.
void lockGpu();

/// Release GPU access. Only truly releases when the ref count reaches zero.
void unlockGpu();

class GpuLockGuard {
 public:
  GpuLockGuard() {
    lockGpu();
  }
  ~GpuLockGuard() {
    unlockGpu();
  }
  GpuLockGuard(const GpuLockGuard&) = delete;
  GpuLockGuard& operator=(const GpuLockGuard&) = delete;
};

} // namespace gluten
