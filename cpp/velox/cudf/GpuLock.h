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

/// Set the executor-wide GPU task admission limit.
void setMaxConcurrentGpuTasks(int n);

/// Get the executor-wide GPU task admission limit.
int getMaxConcurrentGpuTasks();

/// Fine-grained GPU locking is bypassed for MPP execution.
void lockGpu();

void unlockGpu();

/// Acquire one executor-wide task admission permit. This is separate from the
/// fine-grained GPU lock, which remains bypassed for MPP execution.
void lockGpuTask();

/// Release one executor-wide task admission permit.
void unlockGpuTask();

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

class GpuTaskLockGuard {
 public:
  GpuTaskLockGuard() {
    lockGpuTask();
  }
  ~GpuTaskLockGuard() {
    unlockGpuTask();
  }
  GpuTaskLockGuard(const GpuTaskLockGuard&) = delete;
  GpuTaskLockGuard& operator=(const GpuTaskLockGuard&) = delete;
};

} // namespace gluten
