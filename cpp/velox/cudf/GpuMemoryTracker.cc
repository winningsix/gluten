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

#include "GpuMemoryTracker.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <mutex>

namespace {

std::mutex& instanceMutex() {
  static std::mutex mutex;
  return mutex;
}

thread_local int64_t tlTaskId = -1;
thread_local std::string tlContext;
thread_local std::string tlOperatorContext;

} // namespace

std::unique_ptr<GpuMemoryTracker> GpuMemoryTracker::instance_;

void GpuMemoryTracker::initialize() {
  std::lock_guard<std::mutex> lock(instanceMutex());
  if (instance_) {
    return;
  }
  instance_.reset(new GpuMemoryTracker());
  LOG(INFO) << "GpuMemoryTracker compatibility shim initialized; "
               "native RMM allocation attribution is disabled";
}

void GpuMemoryTracker::shutdown() {
  std::lock_guard<std::mutex> lock(instanceMutex());
  instance_.reset();
  LOG(INFO) << "GpuMemoryTracker compatibility shim shut down";
}

void GpuMemoryTracker::setCurrentTask(int64_t taskId) {
  tlTaskId = taskId;
}

void GpuMemoryTracker::clearCurrentTask() {
  tlTaskId = -1;
}

void GpuMemoryTracker::setCurrentContext(const std::string& context) {
  tlContext = context;
}

void GpuMemoryTracker::clearCurrentContext() {
  tlContext.clear();
}

void GpuMemoryTracker::setCurrentOperatorContext(const std::string& context) {
  tlOperatorContext = context;
}

void GpuMemoryTracker::clearCurrentOperatorContext() {
  tlOperatorContext.clear();
}

GpuMemoryTracker* GpuMemoryTracker::instance() {
  std::lock_guard<std::mutex> lock(instanceMutex());
  return instance_.get();
}

void GpuMemoryTracker::setMaxConcurrentGpuTasks(int) {}

bool GpuMemoryTracker::diagnosticsEnabled() {
  return instance() != nullptr;
}

void GpuMemoryTracker::dumpDiagnosticsToLog(const std::string& prefix) {
  if (instance() == nullptr) {
    return;
  }

  size_t freeBytes = 0;
  size_t totalBytes = 0;
  auto status = cudaMemGetInfo(&freeBytes, &totalBytes);
  if (status == cudaSuccess) {
    LOG(ERROR) << prefix
               << " gpuMemory freeBytes=" << freeBytes
               << " totalBytes=" << totalBytes
               << " usedBytes=" << (totalBytes - freeBytes)
               << " currentTaskId=" << tlTaskId
               << " context=" << tlContext
               << " operatorContext=" << tlOperatorContext
               << " attribution=disabled";
  } else {
    LOG(ERROR) << prefix << " gpuMemory cudaMemGetInfo failed: "
               << cudaGetErrorString(status)
               << " currentTaskId=" << tlTaskId
               << " context=" << tlContext
               << " operatorContext=" << tlOperatorContext
               << " attribution=disabled";
  }
}

GpuMemoryTracker::ScopedContext::ScopedContext(const std::string& context)
    : previousContext_(tlContext) {
  setCurrentContext(context);
}

GpuMemoryTracker::ScopedContext::~ScopedContext() {
  setCurrentContext(previousContext_);
}

GpuMemoryTracker::ScopedOperatorContext::ScopedOperatorContext(
    const std::string& context)
    : previousContext_(tlOperatorContext) {
  setCurrentOperatorContext(context);
}

GpuMemoryTracker::ScopedOperatorContext::~ScopedOperatorContext() {
  setCurrentOperatorContext(previousContext_);
}

int64_t GpuMemoryTracker::getMaxTaskMemory(int64_t) {
  return 0;
}

int64_t GpuMemoryTracker::clearTaskMemory(int64_t) {
  return 0;
}

int64_t GpuMemoryTracker::getTotalAllocated() const {
  return 0;
}

std::vector<GpuMemoryTracker::TaskSnapshot> GpuMemoryTracker::getTaskSnapshots() {
  return {};
}

std::vector<GpuMemoryTracker::LargeAllocationEvent>
GpuMemoryTracker::getLargeAllocationEvents(std::size_t) {
  return {};
}
