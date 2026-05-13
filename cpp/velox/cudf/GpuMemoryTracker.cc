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

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <execinfo.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <optional>
#include <sstream>
#include <thread>
#include <typeinfo>

namespace {

constexpr const char* kDiagnosticsEnv = "GLUTEN_GPU_MEMORY_DIAGNOSTICS";
constexpr const char* kLargeAllocThresholdEnv =
    "GLUTEN_GPU_MEMORY_DIAGNOSTICS_LARGE_ALLOC_THRESHOLD_BYTES";
constexpr const char* kLargeAllocEventsEnv =
    "GLUTEN_GPU_MEMORY_DIAGNOSTICS_EVENTS";
constexpr const char* kLargeAllocTopNEnv =
    "GLUTEN_GPU_MEMORY_DIAGNOSTICS_TOP_N";
constexpr const char* kRetryTrimOnOomEnv =
    "GLUTEN_GPU_MEMORY_RETRY_TRIM_ON_OOM";
constexpr const char* kAsyncPoolReleaseThresholdEnv =
    "GLUTEN_GPU_MEMORY_ASYNC_POOL_RELEASE_THRESHOLD_BYTES";
constexpr const char* kAsyncPoolReuseFollowEventDependenciesEnv =
    "GLUTEN_GPU_MEMORY_ASYNC_POOL_REUSE_FOLLOW_EVENT_DEPENDENCIES";
constexpr const char* kAsyncPoolReuseAllowOpportunisticEnv =
    "GLUTEN_GPU_MEMORY_ASYNC_POOL_REUSE_ALLOW_OPPORTUNISTIC";
constexpr const char* kAsyncPoolReuseAllowInternalDependenciesEnv =
    "GLUTEN_GPU_MEMORY_ASYNC_POOL_REUSE_ALLOW_INTERNAL_DEPENDENCIES";
constexpr const char* kUnattributedContext = "unattributed";

bool envFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  std::string normalized(value);
  std::transform(
      normalized.begin(),
      normalized.end(),
      normalized.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized != "0" && normalized != "false" && normalized != "off" &&
      normalized != "no";
}

std::size_t envSizeOrDefault(const char* name, std::size_t defaultValue) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return defaultValue;
  }
  char* end = nullptr;
  auto parsed = std::strtoull(value, &end, 10);
  if (end == value || parsed == 0) {
    LOG(WARNING) << "Ignoring invalid " << name << "='" << value << "'";
    return defaultValue;
  }
  return static_cast<std::size_t>(parsed);
}

std::optional<uint64_t> envUint64(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  char* end = nullptr;
  auto parsed = std::strtoull(value, &end, 10);
  if (end == value) {
    LOG(WARNING) << "Ignoring invalid " << name << "='" << value << "'";
    return std::nullopt;
  }
  return static_cast<uint64_t>(parsed);
}

std::optional<int> envBoolInt(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return envFlagEnabled(name) ? 1 : 0;
}

std::string threadIdString() {
  std::ostringstream out;
  out << std::this_thread::get_id();
  return out.str();
}

std::string contextOrUnattributed(const std::string& context) {
  return context.empty() ? kUnattributedContext : context;
}

std::string captureStackTrace() {
  constexpr int kMaxFrames = 16;
  void* frames[kMaxFrames];
  const int frameCount = backtrace(frames, kMaxFrames);
  if (frameCount <= 0) {
    return "unavailable";
  }

  char** symbols = backtrace_symbols(frames, frameCount);
  if (symbols == nullptr) {
    return "unavailable";
  }

  std::ostringstream out;
  for (int i = 2; i < frameCount; ++i) {
    if (i > 2) {
      out << " | ";
    }
    out << symbols[i];
  }
  std::free(symbols);
  return out.str();
}

void logCudaMemPoolStats(
    const std::string& prefix,
    const char* label,
    cudaMemPool_t pool,
    int device) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
  if (pool == nullptr) {
    LOG(ERROR) << prefix << " cudaMemPool[" << label << "] device=" << device
               << " unavailable=null";
    return;
  }

  auto getAttribute = [&](cudaMemPoolAttr attr, const char* name) {
    uint64_t value = 0;
    auto attrErr = cudaMemPoolGetAttribute(pool, attr, &value);
    if (attrErr != cudaSuccess) {
      LOG(ERROR) << prefix << " cudaMemPool[" << label << "] device=" << device
                 << " attr=" << name
                 << " error=" << cudaGetErrorString(attrErr);
      return value;
    }
    return value;
  };

  const auto reservedCurrent = getAttribute(
      cudaMemPoolAttrReservedMemCurrent, "ReservedMemCurrent");
  const auto reservedHigh =
      getAttribute(cudaMemPoolAttrReservedMemHigh, "ReservedMemHigh");
  const auto usedCurrent =
      getAttribute(cudaMemPoolAttrUsedMemCurrent, "UsedMemCurrent");
  const auto usedHigh = getAttribute(cudaMemPoolAttrUsedMemHigh, "UsedMemHigh");

  LOG(ERROR) << prefix << " cudaMemPool[" << label << "] device=" << device
             << " reservedCurrentBytes=" << reservedCurrent
             << " reservedCurrentMiB=" << (reservedCurrent / 1048576.0)
             << " reservedHighBytes=" << reservedHigh
             << " reservedHighMiB=" << (reservedHigh / 1048576.0)
             << " usedCurrentBytes=" << usedCurrent
             << " usedCurrentMiB=" << (usedCurrent / 1048576.0)
             << " usedHighBytes=" << usedHigh
             << " usedHighMiB=" << (usedHigh / 1048576.0);
#else
#if defined(CUDART_VERSION)
  LOG(ERROR) << prefix << " cudaMemPool unavailable CUDART_VERSION="
             << CUDART_VERSION;
#else
  LOG(ERROR) << prefix << " cudaMemPool unavailable CUDART_VERSION=undefined";
#endif
#endif
}

void logDefaultCudaMemPoolStats(const std::string& prefix) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
  int device = -1;
  auto err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    LOG(ERROR) << prefix << " cudaMemPool[default] cudaGetDeviceFailed="
               << cudaGetErrorString(err);
    return;
  }

  cudaMemPool_t pool = nullptr;
  err = cudaDeviceGetDefaultMemPool(&pool, device);
  if (err != cudaSuccess) {
    LOG(ERROR) << prefix << " cudaMemPool[default] device=" << device
               << " cudaDeviceGetDefaultMemPoolFailed="
               << cudaGetErrorString(err);
    return;
  }
  logCudaMemPoolStats(prefix, "default", pool, device);
#else
#if defined(CUDART_VERSION)
  LOG(ERROR) << prefix << " cudaMemPool[default] unavailable CUDART_VERSION="
             << CUDART_VERSION;
#else
  LOG(ERROR) << prefix << " cudaMemPool[default] unavailable CUDART_VERSION=undefined";
#endif
#endif
}

void logRmmUpstreamCudaMemPoolStats(
    const std::string& prefix,
    rmm::mr::device_memory_resource* upstream) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
  int device = -1;
  auto err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    LOG(ERROR) << prefix << " cudaMemPool[rmm-upstream] cudaGetDeviceFailed="
               << cudaGetErrorString(err);
    return;
  }

  if (auto* asyncMr =
          dynamic_cast<rmm::mr::cuda_async_memory_resource*>(upstream)) {
    logCudaMemPoolStats(prefix, "rmm-upstream", asyncMr->pool_handle(), device);
    return;
  }
  if (auto* asyncViewMr =
          dynamic_cast<rmm::mr::cuda_async_view_memory_resource*>(upstream)) {
    logCudaMemPoolStats(
        prefix, "rmm-upstream-view", asyncViewMr->pool_handle(), device);
    return;
  }

  LOG(ERROR) << prefix << " cudaMemPool[rmm-upstream] unavailable upstreamType="
             << (upstream == nullptr ? "null" : typeid(*upstream).name());
#else
#if defined(CUDART_VERSION)
  LOG(ERROR) << prefix << " cudaMemPool[rmm-upstream] unavailable CUDART_VERSION="
             << CUDART_VERSION;
#else
  LOG(ERROR) << prefix
             << " cudaMemPool[rmm-upstream] unavailable CUDART_VERSION=undefined";
#endif
#endif
}

bool trimRmmUpstreamCudaMemPool(
    const std::string& prefix,
    rmm::mr::device_memory_resource* upstream) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
  cudaMemPool_t pool = nullptr;
  const char* label = "rmm-upstream";
  if (auto* asyncMr =
          dynamic_cast<rmm::mr::cuda_async_memory_resource*>(upstream)) {
    pool = asyncMr->pool_handle();
  } else if (auto* asyncViewMr =
                 dynamic_cast<rmm::mr::cuda_async_view_memory_resource*>(
                     upstream)) {
    pool = asyncViewMr->pool_handle();
    label = "rmm-upstream-view";
  }
  if (pool == nullptr) {
    LOG(WARNING) << prefix
                 << " cudaMemPool trim skipped, unsupported upstreamType="
                 << (upstream == nullptr ? "null" : typeid(*upstream).name());
    return false;
  }

  int device = -1;
  auto err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    LOG(WARNING) << prefix << " cudaMemPool trim cudaGetDeviceFailed="
                 << cudaGetErrorString(err);
    return false;
  }

  logCudaMemPoolStats(prefix + " before-trim", label, pool, device);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    LOG(WARNING) << prefix << " cudaMemPool trim cudaDeviceSynchronizeFailed="
                 << cudaGetErrorString(err);
    return false;
  }
  err = cudaMemPoolTrimTo(pool, 0);
  if (err != cudaSuccess) {
    LOG(WARNING) << prefix << " cudaMemPool trim failed="
                 << cudaGetErrorString(err);
    return false;
  }
  logCudaMemPoolStats(prefix + " after-trim", label, pool, device);
  return true;
#else
#if defined(CUDART_VERSION)
  LOG(WARNING) << prefix << " cudaMemPool trim unavailable CUDART_VERSION="
               << CUDART_VERSION;
#else
  LOG(WARNING) << prefix
               << " cudaMemPool trim unavailable CUDART_VERSION=undefined";
#endif
  return false;
#endif
}

void configureRmmUpstreamCudaMemPool(
    const std::string& prefix,
    rmm::mr::device_memory_resource* upstream) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
  cudaMemPool_t pool = nullptr;
  const char* label = "rmm-upstream";
  if (auto* asyncMr =
          dynamic_cast<rmm::mr::cuda_async_memory_resource*>(upstream)) {
    pool = asyncMr->pool_handle();
  } else if (auto* asyncViewMr =
                 dynamic_cast<rmm::mr::cuda_async_view_memory_resource*>(
                     upstream)) {
    pool = asyncViewMr->pool_handle();
    label = "rmm-upstream-view";
  }
  if (pool == nullptr) {
    return;
  }

  auto setUint64Attr = [&](cudaMemPoolAttr attr,
                           const char* attrName,
                           std::optional<uint64_t> value) {
    if (!value.has_value()) {
      return;
    }
    uint64_t attrValue = value.value();
    auto err = cudaMemPoolSetAttribute(pool, attr, &attrValue);
    if (err != cudaSuccess) {
      LOG(WARNING) << prefix << " cudaMemPool[" << label << "] set "
                   << attrName << "=" << attrValue
                   << " failed=" << cudaGetErrorString(err);
    } else {
      LOG(WARNING) << prefix << " cudaMemPool[" << label << "] set "
                   << attrName << "=" << attrValue;
    }
  };
  auto setIntAttr = [&](cudaMemPoolAttr attr,
                        const char* attrName,
                        std::optional<int> value) {
    if (!value.has_value()) {
      return;
    }
    int attrValue = value.value();
    auto err = cudaMemPoolSetAttribute(pool, attr, &attrValue);
    if (err != cudaSuccess) {
      LOG(WARNING) << prefix << " cudaMemPool[" << label << "] set "
                   << attrName << "=" << attrValue
                   << " failed=" << cudaGetErrorString(err);
    } else {
      LOG(WARNING) << prefix << " cudaMemPool[" << label << "] set "
                   << attrName << "=" << attrValue;
    }
  };

  setUint64Attr(
      cudaMemPoolAttrReleaseThreshold,
      "ReleaseThreshold",
      envUint64(kAsyncPoolReleaseThresholdEnv));
  setIntAttr(
      cudaMemPoolReuseFollowEventDependencies,
      "ReuseFollowEventDependencies",
      envBoolInt(kAsyncPoolReuseFollowEventDependenciesEnv));
  setIntAttr(
      cudaMemPoolReuseAllowOpportunistic,
      "ReuseAllowOpportunistic",
      envBoolInt(kAsyncPoolReuseAllowOpportunisticEnv));
  setIntAttr(
      cudaMemPoolReuseAllowInternalDependencies,
      "ReuseAllowInternalDependencies",
      envBoolInt(kAsyncPoolReuseAllowInternalDependenciesEnv));
#endif
}

} // namespace

// ── Static data ──────────────────────────────────────────────────────────────

std::unique_ptr<GpuMemoryTracker> GpuMemoryTracker::instance_;
rmm::mr::device_memory_resource* GpuMemoryTracker::originalUpstream_ = nullptr;

// ── Constructor ──────────────────────────────────────────────────────────────

GpuMemoryTracker::GpuMemoryTracker(rmm::mr::device_memory_resource* upstream)
    : upstream_(upstream),
      diagnosticsEnabled_(envFlagEnabled(kDiagnosticsEnv)),
      retryTrimOnOom_(envFlagEnabled(kRetryTrimOnOomEnv)),
      largeAllocationThresholdBytes_(envSizeOrDefault(
          kLargeAllocThresholdEnv,
          largeAllocationThresholdBytes_)),
      largeAllocationEventCapacity_(
          envSizeOrDefault(kLargeAllocEventsEnv, largeAllocationEventCapacity_)),
      largeAllocationTopN_(
          envSizeOrDefault(kLargeAllocTopNEnv, largeAllocationTopN_)) {}

// ── Static API ───────────────────────────────────────────────────────────────

void GpuMemoryTracker::initialize() {
  if (instance_) {
    LOG(WARNING) << "GpuMemoryTracker already initialized";
    return;
  }
  originalUpstream_ = rmm::mr::get_current_device_resource();
  if (!originalUpstream_) {
    LOG(WARNING) << "No RMM device resource found, skipping GpuMemoryTracker";
    return;
  }
  instance_.reset(new GpuMemoryTracker(originalUpstream_));
  rmm::mr::set_current_device_resource(instance_.get());
  configureRmmUpstreamCudaMemPool("GpuMemoryTracker initialize", originalUpstream_);
  LOG(INFO) << "GpuMemoryTracker initialized, wrapping upstream resource"
            << " diagnosticsEnabled=" << instance_->diagnosticsEnabled_
            << " retryTrimOnOom=" << instance_->retryTrimOnOom_
            << " currentType=" << instance_->currentResourceType()
            << " upstreamType=" << typeid(*originalUpstream_).name()
            << " largeAllocThresholdBytes="
            << instance_->largeAllocationThresholdBytes_
            << " largeAllocEventCapacity="
            << instance_->largeAllocationEventCapacity_;
}

void GpuMemoryTracker::shutdown() {
  if (!instance_) return;
  if (originalUpstream_) {
    rmm::mr::set_current_device_resource(originalUpstream_);
  }
  instance_.reset();
  originalUpstream_ = nullptr;
  LOG(INFO) << "GpuMemoryTracker shut down";
}

void GpuMemoryTracker::setCurrentTask(int64_t taskId) {
  auto* inst = instance_.get();
  if (!inst) return;
  const auto context = std::string("spark-task:") + std::to_string(taskId);
  auto& metrics = inst->getOrCreateContextMetrics(context);
  tlTaskMetrics() = &metrics;
  tlTaskId() = taskId;
  tlContext() = context;
}

void GpuMemoryTracker::clearCurrentTask() {
  tlTaskMetrics() = nullptr;
  tlTaskId() = -1;
  tlContext().clear();
}

void GpuMemoryTracker::setCurrentContext(const std::string& context) {
  auto* inst = instance_.get();
  if (!inst) return;
  auto& metrics = inst->getOrCreateContextMetrics(contextOrUnattributed(context));
  tlTaskMetrics() = &metrics;
  tlTaskId() = -1;
  tlContext() = contextOrUnattributed(context);
}

void GpuMemoryTracker::clearCurrentContext() {
  clearCurrentTask();
}

void GpuMemoryTracker::setCurrentOperatorContext(const std::string& context) {
  tlOperatorContext() = context;
}

void GpuMemoryTracker::clearCurrentOperatorContext() {
  tlOperatorContext().clear();
}

GpuMemoryTracker* GpuMemoryTracker::instance() {
  return instance_.get();
}

void GpuMemoryTracker::setMaxConcurrentGpuTasks(int) {
  // Concurrency is managed by GpuLock, not by the memory tracker.
}

bool GpuMemoryTracker::diagnosticsEnabled() {
  auto* inst = instance_.get();
  return inst != nullptr && inst->diagnosticsEnabled_;
}

void GpuMemoryTracker::dumpDiagnosticsToLog(const std::string& prefix) {
  auto* inst = instance_.get();
  if (inst == nullptr || !inst->diagnosticsEnabled_) {
    return;
  }
  inst->dumpDiagnostics(prefix);
}

GpuMemoryTracker::ScopedContext::ScopedContext(const std::string& context)
    : previousTaskMetrics_(tlTaskMetrics()),
      previousTaskId_(tlTaskId()),
      previousContext_(tlContext()) {
  if (diagnosticsEnabled()) {
    setCurrentContext(context);
  }
}

GpuMemoryTracker::ScopedContext::~ScopedContext() {
  tlTaskMetrics() = previousTaskMetrics_;
  tlTaskId() = previousTaskId_;
  tlContext() = previousContext_;
}

GpuMemoryTracker::ScopedOperatorContext::ScopedOperatorContext(
    const std::string& context)
    : previousContext_(tlOperatorContext()) {
  if (diagnosticsEnabled()) {
    setCurrentOperatorContext(context);
  }
}

GpuMemoryTracker::ScopedOperatorContext::~ScopedOperatorContext() {
  tlOperatorContext() = previousContext_;
}

// ── Instance API ─────────────────────────────────────────────────────────────

int64_t GpuMemoryTracker::getMaxTaskMemory(int64_t taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it =
      taskMetrics_.find(std::string("spark-task:") + std::to_string(taskId));
  if (it == taskMetrics_.end()) return 0;
  return it->second->peak.load(std::memory_order_relaxed);
}

int64_t GpuMemoryTracker::clearTaskMemory(int64_t taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it =
      taskMetrics_.find(std::string("spark-task:") + std::to_string(taskId));
  if (it == taskMetrics_.end()) return 0;
  auto peak = it->second->peak.load(std::memory_order_relaxed);
  if (!diagnosticsEnabled_) {
    taskMetrics_.erase(it);
  }
  return peak;
}

int64_t GpuMemoryTracker::getTotalAllocated() const {
  return totalAllocated_.load(std::memory_order_relaxed);
}

std::vector<GpuMemoryTracker::TaskSnapshot> GpuMemoryTracker::getTaskSnapshots() {
  std::vector<TaskSnapshot> snapshots;
  std::lock_guard<std::mutex> lock(mutex_);
  snapshots.reserve(taskMetrics_.size());
  for (const auto& entry : taskMetrics_) {
    snapshots.push_back(TaskSnapshot{
        entry.first,
        entry.second->current.load(std::memory_order_relaxed),
        entry.second->peak.load(std::memory_order_relaxed)});
  }
  std::sort(
      snapshots.begin(),
      snapshots.end(),
      [](const auto& left, const auto& right) {
        if (left.current != right.current) {
          return left.current > right.current;
        }
        return left.peak > right.peak;
      });
  return snapshots;
}

std::vector<GpuMemoryTracker::LargeAllocationEvent>
GpuMemoryTracker::getLargeAllocationEvents(std::size_t limit) {
  std::vector<LargeAllocationEvent> events;
  {
    std::lock_guard<std::mutex> lock(diagnosticsMutex_);
    events = largeAllocationEvents_;
  }
  std::sort(events.begin(), events.end(), [](const auto& left, const auto& right) {
    if (left.bytes != right.bytes) {
      return left.bytes > right.bytes;
    }
    return left.sequence > right.sequence;
  });
  if (limit > 0 && events.size() > limit) {
    events.resize(limit);
  }
  return events;
}

// ── device_memory_resource overrides ─────────────────────────────────────────

void* GpuMemoryTracker::do_allocate(
    std::size_t bytes,
    rmm::cuda_stream_view stream) {
  const bool diagnostics = shouldRecordDiagnostics();
  const bool recordLarge = shouldRecordLargeAllocation(bytes);
  std::size_t freeBytesBefore = 0;
  std::size_t totalBytesBefore = 0;
  if (recordLarge) {
    cudaMemGetInfo(&freeBytesBefore, &totalBytesBefore);
  }

  void* ptr = nullptr;
  try {
    ptr = upstream_->allocate(stream, bytes);
  } catch (const std::exception& e) {
    if (retryTrimOnOom_ && bytes >= largeAllocationThresholdBytes_ &&
        trimRmmUpstreamCudaMemPool(
            std::string("GpuMemoryTracker allocation retry bytes=") +
                std::to_string(bytes),
            upstream_)) {
      try {
        ptr = upstream_->allocate(stream, bytes);
        LOG(WARNING) << "GpuMemoryTracker allocation retry succeeded bytes="
                     << bytes;
      } catch (const std::exception& retryError) {
        if (recordLarge) {
          try {
            recordLargeAllocationEvent(
                bytes, false, freeBytesBefore, 0, retryError.what());
          } catch (...) {
            LOG(ERROR) << "GpuMemoryTracker failed to record failed allocation";
          }
        }
        throw;
      }
    } else {
      if (recordLarge) {
        try {
          recordLargeAllocationEvent(bytes, false, freeBytesBefore, 0, e.what());
        } catch (...) {
          LOG(ERROR) << "GpuMemoryTracker failed to record failed allocation";
        }
      }
      throw;
    }
  } catch (...) {
    if (retryTrimOnOom_ && bytes >= largeAllocationThresholdBytes_ &&
        trimRmmUpstreamCudaMemPool(
            std::string("GpuMemoryTracker allocation retry bytes=") +
                std::to_string(bytes),
            upstream_)) {
      try {
        ptr = upstream_->allocate(stream, bytes);
        LOG(WARNING) << "GpuMemoryTracker allocation retry succeeded bytes="
                     << bytes;
      } catch (...) {
        if (recordLarge) {
          try {
            recordLargeAllocationEvent(
                bytes, false, freeBytesBefore, 0, "unknown exception");
          } catch (...) {
            LOG(ERROR) << "GpuMemoryTracker failed to record failed allocation";
          }
        }
        throw;
      }
    } else {
      if (recordLarge) {
        try {
          recordLargeAllocationEvent(
              bytes, false, freeBytesBefore, 0, "unknown exception");
        } catch (...) {
          LOG(ERROR) << "GpuMemoryTracker failed to record failed allocation";
        }
      }
      throw;
    }
  }

  totalAllocated_.fetch_add(
      static_cast<int64_t>(bytes), std::memory_order_relaxed);
  if (diagnostics) {
    try {
      recordActiveAllocation(ptr, bytes);
    } catch (...) {
      LOG(ERROR) << "GpuMemoryTracker failed to record diagnostic allocation";
    }
  } else if (auto* tm = tlTaskMetrics()) {
    tm->recordAlloc(static_cast<int64_t>(bytes));
  }

  if (recordLarge) {
    std::size_t freeBytesAfter = 0;
    std::size_t totalBytesAfter = 0;
    cudaMemGetInfo(&freeBytesAfter, &totalBytesAfter);
    try {
      recordLargeAllocationEvent(
          bytes, true, freeBytesBefore, freeBytesAfter, nullptr);
    } catch (...) {
      LOG(ERROR) << "GpuMemoryTracker failed to record large allocation";
    }
  }
  return ptr;
}

void GpuMemoryTracker::do_deallocate(
    void* ptr,
    std::size_t bytes,
    rmm::cuda_stream_view stream) noexcept {
  upstream_->deallocate(stream, ptr, bytes);
  totalAllocated_.fetch_sub(
      static_cast<int64_t>(bytes), std::memory_order_relaxed);
  if (shouldRecordDiagnostics()) {
    try {
      recordActiveDeallocation(ptr, bytes);
    } catch (...) {
      LOG(ERROR) << "GpuMemoryTracker failed to record diagnostic deallocation";
    }
  } else if (auto* tm = tlTaskMetrics()) {
    tm->recordDealloc(static_cast<int64_t>(bytes));
  }
}

bool GpuMemoryTracker::do_is_equal(
    device_memory_resource const& other) const noexcept {
  if (auto* o = dynamic_cast<const GpuMemoryTracker*>(&other)) {
    return upstream_->is_equal(*o->upstream_);
  }
  return false;
}

// ── Diagnostics helpers ──────────────────────────────────────────────────────

bool GpuMemoryTracker::shouldRecordDiagnostics() const {
  return diagnosticsEnabled_;
}

bool GpuMemoryTracker::shouldRecordLargeAllocation(std::size_t bytes) const {
  return diagnosticsEnabled_ && largeAllocationEventCapacity_ > 0 &&
      bytes >= largeAllocationThresholdBytes_;
}

void GpuMemoryTracker::recordActiveAllocation(void* ptr, std::size_t bytes) {
  const auto context = contextOrUnattributed(tlContext());
  auto& metrics = getOrCreateContextMetrics(context);
  metrics.recordAlloc(static_cast<int64_t>(bytes));

  std::lock_guard<std::mutex> lock(diagnosticsMutex_);
  activeAllocations_[ptr] = ActiveAllocation{bytes, context, captureStackTrace()};
}

void GpuMemoryTracker::recordActiveDeallocation(void* ptr, std::size_t bytes) {
  std::string context;
  std::size_t actualBytes = bytes;
  {
    std::lock_guard<std::mutex> lock(diagnosticsMutex_);
    auto it = activeAllocations_.find(ptr);
    if (it != activeAllocations_.end()) {
      context = it->second.context;
      actualBytes = it->second.bytes;
      activeAllocations_.erase(it);
    }
  }

  if (context.empty()) {
    context = contextOrUnattributed(tlContext());
  }
  auto& metrics = getOrCreateContextMetrics(context);
  metrics.recordDealloc(static_cast<int64_t>(actualBytes));
}

void GpuMemoryTracker::recordLargeAllocationEvent(
    std::size_t bytes,
    bool success,
    std::size_t freeBytesBefore,
    std::size_t freeBytesAfter,
    const char* error) {
  if (largeAllocationEventCapacity_ == 0) {
    return;
  }

  LargeAllocationEvent event{
      nextEventSequence_.fetch_add(1, std::memory_order_relaxed),
      success,
      bytes,
      tlTaskId(),
      contextOrUnattributed(tlContext()),
      tlOperatorContext(),
      threadIdString(),
      freeBytesBefore,
      freeBytesAfter,
      error == nullptr ? std::string() : std::string(error),
      captureStackTrace()};

  std::lock_guard<std::mutex> lock(diagnosticsMutex_);
  if (largeAllocationEvents_.size() >= largeAllocationEventCapacity_) {
    largeAllocationEvents_.erase(largeAllocationEvents_.begin());
  }
  largeAllocationEvents_.push_back(std::move(event));
}

void GpuMemoryTracker::dumpDiagnostics(const std::string& prefix) {
  std::size_t freeMem = 0;
  std::size_t totalMem = 0;
  auto err = cudaMemGetInfo(&freeMem, &totalMem);
  if (err == cudaSuccess) {
    const auto usedMem = totalMem - freeMem;
    LOG(ERROR) << prefix << " cudaMemory freeBytes=" << freeMem
               << " usedBytes=" << usedMem
               << " totalBytes=" << totalMem
               << " freeMiB=" << (freeMem / 1048576.0)
               << " usedMiB=" << (usedMem / 1048576.0)
               << " totalMiB=" << (totalMem / 1048576.0);
  } else {
    LOG(ERROR) << prefix << " cudaMemory cudaMemGetInfoFailed="
               << cudaGetErrorString(err);
  }
  logDefaultCudaMemPoolStats(prefix);
  logRmmUpstreamCudaMemPoolStats(prefix, upstream_);

  std::size_t activeAllocations = 0;
  struct ActiveStackSummary {
    std::size_t bytes{0};
    std::size_t count{0};
  };
  std::vector<std::pair<std::string, ActiveStackSummary>> activeStackSummaries;
  {
    std::lock_guard<std::mutex> lock(diagnosticsMutex_);
    activeAllocations = activeAllocations_.size();
    std::unordered_map<std::string, ActiveStackSummary> byStack;
    for (const auto& entry : activeAllocations_) {
      auto& summary = byStack[entry.second.stackTrace];
      summary.bytes += entry.second.bytes;
      ++summary.count;
    }
    activeStackSummaries.reserve(byStack.size());
    for (auto& entry : byStack) {
      activeStackSummaries.emplace_back(std::move(entry));
    }
  }

  LOG(ERROR) << prefix << " rmmResource currentType=" << currentResourceType()
             << " upstreamType="
             << (upstream_ == nullptr ? "null" : typeid(*upstream_).name())
             << " poolReservedBytes=unavailable"
             << " poolUsedBytes=unavailable";
  LOG(ERROR) << prefix << " tracker totalActiveBytes=" << getTotalAllocated()
             << " totalActiveMiB=" << (getTotalAllocated() / 1048576.0)
             << " activeAllocationRecords=" << activeAllocations
             << " largeAllocationThresholdBytes="
             << largeAllocationThresholdBytes_;

  auto snapshots = getTaskSnapshots();
  bool sawSnapshot = false;
  for (const auto& snapshot : snapshots) {
    if (snapshot.current == 0 && snapshot.peak == 0) {
      continue;
    }
    sawSnapshot = true;
    LOG(ERROR) << prefix << " taskMemory context=" << snapshot.context
               << " currentBytes=" << snapshot.current
               << " peakBytes=" << snapshot.peak
               << " currentMiB=" << (snapshot.current / 1048576.0)
               << " peakMiB=" << (snapshot.peak / 1048576.0);
  }
  if (!sawSnapshot) {
    LOG(ERROR) << prefix << " taskMemory none";
  }

  std::sort(
      activeStackSummaries.begin(),
      activeStackSummaries.end(),
      [](const auto& left, const auto& right) {
        if (left.second.bytes != right.second.bytes) {
          return left.second.bytes > right.second.bytes;
        }
        return left.second.count > right.second.count;
      });
  if (activeStackSummaries.empty()) {
    LOG(ERROR) << prefix << " activeAllocationStack none";
  } else {
    const auto limit =
        std::min(activeStackSummaries.size(), largeAllocationTopN_);
    for (std::size_t i = 0; i < limit; ++i) {
      const auto& summary = activeStackSummaries[i].second;
      LOG(ERROR) << prefix << " activeAllocationStack rank=" << i
                 << " bytes=" << summary.bytes
                 << " MiB=" << (summary.bytes / 1048576.0)
                 << " count=" << summary.count
                 << " stack=" << activeStackSummaries[i].first;
    }
  }

  auto events = getLargeAllocationEvents(largeAllocationTopN_);
  if (events.empty()) {
    LOG(ERROR) << prefix << " largeAllocationEvents none";
    return;
  }
  for (const auto& event : events) {
    LOG(ERROR) << prefix << " largeAllocationEvent seq=" << event.sequence
               << " success=" << event.success
               << " bytes=" << event.bytes
               << " MiB=" << (event.bytes / 1048576.0)
               << " taskId=" << event.taskId
               << " context=" << event.context
               << " operatorContext="
               << (event.operatorContext.empty() ? "unknown" : event.operatorContext)
               << " threadId=" << event.threadId
               << " freeBeforeBytes=" << event.freeBytesBefore
               << " freeAfterBytes=" << event.freeBytesAfter
               << " error=" << (event.error.empty() ? "none" : event.error)
               << " stack=" << event.stackTrace;
  }
}

std::string GpuMemoryTracker::currentResourceType() const {
  auto* current = rmm::mr::get_current_device_resource();
  if (current == nullptr) {
    return "null";
  }
  return typeid(*current).name();
}

// ── Helpers ──────────────────────────────────────────────────────────────────

GpuMemoryTracker::TaskMetrics&
GpuMemoryTracker::getOrCreateTaskMetrics(int64_t taskId) {
  return getOrCreateContextMetrics(std::string("spark-task:") + std::to_string(taskId));
}

GpuMemoryTracker::TaskMetrics& GpuMemoryTracker::getOrCreateContextMetrics(
    const std::string& context) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& ptr = taskMetrics_[contextOrUnattributed(context)];
  if (!ptr) {
    ptr = std::make_unique<TaskMetrics>();
  }
  return *ptr;
}

GpuMemoryTracker::TaskMetrics*& GpuMemoryTracker::tlTaskMetrics() {
  thread_local TaskMetrics* tm = nullptr;
  return tm;
}

int64_t& GpuMemoryTracker::tlTaskId() {
  thread_local int64_t id = -1;
  return id;
}

std::string& GpuMemoryTracker::tlContext() {
  thread_local std::string context;
  return context;
}

std::string& GpuMemoryTracker::tlOperatorContext() {
  thread_local std::string context;
  return context;
}
