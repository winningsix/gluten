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
#include "MemoryManager.h"
#include "utils/Registry.h"
#include <glog/logging.h>
#include <mutex>
#include <unordered_map>

namespace gluten {

namespace {
Registry<MemoryManager::Factory>& memoryManagerFactories() {
  static Registry<MemoryManager::Factory> registry;
  return registry;
}
Registry<MemoryManager::Releaser>& memoryManagerReleasers() {
  static Registry<MemoryManager::Releaser> registry;
  return registry;
}

struct DeferredMemoryManagerReleaseState {
  uint32_t retainCount{0};
  bool releaseRequested{false};
  std::string kind;
  std::string lastReason;
};

std::mutex& deferredMemoryManagerReleaseMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<MemoryManager*, DeferredMemoryManagerReleaseState>& deferredMemoryManagerReleases() {
  static auto* releases = new std::unordered_map<MemoryManager*, DeferredMemoryManagerReleaseState>();
  return *releases;
}

void releaseDirect(MemoryManager* memoryManager) {
  if (memoryManager == nullptr) {
    return;
  }
  const std::string kind = memoryManager->kind();
  auto& releaser = memoryManagerReleasers().get(kind);
  releaser(memoryManager);
}
} // namespace

void MemoryManager::registerFactory(const std::string& kind, MemoryManager::Factory factory, Releaser releaser) {
  memoryManagerFactories().registerObj(kind, std::move(factory));
  memoryManagerReleasers().registerObj(kind, std::move(releaser));
}

MemoryManager* MemoryManager::create(const std::string& kind, std::unique_ptr<AllocationListener> listener) {
  auto& factory = memoryManagerFactories().get(kind);
  return factory(kind, std::move(listener));
}

void MemoryManager::release(MemoryManager* memoryManager) {
  releaseDirect(memoryManager);
}

void MemoryManager::releaseOrDefer(MemoryManager* memoryManager) {
  if (memoryManager == nullptr) {
    return;
  }

  bool shouldRelease = false;
  {
    std::lock_guard<std::mutex> l(deferredMemoryManagerReleaseMutex());
    auto& releases = deferredMemoryManagerReleases();
    auto it = releases.find(memoryManager);
    if (it == releases.end() || it->second.retainCount == 0) {
      if (it != releases.end()) {
        releases.erase(it);
      }
      shouldRelease = true;
    } else {
      it->second.releaseRequested = true;
      LOG(WARNING) << "[MEM-DEFER] deferring native memory manager release"
                   << " manager=" << static_cast<const void*>(memoryManager)
                   << " kind=" << it->second.kind
                   << " retainCount=" << it->second.retainCount
                   << " reason=" << it->second.lastReason;
    }
  }

  if (shouldRelease) {
    releaseDirect(memoryManager);
  }
}

void MemoryManager::retainForAsyncTask(MemoryManager* memoryManager, const std::string& reason) {
  if (memoryManager == nullptr) {
    return;
  }

  uint32_t retainCount = 0;
  const auto kind = memoryManager->kind();
  {
    std::lock_guard<std::mutex> l(deferredMemoryManagerReleaseMutex());
    auto& state = deferredMemoryManagerReleases()[memoryManager];
    state.kind = kind;
    state.lastReason = reason;
    retainCount = ++state.retainCount;
  }
  LOG(WARNING) << "[MEM-DEFER] retained native memory manager for async task"
               << " manager=" << static_cast<const void*>(memoryManager)
               << " kind=" << kind << " retainCount=" << retainCount
               << " reason=" << reason;
}

void MemoryManager::releaseAsyncTaskRetain(MemoryManager* memoryManager, const std::string& reason) {
  if (memoryManager == nullptr) {
    return;
  }

  bool shouldRelease = false;
  uint32_t retainCount = 0;
  std::string kind;
  {
    std::lock_guard<std::mutex> l(deferredMemoryManagerReleaseMutex());
    auto& releases = deferredMemoryManagerReleases();
    auto it = releases.find(memoryManager);
    if (it == releases.end() || it->second.retainCount == 0) {
      LOG(WARNING) << "[MEM-DEFER] async memory manager retain already released"
                   << " manager=" << static_cast<const void*>(memoryManager)
                   << " reason=" << reason;
      return;
    }

    kind = it->second.kind;
    retainCount = --it->second.retainCount;
    if (retainCount == 0 && it->second.releaseRequested) {
      shouldRelease = true;
      releases.erase(it);
    }
  }

  LOG(WARNING) << "[MEM-DEFER] released async native memory manager retain"
               << " manager=" << static_cast<const void*>(memoryManager)
               << " kind=" << kind << " retainCount=" << retainCount
               << " reason=" << reason
               << " pendingRelease=" << shouldRelease;

  if (shouldRelease) {
    releaseDirect(memoryManager);
  }
}

} // namespace gluten
