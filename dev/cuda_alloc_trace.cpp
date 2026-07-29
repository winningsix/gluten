// Diagnostic-only CUDA allocation tracer.
//
// Build:
//   g++ -std=c++17 -O2 -g -shared -fPIC -o libcuda_alloc_trace.so
//   cuda_alloc_trace.cpp -ldl -pthread
// Enable only for a focused executor run through LD_PRELOAD.

#include <dlfcn.h>
#include <execinfo.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using CudaError = int;
using CudaStream = void *;
using CuDevicePtr = std::uint64_t;
using CuMemoryPool = void *;

std::mutex allocationMutex;
struct Allocation {
  std::size_t bytes;
  uint32_t context;
};

struct ContextStats {
  std::string name;
  uint64_t currentBytes{0};
  uint64_t peakBytes{0};
  uint64_t currentAllocations{0};
};

std::unordered_map<void *, Allocation> allocations;
std::unordered_map<std::string, uint32_t> contextIds;
std::vector<ContextStats> contexts{{"unattributed"}};
std::atomic<uint64_t> currentBytes{0};
std::atomic<uint64_t> peakBytes{0};
std::atomic<uint64_t> totalBytes{0};
std::atomic<uint64_t> allocationCount{0};
std::atomic<uint64_t> allocationFailureCount{0};
std::atomic<uint64_t> freeCount{0};
std::atomic<uint64_t> freeFailureCount{0};
std::atomic<uint64_t> reportedPeakGiB{0};
thread_local bool insideHook = false;
thread_local std::vector<uint32_t> contextStack{0};

void printTopContexts() {
  std::vector<ContextStats> snapshot;
  {
    std::lock_guard<std::mutex> lock(allocationMutex);
    snapshot = contexts;
  }
  std::sort(snapshot.begin(), snapshot.end(),
            [](const auto &left, const auto &right) {
              return left.currentBytes > right.currentBytes;
            });
  const auto count = std::min<std::size_t>(snapshot.size(), 12);
  for (std::size_t i = 0; i < count; ++i) {
    if (snapshot[i].currentBytes == 0) {
      break;
    }
    dprintf(
        STDERR_FILENO,
        "CUDA_ALLOC_TRACE_CONTEXT rank=%zu currentBytes=%llu peakBytes=%llu "
        "currentAllocations=%llu name=%s\n",
        i + 1, static_cast<unsigned long long>(snapshot[i].currentBytes),
        static_cast<unsigned long long>(snapshot[i].peakBytes),
        static_cast<unsigned long long>(snapshot[i].currentAllocations),
        snapshot[i].name.c_str());
  }
}

template <typename Function> Function loadNext(const char *name) {
  auto *symbol = dlsym(RTLD_NEXT, name);
  if (symbol == nullptr) {
    // Spark loads libgluten with a classloader-local dlopen scope. Its
    // libcudart dependency is therefore not necessarily visible through
    // RTLD_NEXT from a process-wide preload. Reopen the already-loaded SONAME
    // globally so the real runtime entry points can be resolved.
    auto *cudart = dlopen("libcudart.so.13", RTLD_NOW | RTLD_GLOBAL);
    if (cudart != nullptr) {
      symbol = dlsym(cudart, name);
    }
  }
  if (symbol == nullptr) {
    dprintf(STDERR_FILENO, "CUDA_ALLOC_TRACE missingSymbol=%s\n", name);
  }
  return reinterpret_cast<Function>(symbol);
}

void printStack() {
  void *frames[48];
  const int count = backtrace(frames, 48);
  backtrace_symbols_fd(frames, count, STDERR_FILENO);
}

void recordAllocation(const char *api, void *pointer, std::size_t bytes,
                      CudaError status) {
  if (status != 0) {
    const auto failures = allocationFailureCount.fetch_add(1) + 1;
    if (failures <= 32) {
      uint32_t context = 0;
      std::string contextName{"unattributed"};
      {
        std::lock_guard<std::mutex> lock(allocationMutex);
        context = contextStack.empty() ? 0 : contextStack.back();
        if (context < contexts.size()) {
          contextName = contexts[context].name;
        }
      }
      dprintf(
          STDERR_FILENO,
          "CUDA_ALLOC_TRACE event=allocationFailure api=%s pid=%d bytes=%zu "
          "status=%d currentBytes=%llu peakBytes=%llu failures=%llu "
          "triggerContext=%s\n",
          api, static_cast<int>(getpid()), bytes, status,
          static_cast<unsigned long long>(currentBytes.load()),
          static_cast<unsigned long long>(peakBytes.load()),
          static_cast<unsigned long long>(failures), contextName.c_str());
      printTopContexts();
    }
    return;
  }
  if (pointer == nullptr || bytes == 0) {
    return;
  }

  bool inserted = false;
  const auto context = contextStack.empty() ? 0 : contextStack.back();
  {
    std::lock_guard<std::mutex> lock(allocationMutex);
    inserted = allocations.emplace(pointer, Allocation{bytes, context}).second;
    if (inserted) {
      auto &stats = contexts[context];
      stats.currentBytes += bytes;
      stats.peakBytes = std::max(stats.peakBytes, stats.currentBytes);
      ++stats.currentAllocations;
    }
  }
  // A runtime API may internally call another interposed allocation API.
  // Count the pointer once while still allowing either layer to provide a
  // useful stack.
  if (!inserted) {
    return;
  }

  const auto current = currentBytes.fetch_add(bytes) + bytes;
  const auto total = totalBytes.fetch_add(bytes) + bytes;
  const auto count = allocationCount.fetch_add(1) + 1;
  auto observedPeak = peakBytes.load(std::memory_order_relaxed);
  while (current > observedPeak &&
         !peakBytes.compare_exchange_weak(observedPeak, current)) {
  }
  const auto peak = peakBytes.load(std::memory_order_relaxed);
  const auto peakGiB = peak >> 30;
  auto reported = reportedPeakGiB.load(std::memory_order_relaxed);
  bool crossedGiB = false;
  while (peakGiB > reported) {
    if (reportedPeakGiB.compare_exchange_weak(reported, peakGiB)) {
      crossedGiB = true;
      break;
    }
  }

  if (crossedGiB || bytes >= (64ULL << 20)) {
    dprintf(STDERR_FILENO,
            "CUDA_ALLOC_TRACE event=allocate api=%s pid=%d ptr=%p bytes=%zu "
            "currentBytes=%llu peakBytes=%llu totalBytes=%llu allocations=%llu "
            "frees=%llu freeFailures=%llu\n",
            api, static_cast<int>(getpid()), pointer, bytes,
            static_cast<unsigned long long>(current),
            static_cast<unsigned long long>(peak),
            static_cast<unsigned long long>(total),
            static_cast<unsigned long long>(count),
            static_cast<unsigned long long>(freeCount.load()),
            static_cast<unsigned long long>(freeFailureCount.load()));
    printStack();
    printTopContexts();
  }
}

void recordFree(const char *api, void *pointer, CudaError status) {
  if (pointer == nullptr) {
    return;
  }
  if (status != 0) {
    const auto failures = freeFailureCount.fetch_add(1) + 1;
    if (failures <= 32) {
      dprintf(STDERR_FILENO,
              "CUDA_ALLOC_TRACE event=freeFailure api=%s pid=%d ptr=%p "
              "status=%d failures=%llu\n",
              api, static_cast<int>(getpid()), pointer, status,
              static_cast<unsigned long long>(failures));
      printStack();
    }
    return;
  }
  freeCount.fetch_add(1);
  std::size_t bytes = 0;
  {
    std::lock_guard<std::mutex> lock(allocationMutex);
    const auto it = allocations.find(pointer);
    if (it != allocations.end()) {
      bytes = it->second.bytes;
      auto &stats = contexts[it->second.context];
      stats.currentBytes -= bytes;
      --stats.currentAllocations;
      allocations.erase(it);
    }
  }
  if (bytes > 0) {
    currentBytes.fetch_sub(bytes);
  }
  (void)api;
}

} // namespace

__attribute__((constructor)) static void cudaAllocTraceLoaded() {
  dprintf(STDERR_FILENO, "CUDA_ALLOC_TRACE event=loaded pid=%d\n",
          static_cast<int>(getpid()));
}

extern "C" void cuda_alloc_trace_push_context(const char *name) {
  if (name == nullptr || name[0] == '\0') {
    contextStack.push_back(0);
    return;
  }
  std::lock_guard<std::mutex> lock(allocationMutex);
  auto [it, inserted] = contextIds.emplace(name, contexts.size());
  if (inserted) {
    contexts.push_back(ContextStats{std::string{name}});
  }
  contextStack.push_back(it->second);
}

extern "C" void cuda_alloc_trace_pop_context() {
  if (contextStack.size() > 1) {
    contextStack.pop_back();
  }
}

extern "C" CudaError cudaMalloc(void **pointer, std::size_t bytes) {
  using Function = CudaError (*)(void **, std::size_t);
  static auto real = loadNext<Function>("cudaMalloc");
  if (real == nullptr) {
    return 999;
  }
  if (insideHook) {
    return real(pointer, bytes);
  }
  insideHook = true;
  const auto status = real(pointer, bytes);
  recordAllocation("cudaMalloc", status == 0 ? *pointer : nullptr, bytes,
                   status);
  insideHook = false;
  return status;
}

extern "C" CudaError cudaMallocManaged(void **pointer, std::size_t bytes,
                                       unsigned int flags) {
  using Function = CudaError (*)(void **, std::size_t, unsigned int);
  static auto real = loadNext<Function>("cudaMallocManaged");
  if (real == nullptr) {
    return 999;
  }
  if (insideHook) {
    return real(pointer, bytes, flags);
  }
  insideHook = true;
  const auto status = real(pointer, bytes, flags);
  recordAllocation("cudaMallocManaged", status == 0 ? *pointer : nullptr, bytes,
                   status);
  insideHook = false;
  return status;
}

extern "C" CudaError cudaMallocAsync(void **pointer, std::size_t bytes,
                                     CudaStream stream) {
  using Function = CudaError (*)(void **, std::size_t, CudaStream);
  static auto real = loadNext<Function>("cudaMallocAsync");
  if (real == nullptr) {
    return 999;
  }
  if (insideHook) {
    return real(pointer, bytes, stream);
  }
  insideHook = true;
  const auto status = real(pointer, bytes, stream);
  recordAllocation("cudaMallocAsync", status == 0 ? *pointer : nullptr, bytes,
                   status);
  insideHook = false;
  return status;
}

extern "C" CudaError cudaFree(void *pointer) {
  using Function = CudaError (*)(void *);
  static auto real = loadNext<Function>("cudaFree");
  if (real == nullptr) {
    return 999;
  }
  if (insideHook) {
    return real(pointer);
  }
  insideHook = true;
  const auto status = real(pointer);
  recordFree("cudaFree", pointer, status);
  insideHook = false;
  return status;
}

extern "C" CudaError cudaFreeAsync(void *pointer, CudaStream stream) {
  using Function = CudaError (*)(void *, CudaStream);
  static auto real = loadNext<Function>("cudaFreeAsync");
  if (real == nullptr) {
    return 999;
  }
  if (insideHook) {
    return real(pointer, stream);
  }
  insideHook = true;
  const auto status = real(pointer, stream);
  recordFree("cudaFreeAsync", pointer, status);
  insideHook = false;
  return status;
}

extern "C" CudaError cuMemAllocAsync(CuDevicePtr *pointer, std::size_t bytes,
                                     CudaStream stream) {
  using Function = CudaError (*)(CuDevicePtr *, std::size_t, CudaStream);
  static auto real = loadNext<Function>("cuMemAllocAsync");
  if (real == nullptr) {
    return 999;
  }
  if (insideHook) {
    return real(pointer, bytes, stream);
  }
  insideHook = true;
  const auto status = real(pointer, bytes, stream);
  recordAllocation("cuMemAllocAsync",
                   status == 0 ? reinterpret_cast<void *>(*pointer) : nullptr,
                   bytes, status);
  insideHook = false;
  return status;
}

extern "C" CudaError cuMemAllocFromPoolAsync(CuDevicePtr *pointer,
                                             std::size_t bytes,
                                             CuMemoryPool pool,
                                             CudaStream stream) {
  using Function =
      CudaError (*)(CuDevicePtr *, std::size_t, CuMemoryPool, CudaStream);
  static auto real = loadNext<Function>("cuMemAllocFromPoolAsync");
  if (real == nullptr) {
    return 999;
  }
  if (insideHook) {
    return real(pointer, bytes, pool, stream);
  }
  insideHook = true;
  const auto status = real(pointer, bytes, pool, stream);
  recordAllocation("cuMemAllocFromPoolAsync",
                   status == 0 ? reinterpret_cast<void *>(*pointer) : nullptr,
                   bytes, status);
  insideHook = false;
  return status;
}

extern "C" CudaError cuMemFreeAsync(CuDevicePtr pointer, CudaStream stream) {
  using Function = CudaError (*)(CuDevicePtr, CudaStream);
  static auto real = loadNext<Function>("cuMemFreeAsync");
  if (real == nullptr) {
    return 999;
  }
  if (insideHook) {
    return real(pointer, stream);
  }
  insideHook = true;
  const auto status = real(pointer, stream);
  recordFree("cuMemFreeAsync", reinterpret_cast<void *>(pointer), status);
  insideHook = false;
  return status;
}

extern "C" CudaError cuMemAlloc_v2(CuDevicePtr *pointer, std::size_t bytes) {
  using Function = CudaError (*)(CuDevicePtr *, std::size_t);
  static auto real = loadNext<Function>("cuMemAlloc_v2");
  if (real == nullptr) {
    return 999;
  }
  if (insideHook) {
    return real(pointer, bytes);
  }
  insideHook = true;
  const auto status = real(pointer, bytes);
  recordAllocation("cuMemAlloc_v2",
                   status == 0 ? reinterpret_cast<void *>(*pointer) : nullptr,
                   bytes, status);
  insideHook = false;
  return status;
}

extern "C" CudaError cuMemFree_v2(CuDevicePtr pointer) {
  using Function = CudaError (*)(CuDevicePtr);
  static auto real = loadNext<Function>("cuMemFree_v2");
  if (real == nullptr) {
    return 999;
  }
  if (insideHook) {
    return real(pointer);
  }
  insideHook = true;
  const auto status = real(pointer);
  recordFree("cuMemFree_v2", reinterpret_cast<void *>(pointer), status);
  insideHook = false;
  return status;
}
