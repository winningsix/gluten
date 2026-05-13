#pragma once

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

/// Lightweight RMM device_memory_resource wrapper that tracks GPU memory
/// allocations. Default hot-path behavior preserves the original low-overhead
/// task counters; opt-in diagnostics add per-context active bytes and a bounded
/// large-allocation event buffer.
class GpuMemoryTracker final : public rmm::mr::device_memory_resource {
 public:
  struct TaskMetrics {
    std::atomic<int64_t> current{0};
    std::atomic<int64_t> peak{0};

    void recordAlloc(int64_t bytes) {
      auto cur = current.fetch_add(bytes, std::memory_order_relaxed) + bytes;
      auto pk = peak.load(std::memory_order_relaxed);
      while (cur > pk &&
             !peak.compare_exchange_weak(
                 pk, cur, std::memory_order_relaxed)) {
      }
    }

    void recordDealloc(int64_t bytes) {
      current.fetch_sub(bytes, std::memory_order_relaxed);
    }
  };

  struct TaskSnapshot {
    std::string context;
    int64_t current;
    int64_t peak;
  };

  struct LargeAllocationEvent {
    uint64_t sequence;
    bool success;
    std::size_t bytes;
    int64_t taskId;
    std::string context;
    std::string operatorContext;
    std::string threadId;
    std::size_t freeBytesBefore;
    std::size_t freeBytesAfter;
    std::string error;
    std::string stackTrace;
  };

  // ── Static API (called from JNI / MPP diagnostics) ─────────────────────

  static void initialize();
  static void shutdown();
  static void setCurrentTask(int64_t taskId);
  static void clearCurrentTask();
  static void setCurrentContext(const std::string& context);
  static void clearCurrentContext();
  static void setCurrentOperatorContext(const std::string& context);
  static void clearCurrentOperatorContext();
  static GpuMemoryTracker* instance();
  static void setMaxConcurrentGpuTasks(int);
  static bool diagnosticsEnabled();
  static void dumpDiagnosticsToLog(const std::string& prefix);

  class ScopedContext {
   public:
    explicit ScopedContext(const std::string& context);
    ~ScopedContext();
    ScopedContext(const ScopedContext&) = delete;
    ScopedContext& operator=(const ScopedContext&) = delete;

   private:
    TaskMetrics* previousTaskMetrics_{nullptr};
    int64_t previousTaskId_{-1};
    std::string previousContext_;
  };

  class ScopedOperatorContext {
   public:
    explicit ScopedOperatorContext(const std::string& context);
    ~ScopedOperatorContext();
    ScopedOperatorContext(const ScopedOperatorContext&) = delete;
    ScopedOperatorContext& operator=(const ScopedOperatorContext&) = delete;

   private:
    std::string previousContext_;
  };

  // ── Instance API ──────────────────────────────────────────────────────

  int64_t getMaxTaskMemory(int64_t taskId);
  int64_t clearTaskMemory(int64_t taskId);
  int64_t getTotalAllocated() const;
  std::vector<TaskSnapshot> getTaskSnapshots();
  std::vector<LargeAllocationEvent> getLargeAllocationEvents(std::size_t limit);

 private:
  explicit GpuMemoryTracker(rmm::mr::device_memory_resource* upstream);

  void* do_allocate(std::size_t bytes,
                    rmm::cuda_stream_view stream) override;

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream) noexcept override;

  [[nodiscard]] bool do_is_equal(
      device_memory_resource const& other) const noexcept override;

  TaskMetrics& getOrCreateTaskMetrics(int64_t taskId);
  TaskMetrics& getOrCreateContextMetrics(const std::string& context);
  bool shouldRecordDiagnostics() const;
  bool shouldRecordLargeAllocation(std::size_t bytes) const;
  void recordActiveAllocation(void* ptr, std::size_t bytes);
  void recordActiveDeallocation(void* ptr, std::size_t bytes);
  void recordLargeAllocationEvent(
      std::size_t bytes,
      bool success,
      std::size_t freeBytesBefore,
      std::size_t freeBytesAfter,
      const char* error);
  void dumpDiagnostics(const std::string& prefix);
  std::string currentResourceType() const;

  static TaskMetrics*& tlTaskMetrics();
  static int64_t& tlTaskId();
  static std::string& tlContext();
  static std::string& tlOperatorContext();

  rmm::mr::device_memory_resource* upstream_;
  std::atomic<int64_t> totalAllocated_{0};
  std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<TaskMetrics>> taskMetrics_;

  struct ActiveAllocation {
    std::size_t bytes;
    std::string context;
    std::string stackTrace;
  };

  std::mutex diagnosticsMutex_;
  std::unordered_map<void*, ActiveAllocation> activeAllocations_;
  std::vector<LargeAllocationEvent> largeAllocationEvents_;
  std::atomic<uint64_t> nextEventSequence_{0};
  bool diagnosticsEnabled_{false};
  bool retryTrimOnOom_{false};
  std::size_t largeAllocationThresholdBytes_{256ULL << 20};
  std::size_t largeAllocationEventCapacity_{128};
  std::size_t largeAllocationTopN_{20};

  static std::unique_ptr<GpuMemoryTracker> instance_;
  static rmm::mr::device_memory_resource* originalUpstream_;
};
