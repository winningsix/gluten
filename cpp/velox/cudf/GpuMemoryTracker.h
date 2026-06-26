#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/// Compatibility shim for the old native GPU memory tracker API.
///
/// RMM 26.06 removed the legacy device_memory_resource base class used by the
/// previous tracker. Keep the JNI-facing API available while avoiding global
/// RMM resource interception. Detailed allocation attribution can be rebuilt
/// later on top of RMM's resource_ref/tracking_resource_adaptor APIs.
class GpuMemoryTracker final {
 public:
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

  int64_t getMaxTaskMemory(int64_t taskId);
  int64_t clearTaskMemory(int64_t taskId);
  int64_t getTotalAllocated() const;
  std::vector<TaskSnapshot> getTaskSnapshots();
  std::vector<LargeAllocationEvent> getLargeAllocationEvents(std::size_t limit);

 private:
  static std::unique_ptr<GpuMemoryTracker> instance_;
};
