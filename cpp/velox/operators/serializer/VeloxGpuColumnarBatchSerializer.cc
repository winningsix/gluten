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

#include "VeloxGpuColumnarBatchSerializer.h"

#include <arrow/buffer.h>
#include <atomic>
#include <chrono>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/error.hpp>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <future>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <lz4.h>
#include <nvcomp/lz4.h>
#include <nvcomp/zstd.h>
#include <zstd.h>

#include "cudf/GpuLock.h"
#include "memory/ArrowMemory.h"
#include "memory/VeloxColumnarBatch.h"
#include "velox/common/memory/Memory.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/arrow/Bridge.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <glog/logging.h>

using namespace facebook::velox;

namespace gluten {
namespace {

constexpr uint64_t kCudfCacheMagic = 0x314548434143464EULL; // "NFCACHE1"
constexpr uint32_t kCudfCacheVersionRaw = 1;
constexpr uint32_t kCudfCacheVersionZstd = 2;
constexpr uint32_t kCudfCacheVersionLz4 = 3;
constexpr uint32_t kCudfCacheVersionChunkedZstd = 4;
constexpr uint32_t kCudfCacheVersionChunkedLz4 = 5;
constexpr uint32_t kCudfCacheVersionParquet = 6;
// nvCOMP recommends 64 KiB input chunks for peak throughput, but larger
// chunks can materially improve the compression ratio for disk-backed Spark
// caches. Keep the production default unchanged and allow benchmark/runtime
// tuning without rebuilding the native library.
constexpr size_t kDefaultNvcompZstdChunkBytes = 64ULL << 10;
constexpr size_t kMinNvcompZstdChunkBytes = 64ULL << 10;
constexpr size_t kMaxNvcompZstdChunkBytes = 16ULL << 20;
constexpr uint32_t kMaxCacheChunks = 65536;
constexpr uint32_t kCacheDecompressionWorkers = 12;

enum class CacheCodec { kZstd, kLz4 };

bool useAsyncParquetFileSink() {
  static const bool enabled = [] {
    const auto* value = std::getenv("GLUTEN_CACHE_ASYNC_FILE_WRITES");
    return value != nullptr && std::strcmp(value, "true") == 0;
  }();
  return enabled;
}

cudf::io::compression_type parquetCacheCompression() {
  static const auto compression = [] {
    const auto* value = std::getenv("GLUTEN_CACHE_CODEC");
    if (value == nullptr || *value == '\0' ||
        std::strcmp(value, "snappy") == 0) {
      return cudf::io::compression_type::SNAPPY;
    }
    if (std::strcmp(value, "zstd") == 0) {
      return cudf::io::compression_type::ZSTD;
    }
    if (std::strcmp(value, "none") == 0 ||
        std::strcmp(value, "uncompressed") == 0) {
      return cudf::io::compression_type::NONE;
    }
    VELOX_FAIL(
        "Unsupported GLUTEN_CACHE_CODEC for Parquet cache: {}", value);
  }();
  return compression;
}

const char* parquetCacheCompressionName() {
  switch (parquetCacheCompression()) {
    case cudf::io::compression_type::ZSTD:
      return "zstd";
    case cudf::io::compression_type::SNAPPY:
      return "snappy";
    case cudf::io::compression_type::NONE:
      return "none";
    default:
      return "unsupported";
  }
}

size_t asyncParquetFileSinkBytes() {
  static const size_t bytes = [] {
    constexpr size_t kDefaultBytes = 1ULL << 30;
    const auto* value = std::getenv("GLUTEN_CACHE_ASYNC_WRITE_BYTES");
    if (value == nullptr || *value == '\0') {
      return kDefaultBytes;
    }
    const auto parsed = std::strtoull(value, nullptr, 10);
    return parsed == 0 ? kDefaultBytes : static_cast<size_t>(parsed);
  }();
  return bytes;
}

// Repeated cudaHostAlloc/cudaFree calls can materially inflate the D2H path
// when libcudf emits many small compressed Parquet sink chunks.
// Keep a bounded, executor-process-wide pool so buffers survive the lifetime
// of an individual cache file and can be reused by the following file.  Using
// power-of-two size classes avoids retaining thousands of almost-identical
// allocations when libcudf varies compressed chunk sizes slightly.
class PinnedCacheBufferPool final
    : public std::enable_shared_from_this<PinnedCacheBufferPool> {
 public:
  struct Stats {
    uint64_t allocations;
    uint64_t reuses;
    uint64_t allocationBytes;
    size_t cachedBytes;
  };

  explicit PinnedCacheBufferPool(size_t maxCachedBytes)
      : maxCachedBytes_(maxCachedBytes) {}

  ~PinnedCacheBufferPool() {
    for (auto& [capacity, buffers] : freeBuffers_) {
      for (auto* buffer : buffers) {
        cudaFreeHost(buffer);
      }
    }
  }

  std::shared_ptr<uint8_t> acquire(size_t size) {
    const auto capacity = sizeClass(size);
    uint8_t* data = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = freeBuffers_.lower_bound(capacity);
      if (it != freeBuffers_.end()) {
        const auto retainedCapacity = it->first;
        data = it->second.back();
        it->second.pop_back();
        if (it->second.empty()) {
          freeBuffers_.erase(it);
        }
        cachedBytes_ -= retainedCapacity;
        reuses_.fetch_add(1, std::memory_order_relaxed);
        return wrap(data, retainedCapacity);
      }
    }

    CUDF_CUDA_TRY(cudaHostAlloc(
        reinterpret_cast<void**>(&data), capacity, cudaHostAllocPortable));
    allocations_.fetch_add(1, std::memory_order_relaxed);
    allocationBytes_.fetch_add(capacity, std::memory_order_relaxed);
    return wrap(data, capacity);
  }

  Stats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return Stats{
        allocations_.load(std::memory_order_relaxed),
        reuses_.load(std::memory_order_relaxed),
        allocationBytes_.load(std::memory_order_relaxed),
        cachedBytes_};
  }

 private:
  static size_t sizeClass(size_t size) {
    constexpr size_t kMinimumClassBytes = 64ULL << 10;
    size_t capacity = kMinimumClassBytes;
    while (
        capacity < size &&
        capacity <= std::numeric_limits<size_t>::max() / 2) {
      capacity <<= 1;
    }
    return capacity < size ? size : capacity;
  }

  std::shared_ptr<uint8_t> wrap(uint8_t* data, size_t capacity) {
    auto self = shared_from_this();
    return std::shared_ptr<uint8_t>(
        data, [self = std::move(self), capacity](uint8_t* buffer) {
          self->release(buffer, capacity);
        });
  }

  void release(uint8_t* data, size_t capacity) noexcept {
    bool retain = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (capacity <= maxCachedBytes_ &&
          cachedBytes_ <= maxCachedBytes_ - capacity) {
        freeBuffers_[capacity].push_back(data);
        cachedBytes_ += capacity;
        retain = true;
      }
    }
    if (!retain) {
      cudaFreeHost(data);
    }
  }

  const size_t maxCachedBytes_;
  mutable std::mutex mutex_;
  std::map<size_t, std::vector<uint8_t*>> freeBuffers_;
  size_t cachedBytes_{0};
  std::atomic<uint64_t> allocations_{0};
  std::atomic<uint64_t> reuses_{0};
  std::atomic<uint64_t> allocationBytes_{0};
};

std::shared_ptr<PinnedCacheBufferPool> parquetCachePinnedBufferPool() {
  static auto pool =
      std::make_shared<PinnedCacheBufferPool>(asyncParquetFileSinkBytes());
  return pool;
}

// libcudf emits one sink write for every compressed Parquet column chunk.  It
// submits all device_write_async calls for a row group before waiting on their
// futures. Buffer each writer append in reusable pinned slabs so all D2H copies
// can be queued first. The first deferred future synchronizes the append and
// hands even its partially-filled tail slab to the background writer. This
// overlaps that write with the next append instead of deferring every tail to
// file close.
class AsyncPinnedFileSink final : public cudf::io::data_sink {
 public:
  explicit AsyncPinnedFileSink(std::string path)
      : path_(std::move(path)),
        maxPendingBytes_(asyncParquetFileSinkBytes()),
        pinnedPool_(parquetCachePinnedBufferPool()) {
    fd_ = ::open(path_.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    VELOX_CHECK_GE(fd_, 0, "Failed to open async Parquet cache file: {}", path_);
    worker_ = std::thread([this] { writeLoop(); });
  }

  ~AsyncPinnedFileSink() override {
    try {
      finish();
    } catch (...) {
    }
    // finish() can observe an asynchronous pwrite failure before the worker
    // thread has been joined. Never let std::thread's destructor turn that
    // recoverable I/O error into std::terminate()/executor exit 134.
    stopAndJoinWorker();
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  void host_write(void const* data, size_t size) override {
    const auto copyStart = std::chrono::steady_clock::now();
    auto* destination = reserve(size);
    std::memcpy(destination, data, size);
    copyNanos_ += elapsedNanos(copyStart);
  }

  [[nodiscard]] bool supports_device_write() const override {
    return true;
  }

  [[nodiscard]] bool is_device_write_preferred(size_t) const override {
    return true;
  }

  void device_write(
      void const* gpuData,
      size_t size,
      rmm::cuda_stream_view stream) override {
    device_write_async(gpuData, size, stream).get();
  }

  std::future<void> device_write_async(
      void const* gpuData,
      size_t size,
      rmm::cuda_stream_view stream) override {
    if (size > 0) {
      auto* destination = reserve(size);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          destination,
          gpuData,
          size,
          cudaMemcpyDeviceToHost,
          stream.value()));
    } else {
      ++zeroCopySubmissions_;
    }
    const auto submission = ++copySubmissions_;
    // libcudf submits all column-chunk copies before waiting on the returned
    // futures. Queue each copy immediately so compression and D2H retain the
    // writer's stream ordering. The first future synchronizes the complete
    // append and publishes its sealed slabs to the background disk writer.
    return std::async(std::launch::deferred, [this, stream, submission] {
      if (submission > synchronizedCopySubmissions_) {
        const auto copyStart = std::chrono::steady_clock::now();
        stream.synchronize();
        synchronizedCopySubmissions_ = copySubmissions_;
        copyNanos_ += elapsedNanos(copyStart);
        if (activeSlab_) {
          activeSlab_->sealed = true;
        }
        enqueueSealedSlabs();
      }
    });
  }

  void flush() override {
    if (finished_) {
      rethrowIfFailed();
      return;
    }
    // libcudf calls data_sink::flush() from chunked_parquet_writer::close().
    // At that point the footer and every D2H copy have already been submitted,
    // but waiting for pwrite here serializes file N with GPU encode for file
    // N+1. Seal and enqueue the tail slab only. The executor-wide finalizer
    // calls finish() after ownership of this sink has been transferred.
    if (activeSlab_) {
      activeSlab_->sealed = true;
    }
    enqueueSealedSlabs();
    rethrowIfFailed();
  }

  void finish() {
    if (finished_) {
      rethrowIfFailed();
      return;
    }
    try {
      flush();
      {
        std::unique_lock<std::mutex> lock(mutex_);
        drained_.wait(lock, [this] {
          return (writeQueue_.empty() && activeWrites_ == 0) || error_;
        });
        rethrowIfFailedLocked();
        stopping_ = true;
      }
      ready_.notify_all();
      if (worker_.joinable()) {
        worker_.join();
      }
      rethrowIfFailed();
      finished_ = true;
    } catch (...) {
      const auto failure = std::current_exception();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_) {
          error_ = failure;
        }
      }
      stopAndJoinWorker();
      throw;
    }
  }

  size_t bytes_written() override {
    rethrowIfFailed();
    return nextOffset_;
  }

  uint64_t copyNanos() const {
    return copyNanos_;
  }

  uint64_t copySubmissions() const {
    return copySubmissions_;
  }

  uint64_t copyBatches() const {
    return copyBatches_;
  }

  uint64_t zeroCopySubmissions() const {
    return zeroCopySubmissions_;
  }

  uint64_t enqueueWaitNanos() const {
    return enqueueWaitNanos_;
  }

  uint64_t writeNanos() const {
    return writeNanos_;
  }

  uint64_t writeChunks() const {
    return writeChunks_;
  }

  size_t peakPendingBytes() const {
    return peakPendingBytes_;
  }

 private:
  using PinnedBuffer = std::shared_ptr<uint8_t>;

  struct FileSlab {
    PinnedBuffer buffer;
    size_t capacity;
    size_t used{0};
    size_t fileOffset{0};
    bool sealed{false};
  };

  using FileSlabPtr = std::shared_ptr<FileSlab>;

  static constexpr size_t kFileSlabBytes = 64ULL << 20;

  uint8_t* reserve(size_t size) {
    VELOX_CHECK(!finished_, "Cannot append to a finished Parquet cache sink");
    VELOX_CHECK_LE(
        nextOffset_ + size,
        maxPendingBytes_,
        "A Parquet cache file exceeds the pinned-file budget");
    if (!activeSlab_ || activeSlab_->capacity - activeSlab_->used < size) {
      if (activeSlab_) {
        activeSlab_->sealed = true;
      }
      const auto capacity = std::max(kFileSlabBytes, size);
      activeSlab_ = std::make_shared<FileSlab>(FileSlab{
          pinnedPool_->acquire(capacity), capacity, 0, nextOffset_, false});
      pendingSlabs_.push_back(activeSlab_);
    }
    const auto offset = activeSlab_->used;
    auto* host = activeSlab_->buffer.get() + offset;
    activeSlab_->used += size;
    nextOffset_ += size;
    return host;
  }

  static uint64_t elapsedNanos(
      std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
  }

  void enqueueSealedSlabs() {
    while (!pendingSlabs_.empty() && pendingSlabs_.front()->sealed) {
      auto slab = std::move(pendingSlabs_.front());
      pendingSlabs_.pop_front();
      const auto waitStart = std::chrono::steady_clock::now();
      std::unique_lock<std::mutex> lock(mutex_);
      capacity_.wait(lock, [this, &slab] {
        return error_ || pendingBytes_ == 0 ||
            pendingBytes_ + slab->used <= maxPendingBytes_;
      });
      enqueueWaitNanos_ += elapsedNanos(waitStart);
      rethrowIfFailedLocked();
      pendingBytes_ += slab->used;
      peakPendingBytes_ = std::max(peakPendingBytes_, pendingBytes_);
      writeQueue_.push_back(std::move(slab));
      lock.unlock();
      ready_.notify_one();
    }
    if (activeSlab_ && activeSlab_->sealed) {
      activeSlab_.reset();
    }
  }

  void writeLoop() noexcept {
    try {
      while (true) {
        FileSlabPtr slab;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          ready_.wait(lock, [this] { return stopping_ || !writeQueue_.empty(); });
          if (writeQueue_.empty()) {
            break;
          }
          slab = std::move(writeQueue_.front());
          writeQueue_.pop_front();
          ++activeWrites_;
        }
        const auto writeStart = std::chrono::steady_clock::now();
        size_t written = 0;
        while (written < slab->used) {
          const auto result = ::pwrite(
              fd_,
              slab->buffer.get() + written,
              slab->used - written,
              static_cast<off_t>(slab->fileOffset + written));
          if (result < 0 && errno == EINTR) {
            continue;
          }
          const auto writeError = result < 0 ? errno : 0;
          VELOX_CHECK_GT(
              result,
              0,
              "Buffered Parquet cache pwrite failed: {}, errno={} ({})",
              path_,
              writeError,
              writeError == 0 ? "short write" : std::strerror(writeError));
          written += static_cast<size_t>(result);
        }
        writeNanos_ += elapsedNanos(writeStart);
        ++writeChunks_;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          pendingBytes_ -= slab->used;
          --activeWrites_;
        }
        slab.reset();
        capacity_.notify_all();
        drained_.notify_all();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = std::current_exception();
      capacity_.notify_all();
      drained_.notify_all();
    }
  }

  void rethrowIfFailed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    rethrowIfFailedLocked();
  }

  void rethrowIfFailedLocked() const {
    if (error_) {
      std::rethrow_exception(error_);
    }
  }

  void stopAndJoinWorker() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    capacity_.notify_all();
    drained_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  std::string path_;
  int fd_{-1};
  const size_t maxPendingBytes_;
  std::shared_ptr<PinnedCacheBufferPool> pinnedPool_;
  FileSlabPtr activeSlab_;
  std::deque<FileSlabPtr> pendingSlabs_;
  std::thread worker_;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable capacity_;
  std::condition_variable drained_;
  std::deque<FileSlabPtr> writeQueue_;
  size_t nextOffset_{0};
  size_t pendingBytes_{0};
  size_t activeWrites_{0};
  size_t peakPendingBytes_{0};
  uint64_t copyNanos_{0};
  uint64_t copySubmissions_{0};
  uint64_t copyBatches_{0};
  uint64_t zeroCopySubmissions_{0};
  uint64_t synchronizedCopySubmissions_{0};
  uint64_t enqueueWaitNanos_{0};
  uint64_t writeNanos_{0};
  uint64_t writeChunks_{0};
  bool finished_{false};
  bool stopping_{false};
  std::exception_ptr error_;
};

bool useParquetCacheFormat() {
  static const bool enabled = [] {
    const auto* value = std::getenv("GLUTEN_CACHE_FORMAT");
    if (value == nullptr || *value == '\0' ||
        std::strcmp(value, "packed") == 0) {
      return false;
    }
    GLUTEN_CHECK(
        std::strcmp(value, "parquet") == 0,
        "GLUTEN_CACHE_FORMAT must be either packed or parquet");
    return true;
  }();
  return enabled;
}

void setParquetColumnMetadata(
    cudf::io::column_in_metadata& metadata,
    const std::string& name,
    const TypePtr& type) {
  metadata.set_name(name);
  if (type->isDecimal()) {
    const auto precision = type->isShortDecimal()
        ? type->asShortDecimal().precision()
        : type->asLongDecimal().precision();
    metadata.set_decimal_precision(static_cast<uint8_t>(precision));
  }
  switch (type->kind()) {
    case TypeKind::ROW: {
      VELOX_CHECK_EQ(metadata.num_children(), type->size());
      const auto& rowType = type->asRow();
      for (auto index = 0; index < type->size(); ++index) {
        setParquetColumnMetadata(
            metadata.child(index),
            rowType.nameOf(index),
            type->childAt(index));
      }
      break;
    }
    case TypeKind::ARRAY:
      VELOX_CHECK_GE(metadata.num_children(), 2);
      setParquetColumnMetadata(
          metadata.child(1), "element", type->childAt(0));
      break;
    case TypeKind::MAP:
      VELOX_CHECK_GE(metadata.num_children(), 2);
      metadata.set_list_column_as_map();
      VELOX_CHECK_EQ(metadata.child(1).num_children(), 2);
      setParquetColumnMetadata(
          metadata.child(1).child(0), "key", type->childAt(0));
      setParquetColumnMetadata(
          metadata.child(1).child(1), "value", type->childAt(1));
      break;
    default:
      break;
  }
}

RowTypePtr selectRowType(
    const RowTypePtr& inputType,
    const std::vector<int32_t>& columnIndices) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(columnIndices.size());
  types.reserve(columnIndices.size());
  for (const auto index : columnIndices) {
    VELOX_CHECK_GE(index, 0);
    VELOX_CHECK_LT(index, inputType->size());
    names.push_back(inputType->nameOf(index));
    types.push_back(inputType->childAt(index));
  }
  return ROW(std::move(names), std::move(types));
}

CacheCodec cacheCodec() {
  static const CacheCodec codec = [] {
    const auto* value = std::getenv("GLUTEN_CACHE_CODEC");
    if (value == nullptr || *value == '\0' ||
        std::strcmp(value, "zstd") == 0) {
      return CacheCodec::kZstd;
    }
    GLUTEN_CHECK(
        std::strcmp(value, "lz4") == 0,
        "GLUTEN_CACHE_CODEC must be either zstd or lz4");
    return CacheCodec::kLz4;
  }();
  return codec;
}

const char* cacheCodecName(CacheCodec codec) {
  return codec == CacheCodec::kZstd ? "Zstd" : "LZ4";
}

nvcompStatus_t cacheCompressGetMaxOutputChunkSize(
    CacheCodec codec,
    size_t chunkBytes,
    size_t* maxCompressedChunkBytes) {
  if (codec == CacheCodec::kZstd) {
    return nvcompBatchedZstdCompressGetMaxOutputChunkSize(
        chunkBytes,
        nvcompBatchedZstdCompressDefaultOpts,
        maxCompressedChunkBytes);
  }
  return nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      chunkBytes,
      nvcompBatchedLZ4CompressDefaultOpts,
      maxCompressedChunkBytes);
}

nvcompStatus_t cacheCompressGetTempSize(
    CacheCodec codec,
    size_t numChunks,
    size_t chunkBytes,
    size_t* tempBytes,
    size_t totalBytes) {
  if (codec == CacheCodec::kZstd) {
    return nvcompBatchedZstdCompressGetTempSizeAsync(
        numChunks,
        chunkBytes,
        nvcompBatchedZstdCompressDefaultOpts,
        tempBytes,
        totalBytes);
  }
  return nvcompBatchedLZ4CompressGetTempSizeAsync(
      numChunks,
      chunkBytes,
      nvcompBatchedLZ4CompressDefaultOpts,
      tempBytes,
      totalBytes);
}

nvcompStatus_t cacheCompressAsync(
    CacheCodec codec,
    const void* const* inputPointers,
    const size_t* inputSizes,
    size_t chunkBytes,
    size_t numChunks,
    void* temp,
    size_t tempBytes,
    void* const* outputPointers,
    size_t* outputSizes,
    cudaStream_t stream) {
  if (codec == CacheCodec::kZstd) {
    return nvcompBatchedZstdCompressAsync(
        inputPointers,
        inputSizes,
        chunkBytes,
        numChunks,
        temp,
        tempBytes,
        outputPointers,
        outputSizes,
        nvcompBatchedZstdCompressDefaultOpts,
        nullptr,
        stream);
  }
  return nvcompBatchedLZ4CompressAsync(
      inputPointers,
      inputSizes,
      chunkBytes,
      numChunks,
      temp,
      tempBytes,
      outputPointers,
      outputSizes,
      nvcompBatchedLZ4CompressDefaultOpts,
      nullptr,
      stream);
}

bool cacheCompressionEnabled() {
  static const bool enabled = [] {
    const auto* value = std::getenv("GLUTEN_CACHE_COMPRESSION");
    return value == nullptr || *value == '\0' ||
        std::strcmp(value, "false") != 0;
  }();
  return enabled;
}

size_t nvcompZstdChunkBytes() {
  static const size_t chunkBytes = [] {
    const auto* value = std::getenv("GLUTEN_CACHE_ZSTD_CHUNK_BYTES");
    if (value == nullptr || *value == '\0') {
      return kDefaultNvcompZstdChunkBytes;
    }
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    GLUTEN_CHECK(
        end != value && *end == '\0' &&
            parsed >= kMinNvcompZstdChunkBytes &&
            parsed <= kMaxNvcompZstdChunkBytes,
        "GLUTEN_CACHE_ZSTD_CHUNK_BYTES must be between 65536 and 16777216");
    return static_cast<size_t>(parsed);
  }();
  return chunkBytes;
}
struct CudfCacheHeader {
  uint64_t magic;
  uint32_t version;
  uint32_t reserved;
  uint64_t metadataSize;
  uint64_t gpuDataSize;
};

static_assert(sizeof(CudfCacheHeader) == 32);

struct ChunkedZstdHeader {
  uint32_t numChunks;
  uint32_t reserved;
};

struct ChunkedZstdEntry {
  uint32_t rawSize;
  uint32_t compressedSize;
};

std::atomic<uint64_t> cacheSerializeBatches{0};
std::atomic<uint64_t> cacheSerializeBytes{0};
std::atomic<uint64_t> cacheRawBytes{0};
std::atomic<uint64_t> cacheAppendNanos{0};
std::atomic<uint64_t> cachePackNanos{0};
std::atomic<uint64_t> cacheD2HNanos{0};
std::atomic<uint64_t> cacheCompressionNanos{0};
std::atomic<uint64_t> cacheSerializeCopyNanos{0};
std::atomic<uint64_t> cacheDeserializePages{0};
std::atomic<uint64_t> cacheDeserializeSelectedPages{0};
std::atomic<uint64_t> cacheDeserializeBytes{0};
std::atomic<uint64_t> cacheDeserializeRows{0};
std::atomic<uint64_t> cacheDeserializeOutputColumns{0};
std::atomic<uint64_t> cacheDeserializeNanos{0};
std::atomic<uint64_t> parquetCacheFillPages{0};
std::atomic<uint64_t> parquetCacheFillAppends{0};
std::atomic<uint64_t> parquetCacheFillBatches{0};
std::atomic<uint64_t> parquetCacheFillRows{0};
std::atomic<uint64_t> parquetCacheFillBytes{0};
std::atomic<uint64_t> parquetCacheFillProducerSyncNanos{0};
std::atomic<uint64_t> parquetCacheFillConversionNanos{0};
std::atomic<uint64_t> parquetCacheFillConcatNanos{0};
std::atomic<uint64_t> parquetCacheFillWriterInitNanos{0};
std::atomic<uint64_t> parquetCacheFillWriteSubmitNanos{0};
std::atomic<uint64_t> parquetCacheFillWriteSyncNanos{0};
std::atomic<uint64_t> parquetCacheFillStatNanos{0};
std::atomic<uint64_t> parquetCacheFillCloseNanos{0};
std::atomic<uint64_t> parquetCacheFillAsyncCopyNanos{0};
std::atomic<uint64_t> parquetCacheFillAsyncCopySubmissions{0};
std::atomic<uint64_t> parquetCacheFillAsyncCopyBatches{0};
std::atomic<uint64_t> parquetCacheFillAsyncZeroCopySubmissions{0};
std::atomic<uint64_t> parquetCacheFillAsyncEnqueueWaitNanos{0};
std::atomic<uint64_t> parquetCacheFillAsyncWriteNanos{0};
std::atomic<uint64_t> parquetCacheFillAsyncWriteChunks{0};
std::atomic<uint64_t> parquetCacheFillAsyncPeakPendingBytes{0};
std::atomic<uint64_t> parquetCacheFillAsyncFinalizeSubmitWaitNanos{0};
std::atomic<uint64_t> parquetCacheFillAsyncFinalizeDrainNanos{0};
std::atomic<uint64_t> parquetCacheFillAsyncFinalizePeakFiles{0};

size_t asyncParquetFileFinalizeDepth() {
  static const size_t depth = [] {
    constexpr size_t kDefaultDepth = 2;
    const auto* value = std::getenv("GLUTEN_CACHE_ASYNC_WRITE_DEPTH");
    if (value == nullptr || *value == '\0') {
      return kDefaultDepth;
    }
    const auto parsed = std::strtoull(value, nullptr, 10);
    return parsed == 0 ? kDefaultDepth : static_cast<size_t>(parsed);
  }();
  return depth;
}

// A Parquet footer must be produced before its page descriptor can be
// returned, but the sink does not need to finish pwrite before the next page
// starts encoding. Keep a small executor-wide set of finalizers. The Spark
// cache iterator drains this service before publishing its partition block,
// preserving materialize durability and asynchronous error propagation.
class AsyncParquetFileFinalizer final {
 public:
  AsyncParquetFileFinalizer() : depth_(asyncParquetFileFinalizeDepth()) {
    workers_.reserve(depth_);
    for (size_t index = 0; index < depth_; ++index) {
      workers_.emplace_back([this] { workerLoop(); });
    }
  }

  ~AsyncParquetFileFinalizer() {
    try {
      drain();
    } catch (...) {
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  void submit(std::unique_ptr<AsyncPinnedFileSink> sink) {
    VELOX_CHECK_NOT_NULL(sink);
    const auto waitStart = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    capacity_.wait(lock, [this] { return error_ || inFlight_ < depth_; });
    parquetCacheFillAsyncFinalizeSubmitWaitNanos.fetch_add(
        elapsedNanos(waitStart), std::memory_order_relaxed);
    rethrowIfFailedLocked();
    queue_.push_back(std::move(sink));
    ++inFlight_;
    auto previousPeak =
        parquetCacheFillAsyncFinalizePeakFiles.load(std::memory_order_relaxed);
    while (previousPeak < inFlight_ &&
           !parquetCacheFillAsyncFinalizePeakFiles.compare_exchange_weak(
               previousPeak, inFlight_, std::memory_order_relaxed)) {
    }
    lock.unlock();
    ready_.notify_one();
  }

  void drain() {
    const auto waitStart = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    drained_.wait(lock, [this] { return error_ || inFlight_ == 0; });
    parquetCacheFillAsyncFinalizeDrainNanos.fetch_add(
        elapsedNanos(waitStart), std::memory_order_relaxed);
    rethrowIfFailedLocked();
  }

 private:
  static uint64_t elapsedNanos(
      std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
  }

  void workerLoop() noexcept {
    while (true) {
      std::unique_ptr<AsyncPinnedFileSink> sink;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) {
          return;
        }
        sink = std::move(queue_.front());
        queue_.pop_front();
      }

      std::exception_ptr failure;
      try {
        sink->finish();
        parquetCacheFillAsyncCopyNanos.fetch_add(
            sink->copyNanos(), std::memory_order_relaxed);
        parquetCacheFillAsyncCopySubmissions.fetch_add(
            sink->copySubmissions(), std::memory_order_relaxed);
        parquetCacheFillAsyncCopyBatches.fetch_add(
            sink->copyBatches(), std::memory_order_relaxed);
        parquetCacheFillAsyncZeroCopySubmissions.fetch_add(
            sink->zeroCopySubmissions(), std::memory_order_relaxed);
        parquetCacheFillAsyncEnqueueWaitNanos.fetch_add(
            sink->enqueueWaitNanos(), std::memory_order_relaxed);
        parquetCacheFillAsyncWriteNanos.fetch_add(
            sink->writeNanos(), std::memory_order_relaxed);
        parquetCacheFillAsyncWriteChunks.fetch_add(
            sink->writeChunks(), std::memory_order_relaxed);
        auto previousPeak = parquetCacheFillAsyncPeakPendingBytes.load(
            std::memory_order_relaxed);
        while (previousPeak < sink->peakPendingBytes() &&
               !parquetCacheFillAsyncPeakPendingBytes.compare_exchange_weak(
                   previousPeak,
                   sink->peakPendingBytes(),
                   std::memory_order_relaxed)) {
        }
      } catch (...) {
        failure = std::current_exception();
      }
      sink.reset();

      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure && !error_) {
          error_ = failure;
        }
        if (inFlight_ > 0) {
          --inFlight_;
        } else if (!error_) {
          error_ = std::make_exception_ptr(std::runtime_error(
              "Parquet cache finalizer completed with no in-flight file"));
        }
      }
      capacity_.notify_all();
      drained_.notify_all();
    }
  }

  void rethrowIfFailedLocked() const {
    if (error_) {
      std::rethrow_exception(error_);
    }
  }

  const size_t depth_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable capacity_;
  std::condition_variable drained_;
  std::deque<std::unique_ptr<AsyncPinnedFileSink>> queue_;
  size_t inFlight_{0};
  bool stopping_{false};
  std::exception_ptr error_;
};

AsyncParquetFileFinalizer& asyncParquetFileFinalizer() {
  static AsyncParquetFileFinalizer finalizer;
  return finalizer;
}

void recordParquetCacheDeserialize(
    uint64_t bytes,
    uint64_t rows,
    uint64_t outputColumns,
    uint64_t inputColumns,
    bool selected,
    std::chrono::steady_clock::time_point start) {
  cacheDeserializeBytes.fetch_add(bytes, std::memory_order_relaxed);
  cacheDeserializeRows.fetch_add(rows, std::memory_order_relaxed);
  cacheDeserializeOutputColumns.fetch_add(
      outputColumns, std::memory_order_relaxed);
  cacheDeserializeNanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count(),
      std::memory_order_relaxed);
  const auto pages =
      cacheDeserializePages.fetch_add(1, std::memory_order_relaxed) + 1;
  if (selected) {
    cacheDeserializeSelectedPages.fetch_add(1, std::memory_order_relaxed);
  }
  if (pages == 1 || (pages & 0x7f) == 0) {
    LOG(WARNING) << "GPU Parquet cache restore: pages=" << pages
                 << " selectedPages="
                 << cacheDeserializeSelectedPages.load(
                        std::memory_order_relaxed)
                 << " parquetBytes="
                 << cacheDeserializeBytes.load(std::memory_order_relaxed)
                 << " rows="
                 << cacheDeserializeRows.load(std::memory_order_relaxed)
                 << " outputColumns="
                 << cacheDeserializeOutputColumns.load(
                        std::memory_order_relaxed)
                 << " latestColumns=" << outputColumns << "/"
                 << inputColumns << " decodeMs="
                 << (cacheDeserializeNanos.load(std::memory_order_relaxed) /
                     1000000.0);
  }
}

} // namespace

struct VeloxGpuColumnarBatchSerializer::ParquetFileWriterState {
  explicit ParquetFileWriterState(std::string filePath)
      : path(std::move(filePath)),
        stream(cudf_velox::cudfGlobalStreamPool().get_stream()) {
    if (useAsyncParquetFileSink()) {
      sink = std::make_unique<AsyncPinnedFileSink>(path);
    }
  }

  std::string path;
  rmm::cuda_stream_view stream;
  RowTypePtr inputType;
  std::unique_ptr<AsyncPinnedFileSink> sink;
  std::unique_ptr<cudf::io::chunked_parquet_writer> writer;
  uint64_t rows{0};
  uint64_t appends{0};
  uint64_t batches{0};
  uint64_t producerSyncNanos{0};
  uint64_t conversionNanos{0};
  uint64_t concatNanos{0};
  uint64_t writerInitNanos{0};
  uint64_t writeSubmitNanos{0};
  uint64_t writeSyncNanos{0};
  uint64_t statNanos{0};
};

VeloxGpuColumnarBatchSerializer::VeloxGpuColumnarBatchSerializer(
    arrow::MemoryPool* arrowPool,
    std::shared_ptr<memory::MemoryPool> veloxPool,
    struct ArrowSchema* cSchema)
    : VeloxColumnarBatchSerializer(arrowPool, veloxPool, cSchema) {
}

VeloxGpuColumnarBatchSerializer::~VeloxGpuColumnarBatchSerializer() {
  if (parquetFileWriterState_ && parquetFileWriterState_->writer) {
    try {
      parquetFileWriterState_->writer->close();
      parquetFileWriterState_->stream.synchronize();
    } catch (...) {
      // Destructors must not mask the original task/JNI failure. The caller
      // deletes the incomplete page when append or finish fails.
    }
  }
}

void VeloxGpuColumnarBatchSerializer::beginParquetFile(
    const std::string& path) {
  VELOX_CHECK(useParquetCacheFormat());
  VELOX_CHECK_NULL(
      parquetFileWriterState_, "A Parquet cache page is already open");
  parquetFileWriterState_ =
      std::make_unique<ParquetFileWriterState>(path);
}

int64_t VeloxGpuColumnarBatchSerializer::appendParquetFile(
    const std::shared_ptr<ColumnarBatch>& batch) {
  return appendParquetFileMany({batch});
}

int64_t VeloxGpuColumnarBatchSerializer::appendParquetFileMany(
    const std::vector<std::shared_ptr<ColumnarBatch>>& batches) {
  VELOX_CHECK(!batches.empty());
  VELOX_CHECK_NOT_NULL(
      parquetFileWriterState_, "No Parquet cache page is open");
  auto& state = *parquetFileWriterState_;
  std::vector<std::shared_ptr<cudf_velox::CudfVector>> cudfVectors;
  std::vector<cudf::table_view> tableViews;
  cudfVectors.reserve(batches.size());
  tableViews.reserve(batches.size());
  RowTypePtr batchType;
  uint64_t rows = 0;
  for (const auto& batch : batches) {
    auto veloxBatch = VeloxColumnarBatch::from(veloxPool_.get(), batch);
    auto rowVector = veloxBatch->getRowVector();
    auto cudfVector =
        std::dynamic_pointer_cast<cudf_velox::CudfVector>(rowVector);
    if (!cudfVector) {
      const auto conversionStart = std::chrono::steady_clock::now();
      auto table = cudf_velox::with_arrow::toCudfTable(
          rowVector,
          veloxPool_.get(),
          state.stream,
          cudf_velox::get_output_mr());
      cudfVector = std::make_shared<cudf_velox::CudfVector>(
          veloxPool_.get(),
          rowVector->type(),
          rowVector->size(),
          std::move(table),
          state.stream);
      state.conversionNanos +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - conversionStart)
              .count();
    }
    const auto inputType = asRowType(rowVector->type());
    if (!batchType) {
      batchType = inputType;
    } else {
      VELOX_CHECK(batchType->equivalent(*inputType));
    }
    rows += rowVector->size();
    tableViews.push_back(cudfVector->getTableView());
    cudfVectors.push_back(std::move(cudfVector));
  }

  std::unique_ptr<cudf::table> concatenated;
  cudf::table_view table;
  const auto concatStart = std::chrono::steady_clock::now();
  if (tableViews.size() == 1) {
    // A single input is not copied. Finish its producer stream before the
    // chunked writer reads it on the writer stream. The input remains retained
    // until the writer stream is synchronized below.
    const auto syncStart = std::chrono::steady_clock::now();
    cudfVectors.front()->stream().synchronize();
    state.producerSyncNanos +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - syncStart)
            .count();
    table = tableViews.front();
  } else {
    // Inputs may originate from different MPP scan-driver streams. Use the
    // common cuDF utility so the writer stream joins every producer stream and
    // source-buffer deallocation is ordered after concatenate. A host-side
    // synchronize before concatenate is insufficient: the producer streams
    // can enqueue their next reads before these retained vectors are released.
    concatenated = cudf_velox::getConcatenatedTable(
        std::move(cudfVectors),
        batchType,
        state.stream,
        cudf_velox::get_output_mr());
    table = concatenated->view();
  }
  state.concatNanos +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - concatStart)
          .count();
  if (!state.writer) {
    const auto writerInitStart = std::chrono::steady_clock::now();
    state.inputType = batchType;
    auto metadata = cudf::io::table_input_metadata(table);
    VELOX_CHECK_EQ(metadata.column_metadata.size(), batchType->size());
    for (auto index = 0; index < batchType->size(); ++index) {
      setParquetColumnMetadata(
          metadata.column_metadata[index],
          batchType->nameOf(index),
          batchType->childAt(index));
    }
    auto sinkInfo = state.sink
        ? cudf::io::sink_info(
              static_cast<cudf::io::data_sink*>(state.sink.get()))
        : cudf::io::sink_info(state.path);
    auto options = cudf::io::chunked_parquet_writer_options::builder(
                       std::move(sinkInfo))
                       .metadata(std::move(metadata))
                       .compression(parquetCacheCompression())
                       .stats_level(cudf::io::STATISTICS_NONE)
                       .utc_timestamps(true)
                       .build();
    state.writer = std::make_unique<cudf::io::chunked_parquet_writer>(
        options, state.stream);
    state.writerInitNanos +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - writerInitStart)
            .count();
  } else {
    VELOX_CHECK(state.inputType->equivalent(*batchType));
  }

  const auto writeStart = std::chrono::steady_clock::now();
  state.writer->write(table);
  const auto writeSubmitted = std::chrono::steady_clock::now();
  state.writeSubmitNanos +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          writeSubmitted - writeStart)
          .count();
  state.stream.synchronize();
  state.writeSyncNanos +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - writeSubmitted)
          .count();
  state.rows += rows;
  ++state.appends;
  state.batches += batches.size();
  const auto statStart = std::chrono::steady_clock::now();
  const auto currentBytes = state.sink
      ? state.sink->bytes_written()
      : std::filesystem::file_size(state.path);
  state.statNanos +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - statStart)
          .count();
  return currentBytes;
}

int64_t VeloxGpuColumnarBatchSerializer::finishParquetFile() {
  VELOX_CHECK_NOT_NULL(
      parquetFileWriterState_, "No Parquet cache page is open");
  auto& state = *parquetFileWriterState_;
  VELOX_CHECK_NOT_NULL(state.writer, "Cannot finish an empty Parquet cache page");
  const auto closeStart = std::chrono::steady_clock::now();
  state.writer->close();
  state.stream.synchronize();
  // close() has emitted the footer and every D2H is complete. Destroy the
  // writer before transferring its sink to the executor-wide finalizer.
  state.writer.reset();
  const auto closeNanos =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - closeStart)
          .count();
  const auto finalBytes = state.sink
      ? static_cast<int64_t>(state.sink->bytes_written())
      : static_cast<int64_t>(std::filesystem::file_size(state.path));
  parquetCacheFillAppends.fetch_add(state.appends, std::memory_order_relaxed);
  parquetCacheFillBatches.fetch_add(state.batches, std::memory_order_relaxed);
  parquetCacheFillRows.fetch_add(state.rows, std::memory_order_relaxed);
  parquetCacheFillBytes.fetch_add(finalBytes, std::memory_order_relaxed);
  parquetCacheFillProducerSyncNanos.fetch_add(
      state.producerSyncNanos, std::memory_order_relaxed);
  parquetCacheFillConversionNanos.fetch_add(
      state.conversionNanos, std::memory_order_relaxed);
  parquetCacheFillConcatNanos.fetch_add(
      state.concatNanos, std::memory_order_relaxed);
  parquetCacheFillWriterInitNanos.fetch_add(
      state.writerInitNanos, std::memory_order_relaxed);
  parquetCacheFillWriteSubmitNanos.fetch_add(
      state.writeSubmitNanos, std::memory_order_relaxed);
  parquetCacheFillWriteSyncNanos.fetch_add(
      state.writeSyncNanos, std::memory_order_relaxed);
  parquetCacheFillStatNanos.fetch_add(
      state.statNanos, std::memory_order_relaxed);
  parquetCacheFillCloseNanos.fetch_add(closeNanos, std::memory_order_relaxed);
  if (state.sink) {
    asyncParquetFileFinalizer().submit(std::move(state.sink));
  }
  const auto pages =
      parquetCacheFillPages.fetch_add(1, std::memory_order_relaxed) + 1;
  if (pages == 1 || (pages & 0x7f) == 0) {
    constexpr double kNanosPerMillisecond = 1000000.0;
    const auto pinnedPoolStats = parquetCachePinnedBufferPool()->stats();
    LOG(WARNING) << "GPU Parquet cache fill: pages=" << pages
                 << " compression=" << parquetCacheCompressionName()
                 << " appends="
                 << parquetCacheFillAppends.load(std::memory_order_relaxed)
                 << " batches="
                 << parquetCacheFillBatches.load(std::memory_order_relaxed)
                 << " rows="
                 << parquetCacheFillRows.load(std::memory_order_relaxed)
                 << " parquetBytes="
                 << parquetCacheFillBytes.load(std::memory_order_relaxed)
                 << " producerSyncMs="
                 << (parquetCacheFillProducerSyncNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " conversionMs="
                 << (parquetCacheFillConversionNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " concatMs="
                 << (parquetCacheFillConcatNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " writerInitMs="
                 << (parquetCacheFillWriterInitNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " writeSubmitMs="
                 << (parquetCacheFillWriteSubmitNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " writeSyncMs="
                 << (parquetCacheFillWriteSyncNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " statMs="
                 << (parquetCacheFillStatNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " closeMs="
                 << (parquetCacheFillCloseNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " asyncCopyMs="
                 << (parquetCacheFillAsyncCopyNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " asyncCopySubmissions="
                 << parquetCacheFillAsyncCopySubmissions.load(
                        std::memory_order_relaxed)
                 << " asyncCopyBatches="
                 << parquetCacheFillAsyncCopyBatches.load(
                        std::memory_order_relaxed)
                 << " asyncZeroCopySubmissions="
                 << parquetCacheFillAsyncZeroCopySubmissions.load(
                        std::memory_order_relaxed)
                 << " asyncEnqueueWaitMs="
                 << (parquetCacheFillAsyncEnqueueWaitNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " asyncWriteMs="
                 << (parquetCacheFillAsyncWriteNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " asyncWriteChunks="
                 << parquetCacheFillAsyncWriteChunks.load(
                        std::memory_order_relaxed)
                 << " asyncPeakPendingBytes="
                 << parquetCacheFillAsyncPeakPendingBytes.load(
                        std::memory_order_relaxed)
                 << " asyncFinalizeSubmitWaitMs="
                 << (parquetCacheFillAsyncFinalizeSubmitWaitNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " asyncFinalizeDrainMs="
                 << (parquetCacheFillAsyncFinalizeDrainNanos.load(
                         std::memory_order_relaxed) /
                     kNanosPerMillisecond)
                 << " asyncFinalizePeakFiles="
                 << parquetCacheFillAsyncFinalizePeakFiles.load(
                        std::memory_order_relaxed)
                 << " pinnedPoolAllocations="
                 << pinnedPoolStats.allocations
                 << " pinnedPoolReuses=" << pinnedPoolStats.reuses
                 << " pinnedPoolAllocationBytes="
                 << pinnedPoolStats.allocationBytes
                 << " pinnedPoolCachedBytes=" << pinnedPoolStats.cachedBytes;
  }
  parquetFileWriterState_.reset();
  return finalBytes;
}

void VeloxGpuColumnarBatchSerializer::drainParquetFileWrites() {
  if (useAsyncParquetFileSink()) {
    asyncParquetFileFinalizer().drain();
  }
}

void VeloxGpuColumnarBatchSerializer::append(
    const std::shared_ptr<ColumnarBatch>& batch) {
  const auto appendStart = std::chrono::steady_clock::now();
  auto veloxBatch = VeloxColumnarBatch::from(veloxPool_.get(), batch);
  auto rowVector = veloxBatch->getRowVector();
  if (auto cudfVector =
          std::dynamic_pointer_cast<cudf_velox::CudfVector>(rowVector)) {
    // Keep the cache payload in cuDF's contiguous packed representation.
    // Converting device columns to a host RowVector and then running Presto
    // serde adds two full column walks plus several copies for this scan-heavy
    // cache-fill path. The packed format preserves nested/null semantics and
    // needs only one contiguous D2H copy.
    GpuLockGuard gpuLock;
    auto stream = cudfVector->stream();
    if (useParquetCacheFormat()) {
      const auto table = cudfVector->getTableView();
      VELOX_CHECK_EQ(table.num_columns(), rowVector->type()->size());
      std::vector<char> parquetBytes;
      auto metadata = cudf::io::table_input_metadata(table);
      const auto inputType = asRowType(rowVector->type());
      VELOX_CHECK_EQ(metadata.column_metadata.size(), inputType->size());
      for (auto index = 0; index < inputType->size(); ++index) {
        setParquetColumnMetadata(
            metadata.column_metadata[index],
            inputType->nameOf(index),
            inputType->childAt(index));
      }
      auto options = cudf::io::parquet_writer_options::builder(
                         cudf::io::sink_info(&parquetBytes), table)
                         .metadata(std::move(metadata))
                         .compression(cudf::io::compression_type::SNAPPY)
                         // Cache pages are addressed by Spark block/page metadata and the
                         // restore path currently performs column pruning only.  Row-group
                         // statistics are never consulted, while computing them for a very
                         // wide cache schema adds a full per-column reduction to every page.
                         .stats_level(cudf::io::STATISTICS_NONE)
                         .utc_timestamps(true)
                         .build();
      cudf::io::write_parquet(options, stream);
      stream.synchronize();
      hostGpuData_.assign(parquetBytes.begin(), parquetBytes.end());
      cacheVersion_ = kCudfCacheVersionParquet;
      cudfCache_ = true;
      cacheRawBytes.fetch_add(
          cudfVector->retainedSize(), std::memory_order_relaxed);
      cacheCompressionNanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - appendStart)
              .count(),
          std::memory_order_relaxed);
      cacheAppendNanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - appendStart)
              .count(),
          std::memory_order_relaxed);
      return;
    }
    if (auto packedTable = cudfVector->releasePacked()) {
      packedColumns_ = std::make_unique<cudf::packed_columns>(
          std::move(packedTable->data));
    } else {
      packedColumns_ = std::make_unique<cudf::packed_columns>(cudf::pack(
          cudfVector->getTableView(),
          stream,
          cudf_velox::get_output_mr()));
    }
    stream.synchronize();
    const auto gpuDataSize = packedColumns_->gpu_data->size();
    GLUTEN_CHECK(
        gpuDataSize <= std::numeric_limits<uint32_t>::max(),
        "A packed cache page exceeds the 4 GiB format limit");
    uncompressedGpuDataSize_ = static_cast<uint32_t>(gpuDataSize);
    cacheRawBytes.fetch_add(gpuDataSize, std::memory_order_relaxed);
    cachePackNanos.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - appendStart)
            .count(),
        std::memory_order_relaxed);

    if (!cacheCompressionEnabled()) {
      hostGpuData_.resize(gpuDataSize);
      const auto d2hStart = std::chrono::steady_clock::now();
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostGpuData_.data(),
          packedColumns_->gpu_data->data(),
          gpuDataSize,
          cudaMemcpyDeviceToHost,
          stream.value()));
      stream.synchronize();
      cacheD2HNanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - d2hStart)
              .count(),
          std::memory_order_relaxed);
      cacheVersion_ = kCudfCacheVersionRaw;
      packedColumns_->gpu_data.reset();
      cacheAppendNanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - appendStart)
              .count(),
          std::memory_order_relaxed);
      return;
    }

    const auto compressionStart = std::chrono::steady_clock::now();
    const auto codec = cacheCodec();
    const auto chunkBytes = nvcompZstdChunkBytes();
    const auto numChunks = static_cast<uint32_t>(
        (gpuDataSize + chunkBytes - 1) / chunkBytes);
    GLUTEN_CHECK(
        numChunks > 0 && numChunks <= kMaxCacheChunks,
        "Invalid number of nvCOMP cache chunks: " +
            std::to_string(numChunks));

    size_t maxCompressedChunkBytes = 0;
    auto nvcompStatus = cacheCompressGetMaxOutputChunkSize(
        codec, chunkBytes, &maxCompressedChunkBytes);
    GLUTEN_CHECK(
        nvcompStatus == nvcompSuccess,
        std::string("nvCOMP failed to size ") + cacheCodecName(codec) +
            " cache output: " +
            std::to_string(static_cast<int>(nvcompStatus)));

    size_t tempBytes = 0;
    nvcompStatus = cacheCompressGetTempSize(
        codec,
        numChunks,
        chunkBytes,
        &tempBytes,
        gpuDataSize);
    GLUTEN_CHECK(
        nvcompStatus == nvcompSuccess,
        std::string("nvCOMP failed to size ") + cacheCodecName(codec) +
            " cache workspace: " +
            std::to_string(static_cast<int>(nvcompStatus)));

    const auto outputCapacity =
        static_cast<size_t>(numChunks) * maxCompressedChunkBytes;
    auto mr = cudf_velox::get_output_mr();
    rmm::device_buffer compressedGpuData(outputCapacity, stream, mr);
    rmm::device_buffer tempGpuData(tempBytes, stream, mr);
    rmm::device_buffer inputPointersGpu(
        numChunks * sizeof(void*), stream, mr);
    rmm::device_buffer inputSizesGpu(
        numChunks * sizeof(size_t), stream, mr);
    rmm::device_buffer outputPointersGpu(
        numChunks * sizeof(void*), stream, mr);
    rmm::device_buffer outputSizesGpu(
        numChunks * sizeof(size_t), stream, mr);

    std::vector<void const*> inputPointers(numChunks);
    std::vector<size_t> inputSizes(numChunks);
    std::vector<void*> outputPointers(numChunks);
    std::vector<size_t> outputSizes(numChunks);
    size_t rawOffset = 0;
    for (uint32_t chunk = 0; chunk < numChunks; ++chunk) {
      const auto chunkSize =
          std::min(chunkBytes, gpuDataSize - rawOffset);
      inputPointers[chunk] =
          static_cast<uint8_t const*>(packedColumns_->gpu_data->data()) +
          rawOffset;
      inputSizes[chunk] = chunkSize;
      outputPointers[chunk] =
          static_cast<uint8_t*>(compressedGpuData.data()) +
          chunk * maxCompressedChunkBytes;
      rawOffset += chunkSize;
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        inputPointersGpu.data(),
        inputPointers.data(),
        numChunks * sizeof(void*),
        cudaMemcpyHostToDevice,
        stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        inputSizesGpu.data(),
        inputSizes.data(),
        numChunks * sizeof(size_t),
        cudaMemcpyHostToDevice,
        stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        outputPointersGpu.data(),
        outputPointers.data(),
        numChunks * sizeof(void*),
        cudaMemcpyHostToDevice,
        stream.value()));
    nvcompStatus = cacheCompressAsync(
        codec,
        static_cast<void const* const*>(inputPointersGpu.data()),
        static_cast<size_t const*>(inputSizesGpu.data()),
        chunkBytes,
        numChunks,
        tempGpuData.data(),
        tempBytes,
        static_cast<void* const*>(outputPointersGpu.data()),
        static_cast<size_t*>(outputSizesGpu.data()),
        stream.value());
    GLUTEN_CHECK(
        nvcompStatus == nvcompSuccess,
        std::string("nvCOMP failed to launch ") + cacheCodecName(codec) +
            " cache compression: " +
            std::to_string(static_cast<int>(nvcompStatus)));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        outputSizes.data(),
        outputSizesGpu.data(),
        numChunks * sizeof(size_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();

    size_t compressedSize =
        sizeof(ChunkedZstdHeader) + numChunks * sizeof(ChunkedZstdEntry);
    for (uint32_t chunk = 0; chunk < numChunks; ++chunk) {
      GLUTEN_CHECK(
          outputSizes[chunk] > 0 &&
              outputSizes[chunk] <= maxCompressedChunkBytes,
          std::string("nvCOMP returned an invalid ") +
              cacheCodecName(codec) + " cache chunk size");
      compressedSize += outputSizes[chunk];
    }
    if (compressedSize < gpuDataSize) {
      const auto payloadOffset =
          sizeof(ChunkedZstdHeader) +
          numChunks * sizeof(ChunkedZstdEntry);
      const auto compressedPayloadSize = compressedSize - payloadOffset;
      rmm::device_buffer compactGpuData(compressedPayloadSize, stream, mr);
      std::vector<void*> compactPointers(numChunks);
      std::vector<void const*> outputSourcePointers(numChunks);
      size_t compactOffset = 0;
      for (uint32_t chunk = 0; chunk < numChunks; ++chunk) {
        compactPointers[chunk] =
            static_cast<uint8_t*>(compactGpuData.data()) + compactOffset;
        outputSourcePointers[chunk] = outputPointers[chunk];
        compactOffset += outputSizes[chunk];
      }
      cudaMemcpyAttributes copyAttributes{};
      copyAttributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      size_t copyAttributesIndex = 0;
      CUDF_CUDA_TRY(cudaMemcpyBatchAsync(
          compactPointers.data(),
          outputSourcePointers.data(),
          outputSizes.data(),
          numChunks,
          &copyAttributes,
          &copyAttributesIndex,
          1,
          stream.value()));

      hostGpuData_.resize(compressedSize);
      const ChunkedZstdHeader chunkedHeader{numChunks, 0};
      std::memcpy(
          hostGpuData_.data(), &chunkedHeader, sizeof(chunkedHeader));
      size_t entryOffset = sizeof(chunkedHeader);
      for (uint32_t chunk = 0; chunk < numChunks; ++chunk) {
        const ChunkedZstdEntry entry{
            static_cast<uint32_t>(inputSizes[chunk]),
            static_cast<uint32_t>(outputSizes[chunk])};
        std::memcpy(hostGpuData_.data() + entryOffset, &entry, sizeof(entry));
        entryOffset += sizeof(entry);
      }
      const auto d2hStart = std::chrono::steady_clock::now();
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostGpuData_.data() + payloadOffset,
          compactGpuData.data(),
          compressedPayloadSize,
          cudaMemcpyDeviceToHost,
          stream.value()));
      stream.synchronize();
      cacheD2HNanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - d2hStart)
              .count(),
          std::memory_order_relaxed);
      cacheVersion_ = codec == CacheCodec::kZstd
          ? kCudfCacheVersionChunkedZstd
          : kCudfCacheVersionChunkedLz4;
    } else {
      hostGpuData_.resize(gpuDataSize);
      const auto d2hStart = std::chrono::steady_clock::now();
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostGpuData_.data(),
          packedColumns_->gpu_data->data(),
          gpuDataSize,
          cudaMemcpyDeviceToHost,
          stream.value()));
      stream.synchronize();
      cacheD2HNanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - d2hStart)
              .count(),
          std::memory_order_relaxed);
      cacheVersion_ = kCudfCacheVersionRaw;
    }
    cacheCompressionNanos.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - compressionStart)
            .count(),
        std::memory_order_relaxed);
    packedColumns_->gpu_data.reset();
    cacheAppendNanos.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - appendStart)
            .count(),
        std::memory_order_relaxed);
    cudfCache_ = true;
    return;
  }
  if (useParquetCacheFormat()) {
    // Spark's cache-write boundary may materialize a device result as an
    // ordinary Velox RowVector even when the scan and projection ran on the
    // GPU. Do not silently fall back to a Presto object stream: convert the
    // bounded page back to a cuDF table and emit the same standalone-Parquet
    // envelope used for a CudfVector input.
    GpuLockGuard gpuLock;
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    auto table = cudf_velox::with_arrow::toCudfTable(
        rowVector, veloxPool_.get(), stream, cudf_velox::get_output_mr());
    const auto tableView = table->view();
    std::vector<char> parquetBytes;
    auto metadata = cudf::io::table_input_metadata(tableView);
    const auto inputType = asRowType(rowVector->type());
    VELOX_CHECK_EQ(metadata.column_metadata.size(), inputType->size());
    for (auto index = 0; index < inputType->size(); ++index) {
      setParquetColumnMetadata(
          metadata.column_metadata[index],
          inputType->nameOf(index),
          inputType->childAt(index));
    }
    auto options = cudf::io::parquet_writer_options::builder(
                       cudf::io::sink_info(&parquetBytes), tableView)
                       .metadata(std::move(metadata))
                       .compression(cudf::io::compression_type::SNAPPY)
                       .stats_level(cudf::io::STATISTICS_NONE)
                       .utc_timestamps(true)
                       .build();
    cudf::io::write_parquet(options, stream);
    stream.synchronize();
    hostGpuData_.assign(parquetBytes.begin(), parquetBytes.end());
    cacheVersion_ = kCudfCacheVersionParquet;
    cudfCache_ = true;
    cacheRawBytes.fetch_add(rowVector->retainedSize(), std::memory_order_relaxed);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - appendStart)
            .count();
    cacheCompressionNanos.fetch_add(elapsed, std::memory_order_relaxed);
    cacheAppendNanos.fetch_add(elapsed, std::memory_order_relaxed);
    return;
  }
  VeloxColumnarBatchSerializer::append(batch);
}

void VeloxGpuColumnarBatchSerializer::appendMany(
    const std::vector<std::shared_ptr<ColumnarBatch>>& batches) {
  VELOX_CHECK(!batches.empty());
  if (batches.size() == 1) {
    append(batches.front());
    return;
  }
  if (!useParquetCacheFormat()) {
    // Grouping is currently enabled only for Parquet cache pages. Preserve
    // the CPU serializer's normal multi-append behavior for a defensive JNI
    // caller using another format.
    for (const auto& batch : batches) {
      VeloxColumnarBatchSerializer::append(batch);
    }
    return;
  }

  const auto appendStart = std::chrono::steady_clock::now();
  GpuLockGuard gpuLock;
  auto conversionStream = cudf_velox::cudfGlobalStreamPool().get_stream();
  std::vector<std::shared_ptr<cudf_velox::CudfVector>> cudfVectors;
  cudfVectors.reserve(batches.size());
  RowTypePtr inputType;
  uint64_t rawBytes = 0;
  for (const auto& batch : batches) {
    auto veloxBatch = VeloxColumnarBatch::from(veloxPool_.get(), batch);
    auto rowVector = veloxBatch->getRowVector();
    auto cudfVector = std::dynamic_pointer_cast<cudf_velox::CudfVector>(rowVector);
    if (!cudfVector) {
      auto table = cudf_velox::with_arrow::toCudfTable(
          rowVector,
          veloxPool_.get(),
          conversionStream,
          cudf_velox::get_output_mr());
      cudfVector = std::make_shared<cudf_velox::CudfVector>(
          veloxPool_.get(),
          rowVector->type(),
          rowVector->size(),
          std::move(table),
          conversionStream);
    }
    const auto batchType = asRowType(rowVector->type());
    if (!inputType) {
      inputType = batchType;
    } else {
      VELOX_CHECK(inputType->equivalent(*batchType));
    }
    rawBytes += cudfVector->retainedSize();
    cudfVectors.push_back(std::move(cudfVector));
  }

  conversionStream.synchronize();
  auto stream = cudfVectors.front()->stream();
  const auto firstTable = cudfVectors.front()->getTableView();
  VELOX_CHECK_EQ(firstTable.num_columns(), inputType->size());
  std::vector<char> parquetBytes;
  auto metadata = cudf::io::table_input_metadata(firstTable);
  VELOX_CHECK_EQ(metadata.column_metadata.size(), inputType->size());
  for (auto index = 0; index < inputType->size(); ++index) {
    setParquetColumnMetadata(
        metadata.column_metadata[index],
        inputType->nameOf(index),
        inputType->childAt(index));
  }
  auto options = cudf::io::chunked_parquet_writer_options::builder(
                     cudf::io::sink_info(&parquetBytes))
                     .metadata(std::move(metadata))
                     .compression(cudf::io::compression_type::SNAPPY)
                     // See the single-batch writer above.  Cache restore does not perform
                     // predicate pushdown, so row-group statistics are pure fill overhead.
                     .stats_level(cudf::io::STATISTICS_NONE)
                     .utc_timestamps(true)
                     .build();
  cudf::io::chunked_parquet_writer writer(options, stream);
  for (const auto& cudfVector : cudfVectors) {
    const auto table = cudfVector->getTableView();
    VELOX_CHECK_EQ(table.num_columns(), inputType->size());
    writer.write(table);
  }
  writer.close();
  stream.synchronize();
  hostGpuData_.assign(parquetBytes.begin(), parquetBytes.end());
  cacheVersion_ = kCudfCacheVersionParquet;
  cudfCache_ = true;
  cacheRawBytes.fetch_add(rawBytes, std::memory_order_relaxed);
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - appendStart)
          .count();
  cacheCompressionNanos.fetch_add(elapsed, std::memory_order_relaxed);
  cacheAppendNanos.fetch_add(elapsed, std::memory_order_relaxed);
}

int64_t VeloxGpuColumnarBatchSerializer::maxSerializedSize() {
  if (!cudfCache_) {
    return VeloxColumnarBatchSerializer::maxSerializedSize();
  }
  return sizeof(CudfCacheHeader) +
      (packedColumns_ == nullptr ? 0 : packedColumns_->metadata->size()) +
      hostGpuData_.size();
}

void VeloxGpuColumnarBatchSerializer::serializeTo(
    uint8_t* address,
    int64_t size) {
  if (!cudfCache_) {
    VeloxColumnarBatchSerializer::serializeTo(address, size);
    return;
  }
  const CudfCacheHeader header{
      kCudfCacheMagic,
      cacheVersion_,
      (cacheVersion_ == kCudfCacheVersionZstd ||
       cacheVersion_ == kCudfCacheVersionLz4 ||
       cacheVersion_ == kCudfCacheVersionChunkedZstd ||
       cacheVersion_ == kCudfCacheVersionChunkedLz4)
          ? uncompressedGpuDataSize_
          : 0,
      packedColumns_ == nullptr ? 0 : packedColumns_->metadata->size(),
      hostGpuData_.size()};
  const auto required = static_cast<int64_t>(
      sizeof(header) + header.metadataSize + header.gpuDataSize);
  GLUTEN_CHECK(
      size >= required,
      "The target cuDF cache buffer is insufficient: " +
          std::to_string(size) + " vs. " + std::to_string(required));
  const auto copyStart = std::chrono::steady_clock::now();
  std::memcpy(address, &header, sizeof(header));
  if (header.metadataSize != 0) {
    std::memcpy(
        address + sizeof(header),
        packedColumns_->metadata->data(),
        header.metadataSize);
  }
  std::memcpy(
      address + sizeof(header) + header.metadataSize,
      hostGpuData_.data(),
      header.gpuDataSize);
  cacheSerializeCopyNanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - copyStart)
          .count(),
      std::memory_order_relaxed);
  cacheSerializeBytes.fetch_add(header.gpuDataSize, std::memory_order_relaxed);
  const auto batches =
      cacheSerializeBatches.fetch_add(1, std::memory_order_relaxed) + 1;
  if ((batches & 0x7f) == 0) {
    LOG(WARNING) << "GPU cache native serialization: batches=" << batches
              << " rawBytes="
              << cacheRawBytes.load(std::memory_order_relaxed)
              << " compressedBytes="
              << cacheSerializeBytes.load(std::memory_order_relaxed)
              << " appendMs="
              << (cacheAppendNanos.load(std::memory_order_relaxed) /
                  1000000.0)
              << " packMs="
              << (cachePackNanos.load(std::memory_order_relaxed) /
                  1000000.0)
              << " d2hMs="
              << (cacheD2HNanos.load(std::memory_order_relaxed) / 1000000.0)
              << " compressionMs="
              << (cacheCompressionNanos.load(std::memory_order_relaxed) /
                  1000000.0)
              << " serializeCopyMs="
              << (cacheSerializeCopyNanos.load(std::memory_order_relaxed) /
                  1000000.0);
  }
  packedColumns_.reset();
  hostGpuData_.clear();
  cudfCache_ = false;
}

std::shared_ptr<ColumnarBatch> VeloxGpuColumnarBatchSerializer::deserialize(uint8_t* data, int32_t size) {
  if (size >= static_cast<int32_t>(sizeof(CudfCacheHeader))) {
    CudfCacheHeader header;
    std::memcpy(&header, data, sizeof(header));
    if (header.magic == kCudfCacheMagic) {
      VELOX_CHECK(
          header.version == kCudfCacheVersionRaw ||
          header.version == kCudfCacheVersionZstd ||
          header.version == kCudfCacheVersionLz4 ||
          header.version == kCudfCacheVersionChunkedZstd ||
          header.version == kCudfCacheVersionChunkedLz4 ||
          header.version == kCudfCacheVersionParquet);
      VELOX_CHECK_EQ(
          static_cast<uint64_t>(size),
          sizeof(header) + header.metadataSize + header.gpuDataSize);
      if (header.version == kCudfCacheVersionParquet) {
        const auto deserializeStart = std::chrono::steady_clock::now();
        GpuLockGuard gpuLock;
        auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
        const auto* parquetData =
            data + sizeof(header) + header.metadataSize;
        auto options = cudf::io::parquet_reader_options::builder(
                           cudf::io::source_info(cudf::host_span<std::byte const>(
                               reinterpret_cast<const std::byte*>(parquetData),
                               header.gpuDataSize)))
                           .build();
        auto result = cudf::io::read_parquet(
            options, stream, cudf_velox::get_output_mr());
        // The caller owns the host-backed cache page and releases it as soon
        // as deserialize() returns.  libcudf may enqueue host-to-device reads
        // from source_info on this stream, so make that ownership boundary
        // explicit before returning a batch which no longer retains `data`.
        // Without this synchronization a later operator can observe a sticky
        // cudaErrorIllegalAddress after DiskStore reuses/frees the page.
        stream.synchronize();
        const auto numRows = result.tbl->num_rows();
        recordParquetCacheDeserialize(
            header.gpuDataSize,
            numRows,
            rowType_->size(),
            rowType_->size(),
            false,
            deserializeStart);
        auto vector = std::make_shared<cudf_velox::CudfVector>(
            veloxPool_.get(), rowType_, numRows, std::move(result.tbl), stream);
        return std::make_shared<VeloxColumnarBatch>(
            std::move(vector), rowType_->size());
      }
      auto metadata = std::make_unique<std::vector<uint8_t>>(
          data + sizeof(header),
          data + sizeof(header) + header.metadataSize);
      GpuLockGuard gpuLock;
      auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
      std::vector<uint8_t> decompressed;
      std::unique_ptr<rmm::device_buffer> gpuData;
      const uint8_t* hostGpuData =
          data + sizeof(header) + header.metadataSize;
      uint64_t deviceDataSize = header.gpuDataSize;
      if (header.version == kCudfCacheVersionZstd) {
        deviceDataSize = header.reserved;
        decompressed.resize(deviceDataSize);
        const auto decompressedSize = ZSTD_decompress(
            decompressed.data(),
            decompressed.size(),
            hostGpuData,
            header.gpuDataSize);
        VELOX_CHECK(
            !ZSTD_isError(decompressedSize),
            "Failed to decompress cuDF cache page: {}",
            ZSTD_getErrorName(decompressedSize));
        VELOX_CHECK_EQ(decompressedSize, deviceDataSize);
        hostGpuData = decompressed.data();
      } else if (header.version == kCudfCacheVersionLz4) {
        deviceDataSize = header.reserved;
        VELOX_CHECK_LE(
            deviceDataSize,
            static_cast<uint64_t>(std::numeric_limits<int>::max()));
        VELOX_CHECK_LE(
            header.gpuDataSize,
            static_cast<uint64_t>(std::numeric_limits<int>::max()));
        decompressed.resize(deviceDataSize);
        const auto decompressedSize = LZ4_decompress_safe(
            reinterpret_cast<const char*>(hostGpuData),
            reinterpret_cast<char*>(decompressed.data()),
            static_cast<int>(header.gpuDataSize),
            static_cast<int>(deviceDataSize));
        VELOX_CHECK_EQ(decompressedSize, deviceDataSize);
        hostGpuData = decompressed.data();
      } else if (header.version == kCudfCacheVersionChunkedZstd) {
        deviceDataSize = header.reserved;
        VELOX_CHECK_GE(header.gpuDataSize, sizeof(ChunkedZstdHeader));
        ChunkedZstdHeader chunkedHeader;
        std::memcpy(&chunkedHeader, hostGpuData, sizeof(chunkedHeader));
        VELOX_CHECK_GT(chunkedHeader.numChunks, 0);
        VELOX_CHECK_LE(chunkedHeader.numChunks, kMaxCacheChunks);
        const auto entriesSize =
            chunkedHeader.numChunks * sizeof(ChunkedZstdEntry);
        const auto payloadOffset =
            sizeof(ChunkedZstdHeader) + entriesSize;
        VELOX_CHECK_GE(header.gpuDataSize, payloadOffset);

        std::vector<ChunkedZstdEntry> entries(chunkedHeader.numChunks);
        std::vector<size_t> compressedSizes(chunkedHeader.numChunks);
        std::vector<size_t> rawSizes(chunkedHeader.numChunks);
        size_t entryOffset = sizeof(ChunkedZstdHeader);
        size_t packedOffset = 0;
        size_t rawOffset = 0;
        size_t alignedCompressedSize = 0;
        size_t maxRawChunkSize = 0;
        for (uint32_t chunk = 0; chunk < chunkedHeader.numChunks; ++chunk) {
          auto& entry = entries[chunk];
          std::memcpy(&entry, hostGpuData + entryOffset, sizeof(entry));
          VELOX_CHECK_GT(entry.compressedSize, 0);
          VELOX_CHECK_GT(entry.rawSize, 0);
          VELOX_CHECK_LE(
              payloadOffset + packedOffset + entry.compressedSize,
              header.gpuDataSize);
          VELOX_CHECK_LE(rawOffset + entry.rawSize, deviceDataSize);
          compressedSizes[chunk] = entry.compressedSize;
          rawSizes[chunk] = entry.rawSize;
          packedOffset += entry.compressedSize;
          rawOffset += entry.rawSize;
          alignedCompressedSize +=
              (static_cast<size_t>(entry.compressedSize) + 7) & ~size_t{7};
          maxRawChunkSize =
              std::max(maxRawChunkSize, static_cast<size_t>(entry.rawSize));
          entryOffset += sizeof(entry);
        }
        VELOX_CHECK_EQ(payloadOffset + packedOffset, header.gpuDataSize);
        VELOX_CHECK_EQ(rawOffset, deviceDataSize);

        auto mr = cudf_velox::get_output_mr();
        rmm::device_buffer packedCompressedGpu(
            packedOffset, stream, mr);
        rmm::device_buffer alignedCompressedGpu(
            alignedCompressedSize, stream, mr);
        gpuData = std::make_unique<rmm::device_buffer>(
            deviceDataSize, stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            packedCompressedGpu.data(),
            hostGpuData + payloadOffset,
            packedOffset,
            cudaMemcpyHostToDevice,
            stream.value()));

        std::vector<void const*> packedSourcePointers(
            chunkedHeader.numChunks);
        std::vector<void*> alignedDestinationPointers(
            chunkedHeader.numChunks);
        std::vector<void*> outputPointers(chunkedHeader.numChunks);
        size_t alignedOffset = 0;
        packedOffset = 0;
        rawOffset = 0;
        for (uint32_t chunk = 0; chunk < chunkedHeader.numChunks; ++chunk) {
          packedSourcePointers[chunk] =
              static_cast<const uint8_t*>(packedCompressedGpu.data()) +
              packedOffset;
          alignedDestinationPointers[chunk] =
              static_cast<uint8_t*>(alignedCompressedGpu.data()) +
              alignedOffset;
          outputPointers[chunk] =
              static_cast<uint8_t*>(gpuData->data()) + rawOffset;
          packedOffset += entries[chunk].compressedSize;
          alignedOffset +=
              (static_cast<size_t>(entries[chunk].compressedSize) + 7) &
              ~size_t{7};
          rawOffset += entries[chunk].rawSize;
        }
        cudaMemcpyAttributes copyAttributes{};
        copyAttributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t copyAttributesIndex = 0;
        CUDF_CUDA_TRY(cudaMemcpyBatchAsync(
            alignedDestinationPointers.data(),
            packedSourcePointers.data(),
            compressedSizes.data(),
            chunkedHeader.numChunks,
            &copyAttributes,
            &copyAttributesIndex,
            1,
            stream.value()));

        rmm::device_buffer inputPointersGpu(
            chunkedHeader.numChunks * sizeof(void*), stream, mr);
        rmm::device_buffer inputSizesGpu(
            chunkedHeader.numChunks * sizeof(size_t), stream, mr);
        rmm::device_buffer outputPointersGpu(
            chunkedHeader.numChunks * sizeof(void*), stream, mr);
        rmm::device_buffer outputCapacitiesGpu(
            chunkedHeader.numChunks * sizeof(size_t), stream, mr);
        rmm::device_buffer outputSizesGpu(
            chunkedHeader.numChunks * sizeof(size_t), stream, mr);
        rmm::device_buffer statusesGpu(
            chunkedHeader.numChunks * sizeof(nvcompStatus_t), stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            inputPointersGpu.data(),
            alignedDestinationPointers.data(),
            chunkedHeader.numChunks * sizeof(void*),
            cudaMemcpyHostToDevice,
            stream.value()));
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            inputSizesGpu.data(),
            compressedSizes.data(),
            chunkedHeader.numChunks * sizeof(size_t),
            cudaMemcpyHostToDevice,
            stream.value()));
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            outputPointersGpu.data(),
            outputPointers.data(),
            chunkedHeader.numChunks * sizeof(void*),
            cudaMemcpyHostToDevice,
            stream.value()));
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            outputCapacitiesGpu.data(),
            rawSizes.data(),
            chunkedHeader.numChunks * sizeof(size_t),
            cudaMemcpyHostToDevice,
            stream.value()));

        const auto opts = nvcompBatchedZstdDecompressDefaultOpts;
        size_t tempBytes = 0;
        auto nvcompStatus = nvcompBatchedZstdDecompressGetTempSizeAsync(
            chunkedHeader.numChunks,
            maxRawChunkSize,
            opts,
            &tempBytes,
            deviceDataSize);
        VELOX_CHECK_EQ(
            static_cast<int>(nvcompStatus),
            static_cast<int>(nvcompSuccess));
        rmm::device_buffer tempGpuData(tempBytes, stream, mr);
        nvcompStatus = nvcompBatchedZstdDecompressAsync(
            static_cast<void const* const*>(inputPointersGpu.data()),
            static_cast<size_t const*>(inputSizesGpu.data()),
            static_cast<size_t const*>(outputCapacitiesGpu.data()),
            static_cast<size_t*>(outputSizesGpu.data()),
            chunkedHeader.numChunks,
            tempGpuData.data(),
            tempBytes,
            static_cast<void* const*>(outputPointersGpu.data()),
            opts,
            static_cast<nvcompStatus_t*>(statusesGpu.data()),
            stream.value());
        VELOX_CHECK_EQ(
            static_cast<int>(nvcompStatus),
            static_cast<int>(nvcompSuccess));
        std::vector<size_t> outputSizes(chunkedHeader.numChunks);
        std::vector<nvcompStatus_t> statuses(chunkedHeader.numChunks);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            outputSizes.data(),
            outputSizesGpu.data(),
            chunkedHeader.numChunks * sizeof(size_t),
            cudaMemcpyDeviceToHost,
            stream.value()));
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            statuses.data(),
            statusesGpu.data(),
            chunkedHeader.numChunks * sizeof(nvcompStatus_t),
            cudaMemcpyDeviceToHost,
            stream.value()));
        stream.synchronize();
        for (uint32_t chunk = 0; chunk < chunkedHeader.numChunks; ++chunk) {
          VELOX_CHECK_EQ(
              static_cast<int>(statuses[chunk]),
              static_cast<int>(nvcompSuccess));
          VELOX_CHECK_EQ(outputSizes[chunk], rawSizes[chunk]);
        }
      } else if (header.version == kCudfCacheVersionChunkedLz4) {
        deviceDataSize = header.reserved;
        decompressed.resize(deviceDataSize);
        VELOX_CHECK_GE(
            header.gpuDataSize, sizeof(ChunkedZstdHeader));
        ChunkedZstdHeader chunkedHeader;
        std::memcpy(&chunkedHeader, hostGpuData, sizeof(chunkedHeader));
        VELOX_CHECK_GT(chunkedHeader.numChunks, 0);
        VELOX_CHECK_LE(chunkedHeader.numChunks, kMaxCacheChunks);
        const auto entriesSize =
            chunkedHeader.numChunks * sizeof(ChunkedZstdEntry);
        VELOX_CHECK_GE(
            header.gpuDataSize,
            sizeof(ChunkedZstdHeader) + entriesSize);
        struct DecodeChunk {
          const uint8_t* compressedData;
          uint8_t* rawData;
          ChunkedZstdEntry entry;
        };
        std::vector<DecodeChunk> chunks;
        chunks.reserve(chunkedHeader.numChunks);
        size_t entryOffset = sizeof(ChunkedZstdHeader);
        size_t compressedOffset =
            sizeof(ChunkedZstdHeader) + entriesSize;
        size_t rawOffset = 0;
        for (uint32_t chunk = 0; chunk < chunkedHeader.numChunks; ++chunk) {
          ChunkedZstdEntry entry;
          std::memcpy(
              &entry, hostGpuData + entryOffset, sizeof(entry));
          VELOX_CHECK_LE(
              compressedOffset + entry.compressedSize,
              header.gpuDataSize);
          VELOX_CHECK_LE(
              rawOffset + entry.rawSize, deviceDataSize);
          VELOX_CHECK_LE(
              entry.compressedSize,
              static_cast<uint32_t>(std::numeric_limits<int>::max()));
          VELOX_CHECK_LE(
              entry.rawSize,
              static_cast<uint32_t>(std::numeric_limits<int>::max()));
          chunks.push_back(DecodeChunk{
              hostGpuData + compressedOffset,
              decompressed.data() + rawOffset,
              entry});
          entryOffset += sizeof(entry);
          compressedOffset += entry.compressedSize;
          rawOffset += entry.rawSize;
        }
        VELOX_CHECK_EQ(compressedOffset, header.gpuDataSize);
        VELOX_CHECK_EQ(rawOffset, deviceDataSize);

        const auto numWorkers = std::min<size_t>(
            kCacheDecompressionWorkers, chunks.size());
        std::vector<std::future<void>> futures;
        futures.reserve(numWorkers);
        for (size_t worker = 0; worker < numWorkers; ++worker) {
          futures.emplace_back(std::async(
              std::launch::async,
              [worker, numWorkers, &chunks]() {
                for (size_t index = worker; index < chunks.size();
                     index += numWorkers) {
                  const auto& chunk = chunks[index];
                  const auto decompressedSize = LZ4_decompress_safe(
                      reinterpret_cast<const char*>(chunk.compressedData),
                      reinterpret_cast<char*>(chunk.rawData),
                      static_cast<int>(chunk.entry.compressedSize),
                      static_cast<int>(chunk.entry.rawSize));
                  VELOX_CHECK_GT(
                      decompressedSize,
                      0,
                      "Failed to decompress cuDF cache LZ4 chunk");
                  VELOX_CHECK_EQ(
                      static_cast<size_t>(decompressedSize),
                      static_cast<size_t>(chunk.entry.rawSize));
                }
              }));
        }
        for (auto& future : futures) {
          future.get();
        }
        hostGpuData = decompressed.data();
      }
      if (gpuData == nullptr) {
        gpuData = std::make_unique<rmm::device_buffer>(
            deviceDataSize, stream, cudf_velox::get_output_mr());
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            gpuData->data(),
            hostGpuData,
            deviceDataSize,
            cudaMemcpyHostToDevice,
            stream.value()));
      }
      cudf::packed_columns columns(std::move(metadata), std::move(gpuData));
      auto tableView = cudf::unpack(columns);
      const auto numRows = tableView.num_rows();
      auto table = std::make_unique<cudf::packed_table>(
          cudf::packed_table{tableView, std::move(columns)});
      auto vector = std::make_shared<cudf_velox::CudfVector>(
          veloxPool_.get(), rowType_, numRows, std::move(table), stream);
      return std::make_shared<VeloxColumnarBatch>(
          std::move(vector), rowType_->size());
    }
  }
  auto vb = VeloxColumnarBatchSerializer::deserialize(data, size);
  auto rv = dynamic_pointer_cast<VeloxColumnarBatch>(vb)->getRowVector();
  auto numRows = rv->size();

  GpuLockGuard gpuLock;
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  // IBM-baseline toCudfTable requires an mr argument; use the cudf output mr.
  auto table = cudf_velox::with_arrow::toCudfTable(
      rv, veloxPool_.get(), stream, cudf_velox::get_output_mr());
  auto vector = std::make_shared<cudf_velox::CudfVector>(
      veloxPool_.get(), rowType_, numRows, std::move(table), stream);
  return std::make_shared<VeloxColumnarBatch>(vector, vb->numColumns());
}

std::shared_ptr<ColumnarBatch> VeloxGpuColumnarBatchSerializer::deserializeSelected(
    uint8_t* data,
    int32_t size,
    const std::vector<int32_t>& columnIndices) {
  if (size >= static_cast<int32_t>(sizeof(CudfCacheHeader))) {
    CudfCacheHeader header;
    std::memcpy(&header, data, sizeof(header));
    if (header.magic == kCudfCacheMagic &&
        header.version == kCudfCacheVersionParquet) {
      const auto deserializeStart = std::chrono::steady_clock::now();
      VELOX_CHECK_EQ(
          static_cast<uint64_t>(size),
          sizeof(header) + header.metadataSize + header.gpuDataSize);
      GpuLockGuard gpuLock;
      auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
      const auto* parquetData =
          data + sizeof(header) + header.metadataSize;
      std::vector<cudf::size_type> cudfIndices;
      cudfIndices.reserve(columnIndices.size());
      for (const auto index : columnIndices) {
        VELOX_CHECK_GE(index, 0);
        VELOX_CHECK_LT(index, rowType_->size());
        cudfIndices.push_back(index);
      }
      auto options = cudf::io::parquet_reader_options::builder(
                         cudf::io::source_info(cudf::host_span<std::byte const>(
                             reinterpret_cast<const std::byte*>(parquetData),
                             header.gpuDataSize)))
                         .column_indices(std::move(cudfIndices))
                         .build();
      auto result = cudf::io::read_parquet(
          options, stream, cudf_velox::get_output_mr());
      // `data` belongs to CachedColumnarBatch and is released immediately by
      // the Scala iterator after this JNI call.  Complete every asynchronous
      // source read before crossing that lifetime boundary.
      stream.synchronize();
      const auto outputType = selectRowType(rowType_, columnIndices);
      const auto numRows = result.tbl->num_rows();
      recordParquetCacheDeserialize(
          header.gpuDataSize,
          numRows,
          columnIndices.size(),
          rowType_->size(),
          true,
          deserializeStart);
      auto vector = std::make_shared<cudf_velox::CudfVector>(
          veloxPool_.get(), outputType, numRows, std::move(result.tbl), stream);
      return std::make_shared<VeloxColumnarBatch>(
          std::move(vector), outputType->size());
    }
  }
  auto batch = std::dynamic_pointer_cast<VeloxColumnarBatch>(
      deserialize(data, size));
  VELOX_CHECK_NOT_NULL(batch, "Expected a VeloxColumnarBatch");
  return batch->select(veloxPool_.get(), columnIndices);
}

std::shared_ptr<ColumnarBatch>
VeloxGpuColumnarBatchSerializer::deserializeParquetFile(
    const std::string& path) {
  const auto deserializeStart = std::chrono::steady_clock::now();
  GpuLockGuard gpuLock;
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  auto options = cudf::io::parquet_reader_options::builder(
                     cudf::io::source_info(path))
                     .build();
  auto result = cudf::io::read_parquet(
      options, stream, cudf_velox::get_output_mr());
  stream.synchronize();
  const auto numRows = result.tbl->num_rows();
  recordParquetCacheDeserialize(
      std::filesystem::file_size(path),
      numRows,
      rowType_->size(),
      rowType_->size(),
      false,
      deserializeStart);
  auto vector = std::make_shared<cudf_velox::CudfVector>(
      veloxPool_.get(), rowType_, numRows, std::move(result.tbl), stream);
  return std::make_shared<VeloxColumnarBatch>(
      std::move(vector), rowType_->size());
}

std::shared_ptr<ColumnarBatch>
VeloxGpuColumnarBatchSerializer::deserializeParquetFileSelected(
    const std::string& path,
    const std::vector<int32_t>& columnIndices) {
  const auto deserializeStart = std::chrono::steady_clock::now();
  std::vector<cudf::size_type> cudfIndices;
  cudfIndices.reserve(columnIndices.size());
  for (const auto index : columnIndices) {
    VELOX_CHECK_GE(index, 0);
    VELOX_CHECK_LT(index, rowType_->size());
    cudfIndices.push_back(index);
  }
  GpuLockGuard gpuLock;
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  auto options = cudf::io::parquet_reader_options::builder(
                     cudf::io::source_info(path))
                     .column_indices(std::move(cudfIndices))
                     .build();
  auto result = cudf::io::read_parquet(
      options, stream, cudf_velox::get_output_mr());
  stream.synchronize();
  const auto outputType = selectRowType(rowType_, columnIndices);
  const auto numRows = result.tbl->num_rows();
  recordParquetCacheDeserialize(
      std::filesystem::file_size(path),
      numRows,
      columnIndices.size(),
      rowType_->size(),
      true,
      deserializeStart);
  auto vector = std::make_shared<cudf_velox::CudfVector>(
      veloxPool_.get(), outputType, numRows, std::move(result.tbl), stream);
  return std::make_shared<VeloxColumnarBatch>(
      std::move(vector), outputType->size());
}

} // namespace gluten
