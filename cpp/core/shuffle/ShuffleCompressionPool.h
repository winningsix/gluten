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

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace gluten {

// Global shared thread pool for async shuffle write compression.
// Call setNumThreads() before first instance() to override the
// default.  Controlled by spark.gluten.sql.columnar.shuffle
// .compression.threads.
class ShuffleCompressionPool {
 public:
  static constexpr int kDefaultNumThreads = 20;

  // Must be called before the first instance() call.
  // Later calls are silently ignored (pool already created).
  static void setNumThreads(int n) {
    numThreads() = n;
  }

  static ShuffleCompressionPool& instance() {
    static ShuffleCompressionPool pool(numThreads());
    return pool;
  }

  template <typename F>
  auto submit(F&& func) -> std::future<decltype(func())> {
    using R = decltype(func());
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(func));
    auto future = task->get_future();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      queueSpaceCv_.wait(lock, [this] { return stopped_ || queue_.size() < maxQueuedTasks_; });
      if (stopped_) {
        throw std::runtime_error("Shuffle compression pool is stopped");
      }
      queue_.push([task]() { (*task)(); });
    }
    cv_.notify_one();
    return future;
  }

  int workerCount() const {
    return static_cast<int>(workers_.size());
  }

  ~ShuffleCompressionPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    cv_.notify_all();
    queueSpaceCv_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

 private:
  using Task = std::function<void()>;

  static int& numThreads() {
    static int n = kDefaultNumThreads;
    return n;
  }

  explicit ShuffleCompressionPool(int numThreads) : maxQueuedTasks_(std::max(numThreads, 1)), stopped_(false) {
    const auto workerThreads = std::max(numThreads, 1);
    for (int i = 0; i < workerThreads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          Task task;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stopped_ || !queue_.empty(); });
            if (stopped_ && queue_.empty()) {
              return;
            }
            task = std::move(queue_.front());
            queue_.pop();
          }
          queueSpaceCv_.notify_one();
          task();
        }
      });
    }
  }

  ShuffleCompressionPool(const ShuffleCompressionPool&) = delete;
  ShuffleCompressionPool& operator=(const ShuffleCompressionPool&) = delete;

  std::vector<std::thread> workers_;
  std::queue<Task> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable queueSpaceCv_;
  const size_t maxQueuedTasks_;
  bool stopped_;
};

} // namespace gluten
