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

#include "MppQueryCoordinator.h"

#include <fmt/format.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/SerializedPage.h"
#include "velox/exec/Task.h"
#include "velox/vector/VectorStream.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace gluten {

namespace {

/// Task ID prefix for local GPU exchange. The "local://" scheme is recognized
/// by Velox's built-in LocalExchangeSource (used in testing). A production
/// GPU exchange would use a custom scheme like "gpu-local://".
constexpr const char* kTaskIdPrefix = "local://";

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MppQueryCoordinator::MppQueryCoordinator(
    std::string queryId,
    std::vector<MppFragmentSpec> fragments,
    std::vector<MppExchangeSpec> exchanges,
    std::shared_ptr<core::QueryCtx> queryCtx,
    folly::Executor* executor)
    : queryId_(std::move(queryId)),
      fragmentSpecs_(std::move(fragments)),
      exchangeSpecs_(std::move(exchanges)),
      queryCtx_(std::move(queryCtx)),
      executor_(executor),
      bufferManager_(OutputBufferManager::getInstanceRef()) {
  VELOX_CHECK(!fragmentSpecs_.empty(), "At least one fragment is required");
  VELOX_CHECK(queryCtx_ != nullptr, "QueryCtx must not be null");
  VELOX_CHECK(executor_ != nullptr, "Executor must not be null");

  // Validate that fragment ids form a contiguous 0-based sequence so we can
  // use them as vector indices.
  for (size_t i = 0; i < fragmentSpecs_.size(); ++i) {
    VELOX_CHECK_EQ(
        fragmentSpecs_[i].id,
        static_cast<int32_t>(i),
        "Fragment ids must be contiguous starting from 0, got {} at index {}",
        fragmentSpecs_[i].id,
        i);
  }
}

std::shared_ptr<MppQueryCoordinator> MppQueryCoordinator::create(
    const std::string& queryId,
    std::vector<MppFragmentSpec> fragments,
    std::vector<MppExchangeSpec> exchanges,
    std::shared_ptr<core::QueryCtx> queryCtx,
    folly::Executor* executor) {
  // Using new + shared_ptr because the constructor is private.
  return std::shared_ptr<MppQueryCoordinator>(new MppQueryCoordinator(
      queryId,
      std::move(fragments),
      std::move(exchanges),
      std::move(queryCtx),
      executor));
}

MppQueryCoordinator::~MppQueryCoordinator() {
  // Best-effort cleanup: abort any tasks still running.
  if (started_) {
    try {
      abort();
    } catch (...) {
      // Swallow exceptions in destructor.
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string MppQueryCoordinator::makeTaskId(int32_t fragmentId) const {
  return fmt::format("{}{}-{}", kTaskIdPrefix, queryId_, fragmentId);
}

bool MppQueryCoordinator::isTerminalState(TaskState state) {
  return state == TaskState::kFinished || state == TaskState::kCanceled ||
      state == TaskState::kAborted || state == TaskState::kFailed;
}

// ---------------------------------------------------------------------------
// start()
// ---------------------------------------------------------------------------

void MppQueryCoordinator::start() {
  VELOX_CHECK(!started_, "MppQueryCoordinator already started");
  started_ = true;

  // Phase 1: Create and start a Velox Task for each fragment.
  //
  // Task::create() + Task::start() internally handles:
  //   - initializePartitionOutput() -> OutputBufferManager::initializeTask()
  //     (for fragments whose plan ends with PartitionedOutputNode)
  //   - createExchangeClientLocked() -> creates ExchangeClient for any
  //     ExchangeNode in the plan
  //
  // We start ALL fragments before wiring exchanges so that producer tasks'
  // output buffers are already registered in OutputBufferManager when
  // consumers try to fetch data.
  tasks_.resize(fragmentSpecs_.size());
  for (auto& spec : fragmentSpecs_) {
    auto taskId = makeTaskId(spec.id);

    auto task = Task::create(
        taskId,
        spec.planFragment,
        spec.destination,
        queryCtx_,
        Task::ExecutionMode::kParallel);

    task->start(spec.numDrivers);
    tasks_[spec.id] = std::move(task);

    LOG(INFO) << "MppQueryCoordinator: started fragment " << spec.id
              << " as task " << taskId
              << " with " << spec.numDrivers << " drivers";
  }

  // Phase 2: Wire exchanges by adding RemoteConnectorSplits.
  //
  // For each exchange, we tell the consumer task where to find the producer's
  // output. This mirrors the pattern in MultiFragmentTest::addRemoteSplits():
  //   task->addSplit(exchangeNodeId, Split(RemoteConnectorSplit(producerTaskId)))
  //   task->noMoreSplits(exchangeNodeId)
  for (auto& exchange : exchangeSpecs_) {
    auto producerTaskId = makeTaskId(exchange.producerFragmentId);
    auto& consumerTask = tasks_[exchange.consumerFragmentId];
    VELOX_CHECK(
        consumerTask != nullptr,
        "Consumer fragment {} not found for exchange {}",
        exchange.consumerFragmentId,
        exchange.id);

    consumerTask->addSplit(
        exchange.exchangeNodeId,
        Split(std::make_shared<RemoteConnectorSplit>(producerTaskId)));
    consumerTask->noMoreSplits(exchange.exchangeNodeId);

    LOG(INFO) << "MppQueryCoordinator: wired exchange " << exchange.id
              << " producer=" << producerTaskId
              << " -> consumer fragment " << exchange.consumerFragmentId
              << " exchange node " << exchange.exchangeNodeId;
  }
}

// ---------------------------------------------------------------------------
// next() - fetch output from the root fragment
// ---------------------------------------------------------------------------

bool MppQueryCoordinator::fetchNextOutputPage(
    std::vector<std::unique_ptr<folly::IOBuf>>& iobufs) {
  auto rootTaskId = makeTaskId(0);
  constexpr int32_t kDestination = 0;
  constexpr uint64_t kMaxBytes = std::numeric_limits<uint64_t>::max();

  bool complete = false;
  auto dataPromise =
      ContinuePromise("MppQueryCoordinator::fetchNextOutputPage");

  auto ok = bufferManager_->getData(
      rootTaskId,
      kDestination,
      kMaxBytes,
      outputSequence_,
      [&](std::vector<std::unique_ptr<folly::IOBuf>> pages,
          int64_t inSequence,
          std::vector<int64_t> /*remainingBytes*/) {
        for (auto& page : pages) {
          if (page != nullptr) {
            ++inSequence;
            iobufs.push_back(std::move(page));
          } else {
            // nullptr page signals end-of-stream.
            complete = true;
          }
        }
        outputSequence_ = inSequence;
        dataPromise.setValue();
      });

  if (!ok) {
    // Task not found in OutputBufferManager. This can happen if the root
    // task finished without producing any output or was already cleaned up.
    noMoreData_ = true;
    return false;
  }

  // Block until the callback fires (either with data or end-of-stream).
  dataPromise.getSemiFuture().wait();

  if (complete) {
    noMoreData_ = true;
    // Acknowledge receipt and clean up the output buffer.
    bufferManager_->acknowledge(rootTaskId, kDestination, outputSequence_);
    bufferManager_->deleteResults(rootTaskId, kDestination);
  }

  // Return true if we got at least one data page.
  return !iobufs.empty();
}

RowVectorPtr MppQueryCoordinator::next() {
  VELOX_CHECK(started_, "Must call start() before next()");

  if (noMoreData_) {
    return nullptr;
  }

  // Keep fetching until we get data pages or hit end-of-stream.
  while (!noMoreData_) {
    std::vector<std::unique_ptr<folly::IOBuf>> iobufs;
    bool gotData = fetchNextOutputPage(iobufs);

    if (!gotData) {
      return nullptr;
    }

    // Deserialize the first IOBuf into a RowVector.
    // Each IOBuf corresponds to one serialized page from PartitionedOutput.
    // For simplicity, we deserialize one page per call. If multiple pages
    // were received in one getData callback, the remaining pages will be
    // fetched in subsequent next() calls (they are already acknowledged
    // by sequence advancement).
    for (auto& iobuf : iobufs) {
      auto page = std::make_unique<PrestoSerializedPage>(std::move(iobuf));
      auto inputStream = page->prepareStreamForDeserialize();

      // Get the output type from the root fragment's plan node.
      auto outputType = std::dynamic_pointer_cast<const RowType>(
          fragmentSpecs_[0].planFragment.planNode->outputType());
      VELOX_CHECK(
          outputType != nullptr, "Root fragment must have RowType output");

      RowVectorPtr result;
      VectorStreamGroup::read(
          inputStream.get(),
          queryCtx_->pool(),
          outputType,
          getVectorSerde(),
          &result,
          /*options=*/nullptr);
      return result;
    }
  }

  return nullptr;
}

// ---------------------------------------------------------------------------
// isFinished()
// ---------------------------------------------------------------------------

bool MppQueryCoordinator::isFinished() const {
  if (!started_) {
    return false;
  }
  for (auto& task : tasks_) {
    if (!isTerminalState(task->state())) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// abort()
// ---------------------------------------------------------------------------

void MppQueryCoordinator::abort() {
  if (!started_) {
    return;
  }

  for (auto& task : tasks_) {
    if (!isTerminalState(task->state())) {
      task->requestAbort();
    }
  }
}

// ---------------------------------------------------------------------------
// waitForCompletion()
// ---------------------------------------------------------------------------

void MppQueryCoordinator::waitForCompletion() {
  VELOX_CHECK(started_, "Must call start() before waitForCompletion()");

  std::exception_ptr firstError;

  for (auto& task : tasks_) {
    if (isTerminalState(task->state())) {
      // Already done; check for error.
      if (auto error = task->error()) {
        if (!firstError) {
          firstError = error;
        }
      }
      continue;
    }

    // Block until this task reaches a terminal state.
    auto future = task->taskCompletionFuture();
    std::move(future).wait();

    if (auto error = task->error()) {
      if (!firstError) {
        firstError = error;
      }
    }
  }

  // Clean up output buffer entries for all tasks.
  for (auto& task : tasks_) {
    bufferManager_->removeTask(task->taskId());
  }

  if (firstError) {
    std::rethrow_exception(firstError);
  }
}

} // namespace gluten
