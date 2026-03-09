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

#include "compute/ResultIterator.h"
#include "cudf/GpuLock.h"
#include "memory/GpuBufferColumnarBatch.h"
#include "memory/VeloxColumnarBatch.h"
#include "utils/GpuBufferBatchResizer.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Task.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace gluten {

class CudfVectorStreamBase {
 public:
  virtual ~CudfVectorStreamBase() = default;

  explicit CudfVectorStreamBase(
      facebook::velox::exec::DriverCtx* driverCtx,
      facebook::velox::memory::MemoryPool* pool,
      ResultIterator* iterator,
      const facebook::velox::RowTypePtr& outputType)
      : driverCtx_(driverCtx), pool_(pool), outputType_(outputType), iterator_(iterator) {}

  bool hasNext();

  // Convert arrow batch to row vector, construct the new Rowvector with new outputType.
  virtual facebook::velox::RowVectorPtr next();

 protected:
  // Get the next batch from iterator_.
  std::shared_ptr<ColumnarBatch> nextInternal();

  facebook::velox::exec::DriverCtx* driverCtx_;
  facebook::velox::memory::MemoryPool* pool_;
  const facebook::velox::RowTypePtr outputType_;
  ResultIterator* iterator_;

  bool finished_{false};
};

class ValueStreamNode final : public facebook::velox::core::PlanNode {
 public:
  ValueStreamNode(
      const facebook::velox::core::PlanNodeId& id,
      const facebook::velox::RowTypePtr& outputType,
      std::shared_ptr<ResultIterator> iterator)
      : facebook::velox::core::PlanNode(id), outputType_(outputType), iterator_(std::move(iterator)) {}

  const facebook::velox::RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<facebook::velox::core::PlanNodePtr>& sources() const override {
    return kEmptySources_;
  };

  ResultIterator* iterator() const {
    return iterator_.get();
  }

  std::string_view name() const override {
    return "ValueStream";
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("ValueStream plan node is not serializable");
  }

 private:
  void addDetails(std::stringstream& stream) const override{};

  const facebook::velox::RowTypePtr outputType_;
  std::shared_ptr<ResultIterator> iterator_;
  const std::vector<facebook::velox::core::PlanNodePtr> kEmptySources_;
};

class CudfVectorStream : public CudfVectorStreamBase {
 public:
  CudfVectorStream(
      facebook::velox::exec::DriverCtx* driverCtx,
      facebook::velox::memory::MemoryPool* pool,
      ResultIterator* iterator,
      const facebook::velox::RowTypePtr& outputType)
      : CudfVectorStreamBase(driverCtx, pool, iterator, outputType) {
    targetBatchBytes_ =
        facebook::velox::cudf_velox::CudfConfig::getInstance().gpuTargetBatchBytes;
    targetBatchRows_ =
        facebook::velox::cudf_velox::CudfConfig::getInstance().gpuTargetBatchRows;
  }

  bool hasPending() const {
    return !pendingRows_.empty() || !pendingGpuBatches_.empty() || stashedCudf_ != nullptr;
  }

  // Convert columnar batch to a CudfVector for downstream GPU operators.
  // Handles three input types with batch accumulation for Cases 2 & 3:
  //   1. VeloxColumnarBatch wrapping a CudfVector  -> re-wrap (returned immediately)
  //   2. VeloxColumnarBatch wrapping a CPU RowVector (e.g. BroadcastExchange)
  //      -> accumulated then batched upload
  //   3. GpuBufferColumnarBatch (shuffle read) -> accumulated then batched upload
  facebook::velox::RowVectorPtr next() override {
    auto belowThreshold = [&]() {
      if (targetBatchBytes_ > 0) {
        return pendingBytes_ < targetBatchBytes_;
      }
      if (targetBatchRows_ > 0) {
        return pendingRowCount_ < targetBatchRows_;
      }
      return pendingRows_.empty() && pendingGpuBatches_.empty();
    };
    while (belowThreshold()) {
      auto cb = nextInternal();
      if (cb == nullptr) {
        break;
      }

      // Case 1: VeloxColumnarBatch wrapping a CudfVector
      if (cb->getType() == "velox") {
        auto vb = std::dynamic_pointer_cast<VeloxColumnarBatch>(cb);
        VELOX_CHECK_NOT_NULL(vb);
        auto vp = vb->getRowVector();
        VELOX_CHECK_NOT_NULL(vp);
        auto cudfVector =
            std::dynamic_pointer_cast<facebook::velox::cudf_velox::CudfVector>(vp);
        if (cudfVector != nullptr) {
          // Already a CudfVector. If we have pending CPU rows, flush them
          // first and stash this CudfVector for the next call.
          if (!pendingRows_.empty()) {
            stashedCudf_ = cudfVector;
            break;
          }
          GpuLockGuard gpuLock;
          return std::make_shared<facebook::velox::cudf_velox::CudfVector>(
              vp->pool(), outputType_, vp->size(), cudfVector->release(), cudfVector->stream());
        }
        // Case 2: CPU RowVector – accumulate for batched upload.
        pendingRows_.push_back(vp);
        pendingBytes_ += vp->estimateFlatSize();
        pendingRowCount_ += vp->size();
        continue;
      }

#ifdef GLUTEN_ENABLE_GPU
      // Case 3: GpuBufferColumnarBatch – accumulate directly for batched
      // Arrow-to-cudf conversion, bypassing the Velox RowVector intermediate.
      if (cb->getType() == "gpu") {
        auto gpuBatch = std::dynamic_pointer_cast<GpuBufferColumnarBatch>(cb);
        VELOX_CHECK_NOT_NULL(gpuBatch);
        pendingBytes_ += gpuBatch->numBytes();
        pendingRowCount_ += gpuBatch->numRows();
        pendingGpuBatches_.push_back(std::move(gpuBatch));
        continue;
      }
#endif
      VELOX_FAIL(
          "Unsupported ColumnarBatch type: '{}', numColumns: {}, numRows: {}",
          cb->getType(),
          cb->numColumns(),
          cb->numRows());
    }

    // If there's a stashed CudfVector and no pending data, return it.
    if (pendingRows_.empty() && pendingGpuBatches_.empty() && stashedCudf_ != nullptr) {
      auto cudf = std::move(stashedCudf_);
      stashedCudf_ = nullptr;
      GpuLockGuard gpuLock;
      return std::make_shared<facebook::velox::cudf_velox::CudfVector>(
          cudf->pool(), outputType_, cudf->size(), cudf->release(), cudf->stream());
    }

#ifdef GLUTEN_ENABLE_GPU
    // Direct GPU path: compose Arrow buffers in pinned memory, then convert
    // to cudf columns without the Velox RowVector intermediate.
    if (!pendingGpuBatches_.empty()) {
      auto composed = GpuBufferColumnarBatch::compose(
          getPinnedArrowMemoryPool(), pendingGpuBatches_, pendingRowCount_);

      GpuLockGuard gpuLock;
      auto vcb = gpuBuffersToCudfVector(
          composed->getRowType(), composed->numRows(),
          composed->buffers(), pool_);
      VELOX_CHECK_NOT_NULL(vcb);

      numCoalescedBatches_ += (pendingGpuBatches_.size() > 1) ? 1 : 0;
      pendingGpuBatches_.clear();
      pendingBytes_ = 0;
      pendingRowCount_ = 0;

      auto vb = std::dynamic_pointer_cast<VeloxColumnarBatch>(vcb);
      VELOX_CHECK_NOT_NULL(vb);
      auto rv = vb->getRowVector();
      auto cudfVec = std::dynamic_pointer_cast<facebook::velox::cudf_velox::CudfVector>(rv);
      VELOX_CHECK_NOT_NULL(cudfVec);
      return std::make_shared<facebook::velox::cudf_velox::CudfVector>(
          pool_, outputType_, cudfVec->size(), cudfVec->release(), cudfVec->stream());
    }
#endif

    if (pendingRows_.empty()) {
      return nullptr;
    }

    // Batched HtoD: N async from_arrow, ONE sync, GPU concatenate.
    {
      GpuLockGuard gpuLock;
      auto stream = facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
      auto tbl = facebook::velox::cudf_velox::with_arrow::toCudfTableBatched(
          pendingRows_, pool_, stream);
      VELOX_CHECK_NOT_NULL(tbl);
      const auto size = tbl->num_rows();

      numCoalescedBatches_ += (pendingRows_.size() > 1) ? 1 : 0;

      pendingRows_.clear();
      pendingBytes_ = 0;
      pendingRowCount_ = 0;

      return std::make_shared<facebook::velox::cudf_velox::CudfVector>(
          pool_, outputType_, size, std::move(tbl), stream);
    }
  }

  int64_t numCoalescedBatches() const {
    return numCoalescedBatches_;
  }

 private:
  int64_t targetBatchBytes_;
  int64_t targetBatchRows_;
  std::vector<facebook::velox::RowVectorPtr> pendingRows_;
  std::vector<std::shared_ptr<GpuBufferColumnarBatch>> pendingGpuBatches_;
  int64_t pendingBytes_ = 0;
  int64_t pendingRowCount_ = 0;
  int64_t numCoalescedBatches_ = 0;
  std::shared_ptr<facebook::velox::cudf_velox::CudfVector> stashedCudf_;
};

// To avoid plan translator uses false node, this one cannot inherit ValueStreamNode.
class CudfValueStreamNode final : public facebook::velox::core::PlanNode {
 public:
  CudfValueStreamNode(
      const facebook::velox::core::PlanNodeId& id,
      const facebook::velox::RowTypePtr& outputType,
      std::shared_ptr<ResultIterator> iterator)
      : facebook::velox::core::PlanNode(id), outputType_(outputType), iterator_(std::move(iterator)) {}

  const facebook::velox::RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<facebook::velox::core::PlanNodePtr>& sources() const override {
    return kEmptySources_;
  };

  ResultIterator* iterator() const {
    return iterator_.get();
  }

  std::string_view name() const override {
    return "CudfValueStream";
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("CudfValueStream plan node is not serializable");
  }

 private:
  void addDetails(std::stringstream& stream) const override{};

  const facebook::velox::RowTypePtr outputType_;
  std::shared_ptr<ResultIterator> iterator_;
  const std::vector<facebook::velox::core::PlanNodePtr> kEmptySources_;
};

// Extends CudfOperator to identify it as GPU node, so not add CudfFormVelox operator.
class CudfValueStream : public facebook::velox::exec::SourceOperator, public facebook::velox::cudf_velox::CudfOperator {
 public:
  CudfValueStream(
      int32_t operatorId,
      facebook::velox::exec::DriverCtx* driverCtx,
      std::shared_ptr<const CudfValueStreamNode> valueStreamNode)
      : facebook::velox::exec::SourceOperator(
            driverCtx,
            valueStreamNode->outputType(),
            operatorId,
            valueStreamNode->id(),
            valueStreamNode->name().data()),
        facebook::velox::cudf_velox::CudfOperator(operatorId, valueStreamNode->id()) {
    ResultIterator* itr = valueStreamNode->iterator();
    rvStream_ = std::make_unique<CudfVectorStream>(driverCtx, pool(), itr, outputType_);
  }

  facebook::velox::RowVectorPtr getOutput() override {
    if (finished_) {
      return nullptr;
    }
    if (rvStream_->hasNext() || rvStream_->hasPending()) {
      auto result = rvStream_->next();
      if (result == nullptr) {
        finished_ = true;
        reportCoalescedBatches();
      }
      return result;
    } else {
      finished_ = true;
      reportCoalescedBatches();
      return nullptr;
    }
  }

  facebook::velox::exec::BlockingReason isBlocked(facebook::velox::ContinueFuture* /* unused */) override {
    return facebook::velox::exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 private:
  void reportCoalescedBatches() {
    auto count = rvStream_->numCoalescedBatches();
    if (count > 0) {
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat(
          "numCoalescedBatches",
          facebook::velox::RuntimeCounter(count));
    }
  }

  bool finished_ = false;
  std::unique_ptr<CudfVectorStream> rvStream_;
};

class CudfVectorStreamOperatorTranslator : public facebook::velox::exec::Operator::PlanNodeTranslator {
  std::unique_ptr<facebook::velox::exec::Operator> toOperator(
      facebook::velox::exec::DriverCtx* ctx,
      int32_t id,
      const facebook::velox::core::PlanNodePtr& node) override {
    if (auto valueStreamNode = std::dynamic_pointer_cast<const CudfValueStreamNode>(node)) {
      return std::make_unique<CudfValueStream>(id, ctx, valueStreamNode);
    }
    return nullptr;
  }
};
} // namespace gluten
