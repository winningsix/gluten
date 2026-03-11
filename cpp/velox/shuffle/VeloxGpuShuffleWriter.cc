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

#include "VeloxGpuShuffleWriter.h"
#include "cudf/GpuLock.h"
#include "memory/VeloxColumnarBatch.h"
#include "utils/Timer.h"

#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/exec/PinnedHostMemory.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cuda_runtime.h>

namespace gluten {

using namespace facebook::velox;
using CudfVector = facebook::velox::cudf_velox::CudfVector;
using PinnedHostBuffer =
    facebook::velox::cudf_velox::PinnedHostBuffer;

namespace {

// arrow::Buffer backed by PinnedHostBuffer for zero-copy D2H.
// Pinned memory enables fast PCIe DMA. The PinnedHostBuffer
// lifetime is tied to this arrow::Buffer via shared_ptr, so
// downstream consumers (compression tasks) keep it alive.
class PinnedArrowBuffer : public arrow::Buffer {
 public:
  explicit PinnedArrowBuffer(
      std::shared_ptr<PinnedHostBuffer> pinned)
      : arrow::Buffer(
            pinned->data(),
            static_cast<int64_t>(pinned->size())),
        pinned_(std::move(pinned)) {}

 private:
  std::shared_ptr<PinnedHostBuffer> pinned_;
};

arrow::Status checkCuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    return arrow::Status::IOError(
        msg, ": ", cudaGetErrorString(err));
  }
  return arrow::Status::OK();
}

#define RETURN_CUDA_ERROR(expr)          \
  RETURN_NOT_OK(checkCuda((expr), #expr))

} // namespace

// Split bool bit to bytes.
void VeloxGpuHashShuffleWriter::splitBoolValueType(const uint8_t* srcAddr, const std::vector<uint8_t*>& dstAddrs) {
  for (auto& pid : partitionUsed_) {
    auto dstaddr = dstAddrs[pid];
    if (dstaddr == nullptr) {
      continue;
    }
    auto dstPidBase = (uint8_t*)(dstaddr + partitionBufferBase_[pid] * sizeof(uint8_t));
    auto pos = partition2RowOffsetBase_[pid];
    auto end = partition2RowOffsetBase_[pid + 1];
    for (; pos < end; ++pos) {
      auto rowId = rowOffset2RowId_[pos];
      uint8_t byte = srcAddr[rowId >> 3];
      uint8_t bit = (byte >> (rowId & 7)) & 0x01;
      *dstPidBase++ = bit;
    }
  }
}

// Split timestamp from int128_t to int64_t, both of them represents the timestamp nanoseconds.
arrow::Status VeloxGpuHashShuffleWriter::splitTimestamp(const uint8_t* srcAddr, const std::vector<uint8_t*>& dstAddrs) {
   for (auto& pid : partitionUsed_) {
      auto dstPidBase = (int64_t*)(dstAddrs[pid] + partitionBufferBase_[pid] * sizeof(int64_t));
      auto pos = partition2RowOffsetBase_[pid];
      auto end = partition2RowOffsetBase_[pid + 1];
      for (; pos < end; ++pos) {
        auto rowId = rowOffset2RowId_[pos];
        auto* src = reinterpret_cast<const int64_t*>(srcAddr) + rowId * 2;
        *dstPidBase++ = src[0] * 1'000'000'000LL + src[1];
      }
    }
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<VeloxShuffleWriter>> VeloxGpuHashShuffleWriter::create(
    uint32_t numPartitions,
    const std::shared_ptr<PartitionWriter>& partitionWriter,
    const std::shared_ptr<ShuffleWriterOptions>& options,
    MemoryManager* memoryManager) {
  if (auto hashOptions = std::dynamic_pointer_cast<GpuHashShuffleWriterOptions>(options)) {
    std::shared_ptr<VeloxGpuHashShuffleWriter> res =
        std::make_shared<VeloxGpuHashShuffleWriter>(numPartitions, partitionWriter, hashOptions, memoryManager);
    RETURN_NOT_OK(res->init());
    LOG(INFO) << "VeloxGpuHashShuffleWriter created: numPartitions=" << numPartitions
              << " gpuPartition=" << (hashOptions->gpuPartition ? "ON" : "OFF");
    return res;
  }
  return arrow::Status::Invalid("Error casting ShuffleWriterOptions to GpuHashShuffleWriterOptions. ");
}

arrow::Status VeloxGpuHashShuffleWriter::write(std::shared_ptr<ColumnarBatch> cb, int64_t memLimit) {
  if (!gpuPartitionEnabled_ ||
      partitioning_ == Partitioning::kSingle) {
    return VeloxHashShuffleWriter::write(cb, memLimit);
  }

  if (cb->getType() == "velox") {
    auto veloxBatch = std::dynamic_pointer_cast<VeloxColumnarBatch>(cb);
    if (veloxBatch) {
      auto rv = veloxBatch->getRowVector();
      auto cudfVec = std::dynamic_pointer_cast<CudfVector>(rv);
      if (cudfVec) {
        if (!gpuSchemaInitialized_) {
          gpuSchemaInitialized_ = true;
          auto& fullRowType = cudfVec->type()->asRow();
          auto typeChildren = fullRowType.children();
          typeChildren.erase(typeChildren.begin());
          auto strippedType =
              ROW(std::move(typeChildren));
          auto emptyRv = RowVector::createEmpty(
              strippedType, veloxPool_.get());
          RETURN_NOT_OK(initFromRowVector(*emptyRv));
        }

        if (hasComplexType_) {
          auto cpuRv =
              cudf_velox::with_arrow::toVeloxColumn(
                  cudfVec->getTableView(),
                  veloxPool_.get(),
                  std::string(""),
                  cudfVec->stream());
          auto cpuBatch =
              std::make_shared<VeloxColumnarBatch>(cpuRv);
          return VeloxHashShuffleWriter::write(
              cpuBatch, memLimit);
        }

        writtenBytes_ = 0;
        ++gpuWriteBatches_;
        // Release all outer shared_ptr references so gpuPartitionAndEvict
        // can free the GPU memory after cudf::partition() creates its copy.
        auto localCudfVec = std::move(cudfVec);
        rv.reset();
        veloxBatch.reset();
        cb.reset();
        return gpuPartitionAndEvict(std::move(localCudfVec));
      }

      // RowVector (not CudfVector): CudfShufflePartition may or may not
      // have been in the pipeline. Fall through to the standard CPU path
      // which correctly handles both pre-partitioned (PID as col 0) and
      // regular (hash as col 0) data via computePid (pid % numPartitions).
    }
  }

  ++cpuFallbackBatches_;
  return VeloxHashShuffleWriter::write(cb, memLimit);
}

arrow::Status VeloxGpuHashShuffleWriter::prePartitionedEvict(
    const facebook::velox::RowVectorPtr& rv) {
  VELOX_CHECK_GT(rv->childrenSize(), 0, "RowVector must have at least PID column");

  // Strip first column (PID) to get data-only RowVector for schema init.
  auto strippedRv = getStrippedRowVector(*rv);

  // Initialize schema from the stripped RowVector (data cols only) on first batch.
  if (!gpuSchemaInitialized_) {
    gpuSchemaInitialized_ = true;
    RETURN_NOT_OK(initFromRowVector(*strippedRv));
    LOG(INFO) << "GPU prePartitionedEvict: schema initialized from "
              << strippedRv->childrenSize() << " data columns";
  }

  // Complex types not supported in this fast path.
  if (hasComplexType_) {
    auto cpuBatch = std::make_shared<VeloxColumnarBatch>(rv);
    return VeloxHashShuffleWriter::write(cpuBatch, 0);
  }

  // First column is sorted PID (0,0,...,0,1,1,...,1,2,2,...).
  auto* pidVector = rv->childAt(0)->asFlatVector<int32_t>();
  VELOX_CHECK_NOT_NULL(pidVector, "First column must be flat int32 (PID)");
  const auto* pidData = pidVector->rawValues();
  const auto numRows = rv->size();

  // Scan sorted PID column to find partition boundaries.
  std::vector<int64_t> offsets(numPartitions_ + 1, 0);
  offsets[numPartitions_] = numRows;
  int32_t currentPid = 0;
  for (int64_t i = 0; i < numRows; ++i) {
    while (currentPid < pidData[i]) {
      offsets[++currentPid] = i;
    }
  }
  while (currentPid < static_cast<int32_t>(numPartitions_)) {
    offsets[++currentPid] = numRows;
  }

  // Sequential extract per partition from the stripped RowVector.
  for (uint32_t pid = 0; pid < numPartitions_; ++pid) {
    auto start = offsets[pid];
    auto count = static_cast<uint32_t>(offsets[pid + 1] - offsets[pid]);
    if (count == 0) {
      continue;
    }
    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    RETURN_NOT_OK(extractBuffersFromRowVector(*strippedRv, start, count, buffers));
    RETURN_NOT_OK(evictBuffers(pid, count, std::move(buffers), false));
  }

  return arrow::Status::OK();
}

arrow::Status VeloxGpuHashShuffleWriter::gpuPartitionAndEvict(
    std::shared_ptr<CudfVector> cudfVec) {
  std::vector<cudf::size_type> offsets;
  cudf::size_type totalRows = 0;
  int numCols = 0;

  struct ColHost {
    bool isString = false;
    bool hasNulls = false;
    int32_t elemSize = 0;
    std::shared_ptr<PinnedArrowBuffer> dataBuf;
    std::shared_ptr<PinnedArrowBuffer> nullBuf;
    std::shared_ptr<PinnedArrowBuffer> strOffsetsBuf;
    std::shared_ptr<PinnedArrowBuffer> strCharsBuf;
  };
  std::vector<ColHost> colHosts;

  try {
  {
    GpuLockGuard gpuLock;

    auto tableView = cudfVec->getTableView();
    auto stream = cudfVec->stream();

    auto firstCol = tableView.column(0);
    std::vector<cudf::column_view> dataCols;
    dataCols.reserve(tableView.num_columns() - 1);
    for (cudf::size_type i = 1;
         i < tableView.num_columns(); ++i) {
      dataCols.push_back(tableView.column(i));
    }
    cudf::table_view dataTable(dataCols);

    std::unique_ptr<cudf::column> pidColOwned;
    cudf::column_view pidColView;
    if (partitioning_ == Partitioning::kHash) {
      auto numPartScalar = cudf::numeric_scalar<int32_t>(
          static_cast<int32_t>(numPartitions_),
          true, stream);
      pidColOwned = cudf::binary_operation(
          firstCol, numPartScalar,
          cudf::binary_operator::PYMOD,
          cudf::data_type{cudf::type_id::INT32},
          stream);
      pidColView = pidColOwned->view();
    } else {
      pidColView = firstCol;
    }

    auto [partitionedTable, partOffsets] =
        cudf::partition(
            dataTable, pidColView,
            static_cast<cudf::size_type>(numPartitions_),
            stream);
    VELOX_CHECK_EQ(
        partOffsets.size(), numPartitions_ + 1);
    offsets = std::move(partOffsets);
    // Sync before freeing input: cudf::partition is async.
    stream.synchronize();
    pidColOwned.reset();
    cudfVec.reset();

    totalRows = partitionedTable->num_rows();
    numCols = partitionedTable->num_columns();
    auto tv = partitionedTable->view();

    colHosts.resize(numCols);

    for (int c = 0; c < numCols; ++c) {
      auto col = tv.column(c);
      auto& ch = colHosts[c];
      auto typeId = col.type().id();
      ch.hasNulls = col.nullable();

      if (ch.hasNulls) {
        auto validitySize = static_cast<int64_t>(
            cudf::bitmask_allocation_size_bytes(
                totalRows));
        auto pinned = std::make_shared<PinnedHostBuffer>(
            validitySize);
        RETURN_CUDA_ERROR(cudaMemcpyAsync(
            pinned->data(),
            col.null_mask(),
            validitySize,
            cudaMemcpyDeviceToHost,
            stream.value()));
        ch.nullBuf =
            std::make_shared<PinnedArrowBuffer>(
                std::move(pinned));
      }

      if (typeId == cudf::type_id::STRING) {
        ch.isString = true;
        auto scv = cudf::strings_column_view(col);
        auto offsCol = scv.offsets();
        VELOX_CHECK(
            offsCol.type().id() == cudf::type_id::INT32,
            "Only INT32 string offsets supported");

        auto offsBytes = static_cast<int64_t>(
            totalRows + 1) * sizeof(int32_t);
        auto pinnedOffs =
            std::make_shared<PinnedHostBuffer>(offsBytes);
        RETURN_CUDA_ERROR(cudaMemcpyAsync(
            pinnedOffs->data(),
            offsCol.data<int32_t>(),
            offsBytes,
            cudaMemcpyDeviceToHost,
            stream.value()));
        ch.strOffsetsBuf =
            std::make_shared<PinnedArrowBuffer>(
                std::move(pinnedOffs));

        auto charsSize = static_cast<int64_t>(
            scv.chars_size(stream));
        if (charsSize > 0) {
          auto pinnedChars =
              std::make_shared<PinnedHostBuffer>(
                  charsSize);
          RETURN_CUDA_ERROR(cudaMemcpyAsync(
              pinnedChars->data(),
              scv.chars_begin(stream),
              charsSize,
              cudaMemcpyDeviceToHost,
              stream.value()));
          ch.strCharsBuf =
              std::make_shared<PinnedArrowBuffer>(
                  std::move(pinnedChars));
        }
      } else {
        ch.elemSize = static_cast<int32_t>(
            cudf::size_of(col.type()));
        auto dataBytes =
            static_cast<int64_t>(totalRows) * ch.elemSize;
        if (dataBytes > 0) {
          auto pinned =
              std::make_shared<PinnedHostBuffer>(
                  dataBytes);
          RETURN_CUDA_ERROR(cudaMemcpyAsync(
              pinned->data(),
              col.data<uint8_t>(),
              dataBytes,
              cudaMemcpyDeviceToHost,
              stream.value()));
          ch.dataBuf =
              std::make_shared<PinnedArrowBuffer>(
                  std::move(pinned));
        }
      }
    }

    RETURN_CUDA_ERROR(
        cudaStreamSynchronize(stream.value()));
  }
  } catch (const std::exception& e) {
    LOG(ERROR) << "gpuPartitionAndEvict failed: "
               << e.what();
    return arrow::Status::ExecutionError(
        "GPU shuffle partition failed: ", e.what());
  }

  // CPU-side: slice per-column host buffers per partition.
  uint64_t totalRowsEvicted = 0;

  for (uint32_t pid = 0; pid < numPartitions_; ++pid) {
    auto start = offsets[pid];
    auto count = offsets[pid + 1] - offsets[pid];
    if (count == 0) {
      continue;
    }
    auto numRows = static_cast<uint32_t>(count);

    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    buffers.reserve(numCols * 2);

    for (int c = 0; c < numCols; ++c) {
      auto& ch = colHosts[c];
      auto arrowTypeId = arrowColumnTypes_[c]->id();
      if (arrowTypeId == arrow::Type::NA) {
        continue;
      }

      if (ch.hasNulls) {
        auto validityBytes =
            arrow::bit_util::BytesForBits(numRows);
        ARROW_ASSIGN_OR_RAISE(
            auto vBuf,
            arrow::AllocateBuffer(
                validityBytes,
                partitionBufferPool_.get()));
        std::memset(
            vBuf->mutable_data(), 0, validityBytes);
        bits::copyBits(
            reinterpret_cast<const uint64_t*>(
                ch.nullBuf->data()),
            start,
            reinterpret_cast<uint64_t*>(
                vBuf->mutable_data()),
            0, numRows);
        buffers.push_back(std::move(vBuf));
      } else {
        buffers.push_back(nullptr);
      }

      if (ch.isString) {
        auto* hostOffs =
            reinterpret_cast<const int32_t*>(
                ch.strOffsetsBuf->data());
        auto lenBytes =
            static_cast<int64_t>(numRows) *
            sizeof(uint32_t);
        ARROW_ASSIGN_OR_RAISE(
            auto lenBuf,
            arrow::AllocateBuffer(
                lenBytes,
                partitionBufferPool_.get()));
        auto* lengths = reinterpret_cast<uint32_t*>(
            lenBuf->mutable_data());
        for (uint32_t i = 0; i < numRows; ++i) {
          lengths[i] = static_cast<uint32_t>(
              hostOffs[start + i + 1] -
              hostOffs[start + i]);
        }
        buffers.push_back(std::move(lenBuf));

        auto charsStart = hostOffs[start];
        auto charsEnd = hostOffs[start + numRows];
        auto charsLen = charsEnd - charsStart;
        if (charsLen > 0 && ch.strCharsBuf) {
          buffers.push_back(arrow::SliceBuffer(
              ch.strCharsBuf, charsStart, charsLen));
        } else {
          buffers.push_back(zeroLengthNullBuffer());
        }
      } else {
        auto byteOffset = start * ch.elemSize;
        auto sliceLen =
            static_cast<int64_t>(numRows) * ch.elemSize;
        if (sliceLen > 0 && ch.dataBuf) {
          buffers.push_back(arrow::SliceBuffer(
              ch.dataBuf, byteOffset, sliceLen));
        } else {
          buffers.push_back(zeroLengthNullBuffer());
        }
      }
    }

    RETURN_NOT_OK(evictBuffers(
        pid, numRows, std::move(buffers), false));
    totalRowsEvicted += numRows;
  }

  return arrow::Status::OK();
}

arrow::Status VeloxGpuHashShuffleWriter::extractBuffersFromRowVector(
    const facebook::velox::RowVector& rv,
    int64_t start,
    uint32_t numRows,
    std::vector<std::shared_ptr<arrow::Buffer>>& outBuffers) {
  auto numFields = schema_->num_fields();
  outBuffers.reserve(fixedWidthColumnCount_ * 2 + binaryColumnIndices_.size() * 3);

  for (int col = 0; col < numFields; ++col) {
    auto arrowTypeId = arrowColumnTypes_[col]->id();

    switch (arrowTypeId) {
      case arrow::Type::NA:
        break;

      case arrow::Type::STRING:
      case arrow::Type::BINARY: {
        auto& column = rv.childAt(col);
        auto* flatVec = column->asFlatVector<StringView>();
        // rawValues() is already offset-adjusted for FlatVector.
        const auto* rawValues = flatVec->rawValues();
        const auto* rawNulls = column->rawNulls();

        // Validity buffer: extract bits [start, start+numRows).
        if (rawNulls != nullptr && column->mayHaveNulls()) {
          auto validityBytes = arrow::bit_util::BytesForBits(numRows);
          ARROW_ASSIGN_OR_RAISE(auto buf, arrow::AllocateBuffer(validityBytes, partitionBufferPool_.get()));
          std::memset(buf->mutable_data(), 0, validityBytes);
          bits::copyBits(rawNulls, start, reinterpret_cast<uint64_t*>(buf->mutable_data()), 0, numRows);
          outBuffers.push_back(std::move(buf));
        } else {
          outBuffers.push_back(nullptr);
        }

        // Length buffer + value buffer.
        {
          auto lengthBytes = static_cast<int64_t>(numRows) * sizeof(uint32_t);
          ARROW_ASSIGN_OR_RAISE(auto lengthBuf, arrow::AllocateBuffer(lengthBytes, partitionBufferPool_.get()));
          auto* lengths = reinterpret_cast<uint32_t*>(lengthBuf->mutable_data());
          int64_t totalValueSize = 0;
          for (uint32_t i = 0; i < numRows; ++i) {
            bool isNull = rawNulls && bits::isBitNull(rawNulls, start + i);
            uint32_t len = isNull ? 0 : static_cast<uint32_t>(rawValues[start + i].size());
            lengths[i] = len;
            totalValueSize += len;
          }
          outBuffers.push_back(std::move(lengthBuf));

          if (totalValueSize > 0) {
            ARROW_ASSIGN_OR_RAISE(auto valueBuf, arrow::AllocateBuffer(totalValueSize, partitionBufferPool_.get()));
            auto* dst = valueBuf->mutable_data();
            for (uint32_t i = 0; i < numRows; ++i) {
              bool isNull = rawNulls && bits::isBitNull(rawNulls, start + i);
              if (!isNull) {
                auto len = rawValues[start + i].size();
                if (len > 0) {
                  std::memcpy(dst, rawValues[start + i].data(), len);
                  dst += len;
                }
              }
            }
            outBuffers.push_back(std::move(valueBuf));
          } else {
            outBuffers.push_back(zeroLengthNullBuffer());
          }
        }
        break;
      }

      case arrow::Type::STRUCT:
      case arrow::Type::MAP:
      case arrow::Type::LIST:
        return arrow::Status::NotImplemented(
            "GPU partition does not support complex types yet. Column: " +
            schema_->field(col)->name());

      default: {
        auto& column = rv.childAt(col);
        const auto* rawNulls = column->rawNulls();

        // Validity buffer: extract bits [start, start+numRows).
        if (rawNulls != nullptr && column->mayHaveNulls()) {
          auto validityBytes = arrow::bit_util::BytesForBits(numRows);
          ARROW_ASSIGN_OR_RAISE(auto buf, arrow::AllocateBuffer(validityBytes, partitionBufferPool_.get()));
          std::memset(buf->mutable_data(), 0, validityBytes);
          bits::copyBits(rawNulls, start, reinterpret_cast<uint64_t*>(buf->mutable_data()), 0, numRows);
          outBuffers.push_back(std::move(buf));
        } else {
          outBuffers.push_back(nullptr);
        }

        auto veloxType = veloxColumnTypes_[col];
        if (arrowTypeId == arrow::Type::BOOL) {
          // Bool bit→byte conversion at [start, start+numRows).
          ARROW_ASSIGN_OR_RAISE(auto buf, arrow::AllocateBuffer(numRows, partitionBufferPool_.get()));
          auto* dst = buf->mutable_data();
          auto* srcBytes = static_cast<const uint8_t*>(column->valuesAsVoid());
          for (uint32_t i = 0; i < numRows; ++i) {
            auto idx = start + i;
            dst[i] = (srcBytes[idx >> 3] >> (idx & 7)) & 0x01;
          }
          outBuffers.push_back(std::move(buf));
        } else if (veloxType->kind() == TypeKind::TIMESTAMP) {
          auto valueBufferSize = static_cast<int64_t>(numRows) * sizeof(int64_t);
          ARROW_ASSIGN_OR_RAISE(auto buf, arrow::AllocateBuffer(valueBufferSize, partitionBufferPool_.get()));
          auto* dst = reinterpret_cast<int64_t*>(buf->mutable_data());
          auto* src = reinterpret_cast<const int64_t*>(column->valuesAsVoid()) + start * 2;
          for (uint32_t i = 0; i < numRows; ++i) {
            dst[i] = src[i * 2] * 1'000'000'000LL + src[i * 2 + 1];
          }
          outBuffers.push_back(std::move(buf));
        } else if (veloxType->isShortDecimal()) {
          auto valueBufferSize = static_cast<int64_t>(numRows) * sizeof(int64_t);
          ARROW_ASSIGN_OR_RAISE(auto buf, arrow::AllocateBuffer(valueBufferSize, partitionBufferPool_.get()));
          auto* src = reinterpret_cast<const int64_t*>(column->valuesAsVoid()) + start;
          std::memcpy(buf->mutable_data(), src, valueBufferSize);
          outBuffers.push_back(std::move(buf));
        } else {
          // Standard fixed-width: zero-copy slice from the contiguous D2H buffer.
          auto byteWidth = arrow::bit_width(arrowTypeId) >> 3;
          auto byteOffset = start * byteWidth;
          auto valueBufferSize = static_cast<int64_t>(numRows) * byteWidth;
          ARROW_ASSIGN_OR_RAISE(auto buf, arrow::AllocateBuffer(valueBufferSize, partitionBufferPool_.get()));
          auto* src = static_cast<const uint8_t*>(column->valuesAsVoid()) + byteOffset;
          std::memcpy(buf->mutable_data(), src, valueBufferSize);
          outBuffers.push_back(std::move(buf));
        }
        break;
      }
    }
  }

  return arrow::Status::OK();
}

} // namespace gluten
