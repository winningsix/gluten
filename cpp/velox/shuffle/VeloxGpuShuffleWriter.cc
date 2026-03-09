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

#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

namespace gluten {

using namespace facebook::velox;
using CudfVector = facebook::velox::cudf_velox::CudfVector;

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
    if (!gpuPartitionDiagLogged_) {
      gpuPartitionDiagLogged_ = true;
      LOG(WARNING) << "GPU partition DISABLED: gpuPartitionEnabled_=" << gpuPartitionEnabled_
                   << " partitioning=" << static_cast<int>(partitioning_);
    }
    return VeloxHashShuffleWriter::write(cb, memLimit);
  }

  // Detect CudfVector before any D2H conversion.
  if (cb->getType() == "velox") {
    auto veloxBatch = std::dynamic_pointer_cast<VeloxColumnarBatch>(cb);
    if (veloxBatch) {
      auto rv = veloxBatch->getRowVector();
      auto cudfVec = std::dynamic_pointer_cast<CudfVector>(rv);
      if (!gpuPartitionDiagLogged_) {
        gpuPartitionDiagLogged_ = true;
        LOG(WARNING) << "GPU partition diag: batchType=" << cb->getType()
                  << " veloxBatch=" << (veloxBatch != nullptr)
                  << " rvType=" << (rv ? rv->type()->toString() : "null")
                  << " rvTypeName=" << (rv ? typeid(*rv).name() : "null")
                  << " isCudfVector=" << (cudfVec != nullptr)
                  << " numRows=" << cb->numRows();
      }
      if (cudfVec) {
        if (!gpuSchemaInitialized_) {
          gpuSchemaInitialized_ = true;
          LOG(WARNING)
              << "GPU partition: first CudfVector batch,"
              << " rows=" << cudfVec->size()
              << " cols="
              << cudfVec->getTableView().num_columns();
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
      if (!gpuPartitionDiagLogged_) {
        LOG(WARNING) << "GPU partition: RowVector (not CudfVector) received, "
                  << "using CPU shuffle path. rows=" << rv->size()
                  << " cols=" << rv->childrenSize();
      }
    }
  } else {
    if (!gpuPartitionDiagLogged_) {
      gpuPartitionDiagLogged_ = true;
      LOG(WARNING) << "GPU partition fallback: batchType=" << cb->getType()
                   << " (not 'velox'), falling back to CPU path";
    }
  }

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
  facebook::velox::RowVectorPtr veloxRv;
  std::vector<cudf::size_type> offsets;

  // GPU-intensive section: partition on device and D2H transfer.
  // Acquire the GPU concurrency semaphore so that concurrent shuffle tasks
  // do not collectively exhaust GPU memory.
  {
    GpuLockGuard gpuLock;

    auto tableView = cudfVec->getTableView();
    auto stream = cudfVec->stream();

    LOG(WARNING) << "gpuPartitionAndEvict: input rows=" << tableView.num_rows()
                 << " cols=" << tableView.num_columns()
                 << " numPartitions=" << numPartitions_
                 << " partitioning=" << static_cast<int>(partitioning_);

    auto firstCol = tableView.column(0);

    std::vector<cudf::column_view> dataCols;
    dataCols.reserve(tableView.num_columns() - 1);
    for (cudf::size_type i = 1; i < tableView.num_columns(); ++i) {
      dataCols.push_back(tableView.column(i));
    }
    cudf::table_view dataTable(dataCols);

    std::unique_ptr<cudf::column> pidColOwned;
    cudf::column_view pidColView;
    if (partitioning_ == Partitioning::kHash) {
      auto numPartScalar = cudf::numeric_scalar<int32_t>(
          static_cast<int32_t>(numPartitions_), true, stream);
      pidColOwned = cudf::binary_operation(
          firstCol,
          numPartScalar,
          cudf::binary_operator::PYMOD,
          cudf::data_type{cudf::type_id::INT32},
          stream);
      pidColView = pidColOwned->view();
    } else {
      pidColView = firstCol;
    }

    auto [partitionedTable, partOffsets] = cudf::partition(
        dataTable, pidColView, static_cast<cudf::size_type>(numPartitions_), stream);
    VELOX_CHECK_EQ(partOffsets.size(), numPartitions_ + 1);
    offsets = std::move(partOffsets);

    // Synchronize before freeing the original input: cudf::partition() is
    // async on `stream` and reads from `dataTable` (which references
    // cudfVec's data). Without this sync, the input's GPU memory may be
    // freed while the partition kernel is still reading from it.
    stream.synchronize();
    pidColOwned.reset();
    cudfVec.reset();

    uint64_t totalEvicted = 0;
    for (uint32_t i = 0; i < numPartitions_; ++i) {
      totalEvicted += (offsets[i + 1] - offsets[i]);
    }
    LOG(WARNING) << "gpuPartitionAndEvict: partitioned rows=" << partitionedTable->num_rows()
                 << " dataCols=" << partitionedTable->num_columns()
                 << " totalInOffsets=" << totalEvicted
                 << " offsets[0]=" << offsets[0]
                 << " offsets[last]=" << offsets[numPartitions_];

    // D2H: convert the partitioned table to a Velox RowVector on host.
    veloxRv = cudf_velox::with_arrow::toVeloxColumn(
        partitionedTable->view(), veloxPool_.get(), std::string(""), stream);

    // partitionedTable (GPU) is released when this scope exits.
  }
  // GpuLockGuard released here — all GPU work is done.

  LOG(WARNING) << "gpuPartitionAndEvict: veloxRv rows=" << veloxRv->size()
               << " children=" << veloxRv->childrenSize();

  // CPU-side: extract buffers per partition directly from the full RowVector
  // using offsets (avoids sliced-vector offset pitfalls and enables zero-copy).
  uint64_t totalRowsEvicted = 0;
  for (uint32_t pid = 0; pid < numPartitions_; ++pid) {
    auto start = static_cast<int64_t>(offsets[pid]);
    auto numRows = static_cast<uint32_t>(offsets[pid + 1] - offsets[pid]);
    if (numRows == 0) {
      continue;
    }

    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    RETURN_NOT_OK(extractBuffersFromRowVector(*veloxRv, start, numRows, buffers));
    RETURN_NOT_OK(evictBuffers(pid, numRows, std::move(buffers), false));
    totalRowsEvicted += numRows;
  }

  LOG(WARNING) << "gpuPartitionAndEvict: totalRowsEvicted=" << totalRowsEvicted;
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
