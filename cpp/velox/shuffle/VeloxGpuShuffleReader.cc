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

#include "VeloxGpuShuffleReader.h"

#include <arrow/array/array_binary.h>
#include <arrow/io/buffered.h>

#include "memory/VeloxColumnarBatch.h"
#include "shuffle/Payload.h"
#include "shuffle/Utils.h"
#include "utils/Common.h"
#include "utils/Macros.h"
#include "utils/Timer.h"
#include "memory/GpuBufferColumnarBatch.h"

#include <algorithm>
#include <nvtx3/nvtx3.hpp>
#include "velox/experimental/cudf/exec/NvtxHelper.h"

using namespace facebook::velox;

namespace gluten {
namespace {

arrow::Result<BlockType> readBlockType(arrow::io::InputStream* inputStream) {
  BlockType type;
  ARROW_ASSIGN_OR_RAISE(auto bytes, inputStream->Read(sizeof(BlockType), &type));
  if (bytes == 0) {
    // Reach EOS.
    return BlockType::kEndOfStream;
  }
  return type;
}

} // namespace

VeloxGpuHashShuffleReaderDeserializer::VeloxGpuHashShuffleReaderDeserializer(
    const std::shared_ptr<StreamReader>& streamReader,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<arrow::util::Codec>& codec,
    const facebook::velox::RowTypePtr& rowType,
    int64_t readerBufferSize,
    VeloxMemoryManager* memoryManager,
    int64_t& deserializeTime,
    int64_t& decompressTime)
    : streamReader_(streamReader),
      schema_(schema),
      codec_(codec),
      rowType_(rowType),
      readerBufferSize_(readerBufferSize),
      memoryManager_(memoryManager),
      deserializeTime_(deserializeTime),
      decompressTime_(decompressTime) {}

bool VeloxGpuHashShuffleReaderDeserializer::resolveNextBlockType() {
  GLUTEN_ASSIGN_OR_THROW(auto blockType, readBlockType(in_.get()));
  switch (blockType) {
    case BlockType::kEndOfStream:
      return false;
    case BlockType::kPlainPayload:
      return true;
    default:
      throw GlutenException(fmt::format("Unsupported block type: {}", static_cast<int32_t>(blockType)));
  }
  return true;
}

void VeloxGpuHashShuffleReaderDeserializer::loadNextStream() {
  if (reachedEos_) {
    return;
  }

  auto in = streamReader_->readNextStream(memoryManager_->defaultArrowMemoryPool());
  if (in == nullptr) {
    reachedEos_ = true;
    return;
  }

  GLUTEN_ASSIGN_OR_THROW(
      in_,
      arrow::io::BufferedInputStream::Create(
          readerBufferSize_, memoryManager_->defaultArrowMemoryPool(), std::move(in)));
}

void
VeloxGpuHashShuffleReaderDeserializer::
    fillPrefetchQueue() {
  while (prefetchQueue_.size() < kMaxPrefetch &&
         !reachedEos_) {
    if (in_ == nullptr) {
      loadNextStream();
      if (reachedEos_) {
        return;
      }
    }
    while (!resolveNextBlockType()) {
      loadNextStream();
      if (reachedEos_) {
        return;
      }
    }
    GLUTEN_ASSIGN_OR_THROW(
        auto pending,
        BlockPayload::startDeserialize(
            in_.get(), codec_,
            memoryManager_->
                defaultArrowMemoryPool(),
            deserializeTime_));
    prefetchQueue_.push_back(
        std::move(pending));
  }
}

std::shared_ptr<ColumnarBatch>
VeloxGpuHashShuffleReaderDeserializer::next() {
  using VD =
      facebook::velox::cudf_velox::VeloxDomain;

  {
    nvtx3::scoped_range_in<VD> prefetchRange(
        nvtx3::event_attributes{
            "ShuffleRead::prefetch",
            nvtx3::rgb{100, 149, 237}});
    fillPrefetchQueue();
  }

  if (prefetchQueue_.empty()) {
    return nullptr;
  }

  auto pending =
      std::move(prefetchQueue_.front());
  prefetchQueue_.pop_front();
  auto numRows = pending.numRows;

  std::vector<std::shared_ptr<arrow::Buffer>>
      arrowBuffers;
  {
    nvtx3::scoped_range_in<VD> collectRange(
        nvtx3::event_attributes{
            "ShuffleRead::collectDecomp",
            nvtx3::rgb{255, 140, 0}});
    GLUTEN_ASSIGN_OR_THROW(
        arrowBuffers,
        BlockPayload::finishDeserialize(
            std::move(pending),
            decompressTime_));
  }

  return std::make_shared<GpuBufferColumnarBatch>(
      rowType_, std::move(arrowBuffers),
      static_cast<int32_t>(numRows));
}

} // namespace gluten
