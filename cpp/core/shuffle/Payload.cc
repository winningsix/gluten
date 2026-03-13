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
#include "shuffle/Payload.h"

#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/util/bitmap.h>
#include <arrow/util/compression.h>
#include <lz4.h>
#include <lz4frame.h>

#include "shuffle/Dictionary.h"
#include "shuffle/Options.h"
#include "shuffle/ShuffleCompressionPool.h"
#include "shuffle/Utils.h"
#include "utils/Exception.h"
#include "utils/Timer.h"

namespace gluten {
namespace {

static const Payload::Type kCompressedType =
    gluten::BlockPayload::kCompressed;
static const Payload::Type kUncompressedType =
    gluten::BlockPayload::kUncompressed;
static const Payload::Type kMergedCompressedType =
    gluten::BlockPayload::kMergedCompressed;

static constexpr int64_t kZeroLengthBuffer = 0;
static constexpr int64_t kNullBuffer = -1;
static constexpr int64_t kUncompressedBuffer = -2;

bool isLz4(arrow::util::Codec* codec) {
  return codec != nullptr &&
      codec->compression_type() ==
          arrow::Compression::LZ4_FRAME;
}

bool isLz4(
    const std::shared_ptr<arrow::util::Codec>& codec) {
  return isLz4(codec.get());
}

int64_t rawLz4MaxCompressedLen(int64_t inputSize) {
  return LZ4_compressBound(static_cast<int>(inputSize));
}

arrow::Result<int64_t> rawLz4Compress(
    const uint8_t* src, int64_t srcSize,
    uint8_t* dst, int64_t dstCapacity) {
  auto result = LZ4_compress_default(
      reinterpret_cast<const char*>(src),
      reinterpret_cast<char*>(dst),
      static_cast<int>(srcSize),
      static_cast<int>(dstCapacity));
  if (result <= 0) {
    return arrow::Status::IOError(
        "Raw LZ4 compression failed");
  }
  return static_cast<int64_t>(result);
}

arrow::Status rawLz4Decompress(
    const uint8_t* src, int64_t /*srcSize*/,
    uint8_t* dst, int64_t originalSize) {
  auto result = LZ4_decompress_fast(
      reinterpret_cast<const char*>(src),
      reinterpret_cast<char*>(dst),
      static_cast<int>(originalSize));
  if (result < 0) {
    return arrow::Status::IOError(
        "Raw LZ4 decompression failed, code=",
        std::to_string(result));
  }
  return arrow::Status::OK();
}

template <typename T>
void write(uint8_t** dst, T data) {
  memcpy(*dst, &data, sizeof(T));
  *dst += sizeof(T);
}

template <typename T>
T* advance(uint8_t** dst) {
  auto ptr = reinterpret_cast<T*>(*dst);
  *dst += sizeof(T);
  return ptr;
}

arrow::Result<uint8_t> readPayloadType(arrow::io::InputStream* is) {
  uint8_t type;
  ARROW_ASSIGN_OR_RAISE(auto bytes, is->Read(sizeof(Payload::Type), &type));
  ARROW_RETURN_IF(bytes == 0, arrow::Status::IOError("Failed to read bytes. Reached EOS."));
  return type;
}

arrow::Result<std::shared_ptr<arrow::Buffer>>
readUncompressedBuffer(
    arrow::io::InputStream* inputStream,
    arrow::MemoryPool* pool,
    int64_t& deserializedTime) {
  ScopedTimer timer(&deserializedTime);
  int64_t bufferLength;
  RETURN_NOT_OK(
      inputStream->Read(sizeof(int64_t), &bufferLength));
  if (bufferLength == kNullBuffer) {
    return nullptr;
  }
  ARROW_ASSIGN_OR_RAISE(
      auto buffer,
      arrow::AllocateResizableBuffer(bufferLength, pool));
  RETURN_NOT_OK(inputStream->Read(
      bufferLength, buffer->mutable_data()));
  return buffer;
}

int64_t bufMaxCompressedLen(
    int64_t rawSize,
    const uint8_t* data,
    arrow::util::Codec* codec) {
  if (isLz4(codec)) {
    return rawLz4MaxCompressedLen(rawSize);
  }
  return codec->MaxCompressedLen(rawSize, data);
}

static const int64_t kCompBufHeader = 2 * sizeof(int64_t);

// Per-buffer wire format:
//   [compressedLen : int64]  -1=null, 0=empty,
//                            -2=stored uncompressed
//   [uncompressedLen : int64]  (only if data)
//   [data]                     (only if data)
// Returns bytes written.
arrow::Result<int64_t> compressBuffer(
    const std::shared_ptr<arrow::Buffer>& buffer,
    uint8_t* output,
    int64_t outputCapacity,
    arrow::util::Codec* codec) {
  auto* out = &output;
  if (!buffer) {
    write<int64_t>(out, kNullBuffer);
    return sizeof(int64_t);
  }
  if (buffer->size() == 0) {
    write<int64_t>(out, kZeroLengthBuffer);
    return sizeof(int64_t);
  }
  auto* compLenSlot = advance<int64_t>(out);
  write<int64_t>(out, buffer->size());

  int64_t dataCapacity =
      outputCapacity - kCompBufHeader;
  int64_t compLen;
  if (isLz4(codec)) {
    ARROW_ASSIGN_OR_RAISE(
        compLen,
        rawLz4Compress(
            buffer->data(), buffer->size(),
            *out, dataCapacity));
  } else {
    ARROW_ASSIGN_OR_RAISE(
        compLen,
        codec->Compress(
            buffer->size(), buffer->data(),
            dataCapacity, *out));
  }
  if (compLen >= buffer->size()) {
    memcpy(*out, buffer->data(), buffer->size());
    *compLenSlot = kUncompressedBuffer;
    return kCompBufHeader + buffer->size();
  }
  *compLenSlot = compLen;
  return kCompBufHeader + compLen;
}

arrow::Result<std::shared_ptr<arrow::Buffer>>
readCompressedBuffer(
    arrow::io::InputStream* inputStream,
    const std::shared_ptr<arrow::util::Codec>& codec,
    arrow::MemoryPool* pool,
    int64_t& deserializeTime,
    int64_t& decompressTime) {
  ScopedTimer timer(&deserializeTime);
  int64_t compLen;
  RETURN_NOT_OK(
      inputStream->Read(sizeof(int64_t), &compLen));
  if (compLen == kNullBuffer) {
    return nullptr;
  }
  if (compLen == kZeroLengthBuffer) {
    return zeroLengthNullBuffer();
  }
  int64_t uncompLen;
  RETURN_NOT_OK(
      inputStream->Read(sizeof(int64_t), &uncompLen));

  if (compLen == kUncompressedBuffer) {
    ARROW_ASSIGN_OR_RAISE(
        auto buf,
        arrow::AllocateResizableBuffer(uncompLen, pool));
    RETURN_NOT_OK(
        inputStream->Read(uncompLen, buf->mutable_data()));
    return buf;
  }

  ARROW_ASSIGN_OR_RAISE(
      auto compressed,
      arrow::AllocateResizableBuffer(compLen, pool));
  RETURN_NOT_OK(inputStream->Read(
      compLen, compressed->mutable_data()));

  timer.switchTo(&decompressTime);
  ARROW_ASSIGN_OR_RAISE(
      auto output,
      arrow::AllocateResizableBuffer(uncompLen, pool));
  if (isLz4(codec)) {
    RETURN_NOT_OK(rawLz4Decompress(
        compressed->data(), compLen,
        output->mutable_data(), uncompLen));
  } else {
    RETURN_NOT_OK(codec->Decompress(
        compLen, compressed->data(),
        uncompLen, output->mutable_data()));
  }
  return output;
}

arrow::Status skipCompressedBuffer(
    arrow::io::InputStream* inputStream,
    int64_t& skipTime) {
  ScopedTimer timer(&skipTime);
  int64_t compLen;
  RETURN_NOT_OK(
      inputStream->Read(sizeof(int64_t), &compLen));
  if (compLen == kNullBuffer ||
      compLen == kZeroLengthBuffer) {
    return arrow::Status::OK();
  }
  int64_t uncompLen;
  RETURN_NOT_OK(
      inputStream->Read(sizeof(int64_t), &uncompLen));
  int64_t skip = (compLen == kUncompressedBuffer)
      ? uncompLen : compLen;
  ARROW_ASSIGN_OR_RAISE(
      auto unused, inputStream->Read(skip));
  return arrow::Status::OK();
}

int64_t maxMergedCompressedLength(
    const std::vector<std::shared_ptr<arrow::Buffer>>&
        buffers) {
  LZ4F_preferences_t prefs{};
  int64_t metaSize =
      static_cast<int64_t>(buffers.size())
          * sizeof(int64_t)
      + 2 * sizeof(int64_t);

  size_t frameBound = LZ4F_HEADER_SIZE_MAX;
  for (auto& buf : buffers) {
    if (buf && buf->size() > 0) {
      frameBound += LZ4F_compressBound(
          static_cast<size_t>(buf->size()), &prefs);
    }
  }
  frameBound += LZ4F_compressBound(0, &prefs);
  return metaSize + static_cast<int64_t>(frameBound);
}

arrow::Result<int64_t> compressBuffersMergedLz4(
    const std::vector<std::shared_ptr<arrow::Buffer>>&
        buffers,
    uint8_t* output,
    int64_t outputCapacity) {
  uint8_t* out = output;
  int64_t totalUncomp = 0;

  for (auto& buf : buffers) {
    int64_t meta;
    if (!buf) {
      meta = kNullBuffer;
    } else if (buf->size() == 0) {
      meta = kZeroLengthBuffer;
    } else {
      meta = buf->size();
      totalUncomp += buf->size();
    }
    write<int64_t>(&out, meta);
  }

  auto* compLenSlot = advance<int64_t>(&out);
  write<int64_t>(&out, totalUncomp);

  if (totalUncomp == 0) {
    *compLenSlot = 0;
    return static_cast<int64_t>(out - output);
  }

  auto* frameStart = out;
  auto remaining =
      outputCapacity - (out - output);

  LZ4F_cctx* ctx = nullptr;
  auto err = LZ4F_createCompressionContext(
      &ctx, LZ4F_VERSION);
  if (LZ4F_isError(err)) {
    return arrow::Status::IOError(
        "LZ4F_createCompressionContext failed: ",
        LZ4F_getErrorName(err));
  }

  LZ4F_preferences_t prefs{};

  auto n = LZ4F_compressBegin(
      ctx, out,
      static_cast<size_t>(remaining), &prefs);
  if (LZ4F_isError(n)) {
    LZ4F_freeCompressionContext(ctx);
    return arrow::Status::IOError(
        "LZ4F_compressBegin failed: ",
        LZ4F_getErrorName(n));
  }
  out += n;
  remaining -= static_cast<int64_t>(n);

  for (auto& buf : buffers) {
    if (!buf || buf->size() == 0) {
      continue;
    }
    n = LZ4F_compressUpdate(
        ctx, out,
        static_cast<size_t>(remaining),
        buf->data(),
        static_cast<size_t>(buf->size()),
        nullptr);
    if (LZ4F_isError(n)) {
      LZ4F_freeCompressionContext(ctx);
      return arrow::Status::IOError(
          "LZ4F_compressUpdate failed: ",
          LZ4F_getErrorName(n));
    }
    out += n;
    remaining -= static_cast<int64_t>(n);
  }

  n = LZ4F_compressEnd(
      ctx, out,
      static_cast<size_t>(remaining), nullptr);
  if (LZ4F_isError(n)) {
    LZ4F_freeCompressionContext(ctx);
    return arrow::Status::IOError(
        "LZ4F_compressEnd failed: ",
        LZ4F_getErrorName(n));
  }
  out += n;

  LZ4F_freeCompressionContext(ctx);

  *compLenSlot =
      static_cast<int64_t>(out - frameStart);
  return static_cast<int64_t>(out - output);
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>>
decompressMergedLz4Buffers(
    arrow::io::InputStream* inputStream,
    uint32_t numBuffers,
    arrow::MemoryPool* pool,
    int64_t& deserializeTime,
    int64_t& decompressTime) {
  ScopedTimer readTimer(&deserializeTime);

  std::vector<int64_t> bufMeta(numBuffers);
  for (uint32_t i = 0; i < numBuffers; i++) {
    RETURN_NOT_OK(inputStream->Read(
        sizeof(int64_t), &bufMeta[i]));
  }

  int64_t compLen;
  RETURN_NOT_OK(
      inputStream->Read(sizeof(int64_t), &compLen));
  int64_t totalUncomp;
  RETURN_NOT_OK(
      inputStream->Read(sizeof(int64_t), &totalUncomp));

  std::shared_ptr<arrow::Buffer> decompShared;
  if (totalUncomp > 0) {
    ARROW_ASSIGN_OR_RAISE(
        auto compressed,
        arrow::AllocateResizableBuffer(
            compLen, pool));
    RETURN_NOT_OK(inputStream->Read(
        compLen, compressed->mutable_data()));

    readTimer.switchTo(&decompressTime);

    ARROW_ASSIGN_OR_RAISE(
        auto decompressed,
        arrow::AllocateResizableBuffer(
            totalUncomp, pool));

    LZ4F_dctx* dctx = nullptr;
    auto err = LZ4F_createDecompressionContext(
        &dctx, LZ4F_VERSION);
    if (LZ4F_isError(err)) {
      return arrow::Status::IOError(
          "LZ4F_createDecompressionContext failed: ",
          LZ4F_getErrorName(err));
    }

    size_t srcSize =
        static_cast<size_t>(compLen);
    size_t dstSize =
        static_cast<size_t>(totalUncomp);
    auto ret = LZ4F_decompress(
        dctx,
        decompressed->mutable_data(), &dstSize,
        compressed->data(), &srcSize,
        nullptr);
    LZ4F_freeDecompressionContext(dctx);

    if (LZ4F_isError(ret)) {
      return arrow::Status::IOError(
          "LZ4F_decompress failed: ",
          LZ4F_getErrorName(ret));
    }
    decompShared = std::move(decompressed);
  }

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.reserve(numBuffers);
  int64_t offset = 0;
  for (uint32_t i = 0; i < numBuffers; i++) {
    if (bufMeta[i] == kNullBuffer) {
      buffers.push_back(nullptr);
    } else if (bufMeta[i] == kZeroLengthBuffer) {
      buffers.push_back(zeroLengthNullBuffer());
    } else {
      buffers.push_back(arrow::SliceBuffer(
          decompShared, offset, bufMeta[i]));
      offset += bufMeta[i];
    }
  }
  return buffers;
}

} // namespace

Payload::Payload(Payload::Type type, uint32_t numRows, const std::vector<bool>* isValidityBuffer)
    : type_(type), numRows_(numRows), isValidityBuffer_(isValidityBuffer) {}

std::string Payload::toString() const {
  switch (type_) {
    case kUncompressed:
      return "Payload::kUncompressed";
    case kCompressed:
      return "Payload::kCompressed";
    case kToBeCompressed:
      return "Payload::kToBeCompressed";
    case kMergedCompressed:
      return "Payload::kMergedCompressed";
    default:
      return "Payload::kRaw";
  }
}

arrow::Result<std::unique_ptr<BlockPayload>>
BlockPayload::fromBuffers(
    Payload::Type payloadType,
    uint32_t numRows,
    std::vector<std::shared_ptr<arrow::Buffer>> buffers,
    const std::vector<bool>* isValidityBuffer,
    arrow::MemoryPool* pool,
    arrow::util::Codec* codec) {
  const uint32_t numBuffers = buffers.size();

  if (payloadType == Payload::Type::kCompressed) {
    Timer compressionTime;
    compressionTime.start();

    if (isLz4(codec)) {
      auto maxLen =
          maxMergedCompressedLength(buffers);
      ARROW_ASSIGN_OR_RAISE(
          auto output,
          arrow::AllocateResizableBuffer(
              maxLen, pool));
      ARROW_ASSIGN_OR_RAISE(
          auto actualLen,
          compressBuffersMergedLz4(
              buffers,
              output->mutable_data(), maxLen));
      RETURN_NOT_OK(output->Resize(actualLen));

      compressionTime.stop();
      auto payload =
          std::unique_ptr<BlockPayload>(
              new BlockPayload(
                  Type::kMergedCompressed,
                  numRows, numBuffers,
                  {std::move(output)},
                  isValidityBuffer));
      payload->setCompressionTime(
          compressionTime.realTimeUsed());
      return payload;
    }

    auto maxLen =
        maxCompressedLength(buffers, codec);
    ARROW_ASSIGN_OR_RAISE(
        auto output,
        arrow::AllocateResizableBuffer(
            maxLen, pool));
    auto* out = output->mutable_data();

    int64_t actualLen = 0;
    for (auto& buf : buffers) {
      auto avail = maxLen - actualLen;
      ARROW_ASSIGN_OR_RAISE(
          auto written,
          compressBuffer(buf, out, avail, codec));
      out += written;
      actualLen += written;
    }
    RETURN_NOT_OK(output->Resize(actualLen));

    compressionTime.stop();
    auto payload = std::unique_ptr<BlockPayload>(
        new BlockPayload(
            Type::kCompressed, numRows, numBuffers,
            {std::move(output)}, isValidityBuffer));
    payload->setCompressionTime(
        compressionTime.realTimeUsed());
    return payload;
  }
  return std::unique_ptr<BlockPayload>(
      new BlockPayload(
          payloadType, numRows, numBuffers,
          std::move(buffers), isValidityBuffer));
}

arrow::Result<std::unique_ptr<BlockPayload>>
BlockPayload::compressBuffersForPool(
    uint32_t numRows,
    std::vector<std::shared_ptr<arrow::Buffer>> buffers,
    const std::vector<bool>* isValidityBuffer,
    arrow::util::Codec* codec) {
  return fromBuffers(
      Payload::kCompressed, numRows,
      std::move(buffers), isValidityBuffer,
      arrow::default_memory_pool(), codec);
}

arrow::Result<BlockPayload::PreparedCompression>
BlockPayload::prepareCompression(
    uint32_t numRows,
    std::vector<std::shared_ptr<arrow::Buffer>> buffers,
    const std::vector<bool>* isValidityBuffer,
    arrow::MemoryPool* pool,
    arrow::util::Codec* codec) {
  PreparedCompression pc;
  pc.numRows = numRows;
  pc.numBuffers = buffers.size();
  pc.isValidityBuffer = isValidityBuffer;
  pc.buffers = std::move(buffers);

  auto maxLen = isLz4(codec)
      ? maxMergedCompressedLength(pc.buffers)
      : maxCompressedLength(pc.buffers, codec);
  ARROW_ASSIGN_OR_RAISE(
      pc.output,
      arrow::AllocateResizableBuffer(maxLen, pool));
  return pc;
}

arrow::Result<std::unique_ptr<BlockPayload>>
BlockPayload::finishCompression(
    PreparedCompression&& pc,
    arrow::util::Codec* codec) {
  Timer compressionTime;
  compressionTime.start();

  if (isLz4(codec)) {
    ARROW_ASSIGN_OR_RAISE(
        auto totalLen,
        compressBuffersMergedLz4(
            pc.buffers,
            pc.output->mutable_data(),
            pc.output->size()));
    pc.buffers.clear();
    RETURN_NOT_OK(pc.output->Resize(totalLen));

    compressionTime.stop();
    auto payload = std::unique_ptr<BlockPayload>(
        new BlockPayload(
            Type::kMergedCompressed, pc.numRows,
            pc.numBuffers,
            {std::move(pc.output)},
            pc.isValidityBuffer));
    payload->setCompressionTime(
        compressionTime.realTimeUsed());
    return payload;
  }

  auto* out = pc.output->mutable_data();
  int64_t totalLen = 0;
  int64_t capacity = pc.output->size();

  for (auto& buf : pc.buffers) {
    ARROW_ASSIGN_OR_RAISE(
        auto written,
        compressBuffer(
            buf, out, capacity - totalLen, codec));
    out += written;
    totalLen += written;
    buf.reset();
  }
  pc.buffers.clear();

  RETURN_NOT_OK(pc.output->Resize(totalLen));

  compressionTime.stop();
  auto payload = std::unique_ptr<BlockPayload>(
      new BlockPayload(
          Type::kCompressed, pc.numRows,
          pc.numBuffers,
          {std::move(pc.output)},
          pc.isValidityBuffer));
  payload->setCompressionTime(
      compressionTime.realTimeUsed());
  return payload;
}

// Wire format header size:
// 1 (BlockType) + 1 (PayloadType) + 4 (numRows) + 4 (numBuffers)
static constexpr int64_t kWireHeaderSize = 10;

arrow::Result<BlockPayload::PreparedCompression>
BlockPayload::prepareWireFormat(
    uint32_t numRows,
    std::vector<std::shared_ptr<arrow::Buffer>> buffers,
    const std::vector<bool>* isValidityBuffer,
    arrow::MemoryPool* pool,
    arrow::util::Codec* codec) {
  PreparedCompression pc;
  pc.numRows = numRows;
  pc.numBuffers = buffers.size();
  pc.isValidityBuffer = isValidityBuffer;
  pc.buffers = std::move(buffers);

  auto maxLen = isLz4(codec)
      ? maxMergedCompressedLength(pc.buffers)
      : maxCompressedLength(pc.buffers, codec);
  ARROW_ASSIGN_OR_RAISE(
      pc.output,
      arrow::AllocateResizableBuffer(
          kWireHeaderSize + maxLen, pool));
  return pc;
}

arrow::Result<std::shared_ptr<arrow::Buffer>>
BlockPayload::finishWireFormat(
    PreparedCompression&& pc,
    arrow::util::Codec* codec) {
  auto* dataStart =
      pc.output->mutable_data() + kWireHeaderSize;
  int64_t capacity =
      pc.output->size() - kWireHeaderSize;
  int64_t totalLen = 0;
  Payload::Type payloadType;

  if (isLz4(codec)) {
    ARROW_ASSIGN_OR_RAISE(
        totalLen,
        compressBuffersMergedLz4(
            pc.buffers, dataStart, capacity));
    pc.buffers.clear();
    payloadType = Payload::kMergedCompressed;
  } else {
    auto* out = dataStart;
    for (auto& buf : pc.buffers) {
      ARROW_ASSIGN_OR_RAISE(
          auto written,
          compressBuffer(
              buf, out, capacity - totalLen,
              codec));
      out += written;
      totalLen += written;
      buf.reset();
    }
    pc.buffers.clear();
    payloadType = Payload::kCompressed;
  }

  auto* hdr = pc.output->mutable_data();
  hdr[0] = static_cast<uint8_t>(
      BlockType::kPlainPayload);
  hdr[1] = static_cast<uint8_t>(payloadType);
  memcpy(hdr + 2, &pc.numRows, sizeof(uint32_t));
  memcpy(hdr + 6, &pc.numBuffers, sizeof(uint32_t));

  RETURN_NOT_OK(pc.output->Resize(
      kWireHeaderSize + totalLen));
  return std::shared_ptr<arrow::Buffer>(
      std::move(pc.output));
}

arrow::Status BlockPayload::serialize(arrow::io::OutputStream* outputStream) {
  switch (type_) {
    case Type::kUncompressed: {
      ScopedTimer timer(&writeTime_);
      RETURN_NOT_OK(outputStream->Write(&kUncompressedType, sizeof(Type)));
      RETURN_NOT_OK(outputStream->Write(&numRows_, sizeof(uint32_t)));
      RETURN_NOT_OK(outputStream->Write(&numBuffers_, sizeof(uint32_t)));
      for (auto& buffer : buffers_) {
        if (!buffer) {
          RETURN_NOT_OK(outputStream->Write(&kNullBuffer, sizeof(int64_t)));
          continue;
        }
        int64_t bufferSize = buffer->size();
        RETURN_NOT_OK(outputStream->Write(&bufferSize, sizeof(int64_t)));
        if (bufferSize > 0) {
          RETURN_NOT_OK(outputStream->Write(std::move(buffer)));
        }
      }
    } break;
    case Type::kToBeCompressed: {
      ScopedTimer timer(&writeTime_);

      // No type and rows metadata for kToBeCompressed payload.
      RETURN_NOT_OK(outputStream->Write(&numBuffers_, sizeof(uint32_t)));

      for (auto& buffer : buffers_) {
        if (!buffer) {
          RETURN_NOT_OK(outputStream->Write(&kNullBuffer, sizeof(int64_t)));
          continue;
        }

        int64_t bufferSize = buffer->size();
        RETURN_NOT_OK(outputStream->Write(&bufferSize, sizeof(int64_t)));
        if (bufferSize > 0) {
          RETURN_NOT_OK(outputStream->Write(std::move(buffer)));
        }
      }
    } break;
    case Type::kCompressed: {
      ScopedTimer timer(&writeTime_);
      RETURN_NOT_OK(outputStream->Write(
          &kCompressedType, sizeof(Type)));
      RETURN_NOT_OK(outputStream->Write(
          &numRows_, sizeof(uint32_t)));
      RETURN_NOT_OK(outputStream->Write(
          &numBuffers_, sizeof(uint32_t)));
      RETURN_NOT_OK(
          outputStream->Write(std::move(buffers_[0])));
    } break;
    case Type::kMergedCompressed: {
      ScopedTimer timer(&writeTime_);
      RETURN_NOT_OK(outputStream->Write(
          &kMergedCompressedType, sizeof(Type)));
      RETURN_NOT_OK(outputStream->Write(
          &numRows_, sizeof(uint32_t)));
      RETURN_NOT_OK(outputStream->Write(
          &numBuffers_, sizeof(uint32_t)));
      RETURN_NOT_OK(
          outputStream->Write(std::move(buffers_[0])));
    } break;
    case Type::kRaw: {
      ScopedTimer timer(&writeTime_);
      RETURN_NOT_OK(outputStream->Write(std::move(buffers_[0])));
    } break;
  }
  buffers_.clear();
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Buffer>>
BlockPayload::readBufferAt(uint32_t pos) {
  if (type_ == Type::kCompressed ||
      type_ == Type::kMergedCompressed) {
    return arrow::Status::Invalid(
        "Cannot read buffer from compressed "
        "BlockPayload.");
  }
  if (type_ == Type::kRaw && pos != 0) {
    return arrow::Status::Invalid("Read buffer pos from raw should only be 0, but got " + std::to_string(pos));
  }
  return std::move(buffers_[pos]);
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>>
BlockPayload::deserialize(
    arrow::io::InputStream* inputStream,
    const std::shared_ptr<arrow::util::Codec>& codec,
    arrow::MemoryPool* pool,
    uint32_t& numRows,
    int64_t& deserializeTime,
    int64_t& decompressTime) {
  auto timer =
      std::make_unique<ScopedTimer>(&deserializeTime);
  ARROW_ASSIGN_OR_RAISE(
      auto type, readPayloadType(inputStream));

  RETURN_NOT_OK(
      inputStream->Read(sizeof(uint32_t), &numRows));
  uint32_t numBuffers;
  RETURN_NOT_OK(
      inputStream->Read(sizeof(uint32_t), &numBuffers));

  if (type == Type::kMergedCompressed) {
    timer.reset();
    return decompressMergedLz4Buffers(
        inputStream, numBuffers, pool,
        deserializeTime, decompressTime);
  }

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.reserve(numBuffers);

  if (type == Type::kCompressed) {
    timer.reset();
    for (uint32_t i = 0; i < numBuffers; i++) {
      ARROW_ASSIGN_OR_RAISE(
          auto buf,
          readCompressedBuffer(
              inputStream, codec, pool,
              deserializeTime, decompressTime));
      buffers.push_back(std::move(buf));
    }
  } else {
    timer.reset();
    for (uint32_t i = 0; i < numBuffers; ++i) {
      buffers.emplace_back();
      ARROW_ASSIGN_OR_RAISE(
          buffers.back(),
          readUncompressedBuffer(
              inputStream, pool, deserializeTime));
    }
  }
  return buffers;
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>>
BlockPayload::deserializeAsync(
    arrow::io::InputStream* inputStream,
    const std::shared_ptr<arrow::util::Codec>& codec,
    arrow::MemoryPool* pool,
    uint32_t& numRows,
    int64_t& deserializeTime,
    int64_t& decompressTime) {
  auto timer =
      std::make_unique<ScopedTimer>(&deserializeTime);
  ARROW_ASSIGN_OR_RAISE(
      auto type, readPayloadType(inputStream));

  RETURN_NOT_OK(
      inputStream->Read(sizeof(uint32_t), &numRows));
  uint32_t numBuffers;
  RETURN_NOT_OK(
      inputStream->Read(sizeof(uint32_t), &numBuffers));

  if (type == Type::kMergedCompressed) {
    timer.reset();

    ScopedTimer readTimer(&deserializeTime);
    std::vector<int64_t> bufMeta(numBuffers);
    for (uint32_t i = 0; i < numBuffers; i++) {
      RETURN_NOT_OK(inputStream->Read(
          sizeof(int64_t), &bufMeta[i]));
    }

    int64_t compLen;
    RETURN_NOT_OK(inputStream->Read(
        sizeof(int64_t), &compLen));
    int64_t totalUncomp;
    RETURN_NOT_OK(inputStream->Read(
        sizeof(int64_t), &totalUncomp));

    if (totalUncomp == 0) {
      std::vector<std::shared_ptr<arrow::Buffer>>
          buffers;
      buffers.reserve(numBuffers);
      for (uint32_t i = 0; i < numBuffers; i++) {
        if (bufMeta[i] == kNullBuffer) {
          buffers.push_back(nullptr);
        } else {
          buffers.push_back(
              zeroLengthNullBuffer());
        }
      }
      return buffers;
    }

    ARROW_ASSIGN_OR_RAISE(
        auto compressed,
        arrow::AllocateResizableBuffer(
            compLen, pool));
    RETURN_NOT_OK(inputStream->Read(
        compLen, compressed->mutable_data()));

    ARROW_ASSIGN_OR_RAISE(
        auto decompBuf,
        arrow::AllocateResizableBuffer(
            totalUncomp, pool));
    auto compShared =
        std::shared_ptr<arrow::Buffer>(
            std::move(compressed));
    auto decompShared =
        std::shared_ptr<arrow::Buffer>(
            std::move(decompBuf));
    auto decompRaw = decompShared->mutable_data();

    auto& pool2 =
        ShuffleCompressionPool::instance();
    auto future = pool2.submit(
        [compShared, decompShared, decompRaw,
         compLen, totalUncomp]()
            -> arrow::Result<
                std::shared_ptr<arrow::Buffer>> {
          LZ4F_dctx* dctx = nullptr;
          auto err =
              LZ4F_createDecompressionContext(
                  &dctx, LZ4F_VERSION);
          if (LZ4F_isError(err)) {
            return arrow::Status::IOError(
                "LZ4F decompress ctx failed: ",
                LZ4F_getErrorName(err));
          }
          size_t srcSz =
              static_cast<size_t>(compLen);
          size_t dstSz =
              static_cast<size_t>(totalUncomp);
          auto ret = LZ4F_decompress(
              dctx, decompRaw, &dstSz,
              compShared->data(), &srcSz,
              nullptr);
          LZ4F_freeDecompressionContext(dctx);
          if (LZ4F_isError(ret)) {
            return arrow::Status::IOError(
                "LZ4F_decompress failed: ",
                LZ4F_getErrorName(ret));
          }
          return decompShared;
        });

    std::shared_ptr<arrow::Buffer> decompResult;
    {
      ScopedTimer decompTimer(&decompressTime);
      ARROW_ASSIGN_OR_RAISE(
          decompResult, future.get());
    }

    std::vector<std::shared_ptr<arrow::Buffer>>
        buffers;
    buffers.reserve(numBuffers);
    int64_t offset = 0;
    for (uint32_t i = 0; i < numBuffers; i++) {
      if (bufMeta[i] == kNullBuffer) {
        buffers.push_back(nullptr);
      } else if (bufMeta[i] == kZeroLengthBuffer) {
        buffers.push_back(zeroLengthNullBuffer());
      } else {
        buffers.push_back(arrow::SliceBuffer(
            decompResult, offset, bufMeta[i]));
        offset += bufMeta[i];
      }
    }
    return buffers;
  }

  if (type != Type::kCompressed) {
    timer.reset();
    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    buffers.reserve(numBuffers);
    for (uint32_t i = 0; i < numBuffers; ++i) {
      buffers.emplace_back();
      ARROW_ASSIGN_OR_RAISE(
          buffers.back(),
          readUncompressedBuffer(
              inputStream, pool, deserializeTime));
    }
    return buffers;
  }

  // Per-buffer async path for non-LZ4 codecs.
  timer.reset();
  auto& decompPool = ShuffleCompressionPool::instance();

  struct PendingDecomp {
    std::future<
        arrow::Result<std::shared_ptr<arrow::Buffer>>>
        future;
    // nullptr for null/zero-length sentinels.
  };
  std::vector<PendingDecomp> pending;
  pending.reserve(numBuffers);

  for (uint32_t i = 0; i < numBuffers; i++) {
    ScopedTimer readTimer(&deserializeTime);
    int64_t compLen;
    RETURN_NOT_OK(
        inputStream->Read(sizeof(int64_t), &compLen));
    if (compLen == kNullBuffer) {
      pending.push_back({});
      continue;
    }
    if (compLen == kZeroLengthBuffer) {
      PendingDecomp pd;
      pd.future = std::async(
          std::launch::deferred,
          []() -> arrow::Result<
              std::shared_ptr<arrow::Buffer>> {
            return zeroLengthNullBuffer();
          });
      pending.push_back(std::move(pd));
      continue;
    }
    int64_t uncompLen;
    RETURN_NOT_OK(
        inputStream->Read(sizeof(int64_t), &uncompLen));

    if (compLen == kUncompressedBuffer) {
      ARROW_ASSIGN_OR_RAISE(
          auto buf,
          arrow::AllocateResizableBuffer(
              uncompLen, pool));
      RETURN_NOT_OK(inputStream->Read(
          uncompLen, buf->mutable_data()));
      PendingDecomp pd;
      auto bufPtr =
          std::shared_ptr<arrow::Buffer>(std::move(buf));
      pd.future = std::async(
          std::launch::deferred,
          [bufPtr]() -> arrow::Result<
              std::shared_ptr<arrow::Buffer>> {
            return bufPtr;
          });
      pending.push_back(std::move(pd));
      continue;
    }

    // Read compressed data (I/O on main thread).
    ARROW_ASSIGN_OR_RAISE(
        auto compressed,
        arrow::AllocateResizableBuffer(compLen, pool));
    RETURN_NOT_OK(inputStream->Read(
        compLen, compressed->mutable_data()));

    // Alloc output on main thread (keeps pool accounting
    // on the calling thread's allocator).
    ARROW_ASSIGN_OR_RAISE(
        auto output,
        arrow::AllocateResizableBuffer(uncompLen, pool));

    // Submit decompress to pool. Captures shared_ptr to
    // compressed (zero-copy handoff) and shared_ptr to
    // output (returned directly from the future).
    auto compShared = std::shared_ptr<arrow::Buffer>(
        std::move(compressed));
    auto outShared = std::shared_ptr<arrow::Buffer>(
        std::move(output));
    auto outRaw = outShared->mutable_data();
    auto codecPtr = codec;
    bool lz4 = isLz4(codec);

    PendingDecomp pd;
    pd.future = decompPool.submit(
        [compShared, outShared, outRaw, compLen,
         uncompLen, codecPtr, lz4]()
            -> arrow::Result<
                std::shared_ptr<arrow::Buffer>> {
          if (lz4) {
            RETURN_NOT_OK(rawLz4Decompress(
                compShared->data(), compLen,
                outRaw, uncompLen));
          } else {
            RETURN_NOT_OK(codecPtr->Decompress(
                compLen, compShared->data(),
                uncompLen, outRaw));
          }
          return outShared;
        });
    pending.push_back(std::move(pd));
  }

  // Phase 2: Collect results.
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.reserve(numBuffers);
  for (uint32_t i = 0; i < numBuffers; i++) {
    auto& pd = pending[i];
    if (!pd.future.valid()) {
      buffers.push_back(nullptr);
      continue;
    }
    ScopedTimer decompTimer(&decompressTime);
    ARROW_ASSIGN_OR_RAISE(auto buf, pd.future.get());
    buffers.push_back(std::move(buf));
  }

  return buffers;
}

arrow::Result<BlockPayload::PendingDecompression>
BlockPayload::startDeserialize(
    arrow::io::InputStream* inputStream,
    const std::shared_ptr<arrow::util::Codec>& codec,
    arrow::MemoryPool* pool,
    int64_t& deserializeTime) {
  PendingDecompression pd;
  {
    ScopedTimer timer(&deserializeTime);
    ARROW_ASSIGN_OR_RAISE(
        auto rawType,
        readPayloadType(inputStream));
    pd.type = static_cast<Type>(rawType);
    RETURN_NOT_OK(inputStream->Read(
        sizeof(uint32_t), &pd.numRows));
    RETURN_NOT_OK(inputStream->Read(
        sizeof(uint32_t), &pd.numBuffers));
  }

  if (pd.type == Type::kMergedCompressed) {
    ScopedTimer readTimer(&deserializeTime);
    pd.bufMeta.resize(pd.numBuffers);
    for (uint32_t i = 0; i < pd.numBuffers; i++) {
      RETURN_NOT_OK(inputStream->Read(
          sizeof(int64_t), &pd.bufMeta[i]));
    }
    int64_t compLen;
    RETURN_NOT_OK(inputStream->Read(
        sizeof(int64_t), &compLen));
    RETURN_NOT_OK(inputStream->Read(
        sizeof(int64_t), &pd.totalUncomp));
    pd.estimatedBytes = pd.totalUncomp;

    if (pd.totalUncomp == 0) {
      pd.readyBuffers.reserve(pd.numBuffers);
      for (uint32_t i = 0;
           i < pd.numBuffers; i++) {
        if (pd.bufMeta[i] == kNullBuffer) {
          pd.readyBuffers.push_back(nullptr);
        } else {
          pd.readyBuffers.push_back(
              zeroLengthNullBuffer());
        }
      }
      return pd;
    }

    ARROW_ASSIGN_OR_RAISE(
        auto compressed,
        arrow::AllocateResizableBuffer(
            compLen, pool));
    RETURN_NOT_OK(inputStream->Read(
        compLen, compressed->mutable_data()));
    ARROW_ASSIGN_OR_RAISE(
        auto decompBuf,
        arrow::AllocateResizableBuffer(
            pd.totalUncomp, pool));
    auto compShared =
        std::shared_ptr<arrow::Buffer>(
            std::move(compressed));
    auto decompShared =
        std::shared_ptr<arrow::Buffer>(
            std::move(decompBuf));
    auto* decompRaw =
        decompShared->mutable_data();
    auto& decompPool =
        ShuffleCompressionPool::instance();
    pd.mergedFuture = decompPool.submit(
        [compShared, decompShared, decompRaw,
         compLen,
         totalUncomp = pd.totalUncomp]()
            -> arrow::Result<
                std::shared_ptr<
                    arrow::Buffer>> {
          LZ4F_dctx* dctx = nullptr;
          auto err =
              LZ4F_createDecompressionContext(
                  &dctx, LZ4F_VERSION);
          if (LZ4F_isError(err)) {
            return arrow::Status::IOError(
                "LZ4F decompress ctx: ",
                LZ4F_getErrorName(err));
          }
          size_t srcSz =
              static_cast<size_t>(compLen);
          size_t dstSz =
              static_cast<size_t>(totalUncomp);
          auto ret = LZ4F_decompress(
              dctx, decompRaw, &dstSz,
              compShared->data(), &srcSz,
              nullptr);
          LZ4F_freeDecompressionContext(dctx);
          if (LZ4F_isError(ret)) {
            return arrow::Status::IOError(
                "LZ4F_decompress: ",
                LZ4F_getErrorName(ret));
          }
          return decompShared;
        });
    return pd;
  }

  if (pd.type != Type::kCompressed) {
    pd.readyBuffers.reserve(pd.numBuffers);
    pd.estimatedBytes = 0;
    for (uint32_t i = 0;
         i < pd.numBuffers; ++i) {
      ARROW_ASSIGN_OR_RAISE(
          auto buf,
          readUncompressedBuffer(
              inputStream, pool,
              deserializeTime));
      if (buf) {
        pd.estimatedBytes += buf->size();
      }
      pd.readyBuffers.push_back(
          std::move(buf));
    }
    return pd;
  }

  // kCompressed: per-buffer async.
  auto& decompPool =
      ShuffleCompressionPool::instance();
  pd.perBufferFutures.reserve(pd.numBuffers);
  pd.estimatedBytes = 0;

  for (uint32_t i = 0;
       i < pd.numBuffers; i++) {
    ScopedTimer readTimer(&deserializeTime);
    int64_t compLen;
    RETURN_NOT_OK(inputStream->Read(
        sizeof(int64_t), &compLen));
    if (compLen == kNullBuffer) {
      pd.perBufferFutures.push_back({});
      continue;
    }
    if (compLen == kZeroLengthBuffer) {
      PendingDecompression::PendingBuffer pb;
      pb.future = std::async(
          std::launch::deferred,
          []() -> arrow::Result<
              std::shared_ptr<arrow::Buffer>> {
            return zeroLengthNullBuffer();
          });
      pd.perBufferFutures.push_back(
          std::move(pb));
      continue;
    }
    int64_t uncompLen;
    RETURN_NOT_OK(inputStream->Read(
        sizeof(int64_t), &uncompLen));
    pd.estimatedBytes += uncompLen;

    if (compLen == kUncompressedBuffer) {
      ARROW_ASSIGN_OR_RAISE(
          auto buf,
          arrow::AllocateResizableBuffer(
              uncompLen, pool));
      RETURN_NOT_OK(inputStream->Read(
          uncompLen, buf->mutable_data()));
      auto bufPtr =
          std::shared_ptr<arrow::Buffer>(
              std::move(buf));
      PendingDecompression::PendingBuffer pb;
      pb.future = std::async(
          std::launch::deferred,
          [bufPtr]() -> arrow::Result<
              std::shared_ptr<arrow::Buffer>> {
            return bufPtr;
          });
      pd.perBufferFutures.push_back(
          std::move(pb));
      continue;
    }

    ARROW_ASSIGN_OR_RAISE(
        auto compressed,
        arrow::AllocateResizableBuffer(
            compLen, pool));
    RETURN_NOT_OK(inputStream->Read(
        compLen, compressed->mutable_data()));
    ARROW_ASSIGN_OR_RAISE(
        auto output,
        arrow::AllocateResizableBuffer(
            uncompLen, pool));
    auto compShared =
        std::shared_ptr<arrow::Buffer>(
            std::move(compressed));
    auto outShared =
        std::shared_ptr<arrow::Buffer>(
            std::move(output));
    auto* outRaw = outShared->mutable_data();
    auto codecPtr = codec;
    bool lz4 = isLz4(codec);

    PendingDecompression::PendingBuffer pb;
    pb.future = decompPool.submit(
        [compShared, outShared, outRaw,
         compLen, uncompLen, codecPtr, lz4]()
            -> arrow::Result<
                std::shared_ptr<
                    arrow::Buffer>> {
          if (lz4) {
            RETURN_NOT_OK(rawLz4Decompress(
                compShared->data(), compLen,
                outRaw, uncompLen));
          } else {
            RETURN_NOT_OK(
                codecPtr->Decompress(
                    compLen,
                    compShared->data(),
                    uncompLen, outRaw));
          }
          return outShared;
        });
    pd.perBufferFutures.push_back(
        std::move(pb));
  }

  return pd;
}

arrow::Result<
    std::vector<std::shared_ptr<arrow::Buffer>>>
BlockPayload::finishDeserialize(
    PendingDecompression&& pending,
    int64_t& decompressTime) {
  if (pending.type == Type::kMergedCompressed) {
    if (!pending.mergedFuture.valid()) {
      return std::move(pending.readyBuffers);
    }

    std::shared_ptr<arrow::Buffer> decompResult;
    {
      ScopedTimer timer(&decompressTime);
      ARROW_ASSIGN_OR_RAISE(
          decompResult,
          pending.mergedFuture.get());
    }

    std::vector<std::shared_ptr<arrow::Buffer>>
        buffers;
    buffers.reserve(pending.numBuffers);
    int64_t offset = 0;
    for (uint32_t i = 0;
         i < pending.numBuffers; i++) {
      if (pending.bufMeta[i] == kNullBuffer) {
        buffers.push_back(nullptr);
      } else if (pending.bufMeta[i] ==
                     kZeroLengthBuffer) {
        buffers.push_back(
            zeroLengthNullBuffer());
      } else {
        buffers.push_back(arrow::SliceBuffer(
            decompResult, offset,
            pending.bufMeta[i]));
        offset += pending.bufMeta[i];
      }
    }
    return buffers;
  }

  if (pending.type == Type::kCompressed) {
    std::vector<std::shared_ptr<arrow::Buffer>>
        buffers;
    buffers.reserve(pending.numBuffers);
    for (auto& pb : pending.perBufferFutures) {
      if (!pb.future.valid()) {
        buffers.push_back(nullptr);
        continue;
      }
      ScopedTimer timer(&decompressTime);
      ARROW_ASSIGN_OR_RAISE(
          auto buf, pb.future.get());
      buffers.push_back(std::move(buf));
    }
    return buffers;
  }

  return std::move(pending.readyBuffers);
}

arrow::Result<BlockPayload::BlockHeader> BlockPayload::readHeader(
    arrow::io::InputStream* inputStream,
    int64_t& deserializeTime) {
  ScopedTimer timer(&deserializeTime);
  BlockHeader header{};
  ARROW_ASSIGN_OR_RAISE(header.type, readPayloadType(inputStream));
  RETURN_NOT_OK(inputStream->Read(sizeof(uint32_t), &header.numRows));
  RETURN_NOT_OK(inputStream->Read(sizeof(uint32_t), &header.numBuffers));
  return header;
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>>
BlockPayload::readSelectedBuffers(
    arrow::io::InputStream* inputStream,
    const BlockHeader& header,
    const std::shared_ptr<arrow::util::Codec>& codec,
    arrow::MemoryPool* pool,
    const std::vector<bool>& bufferProjection,
    int64_t& deserializeTime,
    int64_t& decompressTime) {
  if (header.type == Type::kMergedCompressed) {
    ARROW_ASSIGN_OR_RAISE(
        auto allBuffers,
        decompressMergedLz4Buffers(
            inputStream, header.numBuffers, pool,
            deserializeTime, decompressTime));
    std::vector<std::shared_ptr<arrow::Buffer>>
        selected;
    selected.reserve(header.numBuffers);
    for (uint32_t i = 0; i < header.numBuffers;
         i++) {
      bool shouldRead =
          i < bufferProjection.size()
              ? bufferProjection[i] : true;
      if (shouldRead) {
        selected.push_back(
            std::move(allBuffers[i]));
      } else {
        selected.push_back(nullptr);
      }
    }
    return selected;
  }

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.reserve(header.numBuffers);

  if (header.type == Type::kCompressed) {
    for (uint32_t i = 0; i < header.numBuffers; i++) {
      bool shouldRead =
          i < bufferProjection.size()
              ? bufferProjection[i] : true;
      if (shouldRead) {
        ARROW_ASSIGN_OR_RAISE(
            auto buf,
            readCompressedBuffer(
                inputStream, codec, pool,
                deserializeTime, decompressTime));
        buffers.push_back(std::move(buf));
      } else {
        buffers.push_back(nullptr);
        RETURN_NOT_OK(skipCompressedBuffer(
            inputStream, deserializeTime));
      }
    }
  } else {
    for (uint32_t i = 0; i < header.numBuffers; ++i) {
      bool shouldRead =
          i < bufferProjection.size()
              ? bufferProjection[i] : true;
      if (shouldRead) {
        buffers.emplace_back();
        ARROW_ASSIGN_OR_RAISE(
            buffers.back(),
            readUncompressedBuffer(
                inputStream, pool, deserializeTime));
      } else {
        buffers.emplace_back(nullptr);
        ScopedTimer timer(&deserializeTime);
        int64_t bufferLength;
        RETURN_NOT_OK(inputStream->Read(
            sizeof(int64_t), &bufferLength));
        if (bufferLength > 0) {
          if (auto* mmapStream =
                  dynamic_cast<MmapFileStream*>(
                      inputStream)) {
            RETURN_NOT_OK(
                mmapStream->Advance(bufferLength));
          } else {
            ARROW_ASSIGN_OR_RAISE(
                auto unused,
                inputStream->Read(bufferLength));
          }
        }
      }
    }
  }
  return buffers;
}

void BlockPayload::setCompressionTime(int64_t compressionTime) {
  compressTime_ = compressionTime;
}

int64_t BlockPayload::rawSize() {
  return getBufferSize(buffers_);
}

int64_t BlockPayload::maxCompressedLength(
    const std::vector<std::shared_ptr<arrow::Buffer>>&
        buffers,
    arrow::util::Codec* codec) {
  int64_t total = 0;
  for (const auto& buf : buffers) {
    if (!buf || buf->size() == 0) {
      total += sizeof(int64_t);
    } else {
      total += kCompBufHeader +
          bufMaxCompressedLen(
              buf->size(), buf->data(), codec);
    }
  }
  return total;
}

arrow::Result<std::unique_ptr<InMemoryPayload>> InMemoryPayload::merge(
    std::unique_ptr<InMemoryPayload> source,
    std::unique_ptr<InMemoryPayload> append,
    arrow::MemoryPool* pool) {
  GLUTEN_DCHECK(source->mergeable() && append->mergeable(), "Cannot merge payloads.");
  auto mergedRows = source->numRows() + append->numRows();
  auto isValidityBuffer = source->isValidityBuffer();

  auto numBuffers = append->numBuffers();
  ARROW_RETURN_IF(
      numBuffers != source->numBuffers(), arrow::Status::Invalid("Number of merging buffers doesn't match."));
  std::vector<std::shared_ptr<arrow::Buffer>> merged;
  merged.resize(numBuffers);
  for (size_t i = 0; i < numBuffers; ++i) {
    ARROW_ASSIGN_OR_RAISE(auto sourceBuffer, source->readBufferAt(i));
    ARROW_ASSIGN_OR_RAISE(auto appendBuffer, append->readBufferAt(i));
    if (isValidityBuffer->at(i)) {
      if (!sourceBuffer) {
        if (!appendBuffer) {
          merged[i] = nullptr;
        } else {
          ARROW_ASSIGN_OR_RAISE(
              auto buffer, arrow::AllocateResizableBuffer(arrow::bit_util::BytesForBits(mergedRows), pool));
          // Source is null, fill all true.
          arrow::bit_util::SetBitsTo(buffer->mutable_data(), 0, source->numRows(), true);
          // Write append bits.
          arrow::internal::CopyBitmap(
              appendBuffer->data(), 0, append->numRows(), buffer->mutable_data(), source->numRows());
          merged[i] = std::move(buffer);
        }
      } else {
        // Because sourceBuffer can be resized, need to save buffer size in advance.
        auto sourceBufferSize = sourceBuffer->size();
        auto resizable = std::dynamic_pointer_cast<arrow::ResizableBuffer>(sourceBuffer);
        auto mergedBytes = arrow::bit_util::BytesForBits(mergedRows);
        if (resizable) {
          // If source is resizable, resize and reuse source.
          RETURN_NOT_OK(resizable->Resize(mergedBytes));
        } else {
          // Otherwise copy source.
          ARROW_ASSIGN_OR_RAISE(resizable, arrow::AllocateResizableBuffer(mergedBytes, pool));
          memcpy(resizable->mutable_data(), sourceBuffer->data(), sourceBufferSize);
        }
        if (!appendBuffer) {
          arrow::bit_util::SetBitsTo(resizable->mutable_data(), source->numRows(), append->numRows(), true);
        } else {
          arrow::internal::CopyBitmap(
              appendBuffer->data(), 0, append->numRows(), resizable->mutable_data(), source->numRows());
        }
        merged[i] = std::move(resizable);
      }
    } else {
      if (appendBuffer->size() == 0) {
        merged[i] = std::move(sourceBuffer);
      } else {
        // Because sourceBuffer can be resized, need to save buffer size in advance.
        auto sourceBufferSize = sourceBuffer->size();
        auto mergedSize = sourceBufferSize + appendBuffer->size();
        auto resizable = std::dynamic_pointer_cast<arrow::ResizableBuffer>(sourceBuffer);
        if (resizable) {
          // If source is resizable, resize and reuse source.
          RETURN_NOT_OK(resizable->Resize(mergedSize));
        } else {
          // Otherwise copy source.
          ARROW_ASSIGN_OR_RAISE(resizable, arrow::AllocateResizableBuffer(mergedSize, pool));
          memcpy(resizable->mutable_data(), sourceBuffer->data(), sourceBufferSize);
        }
        // Copy append.
        memcpy(resizable->mutable_data() + sourceBufferSize, appendBuffer->data(), appendBuffer->size());
        merged[i] = std::move(resizable);
      }
    }
  }
  return std::make_unique<InMemoryPayload>(mergedRows, isValidityBuffer, source->schema(), std::move(merged));
}

arrow::Result<std::unique_ptr<BlockPayload>>
InMemoryPayload::toBlockPayload(
    Payload::Type payloadType,
    arrow::MemoryPool* pool,
    arrow::util::Codec* codec) {
  return BlockPayload::fromBuffers(
      payloadType, numRows_, std::move(buffers_), isValidityBuffer_, pool, codec);
}

arrow::Status InMemoryPayload::serialize(arrow::io::OutputStream* outputStream) {
  for (auto& buffer : buffers_) {
    RETURN_NOT_OK(outputStream->Write(buffer->data(), buffer->size()));
    buffer = nullptr;
  }
  buffers_.clear();
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> InMemoryPayload::readBufferAt(uint32_t index) {
  GLUTEN_CHECK(
      index < buffers_.size(),
      "buffer index out of range: index = " + std::to_string(index) +
          " vs buffer size = " + std::to_string(buffers_.size()));
  return std::move(buffers_[index]);
}

arrow::Status InMemoryPayload::copyBuffers(arrow::MemoryPool* pool) {
  for (auto& buffer : buffers_) {
    if (!buffer) {
      continue;
    }
    if (buffer->size() == 0) {
      buffer = zeroLengthNullBuffer();
      continue;
    }
    ARROW_ASSIGN_OR_RAISE(auto copy, arrow::AllocateResizableBuffer(buffer->size(), pool));
    memcpy(copy->mutable_data(), buffer->data(), buffer->size());
    buffer = std::move(copy);
  }
  return arrow::Status::OK();
}

int64_t InMemoryPayload::rawSize() {
  return getBufferSize(buffers_);
}

uint32_t InMemoryPayload::numBuffers() const {
  return buffers_.size();
}

int64_t InMemoryPayload::rawCapacity() const {
  return getBufferCapacity(buffers_);
}

bool InMemoryPayload::mergeable() const {
  return !hasComplexType_;
}

std::shared_ptr<arrow::Schema> InMemoryPayload::schema() const {
  return schema_;
}

arrow::Status InMemoryPayload::createDictionaries(const std::shared_ptr<ShuffleDictionaryWriter>& dictionaryWriter) {
  ARROW_ASSIGN_OR_RAISE(buffers_, dictionaryWriter->updateAndGet(schema_, numRows_, buffers_));
  return arrow::Status::OK();
}

UncompressedDiskBlockPayload::UncompressedDiskBlockPayload(
    Type type,
    uint32_t numRows,
    const std::vector<bool>* isValidityBuffer,
    arrow::io::InputStream*& inputStream,
    uint64_t rawSize,
    arrow::MemoryPool* pool,
    arrow::util::Codec* codec)
    : Payload(type, numRows, isValidityBuffer),
      inputStream_(inputStream),
      rawSize_(rawSize),
      pool_(pool),
      codec_(codec) {}

arrow::Status UncompressedDiskBlockPayload::serialize(
    arrow::io::OutputStream* outputStream) {
  ARROW_RETURN_IF(
      inputStream_ == nullptr,
      arrow::Status::Invalid(
          "inputStream_ is uninitialized before"
          " calling serialize()."));

  if (type_ == Payload::kUncompressed) {
    ARROW_ASSIGN_OR_RAISE(
        auto block, inputStream_->Read(rawSize_));
    RETURN_NOT_OK(outputStream->Write(block));
    return arrow::Status::OK();
  }

  GLUTEN_DCHECK(
      type_ == Payload::kToBeCompressed,
      "Invalid payload type: " +
          std::to_string(type_) +
          ", should be either kUncompressed"
          " or kToBeCompressed");

  GLUTEN_CHECK(
      codec_ != nullptr,
      "Codec is null when serializing"
      " Payload::kToBeCompressed.");

  uint8_t blockType;
  ARROW_ASSIGN_OR_RAISE(
      auto bytes,
      inputStream_->Read(sizeof(blockType), &blockType));
  ARROW_RETURN_IF(
      bytes == 0,
      arrow::Status::Invalid(
          "Cannot serialize payload. Reached EOS."));

  uint32_t numBuffers = 0;
  ARROW_ASSIGN_OR_RAISE(
      bytes,
      inputStream_->Read(
          sizeof(uint32_t), &numBuffers));
  ARROW_RETURN_IF(
      bytes == 0 || numBuffers == 0,
      arrow::Status::Invalid(
          "Cannot serialize with 0 buffers."));

  ARROW_ASSIGN_OR_RAISE(auto start, inputStream_->Tell());
  auto pos = start;
  auto rawBufBytes =
      rawSize_ - sizeof(blockType) - sizeof(numBuffers);

  std::vector<std::shared_ptr<arrow::Buffer>> rawBufs;
  rawBufs.reserve(numBuffers);

  {
    ScopedTimer wt(&writeTime_);
    while (pos - start < rawBufBytes) {
      ARROW_ASSIGN_OR_RAISE(
          auto buf, readUncompressedBuffer());
      ARROW_ASSIGN_OR_RAISE(
          pos, inputStream_->Tell());
      rawBufs.push_back(std::move(buf));
    }
  }

  GLUTEN_CHECK(
      pos - start == rawBufBytes,
      "Not all data is read from input stream.");

  // Write header.
  ScopedTimer wt(&writeTime_);
  RETURN_NOT_OK(
      outputStream->Write(
          &blockType, sizeof(blockType)));
  RETURN_NOT_OK(outputStream->Write(
      &kCompressedType, sizeof(kCompressedType)));
  RETURN_NOT_OK(
      outputStream->Write(
          &numRows_, sizeof(uint32_t)));
  auto numBufs =
      static_cast<uint32_t>(rawBufs.size());
  RETURN_NOT_OK(outputStream->Write(
      &numBufs, sizeof(uint32_t)));

  // Per-buffer compress and write.
  wt.switchTo(&compressTime_);
  for (auto& buf : rawBufs) {
    int64_t maxLen = sizeof(int64_t);
    if (buf && buf->size() > 0) {
      maxLen = kCompBufHeader +
          bufMaxCompressedLen(
              buf->size(), buf->data(), codec_);
    }
    ARROW_ASSIGN_OR_RAISE(
        auto compBuf,
        arrow::AllocateResizableBuffer(
            maxLen, pool_));
    ARROW_ASSIGN_OR_RAISE(
        auto written,
        compressBuffer(
            buf, compBuf->mutable_data(),
            maxLen, codec_));
    wt.switchTo(&writeTime_);
    RETURN_NOT_OK(outputStream->Write(
        compBuf->data(), written));
    wt.switchTo(&compressTime_);
  }

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> UncompressedDiskBlockPayload::readUncompressedBuffer() {
  ScopedTimer timer(&writeTime_);

  int64_t bufferLength;
  RETURN_NOT_OK(inputStream_->Read(sizeof(int64_t), &bufferLength));
  if (bufferLength == kNullBuffer) {
    return nullptr;
  }
  if (bufferLength == 0) {
    return zeroLengthNullBuffer();
  }
  ARROW_ASSIGN_OR_RAISE(auto buffer, inputStream_->Read(bufferLength));
  return buffer;
}

int64_t UncompressedDiskBlockPayload::rawSize() {
  return rawSize_;
}

CompressedDiskBlockPayload::CompressedDiskBlockPayload(
    uint32_t numRows,
    const std::vector<bool>* isValidityBuffer,
    arrow::io::InputStream*& inputStream,
    int64_t rawSize,
    arrow::MemoryPool* /* pool */)
    : Payload(Type::kCompressed, numRows, isValidityBuffer), inputStream_(inputStream), rawSize_(rawSize) {}

arrow::Status CompressedDiskBlockPayload::serialize(arrow::io::OutputStream* outputStream) {
  ARROW_RETURN_IF(
      inputStream_ == nullptr, arrow::Status::Invalid("inputStream_ is uninitialized before calling serialize()."));
  ScopedTimer timer(&writeTime_);
  ARROW_ASSIGN_OR_RAISE(auto block, inputStream_->Read(rawSize_));
  RETURN_NOT_OK(outputStream->Write(block));
  return arrow::Status::OK();
}

int64_t CompressedDiskBlockPayload::rawSize() {
  return rawSize_;
}
} // namespace gluten
