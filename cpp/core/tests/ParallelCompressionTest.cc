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
#include <arrow/memory_pool.h>
#include <arrow/util/compression.h>
#include <gtest/gtest.h>

#include <random>
#include <vector>

namespace gluten {

class CompressionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = arrow::default_memory_pool();
    auto codecResult = arrow::util::Codec::Create(
        arrow::Compression::LZ4_FRAME);
    ASSERT_TRUE(codecResult.ok());
    codec_ = std::move(*codecResult);
  }

  std::shared_ptr<arrow::Buffer> makeRandomBuffer(
      int64_t size, uint32_t seed = 42) {
    auto result = arrow::AllocateResizableBuffer(size, pool_);
    EXPECT_TRUE(result.ok());
    auto buf = std::move(*result);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    auto* data = buf->mutable_data();
    for (int64_t i = 0; i < size; i++) {
      data[i] = dist(gen);
    }
    return std::move(buf);
  }

  std::shared_ptr<arrow::Buffer> makeCompressibleBuffer(
      int64_t size, uint32_t seed = 42) {
    auto result = arrow::AllocateResizableBuffer(size, pool_);
    EXPECT_TRUE(result.ok());
    auto buf = std::move(*result);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint8_t> dist(0, 10);
    auto* data = buf->mutable_data();
    for (int64_t i = 0; i < size; i++) {
      data[i] = dist(gen);
    }
    return std::move(buf);
  }

  std::shared_ptr<arrow::Buffer> serializePayload(
      std::unique_ptr<BlockPayload>& payload) {
    auto sink =
        arrow::io::BufferOutputStream::Create(1024 * 1024, pool_);
    EXPECT_TRUE(sink.ok());
    auto os = *sink;
    auto status = payload->serialize(os.get());
    EXPECT_TRUE(status.ok());
    auto result = os->Finish();
    EXPECT_TRUE(result.ok());
    return *result;
  }

  arrow::MemoryPool* pool_;
  std::unique_ptr<arrow::util::Codec> codec_;
};

TEST_F(CompressionTest, BasicCompression) {
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  std::vector<bool> isValidity = {false, false, false};
  for (int i = 0; i < 3; i++) {
    buffers.push_back(makeCompressibleBuffer(64 * 1024, i));
  }

  auto result = BlockPayload::fromBuffers(
      Payload::kCompressed, 100, std::move(buffers),
      &isValidity, pool_, codec_.get());
  ASSERT_TRUE(result.ok());
  auto payload = std::move(*result);
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->type(), Payload::kCompressed);
  EXPECT_GT(payload->getCompressTime(), 0);
}

TEST_F(CompressionTest, WithNullAndEmptyBuffers) {
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  std::vector<bool> isValidity =
      {false, false, false, false, false};

  buffers.push_back(makeCompressibleBuffer(64 * 1024, 0));
  buffers.push_back(nullptr);
  buffers.push_back(makeCompressibleBuffer(128 * 1024, 2));
  buffers.push_back(
      std::make_shared<arrow::Buffer>(nullptr, 0));
  buffers.push_back(makeCompressibleBuffer(32 * 1024, 4));

  auto result = BlockPayload::fromBuffers(
      Payload::kCompressed, 50, std::move(buffers),
      &isValidity, pool_, codec_.get());
  ASSERT_TRUE(result.ok());
  auto payload = std::move(*result);
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->type(), Payload::kCompressed);
}

TEST_F(CompressionTest, SingleBuffer) {
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  std::vector<bool> isValidity = {false};
  buffers.push_back(makeCompressibleBuffer(64 * 1024, 0));

  auto result = BlockPayload::fromBuffers(
      Payload::kCompressed, 10, std::move(buffers),
      &isValidity, pool_, codec_.get());
  ASSERT_TRUE(result.ok());
  auto payload = std::move(*result);
  ASSERT_NE(payload, nullptr);
}

TEST_F(CompressionTest, UncompressedType) {
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  std::vector<bool> isValidity = {false, false};
  buffers.push_back(makeRandomBuffer(1024, 0));
  buffers.push_back(makeRandomBuffer(2048, 1));

  auto result = BlockPayload::fromBuffers(
      Payload::kUncompressed, 10, std::move(buffers),
      &isValidity, pool_, codec_.get());
  ASSERT_TRUE(result.ok());
  auto payload = std::move(*result);
  EXPECT_EQ(payload->type(), Payload::kUncompressed);
}

TEST_F(CompressionTest, DeserializationRoundTrip) {
  const int numBuffers = 6;
  const int64_t bufferSize = 64 * 1024;
  std::vector<bool> isValidity(numBuffers, false);

  std::vector<std::vector<uint8_t>> originalData(numBuffers);
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  for (int i = 0; i < numBuffers; i++) {
    auto buf = makeCompressibleBuffer(bufferSize, i + 100);
    originalData[i].assign(
        buf->data(), buf->data() + buf->size());
    buffers.push_back(buf);
  }

  auto compressResult = BlockPayload::fromBuffers(
      Payload::kCompressed, 300, std::move(buffers),
      &isValidity, pool_, codec_.get());
  ASSERT_TRUE(compressResult.ok());
  auto payload = std::move(*compressResult);

  auto bytes = serializePayload(payload);

  auto readCodecResult = arrow::util::Codec::Create(
      arrow::Compression::LZ4_FRAME);
  ASSERT_TRUE(readCodecResult.ok());
  auto readCodec = std::shared_ptr<arrow::util::Codec>(
      std::move(*readCodecResult));

  auto inputStream =
      std::make_shared<arrow::io::BufferReader>(bytes);
  uint32_t numRows = 0;
  int64_t deserializeTime = 0;
  int64_t decompressTime = 0;

  auto decompressResult = BlockPayload::deserialize(
      inputStream.get(), readCodec, pool_,
      numRows, deserializeTime, decompressTime);
  ASSERT_TRUE(decompressResult.ok());
  auto decompressedBuffers = std::move(*decompressResult);

  ASSERT_EQ(numRows, 300);
  ASSERT_EQ(decompressedBuffers.size(), numBuffers);

  for (int i = 0; i < numBuffers; i++) {
    ASSERT_NE(decompressedBuffers[i], nullptr);
    ASSERT_EQ(
        decompressedBuffers[i]->size(),
        static_cast<int64_t>(originalData[i].size()));
    EXPECT_EQ(
        memcmp(
            decompressedBuffers[i]->data(),
            originalData[i].data(),
            originalData[i].size()),
        0)
        << "Buffer " << i
        << " mismatch after compress -> decompress roundtrip";
  }
}

TEST_F(CompressionTest, ManyBuffers) {
  const int numBuffers = 16;
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  std::vector<bool> isValidity(numBuffers, false);
  for (int i = 0; i < numBuffers; i++) {
    buffers.push_back(makeCompressibleBuffer(32 * 1024, i));
  }

  auto result = BlockPayload::fromBuffers(
      Payload::kCompressed, 500, std::move(buffers),
      &isValidity, pool_, codec_.get());
  ASSERT_TRUE(result.ok());
  auto payload = std::move(*result);
  ASSERT_NE(payload, nullptr);
}

TEST_F(CompressionTest, WithZstdCodec) {
  auto zstdResult =
      arrow::util::Codec::Create(arrow::Compression::ZSTD);
  ASSERT_TRUE(zstdResult.ok());
  auto zstdCodec = std::move(*zstdResult);

  const int numBuffers = 4;
  std::vector<bool> isValidity(numBuffers, false);

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  for (int i = 0; i < numBuffers; i++) {
    buffers.push_back(
        makeCompressibleBuffer(64 * 1024, i + 200));
  }

  auto result = BlockPayload::fromBuffers(
      Payload::kCompressed, 100, std::move(buffers),
      &isValidity, pool_, zstdCodec.get());
  ASSERT_TRUE(result.ok());
  auto payload = std::move(*result);
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->type(), Payload::kCompressed);
}

} // namespace gluten
