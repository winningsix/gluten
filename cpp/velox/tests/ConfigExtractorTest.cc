/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0.
 */

#include "utils/ConfigExtractor.h"

#include <gtest/gtest.h>

#ifdef ENABLE_S3
#include "velox/connectors/hive/storage_adapters/s3fs/S3Config.h"
#endif

namespace gluten {
namespace {

#ifdef ENABLE_S3
using facebook::velox::config::ConfigBase;
using facebook::velox::filesystems::S3Config;

TEST(ConfigExtractorTest, mapsMultipartWriterConfiguration) {
  auto input = std::make_shared<ConfigBase>(std::unordered_map<std::string, std::string>{
      {"spark.hadoop.fs.s3a.multipart.size", "64m"}, {"spark.hadoop.fs.s3a.multipart-upload-threads", "8"}});

  const auto output = createHiveConnectorConfig(input, FileSystemType::kS3);
  EXPECT_EQ(output->get<std::string>(S3Config::baseConfigKey(S3Config::Keys::kMultipartMinPartSize)).value(), "64MB");
  EXPECT_EQ(output->get<std::string>(S3Config::baseConfigKey(S3Config::Keys::kMultipartUploadThreads)).value(), "8");
}

TEST(ConfigExtractorTest, mapsBareMultipartSizeAsBytes) {
  auto input = std::make_shared<ConfigBase>(
      std::unordered_map<std::string, std::string>{{"spark.hadoop.fs.s3a.multipart.size", "67108864"}});

  const auto output = createHiveConnectorConfig(input, FileSystemType::kS3);
  EXPECT_EQ(
      output->get<std::string>(S3Config::baseConfigKey(S3Config::Keys::kMultipartMinPartSize)).value(), "67108864B");
}

TEST(ConfigExtractorTest, acceptsLegacyAwsMultipartWriterConfiguration) {
  auto input = std::make_shared<ConfigBase>(std::unordered_map<std::string, std::string>{
      {"spark.hadoop.hive.s3.min-part-size", "32MB"}, {"spark.hadoop.hive.s3.multipart-upload-threads", "6"}});

  const auto output = createHiveConnectorConfig(input, FileSystemType::kS3);
  EXPECT_EQ(output->get<std::string>(S3Config::baseConfigKey(S3Config::Keys::kMultipartMinPartSize)).value(), "32MB");
  EXPECT_EQ(output->get<std::string>(S3Config::baseConfigKey(S3Config::Keys::kMultipartUploadThreads)).value(), "6");
}
#endif

} // namespace
} // namespace gluten
