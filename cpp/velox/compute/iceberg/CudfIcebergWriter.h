/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "velox/common/compression/Compression.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/dwio/common/ParquetFieldId.h"
#include "velox/vector/ComplexVector.h"

namespace gluten {

class CudfIcebergWriter {
 public:
  CudfIcebergWriter(
      facebook::velox::RowTypePtr rowType,
      std::string outputDirectory,
      facebook::velox::common::CompressionKind compressionKind,
      int32_t partitionId,
      int64_t taskId,
      std::string operationId,
      facebook::velox::connector::hive::iceberg::IcebergPartitionSpecPtr spec,
      facebook::velox::parquet::ParquetFieldId field,
      const std::unordered_map<std::string, std::string>& sparkConfs,
      std::shared_ptr<facebook::velox::memory::MemoryPool> sinkPool);

  ~CudfIcebergWriter();

  void write(const facebook::velox::RowVectorPtr& input);

  std::vector<std::string> commit();

  void abort();

  uint64_t numWrittenBytes() const;

  uint32_t numWrittenFiles() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace gluten
