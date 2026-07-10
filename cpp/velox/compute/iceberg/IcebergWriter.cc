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

#include "IcebergWriter.h"

#include "IcebergPartitionSpec.pb.h"
#include "compute/ProtobufUtils.h"
#include "compute/iceberg/IcebergFormat.h"
#include "config/VeloxConfig.h"
#include "utils/ConfigExtractor.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/expression/Expr.h"

#ifdef GLUTEN_ENABLE_GPU
#include "compute/iceberg/CudfIcebergWriter.h"
#endif

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::connector::hive::iceberg;
namespace {
parquet::ParquetFieldId convertToParquetFieldId(const gluten::IcebergNestedField& protoField) {
  parquet::ParquetFieldId result;
  result.fieldId = protoField.id();

  // Recursively convert children
  result.children.reserve(protoField.children_size());
  for (const auto& protoChild : protoField.children()) {
    result.children.push_back(convertToParquetFieldId(protoChild));
  }

  return result;
}

std::shared_ptr<IcebergInsertTableHandle> createIcebergInsertTableHandle(
    const RowTypePtr& outputRowType,
    const std::string& outputDirectoryPath,
    dwio::common::FileFormat fileFormat,
    facebook::velox::common::CompressionKind compressionKind,
    std::shared_ptr<const IcebergPartitionSpec> spec,
    const parquet::ParquetFieldId& nestedField) {
  std::vector<std::shared_ptr<const iceberg::IcebergColumnHandle>> columnHandles;

  std::vector<std::string> columnNames = outputRowType->names();
  columnHandles.reserve(columnNames.size());
  std::vector<TypePtr> columnTypes = outputRowType->children();
  std::vector<std::string> partitionColumns;
  partitionColumns.reserve(spec->fields.size());
  for (const auto& field : spec->fields) {
    partitionColumns.push_back(field.name);
  }
  for (auto i = 0; i < columnNames.size(); ++i) {
    if (std::find(partitionColumns.begin(), partitionColumns.end(), columnNames[i]) != partitionColumns.end()) {
      columnHandles.push_back(
          std::make_shared<iceberg::IcebergColumnHandle>(
              columnNames.at(i),
              connector::hive::HiveColumnHandle::ColumnType::kPartitionKey,
              columnTypes.at(i),
              nestedField.children[i]));
    } else {
      columnHandles.push_back(
          std::make_shared<iceberg::IcebergColumnHandle>(
              columnNames.at(i),
              connector::hive::HiveColumnHandle::ColumnType::kRegular,
              columnTypes.at(i),
              nestedField.children[i]));
    }
  }
  
  std::shared_ptr<const connector::hive::LocationHandle> locationHandle =
      std::make_shared<connector::hive::LocationHandle>(
          outputDirectoryPath, outputDirectoryPath, connector::hive::LocationHandle::TableType::kExisting);
  const std::unordered_map<std::string, std::string> serdeParameters;
  return std::make_shared<connector::hive::iceberg::IcebergInsertTableHandle>(
      columnHandles, locationHandle, fileFormat, spec, compressionKind, serdeParameters);
}

} // namespace

namespace gluten {
IcebergWriter::IcebergWriter(
    const RowTypePtr& rowType,
    int32_t format,
    const std::string& outputDirectory,
    facebook::velox::common::CompressionKind compressionKind,
    int32_t partitionId,
    int64_t taskId,
    const std::string& operationId,
    std::shared_ptr<const iceberg::IcebergPartitionSpec> spec,
    const gluten::IcebergNestedField& field,
    const std::unordered_map<std::string, std::string>& sparkConfs,
    std::shared_ptr<facebook::velox::memory::MemoryPool> memoryPool,
    std::shared_ptr<facebook::velox::memory::MemoryPool> connectorPool)
    : rowType_(rowType),
      field_(convertToParquetFieldId(field)),
      partitionId_(partitionId),
      taskId_(taskId),
      operationId_(operationId),
      pool_(memoryPool),
      connectorPool_(connectorPool),
      createTimeNs_(getCurrentTimeNano()) {
#ifdef GLUTEN_ENABLE_GPU
  VELOX_USER_CHECK_EQ(
      icebergFormatToVelox(format),
      dwio::common::FileFormat::PARQUET,
      "Iceberg GPU writer supports Parquet only");
  cudfWriter_ = std::make_unique<CudfIcebergWriter>(
      rowType_,
      outputDirectory,
      compressionKind,
      partitionId_,
      taskId_,
      operationId_,
      std::move(spec),
      field_,
      sparkConfs,
      pool_);
#else
  auto veloxCfg = std::make_shared<facebook::velox::config::ConfigBase>(
      std::unordered_map<std::string, std::string>(sparkConfs));
  connectorSessionProperties_ = createHiveConnectorSessionConfig(veloxCfg);
  connectorConfig_ =
      std::make_shared<facebook::velox::connector::hive::HiveConfig>(
          createHiveConnectorConfig(veloxCfg));
  icebergConfig_ = std::make_shared<
      facebook::velox::connector::hive::iceberg::IcebergConfig>(veloxCfg);
  queryCtx_ = facebook::velox::core::QueryCtx::create(
      nullptr,
      facebook::velox::core::QueryConfig(veloxCfg->rawConfigs()));
  connectorQueryCtx_ = std::make_unique<connector::ConnectorQueryCtx>(
      pool_.get(),
      connectorPool_.get(),
      connectorSessionProperties_.get(),
      nullptr,
      common::PrefixSortConfig(),
      std::make_unique<facebook::velox::exec::SimpleExpressionEvaluator>(
          queryCtx_.get(), pool_.get()),
      nullptr,
      "query.IcebergDataSink",
      "task.IcebergDataSink",
      "planNodeId.IcebergDataSink",
      0,
      "");

  dataSink_ = std::make_unique<IcebergDataSink>(
      rowType_,
      createIcebergInsertTableHandle(
          rowType_, outputDirectory, icebergFormatToVelox(format), compressionKind, spec, field_),
      connectorQueryCtx_.get(),
      facebook::velox::connector::CommitStrategy::kNoCommit,
      connectorConfig_,
      icebergConfig_);
#endif
}

IcebergWriter::~IcebergWriter() = default;

void IcebergWriter::write(const VeloxColumnarBatch& batch) {
  const auto input = batch.getRowVector();
#ifdef GLUTEN_ENABLE_GPU
  cudfWriter_->write(input);
#else
  VELOX_USER_CHECK_EQ(
      input->childrenSize(),
      rowType_->size(),
      "Iceberg input column count does not match the table schema");
  for (auto i = 0; i < input->childrenSize(); ++i) {
    VELOX_USER_CHECK(
        input->childAt(i)->type()->kindEquals(rowType_->childAt(i)),
        "Iceberg input column {} has type {}, expected {}",
        i,
        input->childAt(i)->type()->toString(),
        rowType_->childAt(i)->toString());
  }

  // Native plans use internal field names (for example, n1_3), while
  // Iceberg partition transforms bind by table field name. Re-wrap the same
  // vectors with the table schema without copying their data.
  auto normalizedInput = std::make_shared<RowVector>(
      pool_.get(),
      rowType_,
      input->nulls(),
      input->size(),
      input->children(),
      input->getNullCount());
  dataSink_->appendData(std::move(normalizedInput));
#endif
}

std::vector<std::string> IcebergWriter::commit() {
#ifdef GLUTEN_ENABLE_GPU
  return cudfWriter_->commit();
#else
  auto finished = dataSink_->finish();
  VELOX_CHECK(finished);
  return dataSink_->close();
#endif
}

void IcebergWriter::abort() {
#ifdef GLUTEN_ENABLE_GPU
  cudfWriter_->abort();
#else
  dataSink_->abort();
#endif
}

WriteStats IcebergWriter::writeStats() const {
  const auto currentTimeNs = getCurrentTimeNano();
  VELOX_CHECK_GE(currentTimeNs, createTimeNs_);
#ifdef GLUTEN_ENABLE_GPU
  return WriteStats(
      cudfWriter_->numWrittenBytes(),
      cudfWriter_->numWrittenFiles(),
      0,
      currentTimeNs - createTimeNs_);
#else
  const auto sinkStats = dataSink_->stats();
  return WriteStats(
      sinkStats.numWrittenBytes,
      sinkStats.numWrittenFiles,
      sinkStats.writeIOTimeUs * 1000,
      currentTimeNs - createTimeNs_);
#endif
}

std::shared_ptr<const iceberg::IcebergPartitionSpec>
parseIcebergPartitionSpec(const uint8_t* data, const int32_t length, RowTypePtr rowType) {
  gluten::IcebergPartitionSpec protoSpec;
  gluten::parseProtobuf(data, length, &protoSpec);
  std::vector<iceberg::IcebergPartitionSpec::Field> fields;
  fields.reserve(protoSpec.fields_size());

  for (const auto& protoField : protoSpec.fields()) {
    // Convert protobuf enum to C++ enum
    iceberg::TransformType transform;
    switch (protoField.transform()) {
      case gluten::IDENTITY:
        transform = iceberg::TransformType::kIdentity;
        break;
      case gluten::YEAR:
        transform = iceberg::TransformType::kYear;
        break;
      case gluten::MONTH:
        transform = iceberg::TransformType::kMonth;
        break;
      case gluten::DAY:
        transform = iceberg::TransformType::kDay;
        break;
      case gluten::HOUR:
        transform = iceberg::TransformType::kHour;
        break;
      case gluten::BUCKET:
        transform = iceberg::TransformType::kBucket;
        break;
      case gluten::TRUNCATE:
        transform = iceberg::TransformType::kTruncate;
        break;
      default:
        throw std::runtime_error("Unknown transform type");
    }

    // Handle optional parameter
    std::optional<int32_t> parameter;
    if (protoField.has_parameter()) {
      parameter = protoField.parameter();
    }

    fields.emplace_back(protoField.name(), rowType->findChild(protoField.name()), transform, parameter);
  }

  return std::make_shared<iceberg::IcebergPartitionSpec>(protoSpec.spec_id(), fields);
}

} // namespace gluten
