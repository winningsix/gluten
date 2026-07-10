/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0.
 */

#include "compute/iceberg/CudfIcebergWriter.h"

#include <atomic>
#include <cctype>
#include <string_view>
#include <unordered_map>

#include <folly/dynamic.h>
#include <folly/json.h>

#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>

#include "utils/ConfigExtractor.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#ifdef ENABLE_S3
#include "velox/connectors/hive/storage_adapters/s3fs/S3WriteFile.h"
#endif

using facebook::velox::RowTypePtr;
using facebook::velox::TypeKind;
using facebook::velox::TypePtr;
using facebook::velox::common::CompressionKind;
using facebook::velox::connector::hive::iceberg::IcebergPartitionSpecPtr;
using facebook::velox::connector::hive::iceberg::TransformType;
using facebook::velox::cudf_velox::CudfVector;
using facebook::velox::parquet::ParquetFieldId;

namespace gluten {
namespace {

std::atomic<uint64_t> nextFileId{0};

cudf::io::compression_type toCudfCompression(CompressionKind compression) {
  switch (compression) {
    case CompressionKind::CompressionKind_NONE:
      return cudf::io::compression_type::NONE;
    case CompressionKind::CompressionKind_SNAPPY:
      return cudf::io::compression_type::SNAPPY;
    case CompressionKind::CompressionKind_ZSTD:
      return cudf::io::compression_type::ZSTD;
    case CompressionKind::CompressionKind_LZ4:
      return cudf::io::compression_type::LZ4;
    case CompressionKind::CompressionKind_GZIP:
      return cudf::io::compression_type::GZIP;
    case CompressionKind::CompressionKind_ZLIB:
      return cudf::io::compression_type::ZLIB;
    default:
      VELOX_USER_FAIL(
          "Unsupported libcudf Parquet compression codec: {}",
          facebook::velox::common::compressionKindToString(compression));
  }
}

std::string safeFileComponent(std::string value) {
  for (auto& ch : value) {
    if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' &&
        ch != '_') {
      ch = '_';
    }
  }
  return value;
}

std::string appendPath(std::string directory, const std::string& fileName) {
  if (!directory.empty() && directory.back() != '/') {
    directory.push_back('/');
  }
  directory.append(fileName);
  return directory;
}

class VeloxFileDataSink final : public cudf::io::data_sink {
 public:
  VeloxFileDataSink(
      std::shared_ptr<facebook::velox::filesystems::FileSystem> fileSystem,
      std::string path,
      facebook::velox::memory::MemoryPool* pool)
      : fileSystem_(std::move(fileSystem)), path_(std::move(path)) {
    VELOX_USER_CHECK_NOT_NULL(fileSystem_);
    facebook::velox::filesystems::FileOptions options;
    options.pool = pool;
    options.shouldCreateParentDirectories = true;
    file_ = fileSystem_->openFileForWrite(path_, options);
  }

  void host_write(const void* data, size_t size) override {
    VELOX_USER_CHECK(
        !closed_, "Cannot write closed Iceberg output file {}", path_);
    file_->append(
        std::string_view(static_cast<const char*>(data), size));
  }

  void flush() override {
    VELOX_USER_CHECK(
        !closed_, "Cannot flush closed Iceberg output file {}", path_);
    file_->flush();
  }

  size_t bytes_written() override {
    return file_->size();
  }

  void close() {
    if (!closed_) {
      file_->close();
      closed_ = true;
    }
  }

  void abort() {
    if (closed_) {
      return;
    }
#ifdef ENABLE_S3
    if (auto* s3File = dynamic_cast<facebook::velox::filesystems::S3WriteFile*>(
            file_.get())) {
      s3File->abort();
      closed_ = true;
      return;
    }
#endif
    close();
  }

  void remove() {
    fileSystem_->remove(path_);
  }

 private:
  std::shared_ptr<facebook::velox::filesystems::FileSystem> fileSystem_;
  std::string path_;
  std::unique_ptr<facebook::velox::WriteFile> file_;
  bool closed_{false};
};

void setColumnMetadata(
    cudf::io::column_in_metadata& metadata,
    const std::string& name,
    const TypePtr& type,
    const ParquetFieldId& field) {
  metadata.set_name(name).set_parquet_field_id(field.fieldId);

  switch (type->kind()) {
    case TypeKind::ROW: {
      VELOX_USER_CHECK_EQ(
          metadata.num_children(),
          type->size(),
          "libcudf struct metadata does not match the Iceberg schema");
      VELOX_USER_CHECK_EQ(
          field.children.size(),
          type->size(),
          "Iceberg struct field IDs do not match the table schema");
      const auto& rowType = type->asRow();
      for (auto i = 0; i < type->size(); ++i) {
        setColumnMetadata(
            metadata.child(i),
            rowType.nameOf(i),
            type->childAt(i),
            field.children[i]);
      }
      return;
    }
    case TypeKind::ARRAY:
      VELOX_USER_CHECK_EQ(
          field.children.size(), 1, "Iceberg list must have one element field ID");
      VELOX_USER_CHECK_GE(
          metadata.num_children(), 2, "libcudf list metadata is missing its element child");
      setColumnMetadata(
          metadata.child(1), "element", type->childAt(0), field.children[0]);
      return;
    case TypeKind::MAP:
      VELOX_USER_CHECK_EQ(
          field.children.size(), 2, "Iceberg map must have key and value field IDs");
      VELOX_USER_CHECK_GE(
          metadata.num_children(), 2, "libcudf map metadata is missing its entry child");
      metadata.set_list_column_as_map();
      VELOX_USER_CHECK_EQ(
          metadata.child(1).num_children(),
          2,
          "libcudf map entry metadata must contain key and value children");
      setColumnMetadata(
          metadata.child(1).child(0), "key", type->childAt(0), field.children[0]);
      setColumnMetadata(
          metadata.child(1).child(1), "value", type->childAt(1), field.children[1]);
      return;
    default:
      VELOX_USER_CHECK(
          field.children.empty(),
          "Primitive Iceberg field {} unexpectedly has nested field IDs",
          name);
  }
}

folly::dynamic partitionValue(
    const cudf::column_view& column,
    const TypePtr& type,
    cudf::size_type index,
    rmm::cuda_stream_view stream) {
  auto scalar = cudf::get_element(
      column,
      index,
      stream,
      facebook::velox::cudf_velox::get_output_mr());
  if (!scalar->is_valid(stream)) {
    return nullptr;
  }

  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return static_cast<bool>(
          static_cast<cudf::numeric_scalar<bool>*>(scalar.get())
              ->value(stream));
    case TypeKind::TINYINT:
      return static_cast<int64_t>(
          static_cast<cudf::numeric_scalar<int8_t>*>(scalar.get())->value(stream));
    case TypeKind::SMALLINT:
      return static_cast<int64_t>(
          static_cast<cudf::numeric_scalar<int16_t>*>(scalar.get())->value(stream));
    case TypeKind::INTEGER:
      if (type->isDate()) {
        const auto value =
            static_cast<cudf::timestamp_scalar<cudf::timestamp_D>*>(scalar.get())
                ->value(stream);
        return static_cast<int64_t>(value.time_since_epoch().count());
      }
      return static_cast<int64_t>(
          static_cast<cudf::numeric_scalar<int32_t>*>(scalar.get())->value(stream));
    case TypeKind::BIGINT:
      VELOX_USER_CHECK(!type->isDecimal(), "Decimal identity partitions are not supported yet");
      return static_cast<int64_t>(
          static_cast<cudf::numeric_scalar<int64_t>*>(scalar.get())->value(stream));
    case TypeKind::TIMESTAMP:
      switch (column.type().id()) {
        case cudf::type_id::TIMESTAMP_SECONDS:
          return static_cast<int64_t>(
                     static_cast<cudf::timestamp_scalar<cudf::timestamp_s>*>(scalar.get())
                         ->value(stream)
                         .time_since_epoch()
                         .count()) *
              1'000'000;
        case cudf::type_id::TIMESTAMP_MILLISECONDS:
          return static_cast<int64_t>(
                     static_cast<cudf::timestamp_scalar<cudf::timestamp_ms>*>(scalar.get())
                         ->value(stream)
                         .time_since_epoch()
                         .count()) *
              1'000;
        case cudf::type_id::TIMESTAMP_MICROSECONDS:
          return static_cast<int64_t>(
              static_cast<cudf::timestamp_scalar<cudf::timestamp_us>*>(scalar.get())
                  ->value(stream)
                  .time_since_epoch()
                  .count());
        case cudf::type_id::TIMESTAMP_NANOSECONDS:
          return static_cast<int64_t>(
                     static_cast<cudf::timestamp_scalar<cudf::timestamp_ns>*>(scalar.get())
                         ->value(stream)
                         .time_since_epoch()
                         .count()) /
              1'000;
        default:
          VELOX_USER_FAIL("Unexpected libcudf timestamp type for Iceberg partition");
      }
    case TypeKind::VARCHAR:
      return static_cast<cudf::string_scalar*>(scalar.get())->to_string(stream);
    default:
      VELOX_USER_FAIL(
          "Unsupported identity partition type for libcudf Iceberg writer: {}",
          type->toString());
  }
}

} // namespace

struct CudfIcebergWriter::Impl {
  struct OpenFile {
    std::string path;
    folly::dynamic partitionValues;
    std::unique_ptr<VeloxFileDataSink> sink;
    std::unique_ptr<cudf::io::chunked_parquet_writer> writer;
    uint64_t rows{0};
    uint64_t bytes{0};
    bool closed{false};
  };

  Impl(
      RowTypePtr rowType,
      std::string outputDirectory,
      CompressionKind compressionKind,
      int32_t partitionId,
      int64_t taskId,
      std::string operationId,
      IcebergPartitionSpecPtr spec,
      ParquetFieldId field,
      const std::unordered_map<std::string, std::string>& sparkConfs,
      std::shared_ptr<facebook::velox::memory::MemoryPool> sinkPool)
      : rowType(std::move(rowType)),
        outputDirectory(std::move(outputDirectory)),
        compressionKind(compressionKind),
        partitionId(partitionId),
        taskId(taskId),
        operationId(safeFileComponent(std::move(operationId))),
        spec(std::move(spec)),
        field(std::move(field)),
        sinkPool(std::move(sinkPool)),
        stream(facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream()) {
    VELOX_USER_CHECK_NOT_NULL(this->spec);
    VELOX_USER_CHECK_EQ(
        this->field.children.size(),
        this->rowType->size(),
        "Iceberg field IDs do not match the write schema");
    VELOX_USER_CHECK_NOT_NULL(this->sinkPool);
    auto veloxConfig = std::make_shared<facebook::velox::config::ConfigBase>(
        std::unordered_map<std::string, std::string>(sparkConfs));
    fileSystemConfig = createHiveConnectorConfig(veloxConfig);
    fileSystem = facebook::velox::filesystems::getFileSystem(
        this->outputDirectory, fileSystemConfig);

    partitionChannels.reserve(this->spec->fields.size());
    partitionTypes.reserve(this->spec->fields.size());
    for (const auto& partitionField : this->spec->fields) {
      VELOX_USER_CHECK(
          partitionField.transformType == TransformType::kIdentity,
          "libcudf Iceberg writer currently supports identity partition transforms only");
      const auto channel = this->rowType->getChildIdx(partitionField.name);
      VELOX_USER_CHECK_GE(
          channel, 0, "Iceberg partition column {} is missing", partitionField.name);
      partitionChannels.push_back(channel);
      partitionTypes.push_back(partitionField.type);
    }
  }

  OpenFile& openFile(
      const std::string& key,
      const folly::dynamic& partitionValues,
      const cudf::table_view& table) {
    auto existing = files.find(key);
    if (existing != files.end()) {
      return *existing->second;
    }

    const auto fileName = fmt::format(
        "{}-p{}-t{}-{}.parquet",
        operationId,
        partitionId,
        taskId,
        nextFileId.fetch_add(1, std::memory_order_relaxed));
    auto file = std::make_unique<OpenFile>();
    file->path = appendPath(outputDirectory, fileName);
    file->partitionValues = partitionValues;

    cudf::io::table_input_metadata metadata(table);
    VELOX_USER_CHECK_EQ(
        metadata.column_metadata.size(),
        rowType->size(),
        "libcudf table column count does not match the Iceberg schema");
    for (auto i = 0; i < rowType->size(); ++i) {
      setColumnMetadata(
          metadata.column_metadata[i],
          rowType->nameOf(i),
          rowType->childAt(i),
          field.children[i]);
    }

    file->sink = std::make_unique<VeloxFileDataSink>(
        fileSystem, file->path, sinkPool.get());
    auto options = cudf::io::chunked_parquet_writer_options::builder(
                       cudf::io::sink_info(file->sink.get()))
                       .metadata(metadata)
                       .compression(toCudfCompression(compressionKind))
                       .utc_timestamps(true)
                       .build();
    file->writer =
        std::make_unique<cudf::io::chunked_parquet_writer>(options, stream);
    auto [inserted, unused] = files.emplace(key, std::move(file));
    return *inserted->second;
  }

  void write(const facebook::velox::RowVectorPtr& input) {
    VELOX_USER_CHECK(!closed, "Cannot write after the libcudf Iceberg writer is closed");
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_USER_CHECK_NOT_NULL(
        cudfInput,
        "Iceberg GPU writer requires a CudfVector; the MPP root materialized data on CPU");
    const auto table = cudfInput->getTableView();
    VELOX_USER_CHECK_EQ(
        table.num_columns(), rowType->size(), "Iceberg GPU write schema column count mismatch");
    VELOX_USER_CHECK_EQ(
        table.num_rows(), input->size(), "Iceberg GPU write row count mismatch");
    for (auto i = 0; i < rowType->size(); ++i) {
      VELOX_USER_CHECK(
          table.column(i).type() ==
              facebook::velox::cudf_velox::veloxToCudfDataType(rowType->childAt(i)),
          "Iceberg GPU write type mismatch for column {}",
          rowType->nameOf(i));
    }

    // MPP operators may use a different stream. The synchronization keeps the
    // input device buffers alive and ready before the chunked writer consumes them.
    cudfInput->stream().synchronize();

    if (partitionChannels.empty()) {
      auto& file = openFile("", folly::dynamic::array, table);
      file.writer->write(table);
      file.rows += table.num_rows();
      stream.synchronize();
      return;
    }

    std::vector<cudf::column_view> keyColumns;
    keyColumns.reserve(partitionChannels.size());
    for (const auto channel : partitionChannels) {
      keyColumns.push_back(table.column(channel));
    }
    cudf::groupby::groupby grouper(
        cudf::table_view(keyColumns), cudf::null_policy::INCLUDE);
    auto groups = grouper.get_groups(
        table, stream, facebook::velox::cudf_velox::get_output_mr());

    for (size_t group = 0; group + 1 < groups.offsets.size(); ++group) {
      const auto begin = groups.offsets[group];
      const auto end = groups.offsets[group + 1];
      folly::dynamic values = folly::dynamic::array;
      for (auto keyIndex = 0; keyIndex < partitionTypes.size(); ++keyIndex) {
        values.push_back(partitionValue(
            groups.keys->view().column(keyIndex),
            partitionTypes[keyIndex],
            begin,
            stream));
      }
      const auto key = folly::toJson(values);
      const auto slices = cudf::slice(groups.values->view(), {begin, end}, stream);
      VELOX_CHECK(slices.size() == 1);
      auto& file = openFile(key, values, slices.front());
      file.writer->write(slices.front());
      file.rows += end - begin;
    }
    stream.synchronize();
  }

  void closeFiles() {
    if (closed) {
      return;
    }
    for (auto& [key, file] : files) {
      file->writer->close();
      file->writer.reset();
      stream.synchronize();
      file->sink->close();
      file->bytes = file->sink->bytes_written();
      file->closed = true;
      totalBytes += file->bytes;
    }
    closed = true;
  }

  std::vector<std::string> commit() {
    closeFiles();
    std::vector<std::string> messages;
    messages.reserve(files.size());
    for (const auto& [key, file] : files) {
      folly::dynamic metrics = folly::dynamic::object;
      metrics["recordCount"] = file->rows;
      metrics["columnSizes"] = folly::dynamic::object;
      metrics["valueCounts"] = folly::dynamic::object;
      metrics["nullValueCounts"] = folly::dynamic::object;
      metrics["nanValueCounts"] = folly::dynamic::object;
      metrics["lowerBounds"] = folly::dynamic::object;
      metrics["upperBounds"] = folly::dynamic::object;

      folly::dynamic message = folly::dynamic::object;
      message["path"] = file->path;
      message["fileSizeInBytes"] = file->bytes;
      message["metrics"] = std::move(metrics);
      message["partitionSpecJson"] = spec->specId;
      message["content"] = "DATA";
      message["splitOffsets"] = folly::dynamic::array;
      if (!partitionChannels.empty()) {
        message["partitionDataJson"] = folly::toJson(
            folly::dynamic::object("partitionValues", file->partitionValues));
      }
      messages.push_back(folly::toJson(message));
    }
    LOG(INFO) << "Iceberg task writer backend=libcudf files=" << files.size()
              << " bytes=" << totalBytes;
    committed = true;
    return messages;
  }

  void abort() {
    if (aborted) {
      return;
    }
    for (auto& [key, file] : files) {
      if (file->writer) {
        try {
          file->writer->close();
          stream.synchronize();
        } catch (const std::exception& error) {
          LOG(WARNING) << "Failed to close aborted libcudf Iceberg file "
                       << file->path << ": " << error.what();
        }
        file->writer.reset();
      }
      try {
        file->sink->abort();
      } catch (const std::exception& error) {
        LOG(WARNING) << "Failed to abort Iceberg output file " << file->path
                     << ": " << error.what();
      }
      try {
        file->sink->remove();
      } catch (const std::exception& error) {
        LOG(WARNING) << "Failed to delete aborted libcudf Iceberg file "
                     << file->path << ": " << error.what();
      }
    }
    aborted = true;
    closed = true;
  }

  RowTypePtr rowType;
  std::string outputDirectory;
  CompressionKind compressionKind;
  int32_t partitionId;
  int64_t taskId;
  std::string operationId;
  IcebergPartitionSpecPtr spec;
  ParquetFieldId field;
  std::shared_ptr<facebook::velox::memory::MemoryPool> sinkPool;
  std::shared_ptr<facebook::velox::config::ConfigBase> fileSystemConfig;
  std::shared_ptr<facebook::velox::filesystems::FileSystem> fileSystem;
  std::vector<facebook::velox::column_index_t> partitionChannels;
  std::vector<TypePtr> partitionTypes;
  rmm::cuda_stream_view stream;
  std::unordered_map<std::string, std::unique_ptr<OpenFile>> files;
  uint64_t totalBytes{0};
  bool closed{false};
  bool committed{false};
  bool aborted{false};
};

CudfIcebergWriter::CudfIcebergWriter(
    RowTypePtr rowType,
    std::string outputDirectory,
    CompressionKind compressionKind,
    int32_t partitionId,
    int64_t taskId,
    std::string operationId,
    IcebergPartitionSpecPtr spec,
    ParquetFieldId field,
    const std::unordered_map<std::string, std::string>& sparkConfs,
    std::shared_ptr<facebook::velox::memory::MemoryPool> sinkPool)
    : impl_(std::make_unique<Impl>(
          std::move(rowType),
          std::move(outputDirectory),
          compressionKind,
          partitionId,
          taskId,
          std::move(operationId),
          std::move(spec),
          std::move(field),
          sparkConfs,
          std::move(sinkPool))) {}

CudfIcebergWriter::~CudfIcebergWriter() {
  if (!impl_->committed && !impl_->aborted) {
    impl_->abort();
  }
}

void CudfIcebergWriter::write(const facebook::velox::RowVectorPtr& input) {
  impl_->write(input);
}

std::vector<std::string> CudfIcebergWriter::commit() {
  return impl_->commit();
}

void CudfIcebergWriter::abort() {
  impl_->abort();
}

uint64_t CudfIcebergWriter::numWrittenBytes() const {
  return impl_->totalBytes;
}

uint32_t CudfIcebergWriter::numWrittenFiles() const {
  return impl_->files.size();
}

} // namespace gluten
