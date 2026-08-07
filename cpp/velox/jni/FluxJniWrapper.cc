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

#include <jni.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <glog/logging.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <nvtx3/nvtx3.hpp>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>

#include <jni/JniCommon.h>
#include <jni/JniError.h>

#include "compute/FluxQueryCoordinator.h"
#include "compute/ProtobufUtils.h"
#include "compute/VeloxBackend.h"
#include "compute/VeloxPlanConverter.h"
#include "compute/VeloxRuntime.h"
#include "config/VeloxConfig.h"
#include "memory/VeloxColumnarBatch.h"
#include "memory/VeloxMemoryManager.h"
#include "substrait/plan.pb.h"
#include "utils/ConfigExtractor.h"
#include "utils/ObjectStore.h"

// Plan node types for tree rewriting.
#include "velox/core/PlanNode.h"
#include "velox/exec/ExchangeSource.h"
#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/RoundRobinPartitionFunction.h"
#include "operators/plannodes/RowVectorStream.h"
#ifdef GLUTEN_ENABLE_GPU
#include "operators/plannodes/CudfVectorStream.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/RangePartitionFunction.h"
#endif

using namespace gluten;
using namespace facebook;

// NVTX domain for gluten FLUX. Keeps our ranges visually distinct from
// the velox / cudf NVTX ranges in nsys timeline.
namespace {
struct GlutenFluxDomain {
  static constexpr char const* name{"gluten-flux"};
};
} // namespace

// ---------------------------------------------------------------------------
// Helper: container for an FluxQueryCoordinator plus the resources it needs
// that must outlive the coordinator (thread pool, memory pool, QueryCtx).
// This is the object stored via ObjectStore and referenced by the jlong handle.
// ---------------------------------------------------------------------------
namespace {

/// Bundles the FluxQueryCoordinator with its owned resources so everything
/// has a clear lifetime: the JNI handle → FluxQueryHandle → coordinator +
/// resources.  Destroying the handle tears everything down in order.
struct FluxQueryHandle {
  /// Thread pool for Velox task execution. Owned here so it outlives the
  /// coordinator and all tasks.
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor;

  /// Query context (memory pool, config, cache).
  std::shared_ptr<velox::core::QueryCtx> queryCtx;

  /// Optional spill executor. QueryCtx stores a raw pointer to this.
  std::shared_ptr<folly::CPUThreadPoolExecutor> spillExecutor;

  /// The coordinator itself.
  std::shared_ptr<FluxQueryCoordinator> coordinator;

  /// Memory pool kept alive for output deserialization.
  std::shared_ptr<velox::memory::MemoryPool> memoryPool;

  ~FluxQueryHandle() {
    nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{
        "jni::~FluxQueryHandle"};
    // Ensure coordinator is destroyed first (aborts any running tasks),
    // then queryCtx, then spillExecutor, then executor.
    {
      nvtx3::scoped_range_in<GlutenFluxDomain> r{
          "jni::~FluxQueryHandle:coordinator.reset"};
      coordinator.reset();
    }
    {
      nvtx3::scoped_range_in<GlutenFluxDomain> r{
          "jni::~FluxQueryHandle:queryCtx.reset"};
      queryCtx.reset();
    }
    {
      nvtx3::scoped_range_in<GlutenFluxDomain> r{
          "jni::~FluxQueryHandle:spillExecutor.reset"};
      spillExecutor.reset();
    }
    {
      nvtx3::scoped_range_in<GlutenFluxDomain> r{
          "jni::~FluxQueryHandle:executor.reset"};
      executor.reset();
    }
  }
};

bool tryGetIteratorIndex(const ::substrait::ReadRel& readRel, int32_t* index) {
  if (!readRel.has_local_files() || readRel.local_files().items_size() == 0) {
    return false;
  }

  const std::string& uri = readRel.local_files().items(0).uri_file();
  constexpr const char* kIteratorPrefix = "iterator:";
  const auto pos = uri.find(kIteratorPrefix);
  if (pos == std::string::npos) {
    return false;
  }

  const auto indexString = uri.substr(pos + std::strlen(kIteratorPrefix));
  try {
    *index = std::stoi(indexString);
  } catch (const std::exception& e) {
    VELOX_FAIL(
        "Invalid FLUX iterator URI '{}' in Substrait ReadRel: {}",
        uri,
        e.what());
  }
  return true;
}

void collectIteratorIndices(
    const ::google::protobuf::Message& message,
    std::vector<int32_t>& indices) {
  if (message.GetDescriptor() == ::substrait::ReadRel::descriptor()) {
    const auto& readRel = static_cast<const ::substrait::ReadRel&>(message);
    int32_t index = -1;
    if (tryGetIteratorIndex(readRel, &index)) {
      indices.push_back(index);
    }
  }

  const auto* reflection = message.GetReflection();
  std::vector<const ::google::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);
  for (const auto* field : fields) {
    if (field->cpp_type() !=
        ::google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) {
      continue;
    }
    if (field->is_repeated()) {
      const int fieldSize = reflection->FieldSize(message, field);
      for (int i = 0; i < fieldSize; ++i) {
        collectIteratorIndices(
            reflection->GetRepeatedMessage(message, field, i), indices);
      }
    } else {
      collectIteratorIndices(reflection->GetMessage(message, field), indices);
    }
  }
}

std::string formatIndices(std::vector<int32_t> indices) {
  std::sort(indices.begin(), indices.end());
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << indices[i];
  }
  out << "]";
  return out.str();
}

std::shared_ptr<velox::config::ConfigBase> createFluxSessionConfig(
    VeloxRuntime* runtime) {
  auto backendConf = VeloxBackend::get()->getBackendConf();
  auto mergedMap = backendConf->rawConfigsCopy();
  for (const auto& [key, val] : runtime->getConfMap()) {
    mergedMap[key] = val;
  }
  return std::make_shared<velox::config::ConfigBase>(std::move(mergedMap));
}

std::unordered_map<std::string, std::string> buildFluxQueryConfig(
    const std::shared_ptr<velox::config::ConfigBase>& veloxCfg,
    uint64_t replicatedCartesianMaxBuildBytes) {
  std::unordered_map<std::string, std::string> configs;

  configs[velox::core::QueryConfig::kPreferredOutputBatchRows] =
      std::to_string(veloxCfg->get<uint32_t>(kSparkBatchSize, 4096));
  configs[velox::core::QueryConfig::kMaxOutputBatchRows] =
      std::to_string(veloxCfg->get<uint32_t>(kSparkBatchSize, 4096));
  configs[velox::core::QueryConfig::kPreferredOutputBatchBytes] =
      std::to_string(veloxCfg->get<uint64_t>(kVeloxPreferredBatchBytes, 10L << 20));

#ifdef GLUTEN_ENABLE_GPU
  // The FLUX coordinator constructs scan splits after planning. Preserve the
  // scan batching knobs in QueryConfig so it can build multi-file Iceberg
  // splits using the same byte target as the regular execution path.
  configs[kCudfGpuTargetBatchBytes] = std::to_string(veloxCfg->get<uint64_t>(
      kCudfGpuTargetBatchBytes,
      std::stoull(kCudfGpuTargetBatchBytesDefault)));
  configs[kCudfIcebergMultiFileTargetBytes] = std::to_string(
      veloxCfg->get<uint64_t>(
          kCudfIcebergMultiFileTargetBytes,
          kCudfIcebergMultiFileTargetBytesDefault));
  configs[kCudfIcebergMultiFileMaxFiles] = std::to_string(
      veloxCfg->get<int32_t>(
          kCudfIcebergMultiFileMaxFiles,
          kCudfIcebergMultiFileMaxFilesDefault));
  configs[kCudfIcebergMultiFileMaxFileBytes] = std::to_string(
      veloxCfg->get<uint64_t>(
          kCudfIcebergMultiFileMaxFileBytes,
          kCudfIcebergMultiFileMaxFileBytesDefault));
  configs[kCudfHiveUseExperimentalReader] = std::to_string(
      veloxCfg->get<bool>(kCudfHiveUseExperimentalReader, false));

  // These groupby controls may be supplied by a query-scoped Spark SQLConf.
  // Flux builds its own QueryCtx, so values merged into the runtime session
  // config must be copied explicitly into QueryConfig for cuDF operators to
  // observe the per-query override instead of the executor-global default.
  configs[velox::cudf_velox::CudfConfig::kCudfGroupbyStreamingMaxDistinctKeys] =
      veloxCfg->get<std::string>(
          kCudfGroupbyStreamingMaxDistinctKeys,
          kCudfGroupbyStreamingMaxDistinctKeysDefault);
  configs[velox::cudf_velox::CudfConfig::kCudfPartialIdentityAggregation] =
      veloxCfg->get<std::string>(
          kCudfPartialIdentityAggregation,
          kCudfPartialIdentityAggregationDefault);
  LOG(INFO) << "MppJniWrapper: query-scoped cuDF groupby config "
            << velox::cudf_velox::CudfConfig::kCudfGroupbyStreamingMaxDistinctKeys
            << "="
            << configs[velox::cudf_velox::CudfConfig::kCudfGroupbyStreamingMaxDistinctKeys]
            << " "
            << velox::cudf_velox::CudfConfig::kCudfPartialIdentityAggregation
            << "="
            << configs[velox::cudf_velox::CudfConfig::kCudfPartialIdentityAggregation];

  // Keep the UCX exchange byte bound independently configurable. Partial
  // identity can emit one state per input row, so its default is bounded at
  // 64 MiB. Regular queries retain the historical GPU compute-batch default;
  // applying the smaller window globally regresses exchange-heavy queries.
  const auto partialIdentityAggregation = veloxCfg->get<bool>(
      kCudfPartialIdentityAggregation,
      false);
  const auto partitionedOutputBatchBytesDefault = partialIdentityAggregation
      ? kCudfPartitionedOutputBatchBytesPartialIdentityDefault
      : veloxCfg->get<uint64_t>(
            kCudfGpuTargetBatchBytes,
            std::stoull(kCudfGpuTargetBatchBytesDefault));
  configs[velox::core::QueryConfig::kUcxPartitionedOutputBatchBytes] =
      std::to_string(veloxCfg->get<uint64_t>(
          kCudfPartitionedOutputBatchBytes,
          partitionedOutputBatchBytesDefault));
#endif

  try {
    configs[velox::core::QueryConfig::kSparkAnsiEnabled] =
        veloxCfg->get<std::string>(kAnsiEnabled, "false");
    configs[velox::core::QueryConfig::kSessionTimezone] =
        veloxCfg->get<std::string>(kSessionTimezone, "");
    configs[velox::core::QueryConfig::kAdjustTimestampToTimezone] = "true";

    auto offHeapMemory =
        veloxCfg->get<int64_t>(kSparkTaskOffHeapMemory, facebook::velox::memory::kMaxMemory);
    auto maxPartialAggregationMemory = std::max<int64_t>(
        1 << 24,
        veloxCfg->get<int64_t>(kMaxPartialAggregationMemory).has_value()
            ? veloxCfg->get<int64_t>(kMaxPartialAggregationMemory).value()
            : static_cast<int64_t>(
                  veloxCfg->get<double>(kMaxPartialAggregationMemoryRatio, 0.1) * offHeapMemory));
    auto maxExtendedPartialAggregationMemory = std::max<int64_t>(
        1 << 26,
        veloxCfg->get<int64_t>(kMaxExtendedPartialAggregationMemory).has_value()
            ? veloxCfg->get<int64_t>(kMaxExtendedPartialAggregationMemory).value()
            : static_cast<int64_t>(
                  veloxCfg->get<double>(kMaxExtendedPartialAggregationMemoryRatio, 0.15) * offHeapMemory));
    configs[velox::core::QueryConfig::kMaxPartialAggregationMemory] =
        std::to_string(maxPartialAggregationMemory);
    configs[velox::core::QueryConfig::kMaxExtendedPartialAggregationMemory] =
        std::to_string(maxExtendedPartialAggregationMemory);
    configs[velox::core::QueryConfig::kAbandonPartialAggregationMinPct] =
        std::to_string(veloxCfg->get<int32_t>(kAbandonPartialAggregationMinPct, 90));
    configs[velox::core::QueryConfig::kAbandonPartialAggregationMinRows] =
        std::to_string(veloxCfg->get<int32_t>(kAbandonPartialAggregationMinRows, 100000));

    const auto spillStrategy =
        veloxCfg->get<std::string>(kSpillStrategy, kSpillStrategyDefaultValue);
    configs[velox::core::QueryConfig::kSpillEnabled] =
        spillStrategy == "none" ? "false" : "true";
    configs[velox::core::QueryConfig::kAggregationSpillEnabled] =
        std::to_string(veloxCfg->get<bool>(kAggregationSpillEnabled, true));
    configs[velox::core::QueryConfig::kJoinSpillEnabled] =
        std::to_string(veloxCfg->get<bool>(kJoinSpillEnabled, true));
    configs[velox::core::QueryConfig::kOrderBySpillEnabled] =
        std::to_string(veloxCfg->get<bool>(kOrderBySpillEnabled, true));
    configs[velox::core::QueryConfig::kWindowSpillEnabled] =
        std::to_string(veloxCfg->get<bool>(kWindowSpillEnabled, true));
    configs[velox::core::QueryConfig::kMaxSpillLevel] =
        std::to_string(veloxCfg->get<int32_t>(kMaxSpillLevel, 4));
    configs[velox::core::QueryConfig::kMaxSpillFileSize] =
        std::to_string(veloxCfg->get<uint64_t>(kMaxSpillFileSize, 1L * 1024 * 1024 * 1024));
    configs[velox::core::QueryConfig::kMaxSpillRunRows] =
        std::to_string(veloxCfg->get<uint64_t>(kMaxSpillRunRows, 3L * 1024 * 1024));
    configs[velox::core::QueryConfig::kMaxSpillBytes] =
        std::to_string(veloxCfg->get<uint64_t>(kMaxSpillBytes, 107374182400LL));
    configs[velox::core::QueryConfig::kSpillWriteBufferSize] =
        std::to_string(veloxCfg->get<uint64_t>(kShuffleSpillDiskWriteBufferSize, 1L * 1024 * 1024));
    configs[velox::core::QueryConfig::kSpillReadBufferSize] =
        std::to_string(veloxCfg->get<int32_t>(kSpillReadBufferSize, 1L * 1024 * 1024));
    configs[velox::core::QueryConfig::kSpillStartPartitionBit] =
        std::to_string(veloxCfg->get<uint8_t>(kSpillStartPartitionBit, 48));
    configs[velox::core::QueryConfig::kSpillNumPartitionBits] =
        std::to_string(veloxCfg->get<uint8_t>(kSpillPartitionBits, 3));
    configs[velox::core::QueryConfig::kSpillableReservationGrowthPct] =
        std::to_string(veloxCfg->get<uint8_t>(kSpillableReservationGrowthPct, 25));
    configs[velox::core::QueryConfig::kSpillPrefixSortEnabled] =
        veloxCfg->get<std::string>(kSpillPrefixSortEnabled, "false");
    if (veloxCfg->get<bool>(kSparkShuffleSpillCompress, true)) {
      configs[velox::core::QueryConfig::kSpillCompressionKind] =
          veloxCfg->get<std::string>(
              kSpillCompressionKind,
              veloxCfg->get<std::string>(kCompressionKind, "lz4"));
    } else {
      configs[velox::core::QueryConfig::kSpillCompressionKind] = "none";
    }

    configs[velox::core::QueryConfig::kSparkBloomFilterExpectedNumItems] =
        std::to_string(veloxCfg->get<int64_t>(kBloomFilterExpectedNumItems, 1000000));
    configs[velox::core::QueryConfig::kSparkBloomFilterNumBits] =
        std::to_string(veloxCfg->get<int64_t>(kBloomFilterNumBits, 8388608));
    configs[velox::core::QueryConfig::kSparkBloomFilterMaxNumBits] =
        std::to_string(veloxCfg->get<int64_t>(kBloomFilterMaxNumBits, 4194304));
    configs[velox::core::QueryConfig::kHashProbeDynamicFilterPushdownEnabled] =
        std::to_string(veloxCfg->get<bool>(kHashProbeDynamicFilterPushdownEnabled, true));
    configs[velox::core::QueryConfig::kHashProbeBloomFilterPushdownMaxSize] =
        std::to_string(veloxCfg->get<uint64_t>(kHashProbeBloomFilterPushdownMaxSize, 0));
    configs[velox::core::QueryConfig::kMaxSplitPreloadPerDriver] =
        std::to_string(veloxCfg->get<int32_t>(kVeloxSplitPreloadPerDriver, 2));
    configs[velox::core::QueryConfig::kMaxSplitPreloadPerTask] =
        std::to_string(veloxCfg->get<int32_t>(
            kVeloxSplitPreloadPerTask, kVeloxSplitPreloadPerTaskDefault));

    configs[velox::core::QueryConfig::kAbandonDedupHashMapMinRows] =
        std::to_string(veloxCfg->get<int32_t>(kAbandonDedupHashMapMinRows, 100000));
    configs[velox::core::QueryConfig::kAbandonDedupHashMapMinPct] =
        std::to_string(veloxCfg->get<int32_t>(kAbandonDedupHashMapMinPct, 0));
    configs[velox::core::QueryConfig::kDriverCpuTimeSliceLimitMs] = "0";

    configs[velox::core::QueryConfig::kSparkLegacyDateFormatter] =
        veloxCfg->get<std::string>(kSparkLegacyTimeParserPolicy, "") == "LEGACY" ? "true" : "false";
    configs[velox::core::QueryConfig::kThrowExceptionOnDuplicateMapKeys] =
        veloxCfg->get<std::string>(kSparkMapKeyDedupPolicy, "") == "EXCEPTION" ? "true" : "false";
    configs[velox::core::QueryConfig::kSparkLegacyStatisticalAggregate] =
        std::to_string(veloxCfg->get<bool>(kSparkLegacyStatisticalAggregate, false));
    configs[velox::core::QueryConfig::kSparkJsonIgnoreNullFields] =
        std::to_string(veloxCfg->get<bool>(kSparkJsonIgnoreNullFields, true));
    configs[velox::core::QueryConfig::kExprMaxCompiledRegexes] =
        std::to_string(veloxCfg->get<int32_t>(kExprMaxCompiledRegexes, 100));

#ifdef GLUTEN_ENABLE_GPU
    configs[velox::cudf_velox::CudfConfig::kCudfEnabled] =
        std::to_string(veloxCfg->get<bool>(kCudfEnabled, false));
    configs[velox::cudf_velox::CudfConfig::kCudfSkipOutputToVelox] =
        std::to_string(veloxCfg->get<bool>(
            kCudfSkipOutputToVelox,
            kCudfSkipOutputToVeloxDefault));
    if (replicatedCartesianMaxBuildBytes > 0) {
      configs[velox::cudf_velox::CudfConfig::kCudfNestedLoopJoinMaxBuildBytes] =
          std::to_string(replicatedCartesianMaxBuildBytes);
    }
#endif

    const auto setIfExists = [&](const std::string& glutenKey, const std::string& veloxKey) {
      const auto valueOptional = veloxCfg->get<std::string>(glutenKey);
      if (valueOptional.has_value()) {
        configs[veloxKey] = valueOptional.value();
      }
    };
    setIfExists(kQueryTraceEnabled, velox::core::QueryConfig::kQueryTraceEnabled);
    setIfExists(kQueryTraceDir, velox::core::QueryConfig::kQueryTraceDir);
    setIfExists(kQueryTraceMaxBytes, velox::core::QueryConfig::kQueryTraceMaxBytes);
    setIfExists(kQueryTraceTaskRegExp, velox::core::QueryConfig::kQueryTraceTaskRegExp);
    setIfExists(kOpTraceDirectoryCreateConfig, velox::core::QueryConfig::kOpTraceDirectoryCreateConfig);

    overwriteVeloxConf(veloxCfg.get(), configs, kDynamicBackendConfPrefix);
  } catch (const std::invalid_argument& err) {
    std::string errDetails = err.what();
    throw std::runtime_error("Invalid FLUX query conf arg: " + errDetails);
  }

  // Apply FLUX exchange backpressure per fragment.  This must stay after the
  // dynamic config copy so both Velox output-buffer limits use one explicit
  // value.  The previous fixed 1 GiB per fragment allowed a 72-fragment query
  // to retain far more than a single GPU's memory.
  const auto fluxMaxOutputBufferSize = veloxCfg->get<uint64_t>(
      kFluxMaxOutputBufferSize, kFluxMaxOutputBufferSizeDefault);
  configs[velox::core::QueryConfig::kMaxOutputBufferSize] =
      std::to_string(fluxMaxOutputBufferSize);
  configs[velox::core::QueryConfig::kMaxPartitionedOutputBufferSize] =
      std::to_string(fluxMaxOutputBufferSize);
  return configs;
}

// ---------------------------------------------------------------------------
// Plan tree rewriting helpers
// ---------------------------------------------------------------------------

/// Find a TableScanNode by plan node ID and return its connector ID.
/// Returns empty string if not found.
std::string getTableScanConnectorId(
    const velox::core::PlanNodePtr& plan,
    const velox::core::PlanNodeId& nodeId) {
  if (auto tableScan =
          std::dynamic_pointer_cast<const velox::core::TableScanNode>(plan)) {
    if (tableScan->id() == nodeId && tableScan->tableHandle()) {
      return tableScan->tableHandle()->connectorId();
    }
  }
  for (const auto& source : plan->sources()) {
    auto result = getTableScanConnectorId(source, nodeId);
    if (!result.empty()) {
      return result;
    }
  }
  return "";
}

/// Check if a PlanNode is a ValueStream leaf node that should be replaced
/// with an ExchangeNode for FLUX execution.
///
/// ValueStream nodes appear in two forms:
///   1. CPU: TableScanNode with "value-stream" connector ID
///   2. GPU: CudfValueStreamNode (custom Gluten node)
bool isValueStreamNode(const velox::core::PlanNodePtr& node) {
  // Check for CudfValueStreamNode (GPU path).
#ifdef GLUTEN_ENABLE_GPU
  if (std::dynamic_pointer_cast<const CudfValueStreamNode>(node) != nullptr) {
    return true;
  }
#endif
  // Check for CPU ValueStream: TableScanNode with "value-stream" connector.
  if (auto tableScan =
          std::dynamic_pointer_cast<const velox::core::TableScanNode>(node)) {
    if (tableScan->tableHandle() &&
        tableScan->tableHandle()->connectorId() == kIteratorConnectorId) {
      return true;
    }
  }
  return false;
}

/// Collect all ValueStream leaf nodes from a plan tree in depth-first order.
/// The order matches the stream index assignment in SubstraitToVeloxPlanConverter.
void collectValueStreamNodes(
    const velox::core::PlanNodePtr& node,
    std::vector<velox::core::PlanNodePtr>& result) {
  if (isValueStreamNode(node)) {
    result.push_back(node);
    return;
  }
  for (const auto& source : node->sources()) {
    collectValueStreamNodes(source, result);
  }
}

/// Returns true when `targetNodeId` reaches a keyed FINAL aggregation through
/// only schema-preserving/filtering unary nodes. Restricting the match to this
/// shape is deliberate: a HASH exchange feeding a join somewhere below a
/// FINAL aggregation is not sufficient proof that the exchange keys equal the
/// aggregation keys. The TPC-H Q17/Q21 problem inputs are direct (optionally
/// Project/Filter wrapped) FINAL inputs.
bool unaryPathToValueStream(
    const velox::core::PlanNodePtr& node,
    const std::string& targetNodeId) {
  if (isValueStreamNode(node)) {
    return node->id() == targetNodeId;
  }
  if (node->sources().size() != 1) {
    return false;
  }
  const bool allowedUnary =
      std::dynamic_pointer_cast<const velox::core::ProjectNode>(node) != nullptr ||
      std::dynamic_pointer_cast<const velox::core::FilterNode>(node) != nullptr;
  return allowedUnary &&
      unaryPathToValueStream(node->sources().front(), targetNodeId);
}

bool feedsKeyedFinalAggregation(
    const velox::core::PlanNodePtr& node,
    const std::string& targetNodeId) {
  if (auto aggregation =
          std::dynamic_pointer_cast<const velox::core::AggregationNode>(node)) {
    if ((aggregation->step() ==
             velox::core::AggregationNode::Step::kFinal ||
         aggregation->step() ==
             velox::core::AggregationNode::Step::kSingle) &&
        !aggregation->groupingKeys().empty() &&
        unaryPathToValueStream(aggregation->sources().front(), targetNodeId)) {
      return true;
    }
  }
  for (const auto& source : node->sources()) {
    if (feedsKeyedFinalAggregation(source, targetNodeId)) {
      return true;
    }
  }
  return false;
}

/// Recognizes the narrow candidate-first existence fragment shape that is
/// safe to run with multiple drivers inside one HASH-consumer task. This is
/// deliberately an operator/capability predicate, not a query or table-name
/// special case. Non-partial aggregation and every order-sensitive or
/// singleton operator are rejected by the whitelist.
struct RightSemiProjectMultiDriverShape {
  int32_t rightSemiProjectCount{0};
  int32_t partialAggregationCount{0};
};

bool isRightSemiProjectMultiDriverSafePlan(
    const velox::core::PlanNodePtr& node,
    RightSemiProjectMultiDriverShape& shape) {
  if (auto join =
          std::dynamic_pointer_cast<const velox::core::HashJoinNode>(node)) {
    if (join->joinType() == velox::core::JoinType::kRightSemiProject &&
        !join->isNullAware()) {
      ++shape.rightSemiProjectCount;
    } else if (join->joinType() != velox::core::JoinType::kInner) {
      return false;
    }
  } else if (auto aggregation =
                 std::dynamic_pointer_cast<const velox::core::AggregationNode>(
                     node)) {
    if (aggregation->step() !=
            velox::core::AggregationNode::Step::kPartial ||
        aggregation->groupingKeys().empty()) {
      return false;
    }
    ++shape.partialAggregationCount;
  } else if (
      std::dynamic_pointer_cast<const velox::core::ProjectNode>(node) ==
          nullptr &&
      std::dynamic_pointer_cast<const velox::core::FilterNode>(node) ==
          nullptr &&
      std::dynamic_pointer_cast<const velox::core::ExchangeNode>(node) ==
          nullptr &&
      std::dynamic_pointer_cast<const velox::core::TableScanNode>(node) ==
          nullptr) {
    return false;
  }

  return std::all_of(
      node->sources().begin(),
      node->sources().end(),
      [&](const auto& source) {
        return isRightSemiProjectMultiDriverSafePlan(source, shape);
      });
}

velox::core::PlanNodePtr stripProjectAndFilter(
    velox::core::PlanNodePtr node) {
  while (
      node->sources().size() == 1 &&
      (std::dynamic_pointer_cast<const velox::core::ProjectNode>(node) !=
           nullptr ||
       std::dynamic_pointer_cast<const velox::core::FilterNode>(node) !=
           nullptr)) {
    node = node->sources().front();
  }
  return node;
}

bool hasNestedRightSemiProjectPartialShape(
    const velox::core::PlanNodePtr& root) {
  auto node = stripProjectAndFilter(root);
  auto aggregation =
      std::dynamic_pointer_cast<const velox::core::AggregationNode>(node);
  if (aggregation == nullptr ||
      aggregation->step() != velox::core::AggregationNode::Step::kPartial ||
      aggregation->groupingKeys().empty()) {
    return false;
  }

  node = stripProjectAndFilter(aggregation->sources().front());
  auto outer = std::dynamic_pointer_cast<const velox::core::HashJoinNode>(node);
  if (outer == nullptr ||
      outer->joinType() != velox::core::JoinType::kRightSemiProject ||
      outer->isNullAware()) {
    return false;
  }

  // RIGHT SEMI PROJECT preserves its right/build rows. Require the second
  // existence join on that exact preserved spine; two unrelated sibling
  // joins must not accidentally enable this capability.
  node = stripProjectAndFilter(outer->sources().at(1));
  auto inner = std::dynamic_pointer_cast<const velox::core::HashJoinNode>(node);
  return inner != nullptr &&
      inner->joinType() == velox::core::JoinType::kRightSemiProject &&
      !inner->isNullAware();
}

bool isRightSemiProjectMultiDriverSafePlan(
    const velox::core::PlanNodePtr& node) {
  RightSemiProjectMultiDriverShape shape;
  return isRightSemiProjectMultiDriverSafePlan(node, shape) &&
      shape.rightSemiProjectCount == 2 &&
      shape.partialAggregationCount == 1 &&
      hasNestedRightSemiProjectPartialShape(node);
}

bool isInnerJoinMultiDriverSafePlan(
    const velox::core::PlanNodePtr& node,
    int32_t& innerJoinCount) {
  if (auto join =
          std::dynamic_pointer_cast<const velox::core::HashJoinNode>(node)) {
    if (join->joinType() != velox::core::JoinType::kInner ||
        join->isNullAware()) {
      return false;
    }
    ++innerJoinCount;
  } else if (
      std::dynamic_pointer_cast<const velox::core::ProjectNode>(node) ==
          nullptr &&
      std::dynamic_pointer_cast<const velox::core::FilterNode>(node) ==
          nullptr &&
      std::dynamic_pointer_cast<const velox::core::ExchangeNode>(node) ==
          nullptr) {
    return false;
  }
  return std::all_of(
      node->sources().begin(),
      node->sources().end(),
      [&](const auto& source) {
        return isInnerJoinMultiDriverSafePlan(source, innerJoinCount);
      });
}

bool isInnerJoinMultiDriverSafePlan(
    const velox::core::PlanNodePtr& node) {
  int32_t innerJoinCount = 0;
  return isInnerJoinMultiDriverSafePlan(node, innerJoinCount) &&
      innerJoinCount > 0;
}

bool hasValidHashKeys(
    const std::vector<int32_t>& keyIndices,
    const velox::RowTypePtr& wireType) {
  if (wireType == nullptr || keyIndices.empty()) {
    return false;
  }
  return std::all_of(
      keyIndices.begin(), keyIndices.end(), [&](const auto index) {
        return index >= 0 && index < static_cast<int32_t>(wireType->size());
      });
}

/// Build a PartitionFunctionSpec from an exchange's partitionType + key
/// indices. Reused by both the multi-task UCX path (PartitionedOutputNode)
/// and the single-task merge path (LocalPartitionNode kRepartition).
///
/// Returns {nullptr, {}} when no key columns are usable (caller should
/// fall back to RoundRobin or treat as no-op).
struct PartitionSpecAndExprs {
  velox::core::PartitionFunctionSpecPtr funcSpec;
  std::vector<velox::core::TypedExprPtr> partitionExprs;
};

PartitionSpecAndExprs buildPartitionFunctionSpec(
    const std::string& partitionType,
    const std::vector<int32_t>& keyIndices,
    const velox::RowTypePtr& outputType,
    int32_t fragmentIdForLogging,
    const std::string& rangeBoundsJson = {}) {
  PartitionSpecAndExprs result;
  const auto numFields = static_cast<int32_t>(outputType->size());

  if (partitionType == "RANGE") {
    VELOX_CHECK(
        !keyIndices.empty(),
        "FLUX RANGE fragment {} has no resolved sort-key indices; refusing "
        "hash/round-robin degradation",
        fragmentIdForLogging);
    VELOX_CHECK(
        !rangeBoundsJson.empty(),
        "FLUX RANGE fragment {} has no Spark boundary descriptor; refusing "
        "hash/round-robin degradation",
        fragmentIdForLogging);
#ifndef GLUTEN_ENABLE_GPU
    VELOX_FAIL("FLUX RANGE_PID requires the cuDF UCX backend");
#endif
  }

  if ((partitionType == "HASH" || partitionType == "RANGE") &&
      !keyIndices.empty()) {
    std::vector<velox::column_index_t> keyChannels;
    keyChannels.reserve(keyIndices.size());
    for (auto idx : keyIndices) {
      if (idx < 0 || idx >= numFields) {
        if (partitionType == "RANGE") {
          VELOX_FAIL(
              "FLUX RANGE fragment {} key index {} is outside {} output "
              "fields; refusing hash/round-robin degradation",
              fragmentIdForLogging,
              idx,
              numFields);
        }
        LOG(WARNING) << "FluxJniWrapper: fragment " << fragmentIdForLogging
                     << " partition key index " << idx
                     << " out of range (output has " << numFields
                     << " fields); falling back to round-robin";
        keyChannels.clear();
        result.partitionExprs.clear();
        break;
      }
      keyChannels.push_back(static_cast<velox::column_index_t>(idx));
      result.partitionExprs.push_back(
          std::make_shared<velox::core::FieldAccessTypedExpr>(
              outputType->childAt(idx), outputType->nameOf(idx)));
    }
    if (!keyChannels.empty()) {
      if (partitionType == "RANGE") {
#ifdef GLUTEN_ENABLE_GPU
        result.funcSpec =
            std::make_shared<velox::ucx_exchange::RangePartitionFunctionSpec>(
                outputType, std::move(keyChannels), rangeBoundsJson);
#endif
      } else {
        result.funcSpec =
            std::make_shared<velox::exec::HashPartitionFunctionSpec>(
                outputType, std::move(keyChannels));
      }
    }
  }

  if (result.funcSpec == nullptr) {
    VELOX_CHECK_NE(
        partitionType,
        "RANGE",
        "FLUX RANGE partition spec construction failed; refusing fallback");
    result.partitionExprs.clear();
    result.funcSpec =
        std::make_shared<velox::exec::RoundRobinPartitionFunctionSpec>();
  }

  return result;
}

/// Recursively walk a Velox plan tree, replacing a SPECIFIC ValueStream leaf
/// node (identified by plan node ID) with an ExchangeNode. Velox PlanNodes
/// are immutable, so when a child changes we must reconstruct the parent.
///
/// The function handles common single-source node types via their Builder
/// pattern. For multi-source nodes (joins, unions) it rebuilds using
/// their Builders. Unrecognized node types with changed children
/// cause a VELOX_FAIL — add explicit support as needed.
velox::core::PlanNodePtr replaceValueStreamWithExchange(
    const velox::core::PlanNodePtr& node,
    const std::string& targetNodeId,
    const std::string& exchangeNodeId,
    const velox::RowTypePtr& producerWireType,
    const velox::core::PlanNodePtr& producerPlanForMerge = nullptr,
    const std::string& mergePartitionType = "SINGLE",
    const std::vector<int32_t>& mergeKeyIndices = {},
    const std::string& mergeRangeBoundsJson = {},
    bool repartitionRemoteHashLocally = false) {
  // Base case: this IS the target ValueStream leaf - replace it.
  if (isValueStreamNode(node) && node->id() == targetNodeId) {
    const auto& consumerType = node->outputType();
    const auto& wireType =
        producerWireType != nullptr ? producerWireType : consumerType;
    velox::core::PlanNodePtr exchange;
    if (producerPlanForMerge != nullptr) {
      // Single-task merge mode: replace ValueStream with a LocalPartitionNode
      // that splices the producer fragment's plan tree directly into this
      // pipeline. Bypasses UcxExchange entirely.
      //
      // Per partition type:
      //   SINGLE       -> LocalPartitionNode::Type::kGather (N-to-1)
      //   HASH         -> kRepartition + HashPartitionFunctionSpec
      //   RANGE        -> kRepartition + RangePartitionFunctionSpec
      //   ROUND_ROBIN  -> kRepartition + RoundRobinPartitionFunctionSpec
      //   BROADCAST    -> caller short-circuits and inlines producer plan
      //                   directly (no LocalPartitionNode); HashJoinBridge
      //                   handles cross-pipeline access. So we shouldn't
      //                   actually reach here for BROADCAST.
      const std::string localId = exchangeNodeId + "_local";
      if (mergePartitionType == "SINGLE") {
        LOG(WARNING) << "FluxJniWrapper: replacing ValueStream node '"
                     << node->id() << "' -> LocalPartition::gather "
                     << "(single-task merge SINGLE) consumerType="
                     << consumerType->toString();
        exchange = velox::core::LocalPartitionNode::gather(
            localId, {producerPlanForMerge});
      } else if (
          mergePartitionType == "BROADCAST") {
        // BROADCAST in single-task mode: just inline the producer plan tree
        // as-is. Velox HashJoinBridge will cross-link build and probe sides
        // of the consumer's HashJoinNode without needing a LocalPartition.
        LOG(WARNING) << "FluxJniWrapper: replacing ValueStream node '"
                     << node->id()
                     << "' -> producer plan (single-task merge BROADCAST, "
                        "no LocalPartition wrap)";
        exchange = producerPlanForMerge;
      } else {
        // HASH / RANGE / ROUND_ROBIN -> kRepartition + appropriate spec.
        auto specPair = buildPartitionFunctionSpec(
            mergePartitionType, mergeKeyIndices, wireType,
            /*fragmentIdForLogging=*/-1, mergeRangeBoundsJson);
        LOG(WARNING) << "FluxJniWrapper: replacing ValueStream node '"
                     << node->id() << "' -> LocalPartition::kRepartition "
                     << "(single-task merge " << mergePartitionType
                     << ") spec="
                     << (specPair.funcSpec ? specPair.funcSpec->toString()
                                           : "null");
        exchange = velox::core::LocalPartitionNode::Builder()
                       .id(localId)
                       .type(velox::core::LocalPartitionNode::Type::kRepartition)
                       .scaleWriter(false)
                       .partitionFunctionSpec(specPair.funcSpec)
                       .sources({producerPlanForMerge})
                       .build();
      }
    } else {
      LOG(WARNING) << "FluxJniWrapper: replacing ValueStream node '"
                   << node->id() << "' -> Exchange '" << exchangeNodeId
                   << "' wireType=" << wireType->toString()
                   << " consumerType=" << consumerType->toString();
      // ExchangeNode advertises the WIRE schema (producer's outputType, may
      // include a synthetic hash_partition_key:int prefix or other Spark-
      // injected partitioning columns). cuDF serdes the wire as-is; if we
      // declared the consumer's narrower type here, cuDF would silently
      // truncate columns and we'd see "Cannot change vector type" /
      // null-row corruption downstream (Q17 v9s, Q18 hang).
      exchange = velox::core::ExchangeNode::Builder()
                     .id(exchangeNodeId)
                     .outputType(wireType)
                     .serdeKind("Presto")
                     .build();
      if (repartitionRemoteHashLocally) {
        VELOX_CHECK_EQ(
            mergePartitionType,
            "HASH",
            "Local repartition is only valid for remote HASH exchanges");
        VELOX_CHECK(
            hasValidHashKeys(mergeKeyIndices, wireType),
            "Remote HASH exchange {} has no valid local repartition keys",
            exchangeNodeId);
        auto specPair = buildPartitionFunctionSpec(
            mergePartitionType,
            mergeKeyIndices,
            wireType,
            /*fragmentIdForLogging=*/-1);
        LOG(WARNING) << "FluxJniWrapper: wrapping remote HASH exchange '"
                     << exchangeNodeId
                     << "' with LocalPartition::kRepartition for keyed FINAL "
                     << "spec=" << specPair.funcSpec->toString();
        exchange = velox::core::LocalPartitionNode::Builder()
                       .id(exchangeNodeId + "_keyed_final_local_hash")
                       .type(velox::core::LocalPartitionNode::Type::kRepartition)
                       .scaleWriter(false)
                       .partitionFunctionSpec(specPair.funcSpec)
                       .sources({exchange})
                       .build();
      }
    }

    // Fast path: wire schema structurally equals consumer's expected. No
    // reshape needed - return ExchangeNode directly so Q6's single-driver
    // IntraNodeTransferRegistry shortcut isn't broken by an extra wrapper.
    bool sameStructure = wireType->size() == consumerType->size();
    if (sameStructure) {
      for (size_t i = 0; i < wireType->size(); ++i) {
        if (!wireType->childAt(i)->equivalent(*consumerType->childAt(i))) {
          sameStructure = false;
          break;
        }
      }
    }
    if (sameStructure) {
      // Names may still differ (wire uses positional "0","1",...; consumer
      // expects nN_M). If they do, project-rename so downstream
      // FieldAccessTypedExpr name lookups resolve.
      bool namesEqual = true;
      for (size_t i = 0; i < wireType->size(); ++i) {
        if (wireType->nameOf(i) != consumerType->nameOf(i)) {
          namesEqual = false;
          break;
        }
      }
      if (namesEqual) {
        return exchange;
      }
      if (producerPlanForMerge == nullptr &&
          !repartitionRemoteHashLocally) {
        // UCX exchange payloads are positional.  When width and child types
        // already match, expose the consumer names directly on ExchangeNode
        // instead of inserting an identity Project solely to rename fields.
        // Velox deliberately leaves such a no-op Project on CPU, which strict
        // cuDF mode would otherwise (incorrectly) report as fallback.
        LOG(WARNING) << "FluxJniWrapper: applying wire->consumer names directly "
                     << "on Exchange " << exchangeNodeId;
        return velox::core::ExchangeNode::Builder()
            .id(exchangeNodeId)
            .outputType(consumerType)
            .serdeKind("Presto")
            .build();
      }
      // A keyed FINAL with multiple local drivers wraps the remote Exchange
      // in a LocalPartitionNode above. Rebuilding a bare Exchange here would
      // silently discard that wrapper, allowing the same grouping key to be
      // finalized independently by multiple drivers. Preserve the wrapper
      // and use the Project rename path below instead.
      // Names differ - inject identity Project that just renames cols.
      std::vector<velox::core::TypedExprPtr> projections;
      std::vector<std::string> projectionNames;
      projections.reserve(consumerType->size());
      projectionNames.reserve(consumerType->size());
      for (size_t i = 0; i < consumerType->size(); ++i) {
        projections.push_back(
            std::make_shared<velox::core::FieldAccessTypedExpr>(
                wireType->childAt(i), wireType->nameOf(i)));
        projectionNames.push_back(consumerType->nameOf(i));
      }
      LOG(WARNING) << "FluxJniWrapper: renaming wire->consumer cols at "
                   << exchangeNodeId;
      return std::make_shared<velox::core::ProjectNode>(
          exchangeNodeId + "_rename",
          std::move(projectionNames),
          std::move(projections),
          std::move(exchange));
    }

    // Wire wider than consumer - the Spark-side FluxCollapseRule injected
    // synthetic prefix column(s) for HASH partitioning that the consumer
    // wasn't told about. Inject a ProjectNode to drop the leading prefix
    // and rename the remaining cols to the consumer's expected names so
    // downstream FieldAccessTypedExpr name lookups resolve.
    if (wireType->size() < consumerType->size()) {
      LOG(WARNING) << "FluxJniWrapper: wire " << wireType->toString()
                   << " NARROWER than consumer " << consumerType->toString()
                   << " for " << exchangeNodeId
                   << " -- emitting bare Exchange (downstream may fail)";
      return exchange;
    }
    const auto skip = wireType->size() - consumerType->size();
    std::vector<velox::core::TypedExprPtr> projections;
    std::vector<std::string> projectionNames;
    projections.reserve(consumerType->size());
    projectionNames.reserve(consumerType->size());
    for (size_t i = 0; i < consumerType->size(); ++i) {
      const auto wireIdx = i + skip;
      projections.push_back(
          std::make_shared<velox::core::FieldAccessTypedExpr>(
              wireType->childAt(wireIdx), wireType->nameOf(wireIdx)));
      projectionNames.push_back(consumerType->nameOf(i));
    }
    LOG(WARNING) << "FluxJniWrapper: stripping " << skip
                 << " prefix col(s) at " << exchangeNodeId;
    return std::make_shared<velox::core::ProjectNode>(
        exchangeNodeId + "_strip",
        std::move(projectionNames),
        std::move(projections),
        std::move(exchange));
  }

  // If this is a leaf node (no children) that is NOT ValueStream, keep it.
  const auto& sources = node->sources();
  if (sources.empty()) {
    return node;
  }

  // Recurse into children. Forward mergePartitionType and mergeKeyIndices
  // so HASH/RANGE/ROUND_ROBIN partition information survives the descent
  // to the target ValueStream leaf. Without the explicit forwarding the
  // 6th/7th parameters fall back to their defaults ("SINGLE", {}) and
  // every recursive call below the outermost reverts the partition type
  // to SINGLE, causing all single-task merges to take the kGather branch
  // even when the original exchange was HASH-partitioned.
  std::vector<velox::core::PlanNodePtr> newSources;
  newSources.reserve(sources.size());
  bool anyChanged = false;
  for (const auto& source : sources) {
    auto newSource = replaceValueStreamWithExchange(
        source,
        targetNodeId,
        exchangeNodeId,
        producerWireType,
        producerPlanForMerge,
        mergePartitionType,
        mergeKeyIndices,
        mergeRangeBoundsJson,
        repartitionRemoteHashLocally);
    if (newSource.get() != source.get()) {
      anyChanged = true;
    }
    newSources.push_back(std::move(newSource));
  }

  // No children changed — return the original node unchanged.
  if (!anyChanged) {
    return node;
  }

  // Children changed — we must reconstruct this node with the new children.
  // Handle common single-source node types using their Builder pattern.
  // The Builder(existingNode) constructor copies all fields, then we
  // override the source.

  // LocalPartitionNode (N-to-1 gather or N-to-M local repartition). Native
  // union is translated to a gather LocalPartition, and single-task FLUX merge
  // inserts repartition LocalPartitions, so ValueStream replacement must be
  // able to preserve these nodes while rewriting descendants.
  if (auto localPartitionNode =
          std::dynamic_pointer_cast<const velox::core::LocalPartitionNode>(
              node)) {
    return velox::core::LocalPartitionNode::Builder(*localPartitionNode)
        .sources(std::move(newSources))
        .build();
  }

  // FilterNode
  if (auto filterNode =
          std::dynamic_pointer_cast<const velox::core::FilterNode>(node)) {
    return velox::core::FilterNode::Builder(*filterNode)
        .source(newSources[0])
        .build();
  }

  // ProjectNode
  if (auto projectNode =
          std::dynamic_pointer_cast<const velox::core::ProjectNode>(node)) {
    return velox::core::ProjectNode::Builder(*projectNode)
        .source(newSources[0])
        .build();
  }

  // AggregationNode
  if (auto aggNode =
          std::dynamic_pointer_cast<const velox::core::AggregationNode>(
              node)) {
    return velox::core::AggregationNode::Builder(*aggNode)
        .source(newSources[0])
        .build();
  }

  // OrderByNode
  if (auto orderByNode =
          std::dynamic_pointer_cast<const velox::core::OrderByNode>(node)) {
    return velox::core::OrderByNode::Builder(*orderByNode)
        .source(newSources[0])
        .build();
  }

  // TopNNode
  if (auto topNNode =
          std::dynamic_pointer_cast<const velox::core::TopNNode>(node)) {
    return velox::core::TopNNode::Builder(*topNNode)
        .source(newSources[0])
        .build();
  }

  // LimitNode
  if (auto limitNode =
          std::dynamic_pointer_cast<const velox::core::LimitNode>(node)) {
    return velox::core::LimitNode::Builder(*limitNode)
        .source(newSources[0])
        .build();
  }

  // HashJoinNode (2 sources: left, right)
  if (auto hashJoinNode =
          std::dynamic_pointer_cast<const velox::core::HashJoinNode>(node)) {
    return velox::core::HashJoinNode::Builder(*hashJoinNode)
        .left(newSources[0])
        .right(newSources[1])
        .build();
  }

  // MergeJoinNode (2 sources: left, right)
  if (auto mergeJoinNode =
          std::dynamic_pointer_cast<const velox::core::MergeJoinNode>(node)) {
    return velox::core::MergeJoinNode::Builder(*mergeJoinNode)
        .left(newSources[0])
        .right(newSources[1])
        .build();
  }

  // NestedLoopJoinNode (2 sources: left, right)
  if (auto nlJoinNode =
          std::dynamic_pointer_cast<const velox::core::NestedLoopJoinNode>(
              node)) {
    return velox::core::NestedLoopJoinNode::Builder(*nlJoinNode)
        .left(newSources[0])
        .right(newSources[1])
        .build();
  }

  // ExpandNode
  if (auto expandNode =
          std::dynamic_pointer_cast<const velox::core::ExpandNode>(node)) {
    return velox::core::ExpandNode::Builder(*expandNode)
        .source(newSources[0])
        .build();
  }

  // RowNumberNode
  if (auto rowNumberNode =
          std::dynamic_pointer_cast<const velox::core::RowNumberNode>(node)) {
    return velox::core::RowNumberNode::Builder(*rowNumberNode)
        .source(newSources[0])
        .build();
  }

  // TopNRowNumberNode
  if (auto topNRowNumberNode =
          std::dynamic_pointer_cast<const velox::core::TopNRowNumberNode>(
              node)) {
    return velox::core::TopNRowNumberNode::Builder(*topNRowNumberNode)
        .source(newSources[0])
        .build();
  }

  // WindowNode
  if (auto windowNode =
          std::dynamic_pointer_cast<const velox::core::WindowNode>(node)) {
    return velox::core::WindowNode::Builder(*windowNode)
        .source(newSources[0])
        .build();
  }

  // MarkDistinctNode
  if (auto markDistinctNode =
          std::dynamic_pointer_cast<const velox::core::MarkDistinctNode>(
              node)) {
    return velox::core::MarkDistinctNode::Builder(*markDistinctNode)
        .source(newSources[0])
        .build();
  }

  // EnforceSingleRowNode
  if (auto enforceSingleRowNode =
          std::dynamic_pointer_cast<const velox::core::EnforceSingleRowNode>(
              node)) {
    return velox::core::EnforceSingleRowNode::Builder(*enforceSingleRowNode)
        .source(newSources[0])
        .build();
  }

  // GroupIdNode
  if (auto groupIdNode =
          std::dynamic_pointer_cast<const velox::core::GroupIdNode>(node)) {
    return velox::core::GroupIdNode::Builder(*groupIdNode)
        .source(newSources[0])
        .build();
  }

  // UnnestNode
  if (auto unnestNode =
          std::dynamic_pointer_cast<const velox::core::UnnestNode>(node)) {
    return velox::core::UnnestNode::Builder(*unnestNode)
        .source(newSources[0])
        .build();
  }

  // TableWriteNode (write-in-FLUX): the final fragment's parquet write. Rebuild with the
  // rewritten source so the write executes inside the pinned FLUX native task -- each peer
  // writes its own slice. Unary node: one source = the data to write.
  if (auto tableWriteNode =
          std::dynamic_pointer_cast<const velox::core::TableWriteNode>(node)) {
    return velox::core::TableWriteNode::Builder(*tableWriteNode)
        .source(newSources[0])
        .build();
  }

  VELOX_FAIL(
      "FluxJniWrapper: unsupported plan node type '{}' (id='{}') encountered "
      "during ValueStream replacement. Add explicit Builder support for this "
      "node type in replaceValueStreamWithExchange().",
      node->name(),
      node->id());
}

bool containsHashJoin(const velox::core::PlanNodePtr& node) {
  if (std::dynamic_pointer_cast<const velox::core::HashJoinNode>(node) !=
      nullptr) {
    return true;
  }
  return std::any_of(
      node->sources().begin(),
      node->sources().end(),
      [&](const auto& source) { return containsHashJoin(source); });
}

/// Protect a keyed FINAL aggregation whose input contains a hash join with a
/// local hash exchange on the aggregation keys.  This creates a pipeline
/// boundary after the join: multiple upstream probe drivers can run without
/// allowing equal FINAL groups to be owned by different downstream drivers.
velox::core::PlanNodePtr insertKeyedFinalLocalRepartitionAfterJoin(
    const velox::core::PlanNodePtr& node,
    bool& inserted) {
  if (auto aggregation =
          std::dynamic_pointer_cast<const velox::core::AggregationNode>(node)) {
    if ((aggregation->step() ==
             velox::core::AggregationNode::Step::kFinal ||
         aggregation->step() ==
             velox::core::AggregationNode::Step::kSingle) &&
        !aggregation->groupingKeys().empty() &&
        containsHashJoin(aggregation->sources().front())) {
      const auto& source = aggregation->sources().front();
      const auto& sourceType = source->outputType();
      std::vector<velox::column_index_t> keyChannels;
      keyChannels.reserve(aggregation->groupingKeys().size());
      for (const auto& key : aggregation->groupingKeys()) {
        velox::column_index_t channel = -1;
        for (size_t i = 0; i < sourceType->size(); ++i) {
          if (sourceType->nameOf(i) == key->name()) {
            channel = static_cast<velox::column_index_t>(i);
            break;
          }
        }
        VELOX_CHECK_GE(
            channel,
            0,
            "Keyed FINAL grouping key '{}' is absent from source schema {}",
            key->name(),
            sourceType->toString());
        keyChannels.push_back(channel);
      }
      auto partitionSpec =
          std::make_shared<velox::exec::HashPartitionFunctionSpec>(
              sourceType, std::move(keyChannels));
      auto localPartition = velox::core::LocalPartitionNode::Builder()
                                .id(aggregation->id() + "_post_join_local_hash")
                                .type(velox::core::LocalPartitionNode::Type::
                                          kRepartition)
                                .scaleWriter(false)
                                .partitionFunctionSpec(partitionSpec)
                                .sources({source})
                                .build();
      inserted = true;
      LOG(WARNING)
          << "MppJniWrapper: inserting keyed FINAL local HASH repartition "
             "after join before aggregation "
          << aggregation->id();
      return velox::core::AggregationNode::Builder(*aggregation)
          .source(std::move(localPartition))
          .build();
    }
  }

  if (node->sources().size() != 1) {
    return node;
  }
  auto newSource =
      insertKeyedFinalLocalRepartitionAfterJoin(node->sources().front(), inserted);
  if (newSource.get() == node->sources().front().get()) {
    return node;
  }
  if (auto project =
          std::dynamic_pointer_cast<const velox::core::ProjectNode>(node)) {
    return velox::core::ProjectNode::Builder(*project)
        .source(std::move(newSource))
        .build();
  }
  if (auto filter =
          std::dynamic_pointer_cast<const velox::core::FilterNode>(node)) {
    return velox::core::FilterNode::Builder(*filter)
        .source(std::move(newSource))
        .build();
  }
  if (auto topN =
          std::dynamic_pointer_cast<const velox::core::TopNNode>(node)) {
    return velox::core::TopNNode::Builder(*topN)
        .source(std::move(newSource))
        .build();
  }
  if (auto orderBy =
          std::dynamic_pointer_cast<const velox::core::OrderByNode>(node)) {
    return velox::core::OrderByNode::Builder(*orderBy)
        .source(std::move(newSource))
        .build();
  }
  if (auto limit =
          std::dynamic_pointer_cast<const velox::core::LimitNode>(node)) {
    return velox::core::LimitNode::Builder(*limit)
        .source(std::move(newSource))
        .build();
  }
  VELOX_FAIL(
      "Unsupported unary node '{}' above keyed FINAL local repartition",
      node->name());
}

bool schemaMatches(const velox::RowTypePtr& a, const velox::RowTypePtr& b) {
  if (a == nullptr || b == nullptr || a->size() != b->size()) {
    return false;
  }
  for (size_t k = 0; k < a->size(); ++k) {
    if (!a->childAt(k)->equivalent(*b->childAt(k))) {
      return false;
    }
  }
  return true;
}

velox::core::PlanNodePtr rewriteValueStreamsForFlux(
    int32_t fragmentId,
    velox::core::PlanNodePtr veloxPlanNode,
    const std::vector<FluxExchangeSpec>& exchangeSpecs,
    const std::unordered_map<int, velox::RowTypePtr>& producerWireTypes,
    const std::unordered_set<int32_t>& broadcastSlotSet,
    int32_t numExchangeInputs,
    int32_t numBroadcastInputs,
    int32_t keyedFinalLocalDrivers) {
  std::vector<const FluxExchangeSpec*> inboundExchanges;
  for (const auto& exchange : exchangeSpecs) {
    if (exchange.consumerFragmentId == fragmentId) {
      inboundExchanges.push_back(&exchange);
    }
  }

  if (inboundExchanges.empty() && broadcastSlotSet.empty()) {
    return veloxPlanNode;
  }

  std::vector<velox::core::PlanNodePtr> valueStreamNodes;
  collectValueStreamNodes(veloxPlanNode, valueStreamNodes);

  VELOX_CHECK_EQ(
      valueStreamNodes.size(),
      static_cast<size_t>(numExchangeInputs + numBroadcastInputs),
      "Fragment {} has {} ValueStream nodes but {} inbound exchanges + "
      "{} fused broadcasts. These must match 1:1 (broadcast slots are "
      "kept as ValueStream, exchange slots are rewritten to ExchangeNode).",
      fragmentId,
      valueStreamNodes.size(),
      inboundExchanges.size(),
      numBroadcastInputs);

  std::vector<size_t> exchangeForStream(
      valueStreamNodes.size(), std::numeric_limits<size_t>::max());
  std::vector<bool> exchangeUsed(inboundExchanges.size(), false);
  for (size_t j = 0; j < valueStreamNodes.size(); ++j) {
    if (broadcastSlotSet.count(static_cast<int32_t>(j)) > 0) {
      continue;
    }
    auto streamType = std::dynamic_pointer_cast<const velox::RowType>(
        valueStreamNodes[j]->outputType());
    for (size_t k = 0; k < inboundExchanges.size(); ++k) {
      if (exchangeUsed[k]) {
        continue;
      }
      velox::RowTypePtr producerWire;
      auto it = producerWireTypes.find(inboundExchanges[k]->producerFragmentId);
      if (it != producerWireTypes.end()) {
        producerWire = it->second;
      }
      bool matched = schemaMatches(streamType, producerWire);
      if (!matched && producerWire != nullptr && streamType != nullptr &&
          producerWire->size() == streamType->size() + 1) {
        std::vector<std::string> n;
        std::vector<velox::TypePtr> t;
        for (size_t kk = 1; kk < producerWire->size(); ++kk) {
          n.push_back(producerWire->nameOf(kk));
          t.push_back(producerWire->childAt(kk));
        }
        auto stripped =
            std::make_shared<const velox::RowType>(std::move(n), std::move(t));
        matched = schemaMatches(streamType, stripped);
      }
      if (matched) {
        exchangeForStream[j] = k;
        exchangeUsed[k] = true;
        break;
      }
    }
  }

  for (size_t j = 0; j < valueStreamNodes.size(); ++j) {
    if (broadcastSlotSet.count(static_cast<int32_t>(j)) > 0 ||
        exchangeForStream[j] != std::numeric_limits<size_t>::max()) {
      continue;
    }
    for (size_t k = 0; k < inboundExchanges.size(); ++k) {
      if (!exchangeUsed[k]) {
        exchangeForStream[j] = k;
        exchangeUsed[k] = true;
        break;
      }
    }
  }

  for (size_t j = 0; j < valueStreamNodes.size(); ++j) {
    if (broadcastSlotSet.count(static_cast<int32_t>(j)) > 0) {
      LOG(WARNING) << "FluxJniWrapper: fragment " << fragmentId << " stream["
                   << j << "] id=" << valueStreamNodes[j]->id()
                   << " type=" << valueStreamNodes[j]->outputType()->toString()
                   << " -> KEEP as ValueStream (fused broadcast slot)";
      continue;
    }
    const auto k = exchangeForStream[j];
    LOG(WARNING) << "FluxJniWrapper: fragment " << fragmentId << " stream["
                 << j << "] id=" << valueStreamNodes[j]->id()
                 << " type=" << valueStreamNodes[j]->outputType()->toString()
                 << " -> exchange[" << k << "] producerF="
                 << inboundExchanges[k]->producerFragmentId
                 << " nodeId=" << inboundExchanges[k]->exchangeNodeId;
  }

  for (size_t j = 0; j < valueStreamNodes.size(); ++j) {
    if (broadcastSlotSet.count(static_cast<int32_t>(j)) > 0) {
      continue;
    }
    const auto k = exchangeForStream[j];
    velox::RowTypePtr producerWire;
    auto it = producerWireTypes.find(inboundExchanges[k]->producerFragmentId);
    if (it != producerWireTypes.end()) {
      producerWire = it->second;
    }
    const bool localRepartition =
        keyedFinalLocalDrivers > 1 &&
        inboundExchanges[k]->partitionType == "HASH" &&
        hasValidHashKeys(
            inboundExchanges[k]->partitionKeyIndices, producerWire) &&
        feedsKeyedFinalAggregation(
            veloxPlanNode, valueStreamNodes[j]->id());
    veloxPlanNode = replaceValueStreamWithExchange(
        veloxPlanNode,
        valueStreamNodes[j]->id(),
        inboundExchanges[k]->exchangeNodeId,
        producerWire,
        /*producerPlanForMerge=*/nullptr,
        inboundExchanges[k]->partitionType,
        inboundExchanges[k]->partitionKeyIndices,
        inboundExchanges[k]->rangeBoundsJson,
        localRepartition);
  }

  LOG(INFO) << "FluxJniWrapper: fragment " << fragmentId
            << " after ValueStream->Exchange replacement: "
            << veloxPlanNode->toString(/*detailed=*/true, /*recursive=*/true);
  return veloxPlanNode;
}

velox::core::PlanNodePtr wrapWithFluxPartitionedOutput(
    int32_t fragmentId,
    velox::core::PlanNodePtr veloxPlanNode,
    const std::vector<FluxExchangeSpec>& exchangeSpecs) {
  int32_t numOutputPartitions = 1;
  const FluxExchangeSpec* outboundExchange = nullptr;
  for (const auto& exchange : exchangeSpecs) {
    if (exchange.producerFragmentId == fragmentId) {
      numOutputPartitions = exchange.numPartitions;
      outboundExchange = &exchange;
      break;
    }
  }

  auto outputNodeId = fmt::format("flux_output_{}", fragmentId);
  const std::string partitionType =
      outboundExchange != nullptr ? outboundExchange->partitionType : std::string("ROOT");
  const bool isBroadcastOutput =
      outboundExchange != nullptr && partitionType == "BROADCAST";
  const char* outputKindHelper =
      isBroadcastOutput ? "broadcast" : (numOutputPartitions == 1 ? "single" : "partitioned");

  LOG(INFO) << "FluxJniWrapper: fragment " << fragmentId
            << " outbound partitionType=" << partitionType
            << " outputKindHelper=" << outputKindHelper
            << " numOutputPartitions=" << numOutputPartitions;

  if (isBroadcastOutput) {
    return velox::core::PartitionedOutputNode::broadcast(
        outputNodeId,
        numOutputPartitions,
        veloxPlanNode->outputType(),
        /*serdeKind=*/"Presto",
        std::string{velox::core::TransportKind::kUcx},
        veloxPlanNode);
  }

  if (numOutputPartitions == 1) {
    const auto transportType = (outboundExchange != nullptr)
        ? std::string{velox::core::TransportKind::kUcx}
        : std::string{velox::core::TransportKind::kInMemory};
    return velox::core::PartitionedOutputNode::single(
        outputNodeId,
        veloxPlanNode->outputType(),
        /*serdeKind=*/"Presto",
        transportType,
        veloxPlanNode);
  }

  const auto& keyIndices =
      outboundExchange != nullptr ? outboundExchange->partitionKeyIndices : std::vector<int32_t>{};
  const auto& outputType = veloxPlanNode->outputType();
  auto specPair = buildPartitionFunctionSpec(
      partitionType,
      keyIndices,
      outputType,
      fragmentId,
      outboundExchange != nullptr ? outboundExchange->rangeBoundsJson
                                  : std::string{});

  LOG(WARNING) << "FluxJniWrapper: fragment " << fragmentId
               << " outbound exchange type=" << partitionType
               << " keyIndices.size=" << keyIndices.size()
               << " func="
               << (specPair.funcSpec ? specPair.funcSpec->toString() : "null")
               << " usingRoundRobinFallback="
               << (partitionType == "HASH" && keyIndices.empty() ? "YES" : "no");

  return std::make_shared<velox::core::PartitionedOutputNode>(
      outputNodeId,
      velox::core::PartitionedOutputNode::Kind::kPartitioned,
      std::move(specPair.partitionExprs),
      numOutputPartitions,
      /*replicateNullsAndAny=*/false,
      std::move(specPair.funcSpec),
      veloxPlanNode->outputType(),
      /*serdeKind=*/"Presto",
      std::string{velox::core::TransportKind::kUcx},
      veloxPlanNode);
}

struct FluxPeerSpec {
  std::string queryId;
  std::string localPeerId{"local"};
  int32_t peerIndex{0};
  int32_t peerCount{1};
  std::vector<FluxPeerEndpoint> producerEndpoints;
};

std::vector<FluxPeerEndpoint> parsePeerEndpointArray(
    const folly::dynamic& endpoints) {
  VELOX_CHECK(endpoints.isArray(), "FLUX peer endpoints must be a JSON array");
  std::vector<FluxPeerEndpoint> parsed;
  parsed.reserve(endpoints.size());
  int32_t fallbackPeerIndex = 0;
  for (const auto& item : endpoints) {
    VELOX_CHECK(item.isObject(), "FLUX peer endpoint must be an object");
    VELOX_CHECK(
        item.count("peerId") > 0,
        "FLUX peer endpoint is missing required peerId");
    VELOX_CHECK(
        item.count("host") > 0,
        "FLUX peer endpoint is missing required host");
    FluxPeerEndpoint endpoint;
    endpoint.peerId = item["peerId"].asString();
    endpoint.host = item["host"].asString();
    endpoint.port = item.count("port") > 0
        ? static_cast<int32_t>(item["port"].asInt())
        : -1;
    endpoint.peerIndex = item.count("peerIndex") > 0
        ? static_cast<int32_t>(item["peerIndex"].asInt())
        : fallbackPeerIndex;
    parsed.push_back(std::move(endpoint));
    ++fallbackPeerIndex;
  }
  return parsed;
}

FluxPeerSpec parseFluxPeerSpec(const uint8_t* data, int32_t size) {
  FluxPeerSpec spec;
  if (data == nullptr || size <= 0) {
    return spec;
  }
  std::string jsonStr(reinterpret_cast<const char*>(data), size);
  auto parsed = folly::parseJson(jsonStr);
  VELOX_CHECK(parsed.isObject(), "fluxPeerSpecJson must be a JSON object");
  if (parsed.count("queryId") > 0) {
    spec.queryId = parsed["queryId"].asString();
  }
  if (parsed.count("localPeerId") > 0) {
    spec.localPeerId = parsed["localPeerId"].asString();
  }
  if (parsed.count("peerIndex") > 0) {
    spec.peerIndex = static_cast<int32_t>(parsed["peerIndex"].asInt());
  }
  if (parsed.count("peerCount") > 0) {
    spec.peerCount = static_cast<int32_t>(parsed["peerCount"].asInt());
  }
  if (parsed.count("peers") > 0) {
    spec.producerEndpoints = parsePeerEndpointArray(parsed["peers"]);
  }
  return spec;
}

/// Parse the exchange specifications from a JSON byte array.
///
/// Expected format:
/// [
///   {
///     "producerFragmentId": 1,
///     "consumerFragmentId": 0,
///     "exchangeNodeId": "n3"
///   },
///   ...
/// ]
std::vector<FluxExchangeSpec> parseExchangeSpecs(
    const uint8_t* data,
    int32_t size) {
  std::string jsonStr(reinterpret_cast<const char*>(data), size);
  auto parsed = folly::parseJson(jsonStr);
  VELOX_CHECK(parsed.isArray(), "exchangeSpecsJson must be a JSON array");

  std::vector<FluxExchangeSpec> specs;
  specs.reserve(parsed.size());
  int32_t exchangeId = 0;
  for (const auto& item : parsed) {
    FluxExchangeSpec spec;
    spec.id = exchangeId++;
    spec.producerFragmentId = item["producerFragmentId"].asInt();
    spec.consumerFragmentId = item["consumerFragmentId"].asInt();
    spec.exchangeNodeId = item["exchangeNodeId"].asString();
    spec.numPartitions =
        item.count("numPartitions") ? item["numPartitions"].asInt() : 1;
    spec.partitionType =
        item.count("exchangeType") ? item["exchangeType"].asString() : "ROUND_ROBIN";
    if (item.count("partitionKeyIndices") &&
        item["partitionKeyIndices"].isArray()) {
      for (const auto& key : item["partitionKeyIndices"]) {
        spec.partitionKeyIndices.push_back(
            static_cast<int32_t>(key.asInt()));
      }
    }
    if (item.count("rangeBoundsJson")) {
      spec.rangeBoundsJson = item["rangeBoundsJson"].asString();
    }
    if (item.count("rangeEffectivePartitions")) {
      spec.rangeEffectivePartitions =
          static_cast<int32_t>(item["rangeEffectivePartitions"].asInt());
    }
    if (spec.partitionType == "RANGE") {
      VELOX_CHECK(
          !spec.rangeBoundsJson.empty(),
          "FLUX RANGE exchange {} is missing Spark boundaries",
          spec.id);
      VELOX_CHECK_GT(
          spec.rangeEffectivePartitions,
          0,
          "FLUX RANGE exchange {} has invalid effective partition count",
          spec.id);
      VELOX_CHECK_LE(
          spec.rangeEffectivePartitions,
          spec.numPartitions,
          "FLUX RANGE exchange {} effective partitions exceed requested",
          spec.id);
    }
    if (item.count("producerEndpoints") &&
        item["producerEndpoints"].isArray()) {
      spec.producerEndpoints = parsePeerEndpointArray(item["producerEndpoints"]);
    }
    specs.push_back(std::move(spec));
  }
  return specs;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// nativeCreateFluxQuery
// ---------------------------------------------------------------------------

JNIEXPORT jlong JNICALL
Java_org_apache_gluten_vectorized_FluxQueryJniWrapper_nativeCreateFluxQuery( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jobjectArray substraitPlansArr,
    jintArray numDriversArr,
    jbyteArray exchangeSpecsJsonArr,
    jbyteArray fluxPeerSpecJsonArr,
    jobjectArray splitInfosPerFragArr,
    jobjectArray broadcastSlotIndicesPerFragArr,
    jobjectArray broadcastIteratorsPerFragArr,
    jlong replicatedCartesianMaxBuildBytes,
    jstring spillRootPathJstr) {
  JNI_METHOD_START
  nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{"jni::nativeCreateFluxQuery"};

  auto ctx = getRuntime(env, wrapper);
  auto runtime = dynamic_cast<VeloxRuntime*>(ctx);
  GLUTEN_CHECK(runtime != nullptr, "FluxQuery requires VeloxRuntime");
  GLUTEN_CHECK(
      replicatedCartesianMaxBuildBytes >= 0,
      "replicated Cartesian max build bytes must be non-negative");
  const auto spillRootPath = jStringToCString(env, spillRootPathJstr);

  // --- Parse inputs ---

  const jsize numFragments = env->GetArrayLength(substraitPlansArr);
  GLUTEN_CHECK(numFragments > 0, "At least one fragment plan is required");

  auto safeNumDrivers = getIntArrayElementsSafe(env, numDriversArr);
  GLUTEN_CHECK(
      env->GetArrayLength(numDriversArr) == numFragments,
      "numDriversPerFragment length must match substraitPlans length");

  // Parse exchange specs JSON.
  auto safeExchangeJson = getByteArrayElementsSafe(env, exchangeSpecsJsonArr);
  auto exchangeSpecs = parseExchangeSpecs(
      reinterpret_cast<const uint8_t*>(safeExchangeJson.elems()),
      env->GetArrayLength(exchangeSpecsJsonArr));
  FluxPeerSpec peerSpec;
  if (fluxPeerSpecJsonArr != nullptr &&
      env->GetArrayLength(fluxPeerSpecJsonArr) > 0) {
    auto safePeerSpecJson = getByteArrayElementsSafe(env, fluxPeerSpecJsonArr);
    peerSpec = parseFluxPeerSpec(
        reinterpret_cast<const uint8_t*>(safePeerSpecJson.elems()),
        env->GetArrayLength(fluxPeerSpecJsonArr));
  }
  if (!peerSpec.producerEndpoints.empty()) {
    for (auto& exchange : exchangeSpecs) {
      if (exchange.producerEndpoints.empty()) {
        exchange.producerEndpoints = peerSpec.producerEndpoints;
      }
    }
  }
  const auto numExchanges = exchangeSpecs.size();

  // Validate fused-broadcast arrays. They are parallel int[][] / Object[][] of
  // length numFragments. Each entry is null OR an array (slotIndices and
  // iterators must agree in length per fragment). When null/empty: that
  // consumer fragment has no fused broadcasts (the common case).
  if (broadcastSlotIndicesPerFragArr != nullptr) {
    GLUTEN_CHECK(
        env->GetArrayLength(broadcastSlotIndicesPerFragArr) == numFragments,
        "broadcastSlotIndicesPerFrag length must match substraitPlans length");
  }
  if (broadcastIteratorsPerFragArr != nullptr) {
    GLUTEN_CHECK(
        env->GetArrayLength(broadcastIteratorsPerFragArr) == numFragments,
        "broadcastIteratorsPerFrag length must match substraitPlans length");
  }

  // --- Convert each Substrait plan to a Velox PlanNode ---
  // Note: no ExchangeSource factory registration needed here. The IBM
  // velox baseline's ExchangeAdapter swaps our plain ExchangeNodes
  // (transportType=kUcx) to UcxExchange at runtime, and UcxExchange
  // creates its own UcxExchangeSource — gluten doesn't register one.

  auto veloxPool = defaultLeafVeloxMemoryPool();
  auto sessionCfg = createFluxSessionConfig(runtime);

  std::vector<FluxFragmentSpec> fragmentSpecs;
  fragmentSpecs.reserve(numFragments);

  // Per-fragment producer wire schema: captured right after substrait->velox
  // conversion (before any plan rewriting), looked up by consumer fragments
  // when building the inbound ExchangeNode + strip-prefix ProjectNode.
  // Fragments are emitted in topological order (producers before consumers)
  // so the producer's entry is always populated before the consumer reads it.
  std::unordered_map<int, velox::RowTypePtr> producerWireTypes;

  // Local/single-task mode: collapse all fragments into one Velox Task,
  // replacing every UcxExchange boundary with a LocalPartitionNode or direct
  // broadcast inline. This is the default path for local FLUX execution: the
  // exchange semantics stay intact, but data stays inside the Velox task
  // instead of going through UCX split wiring.
  //
  // When active, "merged producer" fragments don't get a PartitionedOutput
  // wrapper and don't appear in fragmentSpecs; their plan tree is inlined
  // into the consumer fragment via LocalPartitionNode.
  // Read from the merged session+backend conf to honor per-query overrides.
  auto preLoopBackendConf = VeloxBackend::get()->getBackendConf();
  auto preLoopMergedMap = preLoopBackendConf->rawConfigsCopy();
  for (const auto& [key, val] : runtime->getConfMap()) {
    preLoopMergedMap[key] = val;
  }
  auto preLoopSessionCfg =
      std::make_shared<velox::config::ConfigBase>(std::move(preLoopMergedMap));
  const bool singleTaskModeRequested = preLoopSessionCfg->get<bool>(
      kFluxSingleTaskMode, kFluxSingleTaskModeDefault);
#ifdef GLUTEN_ENABLE_GPU
  const bool keepDeviceRootOutput = preLoopSessionCfg->get<bool>(
      kCudfSkipOutputToVelox, kCudfSkipOutputToVeloxDefault);
#else
  const bool keepDeviceRootOutput = false;
#endif
  const int32_t keyedFinalLocalDrivers = std::max(
      1,
      preLoopSessionCfg->get<int32_t>(
          kFluxKeyedFinalLocalDrivers, kFluxKeyedFinalLocalDriversDefault));
  bool singleTaskMode = singleTaskModeRequested;
  if (singleTaskMode) {
    // All known partition types are supported in single-task mode:
    //   SINGLE       -> LocalPartitionNode::Type::kGather
    //   HASH         -> kRepartition + HashPartitionFunctionSpec
    //   RANGE        -> kRepartition + RangePartitionFunctionSpec
    //   ROUND_ROBIN  -> kRepartition + RoundRobinPartitionFunctionSpec
    //   BROADCAST    -> producer plan inlined as-is (HashJoinBridge handles
    //                   the cross-pipeline access)
    // Any unknown partition type falls back to multi-task UCX path.
    static const std::set<std::string> kSupportedPartitionTypes{
        "SINGLE", "HASH", "RANGE", "ROUND_ROBIN", "BROADCAST"};
    for (const auto& exch : exchangeSpecs) {
      if (kSupportedPartitionTypes.count(exch.partitionType) == 0) {
        LOG(WARNING) << "FluxJniWrapper: singleTaskMode requested but exchange "
                     << exch.id << " partitionType='" << exch.partitionType
                     << "' not in supported set; falling back to multi-task";
        singleTaskMode = false;
        break;
      }
    }
  }
  // Fragment IDs that are "merged producers" — their plan tree gets inlined
  // into the consumer via LocalPartitionNode and they don't get their own
  // Velox Task.
  std::set<int32_t> mergedProducerIds;
  // Per-fragment unwrapped Velox plan, captured before PartitionedOutput
  // wrapping (and before ValueStream replacement for fragments that have no
  // inbound exchanges, i.e., leaf producers). Used by consumer fragments in
  // single-task mode.
  std::unordered_map<int, velox::core::PlanNodePtr> unwrappedFragmentPlans;
  // Per-fragment scan info accumulated for merged producers — these flow
  // into the surviving root fragment's scanInfos so FluxQueryCoordinator
  // injects their splits into the merged plan.
  std::unordered_map<int, std::vector<std::shared_ptr<SplitInfo>>>
      mergedProducerScanInfos;
  std::unordered_map<int, std::vector<velox::core::PlanNodeId>>
      mergedProducerScanNodeIds;
  int32_t singleTaskThreadPoolDriverBudget = 0;
  if (singleTaskMode) {
    for (const auto& exch : exchangeSpecs) {
      mergedProducerIds.insert(exch.producerFragmentId);
    }
    LOG(WARNING) << "FluxJniWrapper: singleTaskMode active, "
                 << mergedProducerIds.size()
                 << " producer fragment(s) will be merged into consumers";
  }

  // Global plan-node-id allocator for single-task merge: each fragment's
  // VeloxPlanConverter starts allocating ids from this value, then bumps
  // the allocator to its high water mark. Ensures unique ids across the
  // spliced plan tree (Velox Task::buildSplitStates VELOX_USER_CHECK
  // requires unique node ids, otherwise: "Plan node IDs must be unique").
  // Default starts at 100 to leave headroom and make the merged ids
  // visually distinguishable from raw substrait conversion ids.
  uint64_t globalPlanNodeIdAllocator = 100;

  for (jsize i = 0; i < numFragments; ++i) {
    auto planByteArray =
        static_cast<jbyteArray>(env->GetObjectArrayElement(substraitPlansArr, i));
    auto safePlanBytes = getByteArrayElementsSafe(env, planByteArray);
    auto planSize = env->GetArrayLength(planByteArray);

    // Parse protobuf Substrait plan.
    ::substrait::Plan substraitPlan;
    GLUTEN_CHECK(
        parseProtobuf(
            reinterpret_cast<const uint8_t*>(safePlanBytes.elems()),
            planSize,
            &substraitPlan),
        fmt::format("Failed to parse Substrait plan for fragment {}", i));

    // Convert Substrait -> Velox PlanNode with backend defaults plus runtime
    // session overrides (e.g., cudf=true).
    LOG(WARNING) << "FluxJniWrapper: fragment " << i
                << " cudf.enabled="
                << sessionCfg->get<std::string>(
                       "spark.gluten.sql.columnar.cudf", "NOT_SET")
                << " cudf.enableTableScan="
                << sessionCfg->get<std::string>(
                       "spark.gluten.sql.columnar.backend.velox.cudf.enableTableScan",
                       "NOT_SET")
                << " cudf.backend.enabled="
                << sessionCfg->get<std::string>(
                       "spark.gluten.sql.columnar.backend.velox.cudf.enabled",
                       "NOT_SET")
                << " confMap.size=" << runtime->getConfMap().size();
    // Count inbound exchanges + fused broadcasts for this fragment. Each
    // ReadRel(iterator:N) in the substrait plan needs a corresponding entry
    // in placeholderIters. Exchange slots stay nullptr (replaced with
    // ExchangeNode below). Broadcast slots get a real ResultIterator built
    // from the JVM-side Iterator[ColumnarBatch] supplied by
    // FluxNativeQueryRDD.materializeFusedBroadcastIteratorImpl. Without this,
    // SubstraitToVeloxPlan.cc:constructCudfValueStreamNode hits
    // `streamIdx N vs size N` OOB the moment the consumer plan's substrait
    // mentions a fused broadcast input.
    int32_t numExchangeInputs = 0;
    for (const auto& exchange : exchangeSpecs) {
      if (exchange.consumerFragmentId == static_cast<int32_t>(i)) {
        numExchangeInputs++;
      }
    }

    // Look up this fragment's broadcast slot indices.
    std::vector<int32_t> broadcastSlotIndicesForFrag;
    jobjectArray broadcastIterForFragArr = nullptr;
    if (broadcastSlotIndicesPerFragArr != nullptr) {
      auto slotsArrObj = static_cast<jintArray>(
          env->GetObjectArrayElement(broadcastSlotIndicesPerFragArr, i));
      if (slotsArrObj != nullptr) {
        auto safeSlots = getIntArrayElementsSafe(env, slotsArrObj);
        jsize n = env->GetArrayLength(slotsArrObj);
        broadcastSlotIndicesForFrag.reserve(n);
        for (jsize s = 0; s < n; ++s) {
          broadcastSlotIndicesForFrag.push_back(safeSlots.elems()[s]);
        }
        env->DeleteLocalRef(slotsArrObj);
      }
    }
    if (broadcastIteratorsPerFragArr != nullptr) {
      broadcastIterForFragArr = static_cast<jobjectArray>(
          env->GetObjectArrayElement(broadcastIteratorsPerFragArr, i));
    }

    const int32_t numBroadcastInputs =
        static_cast<int32_t>(broadcastSlotIndicesForFrag.size());
    if (broadcastIterForFragArr != nullptr) {
      jsize iterLen = env->GetArrayLength(broadcastIterForFragArr);
      GLUTEN_CHECK(
          iterLen == numBroadcastInputs,
          fmt::format(
              "Fragment {} broadcastSlotIndices length {} != "
              "broadcastIterators length {}",
              i,
              numBroadcastInputs,
              iterLen));
    }

    const int32_t numStreamInputs = numExchangeInputs + numBroadcastInputs;
    std::vector<int32_t> iteratorIndices;
    collectIteratorIndices(substraitPlan, iteratorIndices);
    int32_t maxIteratorIndex = -1;
    for (const auto index : iteratorIndices) {
      maxIteratorIndex = std::max(maxIteratorIndex, index);
    }
    VELOX_CHECK_LT(
        maxIteratorIndex,
        numStreamInputs,
        "Fragment {} Substrait has ReadRel iterator slot(s) {} but JNI "
        "prepared only {} FLUX stream input(s): {} inbound exchange(s) + {} "
        "fused broadcast(s). This indicates the fragment extractor dropped "
        "an InputIteratorTransformer boundary before native plan conversion.",
        i,
        formatIndices(iteratorIndices),
        numStreamInputs,
        numExchangeInputs,
        numBroadcastInputs);
    std::vector<std::shared_ptr<ResultIterator>> placeholderIters(
        numStreamInputs, nullptr);

    // Materialize broadcast slot iterators via the standard Gluten JNI bridge.
    // Each Iterator[ColumnarBatch] becomes a ResultIterator that, when Velox's
    // ValueStream operator pulls next(), call back into the JVM via JNIEnv to
    // fetch the next ColumnarBatch from the broadcasted BuildSideRelation.
    std::unordered_set<int32_t> broadcastSlotSet;
    for (int32_t b = 0; b < numBroadcastInputs; ++b) {
      const int32_t slotIdx = broadcastSlotIndicesForFrag[b];
      GLUTEN_CHECK(
          slotIdx >= 0 && slotIdx < numStreamInputs,
          fmt::format(
              "Fragment {} broadcast slot {} out of range "
              "(numStreamInputs={})",
              i,
              slotIdx,
              numStreamInputs));
      auto jIter = env->GetObjectArrayElement(broadcastIterForFragArr, b);
      GLUTEN_CHECK(
          jIter != nullptr,
          fmt::format(
              "Fragment {} broadcast iterator at index {} is null",
              i,
              b));
      auto wrapped = makeJniColumnarBatchIterator(env, jIter, ctx);
      placeholderIters[slotIdx] =
          std::make_shared<ResultIterator>(std::move(wrapped));
      broadcastSlotSet.insert(slotIdx);
      LOG(WARNING) << "FluxJniWrapper: fragment " << i
                   << " fused broadcast wired at slot " << slotIdx
                   << " (" << b + 1 << "/" << numBroadcastInputs << ")";
      env->DeleteLocalRef(jIter);
    }
    if (broadcastIterForFragArr != nullptr) {
      env->DeleteLocalRef(broadcastIterForFragArr);
    }

    VeloxPlanConverter converter(
        veloxPool.get(),
        sessionCfg.get(),
        placeholderIters,
        /*writeFilesTempPath=*/*Runtime::localWriteFilesTempPath(),
        /*writeFileName=*/*Runtime::localWriteFileName(),
        /*validationMode=*/false);
    // In single-task mode, fragments will be spliced into a single Velox plan
    // tree, so plan node ids must be globally unique across fragments. Advance
    // each fragment's converter id allocator so it starts above the previous
    // fragment's high-water mark. See nextPlanNodeIdValue/setNextPlanNodeId.
    if (singleTaskMode) {
      converter.setNextPlanNodeId(
          static_cast<int>(globalPlanNodeIdAllocator));
    }

    // Parse split infos for this fragment from byte[][][] parameter.
    std::vector<::substrait::ReadRel_LocalFiles> localFiles;
    if (splitInfosPerFragArr != nullptr) {
      auto fragSplitArr = static_cast<jobjectArray>(
          env->GetObjectArrayElement(splitInfosPerFragArr, i));
      if (fragSplitArr != nullptr) {
        jsize numSplits = env->GetArrayLength(fragSplitArr);
        for (jsize j = 0; j < numSplits; ++j) {
          auto splitBytes = static_cast<jbyteArray>(
              env->GetObjectArrayElement(fragSplitArr, j));
          auto safeSplitBytes = getByteArrayElementsSafe(env, splitBytes);
          auto splitSize = env->GetArrayLength(splitBytes);
          ::substrait::ReadRel_LocalFiles localFile;
          GLUTEN_CHECK(
              parseProtobuf(
                  reinterpret_cast<const uint8_t*>(safeSplitBytes.elems()),
                  splitSize,
                  &localFile),
              fmt::format(
                  "Failed to parse split info for fragment {} split {}", i, j));
          LOG(INFO) << "FluxJniWrapper: fragment " << i
                    << " split " << j << " has "
                    << localFile.items_size() << " file items";
          localFiles.push_back(std::move(localFile));
          env->DeleteLocalRef(splitBytes);
        }
        env->DeleteLocalRef(fragSplitArr);
      }
    }

    auto veloxPlanNode = converter.toVeloxPlan(substraitPlan, localFiles);
    bool fragmentUsesKeyedFinalLocalRepartition = false;
    if (singleTaskMode) {
      // Bump the global id allocator above this fragment's high-water mark
      // so the next fragment's converter doesn't reuse ids.
      globalPlanNodeIdAllocator =
          static_cast<uint64_t>(converter.nextPlanNodeIdValue());
    }
    // Capture this fragment's wire schema BEFORE plan rewriting. Consumer
    // fragments processed later look this up by exchange.producerFragmentId.
    producerWireTypes[static_cast<int>(i)] = veloxPlanNode->outputType();

    // Extract scan split info from the converter BEFORE tree rewriting.
    // For scan-containing fragments, this captures file scan node IDs and
    // their associated SplitInfo (paths, starts, lengths, format).
    std::vector<std::shared_ptr<SplitInfo>> fragScanInfos;
    std::vector<velox::core::PlanNodeId> fragScanNodeIds;
    {
      std::vector<velox::core::PlanNodeId> streamIds; // unused for FLUX
      VeloxRuntime::getInfoAndIds(
          converter.splitInfos(),
          veloxPlanNode->leafPlanNodeIds(),
          fragScanInfos,
          fragScanNodeIds,
          streamIds);
      if (!fragScanNodeIds.empty()) {
        LOG(INFO) << "FluxJniWrapper: fragment " << i
                  << " has " << fragScanNodeIds.size() << " scan node(s)";
        for (size_t si = 0; si < fragScanNodeIds.size(); ++si) {
          LOG(INFO) << "  scan node " << fragScanNodeIds[si]
                    << ": " << fragScanInfos[si]->paths.size() << " file(s)";
        }
      }
    }

    LOG(INFO) << "FluxJniWrapper: fragment " << i
              << " raw Velox plan: "
              << veloxPlanNode->toString(/*detailed=*/true, /*recursive=*/true);

    std::vector<const FluxExchangeSpec*> inboundExchanges;
    for (const auto& exchange : exchangeSpecs) {
      if (exchange.consumerFragmentId == static_cast<int32_t>(i)) {
        inboundExchanges.push_back(&exchange);
      }
    }

    if (!inboundExchanges.empty() || !broadcastSlotSet.empty()) {
      // Collect ValueStream leaf nodes in depth-first order.
      // The order matches SubstraitToVeloxPlanConverter's stream index
      // assignment (iterator:0, iterator:1, ...). With fused broadcasts in
      // play, valueStreamNodes contains BOTH exchange-backed slots (which
      // we must replace with ExchangeNode below) AND broadcast-backed slots
      // (which we must leave alone -- their placeholderIters[slot] entry is
      // a live JNI-backed iterator that Velox's ValueStream operator pulls
      // from at runtime). broadcastSlotSet identifies the latter.
      std::vector<velox::core::PlanNodePtr> valueStreamNodes;
      collectValueStreamNodes(veloxPlanNode, valueStreamNodes);

      VELOX_CHECK_EQ(
          valueStreamNodes.size(),
          static_cast<size_t>(numExchangeInputs + numBroadcastInputs),
          "Fragment {} has {} ValueStream nodes but {} inbound exchanges + "
          "{} fused broadcasts. These must match 1:1 (broadcast slots are "
          "kept as ValueStream, exchange slots are rewritten to ExchangeNode).",
          i,
          valueStreamNodes.size(),
          inboundExchanges.size(),
          numBroadcastInputs);

      // Match each ValueStream to its inbound exchange by structural
      // schema equivalence, NOT by DFS position. Spark enumerates
      // ExchangeSpecs in BSP plan walk order (left child first); the
      // Velox plan's ValueStream order depends on how Gluten emitted
      // the substrait join -- Gluten's HashJoin transformer can put the
      // build side on Velox's left, opposite Spark's streamed-first BSP
      // layout. On Q2 SF1K this surfaces as fragment 2 (part join partsupp):
      // inboundExchanges[0] is F0->F2 (part, 2 cols) but the leftmost
      // Velox ValueStream expects partsupp's 3 cols, so HashJoinNode::
      // validate fails with "left side join key not found: n0_0".
      //
      // Pass the producer's wire outputType to replaceValueStreamWithExchange
      // so it can build the ExchangeNode against the actual wire format and
      // synthesize a ProjectNode to strip any synthetic prefix columns /
      // rename to consumer-expected names. Without this Q17 silently returns
      // null and Q18 hangs on cuDF kindEquals at the fragment boundary.
      //
      // Single-task merge mode: instead of an ExchangeNode, pass the
      // producer fragment's already-converted unwrapped plan to splice it
      // in via LocalPartitionNode.
      auto schemaMatches = [](const velox::RowTypePtr& a,
                              const velox::RowTypePtr& b) {
        if (a == nullptr || b == nullptr || a->size() != b->size()) {
          return false;
        }
        for (size_t k = 0; k < a->size(); ++k) {
          if (!a->childAt(k)->equivalent(*b->childAt(k))) {
            return false;
          }
        }
        return true;
      };
      std::vector<size_t> exchangeForStream(
          valueStreamNodes.size(), std::numeric_limits<size_t>::max());
      std::vector<bool> exchangeUsed(inboundExchanges.size(), false);
      // Pass 1: greedy structural schema match against inbound exchanges.
      // Skip ValueStream slots that correspond to fused broadcasts -- those
      // stay as ValueStream and consume placeholderIters[slot] at runtime.
      // The producer wire may carry a leading synthetic hash_partition_key
      // column the consumer doesn't see, so also try the prefix-stripped
      // form.
      for (size_t j = 0; j < valueStreamNodes.size(); ++j) {
        if (broadcastSlotSet.count(static_cast<int32_t>(j)) > 0) {
          continue; // broadcast slot, leave alone
        }
        auto streamType = std::dynamic_pointer_cast<const velox::RowType>(
            valueStreamNodes[j]->outputType());
        for (size_t k = 0; k < inboundExchanges.size(); ++k) {
          if (exchangeUsed[k]) continue;
          velox::RowTypePtr producerWire;
          auto it = producerWireTypes.find(
              inboundExchanges[k]->producerFragmentId);
          if (it != producerWireTypes.end()) {
            producerWire = it->second;
          }
          bool matched = schemaMatches(streamType, producerWire);
          if (!matched && producerWire != nullptr && streamType != nullptr &&
              producerWire->size() == streamType->size() + 1) {
            std::vector<std::string> n;
            std::vector<velox::TypePtr> t;
            for (size_t kk = 1; kk < producerWire->size(); ++kk) {
              n.push_back(producerWire->nameOf(kk));
              t.push_back(producerWire->childAt(kk));
            }
            auto stripped = std::make_shared<const velox::RowType>(
                std::move(n), std::move(t));
            matched = schemaMatches(streamType, stripped);
          }
          if (matched) {
            exchangeForStream[j] = k;
            exchangeUsed[k] = true;
            break;
          }
        }
      }
      // Pass 2: positional fallback for any leftover non-broadcast slot.
      for (size_t j = 0; j < valueStreamNodes.size(); ++j) {
        if (broadcastSlotSet.count(static_cast<int32_t>(j)) > 0) {
          continue;
        }
        if (exchangeForStream[j] != std::numeric_limits<size_t>::max()) {
          continue;
        }
        for (size_t k = 0; k < inboundExchanges.size(); ++k) {
          if (!exchangeUsed[k]) {
            exchangeForStream[j] = k;
            exchangeUsed[k] = true;
            break;
          }
        }
      }
      for (size_t j = 0; j < valueStreamNodes.size(); ++j) {
        if (broadcastSlotSet.count(static_cast<int32_t>(j)) > 0) {
          LOG(WARNING) << "FluxJniWrapper: fragment " << i << " stream[" << j
                       << "] id=" << valueStreamNodes[j]->id()
                       << " type=" << valueStreamNodes[j]->outputType()->toString()
                       << " -> KEEP as ValueStream (fused broadcast slot)";
          continue;
        }
        const auto k = exchangeForStream[j];
        LOG(WARNING) << "FluxJniWrapper: fragment " << i << " stream[" << j
                     << "] id=" << valueStreamNodes[j]->id()
                     << " type=" << valueStreamNodes[j]->outputType()->toString()
                     << " -> exchange[" << k << "] producerF="
                     << inboundExchanges[k]->producerFragmentId
                     << " nodeId=" << inboundExchanges[k]->exchangeNodeId;
      }
      // Replace each non-broadcast ValueStream node with the matched
      // ExchangeNode. Broadcast slots remain as ValueStream and Velox reads
      // the broadcasted batches via placeholderIters[slot] at runtime.
      for (size_t j = 0; j < valueStreamNodes.size(); ++j) {
        if (broadcastSlotSet.count(static_cast<int32_t>(j)) > 0) {
          continue;
        }
        const auto k = exchangeForStream[j];
        velox::RowTypePtr producerWire;
        auto it = producerWireTypes.find(
            inboundExchanges[k]->producerFragmentId);
        if (it != producerWireTypes.end()) {
          producerWire = it->second;
        }
        const bool localRepartition =
            !singleTaskMode && keyedFinalLocalDrivers > 1 &&
            inboundExchanges[k]->partitionType == "HASH" &&
            hasValidHashKeys(
                inboundExchanges[k]->partitionKeyIndices, producerWire) &&
            feedsKeyedFinalAggregation(
                veloxPlanNode, valueStreamNodes[j]->id());
        fragmentUsesKeyedFinalLocalRepartition |= localRepartition;
        velox::core::PlanNodePtr producerPlanForMerge;
        if (singleTaskMode) {
          auto pit = unwrappedFragmentPlans.find(
              inboundExchanges[k]->producerFragmentId);
          VELOX_CHECK(
              pit != unwrappedFragmentPlans.end(),
              "single-task mode: producer fragment {} plan missing for exchange {}",
              inboundExchanges[k]->producerFragmentId,
              inboundExchanges[k]->id);
          producerPlanForMerge = pit->second;
          // Roll the producer fragment's scanInfos / scanNodeIds /
          // scanConnectorIds into this consumer's accumulator so when this
          // (root) fragment becomes the single Velox Task, FluxQueryCoordinator
          // injects the merged-in scan splits onto the right plan node.
          auto& mInfos = mergedProducerScanInfos[
              inboundExchanges[k]->producerFragmentId];
          auto& mIds = mergedProducerScanNodeIds[
              inboundExchanges[k]->producerFragmentId];
          if (!mInfos.empty()) {
            for (size_t mi = 0; mi < mInfos.size(); ++mi) {
              fragScanInfos.push_back(mInfos[mi]);
              fragScanNodeIds.push_back(mIds[mi]);
            }
            LOG(WARNING)
                << "FluxJniWrapper: single-task merge folded "
                << mInfos.size() << " scan(s) from fragment "
                << inboundExchanges[k]->producerFragmentId
                << " into fragment " << i;
          }
        }
        veloxPlanNode = replaceValueStreamWithExchange(
            veloxPlanNode,
            valueStreamNodes[j]->id(),
            inboundExchanges[k]->exchangeNodeId,
            producerWire,
            producerPlanForMerge,
            inboundExchanges[k]->partitionType,
            inboundExchanges[k]->partitionKeyIndices,
            inboundExchanges[k]->rangeBoundsJson,
            localRepartition);
      }

      LOG(INFO) << "FluxJniWrapper: fragment " << i
                << " after ValueStream->Exchange replacement: "
                << veloxPlanNode->toString(
                       /*detailed=*/true, /*recursive=*/true);
    }
    if (!singleTaskMode && keyedFinalLocalDrivers > 1) {
      bool insertedAfterJoin = false;
      veloxPlanNode = insertKeyedFinalLocalRepartitionAfterJoin(
          veloxPlanNode, insertedAfterJoin);
      fragmentUsesKeyedFinalLocalRepartition |= insertedAfterJoin;
    }

    // Capture the post-rewrite, pre-wrap plan for use by downstream consumer
    // fragments in single-task merge mode. This is what gets inlined via
    // LocalPartitionNode when a consumer's ValueStream is replaced.
    if (singleTaskMode) {
      unwrappedFragmentPlans[static_cast<int>(i)] = veloxPlanNode;
      // Also track scan info so we can fold it into the consumer's
      // scanInfos when the consumer is being merged into the root.
      mergedProducerScanInfos[static_cast<int>(i)] = fragScanInfos;
      mergedProducerScanNodeIds[static_cast<int>(i)] = fragScanNodeIds;
    }

    // Skip the PartitionedOutput wrap and fragmentSpecs push for fragments
    // that are getting merged into a downstream consumer. The consumer will
    // own the merged plan as a single Velox Task.
    if (singleTaskMode &&
        mergedProducerIds.find(static_cast<int32_t>(i)) !=
            mergedProducerIds.end()) {
      LOG(WARNING) << "FluxJniWrapper: fragment " << i
                   << " is a merged producer in single-task mode, "
                   << "skipping PartitionedOutput wrap and spec push";
      env->DeleteLocalRef(planByteArray);
      continue;
    }

    // --- Problem 1: Wrap with PartitionedOutputNode ---
    //
    // Every fragment needs a PartitionedOutputNode at the root so that
    // DefaultOutputBufferManager gets initialized when the Task starts.
    // Without
    // this, the coordinator cannot read output from the root fragment and
    // consumer fragments cannot fetch data from producer fragments.
    //
    // - Root fragment (id=0): single partition output (gather to coordinator)
    // - Producer fragments: partition count from exchange spec
    int32_t numOutputPartitions = 1;
    const FluxExchangeSpec* outboundExchange = nullptr;
    for (const auto& exchange : exchangeSpecs) {
      if (exchange.producerFragmentId == static_cast<int32_t>(i)) {
        numOutputPartitions = exchange.numPartitions;
        outboundExchange = &exchange;
        break;
      }
    }

    auto outputNodeId = fmt::format("flux_output_{}", i);
    velox::core::PlanNodePtr wrappedPlan;
    const std::string partitionType = outboundExchange != nullptr
        ? outboundExchange->partitionType
        : std::string("ROOT");
    const bool isBroadcastOutput =
        outboundExchange != nullptr && partitionType == "BROADCAST";
    const char* outputKindHelper = isBroadcastOutput
        ? "broadcast"
        : (numOutputPartitions == 1 ? "single" : "partitioned");

    LOG(INFO) << "FluxJniWrapper: fragment " << i
              << " outbound partitionType=" << partitionType
              << " outputKindHelper=" << outputKindHelper
              << " numOutputPartitions=" << numOutputPartitions;

    if (isBroadcastOutput) {
      // Broadcast output must be tagged as kBroadcast even when the exchange
      // spec carries one producer partition; coordinator fanout is applied via
      // updateOutputBuffers(N, true) after wiring.
      wrappedPlan = velox::core::PartitionedOutputNode::broadcast(
          outputNodeId,
          numOutputPartitions,
          veloxPlanNode->outputType(),
          /*serdeKind=*/"Presto",
          std::string{velox::core::TransportKind::kUcx},
          veloxPlanNode);
    } else if (numOutputPartitions == 1) {
      // Single-partition gather output. Two cases:
      //   1. Producer fragment with a SINGLE-gather outbound exchange
      //      (outboundExchange != nullptr) -> route through IBM cudf's
      //      PartitionedOutputAdapter -> UcxPartitionedOutput so the
      //      consumer fragment's UcxExchange can pull GPU pages via the
      //      IntraNodeTransferRegistry fast path.
      //   2. Root fragment (outboundExchange == nullptr) normally uses kHttp
      //      so the coordinator receives CPU Presto pages. A GPU write sink
      //      opts into kUcx so the coordinator receives packed device columns
      //      and hands a CudfVector directly to the libcudf writer.
      const auto transportType =
          (outboundExchange != nullptr || keepDeviceRootOutput)
          ? std::string{velox::core::TransportKind::kUcx}
          : std::string{velox::core::TransportKind::kInMemory};
      wrappedPlan = velox::core::PartitionedOutputNode::single(
          outputNodeId,
          veloxPlanNode->outputType(),
          /*serdeKind=*/"Presto",
          transportType,
          veloxPlanNode);
    } else {
      // Multi-partition output. RANGE uses a dedicated Spark-boundary PID
      // function; it is never substituted with hash or round-robin.
      const auto& keyIndices = outboundExchange != nullptr
          ? outboundExchange->partitionKeyIndices
          : std::vector<int32_t>{};

      const auto& outputType = veloxPlanNode->outputType();
      auto specPair = buildPartitionFunctionSpec(
          partitionType,
          keyIndices,
          outputType,
          static_cast<int32_t>(i),
          outboundExchange != nullptr ? outboundExchange->rangeBoundsJson
                                      : std::string{});

      LOG(WARNING) << "FluxJniWrapper: fragment " << i
                   << " outbound exchange type=" << partitionType
                   << " keyIndices.size=" << keyIndices.size()
                   << " keyChannels=["
                   << [&] {
                        std::string s;
                        for (size_t k = 0; k < keyIndices.size(); ++k) {
                          if (k) s += ",";
                          s += std::to_string(keyIndices[k]);
                        }
                        return s;
                      }()
                   << "] func="
                   << (specPair.funcSpec ? specPair.funcSpec->toString() : "null")
                   << " usingRoundRobinFallback="
                   << (partitionType == "HASH" && keyIndices.empty() ? "YES" : "no");

      // Emit a plain velox PartitionedOutputNode tagged with
      // TransportType::kUcx. IBM cudf's PartitionedOutputAdapter swaps
      // this to UcxPartitionedOutput at runtime; UcxPartitionedOutput
      // detects the same-Communicator-instance case via
      // IntraNodeTransferRegistry and hands cudf::packed_columns
      // shared_ptrs to the consumer without any UCX wire transfer or
      // serde, so the same code path covers both CPU and GPU runs.
      wrappedPlan = std::make_shared<velox::core::PartitionedOutputNode>(
          outputNodeId,
          velox::core::PartitionedOutputNode::Kind::kPartitioned,
          std::move(specPair.partitionExprs),
          numOutputPartitions,
          /*replicateNullsAndAny=*/false,
          std::move(specPair.funcSpec),
          veloxPlanNode->outputType(),
          /*serdeKind=*/"Presto",
          std::string{velox::core::TransportKind::kUcx},
          veloxPlanNode);
    }

    LOG(INFO) << "FluxJniWrapper: fragment " << i
              << " final plan (with PartitionedOutput): "
              << wrappedPlan->toString(/*detailed=*/true, /*recursive=*/true);

    // Build PlanFragment with ungrouped execution.
    std::unordered_set<velox::core::PlanNodeId> emptyGroupedIds;
    velox::core::PlanFragment planFragment{
        wrappedPlan,
        velox::core::ExecutionStrategy::kUngrouped,
        1, // numSplitGroups
        emptyGroupedIds};

    const bool rightSemiProjectMultiDriverSafe =
        outboundExchange != nullptr && partitionType == "HASH" &&
        hasValidHashKeys(
            outboundExchange->partitionKeyIndices,
            veloxPlanNode->outputType()) &&
        isRightSemiProjectMultiDriverSafePlan(veloxPlanNode);
    if (rightSemiProjectMultiDriverSafe) {
      LOG(WARNING) << "FluxJniWrapper: fragment " << i
                   << " is a RIGHT_SEMI_PROJECT HASH-join shape that is "
                      "safe for intra-task multi-driver execution";
    }
    const bool innerJoinMultiDriverSafe =
        outboundExchange != nullptr && partitionType == "HASH" &&
        hasValidHashKeys(
            outboundExchange->partitionKeyIndices,
            veloxPlanNode->outputType()) &&
        isInnerJoinMultiDriverSafePlan(veloxPlanNode);
    if (innerJoinMultiDriverSafe) {
      LOG(WARNING) << "MppJniWrapper: fragment " << i
                   << " is a pure INNER HASH-join shape that is safe for "
                      "intra-task multi-driver execution";
    }

    FluxFragmentSpec fragSpec;
    // In single-task mode, merged producers are skipped (continue) above, so
    // fragment IDs in the surviving fragmentSpecs would be non-contiguous
    // (e.g. just {1}). FluxQueryCoordinator requires contiguous IDs starting
    // at 0, so renumber here based on the position in fragmentSpecs.
    fragSpec.id = singleTaskMode
        ? static_cast<int32_t>(fragmentSpecs.size())
        : static_cast<int32_t>(i);
    fragSpec.planFragment = std::move(planFragment);
    // numDrivers is the max-drivers parameter passed to Task::start. Velox
    // splits the plan into pipelines at LocalPartitionNode boundaries; each
    // pipeline gets up to numDrivers driver instances. For single-task merge,
    // we pick the max parallelism across all merged-in fragments so the
    // scan pipeline (e.g. parallelism=4) gets enough drivers, even though
    // the consumer pipeline (parallelism=1, e.g. SINGLE final agg) only uses
    // 1. Velox's pipeline-aware driver assignment handles the per-pipeline
    // narrowing internally.
    int32_t mergedNumDrivers = safeNumDrivers.elems()[i];
    if (singleTaskMode) {
      for (auto producerId : mergedProducerIds) {
        if (producerId >= 0 && producerId < numFragments) {
          mergedNumDrivers = std::max(
              mergedNumDrivers, safeNumDrivers.elems()[producerId]);
        }
      }
      // Cap to IBM's GPU-friendly default (2). Without this, queries that
      // inherit Spark default `spark.sql.shuffle.partitions=200` would set
      // mergedNumDrivers=200 and Velox would create 200 driver lanes per
      // pipeline — way more than a single GPU can usefully drive (cuDF
      // SM saturation + RMM mutex contention). Configurable via
      // kFluxSingleTaskMaxDrivers conf if user wants to override.
      const int32_t driverCap = preLoopSessionCfg->get<int32_t>(
          kFluxSingleTaskMaxDrivers, kFluxSingleTaskMaxDriversDefault);
      if (mergedNumDrivers > driverCap) {
        LOG(WARNING) << "FluxJniWrapper: capping mergedNumDrivers from "
                     << mergedNumDrivers << " to " << driverCap
                     << " (singleTaskMaxDrivers)";
        mergedNumDrivers = driverCap;
      }

      // Keep the single-GPU executor budget tied to the capped per-pipeline
      // driver count. Multiplying by folded pipeline count can start many
      // independent cuDF-heavy pipelines concurrently in one merged task,
      // increasing peak RMM pressure without increasing useful GPU parallelism.
      singleTaskThreadPoolDriverBudget = std::max(
          singleTaskThreadPoolDriverBudget,
          std::max(1, mergedNumDrivers));
    }
    if (fragmentUsesKeyedFinalLocalRepartition) {
      mergedNumDrivers = std::max(mergedNumDrivers, keyedFinalLocalDrivers);
      LOG(WARNING) << "FluxJniWrapper: fragment " << i
                   << " enabling keyed FINAL local HASH repartition with "
                   << mergedNumDrivers << " drivers";
    }
    fragSpec.numDrivers = mergedNumDrivers;
    fragSpec.keyedFinalLocalRepartition =
        fragmentUsesKeyedFinalLocalRepartition;
    fragSpec.rightSemiProjectMultiDriverSafe =
        rightSemiProjectMultiDriverSafe;
    fragSpec.innerJoinMultiDriverSafe =
        innerJoinMultiDriverSafe;
    fragSpec.scanInfos = std::move(fragScanInfos);
    fragSpec.scanNodeIds = std::move(fragScanNodeIds);
    // Determine connector IDs for scan nodes from the converted Velox plan.
    // Split connector IDs must match the TableScan table handle connector.
    // cuDF eligibility is decided during Substrait -> Velox conversion.
    for (const auto& scanNodeId : fragSpec.scanNodeIds) {
      auto connectorId = getTableScanConnectorId(veloxPlanNode, scanNodeId);
      LOG(WARNING) << "FluxJniWrapper: fragment " << i
                   << " scan node " << scanNodeId
                   << " connector: '" << connectorId << "'";
      fragSpec.scanConnectorIds.push_back(std::move(connectorId));
    }

    fragmentSpecs.push_back(std::move(fragSpec));

    env->DeleteLocalRef(planByteArray);
  }

  // --- Create execution resources ---

  // Thread pool for task execution. Must account for consumer-fragment
  // replication: a fragment that consumes an N-partition exchange is
  // materialized by FluxQueryCoordinator as N sibling Tasks (one per
  // destination), so its driver count is replicaCount * numDrivers, not
  // just numDrivers. Mirror the replica-count derivation from
  // FluxQueryCoordinator::start() so the pool is sized for the *physical*
  // driver count the coordinator will actually spawn.
  std::vector<int32_t> replicaCount(fragmentSpecs.size(), 1);
  for (const auto& exchange : exchangeSpecs) {
    const auto consumer = exchange.consumerFragmentId;
    const auto n = std::max(1, exchange.numPartitions);
    if (replicaCount[consumer] == 1) {
      replicaCount[consumer] = n;
    }
    // Consistency check deferred to coordinator (VELOX_CHECK_EQ there).
  }
  int32_t totalPhysicalDrivers = 0;
  for (size_t i = 0; i < fragmentSpecs.size(); ++i) {
    // Replicated fragments use 1 driver per replica (matches
    // GpuMultiFragmentTest convention); non-replicated fragments use
    // spec.numDrivers.
    int32_t perReplicaDrivers =
        replicaCount[i] == 1 ? std::max(1, fragmentSpecs[i].numDrivers) : 1;
    totalPhysicalDrivers += replicaCount[i] * perReplicaDrivers;
  }
  // At least 4 threads. Size = 2x physical drivers for exchange I/O +
  // producer-wait headroom. No hard ceiling — Velox drivers yield on
  // GPU/exchange waits, so oversubscribing a few hundred threads is fine
  // and cheaper than starving.
  const int32_t effectivePhysicalDrivers =
      std::max(totalPhysicalDrivers, singleTaskThreadPoolDriverBudget);
  int32_t poolSize = std::max(4, effectivePhysicalDrivers * 2);
  LOG(WARNING) << "FluxJniWrapper: threadPool size=" << poolSize
               << " physicalDrivers=" << totalPhysicalDrivers
               << " effectivePhysicalDrivers=" << effectivePhysicalDrivers
               << " singleTaskBudget=" << singleTaskThreadPoolDriverBudget
               << " fragments=" << fragmentSpecs.size();
  auto executor =
      std::make_shared<folly::CPUThreadPoolExecutor>(poolSize);

  // Generate the query id before allocating the query memory pool. Multiple
  // FLUX query RDDs can run concurrently in one executor (for example, a
  // broadcast input while a write query is still draining). Velox requires
  // sibling memory-pool names to be unique, so the old fixed "FluxQuery" name
  // made the second query fail in addAggregateChild().
  static std::atomic<uint64_t> gFluxQueryCounter{0};
  const auto localQueryOrdinal =
      gFluxQueryCounter.fetch_add(1, std::memory_order_relaxed);
  auto queryId = peerSpec.queryId.empty()
      ? fmt::format("flux-{}", localQueryOrdinal)
      : peerSpec.queryId;
  const auto fluxPoolName = fmt::format("FluxQuery.{}", localQueryOrdinal);

  // Create QueryCtx. We pass nullptr for the executor in QueryCtx::create
  // since Velox tasks use the executor passed to Task::start() (the
  // coordinator calls task->start(numDrivers) which uses Task's internal
  // executor registration). The spill executor is optional.
  // Create a dedicated aggregate child pool for the FLUX query.
  // QueryCtx needs an aggregate pool so it can create leaf child pools
  // for each Task's operators.
  auto rootPool = runtime->memoryManager()->getAggregateMemoryPool();
  auto fluxPool = rootPool->addAggregateChild(fluxPoolName);
  std::unordered_map<std::string, std::shared_ptr<velox::config::ConfigBase>>
      connectorConfigs;
  auto hiveConnectorSessionConfig = createHiveConnectorSessionConfig(sessionCfg);
  connectorConfigs[kHiveConnectorId] = hiveConnectorSessionConfig;
#ifdef GLUTEN_ENABLE_GPU
  connectorConfigs[kCudfHiveConnectorId] = hiveConnectorSessionConfig;
  connectorConfigs[kCudfIcebergConnectorId] = hiveConnectorSessionConfig;
#endif
  // Velox's OutputBuffer.bufferedBytes_ is a single scalar shared across ALL
  // destinations of a Task; enqueue blocks when it exceeds max_*_buffer_size.
  // Default 32 MB is catastrophic at N>=16: 32MB/N per destination trips
  // backpressure on the first few pages, stalling the producer pipeline and
  // never firing noMoreData to downstream. Raise to 1 GB to give chained
  // exchanges breathing room at N up to ~200.
  auto queryConfigMap = buildFluxQueryConfig(
      sessionCfg, static_cast<uint64_t>(replicatedCartesianMaxBuildBytes));
  std::shared_ptr<folly::CPUThreadPoolExecutor> spillExecutor;
  const auto spillThreadNum =
      sessionCfg->get<uint32_t>(kSpillThreadNum, kSpillThreadNumDefaultValue);
  if (spillThreadNum > 0) {
    spillExecutor = std::make_shared<folly::CPUThreadPoolExecutor>(spillThreadNum);
  }
  LOG(WARNING) << "FluxJniWrapper: QueryCtx configs=" << queryConfigMap.size()
               << " spillStrategy="
               << sessionCfg->get<std::string>(kSpillStrategy, kSpillStrategyDefaultValue)
               << " spillThreads=" << spillThreadNum;
  auto queryCtx = velox::core::QueryCtx::create(
      executor.get(),
      velox::core::QueryConfig{std::move(queryConfigMap)},
      connectorConfigs,
      VeloxBackend::get()->getAsyncDataCache(),
      fluxPool,
      spillExecutor.get(),
      fluxPoolName);

  std::optional<velox::common::SpillDiskOptions> spillDiskOpts;
  const auto spillStrategy =
      sessionCfg->get<std::string>(kSpillStrategy, kSpillStrategyDefaultValue);
  if (spillStrategy != "none") {
    VELOX_CHECK(
        !spillRootPath.empty(),
        "FLUX spill is enabled but Spark did not provide a local spill root");
    const auto spillDir = std::filesystem::path(spillRootPath);
    std::filesystem::create_directories(spillDir);
    velox::common::SpillDiskOptions opts;
    opts.spillDirPath = spillDir.string();
    opts.spillDirCreated = true;
    opts.spillDirCreateCb = nullptr;
    spillDiskOpts = std::move(opts);
    LOG(WARNING) << "FluxJniWrapper: spill disk enabled for " << queryId
                 << " dir=" << spillDir.string();
  } else {
    if (!spillRootPath.empty()) {
      std::error_code error;
      std::filesystem::remove_all(spillRootPath, error);
      if (error) {
        LOG(WARNING) << "FluxJniWrapper: failed to remove unused spill root "
                     << spillRootPath << ": " << error.message();
      }
    }
    LOG(WARNING) << "FluxJniWrapper: spill disk disabled for " << queryId;
  }

  // In single-task mode every exchange has been inlined as a LocalPartition,
  // and the producer fragments have been folded into the surviving root.
  // Clear exchangeSpecs so FluxQueryCoordinator's wiring loop skips and the
  // assertion that consumer/producer fragment ids be valid passes.
  if (singleTaskMode) {
    LOG(WARNING) << "FluxJniWrapper: clearing " << exchangeSpecs.size()
                 << " exchangeSpec(s) for single-task mode";
    exchangeSpecs.clear();
  }

  // Create the coordinator.
  auto coordinator = FluxQueryCoordinator::create(
      queryId,
      std::move(fragmentSpecs),
      std::move(exchangeSpecs),
      queryCtx,
      executor.get(),
      std::move(spillDiskOpts),
      peerSpec.localPeerId,
      peerSpec.peerIndex,
      peerSpec.peerCount);

  // Bundle into a handle.
  auto handle = std::make_shared<FluxQueryHandle>();
  handle->executor = std::move(executor);
  handle->queryCtx = std::move(queryCtx);
  handle->spillExecutor = std::move(spillExecutor);
  handle->coordinator = std::move(coordinator);
  handle->memoryPool = std::move(veloxPool);

  auto handleId = ctx->saveObject(std::static_pointer_cast<void>(handle));

  LOG(INFO) << "FluxJniWrapper: created FLUX query " << queryId
            << " with " << numFragments << " fragments, "
            << numExchanges << " exchanges, handle=" << handleId;

  return handleId;

  JNI_METHOD_END(kInvalidObjectHandle)
}

// ---------------------------------------------------------------------------
// nativeExplainFluxQuery
// ---------------------------------------------------------------------------

JNIEXPORT jobjectArray JNICALL
Java_org_apache_gluten_vectorized_FluxQueryJniWrapper_nativeExplainFluxQuery( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jobjectArray substraitPlansArr,
    jintArray numDriversArr,
    jbyteArray exchangeSpecsJsonArr,
    jobjectArray splitInfosPerFragArr,
    jobjectArray broadcastSlotIndicesPerFragArr,
    jobjectArray broadcastIteratorsPerFragArr) {
  JNI_METHOD_START

  auto ctx = getRuntime(env, wrapper);
  auto runtime = dynamic_cast<VeloxRuntime*>(ctx);
  GLUTEN_CHECK(runtime != nullptr, "FluxQuery explain requires VeloxRuntime");

  const jsize numFragments = env->GetArrayLength(substraitPlansArr);
  GLUTEN_CHECK(numFragments > 0, "At least one fragment plan is required");
  GLUTEN_CHECK(
      env->GetArrayLength(numDriversArr) == numFragments,
      "numDriversPerFragment length must match substraitPlans length");

  auto safeExchangeJson = getByteArrayElementsSafe(env, exchangeSpecsJsonArr);
  auto exchangeSpecs = parseExchangeSpecs(
      reinterpret_cast<const uint8_t*>(safeExchangeJson.elems()),
      env->GetArrayLength(exchangeSpecsJsonArr));

  if (broadcastSlotIndicesPerFragArr != nullptr) {
    GLUTEN_CHECK(
        env->GetArrayLength(broadcastSlotIndicesPerFragArr) == numFragments,
        "broadcastSlotIndicesPerFrag length must match substraitPlans length");
  }
  if (broadcastIteratorsPerFragArr != nullptr) {
    GLUTEN_CHECK(
        env->GetArrayLength(broadcastIteratorsPerFragArr) == numFragments,
        "broadcastIteratorsPerFrag length must match substraitPlans length");
  }

  auto veloxPool = defaultLeafVeloxMemoryPool();
  auto sessionCfg = createFluxSessionConfig(runtime);
  const int32_t keyedFinalLocalDrivers = std::max(
      1,
      sessionCfg->get<int32_t>(
          kFluxKeyedFinalLocalDrivers, kFluxKeyedFinalLocalDriversDefault));
  std::unordered_map<int, velox::RowTypePtr> producerWireTypes;
  std::vector<std::string> finalPlans;
  finalPlans.reserve(numFragments);

  for (jsize i = 0; i < numFragments; ++i) {
    auto planByteArray =
        static_cast<jbyteArray>(env->GetObjectArrayElement(substraitPlansArr, i));
    auto safePlanBytes = getByteArrayElementsSafe(env, planByteArray);
    auto planSize = env->GetArrayLength(planByteArray);

    ::substrait::Plan substraitPlan;
    GLUTEN_CHECK(
        parseProtobuf(
            reinterpret_cast<const uint8_t*>(safePlanBytes.elems()),
            planSize,
            &substraitPlan),
        fmt::format("Failed to parse Substrait plan for fragment {}", i));

    int32_t numExchangeInputs = 0;
    for (const auto& exchange : exchangeSpecs) {
      if (exchange.consumerFragmentId == static_cast<int32_t>(i)) {
        numExchangeInputs++;
      }
    }

    std::vector<int32_t> broadcastSlotIndicesForFrag;
    jobjectArray broadcastIterForFragArr = nullptr;
    if (broadcastSlotIndicesPerFragArr != nullptr) {
      auto slotsArrObj = static_cast<jintArray>(
          env->GetObjectArrayElement(broadcastSlotIndicesPerFragArr, i));
      if (slotsArrObj != nullptr) {
        auto safeSlots = getIntArrayElementsSafe(env, slotsArrObj);
        jsize n = env->GetArrayLength(slotsArrObj);
        broadcastSlotIndicesForFrag.reserve(n);
        for (jsize s = 0; s < n; ++s) {
          broadcastSlotIndicesForFrag.push_back(safeSlots.elems()[s]);
        }
        env->DeleteLocalRef(slotsArrObj);
      }
    }
    if (broadcastIteratorsPerFragArr != nullptr) {
      broadcastIterForFragArr = static_cast<jobjectArray>(
          env->GetObjectArrayElement(broadcastIteratorsPerFragArr, i));
    }

    const int32_t numBroadcastInputs =
        static_cast<int32_t>(broadcastSlotIndicesForFrag.size());
    if (broadcastIterForFragArr != nullptr) {
      jsize iterLen = env->GetArrayLength(broadcastIterForFragArr);
      GLUTEN_CHECK(
          iterLen == numBroadcastInputs,
          fmt::format(
              "Fragment {} broadcastSlotIndices length {} != broadcastIterators length {}",
              i,
              numBroadcastInputs,
              iterLen));
    }

    const int32_t numStreamInputs = numExchangeInputs + numBroadcastInputs;
    std::vector<int32_t> iteratorIndices;
    collectIteratorIndices(substraitPlan, iteratorIndices);
    int32_t maxIteratorIndex = -1;
    for (const auto index : iteratorIndices) {
      maxIteratorIndex = std::max(maxIteratorIndex, index);
    }
    VELOX_CHECK_LT(
        maxIteratorIndex,
        numStreamInputs,
        "Fragment {} Substrait has ReadRel iterator slot(s) {} but JNI "
        "prepared only {} FLUX stream input(s): {} inbound exchange(s) + {} "
        "fused broadcast(s).",
        i,
        formatIndices(iteratorIndices),
        numStreamInputs,
        numExchangeInputs,
        numBroadcastInputs);

    std::vector<std::shared_ptr<ResultIterator>> placeholderIters(
        numStreamInputs, nullptr);
    std::unordered_set<int32_t> broadcastSlotSet;
    for (int32_t b = 0; b < numBroadcastInputs; ++b) {
      const int32_t slotIdx = broadcastSlotIndicesForFrag[b];
      GLUTEN_CHECK(
          slotIdx >= 0 && slotIdx < numStreamInputs,
          fmt::format(
              "Fragment {} broadcast slot {} out of range (numStreamInputs={})",
              i,
              slotIdx,
              numStreamInputs));
      auto jIter = env->GetObjectArrayElement(broadcastIterForFragArr, b);
      GLUTEN_CHECK(
          jIter != nullptr,
          fmt::format("Fragment {} broadcast iterator at index {} is null", i, b));
      auto wrapped = makeJniColumnarBatchIterator(env, jIter, ctx);
      placeholderIters[slotIdx] =
          std::make_shared<ResultIterator>(std::move(wrapped));
      broadcastSlotSet.insert(slotIdx);
      env->DeleteLocalRef(jIter);
    }
    if (broadcastIterForFragArr != nullptr) {
      env->DeleteLocalRef(broadcastIterForFragArr);
    }

    VeloxPlanConverter converter(
        veloxPool.get(),
        sessionCfg.get(),
        placeholderIters,
        /*writeFilesTempPath=*/*Runtime::localWriteFilesTempPath(),
        /*writeFileName=*/*Runtime::localWriteFileName(),
        /*validationMode=*/false);

    std::vector<::substrait::ReadRel_LocalFiles> localFiles;
    if (splitInfosPerFragArr != nullptr) {
      auto fragSplitArr = static_cast<jobjectArray>(
          env->GetObjectArrayElement(splitInfosPerFragArr, i));
      if (fragSplitArr != nullptr) {
        jsize numSplits = env->GetArrayLength(fragSplitArr);
        for (jsize j = 0; j < numSplits; ++j) {
          auto splitBytes = static_cast<jbyteArray>(
              env->GetObjectArrayElement(fragSplitArr, j));
          auto safeSplitBytes = getByteArrayElementsSafe(env, splitBytes);
          auto splitSize = env->GetArrayLength(splitBytes);
          ::substrait::ReadRel_LocalFiles localFile;
          GLUTEN_CHECK(
              parseProtobuf(
                  reinterpret_cast<const uint8_t*>(safeSplitBytes.elems()),
                  splitSize,
                  &localFile),
              fmt::format(
                  "Failed to parse split info for fragment {} split {}", i, j));
          localFiles.push_back(std::move(localFile));
          env->DeleteLocalRef(splitBytes);
        }
        env->DeleteLocalRef(fragSplitArr);
      }
    }

    auto veloxPlanNode = converter.toVeloxPlan(substraitPlan, localFiles);
    producerWireTypes[static_cast<int>(i)] = veloxPlanNode->outputType();
    veloxPlanNode = rewriteValueStreamsForFlux(
        static_cast<int32_t>(i),
        veloxPlanNode,
        exchangeSpecs,
        producerWireTypes,
        broadcastSlotSet,
        numExchangeInputs,
        numBroadcastInputs,
        keyedFinalLocalDrivers);
    auto wrappedPlan = wrapWithFluxPartitionedOutput(
        static_cast<int32_t>(i), veloxPlanNode, exchangeSpecs);
    finalPlans.push_back(
        wrappedPlan->toString(/*detailed=*/true, /*recursive=*/true));
    env->DeleteLocalRef(planByteArray);
  }

  auto stringClass = env->FindClass("java/lang/String");
  auto result = env->NewObjectArray(numFragments, stringClass, nullptr);
  for (jsize i = 0; i < numFragments; ++i) {
    auto planString = env->NewStringUTF(finalPlans[i].c_str());
    env->SetObjectArrayElement(result, i, planString);
    env->DeleteLocalRef(planString);
  }
  env->DeleteLocalRef(stringClass);
  return result;

  JNI_METHOD_END(nullptr)
}

// ---------------------------------------------------------------------------
// nativeStartFluxQuery
// ---------------------------------------------------------------------------

JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_FluxQueryJniWrapper_nativeStartFluxQuery( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong handle) {
  JNI_METHOD_START
  nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{"jni::nativeStartFluxQuery"};

  auto fluxHandle = ObjectStore::retrieve<FluxQueryHandle>(handle);
  GLUTEN_CHECK(fluxHandle != nullptr, "Invalid FLUX query handle");
  GLUTEN_CHECK(fluxHandle->coordinator != nullptr, "FLUX coordinator is null");

  fluxHandle->coordinator->start();

  LOG(INFO) << "FluxJniWrapper: started FLUX query, handle=" << handle;

  JNI_METHOD_END()
}

// ---------------------------------------------------------------------------
// nativeGetFluxOutput
// ---------------------------------------------------------------------------

JNIEXPORT jlong JNICALL
Java_org_apache_gluten_vectorized_FluxQueryJniWrapper_nativeGetFluxOutput( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong handle) {
  JNI_METHOD_START
  nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{"jni::nativeGetFluxOutput"};

  auto ctx = getRuntime(env, wrapper);
  auto fluxHandle = ObjectStore::retrieve<FluxQueryHandle>(handle);
  GLUTEN_CHECK(fluxHandle != nullptr, "Invalid FLUX query handle");
  GLUTEN_CHECK(fluxHandle->coordinator != nullptr, "FLUX coordinator is null");

  auto rowVector = fluxHandle->coordinator->next();
  if (rowVector == nullptr) {
    // No more data — all fragments finished.
    return 0L;
  }

  // Wrap the RowVector in a VeloxColumnarBatch and save it in the
  // ObjectStore so Java can reference it by handle.
#ifdef GLUTEN_ENABLE_GPU
  // CudfVector intentionally has no Velox child vectors. Preserve the logical
  // column count so ColumnarBatches.create() keeps the native handle instead
  // of misclassifying this as a zero-column batch.
  auto batch = std::make_shared<VeloxColumnarBatch>(
      rowVector,
      static_cast<int32_t>(rowVector->type()->size()));
#else
  auto batch = std::make_shared<VeloxColumnarBatch>(rowVector);
#endif
  return ctx->saveObject(batch);

  JNI_METHOD_END(kInvalidObjectHandle)
}

// ---------------------------------------------------------------------------
// nativeAbortFluxQuery
// ---------------------------------------------------------------------------

/// Explicit pre-close abort hook. The JVM-side caller (e.g.
/// FluxNativeQueryExec.close, an iterator close-listener, or a
/// failAfter-driven cleanup path) invokes this BEFORE
/// nativeCloseFluxQuery so that the C++ coordinator gets a chance to call
/// Task::requestAbort() on every native Velox Task and wait (bounded) for
/// it to reach a terminal state. This is the only path that reliably
/// releases per-task MemoryPool reservations before JVM shutdown — without
/// it, a `failAfter`-style timeout returns from the test method while the
/// native Tasks keep running, and process exit later trips
/// MemoryManager::removePool()'s reservedBytes==0 VELOX_CHECK and
/// terminate()s the JVM.
///
/// Idempotent and safe to call without a matching close: the underlying
/// coordinator->abort() guards against re-entry. Does NOT release the
/// handle; the caller must still invoke nativeCloseFluxQuery to free the
/// ObjectStore slot and the native resources.
JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_FluxQueryJniWrapper_nativeAbortFluxQuery( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong handle) {
  JNI_METHOD_START
  nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{"jni::nativeAbortFluxQuery"};

  LOG(INFO) << "FluxJniWrapper: aborting FLUX query, handle=" << handle;

  auto fluxHandle = ObjectStore::retrieve<FluxQueryHandle>(handle);
  if (fluxHandle == nullptr) {
    LOG(WARNING) << "FluxJniWrapper: nativeAbortFluxQuery on unknown handle="
                 << handle << " (already released?)";
    return;
  }
  if (fluxHandle->coordinator == nullptr) {
    LOG(WARNING) << "FluxJniWrapper: nativeAbortFluxQuery handle=" << handle
                 << " has null coordinator";
    return;
  }

  try {
    fluxHandle->coordinator->abort();
  } catch (const std::exception& e) {
    LOG(ERROR) << "FluxJniWrapper: nativeAbortFluxQuery handle=" << handle
               << " coordinator->abort() threw: " << e.what();
  } catch (...) {
    LOG(ERROR) << "FluxJniWrapper: nativeAbortFluxQuery handle=" << handle
               << " coordinator->abort() threw unknown exception";
  }

  JNI_METHOD_END()
}

// ---------------------------------------------------------------------------
// nativeCloseFluxQuery
// ---------------------------------------------------------------------------

JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_FluxQueryJniWrapper_nativeCloseFluxQuery( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong handle) {
  JNI_METHOD_START
  nvtx3::scoped_range_in<GlutenFluxDomain> nvtxRange{"jni::nativeCloseFluxQuery"};

  LOG(INFO) << "FluxJniWrapper: closing FLUX query, handle=" << handle;

  // Drive the bounded abort+wait BEFORE dropping the shared_ptr. ~Flux-
  // QueryHandle's destructor would also reach this via coordinator.reset()
  // -> ~FluxQueryCoordinator, but doing it here makes the cleanup point
  // explicit and gives us a clearly-attributed log line if an abort hangs
  // at JVM-exit. abort() is idempotent so callers that have already issued
  // nativeAbortFluxQuery just see the aborted_ short-circuit.
  auto fluxHandle = ObjectStore::retrieve<FluxQueryHandle>(handle);
  if (fluxHandle != nullptr && fluxHandle->coordinator != nullptr) {
    try {
      fluxHandle->coordinator->logOperatorMetrics();
    } catch (const std::exception& e) {
      LOG(ERROR) << "FluxJniWrapper: nativeCloseFluxQuery handle=" << handle
                 << " coordinator->logOperatorMetrics() threw: " << e.what();
    } catch (...) {
      LOG(ERROR) << "FluxJniWrapper: nativeCloseFluxQuery handle=" << handle
                 << " coordinator->logOperatorMetrics() threw unknown exception";
    }
    try {
      fluxHandle->coordinator->abort();
    } catch (const std::exception& e) {
      LOG(ERROR) << "FluxJniWrapper: nativeCloseFluxQuery handle=" << handle
                 << " coordinator->abort() threw: " << e.what();
    } catch (...) {
      LOG(ERROR) << "FluxJniWrapper: nativeCloseFluxQuery handle=" << handle
                 << " coordinator->abort() threw unknown exception";
    }
  }
  // Drop the local retrieve() reference before release() so the only
  // remaining strong ref is the one inside the ObjectStore.
  fluxHandle.reset();

  // Release from the ObjectStore. This drops the shared_ptr<FluxQueryHandle>,
  // which triggers ~FluxQueryHandle -> coordinator->abort() (no-op now) -> cleanup.
  ObjectStore::release(handle);

  JNI_METHOD_END()
}

#ifdef __cplusplus
}
#endif
