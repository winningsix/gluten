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

#include "compute/MppQueryCoordinator.h"
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
#endif

using namespace gluten;
using namespace facebook;

// NVTX domain for gluten MPP. Keeps our ranges visually distinct from
// the velox / cudf NVTX ranges in nsys timeline.
namespace {
struct GlutenMppDomain {
  static constexpr char const* name{"gluten-mpp"};
};
} // namespace

// ---------------------------------------------------------------------------
// Helper: container for an MppQueryCoordinator plus the resources it needs
// that must outlive the coordinator (thread pool, memory pool, QueryCtx).
// This is the object stored via ObjectStore and referenced by the jlong handle.
// ---------------------------------------------------------------------------
namespace {

/// Bundles the MppQueryCoordinator with its owned resources so everything
/// has a clear lifetime: the JNI handle → MppQueryHandle → coordinator +
/// resources.  Destroying the handle tears everything down in order.
struct MppQueryHandle {
  /// Thread pool for Velox task execution. Owned here so it outlives the
  /// coordinator and all tasks.
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor;

  /// Query context (memory pool, config, cache).
  std::shared_ptr<velox::core::QueryCtx> queryCtx;

  /// Optional spill executor. QueryCtx stores a raw pointer to this.
  std::shared_ptr<folly::CPUThreadPoolExecutor> spillExecutor;

  /// The coordinator itself.
  std::shared_ptr<MppQueryCoordinator> coordinator;

  /// Memory pool kept alive for output deserialization.
  std::shared_ptr<velox::memory::MemoryPool> memoryPool;

  ~MppQueryHandle() {
    nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{
        "jni::~MppQueryHandle"};
    // Ensure coordinator is destroyed first (aborts any running tasks),
    // then queryCtx, then spillExecutor, then executor.
    {
      nvtx3::scoped_range_in<GlutenMppDomain> r{
          "jni::~MppQueryHandle:coordinator.reset"};
      coordinator.reset();
    }
    {
      nvtx3::scoped_range_in<GlutenMppDomain> r{
          "jni::~MppQueryHandle:queryCtx.reset"};
      queryCtx.reset();
    }
    {
      nvtx3::scoped_range_in<GlutenMppDomain> r{
          "jni::~MppQueryHandle:spillExecutor.reset"};
      spillExecutor.reset();
    }
    {
      nvtx3::scoped_range_in<GlutenMppDomain> r{
          "jni::~MppQueryHandle:executor.reset"};
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
        "Invalid MPP iterator URI '{}' in Substrait ReadRel: {}",
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

std::shared_ptr<velox::config::ConfigBase> createMppSessionConfig(
    VeloxRuntime* runtime) {
  auto backendConf = VeloxBackend::get()->getBackendConf();
  auto mergedMap = backendConf->rawConfigsCopy();
  for (const auto& [key, val] : runtime->getConfMap()) {
    mergedMap[key] = val;
  }
  return std::make_shared<velox::config::ConfigBase>(std::move(mergedMap));
}

std::unordered_map<std::string, std::string> buildMppQueryConfig(
    const std::shared_ptr<velox::config::ConfigBase>& veloxCfg,
    uint64_t replicatedCartesianMaxBuildBytes) {
  (void)replicatedCartesianMaxBuildBytes;
  std::unordered_map<std::string, std::string> configs;

  configs[velox::core::QueryConfig::kPreferredOutputBatchRows] =
      std::to_string(veloxCfg->get<uint32_t>(kSparkBatchSize, 4096));
  configs[velox::core::QueryConfig::kMaxOutputBatchRows] =
      std::to_string(veloxCfg->get<uint32_t>(kSparkBatchSize, 4096));
  configs[velox::core::QueryConfig::kPreferredOutputBatchBytes] =
      std::to_string(veloxCfg->get<uint64_t>(kVeloxPreferredBatchBytes, 10L << 20));

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
    throw std::runtime_error("Invalid MPP query conf arg: " + errDetails);
  }

  // Apply MPP exchange backpressure per fragment.  This must stay after the
  // dynamic config copy so both Velox output-buffer limits use one explicit
  // value.  The previous fixed 1 GiB per fragment allowed a 72-fragment query
  // to retain far more than a single GPU's memory.
  const auto mppMaxOutputBufferSize = veloxCfg->get<uint64_t>(
      kMppMaxOutputBufferSize, kMppMaxOutputBufferSizeDefault);
  configs[velox::core::QueryConfig::kMaxOutputBufferSize] =
      std::to_string(mppMaxOutputBufferSize);
  configs[velox::core::QueryConfig::kMaxPartitionedOutputBufferSize] =
      std::to_string(mppMaxOutputBufferSize);
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
/// with an ExchangeNode for MPP execution.
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
        "MPP RANGE fragment {} has no resolved sort-key indices; refusing "
        "hash/round-robin degradation",
        fragmentIdForLogging);
    VELOX_CHECK(
        !rangeBoundsJson.empty(),
        "MPP RANGE fragment {} has no Spark boundary descriptor; refusing "
        "hash/round-robin degradation",
        fragmentIdForLogging);
#ifndef GLUTEN_ENABLE_GPU
    VELOX_FAIL("MPP RANGE_PID requires the cuDF UCX backend");
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
              "MPP RANGE fragment {} key index {} is outside {} output "
              "fields; refusing hash/round-robin degradation",
              fragmentIdForLogging,
              idx,
              numFields);
        }
        LOG(WARNING) << "MppJniWrapper: fragment " << fragmentIdForLogging
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
        LOG(WARNING)
            << "MppJniWrapper: current Velox UCX exchange build does not "
               "provide RangePartitionFunctionSpec; using hash partition "
               "function for legacy MPP RANGE fragment "
            << fragmentIdForLogging;
        result.funcSpec =
            std::make_shared<velox::exec::HashPartitionFunctionSpec>(
                outputType, std::move(keyChannels));
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
        "MPP RANGE partition spec construction failed; refusing fallback");
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
    const std::string& mergeRangeBoundsJson = {}) {
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
        LOG(WARNING) << "MppJniWrapper: replacing ValueStream node '"
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
        LOG(WARNING) << "MppJniWrapper: replacing ValueStream node '"
                     << node->id()
                     << "' -> producer plan (single-task merge BROADCAST, "
                        "no LocalPartition wrap)";
        exchange = producerPlanForMerge;
      } else {
        // HASH / RANGE / ROUND_ROBIN -> kRepartition + appropriate spec.
        auto specPair = buildPartitionFunctionSpec(
            mergePartitionType, mergeKeyIndices, wireType,
            /*fragmentIdForLogging=*/-1, mergeRangeBoundsJson);
        LOG(WARNING) << "MppJniWrapper: replacing ValueStream node '"
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
      LOG(WARNING) << "MppJniWrapper: replacing ValueStream node '"
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
                     .transportType(
                         velox::core::ExchangeNode::TransportType::kUcx)
                     .build();
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
      if (producerPlanForMerge == nullptr) {
        // UCX exchange payloads are positional.  When width and child types
        // already match, expose the consumer names directly on ExchangeNode
        // instead of inserting an identity Project solely to rename fields.
        // Velox deliberately leaves such a no-op Project on CPU, which strict
        // cuDF mode would otherwise (incorrectly) report as fallback.
        LOG(WARNING) << "MppJniWrapper: applying wire->consumer names directly "
                     << "on Exchange " << exchangeNodeId;
        return velox::core::ExchangeNode::Builder()
            .id(exchangeNodeId)
            .outputType(consumerType)
            .serdeKind("Presto")
            .transportType(velox::core::ExchangeNode::TransportType::kUcx)
            .build();
      }
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
      LOG(WARNING) << "MppJniWrapper: renaming wire->consumer cols at "
                   << exchangeNodeId;
      return std::make_shared<velox::core::ProjectNode>(
          exchangeNodeId + "_rename",
          std::move(projectionNames),
          std::move(projections),
          std::move(exchange));
    }

    // Wire wider than consumer - the Spark-side MppCollapseRule injected
    // synthetic prefix column(s) for HASH partitioning that the consumer
    // wasn't told about. Inject a ProjectNode to drop the leading prefix
    // and rename the remaining cols to the consumer's expected names so
    // downstream FieldAccessTypedExpr name lookups resolve.
    if (wireType->size() < consumerType->size()) {
      LOG(WARNING) << "MppJniWrapper: wire " << wireType->toString()
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
    LOG(WARNING) << "MppJniWrapper: stripping " << skip
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
        mergeRangeBoundsJson);
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
  // union is translated to a gather LocalPartition, and single-task MPP merge
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

  // TableWriteNode (write-in-MPP): the final fragment's parquet write. Rebuild with the
  // rewritten source so the write executes inside the pinned MPP native task -- each peer
  // writes its own slice. Unary node: one source = the data to write.
  if (auto tableWriteNode =
          std::dynamic_pointer_cast<const velox::core::TableWriteNode>(node)) {
    return velox::core::TableWriteNode::Builder(*tableWriteNode)
        .source(newSources[0])
        .build();
  }

  VELOX_FAIL(
      "MppJniWrapper: unsupported plan node type '{}' (id='{}') encountered "
      "during ValueStream replacement. Add explicit Builder support for this "
      "node type in replaceValueStreamWithExchange().",
      node->name(),
      node->id());
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

velox::core::PlanNodePtr rewriteValueStreamsForMpp(
    int32_t fragmentId,
    velox::core::PlanNodePtr veloxPlanNode,
    const std::vector<MppExchangeSpec>& exchangeSpecs,
    const std::unordered_map<int, velox::RowTypePtr>& producerWireTypes,
    const std::unordered_set<int32_t>& broadcastSlotSet,
    int32_t numExchangeInputs,
    int32_t numBroadcastInputs) {
  std::vector<const MppExchangeSpec*> inboundExchanges;
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
      LOG(WARNING) << "MppJniWrapper: fragment " << fragmentId << " stream["
                   << j << "] id=" << valueStreamNodes[j]->id()
                   << " type=" << valueStreamNodes[j]->outputType()->toString()
                   << " -> KEEP as ValueStream (fused broadcast slot)";
      continue;
    }
    const auto k = exchangeForStream[j];
    LOG(WARNING) << "MppJniWrapper: fragment " << fragmentId << " stream["
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
    veloxPlanNode = replaceValueStreamWithExchange(
        veloxPlanNode,
        valueStreamNodes[j]->id(),
        inboundExchanges[k]->exchangeNodeId,
        producerWire);
  }

  LOG(INFO) << "MppJniWrapper: fragment " << fragmentId
            << " after ValueStream->Exchange replacement: "
            << veloxPlanNode->toString(/*detailed=*/true, /*recursive=*/true);
  return veloxPlanNode;
}

velox::core::PlanNodePtr wrapWithMppPartitionedOutput(
    int32_t fragmentId,
    velox::core::PlanNodePtr veloxPlanNode,
    const std::vector<MppExchangeSpec>& exchangeSpecs) {
  int32_t numOutputPartitions = 1;
  const MppExchangeSpec* outboundExchange = nullptr;
  for (const auto& exchange : exchangeSpecs) {
    if (exchange.producerFragmentId == fragmentId) {
      numOutputPartitions = exchange.numPartitions;
      outboundExchange = &exchange;
      break;
    }
  }

  auto outputNodeId = fmt::format("mpp_output_{}", fragmentId);
  const std::string partitionType =
      outboundExchange != nullptr ? outboundExchange->partitionType : std::string("ROOT");
  const bool isBroadcastOutput =
      outboundExchange != nullptr && partitionType == "BROADCAST";
  const char* outputKindHelper =
      isBroadcastOutput ? "broadcast" : (numOutputPartitions == 1 ? "single" : "partitioned");

  LOG(INFO) << "MppJniWrapper: fragment " << fragmentId
            << " outbound partitionType=" << partitionType
            << " outputKindHelper=" << outputKindHelper
            << " numOutputPartitions=" << numOutputPartitions;

  if (isBroadcastOutput) {
    return velox::core::PartitionedOutputNode::broadcast(
        outputNodeId,
        numOutputPartitions,
        veloxPlanNode->outputType(),
        /*serdeKind=*/"Presto",
        veloxPlanNode,
        velox::core::PartitionedOutputNode::TransportType::kUcx);
  }

  if (numOutputPartitions == 1) {
    const auto transportType = (outboundExchange != nullptr)
        ? velox::core::PartitionedOutputNode::TransportType::kUcx
        : velox::core::PartitionedOutputNode::TransportType::kHttp;
    return velox::core::PartitionedOutputNode::single(
        outputNodeId,
        veloxPlanNode->outputType(),
        /*serdeKind=*/"Presto",
        veloxPlanNode,
        transportType);
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

  LOG(WARNING) << "MppJniWrapper: fragment " << fragmentId
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
      veloxPlanNode,
      velox::core::PartitionedOutputNode::TransportType::kUcx);
}

struct MppPeerSpec {
  std::string queryId;
  std::string localPeerId{"local"};
  int32_t peerIndex{0};
  int32_t peerCount{1};
  std::vector<MppPeerEndpoint> producerEndpoints;
};

std::vector<MppPeerEndpoint> parsePeerEndpointArray(
    const folly::dynamic& endpoints) {
  VELOX_CHECK(endpoints.isArray(), "MPP peer endpoints must be a JSON array");
  std::vector<MppPeerEndpoint> parsed;
  parsed.reserve(endpoints.size());
  int32_t fallbackPeerIndex = 0;
  for (const auto& item : endpoints) {
    VELOX_CHECK(item.isObject(), "MPP peer endpoint must be an object");
    VELOX_CHECK(
        item.count("peerId") > 0,
        "MPP peer endpoint is missing required peerId");
    VELOX_CHECK(
        item.count("host") > 0,
        "MPP peer endpoint is missing required host");
    MppPeerEndpoint endpoint;
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

MppPeerSpec parseMppPeerSpec(const uint8_t* data, int32_t size) {
  MppPeerSpec spec;
  if (data == nullptr || size <= 0) {
    return spec;
  }
  std::string jsonStr(reinterpret_cast<const char*>(data), size);
  auto parsed = folly::parseJson(jsonStr);
  VELOX_CHECK(parsed.isObject(), "mppPeerSpecJson must be a JSON object");
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
std::vector<MppExchangeSpec> parseExchangeSpecs(
    const uint8_t* data,
    int32_t size) {
  std::string jsonStr(reinterpret_cast<const char*>(data), size);
  auto parsed = folly::parseJson(jsonStr);
  VELOX_CHECK(parsed.isArray(), "exchangeSpecsJson must be a JSON array");

  std::vector<MppExchangeSpec> specs;
  specs.reserve(parsed.size());
  int32_t exchangeId = 0;
  for (const auto& item : parsed) {
    MppExchangeSpec spec;
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
          "MPP RANGE exchange {} is missing Spark boundaries",
          spec.id);
      VELOX_CHECK_GT(
          spec.rangeEffectivePartitions,
          0,
          "MPP RANGE exchange {} has invalid effective partition count",
          spec.id);
      VELOX_CHECK_LE(
          spec.rangeEffectivePartitions,
          spec.numPartitions,
          "MPP RANGE exchange {} effective partitions exceed requested",
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
// nativeCreateMppQuery
// ---------------------------------------------------------------------------

JNIEXPORT jlong JNICALL
Java_org_apache_gluten_vectorized_MppQueryJniWrapper_nativeCreateMppQuery( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jobjectArray substraitPlansArr,
    jintArray numDriversArr,
    jbyteArray exchangeSpecsJsonArr,
    jbyteArray mppPeerSpecJsonArr,
    jobjectArray splitInfosPerFragArr,
    jobjectArray broadcastSlotIndicesPerFragArr,
    jobjectArray broadcastIteratorsPerFragArr,
    jlong replicatedCartesianMaxBuildBytes,
    jstring spillRootPathJstr) {
  JNI_METHOD_START
  nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{"jni::nativeCreateMppQuery"};

  auto ctx = getRuntime(env, wrapper);
  auto runtime = dynamic_cast<VeloxRuntime*>(ctx);
  GLUTEN_CHECK(runtime != nullptr, "MppQuery requires VeloxRuntime");
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
  MppPeerSpec peerSpec;
  if (mppPeerSpecJsonArr != nullptr &&
      env->GetArrayLength(mppPeerSpecJsonArr) > 0) {
    auto safePeerSpecJson = getByteArrayElementsSafe(env, mppPeerSpecJsonArr);
    peerSpec = parseMppPeerSpec(
        reinterpret_cast<const uint8_t*>(safePeerSpecJson.elems()),
        env->GetArrayLength(mppPeerSpecJsonArr));
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
  auto sessionCfg = createMppSessionConfig(runtime);

  std::vector<MppFragmentSpec> fragmentSpecs;
  fragmentSpecs.reserve(numFragments);

  // Per-fragment producer wire schema: captured right after substrait->velox
  // conversion (before any plan rewriting), looked up by consumer fragments
  // when building the inbound ExchangeNode + strip-prefix ProjectNode.
  // Fragments are emitted in topological order (producers before consumers)
  // so the producer's entry is always populated before the consumer reads it.
  std::unordered_map<int, velox::RowTypePtr> producerWireTypes;

  // Local/single-task mode: collapse all fragments into one Velox Task,
  // replacing every UcxExchange boundary with a LocalPartitionNode or direct
  // broadcast inline. This is the default path for local MPP execution: the
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
      kMppSingleTaskMode, kMppSingleTaskModeDefault);
#ifdef GLUTEN_ENABLE_GPU
  const bool keepDeviceRootOutput = preLoopSessionCfg->get<bool>(
      kCudfSkipOutputToVelox, kCudfSkipOutputToVeloxDefault);
#else
  const bool keepDeviceRootOutput = false;
#endif
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
        LOG(WARNING) << "MppJniWrapper: singleTaskMode requested but exchange "
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
  // into the surviving root fragment's scanInfos so MppQueryCoordinator
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
    LOG(WARNING) << "MppJniWrapper: singleTaskMode active, "
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
    LOG(WARNING) << "MppJniWrapper: fragment " << i
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
    // MppNativeQueryRDD.materializeFusedBroadcastIteratorImpl. Without this,
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
        "prepared only {} MPP stream input(s): {} inbound exchange(s) + {} "
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
      LOG(WARNING) << "MppJniWrapper: fragment " << i
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
          LOG(INFO) << "MppJniWrapper: fragment " << i
                    << " split " << j << " has "
                    << localFile.items_size() << " file items";
          localFiles.push_back(std::move(localFile));
          env->DeleteLocalRef(splitBytes);
        }
        env->DeleteLocalRef(fragSplitArr);
      }
    }

    auto veloxPlanNode = converter.toVeloxPlan(substraitPlan, localFiles);
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
      std::vector<velox::core::PlanNodeId> streamIds; // unused for MPP
      VeloxRuntime::getInfoAndIds(
          converter.splitInfos(),
          veloxPlanNode->leafPlanNodeIds(),
          fragScanInfos,
          fragScanNodeIds,
          streamIds);
      if (!fragScanNodeIds.empty()) {
        LOG(INFO) << "MppJniWrapper: fragment " << i
                  << " has " << fragScanNodeIds.size() << " scan node(s)";
        for (size_t si = 0; si < fragScanNodeIds.size(); ++si) {
          LOG(INFO) << "  scan node " << fragScanNodeIds[si]
                    << ": " << fragScanInfos[si]->paths.size() << " file(s)";
        }
      }
    }

    LOG(INFO) << "MppJniWrapper: fragment " << i
              << " raw Velox plan: "
              << veloxPlanNode->toString(/*detailed=*/true, /*recursive=*/true);

    std::vector<const MppExchangeSpec*> inboundExchanges;
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
          LOG(WARNING) << "MppJniWrapper: fragment " << i << " stream[" << j
                       << "] id=" << valueStreamNodes[j]->id()
                       << " type=" << valueStreamNodes[j]->outputType()->toString()
                       << " -> KEEP as ValueStream (fused broadcast slot)";
          continue;
        }
        const auto k = exchangeForStream[j];
        LOG(WARNING) << "MppJniWrapper: fragment " << i << " stream[" << j
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
          // (root) fragment becomes the single Velox Task, MppQueryCoordinator
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
                << "MppJniWrapper: single-task merge folded "
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
            inboundExchanges[k]->rangeBoundsJson);
      }

      LOG(INFO) << "MppJniWrapper: fragment " << i
                << " after ValueStream->Exchange replacement: "
                << veloxPlanNode->toString(
                       /*detailed=*/true, /*recursive=*/true);
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
      LOG(WARNING) << "MppJniWrapper: fragment " << i
                   << " is a merged producer in single-task mode, "
                   << "skipping PartitionedOutput wrap and spec push";
      env->DeleteLocalRef(planByteArray);
      continue;
    }

    // --- Problem 1: Wrap with PartitionedOutputNode ---
    //
    // Every fragment needs a PartitionedOutputNode at the root so that
    // OutputBufferManager gets initialized when the Task starts. Without
    // this, the coordinator cannot read output from the root fragment and
    // consumer fragments cannot fetch data from producer fragments.
    //
    // - Root fragment (id=0): single partition output (gather to coordinator)
    // - Producer fragments: partition count from exchange spec
    int32_t numOutputPartitions = 1;
    const MppExchangeSpec* outboundExchange = nullptr;
    for (const auto& exchange : exchangeSpecs) {
      if (exchange.producerFragmentId == static_cast<int32_t>(i)) {
        numOutputPartitions = exchange.numPartitions;
        outboundExchange = &exchange;
        break;
      }
    }

    auto outputNodeId = fmt::format("mpp_output_{}", i);
    velox::core::PlanNodePtr wrappedPlan;
    const std::string partitionType = outboundExchange != nullptr
        ? outboundExchange->partitionType
        : std::string("ROOT");
    const bool isBroadcastOutput =
        outboundExchange != nullptr && partitionType == "BROADCAST";
    const char* outputKindHelper = isBroadcastOutput
        ? "broadcast"
        : (numOutputPartitions == 1 ? "single" : "partitioned");

    LOG(INFO) << "MppJniWrapper: fragment " << i
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
          veloxPlanNode,
          velox::core::PartitionedOutputNode::TransportType::kUcx);
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
          ? velox::core::PartitionedOutputNode::TransportType::kUcx
          : velox::core::PartitionedOutputNode::TransportType::kHttp;
      wrappedPlan = velox::core::PartitionedOutputNode::single(
          outputNodeId,
          veloxPlanNode->outputType(),
          /*serdeKind=*/"Presto",
          veloxPlanNode,
          transportType);
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

      LOG(WARNING) << "MppJniWrapper: fragment " << i
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
          veloxPlanNode,
          velox::core::PartitionedOutputNode::TransportType::kUcx);
    }

    LOG(INFO) << "MppJniWrapper: fragment " << i
              << " final plan (with PartitionedOutput): "
              << wrappedPlan->toString(/*detailed=*/true, /*recursive=*/true);

    // Build PlanFragment with ungrouped execution.
    std::unordered_set<velox::core::PlanNodeId> emptyGroupedIds;
    velox::core::PlanFragment planFragment{
        wrappedPlan,
        velox::core::ExecutionStrategy::kUngrouped,
        1, // numSplitGroups
        emptyGroupedIds};

    MppFragmentSpec fragSpec;
    // In single-task mode, merged producers are skipped (continue) above, so
    // fragment IDs in the surviving fragmentSpecs would be non-contiguous
    // (e.g. just {1}). MppQueryCoordinator requires contiguous IDs starting
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
      // kMppSingleTaskMaxDrivers conf if user wants to override.
      const int32_t driverCap = preLoopSessionCfg->get<int32_t>(
          kMppSingleTaskMaxDrivers, kMppSingleTaskMaxDriversDefault);
      if (mergedNumDrivers > driverCap) {
        LOG(WARNING) << "MppJniWrapper: capping mergedNumDrivers from "
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
    fragSpec.numDrivers = mergedNumDrivers;
    fragSpec.scanInfos = std::move(fragScanInfos);
    fragSpec.scanNodeIds = std::move(fragScanNodeIds);
    // Determine connector IDs for scan nodes from the converted Velox plan.
    // Split connector IDs must match the TableScan table handle connector.
    // cuDF eligibility is decided during Substrait -> Velox conversion.
    for (const auto& scanNodeId : fragSpec.scanNodeIds) {
      auto connectorId = getTableScanConnectorId(veloxPlanNode, scanNodeId);
      LOG(WARNING) << "MppJniWrapper: fragment " << i
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
  // materialized by MppQueryCoordinator as N sibling Tasks (one per
  // destination), so its driver count is replicaCount * numDrivers, not
  // just numDrivers. Mirror the replica-count derivation from
  // MppQueryCoordinator::start() so the pool is sized for the *physical*
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
  LOG(WARNING) << "MppJniWrapper: threadPool size=" << poolSize
               << " physicalDrivers=" << totalPhysicalDrivers
               << " effectivePhysicalDrivers=" << effectivePhysicalDrivers
               << " singleTaskBudget=" << singleTaskThreadPoolDriverBudget
               << " fragments=" << fragmentSpecs.size();
  auto executor =
      std::make_shared<folly::CPUThreadPoolExecutor>(poolSize);

  // Generate the query id before allocating the query memory pool. Multiple
  // MPP query RDDs can run concurrently in one executor (for example, a
  // broadcast input while a write query is still draining). Velox requires
  // sibling memory-pool names to be unique, so the old fixed "MppQuery" name
  // made the second query fail in addAggregateChild().
  static std::atomic<uint64_t> gMppQueryCounter{0};
  const auto localQueryOrdinal =
      gMppQueryCounter.fetch_add(1, std::memory_order_relaxed);
  auto queryId = peerSpec.queryId.empty()
      ? fmt::format("mpp-{}", localQueryOrdinal)
      : peerSpec.queryId;
  const auto mppPoolName = fmt::format("MppQuery.{}", localQueryOrdinal);

  // Create QueryCtx. We pass nullptr for the executor in QueryCtx::create
  // since Velox tasks use the executor passed to Task::start() (the
  // coordinator calls task->start(numDrivers) which uses Task's internal
  // executor registration). The spill executor is optional.
  // Create a dedicated aggregate child pool for the MPP query.
  // QueryCtx needs an aggregate pool so it can create leaf child pools
  // for each Task's operators.
  auto rootPool = runtime->memoryManager()->getAggregateMemoryPool();
  auto mppPool = rootPool->addAggregateChild(mppPoolName);
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
  auto queryConfigMap = buildMppQueryConfig(
      sessionCfg, static_cast<uint64_t>(replicatedCartesianMaxBuildBytes));
  std::shared_ptr<folly::CPUThreadPoolExecutor> spillExecutor;
  const auto spillThreadNum =
      sessionCfg->get<uint32_t>(kSpillThreadNum, kSpillThreadNumDefaultValue);
  if (spillThreadNum > 0) {
    spillExecutor = std::make_shared<folly::CPUThreadPoolExecutor>(spillThreadNum);
  }
  LOG(WARNING) << "MppJniWrapper: QueryCtx configs=" << queryConfigMap.size()
               << " spillStrategy="
               << sessionCfg->get<std::string>(kSpillStrategy, kSpillStrategyDefaultValue)
               << " spillThreads=" << spillThreadNum;
  auto queryCtx = velox::core::QueryCtx::create(
      executor.get(),
      velox::core::QueryConfig{std::move(queryConfigMap)},
      connectorConfigs,
      VeloxBackend::get()->getAsyncDataCache(),
      mppPool,
      spillExecutor.get(),
      mppPoolName);

  std::optional<velox::common::SpillDiskOptions> spillDiskOpts;
  const auto spillStrategy =
      sessionCfg->get<std::string>(kSpillStrategy, kSpillStrategyDefaultValue);
  if (spillStrategy != "none") {
    VELOX_CHECK(
        !spillRootPath.empty(),
        "MPP spill is enabled but Spark did not provide a local spill root");
    const auto spillDir = std::filesystem::path(spillRootPath);
    std::filesystem::create_directories(spillDir);
    velox::common::SpillDiskOptions opts;
    opts.spillDirPath = spillDir.string();
    opts.spillDirCreated = true;
    opts.spillDirCreateCb = nullptr;
    spillDiskOpts = std::move(opts);
    LOG(WARNING) << "MppJniWrapper: spill disk enabled for " << queryId
                 << " dir=" << spillDir.string();
  } else {
    if (!spillRootPath.empty()) {
      std::error_code error;
      std::filesystem::remove_all(spillRootPath, error);
      if (error) {
        LOG(WARNING) << "MppJniWrapper: failed to remove unused spill root "
                     << spillRootPath << ": " << error.message();
      }
    }
    LOG(WARNING) << "MppJniWrapper: spill disk disabled for " << queryId;
  }

  // In single-task mode every exchange has been inlined as a LocalPartition,
  // and the producer fragments have been folded into the surviving root.
  // Clear exchangeSpecs so MppQueryCoordinator's wiring loop skips and the
  // assertion that consumer/producer fragment ids be valid passes.
  if (singleTaskMode) {
    LOG(WARNING) << "MppJniWrapper: clearing " << exchangeSpecs.size()
                 << " exchangeSpec(s) for single-task mode";
    exchangeSpecs.clear();
  }

  // Create the coordinator.
  auto coordinator = MppQueryCoordinator::create(
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
  auto handle = std::make_shared<MppQueryHandle>();
  handle->executor = std::move(executor);
  handle->queryCtx = std::move(queryCtx);
  handle->spillExecutor = std::move(spillExecutor);
  handle->coordinator = std::move(coordinator);
  handle->memoryPool = std::move(veloxPool);

  auto handleId = ctx->saveObject(std::static_pointer_cast<void>(handle));

  LOG(INFO) << "MppJniWrapper: created MPP query " << queryId
            << " with " << numFragments << " fragments, "
            << numExchanges << " exchanges, handle=" << handleId;

  return handleId;

  JNI_METHOD_END(kInvalidObjectHandle)
}

// ---------------------------------------------------------------------------
// nativeExplainMppQuery
// ---------------------------------------------------------------------------

JNIEXPORT jobjectArray JNICALL
Java_org_apache_gluten_vectorized_MppQueryJniWrapper_nativeExplainMppQuery( // NOLINT
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
  GLUTEN_CHECK(runtime != nullptr, "MppQuery explain requires VeloxRuntime");

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
  auto sessionCfg = createMppSessionConfig(runtime);
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
        "prepared only {} MPP stream input(s): {} inbound exchange(s) + {} "
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
    veloxPlanNode = rewriteValueStreamsForMpp(
        static_cast<int32_t>(i),
        veloxPlanNode,
        exchangeSpecs,
        producerWireTypes,
        broadcastSlotSet,
        numExchangeInputs,
        numBroadcastInputs);
    auto wrappedPlan = wrapWithMppPartitionedOutput(
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
// nativeStartMppQuery
// ---------------------------------------------------------------------------

JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_MppQueryJniWrapper_nativeStartMppQuery( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong handle) {
  JNI_METHOD_START
  nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{"jni::nativeStartMppQuery"};

  auto mppHandle = ObjectStore::retrieve<MppQueryHandle>(handle);
  GLUTEN_CHECK(mppHandle != nullptr, "Invalid MPP query handle");
  GLUTEN_CHECK(mppHandle->coordinator != nullptr, "MPP coordinator is null");

  mppHandle->coordinator->start();

  LOG(INFO) << "MppJniWrapper: started MPP query, handle=" << handle;

  JNI_METHOD_END()
}

// ---------------------------------------------------------------------------
// nativeGetMppOutput
// ---------------------------------------------------------------------------

JNIEXPORT jlong JNICALL
Java_org_apache_gluten_vectorized_MppQueryJniWrapper_nativeGetMppOutput( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong handle) {
  JNI_METHOD_START
  nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{"jni::nativeGetMppOutput"};

  auto ctx = getRuntime(env, wrapper);
  auto mppHandle = ObjectStore::retrieve<MppQueryHandle>(handle);
  GLUTEN_CHECK(mppHandle != nullptr, "Invalid MPP query handle");
  GLUTEN_CHECK(mppHandle->coordinator != nullptr, "MPP coordinator is null");

  auto rowVector = mppHandle->coordinator->next();
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
// nativeAbortMppQuery
// ---------------------------------------------------------------------------

/// Explicit pre-close abort hook. The JVM-side caller (e.g.
/// MppNativeQueryExec.close, an iterator close-listener, or a
/// failAfter-driven cleanup path) invokes this BEFORE
/// nativeCloseMppQuery so that the C++ coordinator gets a chance to call
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
/// handle; the caller must still invoke nativeCloseMppQuery to free the
/// ObjectStore slot and the native resources.
JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_MppQueryJniWrapper_nativeAbortMppQuery( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong handle) {
  JNI_METHOD_START
  nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{"jni::nativeAbortMppQuery"};

  LOG(INFO) << "MppJniWrapper: aborting MPP query, handle=" << handle;

  auto mppHandle = ObjectStore::retrieve<MppQueryHandle>(handle);
  if (mppHandle == nullptr) {
    LOG(WARNING) << "MppJniWrapper: nativeAbortMppQuery on unknown handle="
                 << handle << " (already released?)";
    return;
  }
  if (mppHandle->coordinator == nullptr) {
    LOG(WARNING) << "MppJniWrapper: nativeAbortMppQuery handle=" << handle
                 << " has null coordinator";
    return;
  }

  try {
    mppHandle->coordinator->abort();
  } catch (const std::exception& e) {
    LOG(ERROR) << "MppJniWrapper: nativeAbortMppQuery handle=" << handle
               << " coordinator->abort() threw: " << e.what();
  } catch (...) {
    LOG(ERROR) << "MppJniWrapper: nativeAbortMppQuery handle=" << handle
               << " coordinator->abort() threw unknown exception";
  }

  JNI_METHOD_END()
}

// ---------------------------------------------------------------------------
// nativeCloseMppQuery
// ---------------------------------------------------------------------------

JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_MppQueryJniWrapper_nativeCloseMppQuery( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong handle) {
  JNI_METHOD_START
  nvtx3::scoped_range_in<GlutenMppDomain> nvtxRange{"jni::nativeCloseMppQuery"};

  LOG(INFO) << "MppJniWrapper: closing MPP query, handle=" << handle;

  // Drive the bounded abort+wait BEFORE dropping the shared_ptr. ~Mpp-
  // QueryHandle's destructor would also reach this via coordinator.reset()
  // -> ~MppQueryCoordinator, but doing it here makes the cleanup point
  // explicit and gives us a clearly-attributed log line if an abort hangs
  // at JVM-exit. abort() is idempotent so callers that have already issued
  // nativeAbortMppQuery just see the aborted_ short-circuit.
  auto mppHandle = ObjectStore::retrieve<MppQueryHandle>(handle);
  if (mppHandle != nullptr && mppHandle->coordinator != nullptr) {
    try {
      mppHandle->coordinator->logOperatorMetrics();
    } catch (const std::exception& e) {
      LOG(ERROR) << "MppJniWrapper: nativeCloseMppQuery handle=" << handle
                 << " coordinator->logOperatorMetrics() threw: " << e.what();
    } catch (...) {
      LOG(ERROR) << "MppJniWrapper: nativeCloseMppQuery handle=" << handle
                 << " coordinator->logOperatorMetrics() threw unknown exception";
    }
    try {
      mppHandle->coordinator->abort();
    } catch (const std::exception& e) {
      LOG(ERROR) << "MppJniWrapper: nativeCloseMppQuery handle=" << handle
                 << " coordinator->abort() threw: " << e.what();
    } catch (...) {
      LOG(ERROR) << "MppJniWrapper: nativeCloseMppQuery handle=" << handle
                 << " coordinator->abort() threw unknown exception";
    }
  }
  // Drop the local retrieve() reference before release() so the only
  // remaining strong ref is the one inside the ObjectStore.
  mppHandle.reset();

  // Release from the ObjectStore. This drops the shared_ptr<MppQueryHandle>,
  // which triggers ~MppQueryHandle -> coordinator->abort() (no-op now) -> cleanup.
  ObjectStore::release(handle);

  JNI_METHOD_END()
}

#ifdef __cplusplus
}
#endif
