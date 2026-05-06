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
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <glog/logging.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
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
#endif

using namespace gluten;
using namespace facebook;

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

  /// The coordinator itself.
  std::shared_ptr<MppQueryCoordinator> coordinator;

  /// Memory pool kept alive for output deserialization.
  std::shared_ptr<velox::memory::MemoryPool> memoryPool;

  ~MppQueryHandle() {
    // Ensure coordinator is destroyed first (aborts any running tasks),
    // then queryCtx, then the executor.
    coordinator.reset();
    queryCtx.reset();
    executor.reset();
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
    const velox::RowTypePtr& producerWireType) {
  // Base case: this IS the target ValueStream leaf - replace it.
  if (isValueStreamNode(node) && node->id() == targetNodeId) {
    const auto& consumerType = node->outputType();
    const auto& wireType =
        producerWireType != nullptr ? producerWireType : consumerType;
    LOG(WARNING) << "MppJniWrapper: replacing ValueStream node '" << node->id()
                 << "' -> Exchange '" << exchangeNodeId
                 << "' wireType=" << wireType->toString()
                 << " consumerType=" << consumerType->toString();
    // ExchangeNode advertises the WIRE schema (producer's outputType, may
    // include a synthetic hash_partition_key:int prefix or other Spark-
    // injected partitioning columns). cuDF serdes the wire as-is; if we
    // declared the consumer's narrower type here, cuDF would silently
    // truncate columns and we'd see "Cannot change vector type" /
    // null-row corruption downstream (Q17 v9s, Q18 hang).
    auto exchange = velox::core::ExchangeNode::Builder()
                        .id(exchangeNodeId)
                        .outputType(wireType)
                        .serdeKind("Presto")
                        .transportType(
                            velox::core::ExchangeNode::TransportType::kUcx)
                        .build();

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

  // Recurse into children.
  std::vector<velox::core::PlanNodePtr> newSources;
  newSources.reserve(sources.size());
  bool anyChanged = false;
  for (const auto& source : sources) {
    auto newSource = replaceValueStreamWithExchange(
        source, targetNodeId, exchangeNodeId, producerWireType);
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

  VELOX_FAIL(
      "MppJniWrapper: unsupported plan node type '{}' (id='{}') encountered "
      "during ValueStream replacement. Add explicit Builder support for this "
      "node type in replaceValueStreamWithExchange().",
      node->name(),
      node->id());
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
    jobjectArray splitInfosPerFragArr,
    jobjectArray broadcastSlotIndicesPerFragArr,
    jobjectArray broadcastIteratorsPerFragArr) {
  JNI_METHOD_START

  auto ctx = getRuntime(env, wrapper);
  auto runtime = dynamic_cast<VeloxRuntime*>(ctx);
  GLUTEN_CHECK(runtime != nullptr, "MppQuery requires VeloxRuntime");

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

  std::vector<MppFragmentSpec> fragmentSpecs;
  fragmentSpecs.reserve(numFragments);

  // Per-fragment producer wire schema: captured right after substrait->velox
  // conversion (before any plan rewriting), looked up by consumer fragments
  // when building the inbound ExchangeNode + strip-prefix ProjectNode.
  // Fragments are emitted in topological order (producers before consumers)
  // so the producer's entry is always populated before the consumer reads it.
  std::unordered_map<int, velox::RowTypePtr> producerWireTypes;

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

    // Convert Substrait -> Velox PlanNode.
    // Merge backend config + runtime session config. Backend config has static
    // settings; session config has per-query overrides (e.g., cudf=true).
    auto backendConf = VeloxBackend::get()->getBackendConf();
    auto mergedMap = backendConf->rawConfigsCopy();
    for (const auto& [key, val] : runtime->getConfMap()) {
      mergedMap[key] = val;
    }
    auto sessionCfg = std::make_shared<velox::config::ConfigBase>(
        std::move(mergedMap));
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
        /*writeFilesTempPath=*/std::nullopt,
        /*writeFileName=*/std::nullopt,
        /*validationMode=*/false);

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

    // --- Problem 2: Replace ValueStream nodes with ExchangeNode ---
    //
    // Consumer fragments (those that receive data from a producer via an
    // exchange) will have ValueStream leaf nodes (from Gluten's
    // InputIteratorTransformer → ReadRel "iterator:N"). For MPP execution
    // these must be replaced with Velox ExchangeNode so that the
    // ExchangeClient + RemoteConnectorSplit mechanism can wire them to the
    // producer task's OutputBufferManager.
    //
    // Collect exchanges targeting this fragment as consumer, ordered by ID.
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
        veloxPlanNode = replaceValueStreamWithExchange(
            veloxPlanNode,
            valueStreamNodes[j]->id(),
            inboundExchanges[k]->exchangeNodeId,
            producerWire);
      }

      LOG(INFO) << "MppJniWrapper: fragment " << i
                << " after ValueStream->Exchange replacement: "
                << veloxPlanNode->toString(
                       /*detailed=*/true, /*recursive=*/true);
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
      //   2. Root fragment (outboundExchange == nullptr, output goes to
      //      the coordinator) -> keep the default kHttp transport so
      //      OutputBufferManager receives pages; MppQueryCoordinator
      //      polls OBM for the final result.
      const auto transportType = (outboundExchange != nullptr)
          ? velox::core::PartitionedOutputNode::TransportType::kUcx
          : velox::core::PartitionedOutputNode::TransportType::kHttp;
      wrappedPlan = velox::core::PartitionedOutputNode::single(
          outputNodeId,
          veloxPlanNode->outputType(),
          /*serdeKind=*/"Presto",
          veloxPlanNode,
          transportType);
    } else {
      // Multi-partition output. Construct the PartitionFunctionSpec from the
      // outbound exchange's partitionType + partitionKeys.
      //   HASH/RANGE -> HashPartitionFunctionSpec(keys)   (RANGE reuses hash
      //                 for now since GPU range partitioning is not yet wired.
      //                 Correctness is preserved — equal keys land in the same
      //                 partition — only intra-partition ordering is lost.)
      //   ROUND_ROBIN/other -> RoundRobinPartitionFunctionSpec()
      const auto& keyIndices = outboundExchange != nullptr
          ? outboundExchange->partitionKeyIndices
          : std::vector<int32_t>{};

      velox::core::PartitionFunctionSpecPtr funcSpec;
      std::vector<velox::core::TypedExprPtr> partitionExprs;
      const auto& outputType = veloxPlanNode->outputType();
      const auto numFields = static_cast<int32_t>(outputType->size());

      if ((partitionType == "HASH" || partitionType == "RANGE") &&
          !keyIndices.empty()) {
        std::vector<velox::column_index_t> keyChannels;
        keyChannels.reserve(keyIndices.size());
        for (auto idx : keyIndices) {
          if (idx < 0 || idx >= numFields) {
            LOG(WARNING) << "MppJniWrapper: fragment " << i
                         << " partition key index " << idx
                         << " out of range (output has " << numFields
                         << " fields); falling back to round-robin";
            keyChannels.clear();
            break;
          }
          keyChannels.push_back(static_cast<velox::column_index_t>(idx));
          partitionExprs.push_back(
              std::make_shared<velox::core::FieldAccessTypedExpr>(
                  outputType->childAt(idx), outputType->nameOf(idx)));
        }
        if (!keyChannels.empty()) {
          funcSpec = std::make_shared<velox::exec::HashPartitionFunctionSpec>(
              outputType, std::move(keyChannels));
        }
      }

      if (funcSpec == nullptr) {
        partitionExprs.clear();
        funcSpec =
            std::make_shared<velox::exec::RoundRobinPartitionFunctionSpec>();
      }

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
                   << "] func=" << (funcSpec ? funcSpec->toString() : "null")
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
          std::move(partitionExprs),
          numOutputPartitions,
          /*replicateNullsAndAny=*/false,
          std::move(funcSpec),
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
    fragSpec.id = static_cast<int32_t>(i);
    fragSpec.planFragment = std::move(planFragment);
    fragSpec.numDrivers = safeNumDrivers.elems()[i];
    fragSpec.scanInfos = std::move(fragScanInfos);
    fragSpec.scanNodeIds = std::move(fragScanNodeIds);
    // Determine connector IDs for scan nodes.
    // When cuDF is enabled, the plan may still have "test-hive" due to
    // useCudfTableHandle() logic. Override to "cudf-hive" when cudf config
    // is fully enabled, since MPP needs the GPU connector for type coercion.
    for (const auto& scanNodeId : fragSpec.scanNodeIds) {
      // For MPP, always use cudf-hive connector when GPU is enabled.
      // The plan may have "test-hive" due to SubstraitToVeloxPlanConverter
      // logic, but CudfHiveConnectorSplit needs cudf-hive connector ID.
#ifdef GLUTEN_ENABLE_GPU
      auto connectorId = std::string(kCudfHiveConnectorId);
#else
      auto connectorId = getTableScanConnectorId(veloxPlanNode, scanNodeId);
#endif
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
  int32_t poolSize = std::max(4, totalPhysicalDrivers * 2);
  LOG(WARNING) << "MppJniWrapper: threadPool size=" << poolSize
               << " physicalDrivers=" << totalPhysicalDrivers
               << " fragments=" << fragmentSpecs.size();
  auto executor =
      std::make_shared<folly::CPUThreadPoolExecutor>(poolSize);

  // Create QueryCtx. We pass nullptr for the executor in QueryCtx::create
  // since Velox tasks use the executor passed to Task::start() (the
  // coordinator calls task->start(numDrivers) which uses Task's internal
  // executor registration). The spill executor is optional.
  // Create a dedicated aggregate child pool for the MPP query.
  // QueryCtx needs an aggregate pool so it can create leaf child pools
  // for each Task's operators.
  auto rootPool = runtime->memoryManager()->getAggregateMemoryPool();
  auto mppPool = rootPool->addAggregateChild("MppQuery");
  std::unordered_map<std::string, std::shared_ptr<velox::config::ConfigBase>>
      connectorConfigs;
  // Velox's OutputBuffer.bufferedBytes_ is a single scalar shared across ALL
  // destinations of a Task; enqueue blocks when it exceeds max_*_buffer_size.
  // Default 32 MB is catastrophic at N>=16: 32MB/N per destination trips
  // backpressure on the first few pages, stalling the producer pipeline and
  // never firing noMoreData to downstream. Raise to 1 GB to give chained
  // exchanges breathing room at N up to ~200.
  std::unordered_map<std::string, std::string> queryConfigMap = {
      {velox::core::QueryConfig::kMaxOutputBufferSize, "1073741824"},
      {velox::core::QueryConfig::kMaxPartitionedOutputBufferSize, "1073741824"},
  };
  auto queryCtx = velox::core::QueryCtx::create(
      executor.get(),
      velox::core::QueryConfig{std::move(queryConfigMap)},
      connectorConfigs,
      VeloxBackend::get()->getAsyncDataCache(),
      mppPool,
      /*spillExecutor=*/nullptr,
      "MppQuery");

  // Generate a process-unique query ID. Previously we used the queryCtx
  // pointer address, but the allocator freely reuses addresses across
  // consecutive queries in the same JVM: when iter N's queryCtx is
  // destructed and iter N+1 happens to allocate at the same address, the
  // two queries end up with identical queryIds and therefore identical
  // taskIds. Stale state keyed by taskId (ExchangeClient remoteTaskIds_,
  // UcxExchangeSource registries, etc.) then bridges the two queries
  // and hangs the second one. Use a monotonically increasing counter
  // instead.
  static std::atomic<uint64_t> gMppQueryCounter{0};
  auto queryId =
      fmt::format("mpp-{}", gMppQueryCounter.fetch_add(1, std::memory_order_relaxed));

  // Create the coordinator.
  auto coordinator = MppQueryCoordinator::create(
      queryId,
      std::move(fragmentSpecs),
      std::move(exchangeSpecs),
      queryCtx,
      executor.get());

  // Bundle into a handle.
  auto handle = std::make_shared<MppQueryHandle>();
  handle->executor = std::move(executor);
  handle->queryCtx = std::move(queryCtx);
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
// nativeStartMppQuery
// ---------------------------------------------------------------------------

JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_MppQueryJniWrapper_nativeStartMppQuery( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong handle) {
  JNI_METHOD_START

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
  auto batch = std::make_shared<VeloxColumnarBatch>(rowVector);
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
