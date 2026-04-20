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

#include <fmt/format.h>
#include <glog/logging.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

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
#include "velox/experimental/cudf/exchange/LocalGpuExchangeSource.h"
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
    const std::string& exchangeNodeId) {
  // Base case: this IS the target ValueStream leaf — replace it.
  if (isValueStreamNode(node) && node->id() == targetNodeId) {
    LOG(INFO) << "MppJniWrapper: replacing ValueStream node '"
              << node->id() << "' with ExchangeNode '"
              << exchangeNodeId << "' outputType="
              << node->outputType()->toString();
    return std::make_shared<velox::core::ExchangeNode>(
        exchangeNodeId,
        node->outputType(),
        velox::VectorSerde::Kind::kPresto);
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
    auto newSource =
        replaceValueStreamWithExchange(source, targetNodeId, exchangeNodeId);
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
    jobjectArray splitInfosPerFragArr) {
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

  // --- Register GPU ExchangeSource for intra-process exchange ---
  // MPP uses Velox's ExchangeNode/OutputBufferManager for streaming data
  // between fragments. The "gpu-local://" prefix is handled by
  // LocalGpuExchangeSource from the cudf exchange module.
#ifdef GLUTEN_ENABLE_GPU
  static std::once_flag gpuExchangeRegistered;
  std::call_once(gpuExchangeRegistered, []() {
    velox::exec::ExchangeSource::registerFactory(
        facebook::velox::cudf_velox::createLocalGpuExchangeSource);
    LOG(INFO) << "MppJniWrapper: registered LocalGpuExchangeSource factory";
  });
#endif

  // --- Convert each Substrait plan to a Velox PlanNode ---

  auto veloxPool = defaultLeafVeloxMemoryPool();

  std::vector<MppFragmentSpec> fragmentSpecs;
  fragmentSpecs.reserve(numFragments);

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
    // Count inbound exchanges for this fragment (= number of stream inputs).
    // Consumer fragments have ValueStream nodes that need input iterators
    // during plan conversion. We provide nullptr placeholders since these
    // nodes get replaced with ExchangeNode after conversion.
    int32_t numStreamInputs = 0;
    for (const auto& exchange : exchangeSpecs) {
      if (exchange.consumerFragmentId == static_cast<int32_t>(i)) {
        numStreamInputs++;
      }
    }
    std::vector<std::shared_ptr<ResultIterator>> placeholderIters(
        numStreamInputs, nullptr);

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

    if (!inboundExchanges.empty()) {
      // Collect ValueStream leaf nodes in depth-first order.
      // The order matches SubstraitToVeloxPlanConverter's stream index
      // assignment (iterator:0, iterator:1, ...).
      std::vector<velox::core::PlanNodePtr> valueStreamNodes;
      collectValueStreamNodes(veloxPlanNode, valueStreamNodes);

      VELOX_CHECK_EQ(
          valueStreamNodes.size(),
          inboundExchanges.size(),
          "Fragment {} has {} ValueStream nodes but {} inbound exchanges. "
          "These must match 1:1.",
          i,
          valueStreamNodes.size(),
          inboundExchanges.size());

      // Replace each ValueStream node with the corresponding ExchangeNode.
      // Match by position: first ValueStream (stream 0) -> first exchange, etc.
      for (size_t j = 0; j < valueStreamNodes.size(); ++j) {
        veloxPlanNode = replaceValueStreamWithExchange(
            veloxPlanNode,
            valueStreamNodes[j]->id(),
            inboundExchanges[j]->exchangeNodeId);
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

    if (numOutputPartitions == 1) {
      // Single-partition gather output (root fragment or single-consumer).
      wrappedPlan = velox::core::PartitionedOutputNode::single(
          outputNodeId,
          veloxPlanNode->outputType(),
          velox::VectorSerde::Kind::kPresto,
          veloxPlanNode);
    } else {
      // Multi-partition output. Construct the PartitionFunctionSpec from the
      // outbound exchange's partitionType + partitionKeys.
      //   HASH/RANGE -> HashPartitionFunctionSpec(keys)   (RANGE reuses hash
      //                 for now since GPU range partitioning is not yet wired.
      //                 Correctness is preserved — equal keys land in the same
      //                 partition — only intra-partition ordering is lost.)
      //   ROUND_ROBIN/other -> RoundRobinPartitionFunctionSpec()
      const std::string& partitionType =
          outboundExchange != nullptr ? outboundExchange->partitionType
                                      : std::string("ROUND_ROBIN");
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

      LOG(INFO) << "MppJniWrapper: fragment " << i
                << " outbound exchange type=" << partitionType
                << " keyIndices=" << keyIndices.size()
                << " func=" << (funcSpec ? funcSpec->toString() : "null");

      wrappedPlan = std::make_shared<velox::core::PartitionedOutputNode>(
          outputNodeId,
          velox::core::PartitionedOutputNode::Kind::kPartitioned,
          std::move(partitionExprs),
          numOutputPartitions,
          /*replicateNullsAndAny=*/false,
          std::move(funcSpec),
          veloxPlanNode->outputType(),
          velox::VectorSerde::Kind::kPresto,
          veloxPlanNode);
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

  // Thread pool for task execution. Size proportional to total drivers.
  int32_t totalDrivers = 0;
  for (auto& spec : fragmentSpecs) {
    totalDrivers += spec.numDrivers;
  }
  // At least 4 threads, at most 32, with some headroom for exchange I/O.
  int32_t poolSize = std::max(4, std::min(32, totalDrivers * 2));
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
  auto queryCtx = velox::core::QueryCtx::create(
      executor.get(),
      velox::core::QueryConfig{{}},
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
  // LocalGpuExchangeSource timeouts_, etc.) then bridges the two queries
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
// nativeCloseMppQuery
// ---------------------------------------------------------------------------

JNIEXPORT void JNICALL
Java_org_apache_gluten_vectorized_MppQueryJniWrapper_nativeCloseMppQuery( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jlong handle) {
  JNI_METHOD_START

  LOG(INFO) << "MppJniWrapper: closing MPP query, handle=" << handle;

  // Release from the ObjectStore. This drops the shared_ptr<MppQueryHandle>,
  // which triggers ~MppQueryHandle -> coordinator->abort() -> cleanup.
  ObjectStore::release(handle);

  JNI_METHOD_END()
}

#ifdef __cplusplus
}
#endif
