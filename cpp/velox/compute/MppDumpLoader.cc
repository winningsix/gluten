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

#include "compute/MppDumpLoader.h"

#include <fstream>
#include <sstream>

#include <fmt/format.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <glog/logging.h>

#include "compute/ProtobufUtils.h"
#include "compute/VeloxBackend.h"
#include "compute/VeloxPlanConverter.h"
#include "compute/VeloxRuntime.h"
#include "config/VeloxConfig.h"
#include "substrait/plan.pb.h"
#include "utils/Exception.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/RoundRobinPartitionFunction.h"
#include "operators/plannodes/RowVectorStream.h"
#ifdef GLUTEN_ENABLE_GPU
#include "operators/plannodes/CudfVectorStream.h"
// IBM-baseline velox dropped velox/experimental/cudf/exchange/. The
// runtime swap to UcxExchange / UcxPartitionedOutput is performed by
// IBM cudf's OperatorAdapters when transportType=kUcx, so this file
// emits plain velox::core::ExchangeNode / PartitionedOutputNode.
#endif

namespace fs = std::filesystem;

using namespace facebook;

namespace gluten {

namespace {

// ---------------------------------------------------------------------------
// Helpers duplicated from MppJniWrapper.cc so the standalone runner doesn't
// depend on JNI code. Keep behaviour byte-identical; if the JNI versions
// evolve, update both.
// ---------------------------------------------------------------------------

std::string readFileToString(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw GlutenException(
        fmt::format("MppDumpLoader: failed to open '{}'", path.string()));
  }
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool isValueStreamNode(const velox::core::PlanNodePtr& node) {
#ifdef GLUTEN_ENABLE_GPU
  if (std::dynamic_pointer_cast<const CudfValueStreamNode>(node) != nullptr) {
    return true;
  }
#endif
  if (auto tableScan =
          std::dynamic_pointer_cast<const velox::core::TableScanNode>(node)) {
    if (tableScan->tableHandle() &&
        tableScan->tableHandle()->connectorId() == kIteratorConnectorId) {
      return true;
    }
  }
  return false;
}

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

velox::core::PlanNodePtr replaceValueStreamWithExchange(
    const velox::core::PlanNodePtr& node,
    const std::string& targetNodeId,
    const std::string& exchangeNodeId) {
  if (isValueStreamNode(node) && node->id() == targetNodeId) {
    LOG(INFO) << "MppDumpLoader: replacing ValueStream node '" << node->id()
              << "' with ExchangeNode '" << exchangeNodeId
              << "' outputType=" << node->outputType()->toString();
    // Tag the consumer-side ExchangeNode with TransportType::kUcx so IBM
    // cudf's ExchangeAdapter swaps it to UcxExchange at runtime. Without
    // this tag the adapter declines (canRunOnGPU requires kUcx) and the
    // node stays as plain CPU Exchange[Presto]; meanwhile MppQueryCoordinator
    // injects UCX-formatted URLs as splits, and Velox's stock
    // ExchangeSource::create finds no factory matching them — every
    // consumer fragment fails on first getSplits() with INVALID_STATE.
    return std::make_shared<velox::core::ExchangeNode>(
        exchangeNodeId,
        node->outputType(),
        std::string{"Presto"},
        velox::core::ExchangeNode::TransportType::kUcx);
  }

  const auto& sources = node->sources();
  if (sources.empty()) {
    return node;
  }

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
  if (!anyChanged) {
    return node;
  }

#define REBUILD_SINGLE(NodeT)                                                 \
  if (auto n = std::dynamic_pointer_cast<const velox::core::NodeT>(node)) {   \
    return velox::core::NodeT::Builder(*n).source(newSources[0]).build();     \
  }
#define REBUILD_BINARY(NodeT)                                                 \
  if (auto n = std::dynamic_pointer_cast<const velox::core::NodeT>(node)) {   \
    return velox::core::NodeT::Builder(*n)                                    \
        .left(newSources[0])                                                  \
        .right(newSources[1])                                                 \
        .build();                                                             \
  }

  REBUILD_SINGLE(FilterNode)
  REBUILD_SINGLE(ProjectNode)
  REBUILD_SINGLE(AggregationNode)
  REBUILD_SINGLE(OrderByNode)
  REBUILD_SINGLE(TopNNode)
  REBUILD_SINGLE(LimitNode)
  REBUILD_BINARY(HashJoinNode)
  REBUILD_BINARY(MergeJoinNode)
  REBUILD_BINARY(NestedLoopJoinNode)
  REBUILD_SINGLE(ExpandNode)
  REBUILD_SINGLE(RowNumberNode)
  REBUILD_SINGLE(TopNRowNumberNode)
  REBUILD_SINGLE(WindowNode)
  REBUILD_SINGLE(MarkDistinctNode)
  REBUILD_SINGLE(EnforceSingleRowNode)
  REBUILD_SINGLE(GroupIdNode)
  REBUILD_SINGLE(UnnestNode)
#undef REBUILD_SINGLE
#undef REBUILD_BINARY

  VELOX_FAIL(
      "MppDumpLoader: unsupported plan node type '{}' (id='{}') encountered "
      "during ValueStream replacement. Add explicit Builder support for this "
      "node type in replaceValueStreamWithExchange().",
      node->name(),
      node->id());
}

std::vector<MppExchangeSpec> parseExchangeSpecs(const std::string& json) {
  auto parsed = folly::parseJson(json);
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
    spec.partitionType = item.count("exchangeType")
        ? item["exchangeType"].asString()
        : "ROUND_ROBIN";
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

// Discover per-fragment split files from manifest.json. Fragment entries are
// ordered by id (matches the on-disk fragment-<id>.pb numbering).
std::vector<std::vector<std::string>> readSplitFilesFromManifest(
    const folly::dynamic& manifest) {
  VELOX_CHECK(manifest.isObject(), "manifest.json must be a JSON object");
  VELOX_CHECK(manifest.count("fragments"), "manifest.json missing 'fragments'");
  const auto& frags = manifest["fragments"];
  VELOX_CHECK(frags.isArray(), "manifest.json 'fragments' must be an array");

  // Index by fragment id (not array order), since the manifest may emit in
  // a different order than the ids suggest.
  size_t maxId = 0;
  for (const auto& f : frags) {
    maxId = std::max(maxId, static_cast<size_t>(f["id"].asInt()));
  }
  std::vector<std::vector<std::string>> out(maxId + 1);
  for (const auto& f : frags) {
    size_t id = f["id"].asInt();
    if (f.count("splitFiles") && f["splitFiles"].isArray()) {
      for (const auto& sf : f["splitFiles"]) {
        out[id].push_back(sf.asString());
      }
    }
  }
  return out;
}

} // namespace

MppDumpLoadResult loadMppQueryFromDump(
    const fs::path& queryDir,
    velox::memory::MemoryPool* veloxPool,
    const std::unordered_map<std::string, std::string>& sessionConf) {
  VELOX_CHECK(fs::is_directory(queryDir),
              "MppDumpLoader: not a directory: {}", queryDir.string());

  // 1. Read manifest.json + exchange-specs.json (both are small text files).
  auto manifestStr = readFileToString(queryDir / "manifest.json");
  auto manifest = folly::parseJson(manifestStr);

  auto exchangeJson = readFileToString(queryDir / "exchange-specs.json");
  auto exchangeSpecs = parseExchangeSpecs(exchangeJson);

  auto splitFilesByFrag = readSplitFilesFromManifest(manifest);

  // Precompute consumer replica counts from non-broadcast inbound exchanges.
  // BROADCAST inbound carries numPartitions=1; it must not drive consumer
  // parallelism (see plan/issue-broadcast-fanout.md).
  // We use this to size broadcast producers' PartitionedOutputNode fan-out below.
  const size_t numFragmentsEarly =
      std::max<size_t>(splitFilesByFrag.size(), 1);
  std::vector<int32_t> consumerReplicas(numFragmentsEarly, 1);
  for (const auto& ex : exchangeSpecs) {
    if (ex.partitionType == "BROADCAST") {
      continue;
    }
    const auto n = std::max(1, ex.numPartitions);
    if (ex.consumerFragmentId >= 0 &&
        static_cast<size_t>(ex.consumerFragmentId) < consumerReplicas.size()) {
      auto& slot = consumerReplicas[ex.consumerFragmentId];
      if (slot == 1) {
        slot = n;
      }
    }
  }

  // Fragment count is the number of fragment-*.pb files (or manifest entries).
  size_t numFragments = splitFilesByFrag.size();
  // If manifest has fragments but no splitFiles, splitFilesByFrag sizing still
  // reflects fragment count (we pre-sized by max id). Use fragments.size() as
  // a cross-check.
  VELOX_CHECK(
      manifest.count("numFragments"),
      "manifest.json missing 'numFragments'");
  auto manifestNumFragments =
      static_cast<size_t>(manifest["numFragments"].asInt());
  VELOX_CHECK_GE(
      manifestNumFragments,
      numFragments,
      "manifest fragment count mismatch (numFragments={}, max id + 1={})",
      manifestNumFragments,
      numFragments);
  numFragments = manifestNumFragments;

  // 2. Build session config the same way nativeCreateMppQuery does: backend
  //    config first, then session overrides on top.
  auto backendConf = VeloxBackend::get()->getBackendConf();
  auto mergedMap = backendConf->rawConfigsCopy();
  for (const auto& [k, v] : sessionConf) {
    mergedMap[k] = v;
  }
  auto sessionCfg = std::make_shared<velox::config::ConfigBase>(
      std::move(mergedMap));

  // 3. For each fragment build an MppFragmentSpec.
  std::vector<MppFragmentSpec> fragmentSpecs;
  fragmentSpecs.reserve(numFragments);

  for (size_t i = 0; i < numFragments; ++i) {
    // 3a. Parse substrait::Plan from fragment-<i>.pb.
    auto planPath = queryDir / fmt::format("fragment-{}.pb", i);
    VELOX_CHECK(
        fs::exists(planPath),
        "MppDumpLoader: missing {}",
        planPath.string());
    auto planStr = readFileToString(planPath);
    ::substrait::Plan substraitPlan;
    GLUTEN_CHECK(
        parseProtobuf(
            reinterpret_cast<const uint8_t*>(planStr.data()),
            static_cast<int32_t>(planStr.size()),
            &substraitPlan),
        fmt::format(
            "MppDumpLoader: failed to parse Substrait plan for fragment {}",
            i));

    // 3b. Parse per-leaf split protobufs (may be empty for consumer fragments).
    std::vector<::substrait::ReadRel_LocalFiles> localFiles;
    if (i < splitFilesByFrag.size()) {
      for (const auto& name : splitFilesByFrag[i]) {
        auto splitPath = queryDir / name;
        VELOX_CHECK(
            fs::exists(splitPath),
            "MppDumpLoader: missing {}",
            splitPath.string());
        auto splitStr = readFileToString(splitPath);
        ::substrait::ReadRel_LocalFiles localFile;
        GLUTEN_CHECK(
            parseProtobuf(
                reinterpret_cast<const uint8_t*>(splitStr.data()),
                static_cast<int32_t>(splitStr.size()),
                &localFile),
            fmt::format(
                "MppDumpLoader: failed to parse split '{}' for fragment {}",
                name,
                i));
        LOG(INFO) << "MppDumpLoader: fragment " << i << " split '" << name
                  << "' has " << localFile.items_size() << " file items";
        localFiles.push_back(std::move(localFile));
      }
    }

    // 3c. Build VeloxPlanConverter with one placeholder iterator per inbound
    //     exchange (for ValueStream construction; we replace those nodes below).
    int32_t numStreamInputs = 0;
    for (const auto& ex : exchangeSpecs) {
      if (ex.consumerFragmentId == static_cast<int32_t>(i)) {
        numStreamInputs++;
      }
    }
    std::vector<std::shared_ptr<ResultIterator>> placeholderIters(
        numStreamInputs, nullptr);

    VeloxPlanConverter converter(
        veloxPool,
        sessionCfg.get(),
        placeholderIters,
        /*writeFilesTempPath=*/std::nullopt,
        /*writeFileName=*/std::nullopt,
        /*validationMode=*/false);

    auto veloxPlanNode = converter.toVeloxPlan(substraitPlan, localFiles);

    // 3d. Extract scan info BEFORE tree rewriting.
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
    }
    LOG(INFO) << "MppDumpLoader: fragment " << i << " raw Velox plan:\n"
              << veloxPlanNode->toString(
                     /*detailed=*/true, /*recursive=*/true);

    // 3e. Replace each ValueStream leaf with an ExchangeNode (consumer side).
    std::vector<const MppExchangeSpec*> inboundExchanges;
    for (const auto& ex : exchangeSpecs) {
      if (ex.consumerFragmentId == static_cast<int32_t>(i)) {
        inboundExchanges.push_back(&ex);
      }
    }
    if (!inboundExchanges.empty()) {
      std::vector<velox::core::PlanNodePtr> valueStreamNodes;
      collectValueStreamNodes(veloxPlanNode, valueStreamNodes);
      VELOX_CHECK_EQ(
          valueStreamNodes.size(),
          inboundExchanges.size(),
          "Fragment {} has {} ValueStream nodes but {} inbound exchanges.",
          i,
          valueStreamNodes.size(),
          inboundExchanges.size());
      for (size_t j = 0; j < valueStreamNodes.size(); ++j) {
        veloxPlanNode = replaceValueStreamWithExchange(
            veloxPlanNode,
            valueStreamNodes[j]->id(),
            inboundExchanges[j]->exchangeNodeId);
      }
    }

    // 3f. Wrap with PartitionedOutputNode. Single partition for root or
    //     single-consumer; otherwise partitioned per the outbound exchange.
    int32_t numOutputPartitions = 1;
    const MppExchangeSpec* outboundExchange = nullptr;
    for (const auto& ex : exchangeSpecs) {
      if (ex.producerFragmentId == static_cast<int32_t>(i)) {
        numOutputPartitions = ex.numPartitions;
        outboundExchange = &ex;
        break;
      }
    }

    auto outputNodeId = fmt::format("mpp_output_{}", i);
    velox::core::PlanNodePtr wrappedPlan;

    if (outboundExchange != nullptr &&
        outboundExchange->partitionType == "BROADCAST") {
      // BROADCAST edge: one producer payload must reach every consumer
      // replica. Emit a kBroadcast PartitionedOutputNode with fanout=N so
      // Velox's OutputBuffer allocates N DestinationBuffers and replicates
      // the shared payload (via enqueueBroadcastOutputLocked).
      //
      // We DO NOT use PartitionedOutputNode::broadcast(id, N): that factory
      // attaches a GatherPartitionFunctionSpec whose create() is
      // VELOX_UNREACHABLE. The CPU exec::PartitionedOutput constructor calls
      // spec.create(N) when N>1 (PartitionedOutput.cpp:172-174), which blows
      // up before our adapter can swap in GpuPartitionedOutput. Work around
      // by attaching a harmless RoundRobinPartitionFunctionSpec that the
      // kBroadcast code path never actually invokes (broadcast bypasses the
      // partition function -- see OutputBuffer::enqueueBroadcastOutputLocked).
      //
      // See plan/issue-broadcast-fanout.md; the coordinator calls
      // updateOutputBuffers(N, noMore=true) after wiring so
      // isFinishedLocked() can terminate.
      const int32_t fanout =
          (outboundExchange->consumerFragmentId >= 0 &&
           static_cast<size_t>(outboundExchange->consumerFragmentId) <
               consumerReplicas.size())
          ? consumerReplicas[outboundExchange->consumerFragmentId]
          : 1;
      // numPartitions must be 1 at plan construction for kBroadcast --
      // Velox's PartitionedOutput ctor asserts VELOX_USER_CHECK_EQ(1,
      // numDestinations_) when !isPartitioned() (PartitionedOutput.cpp:200-201).
      // The actual fan-out to N consumer replicas is set later by
      // MppQueryCoordinator calling task->updateOutputBuffers(fanout, true)
      // after Phase-2 wiring. The `fanout` value here is kept just for
      // bookkeeping / future reference; it is not passed to the plan node.
      (void)fanout;
      // Broadcast producer is always non-root (it feeds a consumer
      // fragment). Tag with kUcx so IBM cudf's PartitionedOutputAdapter
      // swaps to UcxPartitionedOutput.
      wrappedPlan = std::make_shared<velox::core::PartitionedOutputNode>(
          outputNodeId,
          velox::core::PartitionedOutputNode::Kind::kBroadcast,
          std::vector<velox::core::TypedExprPtr>{},
          /*numPartitions=*/1,
          /*replicateNullsAndAny=*/false,
          std::make_shared<velox::exec::RoundRobinPartitionFunctionSpec>(),
          veloxPlanNode->outputType(),
          std::string{"Presto"},
          veloxPlanNode,
          velox::core::PartitionedOutputNode::TransportType::kUcx);
    } else if (numOutputPartitions == 1) {
      // Single-partition gather. Two cases (mirrors MppJniWrapper.cc:646-665):
      //   - producer with SINGLE-gather outbound exchange (outboundExchange!=nullptr)
      //     → kUcx so the consumer's UcxExchange picks it up via
      //     IntraNodeTransferRegistry.
      //   - root fragment (outboundExchange==nullptr, output goes to the
      //     coordinator) → kHttp so OutputBufferManager receives pages.
      const auto transportType = (outboundExchange != nullptr)
          ? velox::core::PartitionedOutputNode::TransportType::kUcx
          : velox::core::PartitionedOutputNode::TransportType::kHttp;
      wrappedPlan = velox::core::PartitionedOutputNode::single(
          outputNodeId,
          veloxPlanNode->outputType(),
          std::string{"Presto"},
          veloxPlanNode,
          transportType);
    } else {
      const std::string& partitionType = outboundExchange
          ? outboundExchange->partitionType
          : std::string("ROUND_ROBIN");
      const auto& keyIndices = outboundExchange
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

      // Multi-partition output is always producer-side; tag with kUcx so
      // IBM cudf's PartitionedOutputAdapter swaps to UcxPartitionedOutput.
      // Without the tag the node stays as plain CPU PartitionedOutput[Presto]
      // which doesn't speak the UCX URL format MppQueryCoordinator wires.
      wrappedPlan = std::make_shared<velox::core::PartitionedOutputNode>(
          outputNodeId,
          velox::core::PartitionedOutputNode::Kind::kPartitioned,
          std::move(partitionExprs),
          numOutputPartitions,
          /*replicateNullsAndAny=*/false,
          std::move(funcSpec),
          veloxPlanNode->outputType(),
          std::string{"Presto"},
          veloxPlanNode,
          velox::core::PartitionedOutputNode::TransportType::kUcx);
    }

    // 3g. Package into MppFragmentSpec.
    std::unordered_set<velox::core::PlanNodeId> emptyGroupedIds;
    velox::core::PlanFragment planFragment{
        wrappedPlan,
        velox::core::ExecutionStrategy::kUngrouped,
        1,
        emptyGroupedIds};

    MppFragmentSpec spec;
    spec.id = static_cast<int32_t>(i);
    spec.planFragment = std::move(planFragment);
    // Parallelism: prefer the per-fragment parallelism from manifest.json
    // (matches Gluten's numDriversPerFragment input to JNI).
    int32_t parallelism = 1;
    for (const auto& f : manifest["fragments"]) {
      if (static_cast<size_t>(f["id"].asInt()) == i) {
        parallelism = static_cast<int32_t>(f["parallelism"].asInt());
        break;
      }
    }
    spec.numDrivers = parallelism;
    spec.scanInfos = std::move(fragScanInfos);
    spec.scanNodeIds = std::move(fragScanNodeIds);

    for (const auto& scanNodeId : spec.scanNodeIds) {
#ifdef GLUTEN_ENABLE_GPU
      auto connectorId = std::string(kCudfHiveConnectorId);
#else
      auto connectorId = getTableScanConnectorId(veloxPlanNode, scanNodeId);
#endif
      spec.scanConnectorIds.push_back(std::move(connectorId));
    }

    fragmentSpecs.push_back(std::move(spec));
  }

  MppDumpLoadResult out;
  out.fragments = std::move(fragmentSpecs);
  out.exchanges = std::move(exchangeSpecs);
  return out;
}

} // namespace gluten
