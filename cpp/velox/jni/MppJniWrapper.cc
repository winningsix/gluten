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
#include "memory/VeloxColumnarBatch.h"
#include "memory/VeloxMemoryManager.h"
#include "substrait/plan.pb.h"
#include "utils/ObjectStore.h"

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
    jbyteArray exchangeSpecsJsonArr) {
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
    // We use an empty input iterator list and no local files since MPP
    // fragments get their input from Exchange nodes, not from file scans
    // or Java iterators.
    VeloxPlanConverter converter(
        veloxPool.get(),
        VeloxBackend::get()->getBackendConf().get(),
        /*rowVectors=*/{},
        /*writeFilesTempPath=*/std::nullopt,
        /*writeFileName=*/std::nullopt,
        /*validationMode=*/false);

    std::vector<::substrait::ReadRel_LocalFiles> emptyLocalFiles;
    auto veloxPlanNode = converter.toVeloxPlan(substraitPlan, emptyLocalFiles);

    // Build PlanFragment with ungrouped execution.
    std::unordered_set<velox::core::PlanNodeId> emptyGroupedIds;
    velox::core::PlanFragment planFragment{
        veloxPlanNode,
        velox::core::ExecutionStrategy::kUngrouped,
        1, // numSplitGroups
        emptyGroupedIds};

    MppFragmentSpec fragSpec;
    fragSpec.id = static_cast<int32_t>(i);
    fragSpec.planFragment = std::move(planFragment);
    fragSpec.destination = 0; // output partition index
    fragSpec.numDrivers = safeNumDrivers.elems()[i];

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
  auto memoryPool = runtime->memoryManager()->getAggregateMemoryPool();
  std::unordered_map<std::string, std::shared_ptr<velox::config::ConfigBase>>
      connectorConfigs;
  auto queryCtx = velox::core::QueryCtx::create(
      executor.get(),
      velox::core::QueryConfig{{}},
      connectorConfigs,
      VeloxBackend::get()->getAsyncDataCache(),
      memoryPool,
      /*spillExecutor=*/nullptr,
      "MppQuery");

  // Generate a unique query ID.
  auto queryId = fmt::format("mpp-{}", reinterpret_cast<uintptr_t>(queryCtx.get()));

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
  handle->memoryPool = std::move(memoryPool);

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
