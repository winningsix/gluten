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

// Standalone C++ runner that replays Gluten-dumped multi-fragment Substrait
// plans directly on Velox-cudf via MppQueryCoordinator.
//
// Goal: fast iteration loop for debugging TPC-H correctness on the MPP path.
// Compile-run-see-result in seconds instead of minutes through the full
// Spark/Gluten/JNI stack.
//
// This is a SKELETON — first-pass focus is:
//   1. CMake + linker integration so the binary builds.
//   2. Compile against the existing MppQueryCoordinator API as-is.
//   3. Run a hand-constructed two-fragment plan end-to-end.
//   4. Leave clear TODOs for the dump-format parser (blocked on Task #3).
//
// Usage:
//   mpp-substrait-runner --dump-dir=<path> [--query=<name>] [--verify=<path>]
//   mpp-substrait-runner --builtin-smoke   (run the hand-constructed test)

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <folly/json.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "compute/MppDumpLoader.h"
#include "compute/MppQueryCoordinator.h"
#include "compute/VeloxBackend.h"
#include "config/VeloxConfig.h"
#include "memory/AllocationListener.h"

#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/core/PlanFragment.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/ExchangeSource.h"
#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/Task.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorEncoding.h"
#include "velox/vector/VectorStream.h"

// GPU exchange bits. The harness is intentionally coupled to the GPU build
// (ENABLE_GPU=ON) since that's the path we're iterating on; we add a soft
// fallback so a non-GPU build still links.
#ifdef GLUTEN_ENABLE_GPU
#include "velox/experimental/cudf/exchange/LocalGpuExchangeSource.h"
#endif

namespace fs = std::filesystem;
using namespace facebook::velox;

DEFINE_string(
    dump_dir,
    "",
    "Root directory containing one sub-directory per dumped query, each with "
    "fragment-{id}.pb files and a manifest.json describing exchanges.");
DEFINE_string(
    query,
    "",
    "Run a single query by directory name. If omitted, iterate all "
    "sub-directories under --dump-dir.");
DEFINE_string(
    verify,
    "",
    "Optional path to expected.csv for row-count sanity check.");
DEFINE_string(log_level, "WARNING", "glog log level (WARNING, INFO, ...)");
DEFINE_bool(
    builtin_smoke,
    false,
    "Run the hand-constructed two-fragment smoke test (no dump-dir needed). "
    "Used while the dump format (Task #3) is not yet available.");
DEFINE_bool(
    plan_only,
    false,
    "Parse dump and build fragment + exchange specs, print a summary, then "
    "exit. Skips QueryCtx/coordinator/GPU execution. Use to sweep all 22 "
    "dumps quickly and catch plan-construction bugs without waiting on data "
    "flow.");
DEFINE_int32(
    force_drivers,
    1,
    "Override numDrivers on every fragment after loading. Use 1 while "
    "iterating single-GPU on SF1000 to avoid OOM from 16x parallel scans. "
    "Set <=0 to honor the manifest's parallelism value.");

namespace {

// ---------------------------------------------------------------------------
// Per-query summary row for the final pass/fail table.
// ---------------------------------------------------------------------------
struct QueryResult {
  std::string name;
  bool passed{false};
  double wallClockMs{0.0};
  int64_t numPages{0};
  int64_t numRows{0};
  int64_t numBytes{0};
  std::string error;
};

// ---------------------------------------------------------------------------
// CLI usage banner.
// ---------------------------------------------------------------------------
void printUsage() {
  std::cerr << R"(mpp-substrait-runner

Replays Gluten-dumped multi-fragment Substrait plans on Velox-cudf via
MppQueryCoordinator, bypassing Spark/Gluten/JNI entirely.

Usage:
  mpp-substrait-runner --dump-dir=<path> [--query=<name>] [--verify=<path>]
  mpp-substrait-runner --builtin-smoke

Flags:
  --dump-dir      Root directory with per-query sub-directories. Each
                  sub-directory contains fragment-{id}.pb and manifest.json.
  --query         Run a single query by directory name (e.g. q01).
  --verify        Optional path to expected.csv for row-count sanity check.
  --log_level     glog level (default: WARNING).
  --builtin-smoke Run the hand-constructed two-fragment smoke test.

Examples:
  mpp-substrait-runner --builtin-smoke
  mpp-substrait-runner --dump-dir=/tmp/gluten-dumps --query=q06
)";
}

// ---------------------------------------------------------------------------
// Default session config for dump replay.
//
// Mirrors the relevant cuDF-related keys from scripts/mpp-gpu.template so the
// replay path picks the same code paths as the live Spark run. Keep this small
// -- only what affects plan conversion / operator selection. Runtime tuning
// knobs (buffer sizes, thread pool sizing) are set in QueryCtx below.
// ---------------------------------------------------------------------------
std::unordered_map<std::string, std::string> defaultSessionConf() {
  return {
      {"spark.gluten.sql.columnar.cudf", "true"},
      {"spark.gluten.sql.columnar.backend.velox.cudf.enabled", "true"},
      {"spark.gluten.sql.columnar.backend.velox.cudf.enableTableScan", "true"},
      {"spark.gluten.debug.enabled.cudf", "true"},
      // CudfHiveDataSource's async Parquet reads need a non-null IO executor,
      // otherwise device_read_async FATALs with INVALID_STATE. The Spark side
      // has this at 0 by default too but workers get their own threadpool
      // somewhere; for the runner we must size it explicitly.
      {"spark.gluten.sql.columnar.backend.velox.IOThreads", "16"},
      // GPU memory tuning. Values copied from scripts/mpp-gpu.template so the
      // runner and Spark path use the same knobs. Defaults in ToCudf.cpp are
      // way too aggressive (2GB target batch x 16 drivers x 122 splits => OOM).
      {"spark.gluten.sql.columnar.backend.velox.cudf.gpuTargetBatchBytes",
       "1073741824"},
      {"spark.gluten.sql.columnar.backend.velox.cudf.concurrentGpuTasks", "4"},
      {"spark.gluten.sql.columnar.backend.velox.cudf.memoryPercent", "50"},
      {"spark.gluten.sql.columnar.backend.velox.cudf.pinnedPoolSize",
       "21474836480"},
      {"spark.gluten.sql.columnar.backend.velox.cudf.hostAsPinnedThreshold",
       "1099511627776"},
      {"spark.gluten.sql.columnar.backend.velox.cudf.gpuSemaphore.enabled",
       "true"},
      {"spark.gluten.sql.columnar.backend.velox.cudf.concurrentGpuTasks.dynamic",
       "false"},
      // Align with Presto GPU (rapidsai/velox-testing etc_worker/config_native.properties):
      // JIT takes a few iterations to warm up -- useless for single-run replay
      // and adds first-iter latency. Presto GPU disables it in the benchmark
      // harness for the same reason.
      {"spark.gluten.sql.columnar.backend.velox.cudf.jit_expression_enabled",
       "false"},
  };
}

// ---------------------------------------------------------------------------
// Hand-constructed smoke test: a single-fragment "pass-through" plan.
//
// This is deliberately simpler than the real target: one fragment with a
// Values → PartitionedOutput(1) layout. It exercises:
//   * ExchangeSource factory registration.
//   * MemoryPool / QueryCtx construction.
//   * MppQueryCoordinator::create/start/next/waitForCompletion.
//   * Root output buffer drain path.
//
// A proper two-fragment test (Values → PartitionedOutput → Exchange →
// PartitionedOutput) requires ValueStream→Exchange plumbing that currently
// lives in MppJniWrapper. Rather than duplicate that plumbing here, we
// prove the coordinator wiring with a single-fragment case and leave the
// multi-fragment case to come from the real Gluten dump once Task #3 lands.
// ---------------------------------------------------------------------------
core::PlanNodePtr makeSingleFragmentPlan(memory::MemoryPool* pool) {
  // Build a tiny RowVector with two INTEGER columns, 3 rows.
  auto rowType = ROW({"a", "b"}, {INTEGER(), INTEGER()});
  auto a = BaseVector::create<FlatVector<int32_t>>(INTEGER(), 3, pool);
  auto b = BaseVector::create<FlatVector<int32_t>>(INTEGER(), 3, pool);
  a->set(0, 1);
  a->set(1, 2);
  a->set(2, 3);
  b->set(0, 10);
  b->set(1, 20);
  b->set(2, 30);
  auto row = std::make_shared<RowVector>(
      pool, rowType, nullptr /*nulls*/, 3, std::vector<VectorPtr>{a, b});

  // ValuesNode → PartitionedOutputNode(kPartitioned=single partition).
  auto values = std::make_shared<core::ValuesNode>(
      "values-0", std::vector<RowVectorPtr>{row});

  // Single-destination gather; coordinator reads from destination 0.
  auto partitionedOut = core::PartitionedOutputNode::single(
      "out-0", rowType, VectorSerde::Kind::kPresto, values);

  return partitionedOut;
}

QueryResult runBuiltinSmoke(
    folly::Executor* executor,
    const std::shared_ptr<memory::MemoryPool>& rootPool) {
  QueryResult r;
  r.name = "builtin-smoke";
  auto start = std::chrono::steady_clock::now();

  try {
    // Leaf pool for building the input RowVector.
    auto leafPool = rootPool->addLeafChild("smoke-builder");
    auto plan = makeSingleFragmentPlan(leafPool.get());

    // Build QueryCtx. rootPool must be an aggregate pool; the Velox
    // TaskManager will add leaf children per operator.
    std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
        connectorConfigs;
    std::unordered_map<std::string, std::string> queryConfig;
    auto queryCtx = core::QueryCtx::create(
        executor,
        core::QueryConfig{std::move(queryConfig)},
        connectorConfigs,
        /*cache=*/nullptr,
        rootPool->addAggregateChild("smoke-query"),
        /*spillExecutor=*/nullptr,
        "mpp-smoke");

    // Wrap into MppFragmentSpec.
    std::unordered_set<core::PlanNodeId> emptyGroupedIds;
    core::PlanFragment planFragment{
        plan, core::ExecutionStrategy::kUngrouped, 1, emptyGroupedIds};

    gluten::MppFragmentSpec spec;
    spec.id = 0;
    spec.planFragment = std::move(planFragment);
    spec.numDrivers = 1;

    std::vector<gluten::MppFragmentSpec> fragments;
    fragments.push_back(std::move(spec));
    std::vector<gluten::MppExchangeSpec> exchanges; // none: single fragment

    auto coord = gluten::MppQueryCoordinator::create(
        "smoke-query", std::move(fragments), std::move(exchanges), queryCtx,
        executor);
    coord->start();

    while (auto batch = coord->next()) {
      r.numPages++;
      r.numRows += batch->size();
      r.numBytes += batch->estimateFlatSize();
    }

    coord->waitForCompletion();
    r.passed = (r.numRows == 3);
    if (!r.passed) {
      r.error = fmt::format(
          "expected 3 rows, got {} (pages={})", r.numRows, r.numPages);
    }
  } catch (const std::exception& e) {
    r.error = e.what();
    r.passed = false;
  }

  auto end = std::chrono::steady_clock::now();
  r.wallClockMs =
      std::chrono::duration<double, std::milli>(end - start).count();
  return r;
}

// ---------------------------------------------------------------------------
// Run a single query loaded from the dump directory.
//
// Flow mirrors MppJniWrapper::nativeCreateMppQuery + nativeStartMppQuery +
// nativeGetMppOutput, collapsed into one synchronous pass:
//   1. gluten::loadMppQueryFromDump() parses the dump and builds specs.
//   2. Build QueryCtx on a dedicated aggregate pool + thread pool sized to
//      the physical driver count.
//   3. MppQueryCoordinator::create -> start -> drain next() -> wait.
// ---------------------------------------------------------------------------
QueryResult runQueryFromDump(
    const fs::path& queryDir,
    const std::shared_ptr<memory::MemoryPool>& rootPool) {
  QueryResult r;
  r.name = queryDir.filename().string();
  auto start = std::chrono::steady_clock::now();

  try {
    // --- 1. Parse dump + build fragment/exchange specs ---
    auto leafPool = rootPool->addLeafChild("mpp-loader-" + r.name);
    auto loaded = gluten::loadMppQueryFromDump(
        queryDir, leafPool.get(), defaultSessionConf());

    // Runtime override: clamp per-fragment numDrivers. Manifest tends to carry
    // Spark's Phase-2 parallelism (16), which at SF1000 produces 16 parallel
    // Parquet scan drivers per fragment and trivially overruns a 32 GB GPU.
    if (FLAGS_force_drivers > 0) {
      for (auto& spec : loaded.fragments) {
        spec.numDrivers = FLAGS_force_drivers;
      }
    }

    // --- Plan-only mode: summarise and exit ---
    if (FLAGS_plan_only) {
      std::cout << fmt::format(
                       "[{}] loaded {} fragments, {} exchanges",
                       r.name,
                       loaded.fragments.size(),
                       loaded.exchanges.size())
                << "\n";
      for (const auto& f : loaded.fragments) {
        std::cout << fmt::format(
                         "  fragment {} parallelism={} scans={} output={}",
                         f.id,
                         f.numDrivers,
                         f.scanNodeIds.size(),
                         f.planFragment.planNode->outputType()->toString())
                  << "\n";
        // Full Velox plan tree for this fragment (detailed=true, recursive=true).
        // Prefixed with "=== fragment N plan ===" so sweep scripts can split.
        std::cout << fmt::format("=== fragment {} plan ===", f.id) << "\n"
                  << f.planFragment.planNode->toString(true, true) << "\n";
      }
      for (const auto& ex : loaded.exchanges) {
        std::cout << fmt::format(
                         "  exchange {} F{}->F{} type={} partitions={} keys={}",
                         ex.id,
                         ex.producerFragmentId,
                         ex.consumerFragmentId,
                         ex.partitionType,
                         ex.numPartitions,
                         ex.partitionKeyIndices.size())
                  << "\n";
      }
      r.passed = true;
      auto end = std::chrono::steady_clock::now();
      r.wallClockMs =
          std::chrono::duration<double, std::milli>(end - start).count();
      return r;
    }

    // --- 2. Thread pool + QueryCtx ---
    // Physical driver count: replica-count logic from MppJniWrapper.
    const auto& fragments = loaded.fragments;
    const auto& exchanges = loaded.exchanges;
    std::vector<int32_t> replicaCount(fragments.size(), 1);
    for (const auto& ex : exchanges) {
      const auto consumer = ex.consumerFragmentId;
      const auto n = std::max(1, ex.numPartitions);
      if (replicaCount[consumer] == 1) {
        replicaCount[consumer] = n;
      }
    }
    int32_t totalPhysicalDrivers = 0;
    for (size_t i = 0; i < fragments.size(); ++i) {
      int32_t perReplicaDrivers = replicaCount[i] == 1
          ? std::max(1, fragments[i].numDrivers)
          : 1;
      totalPhysicalDrivers += replicaCount[i] * perReplicaDrivers;
    }
    const int32_t poolSize = std::max(4, totalPhysicalDrivers * 2);
    auto executor =
        std::make_shared<folly::CPUThreadPoolExecutor>(poolSize);

    // Aggregate pool for the query, matching the JNI path.
    auto mppPool = rootPool->addAggregateChild("mpp-query-" + r.name);
    std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
        connectorConfigs;
    std::unordered_map<std::string, std::string> queryConfigMap = {
        {core::QueryConfig::kMaxOutputBufferSize, "1073741824"},
        {core::QueryConfig::kMaxPartitionedOutputBufferSize, "1073741824"},
    };
    auto queryCtx = core::QueryCtx::create(
        executor.get(),
        core::QueryConfig{std::move(queryConfigMap)},
        connectorConfigs,
        gluten::VeloxBackend::get()->getAsyncDataCache(),
        mppPool,
        /*spillExecutor=*/nullptr,
        "mpp-runner-" + r.name);

    // --- 3. Coordinator + drain ---
    // Use a monotonically increasing query id like the JNI path does, to avoid
    // taskId collisions between consecutive queries in the same process.
    static std::atomic<uint64_t> gRunnerQueryCounter{0};
    auto queryId = fmt::format(
        "runner-{}",
        gRunnerQueryCounter.fetch_add(1, std::memory_order_relaxed));

    auto coord = gluten::MppQueryCoordinator::create(
        queryId,
        std::vector<gluten::MppFragmentSpec>(loaded.fragments),
        std::vector<gluten::MppExchangeSpec>(loaded.exchanges),
        queryCtx,
        executor.get());
    coord->start();

    while (auto batch = coord->next()) {
      r.numPages++;
      r.numRows += batch->size();
      r.numBytes += batch->estimateFlatSize();
    }
    coord->waitForCompletion();

    r.passed = true;
  } catch (const std::exception& e) {
    r.passed = false;
    r.error = e.what();
  }

  auto end = std::chrono::steady_clock::now();
  r.wallClockMs =
      std::chrono::duration<double, std::milli>(end - start).count();
  return r;
}

// ---------------------------------------------------------------------------
// Print final summary table.
// ---------------------------------------------------------------------------
void printSummary(const std::vector<QueryResult>& results) {
  std::cout << "\n========== MppSubstraitRunner Summary ==========\n";
  std::cout << fmt::format(
      "{:<20} {:<8} {:>10} {:>8} {:>10} {:>12}\n",
      "Query",
      "Status",
      "WallMs",
      "Pages",
      "Rows",
      "Bytes");
  std::cout << std::string(74, '-') << "\n";
  int passed = 0;
  for (const auto& r : results) {
    std::cout << fmt::format(
        "{:<20} {:<8} {:>10.2f} {:>8} {:>10} {:>12}\n",
        r.name,
        r.passed ? "PASS" : "FAIL",
        r.wallClockMs,
        r.numPages,
        r.numRows,
        r.numBytes);
    if (!r.passed && !r.error.empty()) {
      std::cout << "    error: " << r.error << "\n";
    }
    if (r.passed) {
      ++passed;
    }
  }
  std::cout << std::string(74, '-') << "\n";
  std::cout << passed << "/" << results.size() << " passed\n";
}

} // namespace

// ---------------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  // If no args, print usage and exit 0 (matches the spec: "Run the binary
  // once with no args and confirm it prints a usage message and exits
  // cleanly.").
  if (argc == 1) {
    printUsage();
    return 0;
  }

  // folly::Init + gflags parse. Keep gflags help_short output.
  gflags::SetUsageMessage(
      "mpp-substrait-runner: replay Gluten-dumped multi-fragment Substrait "
      "plans on Velox-cudf via MppQueryCoordinator.");
  folly::Init init(&argc, &argv, /*removeFlags=*/false);
  google::SetLogDestination(google::GLOG_INFO, "/tmp/mpp_runner_info.log");

  if (FLAGS_dump_dir.empty() && !FLAGS_builtin_smoke) {
    std::cerr << "error: one of --dump-dir or --builtin-smoke is required.\n\n";
    printUsage();
    return 2;
  }

  // --- Phase 1: process-level initialization ---
  //
  // For dump replay we need a fully-initialized VeloxBackend because it
  // registers filesystems (local/file://), Hive + CudfHive connectors, the
  // Parquet reader, Gluten's memory manager, and cuDF init. The built-in
  // smoke test doesn't strictly need it, but the cost is small and keeping
  // a single init path simplifies the harness.
  gluten::VeloxBackend::create(
      gluten::AllocationListener::noop(), defaultSessionConf());
  auto rootPool = gluten::VeloxBackend::get()
                      ->getGlobalMemoryManager()
                      ->getAggregateMemoryPool();

  // PrestoVectorSerde is needed both for PartitionedOutputNode encoding and
  // for MppQueryCoordinator::next() deserialization. Idempotency-guarded.
  if (!isRegisteredVectorSerde()) {
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde(VectorSerde::Kind::kPresto)) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }

#ifdef GLUTEN_ENABLE_GPU
  // Register the GPU exchange source factory. The coordinator uses
  // "gpu-local://" task IDs which are routed to this factory.
  exec::ExchangeSource::registerFactory(
      cudf_velox::createLocalGpuExchangeSource);
  cudf_velox::testingStartLocalGpuExchangeSource();
  LOG(WARNING) << "mpp-substrait-runner: registered LocalGpuExchangeSource "
               << "factory and called testingStartLocalGpuExchangeSource";
#else
  LOG(WARNING)
      << "mpp-substrait-runner: built without GLUTEN_ENABLE_GPU; "
      << "GPU exchange source factory NOT registered. "
         "Only CPU paths will work.";
#endif

  // Shared thread pool for all queries in this run.
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(
      std::max(4u, std::thread::hardware_concurrency()));

  std::vector<QueryResult> results;

  // --- Phase 2: run queries ---
  if (FLAGS_builtin_smoke) {
    results.push_back(runBuiltinSmoke(executor.get(), rootPool));
  }

  if (!FLAGS_dump_dir.empty()) {
    fs::path dumpRoot(FLAGS_dump_dir);
    if (!fs::exists(dumpRoot) || !fs::is_directory(dumpRoot)) {
      std::cerr << "error: --dump-dir does not exist or is not a directory: "
                << FLAGS_dump_dir << "\n";
      return 3;
    }

    std::vector<fs::path> queryDirs;
    if (!FLAGS_query.empty()) {
      queryDirs.push_back(dumpRoot / FLAGS_query);
    } else {
      for (const auto& entry : fs::directory_iterator(dumpRoot)) {
        if (entry.is_directory()) {
          queryDirs.push_back(entry.path());
        }
      }
      std::sort(queryDirs.begin(), queryDirs.end());
    }

    for (const auto& qd : queryDirs) {
      if (!fs::is_directory(qd)) {
        QueryResult r;
        r.name = qd.filename().string();
        r.passed = false;
        r.error = "not a directory: " + qd.string();
        results.push_back(r);
        continue;
      }
      results.push_back(runQueryFromDump(qd, rootPool));
    }
  }

  printSummary(results);

#ifdef GLUTEN_ENABLE_GPU
  cudf_velox::testingShutdownLocalGpuExchangeSource();
#endif

  // Non-zero exit if any query failed.
  for (const auto& r : results) {
    if (!r.passed) {
      return 1;
    }
  }
  return 0;
}
