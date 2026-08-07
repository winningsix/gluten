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

#pragma once

// Load a Gluten-dumped Substrait plan directory into FLUX fragment + exchange
// specs suitable for FluxQueryCoordinator. Mirrors the fragment-building stage
// of FluxJniWrapper::nativeCreateFluxQuery, but reads from disk instead of
// JNI arrays, so that flux-substrait-runner (or any other offline harness)
// can replay a captured plan.
//
// Expected on-disk layout (produced by FluxNativeQueryExec.dumpPlanIfEnabled):
//   <queryDir>/fragment-<id>.pb            raw substrait::Plan bytes per fragment
//   <queryDir>/splits-frag<id>-leaf<m>.pb  raw ReadRel::LocalFiles per leaf scan
//   <queryDir>/exchange-specs.json         byte-identical to JNI exchangeSpecsJsonArr
//   <queryDir>/manifest.json               descriptor (fragment count, splitFiles[])

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "compute/FluxQueryCoordinator.h"
#include "velox/common/memory/MemoryPool.h"

namespace gluten {

/// Result of loading a dumped FLUX plan. The caller owns these and feeds them
/// to FluxQueryCoordinator::create along with a QueryCtx and executor.
struct FluxDumpLoadResult {
  std::vector<FluxFragmentSpec> fragments;
  std::vector<FluxExchangeSpec> exchanges;
};

/// Load a single dumped query from disk. Throws on invalid layout or
/// Substrait parse/conversion failure.
///
/// @param queryDir       directory holding fragment-*.pb, splits-*.pb,
///                       exchange-specs.json, manifest.json
/// @param veloxPool      leaf memory pool used by VeloxPlanConverter during
///                       plan conversion; must outlive the returned specs
///                       only until the coordinator is created
/// @param sessionConf    session-level config map (cudf flags, etc.). Merged
///                       on top of VeloxBackend's backend conf the same way
///                       nativeCreateFluxQuery does for the runtime-session map
FluxDumpLoadResult loadFluxQueryFromDump(
    const std::filesystem::path& queryDir,
    facebook::velox::memory::MemoryPool* veloxPool,
    const std::unordered_map<std::string, std::string>& sessionConf);

} // namespace gluten
