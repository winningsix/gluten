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

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <folly/dynamic.h>
#include "velox/common/base/RuntimeMetrics.h"

namespace gluten::detail {

inline folly::dynamic serializeFluxOperatorRuntimeStats(
    const std::unordered_map<std::string, facebook::velox::RuntimeMetric>& runtimeStats) {
  std::vector<std::string> names;
  names.reserve(runtimeStats.size());
  for (const auto& entry : runtimeStats) {
    names.push_back(entry.first);
  }
  std::sort(names.begin(), names.end());

  folly::dynamic result = folly::dynamic::object;
  for (const auto& name : names) {
    const auto& metric = runtimeStats.at(name);
    folly::dynamic value = folly::dynamic::object;
    value["sum"] = metric.sum;
    value["count"] = static_cast<int64_t>(metric.count);
    value["min"] = metric.min;
    value["max"] = metric.max;
    result[name] = std::move(value);
  }
  return result;
}

} // namespace gluten::detail
