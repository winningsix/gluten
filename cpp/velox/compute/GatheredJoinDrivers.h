/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
#include <string_view>

namespace gluten {

// Experimental direct INNER-join + single gathered writer only. Do not use
// this limit for ordinary HASH consumers or broadcast producers.
inline int32_t parseGatheredJoinDrivers(std::string_view value) {
  if (value.empty() || value == "1")
    return 1;
  if (value == "2")
    return 2;
  if (value == "4")
    return 4;
  return 0; // caller reports the invalid setting, never silently clamps it
}

inline int32_t gatheredJoinDrivers(int32_t requested, int32_t replicas) {
  return replicas == 1 ? std::clamp(requested, 1, 4) : 1;
}

} // namespace gluten
