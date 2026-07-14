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
 *
 * Parser state transitions are derived from json-smart 2.2.1
 * JSONParserBase/JSONParserMemory/JSONParserString:
 *   Copyright 2011 JSON-SMART authors, Apache License 2.0
 *   https://github.com/netplex/json-smart-v2, release 2.2.1
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gluten::netflix_json {

/**
 * A plan-time JSONPath in the exact first native slice accepted for
 * NfJsonExtractScalar: `$.field(.field|[nonnegative-index])*`.
 *
 * Field names are ASCII identifiers bounded to 128 bytes, indexes fit a
 * signed INT, total path text is bounded to 4096 bytes, and paths have at most
 * sixteen selections. The disconnected production Catalyst gate must apply
 * the same restrictions before this native expression can be wired.
 */
class SimplePath {
 public:
  enum class StepKind : uint8_t { kField, kIndex };

  struct Step {
    StepKind kind;
    std::string field;
    int32_t index{-1};
  };

  static std::optional<SimplePath> parse(std::string_view path);

  const std::vector<Step>& steps() const {
    return steps_;
  }

 private:
  explicit SimplePath(std::vector<Step> steps) : steps_(std::move(steps)) {}

  std::vector<Step> steps_;
};

/**
 * Extracts the selected json-smart 2.2.1 MODE_PERMISSIVE scalar.
 *
 * The complete first root is validated; bytes following that completed root
 * are ignored. Strings, Numbers and Booleans return their Netflix/JVM string
 * representation. Missing paths, JSON null, containers and invalid input
 * return std::nullopt. Nesting beyond 512 levels fails closed to bound native
 * stack use.
 */
std::optional<std::string> extractScalar(std::string_view json, const SimplePath& path);

/** Parses the bounded path and extracts, or returns std::nullopt on either error. */
std::optional<std::string> extractScalar(std::string_view json, std::string_view path);

} // namespace gluten::netflix_json
