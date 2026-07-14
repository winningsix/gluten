/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0.
 *
 * Portions derived from json-smart 2.2.1:
 *   Copyright 2011 JSON-SMART authors
 * Number formatting includes spark-rapids-jni 26.06's Apache-2.0 Ryu port:
 *   Copyright 2018 Ulf Adams
 *   Copyright (c) 2023-2026, NVIDIA CORPORATION
 * Both upstream works are licensed under Apache License 2.0. Exact source
 * references and semantic deltas are recorded in NfJsonExtractScalar.cu.
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace gluten::nfjson {

/** The result class is deliberately separate from scalar type. */
enum class ExtractStatus : int8_t {
  INVALID_JSON = 0,
  INPUT_NULL = 1,
  NO_MATCH = 2,
  JSON_NULL = 3,
  CONTAINER = 4,
  SCALAR = 5,
};

enum class ScalarType : int8_t {
  NONE = 0,
  STRING = 1,
  INTEGER = 2,
  FLOATING = 3,
  BOOLEAN = 4,
};

enum class PathInstructionType : int8_t { NAMED = 0, INDEX = 1 };

struct PathInstruction {
  PathInstructionType type;
  std::string name;
  int32_t index{-1};

  static PathInstruction named(std::string field)
  {
    return PathInstruction{PathInstructionType::NAMED, std::move(field), -1};
  }

  static PathInstruction at(int32_t position)
  {
    return PathInstruction{PathInstructionType::INDEX, {}, position};
  }
};

/**
 * Prototype result. `values` is valid only where status == SCALAR; every
 * other row is SQL null. `statuses` and `types` are non-null INT8 columns.
 * Keeping both sidecars prevents a decoded string beginning with '{'/'['
 * from being confused with a JSON container and lets number rendering handle
 * Infinity without JSON quotes.
 */
struct ExtractResult {
  std::unique_ptr<::cudf::column> values;
  std::unique_ptr<::cudf::column> statuses;
  std::unique_ptr<::cudf::column> types;
};

/**
 * Extract a scalar using a bounded sequence of named/non-negative-index path
 * instructions. Empty instructions mean root `$`; callers are expected to
 * enforce the production planner domain separately.
 *
 * The parser validates the complete first root value, even after finding the
 * target. Bytes following that completed root are intentionally ignored to
 * match json-smart 2.2.1 MODE_PERMISSIVE as used by the Netflix JAR. JSON
 * nesting is fail-closed above 512 total object/array levels; the prototype
 * uses a bounded, preallocated device workspace instead of CUDA recursion.
 */
ExtractResult extract_scalar(
    ::cudf::strings_column_view const& input,
    std::vector<PathInstruction> const& path,
    rmm::cuda_stream_view stream = ::cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = ::cudf::get_current_device_resource_ref());

} // namespace gluten::nfjson
