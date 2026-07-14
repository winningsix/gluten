/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0.
 */

#include "cudf/NfJsonExtractScalar.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nfjson = gluten::nfjson;

namespace {

std::string raw(std::initializer_list<unsigned char> bytes)
{
  return std::string(
      reinterpret_cast<char const*>(bytes.begin()), bytes.size());
}

std::unique_ptr<cudf::column> make_strings(
    std::vector<std::string> const& strings,
    rmm::cuda_stream_view stream,
    int32_t null_row = -1)
{
  std::vector<cudf::size_type> offsets(strings.size() + 1, 0);
  std::string chars;
  for (size_t i = 0; i < strings.size(); ++i) {
    chars += strings[i];
    offsets[i + 1] = static_cast<cudf::size_type>(chars.size());
  }
  auto mr = cudf::get_current_device_resource_ref();
  rmm::device_buffer offsets_data(offsets.size() * sizeof(cudf::size_type), stream, mr);
  rmm::device_buffer chars_data(chars.size(), stream, mr);
  cudaMemcpyAsync(
      offsets_data.data(),
      offsets.data(),
      offsets_data.size(),
      cudaMemcpyHostToDevice,
      stream.value());
  if (!chars.empty()) {
    cudaMemcpyAsync(
        chars_data.data(),
        chars.data(),
        chars.size(),
        cudaMemcpyHostToDevice,
        stream.value());
  }
  auto offsets_column = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(offsets.size()),
      std::move(offsets_data),
      rmm::device_buffer{},
      0);
  rmm::device_buffer null_mask;
  cudf::size_type null_count = 0;
  if (null_row >= 0) {
    if (null_row >= static_cast<int32_t>(strings.size())) {
      throw std::runtime_error("null row is outside the input column");
    }
    std::vector<uint8_t> host_mask(
        cudf::bitmask_allocation_size_bytes(strings.size()), 0xff);
    host_mask[null_row / 8] &= static_cast<uint8_t>(~(1u << (null_row % 8)));
    null_mask = rmm::device_buffer(host_mask.size(), stream, mr);
    cudaMemcpyAsync(
        null_mask.data(),
        host_mask.data(),
        host_mask.size(),
        cudaMemcpyHostToDevice,
        stream.value());
    null_count = 1;
  }
  return cudf::make_strings_column(
      static_cast<cudf::size_type>(strings.size()),
      std::move(offsets_column),
      std::move(chars_data),
      null_count,
      std::move(null_mask));
}

template <typename T>
std::vector<T> copy_fixed(cudf::column_view const& column, rmm::cuda_stream_view stream)
{
  std::vector<T> host(column.size());
  if (!host.empty()) {
    cudaMemcpyAsync(
        host.data(),
        column.data<T>(),
        host.size() * sizeof(T),
        cudaMemcpyDeviceToHost,
        stream.value());
  }
  stream.synchronize();
  return host;
}

std::vector<std::string> copy_strings(
    cudf::column_view const& column, rmm::cuda_stream_view stream)
{
  cudf::strings_column_view strings(column);
  std::vector<cudf::size_type> offsets(column.size() + 1);
  cudaMemcpyAsync(
      offsets.data(),
      strings.offsets().data<cudf::size_type>(),
      offsets.size() * sizeof(cudf::size_type),
      cudaMemcpyDeviceToHost,
      stream.value());
  std::vector<char> chars(static_cast<size_t>(strings.chars_size(stream)));
  if (!chars.empty()) {
    cudaMemcpyAsync(
        chars.data(),
        strings.chars_begin(stream),
        chars.size(),
        cudaMemcpyDeviceToHost,
        stream.value());
  }
  stream.synchronize();
  std::vector<std::string> result(column.size());
  for (cudf::size_type i = 0; i < column.size(); ++i) {
    result[i] = std::string(
        chars.data() + offsets[i],
        static_cast<size_t>(offsets[i + 1] - offsets[i]));
  }
  return result;
}

struct Expected {
  nfjson::ExtractStatus status;
  nfjson::ScalarType type;
  std::string value;
};

void check(
    std::string const& label,
    nfjson::ExtractResult const& result,
    std::vector<Expected> const& expected,
    rmm::cuda_stream_view stream)
{
  auto statuses = copy_fixed<int8_t>(result.statuses->view(), stream);
  auto types = copy_fixed<int8_t>(result.types->view(), stream);
  auto values = copy_strings(result.values->view(), stream);
  std::vector<uint8_t> value_mask(
      cudf::bitmask_allocation_size_bytes(result.values->size()), 0xff);
  if (result.values->nullable() && !value_mask.empty()) {
    cudaMemcpyAsync(
        value_mask.data(),
        result.values->view().null_mask(),
        value_mask.size(),
        cudaMemcpyDeviceToHost,
        stream.value());
    stream.synchronize();
  }
  if (statuses.size() != expected.size()) {
    throw std::runtime_error(label + ": row count mismatch");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    auto const actual_status = static_cast<nfjson::ExtractStatus>(statuses[i]);
    auto const actual_type = static_cast<nfjson::ScalarType>(types[i]);
    bool const value_valid = (value_mask[i / 8] & (1u << (i % 8))) != 0;
    if (actual_status != expected[i].status || actual_type != expected[i].type ||
        value_valid != (actual_status == nfjson::ExtractStatus::SCALAR) ||
        (actual_status == nfjson::ExtractStatus::SCALAR &&
         values[i] != expected[i].value)) {
      throw std::runtime_error(
          label + " row " + std::to_string(i) + " mismatch: status=" +
          std::to_string(statuses[i]) + " type=" + std::to_string(types[i]) +
          " valid=" + std::to_string(value_valid) + " value=[" + values[i] + "]");
    }
  }
}

Expected scalar(nfjson::ScalarType type, std::string value)
{
  return Expected{nfjson::ExtractStatus::SCALAR, type, std::move(value)};
}

void test_named_path(rmm::cuda_stream_view stream)
{
  std::vector<std::string> input{
      R"({"a":"ok"})",
      R"({"a":"ok"} trailing)",
      R"({"a":"ok"} {"b":1})",
      R"({"a":"ok")",
      R"({"a":"ok","b":[})",
      R"({"a":1,})",
      R"({'a':'ok'})",
      R"({a:'ok'})",
      R"({"a":"x\n\t\/\\\""})",
      R"({"a":"\u00e9\u2603"})",
      R"({"a":"\uD83E\uDD66"})",
      R"({"a":"\uD83E"})",
      R"({"a":"\uDD66"})",
      R"({"a":"a\qz"})",
      R"({"a":"\x41"})",
      R"({"a":"{text"})",
      R"({"a":"[text"})",
      R"({"a":{"x":1}})",
      R"({"a":[1]})",
      R"({"a":null})",
      R"({"a":true})",
      R"({"a":-0})",
      R"({"a":-0.0})",
      R"({"a":1e3})",
      R"({"a":1e309})",
      R"({"a":1e-4000})",
      R"({"a":9223372036854775808})",
      R"({"a":12345678900000000000.0})",
      R"({"a":0.1234567890123456789})",
      R"({"a":1.000000000000000000})",
      R"({"a":1.23456789012345e100})",
      R"({"a":NaN})",
      R"({"a":0001})",
      R"({"a":-})",
      R"({"a":5e-324})",
      R"({"a":9e-324})",
      R"({"a":1e-323})",
      R"({"a":1.1e-323})",
      R"({"a":1.01e-323})",
      R"({"a":1e-322})",
      R"({"a":1.01e-322})",
      R"({"a":-1.04323139e-323})",
      R"({"a":1e23})",
      R"({"a":0.001})",
      R"({"a":9999999.0})",
      R"({"a":10000000.0})",
      R"({"a":9007199254740992.0})",
      R"({"a":3.141592653589793})",
      R"({"a":2.2250738585e-308})",
      R"({"a":1.7976931348e308})",
      R"({"a":-1e-4000})",
      std::string("{\"a\":\"A") + '\x01' + "B\"}",
      std::string("{\"a\":\"A\\q") + '\x01' + "B\"}",
      raw({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xe0, 0x80, 0x59, 0x22, 0x7d}),
      raw({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xf0, 0x80, 0x59, 0x22, 0x7d}),
      raw({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xf4, 0xbf, 0x59, 0x22, 0x7d}),
      raw({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xe1, 0x80, 0x59, 0x22, 0x7d}),
      raw({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xe1, 0x80, 0x41, 0x59, 0x22, 0x7d}),
      raw({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xf1, 0x80, 0x80, 0x59, 0x22, 0x7d}),
      raw({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xed, 0xa0, 0x80, 0x59, 0x22, 0x7d}),
      raw({0x7b, 0x61, 0x3a, 0x22, 0x58, 0x5c, 0xc3, 0xa9, 0x59, 0x22, 0x7d}),
      raw({0x7b, 0x61, 0x3a, 0x22, 0x58, 0x5c, 0xf0, 0x9f, 0x98, 0x83, 0x59, 0x22, 0x7d}),
      raw({0x7b, 0x61, 0x3a, 0x22, 0x58, 0x5c, 0xe0, 0x80, 0x59, 0x22, 0x7d}),
      R"({"a":"first","a":"last"})",
      R"({"b":1})",
      R"({"a":"masked"})",
  };
  auto column = make_strings(input, stream, static_cast<int32_t>(input.size() - 1));
  auto result = nfjson::extract_scalar(
      cudf::strings_column_view(column->view()),
      {nfjson::PathInstruction::named("a")},
      stream);
  std::vector<Expected> expected{
      scalar(nfjson::ScalarType::STRING, "ok"),
      scalar(nfjson::ScalarType::STRING, "ok"),
      scalar(nfjson::ScalarType::STRING, "ok"),
      {nfjson::ExtractStatus::INVALID_JSON, nfjson::ScalarType::NONE, {}},
      {nfjson::ExtractStatus::INVALID_JSON, nfjson::ScalarType::NONE, {}},
      scalar(nfjson::ScalarType::INTEGER, "1"),
      scalar(nfjson::ScalarType::STRING, "ok"),
      scalar(nfjson::ScalarType::STRING, "ok"),
      scalar(nfjson::ScalarType::STRING, "x\n\t/\\\""),
      scalar(nfjson::ScalarType::STRING, "é☃"),
      scalar(nfjson::ScalarType::STRING, "🥦"),
      scalar(nfjson::ScalarType::STRING, "?"),
      scalar(nfjson::ScalarType::STRING, "?"),
      scalar(nfjson::ScalarType::STRING, "az"),
      scalar(nfjson::ScalarType::STRING, "A"),
      scalar(nfjson::ScalarType::STRING, "{text"),
      scalar(nfjson::ScalarType::STRING, "[text"),
      {nfjson::ExtractStatus::CONTAINER, nfjson::ScalarType::NONE, {}},
      {nfjson::ExtractStatus::CONTAINER, nfjson::ScalarType::NONE, {}},
      {nfjson::ExtractStatus::JSON_NULL, nfjson::ScalarType::NONE, {}},
      scalar(nfjson::ScalarType::BOOLEAN, "true"),
      scalar(nfjson::ScalarType::INTEGER, "0"),
      scalar(nfjson::ScalarType::FLOATING, "-0.0"),
      scalar(nfjson::ScalarType::FLOATING, "1000.0"),
      scalar(nfjson::ScalarType::FLOATING, "Infinity"),
      scalar(nfjson::ScalarType::FLOATING, "0.0"),
      scalar(nfjson::ScalarType::INTEGER, "9223372036854775808"),
      scalar(nfjson::ScalarType::FLOATING, "12345678900000000000.0"),
      scalar(nfjson::ScalarType::FLOATING, "0.1234567890123456789"),
      scalar(nfjson::ScalarType::FLOATING, "1.000000000000000000"),
      scalar(nfjson::ScalarType::FLOATING, "1.23456789012345E+100"),
      scalar(nfjson::ScalarType::FLOATING, "NaN"),
      scalar(nfjson::ScalarType::INTEGER, "1"),
      scalar(nfjson::ScalarType::INTEGER, "0"),
      scalar(nfjson::ScalarType::FLOATING, "4.9E-324"),
      scalar(nfjson::ScalarType::FLOATING, "1.0E-323"),
      scalar(nfjson::ScalarType::FLOATING, "1.0E-323"),
      scalar(nfjson::ScalarType::FLOATING, "1.0E-323"),
      scalar(nfjson::ScalarType::FLOATING, "1.0E-323"),
      scalar(nfjson::ScalarType::FLOATING, "1.0E-322"),
      scalar(nfjson::ScalarType::FLOATING, "1.0E-322"),
      scalar(nfjson::ScalarType::FLOATING, "-1.0E-323"),
      scalar(nfjson::ScalarType::FLOATING, "9.999999999999999E22"),
      scalar(nfjson::ScalarType::FLOATING, "0.001"),
      scalar(nfjson::ScalarType::FLOATING, "9999999.0"),
      scalar(nfjson::ScalarType::FLOATING, "1.0E7"),
      scalar(nfjson::ScalarType::FLOATING, "9.007199254740992E15"),
      scalar(nfjson::ScalarType::FLOATING, "3.141592653589793"),
      scalar(nfjson::ScalarType::FLOATING, "2.2250738585E-308"),
      scalar(nfjson::ScalarType::FLOATING, "1.7976931348E308"),
      scalar(nfjson::ScalarType::FLOATING, "-0.0"),
      scalar(nfjson::ScalarType::STRING, std::string("A") + '\x01' + "B"),
      scalar(nfjson::ScalarType::STRING, "AB"),
      scalar(nfjson::ScalarType::STRING, "X��Y"),
      scalar(nfjson::ScalarType::STRING, "X��Y"),
      scalar(nfjson::ScalarType::STRING, "X��Y"),
      scalar(nfjson::ScalarType::STRING, "X�Y"),
      scalar(nfjson::ScalarType::STRING, "X�AY"),
      scalar(nfjson::ScalarType::STRING, "X�Y"),
      scalar(nfjson::ScalarType::STRING, "X�Y"),
      scalar(nfjson::ScalarType::STRING, "XY"),
      scalar(nfjson::ScalarType::STRING, "X?Y"),
      scalar(nfjson::ScalarType::STRING, "X�Y"),
      scalar(nfjson::ScalarType::STRING, "last"),
      {nfjson::ExtractStatus::NO_MATCH, nfjson::ScalarType::NONE, {}},
      {nfjson::ExtractStatus::INPUT_NULL, nfjson::ScalarType::NONE, {}},
  };
  check("named-path", result, expected, stream);
}

void test_named_index_path(rmm::cuda_stream_view stream)
{
  std::vector<std::string> input{
      R"({"a":[,{"b":1},,{"b":"x"},]})",
      R"({"a":[{"b":1}]})",
      R"({"a":{"1":{"b":"wrong-container"}}})",
      R"({"a":[{"b":0},{"b":"x"}],"tail":[})",
  };
  auto column = make_strings(input, stream);
  auto result = nfjson::extract_scalar(
      cudf::strings_column_view(column->view()),
      {nfjson::PathInstruction::named("a"),
       nfjson::PathInstruction::at(1),
       nfjson::PathInstruction::named("b")},
      stream);
  check(
      "named-index-path",
      result,
      {scalar(nfjson::ScalarType::STRING, "x"),
       {nfjson::ExtractStatus::NO_MATCH, nfjson::ScalarType::NONE, {}},
       {nfjson::ExtractStatus::NO_MATCH, nfjson::ScalarType::NONE, {}},
       {nfjson::ExtractStatus::INVALID_JSON, nfjson::ScalarType::NONE, {}}},
      stream);
}

void test_root(rmm::cuda_stream_view stream)
{
  std::vector<std::string> input{
      R"("x\n\u2603" trailing)", "not-json", "   ", R"({"a":1})", "1e3"};
  auto column = make_strings(input, stream);
  auto result = nfjson::extract_scalar(
      cudf::strings_column_view(column->view()), {}, stream);
  check(
      "root",
      result,
      {scalar(nfjson::ScalarType::STRING, "x\n☃"),
       scalar(nfjson::ScalarType::STRING, "not-json"),
       scalar(nfjson::ScalarType::STRING, ""),
       {nfjson::ExtractStatus::CONTAINER, nfjson::ScalarType::NONE, {}},
       scalar(nfjson::ScalarType::FLOATING, "1000.0")},
      stream);
}

void test_depth_cap(rmm::cuda_stream_view stream)
{
  auto nested = [](int32_t depth) {
    return std::string(static_cast<size_t>(depth), '[') + "0" +
           std::string(static_cast<size_t>(depth), ']');
  };
  auto column = make_strings({nested(512), nested(513)}, stream);
  auto result = nfjson::extract_scalar(
      cudf::strings_column_view(column->view()), {}, stream);
  check(
      "depth-cap",
      result,
      {{nfjson::ExtractStatus::CONTAINER, nfjson::ScalarType::NONE, {}},
       {nfjson::ExtractStatus::INVALID_JSON, nfjson::ScalarType::NONE, {}}},
      stream);
}

void test_irrelevant_depth_cap(rmm::cuda_stream_view stream)
{
  auto object_with_nested_tail = [](int32_t array_depth) {
    return std::string(R"({"a":"ok","tail":)") +
           std::string(static_cast<size_t>(array_depth), '[') + "0" +
           std::string(static_cast<size_t>(array_depth), ']') + "}";
  };
  // The root object counts as one level: 511 tail arrays are the last valid
  // total depth, while 512 tail arrays fail closed even though $.a matched.
  auto column = make_strings(
      {object_with_nested_tail(511), object_with_nested_tail(512)}, stream);
  auto result = nfjson::extract_scalar(
      cudf::strings_column_view(column->view()),
      {nfjson::PathInstruction::named("a")},
      stream);
  check(
      "irrelevant-depth-cap",
      result,
      {scalar(nfjson::ScalarType::STRING, "ok"),
       {nfjson::ExtractStatus::INVALID_JSON, nfjson::ScalarType::NONE, {}}},
      stream);
}

} // namespace

int main()
{
  try {
    cudaSetDevice(0);
    auto stream = cudf::get_default_stream();
    test_named_path(stream);
    test_named_index_path(stream);
    test_root(stream);
    test_depth_cap(stream);
    test_irrelevant_depth_cap(stream);
    stream.synchronize();
    std::cout << "NfJsonExtractScalarGpuTest: PASS\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "NfJsonExtractScalarGpuTest: FAIL: " << error.what() << '\n';
    return 1;
  }
}
