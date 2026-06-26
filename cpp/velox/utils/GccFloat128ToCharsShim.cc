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

#include <algorithm>
#include <charconv>
#include <cstring>
#include <vector>

#include <quadmath.h>

namespace {

const char* formatSpec(std::chars_format fmt) {
  switch (fmt) {
    case std::chars_format::fixed:
      return "%.*Qf";
    case std::chars_format::scientific:
      return "%.*Qe";
    case std::chars_format::hex:
      return "%.*Qa";
    case std::chars_format::general:
    default:
      return "%.*Qg";
  }
}

std::to_chars_result formatFloat128(
    char* first,
    char* last,
    _Float128 value,
    std::chars_format fmt,
    int precision) {
  if (last < first) {
    return {first, std::errc::invalid_argument};
  }

  const auto capacity = static_cast<size_t>(last - first);
  const auto scratchSize = std::max<size_t>(256, static_cast<size_t>(std::max(precision, 0)) + 64);
  std::vector<char> scratch(scratchSize);
  const int size = quadmath_snprintf(
      scratch.data(), scratch.size(), formatSpec(fmt), precision, static_cast<__float128>(value));
  if (size < 0) {
    return {first, std::errc::invalid_argument};
  }
  const auto bytes = static_cast<size_t>(size);
  if (bytes > capacity) {
    return {last, std::errc::value_too_large};
  }

  std::memcpy(first, scratch.data(), bytes);
  return {first + bytes, std::errc{}};
}

} // namespace

namespace std {

to_chars_result to_chars(char* first, char* last, _Float128 value) {
  return formatFloat128(first, last, value, chars_format::general, 36);
}

to_chars_result to_chars(char* first, char* last, _Float128 value, chars_format fmt) {
  return formatFloat128(first, last, value, fmt, 36);
}

to_chars_result to_chars(char* first, char* last, _Float128 value, chars_format fmt, int precision) {
  return formatFloat128(first, last, value, fmt, precision);
}

} // namespace std
