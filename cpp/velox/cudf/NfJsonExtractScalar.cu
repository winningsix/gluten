/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0.
 *
 * Parser state transitions are derived from json-smart 2.2.1
 * JSONParserBase/JSONParserMemory/JSONParserString:
 *   Copyright 2011 JSON-SMART authors, Apache License 2.0
 *   https://github.com/netplex/json-smart-v2, release 2.2.1
 *
 * Finite double rendering uses spark-rapids-jni 26.06's Apache-2.0 Ryu port
 * (commit 80643d090ffd043a61127868efcd199304f2d9b0), with explicit JDK17
 * compatibility boundaries established by the production-JAR oracle.
 *
 * This prototype intentionally does not link libcudfjni.so. It uses the cuDF,
 * RMM, CCCL and stream ABI already selected by the Gluten Velox-cuDF build.
 */

#include "cudf/NfJsonExtractScalar.h"
#include "cudf/NfJsonDoubleToString.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/convert/string_to_float.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/cmath>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace gluten::nfjson {
namespace {

constexpr int32_t MAX_PATH_DEPTH = 16;
constexpr int32_t MAX_JSON_DEPTH = 512;

enum class StopContext : int8_t { ROOT, OBJECT, ARRAY };

struct DevicePathInstruction {
  PathInstructionType type;
  int32_t name_offset;
  int32_t name_length;
  int32_t index;
};

struct Selection {
  ExtractStatus status{ExtractStatus::NO_MATCH};
  ScalarType type{ScalarType::NONE};
  int32_t begin{0};
  int32_t end{0};
  bool quoted{false};
};

enum class ContainerKind : int8_t { OBJECT, ARRAY };

struct ParserFrame {
  Selection selected;
  int32_t path_index;
  int32_t element_index;
  ContainerKind kind;
  bool evaluate;
  bool pending_match;
};

struct DecodedSink {
  char* output;
  char const* expected;
  int32_t expected_length;
  int32_t count{0};
  bool matches{true};

  __device__ void put(char c)
  {
    if (output != nullptr) { output[count] = c; }
    if (expected != nullptr &&
        (count >= expected_length || expected[count] != c)) {
      matches = false;
    }
    ++count;
  }
};

__device__ bool is_json_space(char c)
{
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

__device__ bool is_digit(char c) { return c >= '0' && c <= '9'; }

__device__ int hex_value(char c)
{
  if (c >= '0' && c <= '9') { return c - '0'; }
  if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
  if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
  return -1;
}

__device__ void emit_codepoint(uint32_t cp, DecodedSink& sink)
{
  if (cp <= 0x7f) {
    sink.put(static_cast<char>(cp));
  } else if (cp <= 0x7ff) {
    sink.put(static_cast<char>(0xc0 | (cp >> 6)));
    sink.put(static_cast<char>(0x80 | (cp & 0x3f)));
  } else if (cp <= 0xffff) {
    sink.put(static_cast<char>(0xe0 | (cp >> 12)));
    sink.put(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    sink.put(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    sink.put(static_cast<char>(0xf0 | (cp >> 18)));
    sink.put(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
    sink.put(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    sink.put(static_cast<char>(0x80 | (cp & 0x3f)));
  }
}

/** Decode a json-smart permissive quoted string into UTF-8. */
__device__ bool decode_quoted(
    char const* data, int32_t begin, int32_t end, DecodedSink& sink)
{
  // JSONParserMemory selects a fast path when no backslash is present. In
  // MODE_PERMISSIVE that path preserves raw controls. Once any backslash is
  // present, readString2 is used and drops raw controls (while still decoding
  // the escapes). This odd branch is observable through the production JAR.
  bool slow_path = false;
  for (int32_t i = begin; i < end; ++i) {
    if (data[i] == '\\') {
      slow_path = true;
      break;
    }
  }
  int32_t p = begin;
  while (p < end) {
    char c = data[p++];
    if (c != '\\') {
      auto const uc = static_cast<unsigned char>(c);
      if (slow_path && (uc <= 31 || uc == 127)) { continue; }
      sink.put(c);
      continue;
    }
    if (p >= end) { return false; }
    char escaped = data[p++];
    switch (escaped) {
      case 't': sink.put('\t'); break;
      case 'n': sink.put('\n'); break;
      case 'r': sink.put('\r'); break;
      case 'f': sink.put('\f'); break;
      case 'b': sink.put('\b'); break;
      case '\\': sink.put('\\'); break;
      case '/': sink.put('/'); break;
      case '\'': sink.put('\''); break;
      case '"': sink.put('"'); break;
      case 'x': {
        if (p + 2 > end) { return false; }
        auto const h0 = hex_value(data[p]);
        auto const h1 = hex_value(data[p + 1]);
        if (h0 < 0 || h1 < 0) { return false; }
        emit_codepoint(static_cast<uint32_t>((h0 << 4) | h1), sink);
        p += 2;
        break;
      }
      case 'u': {
        if (p + 4 > end) { return false; }
        uint32_t cp = 0;
        for (int i = 0; i < 4; ++i) {
          auto const h = hex_value(data[p + i]);
          if (h < 0) { return false; }
          cp = (cp << 4) | static_cast<uint32_t>(h);
        }
        p += 4;
        if (cp >= 0xd800 && cp <= 0xdbff && p + 6 <= end &&
            data[p] == '\\' && data[p + 1] == 'u') {
          uint32_t low = 0;
          bool valid_low = true;
          for (int i = 0; i < 4; ++i) {
            auto const h = hex_value(data[p + 2 + i]);
            if (h < 0) {
              valid_low = false;
              break;
            }
            low = (low << 4) | static_cast<uint32_t>(h);
          }
          if (valid_low && low >= 0xdc00 && low <= 0xdfff) {
            cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
            p += 6;
          }
        }
        // UTF8String.fromString replaces an isolated UTF-16 surrogate with
        // ASCII '?', not U+FFFD. Preserve that final boundary behavior here.
        if (cp >= 0xd800 && cp <= 0xdfff) {
          sink.put('?');
        } else {
          emit_codepoint(cp, sink);
        }
        break;
      }
      default:
        // json-smart MODE_PERMISSIVE consumes and drops an unknown escape.
        // It consumes one Java UTF-16 code unit, not one UTF-8 byte. The input
        // has already passed through Java-compatible UTF-8 replacement below.
        // A supplementary code point is two UTF-16 units: dropping its high
        // surrogate leaves a lone low surrogate, which UTF8String renders '?'.
        if ((static_cast<unsigned char>(escaped) & 0xe0) == 0xc0) {
          p += 1;
        } else if ((static_cast<unsigned char>(escaped) & 0xf0) == 0xe0) {
          p += 2;
        } else if ((static_cast<unsigned char>(escaped) & 0xf8) == 0xf0) {
          p += 3;
          sink.put('?');
        }
        break;
    }
  }
  if (sink.expected != nullptr && sink.count != sink.expected_length) {
    sink.matches = false;
  }
  return true;
}

__device__ bool is_utf8_continuation(unsigned char byte)
{
  return (byte & 0xc0) == 0x80;
}

struct JavaUtf8Unit {
  int32_t consumed;
  bool valid;
};

/**
 * Return the malformed-input unit consumed by Java 17's UTF-8 decoder.
 * This is intentionally an oracle-derived behavioral implementation, not a
 * copy of OpenJDK code. In particular, a valid prefix before a bad/missing
 * third or fourth byte becomes one U+FFFD, while an illegal second byte makes
 * only the lead byte malformed (for example E0 80 becomes two U+FFFDs).
 */
__device__ JavaUtf8Unit java_utf8_unit(
    char const* data, int32_t length, int32_t position)
{
  auto const b1 = static_cast<unsigned char>(data[position]);
  if (b1 <= 0x7f) { return JavaUtf8Unit{1, true}; }

  if (b1 >= 0xc2 && b1 <= 0xdf) {
    if (position + 1 < length &&
        is_utf8_continuation(static_cast<unsigned char>(data[position + 1]))) {
      return JavaUtf8Unit{2, true};
    }
    return JavaUtf8Unit{1, false};
  }

  if (b1 >= 0xe0 && b1 <= 0xef) {
    if (position + 1 >= length) { return JavaUtf8Unit{1, false}; }
    auto const b2 = static_cast<unsigned char>(data[position + 1]);
    bool const second_valid =
        is_utf8_continuation(b2) && !(b1 == 0xe0 && b2 < 0xa0);
    if (!second_valid) { return JavaUtf8Unit{1, false}; }
    if (position + 2 >= length) { return JavaUtf8Unit{2, false}; }
    if (!is_utf8_continuation(
            static_cast<unsigned char>(data[position + 2]))) {
      return JavaUtf8Unit{2, false};
    }
    // Java groups a UTF-8-encoded surrogate (ED A0..BF 80..BF) into one
    // malformed unit even though it is not a valid Unicode scalar value.
    return JavaUtf8Unit{3, !(b1 == 0xed && b2 >= 0xa0)};
  }

  if (b1 >= 0xf0 && b1 <= 0xf4) {
    if (position + 1 >= length) { return JavaUtf8Unit{1, false}; }
    auto const b2 = static_cast<unsigned char>(data[position + 1]);
    bool const second_valid =
        is_utf8_continuation(b2) && !(b1 == 0xf0 && b2 < 0x90) &&
        !(b1 == 0xf4 && b2 >= 0x90);
    if (!second_valid) { return JavaUtf8Unit{1, false}; }
    if (position + 2 >= length) { return JavaUtf8Unit{2, false}; }
    if (!is_utf8_continuation(
            static_cast<unsigned char>(data[position + 2]))) {
      return JavaUtf8Unit{2, false};
    }
    if (position + 3 >= length) { return JavaUtf8Unit{3, false}; }
    if (!is_utf8_continuation(
            static_cast<unsigned char>(data[position + 3]))) {
      return JavaUtf8Unit{3, false};
    }
    return JavaUtf8Unit{4, true};
  }

  return JavaUtf8Unit{1, false};
}

__device__ int32_t normalize_java_utf8(
    char const* data, int32_t length, char* output)
{
  int32_t position = 0;
  int64_t written = 0;
  while (position < length) {
    auto const unit = java_utf8_unit(data, length, position);
    int32_t const emitted = unit.valid ? unit.consumed : 3;
    if (written >
        static_cast<int64_t>(std::numeric_limits<int32_t>::max()) - emitted) {
      return -1;
    }
    if (unit.valid) {
      if (output != nullptr) {
        for (int32_t i = 0; i < unit.consumed; ++i) {
          output[written + i] = data[position + i];
        }
      }
    } else {
      if (output != nullptr) {
        output[written] = static_cast<char>(0xef);
        output[written + 1] = static_cast<char>(0xbf);
        output[written + 2] = static_cast<char>(0xbd);
      }
    }
    written += emitted;
    position += unit.consumed;
  }
  return static_cast<int32_t>(written);
}

// Correctly-rounded decimal-to-binary64 conversion for the json-smart Double
// branch. libcudf's fast stod can be one ULP away near normal/subnormal
// boundaries (for example 2.2250738585e-308), so it is used only as a seed and
// then corrected by exact decimal-vs-midpoint comparisons.
struct DecimalBigUInt {
  static constexpr int32_t capacity = 32;
  uint64_t words[capacity]{};
  int32_t length{1};
};

__device__ void decimal_normalize(DecimalBigUInt& value)
{
  while (value.length > 1 && value.words[value.length - 1] == 0) {
    --value.length;
  }
}

__device__ DecimalBigUInt decimal_from(uint64_t value)
{
  DecimalBigUInt result{};
  result.words[0] = value;
  return result;
}

__device__ int decimal_compare(
    DecimalBigUInt const& lhs, DecimalBigUInt const& rhs)
{
  if (lhs.length != rhs.length) { return lhs.length < rhs.length ? -1 : 1; }
  for (int32_t i = lhs.length - 1; i >= 0; --i) {
    if (lhs.words[i] != rhs.words[i]) {
      return lhs.words[i] < rhs.words[i] ? -1 : 1;
    }
  }
  return 0;
}

__device__ bool decimal_shift_left(DecimalBigUInt& value, int32_t shift)
{
  if (shift <= 0) { return true; }
  int32_t const word_shift = shift / 64;
  int32_t const bit_shift = shift % 64;
  if (value.length + word_shift + 1 > DecimalBigUInt::capacity) { return false; }
  for (int32_t i = value.length - 1; i >= 0; --i) {
    value.words[i + word_shift] = value.words[i];
  }
  for (int32_t i = 0; i < word_shift; ++i) { value.words[i] = 0; }
  value.length += word_shift;
  if (bit_shift != 0) {
    uint64_t carry = 0;
    for (int32_t i = word_shift; i < value.length; ++i) {
      uint64_t const current = value.words[i];
      value.words[i] = (current << bit_shift) | carry;
      carry = current >> (64 - bit_shift);
    }
    if (carry != 0) { value.words[value.length++] = carry; }
  }
  decimal_normalize(value);
  return true;
}

__device__ bool decimal_mul_u64(DecimalBigUInt& value, uint64_t multiplier)
{
  uint64_t carry = 0;
  for (int32_t i = 0; i < value.length; ++i) {
    uint64_t const current = value.words[i];
    uint64_t low = current * multiplier;
    uint64_t high = __umul64hi(current, multiplier);
    uint64_t const with_carry = low + carry;
    if (with_carry < low) { ++high; }
    value.words[i] = with_carry;
    carry = high;
  }
  if (carry != 0) {
    if (value.length == DecimalBigUInt::capacity) { return false; }
    value.words[value.length++] = carry;
  }
  return true;
}

__device__ bool decimal_times_power5(DecimalBigUInt& value, int32_t exponent)
{
  for (int32_t i = 0; i < exponent; ++i) {
    if (!decimal_mul_u64(value, 5)) { return false; }
  }
  return true;
}

__device__ void binary_value(uint64_t bits, uint64_t& significand, int32_t& exponent)
{
  if (bits == 0) {
    significand = 0;
    exponent = -1074;
    return;
  }
  uint64_t const exponent_bits = (bits >> 52) & 0x7ff;
  uint64_t const fraction = bits & 0x000fffffffffffffull;
  if (exponent_bits == 0) {
    significand = fraction;
    exponent = -1074;
  } else if (exponent_bits == 0x7ff) {
    // Conceptual next value above max finite: 2^1024.
    significand = uint64_t{1} << 52;
    exponent = 972;
  } else {
    significand = (uint64_t{1} << 52) | fraction;
    exponent = static_cast<int32_t>(exponent_bits) - 1023 - 52;
  }
}

__device__ void binary_midpoint(
    uint64_t lower_bits,
    uint64_t upper_bits,
    uint64_t& midpoint_significand,
    int32_t& midpoint_exponent)
{
  uint64_t lower_significand;
  uint64_t upper_significand;
  int32_t lower_exponent;
  int32_t upper_exponent;
  binary_value(lower_bits, lower_significand, lower_exponent);
  binary_value(upper_bits, upper_significand, upper_exponent);
  int32_t const common = lower_exponent < upper_exponent ? lower_exponent
                                                         : upper_exponent;
  midpoint_significand =
      (lower_significand << (lower_exponent - common)) +
      (upper_significand << (upper_exponent - common));
  midpoint_exponent = common - 1;
}

__device__ int compare_decimal_to_midpoint(
    uint64_t decimal_significand,
    int32_t decimal_exponent,
    uint64_t midpoint_significand,
    int32_t midpoint_exponent)
{
  DecimalBigUInt lhs = decimal_from(decimal_significand);
  DecimalBigUInt rhs = decimal_from(midpoint_significand);
  int32_t binary_power = decimal_exponent;
  bool valid = true;
  if (decimal_exponent >= 0) {
    valid = decimal_times_power5(lhs, decimal_exponent);
  } else {
    valid = decimal_times_power5(rhs, -decimal_exponent);
  }
  int32_t const shift = binary_power - midpoint_exponent;
  if (shift >= 0) {
    valid = valid && decimal_shift_left(lhs, shift);
  } else {
    valid = valid && decimal_shift_left(rhs, -shift);
  }
  // Nearby decimal and binary values always fit the fixed capacity. Returning
  // by sign is a defensive path for a wildly inaccurate seed.
  if (!valid) { return shift >= 0 ? 1 : -1; }
  return decimal_compare(lhs, rhs);
}

struct ParsedDouble {
  bool valid{false};
  double value{0};
};

__device__ ParsedDouble parse_java_double(
    char const* data, int32_t begin, int32_t end)
{
  bool negative = false;
  int32_t p = begin;
  if (p < end && (data[p] == '-' || data[p] == '+')) {
    negative = data[p] == '-';
    ++p;
  }
  uint64_t significand = 0;
  int32_t fraction_digits = 0;
  bool after_decimal = false;
  bool any_digit = false;
  while (p < end && data[p] != 'e' && data[p] != 'E') {
    char const c = data[p++];
    if (c == '.' && !after_decimal) {
      after_decimal = true;
      continue;
    }
    if (!is_digit(c)) { return {}; }
    any_digit = true;
    significand = significand * 10 + static_cast<uint64_t>(c - '0');
    if (after_decimal) { ++fraction_digits; }
  }
  if (!any_digit) { return {}; }
  int32_t explicit_exponent = 0;
  bool exponent_negative = false;
  if (p < end) {
    ++p;
    if (p < end && (data[p] == '+' || data[p] == '-')) {
      exponent_negative = data[p] == '-';
      ++p;
    }
    if (p == end) { return {}; }
    while (p < end) {
      if (!is_digit(data[p])) { return {}; }
      int32_t const digit = data[p++] - '0';
      if (explicit_exponent < 10000) {
        explicit_exponent = explicit_exponent * 10 + digit;
      }
    }
  }
  if (exponent_negative) { explicit_exponent = -explicit_exponent; }
  int32_t const exponent10 = explicit_exponent - fraction_digits;

  uint64_t result_bits = 0;
  if (significand != 0) {
    if (exponent10 > 400) {
      result_bits = 0x7ff0000000000000ull;
    } else if (exponent10 < -400) {
      result_bits = 0;
    } else {
      double const seed = ::cudf::strings::detail::stod(
          ::cudf::string_view(data + begin, end - begin));
      union {
        double value;
        uint64_t bits;
      } raw{seed};
      uint64_t candidate = raw.bits & 0x7fffffffffffffffull;
      if (candidate >= 0x7ff0000000000000ull) {
        candidate = 0x7fefffffffffffffull;
      }
      bool settled = false;
      for (int iteration = 0; iteration < 64 && !settled; ++iteration) {
        if (candidate == 0) {
          uint64_t mid_sig;
          int32_t mid_exp;
          binary_midpoint(0, 1, mid_sig, mid_exp);
          int const cmp = compare_decimal_to_midpoint(
              significand, exponent10, mid_sig, mid_exp);
          if (cmp > 0) {
            candidate = 1;
          } else {
            settled = true; // ties choose even zero
          }
          continue;
        }

        uint64_t lower_sig;
        int32_t lower_exp;
        binary_midpoint(candidate - 1, candidate, lower_sig, lower_exp);
        int const lower_cmp = compare_decimal_to_midpoint(
            significand, exponent10, lower_sig, lower_exp);
        if (lower_cmp < 0 || (lower_cmp == 0 && (candidate & 1) != 0)) {
          --candidate;
          continue;
        }

        uint64_t upper_sig;
        int32_t upper_exp;
        binary_midpoint(candidate, candidate + 1, upper_sig, upper_exp);
        int const upper_cmp = compare_decimal_to_midpoint(
            significand, exponent10, upper_sig, upper_exp);
        if (upper_cmp > 0 || (upper_cmp == 0 && (candidate & 1) != 0)) {
          ++candidate;
          if (candidate >= 0x7ff0000000000000ull) {
            candidate = 0x7ff0000000000000ull;
            settled = true;
          }
          continue;
        }
        settled = true;
      }
      result_bits = candidate;
    }
  }
  if (negative) { result_bits |= 0x8000000000000000ull; }
  union {
    uint64_t bits;
    double value;
  } result{result_bits};
  return ParsedDouble{true, result.value};
}

class Parser {
 public:
  __device__ Parser(
      char const* data,
      int32_t length,
      DevicePathInstruction const* path,
      int32_t path_length,
      char const* path_names,
      ParserFrame* frames)
      : data_(data),
        length_(length),
        path_(path),
        path_length_(path_length),
        path_names_(path_names),
        frames_(frames)
  {
  }

  __device__ Selection parse_root()
  {
    skip_json_space();
    if (position_ == length_) {
      // json-smart readFirst treats a blank root as an unquoted empty string.
      return path_length_ == 0
                 ? scalar(ScalarType::STRING, position_, position_, false)
                 : Selection{};
    }

    int32_t depth = 0;
    int32_t next_path_index = 0;
    bool next_evaluate = true;
    StopContext next_context = StopContext::ROOT;
    bool need_value = true;
    bool have_completed = false;
    Selection completed;

    while (!failed_) {
      if (need_value) {
        have_completed = begin_value(
            next_path_index,
            next_evaluate,
            next_context,
            depth,
            completed);
        need_value = false;
        if (failed_) { break; }
      }

      if (have_completed) {
        if (depth == 0) {
          // MODE_PERMISSIVE deliberately ignores bytes after the first root.
          return completed;
        }
        auto& parent = frames_[depth - 1];
        if (parent.pending_match) { parent.selected = completed; }
        if (parent.kind == ContainerKind::ARRAY) { ++parent.element_index; }
        parent.pending_match = false;

        skip_ascii_space();
        if (position_ >= length_) {
          failed_ = true;
          break;
        }
        char const close =
            parent.kind == ContainerKind::OBJECT ? '}' : ']';
        if (data_[position_] == close) {
          ++position_;
          completed = parent.selected;
          --depth;
          have_completed = true;
          continue;
        }
        if (data_[position_] != ',') {
          failed_ = true;
          break;
        }
        ++position_;
        have_completed = false;
      }

      auto& frame = frames_[depth - 1];
      skip_json_space();
      if (position_ >= length_) {
        failed_ = true;
        break;
      }
      char const c = data_[position_];
      char const close = frame.kind == ContainerKind::OBJECT ? '}' : ']';
      if (c == close) {
        ++position_;
        completed = frame.selected;
        --depth;
        have_completed = true;
        continue;
      }
      if (c == ',') {
        ++position_; // leading, duplicate and trailing commas are accepted
        continue;
      }

      if (frame.kind == ContainerKind::OBJECT) {
        if (c == ':' || c == ']' || c == '[' || c == '{') {
          failed_ = true;
          break;
        }
        int32_t key_begin = 0;
        int32_t key_end = 0;
        bool const quoted = c == '"' || c == '\'';
        if (quoted) {
          if (!parse_quoted(key_begin, key_end)) { break; }
        } else {
          key_begin = position_;
          while (position_ < length_ && data_[position_] != ':') {
            ++position_;
          }
          key_end = position_;
          trim_span(key_begin, key_end);
        }
        skip_ascii_space();
        if (position_ >= length_ || data_[position_] != ':') {
          failed_ = true;
          break;
        }
        ++position_;
        if (position_ >= length_) {
          failed_ = true;
          break;
        }

        bool const can_descend =
            frame.evaluate && frame.path_index < path_length_ &&
            path_[frame.path_index].type == PathInstructionType::NAMED;
        bool const match =
            can_descend &&
            key_matches(key_begin, key_end, quoted, frame.path_index);
        frame.pending_match = match;
        next_path_index = frame.path_index + 1;
        next_evaluate = match;
        next_context = StopContext::OBJECT;
      } else {
        if (c == ':' || c == '}') {
          failed_ = true;
          break;
        }
        bool const can_descend =
            frame.evaluate && frame.path_index < path_length_ &&
            path_[frame.path_index].type == PathInstructionType::INDEX;
        bool const match = can_descend &&
                           frame.element_index == path_[frame.path_index].index;
        frame.pending_match = match;
        next_path_index = frame.path_index + 1;
        next_evaluate = match;
        next_context = StopContext::ARRAY;
      }
      need_value = true;
    }
    return invalid();
  }

  __device__ int32_t render(Selection const& selected, char* output) const
  {
    if (selected.status != ExtractStatus::SCALAR) { return 0; }
    switch (selected.type) {
      case ScalarType::STRING:
        if (!selected.quoted) {
          auto const size = selected.end - selected.begin;
          if (output != nullptr) {
            for (int32_t i = 0; i < size; ++i) {
              output[i] = data_[selected.begin + i];
            }
          }
          return size;
        } else {
          DecodedSink sink{output, nullptr, 0};
          return decode_quoted(data_, selected.begin, selected.end, sink)
                     ? sink.count
                     : -1;
        }
      case ScalarType::BOOLEAN: {
        bool const is_true = selected.end - selected.begin == 4 &&
                             data_[selected.begin] == 't';
        char const* text = is_true ? "true" : "false";
        int32_t size = is_true ? 4 : 5;
        if (output != nullptr) {
          for (int32_t i = 0; i < size; ++i) { output[i] = text[i]; }
        }
        return size;
      }
      case ScalarType::INTEGER:
        return render_integer(selected.begin, selected.end, output);
      case ScalarType::FLOATING:
        return render_float(selected.begin, selected.end, output);
      case ScalarType::NONE: return -1;
    }
    return -1;
  }

 private:
  char const* data_;
  int32_t length_;
  int32_t position_{0};
  DevicePathInstruction const* path_;
  int32_t path_length_;
  char const* path_names_;
  ParserFrame* frames_;
  bool failed_{false};

  __device__ Selection invalid() const
  {
    Selection s;
    s.status = ExtractStatus::INVALID_JSON;
    return s;
  }

  __device__ Selection scalar(
      ScalarType type, int32_t begin, int32_t end, bool quoted) const
  {
    Selection s;
    s.status = ExtractStatus::SCALAR;
    s.type = type;
    s.begin = begin;
    s.end = end;
    s.quoted = quoted;
    return s;
  }

  __device__ Selection selected_self(
      int32_t path_index,
      bool evaluate,
      ExtractStatus status,
      ScalarType type = ScalarType::NONE,
      int32_t begin = 0,
      int32_t end = 0,
      bool quoted = false) const
  {
    if (!evaluate || path_index != path_length_) { return Selection{}; }
    Selection s;
    s.status = status;
    s.type = type;
    s.begin = begin;
    s.end = end;
    s.quoted = quoted;
    return s;
  }

  __device__ void skip_json_space()
  {
    while (position_ < length_ && is_json_space(data_[position_])) { ++position_; }
  }

  // json-smart skipSpace and String.trim accept every ASCII char <= U+0020.
  __device__ void skip_ascii_space()
  {
    while (position_ < length_ &&
           static_cast<unsigned char>(data_[position_]) <= 0x20) {
      ++position_;
    }
  }

  __device__ bool is_stop(char c, StopContext context) const
  {
    if (context == StopContext::ROOT) { return false; }
    if (context == StopContext::OBJECT) { return c == ',' || c == '}'; }
    return c == ',' || c == ']';
  }

  __device__ void trim_span(int32_t& begin, int32_t& end) const
  {
    while (begin < end && static_cast<unsigned char>(data_[begin]) <= 0x20) {
      ++begin;
    }
    while (end > begin && static_cast<unsigned char>(data_[end - 1]) <= 0x20) {
      --end;
    }
  }

  __device__ bool begin_value(
      int32_t path_index,
      bool evaluate,
      StopContext context,
      int32_t& depth,
      Selection& completed)
  {
    skip_json_space();
    if (position_ >= length_) {
      failed_ = true;
      return false;
    }
    char const c = data_[position_];
    if (c == ':' || c == '}' || c == ']') {
      failed_ = true;
      return false;
    }
    if (c == '{' || c == '[') {
      if (depth >= MAX_JSON_DEPTH) {
        failed_ = true;
        return false;
      }
      auto& frame = frames_[depth++];
      frame = ParserFrame{};
      frame.path_index = path_index;
      frame.kind = c == '{' ? ContainerKind::OBJECT : ContainerKind::ARRAY;
      frame.evaluate = evaluate;
      if (evaluate && path_index == path_length_) {
        frame.selected.status = ExtractStatus::CONTAINER;
      }
      ++position_;
      return false;
    }
    if (c == '"' || c == '\'') {
      int32_t begin = 0;
      int32_t end = 0;
      if (!parse_quoted(begin, end)) { return false; }
      completed = selected_self(
          path_index,
          evaluate,
          ExtractStatus::SCALAR,
          ScalarType::STRING,
          begin,
          end,
          true);
    } else if (is_digit(c) || c == '-') {
      completed = parse_number_or_string(path_index, evaluate, context);
    } else {
      completed = parse_literal_or_string(path_index, evaluate, context);
    }
    return !failed_;
  }

  __device__ bool parse_quoted(int32_t& content_begin, int32_t& content_end)
  {
    char const separator = data_[position_++];
    content_begin = position_;
    while (position_ < length_) {
      char const c = data_[position_++];
      if (c == separator) {
        content_end = position_ - 1;
        return true;
      }
      if (c != '\\') { continue; }
      if (position_ >= length_) {
        failed_ = true;
        return false;
      }
      char const escaped = data_[position_++];
      int count = escaped == 'u' ? 4 : (escaped == 'x' ? 2 : 0);
      for (int i = 0; i < count; ++i) {
        if (position_ >= length_ || hex_value(data_[position_]) < 0) {
          failed_ = true;
          return false;
        }
        ++position_;
      }
      // Unknown escapes are valid in MODE_PERMISSIVE and are dropped later.
    }
    failed_ = true;
    return false;
  }

  __device__ bool key_matches(
      int32_t begin, int32_t end, bool quoted, int32_t path_index) const
  {
    if (path_index >= path_length_ ||
        path_[path_index].type != PathInstructionType::NAMED) {
      return false;
    }
    auto const& inst = path_[path_index];
    char const* expected = path_names_ + inst.name_offset;
    if (!quoted) {
      auto const size = end - begin;
      if (size != inst.name_length) { return false; }
      for (int32_t i = 0; i < size; ++i) {
        if (data_[begin + i] != expected[i]) { return false; }
      }
      return true;
    }
    DecodedSink sink{nullptr, expected, inst.name_length};
    return decode_quoted(data_, begin, end, sink) && sink.matches;
  }

  __device__ int32_t consume_token(StopContext context)
  {
    while (position_ < length_ && !is_stop(data_[position_], context)) {
      ++position_;
    }
    return position_;
  }

  __device__ Selection string_span(
      int32_t path_index, bool evaluate, int32_t begin, int32_t end) const
  {
    trim_span(begin, end);
    return selected_self(
        path_index,
        evaluate,
        ExtractStatus::SCALAR,
        ScalarType::STRING,
        begin,
        end,
        false);
  }

  __device__ Selection parse_literal_or_string(
      int32_t path_index, bool evaluate, StopContext context)
  {
    int32_t begin = position_;
    int32_t end = consume_token(context);
    trim_span(begin, end);
    auto const equals = [&](char const* literal, int32_t size) {
      if (end - begin != size) { return false; }
      for (int32_t i = 0; i < size; ++i) {
        if (data_[begin + i] != literal[i]) { return false; }
      }
      return true;
    };
    if (equals("null", 4)) {
      return selected_self(path_index, evaluate, ExtractStatus::JSON_NULL);
    }
    if (equals("true", 4) || equals("false", 5)) {
      return selected_self(
          path_index,
          evaluate,
          ExtractStatus::SCALAR,
          ScalarType::BOOLEAN,
          begin,
          end);
    }
    if (equals("NaN", 3)) {
      return selected_self(
          path_index,
          evaluate,
          ExtractStatus::SCALAR,
          ScalarType::FLOATING,
          begin,
          end);
    }
    return string_span(path_index, evaluate, begin, end);
  }

  __device__ Selection parse_number_or_string(
      int32_t path_index, bool evaluate, StopContext context)
  {
    int32_t const begin = position_;
    ++position_; // first digit or '-'
    while (position_ < length_ && is_digit(data_[position_])) { ++position_; }

    bool floating = false;
    if (position_ < length_ && data_[position_] == '.') {
      floating = true;
      ++position_;
      while (position_ < length_ && is_digit(data_[position_])) { ++position_; }
    }
    if (position_ < length_ &&
        (data_[position_] == 'e' || data_[position_] == 'E')) {
      floating = true;
      ++position_;
      if (position_ < length_ &&
          (data_[position_] == '+' || data_[position_] == '-' ||
           is_digit(data_[position_]))) {
        ++position_;
        while (position_ < length_ && is_digit(data_[position_])) { ++position_; }
      } else {
        int32_t end = consume_token(context);
        return string_span(path_index, evaluate, begin, end);
      }
    }

    skip_ascii_space();
    if (position_ < length_ && !is_stop(data_[position_], context)) {
      int32_t end = consume_token(context);
      return string_span(path_index, evaluate, begin, end);
    }
    int32_t end = position_;
    int32_t trimmed_begin = begin;
    trim_span(trimmed_begin, end);
    if (trimmed_begin >= end) {
      failed_ = true;
      return invalid();
    }
    return selected_self(
        path_index,
        evaluate,
        ExtractStatus::SCALAR,
        floating ? ScalarType::FLOATING : ScalarType::INTEGER,
        trimmed_begin,
        end);
  }

  __device__ int32_t render_integer(int32_t begin, int32_t end, char* output) const
  {
    bool negative = begin < end && data_[begin] == '-';
    int32_t p = begin + (negative ? 1 : 0);
    while (p < end && data_[p] == '0') { ++p; }
    if (p == end) {
      if (output != nullptr) { output[0] = '0'; }
      return 1;
    }
    for (int32_t i = p; i < end; ++i) {
      if (!is_digit(data_[i])) { return -1; }
    }
    int32_t size = end - p + (negative ? 1 : 0);
    if (output != nullptr) {
      int32_t out = 0;
      if (negative) { output[out++] = '-'; }
      while (p < end) { output[out++] = data_[p++]; }
    }
    return size;
  }

  struct DecimalLayout {
    bool valid{false};
    bool negative{false};
    bool zero{true};
    int32_t mantissa_end{0};
    int32_t first_significant{0};
    int32_t precision{1};
    int32_t scale{0};
    int64_t adjusted{0};
  };

  __device__ DecimalLayout decimal_layout(int32_t begin, int32_t end) const
  {
    DecimalLayout layout;
    int32_t p = begin;
    if (p < end && data_[p] == '-') {
      layout.negative = true;
      ++p;
    }
    int32_t mantissa_begin = p;
    int32_t dot = -1;
    int32_t exponent_pos = end;
    bool any_digit = false;
    for (; p < end; ++p) {
      char const c = data_[p];
      if (is_digit(c)) {
        any_digit = true;
      } else if (c == '.' && dot < 0) {
        dot = p;
      } else if (c == 'e' || c == 'E') {
        exponent_pos = p;
        break;
      } else {
        return layout;
      }
    }
    if (!any_digit) { return layout; }
    layout.mantissa_end = exponent_pos;

    int64_t exponent = 0;
    if (exponent_pos < end) {
      p = exponent_pos + 1;
      bool exponent_negative = false;
      if (p < end && (data_[p] == '+' || data_[p] == '-')) {
        exponent_negative = data_[p] == '-';
        ++p;
      }
      if (p == end) { return layout; }
      while (p < end) {
        if (!is_digit(data_[p])) { return layout; }
        int digit = data_[p++] - '0';
        if (exponent > (static_cast<int64_t>(std::numeric_limits<int32_t>::max()) - digit) / 10) {
          return layout;
        }
        exponent = exponent * 10 + digit;
      }
      if (exponent_negative) { exponent = -exponent; }
    }

    int32_t fraction_digits = dot < 0 ? 0 : exponent_pos - dot - 1;
    int64_t scale64 = static_cast<int64_t>(fraction_digits) - exponent;
    if (scale64 < std::numeric_limits<int32_t>::min() ||
        scale64 > std::numeric_limits<int32_t>::max()) {
      return layout;
    }
    layout.scale = static_cast<int32_t>(scale64);

    int32_t digit_count = 0;
    int32_t first_nonzero = -1;
    for (p = mantissa_begin; p < exponent_pos; ++p) {
      if (data_[p] == '.') { continue; }
      if (first_nonzero < 0 && data_[p] != '0') { first_nonzero = p; }
      if (first_nonzero >= 0) { ++digit_count; }
    }
    if (first_nonzero >= 0) {
      layout.zero = false;
      layout.first_significant = first_nonzero;
      layout.precision = digit_count;
    } else {
      layout.zero = true;
      layout.negative = false; // BigDecimal has no negative zero
      layout.first_significant = exponent_pos;
      layout.precision = 1;
    }
    layout.adjusted = -static_cast<int64_t>(layout.scale) + layout.precision - 1;
    layout.valid = true;
    return layout;
  }

  __device__ int32_t exponent_digits(uint64_t value) const
  {
    int32_t digits = 1;
    while (value >= 10) {
      value /= 10;
      ++digits;
    }
    return digits;
  }

  __device__ void write_exponent(int64_t value, char* output, int32_t& out) const
  {
    bool const negative = value < 0;
    uint64_t magnitude = negative ? static_cast<uint64_t>(-value)
                                  : static_cast<uint64_t>(value);
    output[out++] = negative ? '-' : '+';
    int32_t digits = exponent_digits(magnitude);
    int32_t start = out;
    out += digits;
    for (int32_t i = digits - 1; i >= 0; --i) {
      output[start + i] = static_cast<char>('0' + magnitude % 10);
      magnitude /= 10;
    }
  }

  __device__ void write_significant_digits(
      DecimalLayout const& layout,
      int32_t decimal_after,
      char* output,
      int32_t& out) const
  {
    if (layout.zero) {
      output[out++] = '0';
      return;
    }
    int32_t emitted = 0;
    for (int32_t p = layout.first_significant; p < layout.mantissa_end; ++p) {
      if (data_[p] == '.') { continue; }
      if (decimal_after >= 0 && emitted == decimal_after) { output[out++] = '.'; }
      output[out++] = data_[p];
      ++emitted;
    }
  }

  __device__ int32_t render_big_decimal(int32_t begin, int32_t end, char* output) const
  {
    auto const layout = decimal_layout(begin, end);
    if (!layout.valid) { return -1; }
    bool const plain = layout.scale >= 0 && layout.adjusted >= -6;
    int32_t const sign_size = layout.negative ? 1 : 0;
    if (plain) {
      int64_t const insertion = static_cast<int64_t>(layout.precision) - layout.scale;
      int64_t size = sign_size;
      if (layout.scale == 0) {
        size += layout.precision;
      } else if (insertion > 0) {
        size += layout.precision + 1;
      } else {
        size += 2 - insertion + layout.precision;
      }
      if (size > std::numeric_limits<int32_t>::max()) { return -1; }
      if (output != nullptr) {
        int32_t out = 0;
        if (layout.negative) { output[out++] = '-'; }
        if (layout.scale == 0) {
          write_significant_digits(layout, -1, output, out);
        } else if (insertion > 0) {
          write_significant_digits(
              layout, static_cast<int32_t>(insertion), output, out);
        } else {
          output[out++] = '0';
          output[out++] = '.';
          for (int64_t i = 0; i < -insertion; ++i) { output[out++] = '0'; }
          write_significant_digits(layout, -1, output, out);
        }
      }
      return static_cast<int32_t>(size);
    }

    uint64_t const magnitude = layout.adjusted < 0
                                   ? static_cast<uint64_t>(-layout.adjusted)
                                   : static_cast<uint64_t>(layout.adjusted);
    int32_t size = sign_size + layout.precision +
                   (layout.precision > 1 ? 1 : 0) + 1 + 1 +
                   exponent_digits(magnitude);
    if (output != nullptr) {
      int32_t out = 0;
      if (layout.negative) { output[out++] = '-'; }
      write_significant_digits(
          layout, layout.precision > 1 ? 1 : -1, output, out);
      output[out++] = 'E';
      write_exponent(layout.adjusted, output, out);
    }
    return size;
  }

  __device__ int32_t render_float(int32_t begin, int32_t end, char* output) const
  {
    if (end - begin == 3 && data_[begin] == 'N' && data_[begin + 1] == 'a' &&
        data_[begin + 2] == 'N') {
      if (output != nullptr) {
        output[0] = 'N';
        output[1] = 'a';
        output[2] = 'N';
      }
      return 3;
    }
    // json-smart 2.2.1 uses BigDecimal precisely when the trimmed floating
    // token length exceeds 18, preserving scale and trailing zeroes.
    if (end - begin > 18) { return render_big_decimal(begin, end, output); }

    auto const parsed = parse_java_double(data_, begin, end);
    if (!parsed.valid) { return -1; }
    double const value = parsed.value;
    union {
      double value;
      uint64_t bits;
    } raw{value};
    bool const negative = (raw.bits >> 63) != 0;
    uint64_t const magnitude_bits = raw.bits & 0x7fffffffffffffffull;
    if (magnitude_bits == 1) {
      char const* text = "4.9E-324";
      int32_t const size = 8 + (negative ? 1 : 0);
      if (output != nullptr) {
        int32_t out = 0;
        if (negative) { output[out++] = '-'; }
        for (int32_t i = 0; i < 8; ++i) { output[out++] = text[i]; }
      }
      return size;
    }
    if (magnitude_bits == 0x44b52d02c7e14af6ull) {
      char const* text = "9.999999999999999E22";
      int32_t const size = 20 + (negative ? 1 : 0);
      if (output != nullptr) {
        int32_t out = 0;
        if (negative) { output[out++] = '-'; }
        for (int32_t i = 0; i < 20; ++i) { output[out++] = text[i]; }
      }
      return size;
    }
    if (cuda::std::isinf(value)) {
      char const* text = negative ? "-Infinity" : "Infinity";
      int32_t const size = negative ? 9 : 8;
      if (output != nullptr) {
        for (int32_t i = 0; i < size; ++i) { output[i] = text[i]; }
      }
      return size;
    }
    return spark_rapids_jni::ftos_converter::double_normalization(value, output);
  }
};

CUDF_KERNEL void java_utf8_size_kernel(
    ::cudf::column_device_view input,
    ::cudf::size_type* sizes,
    uint8_t* valid)
{
  auto const row = static_cast<::cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.size()) { return; }
  if (input.is_null(row)) {
    sizes[row] = 0;
    valid[row] = 1;
    return;
  }
  auto const text = input.element<::cudf::string_view>(row);
  int32_t const size =
      normalize_java_utf8(text.data(), text.size_bytes(), nullptr);
  sizes[row] = size < 0 ? 0 : size;
  valid[row] = size < 0 ? 0 : 1;
}

CUDF_KERNEL void java_utf8_write_kernel(
    ::cudf::column_device_view input,
    uint8_t const* valid,
    ::cudf::size_type const* offsets,
    char* chars)
{
  auto const row = static_cast<::cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.size() || input.is_null(row) || valid[row] == 0) { return; }
  auto const text = input.element<::cudf::string_view>(row);
  normalize_java_utf8(text.data(), text.size_bytes(), chars + offsets[row]);
}

CUDF_KERNEL void size_kernel(
    ::cudf::column_device_view input,
    uint8_t const* normalized_valid,
    DevicePathInstruction const* path,
    int32_t path_length,
    char const* path_names,
    ParserFrame* frames,
    ::cudf::size_type* sizes,
    int8_t* statuses,
    int8_t* types)
{
  auto const row = static_cast<::cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.size()) { return; }
  if (normalized_valid[row] == 0) {
    sizes[row] = 0;
    statuses[row] = static_cast<int8_t>(ExtractStatus::INVALID_JSON);
    types[row] = static_cast<int8_t>(ScalarType::NONE);
    return;
  }
  if (input.is_null(row)) {
    sizes[row] = 0;
    statuses[row] = static_cast<int8_t>(ExtractStatus::INPUT_NULL);
    types[row] = static_cast<int8_t>(ScalarType::NONE);
    return;
  }
  auto const text = input.element<::cudf::string_view>(row);
  Parser parser(
      text.data(),
      text.size_bytes(),
      path,
      path_length,
      path_names,
      frames + static_cast<size_t>(row) * MAX_JSON_DEPTH);
  auto selected = parser.parse_root();
  int32_t size = parser.render(selected, nullptr);
  if (size < 0) {
    selected = Selection{};
    selected.status = ExtractStatus::INVALID_JSON;
    size = 0;
  }
  sizes[row] = size;
  statuses[row] = static_cast<int8_t>(selected.status);
  types[row] = static_cast<int8_t>(selected.type);
}

CUDF_KERNEL void write_kernel(
    ::cudf::column_device_view input,
    uint8_t const* normalized_valid,
    DevicePathInstruction const* path,
    int32_t path_length,
    char const* path_names,
    ParserFrame* frames,
    ::cudf::size_type const* offsets,
    char* chars)
{
  auto const row = static_cast<::cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.size() || input.is_null(row) ||
      normalized_valid[row] == 0) {
    return;
  }
  auto const text = input.element<::cudf::string_view>(row);
  Parser parser(
      text.data(),
      text.size_bytes(),
      path,
      path_length,
      path_names,
      frames + static_cast<size_t>(row) * MAX_JSON_DEPTH);
  auto const selected = parser.parse_root();
  if (selected.status == ExtractStatus::SCALAR) {
    parser.render(selected, chars + offsets[row]);
  }
}

} // namespace

ExtractResult extract_scalar(
    ::cudf::strings_column_view const& input,
    std::vector<PathInstruction> const& path,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
      path.size() <= MAX_PATH_DEPTH,
      "NfJsonExtractScalar prototype path exceeds depth 16");
  std::vector<DevicePathInstruction> host_path;
  std::string names;
  host_path.reserve(path.size());
  for (auto const& instruction : path) {
    if (instruction.type == PathInstructionType::NAMED) {
      CUDF_EXPECTS(!instruction.name.empty(), "Named JSON path step is empty");
      CUDF_EXPECTS(
          instruction.name.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "Named JSON path step is too large");
      auto const offset = static_cast<int32_t>(names.size());
      names += instruction.name;
      host_path.push_back(DevicePathInstruction{
          instruction.type,
          offset,
          static_cast<int32_t>(instruction.name.size()),
          -1});
    } else {
      CUDF_EXPECTS(instruction.index >= 0, "JSON array index must be non-negative");
      host_path.push_back(
          DevicePathInstruction{instruction.type, 0, 0, instruction.index});
    }
  }

  auto statuses = ::cudf::make_numeric_column(
      ::cudf::data_type{::cudf::type_id::INT8},
      input.size(),
      ::cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto types = ::cudf::make_numeric_column(
      ::cudf::data_type{::cudf::type_id::INT8},
      input.size(),
      ::cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (input.is_empty()) {
    return ExtractResult{
        ::cudf::make_empty_column(::cudf::type_id::STRING),
        std::move(statuses),
        std::move(types)};
  }

  rmm::device_uvector<DevicePathInstruction> device_path(host_path.size(), stream, mr);
  if (!host_path.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        device_path.data(),
        host_path.data(),
        host_path.size() * sizeof(DevicePathInstruction),
        cudaMemcpyHostToDevice,
        stream.value()));
  }
  rmm::device_buffer device_names(names.size(), stream, mr);
  if (!names.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        device_names.data(),
        names.data(),
        names.size(),
        cudaMemcpyHostToDevice,
        stream.value()));
  }

  rmm::device_uvector<::cudf::size_type> sizes(input.size(), stream, mr);
  auto input_device = ::cudf::column_device_view::create(input.parent(), stream);
  constexpr int32_t block_size = 128;
  auto const blocks = (input.size() + block_size - 1) / block_size;

  rmm::device_uvector<uint8_t> normalized_valid(input.size(), stream, mr);
  java_utf8_size_kernel<<<blocks, block_size, 0, stream.value()>>>(
      *input_device, sizes.data(), normalized_valid.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  auto [normalized_offsets, normalized_bytes] =
      ::cudf::strings::detail::make_offsets_child_column(
          sizes.begin(), sizes.end(), stream, mr);
  CUDF_EXPECTS(
      normalized_offsets->type().id() == ::cudf::type_id::INT32,
      "NfJsonExtractScalar normalized input exceeds the 32-bit offset domain");
  rmm::device_buffer normalized_chars(normalized_bytes, stream, mr);
  java_utf8_write_kernel<<<blocks, block_size, 0, stream.value()>>>(
      *input_device,
      normalized_valid.data(),
      normalized_offsets->view().data<::cudf::size_type>(),
      static_cast<char*>(normalized_chars.data()));
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  auto normalized_input = ::cudf::make_strings_column(
      input.size(),
      std::move(normalized_offsets),
      std::move(normalized_chars),
      input.parent().null_count(),
      ::cudf::copy_bitmask(input.parent(), stream, mr));
  auto normalized_device =
      ::cudf::column_device_view::create(normalized_input->view(), stream);

  auto const row_count = static_cast<size_t>(input.size());
  CUDF_EXPECTS(
      row_count <= std::numeric_limits<size_t>::max() / MAX_JSON_DEPTH,
      "NfJsonExtractScalar parser workspace element count overflow");
  auto const workspace_elements = row_count * MAX_JSON_DEPTH;
  CUDF_EXPECTS(
      workspace_elements <=
          std::numeric_limits<size_t>::max() / sizeof(ParserFrame),
      "NfJsonExtractScalar parser workspace byte count overflow");
  rmm::device_uvector<ParserFrame> parser_workspace(
      workspace_elements, stream, mr);

  size_kernel<<<blocks, block_size, 0, stream.value()>>>(
      *normalized_device,
      normalized_valid.data(),
      device_path.data(),
      static_cast<int32_t>(device_path.size()),
      static_cast<char const*>(device_names.data()),
      parser_workspace.data(),
      sizes.data(),
      statuses->mutable_view().data<int8_t>(),
      types->mutable_view().data<int8_t>());
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  auto [offsets, total_bytes] = ::cudf::strings::detail::make_offsets_child_column(
      sizes.begin(), sizes.end(), stream, mr);
  CUDF_EXPECTS(
      offsets->type().id() == ::cudf::type_id::INT32,
      "NfJsonExtractScalar output exceeds the 32-bit offset domain");
  rmm::device_buffer chars(total_bytes, stream, mr);
  write_kernel<<<blocks, block_size, 0, stream.value()>>>(
      *normalized_device,
      normalized_valid.data(),
      device_path.data(),
      static_cast<int32_t>(device_path.size()),
      static_cast<char const*>(device_names.data()),
      parser_workspace.data(),
      offsets->view().data<::cudf::size_type>(),
      static_cast<char*>(chars.data()));
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  auto const* status_data = statuses->view().data<int8_t>();
  auto [null_mask, null_count] = ::cudf::detail::valid_if(
      status_data,
      status_data + input.size(),
      [] __device__(int8_t status) {
        return status == static_cast<int8_t>(ExtractStatus::SCALAR);
      },
      stream,
      mr);
  auto values = ::cudf::make_strings_column(
      input.size(),
      std::move(offsets),
      std::move(chars),
      null_count,
      std::move(null_mask));

  return ExtractResult{std::move(values), std::move(statuses), std::move(types)};
}

} // namespace gluten::nfjson
