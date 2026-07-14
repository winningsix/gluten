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
 *
 * Legacy finite-double boundary behavior is derived from Apache Harmony
 * NumberConverter.java and dblparse.c at commit
 * ee663b95e84093405985d9557a925b651a0f69b3, Apache License 2.0:
 *   https://github.com/apache/harmony-classlib
 * JDK 17's integral-double fast-path semantics were independently
 * reimplemented with exact integer arithmetic after review of:
 *   https://github.com/openjdk/jdk17u/blob/master/src/java.base/share/classes/
 *   jdk/internal/math/FloatingDecimal.java
 * No OpenJDK source code is incorporated.
 */
#include "operators/functions/NetflixJson.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <locale.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace gluten::netflix_json {
namespace {

using boost::multiprecision::cpp_int;

constexpr int32_t kMaxPathDepth = 16;
constexpr int32_t kMaxNestingDepth = 512;
constexpr size_t kMaxPathBytes = 4096;
constexpr size_t kMaxFieldBytes = 128;

bool isFieldStart(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

bool isAsciiDigit(char c) {
  return c >= '0' && c <= '9';
}

bool isFieldPart(char c) {
  return isFieldStart(c) || isAsciiDigit(c);
}

void appendUtf8(uint32_t codePoint, std::string& output) {
  if (codePoint <= 0x7f) {
    output.push_back(static_cast<char>(codePoint));
  } else if (codePoint <= 0x7ff) {
    output.push_back(static_cast<char>(0xc0 | (codePoint >> 6)));
    output.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
  } else if (codePoint <= 0xffff) {
    output.push_back(static_cast<char>(0xe0 | (codePoint >> 12)));
    output.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
  } else {
    output.push_back(static_cast<char>(0xf0 | (codePoint >> 18)));
    output.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
  }
}

int32_t utf8SequenceLength(unsigned char first) {
  if (first < 0x80) {
    return 1;
  }
  if (first >= 0xc2 && first <= 0xdf) {
    return 2;
  }
  if (first >= 0xe0 && first <= 0xef) {
    return 3;
  }
  if (first >= 0xf0 && first <= 0xf4) {
    return 4;
  }
  return -1;
}

bool isUtf8Continuation(unsigned char byte) {
  return (byte & 0xc0) == 0x80;
}

struct JavaUtf8Unit {
  size_t consumed;
  bool valid;
};

/**
 * Returns the malformed-input unit consumed by Java 17's UTF-8 decoder.
 *
 * This is an oracle-derived behavioral implementation, not copied OpenJDK
 * code. A valid prefix before a bad or missing third/fourth byte becomes one
 * U+FFFD. An illegal second byte makes only the lead byte malformed. A fully
 * encoded UTF-8 surrogate is consumed as one malformed unit.
 */
JavaUtf8Unit javaUtf8Unit(std::string_view input, size_t position) {
  const auto first = static_cast<unsigned char>(input[position]);
  if (first <= 0x7f) {
    return {1, true};
  }
  if (first >= 0xc2 && first <= 0xdf) {
    if (position + 1 < input.size() && isUtf8Continuation(static_cast<unsigned char>(input[position + 1]))) {
      return {2, true};
    }
    return {1, false};
  }
  if (first >= 0xe0 && first <= 0xef) {
    if (position + 1 >= input.size()) {
      return {1, false};
    }
    const auto second = static_cast<unsigned char>(input[position + 1]);
    const bool secondValid = isUtf8Continuation(second) && !(first == 0xe0 && second < 0xa0);
    if (!secondValid) {
      return {1, false};
    }
    if (position + 2 >= input.size()) {
      return {2, false};
    }
    if (!isUtf8Continuation(static_cast<unsigned char>(input[position + 2]))) {
      return {2, false};
    }
    return {3, !(first == 0xed && second >= 0xa0)};
  }
  if (first >= 0xf0 && first <= 0xf4) {
    if (position + 1 >= input.size()) {
      return {1, false};
    }
    const auto second = static_cast<unsigned char>(input[position + 1]);
    const bool secondValid =
        isUtf8Continuation(second) && !(first == 0xf0 && second < 0x90) && !(first == 0xf4 && second >= 0x90);
    if (!secondValid) {
      return {1, false};
    }
    if (position + 2 >= input.size()) {
      return {2, false};
    }
    if (!isUtf8Continuation(static_cast<unsigned char>(input[position + 2]))) {
      return {2, false};
    }
    if (position + 3 >= input.size()) {
      return {3, false};
    }
    if (!isUtf8Continuation(static_cast<unsigned char>(input[position + 3]))) {
      return {3, false};
    }
    return {4, true};
  }
  return {1, false};
}

/** Mirrors UTF8String.toString followed by String.getBytes(UTF_8). */
std::string normalizeJavaUtf8(std::string_view input) {
  std::string output;
  output.reserve(input.size());
  size_t position = 0;
  while (position < input.size()) {
    const auto unit = javaUtf8Unit(input, position);
    if (unit.valid) {
      output.append(input.data() + position, unit.consumed);
    } else {
      appendUtf8(0xfffd, output);
    }
    position += unit.consumed;
  }
  return output;
}

cpp_int powerOfTen(int32_t exponent) {
  cpp_int result = 1;
  for (int32_t i = 0; i < exponent; ++i) {
    result *= 10;
  }
  return result;
}

struct LegacyDoubleDigits {
  std::string digits;
  int32_t firstExponent;
};

std::optional<LegacyDoubleDigits> legacyEasyIntegerDigits(uint64_t absoluteBits) {
  constexpr uint64_t kFractionMask = (uint64_t{1} << 52) - 1;
  constexpr uint64_t kHiddenBit = uint64_t{1} << 52;
  const uint64_t exponentBits = (absoluteBits >> 52) & 0x7ff;
  if (exponentBits == 0) {
    return std::nullopt;
  }
  const int32_t binaryExponent = static_cast<int32_t>(exponentBits) - 1023;
  if (binaryExponent < -21 || binaryExponent > 62) {
    return std::nullopt;
  }
  uint64_t significand = (absoluteBits & kFractionMask) | kHiddenBit;
  uint64_t trailing = significand;
  int32_t trailingZeros = 0;
  while ((trailing & 1) == 0) {
    trailing >>= 1;
    ++trailingZeros;
  }
  const int32_t fractionBits = 53 - trailingZeros;
  if (std::max(0, fractionBits - binaryExponent - 1) != 0) {
    return std::nullopt;
  }

  int32_t insignificantDigits = 0;
  if (binaryExponent > 53) {
    uint64_t insignificant = uint64_t{1} << (binaryExponent - 53 - 1);
    while (insignificant >= 10) {
      insignificant /= 10;
      ++insignificantDigits;
    }
  }
  uint64_t integer = binaryExponent >= 52 ? significand << (binaryExponent - 52) : significand >> (52 - binaryExponent);
  int32_t strippedZeros = 0;
  if (insignificantDigits > 0) {
    uint64_t scale = 1;
    for (int32_t i = 0; i < insignificantDigits; ++i) {
      scale *= 10;
    }
    const uint64_t residue = integer % scale;
    integer /= scale;
    strippedZeros += insignificantDigits;
    if (residue >= scale / 2) {
      ++integer;
    }
  }
  while (integer % 10 == 0) {
    integer /= 10;
    ++strippedZeros;
  }
  auto digits = std::to_string(integer);
  const int32_t firstExponent = strippedZeros + static_cast<int32_t>(digits.size()) - 1;
  return LegacyDoubleDigits{std::move(digits), firstExponent};
}

struct Rational {
  cpp_int numerator;
  cpp_int denominator;
};

Rational binaryRational(uint64_t absoluteBits) {
  const uint64_t fraction = absoluteBits & ((uint64_t{1} << 52) - 1);
  const uint64_t exponentBits = (absoluteBits >> 52) & 0x7ff;
  const uint64_t significand = exponentBits == 0 ? fraction : (uint64_t{1} << 52) | fraction;
  const int32_t binaryExponent = exponentBits == 0 ? -1074 : static_cast<int32_t>(exponentBits) - 1023 - 52;
  return binaryExponent >= 0 ? Rational{cpp_int(significand) << binaryExponent, 1}
                             : Rational{significand, cpp_int(1) << -binaryExponent};
}

Rational midpoint(const Rational& left, const Rational& right) {
  return {
      left.numerator * right.denominator + right.numerator * left.denominator,
      (left.denominator * right.denominator) << 1};
}

int compare(const Rational& left, const Rational& right) {
  const auto difference = left.numerator * right.denominator - right.numerator * left.denominator;
  return difference < 0 ? -1 : (difference > 0 ? 1 : 0);
}

Rational decimalRational(const cpp_int& digits, int32_t exponent) {
  return exponent >= 0 ? Rational{digits * powerOfTen(exponent), 1} : Rational{digits, powerOfTen(-exponent)};
}

std::pair<cpp_int, int32_t> roundedDecimal(const Rational& value, int32_t adjustedExponent, int32_t precision) {
  int32_t decimalExponent = adjustedExponent - precision + 1;
  auto scaled = value;
  if (decimalExponent >= 0) {
    scaled.denominator *= powerOfTen(decimalExponent);
  } else {
    scaled.numerator *= powerOfTen(-decimalExponent);
  }
  cpp_int quotient = scaled.numerator / scaled.denominator;
  const cpp_int remainder = scaled.numerator % scaled.denominator;
  const cpp_int twiceRemainder = remainder << 1;
  if (twiceRemainder > scaled.denominator ||
      (twiceRemainder == scaled.denominator && static_cast<bool>(quotient & 1))) {
    ++quotient;
  }
  if (quotient >= powerOfTen(precision)) {
    quotient /= 10;
    ++decimalExponent;
  }
  return {std::move(quotient), decimalExponent};
}

std::string formatJavaDouble(double value) {
  if (std::isnan(value)) {
    return "NaN";
  }
  if (std::isinf(value)) {
    return std::signbit(value) ? "-Infinity" : "Infinity";
  }
  if (value == 0) {
    return std::signbit(value) ? "-0.0" : "0.0";
  }

  uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  const bool negative = (bits >> 63) != 0;
  const uint64_t absoluteBits = bits & ~(uint64_t{1} << 63);
  LegacyDoubleDigits generated;
  if (auto easy = legacyEasyIntegerDigits(absoluteBits)) {
    generated = std::move(*easy);
  } else {
    const auto exact = binaryRational(absoluteBits);
    const auto previous = binaryRational(absoluteBits - 1);
    const auto lower = midpoint(previous, exact);
    Rational upper;
    constexpr uint64_t kMaxFiniteBits = UINT64_C(0x7fefffffffffffff);
    if (absoluteBits == kMaxFiniteBits) {
      const Rational distance{
          exact.numerator * previous.denominator - previous.numerator * exact.denominator,
          exact.denominator * previous.denominator};
      upper = {
          exact.numerator * (distance.denominator << 1) + distance.numerator * exact.denominator,
          exact.denominator * (distance.denominator << 1)};
    } else {
      upper = midpoint(exact, binaryRational(absoluteBits + 1));
    }

    int32_t adjustedExponent = static_cast<int32_t>(std::floor(std::log10(std::fabs(value))));
    while (compare(exact, decimalRational(1, adjustedExponent)) < 0) {
      --adjustedExponent;
    }
    while (compare(exact, decimalRational(1, adjustedExponent + 1)) >= 0) {
      ++adjustedExponent;
    }
    cpp_int digits;
    int32_t decimalExponent = 0;
    for (int32_t precision = 1; precision <= 17; ++precision) {
      auto candidate = roundedDecimal(exact, adjustedExponent, precision);
      const auto decimal = decimalRational(candidate.first, candidate.second);
      if (compare(decimal, lower) > 0 && compare(decimal, upper) < 0) {
        if (precision == 1) {
          auto twoDigit = roundedDecimal(exact, adjustedExponent, 2);
          const auto twoDigitDecimal = decimalRational(twoDigit.first, twoDigit.second);
          const int32_t firstCandidateExponent =
              candidate.second + static_cast<int32_t>(candidate.first.convert_to<std::string>().size()) - 1;
          const int32_t twoDigitExponent =
              twoDigit.second + static_cast<int32_t>(twoDigit.first.convert_to<std::string>().size()) - 1;
          if (firstCandidateExponent == twoDigitExponent && compare(twoDigitDecimal, lower) > 0 &&
              compare(twoDigitDecimal, upper) < 0) {
            candidate = std::move(twoDigit);
          }
        }
        digits = std::move(candidate.first);
        decimalExponent = candidate.second;
        break;
      }
    }
    if (digits == 0) {
      auto fallback = roundedDecimal(exact, adjustedExponent, 17);
      digits = std::move(fallback.first);
      decimalExponent = fallback.second;
    }
    auto digitText = digits.convert_to<std::string>();
    while (digitText.size() > 1 && digitText.back() == '0') {
      digitText.pop_back();
      ++decimalExponent;
    }
    const int32_t firstExponent = decimalExponent + static_cast<int32_t>(digitText.size()) - 1;
    generated = {std::move(digitText), firstExponent};
  }
  const auto& digitText = generated.digits;
  const int32_t exponent = generated.firstExponent;

  std::string output;
  output.reserve(digitText.size() + 16);
  if (negative) {
    output.push_back('-');
  }
  if (exponent < -3 || exponent >= 7) {
    output.push_back(digitText.front());
    output.push_back('.');
    if (digitText.size() == 1) {
      output.push_back('0');
    } else {
      output.append(digitText.data() + 1, digitText.size() - 1);
    }
    output.push_back('E');
    output.append(std::to_string(exponent));
    return output;
  }

  const int32_t decimalPosition = exponent + 1;
  if (decimalPosition <= 0) {
    output.append("0.");
    output.append(-decimalPosition, '0');
    output.append(digitText);
  } else if (decimalPosition >= static_cast<int32_t>(digitText.size())) {
    output.append(digitText);
    output.append(decimalPosition - digitText.size(), '0');
    output.append(".0");
  } else {
    output.append(digitText.data(), decimalPosition);
    output.push_back('.');
    output.append(digitText.data() + decimalPosition, digitText.size() - decimalPosition);
  }
  return output;
}

std::optional<double> parseDouble(std::string_view token) {
  std::string copy(token);
#if defined(__linux__)
  static locale_t cLocale = newlocale(LC_NUMERIC_MASK, "C", nullptr);
  char* end = nullptr;
  errno = 0;
  const double value = strtod_l(copy.c_str(), &end, cLocale);
#else
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(copy.c_str(), &end);
#endif
  if (end != copy.data() + copy.size()) {
    return std::nullopt;
  }
  return value;
}

std::optional<std::string> formatBigDecimal(std::string_view token) {
  size_t position = 0;
  bool negative = false;
  if (position < token.size() && token[position] == '-') {
    negative = true;
    ++position;
  }
  const size_t mantissaBegin = position;
  size_t dot = std::string_view::npos;
  size_t exponentPosition = token.size();
  bool anyDigit = false;
  for (; position < token.size(); ++position) {
    const char c = token[position];
    if (isAsciiDigit(c)) {
      anyDigit = true;
    } else if (c == '.' && dot == std::string_view::npos) {
      dot = position;
    } else if (c == 'e' || c == 'E') {
      exponentPosition = position;
      break;
    } else {
      return std::nullopt;
    }
  }
  if (!anyDigit) {
    return std::nullopt;
  }

  int64_t parsedExponent = 0;
  if (exponentPosition < token.size()) {
    position = exponentPosition + 1;
    bool exponentNegative = false;
    if (position < token.size() && (token[position] == '+' || token[position] == '-')) {
      exponentNegative = token[position] == '-';
      ++position;
    }
    if (position == token.size()) {
      return std::nullopt;
    }
    while (position < token.size()) {
      if (!isAsciiDigit(token[position])) {
        return std::nullopt;
      }
      const int32_t digit = token[position++] - '0';
      if (parsedExponent > (static_cast<int64_t>(std::numeric_limits<int32_t>::max()) - digit) / 10) {
        return std::nullopt;
      }
      parsedExponent = parsedExponent * 10 + digit;
    }
    if (exponentNegative) {
      parsedExponent = -parsedExponent;
    }
  }

  const int64_t fractionDigits = dot == std::string_view::npos ? 0 : static_cast<int64_t>(exponentPosition - dot - 1);
  const int64_t scale = fractionDigits - parsedExponent;
  if (scale < std::numeric_limits<int32_t>::min() || scale > std::numeric_limits<int32_t>::max()) {
    return std::nullopt;
  }

  std::string significant;
  significant.reserve(exponentPosition - mantissaBegin);
  bool foundNonzero = false;
  for (position = mantissaBegin; position < exponentPosition; ++position) {
    const char c = token[position];
    if (c == '.') {
      continue;
    }
    if (!foundNonzero && c == '0') {
      continue;
    }
    foundNonzero = true;
    significant.push_back(c);
  }
  if (!foundNonzero) {
    negative = false;
    significant = "0";
  }

  const int64_t precision = significant.size();
  const int64_t adjustedExponent = -scale + precision - 1;
  const bool plain = scale >= 0 && adjustedExponent >= -6;
  std::string output;
  output.reserve(significant.size() + 32);
  if (negative) {
    output.push_back('-');
  }
  if (plain) {
    const int64_t insertion = precision - scale;
    if (scale == 0) {
      output.append(significant);
    } else if (insertion > 0) {
      output.append(significant.data(), insertion);
      output.push_back('.');
      output.append(significant.data() + insertion, precision - insertion);
    } else {
      output.append("0.");
      output.append(-insertion, '0');
      output.append(significant);
    }
    return output;
  }

  output.push_back(significant.front());
  if (significant.size() > 1) {
    output.push_back('.');
    output.append(significant.data() + 1, significant.size() - 1);
  }
  output.push_back('E');
  if (adjustedExponent >= 0) {
    output.push_back('+');
  }
  output.append(std::to_string(adjustedExponent));
  return output;
}

enum class StopContext : uint8_t { kRoot, kObject, kArray };

class Parser {
 public:
  Parser(std::string input, const SimplePath& path) : input_(std::move(input)), steps_(path.steps()) {}

  std::optional<std::string> parse() {
    skipJsonSpace();
    if (position_ == input_.size()) {
      return std::nullopt;
    }
    auto result = parseValue(0, true, StopContext::kRoot, 0);
    return failed_ ? std::nullopt : result;
  }

 private:
  std::string input_;
  const std::vector<SimplePath::Step>& steps_;
  size_t position_{0};
  bool failed_{false};

  static bool isJsonSpace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }

  static int32_t hexValue(char c) {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
      return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
      return c - 'A' + 10;
    }
    return -1;
  }

  void skipJsonSpace() {
    while (position_ < input_.size() && isJsonSpace(input_[position_])) {
      ++position_;
    }
  }

  void skipAsciiSpace() {
    while (position_ < input_.size() && static_cast<unsigned char>(input_[position_]) != 0x1a &&
           static_cast<unsigned char>(input_[position_]) <= 0x20) {
      ++position_;
    }
  }

  static void trimSpan(std::string_view input, size_t& begin, size_t& end) {
    while (begin < end && static_cast<unsigned char>(input[begin]) <= 0x20) {
      ++begin;
    }
    while (end > begin && static_cast<unsigned char>(input[end - 1]) <= 0x20) {
      --end;
    }
  }

  static bool isStop(char c, StopContext context) {
    if (context == StopContext::kObject) {
      return c == ',' || c == '}';
    }
    if (context == StopContext::kArray) {
      return c == ',' || c == ']';
    }
    return false;
  }

  void emitUtf16Unit(uint16_t unit, std::optional<uint16_t>& pendingHigh, std::string& output) {
    if (pendingHigh.has_value()) {
      if (unit >= 0xdc00 && unit <= 0xdfff) {
        const uint32_t codePoint = 0x10000 + ((*pendingHigh - 0xd800) << 10) + (unit - 0xdc00);
        appendUtf8(codePoint, output);
        pendingHigh.reset();
        return;
      }
      output.push_back('?');
      pendingHigh.reset();
    }
    if (unit >= 0xd800 && unit <= 0xdbff) {
      pendingHigh = unit;
    } else if (unit >= 0xdc00 && unit <= 0xdfff) {
      output.push_back('?');
    } else {
      appendUtf8(unit, output);
    }
  }

  void flushPending(std::optional<uint16_t>& pendingHigh, std::string& output) {
    if (pendingHigh.has_value()) {
      output.push_back('?');
      pendingHigh.reset();
    }
  }

  std::optional<std::string> parseQuoted() {
    const char separator = input_[position_++];
    const size_t contentBegin = position_;
    const size_t firstSeparator = input_.find(separator, contentBegin);
    if (firstSeparator == std::string::npos) {
      failed_ = true;
      return std::nullopt;
    }
    if (input_.find('\\', contentBegin) >= firstSeparator) {
      position_ = firstSeparator + 1;
      return input_.substr(contentBegin, firstSeparator - contentBegin);
    }

    std::string output;
    output.reserve(firstSeparator - contentBegin);
    std::optional<uint16_t> pendingHigh;
    position_ = contentBegin;
    while (position_ < input_.size()) {
      const auto c = static_cast<unsigned char>(input_[position_++]);
      if (c == static_cast<unsigned char>(separator)) {
        flushPending(pendingHigh, output);
        return output;
      }
      // json-smart uses U+001A as its in-memory EOI sentinel.
      if (c == 0x1a) {
        failed_ = true;
        return std::nullopt;
      }
      if (c != '\\') {
        if (c <= 0x1f || c == 0x7f) {
          // MODE_PERMISSIVE drops raw controls on the escaped slow path.
          continue;
        }
        flushPending(pendingHigh, output);
        const int32_t length = utf8SequenceLength(c);
        if (length <= 0 || position_ - 1 + length > input_.size()) {
          failed_ = true;
          return std::nullopt;
        }
        output.append(input_.data() + position_ - 1, length);
        position_ += length - 1;
        continue;
      }
      if (position_ >= input_.size()) {
        failed_ = true;
        return std::nullopt;
      }
      const char escaped = input_[position_++];
      switch (escaped) {
        case 't':
          flushPending(pendingHigh, output);
          output.push_back('\t');
          break;
        case 'n':
          flushPending(pendingHigh, output);
          output.push_back('\n');
          break;
        case 'r':
          flushPending(pendingHigh, output);
          output.push_back('\r');
          break;
        case 'f':
          flushPending(pendingHigh, output);
          output.push_back('\f');
          break;
        case 'b':
          flushPending(pendingHigh, output);
          output.push_back('\b');
          break;
        case '\\':
        case '/':
        case '\'':
        case '"':
          flushPending(pendingHigh, output);
          output.push_back(escaped);
          break;
        case 'x':
        case 'u': {
          const int32_t count = escaped == 'x' ? 2 : 4;
          if (position_ + count > input_.size()) {
            failed_ = true;
            return std::nullopt;
          }
          uint16_t unit = 0;
          for (int32_t i = 0; i < count; ++i) {
            const int32_t value = hexValue(input_[position_ + i]);
            if (value < 0) {
              failed_ = true;
              return std::nullopt;
            }
            unit = static_cast<uint16_t>((unit << 4) | value);
          }
          position_ += count;
          emitUtf16Unit(unit, pendingHigh, output);
          break;
        }
        default:
          // json-smart consumes and drops exactly one Java UTF-16 code unit.
          // A normalized BMP code point is one unit and is dropped whole. A
          // supplementary code point is a surrogate pair: its high unit is
          // dropped, while the remaining low unit becomes '?' unless it can
          // pair with a preceding pending high surrogate.
          if (static_cast<unsigned char>(escaped) >= 0x80) {
            const int32_t length = utf8SequenceLength(static_cast<unsigned char>(escaped));
            if (length < 2 || position_ + length - 1 > input_.size()) {
              failed_ = true;
              return std::nullopt;
            }
            uint32_t codePoint = static_cast<unsigned char>(escaped) & ((1u << (7 - length)) - 1);
            for (int32_t i = 1; i < length; ++i) {
              const auto continuation = static_cast<unsigned char>(input_[position_ + i - 1]);
              if (!isUtf8Continuation(continuation)) {
                failed_ = true;
                return std::nullopt;
              }
              codePoint = (codePoint << 6) | (continuation & 0x3f);
            }
            position_ += length - 1;
            if (codePoint > 0xffff) {
              const uint16_t low = static_cast<uint16_t>(0xdc00 + ((codePoint - 0x10000) & 0x3ff));
              emitUtf16Unit(low, pendingHigh, output);
            }
          }
          break;
      }
    }
    failed_ = true;
    return std::nullopt;
  }

  size_t consumeToken(StopContext context) {
    while (position_ < input_.size() && input_[position_] != char(0x1a) && !isStop(input_[position_], context)) {
      ++position_;
    }
    return position_;
  }

  std::optional<std::string> selectedScalar(size_t pathIndex, bool evaluate, std::string value) {
    if (evaluate && pathIndex == steps_.size()) {
      return value;
    }
    return std::nullopt;
  }

  std::optional<std::string> parseLiteralOrString(size_t pathIndex, bool evaluate, StopContext context) {
    size_t begin = position_;
    size_t end = consumeToken(context);
    trimSpan(input_, begin, end);
    const std::string_view token(input_.data() + begin, end - begin);
    if (token == "null") {
      return std::nullopt;
    }
    if (token == "true" || token == "false") {
      return selectedScalar(pathIndex, evaluate, std::string(token));
    }
    if (token == "NaN") {
      return selectedScalar(pathIndex, evaluate, "NaN");
    }
    return selectedScalar(pathIndex, evaluate, std::string(token));
  }

  std::optional<std::string> formatInteger(std::string_view token) {
    bool negative = !token.empty() && token.front() == '-';
    size_t position = negative ? 1 : 0;
    while (position < token.size() && token[position] == '0') {
      ++position;
    }
    if (position == token.size()) {
      // Includes json-smart's surprising parseNumber("-") == Integer(0).
      return "0";
    }
    for (size_t i = position; i < token.size(); ++i) {
      if (!isAsciiDigit(token[i])) {
        return std::nullopt;
      }
    }
    std::string result;
    result.reserve(token.size() - position + negative);
    if (negative) {
      result.push_back('-');
    }
    result.append(token.data() + position, token.size() - position);
    return result;
  }

  std::optional<std::string> formatFloating(std::string_view token) {
    if (token.size() > 18) {
      return formatBigDecimal(token);
    }
    const auto value = parseDouble(token);
    if (!value.has_value()) {
      return std::nullopt;
    }
    return formatJavaDouble(*value);
  }

  std::optional<std::string> parseNumberOrString(size_t pathIndex, bool evaluate, StopContext context) {
    const size_t begin = position_;
    ++position_;
    while (position_ < input_.size() && isAsciiDigit(input_[position_])) {
      ++position_;
    }

    bool floating = false;
    if (position_ < input_.size() && input_[position_] == '.') {
      floating = true;
      ++position_;
      while (position_ < input_.size() && isAsciiDigit(input_[position_])) {
        ++position_;
      }
    }
    if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
      floating = true;
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-' || isAsciiDigit(input_[position_]))) {
        ++position_;
        while (position_ < input_.size() && isAsciiDigit(input_[position_])) {
          ++position_;
        }
      } else {
        const size_t end = consumeToken(context);
        size_t trimmedBegin = begin;
        size_t trimmedEnd = end;
        trimSpan(input_, trimmedBegin, trimmedEnd);
        return selectedScalar(pathIndex, evaluate, input_.substr(trimmedBegin, trimmedEnd - trimmedBegin));
      }
    }

    skipAsciiSpace();
    if (position_ < input_.size() && static_cast<unsigned char>(input_[position_]) < 126 &&
        input_[position_] != char(0x1a) && !isStop(input_[position_], context)) {
      const size_t end = consumeToken(context);
      size_t trimmedBegin = begin;
      size_t trimmedEnd = end;
      trimSpan(input_, trimmedBegin, trimmedEnd);
      return selectedScalar(pathIndex, evaluate, input_.substr(trimmedBegin, trimmedEnd - trimmedBegin));
    }

    size_t end = position_;
    size_t trimmedBegin = begin;
    trimSpan(input_, trimmedBegin, end);
    const std::string_view token(input_.data() + trimmedBegin, end - trimmedBegin);
    auto rendered = floating ? formatFloating(token) : formatInteger(token);
    if (!rendered.has_value()) {
      failed_ = true;
      return std::nullopt;
    }
    return selectedScalar(pathIndex, evaluate, std::move(*rendered));
  }

  std::optional<std::string> parseObject(size_t pathIndex, bool evaluate, int32_t depth) {
    ++position_;
    const bool canDescend =
        evaluate && pathIndex < steps_.size() && steps_[pathIndex].kind == SimplePath::StepKind::kField;
    std::optional<std::string> selected;
    while (true) {
      skipJsonSpace();
      if (position_ >= input_.size()) {
        failed_ = true;
        return std::nullopt;
      }
      const char current = input_[position_];
      if (current == '}') {
        ++position_;
        return selected;
      }
      if (current == ',') {
        ++position_;
        continue;
      }
      if (current == ':' || current == ']' || current == '[' || current == '{') {
        failed_ = true;
        return std::nullopt;
      }

      std::string key;
      if (current == '"' || current == '\'') {
        auto parsed = parseQuoted();
        if (failed_) {
          return std::nullopt;
        }
        key = std::move(*parsed);
      } else {
        size_t begin = position_;
        while (position_ < input_.size() && input_[position_] != char(0x1a) && input_[position_] != ':') {
          ++position_;
        }
        size_t end = position_;
        trimSpan(input_, begin, end);
        key.assign(input_.data() + begin, end - begin);
      }
      skipAsciiSpace();
      if (position_ >= input_.size() || input_[position_] != ':') {
        failed_ = true;
        return std::nullopt;
      }
      ++position_;
      if (position_ >= input_.size()) {
        failed_ = true;
        return std::nullopt;
      }

      const bool matches = canDescend && key == steps_[pathIndex].field;
      auto child = parseValue(pathIndex + 1, matches, StopContext::kObject, depth + 1);
      if (failed_) {
        return std::nullopt;
      }
      if (matches) {
        // OrderedJsonObject has Map overwrite semantics: the final duplicate
        // key replaces even a previous scalar with null/container/no-match.
        selected = std::move(child);
      }

      skipAsciiSpace();
      if (position_ >= input_.size()) {
        failed_ = true;
        return std::nullopt;
      }
      if (input_[position_] == '}') {
        ++position_;
        return selected;
      }
      if (input_[position_] != ',') {
        failed_ = true;
        return std::nullopt;
      }
      ++position_;
    }
  }

  std::optional<std::string> parseArray(size_t pathIndex, bool evaluate, int32_t depth) {
    ++position_;
    const bool canDescend =
        evaluate && pathIndex < steps_.size() && steps_[pathIndex].kind == SimplePath::StepKind::kIndex;
    const int32_t wanted = canDescend ? steps_[pathIndex].index : -1;
    int64_t elementIndex = 0;
    std::optional<std::string> selected;
    while (true) {
      skipJsonSpace();
      if (position_ >= input_.size()) {
        failed_ = true;
        return std::nullopt;
      }
      const char current = input_[position_];
      if (current == ']') {
        ++position_;
        return selected;
      }
      if (current == ',') {
        ++position_;
        continue;
      }
      if (current == ':' || current == '}') {
        failed_ = true;
        return std::nullopt;
      }

      const bool matches = canDescend && elementIndex == wanted;
      auto child = parseValue(pathIndex + 1, matches, StopContext::kArray, depth + 1);
      if (failed_) {
        return std::nullopt;
      }
      if (matches) {
        selected = std::move(child);
      }
      ++elementIndex;
      // JSONParserBase.readArray loops on the current character after a
      // value. Consequently quoted/container values may be followed by the
      // next value without a comma; the next iteration parses it directly.
    }
  }

  std::optional<std::string> parseValue(size_t pathIndex, bool evaluate, StopContext context, int32_t depth) {
    if (depth > kMaxNestingDepth) {
      failed_ = true;
      return std::nullopt;
    }
    skipJsonSpace();
    if (position_ >= input_.size()) {
      failed_ = true;
      return std::nullopt;
    }
    const char current = input_[position_];
    if (current == char(0x1a) || current == ':' || current == '}' || current == ']') {
      failed_ = true;
      return std::nullopt;
    }
    if (current == '{') {
      return parseObject(pathIndex, evaluate, depth);
    }
    if (current == '[') {
      return parseArray(pathIndex, evaluate, depth);
    }
    if (current == '"' || current == '\'') {
      auto value = parseQuoted();
      if (failed_) {
        return std::nullopt;
      }
      return selectedScalar(pathIndex, evaluate, std::move(*value));
    }
    if (isAsciiDigit(current) || current == '-') {
      return parseNumberOrString(pathIndex, evaluate, context);
    }
    return parseLiteralOrString(pathIndex, evaluate, context);
  }
};

} // namespace

std::optional<SimplePath> SimplePath::parse(std::string_view path) {
  if (path.size() < 3 || path.size() > kMaxPathBytes || path[0] != '$' || path[1] != '.') {
    return std::nullopt;
  }
  size_t position = 2;
  std::vector<Step> steps;
  steps.reserve(4);

  auto parseField = [&]() -> bool {
    if (position >= path.size() || !isFieldStart(path[position])) {
      return false;
    }
    const size_t begin = position++;
    while (position < path.size() && isFieldPart(path[position])) {
      ++position;
    }
    if (position - begin > kMaxFieldBytes) {
      return false;
    }
    steps.push_back({StepKind::kField, std::string(path.substr(begin, position - begin)), -1});
    return steps.size() <= kMaxPathDepth;
  };

  if (!parseField()) {
    return std::nullopt;
  }
  while (position < path.size()) {
    if (path[position] == '.') {
      ++position;
      if (!parseField()) {
        return std::nullopt;
      }
      continue;
    }
    if (path[position] != '[') {
      return std::nullopt;
    }
    ++position;
    const size_t begin = position;
    while (position < path.size() && isAsciiDigit(path[position])) {
      ++position;
    }
    if (begin == position || position >= path.size() || path[position] != ']') {
      return std::nullopt;
    }
    const auto indexText = path.substr(begin, position - begin);
    if ((indexText.size() > 1 && indexText.front() == '0') || indexText.size() > 10 ||
        (indexText.size() == 10 && indexText > "2147483647")) {
      return std::nullopt;
    }
    int32_t index = 0;
    for (char c : indexText) {
      index = index * 10 + (c - '0');
    }
    steps.push_back({StepKind::kIndex, {}, index});
    if (steps.size() > kMaxPathDepth) {
      return std::nullopt;
    }
    ++position;
  }
  return SimplePath(std::move(steps));
}

std::optional<std::string> extractScalar(std::string_view json, const SimplePath& path) {
  return Parser(normalizeJavaUtf8(json), path).parse();
}

std::optional<std::string> extractScalar(std::string_view json, std::string_view path) {
  auto parsedPath = SimplePath::parse(path);
  if (!parsedPath.has_value()) {
    return std::nullopt;
  }
  return extractScalar(json, *parsedPath);
}

} // namespace gluten::netflix_json
