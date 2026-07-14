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
#include "operators/functions/NetflixJson.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace gluten::netflix_json {
namespace {

void expectValue(std::string_view json, std::string_view path, std::string_view expected) {
  const auto actual = extractScalar(json, path);
  ASSERT_TRUE(actual.has_value()) << json << " / " << path;
  EXPECT_EQ(*actual, expected) << json << " / " << path;
}

void expectNull(std::string_view json, std::string_view path = "$.a") {
  EXPECT_FALSE(extractScalar(json, path).has_value()) << json << " / " << path;
}

std::string rawBytes(std::initializer_list<uint8_t> bytes) {
  std::string result;
  result.reserve(bytes.size());
  for (const auto byte : bytes) {
    result.push_back(static_cast<char>(byte));
  }
  return result;
}

TEST(NetflixJsonTest, AcceptsOnlyThePlannerSimplePathSlice) {
  for (const std::string_view path : {
           "$.a",
           "$._field0",
           "$.a.b[0].c[2147483647]",
       }) {
    EXPECT_TRUE(SimplePath::parse(path).has_value()) << path;
  }

  std::string maxDepth = "$.a";
  for (int32_t i = 1; i < 16; ++i) {
    maxDepth.append(".a");
  }
  EXPECT_TRUE(SimplePath::parse(maxDepth).has_value());
  maxDepth.append(".a");
  EXPECT_FALSE(SimplePath::parse(maxDepth).has_value());

  const std::string maxField = "$." + std::string(128, 'a');
  EXPECT_TRUE(SimplePath::parse(maxField).has_value());
  EXPECT_FALSE(SimplePath::parse("$." + std::string(129, 'a')).has_value());

  std::string overTotalBytes(4097, 'a');
  overTotalBytes[0] = '$';
  overTotalBytes[1] = '.';
  EXPECT_FALSE(SimplePath::parse(overTotalBytes).has_value());

  for (const std::string_view path : {
           "$",
           "$.",
           "$[0]",
           "$.0a",
           "$.a..b",
           "$.a[-1]",
           "$.a[01]",
           "$.a[2147483648]",
           "$.a[*]",
           "$['a']",
           "$.a ",
       }) {
    EXPECT_FALSE(SimplePath::parse(path).has_value()) << path;
  }
}

TEST(NetflixJsonTest, MatchesPermissiveRootAndContainerOracle) {
  // Oracle: spark-udf_4.0_2.13-1.11.6-rc.123.jar, which constructs
  // JsonSmartJsonProvider(MODE_PERMISSIVE == -1).
  expectValue("{a:1}garbage", "$.a", "1");
  expectValue("{a:1}{a:2}", "$.a", "1");
  expectValue("{,a:1,}", "$.a", "1");
  expectValue("{a:0,,a:1}", "$.a", "1");
  expectValue("{'a':'it\\'s'}", "$.a", "it's");
  expectValue("{a: hello world }", "$.a", "hello world");
  expectValue("{a: hello:world}", "$.a", "hello:world");
  expectValue("{a:[,1,,2,]}", "$.a[0]", "1");
  expectValue("{a:[,1,,2,]}", "$.a[1]", "2");
  expectValue("{a:['x' 2]}", "$.a[1]", "2");
  expectValue("{a:[{x:1} {x:2}]}", "$.a[1].x", "2");
  std::string verticalTabArray = "{a:['x'";
  verticalTabArray.push_back(char(11));
  verticalTabArray.append(",2]}");
  expectValue(verticalTabArray, "$.a[1]", "");
  expectValue(verticalTabArray, "$.a[2]", "2");
  expectValue("{a:[1\xC3\xA9,2]}", "$.a[1]", "\xC3\xA9");
  expectValue("{a:[1~,2]}", "$.a[1]", "~");
  expectValue("{a:1,a:2}", "$.a", "2");
  expectValue("{a:{b:true}}", "$.a.b", "true");
  expectValue("{a:{b:false}}", "$.a.b", "false");

  expectNull("{a:1");
  expectNull("{a:1,b:[}");
  expectNull("{a:null}");
  expectNull("{a:{b:1}}");
  expectNull("{a:[1]}");
  expectNull("{a:1,a:null}");
  expectNull("{a:{b:1},a:{}}", "$.a.b");
  expectNull("{a:1}", "$.missing");

  std::string bounded = "{a:1,b:";
  bounded.append(511, '[');
  bounded.push_back('0');
  bounded.append(511, ']');
  bounded.push_back('}');
  expectValue(bounded, "$.a", "1");

  bounded = "{a:1,b:";
  bounded.append(512, '[');
  bounded.push_back('0');
  bounded.append(512, ']');
  bounded.push_back('}');
  expectNull(bounded);
}

TEST(NetflixJsonTest, MatchesStringEscapeAndUtf8StringOracle) {
  // Exact outputs captured from the production spark-udf 4.0/2.13
  // 1.11.6-rc.123 JAR with json-smart 2.2.1 MODE_PERMISSIVE.
  expectValue(R"({a:'\b\f\n\r\t\"\'\\\/'})", "$.a", std::string("\b\f\n\r\t\"'\\/", 9));
  expectValue(R"({a:'x\qy'})", "$.a", "xy");
  expectValue(R"({a:'\x41\x7a'})", "$.a", "Az");
  expectValue(R"({a:'\uD83D\uDE03'})", "$.a", "\xF0\x9F\x98\x83");
  expectValue(R"({a:'\uD83D\q\uDE03'})", "$.a", "\xF0\x9F\x98\x83");
  expectValue(R"({a:'x\uD83Dy'})", "$.a", "x?y");
  expectValue(R"({a:'x\uDE03y'})", "$.a", "x?y");
  expectValue("{a:'\xC3\xA9\xF0\x9F\x98\x83'}", "$.a", "\xC3\xA9\xF0\x9F\x98\x83");

  std::string escapedControls(
      "a\0\1\x7f"
      "b",
      5);
  expectValue(R"({a:'a\u0000\u0001\u007Fb'})", "$.a", escapedControls);
  expectNull(R"({a:'\x4'})");
  expectNull(R"({a:'\x4g'})");
  expectNull(R"({a:'\u12xz'})");

  // json-smart's no-backslash String fast path preserves raw controls. Once a
  // backslash selects readString2, MODE_PERMISSIVE drops those same controls.
  std::string raw = "{a:'a";
  raw.append({char(0), char(1), char(9), char(10), char(31), char(127)});
  raw.append("b'}");
  std::string rawExpected = "a";
  rawExpected.append({char(0), char(1), char(9), char(10), char(31), char(127)});
  rawExpected.push_back('b');
  expectValue(raw, "$.a", rawExpected);

  std::string slow = "{a:'a\\n";
  slow.append({char(0), char(1), char(9), char(10), char(31), char(127)});
  slow.append("b'}");
  expectValue(slow, "$.a", "a\nb");

  std::string fastSubstitute = "{a:'a";
  fastSubstitute.push_back(char(0x1a));
  fastSubstitute.append("b'}");
  std::string fastSubstituteExpected = "a";
  fastSubstituteExpected.push_back(char(0x1a));
  fastSubstituteExpected.push_back('b');
  expectValue(fastSubstitute, "$.a", fastSubstituteExpected);

  std::string slowSubstitute = "{a:'a\\n";
  slowSubstitute.push_back(char(0x1a));
  slowSubstitute.append("b'}");
  expectNull(slowSubstitute);

  std::string containerSubstitute = "{a:1,";
  containerSubstitute.push_back(char(0x1a));
  containerSubstitute.append("b:2}");
  expectNull(containerSubstitute);
  expectNull(rawBytes({0x7b, 0x61, 0x3a, 0x5b, 0x1a, 0x5d, 0x7d}));

  const std::string replacement = "\xEF\xBF\xBD";
  std::string overlong = "{a:'";
  overlong.append({char(0xe0), char(0x80), char(0x80)});
  overlong.append("'}");
  expectValue(overlong, "$.a", replacement + replacement + replacement);

  std::string encodedSurrogate = "{a:'";
  encodedSurrogate.append({char(0xed), char(0xa0), char(0x80)});
  encodedSurrogate.append("'}");
  expectValue(encodedSurrogate, "$.a", replacement);

  const std::vector<std::pair<std::string, std::string>> malformedOracle = {
      {rawBytes({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xe0, 0x80, 0x59, 0x22, 0x7d}), "X" + replacement + replacement + "Y"},
      {rawBytes({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xf0, 0x80, 0x59, 0x22, 0x7d}), "X" + replacement + replacement + "Y"},
      {rawBytes({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xf4, 0xbf, 0x59, 0x22, 0x7d}), "X" + replacement + replacement + "Y"},
      {rawBytes({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xe1, 0x80, 0x59, 0x22, 0x7d}), "X" + replacement + "Y"},
      {rawBytes({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xe1, 0x80, 0x41, 0x59, 0x22, 0x7d}), "X" + replacement + "AY"},
      {rawBytes({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xf1, 0x80, 0x80, 0x59, 0x22, 0x7d}), "X" + replacement + "Y"},
      {rawBytes({0x7b, 0x61, 0x3a, 0x22, 0x58, 0xed, 0xa0, 0x80, 0x59, 0x22, 0x7d}), "X" + replacement + "Y"},
      // json-smart drops one Java UTF-16 unit for an unknown escape. A BMP
      // character is one unit; a supplementary character leaves a lone low
      // surrogate, which UTF8String renders as '?'.
      {rawBytes({0x7b, 0x61, 0x3a, 0x22, 0x58, 0x5c, 0xc3, 0xa9, 0x59, 0x22, 0x7d}), "XY"},
      {rawBytes({0x7b, 0x61, 0x3a, 0x22, 0x58, 0x5c, 0xf0, 0x9f, 0x98, 0x83, 0x59, 0x22, 0x7d}), "X?Y"},
      {rawBytes({0x7b, 0x61, 0x3a, 0x22, 0x58, 0x5c, 0xe0, 0x80, 0x59, 0x22, 0x7d}), "X" + replacement + "Y"},
  };
  for (const auto& [json, expected] : malformedOracle) {
    expectValue(json, "$.a", expected);
  }
}

TEST(NetflixJsonTest, MatchesJavaNumberToStringOracle) {
  // Exact Number.toString outputs captured from the same production JAR on
  // JDK 17. This includes FloatingDecimal behavior that differs from modern
  // shortest-representation implementations.
  const std::vector<std::pair<std::string_view, std::string_view>> corpus = {
      {"0", "0"},
      {"-0", "0"},
      {"-", "0"},
      {"001", "1"},
      {"2147483648", "2147483648"},
      {"9223372036854775807", "9223372036854775807"},
      {"9223372036854775808", "9223372036854775808"},
      {"123456789012345678901234567890", "123456789012345678901234567890"},
      {"0.0", "0.0"},
      {"-0.0", "-0.0"},
      {"1.", "1.0"},
      {"-.5", "-0.5"},
      {"1.00", "1.0"},
      {"1e3", "1000.0"},
      {"1e-3", "0.001"},
      {"1e-4", "1.0E-4"},
      {"9999999.0", "9999999.0"},
      {"10000000.0", "1.0E7"},
      {"5e-324", "4.9E-324"},
      {"9e-324", "1.0E-323"},
      {"1e-323", "1.0E-323"},
      {"1.1e-323", "1.0E-323"},
      {"1.01e-323", "1.0E-323"},
      {"-1.04323139e-323", "-1.0E-323"},
      {"1e-322", "1.0E-322"},
      {"1.01e-322", "1.0E-322"},
      {"-58.690692E-324", "-5.9E-323"},
      {"91.e-324", "8.9E-323"},
      {"76622.1289e-327", "7.9E-323"},
      {"1e-324", "0.0"},
      {"1e309", "Infinity"},
      // JDK 17 FloatingDecimal deliberately excludes the exact upper
      // midpoint accepted by modern shortest-representation algorithms.
      {"1e23", "9.999999999999999E22"},
      // JDK 17's integral-double fast path rounds insignificant binary
      // low-order digits before decimal rendering.
      {"75207785.380929E11", "7.5207785380929004E18"},
      {"05665897.61013e+12", "5.6658976101299999E18"},
      {"-6429913.27729e11", "-6.4299132772899994E17"},
      {"NaN", "NaN"},
  };
  for (const auto& [token, expected] : corpus) {
    expectValue(std::string("{a:") + std::string(token) + "}", "$.a", expected);
  }

  expectValue("{a:.5}", "$.a", ".5");
  expectValue("{a:+1}", "$.a", "+1");
  expectValue("{a:-Infinity}", "$.a", "-Infinity");
  expectValue("{a:1e}", "$.a", "1e");
  expectNull("{a:1e+}");
}

TEST(NetflixJsonTest, MatchesBigDecimalToStringOracle) {
  const std::vector<std::pair<std::string_view, std::string_view>> corpus = {
      {"1.00000000000000000", "1.00000000000000000"},
      {"1.000000000000000000", "1.000000000000000000"},
      {"123456789012345678.0", "123456789012345678.0"},
      {"0.000000123456789012345678", "1.23456789012345678E-7"},
      {"0.0000000123456789012345678", "1.23456789012345678E-8"},
      {"1.230000000000000000e3", "1230.000000000000000"},
      {"1.230000000000000000e30", "1.230000000000000000E+30"},
      {"1.230000000000000000e-30", "1.230000000000000000E-30"},
      {"1234567890123456789E+3", "1.234567890123456789E+21"},
      {"0.000000000000000000e100", "0E+82"},
      {"-0.000000000000000000", "0E-18"},
  };
  for (const auto& [token, expected] : corpus) {
    expectValue(std::string("{a:") + std::string(token) + "}", "$.a", expected);
  }
}

TEST(NetflixJsonTest, FailsClosedUnderDeterministicMalformedByteFuzz) {
  std::mt19937_64 random(0x4e465f4a534f4eULL);
  for (int32_t iteration = 0; iteration < 256; ++iteration) {
    const size_t size = random() % 257;
    std::string json(size, '\0');
    for (auto& byte : json) {
      byte = static_cast<char>(random() & 0xff);
    }
    EXPECT_NO_THROW((void)extractScalar(json, "$.a[0].b")) << "iteration " << iteration;
  }
}

} // namespace
} // namespace gluten::netflix_json
