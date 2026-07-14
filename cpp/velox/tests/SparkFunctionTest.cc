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

#include <limits>
#include <vector>

#include "operators/functions/RegistrationAllFunctions.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::functions::sparksql::test;
using namespace facebook::velox;

class SparkFunctionTest : public SparkFunctionBaseTest {
 public:
  SparkFunctionTest() {
    gluten::registerAllFunctions();
  }

 protected:
  template <typename T>
  void runRoundTest(const std::vector<std::tuple<T, T>>& data) {
    auto result = evaluate<SimpleVector<T>>("round(c0)", makeRowVector({makeFlatVector<T, 0>(data)}));
    for (int32_t i = 0; i < data.size(); ++i) {
      ASSERT_EQ(result->valueAt(i), std::get<1>(data[i]));
    }
  }

  template <typename T>
  void runRoundWithDecimalTest(const std::vector<std::tuple<T, int32_t, T>>& data) {
    auto result = evaluate<SimpleVector<T>>(
        "round(c0, c1)", makeRowVector({makeFlatVector<T, 0>(data), makeFlatVector<int32_t, 1>(data)}));
    for (int32_t i = 0; i < data.size(); ++i) {
      ASSERT_EQ(result->valueAt(i), std::get<2>(data[i]));
    }
  }

  template <typename T>
  std::vector<std::tuple<T, T>> testRoundFloatData() {
    return {
        {1.0, 1.0},
        {1.9, 2.0},
        {1.3, 1.0},
        {0.0, 0.0},
        {0.9999, 1.0},
        {-0.9999, -1.0},
        {1.0 / 9999999, 0},
        {123123123.0 / 9999999, 12.0}};
  }

  template <typename T>
  std::vector<std::tuple<T, T>> testRoundIntegralData() {
    return {{1, 1}, {0, 0}, {-1, -1}};
  }

  template <typename T>
  std::vector<std::tuple<T, int32_t, T>> testRoundWithDecFloatAndDoubleData() {
    return {{1.122112, 0, 1},       {1.129, 1, 1.1},        {1.129, 2, 1.13},         {1.0 / 3, 0, 0.0},
            {1.0 / 3, 1, 0.3},      {1.0 / 3, 2, 0.33},     {1.0 / 3, 6, 0.333333},   {-1.122112, 0, -1},
            {-1.129, 1, -1.1},      {-1.129, 2, -1.13},     {-1.129, 2, -1.13},       {-1.0 / 3, 0, 0.0},
            {-1.0 / 3, 1, -0.3},    {-1.0 / 3, 2, -0.33},   {-1.0 / 3, 6, -0.333333}, {1.0, -1, 0.0},
            {0.0, -2, 0.0},         {-1.0, -3, 0.0},        {11111.0, -1, 11110.0},   {11111.0, -2, 11100.0},
            {11111.0, -3, 11000.0}, {11111.0, -4, 10000.0}, {0.575, 2, 0.58},         {0.574, 2, 0.57},
            {-0.575, 2, -0.58},     {-0.574, 2, -0.57}};
  }

  template <typename T>
  std::vector<std::tuple<T, int32_t, T>> testRoundWithDecIntegralData() {
    return {
        {1, 0, 1},
        {0, 0, 0},
        {-1, 0, -1},
        {1, 1, 1},
        {0, 1, 0},
        {-1, 1, -1},
        {1, 10, 1},
        {0, 10, 0},
        {-1, 10, -1},
        {1, -1, 0},
        {0, -2, 0},
        {-1, -3, 0}};
  }
};

TEST_F(SparkFunctionTest, round) {
  runRoundTest<float>(testRoundFloatData<float>());
  runRoundTest<double>(testRoundFloatData<double>());
  runRoundTest<int64_t>(testRoundIntegralData<int64_t>());
  runRoundTest<int32_t>(testRoundIntegralData<int32_t>());
  runRoundTest<int16_t>(testRoundIntegralData<int16_t>());
  runRoundTest<int8_t>(testRoundIntegralData<int8_t>());
}

TEST_F(SparkFunctionTest, roundWithDecimal) {
  runRoundWithDecimalTest<float>(testRoundWithDecFloatAndDoubleData<float>());
  runRoundWithDecimalTest<double>(testRoundWithDecFloatAndDoubleData<double>());
  runRoundWithDecimalTest<int64_t>(testRoundWithDecIntegralData<int64_t>());
  runRoundWithDecimalTest<int32_t>(testRoundWithDecIntegralData<int32_t>());
  runRoundWithDecimalTest<int16_t>(testRoundWithDecIntegralData<int16_t>());
  runRoundWithDecimalTest<int8_t>(testRoundWithDecIntegralData<int8_t>());
}

TEST_F(SparkFunctionTest, netflixToUnixTimeIntegralDefaultFormatUtc) {
  auto dateInts = makeNullableFlatVector<int32_t>(
      {19700101, 20240101, 20230229, std::nullopt});
  auto seconds = evaluate<FlatVector<int64_t>>(
      "nf_to_unixtime(c0, '-', 'Etc/UTC')", makeRowVector({dateInts}));
  auto millis = evaluate<FlatVector<int64_t>>(
      "nf_to_unixtime_ms(c0, '-', 'UTC')", makeRowVector({dateInts}));

  ASSERT_EQ(seconds->valueAt(0), 0);
  ASSERT_EQ(seconds->valueAt(1), 1704067200);
  ASSERT_TRUE(seconds->isNullAt(2));
  ASSERT_TRUE(seconds->isNullAt(3));
  ASSERT_EQ(millis->valueAt(0), 0);
  ASSERT_EQ(millis->valueAt(1), 1704067200000);
  ASSERT_TRUE(millis->isNullAt(2));
  ASSERT_TRUE(millis->isNullAt(3));
}

TEST_F(SparkFunctionTest, netflixDateIntJob175IntegralDefaultFormatUtc) {
  auto values = makeNullableFlatVector<int32_t>({
      10000101,
      19700101,
      20000229,
      20260531,
      99991231,
      20230229,
      10'000'000,
      100'000'000,
      100'000'001,
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::min(),
      std::nullopt});
  auto utc = evaluate<FlatVector<int32_t>>(
      "nf_dateint(c0, '-', 'UTC')", makeRowVector({values}));
  auto etcUtc = evaluate<FlatVector<int32_t>>(
      "nf_dateint(c0, '-', 'Etc/UTC')", makeRowVector({values}));

  const std::vector<std::optional<int32_t>> expected{
      10000101,
      19700101,
      20000229,
      20260531,
      99991231,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      19730303,
      20380119,
      std::nullopt,
      std::nullopt};
  for (size_t row = 0; row < expected.size(); ++row) {
    if (expected[row].has_value()) {
      ASSERT_EQ(utc->valueAt(row), *expected[row]);
      ASSERT_EQ(etcUtc->valueAt(row), *expected[row]);
    } else {
      ASSERT_TRUE(utc->isNullAt(row));
      ASSERT_TRUE(etcUtc->isNullAt(row));
    }
  }
}

TEST_F(SparkFunctionTest, netflixDateIntRejectsFormatAndTimezoneOutsideSlice) {
  auto values = makeNullableFlatVector<int32_t>({20260531, std::nullopt});
  auto explicitFormat = evaluate<FlatVector<int32_t>>(
      "nf_dateint(c0, 'yyyyMMdd', 'UTC')", makeRowVector({values}));
  auto nonUtc = evaluate<FlatVector<int32_t>>(
      "nf_dateint(c0, '-', 'America/Los_Angeles')", makeRowVector({values}));
  auto zulu = evaluate<FlatVector<int32_t>>(
      "nf_dateint(c0, '-', 'Z')", makeRowVector({values}));

  ASSERT_TRUE(explicitFormat->isNullAt(0));
  ASSERT_TRUE(explicitFormat->isNullAt(1));
  ASSERT_TRUE(nonUtc->isNullAt(0));
  ASSERT_TRUE(nonUtc->isNullAt(1));
  ASSERT_TRUE(zulu->isNullAt(0));
  ASSERT_TRUE(zulu->isNullAt(1));
}

TEST_F(SparkFunctionTest, netflixToUnixTimeEpochSecondsAndMillis) {
  auto epochs = makeNullableFlatVector<int64_t>(
      {1704067200, 1704067200000, 100000000000000, std::nullopt});
  auto seconds = evaluate<FlatVector<int64_t>>(
      "nf_to_unixtime(c0, '-', 'UTC')", makeRowVector({epochs}));
  auto millis = evaluate<FlatVector<int64_t>>(
      "nf_to_unixtime_ms(c0, '-', 'Etc/UTC')", makeRowVector({epochs}));

  ASSERT_EQ(seconds->valueAt(0), 1704067200);
  ASSERT_EQ(seconds->valueAt(1), 1704067200);
  ASSERT_TRUE(seconds->isNullAt(2));
  ASSERT_TRUE(seconds->isNullAt(3));
  ASSERT_EQ(millis->valueAt(0), 1704067200000);
  ASSERT_EQ(millis->valueAt(1), 1704067200000);
  ASSERT_TRUE(millis->isNullAt(2));
  ASSERT_TRUE(millis->isNullAt(3));
}

TEST_F(SparkFunctionTest, netflixToUnixTimeIntegralThresholds) {
  auto values = makeNullableFlatVector<int64_t>({
      100000000,
      100000001,
      99999999999,
      100000000000,
      99999999999999,
      100000000000000,
      std::numeric_limits<int64_t>::min(),
      std::nullopt});
  auto seconds = evaluate<FlatVector<int64_t>>(
      "nf_to_unixtime(c0, '-', 'UTC')", makeRowVector({values}));
  auto millis = evaluate<FlatVector<int64_t>>(
      "nf_to_unixtime_ms(c0, '-', 'Etc/UTC')", makeRowVector({values}));

  ASSERT_TRUE(seconds->isNullAt(0));
  ASSERT_EQ(seconds->valueAt(1), 100000001);
  ASSERT_EQ(seconds->valueAt(2), 99999999999);
  ASSERT_EQ(seconds->valueAt(3), 100000000);
  ASSERT_EQ(seconds->valueAt(4), 99999999999);
  ASSERT_TRUE(seconds->isNullAt(5));
  ASSERT_TRUE(seconds->isNullAt(6));
  ASSERT_TRUE(seconds->isNullAt(7));

  ASSERT_TRUE(millis->isNullAt(0));
  ASSERT_EQ(millis->valueAt(1), 100000001000);
  ASSERT_EQ(millis->valueAt(2), 99999999999000);
  ASSERT_EQ(millis->valueAt(3), 100000000000);
  ASSERT_EQ(millis->valueAt(4), 99999999999999);
  ASSERT_TRUE(millis->isNullAt(5));
  ASSERT_TRUE(millis->isNullAt(6));
  ASSERT_TRUE(millis->isNullAt(7));
}
