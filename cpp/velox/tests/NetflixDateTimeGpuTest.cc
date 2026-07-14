/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0.
 */

#include "operators/functions/RegistrationAllFunctions.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"
#include "velox/parse/TypeResolver.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;

namespace {

class NetflixDateTimeGpuTest : public CudfFunctionBaseTest {
 protected:
  static void SetUpTestSuite() {
    parse::registerTypeResolver();
    memory::MemoryManager::testingSetInstance(
        memory::MemoryManager::Options{});
    registerCudf();
    gluten::registerAllFunctions();
  }

  static void TearDownTestSuite() {
    unregisterFunctions();
    unregisterCudf();
  }
};

TEST_F(NetflixDateTimeGpuTest, DateIntIntegralChronoCastsMatchCpu) {
  auto input = makeRowVector({makeNullableFlatVector<int32_t>({
      10000101,
      19700101,
      20000229,
      20260530,
      20260531,
      20230229,
      100'000'000,
      100'000'001,
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::min(),
      std::nullopt,
  })});

  assertExpressionMatchesCpu(
      "nf_dateint(c0, '-', 'Etc/UTC')", input, input->rowType());
  assertExpressionMatchesCpu(
      "nf_dateint(c0, '-', 'Etc/UTC') between 20260530 and 20260531",
      input,
      input->rowType());
}

TEST_F(NetflixDateTimeGpuTest, UnixTimeDateIntChronoCastsMatchCpu) {
  auto input = makeRowVector({makeNullableFlatVector<int64_t>({
      19700101,
      20000229,
      20240101,
      20230229,
      1'704'067'200,
      1'704'067'200'000,
      std::nullopt,
  })});

  assertExpressionMatchesCpu(
      "nf_to_unixtime(c0, '-', 'UTC')", input, input->rowType());
  assertExpressionMatchesCpu(
      "nf_to_unixtime_ms(c0, '-', 'Etc/UTC')", input, input->rowType());
}

TEST_F(NetflixDateTimeGpuTest, FusedFromUnixTimeTzFieldsMatchCpu) {
  auto input = makeRowVector(
      {makeNullableFlatVector<int64_t>(
           {1'704'067'200,
            1'704'067'200'000,
            1'710'063'000,
            1'710'066'600,
            1'704'067'200,
            1'704'067'200,
            12'622'807'800,
            -2'717'683'500,
            100'000'000'000'000,
            std::nullopt}),
       makeNullableFlatVector<StringView>(
           {"UTC",
            "America/Los_Angeles",
            "America/Los_Angeles",
            "America/Los_Angeles",
            "PST",
            "+05:30",
            "America/Los_Angeles",
            "America/Los_Angeles",
            "UTC",
            std::nullopt})});

  assertExpressionMatchesCpu("nf_dateint_from_unixtime_tz(c0, c1, '-', 'UTC')", input, input->rowType());
  assertExpressionMatchesCpu("nf_hour_from_unixtime_tz(c0, c1, 'Etc/UTC')", input, input->rowType());
  assertExpressionMatchesCpu(
      "nf_dateint_from_unixtime_tz(c0, 'America/Los_Angeles', '-', 'UTC')",
      makeRowVector({input->childAt(0)}),
      ROW({"c0"}, {BIGINT()}));

  auto nullZones = makeRowVector(
      {makeFlatVector<int64_t>({1'704'067'200, 1'704'067'201}),
       makeNullableFlatVector<StringView>({std::nullopt, std::nullopt})});
  assertExpressionMatchesCpu("nf_dateint_from_unixtime_tz(c0, c1, '-', 'UTC')", nullZones, nullZones->rowType());

  auto emptyZone = makeRowVector({makeFlatVector<int64_t>({1'704'067'200})});
  assertExpressionMatchesCpu("nf_hour_from_unixtime_tz(c0, '', 'UTC')", emptyZone, emptyZone->rowType());
}

} // namespace
