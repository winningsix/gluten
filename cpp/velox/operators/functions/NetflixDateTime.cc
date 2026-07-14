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
#include "operators/functions/NetflixDateTime.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/Macros.h"
#include "velox/type/Timestamp.h"

#include <cstdint>
#include <ctime>
#include <limits>
#include <optional>
#include <string_view>

#ifdef GLUTEN_ENABLE_GPU
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/expression/FunctionSignature.h"

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/unary.hpp>
#endif

namespace gluten {
namespace {

constexpr int64_t kDateIntThreshold = 100'000'000;
constexpr int64_t kEpochSecondsMax = 99'999'999'999;
constexpr int64_t kEpochMillisMax = 99'999'999'999'999;
constexpr int64_t kSecondsPerDay = 86'400;

bool isUtc(std::string_view zone) {
  return zone == "UTC" || zone == "Etc/UTC" || zone == "Z";
}

bool isNfDateIntUtc(std::string_view zone) {
  return zone == "UTC" || zone == "Etc/UTC";
}

bool isLeapYear(int32_t year) {
  return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

std::optional<int64_t> dateIntToEpochSeconds(int64_t input) {
  if (input < 10'000'000 || input >= kDateIntThreshold) {
    return std::nullopt;
  }
  const auto year = static_cast<int32_t>(input / 10'000);
  const auto month = static_cast<int32_t>((input / 100) % 100);
  const auto day = static_cast<int32_t>(input % 100);
  if (month < 1 || month > 12) {
    return std::nullopt;
  }
  constexpr int32_t kDaysPerMonth[] = {
      31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  const auto daysInMonth =
      month == 2 && isLeapYear(year) ? 29 : kDaysPerMonth[month - 1];
  if (day < 1 || day > daysInMonth) {
    return std::nullopt;
  }

  // Howard Hinnant's civil-date conversion, with 1970-01-01 as day zero.
  auto adjustedYear = year - (month <= 2);
  const auto era =
      (adjustedYear >= 0 ? adjustedYear : adjustedYear - 399) / 400;
  const auto yearOfEra = static_cast<uint32_t>(adjustedYear - era * 400);
  const auto dayOfYear =
      (153 * static_cast<uint32_t>(month + (month > 2 ? -3 : 9)) + 2) /
          5 +
      static_cast<uint32_t>(day - 1);
  const auto dayOfEra = yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 +
      dayOfYear;
  const auto epochDay =
      static_cast<int64_t>(era) * 146'097 + dayOfEra - 719'468;
  return epochDay * kSecondsPerDay;
}

std::optional<int64_t> integralToEpochMillis(int64_t input) {
  // This mirrors NetflixDateTimeUtils.getEpochMs for the positive integral
  // domain reached by NfToUnixTime{,Ms}. Values with at most 11 digits are
  // seconds; 12-14 digits are milliseconds.
  if (input <= kDateIntThreshold || input > kEpochMillisMax) {
    return std::nullopt;
  }
  if (input <= kEpochSecondsMax) {
    return input * 1'000;
  }
  return input;
}

std::optional<int32_t> epochSecondsToDateInt(int64_t epochSeconds) {
  std::tm utc{};
  if (!facebook::velox::Timestamp::epochToCalendarUtc(epochSeconds, utc)) {
    return std::nullopt;
  }
  const auto year = static_cast<int64_t>(utc.tm_year) + 1900;
  const auto dateInt = year * 10'000 +
      (static_cast<int64_t>(utc.tm_mon) + 1) * 100 + utc.tm_mday;
  if (dateInt < std::numeric_limits<int32_t>::min() ||
      dateInt > std::numeric_limits<int32_t>::max()) {
    return std::nullopt;
  }
  return static_cast<int32_t>(dateInt);
}

struct NfDateIntFunction {
  template <typename T>
  struct udf {
    VELOX_DEFINE_FUNCTION_TYPES(T);

    FOLLY_ALWAYS_INLINE bool call(
        int32_t& result,
        int32_t input,
        const arg_type<facebook::velox::Varchar>& format,
        const arg_type<facebook::velox::Varchar>& timeZone) {
      const std::string_view formatView(format.data(), format.size());
      const std::string_view timeZoneView(timeZone.data(), timeZone.size());
      if (formatView != "-" || !isNfDateIntUtc(timeZoneView)) {
        return false;
      }

      if (input > kDateIntThreshold) {
        const auto epochMillis = integralToEpochMillis(input);
        if (!epochMillis.has_value()) {
          return false;
        }
        const auto dateInt = epochSecondsToDateInt(*epochMillis / 1'000);
        if (!dateInt.has_value()) {
          return false;
        }
        result = *dateInt;
        return true;
      }

      // Netflix validates an integral dateint through toLocalDate and then returns the original
      // INT. Preserve that behavior rather than normalizing malformed values.
      if (!dateIntToEpochSeconds(input).has_value()) {
        return false;
      }
      result = input;
      return true;
    }
  };
};

template <bool kMilliseconds>
struct NfToUnixTimeFunction {
  template <typename T>
  struct udf {
    VELOX_DEFINE_FUNCTION_TYPES(T);

    FOLLY_ALWAYS_INLINE bool call(
        int64_t& result,
        int32_t input,
        const arg_type<facebook::velox::Varchar>& format,
        const arg_type<facebook::velox::Varchar>& timeZone) {
      return callImpl(result, input, format, timeZone);
    }

    FOLLY_ALWAYS_INLINE bool call(
        int64_t& result,
        int64_t input,
        const arg_type<facebook::velox::Varchar>& format,
        const arg_type<facebook::velox::Varchar>& timeZone) {
      return callImpl(result, input, format, timeZone);
    }

   private:
    FOLLY_ALWAYS_INLINE bool callImpl(
        int64_t& result,
        int64_t input,
        const arg_type<facebook::velox::Varchar>& format,
        const arg_type<facebook::velox::Varchar>& timeZone) {
      const std::string_view formatView(format.data(), format.size());
      const std::string_view timeZoneView(timeZone.data(), timeZone.size());
      if (formatView != "-" || !isUtc(timeZoneView)) {
        return false;
      }

      if (input > kDateIntThreshold) {
        const auto epochMillis = integralToEpochMillis(input);
        if (!epochMillis.has_value()) {
          return false;
        }
        result = kMilliseconds ? *epochMillis : *epochMillis / 1'000;
        return true;
      }

      const auto epochSeconds = dateIntToEpochSeconds(input);
      if (!epochSeconds.has_value()) {
        return false;
      }
      result = kMilliseconds ? *epochSeconds * 1'000 : *epochSeconds;
      return true;
    }
  };
};

#ifdef GLUTEN_ENABLE_GPU

using facebook::velox::cudf_velox::ColumnOrView;
using facebook::velox::cudf_velox::CudfFunction;

std::unique_ptr<cudf::column> cudfDateIntFromDays(
    cudf::column_view days,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto yearsRaw = cudf::datetime::extract_datetime_component(
      days, cudf::datetime::datetime_component::YEAR, stream, mr);
  auto monthsRaw = cudf::datetime::extract_datetime_component(
      days, cudf::datetime::datetime_component::MONTH, stream, mr);
  auto daysRaw = cudf::datetime::extract_datetime_component(
      days, cudf::datetime::datetime_component::DAY, stream, mr);
  auto years = cudf::cast(
      yearsRaw->view(), cudf::data_type(cudf::type_id::INT32), stream, mr);
  auto months = cudf::cast(
      monthsRaw->view(), cudf::data_type(cudf::type_id::INT32), stream, mr);
  auto monthDays = cudf::cast(
      daysRaw->view(), cudf::data_type(cudf::type_id::INT32), stream, mr);
  cudf::numeric_scalar<int32_t> tenThousand(10'000, true, stream, mr);
  cudf::numeric_scalar<int32_t> hundred(100, true, stream, mr);
  auto yearPart = cudf::binary_operation(
      years->view(),
      tenThousand,
      cudf::binary_operator::MUL,
      cudf::data_type(cudf::type_id::INT32),
      stream,
      mr);
  auto monthPart = cudf::binary_operation(
      months->view(),
      hundred,
      cudf::binary_operator::MUL,
      cudf::data_type(cudf::type_id::INT32),
      stream,
      mr);
  auto yearMonth = cudf::binary_operation(
      yearPart->view(),
      monthPart->view(),
      cudf::binary_operator::ADD,
      cudf::data_type(cudf::type_id::INT32),
      stream,
      mr);
  return cudf::binary_operation(
      yearMonth->view(),
      monthDays->view(),
      cudf::binary_operator::ADD,
      cudf::data_type(cudf::type_id::INT32),
      stream,
      mr);
}

class CudfNfDateIntFunction final : public CudfFunction {
 public:
  explicit CudfNfDateIntFunction(
      const std::shared_ptr<facebook::velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(),
        3,
        "nf_dateint native function expects input, format, and timezone");
    const auto formatExpr =
        std::dynamic_pointer_cast<facebook::velox::exec::ConstantExpr>(
            expr->inputs()[1]);
    const auto timeZoneExpr =
        std::dynamic_pointer_cast<facebook::velox::exec::ConstantExpr>(
            expr->inputs()[2]);
    VELOX_CHECK_NOT_NULL(formatExpr, "nf_dateint format must be a constant");
    VELOX_CHECK_NOT_NULL(
        timeZoneExpr, "nf_dateint timezone must be a constant");
    VELOX_CHECK_EQ(
        formatExpr->value()->toString(0),
        "-",
        "nf_dateint currently supports only format '-'");
    VELOX_CHECK(
        isNfDateIntUtc(timeZoneExpr->value()->toString(0)),
        "nf_dateint currently supports only UTC");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    auto input = facebook::velox::cudf_velox::asView(inputColumns[0]);

    auto dateStrings = cudf::strings::from_integers(input, stream, mr);
    auto validDate = cudf::strings::is_timestamp(
        dateStrings->view(), "%Y%m%d", stream, mr);
    auto parsedDateDays = cudf::strings::to_timestamps(
        dateStrings->view(),
        cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
        "%Y%m%d",
        stream,
        mr);
    auto parsedDateInts =
        cudfDateIntFromDays(parsedDateDays->view(), stream, mr);
    cudf::numeric_scalar<int32_t> nullInt(0, false, stream, mr);
    auto nullableParsedDateInts = cudf::copy_if_else(
        parsedDateInts->view(), nullInt, validDate->view(), stream, mr);

    // cuDF intentionally rejects integral <-> timestamp casts.  Preserve the
    // integral representation by crossing the chrono boundary through the
    // timestamp's duration type.
    auto epochDurationSeconds = cudf::cast(
        input,
        cudf::data_type(cudf::type_id::DURATION_SECONDS),
        stream,
        mr);
    auto epochSeconds = cudf::cast(
        epochDurationSeconds->view(),
        cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS),
        stream,
        mr);
    auto epochDays = cudf::cast(
        epochSeconds->view(),
        cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
        stream,
        mr);
    auto epochDateInts = cudfDateIntFromDays(epochDays->view(), stream, mr);

    cudf::numeric_scalar<int32_t> dateIntThreshold(
        kDateIntThreshold, true, stream, mr);
    auto epochMask = cudf::binary_operation(
        input,
        dateIntThreshold,
        cudf::binary_operator::GREATER,
        cudf::data_type(cudf::type_id::BOOL8),
        stream,
        mr);
    return cudf::copy_if_else(
        epochDateInts->view(),
        nullableParsedDateInts->view(),
        epochMask->view(),
        stream,
        mr);
  }
};

class CudfNfToUnixTimeFunction final : public CudfFunction {
 public:
  CudfNfToUnixTimeFunction(
      const std::shared_ptr<facebook::velox::exec::Expr>& expr,
      bool milliseconds)
      : milliseconds_(milliseconds) {
    VELOX_CHECK_EQ(
        expr->inputs().size(),
        3,
        "nf_to_unixtime native function expects input, format, and timezone");
    const auto formatExpr =
        std::dynamic_pointer_cast<facebook::velox::exec::ConstantExpr>(
            expr->inputs()[1]);
    const auto timeZoneExpr =
        std::dynamic_pointer_cast<facebook::velox::exec::ConstantExpr>(
            expr->inputs()[2]);
    VELOX_CHECK_NOT_NULL(
        formatExpr, "nf_to_unixtime format must be a constant");
    VELOX_CHECK_NOT_NULL(
        timeZoneExpr, "nf_to_unixtime timezone must be a constant");
    VELOX_CHECK_EQ(
        formatExpr->value()->toString(0),
        "-",
        "nf_to_unixtime currently supports only format '-'");
    VELOX_CHECK(
        isUtc(timeZoneExpr->value()->toString(0)),
        "nf_to_unixtime currently supports only UTC");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    auto input = facebook::velox::cudf_velox::asView(inputColumns[0]);
    auto longs = cudf::cast(
        input, cudf::data_type(cudf::type_id::INT64), stream, mr);

    cudf::numeric_scalar<int64_t> dateIntThreshold(
        kDateIntThreshold, true, stream, mr);
    auto epochMask = cudf::binary_operation(
        longs->view(),
        dateIntThreshold,
        cudf::binary_operator::GREATER,
        cudf::data_type(cudf::type_id::BOOL8),
        stream,
        mr);

    auto dateStrings =
        cudf::strings::from_integers(longs->view(), stream, mr);
    auto validDate = cudf::strings::is_timestamp(
        dateStrings->view(), "%Y%m%d", stream, mr);
    auto dateDays = cudf::strings::to_timestamps(
        dateStrings->view(),
        cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
        "%Y%m%d",
        stream,
        mr);
    // cuDF requires timestamp -> duration -> integral for the reverse chrono
    // conversion as well.
    auto durationDays = cudf::cast(
        dateDays->view(),
        cudf::data_type(cudf::type_id::DURATION_DAYS),
        stream,
        mr);
    auto epochDays = cudf::cast(
        durationDays->view(),
        cudf::data_type(cudf::type_id::INT64),
        stream,
        mr);
    cudf::numeric_scalar<int64_t> dateMultiplier(
        milliseconds_ ? kSecondsPerDay * 1'000 : kSecondsPerDay,
        true,
        stream,
        mr);
    auto dateResult = cudf::binary_operation(
        epochDays->view(),
        dateMultiplier,
        cudf::binary_operator::MUL,
        cudf::data_type(cudf::type_id::INT64),
        stream,
        mr);
    cudf::numeric_scalar<int64_t> nullLong(0, false, stream, mr);
    auto nullableDateResult = cudf::copy_if_else(
        dateResult->view(), nullLong, validDate->view(), stream, mr);

    cudf::numeric_scalar<int64_t> secondsMax(
        kEpochSecondsMax, true, stream, mr);
    auto secondsMask = cudf::binary_operation(
        longs->view(),
        secondsMax,
        cudf::binary_operator::LESS_EQUAL,
        cudf::data_type(cudf::type_id::BOOL8),
        stream,
        mr);
    cudf::numeric_scalar<int64_t> thousand(1'000, true, stream, mr);
    auto scaledEpoch = cudf::binary_operation(
        longs->view(),
        thousand,
        milliseconds_ ? cudf::binary_operator::MUL
                      : cudf::binary_operator::DIV,
        cudf::data_type(cudf::type_id::INT64),
        stream,
        mr);
    auto epochResult = milliseconds_
        ? cudf::copy_if_else(
              scaledEpoch->view(), longs->view(), secondsMask->view(), stream, mr)
        : cudf::copy_if_else(
              longs->view(), scaledEpoch->view(), secondsMask->view(), stream, mr);
    cudf::numeric_scalar<int64_t> millisMax(
        kEpochMillisMax, true, stream, mr);
    auto validEpoch = cudf::binary_operation(
        longs->view(),
        millisMax,
        cudf::binary_operator::LESS_EQUAL,
        cudf::data_type(cudf::type_id::BOOL8),
        stream,
        mr);
    auto nullableEpochResult = cudf::copy_if_else(
        epochResult->view(), nullLong, validEpoch->view(), stream, mr);

    return cudf::copy_if_else(
        nullableEpochResult->view(),
        nullableDateResult->view(),
        epochMask->view(),
        stream,
        mr);
  }

 private:
  bool milliseconds_;
};

void registerCudfNetflixDateTimeFunction(
    const std::string& name,
    bool milliseconds) {
  using facebook::velox::exec::FunctionSignatureBuilder;
  const std::vector<facebook::velox::exec::FunctionSignaturePtr> signatures{
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("integer")
          .constantArgumentType("varchar")
          .constantArgumentType("varchar")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .constantArgumentType("varchar")
          .constantArgumentType("varchar")
          .build()};
  facebook::velox::cudf_velox::registerCudfFunction(
      name,
      [milliseconds](
          const std::string&,
          const std::shared_ptr<facebook::velox::exec::Expr>& expr) {
        return std::make_shared<CudfNfToUnixTimeFunction>(
            expr, milliseconds);
      },
      signatures);
}

void registerCudfNfDateIntFunction() {
  using facebook::velox::exec::FunctionSignatureBuilder;
  const std::vector<facebook::velox::exec::FunctionSignaturePtr> signatures{
      FunctionSignatureBuilder()
          .returnType("integer")
          .argumentType("integer")
          .constantArgumentType("varchar")
          .constantArgumentType("varchar")
          .build()};
  facebook::velox::cudf_velox::registerCudfFunction(
      "nf_dateint",
      [](
          const std::string&,
          const std::shared_ptr<facebook::velox::exec::Expr>& expr) {
        return std::make_shared<CudfNfDateIntFunction>(expr);
      },
      signatures);
}
#endif

} // namespace

void registerNetflixDateTimeFunctions() {
  facebook::velox::registerFunction<
      NfDateIntFunction,
      int32_t,
      int32_t,
      facebook::velox::Varchar,
      facebook::velox::Varchar>({"nf_dateint"});
  facebook::velox::registerFunction<
      NfToUnixTimeFunction<false>,
      int64_t,
      int32_t,
      facebook::velox::Varchar,
      facebook::velox::Varchar>({"nf_to_unixtime"});
  facebook::velox::registerFunction<
      NfToUnixTimeFunction<false>,
      int64_t,
      int64_t,
      facebook::velox::Varchar,
      facebook::velox::Varchar>({"nf_to_unixtime"});
  facebook::velox::registerFunction<
      NfToUnixTimeFunction<true>,
      int64_t,
      int32_t,
      facebook::velox::Varchar,
      facebook::velox::Varchar>({"nf_to_unixtime_ms"});
  facebook::velox::registerFunction<
      NfToUnixTimeFunction<true>,
      int64_t,
      int64_t,
      facebook::velox::Varchar,
      facebook::velox::Varchar>({"nf_to_unixtime_ms"});

#ifdef GLUTEN_ENABLE_GPU
  registerCudfNfDateIntFunction();
  registerCudfNetflixDateTimeFunction("nf_to_unixtime", false);
  registerCudfNetflixDateTimeFunction("nf_to_unixtime_ms", true);
#endif
}

} // namespace gluten
