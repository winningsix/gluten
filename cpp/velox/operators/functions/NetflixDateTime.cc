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
#include "velox/functions/Macros.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/type/Timestamp.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#ifdef GLUTEN_ENABLE_GPU
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/expression/FunctionSignature.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/timezone.hpp>
#include <cudf/unary.hpp>
#endif

namespace gluten {
namespace {

constexpr int64_t kDateIntThreshold = 100'000'000;
constexpr int64_t kEpochSecondsMax = 99'999'999'999;
constexpr int64_t kEpochMillisMax = 99'999'999'999'999;
constexpr int64_t kSecondsPerDay = 86'400;

const std::unordered_map<std::string_view, std::string_view>& javaShortTimeZones() {
  static const std::unordered_map<std::string_view, std::string_view> zones{
      {"ACT", "Australia/Darwin"},
      {"AET", "Australia/Sydney"},
      {"AGT", "America/Argentina/Buenos_Aires"},
      {"ART", "Africa/Cairo"},
      {"AST", "America/Anchorage"},
      {"BET", "America/Sao_Paulo"},
      {"BST", "Asia/Dhaka"},
      {"CAT", "Africa/Harare"},
      {"CNT", "America/St_Johns"},
      {"CST", "America/Chicago"},
      {"CTT", "Asia/Shanghai"},
      {"EAT", "Africa/Addis_Ababa"},
      {"ECT", "Europe/Paris"},
      {"EST", "-05:00"},
      {"HST", "-10:00"},
      {"IET", "America/Indiana/Indianapolis"},
      {"IST", "Asia/Kolkata"},
      {"JST", "Asia/Tokyo"},
      {"MIT", "Pacific/Apia"},
      {"MST", "-07:00"},
      {"NET", "Asia/Yerevan"},
      {"NST", "Pacific/Auckland"},
      {"PLT", "Asia/Karachi"},
      {"PNT", "America/Phoenix"},
      {"PRT", "America/Puerto_Rico"},
      {"PST", "America/Los_Angeles"},
      {"SST", "Pacific/Guadalcanal"},
      {"VST", "Asia/Ho_Chi_Minh"}};
  return zones;
}

std::string normalizeJavaTimeZone(std::string_view zone) {
  if (const auto it = javaShortTimeZones().find(zone); it != javaShortTimeZones().end()) {
    return std::string(it->second);
  }
  if (zone == "Z" || zone == "UTC" || zone == "Etc/UTC" || zone == "Etc/UCT" || zone == "UCT" || zone == "GMT" ||
      zone == "Etc/GMT" || zone == "GMT0" || zone == "Greenwich" || zone == "Universal" || zone == "Zulu") {
    return "UTC";
  }
  return std::string(zone);
}

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

std::optional<int64_t> fusedEpochMillis(int64_t input) {
  // NetflixDateTimeUtils.getEpochMs classifies by decimal digit count of the
  // absolute value. At most 11 digits means seconds; 12-14 means millis.
  // Expressing the same domain as signed bounds avoids abs(Long.MIN_VALUE).
  if (input >= -kEpochSecondsMax && input <= kEpochSecondsMax) {
    return input * 1'000;
  }
  if (input >= -kEpochMillisMax && input <= kEpochMillisMax) {
    return input;
  }
  return std::nullopt;
}

int64_t floorDiv(int64_t value, int64_t divisor) {
  auto quotient = value / divisor;
  const auto remainder = value % divisor;
  if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
    --quotient;
  }
  return quotient;
}

std::optional<int64_t> fusedLocalEpochSeconds(int64_t input, std::string_view zoneName) {
  const auto epochMillis = fusedEpochMillis(input);
  if (!epochMillis.has_value()) {
    return std::nullopt;
  }
  const auto normalizedZone = normalizeJavaTimeZone(zoneName);
  const auto* zone = facebook::velox::tz::locateZone(normalizedZone, false);
  if (zone == nullptr) {
    return std::nullopt;
  }
  try {
    const auto localMillis = zone->to_local(std::chrono::milliseconds(*epochMillis)).count();
    return floorDiv(localMillis, 1'000);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<int32_t> historicalTimeZoneOffsetSeconds(std::string_view zoneName) {
  const auto normalizedZone = normalizeJavaTimeZone(zoneName);
  const auto* zone = facebook::velox::tz::locateZone(normalizedZone, false);
  if (zone == nullptr) {
    return std::nullopt;
  }
  try {
    // Probe well before civil time-zone transitions while staying inside the
    // range supported by Velox's date library. The first TZif offset extends
    // backwards from the first recorded transition.
    constexpr int64_t kHistoricalProbeSeconds = -10'000'000'000LL;
    const auto local = zone->to_local(std::chrono::seconds(kHistoricalProbeSeconds)).count();
    const auto offset = local - kHistoricalProbeSeconds;
    if (offset < std::numeric_limits<int32_t>::min() || offset > std::numeric_limits<int32_t>::max()) {
      return std::nullopt;
    }
    return static_cast<int32_t>(offset);
  } catch (const std::exception&) {
    return std::nullopt;
  }
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

struct NfDateIntFromUnixTimeTzFunction {
  template <typename T>
  struct udf {
    VELOX_DEFINE_FUNCTION_TYPES(T);

    FOLLY_ALWAYS_INLINE bool call(
        int32_t& result,
        int64_t input,
        const arg_type<facebook::velox::Varchar>& rowTimeZone,
        const arg_type<facebook::velox::Varchar>& format,
        const arg_type<facebook::velox::Varchar>& outerTimeZone) {
      const std::string_view formatView(format.data(), format.size());
      const std::string_view outerZoneView(outerTimeZone.data(), outerTimeZone.size());
      if (formatView != "-" || !isNfDateIntUtc(outerZoneView)) {
        return false;
      }
      const std::string_view rowZoneView(rowTimeZone.data(), rowTimeZone.size());
      const auto localSeconds = fusedLocalEpochSeconds(input, rowZoneView);
      if (!localSeconds.has_value()) {
        return false;
      }
      const auto dateInt = epochSecondsToDateInt(*localSeconds);
      if (!dateInt.has_value()) {
        return false;
      }
      result = *dateInt;
      return true;
    }
  };
};

struct NfHourFromUnixTimeTzFunction {
  template <typename T>
  struct udf {
    VELOX_DEFINE_FUNCTION_TYPES(T);

    FOLLY_ALWAYS_INLINE bool call(
        int32_t& result,
        int64_t input,
        const arg_type<facebook::velox::Varchar>& rowTimeZone,
        const arg_type<facebook::velox::Varchar>& outerTimeZone) {
      const std::string_view outerZoneView(outerTimeZone.data(), outerTimeZone.size());
      if (!isNfDateIntUtc(outerZoneView)) {
        return false;
      }
      const std::string_view rowZoneView(rowTimeZone.data(), rowTimeZone.size());
      const auto localSeconds = fusedLocalEpochSeconds(input, rowZoneView);
      if (!localSeconds.has_value()) {
        return false;
      }
      const auto secondOfDay = ((*localSeconds % kSecondsPerDay) + kSecondsPerDay) % kSecondsPerDay;
      result = static_cast<int32_t>(secondOfDay / 3'600);
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

std::optional<int32_t> parseFixedOffsetSeconds(std::string_view zone) {
  if (normalizeJavaTimeZone(zone) == "UTC") {
    return 0;
  }

  auto value = zone;
  for (const auto prefix : {std::string_view("GMT"), std::string_view("UTC"), std::string_view("UT")}) {
    if (value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix) {
      value.remove_prefix(prefix.size());
      break;
    }
  }
  if (value.size() < 2 || (value.front() != '+' && value.front() != '-')) {
    return std::nullopt;
  }
  const bool negative = value.front() == '-';
  value.remove_prefix(1);

  auto parsePart = [](std::string_view part) -> std::optional<int32_t> {
    if (part.empty() || part.size() > 2) {
      return std::nullopt;
    }
    int32_t parsed = 0;
    for (const char c : part) {
      if (c < '0' || c > '9') {
        return std::nullopt;
      }
      parsed = parsed * 10 + (c - '0');
    }
    return parsed;
  };

  int32_t hour = 0;
  int32_t minute = 0;
  int32_t second = 0;
  if (const auto firstColon = value.find(':'); firstColon != std::string_view::npos) {
    if ((value.size() != 5 && value.size() != 8) || firstColon != 2 || (value.size() == 8 && value[5] != ':')) {
      return std::nullopt;
    }
    const auto secondColon = value.find(':', firstColon + 1);
    const auto parsedHour = parsePart(value.substr(0, firstColon));
    const auto parsedMinute = parsePart(value.substr(
        firstColon + 1, secondColon == std::string_view::npos ? std::string_view::npos : secondColon - firstColon - 1));
    if (!parsedHour.has_value() || !parsedMinute.has_value()) {
      return std::nullopt;
    }
    hour = *parsedHour;
    minute = *parsedMinute;
    if (secondColon != std::string_view::npos) {
      const auto parsedSecond = parsePart(value.substr(secondColon + 1));
      if (!parsedSecond.has_value()) {
        return std::nullopt;
      }
      second = *parsedSecond;
    }
  } else {
    if (value.size() <= 2) {
      const auto parsedHour = parsePart(value);
      if (!parsedHour.has_value()) {
        return std::nullopt;
      }
      hour = *parsedHour;
    } else if (value.size() == 4 || value.size() == 6) {
      const auto parsedHour = parsePart(value.substr(0, 2));
      const auto parsedMinute = parsePart(value.substr(2, 2));
      if (!parsedHour.has_value() || !parsedMinute.has_value()) {
        return std::nullopt;
      }
      hour = *parsedHour;
      minute = *parsedMinute;
      if (value.size() == 6) {
        const auto parsedSecond = parsePart(value.substr(4, 2));
        if (!parsedSecond.has_value()) {
          return std::nullopt;
        }
        second = *parsedSecond;
      }
    } else {
      return std::nullopt;
    }
  }
  if (hour > 18 || minute > 59 || second > 59 || (hour == 18 && (minute != 0 || second != 0))) {
    return std::nullopt;
  }
  const int32_t total = hour * 3'600 + minute * 60 + second;
  return negative ? -total : total;
}

std::vector<std::string>
distinctZonesToHost(cudf::column_view zones, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) {
  if (zones.is_empty() || zones.null_count() == zones.size()) {
    return {};
  }
  auto distinct = cudf::distinct(
      cudf::table_view{{zones}},
      {0},
      cudf::duplicate_keep_option::KEEP_ANY,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream,
      mr);
  const cudf::strings_column_view strings(distinct->view().column(0));
  const auto chars = cudf::detail::make_host_vector(
      cudf::device_span<const char>(strings.chars_begin(stream), strings.chars_size(stream)), stream);

  std::vector<std::string> result;
  result.reserve(strings.size());
  const auto offsets = strings.offsets();
  if (offsets.type().id() == cudf::type_id::INT32) {
    const auto hostOffsets = cudf::detail::make_host_vector(
        cudf::device_span<const int32_t>(offsets.data<int32_t>() + strings.offset(), strings.size() + 1), stream);
    for (cudf::size_type i = 0; i < strings.size(); ++i) {
      const auto length = hostOffsets[i + 1] - hostOffsets[i];
      if (length == 0) {
        result.emplace_back();
      } else {
        result.emplace_back(chars.data() + hostOffsets[i], static_cast<size_t>(length));
      }
    }
  } else {
    VELOX_CHECK(
        offsets.type().id() == cudf::type_id::INT64, "fused Netflix timezone strings require INT32 or INT64 offsets");
    const auto hostOffsets = cudf::detail::make_host_vector(
        cudf::device_span<const int64_t>(offsets.data<int64_t>() + strings.offset(), strings.size() + 1), stream);
    for (cudf::size_type i = 0; i < strings.size(); ++i) {
      const auto length = hostOffsets[i + 1] - hostOffsets[i];
      if (length == 0) {
        result.emplace_back();
      } else {
        result.emplace_back(chars.data() + hostOffsets[i], static_cast<size_t>(length));
      }
    }
  }
  return result;
}

std::unique_ptr<cudf::column>
cudfFusedEpochSeconds(cudf::column_view input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) {
  cudf::numeric_scalar<int64_t> secondsMin(-kEpochSecondsMax, true, stream, mr);
  cudf::numeric_scalar<int64_t> secondsMax(kEpochSecondsMax, true, stream, mr);
  cudf::numeric_scalar<int64_t> millisMin(-kEpochMillisMax, true, stream, mr);
  cudf::numeric_scalar<int64_t> millisMax(kEpochMillisMax, true, stream, mr);
  cudf::numeric_scalar<int64_t> thousand(1'000, true, stream, mr);
  cudf::numeric_scalar<int64_t> nullLong(0, false, stream, mr);

  auto secondsLower = cudf::binary_operation(
      input, secondsMin, cudf::binary_operator::GREATER_EQUAL, cudf::data_type(cudf::type_id::BOOL8), stream, mr);
  auto secondsUpper = cudf::binary_operation(
      input, secondsMax, cudf::binary_operator::LESS_EQUAL, cudf::data_type(cudf::type_id::BOOL8), stream, mr);
  auto isSeconds = cudf::binary_operation(
      secondsLower->view(),
      secondsUpper->view(),
      cudf::binary_operator::LOGICAL_AND,
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      mr);
  auto millisLower = cudf::binary_operation(
      input, millisMin, cudf::binary_operator::GREATER_EQUAL, cudf::data_type(cudf::type_id::BOOL8), stream, mr);
  auto millisUpper = cudf::binary_operation(
      input, millisMax, cudf::binary_operator::LESS_EQUAL, cudf::data_type(cudf::type_id::BOOL8), stream, mr);
  auto isValid = cudf::binary_operation(
      millisLower->view(),
      millisUpper->view(),
      cudf::binary_operator::LOGICAL_AND,
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      mr);
  auto millisAsSeconds = cudf::binary_operation(
      input, thousand, cudf::binary_operator::FLOOR_DIV, cudf::data_type(cudf::type_id::INT64), stream, mr);
  auto classified = cudf::copy_if_else(input, millisAsSeconds->view(), isSeconds->view(), stream, mr);
  auto nullable = cudf::copy_if_else(classified->view(), nullLong, isValid->view(), stream, mr);
  auto duration = cudf::cast(nullable->view(), cudf::data_type(cudf::type_id::DURATION_SECONDS), stream, mr);
  return cudf::cast(duration->view(), cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS), stream, mr);
}

std::unique_ptr<cudf::column> clampUpperBoundIndex(
    std::unique_ptr<cudf::column> upperBound,
    int32_t zeroIndex,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  cudf::numeric_scalar<int32_t> zero(0, true, stream, mr);
  cudf::numeric_scalar<int32_t> zeroReplacement(zeroIndex, true, stream, mr);
  cudf::numeric_scalar<int32_t> one(1, true, stream, mr);
  auto isZero = cudf::binary_operation(
      upperBound->view(), zero, cudf::binary_operator::EQUAL, cudf::data_type(cudf::type_id::BOOL8), stream, mr);
  auto previous = cudf::binary_operation(
      upperBound->view(), one, cudf::binary_operator::SUB, cudf::data_type(cudf::type_id::INT32), stream, mr);
  return cudf::copy_if_else(zeroReplacement, previous->view(), isZero->view(), stream, mr);
}

std::unique_ptr<cudf::column> shiftByTransitionTable(
    cudf::column_view epochSeconds,
    cudf::table_view transitions,
    int32_t historicalOffsetSeconds,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(transitions.num_columns(), 2);
  VELOX_CHECK_GE(transitions.num_rows(), static_cast<cudf::size_type>(cudf::solar_cycle_entry_count + 1));
  const auto fileCount = transitions.num_rows() - static_cast<cudf::size_type>(cudf::solar_cycle_entry_count);
  const auto transitionTimes = transitions.column(0);
  const auto fileTimes = cudf::slice(transitionTimes, {0, fileCount}).front();
  const auto cycleTimes = cudf::slice(transitionTimes, {fileCount, transitions.num_rows()}).front();

  auto fileIndex = clampUpperBoundIndex(
      cudf::upper_bound(
          cudf::table_view{{fileTimes}},
          cudf::table_view{{epochSeconds}},
          {cudf::order::ASCENDING},
          {cudf::null_order::BEFORE},
          stream,
          mr),
      0,
      stream,
      mr);

  auto epochDuration = cudf::cast(epochSeconds, cudf::data_type(cudf::type_id::DURATION_SECONDS), stream, mr);
  auto epochIntegral = cudf::cast(epochDuration->view(), cudf::data_type(cudf::type_id::INT64), stream, mr);
  constexpr int64_t kSolarCycleSeconds = 12'622'780'800LL;
  cudf::numeric_scalar<int64_t> cycleSeconds(kSolarCycleSeconds, true, stream, mr);
  auto plusCycle = cudf::binary_operation(
      epochIntegral->view(),
      cycleSeconds,
      cudf::binary_operator::ADD,
      cudf::data_type(cudf::type_id::INT64),
      stream,
      mr);
  auto projectedIntegral = cudf::binary_operation(
      plusCycle->view(), cycleSeconds, cudf::binary_operator::PMOD, cudf::data_type(cudf::type_id::INT64), stream, mr);
  auto projectedDuration =
      cudf::cast(projectedIntegral->view(), cudf::data_type(cudf::type_id::DURATION_SECONDS), stream, mr);
  auto projectedTimestamp =
      cudf::cast(projectedDuration->view(), cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS), stream, mr);
  auto cycleIndex = clampUpperBoundIndex(
      cudf::upper_bound(
          cudf::table_view{{cycleTimes}},
          cudf::table_view{{projectedTimestamp->view()}},
          {cudf::order::ASCENDING},
          {cudf::null_order::BEFORE},
          stream,
          mr),
      static_cast<int32_t>(cudf::solar_cycle_entry_count - 1),
      stream,
      mr);
  cudf::numeric_scalar<int32_t> fileCountScalar(fileCount, true, stream, mr);
  auto absoluteCycleIndex = cudf::binary_operation(
      cycleIndex->view(),
      fileCountScalar,
      cudf::binary_operator::ADD,
      cudf::data_type(cudf::type_id::INT32),
      stream,
      mr);

  auto lastFileTime = cudf::get_element(transitionTimes, fileCount - 1, stream, mr);
  auto inFileRange = cudf::binary_operation(
      epochSeconds,
      *lastFileTime,
      cudf::binary_operator::LESS_EQUAL,
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      mr);
  cudf::numeric_scalar<bool> falseValue(false, true, stream, mr);
  auto nonNullInFileRange = cudf::replace_nulls(inFileRange->view(), falseValue, stream, mr);
  auto selectedIndex =
      cudf::copy_if_else(fileIndex->view(), absoluteCycleIndex->view(), nonNullInFileRange->view(), stream, mr);
  auto gatheredOffsets = cudf::gather(
      cudf::table_view{{transitions.column(1)}},
      selectedIndex->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      mr);
  auto offsetColumns = gatheredOffsets->release();
  auto firstFileTime = cudf::get_element(transitionTimes, 0, stream, mr);
  auto beforeFirstFileTime = cudf::binary_operation(
      epochSeconds, *firstFileTime, cudf::binary_operator::LESS, cudf::data_type(cudf::type_id::BOOL8), stream, mr);
  auto nonNullBeforeFirstFileTime = cudf::replace_nulls(beforeFirstFileTime->view(), falseValue, stream, mr);
  cudf::duration_scalar<cudf::duration_s> historicalOffset(cudf::duration_s{historicalOffsetSeconds}, true, stream, mr);
  auto effectiveOffsets = cudf::copy_if_else(
      historicalOffset, offsetColumns.front()->view(), nonNullBeforeFirstFileTime->view(), stream, mr);
  auto shiftedDuration = cudf::binary_operation(
      epochDuration->view(),
      effectiveOffsets->view(),
      cudf::binary_operator::ADD,
      cudf::data_type(cudf::type_id::DURATION_SECONDS),
      stream,
      mr);
  return cudf::cast(shiftedDuration->view(), cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS), stream, mr);
}

std::unique_ptr<cudf::column> shiftedTimestampForZone(
    cudf::column_view epochSeconds,
    std::string_view rowZone,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto normalized = normalizeJavaTimeZone(rowZone);
  if (normalized.empty()) {
    throw cudf::logic_error("empty timezone is invalid");
  }
  if (const auto fixedOffset = parseFixedOffsetSeconds(normalized); fixedOffset.has_value()) {
    auto duration = cudf::cast(epochSeconds, cudf::data_type(cudf::type_id::DURATION_SECONDS), stream, mr);
    cudf::duration_scalar<cudf::duration_s> offset(cudf::duration_s{*fixedOffset}, true, stream, mr);
    auto shifted = cudf::binary_operation(
        duration->view(),
        offset,
        cudf::binary_operator::ADD,
        cudf::data_type(cudf::type_id::DURATION_SECONDS),
        stream,
        mr);
    return cudf::cast(shifted->view(), cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS), stream, mr);
  }
  auto transitions = cudf::make_timezone_transition_table(std::nullopt, normalized, stream, mr);
  if (transitions->num_columns() == 0) {
    return std::make_unique<cudf::column>(epochSeconds, stream, mr);
  }
  const auto historicalOffset = historicalTimeZoneOffsetSeconds(normalized);
  if (!historicalOffset.has_value()) {
    throw cudf::logic_error("timezone offset is unavailable");
  }
  return shiftByTransitionTable(epochSeconds, transitions->view(), *historicalOffset, stream, mr);
}

enum class FusedDateTimeField { kDateInt, kHour };

std::unique_ptr<cudf::column> extractFusedField(
    cudf::column_view shiftedTimestamp,
    FusedDateTimeField field,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (field == FusedDateTimeField::kDateInt) {
    auto days = cudf::cast(shiftedTimestamp, cudf::data_type(cudf::type_id::TIMESTAMP_DAYS), stream, mr);
    return cudfDateIntFromDays(days->view(), stream, mr);
  }
  auto hour = cudf::datetime::extract_datetime_component(
      shiftedTimestamp, cudf::datetime::datetime_component::HOUR, stream, mr);
  return cudf::cast(hour->view(), cudf::data_type(cudf::type_id::INT32), stream, mr);
}

class CudfNfFromUnixTimeTzFieldFunction final : public CudfFunction {
 public:
  CudfNfFromUnixTimeTzFieldFunction(const std::shared_ptr<facebook::velox::exec::Expr>& expr, FusedDateTimeField field)
      : field_(field) {
    const auto expectedInputs = field == FusedDateTimeField::kDateInt ? 4 : 3;
    VELOX_CHECK_EQ(expr->inputs().size(), expectedInputs);
    const auto rowZoneExpr = std::dynamic_pointer_cast<facebook::velox::exec::ConstantExpr>(expr->inputs()[1]);
    if (rowZoneExpr != nullptr) {
      constantRowZone_ = rowZoneExpr->value()->toString(0);
    }

    const auto outerZoneIndex = expectedInputs - 1;
    const auto outerZoneExpr =
        std::dynamic_pointer_cast<facebook::velox::exec::ConstantExpr>(expr->inputs()[outerZoneIndex]);
    VELOX_CHECK_NOT_NULL(outerZoneExpr, "fused Netflix outer timezone must be constant");
    VELOX_CHECK(
        isNfDateIntUtc(outerZoneExpr->value()->toString(0)),
        "fused Netflix datetime currently supports only outer UTC");

    if (field == FusedDateTimeField::kDateInt) {
      const auto formatExpr = std::dynamic_pointer_cast<facebook::velox::exec::ConstantExpr>(expr->inputs()[2]);
      VELOX_CHECK_NOT_NULL(formatExpr, "fused nf_dateint format must be constant");
      VELOX_CHECK_EQ(formatExpr->value()->toString(0), "-", "fused nf_dateint currently supports only format '-'");
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    const auto expectedColumns = constantRowZone_.has_value() ? 1 : 2;
    VELOX_CHECK_EQ(inputColumns.size(), expectedColumns);
    const auto epochInput = facebook::velox::cudf_velox::asView(inputColumns[0]);
    auto epochSeconds = cudfFusedEpochSeconds(epochInput, stream, mr);

    if (constantRowZone_.has_value()) {
      try {
        auto shifted = shiftedTimestampForZone(epochSeconds->view(), *constantRowZone_, stream, mr);
        return extractFusedField(shifted->view(), field_, stream, mr);
      } catch (const cudf::logic_error&) {
        return cudf::make_numeric_column(
            cudf::data_type(cudf::type_id::INT32), epochInput.size(), cudf::mask_state::ALL_NULL, stream, mr);
      }
    }

    const auto zones = facebook::velox::cudf_velox::asView(inputColumns[1]);
    auto result = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::INT32), epochInput.size(), cudf::mask_state::ALL_NULL, stream, mr);
    for (const auto& originalZone : distinctZonesToHost(zones, stream, mr)) {
      if (originalZone.empty()) {
        continue;
      }
      try {
        auto shifted = shiftedTimestampForZone(epochSeconds->view(), originalZone, stream, mr);
        auto fieldValues = extractFusedField(shifted->view(), field_, stream, mr);
        cudf::string_scalar zoneScalar(originalZone, true, stream, mr);
        auto selectedZone = cudf::binary_operation(
            zones, zoneScalar, cudf::binary_operator::EQUAL, cudf::data_type(cudf::type_id::BOOL8), stream, mr);
        result = cudf::copy_if_else(fieldValues->view(), result->view(), selectedZone->view(), stream, mr);
      } catch (const cudf::logic_error&) {
        // Netflix returns null for an invalid per-row timezone in non-strict
        // mode. Rows for this distinct zone remain null in the accumulator.
      }
    }
    return result;
  }

 private:
  FusedDateTimeField field_;
  std::optional<std::string> constantRowZone_;
};

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

void registerCudfNfFromUnixTimeTzFieldFunction(const std::string& name, FusedDateTimeField field) {
  using facebook::velox::exec::FunctionSignatureBuilder;
  const auto signature = field == FusedDateTimeField::kDateInt ? FunctionSignatureBuilder()
                                                                     .returnType("integer")
                                                                     .argumentType("bigint")
                                                                     .argumentType("varchar")
                                                                     .constantArgumentType("varchar")
                                                                     .constantArgumentType("varchar")
                                                                     .build()
                                                               : FunctionSignatureBuilder()
                                                                     .returnType("integer")
                                                                     .argumentType("bigint")
                                                                     .argumentType("varchar")
                                                                     .constantArgumentType("varchar")
                                                                     .build();
  facebook::velox::cudf_velox::registerCudfFunction(
      name,
      [field](const std::string&, const std::shared_ptr<facebook::velox::exec::Expr>& expr) {
        return std::make_shared<CudfNfFromUnixTimeTzFieldFunction>(expr, field);
      },
      {signature});
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
      NfDateIntFromUnixTimeTzFunction,
      int32_t,
      int64_t,
      facebook::velox::Varchar,
      facebook::velox::Varchar,
      facebook::velox::Varchar>({"nf_dateint_from_unixtime_tz"});
  facebook::velox::registerFunction<
      NfHourFromUnixTimeTzFunction,
      int32_t,
      int64_t,
      facebook::velox::Varchar,
      facebook::velox::Varchar>({"nf_hour_from_unixtime_tz"});
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
  registerCudfNfFromUnixTimeTzFieldFunction("nf_dateint_from_unixtime_tz", FusedDateTimeField::kDateInt);
  registerCudfNfFromUnixTimeTzFieldFunction("nf_hour_from_unixtime_tz", FusedDateTimeField::kHour);
  registerCudfNetflixDateTimeFunction("nf_to_unixtime", false);
  registerCudfNetflixDateTimeFunction("nf_to_unixtime_ms", true);
#endif
}

} // namespace gluten
