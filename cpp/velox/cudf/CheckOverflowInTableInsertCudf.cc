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

#include "cudf/CheckOverflowInTableInsertCudf.h"

#include <limits>

#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>

#include "operators/functions/CheckOverflowInTableInsert.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FunctionSignature.h"

namespace gluten {
namespace {

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;

constexpr const char* kUnsupported =
    "check_overflow_in_table_insert cuDF adapter only supports "
    "(BIGINT, constant VARCHAR) -> INTEGER";

class CheckOverflowInTableInsertCudfFunction final : public CudfFunction {
 public:
  explicit CheckOverflowInTableInsertCudfFunction(const std::shared_ptr<exec::Expr>& expr) {
    if (expr->inputs().size() != 2 || expr->type()->kind() != TypeKind::INTEGER ||
        expr->inputs()[0]->type()->kind() != TypeKind::BIGINT ||
        expr->inputs()[1]->type()->kind() != TypeKind::VARCHAR) {
      VELOX_UNSUPPORTED("{}", kUnsupported);
    }

    const auto columnNameExpr = std::dynamic_pointer_cast<exec::ConstantExpr>(expr->inputs()[1]);
    if (columnNameExpr == nullptr || columnNameExpr->value() == nullptr || columnNameExpr->value()->isNullAt(0)) {
      VELOX_UNSUPPORTED("{}", kUnsupported);
    }
    columnName_ = columnNameExpr->value()->toString(0);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1, "check_overflow_in_table_insert expects one non-constant input");
    const auto input = asView(inputColumns[0]);
    VELOX_CHECK_EQ(static_cast<int>(input.type().id()), static_cast<int>(cudf::type_id::INT64));

    auto [minimum, maximum] = cudf::minmax(input, stream, mr);
    if (minimum->is_valid(stream)) {
      VELOX_CHECK_EQ(static_cast<int>(minimum->type().id()), static_cast<int>(cudf::type_id::INT64));
      VELOX_CHECK_EQ(static_cast<int>(maximum->type().id()), static_cast<int>(cudf::type_id::INT64));
      const auto minValue = static_cast<const cudf::numeric_scalar<int64_t>&>(*minimum).value(stream);
      const auto maxValue = static_cast<const cudf::numeric_scalar<int64_t>&>(*maximum).value(stream);
      if (minValue < std::numeric_limits<int32_t>::min() || maxValue > std::numeric_limits<int32_t>::max()) {
        VELOX_USER_FAIL("{}", checkOverflowInTableInsertMessage(columnName_));
      }
    }

    return cudf::cast(input, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  }

 private:
  std::string columnName_;
};

} // namespace

void registerCheckOverflowInTableInsertCudfFunction(const std::string& prefix) {
  using facebook::velox::exec::FunctionSignatureBuilder;

  facebook::velox::cudf_velox::registerCudfFunction(
      prefix + kCheckOverflowInTableInsert,
      [](const std::string&, const std::shared_ptr<exec::Expr>& expr) {
        return std::make_shared<CheckOverflowInTableInsertCudfFunction>(expr);
      },
      {FunctionSignatureBuilder().returnType("integer").argumentType("bigint").constantArgumentType("varchar").build()},
      false);
}

} // namespace gluten
