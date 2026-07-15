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

#include "operators/functions/CheckOverflowInTableInsert.h"

#include <limits>

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/SpecialForm.h"
#include "velox/expression/SpecialFormRegistry.h"
#include "velox/functions/sparksql/specialforms/SparkCastExpr.h"
#include "velox/functions/sparksql/specialforms/SparkCastHooks.h"
#include "velox/vector/DecodedVector.h"

namespace gluten {
namespace {

using namespace facebook::velox;

constexpr const char* kUnsupported =
    "check_overflow_in_table_insert only supports "
    "(BIGINT, constant VARCHAR) -> INTEGER";

class CheckOverflowInTableInsertExpr final : public exec::SpecialForm {
 public:
  CheckOverflowInTableInsertExpr(
      TypePtr type,
      std::vector<exec::ExprPtr>&& inputs,
      std::string columnName,
      bool trackCpuUsage,
      const core::QueryConfig& config)
      : SpecialForm(
            exec::SpecialFormKind::kCustom,
            std::move(type),
            std::move(inputs),
            kCheckOverflowInTableInsert,
            false,
            trackCpuUsage),
        columnName_(std::move(columnName)) {
    exec::ExprPtr source = inputs_[0];
    checkedCast_ = std::make_unique<functions::sparksql::SparkCastExpr>(
        type_,
        std::move(source),
        trackCpuUsage,
        false,
        std::make_shared<functions::sparksql::SparkCastHooks>(config, false));
  }

  void evalSpecialForm(const SelectivityVector& rows, exec::EvalCtx& context, VectorPtr& result) override {
    VectorPtr input;
    inputs_[0]->eval(rows, context, input);

    DecodedVector decoded(*input, rows);
    rows.applyToSelected([&](vector_size_t row) {
      if (decoded.isNullAt(row)) {
        return;
      }
      const auto value = decoded.valueAt<int64_t>(row);
      if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
        VELOX_USER_FAIL("{}", checkOverflowInTableInsertMessage(columnName_));
      }
    });

    checkedCast_->apply(rows, input, context, BIGINT(), INTEGER(), result);
    context.releaseVector(input);
  }

  void computePropagatesNulls() override {
    propagatesNulls_ = true;
  }

 private:
  const std::string columnName_;
  std::unique_ptr<functions::sparksql::SparkCastExpr> checkedCast_;
};

} // namespace

std::string checkOverflowInTableInsertMessage(const std::string& quotedColumnName) {
  return fmt::format(
      "[CAST_OVERFLOW_IN_TABLE_INSERT] Fail to assign a value of \"BIGINT\" "
      "type to the \"INT\" type column or variable {} due to an overflow. "
      "Use `try_cast` on the input value to tolerate overflow and return NULL "
      "instead. SQLSTATE: 22003",
      quotedColumnName);
}

facebook::velox::TypePtr CheckOverflowInTableInsertCallToSpecialForm::resolveType(
    const std::vector<facebook::velox::TypePtr>& argTypes) {
  if (argTypes.size() != 2 || argTypes[0]->kind() != TypeKind::BIGINT || argTypes[1]->kind() != TypeKind::VARCHAR) {
    VELOX_UNSUPPORTED("{}", kUnsupported);
  }
  return INTEGER();
}

facebook::velox::exec::ExprPtr CheckOverflowInTableInsertCallToSpecialForm::constructSpecialForm(
    const facebook::velox::TypePtr& type,
    std::vector<facebook::velox::exec::ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const facebook::velox::core::QueryConfig& config) {
  if (compiledChildren.size() != 2 || type->kind() != TypeKind::INTEGER ||
      compiledChildren[0]->type()->kind() != TypeKind::BIGINT ||
      compiledChildren[1]->type()->kind() != TypeKind::VARCHAR) {
    VELOX_UNSUPPORTED("{}", kUnsupported);
  }

  const auto columnNameExpr = std::dynamic_pointer_cast<facebook::velox::exec::ConstantExpr>(compiledChildren[1]);
  if (columnNameExpr == nullptr || columnNameExpr->value() == nullptr || columnNameExpr->value()->isNullAt(0)) {
    VELOX_UNSUPPORTED("{}", kUnsupported);
  }
  auto columnName = columnNameExpr->value()->toString(0);

  return std::make_shared<CheckOverflowInTableInsertExpr>(
      type, std::move(compiledChildren), std::move(columnName), trackCpuUsage, config);
}

void registerCheckOverflowInTableInsert() {
  facebook::velox::exec::registerFunctionCallToSpecialForm(
      kCheckOverflowInTableInsert, std::make_unique<CheckOverflowInTableInsertCallToSpecialForm>());
}

} // namespace gluten
