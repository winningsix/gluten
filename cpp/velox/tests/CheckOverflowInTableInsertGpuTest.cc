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
#include <optional>

#include "cudf/CheckOverflowInTableInsertCudf.h"
#include "operators/functions/CheckOverflowInTableInsert.h"
#include "operators/functions/RegistrationAllFunctions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"
#include "velox/parse/TypeResolver.h"

namespace gluten {
namespace {

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;

class CheckOverflowInTableInsertGpuTest : public CudfFunctionBaseTest {
 protected:
  using functions::test::FunctionBaseTest::evaluate;

  static void SetUpTestCase() {
    parse::registerTypeResolver();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    registerAllFunctions();
    cudf_velox::registerCudf();
    registerCheckOverflowInTableInsertCudfFunction(CudfConfig::getInstance().functionNamePrefix);
  }

  static void TearDownTestCase() {
    cudf_velox::unregisterFunctions();
    cudf_velox::unregisterCudf();
  }
};

TEST_F(CheckOverflowInTableInsertGpuTest, preservesCustomRootAndSafeValues) {
  const auto expression = "check_overflow_in_table_insert(c0, '`target_col`')";
  auto exprSet = compileExpression(expression, ROW({"c0"}, {BIGINT()}));
  const auto& compiled = exprSet->exprs()[0];
  ASSERT_EQ(compiled->name(), kCheckOverflowInTableInsert);
  ASSERT_EQ(compiled->inputs().size(), 2);
  ASSERT_EQ(compiled->inputs()[0]->type()->kind(), TypeKind::BIGINT);
  ASSERT_EQ(compiled->inputs()[1]->type()->kind(), TypeKind::VARCHAR);
  ASSERT_TRUE(canBeEvaluatedByCudf(compiled, true));

  const auto input = makeNullableFlatVector<int64_t>(
      {std::numeric_limits<int32_t>::min(), -1, 0, std::numeric_limits<int32_t>::max(), std::nullopt});
  const auto result = evaluate<SimpleVector<int32_t>>(expression, makeRowVector({input}));
  ASSERT_EQ(result->valueAt(0), std::numeric_limits<int32_t>::min());
  ASSERT_EQ(result->valueAt(1), -1);
  ASSERT_EQ(result->valueAt(2), 0);
  ASSERT_EQ(result->valueAt(3), std::numeric_limits<int32_t>::max());
  ASSERT_TRUE(result->isNullAt(4));
}

TEST_F(CheckOverflowInTableInsertGpuTest, rejectsBothOverflowDirections) {
  const auto expression =
      "check_overflow_in_table_insert(c0, "
      "'`target_db`.`is_accounting_expired`')";
  VELOX_ASSERT_THROW(
      evaluate<SimpleVector<int32_t>>(
          expression,
          makeRowVector({makeFlatVector<int64_t>({static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1})})),
      "[CAST_OVERFLOW_IN_TABLE_INSERT]");
  VELOX_ASSERT_THROW(
      evaluate<SimpleVector<int32_t>>(
          expression,
          makeRowVector({makeFlatVector<int64_t>({static_cast<int64_t>(std::numeric_limits<int32_t>::min()) - 1})})),
      "`target_db`.`is_accounting_expired`");
}

TEST_F(CheckOverflowInTableInsertGpuTest, allNullAndEmptyInputsDoNotOverflow) {
  const auto expression = "check_overflow_in_table_insert(c0, '`target_col`')";
  const auto allNullResult = evaluate<SimpleVector<int32_t>>(
      expression, makeRowVector({makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt})}));
  ASSERT_EQ(allNullResult->size(), 2);
  ASSERT_TRUE(allNullResult->isNullAt(0));
  ASSERT_TRUE(allNullResult->isNullAt(1));

  const auto emptyResult =
      evaluate<SimpleVector<int32_t>>(expression, makeRowVector({makeFlatVector<int64_t>(std::vector<int64_t>{})}));
  ASSERT_EQ(emptyResult->size(), 0);
}

} // namespace
} // namespace gluten
