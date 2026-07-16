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
package org.apache.spark.sql.catalyst.expressions

import org.apache.gluten.exception.GlutenNotSupportException
import org.apache.gluten.expression.{CheckOverflowInTableInsertTransformer, ExpressionConverter}
import org.apache.gluten.substrait.SubstraitContext

import org.apache.spark.SparkFunSuite
import org.apache.spark.sql.types.{DataType, FloatType, IntegerType, LongType}

class GlutenCheckOverflowInTableInsertSuite extends SparkFunSuite {
  private val source = AttributeReference("source_value", LongType, nullable = true)()

  private def checkOverflow(
      sourceExpression: Expression,
      targetType: DataType,
      columnName: String = "`target_col`",
      evalMode: EvalMode.Value = EvalMode.ANSI): CheckOverflowInTableInsert = {
    CheckOverflowInTableInsert(Cast(sourceExpression, targetType, evalMode = evalMode), columnName)
  }

  test("emit raw BIGINT source and Spark-quoted column path instead of nested Cast") {
    val quotedColumnPath = "`target_db`.`is_accounting_expired`"
    val expression = checkOverflow(source, IntegerType, quotedColumnPath)
    val transformer = ExpressionConverter
      .replaceWithExpressionTransformer(expression, Seq(source))
      .asInstanceOf[CheckOverflowInTableInsertTransformer]

    assert(transformer.source.dataType == LongType)
    assert(transformer.dataType == IntegerType)
    assert(transformer.columnName == quotedColumnPath)

    val context = new SubstraitContext
    val scalar = transformer.doTransform(context).toProtobuf.getScalarFunction
    assert(context.registeredFunction.containsKey("check_overflow_in_table_insert:i64_str"))
    assert(scalar.getArgumentsCount == 2)
    assert(scalar.getArguments(0).getValue.hasSelection)
    assert(scalar.getArguments(1).getValue.getLiteral.getString == quotedColumnPath)
    assert(scalar.getOutputType.hasI32)
  }

  test("fail closed for table-insert casts outside BIGINT to INT") {
    val floatSource = AttributeReference("float_value", FloatType, nullable = true)()
    val error = intercept[GlutenNotSupportException] {
      ExpressionConverter.replaceWithExpressionTransformer(
        checkOverflow(floatSource, IntegerType),
        Seq(floatSource))
    }
    assert(error.getMessage.contains("limited to ANSI BIGINT -> INT"))

    val wrongTarget = intercept[GlutenNotSupportException] {
      ExpressionConverter.replaceWithExpressionTransformer(
        checkOverflow(source, LongType),
        Seq(source))
    }
    assert(wrongTarget.getMessage.contains("limited to ANSI BIGINT -> INT"))

    Seq(EvalMode.LEGACY, EvalMode.TRY).foreach {
      evalMode =>
        val wrongEvalMode = intercept[GlutenNotSupportException] {
          ExpressionConverter.replaceWithExpressionTransformer(
            checkOverflow(source, IntegerType, evalMode = evalMode),
            Seq(source))
        }
        assert(wrongEvalMode.getMessage.contains("limited to ANSI BIGINT -> INT"))
    }
  }
}
