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
package org.apache.gluten.execution

import org.apache.gluten.expression.{ConverterUtils, ExpressionNames}
import org.apache.gluten.expression.ConverterUtils.FunctionConfig
import org.apache.gluten.substrait.SubstraitContext

import org.apache.spark.sql.catalyst.expressions.Literal
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateMode, Count, Final, Partial, PartialMerge}
import org.apache.spark.sql.types.{IntegerType, LongType}

import org.scalatest.funsuite.AnyFunSuite

class VeloxAggregateFunctionsBuilderSuite extends AnyFunSuite {
  test("count merge modes register sum companion") {
    Seq(PartialMerge, Final).foreach { mode =>
      val registered = registerCount(mode)

      assert(registered.contains(sumMergeFunction))
      assert(!registered.contains(countMergeFunction))
    }
  }

  test("count non-merge mode keeps count function") {
    assert(registerCount(Partial).contains(countInputFunction))
  }

  private def registerCount(mode: AggregateMode): Set[String] = {
    val context = new SubstraitContext
    VeloxAggregateFunctionsBuilder.create(context, Count(Seq(Literal(1))), mode)
    context.registeredFunction.keySet().toArray.map(_.toString).toSet
  }

  private def sumMergeFunction: String =
    ConverterUtils.makeFuncName(ExpressionNames.SUM, Seq(LongType), FunctionConfig.REQ)

  private def countMergeFunction: String =
    ConverterUtils.makeFuncName(ExpressionNames.COUNT, Seq(LongType), FunctionConfig.REQ)

  private def countInputFunction: String =
    ConverterUtils.makeFuncName(ExpressionNames.COUNT, Seq(IntegerType), FunctionConfig.REQ)
}
