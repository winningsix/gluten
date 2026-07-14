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
package org.apache.gluten.expression

import org.apache.gluten.exception.GlutenNotSupportException
import org.apache.gluten.substrait.SubstraitContext

import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Expression, Literal}
import org.apache.spark.sql.types.{DateType, IntegerType, StringType, TimestampType}

import com.netflix.bdp.expressions.{NfDateInt, NfToUnixTime, NfToUnixTimeMs}
import org.scalatest.funsuite.AnyFunSuite

class NetflixDateTimeExpressionTransformerSuite extends AnyFunSuite {
  private def transformChild(expression: Expression): ExpressionTransformer = expression match {
    case literal: Literal => LiteralTransformer(literal)
    case other => fail(s"unexpected child in focused test: $other")
  }

  test("maps exact Netflix classes and propagates timezone metadata") {
    val seconds = NetflixDateTimeExpressionTransformer
      .tryTransform(
        NfToUnixTime(Literal(20240101), Literal("-"), Some("Etc/UTC")),
        transformChild,
        "America/Los_Angeles")
      .get
    val millis = NetflixDateTimeExpressionTransformer
      .tryTransform(
        NfToUnixTimeMs(Literal(1704067200000L), Literal("-"), None),
        transformChild,
        "UTC")
      .get

    assert(seconds.substraitExprName === "nf_to_unixtime")
    assert(seconds.children.size === 3)
    assert(seconds.children.last.original === Literal("Etc/UTC"))
    assert(millis.substraitExprName === "nf_to_unixtime_ms")
    assert(millis.children.last.original === Literal("UTC"))

    val context = new SubstraitContext
    seconds.doTransform(context)
    assert(context.registeredFunction.keySet().toArray.exists {
      _.toString.startsWith("nf_to_unixtime:i32_str_str")
    })
  }

  test("maps the exact NfDateInt Job175 INT slice") {
    val dateInt = NetflixDateTimeExpressionTransformer
      .tryTransform(
        NfDateInt(Literal(20260531), Literal("-"), Some("Etc/UTC")),
        transformChild,
        "America/Los_Angeles")
      .get

    assert(dateInt.substraitExprName === "nf_dateint")
    assert(dateInt.dataType === IntegerType)
    assert(dateInt.nullable)
    assert(dateInt.children.size === 3)
    assert(dateInt.children.last.original === Literal("Etc/UTC"))

    val context = new SubstraitContext
    dateInt.doTransform(context)
    assert(context.registeredFunction.keySet().toArray.exists {
      _.toString.startsWith("nf_dateint:i32_str_str")
    })
  }

  test("does not match a class by simple name") {
    case class NfToUnixTime(date: Expression, format: Expression) extends BinaryExpressionForTest
    case class NfDateInt(date: Expression, format: Expression) extends BinaryExpressionForTest
    val expressions =
      Seq(NfToUnixTime(Literal(20240101), Literal("-")), NfDateInt(Literal(20260531), Literal("-")))

    expressions.foreach {
      expression =>
        assert(
          NetflixDateTimeExpressionTransformer
            .tryTransform(expression, transformChild, "UTC")
            .isEmpty)
    }
  }

  test("rejects unsupported format and timezone explicitly") {
    val explicitFormat = NfToUnixTime(Literal(20240101), Literal("yyyyMMdd"), Some("UTC"))
    val nonUtc =
      NfToUnixTimeMs(Literal(1704067200000L), Literal("-"), Some("America/Los_Angeles"))

    val formatError = intercept[GlutenNotSupportException] {
      NetflixDateTimeExpressionTransformer
        .tryTransform(explicitFormat, transformChild, "UTC")
    }
    val timezoneError = intercept[GlutenNotSupportException] {
      NetflixDateTimeExpressionTransformer
        .tryTransform(nonUtc, transformChild, "UTC")
    }

    assert(formatError.getMessage.contains("supports only the default '-' format"))
    assert(timezoneError.getMessage.contains("supports only UTC"))
  }

  test("NfDateInt rejects every input type outside the Job175 INT slice") {
    Seq(
      Literal(20260531L),
      Literal("20260531"),
      AttributeReference("date", DateType, nullable = true)(),
      AttributeReference("timestamp", TimestampType, nullable = true)()
    )
      .foreach {
        input =>
          val error = intercept[GlutenNotSupportException] {
            NetflixDateTimeExpressionTransformer.tryTransform(
              NfDateInt(input, Literal("-"), Some("UTC")),
              transformChild,
              "UTC")
          }
          assert(error.getMessage.contains("supports INT input"))
      }
  }

  test("NfDateInt rejects dynamic or non-default format and non-UTC timezone") {
    val dynamicFormat = AttributeReference("format", StringType, nullable = false)()
    val cases = Seq(
      NfDateInt(Literal(20260531), dynamicFormat, Some("UTC")) ->
        "format must be a plan-time string literal",
      NfDateInt(Literal(20260531), Literal("yyyyMMdd"), Some("UTC")) ->
        "supports only the default '-' format",
      NfDateInt(Literal(20260531), Literal("-"), Some("America/Los_Angeles")) ->
        "supports only UTC",
      NfDateInt(Literal(20260531), Literal("-"), Some("Z")) ->
        "supports only UTC"
    )

    cases.foreach {
      case (expression, expected) =>
        val error = intercept[GlutenNotSupportException] {
          NetflixDateTimeExpressionTransformer
            .tryTransform(expression, transformChild, "UTC")
        }
        assert(error.getMessage.contains(expected))
    }
  }
}

private trait BinaryExpressionForTest
  extends org.apache.spark.sql.catalyst.expressions.BinaryExpression
  with org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback {
  def date: Expression
  def format: Expression
  override def left: Expression = date
  override def right: Expression = format
  override def dataType: org.apache.spark.sql.types.DataType =
    org.apache.spark.sql.types.LongType
  override def nullable: Boolean = true
  override def nullSafeEval(dateValue: Any, formatValue: Any): Any = null
  override protected def withNewChildrenInternal(
      newLeft: Expression,
      newRight: Expression): Expression = this
}
