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
import org.apache.gluten.substrait.expression.{ExpressionBuilder, ExpressionNode}

import org.apache.spark.sql.catalyst.expressions.{Expression, Literal, TimeZoneAwareExpression}
import org.apache.spark.sql.types.{DataType, IntegerType, LongType, StringType, TimestampType}
import org.apache.spark.unsafe.types.UTF8String

import scala.collection.JavaConverters._

/**
 * Bounded conversion for Netflix's Catalyst date/time expressions.
 *
 * Gluten intentionally has no compile-time dependency on the Netflix UDF jar. Matching is by the
 * exact Catalyst class name, while timezone metadata is read through Spark's public
 * [[TimeZoneAwareExpression]] contract. The timezone is appended as a literal native-function
 * argument because it is metadata rather than a Catalyst child.
 *
 * The first native slices deliberately accept only the domains exercised by Netflix jobs 321, 215,
 * 216, and 175: bounded integral input, the default `-` format, and UTC. Other domains fail
 * validation instead of silently using different date/time semantics.
 */
private[expression] object NetflixDateTimeExpressionTransformer {
  private val NfDateIntClassName = "com.netflix.bdp.expressions.NfDateInt"
  private val NfFromUnixTimeTzClassName = "com.netflix.bdp.expressions.NfFromUnixTimeTz"
  private val NfHourClassName = "com.netflix.bdp.expressions.NfHour"

  private val NfDateIntFromUnixTimeTzFunctionName = "nf_dateint_from_unixtime_tz"
  private val NfHourFromUnixTimeTzFunctionName = "nf_hour_from_unixtime_tz"

  private case class FunctionSpec(
      functionName: String,
      inputTypes: Set[DataType],
      resultType: DataType,
      inputDomain: String,
      utcZoneIds: Set[String])

  private val StandardUtcZoneIds = Set("UTC", "Etc/UTC", "Z")
  private val NfDateIntUtcZoneIds = Set("UTC", "Etc/UTC")

  private val FunctionsByClassName = Map(
    "com.netflix.bdp.expressions.NfToUnixTime" ->
      FunctionSpec(
        "nf_to_unixtime",
        Set(IntegerType, LongType),
        LongType,
        "INT/BIGINT",
        StandardUtcZoneIds),
    "com.netflix.bdp.expressions.NfToUnixTimeMs" ->
      FunctionSpec(
        "nf_to_unixtime_ms",
        Set(IntegerType, LongType),
        LongType,
        "INT/BIGINT",
        StandardUtcZoneIds),
    // Job 175's snapshot_utc_date is a nullable Iceberg INT. Keep this deliberately narrower than
    // Netflix NfDateInt's full JVM domain: strings, Spark dates/timestamps, and BIGINT remain on
    // the fail-closed path until each domain has independent native parity coverage.
    NfDateIntClassName ->
      FunctionSpec("nf_dateint", Set(IntegerType), IntegerType, "INT", NfDateIntUtcZoneIds)
  )

  def tryTransform(
      expr: Expression,
      transformChild: Expression => ExpressionTransformer,
      sessionTimeZone: => String): Option[ExpressionTransformer] = {
    tryTransformFromUnixTimeTzFusion(expr, transformChild, sessionTimeZone).orElse {
      FunctionsByClassName.get(expr.getClass.getName).map {
        spec =>
          val functionName = spec.functionName
          if (expr.children.size != 2) {
            unsupported(
              functionName,
              s"expected exactly two Catalyst children (date, format), got ${expr.children.size}")
          }

          val Seq(date, format) = expr.children
          if (!spec.inputTypes.contains(date.dataType)) {
            unsupported(
              functionName,
              s"first native slice supports ${spec.inputDomain} input, " +
                s"got ${date.dataType.catalogString}")
          }
          validateResultType(expr, spec.resultType, functionName)
          validateDefaultFormat(format, functionName)
          val timeZoneId =
            validatedOuterTimeZone(expr, spec.utcZoneIds, functionName, sessionTimeZone)

          NetflixDateTimeTransformer(
            functionName,
            Seq(transformChild(date), transformChild(format), transformChild(Literal(timeZoneId))),
            expr)
      }
    }
  }

  /**
   * Lower only the exact nested Netflix expressions used by Jobs 215/216. Standalone TIMESTAMP
   * NfDateInt remains outside the existing Job175 INT slice; accepting this composition does not
   * broaden either expression independently.
   */
  private def tryTransformFromUnixTimeTzFusion(
      expr: Expression,
      transformChild: Expression => ExpressionTransformer,
      sessionTimeZone: => String): Option[ExpressionTransformer] = {
    val fusion = expr.getClass.getName match {
      case NfDateIntClassName if expr.children.size == 2 =>
        val Seq(timestamp, format) = expr.children
        if (timestamp.getClass.getName == NfFromUnixTimeTzClassName) {
          Some((NfDateIntFromUnixTimeTzFunctionName, timestamp, Some(format)))
        } else {
          None
        }
      case NfHourClassName if expr.children.size == 1 =>
        val timestamp = expr.children.head
        if (timestamp.getClass.getName == NfFromUnixTimeTzClassName) {
          Some((NfHourFromUnixTimeTzFunctionName, timestamp, None))
        } else {
          None
        }
      case _ => None
    }

    fusion.map {
      case (functionName, timestamp, format) =>
        validateResultType(expr, IntegerType, functionName)
        if (timestamp.dataType != TimestampType) {
          unsupported(
            functionName,
            s"expected NfFromUnixTimeTz to produce TIMESTAMP, got " +
              timestamp.dataType.catalogString)
        }
        if (timestamp.children.size != 2) {
          unsupported(
            functionName,
            s"expected NfFromUnixTimeTz(date, timezone), got " +
              s"${timestamp.children.size} children")
        }

        val Seq(epoch, rowTimeZone) = timestamp.children
        if (epoch.dataType != LongType) {
          unsupported(
            functionName,
            s"first fused native slice requires BIGINT epoch input, got " +
              epoch.dataType.catalogString)
        }
        if (rowTimeZone.dataType != StringType) {
          unsupported(
            functionName,
            s"first fused native slice requires STRING timezone expression, got " +
              rowTimeZone.dataType.catalogString)
        }

        format.foreach(validateDefaultFormat(_, functionName))
        val outerTimeZone =
          validatedOuterTimeZone(expr, NfDateIntUtcZoneIds, functionName, sessionTimeZone)
        val nativeChildren =
          Seq(transformChild(epoch), transformChild(rowTimeZone)) ++
            format.map(transformChild) ++
            Seq(transformChild(Literal(outerTimeZone)))
        NetflixDateTimeTransformer(functionName, nativeChildren, expr)
    }
  }

  private def validateResultType(
      expr: Expression,
      expected: DataType,
      functionName: String): Unit = {
    if (expr.dataType != expected) {
      unsupported(
        functionName,
        s"expected ${expected.catalogString} output, got ${expr.dataType.catalogString}")
    }
  }

  private def validateDefaultFormat(format: Expression, functionName: String): Unit = {
    val formatText = format match {
      case Literal(value: UTF8String, StringType) => value.toString
      case Literal(value: String, StringType) => value
      case Literal(null, StringType) =>
        unsupported(functionName, "format must be the non-null plan-time literal '-'")
      case _ =>
        unsupported(functionName, "format must be a plan-time string literal")
    }
    if (formatText != "-") {
      unsupported(
        functionName,
        s"first native slice supports only the default '-' format, got '$formatText'")
    }
  }

  private def validatedOuterTimeZone(
      expr: Expression,
      supportedZoneIds: Set[String],
      functionName: String,
      sessionTimeZone: => String): String = {
    val timeZoneId = expr match {
      case timezoneExpression: TimeZoneAwareExpression =>
        timezoneExpression.timeZoneId.getOrElse(sessionTimeZone)
      case _ =>
        unsupported(
          functionName,
          s"${expr.getClass.getName} must implement TimeZoneAwareExpression")
    }
    if (!supportedZoneIds.contains(timeZoneId)) {
      unsupported(functionName, s"first native slice supports only UTC, got '$timeZoneId'")
    }
    timeZoneId
  }

  private def unsupported(functionName: String, reason: String): Nothing = {
    throw new GlutenNotSupportException(
      s"Netflix expression $functionName is not supported: $reason")
  }
}

private case class NetflixDateTimeTransformer(
    substraitExprName: String,
    children: Seq[ExpressionTransformer],
    original: Expression)
  extends ExpressionTransformer {
  override def doTransform(context: SubstraitContext): ExpressionNode = {
    // Netflix Catalyst expressions carry timezone metadata outside their children. Build the
    // Substrait signature from transformer children so regular and fused native calls include
    // every explicit data or metadata argument.
    val functionName = ConverterUtils.makeFuncName(substraitExprName, children.map(_.dataType))
    val functionId = context.registerFunction(functionName)
    val childNodes = children.map(_.doTransform(context)).asJava
    ExpressionBuilder.makeScalarFunction(
      functionId,
      childNodes,
      ConverterUtils.getTypeNode(dataType, nullable))
  }
}
