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
package com.netflix.bdp.expressions

import org.apache.spark.sql.catalyst.expressions.{BinaryExpression, Expression, TimeZoneAwareExpression, UnaryExpression}
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.types.{DataType, IntegerType, LongType, TimestampType}

/** Test-only stand-in that keeps Gluten's production module independent of the Netflix jar. */
case class NfToUnixTime(date: Expression, format: Expression, timeZoneId: Option[String] = None)
  extends BinaryExpression
  with TimeZoneAwareExpression
  with CodegenFallback {
  override def left: Expression = date
  override def right: Expression = format
  override def dataType: DataType = LongType
  override def nullable: Boolean = true
  override def nullSafeEval(dateValue: Any, formatValue: Any): Any = null
  override def withTimeZone(timeZoneId: String): TimeZoneAwareExpression =
    copy(timeZoneId = Some(timeZoneId))
  override protected def withNewChildrenInternal(
      newLeft: Expression,
      newRight: Expression): Expression = copy(date = newLeft, format = newRight)
}

case class NfToUnixTimeMs(date: Expression, format: Expression, timeZoneId: Option[String] = None)
  extends BinaryExpression
  with TimeZoneAwareExpression
  with CodegenFallback {
  override def left: Expression = date
  override def right: Expression = format
  override def dataType: DataType = LongType
  override def nullable: Boolean = true
  override def nullSafeEval(dateValue: Any, formatValue: Any): Any = null
  override def withTimeZone(timeZoneId: String): TimeZoneAwareExpression =
    copy(timeZoneId = Some(timeZoneId))
  override protected def withNewChildrenInternal(
      newLeft: Expression,
      newRight: Expression): Expression = copy(date = newLeft, format = newRight)
}

case class NfDateInt(date: Expression, format: Expression, timeZoneId: Option[String] = None)
  extends BinaryExpression
  with TimeZoneAwareExpression
  with CodegenFallback {
  override def left: Expression = date
  override def right: Expression = format
  override def dataType: DataType = IntegerType
  override def nullable: Boolean = true
  override def nullSafeEval(dateValue: Any, formatValue: Any): Any = null
  override def withTimeZone(timeZoneId: String): TimeZoneAwareExpression =
    copy(timeZoneId = Some(timeZoneId))
  override protected def withNewChildrenInternal(
      newLeft: Expression,
      newRight: Expression): Expression = copy(date = newLeft, format = newRight)
}

case class NfFromUnixTimeTz(date: Expression, timezone: Expression)
  extends BinaryExpression
  with CodegenFallback {
  override def left: Expression = date
  override def right: Expression = timezone
  override def dataType: DataType = TimestampType
  override def nullable: Boolean = true
  override def nullSafeEval(dateValue: Any, timezoneValue: Any): Any = null
  override protected def withNewChildrenInternal(
      newLeft: Expression,
      newRight: Expression): Expression = copy(date = newLeft, timezone = newRight)
}

case class NfHour(date: Expression, timeZoneId: Option[String] = None)
  extends UnaryExpression
  with TimeZoneAwareExpression
  with CodegenFallback {
  override def child: Expression = date
  override def dataType: DataType = IntegerType
  override def nullable: Boolean = true
  override def nullSafeEval(dateValue: Any): Any = null
  override def withTimeZone(timeZoneId: String): TimeZoneAwareExpression =
    copy(timeZoneId = Some(timeZoneId))
  override protected def withNewChildInternal(newChild: Expression): Expression =
    copy(date = newChild)
}
