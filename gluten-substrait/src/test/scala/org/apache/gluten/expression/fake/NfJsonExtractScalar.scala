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
package org.apache.gluten.expression.fake

import org.apache.spark.sql.catalyst.expressions.{BinaryExpression, Expression}
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.types.{DataType, StringType}

/** Same simple name as Netflix's class; used to prove that the gate requires the exact FQCN. */
case class NfJsonExtractScalar(json: Expression, path: Expression)
  extends BinaryExpression
  with CodegenFallback {
  override def left: Expression = json
  override def right: Expression = path
  override def dataType: DataType = StringType
  override def nullable: Boolean = true
  override def nullSafeEval(leftValue: Any, rightValue: Any): Any = null
  override protected def withNewChildrenInternal(
      newLeft: Expression,
      newRight: Expression): Expression = copy(json = newLeft, path = newRight)
}
