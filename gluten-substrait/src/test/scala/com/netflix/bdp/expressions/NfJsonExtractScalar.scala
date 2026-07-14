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

import org.apache.spark.sql.catalyst.expressions.{BinaryExpression, Expression}
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.types.{DataType, StringType}

/** Test-only exact-FQCN stand-in; production Gluten remains independent of the Netflix jar. */
case class NfJsonExtractScalar(json: Expression, jsonPath: Expression)
  extends BinaryExpression
  with CodegenFallback {
  override def left: Expression = json
  override def right: Expression = jsonPath
  override def dataType: DataType = StringType
  override def nullable: Boolean = true
  override def nullSafeEval(jsonValue: Any, pathValue: Any): Any = null
  override protected def withNewChildrenInternal(
      newLeft: Expression,
      newRight: Expression): Expression = copy(json = newLeft, jsonPath = newRight)
}
