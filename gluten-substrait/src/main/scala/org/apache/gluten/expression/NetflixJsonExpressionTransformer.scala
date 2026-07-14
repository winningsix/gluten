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

import org.apache.spark.sql.catalyst.expressions.{Expression, Literal}
import org.apache.spark.sql.types.StringType
import org.apache.spark.unsafe.types.UTF8String

/**
 * Exact, bounded Catalyst-side domain for Netflix's scalar JSON extraction expression.
 *
 * This object deliberately has no compile-time dependency on the Netflix UDF jar. It is not wired
 * into [[ExpressionConverter]] until both the Velox and cuDF implementations pass the Netflix JAR
 * oracle. Keeping the converter disconnected is important: emitting a Substrait function whose GPU
 * implementation is only approximately compatible would turn a safe planning fallback into a silent
 * data-corruption risk.
 */
private[expression] object NetflixJsonExpressionTransformer {
  private val NfJsonExtractScalarClassName =
    "com.netflix.bdp.expressions.NfJsonExtractScalar"
  private val NativeFunctionName = "nf_json_extract_scalar"
  private val MaxPathDepth = 16

  def tryTransformFirstSlice(
      expr: Expression,
      transformChild: Expression => ExpressionTransformer): Option[ExpressionTransformer] = {
    if (expr.getClass.getName != NfJsonExtractScalarClassName) {
      return None
    }

    validateFirstSlice(expr)
    Some(GenericExpressionTransformer(NativeFunctionName, expr.children.map(transformChild), expr))
  }

  def isExactFirstSliceDomain(expr: Expression): Boolean = {
    if (expr.getClass.getName != NfJsonExtractScalarClassName) {
      return false
    }
    try {
      validateFirstSlice(expr)
      true
    } catch {
      case _: GlutenNotSupportException => false
    }
  }

  private def validateFirstSlice(expr: Expression): Unit = {
    if (expr.children.size != 2) {
      unsupported(s"expected exactly two Catalyst children, got ${expr.children.size}")
    }
    if (expr.dataType != StringType) {
      unsupported(s"expected STRING output, got ${expr.dataType.catalogString}")
    }

    val Seq(json, path) = expr.children
    if (json.dataType != StringType) {
      unsupported(s"expected a STRING JSON child, got ${json.dataType.catalogString}")
    }

    val pathText = path match {
      case Literal(value: UTF8String, StringType) if value != null => value.toString
      case Literal(value: String, StringType) if value != null => value
      case Literal(null, StringType) =>
        unsupported("path must be a non-null plan-time STRING literal")
      case literal: Literal =>
        unsupported(s"path must be a STRING literal, got ${literal.dataType.catalogString}")
      case _ => unsupported("path must be a non-null plan-time STRING literal")
    }
    validateSimplePath(pathText)
  }

  /**
   * Accept only `$.field(.field|[nonnegative-index])*` with at most sixteen selections.
   *
   * Fields use the deliberately narrow ASCII identifier domain. Quoted fields and every JSONPath
   * operator beyond a direct child or array index stay on Spark's original expression path.
   */
  private[expression] def validateSimplePath(path: String): Unit = {
    var offset = 0
    var depth = 0

    if (path.length < 3 || path.charAt(0) != '$' || path.charAt(1) != '.') {
      unsupported("path must start with '$.' and select at least one field")
    }
    offset = 2

    def parseField(): Unit = {
      if (offset >= path.length || !isFieldStart(path.charAt(offset))) {
        unsupported(s"unsupported field selection in path '$path'")
      }
      offset += 1
      while (offset < path.length && isFieldPart(path.charAt(offset))) {
        offset += 1
      }
      depth += 1
      checkDepth(path, depth)
    }

    parseField()
    while (offset < path.length) {
      path.charAt(offset) match {
        case '.' =>
          offset += 1
          parseField()
        case '[' =>
          offset += 1
          val indexStart = offset
          while (offset < path.length && isAsciiDigit(path.charAt(offset))) {
            offset += 1
          }
          if (indexStart == offset || offset >= path.length || path.charAt(offset) != ']') {
            unsupported(s"unsupported array selection in path '$path'")
          }
          val indexText = path.substring(indexStart, offset)
          if (
            (indexText.length > 1 && indexText.charAt(0) == '0') ||
            indexText.length > 10 ||
            (indexText.length == 10 && indexText > Int.MaxValue.toString)
          ) {
            unsupported(s"array index is outside the bounded nonnegative INT domain in '$path'")
          }
          offset += 1
          depth += 1
          checkDepth(path, depth)
        case _ => unsupported(s"unsupported JSONPath syntax in '$path'")
      }
    }
  }

  private def checkDepth(path: String, depth: Int): Unit = {
    if (depth > MaxPathDepth) {
      unsupported(s"path depth exceeds $MaxPathDepth in '$path'")
    }
  }

  private def isFieldStart(char: Char): Boolean =
    (char >= 'A' && char <= 'Z') || (char >= 'a' && char <= 'z') || char == '_'

  private def isFieldPart(char: Char): Boolean = isFieldStart(char) || isAsciiDigit(char)

  private def isAsciiDigit(char: Char): Boolean = char >= '0' && char <= '9'

  private def unsupported(reason: String): Nothing = {
    throw new GlutenNotSupportException(
      s"Netflix expression $NativeFunctionName is not supported: $reason")
  }
}
