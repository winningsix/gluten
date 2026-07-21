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
package org.apache.spark.sql.hive

import org.apache.gluten.exception.GlutenNotSupportException
import org.apache.gluten.expression.{ExpressionConverter, ExpressionTransformer, GenericExpressionTransformer, UDFMappings}

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.util.Utils

import java.util.Locale

object HiveUDFTransformer {
  private val HIVE_SIMPLE_UDF_CLASS: Option[Class[_]] =
    optionalClass("org.apache.spark.sql.hive.HiveSimpleUDF")
  private val HIVE_GENERIC_UDF_CLASS: Option[Class[_]] =
    optionalClass("org.apache.spark.sql.hive.HiveGenericUDF")

  private def optionalClass(className: String): Option[Class[_]] = {
    try {
      Some(Utils.classForName(className))
    } catch {
      case _: ClassNotFoundException | _: NoClassDefFoundError => None
    }
  }

  def isHiveUDF(expr: Expression): Boolean = {
    HIVE_SIMPLE_UDF_CLASS.exists(_.isAssignableFrom(expr.getClass)) ||
    HIVE_GENERIC_UDF_CLASS.exists(_.isAssignableFrom(expr.getClass))
  }

  def getHiveUDFNameAndClassName(expr: Expression): (String, String) = {
    if (!isHiveUDF(expr)) {
      throw new GlutenNotSupportException(
        s"Expression $expr is not a HiveSimpleUDF or HiveGenericUDF")
    }
    val funcWrapper = invokeNoArg[AnyRef](expr, "funcWrapper")
    (
      invokeNoArg[String](expr, "name").stripPrefix("default."),
      invokeNoArg[String](funcWrapper, "functionClassName"))
  }

  private def invokeNoArg[T](target: AnyRef, methodName: String): T = {
    val method = target.getClass.getMethods
      .find(method => method.getName == methodName && method.getParameterCount == 0)
      .getOrElse {
        val declaredMethod = target.getClass.getDeclaredMethod(methodName)
        declaredMethod.setAccessible(true)
        declaredMethod
      }
    method.invoke(target).asInstanceOf[T]
  }

  def replaceWithExpressionTransformer(
      expr: Expression,
      attributeSeq: Seq[Attribute]): ExpressionTransformer = {
    val (udfName, _) = getHiveUDFNameAndClassName(expr)
    genTransformerFromUDFMappings(udfName, expr, attributeSeq)
  }

  def genTransformerFromUDFMappings(
      udfName: String,
      expr: Expression,
      attributeSeq: Seq[Attribute]): GenericExpressionTransformer = {
    UDFMappings.hiveUDFMap.get(udfName.toLowerCase(Locale.ROOT)) match {
      case Some(name) =>
        GenericExpressionTransformer(
          name,
          ExpressionConverter.replaceWithExpressionTransformer(expr.children, attributeSeq),
          expr)
      case _ =>
        throw new GlutenNotSupportException(
          s"Not supported hive udf:$expr"
            + s" name:$udfName hiveUDFMap:${UDFMappings.hiveUDFMap}")
    }
  }
}
