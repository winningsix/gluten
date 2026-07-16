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

import org.apache.spark.sql.catalyst.expressions.Expression

import java.util.ServiceLoader

import scala.collection.JavaConverters._

/**
 * Service-provider interface for expression mappings owned by an external library.
 *
 * Providers are discovered from `META-INF/services` with the thread context class loader. This
 * keeps application-specific planning rules and native implementations outside the Gluten bundle.
 */
trait ExpressionTransformerProvider {
  def tryTransform(
      expression: Expression,
      transformChild: Expression => ExpressionTransformer): Option[ExpressionTransformer]

  /**
   * Opt in only expressions that the provider can safely materialize before an Arrow scalar UDF.
   */
  def isSafeArrowPreProjection(expression: Expression): Boolean = false
}

object ExpressionTransformerProvider {
  private lazy val providers: Seq[ExpressionTransformerProvider] = {
    val contextLoader = Option(Thread.currentThread().getContextClassLoader)
      .getOrElse(getClass.getClassLoader)
    ServiceLoader
      .load(classOf[ExpressionTransformerProvider], contextLoader)
      .iterator()
      .asScala
      .toSeq
  }

  def tryTransform(
      expression: Expression,
      transformChild: Expression => ExpressionTransformer): Option[ExpressionTransformer] =
    providers.iterator
      .map(_.tryTransform(expression, transformChild))
      .collectFirst { case Some(transformer) => transformer }

  def isSafeArrowPreProjection(expression: Expression): Boolean =
    providers.exists(_.isSafeArrowPreProjection(expression))
}
