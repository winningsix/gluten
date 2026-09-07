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
package org.apache.spark.sql.execution.adaptive

/** Compatibility helpers for the upstream and EMR Spark 4.0 AQE case-class layouts. */
object AdaptiveSparkPlanCompat {
  def isSubquery(plan: AdaptiveSparkPlanExec): Boolean = {
    noArgMethod(plan, "isSubquery") match {
      case Some(method) => method.invoke(plan).asInstanceOf[Boolean]
      case None => !requiredNoArgMethod(plan, "isMainQuery").invoke(plan).asInstanceOf[Boolean]
    }
  }

  def withSupportsColumnar(
      plan: AdaptiveSparkPlanExec,
      supportsColumnar: Boolean): AdaptiveSparkPlanExec = {
    val copyMethods = plan.getClass.getMethods.filter(_.getName == "copy")
    copyMethods.find(_.getParameterCount == 5) match {
      case Some(copy) =>
        copy
          .invoke(
            plan,
            invoke(plan, "inputPlan"),
            invoke(plan, "context"),
            invoke(plan, "preprocessingRules"),
            Boolean.box(isSubquery(plan)),
            Boolean.box(supportsColumnar))
          .asInstanceOf[AdaptiveSparkPlanExec]
      case None =>
        val copy = copyMethods
          .find(_.getParameterCount == 7)
          .getOrElse(unsupported(plan, "expected a five- or seven-field copy method"))
        copy
          .invoke(
            plan,
            invoke(plan, "inputPlan"),
            invoke(plan, "context"),
            invoke(plan, "onReOptimizeRuleProvider"),
            invoke(plan, "sharedPostStageCreationRules"),
            Boolean.box(!isSubquery(plan)),
            invoke(plan, "requiredDistribution"),
            Boolean.box(supportsColumnar)
          )
          .asInstanceOf[AdaptiveSparkPlanExec]
    }
  }

  private def invoke(plan: AdaptiveSparkPlanExec, name: String): AnyRef =
    requiredNoArgMethod(plan, name).invoke(plan)

  private def noArgMethod(plan: AdaptiveSparkPlanExec, name: String) =
    plan.getClass.getMethods.find(method => method.getName == name && method.getParameterCount == 0)

  private def requiredNoArgMethod(plan: AdaptiveSparkPlanExec, name: String) =
    noArgMethod(plan, name).getOrElse(unsupported(plan, s"missing $name accessor"))

  private def unsupported(plan: AdaptiveSparkPlanExec, detail: String): Nothing =
    throw new IllegalStateException(
      s"Unsupported AdaptiveSparkPlanExec layout ${plan.getClass.getName}: $detail")
}
