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
package org.apache.gluten.extension.columnar

import org.apache.gluten.config.GlutenConfig

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{AttributeSet, EqualNullSafe, EqualTo, Expression, In, InSet, PredicateHelper}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{Filter, Join, JoinHint, LogicalPlan, Project, SubqueryAlias}
import org.apache.spark.sql.catalyst.rules.Rule

/**
 * Conservatively rotates a Q11-like join chain so a selective dimension joins its dimension parent
 * before the fact table is probed. It changes `(fact JOIN dimension) JOIN filtered_dimension` into
 * `fact JOIN (dimension JOIN filtered_dimension)`.
 *
 * Presto-GPU Q11 materializes nation->supplier before probing partsupp, so the first hash join sees
 * ~32M rows instead of ~800M. Spark's default join order keeps partsupp on the probe side of an
 * unfiltered supplier join first.
 *
 * The rule is intentionally narrow: it only handles inner joins without hints, keeps conditions on
 * the same attribute sets, and requires the filtered dimension to carry an obvious selective
 * literal predicate plus small table statistics.
 */
case class SelectiveDimensionJoinReorder(spark: SparkSession)
  extends Rule[LogicalPlan]
  with PredicateHelper
  with Logging {

  private val maxFilteredDimensionSizeInBytes = BigInt(64L * 1024L * 1024L)
  private val maxFilteredDimensionRows = BigInt(100000)
  private val maxSelectiveInListValues = 16
  private val maxFilteredDimensionWrapperDepth = 4

  override def apply(plan: LogicalPlan): LogicalPlan = {
    val glutenConfig = new GlutenConfig(spark.sessionState.conf)
    if (!glutenConfig.enableSelectiveDimensionJoinReorder || !plan.resolved) {
      return plan
    }

    plan.transformUp {
      case join: Join =>
        reorder(join).getOrElse(join)
    }
  }

  private def reorder(join: Join): Option[LogicalPlan] = {
    join match {
      case Join(
            Join(fact, dimension, Inner, Some(factDimensionCondition), leftHint),
            filteredDimension,
            Inner,
            Some(dimensionFilterCondition),
            topHint)
          if leftHint == JoinHint.NONE &&
            topHint == JoinHint.NONE &&
            factDimensionCondition.deterministic &&
            dimensionFilterCondition.deterministic &&
            isSelectiveFilteredDimension(filteredDimension) &&
            conditionStaysBetween(factDimensionCondition, fact.outputSet, dimension.outputSet) &&
            conditionStaysBetween(
              dimensionFilterCondition,
              dimension.outputSet,
              filteredDimension.outputSet) =>
        val dimensionJoin =
          Join(dimension, filteredDimension, Inner, Some(dimensionFilterCondition), JoinHint.NONE)
        logInfo("Reordered selective dimension join before fact-table probe.")
        Some(Join(fact, dimensionJoin, Inner, Some(factDimensionCondition), JoinHint.NONE))
      case _ =>
        None
    }
  }

  private def isSelectiveFilteredDimension(plan: LogicalPlan): Boolean = {
    isSelectiveFilteredDimension(plan, 0)
  }

  private def isSelectiveFilteredDimension(plan: LogicalPlan, wrapperDepth: Int): Boolean = {
    plan match {
      case Filter(condition, child) if condition.deterministic =>
        hasSelectiveLiteralPredicate(condition, child.outputSet) && hasSmallDimensionStats(child)
      case Project(projectList, child)
          if wrapperDepth < maxFilteredDimensionWrapperDepth &&
            projectList.forall(_.deterministic) =>
        isSelectiveFilteredDimension(child, wrapperDepth + 1)
      case SubqueryAlias(_, child) if wrapperDepth < maxFilteredDimensionWrapperDepth =>
        isSelectiveFilteredDimension(child, wrapperDepth + 1)
      case _ =>
        false
    }
  }

  private def hasSmallDimensionStats(plan: LogicalPlan): Boolean = {
    val stats = plan.stats
    stats.rowCount.exists(_ <= maxFilteredDimensionRows) ||
    stats.sizeInBytes <= maxFilteredDimensionSizeInBytes
  }

  private def conditionStaysBetween(
      condition: Expression,
      left: AttributeSet,
      right: AttributeSet): Boolean = {
    val predicates = splitConjunctivePredicates(condition)
    predicates.nonEmpty &&
    predicates.forall(predicate => referencesOnly(predicate, left, right)) &&
    predicates.exists(predicate => referencesBoth(predicate, left, right))
  }

  private def hasSelectiveLiteralPredicate(
      condition: Expression,
      dimensionAttributes: AttributeSet): Boolean = {
    splitConjunctivePredicates(condition).exists {
      case EqualTo(left, right) =>
        literalComparison(left, right, dimensionAttributes)
      case EqualNullSafe(left, right) =>
        literalComparison(left, right, dimensionAttributes)
      case In(value, list) =>
        list.nonEmpty &&
        list.length <= maxSelectiveInListValues &&
        referencesOnly(value, dimensionAttributes) &&
        list.forall(_.foldable)
      case InSet(value, values) =>
        values.nonEmpty &&
        values.size <= maxSelectiveInListValues &&
        referencesOnly(value, dimensionAttributes)
      case _ =>
        false
    }
  }

  private def literalComparison(
      left: Expression,
      right: Expression,
      dimensionAttributes: AttributeSet): Boolean = {
    (referencesOnly(left, dimensionAttributes) && right.foldable) ||
    (referencesOnly(right, dimensionAttributes) && left.foldable)
  }

  private def referencesOnly(expression: Expression, allowed: AttributeSet): Boolean = {
    expression.references.nonEmpty && expression.references.forall(allowed.contains)
  }

  private def referencesOnly(
      expression: Expression,
      left: AttributeSet,
      right: AttributeSet): Boolean = {
    expression.references.forall(attribute => left.contains(attribute) || right.contains(attribute))
  }

  private def referencesBoth(
      expression: Expression,
      left: AttributeSet,
      right: AttributeSet): Boolean = {
    expression.references.exists(left.contains) && expression.references.exists(right.contains)
  }
}
