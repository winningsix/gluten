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
package org.apache.gluten.extension

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, Expression, IsNotNull, NamedExpression, PredicateHelper, RowFrame, ScalarSubquery, SpecifiedWindowFrame, UnboundedFollowing, UnboundedPreceding, WindowExpression, WindowSpecDefinition}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Max, Sum}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, LogicalPlan, Project, Window}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf

/**
 * Reuse a grouped sum when an uncorrelated scalar subquery recomputes the same groups only to take
 * their maximum.
 *
 * Spark inlines a CTE at every reference. A query such as TPC-H Q15 consequently has this shape:
 *
 * {{{
 * Filter(groupedSum.value = ScalarSubquery(Aggregate(max, duplicateGroupedSum)), groupedSum)
 * }}}
 *
 * Recomputing a floating-point grouped sum twice is not merely expensive. Parallel GPU reduction
 * order is not deterministic, so the two copies can differ in their low bits and an exact equality
 * can intermittently return no rows. Derive the maximum from the already-computed grouped result:
 *
 * {{{
 * Filter(value = maxValue,
 *   Window(max(value) over unbounded rows,
 *     groupedSum))
 * }}}
 *
 * This keeps exact SQL equality, preserves every tied maximum, and removes one fact scan and one
 * grouped aggregate. The rewrite is deliberately narrow:
 *   - the outer child groups by one direct attribute and outputs that key plus one unfiltered,
 *     non-distinct sum;
 *   - the scalar subquery is an uncorrelated global max over another one-key grouped sum;
 *   - after removing only an optimizer-inferred isnotnull(groupKey), both grouped sums have the
 *     same canonical result.
 *
 * Gated by `spark.gluten.sql.optimizer.reuseGroupedAggregateForScalarMax.enabled` (default true).
 */
case class ReuseGroupedAggregateForScalarMax(spark: SparkSession)
  extends Rule[LogicalPlan]
  with Logging
  with PredicateHelper {

  private case class MainGroupedSum(
      aggregate: Aggregate,
      key: Attribute,
      keyAlias: Alias,
      valueAlias: Alias,
      sum: AggregateExpression)

  private case class ScalarGroupedMax(
      subquery: ScalarSubquery,
      groupedAggregate: Aggregate,
      key: Attribute,
      valueAlias: Alias,
      sum: AggregateExpression)

  private val confKey =
    "spark.gluten.sql.optimizer.reuseGroupedAggregateForScalarMax.enabled"

  override def apply(plan: LogicalPlan): LogicalPlan = {
    registerPostSubqueryPass()
    if (!SQLConf.get.getConfString(confKey, "true").toBoolean || !plan.resolved) {
      return plan
    }

    plan.transformUp {
      case filter @ Filter(condition, aggregate: Aggregate) =>
        rewrite(filter, condition, aggregate).getOrElse(filter)
    }
  }

  private def registerPostSubqueryPass(): Unit = {
    val experimental = spark.experimental
    experimental.synchronized {
      if (
        !experimental.extraOptimizations.exists(
          _.isInstanceOf[ReuseGroupedAggregateForScalarMax])
      ) {
        experimental.extraOptimizations = experimental.extraOptimizations :+ this
        logDebug(
          "ReuseGroupedAggregateForScalarMax: self-registered into " +
            "spark.experimental.extraOptimizations for post-subquery pass")
      }
    }
  }

  private def rewrite(
      filter: Filter,
      condition: Expression,
      aggregate: Aggregate): Option[LogicalPlan] = {
    val main = extractMainGroupedSum(aggregate).getOrElse(return None)
    val scalarCandidates = condition.collect {
      case subquery: ScalarSubquery if !subquery.isCorrelated => subquery
    }.distinct
    val matching = scalarCandidates.flatMap {
      subquery =>
        extractScalarGroupedMax(subquery).filter(scalar => sameGroupedSum(main, scalar))
    }
    if (matching.size != 1) {
      return None
    }
    val scalar = matching.head

    val sharedKeyAlias = copyAlias(main.keyAlias, scalar.key)
    val sharedValueAlias = copyAlias(main.valueAlias, scalar.sum)
    val sharedAggregate = scalar.groupedAggregate.copy(
      aggregateExpressions = Seq(sharedKeyAlias, sharedValueAlias))

    val windowSpec = WindowSpecDefinition(
      Nil,
      Nil,
      SpecifiedWindowFrame(RowFrame, UnboundedPreceding, UnboundedFollowing))
    val maxValue = Alias(
      WindowExpression(
        Max(sharedValueAlias.toAttribute).toAggregateExpression(),
        windowSpec),
      s"_reused_grouped_max_${sharedValueAlias.name}")()
    val window = Window(Seq(maxValue), Nil, Nil, sharedAggregate)
    val rewrittenCondition = condition.transformDown {
      case current: ScalarSubquery if current.eq(scalar.subquery) =>
        maxValue.toAttribute
    }
    val rewrittenFilter = filter.copy(condition = rewrittenCondition, child = window)

    logDebug(
      s"ReuseGroupedAggregateForScalarMax: reused grouped sum by ${main.key.name}; " +
        s"removed one ${main.aggregate.child.nodeName} branch and made max equality deterministic")
    Some(Project(main.aggregate.output, rewrittenFilter))
  }

  private def extractMainGroupedSum(aggregate: Aggregate): Option[MainGroupedSum] = {
    if (aggregate.hint.nonEmpty) {
      return None
    }
    val key = aggregate.groupingExpressions match {
      case Seq(attribute: Attribute) => attribute
      case _ => return None
    }
    val keyAliases = aggregate.aggregateExpressions.collect {
      case alias @ Alias(attribute: Attribute, _)
          if attribute.semanticEquals(key) =>
        alias
    }
    val sums = aggregate.aggregateExpressions.flatMap(extractSum)
    (keyAliases, sums, aggregate.aggregateExpressions.size) match {
      case (Seq(keyAlias), Seq((valueAlias, sum)), 2) =>
        Some(MainGroupedSum(aggregate, key, keyAlias, valueAlias, sum))
      case _ => None
    }
  }

  private def extractScalarGroupedMax(
      subquery: ScalarSubquery): Option[ScalarGroupedMax] = {
    val outer = subquery.plan match {
      case aggregate: Aggregate
          if aggregate.hint.isEmpty && aggregate.groupingExpressions.isEmpty =>
        aggregate
      case _ => return None
    }
    val (maxInput, grouped) = outer.aggregateExpressions match {
      case Seq(Alias(AggregateExpression(max: Max, _, false, None, _), _))
          if max.child.isInstanceOf[Attribute] =>
        outer.child match {
          case aggregate: Aggregate if aggregate.hint.isEmpty =>
            max.child.asInstanceOf[Attribute] -> aggregate
          case _ => return None
        }
      case _ => return None
    }
    val key = grouped.groupingExpressions match {
      case Seq(attribute: Attribute) => attribute
      case _ => return None
    }
    val sums = grouped.aggregateExpressions.flatMap(extractSum)
    (sums, grouped.aggregateExpressions.size) match {
      case (Seq((valueAlias, sum)), 1)
          if maxInput.semanticEquals(valueAlias.toAttribute) =>
        Some(ScalarGroupedMax(subquery, grouped, key, valueAlias, sum))
      case _ => None
    }
  }

  private def extractSum(expression: NamedExpression): Option[(Alias, AggregateExpression)] = {
    expression match {
      case alias @ Alias(
            aggregate @ AggregateExpression(_: Sum, _, false, None, _),
            _) =>
        Some(alias -> aggregate)
      case _ => None
    }
  }

  private def sameGroupedSum(
      main: MainGroupedSum,
      scalar: ScalarGroupedMax): Boolean = {
    if (
      main.valueAlias.dataType != scalar.valueAlias.dataType ||
      main.key.dataType != scalar.key.dataType
    ) {
      return false
    }

    val normalizedMain = main.aggregate.copy(
      child = removeGroupingKeyNullCheck(main.aggregate.child, main.key))
    val comparableScalar = scalar.groupedAggregate.copy(
      aggregateExpressions = Seq(
        copyAlias(main.keyAlias, scalar.key),
        copyAlias(main.valueAlias, scalar.sum)))
    normalizedMain.sameResult(comparableScalar)
  }

  private def removeGroupingKeyNullCheck(
      plan: LogicalPlan,
      key: Attribute): LogicalPlan = {
    plan.transformUp {
      case Filter(condition, child) =>
        val remaining = splitConjunctivePredicates(condition).filterNot {
          case IsNotNull(attribute: Attribute) if attribute.semanticEquals(key) => true
          case _ => false
        }
        remaining.reduceOption(And).map(Filter(_, child)).getOrElse(child)
    }
  }

  private def copyAlias(alias: Alias, child: Expression): Alias = {
    alias.copy(child = child)(
      alias.exprId,
      alias.qualifier,
      alias.explicitMetadata,
      alias.nonInheritableMetadataKeys)
  }
}
