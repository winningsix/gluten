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
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, EqualNullSafe, EqualTo, Expression, If, IsNotNull, Literal, NamedExpression, Not}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Max, Min}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftAnti}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, JoinHint, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf

/**
 * Merge paired existence summaries over the same relation into one conditional aggregate.
 *
 * `RewriteExistenceJoinRhsDedup` converts a common `EXISTS` / `NOT EXISTS` pair into this shape:
 *
 * {{{
 * LeftAnti(
 *   Inner(main, allRowsWithMultipleValues, main.key = all.key),
 *   filteredRowsWithMultipleValues,
 *   main.key = filtered.key)
 * }}}
 *
 * Both build sides group the same relation by the same key and compute min/max of the same value.
 * The first summary keeps groups with multiple values; the anti side removes groups whose filtered
 * rows contain multiple values. A single aggregate can compute both pairs:
 *
 * {{{
 * Inner(
 *   main,
 *   Filter(
 *     allMin != allMax AND filteredMin = filteredMax,
 *     Aggregate(key, min/max(value), min/max(if(filteredPredicate, value, null)), relation)),
 *   main.key = summary.key)
 * }}}
 *
 * The motivating query is TPC-H Q21. Its `lineitem` existence branches become one scan, one
 * aggregate shuffle, and one join instead of two of each. The rewrite is deliberately narrow:
 * both summaries must have exactly the min/max form emitted by the existence rule, the unfiltered
 * branch may contain only key/value null checks, the filtered branch must add a deterministic
 * predicate, and both source leaves and key/value columns must match.
 *
 * Gated by `spark.gluten.sql.optimizer.mergeExistenceJoinSummaries.enabled` (default true).
 */
case class MergeExistenceJoinSummaries(spark: SparkSession)
  extends Rule[LogicalPlan]
  with Logging {

  private case class SourceBranch(
      leaf: LogicalPlan,
      key: Attribute,
      value: Attribute,
      predicates: Seq[Expression])

  private case class MinMaxSummary(
      outputKey: Attribute,
      source: SourceBranch)

  private case class MergedSummaryJoin(
      main: LogicalPlan,
      summary: LogicalPlan,
      condition: Expression,
      hint: JoinHint)

  private val confKey =
    "spark.gluten.sql.optimizer.mergeExistenceJoinSummaries.enabled"

  override def apply(plan: LogicalPlan): LogicalPlan = {
    registerPostSubqueryPass()
    if (!SQLConf.get.getConfString(confKey, "true").toBoolean || !plan.resolved) {
      return plan
    }

    plan.transformUp {
      case anti @ Join(
            inner @ Join(main, allSummaryPlan, Inner, Some(allCondition), innerHint),
            filteredSummaryPlan,
            LeftAnti,
            Some(filteredCondition),
            antiHint)
          if innerHint == JoinHint.NONE && antiHint == JoinHint.NONE =>
        rewrite(
          anti,
          inner,
          main,
          allSummaryPlan,
          filteredSummaryPlan,
          allCondition,
          filteredCondition).getOrElse(anti)
      case join @ Join(_, _, Inner, Some(_), _) =>
        moveMergedSummaryAfterSelectiveJoin(join).getOrElse(join)
    }
  }

  private def registerPostSubqueryPass(): Unit = {
    val experimental = spark.experimental
    experimental.synchronized {
      if (!experimental.extraOptimizations.exists(_.isInstanceOf[MergeExistenceJoinSummaries])) {
        experimental.extraOptimizations = experimental.extraOptimizations :+ this
        logDebug(
          "MergeExistenceJoinSummaries: self-registered into " +
            "spark.experimental.extraOptimizations for post-subquery pass")
      }
    }
  }

  private def rewrite(
      anti: Join,
      inner: Join,
      main: LogicalPlan,
      allSummaryPlan: LogicalPlan,
      filteredSummaryPlan: LogicalPlan,
      allCondition: Expression,
      filteredCondition: Expression): Option[LogicalPlan] = {
    val allSummary = extractMinMaxSummary(allSummaryPlan).getOrElse(return None)
    val filteredSummary =
      extractMinMaxSummary(filteredSummaryPlan).getOrElse(return None)
    val (mainAllKey, allJoinKey) =
      extractSingleEquality(allCondition, main, allSummaryPlan).getOrElse(return None)
    val (mainFilteredKey, filteredJoinKey) =
      extractSingleEquality(filteredCondition, inner, filteredSummaryPlan).getOrElse(return None)

    if (
      !mainAllKey.semanticEquals(mainFilteredKey) ||
      !allJoinKey.semanticEquals(allSummary.outputKey) ||
      !filteredJoinKey.semanticEquals(filteredSummary.outputKey) ||
      !sameSourceColumns(allSummary.source, filteredSummary.source)
    ) {
      return None
    }

    val allPredicates = removeKeyValueNullChecks(allSummary.source)
    val filteredPredicates = removeKeyValueNullChecks(filteredSummary.source)
    if (allPredicates.nonEmpty || filteredPredicates.isEmpty) {
      return None
    }
    val mappedFilteredPredicate =
      mapToSource(
        filteredPredicates.reduce(And),
        filteredSummary.source.leaf,
        allSummary.source.leaf)
        .getOrElse(return None)
    if (!mappedFilteredPredicate.deterministic) {
      return None
    }

    val source = allSummary.source
    val allMin = Alias(
      Min(source.value).toAggregateExpression(),
      s"_merged_existence_all_min_${source.value.name}")()
    val allMax = Alias(
      Max(source.value).toAggregateExpression(),
      s"_merged_existence_all_max_${source.value.name}")()
    val filteredValue = If(
      mappedFilteredPredicate,
      source.value,
      Literal.create(null, source.value.dataType))
    val filteredMin = Alias(
      Min(filteredValue).toAggregateExpression(),
      s"_merged_existence_filtered_min_${source.value.name}")()
    val filteredMax = Alias(
      Max(filteredValue).toAggregateExpression(),
      s"_merged_existence_filtered_max_${source.value.name}")()
    val requiredSourceAttributes = source.leaf.output.filter {
      attribute =>
        attribute.semanticEquals(source.key) ||
        attribute.semanticEquals(source.value) ||
        mappedFilteredPredicate.references.contains(attribute)
    }
    val projectedSource = Project(requiredSourceAttributes, source.leaf)
    val aggregateInput = Filter(
      And(IsNotNull(source.key), IsNotNull(source.value)),
      projectedSource)
    val mergedAggregate = Aggregate(
      Seq(source.key),
      Seq(source.key, allMin, allMax, filteredMin, filteredMax),
      aggregateInput)
    val qualifyingSummary = Filter(
      And(
        Not(EqualTo(allMin.toAttribute, allMax.toAttribute)),
        EqualNullSafe(filteredMin.toAttribute, filteredMax.toAttribute)),
      mergedAggregate)
    val mergedJoin = inner.copy(
      left = main,
      right = qualifyingSummary,
      condition = Some(EqualTo(mainAllKey, source.key)))

    if (!anti.outputSet.subsetOf(mergedJoin.outputSet)) {
      return None
    }

    logDebug(
      s"MergeExistenceJoinSummaries: merged paired min/max summaries by " +
        s"${source.key.name}; removed one ${source.leaf.nodeName} scan, aggregate, and join")
    Some(Project(anti.output, mergedJoin))
  }

  /**
   * The summary merge happens after Spark's CBO batch because the input shape is created while
   * rewriting subqueries. Move the now-single summary past later selective joins when that is
   * semantically independent. This restores the order CBO chooses for an equivalent SQL CTE:
   * filter the main branch through dimensions/orders first, then join the per-key summary.
   */
  private def moveMergedSummaryAfterSelectiveJoin(outer: Join): Option[LogicalPlan] = {
    val MergedSummaryJoin(main, summary, summaryCondition, summaryHint) =
      extractMergedSummaryJoin(outer.left).getOrElse(return None)
    val outerCondition = outer.condition.getOrElse(return None)
    val mainAndOtherOutput = main.outputSet ++ outer.right.outputSet

    if (
      outerCondition.references.exists(summary.outputSet.contains) ||
      !outerCondition.references.subsetOf(mainAndOtherOutput) ||
      !summaryCondition.references.subsetOf(main.outputSet ++ summary.outputSet)
    ) {
      return None
    }

    val filteredMain = outer.copy(left = main)
    val relocated = Join(
      filteredMain,
      summary,
      Inner,
      Some(summaryCondition),
      summaryHint)
    if (!outer.outputSet.subsetOf(relocated.outputSet)) {
      return None
    }

    logDebug(
      "MergeExistenceJoinSummaries: moved merged summary after an independent selective join")
    Some(Project(outer.output, relocated))
  }

  private def extractMergedSummaryJoin(plan: LogicalPlan): Option[MergedSummaryJoin] = {
    def stripAttributeProjects(current: LogicalPlan): LogicalPlan = current match {
      case Project(projectList, child) if projectList.forall(_.isInstanceOf[Attribute]) =>
        stripAttributeProjects(child)
      case other => other
    }
    val candidate = stripAttributeProjects(plan)
    candidate match {
      case Join(main, summary, Inner, Some(condition), hint)
          if hint == JoinHint.NONE && isMergedSummary(summary) =>
        Some(MergedSummaryJoin(main, summary, condition, hint))
      case _ => None
    }
  }

  private def isMergedSummary(plan: LogicalPlan): Boolean = plan match {
    case Filter(_, aggregate: Aggregate) =>
      aggregate.aggregateExpressions.count {
        _.name.startsWith("_merged_existence_")
      } == 4
    case _ => false
  }

  private def extractMinMaxSummary(plan: LogicalPlan): Option[MinMaxSummary] = {
    val (outputKey, summaryCore) = stripKeyOnlyWrappers(plan).getOrElse(return None)
    val aggregate = summaryCore match {
      case Filter(Not(EqualTo(left: Attribute, right: Attribute)), candidate: Aggregate)
          if candidate.outputSet.contains(left) && candidate.outputSet.contains(right) =>
        candidate
      case _ => return None
    }
    val groupingKey = aggregate.groupingExpressions match {
      case Seq(key: Attribute) => key
      case _ => return None
    }
    val minValues = aggregate.aggregateExpressions.flatMap(extractMinValue)
    val maxValues = aggregate.aggregateExpressions.flatMap(extractMaxValue)
    val value = (minValues, maxValues) match {
      case (Seq(minValue), Seq(maxValue)) if minValue.semanticEquals(maxValue) => minValue
      case _ => return None
    }
    val source = extractSourceBranch(aggregate.child, groupingKey, value).getOrElse(return None)
    Some(MinMaxSummary(outputKey, source))
  }

  private def stripKeyOnlyWrappers(plan: LogicalPlan): Option[(Attribute, LogicalPlan)] = {
    plan match {
      case Project(Seq(key: Attribute), child) =>
        stripKeyOnlyWrappers(child).map {
          case (_, core) => key -> core
        }
      case Aggregate(Seq(groupingKey: Attribute), Seq(outputKey: Attribute), child, _)
          if groupingKey.semanticEquals(outputKey) =>
        stripKeyOnlyWrappers(child).map {
          case (_, core) => outputKey -> core
        }
      case Filter(Not(EqualTo(_: Attribute, _: Attribute)), _: Aggregate) =>
        val key = plan.output.headOption.collect {
          case attribute: Attribute => attribute
        }.getOrElse(return None)
        Some(key -> plan)
      case _ => None
    }
  }

  private def extractMinValue(expression: NamedExpression): Option[Attribute] = expression match {
    case Alias(AggregateExpression(Min(value: Attribute), _, false, None, _), _) => Some(value)
    case _ => None
  }

  private def extractMaxValue(expression: NamedExpression): Option[Attribute] = expression match {
    case Alias(AggregateExpression(Max(value: Attribute), _, false, None, _), _) => Some(value)
    case _ => None
  }

  private def extractSourceBranch(
      plan: LogicalPlan,
      key: Attribute,
      value: Attribute): Option[SourceBranch] = {
    def loop(
        current: LogicalPlan,
        predicates: Vector[Expression]): Option[(LogicalPlan, Vector[Expression])] = {
      current match {
        case Project(projectList, child)
            if projectList.forall(_.isInstanceOf[Attribute]) =>
          loop(child, predicates)
        case Filter(condition, child) if condition.deterministic =>
          loop(child, predicates ++ splitAnd(condition))
        case leaf if leaf.children.isEmpty =>
          Some(leaf -> predicates)
        case _ => None
      }
    }

    val (leaf, predicates) = loop(plan, Vector.empty).getOrElse(return None)
    val leafKey = findMatchingLeafAttribute(key, leaf).getOrElse(return None)
    val leafValue = findMatchingLeafAttribute(value, leaf).getOrElse(return None)
    Some(SourceBranch(leaf, leafKey, leafValue, predicates))
  }

  private def extractSingleEquality(
      condition: Expression,
      left: LogicalPlan,
      right: LogicalPlan): Option[(Attribute, Attribute)] = {
    splitAnd(condition) match {
      case Seq(EqualTo(leftKey: Attribute, rightKey: Attribute))
          if left.outputSet.contains(leftKey) && right.outputSet.contains(rightKey) =>
        Some(leftKey -> rightKey)
      case Seq(EqualTo(rightKey: Attribute, leftKey: Attribute))
          if left.outputSet.contains(leftKey) && right.outputSet.contains(rightKey) =>
        Some(leftKey -> rightKey)
      case _ => None
    }
  }

  private def sameSourceColumns(left: SourceBranch, right: SourceBranch): Boolean = {
    left.leaf.sameResult(right.leaf) &&
    left.key.name == right.key.name &&
    left.key.dataType == right.key.dataType &&
    left.value.name == right.value.name &&
    left.value.dataType == right.value.dataType
  }

  private def removeKeyValueNullChecks(source: SourceBranch): Seq[Expression] = {
    source.predicates.filterNot {
      case IsNotNull(attribute: Attribute) =>
        attribute.semanticEquals(source.key) || attribute.semanticEquals(source.value)
      case _ => false
    }
  }

  private def mapToSource(
      expression: Expression,
      from: LogicalPlan,
      to: LogicalPlan): Option[Expression] = {
    val mapping = expression.references.toSeq.map {
      attribute =>
        val sourceAttribute =
          findMatchingLeafAttribute(attribute, from).getOrElse(return None)
        val targetAttribute =
          findMatchingLeafAttribute(sourceAttribute, to).getOrElse(return None)
        attribute.exprId -> targetAttribute
    }.toMap
    Some(
      expression.transformUp {
        case attribute: Attribute if mapping.contains(attribute.exprId) =>
          mapping(attribute.exprId)
      })
  }

  private def findMatchingLeafAttribute(
      attribute: Attribute,
      leaf: LogicalPlan): Option[Attribute] = {
    leaf.output.find(_.semanticEquals(attribute)).orElse {
      leaf.output.filter {
        candidate =>
          candidate.name == attribute.name && candidate.dataType == attribute.dataType
      } match {
        case Seq(candidate) => Some(candidate)
        case _ => None
      }
    }
  }

  private def splitAnd(expression: Expression): Seq[Expression] = expression match {
    case And(left, right) => splitAnd(left) ++ splitAnd(right)
    case other => Seq(other)
  }
}
