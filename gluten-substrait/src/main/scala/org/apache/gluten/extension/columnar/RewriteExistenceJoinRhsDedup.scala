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
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, EqualTo, Exists, Expression, GreaterThan, IsNotNull, Literal, Not, Or, OuterReference, PredicateHelper}
import org.apache.spark.sql.catalyst.expressions.aggregate.{Count, Max, Min}
import org.apache.spark.sql.catalyst.plans.{ExistenceJoin, LeftAnti, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule

/**
 * Deduplicate the build side of existence joins before the join.
 *
 * Spark decorrelates EXISTS / NOT EXISTS into left-semi, left-anti, or existence joins. These joins
 * only care whether a matching RHS row exists and do not output RHS columns, so duplicate RHS rows
 * with identical condition-relevant values are redundant. Deduplicating them early lets Spark plan
 * a normal two-phase aggregate before the join instead of pushing large duplicate streams through
 * later join and aggregate stages.
 *
 * This rule must run as an optimizer rule, not as an analyzer rule. It uses whole-plan transforms,
 * which Spark 3.5 deliberately rejects while executing post-hoc resolution rules. The injected
 * operator-optimization and pre-CBO passes see predicate subqueries before RewriteSubquery. After
 * the rule observes an Exists, its self-registered user optimizer pass also provides
 * post-RewriteSubquery coverage on subsequent optimizer executions.
 */
case class RewriteExistenceJoinRhsDedup(spark: SparkSession)
  extends Rule[LogicalPlan]
  with PredicateHelper
  with Logging {

  private case class EqualityKey(left: Expression, right: Attribute)
  private case class NotEqualKey(left: Expression, right: Attribute)
  private case class CorrelatedEqualityKey(outer: Expression, right: Attribute)
  private case class CorrelatedNotEqualKey(outer: Expression, right: Attribute)

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!GlutenConfig.get.enableExistenceJoinRhsDedup) {
      return plan
    }

    val rewrittenSubqueries = plan.transformUp {
      case Filter(condition, child) =>
        Filter(rewriteExistsExpression(condition), child)
    }
    if (hasExistsExpression(rewrittenSubqueries)) {
      registerPostSubqueryPass()
    }

    if (!rewrittenSubqueries.resolved) {
      return rewrittenSubqueries
    }

    rewrittenSubqueries.transformUp {
      case join @ Join(_, right, joinType, Some(condition), _)
          if isExistenceJoin(joinType) && condition.deterministic =>
        rewriteJoin(join, right, condition)
          .getOrElse(join)
    }
  }

  private def registerPostSubqueryPass(): Unit = {
    val experimental = spark.experimental
    experimental.synchronized {
      if (!experimental.extraOptimizations.exists(_.isInstanceOf[RewriteExistenceJoinRhsDedup])) {
        experimental.extraOptimizations = experimental.extraOptimizations :+ this
        logDebug(
          "RewriteExistenceJoinRhsDedup: self-registered into " +
            "spark.experimental.extraOptimizations for post-RewriteSubquery pass")
      }
    }
  }

  private def hasExistsExpression(plan: LogicalPlan): Boolean = {
    plan.exists {
      node =>
        node.expressions.exists {
          expression =>
            expression.exists {
              case _: Exists => true
              case _ => false
            }
        }
    }
  }

  private def isExistenceJoin(joinType: org.apache.spark.sql.catalyst.plans.JoinType): Boolean = {
    joinType match {
      case LeftSemi | LeftAnti | _: ExistenceJoin => true
      case _ => false
    }
  }

  private def rewriteJoin(join: Join, right: LogicalPlan, condition: Expression): Option[Join] = {
    val summarized =
      if (summarizeNotEqualExistenceJoinEnabled) {
        summarizeSingleNotEqualJoin(join, right, condition)
      } else {
        None
      }
    summarized.orElse {
      deduplicateJoinRhs(join, right, condition)
    }
  }

  private def rewriteExistsExpression(expression: Expression): Expression = {
    expression.transformUp {
      case exists: Exists =>
        rewriteExists(exists)
    }
  }

  private def rewriteExists(exists: Exists): Exists = {
    rewriteExistsSubquery(exists.plan)
      .orElse(rewriteExistsJoinConditions(exists.plan, exists.joinCond))
      .map(exists.withNewPlan)
      .getOrElse(exists)
  }

  private def rewriteExistsSubquery(subqueryPlan: LogicalPlan): Option[LogicalPlan] = {
    if (!summarizeNotEqualExistenceJoinEnabled) {
      return None
    }
    subqueryPlan match {
      case Project(_, Filter(condition, right)) =>
        summarizeCorrelatedSingleNotEqualSubquery(right, condition)
      case Filter(condition, right) =>
        summarizeCorrelatedSingleNotEqualSubquery(right, condition)
      case _ => None
    }
  }

  private def rewriteExistsJoinConditions(
      subqueryPlan: LogicalPlan,
      joinConditions: Seq[Expression]): Option[LogicalPlan] = {
    if (!summarizeNotEqualExistenceJoinEnabled) {
      return None
    }
    joinConditions.reduceOption(And).flatMap {
      condition => summarizeCorrelatedSingleNotEqualSubquery(subqueryPlan, condition)
    }
  }

  private def summarizeNotEqualExistenceJoinEnabled: Boolean = {
    spark.sessionState.conf
      .getConfString(
        "spark.gluten.sql.optimizer.existenceJoinRhsDedup.summarizeNotEqual.enabled",
        "true")
      .toBoolean
  }

  private def summarizeCorrelatedSingleNotEqualSubquery(
      right: LogicalPlan,
      condition: Expression): Option[LogicalPlan] = {
    val predicates = splitConjunctivePredicates(condition)
    val rightOnlyPredicates = predicates.filter(isRightOnlyPredicate(_, right))
    val mixedPredicates = predicates.filterNot(rightOnlyPredicates.contains)
    val equalityKeys = mixedPredicates.flatMap(extractCorrelatedEqualityKey(_, right))
    val notEqualKeys = mixedPredicates.flatMap(extractCorrelatedNotEqualKey(_, right))

    if (
      equalityKeys.isEmpty || notEqualKeys.length != 1 ||
      equalityKeys.length + notEqualKeys.length != mixedPredicates.length ||
      isExistenceSummary(right)
    ) {
      return None
    }

    val notEqualKey = notEqualKeys.head
    if (!supportsMinMaxSummary(notEqualKey.right)) {
      return None
    }
    val groupingAttributes = right.output.filter {
      attr => equalityKeys.exists(_.right.semanticEquals(attr))
    }
    if (groupingAttributes.length != equalityKeys.length) {
      return None
    }

    val filteredRight =
      filtersBeforeAggregate(rightOnlyPredicates :+ IsNotNull(notEqualKey.right), right)
    if (isLikelySelfCorrelation(equalityKeys, notEqualKey)) {
      val minNotEqualKey =
        Alias(
          Min(notEqualKey.right).toAggregateExpression(),
          s"_existence_min_${notEqualKey.right.name}")()
      val maxNotEqualKey =
        Alias(
          Max(notEqualKey.right).toAggregateExpression(),
          s"_existence_max_${notEqualKey.right.name}")()
      val summarizedRight = Aggregate(
        groupingAttributes,
        groupingAttributes ++ Seq(minNotEqualKey, maxNotEqualKey),
        filteredRight)
      val filteredSummary =
        Filter(
          Not(EqualTo(minNotEqualKey.toAttribute, maxNotEqualKey.toAttribute)),
          summarizedRight)
      val equalityConditions = equalityKeys.map(key => EqualTo(key.right, key.outer): Expression)
      return Some(Filter(equalityConditions.reduce(And), filteredSummary))
    }

    val distinctRight = Aggregate(
      groupingAttributes :+ notEqualKey.right,
      groupingAttributes :+ notEqualKey.right,
      filteredRight)
    val distinctNotEqualKey = distinctRight.output
      .find(_.semanticEquals(notEqualKey.right))
      .getOrElse(return None)
    val minNotEqualKey =
      Alias(
        Min(distinctNotEqualKey).toAggregateExpression(),
        s"_existence_min_${notEqualKey.right.name}")()
    val distinctNotEqualKeyCount =
      Alias(
        Count(Literal(1)).toAggregateExpression(),
        s"_existence_count_${notEqualKey.right.name}")()
    val summarizedRight = Aggregate(
      groupingAttributes,
      groupingAttributes ++ Seq(minNotEqualKey, distinctNotEqualKeyCount),
      distinctRight)

    val equalityConditions = equalityKeys.map(key => EqualTo(key.right, key.outer): Expression)
    val notEqualSummaryCondition =
      And(
        IsNotNull(notEqualKey.outer),
        Or(
          GreaterThan(distinctNotEqualKeyCount.toAttribute, Literal(1L)),
          Not(EqualTo(notEqualKey.outer, minNotEqualKey.toAttribute)))
      )
    Some(Filter((equalityConditions :+ notEqualSummaryCondition).reduce(And), summarizedRight))
  }

  private def summarizeSingleNotEqualJoin(
      join: Join,
      right: LogicalPlan,
      condition: Expression): Option[Join] = {
    val predicates = splitConjunctivePredicates(condition)
    val rightOnlyPredicates = predicates.filter(isRightOnlyPredicate(_, right))
    val leftOnlyPredicates = predicates.filter(isLeftOnlyPredicate(_, join.left))
    val mixedPredicates =
      predicates.filterNot(
        predicate =>
          rightOnlyPredicates.contains(predicate) || leftOnlyPredicates.contains(predicate))
    val equalityKeys = mixedPredicates.flatMap(extractEqualityKey(_, join.left, right))
    val notEqualKeys = mixedPredicates.flatMap(extractNotEqualKey(_, join.left, right))

    if (
      equalityKeys.isEmpty || notEqualKeys.length != 1 ||
      equalityKeys.length + notEqualKeys.length != mixedPredicates.length
    ) {
      return None
    }

    val notEqualKey = notEqualKeys.head
    if (!supportsMinMaxSummary(notEqualKey.right)) {
      return None
    }
    val groupingAttributes = right.output.filter {
      attr => equalityKeys.exists(_.right.semanticEquals(attr))
    }
    if (groupingAttributes.length != equalityKeys.length) {
      return None
    }

    val filteredRight =
      filtersBeforeAggregate(rightOnlyPredicates :+ IsNotNull(notEqualKey.right), right)
    if (isLikelySelfJoin(equalityKeys, notEqualKey)) {
      val minNotEqualKey =
        Alias(
          Min(notEqualKey.right).toAggregateExpression(),
          s"_existence_min_${notEqualKey.right.name}")()
      val maxNotEqualKey =
        Alias(
          Max(notEqualKey.right).toAggregateExpression(),
          s"_existence_max_${notEqualKey.right.name}")()
      val summarizedRight = Aggregate(
        groupingAttributes,
        groupingAttributes ++ Seq(minNotEqualKey, maxNotEqualKey),
        filteredRight)
      val filteredSummary =
        Filter(
          Not(EqualTo(minNotEqualKey.toAttribute, maxNotEqualKey.toAttribute)),
          summarizedRight)
      val equalityConditions = equalityKeys.map(key => EqualTo(key.left, key.right): Expression)
      val newCondition = (leftOnlyPredicates ++ equalityConditions).reduce(And)
      return Some(join.copy(right = filteredSummary, condition = Some(newCondition)))
    }

    val distinctRight = Aggregate(
      groupingAttributes :+ notEqualKey.right,
      groupingAttributes :+ notEqualKey.right,
      filteredRight)
    val distinctNotEqualKey = distinctRight.output
      .find(_.semanticEquals(notEqualKey.right))
      .getOrElse(return None)
    val minNotEqualKey =
      Alias(
        Min(distinctNotEqualKey).toAggregateExpression(),
        s"_existence_min_${notEqualKey.right.name}")()
    val distinctNotEqualKeyCount =
      Alias(
        Count(Literal(1)).toAggregateExpression(),
        s"_existence_count_${notEqualKey.right.name}")()
    val summarizedRight = Aggregate(
      groupingAttributes,
      groupingAttributes ++ Seq(minNotEqualKey, distinctNotEqualKeyCount),
      distinctRight)

    val equalityConditions = equalityKeys.map(key => EqualTo(key.left, key.right): Expression)
    val notEqualSummaryCondition =
      And(
        IsNotNull(notEqualKey.left),
        Or(
          GreaterThan(distinctNotEqualKeyCount.toAttribute, Literal(1L)),
          Not(EqualTo(notEqualKey.left, minNotEqualKey.toAttribute)))
      )
    val newCondition =
      (leftOnlyPredicates ++ equalityConditions :+ notEqualSummaryCondition).reduce(And)
    Some(join.copy(right = summarizedRight, condition = Some(newCondition)))
  }

  private def deduplicateJoinRhs(
      join: Join,
      right: LogicalPlan,
      condition: Expression): Option[Join] = {
    val predicates = splitConjunctivePredicates(condition)
    val rightOnlyPredicates = predicates.filter(isRightOnlyPredicate(_, right))
    val joinPredicates = predicates.filterNot(rightOnlyPredicates.contains)
    val newCondition = joinPredicates.reduceOption(And)

    if (newCondition.isEmpty) {
      return None
    }

    val dedupAttributes = dedupKeyAttributes(newCondition.get, right)
    if (
      dedupAttributes.isEmpty || isAlreadyDeduplicated(right, dedupAttributes) ||
      isExistenceSummary(right)
    ) {
      return None
    }

    val filteredRight = filtersBeforeAggregate(rightOnlyPredicates, right)
    val dedupRight = Aggregate(dedupAttributes, dedupAttributes, filteredRight)
    Some(join.copy(right = dedupRight, condition = newCondition))
  }

  private def filtersBeforeAggregate(
      predicates: Seq[Expression],
      child: LogicalPlan): LogicalPlan = {
    predicates.reduceOption(And) match {
      case Some(filterCondition) => Filter(filterCondition, child)
      case None => child
    }
  }

  private def supportsMinMaxSummary(attribute: Attribute): Boolean = {
    Min(attribute).checkInputDataTypes().isSuccess &&
    Max(attribute).checkInputDataTypes().isSuccess &&
    Count(attribute).checkInputDataTypes().isSuccess
  }

  private def isLikelySelfCorrelation(
      equalityKeys: Seq[CorrelatedEqualityKey],
      notEqualKey: CorrelatedNotEqualKey): Boolean = {
    equalityKeys.forall(key => hasSameAttributeName(key.outer, key.right)) &&
    hasSameAttributeName(notEqualKey.outer, notEqualKey.right)
  }

  private def isLikelySelfJoin(
      equalityKeys: Seq[EqualityKey],
      notEqualKey: NotEqualKey): Boolean = {
    equalityKeys.forall(key => hasSameAttributeName(key.left, key.right)) &&
    hasSameAttributeName(notEqualKey.left, notEqualKey.right)
  }

  private def hasSameAttributeName(expression: Expression, attribute: Attribute): Boolean = {
    expressionAttributeName(expression).contains(attribute.name)
  }

  private def expressionAttributeName(expression: Expression): Option[String] = {
    expression match {
      case attr: Attribute => Some(attr.name)
      case OuterReference(attr: Attribute) => Some(attr.name)
      case _ => None
    }
  }

  private def isRightOnlyPredicate(predicate: Expression, right: LogicalPlan): Boolean = {
    predicate.references.nonEmpty &&
    predicate.references.forall(isOutputAttribute(_, right)) &&
    !containsOuterReference(predicate)
  }

  private def isLeftOnlyPredicate(predicate: Expression, left: LogicalPlan): Boolean = {
    predicate.references.nonEmpty &&
    predicate.references.forall(isOutputAttribute(_, left))
  }

  private def dedupKeyAttributes(condition: Expression, right: LogicalPlan): Seq[Attribute] = {
    val referenced = condition.references.intersect(right.outputSet)
    right.output.filter(referenced.contains)
  }

  private def extractEqualityKey(
      predicate: Expression,
      left: LogicalPlan,
      right: LogicalPlan): Option[EqualityKey] = {
    predicate match {
      case EqualTo(l, r) =>
        extractMixedAttributePair(l, r, left, right).map {
          case (leftExpression, rightAttribute) =>
            EqualityKey(leftExpression, rightAttribute)
        }
      case _ => None
    }
  }

  private def extractNotEqualKey(
      predicate: Expression,
      left: LogicalPlan,
      right: LogicalPlan): Option[NotEqualKey] = {
    predicate match {
      case Not(EqualTo(l, r)) =>
        extractMixedAttributePair(l, r, left, right).map {
          case (leftExpression, rightAttribute) =>
            NotEqualKey(leftExpression, rightAttribute)
        }
      case _ => None
    }
  }

  private def extractMixedAttributePair(
      first: Expression,
      second: Expression,
      left: LogicalPlan,
      right: LogicalPlan): Option[(Expression, Attribute)] = {
    (asLeftExpression(first, left), asRightAttribute(second, right)) match {
      case (Some(leftExpression), Some(rightAttribute)) => Some(leftExpression -> rightAttribute)
      case _ =>
        (asLeftExpression(second, left), asRightAttribute(first, right)) match {
          case (Some(leftExpression), Some(rightAttribute)) =>
            Some(leftExpression -> rightAttribute)
          case _ => None
        }
    }
  }

  private def extractCorrelatedEqualityKey(
      predicate: Expression,
      right: LogicalPlan): Option[CorrelatedEqualityKey] = {
    predicate match {
      case EqualTo(l, r) =>
        extractCorrelatedAttributePair(l, r, right).map {
          case (outerExpression, rightAttribute) =>
            CorrelatedEqualityKey(outerExpression, rightAttribute)
        }
      case _ => None
    }
  }

  private def extractCorrelatedNotEqualKey(
      predicate: Expression,
      right: LogicalPlan): Option[CorrelatedNotEqualKey] = {
    predicate match {
      case Not(EqualTo(l, r)) =>
        extractCorrelatedAttributePair(l, r, right).map {
          case (outerExpression, rightAttribute) =>
            CorrelatedNotEqualKey(outerExpression, rightAttribute)
        }
      case _ => None
    }
  }

  private def extractCorrelatedAttributePair(
      first: Expression,
      second: Expression,
      right: LogicalPlan): Option[(Expression, Attribute)] = {
    (asOuterExpression(first, right), asRightAttribute(second, right)) match {
      case (Some(outerExpression), Some(rightAttribute)) => Some(outerExpression -> rightAttribute)
      case _ =>
        (asOuterExpression(second, right), asRightAttribute(first, right)) match {
          case (Some(outerExpression), Some(rightAttribute)) =>
            Some(outerExpression -> rightAttribute)
          case _ => None
        }
    }
  }

  private def asOuterExpression(expression: Expression, right: LogicalPlan): Option[Expression] = {
    if (
      containsOuterReference(expression) &&
      !expression.references.exists(isOutputAttribute(_, right))
    ) {
      Some(expression)
    } else {
      None
    }
  }

  private def asLeftExpression(expression: Expression, left: LogicalPlan): Option[Expression] = {
    if (expression.references.nonEmpty && expression.references.subsetOf(left.outputSet)) {
      Some(expression)
    } else {
      None
    }
  }

  private def asRightAttribute(expression: Expression, right: LogicalPlan): Option[Attribute] = {
    expression match {
      case attr: Attribute if isOutputAttribute(attr, right) => Some(attr)
      case _ => None
    }
  }

  private def isOutputAttribute(attribute: Attribute, plan: LogicalPlan): Boolean = {
    plan.output.exists(_.semanticEquals(attribute))
  }

  private def containsOuterReference(expression: Expression): Boolean = {
    expression.exists {
      case _: OuterReference => true
      case _ => false
    }
  }

  private def isAlreadyDeduplicated(
      right: LogicalPlan,
      dedupAttributes: Seq[Attribute]): Boolean = {
    right match {
      case aggregate: Aggregate
          if aggregate.groupingExpressions.length == dedupAttributes.length &&
            aggregate.aggregateExpressions.length == dedupAttributes.length =>
        sameExpressions(aggregate.groupingExpressions, dedupAttributes) &&
        sameExpressions(aggregate.aggregateExpressions, dedupAttributes)
      case _ => false
    }
  }

  private def isExistenceSummary(right: LogicalPlan): Boolean = {
    right match {
      case aggregate: Aggregate =>
        aggregate.aggregateExpressions.exists(_.name.startsWith("_existence_min_")) &&
        (aggregate.aggregateExpressions.exists(_.name.startsWith("_existence_max_")) ||
          aggregate.aggregateExpressions.exists(_.name.startsWith("_existence_count_")))
      case Project(_, child) => isExistenceSummary(child)
      case Filter(_, child) => isExistenceSummary(child)
      case _ => false
    }
  }

  private def sameExpressions(left: Seq[Expression], right: Seq[Expression]): Boolean = {
    left.length == right.length && left.zip(right).forall { case (l, r) => l.semanticEquals(r) }
  }
}
