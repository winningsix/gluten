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
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, EqualTo, Expression, IsNotNull}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, JoinHint, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf

/**
 * Restrict a per-key aggregate to a selective dimension key set that the parent join already
 * enforces.
 *
 * The motivating shape is TPC-H Q17 after Spark decorrelates its scalar subquery:
 *
 * {{{
 * Join(
 *   Join(fact, filteredDimension, fact.key = dimension.key),
 *   Aggregate(otherFact.key, aggregate(otherFact.measure)),
 *   dimension.key = otherFact.key AND residual)
 * }}}
 *
 * Every aggregate group consumed by the parent join must have a key present in
 * `filteredDimension`. It is therefore equivalent to add an inner join with
 * `Distinct(filteredDimension.key)` below the aggregate. DISTINCT is essential: it prevents
 * duplicate dimension rows from changing the aggregate input multiplicity. The original
 * dimension join is retained, so its output multiplicity is unchanged.
 *
 * This rule deliberately targets only a narrow shape:
 *   - the parent and dimension joins are inner joins with one direct key equality;
 *   - the aggregate groups by one direct attribute and is over a unary fact scan branch;
 *   - the outer and aggregate fact leaves are the same relation and use same-named key columns;
 *   - the dimension branch is a single Project/Filter chain over one different leaf relation;
 *   - the dimension has a deterministic predicate beyond null checks;
 *   - the parent join also consumes a non-key aggregate output in a residual predicate.
 *
 * Gated by `spark.gluten.sql.optimizer.pushFilteredKeysIntoAggregate.enabled` (default true).
 */
case class PushFilteredKeysIntoAggregate(spark: SparkSession)
  extends Rule[LogicalPlan]
  with Logging {

  private case class AggregateBranch(
      plan: LogicalPlan,
      aggregate: Aggregate,
      key: Attribute,
      factLeaf: LogicalPlan)

  private case class DimensionCandidate(
      plan: LogicalPlan,
      key: Attribute,
      factKey: Attribute)

  private val confKey =
    "spark.gluten.sql.optimizer.pushFilteredKeysIntoAggregate.enabled"

  override def apply(plan: LogicalPlan): LogicalPlan = {
    registerPostSubqueryPass()
    if (!SQLConf.get.getConfString(confKey, "true").toBoolean || !plan.resolved) {
      return plan
    }

    plan.transformUp {
      case join @ Join(left, right, Inner, Some(condition), _)
          if condition.deterministic =>
        rewrite(join, left, right, condition).getOrElse(join)
    }
  }

  /**
   * Q17 becomes the matching join/aggregate tree only after Spark's Subquery optimizer batch.
   * Register one idempotent user-optimizer pass so the rule also sees that final shape.
   */
  private def registerPostSubqueryPass(): Unit = {
    val experimental = spark.experimental
    experimental.synchronized {
      if (!experimental.extraOptimizations.exists(_.isInstanceOf[PushFilteredKeysIntoAggregate])) {
        experimental.extraOptimizations = experimental.extraOptimizations :+ this
        logDebug(
          "PushFilteredKeysIntoAggregate: self-registered into " +
            "spark.experimental.extraOptimizations for post-subquery pass")
      }
    }
  }

  private def rewrite(
      join: Join,
      left: LogicalPlan,
      right: LogicalPlan,
      condition: Expression): Option[LogicalPlan] = {
    val aggregateBranch = extractAggregateBranch(right).getOrElse(return None)
    val (outerKey, aggregateKey) =
      extractParentKeys(condition, left, aggregateBranch).getOrElse(return None)

    if (!aggregateKey.semanticEquals(aggregateBranch.key)) {
      return None
    }

    val aggregateNonKeyOutput = aggregateBranch.aggregate.outputSet -
      aggregateBranch.key
    val residuals = splitAnd(condition).filterNot(isKeyEquality(_, outerKey, aggregateKey))
    if (
      aggregateNonKeyOutput.isEmpty ||
      !residuals.exists(_.references.intersect(aggregateNonKeyOutput).nonEmpty)
    ) {
      return None
    }

    val dimensions =
      findDimensionCandidates(left, outerKey, aggregateBranch.key, aggregateBranch.factLeaf)
    if (dimensions.length != 1) {
      return None
    }
    val dimension = dimensions.head

    val distinctKeys = Aggregate(
      Seq(dimension.key),
      Seq(dimension.key),
      Project(Seq(dimension.key), dimension.plan))
    val restrictedInput = Join(
      aggregateBranch.aggregate.child,
      distinctKeys,
      Inner,
      Some(EqualTo(aggregateBranch.key, dimension.key)),
      JoinHint.NONE)
    val rewrittenAggregate = aggregateBranch.aggregate.copy(child = restrictedInput)
    val rewrittenRight =
      replaceAggregate(aggregateBranch.plan, aggregateBranch.aggregate, rewrittenAggregate)

    if (!rewrittenRight.resolved || !rewrittenRight.outputSet.subsetOf(right.outputSet)) {
      return None
    }

    logDebug(
      s"PushFilteredKeysIntoAggregate: restricted aggregate by distinct " +
        s"${dimension.key.name} keys before grouping ${aggregateBranch.key.name}")
    Some(join.copy(right = rewrittenRight))
  }

  private def extractAggregateBranch(plan: LogicalPlan): Option[AggregateBranch] = {
    def loop(current: LogicalPlan): Option[Aggregate] = current match {
      case aggregate: Aggregate => Some(aggregate)
      case Project(_, child) => loop(child)
      case Filter(condition, child) if condition.deterministic => loop(child)
      case _ => None
    }

    val aggregate = loop(plan).getOrElse(return None)
    val key = aggregate.groupingExpressions match {
      case Seq(attribute: Attribute) => attribute
      case _ => return None
    }
    if (
      !aggregate.output.exists(_.semanticEquals(key)) ||
      !aggregate.child.outputSet.contains(key)
    ) {
      return None
    }
    val factLeaf = extractUnaryLeaf(aggregate.child).getOrElse(return None)
    Some(AggregateBranch(plan, aggregate, key, factLeaf))
  }

  private def extractParentKeys(
      condition: Expression,
      left: LogicalPlan,
      aggregateBranch: AggregateBranch): Option[(Attribute, Attribute)] = {
    val candidates = splitAnd(condition).flatMap {
      case EqualTo(leftKey: Attribute, rightKey: Attribute)
          if left.outputSet.contains(leftKey) &&
            aggregateBranch.plan.outputSet.contains(rightKey) =>
        Some(leftKey -> rightKey)
      case EqualTo(rightKey: Attribute, leftKey: Attribute)
          if left.outputSet.contains(leftKey) &&
            aggregateBranch.plan.outputSet.contains(rightKey) =>
        Some(leftKey -> rightKey)
      case _ => None
    }
    candidates match {
      case Seq(candidate) => Some(candidate)
      case _ => None
    }
  }

  private def findDimensionCandidates(
      plan: LogicalPlan,
      outerKey: Attribute,
      aggregateKey: Attribute,
      aggregateFactLeaf: LogicalPlan): Seq[DimensionCandidate] = {
    val candidates = plan.collect {
      case Join(joinLeft, joinRight, Inner, Some(condition), _)
          if condition.deterministic =>
        extractDimensionCandidate(
          joinLeft,
          joinRight,
          condition,
          outerKey,
          aggregateKey,
          aggregateFactLeaf)
          .orElse(
            extractDimensionCandidate(
              joinRight,
              joinLeft,
              condition,
              outerKey,
              aggregateKey,
              aggregateFactLeaf))
    }.flatten
    candidates.distinct
  }

  private def extractDimensionCandidate(
      fact: LogicalPlan,
      dimension: LogicalPlan,
      condition: Expression,
      outerKey: Attribute,
      aggregateKey: Attribute,
      aggregateFactLeaf: LogicalPlan): Option[DimensionCandidate] = {
    if (
      dimension.output.length != 1 ||
      !dimension.output.head.semanticEquals(outerKey) ||
      !isDeterministic(dimension) ||
      !hasSelectiveFilter(dimension)
    ) {
      return None
    }

    val dimensionKey = dimension.output.head
    val factKey = extractDirectEquality(condition, fact, dimension, dimensionKey)
      .getOrElse(return None)
    if (
      factKey.name != aggregateKey.name ||
      factKey.dataType != aggregateKey.dataType ||
      dimensionKey.dataType != aggregateKey.dataType
    ) {
      return None
    }

    val outerFactLeaf = extractUnaryLeaf(fact).getOrElse(return None)
    val dimensionLeaf = extractUnaryLeaf(dimension).getOrElse(return None)
    if (
      !outerFactLeaf.sameResult(aggregateFactLeaf) ||
      outerFactLeaf.sameResult(dimensionLeaf)
    ) {
      return None
    }
    Some(DimensionCandidate(dimension, dimensionKey, factKey))
  }

  private def extractDirectEquality(
      condition: Expression,
      left: LogicalPlan,
      right: LogicalPlan,
      expectedRightKey: Attribute): Option[Attribute] = {
    condition match {
      case EqualTo(leftKey: Attribute, rightKey: Attribute)
          if left.outputSet.contains(leftKey) &&
            right.outputSet.contains(rightKey) &&
            rightKey.semanticEquals(expectedRightKey) =>
        Some(leftKey)
      case EqualTo(rightKey: Attribute, leftKey: Attribute)
          if left.outputSet.contains(leftKey) &&
            right.outputSet.contains(rightKey) &&
            rightKey.semanticEquals(expectedRightKey) =>
        Some(leftKey)
      case _ => None
    }
  }

  private def extractUnaryLeaf(plan: LogicalPlan): Option[LogicalPlan] = plan match {
    case Project(_, child) => extractUnaryLeaf(child)
    case Filter(condition, child) if condition.deterministic => extractUnaryLeaf(child)
    case leaf if leaf.children.isEmpty => Some(leaf)
    case _ => None
  }

  private def replaceAggregate(
      plan: LogicalPlan,
      oldAggregate: Aggregate,
      newAggregate: Aggregate): LogicalPlan = {
    plan.transformUp {
      case aggregate: Aggregate if aggregate.eq(oldAggregate) => newAggregate
    }
  }

  private def hasSelectiveFilter(plan: LogicalPlan): Boolean = {
    plan.exists {
      case Filter(condition, _) =>
        splitAnd(condition).exists {
          case IsNotNull(_) => false
          case predicate => predicate.deterministic
        }
      case _ => false
    }
  }

  private def isDeterministic(plan: LogicalPlan): Boolean = {
    !plan.exists(_.expressions.exists(expression => !expression.deterministic))
  }

  private def isKeyEquality(
      expression: Expression,
      leftKey: Attribute,
      rightKey: Attribute): Boolean = expression match {
    case EqualTo(left: Attribute, right: Attribute) =>
      (left.semanticEquals(leftKey) && right.semanticEquals(rightKey)) ||
      (left.semanticEquals(rightKey) && right.semanticEquals(leftKey))
    case _ => false
  }

  private def splitAnd(expression: Expression): Seq[Expression] = expression match {
    case And(left, right) => splitAnd(left) ++ splitAnd(right)
    case other => Seq(other)
  }
}
