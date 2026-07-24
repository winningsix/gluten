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
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, AttributeSet, EqualTo, Expression, IsNotNull, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, AggregateFunction, Sum}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, JoinHint, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf

/**
 * Reuse a filtered per-key aggregate when a parent aggregate rejoins the same unfiltered relation
 * only to recompute that aggregate.
 *
 * The motivating shape is TPC-H Q18:
 *
 * {{{
 * Aggregate(groupKeys, sum(fact.measure))
 *   Join(leftWithQualifiedFactKeys, fact, left.key = fact.key)
 *
 * leftWithQualifiedFactKeys contains:
 *   Join(left, Distinct(Filter(having, Aggregate(fact.key, sum(fact.measure), fact))))
 * }}}
 *
 * The inner aggregate is already unique by key and already has the exact sum needed by the parent.
 * Carrying that sum through the left branch lets us remove the second fact scan, its join, and its
 * shuffle. The parent aggregate remains in place, so duplicate rows introduced elsewhere in the
 * left branch retain their original multiplicity.
 *
 * This rule deliberately recognizes only a narrow, provably equivalent shape:
 *   - both joins are unhinted inner joins on one direct attribute equality;
 *   - both fact branches resolve to the same leaf relation;
 *   - projections use the same direct key and measure columns;
 *   - fact-side filters are absent except for `isnotnull(key)`;
 *   - both aggregates are non-distinct, unfiltered `sum(measure)`;
 *   - the qualifying side is exactly `Distinct(key)` over a filter on that aggregate.
 *
 * Gated by `spark.gluten.sql.optimizer.reuseFilteredAggregate.enabled` (default true).
 */
case class RewriteFilteredAggregateRejoin(spark: SparkSession)
  extends Rule[LogicalPlan]
  with Logging {

  private case class ReusableAggregate(
      key: Attribute,
      sumAttribute: Attribute,
      measure: Attribute,
      filteredAggregate: LogicalPlan,
      factLeaf: LogicalPlan)

  private val confKey = "spark.gluten.sql.optimizer.reuseFilteredAggregate.enabled"

  override def apply(plan: LogicalPlan): LogicalPlan = {
    registerPostSubqueryPass()
    if (!SQLConf.get.getConfString(confKey, "true").toBoolean || !plan.resolved) {
      return plan
    }

    plan.transformUp {
      case aggregate @ Aggregate(
            groupingExpressions,
            aggregateExpressions,
            Project(
              projectList,
              Join(left, right, Inner, Some(condition), hint)),
            _)
          if !hasUserHint(hint) =>
        rewrite(
          aggregate,
          groupingExpressions,
          aggregateExpressions,
          projectList,
          left,
          right,
          condition).getOrElse(aggregate)
    }
  }

  /**
   * Spark's ordinary optimizer rules run before its Subquery batch, while Q18 becomes the matching
   * pair of inner joins only after RewritePredicateSubquery. Register an idempotent user optimizer
   * pass so the rule also sees that final logical shape.
   */
  private def registerPostSubqueryPass(): Unit = {
    val experimental = spark.experimental
    experimental.synchronized {
      if (!experimental.extraOptimizations.exists(_.isInstanceOf[RewriteFilteredAggregateRejoin])) {
        experimental.extraOptimizations = experimental.extraOptimizations :+ this
        logDebug(
          "RewriteFilteredAggregateRejoin: self-registered into " +
            "spark.experimental.extraOptimizations for post-subquery pass")
      }
    }
  }

  private def rewrite(
      aggregate: Aggregate,
      groupingExpressions: Seq[Expression],
      aggregateExpressions: Seq[NamedExpression],
      projectList: Seq[NamedExpression],
      left: LogicalPlan,
      right: LogicalPlan,
      condition: Expression): Option[LogicalPlan] = {
    val (leftKey, rightKey) =
      extractSingleEquiJoinKey(condition, left, right).getOrElse(return None)
    val (outerSum, rightMeasure) =
      extractSingleOuterSum(aggregateExpressions, right).getOrElse(return None)

    if (
      rightKey.semanticEquals(rightMeasure) ||
      references(groupingExpressions).intersect(right.outputSet).nonEmpty ||
      references(aggregateExpressions).intersect(right.outputSet) !=
        AttributeSet(Seq(rightMeasure)) ||
      references(projectList).intersect(right.outputSet) != AttributeSet(Seq(rightMeasure)) ||
      projectList.count {
        case attribute: Attribute => attribute.semanticEquals(rightMeasure)
        case _ => false
      } != 1
    ) {
      return None
    }

    val rightFactLeaf = extractFactLeaf(right, rightKey, rightMeasure).getOrElse(return None)
    val (rewrittenLeft, carriedSum) =
      carryReusableAggregate(left, leftKey, rightKey, rightMeasure, rightFactLeaf)
        .getOrElse(return None)

    val rewrittenAggregateExpressions = aggregateExpressions.map {
      expression =>
        expression
          .transformDown {
            case current: AggregateExpression if current.eq(outerSum) =>
              val rewrittenFunction = current.aggregateFunction
                .withNewChildren(Seq(carriedSum))
                .asInstanceOf[AggregateFunction]
              current.copy(aggregateFunction = rewrittenFunction)
          }
          .asInstanceOf[NamedExpression]
    }
    val rewrittenProjectList = projectList.map {
      case attribute: Attribute if attribute.semanticEquals(rightMeasure) => carriedSum
      case other => other
    }
    val rewrittenProject = Project(rewrittenProjectList, rewrittenLeft)

    val available = rewrittenProject.outputSet
    if (
      !references(groupingExpressions).subsetOf(available) ||
      !references(rewrittenAggregateExpressions).subsetOf(available)
    ) {
      return None
    }

    logDebug(
      s"RewriteFilteredAggregateRejoin: reused sum(${rightMeasure.name}) by ${rightKey.name}; " +
        s"removed one ${rightFactLeaf.nodeName} scan and inner join")
    Some(
      aggregate.copy(
        aggregateExpressions = rewrittenAggregateExpressions,
        child = rewrittenProject))
  }

  private def extractSingleOuterSum(
      aggregateExpressions: Seq[NamedExpression],
      right: LogicalPlan): Option[(AggregateExpression, Attribute)] = {
    val candidates = aggregateExpressions.flatMap {
      expression =>
        expression.collect {
          case aggregateExpression @ AggregateExpression(
                sum: Sum,
                _,
                false,
                None,
                _)
              if sum.child.isInstanceOf[Attribute] &&
                right.outputSet.contains(sum.child.asInstanceOf[Attribute]) =>
            aggregateExpression -> sum.child.asInstanceOf[Attribute]
        }
    }
    candidates match {
      case Seq(candidate) => Some(candidate)
      case _ => None
    }
  }

  private def carryReusableAggregate(
      plan: LogicalPlan,
      outerKey: Attribute,
      rightKey: Attribute,
      rightMeasure: Attribute,
      rightFactLeaf: LogicalPlan): Option[(LogicalPlan, Attribute)] = {
    plan match {
      case join @ Join(joinLeft, candidate, Inner, Some(condition), hint)
          if !hasUserHint(hint) =>
        extractReusableAggregate(candidate).flatMap {
          reusable =>
            if (
              matchesCandidateJoin(condition, joinLeft, candidate, outerKey, reusable.key) &&
              sameFactColumns(reusable, rightKey, rightMeasure) &&
              reusable.factLeaf.sameResult(rightFactLeaf)
            ) {
              Some(
                join.copy(right = reusable.filteredAggregate) ->
                  reusable.sumAttribute)
            } else {
              None
            }
        }.orElse {
          val rewrittenLeft =
            carryReusableAggregate(joinLeft, outerKey, rightKey, rightMeasure, rightFactLeaf)
          val rewrittenRight =
            carryReusableAggregate(candidate, outerKey, rightKey, rightMeasure, rightFactLeaf)
          (rewrittenLeft, rewrittenRight) match {
            case (Some((newLeft, carried)), None) =>
              Some(join.copy(left = newLeft) -> carried)
            case (None, Some((newRight, carried))) =>
              Some(join.copy(right = newRight) -> carried)
            // Refuse an ambiguous tree with more than one reusable candidate.
            case _ => None
          }
        }

      case project @ Project(projectList, child) =>
        carryReusableAggregate(child, outerKey, rightKey, rightMeasure, rightFactLeaf).map {
          case (newChild, carried) =>
            val rewrittenList =
              if (projectList.exists(_.toAttribute.semanticEquals(carried))) {
                projectList
              } else {
                projectList :+ carried
              }
            project.copy(projectList = rewrittenList, child = newChild) -> carried
        }

      case filter @ Filter(_, child) =>
        carryReusableAggregate(child, outerKey, rightKey, rightMeasure, rightFactLeaf).map {
          case (newChild, carried) => filter.copy(child = newChild) -> carried
        }

      case _ => None
    }
  }

  private def extractReusableAggregate(plan: LogicalPlan): Option[ReusableAggregate] = {
    plan match {
      case Aggregate(
            Seq(distinctKey: Attribute),
            Seq(distinctOutput: Attribute),
            Project(
              Seq(projectKey: Attribute),
              Filter(havingCondition, innerAggregate: Aggregate)),
            _)
          if distinctOutput.semanticEquals(distinctKey) &&
            projectKey.semanticEquals(distinctKey) &&
            havingCondition.deterministic =>
        val groupingExpressions = innerAggregate.groupingExpressions
        val aggregateExpressions = innerAggregate.aggregateExpressions
        if (groupingExpressions.length != 1 || aggregateExpressions.length != 2) {
          return None
        }
        val innerKey = groupingExpressions.head match {
          case attribute: Attribute => attribute
          case _ => return None
        }
        if (
          !innerKey.semanticEquals(distinctKey) ||
          !aggregateExpressions.exists {
            case attribute: Attribute => attribute.semanticEquals(innerKey)
            case _ => false
          }
        ) {
          return None
        }

        val sums = aggregateExpressions.collect {
          case named: NamedExpression
              if named.collect {
                case aggregateExpression @ AggregateExpression(
                      sum: Sum,
                      _,
                      false,
                      None,
                      _)
                    if sum.child.isInstanceOf[Attribute] =>
                  aggregateExpression -> sum.child.asInstanceOf[Attribute]
              }.size == 1 =>
            val aggregateExpression = named.collect {
              case expression: AggregateExpression => expression
            }.head
            val sum = aggregateExpression.aggregateFunction.asInstanceOf[Sum]
            (named.toAttribute, sum.child.asInstanceOf[Attribute])
        }
        if (sums.length != 1) {
          return None
        }
        val (sumAttribute, measure) = sums.head
        if (
          !havingCondition.references.contains(sumAttribute) ||
          !havingCondition.references.subsetOf(innerAggregate.outputSet)
        ) {
          return None
        }
        val factLeaf =
          extractFactLeaf(innerAggregate.child, innerKey, measure).getOrElse(return None)
        Some(
          ReusableAggregate(
            innerKey,
            sumAttribute,
            measure,
            Filter(havingCondition, innerAggregate),
            factLeaf))
      case _ => None
    }
  }

  private def extractFactLeaf(
      plan: LogicalPlan,
      key: Attribute,
      measure: Attribute): Option[LogicalPlan] = {
    plan match {
      case Project(projectList, child)
          if projectList.length == 2 &&
            projectList.forall(_.isInstanceOf[Attribute]) &&
            plan.outputSet == AttributeSet(Seq(key, measure)) =>
        extractFactLeaf(child, key, measure)
      case Filter(condition, child) if onlyKeyNullChecks(condition, key) =>
        extractFactLeaf(child, key, measure)
      case leaf
          if leaf.children.isEmpty &&
            leaf.outputSet.contains(key) &&
            leaf.outputSet.contains(measure) =>
        Some(leaf)
      case _ => None
    }
  }

  private def onlyKeyNullChecks(condition: Expression, key: Attribute): Boolean = {
    splitAnd(condition).forall {
      case IsNotNull(attribute: Attribute) => attribute.semanticEquals(key)
      case _ => false
    }
  }

  private def sameFactColumns(
      reusable: ReusableAggregate,
      rightKey: Attribute,
      rightMeasure: Attribute): Boolean = {
    reusable.key.name == rightKey.name &&
    reusable.key.dataType == rightKey.dataType &&
    reusable.measure.name == rightMeasure.name &&
    reusable.measure.dataType == rightMeasure.dataType
  }

  private def extractSingleEquiJoinKey(
      condition: Expression,
      left: LogicalPlan,
      right: LogicalPlan): Option[(Attribute, Attribute)] = {
    condition match {
      case EqualTo(leftAttribute: Attribute, rightAttribute: Attribute)
          if left.outputSet.contains(leftAttribute) && right.outputSet.contains(rightAttribute) =>
        Some(leftAttribute -> rightAttribute)
      case EqualTo(rightAttribute: Attribute, leftAttribute: Attribute)
          if left.outputSet.contains(leftAttribute) && right.outputSet.contains(rightAttribute) =>
        Some(leftAttribute -> rightAttribute)
      case _ => None
    }
  }

  private def matchesCandidateJoin(
      condition: Expression,
      left: LogicalPlan,
      right: LogicalPlan,
      outerKey: Attribute,
      candidateKey: Attribute): Boolean = {
    extractSingleEquiJoinKey(condition, left, right).exists {
      case (leftKey, rightKey) =>
        leftKey.semanticEquals(outerKey) && rightKey.semanticEquals(candidateKey)
    }
  }

  private def references(expressions: Seq[Expression]): AttributeSet = {
    AttributeSet(expressions.flatMap(_.references))
  }

  private def splitAnd(expression: Expression): Seq[Expression] = expression match {
    case And(left, right) => splitAnd(left) ++ splitAnd(right)
    case other => Seq(other)
  }

  private def hasUserHint(hint: JoinHint): Boolean = {
    hint.leftHint.exists(_.strategy.isDefined) ||
    hint.rightHint.exists(_.strategy.isDefined)
  }
}
