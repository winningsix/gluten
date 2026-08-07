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

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.analysis.MultiInstanceRelation
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, EqualTo, Expression, ExprId, IsNotNull, Literal, NamedExpression, PredicateHelper}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Sum}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, LeafNode, LogicalPlan, Project, SubqueryAlias}
import org.apache.spark.sql.catalyst.rules.Rule

import scala.collection.mutable

/**
 * Reuse a self-aggregate instead of scanning and aggregating the same relation a second time.
 *
 * The admitted shape is deliberately narrow:
 *
 * Aggregate(..., sum(rhs.value), Project(..., Join(leftWithSummary, rhs, Inner, key = key)))
 *
 * `leftWithSummary` must already contain a one-row-per-key `sum(value)` from the same source
 * relation and source ordinals, optionally filtered and joined to other relations. For every row
 * produced by `leftWithSummary`, joining `rhs` and summing `rhs.value` is exactly the precomputed
 * per-key sum. Replacing the second scan with that sum preserves bag semantics even when the
 * surrounding dimension joins duplicate a key: the outer Aggregate still sums one precomputed value
 * per left row.
 *
 * This is a relational optimization, not a query-name or SQL-text rewrite. Same-source,
 * key/value-ordinal, equality-path, deterministic-filter, and projection guards must all prove the
 * substitution before it fires.
 */
case class EliminateRedundantSelfAggregateJoin(spark: SparkSession)
  extends Rule[LogicalPlan]
  with PredicateHelper
  with Logging {

  private case class SourceView(
      source: LeafNode,
      outputOrdinals: Map[ExprId, Int],
      sourcePredicates: Seq[Expression]) {
    def sourceOrdinal(attribute: Attribute): Option[Int] = {
      outputOrdinals
        .get(attribute.exprId)
        .orElse {
          source.output.zipWithIndex.collectFirst {
            case (output, index) if output.semanticEquals(attribute) => index
          }
        }
    }
  }

  private case class Summary(
      aggregate: Aggregate,
      key: Attribute,
      value: Attribute,
      sum: Attribute,
      source: SourceView)

  private val enabled = spark.sessionState.conf
    .getConfString("spark.gluten.sql.optimizer.eliminateRedundantSelfAggregateJoin.enabled", "true")
    .toBoolean

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!enabled || !plan.resolved || plan.isStreaming) {
      return plan
    }
    plan.transformUp {
      case outer: Aggregate =>
        outer.child match {
          case project @ Project(
                projectList,
                join @ Join(left, right, Inner, Some(condition), _)) =>
            rewrite(outer, outer.aggregateExpressions, projectList, left, right, condition)
              .getOrElse(outer)
          case _ => outer
        }
    }
  }

  private def rewrite(
      outer: Aggregate,
      aggregateExpressions: Seq[NamedExpression],
      projectList: Seq[NamedExpression],
      left: LogicalPlan,
      right: LogicalPlan,
      joinCondition: Expression): Option[LogicalPlan] = {
    val rightValueSums = aggregateExpressions.flatMap {
      expression =>
        expression.collect {
          case aggregate @ AggregateExpression(sum @ Sum(value: Attribute, _), _, false, None, _)
              if isOutput(value, right) =>
            (aggregate, sum, value)
        }
    }
    if (rightValueSums.length != 1) {
      return None
    }
    val (outerAggregateExpression, _, rightValue) = rightValueSums.head

    // The removable RHS value may be consumed only by the SUM that is replaced below. Once the
    // join is gone, the child no longer produces rightValue, so grouping by it or referencing it
    // from another aggregate would leave an unresolved (or, worse, mis-bound) expression.
    def referencesRightValue(expression: Expression): Boolean =
      expression.references.exists(_.semanticEquals(rightValue))
    val hasUnsupportedOuterReference =
      outer.groupingExpressions.exists(referencesRightValue) ||
        aggregateExpressions.exists {
          expression =>
            val withoutTargetSum = expression.transformDown {
              case aggregate: AggregateExpression
                  if aggregate.resultId == outerAggregateExpression.resultId =>
                Literal.create(null, aggregate.dataType)
            }
            referencesRightValue(withoutTargetSum)
        }
    if (hasUnsupportedOuterReference) {
      return None
    }

    val equalityPairs = splitConjunctivePredicates(joinCondition).flatMap {
      case EqualTo(first: Attribute, second: Attribute)
          if isOutput(first, left) && isOutput(second, right) =>
        Some(first -> second)
      case EqualTo(first: Attribute, second: Attribute)
          if isOutput(second, left) && isOutput(first, right) =>
        Some(second -> first)
      case _ => None
    }
    if (
      equalityPairs.length != 1 ||
      splitConjunctivePredicates(joinCondition).length != equalityPairs.length
    ) {
      return None
    }
    val (leftKey, rightKey) = equalityPairs.head

    // The Project above the removable join may consume only left expressions plus the direct value
    // attribute whose SUM is being substituted. Checking flattened references is insufficient: an
    // expression such as Alias(rightValue + 1) contains the admitted attribute, but would remain
    // bound to the discarded RHS after the join is removed.
    val rightConsumers = projectList.filter {
      expression => expression.references.exists(isOutput(_, right))
    }
    if (
      rightConsumers.exists {
        case attribute: Attribute => !attribute.semanticEquals(rightValue)
        case _ => true
      } ||
      !projectList.exists {
        case attribute: Attribute => attribute.semanticEquals(rightValue)
        case _ => false
      }
    ) {
      return None
    }

    val rightView = sourceView(right).getOrElse(return None)
    val rightKeyOrdinal = rightView.sourceOrdinal(rightKey).getOrElse(return None)
    val rightValueOrdinal = rightView.sourceOrdinal(rightValue).getOrElse(return None)
    if (!predicatesAreRedundantKeyNullChecks(rightView, rightKeyOrdinal)) {
      return None
    }

    val summaries = collectSummaries(left).filter {
      summary =>
        sameSource(summary.source, rightView) &&
        summary.source.sourceOrdinal(summary.key).contains(rightKeyOrdinal) &&
        summary.source.sourceOrdinal(summary.value).contains(rightValueOrdinal) &&
        predicatesAreRedundantKeyNullChecks(summary.source, rightKeyOrdinal) &&
        hasFilterReferencing(left, summary.aggregate, summary.sum) &&
        keysAreConnected(left, leftKey, summary.key)
    }
    if (summaries.length != 1) {
      return None
    }
    val summary = summaries.head
    val (newLeft, carriedSum) =
      propagateSummary(left, summary.aggregate, summary.key, summary.sum).getOrElse(return None)

    val newProjectList = projectList.filterNot {
      case attribute: Attribute => attribute.semanticEquals(rightValue)
      case _ => false
    } :+ carriedSum
    val newProject = Project(newProjectList, newLeft)

    var substitutions = 0
    val newAggregateExpressions = aggregateExpressions.map {
      named =>
        named
          .transformUp {
            case aggregate: AggregateExpression
                if aggregate.resultId == outerAggregateExpression.resultId =>
              substitutions += 1
              aggregate.copy(aggregateFunction = Sum(carriedSum))
          }
          .asInstanceOf[NamedExpression]
    }
    if (substitutions != 1) {
      return None
    }

    logInfo(
      "EliminateRedundantSelfAggregateJoin: removed one self scan and reused " +
        s"${summary.sum.name} for source=${summary.source.source.nodeName} " +
        s"keyOrdinal=$rightKeyOrdinal valueOrdinal=$rightValueOrdinal")
    Some(outer.copy(aggregateExpressions = newAggregateExpressions, child = newProject))
  }

  private def collectSummaries(plan: LogicalPlan): Seq[Summary] = {
    plan.collect {
      case aggregate: Aggregate if aggregate.groupingExpressions.length == 1 =>
        val grouping = aggregate.groupingExpressions
        val aggregateExpressions = aggregate.aggregateExpressions
        val child = aggregate.child
        val key = grouping.head match {
          case attribute: Attribute => attribute
          case _ => null
        }
        if (key == null) {
          None
        } else {
          val sums = aggregateExpressions.flatMap {
            case alias @ Alias(
                  AggregateExpression(Sum(value: Attribute, _), _, false, None, _),
                  _) =>
              Some((value, alias.toAttribute))
            case _ => None
          }
          if (sums.length != 1) {
            None
          } else {
            sourceView(child).flatMap {
              view =>
                val (value, sum) = sums.head
                if (
                  view.sourceOrdinal(key).isDefined &&
                  view.sourceOrdinal(value).isDefined
                ) {
                  Some(Summary(aggregate, key, value, sum, view))
                } else {
                  None
                }
            }
          }
        }
    }.flatten
  }

  private def sourceView(plan: LogicalPlan): Option[SourceView] = {
    def loop(current: LogicalPlan): Option[SourceView] = current match {
      case leaf: LeafNode if leaf.output.nonEmpty && !leaf.isStreaming =>
        Some(SourceView(leaf, leaf.output.map(_.exprId).zipWithIndex.toMap, Seq.empty))

      case Filter(condition, child) if condition.deterministic =>
        loop(child).flatMap {
          view =>
            replaceWithSourceAttributes(condition, view).map {
              sourceCondition =>
                view.copy(
                  outputOrdinals = passthroughOrdinals(current.output, child.output, view),
                  sourcePredicates =
                    view.sourcePredicates ++ splitConjunctivePredicates(sourceCondition)
                )
            }
        }

      case Project(projectList, child) =>
        loop(child).flatMap {
          view =>
            val ordinals = projectList.map(projectOrdinal(_, view))
            if (ordinals.forall(_.isDefined)) {
              Some(
                view.copy(
                  outputOrdinals = current.output.map(_.exprId).zip(ordinals.flatten).toMap))
            } else {
              None
            }
        }

      case SubqueryAlias(_, child) =>
        loop(child).map {
          view =>
            view.copy(outputOrdinals = passthroughOrdinals(current.output, child.output, view))
        }

      case _ => None
    }
    loop(plan)
  }

  private def passthroughOrdinals(
      output: Seq[Attribute],
      childOutput: Seq[Attribute],
      childView: SourceView): Map[ExprId, Int] = {
    output
      .zip(childOutput)
      .flatMap { case (out, child) => childView.sourceOrdinal(child).map(out.exprId -> _) }
      .toMap
  }

  private def projectOrdinal(expression: NamedExpression, childView: SourceView): Option[Int] =
    expression match {
      case attribute: Attribute => childView.sourceOrdinal(attribute)
      case _ => None
    }

  private def replaceWithSourceAttributes(
      expression: Expression,
      view: SourceView): Option[Expression] = {
    if (!expression.references.forall(view.sourceOrdinal(_).isDefined)) {
      return None
    }
    Some(expression.transform {
      case attribute: Attribute =>
        view.source.output(view.sourceOrdinal(attribute).get)
    })
  }

  private def sameSource(left: SourceView, right: SourceView): Boolean = {
    (left.source, right.source) match {
      case (_: MultiInstanceRelation, _: MultiInstanceRelation) =>
        left.source.getClass == right.source.getClass &&
        left.source.output.length == right.source.output.length &&
        left.source.output.zip(right.source.output).forall {
          case (l, r) => l.dataType == r.dataType && l.nullable == r.nullable
        } && left.source.sameResult(right.source)
      case _ => false
    }
  }

  private def predicatesAreRedundantKeyNullChecks(view: SourceView, keyOrdinal: Int): Boolean = {
    view.sourcePredicates.forall {
      case IsNotNull(attribute: Attribute) =>
        view.source.output.zipWithIndex.exists {
          case (sourceAttribute, ordinal) =>
            ordinal == keyOrdinal && sourceAttribute.semanticEquals(attribute)
        }
      case _ => false
    }
  }

  private def hasFilterReferencing(
      root: LogicalPlan,
      aggregate: Aggregate,
      sum: Attribute): Boolean = {
    root.exists {
      case Filter(condition, child) =>
        condition.deterministic &&
        condition.references.exists(_.semanticEquals(sum)) &&
        child.exists(_ eq aggregate)
      case _ => false
    }
  }

  private def keysAreConnected(plan: LogicalPlan, first: Attribute, second: Attribute): Boolean = {
    val graph = mutable.HashMap.empty[ExprId, mutable.Set[ExprId]]
    plan.foreach {
      case Join(_, _, Inner, Some(condition), _) =>
        splitConjunctivePredicates(condition).foreach {
          case EqualTo(left: Attribute, right: Attribute) =>
            graph.getOrElseUpdate(left.exprId, mutable.Set.empty) += right.exprId
            graph.getOrElseUpdate(right.exprId, mutable.Set.empty) += left.exprId
          case _ =>
        }
      case _ =>
    }
    val visited = mutable.Set(first.exprId)
    val pending = mutable.Queue(first.exprId)
    while (pending.nonEmpty) {
      val current = pending.dequeue()
      graph.getOrElse(current, mutable.Set.empty).foreach {
        next =>
          if (!visited.contains(next)) {
            visited += next
            pending.enqueue(next)
          }
      }
    }
    visited.contains(second.exprId)
  }

  private def propagateSummary(
      plan: LogicalPlan,
      target: Aggregate,
      targetKey: Attribute,
      targetSum: Attribute): Option[(LogicalPlan, Attribute)] = {
    if (plan eq target) {
      return Some(plan -> targetSum)
    }
    plan match {
      case Project(projectList, child) if child.exists(_ eq target) =>
        propagateSummary(child, target, targetKey, targetSum).map {
          case (newChild, carried) =>
            val newProjects =
              if (projectList.exists(_.toAttribute.semanticEquals(carried))) projectList
              else projectList :+ carried
            val rewritten = Project(newProjects, newChild)
            rewritten -> rewritten.output.find(_.exprId == carried.exprId).get
        }

      case Filter(condition, child) if child.exists(_ eq target) =>
        propagateSummary(child, target, targetKey, targetSum).map {
          case (newChild, carried) => Filter(condition, newChild) -> carried
        }

      case alias @ SubqueryAlias(_, child) if child.exists(_ eq target) =>
        propagateSummary(child, target, targetKey, targetSum).map {
          case (newChild, carried) =>
            val rewritten = alias.withNewChildren(Seq(newChild)).asInstanceOf[SubqueryAlias]
            rewritten -> rewritten.output.find(_.exprId == carried.exprId).get
        }

      case aggregate: Aggregate if aggregate.child.exists(_ eq target) =>
        val grouping = aggregate.groupingExpressions
        val outputs = aggregate.aggregateExpressions
        val child = aggregate.child
        propagateSummary(child, target, targetKey, targetSum).flatMap {
          case (newChild, carried)
              if isKeyOnlyDedup(aggregate, targetKey) &&
                pathPreservesKeyUniqueness(child, target) &&
                !outputs.exists(_.toAttribute.semanticEquals(carried)) =>
            // The target Aggregate already emits one row per key, and every node between it and
            // this IN-subquery dedup is cardinality preserving. Keeping another Aggregate over
            // hundreds of millions of summary rows is both redundant and expensive.
            val rewritten = Project(outputs :+ carried, newChild)
            Some(rewritten -> rewritten.output.find(_.exprId == carried.exprId).get)
          case (newChild, carried)
              if isKeyOnlyDedup(aggregate, targetKey) &&
                pathPreservesKeyUniqueness(child, target) &&
                outputs.exists(_.toAttribute.semanticEquals(carried)) =>
            val rewritten = Project(outputs, newChild)
            Some(rewritten -> rewritten.output.find(_.exprId == carried.exprId).get)
          case (newChild, carried)
              if isKeyOnlyDedup(aggregate, targetKey) &&
                !outputs.exists(_.toAttribute.semanticEquals(carried)) =>
            val rewritten =
              aggregate.copy(
                groupingExpressions = grouping :+ carried,
                aggregateExpressions = outputs :+ carried,
                child = newChild)
            Some(rewritten -> rewritten.output.find(_.exprId == carried.exprId).get)
          case (newChild, carried) if outputs.exists(_.toAttribute.semanticEquals(carried)) =>
            val rewritten = aggregate.copy(child = newChild)
            Some(rewritten -> rewritten.output.find(_.exprId == carried.exprId).get)
          case _ => None
        }

      case join: Join =>
        val inLeft = join.left.exists(_ eq target)
        val inRight = join.right.exists(_ eq target)
        if (inLeft == inRight) {
          None
        } else if (inLeft) {
          propagateSummary(join.left, target, targetKey, targetSum).map {
            case (newLeft, carried) => join.copy(left = newLeft) -> carried
          }
        } else {
          propagateSummary(join.right, target, targetKey, targetSum).map {
            case (newRight, carried) => join.copy(right = newRight) -> carried
          }
        }

      case _ => None
    }
  }

  private def isKeyOnlyDedup(aggregate: Aggregate, targetKey: Attribute): Boolean = {
    aggregate.aggregateExpressions.forall {
      case attribute: Attribute =>
        aggregate.groupingExpressions.exists(_.semanticEquals(attribute))
      case _ => false
    } &&
    aggregate.groupingExpressions.exists {
      case attribute: Attribute => attribute.semanticEquals(targetKey)
      case _ => false
    }
  }

  private def pathPreservesKeyUniqueness(plan: LogicalPlan, target: Aggregate): Boolean = {
    if (plan eq target) {
      true
    } else {
      plan match {
        case Project(_, child) => pathPreservesKeyUniqueness(child, target)
        case Filter(_, child) => pathPreservesKeyUniqueness(child, target)
        case SubqueryAlias(_, child) => pathPreservesKeyUniqueness(child, target)
        case _ => false
      }
    }
  }

  private def isOutput(attribute: Attribute, plan: LogicalPlan): Boolean = {
    plan.output.exists(_.semanticEquals(attribute))
  }
}
