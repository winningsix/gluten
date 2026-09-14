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
package org.apache.gluten.execution

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, AttributeSet, EqualTo, Expression, Literal, RowNumber, SortOrder, WindowExpression}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.window.GlutenFinal

/**
 * Removes the semantic row_number Window and rank Filter after an exact Final WindowGroupLimit
 * top-1.
 *
 * Spark deliberately retains Window + Filter after InferWindowGroupLimit because the pruning node
 * is normally only an implementation aid. For the exact `row_number = 1` shape, however, Final
 * WindowGroupLimit has already emitted exactly the one winning row for every partition. If the
 * parent project drops the rank column, regenerating a column full of ones and filtering it again
 * is redundant. Preserve the local Sort by default because it can materially improve Parquet
 * compression of wide rows. A query-scoped opt-out is available when the sort itself dominates
 * execution time and downstream consumers do not require ordering.
 *
 * Keep the match intentionally narrow: one row_number expression, an equality-to-one filter,
 * matching Final top-1 partition/order contracts, a valid local sort, and a deterministic parent
 * project that does not reference the generated rank attribute.
 */
case class CudfRankOneTopNElisionRule(session: SparkSession) extends Rule[SparkPlan] {
  override def apply(plan: SparkPlan): SparkPlan = {
    val enabled = session.conf.getOption(CudfRankOneTopNElisionRule.EnabledKey).exists(_.toBoolean)
    val preserveSort = session.conf
      .getOption(CudfRankOneTopNElisionRule.PreserveSortKey)
      .forall(_.toBoolean)
    CudfRankOneTopNElisionRule.rewrite(plan, enabled, preserveSort)._1
  }
}

object CudfRankOneTopNElisionRule {
  val EnabledKey: String =
    "spark.gluten.sql.columnar.backend.velox.cudf.rankOneTopNElision.enabled"
  val PreserveSortKey: String =
    "spark.gluten.sql.columnar.backend.velox.cudf.rankOneTopNElision.preserveSort.enabled"

  private[execution] case class RewriteStats(elidedWindows: Int)

  private[execution] def rewrite(
      plan: SparkPlan,
      enabled: Boolean,
      preserveSort: Boolean = true): (SparkPlan, RewriteStats) = {
    if (!enabled) {
      return plan -> RewriteStats(0)
    }

    var elided = 0
    val rewritten = plan.transformUp {
      case project: ProjectExecTransformer =>
        collapse(project, preserveSort) match {
          case Some(child) =>
            elided += 1
            project.withNewChildren(Seq(child))
          case None => project
        }
    }
    rewritten -> RewriteStats(elided)
  }

  private def collapse(
      project: ProjectExecTransformer,
      preserveSort: Boolean): Option[SparkPlan] = {
    if (!project.projectList.forall(_.deterministic)) {
      return None
    }

    project.child match {
      case FilterExecTransformer(condition, window: WindowExecTransformer) =>
        exactRowNumberAlias(window).flatMap {
          rankAlias =>
            val rankAttribute = rankAlias.toAttribute
            if (!isEqualToOne(condition, rankAttribute)) {
              None
            } else if (project.projectList.exists(_.references.contains(rankAttribute))) {
              None
            } else {
              window.child match {
                case sort: SortExecTransformer
                    if !sort.global &&
                      hasRequiredOrdering(sort, window.partitionSpec, window.orderSpec) =>
                  sort.child match {
                    case groupLimit: WindowGroupLimitExecTransformer
                        if groupLimit.mode == GlutenFinal &&
                          groupLimit.limit == 1 &&
                          groupLimit.rankLikeFunction.isInstanceOf[RowNumber] &&
                          sameExpressions(groupLimit.partitionSpec, window.partitionSpec) &&
                          sameExpressions(groupLimit.orderSpec, window.orderSpec) &&
                          projectReferencesAvailable(project, groupLimit) =>
                      Some(if (preserveSort) sort else groupLimit)
                    case _ => None
                  }
                case _ => None
              }
            }
        }
      case _ => None
    }
  }

  private def exactRowNumberAlias(window: WindowExecTransformer): Option[Alias] = {
    window.windowExpression match {
      case Seq(alias @ Alias(WindowExpression(_: RowNumber, _), _)) => Some(alias)
      case _ => None
    }
  }

  private def isEqualToOne(condition: Expression, rank: Expression): Boolean = condition match {
    case EqualTo(left, literal: Literal) if left.semanticEquals(rank) => isOne(literal)
    case EqualTo(literal: Literal, right) if right.semanticEquals(rank) => isOne(literal)
    case _ => false
  }

  private def isOne(literal: Literal): Boolean = literal.value match {
    case value: Byte => value == 1
    case value: Short => value == 1
    case value: Int => value == 1
    case value: Long => value == 1L
    case _ => false
  }

  private def projectReferencesAvailable(
      project: ProjectExecTransformer,
      child: SparkPlan): Boolean = {
    val references = project.projectList.foldLeft(AttributeSet.empty) {
      case (all, expression) => all ++ expression.references
    }
    references.subsetOf(AttributeSet(child.output))
  }

  private def sameExpressions(left: Seq[Expression], right: Seq[Expression]): Boolean = {
    left.length == right.length && left.zip(right).forall {
      case (leftExpression, rightExpression) => leftExpression.semanticEquals(rightExpression)
    }
  }

  /**
   * Keep this check local instead of depending on MppWindowInputOrdering. The rule is deployed as
   * an incremental class update onto released Gluten bundles where that helper is not part of the
   * binary interface yet.
   */
  private def hasRequiredOrdering(
      sort: SortExecTransformer,
      partitionSpec: Seq[Expression],
      orderSpec: Seq[SortOrder]): Boolean = {
    val requiredOrdering = partitionSpec.map(SortOrder(_, Ascending)) ++ orderSpec
    sort.sortOrder.length == requiredOrdering.length &&
    (SortOrder.orderingSatisfies(sort.sortOrder, requiredOrdering) ||
      sort.sortOrder.zip(requiredOrdering).forall {
        case (actual, required) =>
          actual.direction == required.direction &&
          actual.nullOrdering == required.nullOrdering &&
          actual.child.semanticEquals(required.child)
      })
  }
}
