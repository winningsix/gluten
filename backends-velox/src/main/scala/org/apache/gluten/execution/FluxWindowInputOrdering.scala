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

import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, Attribute, AttributeSet, CurrentRow, Expression, Literal, NamedExpression, RangeFrame, Rank, RowFrame, RowNumber, SortOrder, SpecifiedWindowFrame, UnboundedFollowing, UnboundedPreceding, WindowExpression, WindowSpecDefinition}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Count, Sum}
import org.apache.spark.sql.execution.{ProjectExec, SparkPlan}
import org.apache.spark.sql.types.{ByteType, DoubleType, IntegerType, LongType, ShortType}

private[execution] object FluxWindowInputOrdering {
  sealed private trait WindowKind
  private case object RankLike extends WindowKind
  private case object FullPartitionCount extends WindowKind
  private case object RunningRangeSum extends WindowKind

  private[execution] case class OrderingStats(
      candidateWindows: Int,
      verifiedSortWindows: Int,
      supportedShapeWindows: Int,
      markedWindows: Int)

  def apply(plan: SparkPlan): (SparkPlan, OrderingStats) = {
    var candidateWindows = 0
    var verifiedSortWindows = 0
    var supportedShapeWindows = 0
    var markedWindows = 0
    plan.foreach {
      case window: WindowExecTransformer =>
        candidateWindows += 1
        val hasSort = hasCompleteLocalSort(window)
        val hasShape = hasSupportedWindowShape(window)
        if (hasSort) {
          verifiedSortWindows += 1
        }
        if (hasShape) {
          supportedShapeWindows += 1
        }
        if (hasSort && hasShape) {
          if (!WindowExecTransformer.inputsSorted(window)) {
            WindowExecTransformer.markInputsSorted(window)
            markedWindows += 1
          }
        }
      case _ =>
    }
    plan -> OrderingStats(
      candidateWindows,
      verifiedSortWindows,
      supportedShapeWindows,
      markedWindows)
  }

  private def hasCompleteLocalSort(window: WindowExecTransformer): Boolean = {
    def find(child: SparkPlan, partitionSpec: Seq[Expression], orderSpec: Seq[SortOrder]): Boolean =
      child match {
        case sort: SortExecTransformer if !sort.global =>
          hasRequiredOrdering(sort, partitionSpec, orderSpec)
        case project: ProjectExecTransformer if project.projectList.forall(_.deterministic) =>
          rewriteOrderingThroughProject(partitionSpec, orderSpec, project.projectList).exists {
            case (childPartitionSpec, childOrderSpec) =>
              find(project.child, childPartitionSpec, childOrderSpec)
          }
        case project: ProjectExec if project.projectList.forall(_.deterministic) =>
          rewriteOrderingThroughProject(partitionSpec, orderSpec, project.projectList).exists {
            case (childPartitionSpec, childOrderSpec) =>
              find(project.child, childPartitionSpec, childOrderSpec)
          }
        case _ => false
      }

    window.partitionSpec.nonEmpty &&
    find(window.child, window.partitionSpec, window.orderSpec)
  }

  private[execution] def hasRequiredOrdering(
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
          resolveProjectLineage(actual.child, sort.child)
            .semanticEquals(resolveProjectLineage(required.child, sort.child))
      })
  }

  private def resolveProjectLineage(expression: Expression, plan: SparkPlan): Expression = {
    plan match {
      case project: ProjectExecTransformer if project.projectList.forall(_.deterministic) =>
        rewriteExpressionsThroughProject(Seq(expression), project.projectList)
          .map(_.head)
          .map(resolveProjectLineage(_, project.child))
          .getOrElse(expression)
      case project: ProjectExec if project.projectList.forall(_.deterministic) =>
        rewriteExpressionsThroughProject(Seq(expression), project.projectList)
          .map(_.head)
          .map(resolveProjectLineage(_, project.child))
          .getOrElse(expression)
      case _ => expression
    }
  }

  private[execution] def rewriteOrderingThroughProject(
      partitionSpec: Seq[Expression],
      orderSpec: Seq[SortOrder],
      projectList: Seq[NamedExpression]): Option[(Seq[Expression], Seq[SortOrder])] = {
    rewriteExpressionsThroughProject(partitionSpec ++ orderSpec, projectList).map {
      rewritten =>
        val (partitions, orders) = rewritten.splitAt(partitionSpec.size)
        partitions -> orders.map(_.asInstanceOf[SortOrder])
    }
  }

  private def rewriteExpressionsThroughProject(
      expressions: Seq[Expression],
      projectList: Seq[NamedExpression]): Option[Seq[Expression]] = {
    val projectOutput = AttributeSet(projectList.map(_.toAttribute))
    val references = expressions.foldLeft(AttributeSet.empty)(_ ++ _.references)
    if (!references.subsetOf(projectOutput)) {
      return None
    }

    val replacements = projectList.map {
      case alias: Alias => alias.exprId -> alias.child
      case attribute: Attribute => attribute.exprId -> attribute
      case expression => expression.exprId -> expression
    }.toMap
    def rewrite(expression: Expression): Expression = {
      expression.transform {
        case attribute: Attribute if replacements.contains(attribute.exprId) =>
          replacements(attribute.exprId)
      }
    }

    Some(expressions.map(rewrite))
  }

  private def hasSupportedWindowShape(window: WindowExecTransformer): Boolean = {
    val kinds = window.windowExpression.map(classify(_, window))
    kinds.nonEmpty && kinds.forall(_.isDefined) && kinds.flatten.distinct.size == 1
  }

  private def classify(expression: Expression, window: WindowExecTransformer): Option[WindowKind] =
    expression match {
      case Alias(WindowExpression(function, spec: WindowSpecDefinition), _) =>
        function match {
          case _: RowNumber | _: Rank
              if window.orderSpec.nonEmpty &&
                hasFrame(spec, UnboundedPreceding, CurrentRow) =>
            Some(RankLike)
          case aggregate: AggregateExpression
              if !aggregate.isDistinct && aggregate.filter.isEmpty =>
            aggregate.aggregateFunction match {
              case count: Count
                  if window.orderSpec.isEmpty &&
                    count.children.size == 1 &&
                    count.children.head.isInstanceOf[Literal] &&
                    count.children.head.asInstanceOf[Literal].value != null &&
                    hasFrame(spec, UnboundedPreceding, UnboundedFollowing, Some(RowFrame)) =>
                Some(FullPartitionCount)
              case sum: Sum
                  if window.orderSpec.nonEmpty &&
                    sum.child.isInstanceOf[Attribute] &&
                    supportedRangeSumType(sum.child.dataType) &&
                    hasFrame(spec, UnboundedPreceding, CurrentRow, Some(RangeFrame)) =>
                Some(RunningRangeSum)
              case _ => None
            }
          case _ => None
        }
      case _ => None
    }

  private def hasFrame(
      spec: WindowSpecDefinition,
      lower: Expression,
      upper: Expression,
      frameType: Option[org.apache.spark.sql.catalyst.expressions.FrameType] = None): Boolean = {
    spec.frameSpecification match {
      case SpecifiedWindowFrame(actualType, actualLower, actualUpper) =>
        frameType.forall(_ == actualType) &&
        actualLower == lower &&
        actualUpper == upper
      case _ => false
    }
  }

  private def supportedRangeSumType(dataType: org.apache.spark.sql.types.DataType): Boolean =
    dataType == ByteType ||
      dataType == ShortType ||
      dataType == IntegerType ||
      dataType == LongType ||
      dataType == DoubleType

}
