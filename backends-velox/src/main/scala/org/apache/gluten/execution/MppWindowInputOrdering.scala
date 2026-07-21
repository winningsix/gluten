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

import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, Attribute, CurrentRow, Expression, Literal, RangeFrame, Rank, RowFrame, RowNumber, SortOrder, SpecifiedWindowFrame, UnboundedFollowing, UnboundedPreceding, WindowExpression, WindowSpecDefinition}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Count, Sum}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.types.{ByteType, DoubleType, IntegerType, LongType, ShortType}

private[execution] object MppWindowInputOrdering {
  sealed private trait WindowKind
  private case object RankLike extends WindowKind
  private case object FullPartitionCount extends WindowKind
  private case object RunningRangeSum extends WindowKind

  private[execution] case class OrderingStats(markedWindows: Int)

  def apply(plan: SparkPlan): (SparkPlan, OrderingStats) = {
    var markedWindows = 0
    plan.foreach {
      case window: WindowExecTransformer
          if hasCompleteLocalSort(window) && hasSupportedWindowShape(window) =>
        if (!WindowExecTransformer.inputsSorted(window)) {
          WindowExecTransformer.markInputsSorted(window)
          markedWindows += 1
        }
      case _ =>
    }
    plan -> OrderingStats(markedWindows)
  }

  private def hasCompleteLocalSort(window: WindowExecTransformer): Boolean = {
    window.partitionSpec.nonEmpty && (window.child match {
      case sort: SortExecTransformer if !sort.global =>
        val requiredOrdering =
          window.partitionSpec.map(SortOrder(_, Ascending)) ++ window.orderSpec
        sort.sortOrder.length == requiredOrdering.length &&
        SortOrder.orderingSatisfies(sort.sortOrder, requiredOrdering)
      case _ => false
    })
  }

  private def hasSupportedWindowShape(window: WindowExecTransformer): Boolean = {
    val kinds = window.windowExpression.map(classify(_, window))
    kinds.nonEmpty && kinds.forall(_.isDefined) && kinds.flatten.distinct.size == 1
  }

  private def classify(expression: Expression, window: WindowExecTransformer): Option[WindowKind] =
    expression match {
      case Alias(WindowExpression(function, spec: WindowSpecDefinition), _)
          if matchesPhysicalWindow(spec, window) =>
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

  private def matchesPhysicalWindow(
      spec: WindowSpecDefinition,
      window: WindowExecTransformer): Boolean = {
    sameExpressions(spec.partitionSpec, window.partitionSpec) &&
    sameSortOrders(spec.orderSpec, window.orderSpec)
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

  private def sameExpressions(left: Seq[Expression], right: Seq[Expression]): Boolean = {
    left.length == right.length && left.zip(right).forall {
      case (leftExpression, rightExpression) => leftExpression.semanticEquals(rightExpression)
    }
  }

  private def sameSortOrders(left: Seq[SortOrder], right: Seq[SortOrder]): Boolean = {
    left.length == right.length && left.zip(right).forall {
      case (leftOrder, rightOrder) => leftOrder.semanticEquals(rightOrder)
    }
  }
}
