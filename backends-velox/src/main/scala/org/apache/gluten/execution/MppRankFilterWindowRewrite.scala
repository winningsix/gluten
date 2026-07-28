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

import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, AttributeSet, Expression, Murmur3Hash, Rank, RowNumber, SortOrder, WindowExpression}
import org.apache.spark.sql.catalyst.plans.physical.{ClusteredDistribution, HashPartitioning}
import org.apache.spark.sql.execution.{ColumnarInputAdapter, ColumnarShuffleExchangeExec, InputIteratorTransformer, ProjectExec, SparkPlan}
import org.apache.spark.sql.execution.exchange.ENSURE_REQUIREMENTS
import org.apache.spark.sql.execution.window.{GlutenFinal, GlutenPartial}

/**
 * Selects the real Window implementation for Spark's rank-filter plan shape.
 *
 * Spark's InferWindowGroupLimit optimization places Partial and Final WindowGroupLimit operators
 * below the original Window. The Velox backend maps those pruning operators to TopNRowNumber. This
 * rewrite removes the redundant pruning operators and keeps the semantic Window, its local Sort and
 * the upper Filter.
 *
 * A partitioned Window must see every row for a partition on the same MPP peer. Spark normally
 * inserts a HASH exchange for the Final WindowGroupLimit. We preserve that exchange and insert one
 * explicitly if the rewritten child no longer satisfies the Window distribution.
 *
 * Global (empty partitionSpec) windows are deliberately left unchanged: changing them here would
 * require a SINGLE exchange and would serialize the query onto one peer.
 */
private[execution] object MppRankFilterWindowRewrite {
  val EnabledKey: String = "spark.gluten.mpp.rankFilterWindow.enabled"

  private[execution] def isEnabled(configured: Option[Boolean], cudfEnabled: Boolean): Boolean = {
    configured.getOrElse(cudfEnabled)
  }

  private[execution] case class RewriteStats(rewrittenWindows: Int, insertedHashExchanges: Int)
  private type HashContract = (Seq[Expression], Seq[Attribute])
  private case class LocalSortPath(
      sort: SortExecTransformer,
      projectsAboveSort: Seq[SparkPlan],
      partitionSpec: Seq[Expression],
      orderSpec: Seq[SortOrder])

  def apply(plan: SparkPlan, enabled: Boolean, numPartitions: Int): (SparkPlan, RewriteStats) = {
    if (!enabled) {
      return plan -> RewriteStats(0, 0)
    }

    var rewrittenWindows = 0
    var insertedHashExchanges = 0
    val rewritten = plan.transformUp {
      case window: WindowExecTransformer
          if window.partitionSpec.nonEmpty &&
            isSupportedRankWindow(window) =>
        findRequiredLocalSort(window).flatMap {
          localSort =>
            stripMatchingGroupLimits(
              localSort.sort.child,
              localSort.partitionSpec,
              localSort.orderSpec,
              window).map(withoutGroupLimits => localSort -> withoutGroupLimits)
        } match {
          case Some((localSort, withoutGroupLimits)) =>
            val partitionReferences = localSort.partitionSpec.foldLeft(AttributeSet.empty) {
              case (references, expression) => references ++ expression.references
            }

            // Do not synthesize an invalid exchange if PullOutPreProject failed to materialize a
            // partition expression below the Window. Keeping the old TopN path is safer.
            if (!partitionReferences.subsetOf(AttributeSet(withoutGroupLimits.output))) {
              window
            } else {
              val distributedChild =
                if (
                  hasCompatibleNativeHashDistribution(withoutGroupLimits, localSort.partitionSpec)
                ) {
                  withoutGroupLimits
                } else {
                  insertedHashExchanges += 1
                  ColumnarShuffleExchangeExec(
                    HashPartitioning(localSort.partitionSpec, math.max(1, numPartitions)),
                    withoutGroupLimits,
                    ENSURE_REQUIREMENTS,
                    withoutGroupLimits.output,
                    None)
                }

              rewrittenWindows += 1
              val rewrittenSort = localSort.sort.withNewChildren(Seq(distributedChild))
              val rewrittenChild =
                localSort.projectsAboveSort.reverse.foldLeft(rewrittenSort: SparkPlan) {
                  case (child, project) => project.withNewChildren(Seq(child))
                }
              window.withNewChildren(Seq(rewrittenChild))
            }
          case None => window
        }
    }

    rewritten -> RewriteStats(rewrittenWindows, insertedHashExchanges)
  }

  private def findRequiredLocalSort(window: WindowExecTransformer): Option[LocalSortPath] = {
    def find(
        child: SparkPlan,
        partitionSpec: Seq[Expression],
        orderSpec: Seq[SortOrder],
        projectsAboveSort: Seq[SparkPlan]): Option[LocalSortPath] = child match {
      case sort: SortExecTransformer if !sort.global =>
        if (MppWindowInputOrdering.hasRequiredOrdering(sort, partitionSpec, orderSpec)) {
          Some(LocalSortPath(sort, projectsAboveSort, partitionSpec, orderSpec))
        } else {
          None
        }
      case project: ProjectExecTransformer if project.projectList.forall(_.deterministic) =>
        MppWindowInputOrdering
          .rewriteOrderingThroughProject(partitionSpec, orderSpec, project.projectList)
          .flatMap {
            case (childPartitionSpec, childOrderSpec) =>
              find(project.child, childPartitionSpec, childOrderSpec, projectsAboveSort :+ project)
          }
      case project: ProjectExec if project.projectList.forall(_.deterministic) =>
        MppWindowInputOrdering
          .rewriteOrderingThroughProject(partitionSpec, orderSpec, project.projectList)
          .flatMap {
            case (childPartitionSpec, childOrderSpec) =>
              find(project.child, childPartitionSpec, childOrderSpec, projectsAboveSort :+ project)
          }
      case _ => None
    }

    find(window.child, window.partitionSpec, window.orderSpec, Seq.empty)
  }

  private def stripMatchingGroupLimits(
      child: SparkPlan,
      partitionSpec: Seq[Expression],
      orderSpec: Seq[SortOrder],
      window: WindowExecTransformer): Option[SparkPlan] = child match {
    case groupLimit: WindowGroupLimitExecTransformer
        if groupLimit.limit == 1 &&
          groupLimit.mode == GlutenFinal &&
          matchesWindow(groupLimit, partitionSpec, orderSpec, window) =>
      stripMatchingPartialGroupLimit(
        groupLimit.child,
        groupLimit.partitionSpec,
        groupLimit.orderSpec,
        window)
    // PullOutPreProject may materialize a non-trivial partition/order expression between the
    // local Sort and WindowGroupLimit. Retain a deterministic project and all of its output
    // ExprIds. A nondeterministic project must keep the original pruning boundary because
    // evaluating it for all rows can change values on the rows that survive the upper filter.
    case project: ProjectExecTransformer if project.projectList.forall(_.deterministic) =>
      MppWindowInputOrdering
        .rewriteOrderingThroughProject(partitionSpec, orderSpec, project.projectList)
        .flatMap {
          case (childPartitionSpec, childOrderSpec) =>
            stripMatchingGroupLimits(project.child, childPartitionSpec, childOrderSpec, window).map(
              rewrittenChild => project.withNewChildren(Seq(rewrittenChild)))
        }
    case project: ProjectExec if project.projectList.forall(_.deterministic) =>
      MppWindowInputOrdering
        .rewriteOrderingThroughProject(partitionSpec, orderSpec, project.projectList)
        .flatMap {
          case (childPartitionSpec, childOrderSpec) =>
            stripMatchingGroupLimits(project.child, childPartitionSpec, childOrderSpec, window).map(
              rewrittenChild => project.withNewChildren(Seq(rewrittenChild)))
        }
    case _ => None
  }

  private def stripMatchingPartialGroupLimit(
      child: SparkPlan,
      partitionSpec: Seq[Expression],
      orderSpec: Seq[SortOrder],
      window: WindowExecTransformer,
      hashContract: Option[HashContract] = None): Option[SparkPlan] = child match {
    case groupLimit: WindowGroupLimitExecTransformer
        if groupLimit.limit == 1 &&
          groupLimit.mode == GlutenPartial &&
          matchesWindow(groupLimit, partitionSpec, orderSpec, window) =>
      Some(groupLimit.child)
    case exchange: ColumnarShuffleExchangeExec =>
      exchange.outputPartitioning match {
        case HashPartitioning(expressions, _) =>
          stripMatchingPartialGroupLimit(
            exchange.child,
            partitionSpec,
            orderSpec,
            window,
            Some(expressions -> exchange.output))
            .map(rewrittenChild => exchange.withNewChildren(Seq(rewrittenChild)))
        case _ => None
      }
    // CollapseTransformStages wraps each native pipeline below a shuffle in this transparent
    // stage container. Preserve the exact wrapper and its stage metadata while looking for the
    // synthetic hash project immediately below it.
    case wholeStage: WholeStageTransformer
        if hashContract.nonEmpty &&
          !wholeStage.wholeStageTransformerContextDefined &&
          wholeStage.child.isInstanceOf[ProjectExecTransformer] =>
      stripMatchingPartialGroupLimit(
        wholeStage.child,
        partitionSpec,
        orderSpec,
        window,
        hashContract)
        .flatMap {
          rewrittenChild =>
            val sameOutput =
              sameExpressions(rewrittenChild.output, wholeStage.child.output) &&
                rewrittenChild.output.map(_.exprId) == wholeStage.child.output.map(_.exprId)
            if (rewrittenChild.isInstanceOf[TransformSupport] && sameOutput) {
              Some(wholeStage.withNewChildren(Seq(rewrittenChild)))
            } else None
        }
    // ColumnarCollapseTransformStages wraps shuffle inputs in this exact adapter pair. Do not
    // admit arbitrary unary nodes here: the pair is a value-stream boundary with no expressions.
    case input: InputIteratorTransformer =>
      input.child match {
        case adapter: ColumnarInputAdapter
            if adapter.child.isInstanceOf[ColumnarShuffleExchangeExec] =>
          stripMatchingPartialGroupLimit(adapter.child, partitionSpec, orderSpec, window)
            .map {
              rewrittenChild =>
                val rewrittenAdapter = adapter.withNewChildren(Seq(rewrittenChild))
                input.withNewChildren(Seq(rewrittenAdapter))
            }
        case _ => None
      }
    // GPU shuffle preparation prepends one private Murmur3 hash column while exposing only the
    // original child output at the exchange. Preserve only that exact identity projection.
    case project: ProjectExecTransformer if hashContract.exists {
          case (hashExpressions, exchangeOutput) =>
            isSyntheticHashProject(project, hashExpressions, exchangeOutput)
        } =>
      stripMatchingPartialGroupLimit(project.child, partitionSpec, orderSpec, window, hashContract)
        .map(rewrittenChild => project.withNewChildren(Seq(rewrittenChild)))
    case _ => None
  }

  private def isSyntheticHashProject(
      project: ProjectExecTransformer,
      hashExpressions: Seq[Expression],
      exchangeOutput: Seq[Attribute]): Boolean = {
    project.projectList match {
      case Alias(hash: Murmur3Hash, "hash_partition_key") +: tail =>
        hash.semanticEquals(new Murmur3Hash(hashExpressions)) &&
        project.projectList.forall(_.deterministic) &&
        sameExpressions(tail, exchangeOutput) &&
        tail.map(_.exprId) == exchangeOutput.map(_.exprId) &&
        sameExpressions(project.child.output, exchangeOutput) &&
        project.child.output.map(_.exprId) == exchangeOutput.map(_.exprId)
      case _ => false
    }
  }

  private def hasCompatibleNativeHashDistribution(
      plan: SparkPlan,
      partitionSpec: Seq[Expression]): Boolean = {
    plan.outputPartitioning.satisfies(ClusteredDistribution(partitionSpec)) && plan.exists {
      case exchange: ColumnarShuffleExchangeExec =>
        exchange.outputPartitioning match {
          case HashPartitioning(expressions, _) => sameExpressions(expressions, partitionSpec)
          case _ => false
        }
      case _ => false
    }
  }

  private def matchesWindow(
      groupLimit: WindowGroupLimitExecTransformer,
      partitionSpec: Seq[Expression],
      orderSpec: Seq[SortOrder],
      window: WindowExecTransformer): Boolean = {
    sameExpressions(groupLimit.partitionSpec, partitionSpec) &&
    sameSortOrders(groupLimit.orderSpec, orderSpec) &&
    sameRankFunction(groupLimit.rankLikeFunction, window)
  }

  private def isSupportedRankWindow(window: WindowExecTransformer): Boolean = {
    window.windowExpression.nonEmpty && window.windowExpression.forall {
      case Alias(WindowExpression(_: RowNumber, _), _) => true
      case Alias(WindowExpression(_: Rank, _), _) => true
      case _ => false
    }
  }

  private def sameRankFunction(
      rankLikeFunction: Expression,
      window: WindowExecTransformer): Boolean = {
    window.windowExpression.forall {
      case Alias(WindowExpression(_: RowNumber, _), _) =>
        rankLikeFunction.isInstanceOf[RowNumber]
      case Alias(WindowExpression(_: Rank, _), _) => rankLikeFunction.isInstanceOf[Rank]
      case _ => false
    }
  }

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
