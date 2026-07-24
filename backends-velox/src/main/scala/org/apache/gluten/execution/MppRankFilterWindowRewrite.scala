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

import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, Attribute, AttributeSet, EqualTo, Expression, Literal, Murmur3Hash, Rank, RowNumber, SortOrder, WindowExpression}
import org.apache.spark.sql.catalyst.plans.physical.{ClusteredDistribution, HashPartitioning}
import org.apache.spark.sql.execution.{ColumnarInputAdapter, ColumnarShuffleExchangeExec, InputIteratorTransformer, ProjectExec, SparkPlan}
import org.apache.spark.sql.execution.exchange.ENSURE_REQUIREMENTS
import org.apache.spark.sql.execution.window.{GlutenFinal, GlutenPartial}

/**
 * Selects the native implementation for Spark's rank-filter plan shape.
 *
 * Spark's InferWindowGroupLimit optimization places Partial and Final WindowGroupLimit operators
 * below the original Window. The Velox backend maps those pruning operators to TopNRowNumber.
 *
 * An exact rank() = 1 subtree can use the Final TopNRowNumber as its semantic operator because it
 * preserves every first-place tie. For that shape, this rewrite removes the redundant Partial
 * TopNRowNumber, local Sort, Window and Filter. Other supported shapes remove both pruning
 * operators while retaining the semantic Window, its local Sort and the upper Filter.
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

  private[execution] case class RewriteStats(
      rewrittenWindows: Int,
      fusedRankFilters: Int,
      insertedHashExchanges: Int)
  private type HashContract = (Seq[Expression], Seq[Attribute])

  def apply(plan: SparkPlan, enabled: Boolean, numPartitions: Int): (SparkPlan, RewriteStats) = {
    if (!enabled) {
      return plan -> RewriteStats(0, 0, 0)
    }

    var rewrittenWindows = 0
    var fusedRankFilters = 0
    var insertedHashExchanges = 0
    val afterRankOneFusion = plan.transformDown {
      case filter: FilterExecTransformer =>
        fuseExactRankOneFilter(filter, numPartitions) match {
          case Some((fused, insertedExchange)) =>
            fusedRankFilters += 1
            if (insertedExchange) {
              insertedHashExchanges += 1
            }
            fused
          case None => filter
        }
    }

    val rewritten = afterRankOneFusion.transformUp {
      case window: WindowExecTransformer
          if window.partitionSpec.nonEmpty &&
            hasRequiredLocalSort(window) &&
            isSupportedRankWindow(window) =>
        val sort = window.child.asInstanceOf[SortExecTransformer]
        stripMatchingGroupLimits(sort.child, window) match {
          case Some(withoutGroupLimits) =>
            val partitionReferences = window.partitionSpec.foldLeft(AttributeSet.empty) {
              case (references, expression) => references ++ expression.references
            }

            // Do not synthesize an invalid exchange if PullOutPreProject failed to materialize a
            // partition expression below the Window. Keeping the old TopN path is safer.
            if (!partitionReferences.subsetOf(AttributeSet(withoutGroupLimits.output))) {
              window
            } else {
              val distributedChild =
                if (hasCompatibleNativeHashDistribution(withoutGroupLimits, window.partitionSpec)) {
                  withoutGroupLimits
                } else {
                  insertedHashExchanges += 1
                  ColumnarShuffleExchangeExec(
                    HashPartitioning(window.partitionSpec, math.max(1, numPartitions)),
                    withoutGroupLimits,
                    ENSURE_REQUIREMENTS,
                    withoutGroupLimits.output,
                    None)
                }

              rewrittenWindows += 1
              val rewrittenSort = sort.withNewChildren(Seq(distributedChild))
              window.withNewChildren(Seq(rewrittenSort))
            }
          case None => window
        }
    }

    rewritten -> RewriteStats(rewrittenWindows, fusedRankFilters, insertedHashExchanges)
  }

  private def fuseExactRankOneFilter(
      filter: FilterExecTransformer,
      numPartitions: Int): Option[(SparkPlan, Boolean)] = filter.child match {
    case window: WindowExecTransformer
        if window.partitionSpec.nonEmpty &&
          hasRequiredLocalSort(window) &&
          exactRankOneAlias(filter.condition, window).nonEmpty =>
      val rankAlias = exactRankOneAlias(filter.condition, window).get
      val sort = window.child.asInstanceOf[SortExecTransformer]
      retainMatchingFinalGroupLimit(sort.child, window, numPartitions).map {
        case (finalGroupLimit, insertedExchange) =>
          val rankOne = rankAlias.copy(child = Literal.create(1, rankAlias.dataType))(
            rankAlias.exprId,
            rankAlias.qualifier,
            rankAlias.explicitMetadata,
            rankAlias.nonInheritableMetadataKeys)
          ProjectExecTransformer(finalGroupLimit.output :+ rankOne, finalGroupLimit) ->
            insertedExchange
      }
    case _ => None
  }

  private def exactRankOneAlias(
      condition: Expression,
      window: WindowExecTransformer): Option[Alias] = {
    window.windowExpression match {
      case Seq(rankAlias @ Alias(WindowExpression(_: Rank, _), _))
          if isRankOnePredicate(condition, rankAlias.toAttribute) =>
        Some(rankAlias)
      case _ => None
    }
  }

  private def isRankOnePredicate(condition: Expression, rankAttribute: Attribute): Boolean = {
    condition match {
      case EqualTo(attribute: Attribute, literal: Literal) =>
        attribute.semanticEquals(rankAttribute) && isIntegralOne(literal)
      case EqualTo(literal: Literal, attribute: Attribute) =>
        attribute.semanticEquals(rankAttribute) && isIntegralOne(literal)
      case _ => false
    }
  }

  private def isIntegralOne(literal: Literal): Boolean = {
    literal.value match {
      case value: Byte => value == 1
      case value: Short => value == 1
      case value: Int => value == 1
      case value: Long => value == 1L
      case _ => false
    }
  }

  private def retainMatchingFinalGroupLimit(
      child: SparkPlan,
      window: WindowExecTransformer,
      numPartitions: Int): Option[(SparkPlan, Boolean)] = child match {
    case groupLimit: WindowGroupLimitExecTransformer
        if groupLimit.limit == 1 &&
          groupLimit.mode == GlutenFinal &&
          matchesWindow(groupLimit, window) =>
      stripMatchingPartialGroupLimit(groupLimit.child, window).flatMap {
        withoutPartial =>
          val partitionReferences = window.partitionSpec.foldLeft(AttributeSet.empty) {
            case (references, expression) => references ++ expression.references
          }
          if (!partitionReferences.subsetOf(AttributeSet(withoutPartial.output))) {
            None
          } else {
            val hasDistribution =
              hasCompatibleNativeHashDistribution(withoutPartial, window.partitionSpec)
            val distributedChild =
              if (hasDistribution) {
                withoutPartial
              } else {
                ColumnarShuffleExchangeExec(
                  HashPartitioning(window.partitionSpec, math.max(1, numPartitions)),
                  withoutPartial,
                  ENSURE_REQUIREMENTS,
                  withoutPartial.output,
                  None)
              }
            Some(groupLimit.withNewChildren(Seq(distributedChild)) -> !hasDistribution)
          }
      }
    case project: ProjectExecTransformer if project.projectList.forall(_.deterministic) =>
      retainMatchingFinalGroupLimit(project.child, window, numPartitions).map {
        case (rewrittenChild, insertedExchange) =>
          project.withNewChildren(Seq(rewrittenChild)) -> insertedExchange
      }
    case project: ProjectExec if project.projectList.forall(_.deterministic) =>
      retainMatchingFinalGroupLimit(project.child, window, numPartitions).map {
        case (rewrittenChild, insertedExchange) =>
          project.withNewChildren(Seq(rewrittenChild)) -> insertedExchange
      }
    case _ => None
  }

  private def hasRequiredLocalSort(window: WindowExecTransformer): Boolean = {
    window.child match {
      case sort: SortExecTransformer if !sort.global =>
        val requiredOrdering =
          window.partitionSpec.map(SortOrder(_, Ascending)) ++ window.orderSpec
        sort.sortOrder.length == requiredOrdering.length &&
        SortOrder.orderingSatisfies(sort.sortOrder, requiredOrdering)
      case _ => false
    }
  }

  private def stripMatchingGroupLimits(
      child: SparkPlan,
      window: WindowExecTransformer): Option[SparkPlan] = child match {
    case groupLimit: WindowGroupLimitExecTransformer
        if groupLimit.limit == 1 &&
          groupLimit.mode == GlutenFinal &&
          matchesWindow(groupLimit, window) =>
      stripMatchingPartialGroupLimit(groupLimit.child, window)
    // PullOutPreProject may materialize a non-trivial partition/order expression between the
    // local Sort and WindowGroupLimit. Retain a deterministic project and all of its output
    // ExprIds. A nondeterministic project must keep the original pruning boundary because
    // evaluating it for all rows can change values on the rows that survive the upper filter.
    case project: ProjectExecTransformer if project.projectList.forall(_.deterministic) =>
      stripMatchingGroupLimits(project.child, window).map(
        rewrittenChild => project.withNewChildren(Seq(rewrittenChild)))
    case project: ProjectExec if project.projectList.forall(_.deterministic) =>
      stripMatchingGroupLimits(project.child, window).map(
        rewrittenChild => project.withNewChildren(Seq(rewrittenChild)))
    case _ => None
  }

  private def stripMatchingPartialGroupLimit(
      child: SparkPlan,
      window: WindowExecTransformer,
      hashContract: Option[HashContract] = None): Option[SparkPlan] = child match {
    case groupLimit: WindowGroupLimitExecTransformer
        if groupLimit.limit == 1 &&
          groupLimit.mode == GlutenPartial &&
          matchesWindow(groupLimit, window) =>
      Some(groupLimit.child)
    case exchange: ColumnarShuffleExchangeExec =>
      exchange.outputPartitioning match {
        case HashPartitioning(expressions, _) =>
          stripMatchingPartialGroupLimit(
            exchange.child,
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
      stripMatchingPartialGroupLimit(wholeStage.child, window, hashContract)
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
          stripMatchingPartialGroupLimit(adapter.child, window).map {
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
      stripMatchingPartialGroupLimit(project.child, window, hashContract)
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
      window: WindowExecTransformer): Boolean = {
    sameExpressions(groupLimit.partitionSpec, window.partitionSpec) &&
    sameSortOrders(groupLimit.orderSpec, window.orderSpec) &&
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
