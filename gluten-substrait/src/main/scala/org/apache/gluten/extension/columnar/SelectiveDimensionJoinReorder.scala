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
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeSet, EqualNullSafe, EqualTo, Expression, In, InSet, PredicateHelper}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{BROADCAST, Filter, HintInfo, Join, JoinHint, LogicalPlan, Project, SubqueryAlias}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.datasources.LogicalRelation

/**
 * Conservatively rotates a Q11-like join chain so a selective dimension joins its dimension parent
 * before the fact table is probed. It changes `(fact JOIN dimension) JOIN filtered_dimension` into
 * `fact JOIN (dimension JOIN filtered_dimension)`.
 *
 * Presto-GPU Q11 materializes nation->supplier before probing partsupp, so the first hash join sees
 * ~32M rows instead of ~800M. Spark's default join order keeps partsupp on the probe side of an
 * unfiltered supplier join first.
 *
 * The rule is intentionally narrow: it only handles inner joins with no hints or broadcast-only
 * hints, keeps conditions on the same attribute sets, and requires the filtered dimension to carry
 * an obvious selective literal predicate plus small table statistics.
 */
case class SelectiveDimensionJoinReorder(spark: SparkSession)
  extends Rule[LogicalPlan]
  with PredicateHelper
  with Logging {

  private val maxFilteredDimensionSizeInBytes = BigInt(64L * 1024L * 1024L)
  private val maxFilteredDimensionRows = BigInt(100000)
  private val maxSelectiveInListValues = 16
  private val maxFilteredDimensionWrapperDepth = 4
  private val selectiveDimensionJoinReorderKey =
    GlutenConfig.SELECTIVE_DIMENSION_JOIN_REORDER_ENABLED.key
  private val fluxEnabledKey = "spark.gluten.mpp.enabled"
  private val singleTaskModeKey = "spark.gluten.sql.columnar.backend.velox.flux.singleTaskMode"

  registerPostCboPass()

  override def apply(plan: LogicalPlan): LogicalPlan = {
    registerPostCboPass()
    if (!enabledForSession || !plan.resolved) {
      return plan
    }

    plan.transformUp {
      case join: Join =>
        reorder(join).getOrElse(join)
    }
  }

  private def reorder(join: Join): Option[LogicalPlan] = {
    join match {
      case Join(left, filteredDimension, Inner, Some(dimensionFilterCondition), topHint)
          if hintAllowsReorder(topHint) =>
        extractFactDimensionJoin(left).flatMap {
          case FactDimensionJoin(fact, dimension, factDimensionCondition, leftHint) =>
            if (
              hintAllowsReorder(leftHint) &&
              factDimensionCondition.deterministic &&
              dimensionFilterCondition.deterministic &&
              isSelectiveFilteredDimension(filteredDimension) &&
              conditionStaysBetween(factDimensionCondition, fact.outputSet, dimension.outputSet) &&
              conditionStaysBetween(
                dimensionFilterCondition,
                dimension.outputSet,
                filteredDimension.outputSet)
            ) {
              val dimensionJoin =
                Join(
                  dimension,
                  filteredDimension,
                  Inner,
                  Some(dimensionFilterCondition),
                  JoinHint.NONE)
              val outerHint =
                dimensionChainBroadcastHint(fact, dimensionJoin).getOrElse(JoinHint.NONE)
              val reordered =
                Join(fact, dimensionJoin, Inner, Some(factDimensionCondition), outerHint)
              logInfo("Reordered selective dimension join before fact-table probe.")
              Some(preserveOutputIfNeeded(join, reordered))
            } else {
              None
            }
        }
      case _ =>
        None
    }
  }

  private case class FactDimensionJoin(
      fact: LogicalPlan,
      dimension: LogicalPlan,
      condition: Expression,
      hint: JoinHint)

  private def extractFactDimensionJoin(plan: LogicalPlan): Option[FactDimensionJoin] = {
    val unwrapped = plan match {
      case Project(projectList, child) if projectList.forall(_.isInstanceOf[Attribute]) => child
      case other => other
    }
    unwrapped match {
      case Join(fact, dimension, Inner, Some(condition), hint) =>
        Some(FactDimensionJoin(fact, dimension, condition, hint))
      case _ =>
        None
    }
  }

  private def dimensionChainBroadcastHint(
      fact: LogicalPlan,
      dimensionJoin: LogicalPlan): Option[JoinHint] = {
    val maxBroadcastBytes = BigInt(spark.sessionState.conf.autoBroadcastJoinThreshold)
    if (maxBroadcastBytes < 0) {
      return None
    }

    val factBytes = leafScanOrStatsBytes(fact)
    val dimensionChainBytes = leafScanOrStatsBytes(dimensionJoin)
    if (
      factBytes > 0 &&
      dimensionChainBytes > 0 &&
      factBytes > dimensionChainBytes &&
      dimensionChainBytes <= maxBroadcastBytes
    ) {
      logWarning(
        "SelectiveDimensionJoinReorder: broadcasting reordered dimension chain " +
          s"(factLeafBytes=$factBytes dimensionChainLeafBytes=$dimensionChainBytes " +
          s"autoBroadcastJoinThreshold=$maxBroadcastBytes)")
      Some(JoinHint(None, Some(HintInfo(strategy = Some(BROADCAST)))))
    } else {
      None
    }
  }

  private def leafScanOrStatsBytes(plan: LogicalPlan): BigInt = {
    var total = BigInt(0)
    plan.foreach {
      case lr: LogicalRelation =>
        total += BigInt(lr.relation.sizeInBytes)
      case leaf if leaf.children.isEmpty =>
        total += leaf.stats.sizeInBytes
      case _ =>
    }
    total
  }

  private def preserveOutputIfNeeded(original: Join, rewritten: LogicalPlan): LogicalPlan = {
    val rewrittenOutput = rewritten.outputSet
    if (original.outputSet.subsetOf(rewrittenOutput) && original.output != rewritten.output) {
      Project(original.output, rewritten)
    } else {
      rewritten
    }
  }

  private def hintAllowsReorder(hint: JoinHint): Boolean = {
    hint == JoinHint.NONE ||
    Seq(hint.leftHint, hint.rightHint).flatten.forall(_.strategy.contains(BROADCAST))
  }

  private def registerPostCboPass(): Unit = {
    if (!enabledForSession) {
      return
    }

    val experimental = spark.experimental
    experimental.synchronized {
      val current = experimental.extraOptimizations
      val existing =
        current.collectFirst { case r: SelectiveDimensionJoinReorder => r }.getOrElse(this)
      val without = current.filterNot(_.isInstanceOf[SelectiveDimensionJoinReorder])
      val hintIndex = without.indexWhere(_.isInstanceOf[FluxFactProbeBroadcastHint])
      val reordered =
        if (hintIndex >= 0) {
          without.patch(hintIndex, Seq(existing), 0)
        } else {
          without :+ existing
        }

      if (reordered != current) {
        experimental.extraOptimizations = reordered
        logDebug("SelectiveDimensionJoinReorder: registered post-CBO rewrite")
      }
    }
  }

  private def enabledForSession: Boolean = {
    val conf = spark.sessionState.conf
    if (conf.getConfString(singleTaskModeKey, "false").toBoolean) {
      false
    } else {
      conf.getAllConfs
        .get(selectiveDimensionJoinReorderKey)
        .map(_.toBoolean)
        .getOrElse(conf.getConfString(fluxEnabledKey, "false").toBoolean)
    }
  }

  private def isSelectiveFilteredDimension(plan: LogicalPlan): Boolean = {
    isSelectiveFilteredDimension(plan, 0)
  }

  private def isSelectiveFilteredDimension(plan: LogicalPlan, wrapperDepth: Int): Boolean = {
    plan match {
      case Filter(condition, child) if condition.deterministic =>
        hasSelectiveLiteralPredicate(condition, child.outputSet) && hasSmallDimensionStats(child)
      case Project(projectList, child)
          if wrapperDepth < maxFilteredDimensionWrapperDepth &&
            projectList.forall(_.deterministic) =>
        isSelectiveFilteredDimension(child, wrapperDepth + 1)
      case SubqueryAlias(_, child) if wrapperDepth < maxFilteredDimensionWrapperDepth =>
        isSelectiveFilteredDimension(child, wrapperDepth + 1)
      case _ =>
        false
    }
  }

  private def hasSmallDimensionStats(plan: LogicalPlan): Boolean = {
    val stats = plan.stats
    stats.rowCount.exists(_ <= maxFilteredDimensionRows) ||
    stats.sizeInBytes <= maxFilteredDimensionSizeInBytes
  }

  private def conditionStaysBetween(
      condition: Expression,
      left: AttributeSet,
      right: AttributeSet): Boolean = {
    val predicates = splitConjunctivePredicates(condition)
    predicates.nonEmpty &&
    predicates.forall(predicate => referencesOnly(predicate, left, right)) &&
    predicates.exists(predicate => referencesBoth(predicate, left, right))
  }

  private def hasSelectiveLiteralPredicate(
      condition: Expression,
      dimensionAttributes: AttributeSet): Boolean = {
    splitConjunctivePredicates(condition).exists {
      case EqualTo(left, right) =>
        literalComparison(left, right, dimensionAttributes)
      case EqualNullSafe(left, right) =>
        literalComparison(left, right, dimensionAttributes)
      case In(value, list) =>
        list.nonEmpty &&
        list.length <= maxSelectiveInListValues &&
        referencesOnly(value, dimensionAttributes) &&
        list.forall(_.foldable)
      case InSet(value, values) =>
        values.nonEmpty &&
        values.size <= maxSelectiveInListValues &&
        referencesOnly(value, dimensionAttributes)
      case _ =>
        false
    }
  }

  private def literalComparison(
      left: Expression,
      right: Expression,
      dimensionAttributes: AttributeSet): Boolean = {
    (referencesOnly(left, dimensionAttributes) && right.foldable) ||
    (referencesOnly(right, dimensionAttributes) && left.foldable)
  }

  private def referencesOnly(expression: Expression, allowed: AttributeSet): Boolean = {
    expression.references.nonEmpty && expression.references.forall(allowed.contains)
  }

  private def referencesOnly(
      expression: Expression,
      left: AttributeSet,
      right: AttributeSet): Boolean = {
    expression.references.forall(attribute => left.contains(attribute) || right.contains(attribute))
  }

  private def referencesBoth(
      expression: Expression,
      left: AttributeSet,
      right: AttributeSet): Boolean = {
    expression.references.exists(left.contains) && expression.references.exists(right.contains)
  }
}
