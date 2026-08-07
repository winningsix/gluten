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
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, EqualTo, Expression, Literal}
import org.apache.spark.sql.catalyst.expressions.SubqueryExpression
import org.apache.spark.sql.catalyst.expressions.aggregate.AggregateExpression
import org.apache.spark.sql.catalyst.plans.{Inner, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, BROADCAST, Filter, HintInfo, Join, JoinHint, LeafNode, LogicalPlan, Project, SubqueryAlias}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreePattern.JOIN
import org.apache.spark.sql.internal.SQLConf

import scala.util.Try

/**
 * Push an already-required selective dimension key into a correlated aggregate input.
 *
 * TPC-H Q17 has the following optimized shape (irrelevant projections omitted):
 * {{{
 *   (fact JOIN filtered_dimension)
 *     JOIN (AGGREGATE same_fact GROUP BY fact_key)
 *       ON aggregate_key = dimension_key
 * }}}
 * The aggregate computes groups for every fact key even though the upper inner joins guarantee that
 * only keys present in `filtered_dimension` can contribute to the result. This rule adds a
 * left-semi key filter below the aggregate:
 * {{{
 *   AGGREGATE (same_fact LEFT SEMI JOIN filtered_dimension) GROUP BY fact_key
 * }}}
 * This is an exact equivalence. Left-semi preserves fact multiplicity (including when the dimension
 * key is accidentally non-unique), and the original outer join remains responsible for the result
 * multiplicity. Null keys are rejected by both the original equality and the semi equality.
 *
 * The match is deliberately narrow to avoid regressions:
 *   - all three joins are inner equi-joins; both original joins and their strategy hints are kept
 *     unchanged, while the newly inserted semi join has its own dimension-side broadcast hint;
 *   - the aggregate has exactly one attribute grouping key and a deterministic, join-free input;
 *   - the aggregate fact and outer fact resolve to the same single leaf relation;
 *   - the dimension is a single-key Project/Filter/leaf chain with a deterministic literal equality
 *     predicate;
 *   - the filtered dimension is eligible for Spark broadcast, and its leaf scan is at most a
 *     configurable fraction of the aggregate input (default 25%).
 *
 * The relative cost guard is intentionally scale-independent. It avoids the fixed row-count gates
 * that change behavior between SF1000 and SF30000 while still bounding the duplicated scan. The
 * inserted semi join has an explicit right BROADCAST hint: allowing Spark to select SHJ here would
 * repartition the raw fact before its partial aggregate and can be worse than the original plan.
 * Disable with `spark.gluten.mpp.pushSelectiveDimensionFilterIntoAggregate=false`.
 */
case class PushSelectiveDimensionFilterIntoAggregate(spark: SparkSession)
  extends Rule[LogicalPlan]
  with Logging {

  private val enabledKey = "spark.gluten.mpp.pushSelectiveDimensionFilterIntoAggregate"
  private val maxScanRatioKey =
    "spark.gluten.mpp.pushSelectiveDimensionFilterIntoAggregate.maxDimensionScanRatio"
  private val minImprovementRatioKey =
    "spark.gluten.mpp.pushSelectiveDimensionFilterIntoAggregate.minImprovementRatio"
  private val maxBuildBytesKey = "spark.gluten.mpp.factProbeBroadcastHint.maxBuildBytes"
  private val fluxPartitionsKey = "spark.gluten.mpp.multiExecutor.numPartitions"

  registerPostCboPass()

  override def apply(plan: LogicalPlan): LogicalPlan = {
    registerPostCboPass()

    val conf = SQLConf.get
    if (
      !conf.getConfString("spark.gluten.mpp.enabled", "false").toBoolean ||
      !conf.getConfString(enabledKey, "true").toBoolean ||
      plan.isStreaming
    ) {
      return plan
    }

    val maxScanRatio = Try(conf.getConfString(maxScanRatioKey, "0.25").toDouble)
      .filter(value => java.lang.Double.isFinite(value) && value > 0.0)
      .getOrElse(0.0)
    if (maxScanRatio == 0.0) {
      return plan
    }

    plan.transformUpWithPruning(_.containsPattern(JOIN)) {
      case join: Join => rewriteJoin(join, maxScanRatio).getOrElse(join)
    }
  }

  private def rewriteJoin(join: Join, maxScanRatio: Double): Option[LogicalPlan] = {
    if (join.joinType != Inner || join.condition.isEmpty) {
      return None
    }

    rewriteOrientation(join, join.left, join.right, maxScanRatio)
      .orElse(rewriteOrientation(join, join.right, join.left, maxScanRatio))
  }

  /** `outerSide` contains fact JOIN dimension; `aggregateSide` contains the grouped same fact. */
  private def rewriteOrientation(
      topJoin: Join,
      outerSide: LogicalPlan,
      aggregateSide: LogicalPlan,
      maxScanRatio: Double): Option[LogicalPlan] = {
    val aggregate = unwrapAggregate(aggregateSide).getOrElse(return None)
    if (!safeAggregate(aggregate)) {
      return None
    }

    val groupingInput = aggregate.groupingExpressions.head.asInstanceOf[Attribute]
    val groupingOutput = aggregate.aggregateExpressions
      .collectFirst {
        case named if named.semanticEquals(groupingInput) => named.toAttribute
      }
      .getOrElse(return None)

    val outerDimensionKey =
      equiKeyFromOtherSide(topJoin.condition.get, outerSide, aggregateSide, groupingOutput)
        .getOrElse(return None)

    val candidate = findOuterFactDimensionJoin(outerSide, outerDimensionKey)
      .getOrElse(return None)
    val (outerFact, dimension, dimensionKey) = candidate

    val sameFact = sameSingleLeafRelation(outerFact, aggregate.child)
    val dimensionIsSafe = safeDimension(dimension, dimensionKey)
    val duplicationIsWorthwhile =
      sameFact && dimensionIsSafe &&
        worthDuplicatingDimension(dimension, aggregate, maxScanRatio)
    if (!sameFact || !dimensionIsSafe || !duplicationIsWorthwhile) {
      logInfo(
        s"PushSelectiveDimensionFilterIntoAggregate: rejected candidate " +
          s"(sameFact=$sameFact dimensionIsSafe=$dimensionIsSafe " +
          s"duplicationIsWorthwhile=$duplicationIsWorthwhile " +
          s"dimensionOutputBytes=${dimension.stats.sizeInBytes} " +
          s"dimensionScanBytes=${dimensionLeafBytes(dimension)} " +
          s"aggregateInputBytes=${aggregate.child.stats.sizeInBytes} " +
          s"aggregateScanBytes=${dimensionLeafBytes(aggregate.child)})")
      return None
    }

    // Do not reapply in the fixed-point optimizer batch.
    if (aggregate.child.exists(_.isInstanceOf[Join])) {
      return None
    }

    val filteredInput = Join(
      aggregate.child,
      dimension,
      LeftSemi,
      Some(EqualTo(groupingInput, dimensionKey)),
      JoinHint(None, Some(HintInfo(strategy = Some(BROADCAST)))))
    val rewrittenAggregate = aggregate.copy(child = filteredInput)
    val rewrittenAggregateSide = aggregateSide.transformDown {
      case node if node eq aggregate => rewrittenAggregate
    }

    logInfo(
      s"PushSelectiveDimensionFilterIntoAggregate: pushed ${dimensionKey.name} as a left-semi " +
        s"filter below aggregate grouping key ${groupingInput.name}; " +
        s"dimensionOutputBytes=${dimension.stats.sizeInBytes}, " +
        s"dimensionScanBytes=${dimensionLeafBytes(dimension)}, " +
        s"aggregateInputBytes=${dimensionLeafBytes(aggregate.child)}, " +
        s"autoBroadcastJoinThreshold=${SQLConf.get.autoBroadcastJoinThreshold}")

    if (aggregateSide eq topJoin.left) {
      Some(topJoin.copy(left = rewrittenAggregateSide))
    } else {
      Some(topJoin.copy(right = rewrittenAggregateSide))
    }
  }

  private def unwrapAggregate(plan: LogicalPlan): Option[Aggregate] = plan match {
    case aggregate: Aggregate => Some(aggregate)
    case Project(projectList, child)
        if projectList.forall(
          expression => expression.deterministic && !hasSubqueryExpression(expression)) =>
      unwrapAggregate(child)
    case Filter(condition, child) if condition.deterministic && !hasSubqueryExpression(condition) =>
      unwrapAggregate(child)
    case SubqueryAlias(_, child) => unwrapAggregate(child)
    case _ => None
  }

  private def safeAggregate(aggregate: Aggregate): Boolean = {
    aggregate.groupingExpressions match {
      case Seq(_: Attribute) =>
        aggregate.expressions.forall(_.deterministic) &&
        !aggregate.child.exists(_.isInstanceOf[Join]) &&
        singleLeaf(aggregate.child).isDefined
      case _ => false
    }
  }

  /** Find `dimensionKey = aggregateGroupingOutput` among otherwise arbitrary top predicates. */
  private def equiKeyFromOtherSide(
      condition: Expression,
      outerSide: LogicalPlan,
      aggregateSide: LogicalPlan,
      groupingOutput: Attribute): Option[Attribute] = {
    splitAnd(condition).collectFirst {
      case EqualTo(left: Attribute, right: Attribute)
          if left.semanticEquals(groupingOutput) &&
            aggregateSide.outputSet.contains(left) && outerSide.outputSet.contains(right) =>
        right
      case EqualTo(left: Attribute, right: Attribute)
          if right.semanticEquals(groupingOutput) &&
            aggregateSide.outputSet.contains(right) && outerSide.outputSet.contains(left) =>
        left
    }
  }

  /**
   * Walk output-preserving wrappers and locate the inner equi-join that produced `dimensionKey`.
   * Returns the opposite fact side and the complete filtered dimension subtree.
   */
  private def findOuterFactDimensionJoin(
      plan: LogicalPlan,
      dimensionKey: Attribute): Option[(LogicalPlan, LogicalPlan, Attribute)] = plan match {
    case Project(_, child) if child.outputSet.contains(dimensionKey) =>
      findOuterFactDimensionJoin(child, dimensionKey)
    case Filter(_, child) if child.outputSet.contains(dimensionKey) =>
      findOuterFactDimensionJoin(child, dimensionKey)
    case SubqueryAlias(_, child) if child.outputSet.contains(dimensionKey) =>
      findOuterFactDimensionJoin(child, dimensionKey)
    case Join(left, right, Inner, Some(condition), _) =>
      val orientation =
        if (right.outputSet.contains(dimensionKey)) Some((left, right))
        else if (left.outputSet.contains(dimensionKey)) Some((right, left))
        else None
      orientation.flatMap {
        case (fact, dimension) =>
          val hasFactDimensionEquality = splitAnd(condition).exists {
            case EqualTo(l: Attribute, r: Attribute) =>
              (fact.outputSet.contains(l) && r.semanticEquals(dimensionKey)) ||
              (fact.outputSet.contains(r) && l.semanticEquals(dimensionKey))
            case _ => false
          }
          if (hasFactDimensionEquality) Some((fact, dimension, dimensionKey)) else None
      }
    case _ => None
  }

  private def safeDimension(dimension: LogicalPlan, key: Attribute): Boolean = {
    dimension.output.size == 1 &&
    dimension.output.head.semanticEquals(key) &&
    singleLeaf(dimension).isDefined &&
    !dimension.exists {
      case _: LeafNode => false
      case project: Project =>
        !project.projectList.forall(
          expression => expression.deterministic && !hasSubqueryExpression(expression))
      case filter: Filter =>
        !filter.condition.deterministic || hasSubqueryExpression(filter.condition)
      case _: SubqueryAlias => false
      case _ => true
    } &&
    dimension.exists {
      case Filter(condition, _) => splitAnd(condition).exists(isLiteralEquality)
      case _ => false
    }
  }

  private def isLiteralEquality(expression: Expression): Boolean = expression match {
    case EqualTo(_: Attribute, _: Literal) => true
    case EqualTo(_: Literal, _: Attribute) => true
    case _ => false
  }

  private def worthDuplicatingDimension(
      dimension: LogicalPlan,
      aggregate: Aggregate,
      maxScanRatio: Double): Boolean = {
    val dimensionOutputBytes = dimension.stats.sizeInBytes
    val dimensionBytes = dimensionLeafBytes(dimension)
    val factBytes = dimensionLeafBytes(aggregate.child)
    val broadcastThreshold = BigInt(SQLConf.get.autoBroadcastJoinThreshold)
    val baseGuard =
      dimensionOutputBytes > 0 && dimensionOutputBytes < BigInt(Long.MaxValue) &&
        broadcastThreshold >= 0 &&
        dimensionBytes > 0 && dimensionBytes < BigInt(Long.MaxValue) &&
        factBytes > 0 && factBytes < BigInt(Long.MaxValue) &&
        BigDecimal(dimensionBytes) <= BigDecimal(factBytes) * BigDecimal(maxScanRatio)
    if (!baseGuard) {
      logInfo(
        s"PushSelectiveDimensionFilterIntoAggregate: rejected base cost guard " +
          s"(dimensionOutputBytes=$dimensionOutputBytes dimensionScanBytes=$dimensionBytes " +
          s"aggregateInputBytes=${aggregate.child.stats.sizeInBytes} " +
          s"aggregateScanBytes=$factBytes maxScanRatio=$maxScanRatio " +
          s"autoBroadcastJoinThreshold=$broadcastThreshold)")
      return false
    }

    // Preserve the original Spark behavior below the standard auto-broadcast threshold. The FLUX
    // extension is deliberately stricter: a larger duplicated build must fit an explicit memory
    // ceiling and save enough aggregate work to repay its scan, hash build and replicated network
    // costs.
    if (dimensionOutputBytes <= broadcastThreshold) {
      return true
    }

    val maxBuildBytes = configuredMaxBuildBytes(broadcastThreshold)
    val literalFilters = distinctLiteralEqualityCount(dimension)
    val aggregateInputBytes = aggregate.child.stats.sizeInBytes
    val peers = configuredFluxPartitions
    if (
      maxBuildBytes <= broadcastThreshold || dimensionOutputBytes > maxBuildBytes ||
      literalFilters < 2 || aggregateInputBytes <= 0 ||
      aggregateInputBytes >= BigInt(Long.MaxValue) || peers <= 1
    ) {
      return false
    }

    val factWorkBytes = aggregateInputBytes.min(factBytes)
    val aggregateMultiplier = 1 + aggregateBufferSlots(aggregate)
    val retainedFraction = BigDecimal("0.25").pow(math.min(literalFilters, 3))
    val baselineCost = BigDecimal(aggregateMultiplier) * BigDecimal(factWorkBytes)
    val rewrittenCost =
      BigDecimal(factWorkBytes) +
        BigDecimal(aggregateMultiplier) * retainedFraction * BigDecimal(factWorkBytes) +
        BigDecimal(dimensionBytes) +
        BigDecimal(2) * BigDecimal(dimensionOutputBytes) +
        BigDecimal(peers - 1) * BigDecimal(dimensionOutputBytes)
    val minImprovementRatio = configuredMinImprovementRatio
    val beneficial = baselineCost >= rewrittenCost * BigDecimal(minImprovementRatio)
    logInfo(
      s"PushSelectiveDimensionFilterIntoAggregate: " +
        s"${if (beneficial) "accepted" else "rejected"} cost-gated build above Spark threshold " +
        s"(dimensionOutputBytes=$dimensionOutputBytes dimensionScanBytes=$dimensionBytes " +
        s"aggregateInputBytes=$aggregateInputBytes factWorkBytes=$factWorkBytes peers=$peers " +
        s"literalFilters=$literalFilters retainedFraction=$retainedFraction " +
        s"aggregateMultiplier=$aggregateMultiplier baselineCost=$baselineCost " +
        s"rewrittenCost=$rewrittenCost minImprovementRatio=$minImprovementRatio " +
        s"maxBuildBytes=$maxBuildBytes autoBroadcastJoinThreshold=$broadcastThreshold)")
    beneficial
  }

  private def aggregateBufferSlots(aggregate: Aggregate): Int = {
    aggregate.aggregateExpressions
      .flatMap(_.collect { case expression: AggregateExpression => expression })
      .groupBy(_.resultId)
      .values
      .map(_.head.aggregateFunction.aggBufferAttributes.size)
      .sum
  }

  private def distinctLiteralEqualityCount(plan: LogicalPlan): Int = {
    plan
      .collect {
        case Filter(condition, _) => splitAnd(condition).flatMap(literalEqualityAttribute)
      }
      .flatten
      .foldLeft(Seq.empty[Attribute]) {
        case (attributes, attribute) if attributes.exists(_.semanticEquals(attribute)) =>
          attributes
        case (attributes, attribute) => attributes :+ attribute
      }
      .size
  }

  private def literalEqualityAttribute(expression: Expression): Option[Attribute] =
    expression match {
      case EqualTo(attribute: Attribute, _: Literal) => Some(attribute)
      case EqualTo(_: Literal, attribute: Attribute) => Some(attribute)
      case _ => None
    }

  private def configuredMaxBuildBytes(autoBroadcastBytes: BigInt): BigInt = {
    val raw = SQLConf.get.getConfString(maxBuildBytesKey, "auto").trim
    if (raw.isEmpty || raw.equalsIgnoreCase("auto")) {
      autoBroadcastBytes
    } else {
      parseBytes(raw).getOrElse(autoBroadcastBytes)
    }
  }

  private def configuredFluxPartitions: Int = {
    val conf = SQLConf.get
    val fallback = conf.numShufflePartitions
    Try(conf.getConfString(fluxPartitionsKey, fallback.toString).toInt).toOption
      .filter(_ > 0)
      .getOrElse(fallback)
  }

  private def configuredMinImprovementRatio: Double = {
    Try(SQLConf.get.getConfString(minImprovementRatioKey, "1.25").trim.toDouble).toOption
      .filter(value => java.lang.Double.isFinite(value) && value >= 1.0)
      .getOrElse(1.25)
  }

  private def parseBytes(raw: String): Option[BigInt] = {
    val normalized = raw.trim.toLowerCase(java.util.Locale.ROOT)
    val BytePattern = "^([0-9]+)([kmgt]?)(i?b)?$".r
    normalized match {
      case BytePattern(value, unit, _) =>
        val shift = unit match {
          case "" => 0
          case "k" => 10
          case "m" => 20
          case "g" => 30
          case "t" => 40
        }
        Some(BigInt(value) << shift)
      case _ => None
    }
  }

  private def dimensionLeafBytes(plan: LogicalPlan): BigInt = {
    singleLeaf(plan).map(_.stats.sizeInBytes).getOrElse(BigInt(Long.MaxValue))
  }

  private def sameSingleLeafRelation(left: LogicalPlan, right: LogicalPlan): Boolean = {
    (singleLeaf(left), singleLeaf(right)) match {
      case (Some(leftLeaf), Some(rightLeaf)) => leftLeaf.sameResult(rightLeaf)
      case _ => false
    }
  }

  private def singleLeaf(plan: LogicalPlan): Option[LeafNode] = {
    val leaves = plan.collect { case leaf: LeafNode => leaf }
    if (leaves.size == 1) Some(leaves.head) else None
  }

  private def splitAnd(expression: Expression): Seq[Expression] = expression match {
    case And(left, right) => splitAnd(left) ++ splitAnd(right)
    case other => Seq(other)
  }

  private def hasSubqueryExpression(expression: Expression): Boolean =
    expression.exists {
      case _: SubqueryExpression => true
      case _ => false
    }

  /**
   * Scalar-subquery decorrelation happens after Spark's normal injected optimizer-rule batch.
   * Register during rule construction so the first query in a fresh session is already covered,
   * then re-check from apply() in case another FLUX rule reordered the user optimizer list.
   */
  private def registerPostCboPass(): Unit = {
    val conf = spark.sessionState.conf
    if (
      !conf.getConfString("spark.gluten.mpp.enabled", "false").toBoolean ||
      !conf.getConfString(enabledKey, "true").toBoolean
    ) {
      return
    }
    val experimental = spark.experimental
    experimental.synchronized {
      if (
        !experimental.extraOptimizations.exists(
          _.isInstanceOf[PushSelectiveDimensionFilterIntoAggregate])
      ) {
        experimental.extraOptimizations = experimental.extraOptimizations :+ this
      }
    }
  }
}
