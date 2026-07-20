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
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Sum}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.datasources.LogicalRelation

import scala.collection.mutable

/**
 * Reassociate a selective two-fact join before an unfiltered wide dimension.
 *
 * Spark cannot cost this shape well when Parquet relations have byte statistics but no row or
 * column statistics. It may join a filtered bridge relation to a wide dimension first, carrying all
 * dimension strings through the subsequent fact join and another HASH exchange:
 *
 * {{{
 *   (wideDimension JOIN filteredBridge) JOIN filteredMeasureFact
 * }}}
 *
 * Inner equi joins are associative, so the strictly equivalent order is:
 *
 * {{{
 *   (filteredBridge JOIN filteredMeasureFact) JOIN wideDimension
 * }}}
 *
 * This rule deliberately does not push TopN or aggregation through a dimension join and therefore
 * does not assume primary-key, foreign-key, uniqueness, or referential-integrity metadata.
 *
 * The guard is intentionally narrow and independent of table/query names:
 *   - one resolved Aggregate over a connected cluster of unhinted inner equi joins;
 *   - all aggregate functions are SUM and all measure inputs come from one filtered fact leaf;
 *   - a second filtered bridge leaf connects that measure leaf to an unfiltered wide leaf;
 *   - the wide leaf contributes at least four grouping attributes and is already joined to the
 *     bridge before the measure leaf in the current tree;
 *   - plan statistics and projected widths prove that the candidate intermediate is at least 1.5x
 *     smaller than the current wide intermediate.
 *
 * The output is projected back to the original child's attributes, including exprIds and order.
 * Gated by `spark.gluten.sql.columnar.reorderFilteredFactBeforeWideDimension.enabled`; it defaults
 * to the MPP enablement state.
 */
case class ReorderFilteredFactBeforeWideDimension(spark: SparkSession)
  extends Rule[LogicalPlan]
  with PredicateHelper
  with Logging {

  private val enabledKey =
    "spark.gluten.sql.columnar.reorderFilteredFactBeforeWideDimension.enabled"
  private val mppEnabledKey = "spark.gluten.mpp.enabled"
  private val minLeafBytes = BigInt(1L << 30)
  private val minWideGroupingAttributes = 4
  private val minCostImprovement = 1.5
  private val maxPeripheralBroadcastBytes = BigInt(64L << 20)
  private val maxClusterItems = 8
  private val maxWrapperDepth = 5

  registerPostCboPass()

  override def apply(plan: LogicalPlan): LogicalPlan = {
    registerPostCboPass()
    if (!enabled || !plan.resolved || plan.isStreaming) {
      return plan
    }
    plan.transformUp {
      case aggregate: Aggregate => rewriteAggregate(aggregate).getOrElse(aggregate)
    }
  }

  private def enabled: Boolean = {
    val conf = spark.sessionState.conf
    conf
      .getConfString(enabledKey, conf.getConfString(mppEnabledKey, "false"))
      .toBoolean
  }

  private def rewriteAggregate(aggregate: Aggregate): Option[LogicalPlan] = {
    if (
      !aggregate.expressions.forall(_.deterministic) ||
      aggregate.expressions.exists(_.exists(_.isInstanceOf[SubqueryExpression]))
    ) {
      return None
    }
    val aggregateFunctions = aggregate.aggregateExpressions.flatMap(_.collect {
      case expression: AggregateExpression => expression
    })
    if (
      aggregateFunctions.isEmpty ||
      !aggregateFunctions.forall(_.aggregateFunction.isInstanceOf[Sum])
    ) {
      return None
    }

    val rawMeasureReferences = AttributeSet(
      aggregateFunctions.flatMap(_.aggregateFunction.references))
    val measureReferences =
      resolveProjectLineage(rawMeasureReferences, aggregate.child).getOrElse(return None)
    if (measureReferences.isEmpty) {
      return None
    }

    val cluster = extractInnerJoinCluster(aggregate.child).getOrElse(return None)
    val (items, predicates, hasUnsafeHints) =
      (cluster.items, cluster.predicates, cluster.hasUnsafeHints)
    if (
      hasUnsafeHints || items.size < 3 || items.size > maxClusterItems ||
      predicates.isEmpty || predicates.exists(!_.deterministic)
    ) {
      return None
    }

    val atomic = deduplicate(predicates.flatMap(splitConjunctivePredicates))
    if (!atomic.forall(isAttributeEquality)) {
      return None
    }

    val measureItems = items.filter(item => measureReferences.exists(item.outputSet.contains))
    if (measureItems.size != 1 || !measureReferences.subsetOf(measureItems.head.outputSet)) {
      return None
    }
    val measure = measureItems.head
    if (!hasSelectiveLiteralFilter(measure) || estimatedBytes(measure) < minLeafBytes) {
      return None
    }

    val groupingReferences = AttributeSet(aggregate.groupingExpressions.flatMap(_.references))
    val wideCandidates = items.filter {
      item =>
        val groupingCount = groupingReferences.count(item.outputSet.contains)
        item.ne(measure) && groupingCount >= minWideGroupingAttributes &&
        !hasSelectiveLiteralFilter(item) && estimatedBytes(item) >= minLeafBytes
    }

    val candidates = for {
      wide <- wideCandidates
      bridge <- items
      if bridge.ne(wide) && bridge.ne(measure)
      if hasSelectiveLiteralFilter(bridge) && estimatedBytes(bridge) >= minLeafBytes
      if edgePredicates(bridge.outputSet, measure.outputSet, atomic).nonEmpty
      if edgePredicates(bridge.outputSet, wide.outputSet, atomic).nonEmpty
      if joinedBeforeMeasure(aggregate.child, wide, bridge, measure)
      cost <- costImprovement(wide, bridge, measure)
      if cost >= minCostImprovement
    } yield Candidate(wide, bridge, measure, cost)

    val candidate = candidates.sortBy(value => -value.improvement).headOption.getOrElse(return None)
    rebuildCluster(items, atomic, cluster.peripheralBroadcasts, candidate) match {
      case Some(rebuilt) =>
        val restored = cluster.restore(rebuilt)
        if (!aggregate.child.outputSet.subsetOf(restored.outputSet)) {
          return None
        }
        val projected =
          if (aggregate.child.sameOutput(restored)) restored
          else Project(aggregate.child.output, restored)
        logWarning(
          "ReorderFilteredFactBeforeWideDimension: reassociated filtered fact pair before " +
            f"wide dimension (estimated intermediate improvement=${candidate.improvement}%.2fx, " +
            s"wideBytes=${estimatedBytes(candidate.wide)}, " +
            s"bridgeBytes=${estimatedBytes(candidate.bridge)}, " +
            s"measureBytes=${estimatedBytes(candidate.measure)})")
        Some(aggregate.withNewChildren(Seq(projected)))
      case _ => None
    }
  }

  private case class Candidate(
      wide: LogicalPlan,
      bridge: LogicalPlan,
      measure: LogicalPlan,
      improvement: Double)

  private case class ExtractedCluster(
      items: Seq[LogicalPlan],
      predicates: Seq[Expression],
      hasUnsafeHints: Boolean,
      peripheralBroadcasts: Seq[AttributeSet],
      restore: LogicalPlan => LogicalPlan)

  /** Flatten only attribute-only wrappers. Expressions remain in their original leaf branches. */
  private def extractInnerJoinCluster(root: LogicalPlan): Option[ExtractedCluster] = {
    // Keep deterministic expression Projects above the join cluster intact. Aggregate inputs may
    // be aliases defined there; flattening them would lose expressions, while refusing them would
    // miss common optimized plans.
    val (clusterRoot, restore) = peelTopWrappers(root, 0)
    val items = mutable.ArrayBuffer.empty[LogicalPlan]
    val predicates = mutable.ArrayBuffer.empty[Expression]
    val peripheralBroadcasts = mutable.ArrayBuffer.empty[AttributeSet]
    var hasUnsafeHints = false

    def collect(plan: LogicalPlan, depth: Int): Unit = plan match {
      case Project(projectList, child)
          if depth < maxWrapperDepth && isAttributeOnly(projectList) && containsInnerJoin(child) =>
        collect(child, depth + 1)
      case Filter(condition, child)
          if depth < maxWrapperDepth && condition.deterministic && containsInnerJoin(child) =>
        predicates += condition
        collect(child, depth + 1)
      case SubqueryAlias(_, child) if depth < maxWrapperDepth && containsInnerJoin(child) =>
        collect(child, depth + 1)
      case Join(left, right, Inner, condition, hint) =>
        val classified = classifyHint(left, right, hint)
        classified.peripheralBroadcast.foreach(peripheralBroadcasts += _)
        hasUnsafeHints = hasUnsafeHints || classified.unsafe
        condition.foreach(predicates += _)
        collect(left, depth)
        collect(right, depth)
      case other => items += other
    }

    collect(clusterRoot, 0)
    if (items.size >= 2) {
      Some(
        ExtractedCluster(
          items.toSeq,
          predicates.toSeq,
          hasUnsafeHints,
          peripheralBroadcasts.toSeq,
          restore))
    } else {
      None
    }
  }

  /**
   * Preserve only a broadcast hint on a tiny, join-free peripheral side. This admits the engine's
   * nation-style broadcast already present in the post-CBO plan, but still rejects hints on any
   * reordered large branch or hints whose strategy the rule cannot reproduce exactly.
   */
  private case class ClassifiedHint(peripheralBroadcast: Option[AttributeSet], unsafe: Boolean)

  private def classifyHint(
      left: LogicalPlan,
      right: LogicalPlan,
      hint: JoinHint): ClassifiedHint = {
    if (hint == JoinHint.NONE) {
      ClassifiedHint(None, unsafe = false)
    } else {
      val leftBroadcast = isBroadcast(hint.leftHint)
      val rightBroadcast = isBroadcast(hint.rightHint)
      if (leftBroadcast && hint.rightHint.isEmpty && isTinyJoinFreePeripheral(left)) {
        ClassifiedHint(Some(left.outputSet), unsafe = false)
      } else if (rightBroadcast && hint.leftHint.isEmpty && isTinyJoinFreePeripheral(right)) {
        ClassifiedHint(Some(right.outputSet), unsafe = false)
      } else {
        ClassifiedHint(None, unsafe = true)
      }
    }
  }

  private def isBroadcast(hint: Option[HintInfo]): Boolean =
    hint.exists(_.strategy.contains(BROADCAST))

  private def isTinyJoinFreePeripheral(plan: LogicalPlan): Boolean = {
    val bytes = realScanBytes(plan)
    !containsInnerJoin(plan) && bytes > 0 && bytes <= maxPeripheralBroadcastBytes
  }

  /** Resolve deterministic Project aliases back to their unique leaf attribute lineage. */
  private def resolveProjectLineage(
      references: AttributeSet,
      plan: LogicalPlan,
      depth: Int = 0): Option[AttributeSet] = plan match {
    case Project(projectList, child)
        if depth < maxWrapperDepth && projectList.forall(_.deterministic) =>
      val byExprId = projectList.map(expression => expression.exprId -> expression).toMap
      val expanded = references.toSeq.flatMap {
        reference => byExprId.get(reference.exprId).toSeq.flatMap(_.references)
      }
      if (expanded.isEmpty) None
      else resolveProjectLineage(AttributeSet(expanded), child, depth + 1)
    case SubqueryAlias(_, child) if depth < maxWrapperDepth =>
      resolveProjectLineage(references, child, depth + 1)
    case _ => Some(references)
  }

  private def peelTopWrappers(
      plan: LogicalPlan,
      depth: Int): (LogicalPlan, LogicalPlan => LogicalPlan) = plan match {
    case project @ Project(projectList, child)
        if depth < maxWrapperDepth && projectList.forall(_.deterministic) &&
          containsInnerJoin(child) =>
      val (cluster, restoreChild) = peelTopWrappers(child, depth + 1)
      (cluster, rewritten => project.withNewChildren(Seq(restoreChild(rewritten))))
    case alias @ SubqueryAlias(_, child) if depth < maxWrapperDepth && containsInnerJoin(child) =>
      val (cluster, restoreChild) = peelTopWrappers(child, depth + 1)
      (cluster, rewritten => alias.withNewChildren(Seq(restoreChild(rewritten))))
    case other => (other, identity)
  }

  private def containsInnerJoin(plan: LogicalPlan): Boolean =
    plan.exists {
      case Join(_, _, Inner, _, _) => true
      case _ => false
    }

  private def isAttributeOnly(expressions: Seq[NamedExpression]): Boolean =
    // Aliases must stay in place because they introduce new exprIds.
    expressions.forall(_.isInstanceOf[Attribute])

  private def isAttributeEquality(expression: Expression): Boolean = expression match {
    case EqualTo(_: Attribute, _: Attribute) => true
    case _ => false
  }

  private def edgePredicates(
      left: AttributeSet,
      right: AttributeSet,
      predicates: Seq[Expression]): Seq[Expression] =
    predicates.filter {
      predicate =>
        val refs = predicate.references
        refs.subsetOf(left ++ right) && refs.exists(left.contains) && refs.exists(right.contains)
    }

  private def joinedBeforeMeasure(
      root: LogicalPlan,
      wide: LogicalPlan,
      bridge: LogicalPlan,
      measure: LogicalPlan): Boolean =
    root.exists {
      case join: Join =>
        containsItem(join, wide) && containsItem(join, bridge) && !containsItem(join, measure)
      case _ => false
    }

  private def containsItem(root: LogicalPlan, item: LogicalPlan): Boolean =
    root.exists(plan => plan.eq(item) || plan.fastEquals(item))

  private def costImprovement(
      wide: LogicalPlan,
      bridge: LogicalPlan,
      measure: LogicalPlan): Option[Double] = {
    val wideWidth = projectedWidth(wide)
    val bridgeWidth = projectedWidth(bridge)
    val measureWidth = projectedWidth(measure)
    if (wideWidth <= 0 || bridgeWidth <= 0 || measureWidth <= 0) {
      return None
    }

    val currentRows = math.min(estimatedRows(wide), estimatedRows(bridge))
    val candidateRows = math.min(estimatedRows(bridge), estimatedRows(measure))
    val currentCost = currentRows * (wideWidth + bridgeWidth)
    val candidateCost = candidateRows * (bridgeWidth + measureWidth)
    if (currentCost <= 0.0 || candidateCost <= 0.0) None
    else Some(currentCost / candidateCost)
  }

  private def projectedWidth(plan: LogicalPlan): Double =
    plan.output.map(_.dataType.defaultSize.toDouble).sum.max(1.0)

  private def estimatedRows(plan: LogicalPlan): Double = {
    val base = plan.stats.rowCount
      .filter(_ > 0)
      .map(_.toDouble)
      .getOrElse(estimatedBytes(plan).toDouble / projectedWidth(plan))
    base * literalFilterSelectivity(plan)
  }

  private def estimatedBytes(plan: LogicalPlan): BigInt = {
    val bytes = plan.stats.sizeInBytes
    if (bytes > 0) bytes else realScanBytes(plan)
  }

  private def realScanBytes(plan: LogicalPlan): BigInt = {
    var total = BigInt(0)
    plan.foreach {
      case relation: LogicalRelation => total += BigInt(relation.relation.sizeInBytes)
      case leaf if leaf.children.isEmpty => total += leaf.stats.sizeInBytes
      case _ =>
    }
    if (total > 0) total else plan.stats.sizeInBytes
  }

  private def hasSelectiveLiteralFilter(plan: LogicalPlan): Boolean =
    literalPredicates(plan).nonEmpty

  private def literalFilterSelectivity(plan: LogicalPlan): Double = {
    val predicates = literalPredicates(plan)
    if (predicates.isEmpty) {
      1.0
    } else {
      predicates
        .map {
          case EqualTo(_: Attribute, _: Literal) | EqualTo(_: Literal, _: Attribute) => 0.1
          case _: In | _: InSet => 0.1
          case _ => 0.2
        }
        .product
        .max(0.001)
    }
  }

  private def literalPredicates(plan: LogicalPlan): Seq[Expression] = {
    val values = mutable.ArrayBuffer.empty[Expression]
    plan.foreach {
      case Filter(condition, _) if condition.deterministic =>
        splitConjunctivePredicates(condition).foreach {
          case predicate @ EqualTo(_: Attribute, _: Literal) => values += predicate
          case predicate @ EqualTo(_: Literal, _: Attribute) => values += predicate
          case predicate @ GreaterThan(_: Attribute, _: Literal) => values += predicate
          case predicate @ GreaterThan(_: Literal, _: Attribute) => values += predicate
          case predicate @ GreaterThanOrEqual(_: Attribute, _: Literal) => values += predicate
          case predicate @ GreaterThanOrEqual(_: Literal, _: Attribute) => values += predicate
          case predicate @ LessThan(_: Attribute, _: Literal) => values += predicate
          case predicate @ LessThan(_: Literal, _: Attribute) => values += predicate
          case predicate @ LessThanOrEqual(_: Attribute, _: Literal) => values += predicate
          case predicate @ LessThanOrEqual(_: Literal, _: Attribute) => values += predicate
          case predicate: In if predicate.list.forall(_.isInstanceOf[Literal]) =>
            values += predicate
          case predicate: InSet => values += predicate
          case _ =>
        }
      case _ =>
    }
    deduplicate(values.toSeq)
  }

  private def rebuildCluster(
      items: Seq[LogicalPlan],
      predicates: Seq[Expression],
      peripheralBroadcasts: Seq[AttributeSet],
      candidate: Candidate): Option[LogicalPlan] = {
    val used = mutable.ArrayBuffer.empty[Expression]

    def takeEdges(left: AttributeSet, right: AttributeSet): Seq[Expression] = {
      val edges = predicates.filter(
        predicate =>
          !used.exists(_.semanticEquals(predicate)) &&
            predicate.references.subsetOf(left ++ right) &&
            predicate.references.exists(left.contains) &&
            predicate.references.exists(right.contains))
      used ++= edges
      edges
    }

    val firstEdges = takeEdges(candidate.bridge.outputSet, candidate.measure.outputSet)
    if (firstEdges.isEmpty) {
      return None
    }
    var current: LogicalPlan =
      Join(candidate.bridge, candidate.measure, Inner, combine(firstEdges), JoinHint.NONE)

    val wideEdges = takeEdges(current.outputSet, candidate.wide.outputSet)
    if (wideEdges.isEmpty) {
      return None
    }
    current = Join(current, candidate.wide, Inner, combine(wideEdges), JoinHint.NONE)

    val remaining = mutable.ArrayBuffer(
      items.filterNot(
        item =>
          item.eq(candidate.bridge) || item.eq(candidate.measure) || item.eq(candidate.wide)): _*)
    while (remaining.nonEmpty) {
      val choices = remaining.flatMap {
        item =>
          val edges = predicates.filter(
            predicate =>
              !used.exists(_.semanticEquals(predicate)) &&
                predicate.references.subsetOf(current.outputSet ++ item.outputSet) &&
                predicate.references.exists(current.outputSet.contains) &&
                predicate.references.exists(item.outputSet.contains))
          if (edges.nonEmpty) Some((item, edges)) else None
      }
      if (choices.isEmpty) {
        return None
      }
      val (next, _) = choices.minBy { case (item, _) => estimatedBytes(item) }
      val edges = takeEdges(current.outputSet, next.outputSet)
      val hint =
        if (peripheralBroadcasts.contains(next.outputSet)) {
          JoinHint(None, Some(HintInfo(strategy = Some(BROADCAST))))
        } else {
          JoinHint.NONE
        }
      current = Join(current, next, Inner, combine(edges), hint)
      remaining -= next
    }

    val residual = predicates.filterNot(predicate => used.exists(_.semanticEquals(predicate)))
    if (residual.exists(predicate => !predicate.references.subsetOf(current.outputSet))) {
      None
    } else if (residual.nonEmpty) {
      Some(Filter(combine(residual).get, current))
    } else {
      Some(current)
    }
  }

  private def combine(predicates: Seq[Expression]): Option[Expression] =
    predicates.reduceOption(And)

  private def deduplicate(expressions: Seq[Expression]): Seq[Expression] =
    expressions.foldLeft(Seq.empty[Expression]) {
      case (result, expression) if result.exists(_.semanticEquals(expression)) => result
      case (result, expression) => result :+ expression
    }

  private def registerPostCboPass(): Unit = {
    if (!enabled) {
      return
    }
    val experimental = spark.experimental
    experimental.synchronized {
      val current = experimental.extraOptimizations
      val existingRewrite = current
        .collectFirst { case rule: ReorderFilteredFactBeforeWideDimension => rule }
        .getOrElse(this)
      val existingPush =
        current.collectFirst { case rule: PushSelectiveDimensionChainBeforeFact => rule }
      val existingHint = current
        .collectFirst { case rule: MppFactProbeBroadcastHint => rule }
        .getOrElse(MppFactProbeBroadcastHint(spark))
      val base = current.filterNot(
        rule =>
          rule.isInstanceOf[ReorderFilteredFactBeforeWideDimension] ||
            rule.isInstanceOf[PushSelectiveDimensionChainBeforeFact] ||
            rule.isInstanceOf[MppFactProbeBroadcastHint])
      val reordered = base ++ Seq(existingRewrite) ++ existingPush.toSeq ++ Seq(existingHint)
      if (reordered != current) {
        experimental.extraOptimizations = reordered
        logDebug(
          "ReorderFilteredFactBeforeWideDimension: registered post-CBO before dimension-chain " +
            "rewrite and MppFactProbeBroadcastHint")
      }
    }
  }
}
