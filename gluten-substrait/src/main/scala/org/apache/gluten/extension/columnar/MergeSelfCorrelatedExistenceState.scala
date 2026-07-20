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
import org.apache.spark.sql.catalyst.analysis.MultiInstanceRelation
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, AttributeReference, AttributeSet, EqualTo, Exists, Expression, ExprId, If, Literal, NamedExpression, Not, PredicateHelper}
import org.apache.spark.sql.catalyst.expressions.aggregate.{Max, Min}
import org.apache.spark.sql.catalyst.plans.{ExistenceJoin, Inner, LeftAnti, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, BROADCAST, Filter, Join, JoinHint, LeafNode, LogicalPlan, Project, SubqueryAlias}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.types.{BooleanType, DoubleType, FloatType}

/**
 * Install the paired-existence rules before optimization starts.
 *
 * Spark materializes LeftSemi / LeftAnti in RewriteSubquery, after injected operator-optimization
 * rules have already run. User Provided Optimizers is the first extensible batch that sees those
 * joins. Registration used to happen from the rules' own apply methods, which was too late for the
 * first query in a fresh session. This post-hoc analyzer rule performs registration only; it never
 * transforms the analyzer plan.
 */
case class RegisterMppExistencePostSubqueryRules(spark: SparkSession) extends Rule[LogicalPlan] {
  override def apply(plan: LogicalPlan): LogicalPlan = {
    MergeSelfCorrelatedExistenceState.registerPostSubqueryRules(spark)
    plan
  }
}

/**
 * Merge a paired self-correlated EXISTS / NOT EXISTS into one state aggregate.
 *
 * The accepted shape is intentionally narrow:
 *
 * {{ left ANTI JOIN delayedRhs ON key = key AND rhsValue != leftValue left SEMI JOIN allRhs ON key
 * \= key AND rhsValue != leftValue candidateRows }}
 *
 * All three inputs must be provably backed by the same leaf relation, with the correlation key and
 * compared value mapped to the same source ordinals. The delayed RHS must have a deterministic
 * source-only predicate which is also an explicit conjunct on the candidate rows. The last guard is
 * essential: it proves that each candidate's value is a member of the delayed set, including the
 * empty-set and SQL NULL cases.
 *
 * For an admitted pair, one aggregate computes min/max of all values and conditional min/max of
 * delayed values. `allMin != allMax` proves that another value exists. `delayedMin == delayedMax`
 * together with `candidateValue == delayedMin` proves that no other delayed value exists. MIN/MAX
 * ignore NULL, which matches the UNKNOWN result of SQL `!=` for a NULL operand. Duplicate RHS rows
 * do not affect either existence predicate.
 */
case class MergeSelfCorrelatedExistenceState(spark: SparkSession)
  extends Rule[LogicalPlan]
  with PredicateHelper
  with Logging {

  import MergeSelfCorrelatedExistenceState._

  private val glutenConfig = new GlutenConfig(spark.sessionState.conf)

  private case class SourceView(
      plan: LogicalPlan,
      source: LeafNode,
      outputOrdinals: Map[ExprId, Int],
      sourcePredicates: Seq[Expression]) {

    def sourceOrdinal(attribute: Attribute): Option[Int] = {
      outputOrdinals
        .get(attribute.exprId)
        .orElse {
          plan.output.zipWithIndex.collectFirst {
            case (output, index) if output.semanticEquals(attribute) =>
              outputOrdinals.getOrElse(output.exprId, index)
          }
        }
        .orElse {
          source.output.zipWithIndex.collectFirst {
            case (output, index) if output.semanticEquals(attribute) => index
          }
        }
    }
  }

  private case class JoinKeys(
      leftEquality: Seq[Attribute],
      rightEquality: Seq[Attribute],
      leftNotEqual: Attribute,
      rightNotEqual: Attribute)

  private case class PairProof(
      candidate: LogicalPlan,
      allRhs: LogicalPlan,
      delayedRhs: LogicalPlan,
      semiCondition: Expression,
      antiCondition: Expression,
      allView: SourceView,
      semiKeys: JoinKeys,
      candidateKeyOrdinals: Seq[Int],
      candidateValueOrdinal: Int,
      sourceValue: Attribute,
      delayedPredicate: Expression,
      sourceKeys: Seq[Attribute])

  /**
   * A safe inner-join/unary context surrounding the original paired existence spine. `candidate` is
   * the same context moved below the existence predicates, while `output` is the exact output that
   * the context exposed before the move. Keeping both lets the final Project preserve exprIds,
   * ordering, aliases, and bag semantics while hidden correlation attributes remain available to
   * the two existence joins.
   */
  private case class CandidateContext(
      proof: PairProof,
      candidate: LogicalPlan,
      output: Seq[Attribute],
      absorbedJoins: Seq[AbsorbedJoin])

  private case class AbsorbedJoin(
      side: LogicalPlan,
      condition: Expression,
      hint: JoinHint,
      pairOnLeft: Boolean,
      pairOutput: Seq[Attribute])

  private case class CandidateFirstCost(
      pairedCostBytes: Double,
      candidateCostBytes: Double,
      improvementRatio: Double,
      stateReadBytes: BigInt,
      rhsReadBytes: BigInt,
      candidateReadBytes: BigInt,
      absorbedSideScanBytes: BigInt,
      literalEqualityFilters: Int)

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!glutenConfig.getConf(GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED)) {
      return plan
    }

    if (!plan.resolved) {
      return plan
    }

    val candidateFirst = glutenConfig.candidateFirstExistenceMode match {
      case "force" => rewriteCandidateFirst(plan, requireCostGuard = false)
      case "auto" => rewriteCandidateFirst(plan, requireCostGuard = true)
      case "off" => plan
    }

    candidateFirst.transformUp {
      case anti @ Join(
            semi @ Join(candidate, allRhs, LeftSemi, Some(semiCondition), JoinHint.NONE),
            delayedRhs,
            LeftAnti,
            Some(antiCondition),
            JoinHint.NONE) =>
        rewritePair(anti, semi, candidate, allRhs, delayedRhs, semiCondition, antiCondition)
          .getOrElse(anti)
    }
  }

  /**
   * Move the deterministic inner-join context around a proven paired existence spine into its
   * candidate side, then evaluate both correlated predicates as ExistenceJoin probes. This is the
   * candidate-first shape needed by Q21: selective supplier/nation and orders joins reduce l1
   * before either full lineitem probe is built.
   *
   * The extraction is deliberately narrow. It only crosses deterministic filters, pass-through
   * projects/aliases, and ordinary inner joins. The original pair is admitted by exactly the same
   * same-source, key/value ordinal, delayed-predicate implication, type, streaming, hint, and cost
   * guards as the paired-state fallback.
   */
  private def rewriteCandidateFirst(plan: LogicalPlan, requireCostGuard: Boolean): LogicalPlan = {
    plan.transformDown {
      case current =>
        extractCandidateContext(current)
          .flatMap {
            context =>
              if (!requireCostGuard) {
                logWarning(
                  "MergeSelfCorrelatedExistenceState: selected forced candidate-first shape " +
                    s"absorbedInnerJoins=${context.absorbedJoins.size} " +
                    s"sourceBytes=${context.proof.allView.source.stats.sizeInBytes}")
                Some(buildCandidateFirst(context))
              } else {
                candidateFirstCost(context).flatMap {
                  cost =>
                    val minRatio = glutenConfig.getConf(
                      GlutenConfig.CANDIDATE_FIRST_EXISTENCE_MIN_COST_IMPROVEMENT_RATIO)
                    if (cost.improvementRatio >= minRatio) {
                      logInfo(
                        "MergeSelfCorrelatedExistenceState: selected cost-gated " +
                          "candidate-first shape " +
                          s"absorbedInnerJoins=${context.absorbedJoins.size} " +
                          f"estimatedImprovement=${cost.improvementRatio}%.3f " +
                          f"pairedCostBytes=${cost.pairedCostBytes}%.0f " +
                          f"candidateCostBytes=${cost.candidateCostBytes}%.0f " +
                          s"stateReadBytes=${cost.stateReadBytes} " +
                          s"rhsReadBytes=${cost.rhsReadBytes} " +
                          s"candidateReadBytes=${cost.candidateReadBytes} " +
                          s"absorbedSideScanBytes=${cost.absorbedSideScanBytes} " +
                          s"literalEqualityFilters=${cost.literalEqualityFilters}")
                      Some(buildCandidateFirst(context))
                    } else {
                      logWarning(
                        "MergeSelfCorrelatedExistenceState: retained paired-state shape; " +
                          f"candidate-first estimated improvement " +
                          f"${cost.improvementRatio}%.3f is below $minRatio%.3f")
                      None
                    }
                }
              }
          }
          .getOrElse(current)
    }
  }

  private def extractCandidateContext(plan: LogicalPlan): Option[CandidateContext] = {
    plan match {
      case anti @ Join(
            semi @ Join(candidate, allRhs, LeftSemi, Some(semiCondition), JoinHint.NONE),
            delayedRhs,
            LeftAnti,
            Some(antiCondition),
            JoinHint.NONE) =>
        analyzePair(anti, semi, candidate, allRhs, delayedRhs, semiCondition, antiCondition)
          .map(proof => CandidateContext(proof, candidate, candidate.output, Seq.empty))

      case project @ Project(projectList, child) if !project.isStreaming =>
        extractCandidateContext(child).flatMap {
          context =>
            val passThrough = projectList.forall {
              case _: Attribute => true
              case Alias(_: Attribute, _) => true
              case _ => false
            }
            if (!passThrough) {
              None
            } else {
              // Preserve every hidden candidate attribute required by the original pair. A final
              // Project below restores the exact output of the pre-rewrite context.
              val visible = projectList.map(_.toAttribute)
              val hidden = context.candidate.output.filterNot {
                attribute => visible.exists(_.semanticEquals(attribute))
              }
              Some(
                context.copy(
                  candidate = Project(projectList ++ hidden, context.candidate),
                  output = project.output))
            }
        }

      case filter @ Filter(condition, child) if !filter.isStreaming && condition.deterministic =>
        extractCandidateContext(child).flatMap {
          context =>
            if (condition.references.subsetOf(AttributeSet(context.output))) {
              Some(
                context
                  .copy(candidate = Filter(condition, context.candidate), output = filter.output))
            } else {
              None
            }
        }

      case alias @ SubqueryAlias(_, child) if !alias.isStreaming =>
        extractCandidateContext(child).map {
          context =>
            context.copy(candidate = alias.copy(child = context.candidate), output = alias.output)
        }

      case join @ Join(left, right, Inner, Some(condition), hint)
          if !join.isStreaming && condition.deterministic =>
        val leftContext = extractCandidateContext(left)
        val rightContext = extractCandidateContext(right)
        (leftContext, rightContext) match {
          case (Some(context), None) if !containsPairedSpine(right) =>
            Some(
              context.copy(
                candidate = Join(context.candidate, right, Inner, Some(condition), hint),
                output = join.output,
                absorbedJoins = context.absorbedJoins :+ AbsorbedJoin(
                  right,
                  condition,
                  hint,
                  pairOnLeft = true,
                  context.output)
              ))
          case (None, Some(context)) if !containsPairedSpine(left) =>
            Some(
              context.copy(
                candidate = Join(left, context.candidate, Inner, Some(condition), hint),
                output = join.output,
                absorbedJoins = context.absorbedJoins :+ AbsorbedJoin(
                  left,
                  condition,
                  hint,
                  pairOnLeft = false,
                  context.output)
              ))
          case _ => None
        }

      case _ => None
    }
  }

  private def containsPairedSpine(plan: LogicalPlan): Boolean = {
    plan.exists {
      case Join(
            Join(_, _, LeftSemi, Some(_), JoinHint.NONE),
            _,
            LeftAnti,
            Some(_),
            JoinHint.NONE) =>
        true
      case _ => false
    }
  }

  /**
   * Compare the two existence implementations using only statistics that remain meaningful when
   * Spark has no column NDV statistics. In that situation Spark's generic inner-join estimate is a
   * Cartesian product (Q21 reaches 1e59 bytes), so using the final candidate `sizeInBytes` would
   * make a CBO decision actively worse than no decision.
   *
   * The model instead prices the physical work that differs between the alternatives:
   *
   *   - paired-state scans the required source columns, builds a large hash aggregate, partitions
   *     its state, and probes that state (five byte passes in total);
   *   - candidate-first scans and probes the all/delayed RHS streams, then builds the filtered
   *     candidate twice;
   *   - scans of absorbed sides are common work, but remain in both totals so a large surrounding
   *     join chain lowers the confidence/improvement ratio.
   *
   * Auto mode additionally requires a safe, costable join chain: at least two pure attribute
   * equi-joins, unary deterministic side plans, no hint on the pair side, only an optional
   * BROADCAST hint on the absorbed side, finite leaf statistics, and at least two literal equality
   * filters. Every absorbed side must be smaller than the self-correlated source and at least one
   * must be at most one eighth of it. These are confidence gates for the cost estimate, not query,
   * table, or SQL-text checks.
   */
  private def candidateFirstCost(context: CandidateContext): Option[CandidateFirstCost] = {
    if (context.absorbedJoins.size < 2) {
      return None
    }
    if (!context.absorbedJoins.forall(safeAbsorbedJoin)) {
      return None
    }

    val sourceBytes = finiteBytes(context.proof.allView.source.stats.sizeInBytes)
      .getOrElse(return None)
    val sideScanBytes = context.absorbedJoins.map(absorbedSideScanBytes)
    if (sideScanBytes.exists(_.isEmpty)) {
      return None
    }
    val resolvedSideBytes = sideScanBytes.flatten
    // Without NDV/uniqueness statistics, require every absorbed side to be smaller than the
    // self-correlated fact and require at least one unambiguous dimension-sized entrance.
    if (
      resolvedSideBytes.exists(_ >= sourceBytes) ||
      !resolvedSideBytes.exists(bytes => bytes * 8 <= sourceBytes)
    ) {
      return None
    }

    // Count selective sides, not raw conjuncts: duplicated predicates on one side must not create
    // an artificial 1/16 estimate.
    val literalFilters = context.absorbedJoins.count(join => literalEqualityFilters(join.side) > 0)
    if (literalFilters < 2) {
      return None
    }

    val candidateView = sourceView(context.proof.candidate).getOrElse(return None)
    val allReadBytes = estimatedSourceReadBytes(Seq(context.proof.allView)).getOrElse(return None)
    val delayedView = sourceView(context.proof.delayedRhs).getOrElse(return None)
    val delayedReadBytes = estimatedSourceReadBytes(Seq(delayedView)).getOrElse(return None)
    val stateReadBytes = estimatedSourceReadBytes(Seq(context.proof.allView, delayedView))
      .getOrElse(return None)
    val candidateReadBytes = estimatedSourceReadBytes(Seq(candidateView)).getOrElse(return None)
    val totalSideBytes = resolvedSideBytes.sum

    // With no histogram/NDV, assign only a conservative 0.25 reduction to a deterministic literal
    // equality. Requiring two such predicates caps the structural fallback at 1/16 before auto can
    // be selected. Cap at three predicates so a long filter list cannot manufacture an
    // arbitrarily small estimate.
    val estimatedCandidateSelectivity = math.pow(0.25, math.min(literalFilters, 3))
    val pairedCost = totalSideBytes.toDouble + 5.0 * stateReadBytes.toDouble
    val rhsReadBytes = allReadBytes + delayedReadBytes
    val baseCandidateWidth = context.proof.candidate.output.map(_.dataType.defaultSize.toLong).sum
    val movedCandidateWidth = context.candidate.output.map(_.dataType.defaultSize.toLong).sum
    if (baseCandidateWidth <= 0 || movedCandidateWidth <= 0) {
      return None
    }
    // The moved candidate also carries dimension payload (for Q21, s_name) through both builds.
    // Inflate the source-read proxy by the logical output-width ratio so a wide dimension chain is
    // not treated as a free selective build.
    val candidateWidthAmplification =
      math.max(1.0, movedCandidateWidth.toDouble / baseCandidateWidth.toDouble)
    val candidateCost = totalSideBytes.toDouble + 2.0 * rhsReadBytes.toDouble +
      2.0 * candidateReadBytes.toDouble * estimatedCandidateSelectivity *
      candidateWidthAmplification
    if (
      !java.lang.Double.isFinite(pairedCost) ||
      !java.lang.Double.isFinite(candidateCost) || candidateCost <= 0.0
    ) {
      return None
    }
    Some(
      CandidateFirstCost(
        pairedCost,
        candidateCost,
        pairedCost / candidateCost,
        stateReadBytes,
        rhsReadBytes,
        candidateReadBytes,
        totalSideBytes,
        literalFilters))
  }

  private def safeAbsorbedJoin(join: AbsorbedJoin): Boolean = {
    if (join.side.isStreaming || !safeAbsorbedSide(join.side)) {
      return false
    }
    val pairHint = if (join.pairOnLeft) join.hint.leftHint else join.hint.rightHint
    val sideHint = if (join.pairOnLeft) join.hint.rightHint else join.hint.leftHint
    if (
      pairHint.exists(_.strategy.isDefined) ||
      sideHint.exists(_.strategy.exists(_ != BROADCAST))
    ) {
      return false
    }
    val predicates = splitConjunctivePredicates(join.condition)
    predicates.nonEmpty && predicates.forall {
      case EqualTo(left: Attribute, right: Attribute) =>
        (isOutput(left, join.pairOutput) && isOutput(right, join.side.output)) ||
        (isOutput(right, join.pairOutput) && isOutput(left, join.side.output))
      case _ => false
    }
  }

  private def safeAbsorbedSide(plan: LogicalPlan): Boolean = {
    !plan.exists {
      case leaf: LeafNode => leaf.output.isEmpty || leaf.isStreaming
      case Project(projectList, _) => projectList.exists(expression => !expression.deterministic)
      case Filter(condition, _) => !condition.deterministic
      case _: SubqueryAlias => false
      case _ => true
    }
  }

  private def absorbedSideScanBytes(join: AbsorbedJoin): Option[BigInt] = {
    val leaves = join.side.collect { case leaf: LeafNode => leaf }
    if (leaves.isEmpty) {
      return None
    }
    val sizes = leaves.map(leaf => finiteBytes(leaf.stats.sizeInBytes))
    if (sizes.exists(_.isEmpty)) None else Some(sizes.flatten.sum)
  }

  private def literalEqualityFilters(plan: LogicalPlan): Int = {
    plan.collect {
      case Filter(condition, _) =>
        splitConjunctivePredicates(condition).count {
          case EqualTo(_: Attribute, literal: Literal) => literal.value != null
          case EqualTo(literal: Literal, _: Attribute) => literal.value != null
          case _ => false
        }
    }.sum
  }

  private def estimatedSourceReadBytes(views: Seq[SourceView]): Option[BigInt] = {
    val source = views.headOption.map(_.source).getOrElse(return None)
    if (!views.forall(view => sameSourceLeaf(source, view.source))) {
      return None
    }
    val sourceBytes = finiteBytes(source.stats.sizeInBytes).getOrElse(return None)
    val outputOrdinals = views.flatMap(view => view.plan.output.flatMap(view.sourceOrdinal))
    val predicateOrdinals = views.flatMap {
      view => view.sourcePredicates.flatMap(_.references).flatMap(view.sourceOrdinal)
    }
    val requiredOrdinals = (outputOrdinals ++ predicateOrdinals).distinct
    if (requiredOrdinals.isEmpty) {
      return None
    }
    val totalWidth = source.output.map(_.dataType.defaultSize.toLong).sum
    val requiredWidth = requiredOrdinals.map(source.output(_).dataType.defaultSize.toLong).sum
    if (totalWidth <= 0 || requiredWidth <= 0) {
      return None
    }
    Some((sourceBytes * requiredWidth / totalWidth).max(BigInt(1)))
  }

  private def sameSourceLeaf(left: LeafNode, right: LeafNode): Boolean = {
    val leftView = SourceView(left, left, left.output.map(_.exprId).zipWithIndex.toMap, Seq.empty)
    val rightView =
      SourceView(right, right, right.output.map(_.exprId).zipWithIndex.toMap, Seq.empty)
    sameSource(leftView, rightView)
  }

  private def finiteBytes(bytes: BigInt): Option[BigInt] = {
    if (bytes > 0 && bytes != BigInt(Long.MaxValue)) Some(bytes) else None
  }

  private def isOutput(attribute: Attribute, output: Seq[Attribute]): Boolean = {
    output.exists(_.semanticEquals(attribute))
  }

  private def buildCandidateFirst(context: CandidateContext): LogicalPlan = {
    val proof = context.proof
    val allExists = AttributeReference(
      s"${CandidateFirstExistenceAttributePrefix}all",
      BooleanType,
      nullable = false)()
    val allProbe = Join(
      context.candidate,
      proof.allRhs,
      ExistenceJoin(allExists),
      Some(proof.semiCondition),
      JoinHint.NONE)
    val candidatesWithAllMatch =
      Project(context.candidate.output, Filter(allExists, allProbe))

    val delayedExists = AttributeReference(
      s"${CandidateFirstExistenceAttributePrefix}delayed",
      BooleanType,
      nullable = false)()
    val delayedProbe = Join(
      candidatesWithAllMatch,
      proof.delayedRhs,
      ExistenceJoin(delayedExists),
      Some(proof.antiCondition),
      JoinHint.NONE)
    val candidatesWithoutDelayedMatch =
      Project(context.candidate.output, Filter(Not(delayedExists), delayedProbe))

    // Restore the exact output contract of the original surrounding inner-join context. Hidden
    // source attributes kept through intermediate Projects do not escape this boundary.
    Project(context.output, candidatesWithoutDelayedMatch)
  }

  private def analyzePair(
      originalAnti: Join,
      originalSemi: Join,
      candidate: LogicalPlan,
      allRhs: LogicalPlan,
      delayedRhs: LogicalPlan,
      semiCondition: Expression,
      antiCondition: Expression): Option[PairProof] = {
    if (
      originalAnti.isStreaming || originalSemi.isStreaming ||
      !semiCondition.deterministic || !antiCondition.deterministic
    ) {
      return None
    }

    val candidateView = sourceView(candidate).getOrElse(return None)
    val allView = sourceView(allRhs).getOrElse(return None)
    val delayedView = sourceView(delayedRhs).getOrElse(return None)
    if (
      candidateView.source.isStreaming || allView.source.isStreaming ||
      delayedView.source.isStreaming ||
      !sameSource(candidateView, allView) || !sameSource(candidateView, delayedView)
    ) {
      return None
    }

    // Start with the strict, high-value case: the positive EXISTS ranges over the entire source.
    // This is also what makes a single source scan sufficient.
    if (allView.sourcePredicates.nonEmpty || delayedView.sourcePredicates.isEmpty) {
      return None
    }

    val semiKeys = parseJoinKeys(semiCondition, candidate, allRhs).getOrElse(return None)
    val antiKeys = parseJoinKeys(antiCondition, originalSemi, delayedRhs).getOrElse(return None)

    val candidateKeyOrdinals =
      semiKeys.leftEquality.map(candidateView.sourceOrdinal).sequence.getOrElse(return None)
    val allKeyOrdinals =
      semiKeys.rightEquality.map(allView.sourceOrdinal).sequence.getOrElse(return None)
    val antiCandidateKeyOrdinals =
      antiKeys.leftEquality.map(candidateView.sourceOrdinal).sequence.getOrElse(return None)
    val delayedKeyOrdinals =
      antiKeys.rightEquality.map(delayedView.sourceOrdinal).sequence.getOrElse(return None)
    val candidateValueOrdinal =
      candidateView.sourceOrdinal(semiKeys.leftNotEqual).getOrElse(return None)
    val allValueOrdinal = allView.sourceOrdinal(semiKeys.rightNotEqual).getOrElse(return None)
    val antiCandidateValueOrdinal =
      candidateView.sourceOrdinal(antiKeys.leftNotEqual).getOrElse(return None)
    val delayedValueOrdinal =
      delayedView.sourceOrdinal(antiKeys.rightNotEqual).getOrElse(return None)

    if (
      candidateKeyOrdinals.isEmpty || candidateKeyOrdinals.distinct.length !=
        candidateKeyOrdinals.length ||
        candidateKeyOrdinals != allKeyOrdinals ||
        candidateKeyOrdinals != antiCandidateKeyOrdinals ||
        candidateKeyOrdinals != delayedKeyOrdinals ||
        candidateValueOrdinal != allValueOrdinal ||
        candidateValueOrdinal != antiCandidateValueOrdinal ||
        candidateValueOrdinal != delayedValueOrdinal ||
        candidateKeyOrdinals.contains(candidateValueOrdinal)
    ) {
      return None
    }

    val sourceValue = allView.source.output(candidateValueOrdinal)
    if (!supportsMinMax(sourceValue)) {
      return None
    }

    val delayedPredicatesOnCandidate = delayedView.sourcePredicates.map(
      remapSourceExpression(_, delayedView.source, candidateView.source))
    if (delayedPredicatesOnCandidate.exists(_.isEmpty)) {
      return None
    }
    val candidatePredicates = candidateView.sourcePredicates
    val candidateImpliesDelayed = delayedPredicatesOnCandidate.flatten.forall {
      required => candidatePredicates.exists(_.semanticEquals(required))
    }
    if (!candidateImpliesDelayed) {
      return None
    }

    if (!passesCostGuard(allView, delayedView)) {
      return None
    }

    val delayedPredicatesOnAll =
      delayedView.sourcePredicates.map(remapSourceExpression(_, delayedView.source, allView.source))
    if (delayedPredicatesOnAll.exists(_.isEmpty)) {
      return None
    }
    val delayedPredicate = delayedPredicatesOnAll.flatten.reduce(And)
    if (!delayedPredicate.deterministic) {
      return None
    }

    val sourceKeys = candidateKeyOrdinals.map(allView.source.output)
    Some(
      PairProof(
        candidate,
        allRhs,
        delayedRhs,
        semiCondition,
        antiCondition,
        allView,
        semiKeys,
        candidateKeyOrdinals,
        candidateValueOrdinal,
        sourceValue,
        delayedPredicate,
        sourceKeys
      ))
  }

  private def rewritePair(
      originalAnti: Join,
      originalSemi: Join,
      candidate: LogicalPlan,
      allRhs: LogicalPlan,
      delayedRhs: LogicalPlan,
      semiCondition: Expression,
      antiCondition: Expression): Option[LogicalPlan] = {
    analyzePair(
      originalAnti,
      originalSemi,
      candidate,
      allRhs,
      delayedRhs,
      semiCondition,
      antiCondition).map(buildPairedState)
  }

  private def buildPairedState(proof: PairProof): LogicalPlan = {
    val PairProof(
      candidate,
      _,
      _,
      _,
      _,
      allView,
      semiKeys,
      candidateKeyOrdinals,
      candidateValueOrdinal,
      sourceValue,
      delayedPredicate,
      sourceKeys) = proof
    // This rule runs in Spark's post-CBO User Provided Optimizers batch, after the normal
    // ColumnPruning batch. Aggregate directly on the leaf and Spark will therefore retain every
    // physical source column (all 16 lineitem columns in canonical Q21). Build the narrow source
    // projection explicitly so the late rewrite scans only grouping, value, and predicate inputs.
    val stateInputReferences =
      AttributeSet(sourceKeys :+ sourceValue) ++ delayedPredicate.references
    val stateInput =
      Project(allView.source.output.filter(stateInputReferences.contains), allView.source)
    val allMin = Alias(
      Min(sourceValue).toAggregateExpression(),
      s"_paired_existence_all_min_${sourceValue.name}")()
    val allMax = Alias(
      Max(sourceValue).toAggregateExpression(),
      s"_paired_existence_all_max_${sourceValue.name}")()
    val delayedValue = If(delayedPredicate, sourceValue, Literal.create(null, sourceValue.dataType))
    val delayedMin = Alias(
      Min(delayedValue).toAggregateExpression(),
      s"_paired_existence_delayed_min_${sourceValue.name}")()
    val delayedMax = Alias(
      Max(delayedValue).toAggregateExpression(),
      s"_paired_existence_delayed_max_${sourceValue.name}")()
    val state =
      Aggregate(sourceKeys, sourceKeys ++ Seq(allMin, allMax, delayedMin, delayedMax), stateInput)
    val eligibleState = Filter(
      And(
        Not(EqualTo(allMin.toAttribute, allMax.toAttribute)),
        EqualTo(delayedMin.toAttribute, delayedMax.toAttribute)),
      state)
    // No later ColumnPruning pass will discard the proof-only min/max states either. Carry only
    // the correlation keys plus delayedMin into the join/exchange, matching the compact state
    // shape produced when the same algebra originates in SQL before optimizer pruning.
    val eligibleJoinState =
      Project(sourceKeys ++ Seq(delayedMin.toAttribute), eligibleState)

    val equalityConditions = semiKeys.leftEquality.zip(sourceKeys).map {
      case (leftKey, sourceKey) => EqualTo(leftKey, sourceKey): Expression
    }
    val valueCondition = EqualTo(semiKeys.leftNotEqual, delayedMin.toAttribute)
    val mergedJoin = Join(
      candidate,
      eligibleJoinState,
      // eligibleJoinState is unique by the correlation keys. LeftSemi is therefore bag- and
      // NULL-equivalent to Inner followed by Project(candidate.output), while also preserving the
      // existence-probe spine for later dimension-chain reordering instead of being flattened
      // into the ordinary inner-join cluster.
      LeftSemi,
      Some((equalityConditions :+ valueCondition).reduce(And)),
      JoinHint.NONE
    )

    logWarning(
      "MergeSelfCorrelatedExistenceState: merged paired self-correlated existence joins " +
        s"keys=${candidateKeyOrdinals.mkString(",")} value=$candidateValueOrdinal " +
        s"sourceBytes=${allView.source.stats.sizeInBytes}")
    Project(candidate.output, mergedJoin)
  }

  private def passesCostGuard(allView: SourceView, delayedView: SourceView): Boolean = {
    val sourceBytes = allView.source.stats.sizeInBytes
    val delayedSourceBytes = delayedView.source.stats.sizeInBytes
    val unknown = BigInt(Long.MaxValue)
    if (
      sourceBytes <= 0 || delayedSourceBytes <= 0 || sourceBytes == unknown ||
      delayedSourceBytes == unknown
    ) {
      return false
    }
    val minBytes = BigInt(
      glutenConfig.getConf(GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SOURCE_BYTES))
    if (sourceBytes < minBytes) {
      return false
    }
    val reductionRatio = (sourceBytes + delayedSourceBytes).toDouble / sourceBytes.toDouble
    reductionRatio >= glutenConfig.getConf(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SCAN_REDUCTION_RATIO)
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

  private def sourceView(plan: LogicalPlan): Option[SourceView] = {
    def loop(current: LogicalPlan): Option[SourceView] = {
      if (current.isStreaming) {
        return None
      }
      current match {
        case leaf: LeafNode if leaf.output.nonEmpty =>
          Some(SourceView(current, leaf, leaf.output.map(_.exprId).zipWithIndex.toMap, Seq.empty))

        case Filter(condition, child) if condition.deterministic =>
          loop(child).flatMap {
            view =>
              replaceWithSourceAttributes(condition, view).map {
                sourceCondition =>
                  view.copy(
                    plan = current,
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
                    plan = current,
                    outputOrdinals = current.output.map(_.exprId).zip(ordinals.flatten).toMap))
              } else {
                None
              }
          }

        case SubqueryAlias(_, child) =>
          loop(child).map {
            view =>
              view.copy(
                plan = current,
                outputOrdinals = passthroughOrdinals(current.output, child.output, view))
          }

        case _ => None
      }
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

  private def projectOrdinal(expression: NamedExpression, childView: SourceView): Option[Int] = {
    expression match {
      case attribute: Attribute => childView.sourceOrdinal(attribute)
      case Alias(attribute: Attribute, _) => childView.sourceOrdinal(attribute)
      case _ => None
    }
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

  private def remapSourceExpression(
      expression: Expression,
      from: LeafNode,
      to: LeafNode): Option[Expression] = {
    val ordinalByExprId = from.output.map(_.exprId).zipWithIndex.toMap
    if (!expression.references.forall(attribute => ordinalByExprId.contains(attribute.exprId))) {
      return None
    }
    Some(expression.transform {
      case attribute: Attribute => to.output(ordinalByExprId(attribute.exprId))
    })
  }

  private def parseJoinKeys(
      condition: Expression,
      left: LogicalPlan,
      right: LogicalPlan): Option[JoinKeys] = {
    val predicates = splitConjunctivePredicates(condition)
    val equalities = predicates.flatMap(extractAttributePair(_, left, right, notEqual = false))
    val notEqualities = predicates.flatMap(extractAttributePair(_, left, right, notEqual = true))
    if (
      equalities.isEmpty || notEqualities.length != 1 ||
      equalities.length + notEqualities.length != predicates.length
    ) {
      return None
    }
    Some(
      JoinKeys(
        equalities.map(_._1),
        equalities.map(_._2),
        notEqualities.head._1,
        notEqualities.head._2))
  }

  private def extractAttributePair(
      predicate: Expression,
      left: LogicalPlan,
      right: LogicalPlan,
      notEqual: Boolean): Option[(Attribute, Attribute)] = {
    val comparison = (predicate, notEqual) match {
      case (EqualTo(l: Attribute, r: Attribute), false) => Some(l -> r)
      case (Not(EqualTo(l: Attribute, r: Attribute)), true) => Some(l -> r)
      case _ => None
    }
    comparison.flatMap {
      case (first, second) if isOutput(first, left) && isOutput(second, right) =>
        Some(first -> second)
      case (first, second) if isOutput(second, left) && isOutput(first, right) =>
        Some(second -> first)
      case _ => None
    }
  }

  private def isOutput(attribute: Attribute, plan: LogicalPlan): Boolean = {
    plan.output.exists(_.semanticEquals(attribute))
  }

  private def supportsMinMax(attribute: Attribute): Boolean = {
    // MIN/MAX do not preserve SQL `!=` set cardinality for NaN and signed zero in all Spark/native
    // combinations. Start with the exact integral Q21 case and other orderable non-FP atomics.
    attribute.dataType != FloatType && attribute.dataType != DoubleType &&
    Min(attribute).checkInputDataTypes().isSuccess && Max(attribute).checkInputDataTypes().isSuccess
  }

  implicit private class OptionSequence[T](options: Seq[Option[T]]) {
    def sequence: Option[Seq[T]] = {
      if (options.forall(_.isDefined)) Some(options.flatten) else None
    }
  }
}

object MergeSelfCorrelatedExistenceState {

  val CandidateFirstExistenceAttributePrefix: String = "_gluten_candidate_first_exists_"

  /** Install one ordered pair of post-RewriteSubquery rules for this Spark session. */
  def registerPostSubqueryRules(spark: SparkSession): Unit = {
    val experimental = spark.experimental
    experimental.synchronized {
      val current = experimental.extraOptimizations
      val firstExisting = current.indexWhere {
        case _: MergeSelfCorrelatedExistenceState => true
        case _: RewriteExistenceJoinRhsDedup => true
        case _ => false
      }
      val retained = current.filterNot {
        case _: MergeSelfCorrelatedExistenceState => true
        case _: RewriteExistenceJoinRhsDedup => true
        case _ => false
      }
      val insertion =
        if (firstExisting < 0) retained.length
        else {
          current.take(firstExisting).count {
            case _: MergeSelfCorrelatedExistenceState => false
            case _: RewriteExistenceJoinRhsDedup => false
            case _ => true
          }
        }
      experimental.extraOptimizations = retained.patch(
        insertion,
        Seq(MergeSelfCorrelatedExistenceState(spark), RewriteExistenceJoinRhsDedup(spark)),
        0)
    }
  }

  /**
   * A cheap pre-RewriteSubquery signal. The full same-source and predicate implication proof is
   * deliberately deferred until Spark has materialized the LeftSemi / LeftAnti joins.
   */
  def hasPotentialPairedExistence(expression: Expression): Boolean = {
    def conjuncts(current: Expression): Seq[Expression] = current match {
      case And(left, right) => conjuncts(left) ++ conjuncts(right)
      case other => Seq(other)
    }
    val terms = conjuncts(expression)
    terms.exists(_.isInstanceOf[Exists]) && terms.exists {
      case Not(_: Exists) => true
      case _ => false
    }
  }
}
