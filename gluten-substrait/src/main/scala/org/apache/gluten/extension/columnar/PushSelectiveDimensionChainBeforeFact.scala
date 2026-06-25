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
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, AttributeSet}
import org.apache.spark.sql.catalyst.expressions.{EqualNullSafe, EqualTo, Expression}
import org.apache.spark.sql.catalyst.expressions.{In, InSet, PredicateHelper}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{BROADCAST, Filter, HintInfo, Join}
import org.apache.spark.sql.catalyst.plans.logical.{JoinHint, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.plans.logical.SubqueryAlias
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.datasources.LogicalRelation

import java.util.concurrent.atomic.AtomicBoolean

/**
 * Prune a large dimension by a filtered dimension chain before it reaches fact-ward joins, for
 * star-chain shapes that [[SelectiveDimensionJoinReorder]] (single hop) cannot reach.
 *
 * The target pattern is generic: a tiny literal-filtered seed joins a small or moderate prunable
 * chain dimension, that chain prunes a large dimension, and the pruned dimension then joins a
 * larger fact-ward branch. Spark's CBO can miss this shape when low-NDV pruning keys inflate the
 * intermediate estimate, even though joining only the filtered keyset into the large dimension is
 * selective.
 *
 * The rewrite moves ONLY the small `(chainDim JOIN seed)` keyset onto the victim dimension:
 * {{{
 *   victimPruned = victim JOIN (chainDim JOIN seed[literal filter])
 * }}}
 * and rebuilds the rest of the inner-join cluster left-deep with victimPruned first. Crucially the
 * rebuild defers any relation reachable only through the low-NDV pruning key until a high-NDV edge
 * is available, otherwise the low-NDV join can become a many-to-many fan-out.
 *
 * Intentionally narrow (`spark.gluten.sql.columnar.pushDimensionChainBeforeFact.enabled`): inner
 * joins only, a single selective literal-filtered seed, a prunable chain dimension, a large victim
 * with a non-pruning (high-NDV) fact-ward edge, and an idempotent canonical output.
 */
case class PushSelectiveDimensionChainBeforeFact(spark: SparkSession)
  extends Rule[LogicalPlan]
  with PredicateHelper
  with Logging {

  private val maxSmallDimensionSizeInBytes = BigInt(64L * 1024L * 1024L)
  private val maxSmallDimensionRows = BigInt(100000)
  private val maxPrunableChainDimensionSizeInBytes = BigInt(1024L * 1024L * 1024L)
  private val maxPrunableChainDimensionRows = BigInt(10000000L)
  private val minVictimRows = BigInt(10000000L) // 10M: only prune a genuinely large dimension
  private val minVictimSizeInBytes = minVictimRows * 16
  private val minBenefitRatio = 2.0 // victim must be >= this multiple of the chain dim to bother
  private val minFactProbeRatio = 4.0
  private val maxClusterItems = 12
  private val maxWrapperDepth = 4
  private val maxSelectiveInListValues = 16
  private val singleTaskModeKey = "spark.gluten.sql.columnar.backend.velox.mpp.singleTaskMode"
  private val firstApplyLog = new AtomicBoolean(false)

  registerPostCboPass()

  override def apply(plan: LogicalPlan): LogicalPlan = {
    registerPostCboPass()
    if (!enabledForSession || !plan.resolved) {
      return plan
    }
    if (firstApplyLog.compareAndSet(false, true)) {
      logDebug(
        s"PushSelectiveDimensionChainBeforeFact: enabled root=${plan.nodeName} " +
          s"resolved=${plan.resolved}")
    }
    rewritePlan(plan, blockedByParentCluster = false)
  }

  private def rewritePlan(plan: LogicalPlan, blockedByParentCluster: Boolean): LogicalPlan = {
    val blockThisCluster = !blockedByParentCluster && hasCompetingSelectiveEntrances(plan)
    val current =
      if (blockedByParentCluster || blockThisCluster) {
        plan
      } else {
        rewriteCurrent(plan)
      }
    val blockChildren = blockedByParentCluster || blockThisCluster
    val rewrittenChildren = current.children.map(rewritePlan(_, blockChildren))
    if (
      rewrittenChildren.zip(current.children).forall {
        case (left, right) => left.fastEquals(right)
      }
    ) {
      current
    } else {
      current.withNewChildren(rewrittenChildren)
    }
  }

  private def rewriteCurrent(plan: LogicalPlan): LogicalPlan = {
    plan match {
      case p @ Project(projectList, child) if projectList.forall(_.deterministic) =>
        extractInnerJoinCluster(child) match {
          case Some((input, conditions)) if input.length >= 4 && input.length <= maxClusterItems =>
            if (input.length >= 6) {
              logDebug(
                "PushSelectiveDimensionChainBeforeFact: candidate cluster " +
                  s"items=${input.map(planSummary).mkString(" | ")} predicates=${conditions.size}")
            }
            val requiredOutput = AttributeSet(projectList.flatMap(_.references))
            tryReorder(child, input, conditions, requiredOutput) match {
              case Some(rewritten) if !rewritten.fastEquals(child) =>
                logWarning(
                  "PushSelectiveDimensionChainBeforeFact: rewrote project-wrapped cluster " +
                    s"projectRefs=${requiredOutput.map(_.name).toSeq.sorted.mkString(",")}")
                Project(projectList, rewritten)
              case _ =>
                if (input.length >= 6) {
                  logDebug("PushSelectiveDimensionChainBeforeFact: candidate cluster not rewritten")
                }
                p
            }
          case _ =>
            p
        }
      case p =>
        extractInnerJoinCluster(p) match {
          case Some((input, conditions)) if input.length >= 4 && input.length <= maxClusterItems =>
            if (input.length >= 6) {
              logDebug(
                "PushSelectiveDimensionChainBeforeFact: candidate cluster " +
                  s"items=${input.map(planSummary).mkString(" | ")} predicates=${conditions.size}")
            }
            tryReorder(p, input, conditions, p.outputSet) match {
              case Some(rewritten) if !rewritten.fastEquals(p) =>
                logWarning(
                  "PushSelectiveDimensionChainBeforeFact: rewrote cluster " +
                    s"outputPreserved=${p.sameOutput(rewritten)}")
                if (p.sameOutput(rewritten)) {
                  rewritten
                } else {
                  Project(p.output, rewritten)
                }
              case _ =>
                if (input.length >= 6) {
                  logDebug("PushSelectiveDimensionChainBeforeFact: candidate cluster not rewritten")
                }
                p
            }
          case _ =>
            p
        }
    }
  }

  private def hasCompetingSelectiveEntrances(plan: LogicalPlan): Boolean = {
    extractInnerJoinCluster(plan) match {
      case Some((items, conditions)) if items.length >= 4 && items.length <= maxClusterItems =>
        if (conditions.exists(!_.deterministic)) {
          false
        } else {
          val atomic = deduplicatePredicates(conditions.flatMap(splitConjunctivePredicates))
          val uf = equivClasses(atomic)
          buildReorderContext(items, atomic, uf) match {
            case Some(context) => hasCompetingSelectiveEntrances(context)
            case None => items.count(isSelectiveFilteredDimension) > 1
          }
        }
      case _ => false
    }
  }

  private def planSummary(plan: LogicalPlan): String = {
    val rows = positiveRowCount(plan).map(_.toString).getOrElse("?")
    val attrs = plan.output.map(_.name).take(4).mkString(",")
    s"${plan.nodeName}[$attrs] rows=$rows bytes=${plan.stats.sizeInBytes} " +
      s"scanBytes=${realScanBytes(plan)}"
  }

  private def extractInnerJoinCluster(
      plan: LogicalPlan): Option[(Seq[LogicalPlan], Seq[Expression])] = {
    val items = scala.collection.mutable.ArrayBuffer.empty[LogicalPlan]
    val conditions = scala.collection.mutable.ArrayBuffer.empty[Expression]

    def collect(p: LogicalPlan): Unit = stripJoinWrapper(p, 0) match {
      case Join(left, right, Inner, condition, _) =>
        condition.foreach(conditions += _)
        collect(left)
        collect(right)
      case Filter(cond, child) if cond.deterministic && containsInnerJoin(child, 0) =>
        conditions += cond
        collect(child)
      case other =>
        items += other
    }

    collect(plan)
    if (items.length >= 2 && conditions.nonEmpty) Some((items.toSeq, conditions.toSeq)) else None
  }

  private def stripJoinWrapper(plan: LogicalPlan, depth: Int): LogicalPlan = plan match {
    case Project(projectList, child)
        if depth < maxWrapperDepth && isAttributeOnlyProject(projectList) &&
          containsInnerJoin(child, depth + 1) =>
      stripJoinWrapper(child, depth + 1)
    case SubqueryAlias(_, child)
        if depth < maxWrapperDepth && containsInnerJoin(child, depth + 1) =>
      stripJoinWrapper(child, depth + 1)
    case other => other
  }

  private def containsInnerJoin(plan: LogicalPlan, depth: Int): Boolean = plan match {
    case Join(_, _, Inner, _, _) => true
    case Project(projectList, child)
        if depth < maxWrapperDepth && isAttributeOnlyProject(projectList) =>
      containsInnerJoin(child, depth + 1)
    case Filter(_, child) if depth < maxWrapperDepth =>
      containsInnerJoin(child, depth + 1)
    case SubqueryAlias(_, child) if depth < maxWrapperDepth =>
      containsInnerJoin(child, depth + 1)
    case _ => false
  }

  private def isAttributeOnlyProject(projectList: Seq[Expression]): Boolean =
    projectList.forall(_.isInstanceOf[Attribute])

  private def alreadyHasVictimPrunedByChainSeed(
      root: LogicalPlan,
      victim: LogicalPlan,
      chainDim: LogicalPlan,
      seed: LogicalPlan): Boolean = {
    root.exists {
      case Join(left, right, Inner, _, _) =>
        (sameBranch(left, victim) && containsBothBranches(right, chainDim, seed)) ||
        (sameBranch(right, victim) && containsBothBranches(left, chainDim, seed))
      case _ => false
    }
  }

  private def containsBothBranches(
      plan: LogicalPlan,
      first: LogicalPlan,
      second: LogicalPlan): Boolean =
    containsBranch(plan, first) && containsBranch(plan, second)

  private def containsBranch(plan: LogicalPlan, target: LogicalPlan): Boolean =
    sameBranch(plan, target) || plan.children.exists(containsBranch(_, target))

  private def sameBranch(left: LogicalPlan, right: LogicalPlan): Boolean =
    left.fastEquals(right) || left.outputSet == right.outputSet

  private def registerPostCboPass(): Unit = {
    // injectOptimizerRule runs in Spark's Operator Optimization batches, before CBO can choose
    // an expensive fact-ward join order. Also register a post-CBO pass in SparkOptimizer's
    // "User Provided Optimizers" batch. MppFactProbeBroadcastHint runs in the earlier
    // operator-optimization batches and may attach hints to the CBO output; accept those hints
    // while matching, then run a post-CBO hint pass after this rewrite so the rebuilt joins still
    // get the all-REPLICATE build-side choices.
    if (!enabledForSession) {
      return
    }

    val experimental = spark.experimental
    experimental.synchronized {
      val current = experimental.extraOptimizations
      val existingRewrite =
        current.collectFirst { case r: PushSelectiveDimensionChainBeforeFact => r }.getOrElse(this)
      val existingHint = current
        .collectFirst { case r: MppFactProbeBroadcastHint => r }
        .getOrElse(MppFactProbeBroadcastHint(spark))
      val reordered = current.filterNot(
        r =>
          r.isInstanceOf[PushSelectiveDimensionChainBeforeFact] ||
            r.isInstanceOf[MppFactProbeBroadcastHint]) :+ existingRewrite :+ existingHint

      if (reordered != current) {
        experimental.extraOptimizations = reordered
        logDebug(
          "PushSelectiveDimensionChainBeforeFact: registered post-CBO rewrite plus " +
            "MppFactProbeBroadcastHint in spark.experimental.extraOptimizations")
      }
    }
  }

  private def enabledForSession: Boolean = {
    val conf = spark.sessionState.conf
    !conf.getConfString(singleTaskModeKey, "false").toBoolean &&
    new GlutenConfig(conf).enablePushDimensionChainBeforeFact
  }

  private def tryReorder(
      root: LogicalPlan,
      items: Seq[LogicalPlan],
      conditions: Seq[Expression],
      requiredOutput: AttributeSet): Option[LogicalPlan] = {
    if (conditions.exists(!_.deterministic)) {
      return None
    }
    val atomic = deduplicatePredicates(conditions.flatMap(splitConjunctivePredicates))

    // Equi-key union-find over base attributes appearing in attr=attr join predicates.
    val uf = equivClasses(atomic)
    val context = buildReorderContext(items, atomic, uf).getOrElse(return None)
    val ReorderContext(seed, others, chainDim, rest, victim, chainAttr, victimAttr, lowNdv, _) =
      context

    if (hasCompetingSelectiveEntrances(context)) {
      logDebug(
        "PushSelectiveDimensionChainBeforeFact: skipped because another selective " +
          "literal filter is not the chosen victim's fact-ward branch: " +
          competingSelectiveEntrances(context).map(planSummary).mkString(" | "))
      return None
    }

    val directChainVictimEdges =
      edgePredicates(chainDim.outputSet, victim.outputSet, atomic)
        .filter(_.references.subsetOf(lowNdv))
    if (
      directChainVictimEdges.nonEmpty &&
      alreadyHasVictimPrunedByChainSeed(root, victim, chainDim, seed)
    ) {
      return None
    }

    val seedEdge = edgePredicates(chainDim.outputSet, seed.outputSet, atomic)
    if (seedEdge.isEmpty) return None

    val replacementAttrs =
      if (directChainVictimEdges.nonEmpty) {
        Seq.empty
      } else {
        chainDim.output.filter(a => uf.connected(a, victimAttr) && !requiredOutput.contains(a))
      }
    val replacements = replacementAttrs.map(a => a.exprId.id -> victimAttr).toMap

    val remaining = rest.filterNot(_ eq victim)
    val dimChainRaw = buildJoin(chainDim, seed, seedEdge.reduceOption(And))
    val rewrittenRemainingConditions = deduplicatePredicates(
      atomic
        .filterNot {
          p =>
            seedEdge.exists(e => samePredicate(e, p)) ||
            directChainVictimEdges.exists(e => samePredicate(e, p)) ||
            (directChainVictimEdges.isEmpty && samePredicate(p, EqualTo(victimAttr, chainAttr)))
        }
        .map(replaceAttributes(_, replacements))
        .filterNot(isTrivialPredicate))

    val chainJoinAttrs =
      if (directChainVictimEdges.nonEmpty) {
        directChainVictimEdges.flatMap(_.references).filter(chainDim.outputSet.contains)
      } else {
        Seq(chainAttr)
      }
    val dimChainRequired =
      AttributeSet(chainJoinAttrs) ++ requiredOutput ++
        AttributeSet(
          rewrittenRemainingConditions.flatMap(_.references).filter(dimChainRaw.outputSet.contains))
    val dimChain = projectForFuture(dimChainRaw, dimChainRequired)

    val pruneCondition =
      if (directChainVictimEdges.nonEmpty) {
        directChainVictimEdges.reduceOption(And)
      } else {
        // victim.key = chainDim.key (transitive within the same equi-class).
        Some(EqualTo(victimAttr, chainAttr))
      }
    val victimPrunedRaw = buildJoin(victim, dimChain, pruneCondition)
    val victimPrunedRequired =
      requiredOutput ++ AttributeSet(rewrittenRemainingConditions.flatMap(_.references))
    val victimPruned = projectForFuture(victimPrunedRaw, victimPrunedRequired)

    Some(
      buildLeftDeep(
        victimPruned,
        victim,
        remaining,
        rewrittenRemainingConditions,
        lowNdv,
        requiredOutput))
  }

  /**
   * Greedy left-deep build, pruned victim first. Never connect an item to acc through a
   * low-NDV-only edge while it or another remaining item still has a high-NDV edge available.
   */
  private def buildLeftDeep(
      first: LogicalPlan,
      victim: LogicalPlan,
      rest: Seq[LogicalPlan],
      conditions: Seq[Expression],
      lowNdv: AttributeSet,
      requiredOutput: AttributeSet): LogicalPlan = {
    var acc = first
    var accIsInitialPrunedBranch = true
    var remainingConditions = conditions
    var remaining = rest
    while (remaining.nonEmpty) {
      val accSet = acc.outputSet
      val good = remaining.filter(it => hasHighNdvEdge(accSet, it.outputSet, conditions, lowNdv))
      val pick =
        if (good.nonEmpty) good.minBy(_.outputSet.toString)
        else {
          val connected = remaining.filter(it => hasEquiEdge(accSet, it.outputSet, conditions))
          if (connected.nonEmpty) connected.minBy(_.outputSet.toString)
          else remaining.minBy(_.outputSet.toString)
        }
      val joinedRefs = accSet ++ pick.outputSet
      val (conds, restConds) = remainingConditions.partition {
        expr => expr.references.subsetOf(joinedRefs) && canEvaluateWithinJoin(expr)
      }
      val joinHint = broadcastSmallerHint(
        acc,
        pick,
        allowLeft =
          accIsInitialPrunedBranch || canBroadcastPrunedBranchIntoMuchLargerFact(victim, acc, pick),
        allowRight = hasPrunableChainDimensionStats(pick)
      )
      val joined = buildJoin(acc, pick, conds.reduceOption(And), joinHint)
      remainingConditions = restConds
      remaining = remaining.filterNot(_ eq pick)
      val futureRefs = requiredOutput ++ AttributeSet(remainingConditions.flatMap(_.references))
      acc = projectForFuture(joined, futureRefs)
      accIsInitialPrunedBranch = false
    }
    if (remainingConditions.nonEmpty) {
      Filter(remainingConditions.reduceLeft(And), acc)
    } else {
      acc
    }
  }

  private def projectForFuture(plan: LogicalPlan, required: AttributeSet): LogicalPlan = {
    val keep = plan.output.filter(required.contains)
    if (keep.nonEmpty && keep.length < plan.output.length) {
      Project(keep, plan)
    } else {
      plan
    }
  }

  private def replaceAttributes(expr: Expression, replacements: Map[Long, Attribute]): Expression =
    if (replacements.isEmpty) {
      expr
    } else {
      expr.transform { case a: Attribute => replacements.getOrElse(a.exprId.id, a) }
    }

  private def isTrivialPredicate(expr: Expression): Boolean = expr match {
    case EqualTo(l: Attribute, r: Attribute) => l.semanticEquals(r)
    case EqualNullSafe(l: Attribute, r: Attribute) => l.semanticEquals(r)
    case _ => false
  }

  private def buildJoin(
      left: LogicalPlan,
      right: LogicalPlan,
      condition: Option[Expression]): Join =
    buildJoin(left, right, condition, broadcastSmallerHint(left, right))

  private def buildJoin(
      left: LogicalPlan,
      right: LogicalPlan,
      condition: Option[Expression],
      hint: JoinHint): Join = {
    Join(left, right, Inner, condition, hint)
  }

  private def broadcastSmallerHint(left: LogicalPlan, right: LogicalPlan): JoinHint = {
    broadcastSmallerHint(left, right, allowLeft = true, allowRight = true)
  }

  private def broadcastSmallerHint(
      left: LogicalPlan,
      right: LogicalPlan,
      allowLeft: Boolean,
      allowRight: Boolean): JoinHint = {
    val leftBytes = realScanBytes(left)
    val rightBytes = realScanBytes(right)
    val broadcastHint = HintInfo(strategy = Some(BROADCAST))
    if (allowLeft && leftBytes > 0 && rightBytes > 0 && leftBytes < rightBytes) {
      JoinHint(Some(broadcastHint), None)
    } else if (allowRight && leftBytes > 0 && rightBytes > 0 && rightBytes < leftBytes) {
      JoinHint(None, Some(broadcastHint))
    } else {
      JoinHint.NONE
    }
  }

  private def realScanBytes(plan: LogicalPlan): BigInt = {
    var total = BigInt(0)
    plan.foreach {
      case lr: LogicalRelation => total += BigInt(lr.relation.sizeInBytes)
      case leaf if leaf.children.isEmpty => total += leaf.stats.sizeInBytes
      case _ =>
    }
    if (total > 0) total else plan.stats.sizeInBytes
  }

  private def positiveRowCount(plan: LogicalPlan): Option[BigInt] =
    plan.stats.rowCount.filter(_ > 0).orElse {
      var total = BigInt(0)
      var found = false
      plan.foreach {
        case leaf if leaf.children.isEmpty =>
          leaf.stats.rowCount.filter(_ > 0).foreach {
            rows =>
              total += rows
              found = true
          }
        case _ =>
      }
      if (found) Some(total) else None
    }

  private def isLargeVictim(plan: LogicalPlan): Boolean =
    positiveRowCount(plan).exists(_ >= minVictimRows) ||
      realScanBytes(plan) >= minVictimSizeInBytes

  private def victimScore(plan: LogicalPlan): BigInt =
    positiveRowCount(plan).getOrElse(realScanBytes(plan) / 16)

  // ---- pattern / edge helpers ----

  /** A chainDim attr (NOT in the seed's equi-class) connected to some victim attr (same class). */
  private def pruningEdge(
      chainDim: LogicalPlan,
      victim: LogicalPlan,
      uf: EquivClasses,
      seed: LogicalPlan): Option[(Attribute, Attribute)] = {
    val seedSet = seed.outputSet
    val pairs = for {
      c <- chainDim.output if !uf.sameClassAsAny(c, seedSet)
      v <- victim.output if uf.connected(c, v)
    } yield (c, v)
    pairs.headOption
  }

  private def equivClasses(atomic: Seq[Expression]): EquivClasses = {
    val uf = new EquivClasses
    atomic.foreach {
      case EqualTo(l: Attribute, r: Attribute) => uf.union(l, r)
      case _ =>
    }
    uf
  }

  private case class ReorderContext(
      seed: LogicalPlan,
      others: Seq[LogicalPlan],
      chainDim: LogicalPlan,
      rest: Seq[LogicalPlan],
      victim: LogicalPlan,
      chainAttr: Attribute,
      victimAttr: Attribute,
      lowNdv: AttributeSet,
      factWardBranch: LogicalPlan)

  private def buildReorderContext(
      items: Seq[LogicalPlan],
      atomic: Seq[Expression],
      uf: EquivClasses): Option[ReorderContext] = {
    // Seed: exactly one item that is a selective literal-filtered small dimension.
    val seeds = items.filter(isSelectiveFilteredDimension)
    if (seeds.length != 1) return None
    val seed = seeds.head
    val others = items.filterNot(_ eq seed)

    // Chain dimension: a prunable item directly equi-joined to the seed.
    val chainCandidates = others.filter(
      it => hasEquiEdge(seed.outputSet, it.outputSet, atomic) && hasPrunableChainDimensionStats(it))
    if (chainCandidates.length != 1) return None
    val chainDim = chainCandidates.head
    val rest = others.filterNot(_ eq chainDim)

    // Victim: largest item joined to chainDim through a non-seed equi-class, that also has
    // a high-NDV fact-ward edge to another item (so pruning it helps the downstream join).
    val victimChoice = rest.flatMap {
      v =>
        pruningEdge(chainDim, v, uf, seed).flatMap {
          case (chainAttr, victimAttr) =>
            val lowNdv = uf.classOf(victimAttr)
            val factWardCandidates = rest.filter(
              f => !(f eq v) && hasHighNdvEdge(v.outputSet, f.outputSet, atomic, lowNdv))
            if (factWardCandidates.nonEmpty && isLargeVictim(v)) {
              Some(
                (
                  v,
                  chainAttr,
                  victimAttr,
                  lowNdv,
                  chooseFactWardBranch(factWardCandidates),
                  victimScore(v)))
            } else {
              None
            }
        }
    }
    if (victimChoice.isEmpty) return None
    val (victim, chainAttr, victimAttr, lowNdv, factWardBranch, _) =
      victimChoice.maxBy { case (v, _, _, _, _, rows) => (rows, v.outputSet.toString) }

    if (!benefitsFromPrune(victim, chainDim)) {
      None
    } else {
      Some(
        ReorderContext(
          seed,
          others,
          chainDim,
          rest,
          victim,
          chainAttr,
          victimAttr,
          lowNdv,
          factWardBranch))
    }
  }

  private def hasCompetingSelectiveEntrances(context: ReorderContext): Boolean =
    competingSelectiveEntrances(context).nonEmpty

  private def competingSelectiveEntrances(context: ReorderContext): Seq[LogicalPlan] =
    context.others
      .filterNot(_ eq context.victim)
      .filterNot(sameBranch(_, context.factWardBranch))
      .filter(hasSelectiveLiteralFilter)

  private def deduplicatePredicates(predicates: Seq[Expression]): Seq[Expression] =
    predicates.foldLeft(Vector.empty[Expression]) {
      (acc, p) => if (acc.exists(existing => samePredicate(existing, p))) acc else acc :+ p
    }

  private def samePredicate(left: Expression, right: Expression): Boolean =
    left.semanticEquals(right) || ((left, right) match {
      case (EqualTo(ll, lr), EqualTo(rl, rr)) =>
        ll.semanticEquals(rr) && lr.semanticEquals(rl)
      case _ => false
    })

  private def hasEquiEdge(a: AttributeSet, b: AttributeSet, atomic: Seq[Expression]): Boolean =
    edgePredicates(a, b, atomic).nonEmpty

  /** An equi-join edge between a and b that references at least one attribute OUTSIDE lowNdv. */
  private def hasHighNdvEdge(
      a: AttributeSet,
      b: AttributeSet,
      atomic: Seq[Expression],
      lowNdv: AttributeSet): Boolean =
    edgePredicates(a, b, atomic).exists(p => !p.references.subsetOf(lowNdv))

  /** equi-join predicates that straddle a and b (one ref each side, all refs within a ++ b). */
  private def edgePredicates(
      a: AttributeSet,
      b: AttributeSet,
      atomic: Seq[Expression]): Seq[Expression] =
    atomic.filter(p => isEquiEdge(p, a, b))

  private def isEquiEdge(p: Expression, a: AttributeSet, b: AttributeSet): Boolean = p match {
    case EqualTo(_, _) =>
      val refs = p.references
      refs.subsetOf(a ++ b) && refs.exists(a.contains) && refs.exists(b.contains)
    case _ => false
  }

  private def isSelectiveFilteredDimension(plan: LogicalPlan): Boolean =
    isSelectiveFilteredDimension(plan, 0)

  private def isSelectiveFilteredDimension(plan: LogicalPlan, depth: Int): Boolean = plan match {
    case Filter(cond, child) if cond.deterministic =>
      hasSelectiveLiteralPredicate(cond, child.outputSet) && hasSmallDimensionStats(child)
    case Project(projectList, child)
        if depth < maxWrapperDepth && projectList.forall(_.deterministic) =>
      isSelectiveFilteredDimension(child, depth + 1)
    case SubqueryAlias(_, child) if depth < maxWrapperDepth =>
      isSelectiveFilteredDimension(child, depth + 1)
    case _ => false
  }

  private def hasSelectiveLiteralFilter(plan: LogicalPlan): Boolean =
    hasSelectiveLiteralFilter(plan, 0)

  private def hasSelectiveLiteralFilter(plan: LogicalPlan, depth: Int): Boolean = plan match {
    case Filter(cond, child) if cond.deterministic =>
      hasSelectiveLiteralPredicate(cond, child.outputSet) ||
      (depth < maxWrapperDepth && hasSelectiveLiteralFilter(child, depth + 1))
    case Project(projectList, child)
        if depth < maxWrapperDepth && projectList.forall(_.deterministic) =>
      hasSelectiveLiteralFilter(child, depth + 1)
    case SubqueryAlias(_, child) if depth < maxWrapperDepth =>
      hasSelectiveLiteralFilter(child, depth + 1)
    case _ => false
  }

  private def hasSmallDimensionStats(plan: LogicalPlan): Boolean =
    positiveRowCount(plan).exists(_ <= maxSmallDimensionRows) ||
      realScanBytes(plan) <= maxSmallDimensionSizeInBytes

  private def hasPrunableChainDimensionStats(plan: LogicalPlan): Boolean =
    hasSmallDimensionStats(plan) ||
      positiveRowCount(plan).exists(_ <= maxPrunableChainDimensionRows) ||
      realScanBytes(plan) <= maxPrunableChainDimensionSizeInBytes

  private def canBroadcastPrunedBranchIntoMuchLargerFact(
      victim: LogicalPlan,
      prunedBranch: LogicalPlan,
      factWardBranch: LogicalPlan): Boolean = {
    val victimBytes = realScanBytes(victim).toDouble.max(1.0)
    val prunedBytes = realScanBytes(prunedBranch)
    val factBytes = realScanBytes(factWardBranch)
    prunedBytes > 0 &&
    factBytes > prunedBytes &&
    factBytes.toDouble / victimBytes >= minFactProbeRatio
  }

  private def chooseFactWardBranch(candidates: Seq[LogicalPlan]): LogicalPlan = {
    val nonSelective = candidates.filterNot(hasSelectiveLiteralFilter)
    val pool = if (nonSelective.nonEmpty) nonSelective else candidates
    pool.minBy(_.outputSet.toString)
  }

  private def benefitsFromPrune(victim: LogicalPlan, chainDim: LogicalPlan): Boolean = {
    (positiveRowCount(victim), positiveRowCount(chainDim)) match {
      case (Some(vRows), Some(cRows)) =>
        vRows >= minVictimRows &&
        (vRows.toDouble / cRows.toDouble.max(1.0)) >= minBenefitRatio
      case _ =>
        val vBytes = realScanBytes(victim).toDouble
        val cBytes = realScanBytes(chainDim).toDouble.max(1.0)
        vBytes >= minVictimSizeInBytes.toDouble && (vBytes / cBytes) >= minBenefitRatio
    }
  }

  private def hasSelectiveLiteralPredicate(cond: Expression, attrs: AttributeSet): Boolean =
    splitConjunctivePredicates(cond).exists {
      case EqualTo(l, r) => literalComparison(l, r, attrs)
      case EqualNullSafe(l, r) => literalComparison(l, r, attrs)
      case In(v, list) =>
        list.nonEmpty && list.length <= maxSelectiveInListValues &&
        referencesOnly(v, attrs) && list.forall(_.foldable)
      case InSet(v, values) =>
        values.nonEmpty && values.size <= maxSelectiveInListValues && referencesOnly(v, attrs)
      case _ => false
    }

  private def literalComparison(l: Expression, r: Expression, attrs: AttributeSet): Boolean =
    (referencesOnly(l, attrs) && r.foldable) || (referencesOnly(r, attrs) && l.foldable)

  private def referencesOnly(e: Expression, allowed: AttributeSet): Boolean =
    e.references.nonEmpty && e.references.forall(allowed.contains)

  /** Tiny union-find over Attributes keyed by exprId. */
  final private class EquivClasses {
    private val parent = scala.collection.mutable.HashMap.empty[Long, Long]
    private val attrOf = scala.collection.mutable.HashMap.empty[Long, Attribute]
    private def touch(a: Attribute): Long = {
      val k = a.exprId.id
      attrOf.getOrElseUpdate(k, a)
      parent.getOrElseUpdate(k, k)
      k
    }
    private def find(x: Long): Long = {
      var r = x
      while (parent.getOrElse(r, r) != r) r = parent(r)
      r
    }
    def union(a: Attribute, b: Attribute): Unit = {
      val ra = find(touch(a))
      val rb = find(touch(b))
      if (ra != rb) parent(rb) = ra
    }
    def connected(a: Attribute, b: Attribute): Boolean =
      attrOf.contains(a.exprId.id) && attrOf.contains(b.exprId.id) &&
        find(a.exprId.id) == find(b.exprId.id)
    def classOf(a: Attribute): AttributeSet = {
      if (!attrOf.contains(a.exprId.id)) return AttributeSet(a :: Nil)
      val root = find(a.exprId.id)
      AttributeSet(attrOf.values.filter(x => find(x.exprId.id) == root).toSeq)
    }
    def sameClassAsAny(a: Attribute, set: AttributeSet): Boolean = set.exists(s => connected(a, s))
  }
}
