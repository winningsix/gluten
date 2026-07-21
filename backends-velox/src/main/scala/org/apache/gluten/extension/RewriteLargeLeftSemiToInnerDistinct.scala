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
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, AttributeSet, EqualTo, Expression}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, JoinHint, LogicalPlan}
import org.apache.spark.sql.catalyst.plans.logical.{Project, SubqueryAlias}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreePattern.JOIN
import org.apache.spark.sql.internal.SQLConf

/**
 * Rewrite `LeftSemiJoin(left, right, equi-condition)` into `InnerJoin(left, Aggregate(rightKeys ->
 * rightKeys, right), equi-condition)` when the right side is estimated to be too large for BHJ.
 * Mirrors what Presto's optimizer auto-produces for `EXISTS` subqueries (TPC-H Q4 plan reference at
 * SF1000: 28 GB lineitem build becomes 1.5 GB distinct-orderkey build -> BHJ-eligible / smaller SHJ
 * hash table).
 *
 * Why this helps cuDF-velox specifically: Spark Catalyst pins LeftSemi to BuildRight as a
 * semantic-driven choice ("right is the existence set"), independent of cost. When the right side
 * is huge (e.g. lineitem after only a date-window filter), cudf::hash_join must materialize the
 * full build table as a single GPU allocation; this trips ~30+ GB allocations on SF1000 Q4 and Q21.
 * Adding `DISTINCT(rightKeys)` to the right input collapses it to one row per join-key value, after
 * which the build either fits in `autoBroadcastJoinThreshold` (Catalyst picks BHJ with the
 * now-small distinct side as build) or yields an InnerJoin where the optimizer can freely flip
 * BuildSide based on stats.
 *
 * Conservative match - only triggers when ALL hold:
 *   - LeftSemiJoin with a single equi-join condition tree (AND of `attr = attr` pairs)
 *   - Each equi conjunct references exactly one attribute from each side (no expressions, no
 *     correlated non-key predicates such as `l_suppkey <> l1.l_suppkey`)
 *   - No user-supplied JoinHint on the Join
 *   - `right.stats.sizeInBytes` exceeds `autoBroadcastJoinThreshold * rewriteThresholdMultiplier`
 *     (default 2x). For unanalyzed parquet temp views Spark reports the raw file-size sum, which is
 *     a usable proxy.
 *
 * Gated by `spark.gluten.mpp.rewriteLargeLeftSemiToInnerDistinct` (default true). Multiplier
 * tunable via `spark.gluten.mpp.leftSemiDistinctThresholdMultiplier`. Q21-style mixed-semi-join
 * (non-equi conjunct) is intentionally NOT covered by this minimal rule.
 */
case class RewriteLargeLeftSemiToInnerDistinct(spark: SparkSession)
  extends Rule[LogicalPlan]
  with Logging {

  private val confKey = "spark.gluten.mpp.rewriteLargeLeftSemiToInnerDistinct"
  private val confDefault = "true"
  private val multiplierKey = "spark.gluten.mpp.leftSemiDistinctThresholdMultiplier"
  private val multiplierDefault = "2"

  override def apply(plan: LogicalPlan): LogicalPlan = {
    // Self-register into spark.experimental.extraOptimizations on first apply so
    // this rule lands in the "User Provided Optimizers" batch (FixedPoint), which
    // runs AFTER the optimizer's "Subquery" batch (where RewriteSubquery converts
    // EXISTS into LeftSemiJoin) -- the only state where our pattern matches.
    //
    // Why self-register vs. SparkSessionExtensions hooks:
    //  - injectOptimizerRule: lands in "Operator Optimization" batches, which run
    //    BEFORE "Subquery" batch in Spark 3.5. Rule sees Filter(Exists(...)),
    //    never LeftSemiJoin, never matches.
    //  - injectPreCBORule: SHOULD work in theory -- in Spark 3.5
    //    BaseSessionStateBuilder.customPreCBORules calls
    //    extensions.buildPreCBORules(session) and chains them into the
    //    SparkOptimizer's "Pre CBO Rules" batch (Once strategy) that runs after
    //    all defaultBatches (including Subquery). Gluten does not override
    //    SessionStateBuilder or SparkOptimizer either. But empirically, switching
    //    this rule's registration to injectPreCBORule made Q4 SF1000 regress from
    //    1.97s (rule fires) to 12s (rule doesn't fire), per the regression sweep
    //    on 2026-05-15. Root cause was not investigated -- candidates: an
    //    intermediate batch changes the plan shape (drops LeftSemi or shrinks
    //    right.stats below the BHJ-threshold-multiplier) before Pre CBO runs,
    //    or Pre CBO sees a different plan than User Provided Optimizers does.
    //    Worth revisiting if Spark adds a cleaner post-Subquery hook or if a
    //    later investigation pins the cause.
    //
    // Self-register is idempotent (reference-equality check) and synchronized to
    // avoid concurrent double-append on first-query startup. Net cost after the
    // first call is one O(n) eq scan of a tiny list; functional behaviour matches
    // having the rule in extraOptimizations from session-init time.
    val experimental = spark.experimental
    experimental.synchronized {
      if (!experimental.extraOptimizations.exists(_ eq this)) {
        experimental.extraOptimizations = experimental.extraOptimizations :+ this
        logDebug(
          "RewriteLargeLeftSemiToInnerDistinct: self-registered into " +
            "spark.experimental.extraOptimizations for post-RewriteSubquery pass")
      }
    }

    val conf = SQLConf.get
    val enabled = conf.getConfString(confKey, confDefault).toBoolean
    if (!enabled) {
      return plan
    }
    val multiplier = scala.util
      .Try(conf.getConfString(multiplierKey, multiplierDefault).toLong)
      .getOrElse(multiplierDefault.toLong)
    val sizeThreshold = conf.autoBroadcastJoinThreshold * multiplier

    plan.transformDownWithPruning(_.containsPattern(JOIN)) {
      case j @ Join(left, right, LeftSemi, Some(condition), hint)
          if !hasUserHint(hint) && right.stats.sizeInBytes > BigInt(sizeThreshold) =>
        extractEquiKeys(condition, left.outputSet, right.outputSet) match {
          case Some((_, rightKeys)) if isAlreadyUniqueOn(right, AttributeSet(rightKeys), 0) =>
            // A grouped existence-state RHS can already be unique on a subset of the equi keys.
            // Adding DISTINCT(rightKeys) in that case creates another full-cardinality hash
            // aggregation after the real FINAL aggregation (canonical Q21 has tens of billions
            // of order keys). Inner is bag-equivalent because at most one RHS row can match; keep
            // the original LeftSemi output contract explicitly.
            logDebug(
              s"RewriteLargeLeftSemiToInnerDistinct: rewrote already-unique LeftSemi -> " +
                s"Project(left.output, Inner) without DISTINCT on right keys " +
                s"${rightKeys.map(_.name).mkString("[", ",", "]")}")
            Project(left.output, Join(left, right, Inner, Some(condition), JoinHint.NONE))
          case Some((_, rightKeys)) =>
            val deduplicatedRight = Aggregate(
              groupingExpressions = rightKeys,
              aggregateExpressions = rightKeys,
              child = right
            )
            logDebug(
              s"RewriteLargeLeftSemiToInnerDistinct: rewrote LeftSemi -> Inner with DISTINCT on " +
                s"right keys ${rightKeys.map(_.name).mkString("[", ",", "]")}; " +
                s"right.sizeInBytes=${right.stats.sizeInBytes} > threshold=$sizeThreshold")
            // DISTINCT makes the inner join bag-equivalent to LeftSemi, but Inner would otherwise
            // expose the RHS keys. Preserve the LeftSemi output contract explicitly, just as the
            // already-unique fast path above does.
            Project(
              left.output,
              Join(left, deduplicatedRight, Inner, Some(condition), JoinHint.NONE))
          case None =>
            // non-equi / mixed-side conjunct (e.g. Q21 l_suppkey <> l1.l_suppkey)
            // -- intentionally NOT rewritten by this minimal rule. Return the
            // original tree node (NOT a freshly-constructed Join with the same
            // fields) so TreeNodeTags / referential identity used by other
            // rules survive.
            j
        }
    }
  }

  private def isAlreadyUniqueOn(plan: LogicalPlan, equiKeys: AttributeSet, depth: Int): Boolean = {
    val maxWrapperDepth = 4
    plan match {
      case Aggregate(groupingExpressions, _, _) if groupingExpressions.nonEmpty =>
        // GROUP BY (orderkey) is unique for a join on (orderkey, delayedMin). The inverse is not
        // true, hence grouping must be a subset of the equi keys rather than merely intersect it.
        groupingExpressions.forall {
          case groupingAttribute: Attribute =>
            equiKeys.exists(_.semanticEquals(groupingAttribute))
          case _ => false
        }
      case Filter(condition, child) if depth < maxWrapperDepth && condition.deterministic =>
        isAlreadyUniqueOn(child, equiKeys, depth + 1)
      case Project(projectList, child)
          if depth < maxWrapperDepth && projectList.forall(_.isInstanceOf[Attribute]) =>
        isAlreadyUniqueOn(child, equiKeys, depth + 1)
      case SubqueryAlias(_, child) if depth < maxWrapperDepth =>
        isAlreadyUniqueOn(child, equiKeys, depth + 1)
      case _ => false
    }
  }

  /**
   * Split `cond` into top-level conjuncts and verify every conjunct is `EqualTo(leftAttr,
   * rightAttr)` (or the symmetric form) where one side references only `leftOut` and the other only
   * `rightOut`. Returns Some((leftKeys, rightKeys)) on match, None otherwise (e.g. when a conjunct
   * mixes both sides like a `<>` correlation, or wraps an expression around the attribute).
   */
  private def extractEquiKeys(
      cond: Expression,
      leftOut: AttributeSet,
      rightOut: AttributeSet): Option[(Seq[Attribute], Seq[Attribute])] = {
    val conjuncts = splitAnd(cond)
    val leftBuilder = scala.collection.mutable.ArrayBuffer.empty[Attribute]
    val rightBuilder = scala.collection.mutable.ArrayBuffer.empty[Attribute]
    val seenRightKeys = scala.collection.mutable.HashSet.empty[Long]
    conjuncts.foreach {
      case EqualTo(l: Attribute, r: Attribute) =>
        if (leftOut.contains(l) && rightOut.contains(r)) {
          leftBuilder += l
          if (seenRightKeys.add(r.exprId.id)) rightBuilder += r
        } else if (leftOut.contains(r) && rightOut.contains(l)) {
          leftBuilder += r
          if (seenRightKeys.add(l.exprId.id)) rightBuilder += l
        } else {
          return None
        }
      case _ => return None
    }
    if (leftBuilder.isEmpty) None else Some((leftBuilder.toSeq, rightBuilder.toSeq))
  }

  private def splitAnd(expr: Expression): Seq[Expression] = expr match {
    case And(l, r) => splitAnd(l) ++ splitAnd(r)
    case other => Seq(other)
  }

  private def hasUserHint(hint: JoinHint): Boolean = {
    hint.leftHint.exists(_.strategy.isDefined) ||
    hint.rightHint.exists(_.strategy.isDefined)
  }
}
