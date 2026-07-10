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

import org.apache.gluten.backendsapi.BackendsApiManager
import org.apache.gluten.execution.{BasicScanExecTransformer, ColumnarToColumnarExec, ColumnarToRowExecBase, FilterExecTransformer, FilterExecTransformerBase, MppNativeQueryExec, MppPreparedChildExec, ProjectExecTransformer, TransformSupport}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Expression, NamedExpression}
import org.apache.spark.sql.catalyst.optimizer.BuildRight
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.physical.IdentityBroadcastMode
import org.apache.spark.sql.execution._

import java.util.Locale

/**
 * Presto-style rewrite of uncorrelated [[ScalarSubquery]] references into an inner
 * [[org.apache.gluten.execution.VeloxBroadcastNestedLoopJoinExecTransformer]].
 *
 * Spark's optimizer decorrelates correlated scalar subqueries into joins, but leaves uncorrelated
 * ones (Q11/Q15/Q22 in TPC-H) as [[ScalarSubquery]] expressions hanging off the enclosing Filter /
 * Project. Spark's physical planner wraps them in a [[BaseSubqueryExec]] sibling the driver has to
 * materialize into a single-value literal before the main plan runs -- that value is then plugged
 * into the filter via [[org.apache.gluten.expression.ScalarSubqueryTransformer]].
 *
 * Keeping the subquery on the driver means:
 *   - the outer plan stays out of MPP until we whitelist the enclosing Filter/Project (S8 patch);
 *   - the subquery itself never runs on GPU even when MPP claims the outer plan.
 *
 * This rule removes both gaps by transforming, for each qualifying Filter/Project node:
 * {{{
 *     Filter(cond, child)                                    [cond refs ScalarSubquery(q)]
 *       =>
 *     Project(originalChildOutput,
 *       FilterExecTransformer(rewrittenCond,
 *         VeloxBroadcastNestedLoopJoin(Inner, BuildRight,
 *           left  = child,
 *           right = ColumnarBroadcastExchange(IdentityBroadcastMode, q.plan.child))))
 * }}}
 * and analogously for [[ProjectExec]] (the outer [[ProjectExec]] is replaced with a
 * [[ProjectExecTransformer]] carrying the rewritten projection list; the restoring Project that
 * strips the scalar column is unnecessary when the parent already used the rewrite output).
 *
 * The rewrite preserves the original Filter/Project's output schema, so the rest of the plan tree
 * (and the driver-side caller) sees the same attributes. The scalar is introduced strictly inside
 * the Filter/Project replacement.
 *
 * Reused subqueries share a single broadcast. Spark's [[ReusedSubqueryExec]] wraps the original
 * [[org.apache.spark.sql.execution.SubqueryExec]]; we key the shared broadcast on the inner
 * subquery child's canonicalized form so multiple [[ScalarSubquery]] expressions against the same
 * physical plan collapse to one [[ColumnarBroadcastExchangeExec]].
 *
 * Correlated scalar subqueries and other [[PlanExpression]] flavours (DynamicPruning,
 * InSubqueryExec, ...) are left alone -- they either don't reach this layer or need different
 * rewrites.
 */
object RewriteUncorrelatedScalarSubquery extends Logging {

  /**
   * Apply the rewrite to every Filter/Project in the plan tree that has an uncorrelated
   * [[ScalarSubquery]] in its expressions. A shared cache ensures identical subqueries reuse the
   * same broadcast across the whole plan.
   */
  def apply(plan: SparkPlan): SparkPlan = {
    val broadcastCache = new BroadcastCache
    plan.transformUp {
      case f: FilterExec if hasUncorrelatedScalarSubquery(f.condition) =>
        rewriteFilter(f, broadcastCache)
      case p: ProjectExec if p.projectList.exists(hasUncorrelatedScalarSubquery) =>
        rewriteProject(p, broadcastCache)
      // By the time MppCollapseRule (Post rule) sees the plan, Gluten's HeuristicTransform /
      // OffloadOthers has already converted vanilla FilterExec / ProjectExec carrying
      // ScalarSubquery into their transformer variants -- ScalarSubquery IS a supported
      // expression in ExpressionConverter, so offload succeeds. Match those too.
      case f: FilterExecTransformerBase if hasUncorrelatedScalarSubquery(f.cond) =>
        rewriteFilterTransformer(f, broadcastCache)
      case p: ProjectExecTransformer if p.projectList.exists(hasUncorrelatedScalarSubquery) =>
        rewriteProjectTransformer(p, broadcastCache)
    }
  }

  private def rewriteFilterTransformer(
      filter: FilterExecTransformerBase,
      cache: BroadcastCache): SparkPlan = {
    val subqueries = collectUncorrelatedScalarSubqueries(filter.cond)
    val scalarFreeChild = stripRewrittenScalarPushdowns(filter.child, subqueries)
    val (joinedChild, replacements) = buildBroadcastStack(scalarFreeChild, subqueries, cache)
    if (replacements.isEmpty) return filter
    val rewrittenCondition = replaceScalarSubqueries(filter.cond, replacements)
    val newFilter = FilterExecTransformer(rewrittenCondition, joinedChild)
    val restoringProjectList: Seq[NamedExpression] = filter.output.map(a => a: NamedExpression)
    ProjectExecTransformer(restoringProjectList, newFilter)
  }

  private def rewriteProjectTransformer(
      project: ProjectExecTransformer,
      cache: BroadcastCache): SparkPlan = {
    val subqueries = project.projectList.flatMap(collectUncorrelatedScalarSubqueries).distinct
    val (joinedChild, replacements) = buildBroadcastStack(project.child, subqueries, cache)
    if (replacements.isEmpty) return project
    val rewritten = project.projectList.map(
      e => replaceScalarSubqueries(e, replacements).asInstanceOf[NamedExpression])
    ProjectExecTransformer(rewritten, joinedChild)
  }

  /**
   * Rewrite a [[FilterExec]] whose condition contains one or more uncorrelated scalar subqueries.
   *
   * The restoring [[ProjectExecTransformer]] on top re-projects the original filter output so
   * callers never see the scalar column.
   */
  private def rewriteFilter(filter: FilterExec, cache: BroadcastCache): SparkPlan = {
    val subqueries = collectUncorrelatedScalarSubqueries(filter.condition)
    val scalarFreeChild = stripRewrittenScalarPushdowns(filter.child, subqueries)
    val (joinedChild, replacements) = buildBroadcastStack(scalarFreeChild, subqueries, cache)
    if (replacements.isEmpty) {
      // All subqueries were correlated / unsupported; leave the node for the fallback path.
      return filter
    }
    val rewrittenCondition = replaceScalarSubqueries(filter.condition, replacements)
    val newFilter = FilterExecTransformer(rewrittenCondition, joinedChild)
    // Strip the scalar columns introduced by the BNLJ so the filter's callers see the original
    // output schema. ProjectExecTransformer is a plain TransformSupport, so MppCollapseRule will
    // absorb it into the native fragment alongside the filter and the BNLJ.
    val restoringProjectList: Seq[NamedExpression] = filter.output.map(a => a: NamedExpression)
    ProjectExecTransformer(restoringProjectList, newFilter)
  }

  /**
   * Gluten's PushDownFilterToScan copies every FilterExecTransformer conjunct into the scan's
   * pushDownFilters. When this rule later replaces a ScalarSubquery conjunct with a broadcast
   * attribute, the copied scan predicate otherwise keeps the original ScalarSubquery alive. Spark
   * then materializes that stale copy even though the rewritten filter computes the same predicate
   * inside the main MPP graph.
   *
   * Remove only pushdown predicates containing a scalar that this filter is about to rewrite. The
   * enclosing filter remains in place with the equivalent broadcast-attribute predicate, so this
   * only gives up a redundant scan pushdown; it does not remove the query predicate.
   */
  private def stripRewrittenScalarPushdowns(
      child: SparkPlan,
      rewrittenSubqueries: Seq[ScalarSubquery]): SparkPlan = {
    val rewrittenExprIds = rewrittenSubqueries.map(_.exprId).toSet
    if (rewrittenExprIds.isEmpty) return child

    child.transformUp {
      case scan: BasicScanExecTransformer if scan.pushDownFilters.nonEmpty =>
        val original = scan.pushDownFilters.get
        val retained = original.filterNot {
          filter =>
            filter.exists {
              case sq: ScalarSubquery => rewrittenExprIds.contains(sq.exprId)
              case _ => false
            }
        }
        if (retained.size == original.size) {
          scan
        } else {
          logDebug(
            s"RewriteUncorrelatedScalarSubquery: removed " +
              s"${original.size - retained.size} stale scalar scan pushdown predicate(s)")
          scan.withNewPushdownFilters(retained)
        }
    }
  }

  /**
   * Rewrite a [[ProjectExec]] whose projection list contains one or more uncorrelated scalar
   * subqueries.
   *
   * The projection itself produces the final output schema (including any attributes the rewritten
   * expressions reference), so no extra restoring project is needed.
   */
  private def rewriteProject(project: ProjectExec, cache: BroadcastCache): SparkPlan = {
    val subqueries = project.projectList.flatMap(collectUncorrelatedScalarSubqueries).distinct
    val (joinedChild, replacements) = buildBroadcastStack(project.child, subqueries, cache)
    if (replacements.isEmpty) {
      return project
    }
    val rewrittenProjectList: Seq[NamedExpression] = project.projectList.map {
      e => replaceScalarSubqueries(e, replacements).asInstanceOf[NamedExpression]
    }
    ProjectExecTransformer(rewrittenProjectList, joinedChild)
  }

  /**
   * Stack one [[org.apache.gluten.execution.VeloxBroadcastNestedLoopJoinExecTransformer]] per
   * unique subquery above `child`, returning the resulting join tree together with a map from each
   * original [[ScalarSubquery]] to the [[AttributeReference]] that should replace it.
   *
   * `child` is the outer operator's child (the streaming side). Each broadcast produces a single
   * row carrying the scalar value; the BNLJ preserves `child`'s cardinality and appends the scalar
   * column. Because the cross-product of `child` with a 1-row relation is `child` itself, filter /
   * project semantics are preserved exactly.
   */
  private def buildBroadcastStack(
      child: SparkPlan,
      subqueries: Seq[ScalarSubquery],
      cache: BroadcastCache): (SparkPlan, Map[ScalarSubquery, AttributeReference]) = {
    val replacements = scala.collection.mutable.Map[ScalarSubquery, AttributeReference]()
    var current: SparkPlan = child

    // Group subqueries by their canonicalized inner plan so we build exactly one broadcast per
    // unique underlying physical plan. Two Spark ScalarSubquery instances that reference the same
    // SubqueryExec (directly or via ReusedSubqueryExec) both point at the same scalar column, so
    // share the join.
    val uniqueKeys = scala.collection.mutable.LinkedHashMap[SparkPlan, Seq[ScalarSubquery]]()
    subqueries.foreach {
      sq =>
        val inner = unwrapSubquery(sq.plan).canonicalized
        uniqueKeys.get(inner) match {
          case Some(existing) => uniqueKeys(inner) = existing :+ sq
          case None => uniqueKeys(inner) = Seq(sq)
        }
    }

    var abort = false
    val iter = uniqueKeys.iterator
    while (!abort && iter.hasNext) {
      val (_, group) = iter.next()
      val representative = group.head
      val innerExec = unwrapSubquery(representative.plan)
      // BaseSubqueryExec.child is typically VeloxColumnarToRowExec (a ColumnarToRowExecBase)
      // wrapping the real native plan -- the driver-side eval() path needs rows. For the
      // broadcast side we want the underlying TransformSupport, so peel row/columnar shims.
      val innerPlan = peelColumnarWrappers(innerExec.child)
      if (isRuntimeBloomFilterSubquery(innerExec.child)) {
        // Runtime-DPP bloom filters are already driven by materializeScalarSubqueries.
        // Keeping them materialized avoids embedding a SINGLE->BROADCAST bloom producer
        // chain inside the main multi-peer MPP graph, which can leave Q21 waiting on
        // cross-peer broadcast/SINGLE completion. Ordinary scalar subqueries still use
        // the exchange-based rewrite.
        logDebug(
          "RewriteUncorrelatedScalarSubquery: leaving runtime bloom-filter scalar " +
            "subquery for materializeScalarSubqueries")
      } else if (!innerPlan.isInstanceOf[TransformSupport]) {
        logWarning(
          "RewriteUncorrelatedScalarSubquery: leaving ScalarSubquery unrewritten because " +
            s"its inner plan root is not TransformSupport: ${innerPlan.getClass.getSimpleName}")
        abort = true
      } else {
        val exchange = cache.getOrBuild(innerPlan)
        // Use the cached exchange's actual output attribute, not the current subquery plan's
        // attribute. Spark may create several ScalarSubquery instances with different exprIds for
        // the same canonicalized plan; the exchange cache intentionally reuses one broadcast for
        // all of them, so rewritten predicates must bind to that broadcast's output.
        val scalarAttr = exchange.output.head match {
          case ar: AttributeReference => ar
          case other => other.toAttribute.asInstanceOf[AttributeReference]
        }
        val broadcast = ColumnarCollapseTransformStages.wrapInputIteratorTransformer(exchange)
        val bnlj = BackendsApiManager.getSparkPlanExecApiInstance
          .genBroadcastNestedLoopJoinExecTransformer(
            left = current,
            right = broadcast,
            buildSide = BuildRight,
            joinType = Inner,
            condition = None)
        current = bnlj
        group.foreach(sq => replacements(sq) = scalarAttr)
      }
    }

    if (abort) (child, Map.empty) else (current, replacements.toMap)
  }

  /**
   * Walk an expression tree and replace any [[ScalarSubquery]] that appears as a key in
   * `replacements` with the corresponding attribute. Other [[ScalarSubquery]] / [[PlanExpression]]
   * instances are left untouched.
   */
  private def replaceScalarSubqueries(
      expr: Expression,
      replacements: Map[ScalarSubquery, AttributeReference]): Expression = {
    expr.transformUp {
      case sq: ScalarSubquery if replacements.contains(sq) =>
        replacements(sq)
    }
  }

  /**
   * Collect every uncorrelated [[ScalarSubquery]] appearing inside an expression tree.
   *
   * Correlated scalar subqueries are decorrelated into joins by Spark's optimizer before physical
   * planning, so a physical [[ScalarSubquery]] is always uncorrelated. As a defensive check we
   * still ignore any [[ScalarSubquery]] whose inner plan's references contain external attrs.
   */
  private def collectUncorrelatedScalarSubqueries(expr: Expression): Seq[ScalarSubquery] = {
    val seen = scala.collection.mutable.ArrayBuffer[ScalarSubquery]()
    expr.foreach {
      case sq: ScalarSubquery if isUncorrelated(sq) => seen += sq
      case _ =>
    }
    seen.toSeq
  }

  private def hasUncorrelatedScalarSubquery(expr: Expression): Boolean = {
    expr.find {
      case sq: ScalarSubquery if isUncorrelated(sq) => true
      case _ => false
    }.isDefined
  }

  /**
   * Peel ColumnarToRow / ColumnarToColumnar wrappers off a plan to get at the underlying
   * TransformSupport node. BaseSubqueryExec always wraps its native plan in a row conversion so
   * eval() can drive it, but we want the native columnar operator on the broadcast side.
   */
  private def peelColumnarWrappers(plan: SparkPlan): SparkPlan = plan match {
    // ColumnarToRowExecBase takes `child: SparkPlan` as a ctor param without `val`, so field
    // access is not possible -- but it extends UnaryExecNode, so children.head is the child.
    case c2r: ColumnarToRowExecBase => peelColumnarWrappers(c2r.children.head)
    case c2c: ColumnarToColumnarExec => peelColumnarWrappers(c2c.child)
    // MppCollapseRule wraps already-collapsed subquery plans in MppNativeQueryExec before the
    // outer plan's rewrite runs. Peel it so we reach the underlying TransformSupport tree that
    // we can wrap in a ColumnarBroadcastExchangeExec on the broadcast side.
    case mpp: MppNativeQueryExec => peelColumnarWrappers(mpp.child)
    case prepared: MppPreparedChildExec => peelColumnarWrappers(prepared.hiddenPlan)
    case other => other
  }

  private def isUncorrelated(sq: ScalarSubquery): Boolean = {
    // An uncorrelated ScalarSubquery has no outer-attr references exposed either as child
    // expressions (decorrelation would have pulled them out) or as a LateralSubquery marker.
    // We also treat anything Spark labels a PlanExpression with non-empty children as correlated.
    sq.children.isEmpty
  }

  private def isRuntimeBloomFilterSubquery(plan: SparkPlan): Boolean = {
    var found = false
    plan.foreach {
      node =>
        if (node.output.exists(_.name.toLowerCase(Locale.ROOT).contains("bloomfilter"))) {
          found = true
        }
        if (
          node.expressions.exists {
            expr =>
              val text = expr.toString.toLowerCase(Locale.ROOT)
              text.contains("bloom_filter_agg") || text.contains("velox_bloom_filter_agg")
          }
        ) {
          found = true
        }
    }
    found
  }

  /** Peel [[ReusedSubqueryExec]] layers to find the underlying [[BaseSubqueryExec]]. */
  private def unwrapSubquery(plan: BaseSubqueryExec): BaseSubqueryExec = plan match {
    case reused: ReusedSubqueryExec => unwrapSubquery(reused.child)
    case other => other
  }

  /**
   * Cache mapping a canonicalized subquery child plan to the [[ColumnarBroadcastExchangeExec]] that
   * should broadcast it. Reusing the same object across every BNLJ we build lets downstream Spark
   * reuse logic (`ReuseExchangeAndSubquery`) dedup identical broadcasts automatically, and also
   * keeps our BNLJ-per-subquery count in sync with how many ColumnarBroadcastExchangeExec instances
   * end up in the fragment graph.
   */
  final private class BroadcastCache {
    private val byCanonicalized =
      scala.collection.mutable.HashMap[SparkPlan, ColumnarBroadcastExchangeExec]()

    def getOrBuild(innerPlan: SparkPlan): ColumnarBroadcastExchangeExec = {
      byCanonicalized.getOrElseUpdate(
        innerPlan.canonicalized,
        ColumnarBroadcastExchangeExec(IdentityBroadcastMode, innerPlan))
    }
  }
}
