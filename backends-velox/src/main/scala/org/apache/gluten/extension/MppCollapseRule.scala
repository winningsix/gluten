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

import org.apache.gluten.config.GlutenConfig
import org.apache.gluten.execution._

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, PlanExpression}
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{FilterExec, ProjectExec, ScalarSubquery, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{BroadcastQueryStageExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.command.DataWritingCommandExec
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeLike, ShuffleExchangeLike}
import org.apache.spark.sql.internal.SQLConf

import java.util.concurrent.atomic.AtomicInteger

/**
 * Data class representing a native execution fragment in the MPP plan.
 *
 * A fragment is a connected subgraph of TransformSupport operators that can be compiled into a
 * single native (Velox) pipeline. Fragments are separated by exchange boundaries.
 *
 * @param id
 *   Unique identifier for this fragment within the query.
 * @param rootOperator
 *   The root SparkPlan of this fragment's TransformSupport subtree.
 * @param outputAttributes
 *   The output schema of this fragment.
 * @param parallelism
 *   The target parallelism (number of partitions) for this fragment.
 */
case class NativeFragment(
    id: Int,
    rootOperator: SparkPlan,
    outputAttributes: Seq[Attribute],
    parallelism: Int)

/**
 * Data class describing an exchange (shuffle/broadcast) between two fragments.
 *
 * @param id
 *   Unique identifier for this exchange.
 * @param producerFragmentId
 *   The fragment that produces data for this exchange.
 * @param consumerFragmentId
 *   The fragment that consumes data from this exchange.
 * @param exchangeType
 *   The type of exchange: "HASH", "BROADCAST", "SINGLE", "ROUND_ROBIN".
 * @param numPartitions
 *   The number of output partitions for the exchange.
 * @param partitionKeys
 *   The partitioning key attributes (for HASH exchanges).
 */
case class ExchangeSpec(
    id: Int,
    producerFragmentId: Int,
    consumerFragmentId: Int,
    exchangeType: String,
    numPartitions: Int,
    partitionKeys: Seq[Attribute])

/**
 * MPP plan collapse rule that replaces [[ColumnarCollapseTransformStages]] for queries that can be
 * fully executed in MPP mode.
 *
 * In BSP (Bulk Synchronous Parallel) mode, Gluten breaks the plan at every ShuffleExchange into
 * separate stages, each wrapped in a [[WholeStageTransformer]]. In MPP mode, ALL stages run
 * concurrently with streaming GPU exchange - no BSP barriers.
 *
 * This rule:
 *   1. Walks the Catalyst physical plan tree (after AQE finalization). 2. Identifies chains of
 *      [[TransformSupport]] operators separated by [[ShuffleExchangeLike]] or
 *      [[BroadcastExchangeLike]]. 3. If ALL operators in the chain are native-supported
 *      (TransformSupport), absorbs the exchanges and wraps the entire plan in a single
 *      [[MppNativeQueryExec]]. 4. If any operator is NOT supported, returns the plan unchanged so
 *      that BSP mode (via [[ColumnarCollapseTransformStages]]) handles it.
 *
 * Gated by config: `spark.gluten.mpp.enabled` (default: false).
 */
case class MppCollapseRule(glutenConf: GlutenConfig) extends Rule[SparkPlan] with Logging {

  private val MPP_ENABLED_KEY = "spark.gluten.mpp.enabled"
  private val MPP_ENABLED_DEFAULT = "true"

  // --- Q21-style broadcast fusion (Presto parity) ---
  //
  // Spark/Gluten emit a broadcast build subtree as its OWN fragment fed by a
  // BROADCAST exchange into the consumer. Presto co-locates a small REPLICATED
  // build with its consumer fragment, saving 2-3 fragments per join-heavy query
  // (TPC-H Q21 in particular: 12 fragments -> 9). This is enabled only when
  // spark.gluten.mpp.fuseBroadcastBuilds is true and the broadcast's estimated
  // size is below spark.gluten.mpp.broadcastFuseThreshold (default 8 GB).
  private val BROADCAST_FUSE_ENABLED_KEY = "spark.gluten.mpp.fuseBroadcastBuilds"
  private val BROADCAST_FUSE_ENABLED_DEFAULT = "false"
  private val BROADCAST_FUSE_THRESHOLD_KEY = "spark.gluten.mpp.broadcastFuseThreshold"
  // 8 GB (bytes). String form so SQLConf parses it deterministically across Spark versions.
  private val BROADCAST_FUSE_THRESHOLD_DEFAULT = (8L * 1024L * 1024L * 1024L).toString

  private def isBroadcastFuseEnabled: Boolean = {
    SQLConf.get
      .getConfString(BROADCAST_FUSE_ENABLED_KEY, BROADCAST_FUSE_ENABLED_DEFAULT)
      .toBoolean
  }

  private def broadcastFuseThresholdBytes: Long = {
    SQLConf.get
      .getConfString(BROADCAST_FUSE_THRESHOLD_KEY, BROADCAST_FUSE_THRESHOLD_DEFAULT)
      .toLong
  }

  /**
   * Decide whether a [[BroadcastExchangeLike]] is small enough to fuse into the consumer fragment.
   * Only fuse when the Spark-computed Statistics.sizeInBytes is both available AND below threshold.
   * If stats throw (broadcast not yet materialized) or report 0, refuse to fuse: with stats unknown
   * we have no way to confirm the build is small, and aggressive fusion in that case has bitten us
   * on Q16 - the consumer fragment's WST was already frozen with an InputIteratorTransformer that
   * emits ReadRel(iterator:0) for the build, but with fusion-decided-true no BROADCAST exchange
   * spec is emitted, so the C++ side has zero placeholder iterators and conversion crashes at
   * SubstraitToVeloxPlan.cc:1357 (streamIdx 0 < inputIters_.size() 0).
   */
  private def canFuseBroadcast(bc: SparkPlan): Boolean = {
    if (!isBroadcastFuseEnabled) return false
    val sizeBytes: Long = bc match {
      case b: BroadcastExchangeLike =>
        try b.runtimeStatistics.sizeInBytes.toLong
        catch { case _: Throwable => -1L }
      case stage: BroadcastQueryStageExec =>
        try stage.computeStats().map(_.sizeInBytes.toLong).getOrElse(-1L)
        catch { case _: Throwable => -1L }
      case _ => -1L
    }
    // sizeBytes <= 0 means "unknown" or "stats-not-yet-populated". Don't fuse in that case.
    val ok = sizeBytes > 0 && sizeBytes <= broadcastFuseThresholdBytes
    if (ok) {
      logWarning(
        s"MppCollapseRule.canFuseBroadcast: fusing (sizeBytes=$sizeBytes " +
          s"<= $broadcastFuseThresholdBytes)")
    } else {
      logWarning(
        s"MppCollapseRule.canFuseBroadcast: NOT fusing (sizeBytes=$sizeBytes " +
          s"unknown or above threshold $broadcastFuseThresholdBytes)")
    }
    ok
  }

  override def apply(plan: SparkPlan): SparkPlan = {
    if (!isMppEnabled) {
      return plan
    }
    // When Plan C (MppStrategy) is enabled, disable Plan D entirely.
    // MppStrategy handles MPP at the Strategy level; MppCollapseRule
    // would interfere with subqueries and cause "cannot transform shuffle node".
    val strategyEnabled =
      SQLConf.get.getConfString("spark.gluten.mpp.strategy.enabled", "false").toBoolean
    if (strategyEnabled) {
      return plan
    }
    // Plan C check: if MppStrategy already claimed this plan
    if (alreadyHandledByMppStrategy(plan)) {
      return plan
    }
    logWarning("MppCollapseRule: attempting MPP collapse on query plan")
    // Presto-style rewrite: turn every uncorrelated ScalarSubquery still dangling off a Filter /
    // Project into a BroadcastExchange + BroadcastNestedLoopJoin(Inner). This folds the subquery
    // into the main plan tree (so MPP absorbs it as an additional fragment) instead of leaving it
    // as a driver-materialized sibling. We apply the rewrite up front so downstream collapse /
    // walk logic sees a tree with no ScalarSubquery expressions left. If MPP collapse ultimately
    // fails we fall back to the ORIGINAL (unrewritten) plan so Spark's BSP path can run the
    // query unchanged.
    val preRewritten = RewriteUncorrelatedScalarSubquery(plan)
    preRewritten match {
      case dwce: DataWritingCommandExec =>
        // Collapse the query subtree; keep DWCE at the root so Spark still drives
        // the file-write path. MppNativeQueryExec produces the result the writer consumes.
        tryCollapseMpp(dwce.child)
          .map(mppChild => dwce.withNewChildren(Seq(mppChild)))
          .getOrElse {
            logWarning("MppCollapseRule: FALLBACK TO BSP (inside DataWritingCommandExec)")
            plan
          }
      case _ =>
        tryCollapseMpp(preRewritten).getOrElse {
          logWarning("MppCollapseRule: FALLBACK TO BSP")
          plan
        }
    }
  }

  /** Check if the plan tree already contains MppNativeQueryExec (set by MppStrategy / Plan C). */
  private def alreadyHandledByMppStrategy(plan: SparkPlan): Boolean = {
    plan match {
      case _: MppNativeQueryExec => true
      case _ => plan.children.exists(alreadyHandledByMppStrategy)
    }
  }

  private def isMppEnabled: Boolean = {
    SQLConf.get.getConfString(MPP_ENABLED_KEY, MPP_ENABLED_DEFAULT).toBoolean
  }

  /**
   * Attempt to collapse the entire plan into a single MppNativeQueryExec.
   *
   * Returns None if any part of the plan cannot be handled in MPP mode (i.e., contains
   * non-TransformSupport operators that are not exchanges).
   */
  private def tryCollapseMpp(plan: SparkPlan): Option[MppNativeQueryExec] = {
    if (!isFullyNativeSupported(plan)) {
      val reason = findFirstUnsupportedOperator(plan).getOrElse("unknown")
      logWarning(
        s"MppCollapseRule: plan contains non-native operators, " +
          s"cannot collapse to MPP. First blocker: $reason. " +
          s"Plan root: ${plan.getClass.getSimpleName}. " +
          s"Plan: ${plan.treeString.take(500)}")
      return None
    }

    logWarning("MppCollapseRule: plan is fully native-supported, wrapping with MppNativeQueryExec")

    // Rewrite any whitelisted vanilla FilterExec/ProjectExec (only present because
    // of a ScalarSubquery in their expressions) into their transformer counterparts
    // so ColumnarCollapseTransformStages, which runs right after this rule, can
    // absorb them into a WholeStageTransformer like any other native operator.
    val rewritten = rewriteSubqueryFilterProject(plan)

    // Phase 1: wrap plan with placeholder fragments. The child plan is NOT modified.
    // MppNativeQueryExec.doExecuteColumnar() will delegate to child.executeColumnar()
    // (BSP execution). True MPP fragment extraction comes in Phase 2.
    val fragments = Seq(
      NativeFragment(
        id = 0,
        rootOperator = null, // placeholder - real extraction in Phase 2
        outputAttributes = rewritten.output,
        parallelism = 4
      ))
    val exchanges = Seq.empty[ExchangeSpec]

    logWarning(
      s"MppCollapseRule: *** MPP MODE ACTIVE *** - collapsed plan into " +
        s"${fragments.size} fragments and ${exchanges.size} exchanges. " +
        s"All stages will run concurrently with streaming exchange.")

    // Plan D: Wrap, don't replace. Keep original plan as child so Spark's
    // shuffle/broadcast validation passes. At execution time, MppNativeQueryExec
    // bypasses child.executeColumnar() and runs via MppQueryCoordinator instead.
    Some(MppNativeQueryExec(child = rewritten, fragments = fragments, exchanges = exchanges))
  }

  /**
   * Replace whitelisted vanilla FilterExec/ProjectExec nodes (those carrying a ScalarSubquery in
   * their expressions) with their transformer equivalents so the normal BSP wrapping path can
   * absorb them into a WholeStageTransformer.
   */
  private def rewriteSubqueryFilterProject(plan: SparkPlan): SparkPlan = {
    plan.transformUp {
      case f: FilterExec if containsScalarSubquery(f.condition) =>
        FilterExecTransformer(f.condition, f.child)
      case p: ProjectExec if p.projectList.exists(containsScalarSubquery) =>
        ProjectExecTransformer(p.projectList, p.child)
    }
  }

  /**
   * Check whether the entire plan tree can be executed natively in MPP mode.
   *
   * A plan is fully native-supported when every non-exchange node is a TransformSupport, and every
   * exchange can be absorbed (both sides are TransformSupport with supported partitioning).
   */
  private def isFullyNativeSupported(plan: SparkPlan): Boolean = {
    plan match {
      // Exchanges that we can absorb into the MPP plan
      case exchange: ShuffleExchangeLike =>
        canAbsorbExchange(exchange) && isFullyNativeSupported(exchange.child)

      // Broadcast exchanges are absorbable when the build side is itself a fully
      // native TransformSupport subtree (recurse into children, same as shuffle).
      // The native MPP runtime treats BROADCAST as a distinct exchange type.
      case bc: BroadcastExchangeLike =>
        val absorbable = canAbsorbExchange(bc) && bc.children.forall(isFullyNativeSupported)
        if (!absorbable) {
          logWarning(
            s"MppCollapseRule: BLOCKED by BroadcastExchangeLike: " +
              s"${bc.getClass.getSimpleName} (build side not fully native)")
        }
        absorbable

      // AQE query stage wrappers - check their underlying plan
      case stage: ShuffleQueryStageExec =>
        isFullyNativeSupported(stage.plan)

      case stage: BroadcastQueryStageExec =>
        isFullyNativeSupported(stage.plan)

      // Native-supported operators
      case _: TransformSupport =>
        plan.children.forall(isFullyNativeSupported)

      // ColumnarToRow at the top of the plan is OK - Spark always adds this
      // to convert columnar output to rows for the driver. Look through it.
      case c2r: ColumnarToRowExecBase =>
        c2r.children.forall(isFullyNativeSupported)

      // Gluten-internal columnar-to-columnar nodes (batch resize, etc.) - look through
      case c2c: ColumnarToColumnarExec =>
        c2c.children.forall(isFullyNativeSupported)

      // TakeOrderedAndProjectExecTransformer wraps a sort+limit+project over a
      // TransformSupport child; treat it as a transparent wrapper and recurse.
      case topk: TakeOrderedAndProjectExecTransformer =>
        isFullyNativeSupported(topk.child)

      // Vanilla FilterExec/ProjectExec only land here when Gluten's columnar
      // validator refused to convert them -- typically because their expression
      // tree contains an uncorrelated ScalarSubquery (Q11, Q22). The subquery's
      // result is materialized by Spark's ReusedSubqueryExec mechanism and
      // injected as a Substrait literal at build time by ScalarSubqueryTransformer.
      // Whitelist these so the outer plan can collapse to MPP.
      case f: FilterExec if containsScalarSubquery(f.condition) =>
        isFullyNativeSupported(f.child)
      case p: ProjectExec if p.projectList.exists(containsScalarSubquery) =>
        isFullyNativeSupported(p.child)

      // Any other non-native operator means we cannot do MPP
      case other =>
        logWarning(
          s"MppCollapseRule: BLOCKED by non-native operator: " +
            s"${other.getClass.getSimpleName} (${other.simpleString(50)})")
        false
    }
  }

  /**
   * Walk an expression tree; return true iff it contains a [[ScalarSubquery]] reference. Used to
   * whitelist vanilla FilterExec/ProjectExec that Gluten could not validate purely because of an
   * uncorrelated scalar subquery -- the Substrait converter injects it as a literal later.
   */
  private def containsScalarSubquery(expr: Expression): Boolean = {
    expr.find {
      case _: ScalarSubquery => true
      case _: PlanExpression[_] => true
      case _ => false
    }.isDefined
  }

  /**
   * Find the first operator in the plan tree that is not natively supported, for diagnostic logging
   * purposes.
   */
  private def findFirstUnsupportedOperator(plan: SparkPlan): Option[String] = {
    plan match {
      case _: ShuffleExchangeLike if canAbsorbExchange(plan) =>
        plan.children.flatMap(findFirstUnsupportedOperator).headOption
      case _: BroadcastExchangeLike if canAbsorbExchange(plan) =>
        plan.children.flatMap(findFirstUnsupportedOperator).headOption
      case _: ShuffleQueryStageExec =>
        findFirstUnsupportedOperator(plan.asInstanceOf[ShuffleQueryStageExec].plan)
      case _: BroadcastQueryStageExec =>
        findFirstUnsupportedOperator(plan.asInstanceOf[BroadcastQueryStageExec].plan)
      case _: TransformSupport =>
        plan.children.flatMap(findFirstUnsupportedOperator).headOption
      case _: ColumnarToRowExecBase =>
        plan.children.flatMap(findFirstUnsupportedOperator).headOption
      case _: ColumnarToColumnarExec =>
        plan.children.flatMap(findFirstUnsupportedOperator).headOption
      case _: TakeOrderedAndProjectExecTransformer =>
        plan.children.flatMap(findFirstUnsupportedOperator).headOption
      case f: FilterExec if containsScalarSubquery(f.condition) =>
        plan.children.flatMap(findFirstUnsupportedOperator).headOption
      case p: ProjectExec if p.projectList.exists(containsScalarSubquery) =>
        plan.children.flatMap(findFirstUnsupportedOperator).headOption
      case other =>
        Some(s"${other.getClass.getSimpleName}: ${other.simpleString(20)}")
    }
  }

  /**
   * Check whether an exchange can be absorbed into the MPP plan.
   *
   * An exchange is absorbable when:
   *   - The child (producer side) is a TransformSupport subtree
   *   - The partitioning type is one we support for GPU streaming exchange
   *   - There are no UDFs that require JVM execution in the exchange
   */
  private def canAbsorbExchange(exchange: SparkPlan): Boolean = {
    exchange match {
      case shuffle: ShuffleExchangeLike =>
        val partitioningSupported = isSupportedPartitioning(shuffle.outputPartitioning)
        val childSupported = shuffle.child.isInstanceOf[TransformSupport] ||
          shuffle.child.isInstanceOf[ShuffleQueryStageExec] ||
          shuffle.child.isInstanceOf[BroadcastQueryStageExec] ||
          shuffle.child.isInstanceOf[ColumnarToColumnarExec]

        if (!partitioningSupported) {
          logWarning(
            s"MppCollapseRule: canAbsorbExchange=false: unsupported partitioning " +
              s"${shuffle.outputPartitioning.getClass.getSimpleName}")
        }
        if (!childSupported) {
          logWarning(
            s"MppCollapseRule: canAbsorbExchange=false: child not supported: " +
              s"${shuffle.child.getClass.getSimpleName}")
        }

        partitioningSupported && childSupported

      case broadcast: BroadcastExchangeLike =>
        // Broadcast exchanges are supported - the native side will handle broadcast
        // as a special exchange type
        broadcast.children.forall {
          child =>
            child.isInstanceOf[TransformSupport] ||
            child.isInstanceOf[ShuffleQueryStageExec] ||
            child.isInstanceOf[BroadcastQueryStageExec]
        }

      case _ => false
    }
  }

  /** Check whether a partitioning scheme is supported for MPP streaming exchange. */
  private def isSupportedPartitioning(partitioning: Partitioning): Boolean = {
    partitioning match {
      case _: HashPartitioning => true
      case _: RoundRobinPartitioning => true
      case _: RangePartitioning => true
      case SinglePartition => true
      case _: BroadcastPartitioning => true
      case _: UnknownPartitioning => false
      case other =>
        logWarning(
          s"MppCollapseRule: unsupported partitioning type: ${other.getClass.getSimpleName}")
        false
    }
  }

  /**
   * Extract NativeFragment and ExchangeSpec lists from the plan tree via DFS walk.
   *
   * The walk cuts at exchange boundaries: each contiguous subtree of TransformSupport operators
   * becomes a fragment, and each exchange becomes an ExchangeSpec linking the producer fragment to
   * the consumer fragment.
   *
   * @return
   *   Tuple of (fragments, exchanges) extracted from the plan.
   */
  private def extractFragments(
      plan: SparkPlan,
      fragmentCounter: AtomicInteger,
      exchangeCounter: AtomicInteger): (Seq[NativeFragment], Seq[ExchangeSpec]) = {

    val fragments = scala.collection.mutable.ArrayBuffer[NativeFragment]()
    val exchanges = scala.collection.mutable.ArrayBuffer[ExchangeSpec]()

    /**
     * Check whether a node is an exchange boundary (or a wrapper around one) that should be handled
     * by `walk` rather than `walkInFragment`. This looks through transparent wrappers:
     * ColumnarToColumnarExec (e.g. VeloxResizeBatchesExec), ColumnarToRowExecBase, and AQE query
     * stage nodes.
     */
    def isExchangeBoundary(node: SparkPlan): Boolean = {
      node match {
        case _: ShuffleExchangeLike | _: BroadcastExchangeLike => true
        case _: ShuffleQueryStageExec | _: BroadcastQueryStageExec => true
        case c2c: ColumnarToColumnarExec => isExchangeBoundary(c2c.child)
        case c2r: ColumnarToRowExecBase => isExchangeBoundary(c2r.child)
        case _ => false
      }
    }

    /**
     * Unwrap transparent wrapper nodes (ColumnarToColumnarExec, ColumnarToRowExecBase, AQE query
     * stages) to reach the underlying exchange or TransformSupport node.
     */
    def unwrapToExchange(node: SparkPlan): SparkPlan = {
      node match {
        case c2c: ColumnarToColumnarExec => unwrapToExchange(c2c.child)
        case c2r: ColumnarToRowExecBase => unwrapToExchange(c2r.child)
        case stage: ShuffleQueryStageExec => unwrapToExchange(stage.plan)
        case stage: BroadcastQueryStageExec => unwrapToExchange(stage.plan)
        case other => other
      }
    }

    // Walk the plan, building fragments. Returns the fragment ID of the current subtree's root.
    def walk(node: SparkPlan): Int = {
      node match {
        case shuffle: ShuffleExchangeLike =>
          // PoC: opt-in fusion of SinglePartition gather shuffles into the parent
          // fragment so Velox LocalPlanner can split internally via LocalExchange
          // instead of cross-fragment Spark exchange. Tracked by
          // spark.gluten.mpp.collapseSingleGather (default false).
          val isSingleGather =
            shuffle.outputPartitioning
              .isInstanceOf[org.apache.spark.sql.catalyst.plans.physical.SinglePartition.type]
          val collapseSingleGather =
            org.apache.spark.sql.SparkSession.getActiveSession
              .exists(_.conf.get("spark.gluten.mpp.collapseSingleGather", "false").toBoolean)
          if (isSingleGather && collapseSingleGather) {
            logWarning(
              "MppCollapseRule.walk: collapsing SinglePartition shuffle into parent fragment " +
                "(opt-in spark.gluten.mpp.collapseSingleGather=true)")
            // Walk through the child as if no shuffle; the shuffle becomes a Velox
            // LocalExchange handled by Velox LocalPlanner (or absorbed if
            // partial-final agg are in the same WST).
            walkInFragment(unwrapToExchange(shuffle.child))
            // -1 sentinel: caller (WST consumer case) treats as "no separate
            // upstream fragment"; same convention used for fused broadcast builds.
            return -1
          }
          // The child of the exchange belongs to the producer fragment
          val producerFragmentId = walk(unwrapToExchange(shuffle.child))

          // Create the consumer fragment (the exchange source side)
          val consumerFragmentId = fragmentCounter.getAndIncrement()
          val exchangeSource = MppExchangeSourceTransformer(
            exchangeCounter.get(),
            shuffle.output
          )
          fragments += NativeFragment(
            id = consumerFragmentId,
            rootOperator = exchangeSource,
            outputAttributes = shuffle.output,
            parallelism = shuffle.outputPartitioning.numPartitions
          )

          // Create exchange spec
          val (exchangeType, partitionKeys) = classifyPartitioning(shuffle.outputPartitioning)
          exchanges += ExchangeSpec(
            id = exchangeCounter.getAndIncrement(),
            producerFragmentId = producerFragmentId,
            consumerFragmentId = consumerFragmentId,
            exchangeType = exchangeType,
            numPartitions = shuffle.outputPartitioning.numPartitions,
            partitionKeys = partitionKeys
          )

          consumerFragmentId

        case broadcast: BroadcastExchangeLike =>
          // Q21-style fusion: when the broadcast build is small enough and the user
          // opted in via spark.gluten.mpp.fuseBroadcastBuilds, fold the build subtree
          // INTO the consumer fragment (no separate fragment, no BROADCAST exchange
          // spec). This matches Presto's REPLICATED distribution and removes 1
          // fragment per fused broadcast (e.g. 12 -> 9 on TPC-H Q21).
          if (canFuseBroadcast(broadcast)) {
            logWarning(
              s"MppCollapseRule: fusing broadcast build (size <= " +
                s"$broadcastFuseThresholdBytes bytes) into consumer fragment")
            val childPlan = broadcast.children.headOption
              .map(unwrapToExchange)
              .getOrElse(broadcast)
            // Walk the build subtree as part of the surrounding fragment by calling
            // walkInFragment directly. We do not allocate a fragment id for the build:
            // the caller already created the consumer fragment and will absorb us.
            walkInFragment(childPlan)
            // Sentinel: -1 means "no separate fragment" -- caller treats this child
            // exactly like an in-fragment TransformSupport child.
            -1
          } else {
            // Default (existing) behavior: build side becomes its own fragment fed
            // into the consumer via a BROADCAST exchange.
            val childPlan = broadcast.children.headOption
              .map(unwrapToExchange)
              .getOrElse(broadcast)
            val producerFragmentId = walk(childPlan)

            val consumerFragmentId = fragmentCounter.getAndIncrement()
            val exchangeSource = MppExchangeSourceTransformer(
              exchangeCounter.get(),
              broadcast.output
            )
            fragments += NativeFragment(
              id = consumerFragmentId,
              rootOperator = exchangeSource,
              outputAttributes = broadcast.output,
              parallelism = 1
            )

            exchanges += ExchangeSpec(
              id = exchangeCounter.getAndIncrement(),
              producerFragmentId = producerFragmentId,
              consumerFragmentId = consumerFragmentId,
              exchangeType = "BROADCAST",
              numPartitions = 1,
              partitionKeys = Seq.empty
            )

            consumerFragmentId
          }

        case stage: ShuffleQueryStageExec =>
          walk(stage.plan)

        case stage: BroadcastQueryStageExec =>
          walk(stage.plan)

        // Look through transparent wrapper nodes to reach the actual operator
        case c2r: ColumnarToRowExecBase =>
          walk(c2r.child)

        case c2c: ColumnarToColumnarExec =>
          walk(c2c.child)

        case transformNode =>
          // This is a TransformSupport node. Walk all children first.
          // Children that are exchanges (or wrappers around exchanges) are processed
          // via walk(). Children that are TransformSupport are part of the same fragment.
          val childFragmentIds = transformNode.children.map {
            child =>
              if (isExchangeBoundary(child)) {
                walk(unwrapToExchange(child))
              } else {
                child match {
                  case _: ShuffleExchangeLike | _: BroadcastExchangeLike |
                      _: ShuffleQueryStageExec | _: BroadcastQueryStageExec =>
                    walk(child)
                  case _ =>
                    // Non-exchange child - belongs to the same fragment, walk recursively
                    walkInFragment(child)
                    -1 // sentinel: not a separate fragment
                }
              }
          }

          // If this node has no exchange children, it's a leaf fragment or
          // part of an existing fragment being built. Create a new fragment for it.
          val fragmentId = fragmentCounter.getAndIncrement()
          val parallelism = inferParallelism(transformNode)
          fragments += NativeFragment(
            id = fragmentId,
            rootOperator = transformNode,
            outputAttributes = transformNode.output,
            parallelism = parallelism
          )

          fragmentId
      }
    }

    /**
     * Walk within a fragment (no exchange boundaries). This just recurses through TransformSupport
     * nodes that are part of the same fragment. We don't create new fragments here - the parent
     * call handles fragment creation.
     *
     * When we encounter ColumnarToColumnarExec or ColumnarToRowExecBase wrapping an exchange, we
     * route back to walk() to handle the exchange boundary properly.
     */
    def walkInFragment(node: SparkPlan): Unit = {
      node.children.foreach {
        child =>
          if (isExchangeBoundary(child)) {
            // This is a wrapper around an exchange - route to walk() to handle it
            walk(unwrapToExchange(child))
          } else {
            child match {
              case _: ShuffleExchangeLike | _: BroadcastExchangeLike | _: ShuffleQueryStageExec |
                  _: BroadcastQueryStageExec =>
                // Bare exchange - route to walk()
                walk(child)
              case _ =>
                walkInFragment(child)
            }
          }
      }
    }

    walk(plan)

    (fragments.toSeq, exchanges.toSeq)
  }

  /** Unwrap AQE query stage wrappers to get the underlying plan. */
  private def unwrapQueryStage(plan: SparkPlan): SparkPlan = {
    plan match {
      case stage: ShuffleQueryStageExec => stage.plan
      case stage: BroadcastQueryStageExec => stage.plan
      case other => other
    }
  }

  /** Classify the partitioning into an exchange type string and extract partition keys. */
  private def classifyPartitioning(partitioning: Partitioning): (String, Seq[Attribute]) = {
    partitioning match {
      case hash: HashPartitioning =>
        val keys = hash.expressions.collect { case attr: Attribute => attr }
        ("HASH", keys)
      case range: RangePartitioning =>
        val keys = range.ordering.map(_.child).collect { case attr: Attribute => attr }
        ("RANGE", keys)
      case _: RoundRobinPartitioning =>
        ("ROUND_ROBIN", Seq.empty)
      case SinglePartition =>
        ("SINGLE", Seq.empty)
      case _: BroadcastPartitioning =>
        ("BROADCAST", Seq.empty)
      case other =>
        logWarning(
          s"MppCollapseRule: unexpected partitioning type: ${other.getClass.getSimpleName}")
        ("HASH", Seq.empty)
    }
  }

  /** Infer the parallelism for a fragment based on its root operator's output partitioning. */
  private def inferParallelism(plan: SparkPlan): Int = {
    plan.outputPartitioning match {
      case p if p.numPartitions > 0 => p.numPartitions
      case _ =>
        // Default parallelism from Spark config
        SQLConf.get.getConfString("spark.sql.shuffle.partitions", "200").toInt
    }
  }
}
