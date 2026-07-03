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
import org.apache.gluten.extension.columnar.UnionTransformerRule

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, PlanExpression}
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarInputAdapter, FilterExec, ProjectExec, ScalarSubquery, SparkPlan}
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

  // --- Broadcast fusion guard ---
  //
  // The old fused path inlined broadcast builds into the consumer fragment, but
  // that path shares one JNI iterator across native fanout drivers. Keep broadcast
  // builds behind native BROADCAST exchange fragments instead.
  private val BROADCAST_FUSE_ENABLED_KEY = "spark.gluten.mpp.fuseBroadcastBuilds"
  private val BROADCAST_FUSE_ENABLED_DEFAULT = "false"

  private val SCALAR_SUBQUERY_REWRITE_ENABLED_KEY =
    "spark.gluten.mpp.rewriteUncorrelatedScalarSubquery"
  private val SCALAR_SUBQUERY_REWRITE_ENABLED_DEFAULT = "true"
  private val MPP_SINGLE_TASK_MODE_KEY =
    "spark.gluten.sql.columnar.backend.velox.mpp.singleTaskMode"

  private def isBroadcastFuseEnabled: Boolean = {
    SQLConf.get
      .getConfString(BROADCAST_FUSE_ENABLED_KEY, BROADCAST_FUSE_ENABLED_DEFAULT)
      .toBoolean
  }

  private def isScalarSubqueryRewriteEnabled: Boolean = {
    val conf = SQLConf.get
    conf.getAllConfs
      .get(SCALAR_SUBQUERY_REWRITE_ENABLED_KEY)
      .map(_.toBoolean)
      .getOrElse(!isExplicitSingleTaskMode && SCALAR_SUBQUERY_REWRITE_ENABLED_DEFAULT.toBoolean)
  }

  private def isExplicitSingleTaskMode: Boolean =
    SQLConf.get.getConfString(MPP_SINGLE_TASK_MODE_KEY, "false").toBoolean

  private def scalarSubqueryRewriteDisabledReason: String = {
    val conf = SQLConf.get
    if (
      isExplicitSingleTaskMode && !conf.getAllConfs.contains(SCALAR_SUBQUERY_REWRITE_ENABLED_KEY)
    ) {
      s"$MPP_SINGLE_TASK_MODE_KEY=true"
    } else {
      s"$SCALAR_SUBQUERY_REWRITE_ENABLED_KEY=false"
    }
  }

  private def rejectUnsafeBroadcastFusionIfRequested(context: String): Unit = {
    if (!isBroadcastFuseEnabled) return
    logWarning(
      s"$context: ignoring spark.gluten.mpp.fuseBroadcastBuilds=true; using native " +
        "BROADCAST exchange to avoid sharing one JNI broadcast iterator across fanout drivers")
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
    if (strategyEnabled && !plan.isInstanceOf[DataWritingCommandExec]) {
      // Plan C (MppStrategy) handles non-write queries at the strategy level. But it
      // SKIPS write commands (InsertIntoHadoopFsRelationCommand / Command), so for a
      // write we let Plan D collapse the write's child into MPP (DataWritingCommandExec
      // case below) instead of letting the whole write subtree fall back to BSP.
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
    // as a driver-materialized sibling. Keep it configurable because singleton scalar broadcasts
    // can be worse than materialized-literal semantics for Q11-like filters.
    val preRewritten =
      if (isScalarSubqueryRewriteEnabled) {
        RewriteUncorrelatedScalarSubquery(plan)
      } else {
        logWarning(
          s"MppCollapseRule: keeping ScalarSubquery expressions materialized by Spark " +
            s"because $scalarSubqueryRewriteDisabledReason")
        plan
      }
    preRewritten match {
      case dwce: DataWritingCommandExec =>
        // Collapse the whole WriteFilesExecTransformer subtree (write + query) INTO MPP: the
        // write becomes the final fragment root, and each pinned peer's native task runs the
        // velox TableWrite for its slice. WriteFilesExecTransformer is a TransformSupport, so
        // isFullyNativeSupported accepts it; generateSubstraitForFragment -> doWholeStageTransform
        // emits the WriteRel automatically. Keeping the write inside the pinned MPP task avoids the
        // VeloxColumnarWriteFilesRDD-on-top scheduling break (write task on the wrong executor ->
        // UCX peer mismatch -> crash).
        var collapsedWrite = false
        val rewrittenWrite = dwce.transformDown {
          // Guard with !collapsedWrite: tryCollapseMpp(wft) returns MppNativeQueryExec(wft), and
          // transformDown recurses into that result -- which still contains wft -- so without the
          // guard it re-matches and re-wraps forever (StackOverflow). Collapse only the first.
          case wft: WriteFilesExecTransformer if !collapsedWrite =>
            tryCollapseMpp(wft) match {
              case Some(mppWrite) =>
                collapsedWrite = true
                mppWrite
              case None => wft
            }
        }
        if (collapsedWrite) {
          rewrittenWrite
        } else {
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
    val unionRewritten = rewriteMppNativeUnion(plan)
    if (!isFullyNativeSupported(unionRewritten)) {
      val reason = findFirstUnsupportedOperator(unionRewritten).getOrElse("unknown")
      logWarning(
        s"MppCollapseRule: plan contains non-native operators, " +
          s"cannot collapse to MPP. First blocker: $reason. " +
          s"Plan root: ${unionRewritten.getClass.getSimpleName}. " +
          s"Plan: ${unionRewritten.treeString.take(500)}")
      return None
    }

    logWarning("MppCollapseRule: plan is fully native-supported, wrapping with MppNativeQueryExec")

    // Rewrite any whitelisted vanilla FilterExec/ProjectExec (only present because
    // of a ScalarSubquery in their expressions) into their transformer counterparts
    // so ColumnarCollapseTransformStages, which runs right after this rule, can
    // absorb them into a WholeStageTransformer like any other native operator.
    val rewritten = rewriteSubqueryFilterProject(unionRewritten)

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

  private def rewriteMppNativeUnion(plan: SparkPlan): SparkPlan = {
    UnionTransformerRule(requireSameNumPartitions = false, requireNativeUnionEnabled = false)(plan)
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

      // ColumnarCollapseTransformStages places this convention adapter below an
      // InputIteratorTransformer at exchange boundaries. It is transparent to MPP fragment
      // extraction, so validate the wrapped exchange/native subtree instead of rejecting the
      // adapter itself. This is also the shape produced by the scalar-subquery broadcast rewrite.
      case cia: ColumnarInputAdapter =>
        isFullyNativeSupported(cia.child)

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
      case cia: ColumnarInputAdapter =>
        findFirstUnsupportedOperator(cia.child)
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

    /**
     * Peel off the synthetic hash-prefix ProjectExecTransformer that
     * VeloxSparkPlanExecApi.genColumnarShuffleExchange injects above the shuffle child for HASH
     * partitioning. The injected project's first column is always aliased "hash_partition_key" and
     * computes Murmur3Hash. In MPP, we recompute that hash inside Velox PartitionedOutputNode's
     * HashPartitionFunctionSpec, so the prefix column is unnecessary and actively harmful: (a) it
     * makes producer/consumer wire schemas drift (producer ships N+1 cols, consumer expects N), (b)
     * it materializes hash_with_seed which cuDF can't replace and falls back to CPU, (c) it wraps
     * the IntraNodeTransferRegistry fast path with an extra ProjectNode.
     */
    def stripSyntheticHashProject(node: SparkPlan): SparkPlan = node match {
      case p: org.apache.gluten.execution.ProjectExecTransformer
          if p.projectList.nonEmpty && p.projectList.head.name == "hash_partition_key" =>
        p.child
      case other => other
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
          // The child of the exchange belongs to the producer fragment.
          // Strip the synthetic hash_partition_key ProjectExecTransformer that
          // VeloxSparkPlanExecApi injects: in MPP we recompute the hash on-the-fly
          // inside HashPartitionFunctionSpec, so the prefix column shouldn't exist
          // on the wire (otherwise producer ships N+1 cols, consumer expects N).
          val producerFragmentId =
            walk(stripSyntheticHashProject(unwrapToExchange(shuffle.child)))

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
          // Build side becomes its own fragment fed into the consumer via a
          // native BROADCAST exchange. Do not inline the build into the consumer:
          // the fused path is unsafe under native fanout.
          rejectUnsafeBroadcastFusionIfRequested("MppCollapseRule")
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
