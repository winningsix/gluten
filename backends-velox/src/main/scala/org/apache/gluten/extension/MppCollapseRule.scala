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
import org.apache.gluten.backendsapi.velox.VeloxBatchType
import org.apache.gluten.config.GlutenConfig
import org.apache.gluten.execution._
import org.apache.gluten.extension.columnar.{FallbackTags, UnionTransformerRule}
import org.apache.gluten.extension.columnar.offload.OffloadOthers.ARROW_SCALAR_NORMALIZATION_REJECTION_TAG
import org.apache.gluten.extension.columnar.rewrite.{PullOutPostProject, PullOutPreProject}
import org.apache.gluten.extension.columnar.transition.InsertTransitions

import org.apache.spark.api.python.ColumnarArrowEvalPythonExec
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, PlanExpression, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarInputAdapter, ColumnarShuffleExchangeExecBase, ColumnarToRowExec, CommandResultExec, DeserializeToObjectExec, FilterExec, GenerateExec, ProjectExec, RowToColumnarExec, ScalarSubquery, SortExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{BroadcastQueryStageExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.aggregate.BaseAggregateExec
import org.apache.spark.sql.execution.command.{DataWritingCommandExec, ExecutedCommandExec}
import org.apache.spark.sql.execution.datasources.v2.{V2CommandExec, V2TableWriteExec}
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeLike, ShuffleExchangeLike}
import org.apache.spark.sql.execution.python.EvalPythonExecTransformer
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
    partitionKeys: Seq[Attribute],
    rangeOrdering: Seq[SortOrder] = Seq.empty,
    @transient rangeSamplePlan: SparkPlan = null,
    rangeBoundsJson: Option[String] = None,
    rangeEffectivePartitions: Option[Int] = None)

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
    if (
      strategyEnabled &&
      !plan.isInstanceOf[DataWritingCommandExec] &&
      !plan.isInstanceOf[V2TableWriteExec]
    ) {
      // Plan C (MppStrategy) handles non-write queries at the strategy level. But it
      // skips write commands, so Plan D must collapse the child of both V1 and V2
      // writes instead of letting the write subtree fall back to BSP.
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
      case v2: V2TableWriteExec if !alreadyHandledByMppStrategy(v2.query) =>
        // DataSource V2 owns the external transaction/commit protocol (Iceberg in
        // particular), so the command node itself is deliberately not a native
        // operator.  That must not force its whole query back to Spark BSP.  Keep
        // the V2 writer as the thin commit boundary and collapse its query child;
        // all scans, relational operators, and exchanges still execute in one MPP
        // query and feed the V2 writer through the normal columnar transition.
        val writeNormalization = MppCollapseRule.normalizeV2WriteExchangeAdapters(v2.query)
        if (writeNormalization.strippedAdapters > 0) {
          logInfo(
            s"MppCollapseRule: removed ${writeNormalization.strippedAdapters} " +
              "V2-write-only ColumnarToRow convention adapter(s); " +
              "the V2 writer remains the row/commit boundary and MppNativeQueryExec.doExecute " +
              "performs its final native-to-row conversion")
        }
        collapseMppOrArrowHybrid(writeNormalization.plan) match {
          case Some(mppQuery) =>
            logWarning(
              s"MppCollapseRule: *** MPP MODE ACTIVE UNDER V2 WRITE *** " +
                s"writer=${v2.getClass.getSimpleName}")
            v2.withNewChildren(Seq(mppQuery))
          case None =>
            val reason =
              s"MppCollapseRule: FALLBACK TO BSP under V2 write " +
                s"${v2.getClass.getSimpleName}"
            logWarning(reason)
            if (failOnFallback) throw new IllegalStateException(reason)
            plan
        }
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
          val reason = "MppCollapseRule: FALLBACK TO BSP (inside DataWritingCommandExec)"
          logWarning(reason)
          if (failOnFallback) throw new IllegalStateException(reason)
          plan
        }
      case _: ExecutedCommandExec =>
        // CREATE OR REPLACE TEMP VIEW and similar command wrappers only mutate
        // driver-side catalog/session metadata; they do not execute their logical
        // query child.  Strict MPP fallback applies to data execution, not to this
        // zero-row command boundary.  The first action against the registered view
        // is planned independently and must still collapse to MPP.
        preRewritten
      case _: CommandResultExec =>
        // Spark replaces an eagerly executed command with CommandResultExec. Its
        // commandPhysicalPlan is not exposed through children, so a second plan
        // inspection cannot discover the MPP query that already ran. No data
        // execution remains at this boundary.
        preRewritten
      case _: V2CommandExec =>
        // Namespace/property/catalog commands are metadata-only. V2 data writes
        // were handled above by the more specific V2TableWriteExec case.
        preRewritten
      case _ =>
        collapseMppOrArrowHybrid(preRewritten).getOrElse {
          val reason = "MppCollapseRule: FALLBACK TO BSP"
          logWarning(reason)
          if (failOnFallback) throw new IllegalStateException(reason)
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

  private def failOnFallback: Boolean = {
    SQLConf.get.getConfString("spark.gluten.mpp.failOnFallback", "false").toBoolean
  }

  /**
   * Collapse a fully native query, or split it at an explicitly supported columnar Arrow Python
   * boundary. The latter is Phase B1: the native producer runs as MPP, the unchanged Python
   * function runs in the external Arrow worker, and the already-columnar Spark/native suffix stays
   * outside that producer. This is an intentional hybrid boundary, not BSP fallback.
   */
  private def collapseMppOrArrowHybrid(plan: SparkPlan): Option[SparkPlan] = {
    plan match {
      // Dataset.rdd is represented by a terminal DeserializeToObjectExec. Keep exactly that
      // object-producing root and its direct C2R outside MPP while requiring the relational
      // producer below them to satisfy the ordinary fully-native contract. Object operators at any
      // nested position, and every other root object operator, remain rejected by the strict
      // recursive validator.
      case deserialize: DeserializeToObjectExec =>
        collapseTerminalObjectEgress(deserialize)
      // Spark materializes an uncorrelated scalar subquery through a final C2R even when its
      // producer is entirely native. A global aggregate without grouping emits at most one row,
      // so retain that semantically required driver boundary while collapsing its child to MPP.
      // This avoids a BSP stage and never introduces the expensive C2R -> R2C round trip.
      case c2r: ColumnarToRowExecBase if MppCollapseRule.isBoundedScalarResult(c2r.child) =>
        collapseMppOrArrowHybrid(c2r.child).map {
          nativeChild =>
            logWarning(
              "MppCollapseRule: *** MPP MODE ACTIVE UNDER BOUNDED SCALAR ROW OUTPUT *** " +
                s"boundary=${c2r.getClass.getSimpleName}; final C2R converts at most one row")
            c2r.withNewChildren(Seq(nativeChild))
        }
      case c2r: ColumnarToRowExec if MppCollapseRule.isBoundedScalarResult(c2r.child) =>
        collapseMppOrArrowHybrid(c2r.child).map {
          nativeChild =>
            logWarning(
              "MppCollapseRule: *** MPP MODE ACTIVE UNDER BOUNDED SCALAR ROW OUTPUT *** " +
                s"boundary=${c2r.getClass.getSimpleName}; Spark C2R removed and " +
                "MppNativeQueryExec performs the final JNI row conversion")
            nativeChild
        }
      // A terminal root C2R is Spark's row-consumer boundary (for example Dataset.rdd /
      // javaToPython, collect, head, or toLocalIterator). The physical plan does not retain which
      // action requested it, so define the contract by its exact position: keep only the plan-root
      // adapter outside MppNativeQueryExec while requiring its entire child to be either native or
      // the separately validated batch ExistingRDD-input hybrid. A nested C2R still reaches the
      // strict validator and is rejected.
      case c2r: ColumnarToRowExecBase =>
        collapseTerminalRowEgress(c2r, c2r.child)
      case c2r: ColumnarToRowExec =>
        collapseTerminalRowEgress(c2r, c2r.child)
      case _ =>
        tryCollapseExistingRddHybrid(plan)
          .orElse(tryCollapseMpp(plan))
          .orElse(tryCollapseArrowPythonBoundary(plan))
    }
  }

  private def collapseTerminalRowEgress(
      boundary: SparkPlan,
      child: SparkPlan): Option[SparkPlan] = {
    tryCollapseExistingRddHybrid(child).orElse(tryCollapseMpp(child)).map {
      nativeChild =>
        val (rowOutput, transitionMessage) = boundary match {
          case c2r: ColumnarToRowExecBase =>
            (c2r.withNewChildren(Seq(nativeChild)), "the Gluten C2R remains outside MPP")
          case _: ColumnarToRowExec =>
            (nativeChild, "Spark C2R removed; MppNativeQueryExec performs JNI row conversion")
        }
        logWarning(
          "MppCollapseRule: *** INTENTIONAL TERMINAL NATIVE ROW OUTPUT *** " +
            s"boundary=${boundary.getClass.getSimpleName}; $transitionMessage")
        rowOutput
    }
  }

  /**
   * Collapse the relational producer of a root Dataset.rdd object adapter.
   *
   * Spark's InsertTransitions contract places one direct C2R between DeserializeToObjectExec and a
   * columnar child. Preserve that required transition outside MPP, but peel it before validating
   * the relational subtree so it is not mistaken for a nested row island. This direct adjacency is
   * defined by Spark's transition contract; diagnostic plan strings can be truncated before the
   * child tree and are not evidence for accepting any looser object shape.
   */
  private def collapseTerminalObjectEgress(boundary: DeserializeToObjectExec): Option[SparkPlan] =
    boundary.child match {
      case c2r: ColumnarToRowExecBase =>
        collapseTerminalObjectEgressThroughC2r(boundary, c2r, c2r.child)
      case c2r: ColumnarToRowExec =>
        tryCollapseExistingRddHybrid(c2r.child).orElse(tryCollapseMpp(c2r.child)).map {
          nativeChild =>
            logWarning(
              "MppCollapseRule: *** INTENTIONAL TERMINAL NATIVE OBJECT OUTPUT *** " +
                s"boundary=${boundary.getClass.getSimpleName}; direct Spark C2R removed and " +
                "MppNativeQueryExec performs the JNI row conversion")
            boundary.withNewChildren(Seq(nativeChild))
        }
      case _ => None
    }

  private def collapseTerminalObjectEgressThroughC2r(
      boundary: DeserializeToObjectExec,
      directRowTransition: SparkPlan,
      relationalChild: SparkPlan): Option[SparkPlan] = {
    tryCollapseExistingRddHybrid(relationalChild).orElse(tryCollapseMpp(relationalChild)).map {
      nativeChild =>
        val objectInput = directRowTransition.withNewChildren(Seq(nativeChild))
        logWarning(
          "MppCollapseRule: *** INTENTIONAL TERMINAL NATIVE OBJECT OUTPUT *** " +
            s"boundary=${boundary.getClass.getSimpleName}; the root object adapter and its " +
            s"direct ${directRowTransition.getClass.getSimpleName} remain outside MPP")
        boundary.withNewChildren(Seq(objectInput))
    }
  }

  private def tryCollapseArrowPythonBoundary(plan: SparkPlan): Option[SparkPlan] = {
    var sawArrowBoundary = false
    var failedProducer = false

    val normalized = normalizeMppNativeOperators(plan)
    val rewritten = normalized.transformUp {
      case arrow: ColumnarArrowEvalPythonExec =>
        sawArrowBoundary = true
        collapseArrowProducer(arrow.child) match {
          case Some(producer) => arrow.withNewChildren(Seq(producer))
          case None =>
            failedProducer = true
            arrow
        }
    }

    if (!sawArrowBoundary || failedProducer || !isSupportedArrowHybridPlan(rewritten)) {
      None
    } else {
      logWarning(
        "MppCollapseRule: *** INTENTIONAL COLUMNAR ARROW PYTHON MPP BOUNDARY *** " +
          "native producer -> ColumnarArrowEvalPythonExec -> columnar native/Spark suffix; " +
          "no row conversion or BSP fallback was introduced")
      Some(rewritten)
    }
  }

  /**
   * Preserve leading batch-to-batch conversion nodes required by the Arrow operator. Collapsing
   * those conversions inside MppNativeQueryExec would make MPP advertise/output its native batch
   * convention directly to a child that requires ArrowJavaBatchType.
   */
  private def collapseArrowProducer(plan: SparkPlan): Option[SparkPlan] = plan match {
    case mpp: MppNativeQueryExec => Some(mpp)
    case c2c: ColumnarToColumnarExec =>
      collapseArrowProducer(c2c.child).map(child => c2c.withNewChildren(Seq(child)))
    case _: ColumnarToRowExecBase | _: RowToColumnarExecBase =>
      None
    case nativeProducer =>
      tryCollapseMpp(nativeProducer)
  }

  /**
   * Validate the B1 suffix without pretending the external Arrow worker is serializable as
   * Substrait. Every other operator must remain columnar/native-compatible, and row transitions are
   * rejected even if normal non-MPP Gluten would otherwise insert them.
   */
  private def isSupportedArrowHybridPlan(plan: SparkPlan): Boolean =
    isSupportedArrowHybridPlan(plan, allowDriverRowOutput = true)

  private def isSupportedArrowHybridPlan(plan: SparkPlan, allowDriverRowOutput: Boolean): Boolean =
    plan match {
      case _: MppNativeQueryExec => true
      case arrow: ColumnarArrowEvalPythonExec =>
        arrow.child.find {
          case _: ColumnarToRowExecBase | _: RowToColumnarExecBase => true
          case _ => false
        }.isEmpty && containsMppProducer(arrow.child)
      case exchange: ShuffleExchangeLike =>
        isSupportedPartitioning(exchange.outputPartitioning) &&
        exchange.children.forall(isSupportedArrowHybridPlan(_, allowDriverRowOutput = false))
      case exchange: BroadcastExchangeLike =>
        exchange.children.forall(isSupportedArrowHybridPlan(_, allowDriverRowOutput = false))
      case stage: ShuffleQueryStageExec =>
        isSupportedArrowHybridPlan(stage.plan, allowDriverRowOutput = false)
      case stage: BroadcastQueryStageExec =>
        isSupportedArrowHybridPlan(stage.plan, allowDriverRowOutput = false)
      case c2r: ColumnarToRowExecBase if allowDriverRowOutput =>
        isSupportedArrowHybridPlan(c2r.child, allowDriverRowOutput = false)
      case _: ColumnarToRowExecBase | _: RowToColumnarExecBase => false
      case c2c: ColumnarToColumnarExec =>
        isSupportedArrowHybridPlan(c2c.child, allowDriverRowOutput = false)
      case cia: ColumnarInputAdapter =>
        isSupportedArrowHybridPlan(cia.child, allowDriverRowOutput = false)
      case _: TransformSupport =>
        plan.children.forall(isSupportedArrowHybridPlan(_, allowDriverRowOutput = false))
      case topk: TakeOrderedAndProjectExecTransformer =>
        isSupportedArrowHybridPlan(topk.child, allowDriverRowOutput = false)
      case _ => false
    }

  private def containsMppProducer(plan: SparkPlan): Boolean = plan match {
    case _: MppNativeQueryExec => true
    case other => other.children.exists(containsMppProducer)
  }

  /**
   * Attempt to collapse the entire plan into a single MppNativeQueryExec.
   *
   * Returns None if any part of the plan cannot be handled in MPP mode (i.e., contains
   * non-TransformSupport operators that are not exchanges).
   */
  private def tryCollapseMpp(plan: SparkPlan): Option[MppNativeQueryExec] = {
    // Gluten's ordinary validation can leave a temporary C2R -> row aggregate/sort/project/filter
    // -> R2C shell when a native operator has a computed argument. Repair only those explicit
    // late-offload shapes before checking row boundaries. Running the broad transition normalizer
    // first would also erase an unrelated nested C2R and weaken the strict-MPP contract.
    val repaired = repairRecoverableRowShells(plan)
    findUnpairedNestedRowOutput(repaired) match {
      case Some(boundary) =>
        logWarning(
          "MppCollapseRule: refusing to normalize an unpaired nested C2R; " +
            "only the explicit root row-output path may retain that boundary. " +
            s"Offending boundary: ${describeExecutionBoundary(boundary, "native-to-row")}. " +
            s"Subtree: ${boundary.treeString.take(500)}")
        return None
      case None => ()
    }
    val normalized = normalizeMppNativeOperators(repaired)
    tryCollapseNormalizedMpp(
      normalized,
      allowExistingRddIngress = false,
      mode = "fully native-supported")
  }

  /**
   * Apply the existing strict native normalizer only to row shells it explicitly knows how to
   * late-offload. This makes computed aggregate/sort expressions reachable without treating an
   * arbitrary nested row consumer as a convention adapter.
   */
  private def repairRecoverableRowShells(plan: SparkPlan): SparkPlan =
    repairRecoverableRowShells(plan, preserveExistingRddIngress = false)

  private def repairRecoverableRowShells(
      plan: SparkPlan,
      preserveExistingRddIngress: Boolean): SparkPlan =
    plan.transformUp {
      case r2c: RowToColumnarExecBase if isRecoverableRowOperator(r2c.child) =>
        normalizeMppNativeOperators(
          r2c,
          preserveExistingRddIngress,
          recoverGenerate = isRecoverableGenerateRowIsland(r2c.child))
      case r2c: RowToColumnarExec if isRecoverableRowOperator(r2c.child) =>
        normalizeMppNativeOperators(
          r2c,
          preserveExistingRddIngress,
          recoverGenerate = isRecoverableGenerateRowIsland(r2c.child))
    }

  private def isRecoverableRowOperator(plan: SparkPlan): Boolean = plan match {
    case _: BaseAggregateExec => true
    case sort: SortExec if !sort.global => true
    case _: ProjectExec | _: FilterExec => true
    case _ => false
  }

  /**
   * Recognize the exact closed row island left when Generate's computed input fails the ordinary
   * heuristic offload as a unit:
   *
   * R2C -> (Project/Filter)* -> Generate -> C2R -> native child
   *
   * The enclosing R2C is matched by [[repairRecoverableRowShells]]. Requiring a supported
   * generator, direct schema-identical C2R, and a columnar child keeps this repair distinct from an
   * arbitrary nested row consumer. The latter must continue to fail strict MPP validation.
   */
  private def isRecoverableGenerateRowIsland(plan: SparkPlan): Boolean = plan match {
    case project: ProjectExec => isRecoverableGenerateRowIsland(project.child)
    case filter: FilterExec => isRecoverableGenerateRowIsland(filter.child)
    case generate: GenerateExec if GenerateExecTransformer.supportsGenerate(generate.generator) =>
      generate.child match {
        case c2r: ColumnarToRowExecBase if c2r.children.size == 1 =>
          val nativeChild = c2r.children.head
          nativeChild.supportsColumnar &&
          MppCollapseRule.sameOutput(c2r.output, nativeChild.output)
        case c2r: ColumnarToRowExec =>
          c2r.child.supportsColumnar &&
          MppCollapseRule.sameOutput(c2r.output, c2r.child.output)
        case _ => false
      }
    case _ => false
  }

  private def tryCollapseExistingRddHybrid(plan: SparkPlan): Option[MppNativeQueryExec] = {
    // Keep the hybrid fail-closed: only run row-shell repair after proving that the original plan
    // contains an exact JVM-backed ingress. A computed aggregate above that ingress can be left as
    // C2R -> row aggregate -> R2C by ordinary Gluten validation; repairing it here composes the two
    // independently supported shapes without admitting an arbitrary row subtree.
    if (!containsExactJvmStreamIngress(plan)) {
      return None
    }
    val repaired = repairRecoverableRowShells(plan, preserveExistingRddIngress = true)
    // Do not run InsertTransitions over this hybrid. Its first step deliberately removes every
    // transition, including the exact R2C that identifies the JVM-backed ingress; rebuilding that
    // transition is not guaranteed before backend component initialization and would erase the
    // boundary this path is required to validate. The plan has already passed Gluten's
    // post-transform rules, so retain the explicit ingress while applying only the transition-safe
    // native rewrites.
    val unionRewritten =
      MppReplicatedCartesianRule()(rewriteMppNativeUnion(MppColumnarTransitionBridge()(repaired)))
    if (!containsExactJvmStreamIngress(unionRewritten)) {
      return None
    }
    // An exchange directly over a JVM-backed ingress still needs a native producer fragment on its
    // side. Without this identity anchor, dynamic extraction sees the consumer ReadRel slot but
    // has neither a producer fragment nor a captured local stream for it. Anchor only the exact
    // R2C child that the strict hybrid validator already admits.
    val exchangeInputsAnchored = unionRewritten.transformUp {
      case exchange: ShuffleExchangeLike if isExactJvmStreamIngress(exchange.child) =>
        exchange.withNewChildren(Seq(ProjectExecTransformer(exchange.child.output, exchange.child)))
    }
    // A DataFrame created directly from a JVM-backed input can reach a V2 writer without any native
    // relational suffix. The exact R2C ingress is a local stream input, not a native fragment by
    // itself, so anchor this otherwise transparent plan with an identity native project.  This
    // preserves output attributes and gives ColumnarCollapseTransformStages a TransformSupport
    // consumer from which dynamic MPP extraction can build one real fragment.
    val nativeAnchored =
      if (exchangeInputsAnchored.find(_.isInstanceOf[TransformSupport]).isEmpty) {
        ProjectExecTransformer(exchangeInputsAnchored.output, exchangeInputsAnchored)
      } else {
        exchangeInputsAnchored
      }
    tryCollapseNormalizedMpp(
      nativeAnchored,
      allowExistingRddIngress = true,
      mode = "native with exact JVM-backed ingress")
  }

  /**
   * An isolated C2R below another operator is an execution boundary, not a convention adapter.
   * Permit only the two existing normalization shapes: an adjacent R2C(C2R(native)) pair, or the
   * identity C2R immediately below a Gluten columnar shuffle. The explicit terminal root C2R has
   * already been peeled by [[collapseTerminalRowEgress]] before this check.
   */
  private[extension] def findUnpairedNestedRowOutput(plan: SparkPlan): Option[SparkPlan] = {
    def isC2r(node: SparkPlan): Boolean = node match {
      case _: ColumnarToRowExecBase | _: ColumnarToRowExec => true
      case _ => false
    }

    def childOf(node: SparkPlan): SparkPlan = node.children.head

    def loop(node: SparkPlan): Option[SparkPlan] = node match {
      case r2c: RowToColumnarExecBase if isC2r(r2c.child) => loop(childOf(r2c.child))
      case r2c: RowToColumnarExec if isC2r(r2c.child) => loop(childOf(r2c.child))
      case exchange: ColumnarShuffleExchangeExecBase if isC2r(exchange.child) =>
        loop(childOf(exchange.child))
      case c2r: ColumnarToRowExecBase => Some(c2r)
      case c2r: ColumnarToRowExec => Some(c2r)
      case other => other.children.iterator.map(loop).collectFirst { case Some(found) => found }
    }

    loop(plan)
  }

  private def tryCollapseNormalizedMpp(
      unionRewritten: SparkPlan,
      allowExistingRddIngress: Boolean,
      mode: String): Option[MppNativeQueryExec] = {
    arrowScalarNormalizationRejection(unionRewritten).foreach {
      reason =>
        if (failOnFallback) {
          throw new IllegalStateException(s"MppCollapseRule: strict MPP rejected plan: $reason")
        }
    }
    if (!isNativeSupported(unionRewritten, allowExistingRddIngress)) {
      val reason =
        findFirstUnsupportedOperator(unionRewritten, allowExistingRddIngress).getOrElse("unknown")
      logWarning(
        s"MppCollapseRule: plan contains non-native operators, " +
          s"cannot collapse to MPP. First blocker: $reason. " +
          s"Plan root: ${unionRewritten.getClass.getSimpleName}. " +
          s"Plan: ${unionRewritten.treeString.take(500)}")
      return None
    }

    logWarning(s"MppCollapseRule: plan is $mode, wrapping with MppNativeQueryExec")

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
        s"All stages will run concurrently with streaming exchange. mode=$mode")

    // Plan D: Wrap, don't replace. Keep original plan as child so Spark's
    // shuffle/broadcast validation passes. At execution time, MppNativeQueryExec
    // bypasses child.executeColumnar() and runs via MppQueryCoordinator instead.
    Some(MppNativeQueryExec(child = rewritten, fragments = fragments, exchanges = exchanges))
  }

  private[extension] def normalizeMppNativeOperators(plan: SparkPlan): SparkPlan =
    normalizeMppNativeOperators(plan, preserveExistingRddIngress = false, recoverGenerate = false)

  private def normalizeMppNativeOperators(
      plan: SparkPlan,
      preserveExistingRddIngress: Boolean,
      recoverGenerate: Boolean = false): SparkPlan =
    normalizeMppNativeOperators(
      plan,
      preserveExistingRddIngress,
      recoverGenerate,
      rewriteNativeUnion = rewriteMppNativeUnion)

  private[extension] def normalizeMppNativeOperators(
      plan: SparkPlan,
      preserveExistingRddIngress: Boolean,
      rewriteNativeUnion: SparkPlan => SparkPlan): SparkPlan =
    normalizeMppNativeOperators(
      plan,
      preserveExistingRddIngress,
      recoverGenerate = false,
      rewriteNativeUnion)

  private def normalizeMppNativeOperators(
      plan: SparkPlan,
      preserveExistingRddIngress: Boolean,
      recoverGenerate: Boolean,
      rewriteNativeUnion: SparkPlan => SparkPlan): SparkPlan = {
    // Spark may insert RowToColumnar(ColumnarToRow(nativeChild)) solely to reconcile the
    // convention expected by an exchange or V2 writer. MPP absorbs that boundary, so eliminate
    // the adjacent inverse transitions before validating/extracting fragments. This preserves the
    // original ColumnarBatch stream; it does not whitelist a real row execution island.
    val transitionBridged = MppColumnarTransitionBridge()(plan)
    // Spark's DataFrame.coalesce(N>1) is only a narrow RDD/file-count hint; it
    // neither changes rows nor establishes a semantic distribution.  Keeping
    // ColumnarCoalesceExec would create a Spark scheduling island inside an
    // otherwise native query when applications request more output files than physical peers.
    // MPP owns peer parallelism and the Iceberg-required exchange immediately
    // above this node still enforces the actual write partitioning.
    val coalesceElided = transitionBridged.transformUp {
      case coalesce: ColumnarCoalesceExec if coalesce.numPartitions > 1 =>
        logInfo(
          s"MppCollapseRule: eliding non-semantic ColumnarCoalesceExec(" +
            s"${coalesce.numPartitions}) inside MPP query")
        coalesce.child
    }
    // Native aggregate and sort relations require their computed arguments to be materialized as
    // fields. The normal heuristic rewrite may leave a row operator behind after one sibling
    // fails validation, so repeat the official pre-project rewrite before force-offloading the
    // strict-MPP subtree. This is expression-preserving and does not broaden native validation.
    val preProjected = coalesceElided.transformUp {
      case agg: BaseAggregateExec => PullOutPreProject.rewrite(agg)
      case sort: SortExec if !sort.global => PullOutPreProject.rewrite(sort)
      case generate: GenerateExec if recoverGenerate => PullOutPreProject.rewrite(generate)
    }
    val postProjected = preProjected.transformUp {
      case generate: GenerateExec if recoverGenerate => PullOutPostProject.rewrite(generate)
    }
    // RAS may leave a vanilla Spark aggregate behind when its generic
    // profitability/validation pass declines a very wide aggregate. MPP has a
    // stricter end-to-end contract and validates the generated native/cuDF plan
    // later, so materialize the native aggregate transformer here instead of
    // accepting a row/BSP island.
    val aggregateRewritten = postProjected.transformUp {
      case agg: BaseAggregateExec if !agg.isInstanceOf[HashAggregateExecBaseTransformer] =>
        HashAggregateExecBaseTransformer.from(agg)
    }
    val lateOffloaded = aggregateRewritten.transformUp {
      case project: ProjectExec => ProjectExecTransformer(project.projectList, project.child)
      case filter: FilterExec => FilterExecTransformer(filter.condition, filter.child)
      case sort: SortExec if !sort.global =>
        SortExecTransformer(sort.sortOrder, global = false, sort.child, sort.testSpillFrequency)
      case generate: GenerateExec
          if recoverGenerate && GenerateExecTransformer.supportsGenerate(generate.generator) =>
        BackendsApiManager.getSparkPlanExecApiInstance.genGenerateTransformer(
          generate.generator,
          generate.requiredChildOutput,
          generate.outer,
          generate.generatorOutput,
          generate.child)
    }
    // The incoming plan's transitions were selected before the row aggregate/sort/project was
    // replaced above. Re-run Gluten's convention planner so it removes only stale transitions and
    // re-inserts every boundary still required by a genuine row operator. This keeps DataFrame.rdd
    // and other real row consumers as strict-MPP negatives.
    // ExistingRDD recovery must see a validated native union rather than ColumnarUnionExec;
    // otherwise its strict native-tree check retains the now-stale C2R/R2C shell. The ordinary
    // path still rewrites after transition insertion, where output partitioning is fully known.
    val unionPrepared =
      if (preserveExistingRddIngress) rewriteNativeUnion(lateOffloaded) else lateOffloaded
    val retransitioned =
      if (preserveExistingRddIngress) {
        bridgeRecoverableExistingRddTransitions(unionPrepared)
      } else {
        InsertTransitions
          .create(outputsColumnar = unionPrepared.supportsColumnar, VeloxBatchType)
          .apply(unionPrepared)
      }
    val unionRewritten = rewriteNativeUnion(retransitioned)
    MppReplicatedCartesianRule()(unionRewritten)
  }

  /**
   * Remove only the stale transitions around a row shell that was just late-offloaded, while
   * retaining the exact R2C leaf that identifies the JVM-backed local stream.
   */
  private def bridgeRecoverableExistingRddTransitions(plan: SparkPlan): SparkPlan = {
    val withoutIngressC2r = plan.transformUp {
      case c2r: ColumnarToRowExecBase if isSupportedExistingRddHybridPlan(c2r.child) =>
        c2r.child
      case c2r: ColumnarToRowExec if isSupportedExistingRddHybridPlan(c2r.child) => c2r.child
    }
    withoutIngressC2r.transformUp {
      case r2c: RowToColumnarExecBase if isSupportedExistingRddHybridPlan(r2c.child) =>
        r2c.child
      case r2c: RowToColumnarExec if isSupportedExistingRddHybridPlan(r2c.child) =>
        r2c.child
    }
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
  private[extension] def isFullyNativeSupported(plan: SparkPlan): Boolean =
    isNativeSupported(plan, allowExistingRddIngress = false)

  /**
   * Validate a native suffix with one of the exact JVM-backed ingresses without weakening the
   * ordinary strict-MPP contract. At least one exact ingress must be present; every other node
   * remains subject to the same native validator.
   */
  private[extension] def isSupportedExistingRddHybridPlan(plan: SparkPlan): Boolean =
    containsExactJvmStreamIngress(plan) &&
      isNativeSupported(plan, allowExistingRddIngress = true)

  private def containsExactJvmStreamIngress(plan: SparkPlan): Boolean =
    plan.find(isExactJvmStreamIngress).isDefined

  private def isExactJvmStreamIngress(plan: SparkPlan): Boolean = plan match {
    case r2c: RowToColumnarExecBase => MppExistingRddStreamInput.rowInput(r2c).isDefined
    case r2c: RowToColumnarExec => MppExistingRddStreamInput.rowInput(r2c).isDefined
    case _ => false
  }

  private def isNativeSupported(plan: SparkPlan, allowExistingRddIngress: Boolean): Boolean = {
    plan match {
      // Exchanges that we can absorb into the MPP plan
      case exchange: ShuffleExchangeLike =>
        canAbsorbExchange(exchange, allowExistingRddIngress) &&
        isNativeSupported(exchange.child, allowExistingRddIngress)

      // Broadcast exchanges are absorbable when the build side is itself a fully
      // native TransformSupport subtree (recurse into children, same as shuffle).
      // The native MPP runtime treats BROADCAST as a distinct exchange type.
      case bc: BroadcastExchangeLike =>
        val absorbable = canAbsorbExchange(bc, allowExistingRddIngress) &&
          bc.children.forall(isNativeSupported(_, allowExistingRddIngress))
        if (!absorbable) {
          logWarning(
            s"MppCollapseRule: BLOCKED by BroadcastExchangeLike: " +
              s"${bc.getClass.getSimpleName} (build side not fully native)")
        }
        absorbable

      // AQE query stage wrappers - check their underlying plan
      case stage: ShuffleQueryStageExec =>
        isNativeSupported(stage.plan, allowExistingRddIngress)

      case stage: BroadcastQueryStageExec =>
        isNativeSupported(stage.plan, allowExistingRddIngress)

      // This transformer intentionally retains Spark's row-UDF eval type because changing a
      // nullable input to the ordinary Arrow scalar protocol is not semantics-preserving. It may
      // implement TransformSupport for registered native UDFs in general, but this tagged instance
      // must not be mistaken for the safe external Arrow B1 boundary.
      case python: EvalPythonExecTransformer
          if python.getTagValue(ARROW_SCALAR_NORMALIZATION_REJECTION_TAG).isDefined =>
        logWarning(
          s"MppCollapseRule: BLOCKED by unsafe ordinary Arrow scalar normalization: " +
            python.getTagValue(ARROW_SCALAR_NORMALIZATION_REJECTION_TAG).get)
        false

      // Native-supported operators
      case _: TransformSupport =>
        plan.children.forall(isNativeSupported(_, allowExistingRddIngress))

      // Exact matched rowInput is the only permitted JVM-backed input slot. Do not recurse into
      // it: MppNativeQueryExec captures and columnarizes the row plan as a local stream.
      case r2c: RowToColumnarExecBase
          if allowExistingRddIngress && MppExistingRddStreamInput.rowInput(r2c).isDefined =>
        true
      case r2c: RowToColumnarExec
          if allowExistingRddIngress && MppExistingRddStreamInput.rowInput(r2c).isDefined =>
        true

      // A remaining row transition is a real execution boundary. The only C2R shape that MPP may
      // elide is an identity C2R directly over Gluten's ColumnarExchange under a V2 write; that
      // narrowly-scoped normalization runs before this validation. In particular, a top-level
      // terminal root consumer is admitted only by collapseTerminalRowEgress, outside this strict
      // recursive validator; the same C2R at any nested position remains rejected.
      case c2r: ColumnarToRowExecBase =>
        logWarning(
          s"MppCollapseRule: BLOCKED by native-to-row execution boundary: " +
            describeExecutionBoundary(c2r, "native-to-row"))
        false
      case c2r: ColumnarToRowExec =>
        logWarning(
          s"MppCollapseRule: BLOCKED by native-to-row execution boundary: " +
            describeExecutionBoundary(c2r, "native-to-row"))
        false
      case r2c: RowToColumnarExecBase =>
        logWarning(
          s"MppCollapseRule: BLOCKED by row-to-native execution boundary: " +
            describeExecutionBoundary(r2c, "row-to-native"))
        false
      case r2c: RowToColumnarExec =>
        logWarning(
          s"MppCollapseRule: BLOCKED by row-to-native execution boundary: " +
            describeExecutionBoundary(r2c, "row-to-native"))
        false

      // Gluten-internal columnar-to-columnar nodes (batch resize, etc.) - look through
      case c2c: ColumnarToColumnarExec =>
        c2c.children.forall(isNativeSupported(_, allowExistingRddIngress))

      // ColumnarCollapseTransformStages places this convention adapter below an
      // InputIteratorTransformer at exchange boundaries. It is transparent to MPP fragment
      // extraction, so validate the wrapped exchange/native subtree instead of rejecting the
      // adapter itself. This is also the shape produced by the scalar-subquery broadcast rewrite.
      case cia: ColumnarInputAdapter =>
        isNativeSupported(cia.child, allowExistingRddIngress)

      // TakeOrderedAndProjectExecTransformer wraps a sort+limit+project over a
      // TransformSupport child; treat it as a transparent wrapper and recurse.
      case topk: TakeOrderedAndProjectExecTransformer =>
        isNativeSupported(topk.child, allowExistingRddIngress)

      // Vanilla FilterExec/ProjectExec only land here when Gluten's columnar
      // validator refused to convert them -- typically because their expression
      // tree contains an uncorrelated ScalarSubquery (Q11, Q22). The subquery's
      // result is materialized by Spark's ReusedSubqueryExec mechanism and
      // injected as a Substrait literal at build time by ScalarSubqueryTransformer.
      // Whitelist these so the outer plan can collapse to MPP.
      case f: FilterExec if containsScalarSubquery(f.condition) =>
        isNativeSupported(f.child, allowExistingRddIngress)
      case p: ProjectExec if p.projectList.exists(containsScalarSubquery) =>
        isNativeSupported(p.child, allowExistingRddIngress)

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
  private def findFirstUnsupportedOperator(
      plan: SparkPlan,
      allowExistingRddIngress: Boolean): Option[String] = {
    def findInChildren(node: SparkPlan): Option[String] =
      node.children
        .flatMap(findFirstUnsupportedOperator(_, allowExistingRddIngress))
        .headOption

    plan match {
      case _: ShuffleExchangeLike if canAbsorbExchange(plan, allowExistingRddIngress) =>
        findInChildren(plan)
      case _: BroadcastExchangeLike if canAbsorbExchange(plan, allowExistingRddIngress) =>
        findInChildren(plan)
      case _: ShuffleQueryStageExec =>
        findFirstUnsupportedOperator(
          plan.asInstanceOf[ShuffleQueryStageExec].plan,
          allowExistingRddIngress)
      case _: BroadcastQueryStageExec =>
        findFirstUnsupportedOperator(
          plan.asInstanceOf[BroadcastQueryStageExec].plan,
          allowExistingRddIngress)
      case python: EvalPythonExecTransformer
          if python.getTagValue(ARROW_SCALAR_NORMALIZATION_REJECTION_TAG).isDefined =>
        Some(
          s"${python.getClass.getSimpleName}: " +
            python.getTagValue(ARROW_SCALAR_NORMALIZATION_REJECTION_TAG).get)
      case _: TransformSupport =>
        findInChildren(plan)
      case r2c: RowToColumnarExecBase
          if allowExistingRddIngress && MppExistingRddStreamInput.rowInput(r2c).isDefined =>
        None
      case r2c: RowToColumnarExec
          if allowExistingRddIngress && MppExistingRddStreamInput.rowInput(r2c).isDefined =>
        None
      case c2r: ColumnarToRowExecBase =>
        Some(describeExecutionBoundary(c2r, "native-to-row"))
      case c2r: ColumnarToRowExec =>
        Some(describeExecutionBoundary(c2r, "native-to-row"))
      case r2c: RowToColumnarExecBase =>
        Some(describeExecutionBoundary(r2c, "row-to-native"))
      case r2c: RowToColumnarExec =>
        Some(describeExecutionBoundary(r2c, "row-to-native"))
      case _: ColumnarToColumnarExec =>
        findInChildren(plan)
      case cia: ColumnarInputAdapter =>
        findFirstUnsupportedOperator(cia.child, allowExistingRddIngress)
      case _: TakeOrderedAndProjectExecTransformer =>
        findInChildren(plan)
      case f: FilterExec if containsScalarSubquery(f.condition) =>
        findInChildren(plan)
      case p: ProjectExec if p.projectList.exists(containsScalarSubquery) =>
        findInChildren(plan)
      case other =>
        Some(s"${other.getClass.getSimpleName}: ${other.simpleString(20)}")
    }
  }

  /**
   * Report the row operator and Gluten validation tag hidden immediately below a convention
   * transition. A transition class alone is not actionable: it says where native execution ends,
   * but not why Spark created the row island. Keep this diagnostic read-only; a real row operator
   * is still rejected by strict MPP.
   */
  private[extension] def describeExecutionBoundary(
      boundary: SparkPlan,
      direction: String): String = {
    val child = boundary.children.headOption
    val childText = child
      .map(plan => s"child=${plan.getClass.getSimpleName}(${plan.simpleString(100)})")
      .getOrElse("child=<none>")
    val fallbackText = child
      .flatMap(firstFallbackReason)
      .map {
        case (node, reason) =>
          s"; firstFallback=${node.getClass.getSimpleName}: $reason"
      }
      .getOrElse("")
    s"${boundary.getClass.getSimpleName}: $direction execution boundary; " +
      s"$childText$fallbackText"
  }

  private def firstFallbackReason(plan: SparkPlan): Option[(SparkPlan, String)] = {
    plan.collect {
      case node if FallbackTags.nonEmpty(node) =>
        (node, FallbackTags.get(node).reason())
    }.headOption
  }

  private def arrowScalarNormalizationRejection(plan: SparkPlan): Option[String] = {
    plan.collect {
      case python: EvalPythonExecTransformer
          if python.getTagValue(ARROW_SCALAR_NORMALIZATION_REJECTION_TAG).isDefined =>
        python.getTagValue(ARROW_SCALAR_NORMALIZATION_REJECTION_TAG).get
    }.headOption
  }

  /**
   * Check whether an exchange can be absorbed into the MPP plan.
   *
   * An exchange is absorbable when:
   *   - The child (producer side) is a TransformSupport subtree
   *   - The partitioning type is one we support for GPU streaming exchange
   *   - There are no UDFs that require JVM execution in the exchange
   */
  private def canAbsorbExchange(exchange: SparkPlan, allowExistingRddIngress: Boolean): Boolean = {
    exchange match {
      case shuffle: ShuffleExchangeLike =>
        val partitioningSupported = isSupportedPartitioning(shuffle.outputPartitioning)
        val childSupported = shuffle.child.isInstanceOf[TransformSupport] ||
          shuffle.child.isInstanceOf[ShuffleQueryStageExec] ||
          shuffle.child.isInstanceOf[BroadcastQueryStageExec] ||
          shuffle.child.isInstanceOf[ColumnarToColumnarExec] ||
          (allowExistingRddIngress && isExactJvmStreamIngress(shuffle.child))

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
        case r2c: RowToColumnarExecBase => isExchangeBoundary(r2c.child)
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
        case r2c: RowToColumnarExecBase => unwrapToExchange(r2c.child)
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
        // Preserve the meaningful projection expressions and remove only the
        // synthetic prefix inserted for Spark's shuffle implementation.
        p.copy(projectList = p.projectList.tail)
      case wst: WholeStageTransformer =>
        val strippedChild = stripSyntheticHashProject(wst.child)
        if (strippedChild eq wst.child) wst
        else wst.withNewChildren(Seq(strippedChild))
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
            partitionKeys = partitionKeys,
            rangeOrdering = shuffle.outputPartitioning match {
              case range: RangePartitioning => range.ordering
              case _ => Seq.empty
            },
            rangeSamplePlan = shuffle.child
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

        case r2c: RowToColumnarExecBase =>
          walk(r2c.child)

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

object MppCollapseRule {

  private[extension] case class V2WriteQueryNormalization(plan: SparkPlan, strippedAdapters: Int)

  /** True only when a final row adapter is statically bounded to a single global-aggregate row. */
  private[extension] def isBoundedScalarResult(plan: SparkPlan): Boolean = plan match {
    case aggregate: BaseAggregateExec => aggregate.groupingExpressions.isEmpty
    case _ => false
  }

  /**
   * Remove convention adapters Spark inserts while preparing a V2 write query.
   *
   * This is deliberately scoped to the V2-write call site. Its schema-identical top-level C2R is
   * not an observable row consumer: the V2 writer immediately consumes the rows, and
   * MppNativeQueryExec.doExecute supplies that one required native-to-row conversion. Removing the
   * explicit adapter lets the complete columnar query (including sort and exchange) collapse to MPP
   * without an intermediate C2R -> R2C round trip. Exact C2R(ColumnarShuffleExchange) adapters may
   * also appear below the root and remain safe to remove. Any adapter around a real row child
   * remains untouched.
   */
  private[extension] def normalizeV2WriteExchangeAdapters(
      plan: SparkPlan): V2WriteQueryNormalization = {
    var stripped = 0
    val withoutTopLevelAdapter = plan match {
      case c2r: ColumnarToRowExec if isTransparentTopLevelWriteAdapter(c2r, c2r.child) =>
        stripped += 1
        c2r.child
      case c2r: ColumnarToRowExecBase
          if c2r.children.size == 1 &&
            isTransparentTopLevelWriteAdapter(c2r, c2r.children.head) =>
        stripped += 1
        c2r.children.head
      case other => other
    }
    val normalized = withoutTopLevelAdapter.transformUp {
      case c2r: ColumnarToRowExec if isTransparentWriteExchangeAdapter(c2r, c2r.child) =>
        stripped += 1
        c2r.child
      case c2r: ColumnarToRowExecBase
          if c2r.children.size == 1 &&
            isTransparentWriteExchangeAdapter(c2r, c2r.children.head) =>
        stripped += 1
        c2r.children.head
    }
    V2WriteQueryNormalization(normalized, stripped)
  }

  private def isTransparentTopLevelWriteAdapter(adapter: SparkPlan, child: SparkPlan): Boolean =
    child.supportsColumnar && sameOutput(adapter.output, child.output)

  private def isTransparentWriteExchangeAdapter(adapter: SparkPlan, child: SparkPlan): Boolean =
    child match {
      case exchange: ColumnarShuffleExchangeExecBase =>
        exchange.supportsColumnar && sameOutput(adapter.output, exchange.output)
      case _ => false
    }

  private def sameOutput(left: Seq[Attribute], right: Seq[Attribute]): Boolean =
    left.length == right.length && left.zip(right).forall {
      case (l, r) =>
        l.exprId == r.exprId && l.dataType == r.dataType && l.nullable == r.nullable
    }
}
