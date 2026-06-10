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
package org.apache.gluten.execution

import org.apache.gluten.backendsapi.BackendsApiManager
import org.apache.gluten.config.GlutenConfig
import org.apache.gluten.events.{GlutenMppPlanEvent, GlutenMppPlanFragmentEvent}
import org.apache.gluten.expression.ConverterUtils
import org.apache.gluten.extension.{ExchangeSpec, MppFinalAggTopNPartialRule, MppParallelSortSplitRule, MppRemoveRedundantShuffleRule, MppSinglePartitionSortRule, NativeFragment, RewriteUncorrelatedScalarSubquery}
import org.apache.gluten.extension.columnar.transition.{Convention, ConventionReq}
import org.apache.gluten.metrics.MetricsUpdater
import org.apache.gluten.mpp.control.{GlutenMppPeerResolution, GlutenMppPeerResolver}
import org.apache.gluten.runtime.Runtimes
import org.apache.gluten.substrait.SubstraitContext
import org.apache.gluten.substrait.plan.PlanBuilder
import org.apache.gluten.substrait.rel.SplitInfo
import org.apache.gluten.utils.SubstraitPlanPrinterUtil
import org.apache.gluten.vectorized.MppQueryJniWrapper

import org.apache.spark.SparkEnv
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.SortOrder
import org.apache.spark.sql.catalyst.expressions.aggregate.{Complete, Final, Partial}
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight, BuildSide, JoinSelectionHelper}
import org.apache.spark.sql.catalyst.plans.{InnerLike, LeftOuter, LeftSemi, RightOuter}
import org.apache.spark.sql.catalyst.plans.logical.{Join, Statistics}
import org.apache.spark.sql.catalyst.plans.physical.{BroadcastPartitioning, HashPartitioning, Partitioning, RangePartitioning, RoundRobinPartitioning, SinglePartition}
import org.apache.spark.sql.execution.{ColumnarBroadcastExchangeExec, ColumnarCollapseTransformStages, ColumnarInputAdapter, ExecSubqueryExpression, InputIteratorTransformer, LeafExecNode, SortExec, SparkPlan, SQLExecution, UnaryExecNode}
import org.apache.spark.sql.execution.adaptive.{BroadcastQueryStageExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeLike, Exchange, ReusedExchangeExec, ShuffleExchangeExec, ShuffleExchangeLike}
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, BuildSideRelation, ShuffledHashJoinExec}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.ui.GlutenUIUtils
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.ColumnarBatch

import com.google.common.collect.Lists
import io.substrait.proto.ReadRel

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * A single SparkPlan node that represents the entire MPP query execution.
 *
 * From Spark's perspective, this is a leaf node producing the final query result. Internally, it
 * contains multiple [[NativeFragment]]s connected by [[ExchangeSpec]]s that describe the streaming
 * GPU exchange topology.
 *
 * During execution (M4 JNI integration phase), this node will:
 *   1. Serialize all fragments' Substrait plans and exchange specs into a single MPP execution
 *      descriptor. 2. Submit the descriptor to the native runtime via JNI. 3. The native runtime
 *      launches all fragments concurrently, connected by GPU streaming exchanges
 *      (OutputBufferManager / GpuExchange). 4. The final fragment's output is returned to Spark as
 *      an RDD[ColumnarBatch].
 *
 * @param fragments
 *   The ordered list of native execution fragments.
 * @param exchanges
 *   The exchange specifications connecting fragments.
 * @param originalPlan
 *   The original SparkPlan before MPP collapse (for explain/debugging).
 */
/**
 * Plan D: Wrap, don't replace.
 *
 * MppNativeQueryExec wraps the original plan as its child (UnaryExecNode). Spark sees the original
 * plan (including ShuffleExchange nodes) still intact, so the "cannot transform shuffle node"
 * validation passes.
 *
 * At execution time, doExecuteColumnar() does NOT call child.executeColumnar(). Instead, it uses
 * the child plan only to extract Substrait fragments, then executes via JNI -> MppQueryCoordinator
 * (streaming exchange).
 */
/**
 * Identifies a fused broadcast: one whose build subtree was inlined into the consumer fragment's
 * substrait (no separate producer fragment), but which still occupies an `iterator:N` ReadRel slot
 * in that consumer's plan. The runtime side must pre-populate `placeholderIters[slotIdx]` with a
 * BatchIterator over the broadcasted ColumnarBatches before plan conversion, otherwise
 * `SubstraitToVeloxPlan.cc:constructCudfValueStreamNode` hits `streamIdx N vs size N` OOB.
 *
 * Slot index is the position in `findExchangeChildren(wst)` walk order -- the same DFS order Gluten
 * uses to assign `iterator:0, iterator:1, ...` in substrait.
 */
case class FusedBroadcast(slotIdx: Int, broadcast: Broadcast[BuildSideRelation])

private case class MppPartitioningPreservingWrapper(
    child: SparkPlan,
    partitioning: Partitioning,
    ordering: Seq[SortOrder],
    exchangeId: Int)
  extends UnaryTransformSupport {

  override def output: Seq[Attribute] = child.output

  override def outputPartitioning: Partitioning = partitioning

  override def outputOrdering: Seq[SortOrder] = ordering

  override def nodeName: String = s"MppPartitioningPreservingWrapper(exchange=$exchangeId)"

  override protected def doTransform(context: SubstraitContext): TransformContext = {
    val childCtx = child.asInstanceOf[TransformSupport].transform(context)
    TransformContext(output, childCtx.root)
  }

  override def metricsUpdater(): MetricsUpdater = MetricsUpdater.None

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = {
    copy(child = newChild)
  }
}

private[gluten] case class MppPreparedChildExec(hiddenPlan: SparkPlan)
  extends LeafExecNode
  with GlutenPlan {
  private lazy val convention = Convention.get(hiddenPlan)

  override def batchType(): Convention.BatchType = convention.batchType
  override def rowType0(): Convention.RowType = convention.rowType
  override def output: Seq[Attribute] = hiddenPlan.output
  override def outputPartitioning: Partitioning = hiddenPlan.outputPartitioning
  override def outputOrdering: Seq[SortOrder] = hiddenPlan.outputOrdering

  override protected def doExecute(): RDD[InternalRow] =
    throw new UnsupportedOperationException("MppPreparedChildExec is only a prepare-time wrapper")

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] =
    throw new UnsupportedOperationException("MppPreparedChildExec is only a prepare-time wrapper")
}

case class MppNativeQueryExec(
    child: SparkPlan,
    fragments: Seq[NativeFragment],
    exchanges: Seq[ExchangeSpec],
    @transient originalLogicalPlan: org.apache.spark.sql.catalyst.plans.logical.LogicalPlan = null
) extends UnaryExecNode
  with GlutenPlan
  with Logging {

  // --- Output schema and partitioning (delegate to child) ---

  private def preparedChildPlan: SparkPlan = unwrapPreparedMppChild(child)

  override def output: Seq[Attribute] = preparedChildPlan.output
  override def outputPartitioning: Partitioning = preparedChildPlan.outputPartitioning
  override def outputOrdering: Seq[SortOrder] = preparedChildPlan.outputOrdering

  // --- Convention support for GlutenPlan ---

  override def batchType(): Convention.BatchType = {
    // MPP execution produces native columnar batches
    org.apache.gluten.backendsapi.BackendsApiManager.getSettings.primaryBatchType
  }

  override def rowType0(): Convention.RowType = Convention.RowType.None

  // Must match children.size (1 child = MppSchemaOnlyExec or planLater)
  override def requiredChildConvention(): Seq[ConventionReq] = {
    children.map(_ => ConventionReq.any)
  }

  private val MPP_PEER_ENDPOINTS_KEY = "spark.gluten.mpp.peerEndpoints"

  private def newNativeMppQueryId(): String = {
    s"spark-${UUID.randomUUID().toString.replace("-", "")}"
  }

  // --- Metrics ---

  @transient
  override lazy val metrics: Map[String, SQLMetric] = Map(
    "totalQueryTimeMs" -> SQLMetrics.createTimingMetric(sparkContext, "total query time (ms)"),
    "numFragments" -> SQLMetrics.createMetric(sparkContext, "number of fragments"),
    "numExchanges" -> SQLMetrics.createMetric(sparkContext, "number of exchanges"),
    "numSparkPartitions" -> SQLMetrics.createMetric(sparkContext, "number of Spark partitions"),
    "outputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "outputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches")
  )

  // --- Execution ---

  override def doPrepare(): Unit = {
    // MPP bypasses child.executeColumnar(), so preparing the physical child tree here is both
    // unnecessary and harmful: Spark eagerly starts stale scalar-subquery and broadcast jobs that
    // the native MPP rewrite will consume through exchange fragments.
    logWarning("MppNativeQueryExec: skipping child prepare; MPP will prepare required inputs")
  }

  override protected def doExecute(): RDD[InternalRow] = {
    // Gluten ColumnarBatch uses IndicatorVectorBase (native-pointer wrappers),
    // which does not support Spark's standard row-access path
    // (ColumnarBatchRow.copy -> isNullAt throws UnsupportedOperationException).
    // Delegate to VeloxColumnarToRowExec, which does the conversion via JNI.
    VeloxColumnarToRowExec(this).doExecute()
  }

  // supportsColumnar is already true via GlutenPlan

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val executionChild = preparedChildPlan
    logWarning(
      s"MppNativeQueryExec: executing with ${fragments.size} fragments " +
        s"and ${exchanges.size} exchanges")

    // Fold uncorrelated scalar subqueries into the native plan as broadcast inputs before any
    // driver-side scalar materialization. Plan D already applies this rewrite in MppCollapseRule,
    // but Plan C wraps a child plan before the post-rule pass can see it.
    val childForMpp = RewriteUncorrelatedScalarSubquery(executionChild)

    // Materialize any remaining ScalarSubquery results in the child plan before driver-side
    // Substrait generation. ScalarSubqueryTransformer.doTransform calls
    // query.eval(InternalRow.empty), which requires updateResult() to have run.
    // In BSP this happens via SparkPlan.prepareSubqueries on the enclosing plan;
    // because MppNativeQueryExec bypasses child.executeColumnar() we drive it
    // explicitly here. Uncorrelated scalar subqueries should have been rewritten above and will
    // flow through native BROADCAST exchanges instead.
    materializeScalarSubqueries(childForMpp)

    // Plan D: child is the original BSP plan (with ShuffleExchange intact).
    // Phase 1: delegate to child BSP execution to prove the wrap chain works.
    // Phase 2: replace with real MPP multi-fragment streaming execution.
    val hasRealFragments = fragments.nonEmpty && fragments.head.rootOperator != null
    if (!hasRealFragments) {
      // Phase 2: Extract fragments from child BSP plan and generate Substrait plans.
      logWarning(s"MppNativeQueryExec: child plan tree:\n${childForMpp.treeString.take(2000)}")

      val (extractedFragments, extractedExchanges, fusedBroadcastsByConsumer) =
        extractFragmentsFromChildPlan(childForMpp)
      logWarning(
        s"MppNativeQueryExec: Phase 2 extracted ${extractedFragments.size} fragments " +
          s"and ${extractedExchanges.size} exchanges " +
          s"and ${fusedBroadcastsByConsumer.values.map(_.size).sum} fused " +
          s"broadcasts from child plan")

      // Log each fragment for debugging
      extractedFragments.foreach {
        frag =>
          val rootName = if (frag.rootOperator != null) {
            frag.rootOperator.getClass.getSimpleName
          } else "<null>"
          logWarning(
            s"  Fragment ${frag.id}: root=$rootName, " +
              s"output=${frag.outputAttributes.map(_.name).mkString("[", ", ", "]")}, " +
              s"parallelism=${frag.parallelism}")
      }
      extractedExchanges.foreach {
        ex =>
          logWarning(
            s"  Exchange ${ex.id}: F${ex.producerFragmentId} -> F${ex.consumerFragmentId} " +
              s"(${ex.exchangeType}, ${ex.numPartitions} partitions)")
      }

      nativeMppFallbackReason(extractedFragments, extractedExchanges).foreach {
        reason =>
          return delegateToBsp(
            executionChild,
            s"MppNativeQueryExec: delegating to BSP because extracted MPP plan is unsafe: $reason")
      }

      // Generate Substrait plan for each fragment
      val fragmentSubstraitPlans = extractedFragments.map {
        frag => generateSubstraitForFragment(frag)
      }

      logWarning(
        s"MppNativeQueryExec: generated ${fragmentSubstraitPlans.size} Substrait plans " +
          s"(sizes: ${fragmentSubstraitPlans.map(_.length).mkString("[", ", ", "]")} bytes)")

      if (extractedFragments.size >= 2 && extractedExchanges.nonEmpty) {
        // We have real fragments with exchanges -- use MPP streaming execution!
        logWarning(
          s"MppNativeQueryExec: *** PHASE 3 MPP EXECUTION *** " +
            s"${extractedFragments.size} fragments, ${extractedExchanges.size} exchanges")
        suppressDeadBroadcastsForNativeMpp(childForMpp)

        val fragmentPlans = fragmentSubstraitPlans.toArray
        val numDriversPerFragment = extractedFragments.map(_.parallelism).toArray
        val exchangeSpecsJson = serializeExchangeSpecs(extractedExchanges, extractedFragments)
        val mppQueryId = newNativeMppQueryId()
        val broadcastProducerFragmentIds = broadcastProducerIds(extractedExchanges)

        val requestedSparkPartitionCount = mppSparkPartitionCount
        val sparkPartitionCount =
          effectiveMppSparkPartitionCount(
            requestedSparkPartitionCount,
            extractedExchanges,
            childForMpp.find(_.isInstanceOf[WriteFilesExecTransformer]).isDefined)
        val peerResolution = resolveMppPeers(sparkPartitionCount)
        val peerInfos = peerResolution.peerInfos
        val peerEndpointsJson = peerResolution.peerEndpointsJson
        metrics("numSparkPartitions") += sparkPartitionCount

        // Extract scan split infos only after endpoint discovery. Some Spark exchange
        // wrappers can lazily start shuffle work while split metadata is inspected; probing first
        // keeps the UCX endpoint job from racing those regular Spark stages.
        val fragmentSplitInfos: Array[Array[Array[Byte]]] =
          extractedFragments.map(extractSplitInfosForFragment).toArray

        logWarning(
          s"MppNativeQueryExec: split infos per fragment: " +
            s"${fragmentSplitInfos.map(_.length).mkString("[", ", ", "]")}")

        // Optional dump-to-disk of the multi-fragment plan for offline C++ replay
        // (spark.gluten.mpp.substraitDumpDir). No-op when the config is empty.
        dumpPlanIfEnabled(
          fragmentPlans,
          numDriversPerFragment,
          extractedFragments,
          extractedExchanges,
          exchangeSpecsJson,
          fragmentSplitInfos)
        postMppPlanEvent(
          fragmentPlans,
          numDriversPerFragment,
          extractedFragments,
          extractedExchanges,
          exchangeSpecsJson,
          fragmentSplitInfos,
          fusedBroadcastsByConsumer)

        return new MppNativeQueryRDD(
          sparkContext,
          fragmentPlans,
          numDriversPerFragment,
          exchangeSpecsJson,
          mppQueryId,
          peerEndpointsJson,
          peerInfos,
          fragmentSplitInfos,
          fusedBroadcastsByConsumer,
          sparkPartitionCount,
          broadcastProducerFragmentIds,
          longMetric("totalQueryTimeMs"),
          longMetric("outputRows"),
          longMetric("outputBatches")
        )
      } else {
        // Single fragment or no exchanges -- BSP is fine
        logWarning(
          s"MppNativeQueryExec: single fragment (${extractedFragments.size} fragments, " +
            s"${extractedExchanges.size} exchanges), delegating to BSP")
        if (childForMpp.supportsColumnar) {
          return childForMpp.executeColumnar()
        } else {
          return childForMpp.execute().mapPartitions(rows => Iterator.empty)
        }
      }
    }

    nativeMppFallbackReason(fragments, exchanges).foreach {
      reason =>
        return delegateToBsp(
          executionChild,
          "MppNativeQueryExec: delegating to BSP because pre-extracted MPP plan is unsafe: " +
            reason)
    }

    suppressDeadBroadcastsForNativeMpp(childForMpp)

    // Update fragment/exchange count metrics.
    metrics("numFragments") += fragments.size
    metrics("numExchanges") += exchanges.size

    // Generate Substrait plans on the DRIVER side where sparkContext is available.
    val fragmentPlans: Array[Array[Byte]] = fragments.map {
      frag => generateSubstraitPlan(frag)
    }.toArray

    val numDriversPerFragment = fragments.map(_.parallelism).toArray
    val exchangeSpecsJson = serializeExchangeSpecs(exchanges, fragments)
    val mppQueryId = newNativeMppQueryId()
    val broadcastProducerFragmentIds = broadcastProducerIds(exchanges)

    logInfo(s"MppNativeQueryExec: generated ${fragmentPlans.length} Substrait plans on driver")

    val requestedSparkPartitionCount = mppSparkPartitionCount
    val sparkPartitionCount =
      effectiveMppSparkPartitionCount(
        requestedSparkPartitionCount,
        exchanges,
        childForMpp.find(_.isInstanceOf[WriteFilesExecTransformer]).isDefined)
    val peerResolution = resolveMppPeers(sparkPartitionCount)
    val peerInfos = peerResolution.peerInfos
    val peerEndpointsJson = peerResolution.peerEndpointsJson
    metrics("numSparkPartitions") += sparkPartitionCount

    // Probe endpoints before split extraction for the same scheduling reason as the child-plan
    // extraction path above.
    val fragmentSplitInfos: Array[Array[Array[Byte]]] =
      fragments.map(extractSplitInfosForFragment).toArray

    // Optional dump-to-disk of the multi-fragment plan for offline C++ replay
    // (spark.gluten.mpp.substraitDumpDir). No-op when the config is empty.
    dumpPlanIfEnabled(
      fragmentPlans,
      numDriversPerFragment,
      fragments,
      exchanges,
      exchangeSpecsJson,
      fragmentSplitInfos)
    postMppPlanEvent(
      fragmentPlans,
      numDriversPerFragment,
      fragments,
      exchanges,
      exchangeSpecsJson,
      fragmentSplitInfos,
      Map.empty[Int, Seq[FusedBroadcast]])

    // RDD only receives serialized bytes - no SparkPlan references.
    // Pre-built fragments path doesn't fuse broadcasts (legacy MppCollapseRule
    // path materializes them as separate fragments via BROADCAST exchange).
    new MppNativeQueryRDD(
      sparkContext,
      fragmentPlans,
      numDriversPerFragment,
      exchangeSpecsJson,
      mppQueryId,
      peerEndpointsJson,
      peerInfos,
      fragmentSplitInfos,
      Map.empty[Int, Seq[FusedBroadcast]],
      sparkPartitionCount,
      broadcastProducerFragmentIds,
      longMetric("totalQueryTimeMs"),
      longMetric("outputRows"),
      longMetric("outputBatches")
    )
  }

  // --- Explain / toString ---

  override def nodeName: String = s"MppNativeQuery"

  override def simpleString(maxFields: Int): String = {
    s"MppNativeQuery(${fragments.size} fragments, ${exchanges.size} exchanges)"
  }

  override def verboseStringWithOperatorId(): String = {
    val sb = new StringBuilder
    sb.append(s"MppNativeQuery (${fragments.size} fragments, ${exchanges.size} exchanges)\n")
    sb.append(fragmentSummary)
    sb.append(exchangeSummary)
    sb.toString()
  }

  override protected def withNewChildInternal(newChild: SparkPlan): MppNativeQueryExec = {
    copy(child = maybeHidePreparedMppChild(newChild), originalLogicalPlan = originalLogicalPlan)
  }

  private def delegateToBsp(plan: SparkPlan, reason: String): RDD[ColumnarBatch] = {
    logWarning(reason)
    if (plan.supportsColumnar) {
      plan.executeColumnar()
    } else {
      plan.execute().mapPartitions(_ => Iterator.empty)
    }
  }

  private def maybeHidePreparedMppChild(plan: SparkPlan): SparkPlan = {
    plan match {
      case _: MppPreparedChildExec => plan
      case other if containsGlutenColumnarPlan(other) =>
        MppPreparedChildExec(other)
      case other =>
        other
    }
  }

  private def unwrapPreparedMppChild(plan: SparkPlan): SparkPlan = {
    plan match {
      case prepared: MppPreparedChildExec => prepared.hiddenPlan
      case other => other
    }
  }

  private def containsGlutenColumnarPlan(plan: SparkPlan): Boolean = {
    plan match {
      case _: MppPreparedChildExec => true
      case _: TransformSupport | _: ColumnarToRowExecBase | _: ColumnarToColumnarExec => true
      case other => other.children.exists(containsGlutenColumnarPlan)
    }
  }

  // --- Phase 2: Fragment extraction from child BSP plan ---

  /**
   * Walk the child BSP plan tree and extract fragments and exchange boundaries.
   *
   * The child plan looks like:
   * {{{
   *   VeloxColumnarToRowExec
   *     WholeStageTransformer [Sort]           <- Fragment (final)
   *       InputIteratorTransformer
   *         ColumnarInputAdapter
   *           ColumnarShuffleExchangeExec      <- Exchange boundary
   *             VeloxResizeBatchesExec
   *               WholeStageTransformer [Agg]  <- Fragment (producer)
   * }}}
   *
   * Each [[WholeStageTransformer]] between exchange boundaries is a "fragment". Each
   * [[ShuffleExchangeLike]] is an "exchange boundary" connecting two fragments.
   *
   * We do NOT modify the child plan tree -- only read it.
   */
  private def extractFragmentsFromChildPlan(
      plan: SparkPlan): (Seq[NativeFragment], Seq[ExchangeSpec], Map[Int, Seq[FusedBroadcast]]) = {
    val extractedFragments = mutable.ArrayBuffer[NativeFragment]()
    val extractedExchanges = mutable.ArrayBuffer[ExchangeSpec]()
    val fragmentCounter = new AtomicInteger(0)
    val exchangeCounter = new AtomicInteger(0)
    // Track fused broadcasts per consumer fragment so we can pre-populate
    // their iterator slot at JNI hand-off time. Slot index is the position
    // in WST.findExchangeChildren walk order, which matches the substrait
    // iterator:N indexing emitted by InputIteratorTransformer.
    val broadcastsByConsumer =
      mutable.HashMap[Int, mutable.ArrayBuffer[FusedBroadcast]]()
    val replicatedJoinBuildExchangeIds = mutable.HashSet[Int]()

    /**
     * Walk the plan tree depth-first. Returns the fragment ID of the subtree rooted at `plan`. At
     * WholeStageTransformer: creates a new fragment. At ShuffleExchangeLike: creates an exchange
     * spec connecting producer to consumer. At wrapper nodes (ColumnarToRow, ColumnarToColumnar,
     * InputIterator, ColumnarInputAdapter): passes through to the child.
     */
    def walk(plan: SparkPlan): Int = {
      plan match {
        case wst: WholeStageTransformer =>
          // This WholeStageTransformer is a fragment. First, walk its children
          // to discover any exchange boundaries below it. The WST's doTransform()
          // stops at InputIteratorTransformer boundaries, which is exactly what
          // we want: each fragment's Substrait covers operators between exchanges.
          val exchangeChildren = findExchangeChildren(wst)
          val childExchangeFragIds = exchangeChildren.map(walk)

          val fragId = fragmentCounter.getAndIncrement()
          // Write-in-MPP: a fragment whose WST contains a WriteFilesExecTransformer must run
          // single-writer per Spark task attempt dir. With parallelism>1, multiple native
          // TableWrite drivers in one task emit the same part-NNNNN-<uuid> filename and the
          // second hits LocalWriteFile "File exists" (Spark's per-task write model is inherently
          // single-writer; local/BSP write never has >1 writer per attempt dir).
          val parallelism =
            if (wst.find(_.isInstanceOf[WriteFilesExecTransformer]).isDefined) {
              logWarning(
                s"MppNativeQueryExec: write fragment $fragId -> parallelism=1 (1 writer/task)")
              1
            } else {
              inferParallelism(wst)
            }
          extractedFragments += NativeFragment(
            id = fragId,
            rootOperator = wst,
            outputAttributes = wst.output,
            parallelism = parallelism
          )

          // Link exchange specs: each exchange child's consumer is this fragment.
          // When broadcast-fusion is enabled and the exchange is a BroadcastExchange
          // whose build was already absorbed (producer id == -1), we skip the
          // ExchangeSpec but capture the broadcast variable with its iterator slot
          // index so the JNI layer can pre-populate inputIters_[slotIdx] with a
          // BatchIterator over the broadcasted ColumnarBatches. Without this the
          // C++ side would short by one (substrait still emits ReadRel for the
          // fused build) -> streamIdx OOB at SubstraitToVeloxPlan.cc:1357.
          childExchangeFragIds.zip(exchangeChildren).zipWithIndex.foreach {
            case ((producerFragId, child), slotIdx) =>
              unwrapToExchange(child) match {
                case Some(bex: BroadcastExchangeLike) if producerFragId < 0 =>
                  // Fused broadcast: capture for executor-side iterator hand-off.
                  try {
                    val bcast = bex.executeBroadcast[BuildSideRelation]()
                    broadcastsByConsumer.getOrElseUpdate(fragId, mutable.ArrayBuffer.empty) +=
                      FusedBroadcast(slotIdx, bcast)
                    logWarning(
                      s"MppNativeQueryExec: captured fused broadcast at " +
                        s"consumerF=$fragId slot=$slotIdx " +
                        s"(${bex.getClass.getSimpleName})")
                  } catch {
                    case t: Throwable =>
                      logWarning(
                        s"MppNativeQueryExec: failed to capture broadcast " +
                          s"variable at consumerF=$fragId slot=$slotIdx " +
                          s"(${bex.getClass.getSimpleName}); will likely hit " +
                          s"streamIdx OOB at native side: ${t.getMessage}",
                        t
                      )
                  }
                case _ if producerFragId < 0 =>
                  // Non-broadcast collapse (e.g. collapseSingleGather sentinel).
                  ()
                case Some(exchangeNode) =>
                  val forceBroadcast =
                    replicatedJoinBuildExchangeIds.contains(exchangeNode.id)
                  val (exchangeType, partitionKeys, numPartitions) =
                    if (forceBroadcast) {
                      logWarning(
                        s"MppNativeQueryExec: forcing exchange ${exchangeNode.id} " +
                          s"(${exchangeNode.getClass.getSimpleName}) to BROADCAST for " +
                          s"replicated MPP hash join build side")
                      ("BROADCAST", Seq.empty[Attribute], 1)
                    } else {
                      val (classifiedType, classifiedKeys) =
                        classifyPartitioning(exchangeNode.outputPartitioning)
                      (
                        classifiedType,
                        classifiedKeys,
                        exchangeNode.outputPartitioning.numPartitions)
                    }
                  extractedExchanges += ExchangeSpec(
                    id = exchangeCounter.getAndIncrement(),
                    producerFragmentId = producerFragId,
                    consumerFragmentId = fragId,
                    exchangeType = exchangeType,
                    numPartitions = numPartitions,
                    partitionKeys = partitionKeys
                  )
                case None if unwrapToTopN(child).isDefined =>
                  // Top-N input slot: this slot's subtree was already walked into a
                  // single-driver TakeOrderedAndProject fragment (see that case) fed
                  // by a SINGLE gather. The parent (e.g. a write) WST consumes that
                  // fragment's single ordered+limited output via a SINGLE exchange.
                  // Without this, unwrapToExchange returns None -> we would throw,
                  // blocking write-in-MPP for top-N (LIMIT) queries (Q2/3/10/18/21).
                  extractedExchanges += ExchangeSpec(
                    id = exchangeCounter.getAndIncrement(),
                    producerFragmentId = producerFragId,
                    consumerFragmentId = fragId,
                    exchangeType = "SINGLE",
                    numPartitions = 1,
                    partitionKeys = Seq.empty[Attribute]
                  )
                case None =>
                  throw new IllegalStateException(
                    s"MppNativeQueryExec: WST fragment $fragId input slot $slotIdx " +
                      s"(${child.getClass.getSimpleName}) did not unwrap to an Exchange. " +
                      s"Refusing to silently drop an ExchangeSpec; this would desynchronize " +
                      s"iterator slots and native ValueStream schemas. Slot tree: " +
                      child.treeString.take(1000))
              }
          }
          fragId

        case exchange: ShuffleExchangeLike =>
          // PoC: when spark.gluten.mpp.collapseSingleGather=true, fuse SinglePartition
          // shuffles into the parent fragment. Returns -1 sentinel (same convention
          // as fused broadcast); the WST consumer case must tolerate -1 producerFragId
          // by suppressing the ExchangeSpec entry.
          val isSingleGather = exchange.outputPartitioning.isInstanceOf[SinglePartition.type]
          val collapseSingleGather =
            org.apache.spark.sql.SparkSession.getActiveSession
              .exists(_.conf.get("spark.gluten.mpp.collapseSingleGather", "false").toBoolean)
          if (isSingleGather && collapseSingleGather) {
            logWarning(
              "MppNativeQueryExec.walk: collapsing SinglePartition shuffle into parent fragment")
            walk(unwrapTransparent(exchange.child))
            -1
          } else {
            // Exchange boundary: walk the producer side for this consumer.
            //
            // Native MPP emits exactly one PartitionedOutputNode for each fragment.
            // Reusing one producer fragment for multiple Spark exchange consumers
            // makes the native side reuse that producer's wire row type and output
            // buffers for all consumers. Duplicate the producer occurrence instead;
            // this recomputes the shared exchange payload but keeps producer and
            // consumer schemas aligned positionally.
            // Strip the synthetic hash_partition_key prefix project before descending
            // (mirrors MppCollapseRule.stripSyntheticHashProject) so the producer
            // fragment's WST output excludes the prefix column. Without this, the
            // partition-key indices computed by serializeExchangeSpecs land at [1,2]
            // of an N+1-wide schema while the C++ side strips the prefix on receive
            // (MppJniWrapper.cc), and Velox's PartitionedOutputNode partitions BEFORE
            // the strip -- misrouting Q1's partial-agg states to all 4 F1 drivers and
            // producing 4 keys * 4 drivers = 16 rows instead of 4.
            walk(stripSyntheticHashProject(unwrapTransparent(exchange.child)))
          }

        case bex: BroadcastExchangeLike =>
          // Keep broadcast builds behind a native producer fragment. The old fused path
          // shared one JNI iterator across fanout drivers and is not safe for MPP.
          rejectUnsafeBroadcastFusionIfRequested("MppNativeQueryExec")
          walk(unwrapTransparent(bex.child))

        case reused: ReusedExchangeExec =>
          // Spark reuses the same Exchange JVM object here, but native MPP cannot
          // share one producer output buffer across multiple consumers. Walking
          // the child creates an independent producer fragment for this occurrence.
          walk(reused.child)

        case topk: TakeOrderedAndProjectExecTransformer =>
          // TopN-at-root: the query ends in a global Sort+Limit(+Project) that
          // BSP implements by gathering every producer partition on a single
          // task. In the MPP dump we model this as a dedicated single-driver
          // fragment fed by a SINGLE exchange, mirroring what
          // TakeOrderedAndProjectExecTransformer.doExecuteColumnar builds at
          // runtime: Project(Limit(Sort(InputIteratorTransformer(producer)))).
          val producerFragId = walk(topk.child)
          val topkFragId = fragmentCounter.getAndIncrement()

          // Build the consumer pipeline. wrapInputIteratorTransformer produces
          // InputIteratorTransformer(ColumnarInputAdapter(plan)); at Substrait
          // time the InputIteratorTransformer terminates the tree with a
          // ReadRel, so the producer WST's operators are NOT re-serialized.
          val inputIter =
            ColumnarCollapseTransformStages.wrapInputIteratorTransformer(topk.child)
          val topNPlan =
            if (topk.offset == 0) {
              TopNTransformer(topk.limit, topk.sortOrder, global = false, inputIter)
            } else {
              val sortPlan = SortExecTransformer(topk.sortOrder, global = false, inputIter)
              LimitExecTransformer(sortPlan, topk.offset.toLong, topk.limit)
            }
          val consumerRoot: SparkPlan =
            if (topk.projectList != topk.child.output) {
              ProjectExecTransformer(topk.projectList, topNPlan)
            } else {
              topNPlan
            }
          val stageCounter = ColumnarCollapseTransformStages.getTransformStageCounter(topk)
          val wrappingWst =
            WholeStageTransformer(consumerRoot)(stageCounter.incrementAndGet())

          extractedFragments += NativeFragment(
            id = topkFragId,
            rootOperator = wrappingWst,
            outputAttributes = topk.output,
            parallelism = 1
          )
          extractedExchanges += ExchangeSpec(
            id = exchangeCounter.getAndIncrement(),
            producerFragmentId = producerFragId,
            consumerFragmentId = topkFragId,
            exchangeType = "SINGLE",
            numPartitions = 1,
            partitionKeys = Seq.empty
          )
          topkFragId

        case c2r: ColumnarToRowExecBase =>
          walk(c2r.child)

        case c2c: ColumnarToColumnarExec =>
          walk(c2c.child)

        case iit: InputIteratorTransformer =>
          walk(iit.child)

        case cia: ColumnarInputAdapter =>
          walk(cia.child)

        case other =>
          // For other nodes, walk all children and return the last fragment ID found
          var lastFragId = -1
          other.children.foreach {
            c =>
              val fid = walk(c)
              if (fid >= 0) lastFragId = fid
          }
          lastFragId
      }
    }

    // Apply cross-cut Catalyst rules (Sort/Skip-shuffle) here. Plan-C MppStrategy
    // intercepts queries at planner time and wraps them in MppNativeQueryExec
    // before the columnar post-rule pass runs, so rules registered via
    // VeloxRuleApi.injectPost don't see the wrapped subtree. Re-running them on
    // `child` ensures plan-shape parity (range->single sort, redundant-shuffle
    // elimination) regardless of injection ordering. Each rule is gated by its
    // own conf key, so this is a no-op when the user hasn't opted in.
    val rewrittenChild = applyCrossCutRules(plan, replicatedJoinBuildExchangeIds)

    walk(rewrittenChild)

    // Sort fragments by ID (ensures topological order: producers before consumers)
    val sortedFragments = extractedFragments.sortBy(_.id).toSeq
    val cappedExchanges = capLocalHashExchangeTasks(extractedExchanges.toSeq)
    val adjustedFragments =
      tunePostJoinFinalAggSplitParallelism(sortedFragments, cappedExchanges)
    val sortedExchanges = cappedExchanges.sortBy(_.id).toSeq
    val frozenBroadcasts = broadcastsByConsumer.iterator.map {
      case (consumerId, buf) => consumerId -> buf.toSeq
    }.toMap
    (adjustedFragments, sortedExchanges, frozenBroadcasts)
  }

  /**
   * Re-run plan-shape parity rules on the wrapped child plan. MppStrategy intercepts before the
   * columnar post-rule pass, so rules registered via VeloxRuleApi.injectPost otherwise miss this
   * subtree. Each rule is conf-gated and a no-op by default.
   */
  private def applyCrossCutRules(
      plan: SparkPlan,
      replicatedJoinBuildExchangeIds: mutable.Set[Int]): SparkPlan = {
    val sortRule = MppSinglePartitionSortRule()
    val skipShuffleRule = MppRemoveRedundantShuffleRule()
    val parallelSortSplitRule = MppParallelSortSplitRule()
    val finalAggTopNPartialRule = MppFinalAggTopNPartialRule()
    val afterSort = sortRule(plan)
    val afterSkipShuffle = skipShuffleRule(afterSort)
    // parallelSortSplit must run AFTER MppSinglePartitionSortRule so the
    // RangePartitioning -> SinglePartition rewrite has already happened.
    val afterParallelSortSplit = parallelSortSplitRule(afterSkipShuffle)
    val afterRootFinalAggSplit = splitPostJoinFinalAgg(afterParallelSortSplit)
    // finalAggTopNPartial inserts local TopN after final agg before the TakeOrdered
    // SINGLE gather; run after post-join agg splitting so it limits the split final output.
    val afterFinalAggTopNPartial = finalAggTopNPartialRule(afterRootFinalAggSplit)
    val afterFinalAggSplit = splitFinalAggBeforeJoinHub(afterFinalAggTopNPartial)
    val afterExistenceSplit = splitExistenceFinalBeforeJoinHub(afterFinalAggSplit)
    val afterNativeLocalSorts = offloadLocalSorts(afterExistenceSplit)
    val afterNativeHashJoins = offloadLocalHashJoins(afterNativeLocalSorts)
    val afterBuildSideNormalization = normalizeMppJoinBuildSide(afterNativeHashJoins)
    val afterReplicatedJoin =
      coLocateReplicatedJoinProbe(afterBuildSideNormalization, replicatedJoinBuildExchangeIds)
    val afterBroadcastJoin =
      coLocateBroadcastJoinProbe(afterReplicatedJoin, replicatedJoinBuildExchangeIds)
    normalizeInputIteratorTransformers(
      ColumnarCollapseTransformStages(new GlutenConfig(SQLConf.get))(afterBroadcastJoin))
  }

  private def splitPostJoinFinalAgg(plan: SparkPlan): SparkPlan = {
    // Separate knob, default OFF: the post-join split inserts a vanilla
    // ShuffleExchangeExec that is not a TransformSupport and crashes the
    // columnar pipeline (TPC-H Q18: ClassCastException). Keep it off until that
    // is fixed; it is not needed for correctness (Q18 is correct without it).
    if (!booleanConf("spark.gluten.mpp.splitPostJoinFinalAgg", defaultValue = false)) {
      return plan
    }

    var splitCount = 0
    val rewritten = plan.transformUp {
      case agg: RegularHashAggregateExecTransformer if shouldSplitPostJoinFinalAgg(agg) =>
        buildPostJoinPartialFinalAgg(agg) match {
          case Some(splitAgg) =>
            splitCount += 1
            splitAgg
          case None =>
            agg
        }
    }
    if (splitCount > 0) {
      logWarning(
        s"MppNativeQueryExec: inserted $splitCount post-join final-aggregate split(s) " +
          s"after join hubs")
    }
    rewritten
  }

  private def shouldSplitPostJoinFinalAgg(agg: HashAggregateExecTransformer): Boolean = {
    if (agg.groupingExpressions.isEmpty || unwrapToExchange(agg.child).isDefined) {
      return false
    }
    if (!canSplitPostJoinAggregateModes(agg)) {
      return false
    }
    countHashJoins(agg.child) >= 2
  }

  private def canSplitPostJoinAggregateModes(agg: HashAggregateExecTransformer): Boolean = {
    agg.aggregateExpressions.nonEmpty &&
    agg.aggregateExpressions.forall {
      expr =>
        !expr.isDistinct &&
        expr.filter.isEmpty &&
        (expr.mode == Complete || expr.mode == Final)
    }
  }

  private def buildPostJoinPartialFinalAgg(
      agg: RegularHashAggregateExecTransformer): Option[SparkPlan] = {
    val partialAggregateExpressions = agg.aggregateExpressions.map(_.copy(mode = Partial))
    val finalAggregateExpressions = agg.aggregateExpressions.map(_.copy(mode = Final))
    val partialAggregateAttributes =
      partialAggregateExpressions.flatMap(_.aggregateFunction.aggBufferAttributes)
    val partialResultExpressions: Seq[NamedExpression] =
      agg.groupingExpressions ++ partialAggregateAttributes
    val partialAgg = agg.copy(
      requiredChildDistributionExpressions = None,
      aggregateExpressions = partialAggregateExpressions,
      aggregateAttributes = partialAggregateAttributes,
      initialInputBufferOffset = 0,
      resultExpressions = partialResultExpressions,
      child = agg.child
    )

    val validation = partialAgg.doValidate()
    if (!validation.ok()) {
      logWarning(
        s"MppNativeQueryExec: post-join partial aggregate validation failed ($validation); " +
          "leaving aggregate unsplit")
      return None
    }

    val shuffledPartialAgg = ShuffleExchangeExec(
      HashPartitioning(agg.groupingExpressions, mppPostJoinFinalAggSplitPartitions()),
      partialAgg)
    Some(
      agg.copy(
        aggregateExpressions = finalAggregateExpressions,
        initialInputBufferOffset = agg.groupingExpressions.length,
        child = shuffledPartialAgg))
  }

  private def countHashJoins(plan: SparkPlan): Int = {
    var count = 0
    plan.foreach {
      case _: ShuffledHashJoinExecTransformer => count += 1
      case _: ShuffledHashJoinExec => count += 1
      case _ =>
    }
    count
  }

  private def mppPostJoinFinalAggSplitPartitions(): Int = {
    val confKey = "spark.gluten.mpp.postJoinFinalAggSplitPartitions"
    math.max(
      1,
      SQLConf.get
        .getConfString(confKey, mppSplitHashPartitions().toString)
        .toInt)
  }

  // Unconditional MPP "EnsureRequirements" for hash-join sides: when the fragment
  // walk fuses a grouping aggregate into a hash-join fragment WITHOUT an
  // intervening exchange, the agg's output is not clustered on the join keys, so
  // build and probe are not co-partitioned and the join silently drops matching
  // rows (TPC-H Q17: avg(l_quantity) per l_partkey -> wrong boundary -> ~9% low).
  // Re-cluster every such fused join side on the join keys via a real COLUMNAR
  // shuffle. No conf gate -- this is a correctness invariant, not an opt-in.
  private def splitFinalAggBeforeJoinHub(plan: SparkPlan): SparkPlan = {
    var splitCount = 0
    val rewritten = plan.transformUp {
      case join: ShuffledHashJoinExecTransformer =>
        val newLeft = maybeSplitFinalAggJoinSide(join.left, join.leftKeys)
        val newRight = maybeSplitFinalAggJoinSide(join.right, join.rightKeys)
        if ((newLeft eq join.left) && (newRight eq join.right)) {
          join
        } else {
          splitCount += splitDelta(newLeft, join.left) + splitDelta(newRight, join.right)
          join.copy(left = newLeft, right = newRight)
        }
    }
    if (splitCount > 0) {
      logWarning(
        s"MppNativeQueryExec: inserted $splitCount final-aggregate split(s) before join hubs")
    }
    rewritten
  }

  private def maybeSplitFinalAggJoinSide(plan: SparkPlan, joinKeys: Seq[Expression]): SparkPlan = {
    // Re-cluster a Filter(HashAggregate) join side on the join keys so build and
    // probe are co-partitioned -- the decorrelated correlated-subquery avg in
    // TPC-H Q17 (l_quantity < 0.2*avg(l_quantity) per l_partkey), whose aggregate
    // IS the join input. insertInputIteratorTransformer turns the inserted row
    // exchange into a columnar MPP HASH exchange during
    // ColumnarCollapseTransformStages (GPU-to-GPU; stays MPP). Trigger is the
    // precise Filter(HashAggregate)-at-root shape: a broader "any descendant
    // aggregate" walk over-fires onto sides whose aggregate sits below another
    // join, inserting spurious shuffles that trip the join's PartitioningCollection
    // (numPartitions mismatch -> write-path failure). A full EnsureRequirements-
    // for-MPP reconciling both sides across all distribution requirements is the
    // follow-up.
    if (joinKeys.isEmpty || !isFilteredAggregateJoinSide(plan)) {
      return plan
    }
    ShuffleExchangeExec(HashPartitioning(joinKeys, mppSplitHashPartitions()), plan)
  }

  private def isFilteredAggregateJoinSide(plan: SparkPlan): Boolean = plan match {
    case FilterExecTransformer(_, _: HashAggregateExecBaseTransformer) => true
    case _ => false
  }

  private def mppSplitHashPartitions(): Int = {
    val defaultPartitions =
      math.max(1, SQLConf.get.getConfString("spark.sql.shuffle.partitions", "200").toInt)
    val configured =
      mppNonNegativeIntConf("spark.gluten.mpp.localHashExchangeTasks", defaultPartitions)
    math.max(1, configured)
  }

  private def splitDelta(newPlan: SparkPlan, oldPlan: SparkPlan): Int = {
    if (newPlan eq oldPlan) 0 else 1
  }

  private def splitExistenceFinalBeforeJoinHub(plan: SparkPlan): SparkPlan = {
    if (!booleanConf("spark.gluten.mpp.splitExistenceFinalBeforeJoinHub", defaultValue = false)) {
      return plan
    }

    var splitCount = 0
    val rewritten = plan.transformUp {
      case join: ShuffledHashJoinExecTransformer =>
        val newLeft = maybeSplitExistenceFinalJoinSide(join.left, join.leftKeys)
        val newRight = maybeSplitExistenceFinalJoinSide(join.right, join.rightKeys)
        if ((newLeft eq join.left) && (newRight eq join.right)) {
          join
        } else {
          splitCount += splitDelta(newLeft, join.left) + splitDelta(newRight, join.right)
          join.copy(left = newLeft, right = newRight)
        }
    }
    if (splitCount > 0) {
      logWarning(
        s"MppNativeQueryExec: inserted $splitCount existence-final split(s) before join hubs")
    }
    rewritten
  }

  private def maybeSplitExistenceFinalJoinSide(
      plan: SparkPlan,
      joinKeys: Seq[Expression]): SparkPlan = {
    if (
      joinKeys.isEmpty ||
      unwrapToExchange(plan).isDefined ||
      !containsQ21ExistenceFinalAggFilter(plan)
    ) {
      return plan
    }
    ShuffleExchangeExec(HashPartitioning(joinKeys, mppSplitHashPartitions()), plan)
  }

  private def containsQ21ExistenceFinalAggFilter(plan: SparkPlan): Boolean = {
    var found = false
    plan.foreach {
      case filter @ FilterExecTransformer(_, _: HashAggregateExecBaseTransformer) =>
        val outputNames = filter.output.map(_.name.toLowerCase(java.util.Locale.ROOT))
        val tree = filter.treeString.toLowerCase(java.util.Locale.ROOT)
        if (
          outputNames.exists(_.contains("exists")) &&
          outputNames.exists(_.contains("unique")) &&
          outputNames.exists(_.contains("count")) &&
          tree.contains("hashaggregatetransformer") &&
          tree.contains("l_orderkey") &&
          tree.contains("l_suppkey")
        ) {
          found = true
        }
      case _ =>
    }
    found
  }

  private def normalizeInputIteratorTransformers(plan: SparkPlan): SparkPlan = {
    var changed = true
    var current = plan
    while (changed) {
      changed = false
      current = current.transformUp {
        case outer: InputIteratorTransformer
            if outer.child.isInstanceOf[InputIteratorTransformer] =>
          changed = true
          outer.child
        case InputIteratorTransformer(ColumnarInputAdapter(wst: WholeStageTransformer)) =>
          changed = true
          wst.child
        case InputIteratorTransformer(ColumnarInputAdapter(ts: TransformSupport)) =>
          changed = true
          ts
      }
    }
    current
  }

  private def offloadLocalSorts(plan: SparkPlan): SparkPlan = {
    plan.transformUp {
      case sort: SortExec if !sort.global =>
        SortExecTransformer(sort.sortOrder, global = false, sort.child, sort.testSpillFrequency)
    }
  }

  private def offloadLocalHashJoins(plan: SparkPlan): SparkPlan = {
    var strippedSorts = 0
    def stripHashJoinInputSort(child: SparkPlan): SparkPlan = child match {
      case sort: SortExec if !sort.global =>
        strippedSorts += 1
        sort.child
      case sort: SortExecTransformer if !sort.global =>
        strippedSorts += 1
        sort.child
      case other =>
        other
    }
    val rewritten = plan.transformUp {
      case join: ShuffledHashJoinExec =>
        val left = stripHashJoinInputSort(join.left)
        val right = stripHashJoinInputSort(join.right)
        ShuffledHashJoinExecTransformer(
          join.leftKeys,
          join.rightKeys,
          join.joinType,
          join.buildSide,
          join.condition,
          left,
          right,
          join.isSkewJoin)
      case join: ShuffledHashJoinExecTransformer =>
        val left = stripHashJoinInputSort(join.left)
        val right = stripHashJoinInputSort(join.right)
        if ((left eq join.left) && (right eq join.right)) {
          join
        } else {
          join.copy(left = left, right = right)
        }
      case join: BroadcastHashJoinExec =>
        BroadcastHashJoinExecTransformer(
          join.leftKeys,
          join.rightKeys,
          join.joinType,
          join.buildSide,
          join.condition,
          join.left,
          join.right,
          join.isNullAwareAntiJoin)
    }
    if (strippedSorts > 0) {
      logWarning(
        s"MppNativeQueryExec: stripped $strippedSorts local sort(s) " +
          s"from hash join inputs")
    }
    rewritten
  }

  /**
   * Optional guardrail for plans where Spark has already collapsed several Presto-like stages into
   * a single native fragment. When enabled, this avoids running one compact fragment that must
   * drain many large HASH/BROADCAST inputs and keep all join/aggregate state live at once.
   */
  private def overloadedFragmentFallbackReason(
      fragments: Seq[NativeFragment],
      exchanges: Seq[ExchangeSpec]): Option[String] = {
    val maxInbound = mppNonNegativeIntConf("spark.gluten.mpp.maxInboundExchangesPerFragment", 0)
    val maxHashInbound =
      mppNonNegativeIntConf("spark.gluten.mpp.maxHashInboundExchangesPerFragment", 0)
    val maxBroadcastInbound =
      mppNonNegativeIntConf("spark.gluten.mpp.maxBroadcastInboundExchangesPerFragment", 0)
    if (maxInbound <= 0 && maxHashInbound <= 0 && maxBroadcastInbound <= 0) {
      return None
    }
    logWarning(
      s"MppNativeQueryExec: inbound exchange guardrails enabled " +
        s"(maxInbound=$maxInbound maxHash=$maxHashInbound maxBroadcast=$maxBroadcastInbound)")

    val fragmentById = fragments.map(f => f.id -> f).toMap
    val overloaded = exchanges
      .groupBy(_.consumerFragmentId)
      .toSeq
      .sortBy(_._1)
      .flatMap {
        case (consumerId, inbound) =>
          val hashCount = inbound.count(_.exchangeType == "HASH")
          val broadcastCount = inbound.count(_.exchangeType == "BROADCAST")
          val totalCount = inbound.size
          val exceeded =
            (maxInbound > 0 && totalCount > maxInbound) ||
              (maxHashInbound > 0 && hashCount > maxHashInbound) ||
              (maxBroadcastInbound > 0 && broadcastCount > maxBroadcastInbound)
          if (exceeded) {
            val output = fragmentById
              .get(consumerId)
              .map(_.outputAttributes.map(_.name).mkString("[", ", ", "]"))
              .getOrElse("[]")
            Some(
              s"fragment $consumerId has $totalCount inbound exchanges " +
                s"(HASH=$hashCount, BROADCAST=$broadcastCount, output=$output), exceeding " +
                s"limits maxInbound=$maxInbound maxHash=$maxHashInbound " +
                s"maxBroadcast=$maxBroadcastInbound")
          } else {
            None
          }
      }

    overloaded.headOption
  }

  private def nativeMppFallbackReason(
      fragments: Seq[NativeFragment],
      exchanges: Seq[ExchangeSpec]): Option[String] = {
    overloadedFragmentFallbackReason(fragments, exchanges)
      .orElse(duplicateAggregateFragmentFallbackReason(fragments, exchanges))
  }

  /**
   * Q18/Q21-style plans can contain multiple large scan-bearing aggregation fragments with the same
   * output shape, usually from duplicated lineitem branches. Native all-stages-up MPP runs those
   * branches at once, unlike Spark's staged BSP path, so keep this opt-in until native branch
   * scheduling exists.
   */
  private def duplicateAggregateFragmentFallbackReason(
      fragments: Seq[NativeFragment],
      exchanges: Seq[ExchangeSpec]): Option[String] = {
    if (
      !booleanConf("spark.gluten.mpp.fallbackOnDuplicateAggregateFragments", defaultValue = false)
    ) {
      return None
    }

    case class AggregateFragment(id: Int, signature: String)

    val singleLinkedFragmentPairs = exchanges
      .filter(_.exchangeType == "SINGLE")
      .flatMap {
        exchange =>
          Seq(
            exchange.producerFragmentId -> exchange.consumerFragmentId,
            exchange.consumerFragmentId -> exchange.producerFragmentId)
      }
      .toSet

    val aggregateFragments = fragments.flatMap {
      fragment =>
        Option(fragment.rootOperator).flatMap {
          root =>
            val tree = root.treeString
            if (
              tree.contains("HashAggregateTransformer") &&
              tree.contains("FileSourceScanExecTransformer") &&
              tree.contains("lineitem")
            ) {
              val signature = fragment.outputAttributes.map(_.name).mkString("[", ", ", "]")
              Some(AggregateFragment(fragment.id, signature))
            } else {
              None
            }
        }
    }

    aggregateFragments
      .groupBy(_.signature)
      .toSeq
      .sortBy(_._1)
      .flatMap {
        case (signature, grouped) =>
          val independentIds = grouped.map(_.id).filter {
            id =>
              grouped.exists {
                other => other.id != id && !singleLinkedFragmentPairs.contains(id -> other.id)
              }
          }
          if (independentIds.size > 1) Some(signature -> independentIds.sorted) else None
      }
      .collectFirst {
        case (signature, fragmentIds) =>
          s"duplicate aggregate fragments output=$signature fragments=" +
            fragmentIds.mkString("[", ",", "]") +
            " and spark.gluten.mpp.fallbackOnDuplicateAggregateFragments=true"
      }
  }

  private case class BuildSideChoice(side: BuildSide, reason: String)

  private case class JoinSideStats(sizeInBytes: BigInt, rowCount: Option[BigInt], source: String)

  private object MppJoinSelectionHelper extends JoinSelectionHelper

  /**
   * Align MPP hash-join fragmenting with Presto's replicated join shape. When one side is small
   * enough to replicate, keep the large streamed/probe side in the same fragment as the join and
   * only ship the build side through a BROADCAST exchange.
   */
  private def coLocateReplicatedJoinProbe(
      plan: SparkPlan,
      replicatedJoinBuildExchangeIds: mutable.Set[Int]): SparkPlan = {
    if (!coLocateReplicatedJoinProbeEnabled) {
      return plan
    }
    plan.transformUp {
      case join: ShuffledHashJoinExecTransformer if shouldCoLocateReplicatedJoinProbe(join) =>
        val streamedExchange = boundaryExchange(join.streamedPlan).get
        val buildExchange = boundaryExchange(join.buildPlan).get
        coLocateJoinProbeSide(
          join,
          join.buildSide,
          streamedExchange,
          buildExchange,
          replicatedJoinBuildExchangeIds)
    }
  }

  /**
   * Presto keeps broadcast joins on the large HASH-partitioned probe branch in the same fragment as
   * the join while only the small build side crosses a BROADCAST exchange. Spark/Gluten often
   * leaves the probe side behind its own fragment boundary, which shows up on Q2/Q5/Q7 as extra
   * exchanges and higher join_hash / scan-I/O overhead.
   */
  private def coLocateBroadcastJoinProbe(
      plan: SparkPlan,
      replicatedJoinBuildExchangeIds: mutable.Set[Int]): SparkPlan = {
    if (!coLocateBroadcastJoinProbeEnabled) {
      return plan
    }
    plan.transformUp {
      case join: BroadcastHashJoinExecTransformer if shouldCoLocateBroadcastJoinProbe(join) =>
        val streamedExchange = boundaryExchange(join.streamedPlan).get
        val buildExchange = boundaryExchange(join.buildPlan).get
        coLocateJoinProbeSide(
          join,
          join.buildSide,
          streamedExchange,
          buildExchange,
          replicatedJoinBuildExchangeIds)
    }
  }

  private def coLocateJoinProbeSide[T <: HashJoinLikeExecTransformer](
      join: T,
      buildSide: BuildSide,
      streamedExchange: Exchange,
      buildExchange: Exchange,
      replicatedJoinBuildExchangeIds: mutable.Set[Int]): T = {
    val inlinedStreamed = inlineExchangeProducer(streamedExchange)
    replicatedJoinBuildExchangeIds += buildExchange.id
    logWarning(
      s"MppNativeQueryExec: co-locating hash join probe side " +
        s"join=${join.nodeName} buildSide=$buildSide " +
        s"streamedExchange=${streamedExchange.id} buildExchange=${buildExchange.id}; " +
        s"build side will use native BROADCAST exchange")
    val relocated = buildSide match {
      case BuildLeft => join.withNewChildren(Array(join.left, inlinedStreamed))
      case BuildRight => join.withNewChildren(Array(inlinedStreamed, join.right))
    }
    relocated.asInstanceOf[T]
  }

  private def shouldCoLocateBroadcastJoinProbe(join: BroadcastHashJoinExecTransformer): Boolean = {
    join.joinType match {
      case _: InnerLike =>
      case _ =>
        logWarning(
          s"MppNativeQueryExec: not co-locating broadcast hash join probe side; " +
            s"join=${join.nodeName} joinType=${join.joinType} is not inner-like")
        return false
    }

    val streamedExchange = boundaryExchange(join.streamedPlan)
    val buildExchange = boundaryExchange(join.buildPlan)
    if (streamedExchange.isEmpty || buildExchange.isEmpty) {
      logWarning(
        s"MppNativeQueryExec: not co-locating broadcast hash join probe side; " +
          s"join=${join.nodeName} buildSide=${join.buildSide} " +
          s"streamedExchangeFound=${streamedExchange.isDefined} " +
          s"buildExchangeFound=${buildExchange.isDefined}")
      return false
    }
    val streamed = streamedExchange.get
    if (!streamed.isInstanceOf[ShuffleExchangeLike]) {
      logWarning(
        s"MppNativeQueryExec: not co-locating broadcast hash join probe side; " +
          s"join=${join.nodeName} buildSide=${join.buildSide} " +
          s"streamedExchange=${streamed.nodeName} " +
          s"class=${streamed.getClass.getName} is not ShuffleExchangeLike")
      return false
    }
    if (!isHashPartitioned(streamed)) {
      logWarning(
        s"MppNativeQueryExec: not co-locating broadcast hash join probe side; " +
          s"join=${join.nodeName} buildSide=${join.buildSide} " +
          s"streamedExchange=${streamed.nodeName} " +
          s"partitioning=${streamed.outputPartitioning} is not hash partitioning")
      return false
    }
    if (!buildExchange.get.isInstanceOf[BroadcastExchangeLike]) {
      logWarning(
        s"MppNativeQueryExec: not co-locating broadcast hash join probe side; " +
          s"join=${join.nodeName} buildSide=${join.buildSide} " +
          s"buildExchange=${buildExchange.get.nodeName} is not broadcast")
      return false
    }
    true
  }

  private def shouldCoLocateReplicatedJoinProbe(join: ShuffledHashJoinExecTransformer): Boolean = {
    join.joinType match {
      case _: InnerLike =>
      case _ =>
        logWarning(
          s"MppNativeQueryExec: not co-locating replicated hash join probe side; " +
            s"join=${join.nodeName} joinType=${join.joinType} is not inner-like")
        return false
    }

    val streamedExchange = boundaryExchange(join.streamedPlan)
    val buildExchange = boundaryExchange(join.buildPlan)
    if (streamedExchange.isEmpty || buildExchange.isEmpty) {
      logWarning(
        s"MppNativeQueryExec: not co-locating replicated hash join probe side; " +
          s"join=${join.nodeName} buildSide=${join.buildSide} " +
          s"streamedExchangeFound=${streamedExchange.isDefined} " +
          s"buildExchangeFound=${buildExchange.isDefined}")
      return false
    }
    if (!streamedExchange.get.isInstanceOf[ShuffleExchangeLike]) {
      logWarning(
        s"MppNativeQueryExec: not co-locating replicated hash join probe side; " +
          s"join=${join.nodeName} buildSide=${join.buildSide} " +
          s"streamedExchange=${streamedExchange.get.nodeName} " +
          s"class=${streamedExchange.get.getClass.getName} is not ShuffleExchangeLike")
      return false
    }

    if (buildExchange.get.isInstanceOf[BroadcastExchangeLike]) {
      logWarning(
        s"MppNativeQueryExec: co-locating replicated hash join probe side; " +
          s"join=${join.nodeName} buildSide=${join.buildSide} " +
          s"buildExchange=${buildExchange.get.nodeName} is already broadcast")
      return true
    }

    val eligible = replicatedJoinStats(join).isDefined
    if (!eligible) {
      logWarning(
        s"MppNativeQueryExec: not co-locating replicated hash join probe side; " +
          s"join=${join.nodeName} buildSide=${join.buildSide} " +
          s"streamedExchange=${streamedExchange.get.nodeName} " +
          s"buildExchange=${buildExchange.get.nodeName} stats not eligible")
    }
    eligible
  }

  private def inlineExchangeProducer(exchange: Exchange): SparkPlan = {
    val producer = stripSyntheticHashProject(unwrapTransparent(exchange.child))
    val inlined = producer match {
      case wst: WholeStageTransformer => wst.child
      case other => other
    }
    MppPartitioningPreservingWrapper(
      inlined,
      exchange.outputPartitioning,
      exchange.outputOrdering,
      exchange.id)
  }

  private def joinSideStats(
      join: ShuffledHashJoinExecTransformer): Option[(JoinSideStats, JoinSideStats)] = {
    joinSideStatsCandidates(join).headOption
  }

  private def replicatedJoinStats(
      join: ShuffledHashJoinExecTransformer): Option[(JoinSideStats, JoinSideStats)] = {
    val maxBuildBytes = replicatedJoinMaxBuildBytes
    val candidates = joinSideStatsCandidates(join)
    val eligible = candidates.collectFirst {
      case (leftStats, rightStats)
          if isReplicatedJoinStatsEligible(join, leftStats, rightStats, maxBuildBytes) =>
        buildAndStreamedStats(join, leftStats, rightStats)
    }
    if (eligible.isEmpty && candidates.nonEmpty) {
      val reasons = candidates
        .map {
          case (leftStats, rightStats) =>
            val (buildStats, streamedStats) = buildAndStreamedStats(join, leftStats, rightStats)
            s"${leftStats.source}: buildSize=${buildStats.sizeInBytes} " +
              s"streamedSize=${streamedStats.sizeInBytes}"
        }
        .mkString("; ")
      logWarning(
        s"MppNativeQueryExec: not co-locating hash join probe side; " +
          s"maxBuildBytes=$maxBuildBytes candidates=[$reasons]")
    }
    eligible
  }

  private def isReplicatedJoinStatsEligible(
      join: ShuffledHashJoinExecTransformer,
      leftStats: JoinSideStats,
      rightStats: JoinSideStats,
      maxBuildBytes: BigInt): Boolean = {
    val (buildStats, streamedStats) = buildAndStreamedStats(join, leftStats, rightStats)
    isConfidentSize(buildStats.sizeInBytes) &&
    isConfidentSize(streamedStats.sizeInBytes) &&
    isSignificantlySmaller(buildStats.sizeInBytes, streamedStats.sizeInBytes) &&
    buildStats.sizeInBytes <= maxBuildBytes
  }

  private def buildAndStreamedStats(
      join: ShuffledHashJoinExecTransformer,
      leftStats: JoinSideStats,
      rightStats: JoinSideStats): (JoinSideStats, JoinSideStats) = {
    join.buildSide match {
      case BuildLeft => (leftStats, rightStats)
      case BuildRight => (rightStats, leftStats)
    }
  }

  private def joinSideStatsCandidates(
      join: ShuffledHashJoinExecTransformer): Seq[(JoinSideStats, JoinSideStats)] = {
    Seq(logicalJoinStats(join), childLogicalStats(join)).flatten
  }

  /**
   * Normalize build/probe choice before Substrait generation. Native Velox/cuDF hash join builds
   * the right input, and Gluten's HashJoin transformer already maps BuildLeft/BuildRight into
   * streamed/build input order. This rule decides the build side in the Spark/Gluten physical plan
   * layer rather than relying on table-name special cases or native operators to reinterpret it.
   */
  private def normalizeMppJoinBuildSide(plan: SparkPlan): SparkPlan = {
    if (!normalizeMppJoinBuildSideEnabled) {
      return plan
    }
    plan.transformUp {
      case join: SortMergeJoinExecTransformer
          if join.joinType == LeftSemi && forceLeftSemiBuildLeftEnabled =>
        logWarning(
          s"MppNativeQueryExec: rewriting LEFT SEMI sort-merge join to " +
            s"shuffled hash join with BuildLeft; " +
            s"spark.gluten.mpp.forceLeftSemiBuildLeft=true")
        ShuffledHashJoinExecTransformer(
          join.leftKeys,
          join.rightKeys,
          join.joinType,
          BuildLeft,
          join.condition,
          join.left,
          join.right,
          join.isSkewJoin)
      case join: ShuffledHashJoinExecTransformer if canNormalizeMppBuildSide(join) =>
        preferredMppBuildSide(join)
          .filter(_.side != join.buildSide)
          .map {
            choice =>
              logWarning(
                s"MppNativeQueryExec: normalizing MPP hash join build side " +
                  s"from ${join.buildSide} to ${choice.side}; ${choice.reason}")
              join.copy(buildSide = choice.side)
          }
          .getOrElse(join)
    }
  }

  private def canNormalizeMppBuildSide(join: ShuffledHashJoinExecTransformer): Boolean = {
    join.joinType match {
      case _: InnerLike => true
      case LeftSemi => true
      case LeftOuter | RightOuter
          if normalizeMppOuterJoinBuildSideEnabled || forceMppOuterJoinPreservedBuildSideEnabled =>
        true
      case _ => false
    }
  }

  private def preferredMppBuildSide(
      join: ShuffledHashJoinExecTransformer): Option[BuildSideChoice] = {
    if (join.joinType == LeftSemi) {
      return preferredLeftSemiBuildSide(join)
    }
    if (forceMppOuterJoinPreservedBuildSideEnabled) {
      preferredOuterJoinPreservedBuildSide(join) match {
        case forced @ Some(_) => return forced
        case None =>
      }
    }
    broadcastBuildSide(join).orElse {
      statsBuildSide(join).orElse {
        sparkJoinSelectionBuildSide(join)
      }
    }
  }

  private def preferredLeftSemiBuildSide(
      join: ShuffledHashJoinExecTransformer): Option[BuildSideChoice] = {
    // Spark normally builds the right side for LeftSemi to preserve left-side output semantics.
    // Velox/cuDF also supports the swapped shape via RIGHT_SEMI, so MPP can build the left side
    // when stats show it is clearly smaller. This avoids Q21-style lineitem RHS hash builds.
    statsBuildSide(join).filter(_.side == BuildLeft).orElse {
      if (forceLeftSemiBuildLeftEnabled) {
        Some(
          BuildSideChoice(
            BuildLeft,
            "spark.gluten.mpp.forceLeftSemiBuildLeft=true forced LEFT SEMI build-left"))
      } else {
        None
      }
    }
  }

  private def forceLeftSemiBuildLeftEnabled: Boolean = {
    booleanConf("spark.gluten.mpp.forceLeftSemiBuildLeft", defaultValue = false)
  }

  private def preferredOuterJoinPreservedBuildSide(
      join: ShuffledHashJoinExecTransformer): Option[BuildSideChoice] = {
    join.joinType match {
      case LeftOuter =>
        Some(
          BuildSideChoice(
            BuildLeft,
            "spark.gluten.mpp.forceOuterJoinPreservedBuildSide=true forced LEFT OUTER " +
              "preserved side build"))
      case RightOuter =>
        Some(
          BuildSideChoice(
            BuildRight,
            "spark.gluten.mpp.forceOuterJoinPreservedBuildSide=true forced RIGHT OUTER " +
              "preserved side build"))
      case _ => None
    }
  }

  private def broadcastBuildSide(join: ShuffledHashJoinExecTransformer): Option[BuildSideChoice] = {
    val leftBroadcast =
      boundaryExchange(join.left).exists(_.isInstanceOf[BroadcastExchangeLike])
    val rightBroadcast =
      boundaryExchange(join.right).exists(_.isInstanceOf[BroadcastExchangeLike])

    if (leftBroadcast != rightBroadcast) {
      val side = if (leftBroadcast) BuildLeft else BuildRight
      return Some(
        BuildSideChoice(
          side,
          s"broadcast child selected as build side " +
            s"(leftBroadcast=$leftBroadcast, rightBroadcast=$rightBroadcast)"))
    }
    None
  }

  private def boundaryExchange(plan: SparkPlan): Option[Exchange] = {
    unwrapToExchange(plan).orElse {
      plan.children match {
        case Seq(child) => boundaryExchange(child)
        case _ => None
      }
    }
  }

  private def sparkJoinSelectionBuildSide(
      join: ShuffledHashJoinExecTransformer): Option[BuildSideChoice] = {
    join.logicalLink.flatMap {
      case logicalJoin: Join =>
        MppJoinSelectionHelper
          .getBroadcastBuildSide(
            logicalJoin.left,
            logicalJoin.right,
            logicalJoin.joinType,
            logicalJoin.hint,
            hintOnly = false,
            SQLConf.get)
          .orElse {
            MppJoinSelectionHelper.getShuffleHashJoinBuildSide(
              logicalJoin.left,
              logicalJoin.right,
              logicalJoin.joinType,
              logicalJoin.hint,
              hintOnly = false,
              SQLConf.get)
          }
          .map {
            side =>
              BuildSideChoice(
                side,
                s"Spark JoinSelection selected build side " +
                  s"(leftSize=${logicalJoin.left.stats.sizeInBytes}, " +
                  s"rightSize=${logicalJoin.right.stats.sizeInBytes})"
              )
          }
      case _ => None
    }
  }

  private def statsBuildSide(join: ShuffledHashJoinExecTransformer): Option[BuildSideChoice] = {
    Seq(logicalJoinStats(join), childLogicalStats(join)).flatten.flatMap {
      case (leftStats, rightStats) =>
        chooseSmallerBuildSide(leftStats, rightStats).map {
          side =>
            BuildSideChoice(
              side,
              s"${leftStats.source} selected smaller build side " +
                s"(leftSize=${leftStats.sizeInBytes}, rightSize=${rightStats.sizeInBytes}, " +
                s"leftRows=${leftStats.rowCount.getOrElse("unknown")}, " +
                s"rightRows=${rightStats.rowCount.getOrElse("unknown")})"
            )
        }
    }.headOption
  }

  private def logicalJoinStats(
      join: ShuffledHashJoinExecTransformer): Option[(JoinSideStats, JoinSideStats)] = {
    join.logicalLink.flatMap {
      case logicalJoin: Join =>
        val leftStats = logicalJoin.left.stats
        val rightStats = logicalJoin.right.stats
        Some(
          (
            JoinSideStats(leftStats.sizeInBytes, leftStats.rowCount, "logical join stats"),
            JoinSideStats(rightStats.sizeInBytes, rightStats.rowCount, "logical join stats")))
      case _ => None
    }
  }

  private def childLogicalStats(
      join: ShuffledHashJoinExecTransformer): Option[(JoinSideStats, JoinSideStats)] = {
    for {
      left <- planLogicalStats(join.left)
      right <- planLogicalStats(join.right)
    } yield {
      (
        JoinSideStats(left.sizeInBytes, left.rowCount, "physical subtree stats"),
        JoinSideStats(right.sizeInBytes, right.rowCount, "physical subtree stats"))
    }
  }

  private def planLogicalStats(plan: SparkPlan): Option[Statistics] = {
    plan.logicalLink.map(_.stats).filter(hasUsableStats).orElse {
      val childStats = plan.children.flatMap(planLogicalStats)
      if (childStats.isEmpty) {
        None
      } else if (childStats.size == 1) {
        Some(childStats.head)
      } else {
        val rowCount =
          if (childStats.forall(_.rowCount.isDefined)) {
            Some(childStats.flatMap(_.rowCount).sum)
          } else {
            None
          }
        Some(Statistics(childStats.map(_.sizeInBytes).sum, rowCount))
      }
    }
  }

  private def hasUsableStats(stats: Statistics): Boolean = {
    isConfidentSize(stats.sizeInBytes) || stats.rowCount.exists(_ > 0)
  }

  private def chooseSmallerBuildSide(
      leftStats: JoinSideStats,
      rightStats: JoinSideStats): Option[BuildSide] = {
    if (isConfidentSize(leftStats.sizeInBytes) && isConfidentSize(rightStats.sizeInBytes)) {
      if (isSignificantlySmaller(leftStats.sizeInBytes, rightStats.sizeInBytes)) {
        return Some(BuildLeft)
      }
      if (isSignificantlySmaller(rightStats.sizeInBytes, leftStats.sizeInBytes)) {
        return Some(BuildRight)
      }
    }

    (leftStats.rowCount, rightStats.rowCount) match {
      case (Some(leftRows), Some(rightRows)) if isSignificantlySmaller(leftRows, rightRows) =>
        Some(BuildLeft)
      case (Some(leftRows), Some(rightRows)) if isSignificantlySmaller(rightRows, leftRows) =>
        Some(BuildRight)
      case _ => None
    }
  }

  private def isConfidentSize(size: BigInt): Boolean = {
    size > 0 && size < BigInt(Long.MaxValue)
  }

  private def isSignificantlySmaller(candidate: BigInt, other: BigInt): Boolean = {
    candidate > 0 && candidate * 2 <= other
  }

  private def normalizeMppJoinBuildSideEnabled: Boolean = {
    booleanConf("spark.gluten.mpp.normalizeJoinBuildSide", defaultValue = true)
  }

  private def normalizeMppOuterJoinBuildSideEnabled: Boolean = {
    booleanConf("spark.gluten.mpp.normalizeOuterJoinBuildSide", defaultValue = false)
  }

  private def forceMppOuterJoinPreservedBuildSideEnabled: Boolean = {
    booleanConf("spark.gluten.mpp.forceOuterJoinPreservedBuildSide", defaultValue = false)
  }

  private def coLocateReplicatedJoinProbeEnabled: Boolean = {
    booleanConf("spark.gluten.mpp.coLocateReplicatedJoinProbe", defaultValue = true)
  }

  private def coLocateBroadcastJoinProbeEnabled: Boolean = {
    booleanConf("spark.gluten.mpp.coLocateBroadcastJoinProbe", defaultValue = false)
  }

  private def replicatedJoinMaxBuildBytes: BigInt = {
    bytesConf("spark.gluten.mpp.replicatedJoinMaxBuildBytes", BigInt(32L) << 30)
  }

  private def capLocalHashExchangeTasks(exchanges: Seq[ExchangeSpec]): Seq[ExchangeSpec] = {
    localHashExchangeTasks match {
      case Some(cap) =>
        exchanges.map {
          spec =>
            if (
              (spec.exchangeType == "HASH" || spec.exchangeType == "RANGE") &&
              spec.numPartitions > cap
            ) {
              logWarning(
                s"MppNativeQueryExec: capping local ${spec.exchangeType} exchange ${spec.id} " +
                  s"F${spec.producerFragmentId}->F${spec.consumerFragmentId} native partitions " +
                  s"from ${spec.numPartitions} to $cap via " +
                  s"spark.gluten.mpp.localHashExchangeTasks")
              spec.copy(numPartitions = cap)
            } else {
              spec
            }
        }
      case None => exchanges
    }
  }

  private def isPostJoinFinalAggSplitExchange(
      spec: ExchangeSpec,
      fragmentById: Map[Int, NativeFragment]): Boolean = {
    val producerPlan = fragmentById.get(spec.producerFragmentId).map(_.rootOperator)
    val consumer = fragmentById.get(spec.consumerFragmentId)
    val consumerPlan = consumer.map(_.rootOperator)

    (spec.exchangeType == "HASH" || spec.exchangeType == "RANGE") &&
    producerPlan.exists(hasJoinHub) &&
    consumerPlan.exists(containsHashAggregate) &&
    consumer.exists(hasAggregateOutput)
  }

  private def tunePostJoinFinalAggSplitParallelism(
      fragments: Seq[NativeFragment],
      exchanges: Seq[ExchangeSpec]): Seq[NativeFragment] = {
    val fragmentById = fragments.map(fragment => fragment.id -> fragment).toMap
    val tunedConsumers = exchanges.collect {
      case spec if isPostJoinFinalAggSplitExchange(spec, fragmentById) =>
        spec.consumerFragmentId -> math.max(1, spec.numPartitions)
    }.toMap

    fragments.map {
      fragment =>
        tunedConsumers.get(fragment.id) match {
          case Some(tunedParallelism) if tunedParallelism > fragment.parallelism =>
            logWarning(
              s"MppNativeQueryExec: raising post-join final-aggregate fragment " +
                s"F${fragment.id} drivers from ${fragment.parallelism} to $tunedParallelism")
            fragment.copy(parallelism = tunedParallelism)
          case _ =>
            fragment
        }
    }
  }

  private def hasJoinHub(plan: SparkPlan): Boolean = {
    countHashJoins(plan) >= 2
  }

  private def containsHashAggregate(plan: SparkPlan): Boolean = {
    plan.treeString.toLowerCase(java.util.Locale.ROOT).contains("hashaggregate")
  }

  private def hasAggregateOutput(fragment: NativeFragment): Boolean = {
    fragment.outputAttributes.exists {
      attr =>
        val name = attr.name.toLowerCase(java.util.Locale.ROOT)
        name.contains("(") && name.contains(")")
    }
  }

  private def localHashExchangeTasks: Option[Int] = {
    positiveIntConf("spark.gluten.mpp.localHashExchangeTasks")
  }

  private def mppSparkPartitionCount: Int = {
    if (!booleanConf("spark.gluten.mpp.multiExecutor.enabled", defaultValue = false)) {
      return 1
    }

    val count = mppNonNegativeIntConf("spark.gluten.mpp.multiExecutor.numPartitions", 2)
    if (count <= 0) {
      throw new IllegalArgumentException(
        "spark.gluten.mpp.multiExecutor.numPartitions must be positive when " +
          "spark.gluten.mpp.multiExecutor.enabled=true")
    }
    logWarning(s"MppNativeQueryExec: multi-executor MPP requested; target Spark partitions=$count")
    count
  }

  private def effectiveMppSparkPartitionCount(
      requestedCount: Int,
      exchanges: Seq[ExchangeSpec],
      isWriteJob: Boolean): Int = {
    if (requestedCount <= 1) {
      return 1
    }
    val hasOnlySinglePeerExchange =
      exchanges.nonEmpty && exchanges.forall {
        exchange => exchange.exchangeType == "SINGLE" || exchange.exchangeType == "BROADCAST"
      }
    // CRITICAL (4-GPU): only DEMOTE a subquery sub-job here, NOT the MAIN query.
    // A scalar/bloom subquery (exchange graph all SINGLE/BROADCAST) runs on a "subquery-"
    // thread via SubqueryExec.relationFuture's lazy executeTake, which launches only
    // partition 0 and would block multi-peer. But a MAIN query whose exchange graph is only
    // SINGLE/BROADCAST is typically a GLOBAL AGGREGATE (TPC-H Q6 = sum over lineitem; Q14 =
    // lineitem><part with a BROADCAST dim + SINGLE final gather): its scan/partial-agg
    // fragment IS a very useful multi-worker candidate and must fan out to all peers, with
    // only the tiny final agg gathered SINGLE. Demoting it ran the whole lineitem scan on ONE
    // GPU (Q6 3.9x / Q14 5.9x slower than Presto-4GPU). The main query runs on the driver/RPC
    // thread and launches ALL peer partitions via executeCollect, so it cannot hang. Mirrors
    // the rootFedBySingleGather narrowing below (commit 19fc16d03).
    if (
      hasOnlySinglePeerExchange &&
      !isWriteJob && Thread.currentThread().getName.startsWith("subquery-")
    ) {
      logWarning(
        s"MppNativeQueryExec: keeping this MPP subquery single-peer despite requested " +
          s"$requestedCount Spark partitions because the exchange graph contains only " +
          "SINGLE/BROADCAST exchanges. " +
          "Scalar/bloom subqueries are not useful multi-worker candidates and can " +
          "otherwise block waiting for root output.")
      return 1
    }
    // Non-write global-gather sub-job (e.g. an uncorrelated scalar subquery whose body
    // has an inner GROUP BY -> HASH exchange, so the SINGLE/BROADCAST-only guard above
    // misses it): its root fragment is still fed by a SINGLE gather (the 1-row scalar
    // result), so it has no multi-worker value. Spark materializes such a sub-job via
    // executeCollect/take, which lazily launches only partition 0; a multi-peer
    // coordinator (peerCount=2) then blocks forever on peer1's cross-peer UCX exchange
    // that the lazy scalar materialization never launches (Q15's max over a per-suppkey
    // sum). Keep it single-peer. A write is excluded: the distributed write MUST stay
    // multi-peer even when its top-N input is a SINGLE gather. CRITICAL: gate on
    // the "subquery-" thread. Only a SubqueryExec.relationFuture lazy executeTake
    // (which runs on a "subquery-" thread) launches partition 0 alone and would
    // hang multi-peer. The MAIN query runs on the driver/RPC thread and launches
    // ALL peer partitions via executeCollect, so it can never hang -- demoting it
    // here wasted the 2nd GPU on ~20/22 queries whose root is a final ORDER BY /
    // global-agg SINGLE gather (the whole workload ran single-peer, GPU idle).
    if (!isWriteJob && Thread.currentThread().getName.startsWith("subquery-")) {
      val producerFragmentIds = exchanges.map(_.producerFragmentId).toSet
      val rootConsumerIds =
        exchanges.map(_.consumerFragmentId).filterNot(producerFragmentIds.contains).distinct
      val rootFedBySingleGather = rootConsumerIds.nonEmpty && rootConsumerIds.forall {
        rootId =>
          exchanges
            .filter(_.consumerFragmentId == rootId)
            .forall(_.exchangeType == "SINGLE")
      }
      if (rootFedBySingleGather) {
        logWarning(
          s"MppNativeQueryExec: keeping this MPP query single-peer despite requested " +
            s"$requestedCount Spark partitions because its root fragment is fed by a SINGLE " +
            "gather and it is not a write (uncorrelated scalar subquery / global gather); " +
            "multi-peer would block on a peer the lazy scalar materialization never launches.")
        return 1
      }
    }
    requestedCount
  }

  private def resolveMppPeers(requestedCount: Int): GlutenMppPeerResolution = {
    val manualPeerEndpointsJson = stringConf(MPP_PEER_ENDPOINTS_KEY, "")
    if (
      requestedCount <= 1 ||
      !booleanConf("spark.gluten.mpp.multiExecutor.enabled", defaultValue = false)
    ) {
      return GlutenMppPeerResolution(Array.empty, manualPeerEndpointsJson)
    }
    GlutenMppPeerResolver.resolve(sparkContext.getConf, requestedCount, manualPeerEndpointsJson)
  }

  private def broadcastProducerIds(exchanges: Seq[ExchangeSpec]): Set[Int] = {
    exchanges
      .filter(_.exchangeType == "BROADCAST")
      .map(_.producerFragmentId)
      .toSet
  }

  /**
   * Find ShuffleExchangeLike nodes that are direct exchange children of a WholeStageTransformer
   * (reachable through InputIteratorTransformer -> ColumnarInputAdapter -> ... ->
   * ShuffleExchangeLike chain).
   */
  private def findExchangeChildren(wst: WholeStageTransformer): Seq[SparkPlan] = {
    val result = mutable.ArrayBuffer[SparkPlan]()
    def collect(plan: SparkPlan): Unit = {
      plan match {
        case _: WholeStageTransformer =>
          // Stop: this is a nested WST (shouldn't happen in normal BSP plans)
          ()
        case join: HashJoinLikeExecTransformer =>
          // HashJoinLikeExecTransformer emits Substrait inputs in streamed/build order, which may
          // differ from Spark's left/right child order when the build side is switched. Keep MPP
          // exchange specs in the same order so native ValueStream replacement cannot swap inputs.
          collect(join.streamedPlan)
          collect(join.buildPlan)
        case iit: InputIteratorTransformer =>
          // InputIteratorTransformer marks a fragment boundary.
          // Its child (ColumnarInputAdapter -> Exchange) is the exchange child.
          result += iit.child
        case other =>
          other.children.foreach(collect)
      }
    }
    // Walk the WST's internal operator tree (wst.child is the root TransformSupport)
    collect(wst.child)
    result.toSeq
  }

  /** Unwrap wrapper nodes to find the Exchange (shuffle or broadcast) underneath. */
  private def unwrapToExchange(plan: SparkPlan): Option[Exchange] = {
    plan match {
      case ex: ShuffleExchangeLike => Some(ex.asInstanceOf[Exchange])
      case ex: BroadcastExchangeLike => Some(ex.asInstanceOf[Exchange])
      case reused: ReusedExchangeExec => unwrapToExchange(reused.child)
      // AQE wraps each finalized exchange in a query-stage node. By the time
      // applyCrossCutRules runs at exec time, every still-AQE-wrapped exchange
      // (e.g. the partial-agg -> final-agg HASH shuffle that the cross-cut
      // rules don't rewrite) is reachable only through ShuffleQueryStageExec /
      // BroadcastQueryStageExec. Without these two cases, unwrapToExchange
      // returns None and the surrounding ExchangeSpec is silently dropped --
      // which strips the HASH redistribution between Partial and Final
      // aggregates on TPC-H Q1 and turns Final into a per-driver pass-through
      // (4 producer drivers x 4 groups = 16 rows instead of 4).
      case stage: ShuffleQueryStageExec => unwrapToExchange(stage.plan)
      case stage: BroadcastQueryStageExec => unwrapToExchange(stage.plan)
      case iit: InputIteratorTransformer => unwrapToExchange(iit.child)
      case cia: ColumnarInputAdapter => unwrapToExchange(cia.child)
      case c2c: ColumnarToColumnarExec => unwrapToExchange(c2c.child)
      case c2r: ColumnarToRowExecBase => unwrapToExchange(c2r.child)
      // Spark inserts RowToColumnar (and its inverse ColumnarToRow) on either
      // side of an Exchange when the exchange operates on rows. Without this
      // case, non-broadcast partial-agg to final-agg gather edges are silently
      // dropped from extractedExchanges and the multi-root planner check fails.
      case r2c: RowToColumnarExecBase => unwrapToExchange(r2c.child)
      case _ => None
    }
  }

  /**
   * Unwrap wrapper nodes to a TakeOrderedAndProject (top-N) under an input slot. Such a slot is
   * walked into its own single-driver top-N fragment fed by a SINGLE gather; the parent WST then
   * consumes it via a SINGLE exchange.
   */
  private def unwrapToTopN(plan: SparkPlan): Option[TakeOrderedAndProjectExecTransformer] = {
    plan match {
      case topk: TakeOrderedAndProjectExecTransformer => Some(topk)
      case reused: ReusedExchangeExec => unwrapToTopN(reused.child)
      case stage: ShuffleQueryStageExec => unwrapToTopN(stage.plan)
      case stage: BroadcastQueryStageExec => unwrapToTopN(stage.plan)
      case iit: InputIteratorTransformer => unwrapToTopN(iit.child)
      case cia: ColumnarInputAdapter => unwrapToTopN(cia.child)
      case c2c: ColumnarToColumnarExec => unwrapToTopN(c2c.child)
      case c2r: ColumnarToRowExecBase => unwrapToTopN(c2r.child)
      case r2c: RowToColumnarExecBase => unwrapToTopN(r2c.child)
      case _ => None
    }
  }

  /**
   * Materialize only subqueries that still appear in executable expressions after the MPP scalar
   * rewrite. Rewritten uncorrelated scalar subqueries are now represented as native exchange inputs
   * (for example Q11's scalar producer fragment) and must not also be launched as driver-side
   * subquery jobs.
   */
  private def materializeScalarSubqueries(plan: SparkPlan): Unit = {
    // Do not call plan.prepare() on the whole child tree here. SparkPlan.prepare()
    // eagerly starts regular BroadcastExchange jobs, but MPP consumes those joins
    // through native BROADCAST exchanges. Preparing the whole plan can therefore
    // driver-collect a large build side before MPP starts, tripping
    // spark.driver.maxResultSize on Q16/Q18. We also do not iterate
    // SparkPlan.subqueries directly: Spark can keep stale BaseSubqueryExec handles on nodes whose
    // ScalarSubquery expressions were rewritten to attributes. Launching those handles races the
    // outer MPP graph and can deadlock the 2-GPU task slots.

    var materializedCount = 0
    plan.foreach {
      node =>
        node.expressions.foreach {
          expr =>
            expr.foreach {
              case sub: ExecSubqueryExpression =>
                materializedCount += 1
                materializeExecSubquery(sub)
              case other =>
                // Runtime-DPP bloom filters extend PlanExpression but NOT
                // ExecSubqueryExpression. Spark exposes the embedded plan via
                // a `plan` field; if the expression has a `updateResult`
                // method, drive it too (mirrors ExecSubqueryExpression).
                // Reflection keeps us insulated from the concrete class name
                // (BloomFilterSubqueryExpression vs DynamicPruningSubquery
                // shifted between Spark releases).
                val cls = other.getClass
                if (
                  cls.getName.contains("BloomFilter") ||
                  cls.getName.contains("DynamicPruning") ||
                  cls.getName.contains("PlanExpression")
                ) {
                  materializedCount += materializePlanExpression(other)
                }
            }
        }
    }
    if (materializedCount == 0) {
      logWarning(
        "materializeScalarSubqueries: no remaining executable subquery expressions; " +
          "rewritten scalars will be supplied by native MPP exchanges")
    }
  }

  private def suppressDeadBroadcastsForNativeMpp(plan: SparkPlan): Unit = {
    var markedCount = 0
    plan.foreach {
      case broadcast: ColumnarBroadcastExchangeExec if !broadcast.isMppSuppressed =>
        broadcast.setTagValue(ColumnarBroadcastExchangeExec.MppSuppressedTag, true)
        markedCount += 1
      case _ =>
    }
    if (markedCount > 0) {
      logWarning(
        s"MppNativeQueryExec: suppressed $markedCount ColumnarBroadcastExchangeExec " +
          "node(s) under native MPP execution")
    }
  }

  private def materializeExecSubquery(sub: ExecSubqueryExpression): Unit = {
    sub.plan.prepare()
    // Idempotent: updateResult blocks on the underlying future and stores the row.
    sub.updateResult()
  }

  private def materializePlanExpression(expr: Expression): Int = {
    val cls = expr.getClass
    var materialized = 0
    try {
      val planField = cls.getMethod("plan")
      val embedded = planField.invoke(expr).asInstanceOf[SparkPlan]
      if (embedded != null) {
        embedded.prepare()
        materialized += 1
      }
    } catch { case _: Throwable => () }
    try {
      cls.getMethod("updateResult").invoke(expr)
    } catch { case _: Throwable => () }
    materialized
  }

  /**
   * Peel off the synthetic hash-prefix ProjectExecTransformer that
   * VeloxSparkPlanExecApi.genColumnarShuffleExchange may inject above the shuffle child for HASH
   * partitioning. The injected project's first column is always aliased "hash_partition_key" and
   * computes Murmur3Hash. In MPP we recompute the hash inside Velox's HashPartitionFunctionSpec, so
   * the prefix column is dead weight and -- worse -- shifts every real partition-key column right
   * by one. The C++ side strips the prefix on receive (MppJniWrapper.cc) but the producer's
   * PartitionedOutputNode partitions BEFORE the strip, so partition-key indices computed against
   * the prefix-shifted schema misroute every record. On TPC-H Q1 SF1K this surfaces as 4 keys * 4
   * active F1 drivers = 16 rows instead of 4. Mirrors MppCollapseRule.stripSyntheticHashProject
   * (Plan D).
   */
  private def stripSyntheticHashProject(plan: SparkPlan): SparkPlan = plan match {
    case p: ProjectExecTransformer
        if p.projectList.nonEmpty && p.projectList.head.name == "hash_partition_key" =>
      p.child
    case other => other
  }

  /** Unwrap transparent wrapper nodes (ColumnarToColumnar, resize batches, etc.). */
  private def unwrapTransparent(plan: SparkPlan): SparkPlan = {
    plan match {
      case c2c: ColumnarToColumnarExec => unwrapTransparent(c2c.child)
      // Q21-style fix: Spark inserts AssignUniqueId before LeftSemi/LeftAnti joins
      // to dedupe; that op declares outputPartitioning = UnknownPartitioning, which
      // makes EnsureRequirements add a SPURIOUS shuffle even when the input is
      // already hash-partitioned on the join key. When opted in
      // (spark.gluten.mpp.assignUniqueIdTransparent) and the child carries a real
      // HashPartitioning, treat it as transparent so the surrounding fragment
      // walk does not see an artificial fragment boundary.
      case other
          if isAssignUniqueIdLike(other) && assignUniqueIdTransparent &&
            isHashPartitioned(other.children.head) =>
        unwrapTransparent(other.children.head)
      case other => other
    }
  }

  /**
   * Class-name match for Spark's AssignUniqueId / AssignUniqueIdExec across versions. We avoid a
   * direct import because the operator's package has shifted between Spark releases (and Gluten
   * supports several); a name match is stable enough for a transparency hint.
   */
  private def isAssignUniqueIdLike(plan: SparkPlan): Boolean = {
    if (plan.children.size != 1) return false
    val cn = plan.getClass.getName
    cn.endsWith(".AssignUniqueId") ||
    cn.endsWith(".AssignUniqueIdExec") ||
    cn.endsWith("AssignUniqueIdExecTransformer")
  }

  private def isHashPartitioned(plan: SparkPlan): Boolean = {
    plan.outputPartitioning match {
      case _: HashPartitioning => true
      case _ => false
    }
  }

  private def assignUniqueIdTransparent: Boolean = {
    SQLConf.get
      .getConfString("spark.gluten.mpp.assignUniqueIdTransparent", "false")
      .toBoolean
  }

  private def fuseBroadcastBuildsEnabled: Boolean = {
    SQLConf.get.getConfString("spark.gluten.mpp.fuseBroadcastBuilds", "false").toBoolean
  }

  private def rejectUnsafeBroadcastFusionIfRequested(context: String): Unit = {
    if (!fuseBroadcastBuildsEnabled) return
    logWarning(
      s"$context: ignoring spark.gluten.mpp.fuseBroadcastBuilds=true; using native " +
        "BROADCAST exchange to avoid sharing one JNI broadcast iterator across fanout drivers")
  }

  /** Classify the partitioning into an exchange type string and extract partition keys. */
  private def classifyPartitioning(partitioning: Partitioning): (String, Seq[Attribute]) = {
    partitioning match {
      case hash: HashPartitioning =>
        val keys = hash.expressions.collect { case attr: Attribute => attr }
        ("HASH", keys)
      case range: RangePartitioning =>
        // GPU range-partitioning is not yet implemented. Treat ORDER BY
        // exchanges as hash-partitioned on the sort keys: equal keys still
        // land in the same partition (correctness preserved), only
        // intra-partition sort order is delegated to the downstream sort.
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
          s"MppNativeQueryExec: unexpected partitioning type: " +
            s"${other.getClass.getSimpleName}")
        ("UNKNOWN", Seq.empty)
    }
  }

  /**
   * Infer the parallelism for a fragment based on its output partitioning.
   *
   * In MPP mode every fragment runs in-process as a set of Velox drivers, so the "parallelism" here
   * becomes the driver count passed to task->start(numDrivers). With the default of
   * spark.sql.shuffle.partitions (=200) on a 16-core single-GPU executor this spawns 200 driver
   * threads per fragment, which (a) thrashes on 16 cores, (b) contends heavily on the GpuSemaphore
   * (maxConcurrentGpuTasks ~= 6), and (c) leaves dozens of drivers with zero scan splits that still
   * have to start/teardown. In practice this also correlates with a producer livelock that prevents
   * F0 from ever signaling noMoreData (observed 2026-04-17).
   *
   * Cap driver count at the executor core count. On multi-executor M2 this still gives the right
   * parallelism (each executor caps locally).
   */
  private def inferParallelism(plan: SparkPlan): Int = {
    val raw = plan.outputPartitioning match {
      case p if p.numPartitions > 0 => p.numPartitions
      case _ =>
        SQLConf.get.getConfString("spark.sql.shuffle.partitions", "200").toInt
    }
    val cores = SQLConf.get.getConfString("spark.executor.cores", "16").toInt
    val coreCapped = math.min(raw, math.max(cores, 1))
    positiveIntConf("spark.gluten.mpp.maxDriversPerFragment") match {
      case Some(maxDrivers) =>
        val capped = math.min(coreCapped, maxDrivers)
        if (capped != coreCapped) {
          logWarning(
            s"MppNativeQueryExec: capping fragment drivers from $coreCapped to $capped " +
              s"(raw=$raw, executorCores=$cores) via " +
              s"spark.gluten.mpp.maxDriversPerFragment")
        }
        capped
      case None => coreCapped
    }
  }

  private def positiveIntConf(key: String): Option[Int] = {
    val value = SQLConf.get.getConfString(key, "").trim match {
      case "" =>
        Option(SparkEnv.get)
          .flatMap(env => Option(env.conf.get(key, null)))
          .orElse(sys.props.get(key))
          .getOrElse("")
          .trim
      case fromSqlConf if fromSqlConf.equalsIgnoreCase("null") => ""
      case fromSqlConf => fromSqlConf
    }
    if (value.isEmpty || value.equalsIgnoreCase("null")) {
      None
    } else {
      val parsed = value.toInt
      if (parsed <= 0) {
        throw new IllegalArgumentException(s"$key must be a positive integer, got $value")
      }
      Some(parsed)
    }
  }

  // spark.gluten.mpp.largeParquetScanChunks (default false): session-level cuDF
  // parquet chunk/pass read limits for MPP scans. See MppNativeQueryRDD companion.

  private def booleanConf(key: String, defaultValue: Boolean): Boolean = {
    SQLConf.get.getConfString(key, "").trim match {
      case "" =>
        Option(SparkEnv.get)
          .flatMap(env => Option(env.conf.get(key, null)))
          .orElse(sys.props.get(key))
          .map(_.trim)
          .filter(_.nonEmpty)
          .filterNot(_.equalsIgnoreCase("null"))
          .map(_.toBoolean)
          .getOrElse(defaultValue)
      case fromSqlConf if fromSqlConf.equalsIgnoreCase("null") => defaultValue
      case fromSqlConf => fromSqlConf.toBoolean
    }
  }

  private def bytesConf(key: String, defaultValue: BigInt): BigInt = {
    val raw = SQLConf.get.getConfString(key, "").trim match {
      case "" =>
        Option(sparkContext.getConf.get(key, null))
          .orElse(Option(SparkEnv.get).flatMap(env => Option(env.conf.get(key, null))))
          .orElse(sys.props.get(key))
          .map(_.trim)
          .filter(_.nonEmpty)
          .filterNot(_.equalsIgnoreCase("null"))
          .getOrElse(defaultValue.toString)
      case fromSqlConf if fromSqlConf.equalsIgnoreCase("null") => defaultValue.toString
      case fromSqlConf => fromSqlConf
    }
    parseBytes(raw).getOrElse {
      logWarning(s"MppNativeQueryExec: invalid byte size for $key=$raw; using $defaultValue")
      defaultValue
    }
  }

  private def parseBytes(raw: String): Option[BigInt] = {
    val normalized = raw.trim.toLowerCase(java.util.Locale.ROOT)
    if (normalized.isEmpty || normalized == "null") {
      return None
    }
    val units = Seq(
      "kb" -> 10L,
      "k" -> 10L,
      "mb" -> 20L,
      "m" -> 20L,
      "gb" -> 30L,
      "g" -> 30L,
      "tb" -> 40L,
      "t" -> 40L,
      "b" -> 0L)
    val (number, shift) = units
      .find { case (suffix, _) => normalized.endsWith(suffix) }
      .map {
        case (suffix, unitShift) =>
          (normalized.stripSuffix(suffix), unitShift)
      }
      .getOrElse((normalized, 0L))
    try {
      val parsed = BigInt(number.trim)
      if (parsed >= 0) {
        Some(parsed << shift.toInt)
      } else {
        None
      }
    } catch {
      case _: NumberFormatException => None
    }
  }

  private def stringConf(key: String, defaultValue: String): String = {
    SQLConf.get.getConfString(key, "").trim match {
      case "" =>
        Option(sparkContext.getConf.get(key, null))
          .orElse(Option(SparkEnv.get).flatMap(env => Option(env.conf.get(key, null))))
          .orElse(sys.props.get(key))
          .map(_.trim)
          .filter(_.nonEmpty)
          .filterNot(_.equalsIgnoreCase("null"))
          .getOrElse(defaultValue)
      case fromSqlConf if fromSqlConf.equalsIgnoreCase("null") => defaultValue
      case fromSqlConf => fromSqlConf
    }
  }

  private def mppNonNegativeIntConf(key: String, defaultValue: Int): Int = {
    // Prefer SparkContext conf: SF1K suites set these on SparkConf; executor-side SQLConf
    // lookups have historically missed them while SQLConf.getConfString stayed empty, which
    // silently disabled inbound-exchange guardrails (see Q18 fragment fan-in).
    def trimConf(value: Option[String]): Option[String] =
      value
        .map(_.trim)
        .filter(_.nonEmpty)
        .filter(!_.equalsIgnoreCase("null"))

    val sqlRaw = SQLConf.get.getConfString(key, "").trim
    val rawJoined = Seq(
      trimConf(Option(sparkContext.getConf.get(key, null))),
      trimConf(Some(sqlRaw)),
      trimConf(Option(SparkEnv.get).flatMap(env => Option(env.conf.get(key, null)))),
      trimConf(sys.props.get(key))
    ).flatten.headOption.getOrElse(defaultValue.toString)
    // Maven scalatest forwards sometimes inject the literal "null" when a POM property is unset.
    val raw =
      Option(rawJoined)
        .map(_.trim)
        .filter(_.nonEmpty)
        .filter(!_.equalsIgnoreCase("null"))
        .getOrElse(defaultValue.toString)
    try {
      val parsed = raw.toInt
      if (parsed < 0) {
        logWarning(
          s"MppNativeQueryExec: invalid negative integer for $key=$raw; using $defaultValue")
        defaultValue
      } else {
        parsed
      }
    } catch {
      case _: NumberFormatException =>
        logWarning(s"MppNativeQueryExec: invalid integer for $key=$raw; using $defaultValue")
        defaultValue
    }
  }

  /**
   * Generate a Substrait plan for a fragment extracted from the child BSP plan.
   *
   * Uses [[WholeStageTransformer.doWholeStageTransform()]] which calls transform() on the WST's
   * internal operator tree. The transform stops at [[InputIteratorTransformer]] boundaries,
   * producing a Substrait plan that covers exactly the operators between exchange boundaries --
   * which is what we want for each MPP fragment.
   *
   * @return
   *   serialized Substrait plan bytes
   */
  private def generateSubstraitForFragment(fragment: NativeFragment): Array[Byte] = {
    fragment.rootOperator match {
      case wst: WholeStageTransformer =>
        logWarning(
          s"generateSubstraitForFragment: fragment ${fragment.id} " +
            s"WST stageId=${wst.stageId}")

        val wsCtx = wst.doWholeStageTransform()
        val planNode = wsCtx.root
        val planBytes = planNode.toProtobuf.toByteArray

        logWarning(
          s"generateSubstraitForFragment: fragment ${fragment.id} " +
            s"Substrait plan size=${planBytes.length} bytes")
        logDebug(
          s"generateSubstraitForFragment: fragment ${fragment.id} " +
            s"Substrait JSON: ${SubstraitPlanPrinterUtil.substraitPlanToJson(planNode.toProtobuf)}")

        planBytes

      case other =>
        throw new IllegalStateException(
          s"MppNativeQueryExec: fragment ${fragment.id} root operator " +
            s"${other.getClass.getSimpleName} is not a WholeStageTransformer. " +
            s"Only WholeStageTransformer fragments are supported in Phase 2.")
    }
  }

  /**
   * Generate a Substrait plan for a single fragment.
   *
   * Must be called on the driver side where SparkPlan.sparkContext is available. Uses the same
   * pattern as [[WholeStageTransformer.doWholeStageTransform()]]:
   *   1. Create a SubstraitContext 2. Call transform() on the fragment's root operator
   *      (TransformSupport) 3. Build a PlanNode and serialize to bytes
   */
  private def generateSubstraitPlan(fragment: NativeFragment): Array[Byte] = {
    // Unwrap non-TransformSupport wrappers to find the actual native operator
    val rootOp = unwrapToTransformSupport(fragment.rootOperator)
    logWarning(
      s"generateSubstraitPlan: fragment ${fragment.id} " +
        s"original=${fragment.rootOperator.getClass.getSimpleName} " +
        s"unwrapped=${rootOp.getClass.getSimpleName}")
    rootOp match {
      case ts: TransformSupport =>
        val substraitContext = new SubstraitContext
        val childCtx = ts.transform(substraitContext)
        if (childCtx == null) {
          throw new IllegalStateException(
            s"MppNativeQueryExec: fragment ${fragment.id} root operator " +
              s"${rootOp.getClass.getSimpleName} returned null from transform()")
        }

        val outNames = childCtx.outputAttributes
          .map(ConverterUtils.genColumnNameWithExprId)
          .asJava

        val planNode = if (BackendsApiManager.getSettings.needOutputSchemaForPlan()) {
          val outputTypeNodes =
            new java.util.ArrayList[org.apache.gluten.substrait.`type`.TypeNode]()
          for (attr <- childCtx.outputAttributes) {
            outputTypeNodes.add(ConverterUtils.getTypeNode(attr.dataType, attr.nullable))
          }
          val outputSchema =
            org.apache.gluten.substrait.`type`.TypeBuilder.makeStruct(false, outputTypeNodes)

          PlanBuilder.makePlan(
            substraitContext,
            Lists.newArrayList(childCtx.root),
            outNames,
            outputSchema,
            null)
        } else {
          PlanBuilder.makePlan(substraitContext, Lists.newArrayList(childCtx.root), outNames)
        }

        logDebug(
          s"MppNativeQueryExec: fragment ${fragment.id} Substrait plan: " +
            SubstraitPlanPrinterUtil.substraitPlanToJson(planNode.toProtobuf))

        planNode.toProtobuf.toByteArray

      case other =>
        throw new IllegalStateException(
          s"MppNativeQueryExec: fragment ${fragment.id} root operator " +
            s"${other.getClass.getSimpleName} is not a TransformSupport")
    }
  }

  /**
   * Unwrap non-TransformSupport wrappers to find the actual native operator. Fragments may have
   * ShuffleExchangeLike, ColumnarToColumnarExec, or ColumnarToRowExecBase as root - we need the
   * TransformSupport child.
   */
  private def unwrapToTransformSupport(plan: SparkPlan): SparkPlan = {
    plan match {
      case ts: TransformSupport => ts
      case c2r: ColumnarToRowExecBase => unwrapToTransformSupport(c2r.child)
      case c2c: ColumnarToColumnarExec => unwrapToTransformSupport(c2c.child)
      case exchange: org.apache.spark.sql.execution.exchange.ShuffleExchangeLike =>
        unwrapToTransformSupport(exchange.child)
      case _ =>
        logWarning(s"unwrapToTransformSupport: cannot unwrap ${plan.getClass.getSimpleName}")
        plan
    }
  }

  /**
   * Serialize exchange specifications to JSON for the native side.
   *
   * Resolves partition keys to column indices against the producer fragment's output schema. The
   * native side uses these indices directly as Velox keyChannels (Velox synthesizes its own column
   * names like n<frag>_<idx>, so Spark-style names such as "l_returnflag#84" would never match).
   */
  private def serializeExchangeSpecs(
      specs: Seq[ExchangeSpec],
      producerFragments: Seq[NativeFragment]): String = {
    val entries = specs.map {
      spec =>
        val producerOutput =
          if (spec.producerFragmentId >= 0 && spec.producerFragmentId < producerFragments.size) {
            producerFragments(spec.producerFragmentId).outputAttributes
          } else {
            Seq.empty[Attribute]
          }
        val indices = spec.partitionKeys.flatMap {
          attr =>
            val idx = producerOutput.indexWhere(_.exprId == attr.exprId)
            if (idx < 0) {
              logWarning(
                s"MppNativeQueryExec: partition key ${attr.name}#${attr.exprId.id} not in " +
                  s"producer fragment ${spec.producerFragmentId} output " +
                  s"(size=${producerOutput.size}); exchange will fall back to round-robin")
              None
            } else {
              Some(idx)
            }
        }
        val keyIndicesJson = indices.mkString("[", ", ", "]")
        s"""{
           |  "id": ${spec.id},
           |  "producerFragmentId": ${spec.producerFragmentId},
           |  "consumerFragmentId": ${spec.consumerFragmentId},
           |  "exchangeType": "${spec.exchangeType}",
           |  "numPartitions": ${spec.numPartitions},
           |  "exchangeNodeId": "mpp_exchange_source_${spec.id}",
           |  "partitionKeyIndices": $keyIndicesJson
           |}""".stripMargin
    }
    entries.mkString("[", ", ", "]")
  }

  private def fragmentSummary: String = {
    fragments
      .map {
        f =>
          val rootStr = if (f.rootOperator != null) f.rootOperator.simpleString(10) else "<pending>"
          s"  Fragment ${f.id}: parallelism=${f.parallelism}, " +
            s"output=${f.outputAttributes.map(_.name).mkString("[", ", ", "]")}, " +
            s"root=$rootStr"
      }
      .mkString("\n")
  }

  private def exchangeSummary: String = {
    if (exchanges.isEmpty) return ""
    "\n" + exchanges
      .map {
        e =>
          s"  Exchange ${e.id}: F${e.producerFragmentId} -> F${e.consumerFragmentId} " +
            s"(${e.exchangeType}, ${e.numPartitions} partitions" +
            (if (e.partitionKeys.nonEmpty) {
               s", keys=${e.partitionKeys.map(_.name).mkString("[", ", ", "]")}"
             } else "") +
            ")"
      }
      .mkString("\n")
  }

  // --- Scan split info extraction for MPP Phase 3 ---

  /**
   * Extract serialized split infos for a fragment's scan nodes.
   *
   * For scan-containing fragments (leaf fragments with file scans), this collects ALL partition
   * file splits from each LeafTransformSupport and merges them into a single protobuf per leaf.
   * This gives the native MppQueryCoordinator all file paths to scan.
   *
   * Non-scan fragments (consumers with exchange inputs) return an empty array.
   *
   * @return
   *   Array of protobuf byte arrays, one per leaf scan in the fragment.
   */
  private def extractSplitInfosForFragment(fragment: NativeFragment): Array[Array[Byte]] = {
    fragment.rootOperator match {
      case wst: WholeStageTransformer =>
        val leafTransformers = findLeafTransformersInWST(wst)
        if (leafTransformers.nonEmpty) {
          logWarning(
            s"extractSplitInfosForFragment: fragment ${fragment.id} has " +
              s"${leafTransformers.size} leaf scan(s)")
          leafTransformers.map {
            leaf =>
              val allPartitionSplits = leaf.getSplitInfos
              logWarning(
                s"  leaf ${leaf.getClass.getSimpleName}: " +
                  s"${allPartitionSplits.size} partition splits")
              mergeSplitInfosToBytes(allPartitionSplits)
          }.toArray
        } else {
          Array.empty[Array[Byte]]
        }
      case _ => Array.empty[Array[Byte]]
    }
  }

  /**
   * Find all [[LeafTransformSupport]] nodes within a WholeStageTransformer. These are the file scan
   * operators (e.g., FileSourceScanExecTransformer). Mirrors
   * WholeStageTransformer.findAllLeafTransformers().
   */
  private def findLeafTransformersInWST(wst: WholeStageTransformer): Seq[LeafTransformSupport] = {
    def collect(plan: SparkPlan): Seq[LeafTransformSupport] = plan match {
      case leaf: LeafTransformSupport => Seq(leaf)
      case ts: TransformSupport => ts.children.flatMap(collect)
      case _ => Seq.empty
    }
    collect(wst.child)
  }

  /**
   * Merge split infos from all partitions into a single protobuf.
   *
   * In BSP, each partition gets its own SplitInfo (a subset of files). For MPP, the coordinator
   * needs ALL files, so we merge all partitions' file items into one ReadRel.LocalFiles protobuf.
   */
  private def mergeSplitInfosToBytes(splitInfos: Seq[SplitInfo]): Array[Byte] = {
    val builder = ReadRel.LocalFiles.newBuilder()
    splitInfos.foreach {
      si =>
        // SplitInfo.toProtobuf returns Message; parse as typed LocalFiles
        val localFiles = ReadRel.LocalFiles.parseFrom(si.toProtobuf.toByteArray)
        builder.addAllItems(localFiles.getItemsList)
    }
    val merged = builder.build()
    logWarning(
      s"mergeSplitInfosToBytes: merged ${splitInfos.size} partitions " +
        s"into ${merged.getItemsCount} file items")
    merged.toByteArray
  }

  // --- Substrait plan dump (offline C++ replay) ---

  /**
   * If `spark.gluten.mpp.substraitDumpDir` is set, write the fully-collapsed multi-fragment
   * Substrait plan plus exchange wiring to that directory. This lets a separate C++ harness
   * (`mpp-substrait-runner`) replay the plan directly against Velox-cudf without going through
   * Spark.
   *
   * Layout per query: `<dumpDir>/<queryId>/`:
   *   - `fragment-<id>.pb` -- raw Substrait Plan protobuf bytes (byte-identical to what the JNI
   *     bridge receives).
   *   - `manifest.json` -- small human-readable descriptor of fragments + exchanges.
   *   - `query.sql` -- best-effort logical / child plan text for debugging.
   *
   * The dump is a driver-side side-effect only. It must never fail the query: any I/O error is
   * logged as a warning and the normal native execution continues.
   */
  private def dumpPlanIfEnabled(
      fragmentPlans: Array[Array[Byte]],
      numDriversPerFragment: Array[Int],
      fragmentSpecs: Seq[NativeFragment],
      exchangeSpecs: Seq[ExchangeSpec],
      exchangeSpecsJson: String,
      fragmentSplitInfos: Array[Array[Array[Byte]]]): Unit = {
    val dumpDir =
      SQLConf.get.getConfString("spark.gluten.mpp.substraitDumpDir", "").trim
    if (dumpDir.isEmpty) {
      return
    }
    try {
      val queryId = deriveQueryId()
      val outDir = Paths.get(dumpDir, queryId)
      Files.createDirectories(outDir)

      // 1. Fragment protobufs -- identical bytes to what goes through JNI.
      var i = 0
      while (i < fragmentPlans.length) {
        val target = outDir.resolve(f"fragment-$i%d.pb")
        Files.write(target, fragmentPlans(i))
        i += 1
      }

      // 1b. Per-fragment scan splits. One file per leaf scan inside a fragment;
      // raw ReadRel.LocalFiles protobuf bytes, byte-identical to what the JNI
      // bridge passes as splitInfosPerFragArr[fragmentId][leafIdx]. Fragments
      // without leaf scans (exchange-only consumer fragments) write no files.
      var fi = 0
      while (fi < fragmentSplitInfos.length) {
        val leaves = fragmentSplitInfos(fi)
        var li = 0
        while (li < leaves.length) {
          val target = outDir.resolve(f"splits-frag$fi%d-leaf$li%d.pb")
          Files.write(target, leaves(li))
          li += 1
        }
        fi += 1
      }

      // 2. manifest.json
      val manifestBytes =
        buildManifestJson(
          queryId,
          fragmentPlans,
          numDriversPerFragment,
          fragmentSpecs,
          exchangeSpecs,
          fragmentSplitInfos)
          .getBytes(StandardCharsets.UTF_8)
      Files.write(outDir.resolve("manifest.json"), manifestBytes)

      // 3. exchange-specs.json -- byte-for-byte the same JSON that currently goes through JNI as
      // the `exchangeSpecsJson` parameter to nativeCreateMppQuery. The C++ harness can feed this
      // directly into the same parseExchangeSpecs() code path.
      Files.write(
        outDir.resolve("exchange-specs.json"),
        exchangeSpecsJson.getBytes(StandardCharsets.UTF_8))

      // 4. query.sql -- best-effort plan text for debugging.
      val sqlBytes = deriveQueryText().getBytes(StandardCharsets.UTF_8)
      Files.write(outDir.resolve("query.sql"), sqlBytes)

      val totalSplitFiles = fragmentSplitInfos.map(_.length).sum
      logWarning(
        s"MppNativeQueryExec: dumped Substrait plan " +
          s"(${fragmentPlans.length} fragments, ${exchangeSpecs.size} exchanges, " +
          s"$totalSplitFiles split files) to $outDir")
    } catch {
      case t: Throwable =>
        logWarning(
          s"MppNativeQueryExec: failed to dump Substrait plan to '$dumpDir' " +
            s"(${t.getClass.getSimpleName}: ${t.getMessage}); continuing with native execution",
          t
        )
    }
  }

  private def postMppPlanEvent(
      fragmentPlans: Array[Array[Byte]],
      numDriversPerFragment: Array[Int],
      fragmentSpecs: Seq[NativeFragment],
      exchangeSpecs: Seq[ExchangeSpec],
      exchangeSpecsJson: String,
      fragmentSplitInfos: Array[Array[Array[Byte]]],
      fusedBroadcastsByConsumer: Map[Int, Seq[FusedBroadcast]]): Unit = {
    val executionId = Option(sparkContext.getLocalProperty(SQLExecution.EXECUTION_ID_KEY))
      .filter(_.nonEmpty)
    if (executionId.isEmpty) {
      logDebug("MppNativeQueryExec: skip MPP plan event because SQL execution id is unavailable")
      return
    }

    val queryId = deriveQueryId()
    val dumpDir = stringConf("spark.gluten.mpp.substraitDumpDir", "")
    val dumpPath = if (dumpDir.nonEmpty) {
      Paths.get(dumpDir, queryId).toString
    } else {
      ""
    }

    val requestedCaptureEnabled =
      booleanConf("spark.gluten.mpp.veloxPlan.eventLog.enabled", false)
    val maxChars = mppNonNegativeIntConf("spark.gluten.mpp.veloxPlan.eventLog.maxChars", 262144)
    val (plans, captureError) =
      if (!requestedCaptureEnabled) {
        (Seq.empty[String], "")
      } else if (Thread.currentThread().getName.startsWith("subquery-")) {
        (
          Seq.empty[String],
          "Final Velox plan capture skipped during scalar subquery materialization.")
      } else if (exchangeSpecs.isEmpty) {
        (Seq.empty[String], "Final Velox plan capture skipped for single-fragment MPP plan.")
      } else if (fusedBroadcastsByConsumer.values.exists(_.nonEmpty)) {
        (
          Seq.empty[String],
          "Final Velox plan capture skipped because fused broadcast inputs require " +
            "runtime iterators during native plan conversion.")
      } else {
        captureFinalVeloxPlans(
          fragmentPlans,
          numDriversPerFragment,
          exchangeSpecsJson,
          fragmentSplitInfos)
      }

    val fragments = truncatePlanFragments(plans, maxChars)
    val event = GlutenMppPlanEvent(
      executionId.get.toLong,
      queryId,
      fragmentSpecs.size,
      exchangeSpecs.size,
      dumpPath,
      plans.map(_.length.toLong).sum,
      if (plans.nonEmpty) sha256Hex(plans.mkString("\n")) else "",
      fragments.exists(_.truncated),
      requestedCaptureEnabled,
      captureError,
      fragments
    )
    GlutenUIUtils.postEvent(sparkContext, event)
  }

  private def captureFinalVeloxPlans(
      fragmentPlans: Array[Array[Byte]],
      numDriversPerFragment: Array[Int],
      exchangeSpecsJson: String,
      fragmentSplitInfos: Array[Array[Array[Byte]]]): (Seq[String], String) = {
    try {
      val runtime = Runtimes.contextInstance(BackendsApiManager.getBackendName, "MppQueryExplain")
      val jniWrapper = MppQueryJniWrapper.create(runtime)
      val plans = jniWrapper.nativeExplainMppQuery(
        fragmentPlans,
        numDriversPerFragment,
        exchangeSpecsJson.getBytes(StandardCharsets.UTF_8),
        fragmentSplitInfos,
        null,
        null)
      (Option(plans).map(_.toSeq).getOrElse(Seq.empty), "")
    } catch {
      case t: Throwable =>
        logWarning(
          s"MppNativeQueryExec: failed to capture final Velox plan for event log " +
            s"(${t.getClass.getSimpleName}: ${t.getMessage}); continuing with native execution",
          t
        )
        (Seq.empty, s"${t.getClass.getSimpleName}: ${Option(t.getMessage).getOrElse("")}")
    }
  }

  private def truncatePlanFragments(
      plans: Seq[String],
      maxChars: Int): Seq[GlutenMppPlanFragmentEvent] = {
    var remaining = maxChars
    plans.zipWithIndex.map {
      case (plan, fragmentId) =>
        val safePlan = Option(plan).getOrElse("")
        val captured = if (remaining <= 0) {
          ""
        } else {
          safePlan.take(remaining)
        }
        remaining = math.max(0, remaining - captured.length)
        GlutenMppPlanFragmentEvent(
          fragmentId,
          captured,
          safePlan.length,
          sha256Hex(safePlan),
          captured.length < safePlan.length)
    }
  }

  private def sha256Hex(value: String): String = {
    val digest = MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8))
    digest.map(b => "%02x".format(b & 0xff)).mkString
  }

  /**
   * Best-effort stable identifier for this query. Uses Spark's SQL execution id when running inside
   * a DataFrame action (the normal path) so the dump directory lines up with the Spark UI query id.
   * Falls back to a random UUID when unavailable (e.g. explain-only callers).
   */
  private def deriveQueryId(): String = {
    val execId = Option(sparkContext.getLocalProperty(SQLExecution.EXECUTION_ID_KEY))
    execId.filter(_.nonEmpty).getOrElse("q-" + UUID.randomUUID().toString)
  }

  /** Render the original logical plan (if we captured it) or the child plan tree as debug SQL. */
  private def deriveQueryText(): String = {
    if (originalLogicalPlan != null) {
      originalLogicalPlan.toString()
    } else {
      preparedChildPlan.treeString
    }
  }

  /**
   * Build a JSON manifest describing the dumped plan. We inline JSON construction to avoid adding a
   * new JSON library dependency; this matches [[serializeExchangeSpecs]] above, which does the
   * same. Strings are escaped for the two characters that actually show up in plan text:
   * backslashes and double quotes. Control characters (newlines, tabs) are passed through inside
   * JSON strings via \n / \t.
   */
  private def buildManifestJson(
      queryId: String,
      fragmentPlans: Array[Array[Byte]],
      numDriversPerFragment: Array[Int],
      fragmentSpecs: Seq[NativeFragment],
      exchangeSpecs: Seq[ExchangeSpec],
      fragmentSplitInfos: Array[Array[Array[Byte]]]): String = {
    def esc(s: String): String = {
      val sb = new StringBuilder(s.length + 2)
      var i = 0
      while (i < s.length) {
        val c = s.charAt(i)
        c match {
          case '\\' => sb.append("\\\\")
          case '"' => sb.append("\\\"")
          case '\n' => sb.append("\\n")
          case '\r' => sb.append("\\r")
          case '\t' => sb.append("\\t")
          case ch if ch < 0x20 => sb.append(f"\\u$ch%04x")
          case ch => sb.append(ch)
        }
        i += 1
      }
      sb.toString()
    }
    def schemaString(attrs: Seq[Attribute]): String = {
      attrs
        .map(a => s"${a.name}:${a.dataType.catalogString}${if (a.nullable) "?" else ""}")
        .mkString(",")
    }

    // Fragment entries. isFinal = the fragment whose output Spark ultimately consumes;
    // by convention fragment id 0 in both the Plan C / Plan D construction paths.
    val fragmentEntries = fragmentSpecs.zipWithIndex.map {
      case (frag, idx) =>
        val parallelism = if (idx < numDriversPerFragment.length) {
          numDriversPerFragment(idx)
        } else frag.parallelism
        val planSize = if (idx < fragmentPlans.length) fragmentPlans(idx).length else 0
        // Split files for this fragment's leaf scans. Empty for exchange-only
        // consumer fragments. Ordering matches JNI splitInfosPerFragArr[fragId].
        val splitFiles = if (idx < fragmentSplitInfos.length) {
          fragmentSplitInfos(idx).indices
            .map(li => s""""splits-frag$idx-leaf$li.pb"""")
            .mkString("[", ", ", "]")
        } else "[]"
        s"""    {
           |      "id": ${frag.id},
           |      "parallelism": $parallelism,
           |      "outputSchema": "${esc(schemaString(frag.outputAttributes))}",
           |      "planBytes": $planSize,
           |      "planFile": "fragment-${frag.id}.pb",
           |      "splitFiles": $splitFiles,
           |      "isFinal": ${frag.id == 0}
           |    }""".stripMargin
    }

    val exchangeEntries = exchangeSpecs.map {
      spec =>
        val keyNames = spec.partitionKeys
          .map(a => "\"" + esc(a.name) + "\"")
          .mkString("[", ", ", "]")
        s"""    {
           |      "id": ${spec.id},
           |      "producer": ${spec.producerFragmentId},
           |      "consumer": ${spec.consumerFragmentId},
           |      "type": "${esc(spec.exchangeType)}",
           |      "numPartitions": ${spec.numPartitions},
           |      "partitioningKeys": $keyNames
           |    }""".stripMargin
    }

    s"""{
       |  "queryId": "${esc(queryId)}",
       |  "numFragments": ${fragmentSpecs.size},
       |  "fragments": [
       |${fragmentEntries.mkString(",\n")}
       |  ],
       |  "exchanges": [
       |${exchangeEntries.mkString(",\n")}
       |  ]
       |}
       |""".stripMargin
  }

}
