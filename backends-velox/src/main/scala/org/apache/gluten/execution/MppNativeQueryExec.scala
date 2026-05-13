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
import org.apache.gluten.events.{GlutenMppPlanEvent, GlutenMppPlanFragmentEvent}
import org.apache.gluten.expression.ConverterUtils
import org.apache.gluten.extension.{ExchangeSpec, MppParallelSortSplitRule, MppRemoveRedundantShuffleRule, MppSinglePartitionSortRule, NativeFragment}
import org.apache.gluten.extension.columnar.transition.{Convention, ConventionReq}
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
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.expressions.SortOrder
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight, BuildSide, JoinSelectionHelper}
import org.apache.spark.sql.catalyst.plans.{InnerLike, LeftOuter, LeftSemi, RightOuter}
import org.apache.spark.sql.catalyst.plans.logical.{Join, Statistics}
import org.apache.spark.sql.catalyst.plans.physical.{BroadcastPartitioning, HashPartitioning, Partitioning, RangePartitioning, RoundRobinPartitioning, SinglePartition}
import org.apache.spark.sql.execution.{ColumnarCollapseTransformStages, ColumnarInputAdapter, ExecSubqueryExpression, InputIteratorTransformer, SparkPlan, SQLExecution, UnaryExecNode}
import org.apache.spark.sql.execution.adaptive.{BroadcastQueryStageExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeLike, Exchange, ReusedExchangeExec, ShuffleExchangeLike}
import org.apache.spark.sql.execution.joins.BuildSideRelation
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

case class MppNativeQueryExec(
    child: SparkPlan,
    fragments: Seq[NativeFragment],
    exchanges: Seq[ExchangeSpec],
    @transient originalLogicalPlan: org.apache.spark.sql.catalyst.plans.logical.LogicalPlan = null
) extends UnaryExecNode
  with GlutenPlan
  with Logging {

  // --- Output schema and partitioning (delegate to child) ---

  override def output: Seq[Attribute] = child.output
  override def outputPartitioning: Partitioning = child.outputPartitioning
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

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

  // --- Metrics ---

  @transient
  override lazy val metrics: Map[String, SQLMetric] = Map(
    "totalQueryTimeMs" -> SQLMetrics.createTimingMetric(sparkContext, "total query time (ms)"),
    "numFragments" -> SQLMetrics.createMetric(sparkContext, "number of fragments"),
    "numExchanges" -> SQLMetrics.createMetric(sparkContext, "number of exchanges"),
    "outputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "outputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches")
  )

  // --- Execution ---

  override protected def doExecute(): RDD[InternalRow] = {
    // Gluten ColumnarBatch uses IndicatorVectorBase (native-pointer wrappers),
    // which does not support Spark's standard row-access path
    // (ColumnarBatchRow.copy -> isNullAt throws UnsupportedOperationException).
    // Delegate to VeloxColumnarToRowExec, which does the conversion via JNI.
    VeloxColumnarToRowExec(this).doExecute()
  }

  // supportsColumnar is already true via GlutenPlan

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    logWarning(
      s"MppNativeQueryExec: executing with ${fragments.size} fragments " +
        s"and ${exchanges.size} exchanges")

    // Materialize any ScalarSubquery results in the child plan before driver-side
    // Substrait generation. ScalarSubqueryTransformer.doTransform calls
    // query.eval(InternalRow.empty), which requires updateResult() to have run.
    // In BSP this happens via SparkPlan.prepareSubqueries on the enclosing plan;
    // because MppNativeQueryExec bypasses child.executeColumnar() we drive it
    // explicitly here so Q11/Q22-style uncorrelated subqueries become literals.
    materializeScalarSubqueries(child)

    // Plan D: child is the original BSP plan (with ShuffleExchange intact).
    // Phase 1: delegate to child BSP execution to prove the wrap chain works.
    // Phase 2: replace with real MPP multi-fragment streaming execution.
    val hasRealFragments = fragments.nonEmpty && fragments.head.rootOperator != null
    if (!hasRealFragments) {
      // Phase 2: Extract fragments from child BSP plan and generate Substrait plans.
      logWarning(s"MppNativeQueryExec: child plan tree:\n${child.treeString.take(2000)}")

      val (extractedFragments, extractedExchanges, fusedBroadcastsByConsumer) =
        extractFragmentsFromChildPlan()
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

        val fragmentPlans = fragmentSubstraitPlans.toArray
        val numDriversPerFragment = extractedFragments.map(_.parallelism).toArray
        val exchangeSpecsJson = serializeExchangeSpecs(extractedExchanges, extractedFragments)

        // Extract scan split infos for each fragment.
        // Scan-containing fragments (leaf fragments) have LeafTransformSupport
        // nodes with file paths. Non-scan fragments get empty arrays.
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
          fragmentSplitInfos,
          fusedBroadcastsByConsumer,
          longMetric("totalQueryTimeMs"),
          longMetric("outputRows"),
          longMetric("outputBatches")
        )
      } else {
        // Single fragment or no exchanges -- BSP is fine
        logWarning(
          s"MppNativeQueryExec: single fragment (${extractedFragments.size} fragments, " +
            s"${extractedExchanges.size} exchanges), delegating to BSP")
        if (child.supportsColumnar) {
          return child.executeColumnar()
        } else {
          return child.execute().mapPartitions(rows => Iterator.empty)
        }
      }
    }

    // Update fragment/exchange count metrics.
    metrics("numFragments") += fragments.size
    metrics("numExchanges") += exchanges.size

    // Generate Substrait plans on the DRIVER side where sparkContext is available.
    val fragmentPlans: Array[Array[Byte]] = fragments.map {
      frag => generateSubstraitPlan(frag)
    }.toArray

    val numDriversPerFragment = fragments.map(_.parallelism).toArray
    val exchangeSpecsJson = serializeExchangeSpecs(exchanges, fragments)

    logInfo(s"MppNativeQueryExec: generated ${fragmentPlans.length} Substrait plans on driver")

    // Extract split infos for pre-built fragments
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
      fragmentSplitInfos,
      Map.empty[Int, Seq[FusedBroadcast]],
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
    copy(child = newChild, originalLogicalPlan = originalLogicalPlan)
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
  private def extractFragmentsFromChildPlan()
      : (Seq[NativeFragment], Seq[ExchangeSpec], Map[Int, Seq[FusedBroadcast]]) = {
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
          val parallelism = inferParallelism(wst)
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
          val exchangeNodes = exchangeChildren.zipWithIndex.map {
            case (child, slotIdx) =>
              unwrapToExchange(child).getOrElse {
                throw new IllegalStateException(
                  s"MppNativeQueryExec: WST fragment $fragId input slot $slotIdx " +
                    s"(${child.getClass.getSimpleName}) did not unwrap to an Exchange. " +
                    s"Refusing to silently drop an ExchangeSpec; this would desynchronize " +
                    s"iterator slots and native ValueStream schemas.")
              }
          }
          childExchangeFragIds.zipWithIndex.zip(exchangeNodes).foreach {
            case ((producerFragId, slotIdx), bex: BroadcastExchangeLike) if producerFragId < 0 =>
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
            case ((producerFragId, _), _) if producerFragId < 0 =>
              // Non-broadcast collapse (e.g. collapseSingleGather sentinel).
              ()
            case ((producerFragId, _), exchangeNode) =>
              val (exchangeType, partitionKeys) =
                classifyPartitioning(exchangeNode.outputPartitioning)
              extractedExchanges += ExchangeSpec(
                id = exchangeCounter.getAndIncrement(),
                producerFragmentId = producerFragId,
                consumerFragmentId = fragId,
                exchangeType = exchangeType,
                numPartitions = exchangeNode.outputPartitioning.numPartitions,
                partitionKeys = partitionKeys
              )
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
          val sortPlan = SortExecTransformer(topk.sortOrder, global = false, inputIter)
          val limitPlan = LimitExecTransformer(sortPlan, topk.offset.toLong, topk.limit)
          val consumerRoot: SparkPlan =
            if (topk.projectList != topk.child.output) {
              ProjectExecTransformer(topk.projectList, limitPlan)
            } else {
              limitPlan
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
    val rewrittenChild = applyCrossCutRules(child)

    walk(rewrittenChild)

    // Sort fragments by ID (ensures topological order: producers before consumers)
    val sortedFragments = extractedFragments.sortBy(_.id).toSeq
    val cappedExchanges = capLocalHashExchangeTasks(extractedExchanges.toSeq)
    val sortedExchanges = cappedExchanges.sortBy(_.id).toSeq
    val frozenBroadcasts = broadcastsByConsumer.iterator.map {
      case (consumerId, buf) => consumerId -> buf.toSeq
    }.toMap
    (sortedFragments, sortedExchanges, frozenBroadcasts)
  }

  /**
   * Re-run plan-shape parity rules on the wrapped child plan. MppStrategy intercepts before the
   * columnar post-rule pass, so rules registered via VeloxRuleApi.injectPost otherwise miss this
   * subtree. Each rule is conf-gated and a no-op by default.
   */
  private def applyCrossCutRules(plan: SparkPlan): SparkPlan = {
    val sortRule = MppSinglePartitionSortRule()
    val skipShuffleRule = MppRemoveRedundantShuffleRule()
    val parallelSortSplitRule = MppParallelSortSplitRule()
    val afterSort = sortRule(plan)
    val afterSkipShuffle = skipShuffleRule(afterSort)
    // parallelSortSplit must run AFTER MppSinglePartitionSortRule so the
    // RangePartitioning -> SinglePartition rewrite has already happened.
    val afterParallelSortSplit = parallelSortSplitRule(afterSkipShuffle)
    normalizeMppJoinBuildSide(afterParallelSortSplit)
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

  private case class BuildSideChoice(side: BuildSide, reason: String)

  private case class JoinSideStats(sizeInBytes: BigInt, rowCount: Option[BigInt], source: String)

  private object MppJoinSelectionHelper extends JoinSelectionHelper

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

  private def localHashExchangeTasks: Option[Int] = {
    positiveIntConf("spark.gluten.mpp.localHashExchangeTasks")
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
   * Walk the plan tree and force every [[ScalarSubquery]] (and any other
   * [[ExecSubqueryExpression]]) to materialize. Spark normally drives this through
   * SparkPlan.prepareSubqueries during the executeQuery() prepare phase; we call it explicitly
   * because [[MppNativeQueryExec]] bypasses child.executeColumnar() and generates Substrait on the
   * driver instead. After this returns, Gluten's ScalarSubqueryTransformer can safely call
   * query.eval(InternalRow.empty) and emit a Substrait Literal. Spark's ReusedSubqueryExec
   * mechanism ensures identical subqueries share a single execution.
   */
  private def materializeScalarSubqueries(plan: SparkPlan): Unit = {
    // Do not call plan.prepare() on the whole child tree here. SparkPlan.prepare()
    // eagerly starts regular BroadcastExchange jobs, but MPP consumes those joins
    // through native BROADCAST exchanges. Preparing the whole plan can therefore
    // driver-collect a large build side before MPP starts, tripping
    // spark.driver.maxResultSize on Q16/Q18. Only materialize real subquery
    // expressions below; regular broadcast builds stay behind native BROADCAST
    // exchange fragments.

    plan.foreach {
      node =>
        // Drive every declared subquery to completion. SparkPlan.subqueries
        // returns the Seq[BaseSubqueryExec] reachable from this node, which
        // covers BOTH classic ExecSubqueryExpression (Q11/Q22 scalar) AND the
        // SubqueryBroadcastExec wrapper that Spark's InjectRuntimeFilter rule
        // emits for VeloxBloomFilterAggregate build sides (Q16/Q20).
        node.subqueries.foreach {
          subPlan =>
            try {
              subPlan.prepare()
              // Force the underlying future to finish. For ScalarSubqueryExec
              // this caches the row; for SubqueryBroadcastExec this awaits
              // the broadcast and caches the relation. Either way, by return
              // any PlanExpression that references subPlan can resolve to a
              // concrete value during Substrait conversion.
              try {
                val m = subPlan.getClass.getMethod("relationFuture")
                val fut = m.invoke(subPlan)
                fut.getClass
                  .getMethod("get")
                  .invoke(fut)
              } catch {
                case _: NoSuchMethodException => // not a broadcast subquery
                case _: Throwable => // best-effort
              }
            } catch {
              case t: Throwable =>
                logWarning(
                  s"materializeScalarSubqueries: subPlan.prepare() failed for " +
                    s"${subPlan.getClass.getSimpleName}: ${t.getMessage}",
                  t)
            }
        }
        node.expressions.foreach {
          expr =>
            expr.foreach {
              case sub: ExecSubqueryExpression =>
                sub.plan.prepare()
                // Idempotent: updateResult blocks on the underlying future and
                // stores the row.
                sub.updateResult()
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
                  try {
                    val planField = cls.getMethod("plan")
                    val embedded = planField.invoke(other).asInstanceOf[SparkPlan]
                    if (embedded != null) embedded.prepare()
                  } catch { case _: Throwable => () }
                  try {
                    cls.getMethod("updateResult").invoke(other)
                  } catch { case _: Throwable => () }
                }
            }
        }
    }
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
      fragments)
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
          t)
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
      child.treeString
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
