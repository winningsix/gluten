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
import org.apache.gluten.expression.ConverterUtils
import org.apache.gluten.extension.{ExchangeSpec, NativeFragment}
import org.apache.gluten.extension.columnar.transition.{Convention, ConventionReq}
import org.apache.gluten.substrait.SubstraitContext
import org.apache.gluten.substrait.plan.PlanBuilder
import org.apache.gluten.substrait.rel.SplitInfo
import org.apache.gluten.utils.SubstraitPlanPrinterUtil

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.{BroadcastPartitioning, HashPartitioning, Partitioning, RangePartitioning, RoundRobinPartitioning, SinglePartition}
import org.apache.spark.sql.execution.{ColumnarCollapseTransformStages, ColumnarInputAdapter, ExecSubqueryExpression, InputIteratorTransformer, SparkPlan, SQLExecution, UnaryExecNode}
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeLike, Exchange, ReusedExchangeExec, ShuffleExchangeLike}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.ColumnarBatch

import com.google.common.collect.Lists
import io.substrait.proto.ReadRel

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
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

      val (extractedFragments, extractedExchanges) = extractFragmentsFromChildPlan()
      logWarning(
        s"MppNativeQueryExec: Phase 2 extracted ${extractedFragments.size} fragments " +
          s"and ${extractedExchanges.size} exchanges from child plan")

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

        return new MppNativeQueryRDD(
          sparkContext,
          fragmentPlans,
          numDriversPerFragment,
          exchangeSpecsJson,
          fragmentSplitInfos,
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

    // RDD only receives serialized bytes - no SparkPlan references.
    new MppNativeQueryRDD(
      sparkContext,
      fragmentPlans,
      numDriversPerFragment,
      exchangeSpecsJson,
      fragmentSplitInfos,
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
  private def extractFragmentsFromChildPlan(): (Seq[NativeFragment], Seq[ExchangeSpec]) = {
    val extractedFragments = mutable.ArrayBuffer[NativeFragment]()
    val extractedExchanges = mutable.ArrayBuffer[ExchangeSpec]()
    // Memoize producer fragment ids by Exchange so ReusedExchangeExec (which shares the
    // same Exchange reference as the original) reuses rather than creates a duplicate.
    val exchangeToProducerFragId = mutable.HashMap[Exchange, Int]()
    val fragmentCounter = new AtomicInteger(0)
    val exchangeCounter = new AtomicInteger(0)

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
          val childExchangeFragIds = findExchangeChildren(wst).map(walk)

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
          // ExchangeSpec entirely -- the build operators were walked into this
          // consumer fragment via walkInFragment-equivalent recursion, so there is
          // no remote producer to wire.
          childExchangeFragIds.zip(findExchangeNodes(wst)).foreach {
            case (producerFragId, _) if producerFragId < 0 =>
              // Fused broadcast: no separate producer fragment, no exchange spec.
              ()
            case (producerFragId, exchangeNode) =>
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
          // Exchange boundary: walk the producer side (exchange.child). Memoize so a
          // subsequent ReusedExchangeExec pointing at the same Exchange reuses it.
          exchangeToProducerFragId.getOrElseUpdate(
            exchange.asInstanceOf[Exchange],
            walk(unwrapTransparent(exchange.child)))

        case bex: BroadcastExchangeLike =>
          // Q21-style fusion: when spark.gluten.mpp.fuseBroadcastBuilds=true and
          // the broadcast's runtime size <= spark.gluten.mpp.broadcastFuseThreshold,
          // do NOT allocate a producer fragment. Return -1 as a sentinel; the WST
          // consumer suppresses the corresponding ExchangeSpec entry. The build
          // operators are still serialized: the consumer WST's doTransform walks
          // through ColumnarInputAdapter -> BroadcastQueryStage normally, so the
          // local-build path inside Velox handles the data side. This saves one
          // fragment per fused broadcast (e.g. 12 -> 9 on TPC-H Q21).
          if (canFuseBroadcastLive(bex)) {
            logWarning(
              s"MppNativeQueryExec: fusing broadcast build (size <= " +
                s"$broadcastFuseThresholdBytes bytes) into consumer fragment")
            -1
          } else {
            exchangeToProducerFragId.getOrElseUpdate(
              bex.asInstanceOf[Exchange],
              walk(unwrapTransparent(bex.child)))
          }

        case reused: ReusedExchangeExec =>
          // The ReusedExchangeExec.child is the same JVM object as the original
          // Exchange. Walk it -- memoization guarantees we return the already-
          // assigned producer fragment id.
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

    walk(child)

    // Sort fragments by ID (ensures topological order: producers before consumers)
    val sortedFragments = extractedFragments.sortBy(_.id).toSeq
    val sortedExchanges = extractedExchanges.sortBy(_.id).toSeq
    (sortedFragments, sortedExchanges)
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

  /**
   * Find the actual ShuffleExchangeLike nodes corresponding to the exchange children found by
   * [[findExchangeChildren]]. Unwraps ColumnarInputAdapter and other wrappers.
   */
  private def findExchangeNodes(wst: WholeStageTransformer): Seq[Exchange] = {
    findExchangeChildren(wst).flatMap(child => unwrapToExchange(child))
  }

  /** Unwrap wrapper nodes to find the Exchange (shuffle or broadcast) underneath. */
  private def unwrapToExchange(plan: SparkPlan): Option[Exchange] = {
    plan match {
      case ex: ShuffleExchangeLike => Some(ex.asInstanceOf[Exchange])
      case ex: BroadcastExchangeLike => Some(ex.asInstanceOf[Exchange])
      case reused: ReusedExchangeExec => unwrapToExchange(reused.child)
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
    plan.foreach {
      node =>
        node.expressions.foreach {
          expr =>
            expr.foreach {
              case sub: ExecSubqueryExpression =>
                sub.plan.prepare()
                // Idempotent: updateResult blocks on the underlying future and stores the row.
                sub.updateResult()
              case _ =>
            }
        }
    }
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

  /** Threshold in bytes below which a broadcast build may be fused. Default 8 GB. */
  private def broadcastFuseThresholdBytes: Long = {
    SQLConf.get
      .getConfString(
        "spark.gluten.mpp.broadcastFuseThreshold",
        (8L * 1024L * 1024L * 1024L).toString)
      .toLong
  }

  /**
   * Decide whether a [[BroadcastExchangeLike]] is small enough to fuse into the consumer fragment.
   * We use `runtimeStatistics.sizeInBytes` when the runtime exposes it; failing that we
   * conservatively refuse to fuse so we never silently turn a too-large broadcast into a fused
   * replicated build (would OOM the consumer).
   */
  private def canFuseBroadcastLive(bc: SparkPlan): Boolean = {
    if (!fuseBroadcastBuildsEnabled) return false
    val sizeBytes: Option[Long] = bc match {
      case b: BroadcastExchangeLike =>
        try { Some(b.runtimeStatistics.sizeInBytes.toLong) }
        catch {
          case _: Throwable => None
        }
      case _ => None
    }
    sizeBytes.exists(_ <= broadcastFuseThresholdBytes)
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
    math.min(raw, math.max(cores, 1))
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
