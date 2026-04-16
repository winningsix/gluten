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
import org.apache.gluten.utils.SubstraitPlanPrinterUtil

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.{HashPartitioning, Partitioning, RoundRobinPartitioning, SinglePartition}
import org.apache.spark.sql.execution.{ColumnarInputAdapter, InputIteratorTransformer, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.exchange.ShuffleExchangeLike
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.ColumnarBatch

import com.google.common.collect.Lists

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
 * MppNativeQueryExec wraps the original plan as its child (UnaryExecNode).
 * Spark sees the original plan (including ShuffleExchange nodes) still intact,
 * so the "cannot transform shuffle node" validation passes.
 *
 * At execution time, doExecuteColumnar() does NOT call child.executeColumnar().
 * Instead, it uses the child plan only to extract Substrait fragments, then
 * executes via JNI → MppQueryCoordinator (streaming exchange).
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
    // Always route through doExecuteColumnar() and convert to rows.
    // This ensures MPP execution is used for both row-based and columnar code paths.
    doExecuteColumnar().mapPartitions { batches =>
      batches.flatMap { batch =>
        val numRows = batch.numRows()
        val rows = new Array[InternalRow](numRows)
        for (i <- 0 until numRows) {
          rows(i) = batch.getRow(i).copy()
        }
        batch.close()
        rows.iterator
      }
    }
  }

  // supportsColumnar is already true via GlutenPlan

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    logWarning(
      s"MppNativeQueryExec: executing with ${fragments.size} fragments " +
        s"and ${exchanges.size} exchanges")

    // Plan D: child is the original BSP plan (with ShuffleExchange intact).
    // Phase 1: delegate to child BSP execution to prove the wrap chain works.
    // Phase 2: replace with real MPP multi-fragment streaming execution.
    val hasRealFragments = fragments.nonEmpty && fragments.head.rootOperator != null
    if (!hasRealFragments) {
      // Phase 2: Extract fragments from child BSP plan and generate Substrait plans.
      logWarning(
        s"MppNativeQueryExec: child plan tree:\n${child.treeString.take(2000)}")

      val (extractedFragments, extractedExchanges) = extractFragmentsFromChildPlan()
      logWarning(
        s"MppNativeQueryExec: Phase 2 extracted ${extractedFragments.size} fragments " +
          s"and ${extractedExchanges.size} exchanges from child plan")

      // Log each fragment for debugging
      extractedFragments.foreach { frag =>
        val rootName = if (frag.rootOperator != null) {
          frag.rootOperator.getClass.getSimpleName
        } else "<null>"
        logWarning(
          s"  Fragment ${frag.id}: root=$rootName, " +
            s"output=${frag.outputAttributes.map(_.name).mkString("[", ", ", "]")}, " +
            s"parallelism=${frag.parallelism}")
      }
      extractedExchanges.foreach { ex =>
        logWarning(
          s"  Exchange ${ex.id}: F${ex.producerFragmentId} -> F${ex.consumerFragmentId} " +
            s"(${ex.exchangeType}, ${ex.numPartitions} partitions)")
      }

      // Generate Substrait plan for each fragment
      val fragmentSubstraitPlans = extractedFragments.map { frag =>
        generateSubstraitForFragment(frag)
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
        val exchangeSpecsJson = serializeExchangeSpecs(extractedExchanges)

        return new MppNativeQueryRDD(
          sparkContext,
          fragmentPlans,
          numDriversPerFragment,
          exchangeSpecsJson,
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
          return child.execute().mapPartitions { rows =>
            Iterator.empty
          }
        }
      }
    }

    // Update fragment/exchange count metrics.
    metrics("numFragments") += fragments.size
    metrics("numExchanges") += exchanges.size

    // Generate Substrait plans on the DRIVER side where sparkContext is available.
    val fragmentPlans: Array[Array[Byte]] = fragments.map { frag =>
      generateSubstraitPlan(frag)
    }.toArray

    val numDriversPerFragment = fragments.map(_.parallelism).toArray
    val exchangeSpecsJson = serializeExchangeSpecs(exchanges)

    logInfo(
      s"MppNativeQueryExec: generated ${fragmentPlans.length} Substrait plans on driver")

    // RDD only receives serialized bytes — no SparkPlan references.
    new MppNativeQueryRDD(
      sparkContext,
      fragmentPlans,
      numDriversPerFragment,
      exchangeSpecsJson,
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
   * Each [[WholeStageTransformer]] between exchange boundaries is a "fragment".
   * Each [[ShuffleExchangeLike]] is an "exchange boundary" connecting two fragments.
   *
   * We do NOT modify the child plan tree -- only read it.
   */
  private def extractFragmentsFromChildPlan(): (Seq[NativeFragment], Seq[ExchangeSpec]) = {
    val extractedFragments = mutable.ArrayBuffer[NativeFragment]()
    val extractedExchanges = mutable.ArrayBuffer[ExchangeSpec]()
    val fragmentCounter = new AtomicInteger(0)
    val exchangeCounter = new AtomicInteger(0)

    /**
     * Walk the plan tree depth-first. Returns the fragment ID of the subtree rooted at `plan`.
     * At WholeStageTransformer: creates a new fragment.
     * At ShuffleExchangeLike: creates an exchange spec connecting producer to consumer.
     * At wrapper nodes (ColumnarToRow, ColumnarToColumnar, InputIterator, ColumnarInputAdapter):
     *   passes through to the child.
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

          // Link exchange specs: each exchange child's consumer is this fragment
          childExchangeFragIds.zip(findExchangeNodes(wst)).foreach {
            case (producerFragId, exchangeNode) =>
              val (exchangeType, partitionKeys) = classifyPartitioning(
                exchangeNode.outputPartitioning)
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
          // Exchange boundary: walk the producer side (exchange.child)
          walk(unwrapTransparent(exchange.child))

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
          other.children.foreach { c =>
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
   * Find ShuffleExchangeLike nodes that are direct exchange children of a
   * WholeStageTransformer (reachable through InputIteratorTransformer ->
   * ColumnarInputAdapter -> ... -> ShuffleExchangeLike chain).
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
   * Find the actual ShuffleExchangeLike nodes corresponding to the exchange children
   * found by [[findExchangeChildren]]. Unwraps ColumnarInputAdapter and other wrappers.
   */
  private def findExchangeNodes(wst: WholeStageTransformer): Seq[ShuffleExchangeLike] = {
    findExchangeChildren(wst).flatMap { child =>
      unwrapToExchange(child)
    }
  }

  /** Unwrap wrapper nodes to find the ShuffleExchangeLike underneath. */
  private def unwrapToExchange(plan: SparkPlan): Option[ShuffleExchangeLike] = {
    plan match {
      case ex: ShuffleExchangeLike => Some(ex)
      case cia: ColumnarInputAdapter => unwrapToExchange(cia.child)
      case c2c: ColumnarToColumnarExec => unwrapToExchange(c2c.child)
      case c2r: ColumnarToRowExecBase => unwrapToExchange(c2r.child)
      case _ => None
    }
  }

  /** Unwrap transparent wrapper nodes (ColumnarToColumnar, resize batches, etc.). */
  private def unwrapTransparent(plan: SparkPlan): SparkPlan = {
    plan match {
      case c2c: ColumnarToColumnarExec => unwrapTransparent(c2c.child)
      case other => other
    }
  }

  /** Classify the partitioning into an exchange type string and extract partition keys. */
  private def classifyPartitioning(partitioning: Partitioning): (String, Seq[Attribute]) = {
    partitioning match {
      case hash: HashPartitioning =>
        val keys = hash.expressions.collect { case attr: Attribute => attr }
        ("HASH", keys)
      case _: RoundRobinPartitioning =>
        ("ROUND_ROBIN", Seq.empty)
      case SinglePartition =>
        ("SINGLE", Seq.empty)
      case other =>
        logWarning(
          s"MppNativeQueryExec: unexpected partitioning type: " +
            s"${other.getClass.getSimpleName}")
        ("UNKNOWN", Seq.empty)
    }
  }

  /** Infer the parallelism for a fragment based on its output partitioning. */
  private def inferParallelism(plan: SparkPlan): Int = {
    plan.outputPartitioning match {
      case p if p.numPartitions > 0 => p.numPartitions
      case _ =>
        SQLConf.get.getConfString("spark.sql.shuffle.partitions", "200").toInt
    }
  }

  /**
   * Generate a Substrait plan for a fragment extracted from the child BSP plan.
   *
   * Uses [[WholeStageTransformer.doWholeStageTransform()]] which calls transform()
   * on the WST's internal operator tree. The transform stops at
   * [[InputIteratorTransformer]] boundaries, producing a Substrait plan that covers
   * exactly the operators between exchange boundaries -- which is what we want for
   * each MPP fragment.
   *
   * @return serialized Substrait plan bytes
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
   * Must be called on the driver side where SparkPlan.sparkContext is available.
   * Uses the same pattern as [[WholeStageTransformer.doWholeStageTransform()]]:
   *   1. Create a SubstraitContext
   *   2. Call transform() on the fragment's root operator (TransformSupport)
   *   3. Build a PlanNode and serialize to bytes
   */
  private def generateSubstraitPlan(fragment: NativeFragment): Array[Byte] = {
    // Unwrap non-TransformSupport wrappers to find the actual native operator
    val rootOp = unwrapToTransformSupport(fragment.rootOperator)
    logWarning(s"generateSubstraitPlan: fragment ${fragment.id} " +
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
          val outputTypeNodes = new java.util.ArrayList[
            org.apache.gluten.substrait.`type`.TypeNode]()
          for (attr <- childCtx.outputAttributes) {
            outputTypeNodes.add(
              ConverterUtils.getTypeNode(attr.dataType, attr.nullable))
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
          PlanBuilder.makePlan(
            substraitContext,
            Lists.newArrayList(childCtx.root),
            outNames)
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
   * Unwrap non-TransformSupport wrappers to find the actual native operator.
   * Fragments may have ShuffleExchangeLike, ColumnarToColumnarExec, or
   * ColumnarToRowExecBase as root — we need the TransformSupport child.
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
   */
  private def serializeExchangeSpecs(specs: Seq[ExchangeSpec]): String = {
    val entries = specs.map { spec =>
      val keys = spec.partitionKeys
        .map(attr => s""""${ConverterUtils.genColumnNameWithExprId(attr)}"""")
        .mkString("[", ", ", "]")
      s"""{
         |  "id": ${spec.id},
         |  "producerFragmentId": ${spec.producerFragmentId},
         |  "consumerFragmentId": ${spec.consumerFragmentId},
         |  "exchangeType": "${spec.exchangeType}",
         |  "numPartitions": ${spec.numPartitions},
         |  "exchangeNodeId": "mpp_exchange_source_${spec.id}",
         |  "partitionKeys": $keys
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

}
