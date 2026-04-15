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
import org.apache.gluten.iterator.Iterators
import org.apache.gluten.substrait.SubstraitContext
import org.apache.gluten.substrait.plan.PlanBuilder
import org.apache.gluten.utils.SubstraitPlanPrinterUtil
import org.apache.gluten.runtime.Runtimes
import org.apache.gluten.vectorized.MppQueryJniWrapper

import org.apache.spark.{Partition, SparkContext, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.vectorized.ColumnarBatch

import com.google.common.collect.Lists

import scala.collection.JavaConverters._

/**
 * RDD that executes an MPP query plan via JNI.
 *
 * This is a single-partition RDD: the native MPP coordinator handles all parallelism internally.
 * Each fragment's Substrait plan is generated at compute time and submitted to the native runtime
 * as a batch of plans connected by exchange specifications.
 *
 * The native side creates an MppQueryCoordinator that:
 *   - Launches all fragment pipelines concurrently
 *   - Connects them via streaming GPU exchanges (OutputBufferManager / GpuExchange)
 *   - Returns the final fragment's output as an iterator of ColumnarBatch
 *
 * @param sc
 *   The SparkContext.
 * @param fragments
 *   The ordered list of native execution fragments.
 * @param exchanges
 *   The exchange specifications connecting fragments.
 * @param pipelineTime
 *   Metric for tracking total pipeline execution time.
 * @param outputRows
 *   Metric for tracking number of output rows.
 * @param outputBatches
 *   Metric for tracking number of output batches.
 */
class MppNativeQueryRDD(
    @transient sc: SparkContext,
    fragments: Seq[NativeFragment],
    exchanges: Seq[ExchangeSpec],
    pipelineTime: SQLMetric,
    outputRows: SQLMetric,
    outputBatches: SQLMetric
) extends RDD[ColumnarBatch](sc, Nil)
  with Logging {

  override protected def getPartitions: Array[Partition] = {
    // Single partition -- the MPP coordinator handles parallelism internally.
    Array(MppPartition(0))
  }

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] = {
    val planBuildStart = System.nanoTime()

    // 1. Generate Substrait plans for each fragment.
    //    Each fragment's rootOperator is a TransformSupport -- call doTransform()
    //    to get the Substrait plan, then serialize to bytes.
    val fragmentPlans: Seq[Array[Byte]] = fragments.map { fragment =>
      generateSubstraitPlan(fragment)
    }

    logInfo(
      s"MppNativeQueryRDD: generated ${fragmentPlans.size} Substrait plans " +
        s"for ${fragments.size} fragments")

    // 2. Serialize exchange specs to JSON for the native side.
    val exchangeSpecsJson = serializeExchangeSpecs(exchanges)
    logDebug(s"MppNativeQueryRDD: exchange specs JSON: $exchangeSpecsJson")

    // 3. Submit all fragment plans to MppQueryCoordinator via JNI.
    //    This launches ALL fragments concurrently (MPP all-stages-up)
    //    and wires them together via OutputBufferManager streaming exchange.
    logInfo(
      s"MppNativeQueryRDD: *** LAUNCHING MPP EXECUTION *** " +
        s"with ${fragmentPlans.size} fragments, ${exchanges.size} exchanges")

    val numDriversPerFragment = fragments.map(_.parallelism).toArray
    val runtime = Runtimes.contextInstance(BackendsApiManager.getBackendName, "MppQuery")
    val jniWrapper = MppQueryJniWrapper.create(runtime)
    val mppHandle = jniWrapper.nativeCreateMppQuery(
      fragmentPlans.toArray,
      numDriversPerFragment,
      exchangeSpecsJson.getBytes("UTF-8"))
    jniWrapper.nativeStartMppQuery(mppHandle)

    logInfo("MppNativeQueryRDD: all MPP fragments started, streaming exchange active")

    val tracker = org.apache.gluten.metrics.TaskWallTimeTracker.get()
    tracker.planBuildNanos += (System.nanoTime() - planBuildStart)

    // 4. Create iterator that pulls batches from MppQueryCoordinator via JNI.
    val mppIter = new Iterator[ColumnarBatch] {
      private var nextHandle: Long = -1L
      private var finished = false

      override def hasNext: Boolean = {
        if (finished) return false
        if (nextHandle != -1L) return true
        nextHandle = jniWrapper.nativeGetMppOutput(mppHandle)
        if (nextHandle == 0L) {
          finished = true
          jniWrapper.nativeCloseMppQuery(mppHandle)
          logInfo("MppNativeQueryRDD: MPP execution complete, all fragments finished")
          false
        } else {
          true
        }
      }

      override def next(): ColumnarBatch = {
        if (!hasNext) throw new NoSuchElementException("MPP iterator exhausted")
        val handle = nextHandle
        nextHandle = -1L
        // Convert native handle to ColumnarBatch using Gluten's standard mechanism
        val batch = org.apache.gluten.columnarbatch.ColumnarBatches.create(handle)
        outputRows += batch.numRows()
        outputBatches += 1
        batch
      }
    }

    // Wrap with lifecycle management
    Iterators
      .wrap(mppIter)
      .protectInvocationFlow()
      .recyclePayload(batch => batch.close())
      .collectLifeMillis(millis => pipelineTime += millis)
      .asInterruptible(context)
      .create()
  }

  /**
   * Generate a Substrait plan for a single fragment.
   *
   * Uses the same pattern as [[WholeStageTransformer.doWholeStageTransform()]]:
   *   1. Create a SubstraitContext
   *   2. Call doTransform() on the fragment's root operator (TransformSupport)
   *   3. Build a PlanNode and serialize to bytes
   */
  private def generateSubstraitPlan(fragment: NativeFragment): Array[Byte] = {
    val rootOp = fragment.rootOperator
    rootOp match {
      case ts: TransformSupport =>
        val substraitContext = new SubstraitContext
        val childCtx = ts.transform(substraitContext)
        if (childCtx == null) {
          throw new IllegalStateException(
            s"MppNativeQueryRDD: fragment ${fragment.id} root operator " +
              s"${rootOp.getClass.getSimpleName} returned null from doTransform()")
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
          s"MppNativeQueryRDD: fragment ${fragment.id} Substrait plan: " +
            SubstraitPlanPrinterUtil.substraitPlanToJson(planNode.toProtobuf))

        planNode.toProtobuf.toByteArray

      case other =>
        throw new IllegalStateException(
          s"MppNativeQueryRDD: fragment ${fragment.id} root operator " +
            s"${other.getClass.getSimpleName} is not a TransformSupport")
    }
  }

  /**
   * Serialize exchange specifications to JSON for the native side.
   *
   * Format:
   * {{{
   * [
   *   {
   *     "id": 0,
   *     "producerFragmentId": 0,
   *     "consumerFragmentId": 1,
   *     "exchangeType": "HASH",
   *     "numPartitions": 200,
   *     "partitionKeys": ["col1#10", "col2#11"]
   *   },
   *   ...
   * ]
   * }}}
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
         |  "partitionKeys": $keys
         |}""".stripMargin
    }
    entries.mkString("[", ", ", "]")
  }
}

/** Simple partition for the single-partition MppNativeQueryRDD. */
private[execution] case class MppPartition(override val index: Int) extends Partition
