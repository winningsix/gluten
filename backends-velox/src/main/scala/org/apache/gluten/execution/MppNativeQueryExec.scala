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

import org.apache.gluten.extension.{ExchangeSpec, NativeFragment}
import org.apache.gluten.extension.columnar.transition.{Convention, ConventionReq}

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.{Partitioning, UnknownPartitioning}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.vectorized.ColumnarBatch

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
case class MppNativeQueryExec(
    fragments: Seq[NativeFragment],
    exchanges: Seq[ExchangeSpec],
    originalPlan: SparkPlan
) extends SparkPlan
  with GlutenPlan
  with Logging {

  // --- Output schema and partitioning ---

  override def output: Seq[Attribute] = originalPlan.output

  override def outputPartitioning: Partitioning = {
    // The final fragment determines the output partitioning.
    // In most cases this is the root fragment's partitioning.
    if (fragments.nonEmpty) {
      fragments.last.rootOperator.outputPartitioning
    } else {
      UnknownPartitioning(0)
    }
  }

  override def outputOrdering: Seq[SortOrder] = {
    if (fragments.nonEmpty) {
      fragments.last.rootOperator.outputOrdering
    } else {
      Nil
    }
  }

  // --- Leaf node from Spark's perspective ---

  override def children: Seq[SparkPlan] = Nil

  // --- Convention support for GlutenPlan ---

  override def batchType(): Convention.BatchType = {
    // MPP execution produces native columnar batches
    org.apache.gluten.backendsapi.BackendsApiManager.getSettings.primaryBatchType
  }

  override def rowType0(): Convention.RowType = Convention.RowType.None

  override def requiredChildConvention(): Seq[ConventionReq] = Nil

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
    throw new UnsupportedOperationException(
      s"MppNativeQueryExec does not support row-based execution. " +
        s"Use executeColumnar() instead.")
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    // TODO (M4): Wire to JNI for native MPP execution.
    // The implementation will:
    // 1. Generate Substrait plans for each fragment via WholeStageTransformContext
    // 2. Serialize fragment plans + exchange specs into MppExecutionDescriptor protobuf
    // 3. Call JNI: NativePlanExecutor.executeMppPlan(descriptor)
    // 4. Return the result RDD from the native runtime
    //
    // For now, this is a stub that fails at runtime if someone tries to execute
    // an MPP plan before JNI integration is complete.
    logInfo(
      s"MppNativeQueryExec: plan has ${fragments.size} fragments " +
        s"and ${exchanges.size} exchanges")
    logInfo(s"MppNativeQueryExec: fragment details:\n$fragmentSummary")
    logInfo(s"MppNativeQueryExec: exchange details:\n$exchangeSummary")

    throw new UnsupportedOperationException(
      "MPP native execution is not yet implemented. " +
        "JNI integration is planned for M4. " +
        s"Plan: ${fragments.size} fragments, ${exchanges.size} exchanges.")
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

  private def fragmentSummary: String = {
    fragments
      .map {
        f =>
          s"  Fragment ${f.id}: parallelism=${f.parallelism}, " +
            s"output=${f.outputAttributes.map(_.name).mkString("[", ", ", "]")}, " +
            s"root=${f.rootOperator.simpleString(10)}"
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

  // --- TreeNode support ---

  // MppNativeQueryExec is a leaf from Spark's perspective, so no withNewChildInternal needed.
  // But SparkPlan requires it for completeness.
  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan = {
    assert(newChildren.isEmpty, "MppNativeQueryExec is a leaf node and has no children")
    this
  }
}
