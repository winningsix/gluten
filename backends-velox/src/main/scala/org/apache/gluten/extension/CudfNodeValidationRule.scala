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

import org.apache.gluten.config.{GlutenConfig, VeloxConfig}
import org.apache.gluten.cudf.VeloxCudfPlanValidatorJniWrapper
import org.apache.gluten.execution._
import org.apache.gluten.extension.CudfNodeValidationRule.{createGPUColumnarExchange, setTagForWholeStageTransformer}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.plans.physical.HashPartitioning
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarShuffleExchangeExec, GPUColumnarShuffleExchangeExec, SparkPlan}
import org.apache.spark.sql.execution.exchange.ShuffleExchangeLike
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{ArrayType, DataType, MapType, StructType}

// Add the node name prefix 'Cudf' to GlutenPlan when can offload to cudf
case class CudfNodeValidationRule(glutenConf: GlutenConfig) extends Rule[SparkPlan] {

  override def apply(plan: SparkPlan): SparkPlan = {
    if (!glutenConf.enableColumnarCudf) {
      return plan
    }
    val gpuPartition = glutenConf.enableCudfGpuPartition
    val preserveExchangeRoot = plan.isInstanceOf[ShuffleExchangeLike]
    val transformedPlan = plan.transformUp {
      case shuffle @ ColumnarShuffleExchangeExec(
            _,
            VeloxResizeBatchesExec(w: WholeStageTransformer, _, _, _),
            _,
            _,
            _) =>
        setTagForWholeStageTransformer(w)
        val isHash = shuffle.outputPartitioning.isInstanceOf[HashPartitioning]
        if (gpuPartition && isHash) {
          val rewritten = createGPUColumnarExchange(shuffle, Some(w), preserveExchangeRoot)
          if (!(rewritten eq shuffle)) {
            w.setTagValue(CudfTag.GpuShuffleStageTag, true)
          }
          rewritten
        } else if (isHash) {
          createGPUColumnarExchange(shuffle, preserveExchange = preserveExchangeRoot)
        } else {
          shuffle
        }
      case shuffle @ ColumnarShuffleExchangeExec(_, w: WholeStageTransformer, _, _, _) =>
        setTagForWholeStageTransformer(w)
        val isHash = shuffle.outputPartitioning.isInstanceOf[HashPartitioning]
        if (gpuPartition && isHash) {
          val rewritten =
            createGPUColumnarExchange(shuffle, preserveExchange = preserveExchangeRoot)
          if (!(rewritten eq shuffle)) {
            w.setTagValue(CudfTag.GpuShuffleStageTag, true)
          }
          rewritten
        } else if (isHash) {
          createGPUColumnarExchange(shuffle, preserveExchange = preserveExchangeRoot)
        } else {
          shuffle
        }
      case transformer: WholeStageTransformer =>
        setTagForWholeStageTransformer(transformer)
        transformer
    }
    transformedPlan
  }
}

object CudfNodeValidationRule extends Logging {
  private def setTagForStage(transformer: WholeStageTransformer, isCudf: Boolean): Unit = {
    transformer.foreach {
      case t: TransformSupport =>
        t.setTagValue(CudfTag.CudfTag, isCudf)
      case _ =>
    }
    transformer.setTagValue(CudfTag.CudfTag, isCudf)
  }

  def setTagForWholeStageTransformer(transformer: WholeStageTransformer): Unit = {
    if (VeloxConfig.get.cudfEnableValidation && !VeloxConfig.get.cudfEnableTableScan) {
      // Validate whether the non-scan operators in this stage can run on cudf.
      // When cudfEnableTableScan is false the scan stays on CPU (Velox) and the
      // ToCudf plan compiler inserts a CudfFromVelox boundary after the
      // TableScan, so subsequent operators still get cudf overrides.
      if (
        VeloxCudfPlanValidatorJniWrapper.validate(transformer.substraitPlan.toProtobuf.toByteArray)
      ) {
        setTagForStage(transformer, isCudf = true)
      }
    } else {
      setTagForStage(transformer, isCudf = true)
    }
  }

  def createGPUColumnarExchange(
      shuffle: ColumnarShuffleExchangeExec,
      childOverride: Option[SparkPlan] = None,
      preserveExchange: Boolean = false
  ): SparkPlan = {
    if (containsComplexType(shuffle.output.map(_.dataType))) {
      logWarning(
        s"CudfNodeValidationRule: keeping ${shuffle.getClass.getSimpleName} because " +
          "GPU hash shuffle does not support complex output types yet")
      return shuffle
    }
    val child = childOverride.getOrElse(shuffle.child)
    val exec = GPUColumnarShuffleExchangeExec(
      shuffle.outputPartitioning,
      child,
      shuffle.shuffleOrigin,
      shuffle.projectOutputAttributes,
      shuffle.advisoryPartitionSize)
    val res = exec.doValidate()
    if (!res.ok()) {
      logWarning(
        s"CudfNodeValidationRule: keeping ${shuffle.getClass.getSimpleName} because " +
          s"GPUColumnarShuffleExchangeExec validation failed: ${res.reason()}")
      return shuffle
    }
    if (!preserveExchange && !SQLConf.get.adaptiveExecutionEnabled) {
      val batchSize = VeloxConfig.get.cudfBatchSize
      val batchSizeInBytes = VeloxConfig.get.cudfBatchSizeInBytes
      GpuResizeBufferColumnarBatchExec(exec, batchSize, batchSizeInBytes)
    } else {
      exec
    }
  }

  private def containsComplexType(types: Seq[DataType]): Boolean = types.exists(containsComplexType)

  private def containsComplexType(dataType: DataType): Boolean = dataType match {
    case _: StructType => true
    case _: ArrayType => true
    case _: MapType => true
    case _ => false
  }
}
