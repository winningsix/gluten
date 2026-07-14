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

import org.apache.spark.Partition
import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.catalyst.expressions.{Ascending, AttributeReference, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.{Partitioning, UnknownPartitioning}
import org.apache.spark.sql.connector.read.Scan
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}

import org.apache.iceberg._
import org.apache.iceberg.expressions.{Expressions, ResidualEvaluator}
import org.apache.iceberg.spark.source.SparkInputPartition
import org.apache.iceberg.types.Types

import scala.collection.JavaConverters._

class IcebergScanTransformerPlanningSuite extends QueryTest with SharedSparkSession {

  private val key = AttributeReference("id", IntegerType, nullable = false)()
  private val schema = new Schema(Types.NestedField.required(1, "id", Types.IntegerType.get()))
  private val spec = PartitionSpec.unpartitioned()

  test("MPP unordered Iceberg transformer coalesces planned partitions") {
    val planned = plannedPartitions(4, bytesPerPartition = 10L)

    withPlanningConf(singleTaskMode = false) {
      val transformer = new PlanningIcebergScanTransformer(planned, Seq.empty)
      val result = transformer.getPartitions

      assert(result.size === 1)
      val combined = result.head.asInstanceOf[SparkDataSourceRDDPartition]
      assert(combined.index === 0)
      assert(combined.inputPartitions === planned.flatMap(_.inputPartitions))
    }
  }

  test("Iceberg transformer preserves ordered planned partitions") {
    val planned = plannedPartitions(4, bytesPerPartition = 10L)

    withPlanningConf(singleTaskMode = false) {
      val transformer = new PlanningIcebergScanTransformer(
        planned,
        Seq(SortOrder(key, Ascending)))
      assert(transformer.getPartitions eq planned)
    }
  }

  test("Iceberg transformer preserves planned partitions in single-task mode") {
    val planned = plannedPartitions(4, bytesPerPartition = 10L)

    withPlanningConf(singleTaskMode = true) {
      val transformer = new PlanningIcebergScanTransformer(planned, Seq.empty)
      assert(transformer.getPartitions eq planned)
    }
  }

  private def withPlanningConf(singleTaskMode: Boolean)(body: => Unit): Unit = {
    withSQLConf(
      "spark.gluten.mpp.enabled" -> "true",
      "spark.gluten.sql.columnar.backend.velox.mpp.singleTaskMode" -> singleTaskMode.toString,
      "spark.sql.files.maxPartitionBytes" -> "64",
      "spark.sql.files.openCostInBytes" -> "1")(body)
  }

  private def plannedPartitions(count: Int, bytesPerPartition: Long): Seq[PlannedPartition] = {
    (0 until count).map {
      index =>
        val task = fileScanTask(index, bytesPerPartition)
        val input = sparkInputPartition(task, Array(s"host-${index % 2}"))
        new PlannedPartition(index, Seq(input))
    }
  }

  private def fileScanTask(index: Int, length: Long): FileScanTask = {
    val dataFile = DataFiles
      .builder(spec)
      .withPath(s"file:/tmp/iceberg-planning-$index.parquet")
      .withFormat(FileFormat.PARQUET)
      .withRecordCount(1L)
      .withFileSizeInBytes(length)
      .build()
    new BaseFileScanTask(
      dataFile,
      Array.empty[DeleteFile],
      SchemaParser.toJson(schema),
      PartitionSpecParser.toJson(spec),
      ResidualEvaluator.of(spec, Expressions.alwaysTrue(), true))
  }

  /** Uses reflection so the test remains source-compatible across Iceberg constructor additions. */
  private def sparkInputPartition(
      task: FileScanTask,
      preferredLocations: Array[String]): SparkInputPartition = {
    val taskGroup: ScanTaskGroup[ScanTask] =
      new BaseScanTaskGroup[ScanTask](Seq(task: ScanTask).asJava)
    val constructor = classOf[SparkInputPartition].getDeclaredConstructors
      .find(_.getParameterTypes.exists(classOf[ScanTaskGroup[_]].isAssignableFrom))
      .getOrElse(throw new IllegalStateException("SparkInputPartition constructor not found"))
    constructor.setAccessible(true)
    val arguments: Array[AnyRef] = constructor.getParameterTypes.map {
      parameterType =>
        if (classOf[ScanTaskGroup[_]].isAssignableFrom(parameterType)) {
          taskGroup.asInstanceOf[AnyRef]
        } else if (parameterType == classOf[Array[String]]) {
          preferredLocations.asInstanceOf[AnyRef]
        } else if (parameterType == java.lang.Boolean.TYPE) {
          Boolean.box(false)
        } else if (parameterType == java.lang.Integer.TYPE) {
          Int.box(0)
        } else if (parameterType == java.lang.Long.TYPE) {
          Long.box(0L)
        } else {
          null
        }
    }
    constructor.newInstance(arguments: _*).asInstanceOf[SparkInputPartition]
  }

  private object EmptyScan extends Scan {
    override def readSchema(): StructType = StructType(Seq(StructField("id", IntegerType)))
  }

  private class PlannedPartition(index: Int, inputPartitions: Seq[SparkInputPartition])
    extends SparkDataSourceRDDPartition(index, inputPartitions)

  private class PlanningIcebergScanTransformer(
      planned: Seq[PlannedPartition],
      exposedOrdering: Seq[SortOrder])
    extends IcebergScanTransformer(
      output = Seq(key),
      scan = EmptyScan,
      runtimeFilters = Seq.empty,
      table = null,
      ordering = Option(exposedOrdering).filter(_.nonEmpty)) {

    override protected def planFinalPartitions(): Seq[Partition] = planned

    override def outputPartitioning: Partitioning = UnknownPartitioning(planned.size)

    override def outputOrdering: Seq[SortOrder] = exposedOrdering
  }
}
