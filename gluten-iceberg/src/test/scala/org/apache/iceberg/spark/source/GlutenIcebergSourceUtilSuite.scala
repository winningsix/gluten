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
package org.apache.iceberg.spark.source

import org.apache.gluten.execution.SparkDataSourceRDDPartition

import org.apache.spark.sql.types.{StringType, StructField, StructType}

import org.apache.iceberg._
import org.apache.iceberg.expressions.{Expressions, ResidualEvaluator}
import org.apache.iceberg.types.Types
import org.scalatest.funsuite.AnyFunSuite

import scala.collection.JavaConverters._

class GlutenIcebergSourceUtilSuite extends AnyFunSuite {

  test("combined input partitions preserve split metadata and preferred-location order") {
    val schema = new Schema(Types.NestedField.required(1, "p", Types.StringType.get()))
    val spec = PartitionSpec.builderFor(schema).identity("p").build()
    val taskA = fileScanTask(
      schema,
      spec,
      dataPath = "file:/tmp/data-a.parquet",
      deletePath = "file:/tmp/delete-a.parquet",
      partitionValue = "a",
      length = 101)
    val taskB = fileScanTask(
      schema,
      spec,
      dataPath = "file:/tmp/data-b.parquet",
      deletePath = "file:/tmp/delete-b.parquet",
      partitionValue = "b",
      length = 202)
    val inputA = sparkInputPartition(taskA, Array("host-a", "host-shared"))
    val inputB = sparkInputPartition(taskB, Array("host-b"))
    val wrapper = new SparkDataSourceRDDPartition(0, Seq(inputA, inputB))
    val readPartitionSchema = StructType(Seq(StructField("p", StringType)))

    val metadata =
      GlutenIcebergSourceUtil.collectPartitionMetadata(wrapper, readPartitionSchema)

    assert(metadata.splits.map(_.path) === Seq(
      "file:/tmp/data-a.parquet",
      "file:/tmp/data-b.parquet"))
    assert(metadata.splits.map(_.start) === Seq(0L, 0L))
    assert(metadata.splits.map(_.length) === Seq(101L, 202L))
    assert(metadata.splits.map(_.partitionColumns.get("p")) === Seq("a", "b"))
    assert(
      metadata.splits.flatMap(_.deleteFiles.asScala).map(_.path().toString) === Seq(
        "file:/tmp/delete-a.parquet",
        "file:/tmp/delete-b.parquet"))
    assert(metadata.preferredLocations === Seq("host-a", "host-shared", "host-b"))

    assert(GlutenIcebergSourceUtil.inputPartitionPlanningInfo(inputA, 7) ===
      Some((116L, FileFormat.PARQUET.toString)))
    assert(GlutenIcebergSourceUtil.inputPartitionPlanningInfo(inputB, 7) ===
      Some((217L, FileFormat.PARQUET.toString)))
  }

  private def fileScanTask(
      schema: Schema,
      spec: PartitionSpec,
      dataPath: String,
      deletePath: String,
      partitionValue: String,
      length: Long): FileScanTask = {
    val dataFile = DataFiles
      .builder(spec)
      .withPath(dataPath)
      .withFormat(FileFormat.PARQUET)
      .withPartitionPath(s"p=$partitionValue")
      .withRecordCount(1)
      .withFileSizeInBytes(length)
      .build()
    val deleteFile = FileMetadata
      .deleteFileBuilder(spec)
      .ofPositionDeletes()
      .withPath(deletePath)
      .withFormat(FileFormat.PARQUET)
      .withPartitionPath(s"p=$partitionValue")
      .withRecordCount(1)
      .withFileSizeInBytes(1)
      .build()
    new BaseFileScanTask(
      dataFile,
      Array(deleteFile),
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
}
