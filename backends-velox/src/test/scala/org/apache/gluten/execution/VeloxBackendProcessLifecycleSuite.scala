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

import org.apache.gluten.config.GlutenConfig

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import org.apache.commons.io.FileUtils
import org.scalatest.funsuite.AnyFunSuite

import java.nio.file.Files

class VeloxBackendProcessLifecycleSuite extends AnyFunSuite {
  test("native validation survives SparkContext recreation in the same JVM") {
    val warehouse = Files.createTempDirectory("velox-backend-lifecycle")
    try {
      validateNativePlanning(createSparkSession(warehouse.toString))
      validateNativePlanning(createSparkSession(warehouse.toString))
    } finally {
      FileUtils.deleteDirectory(warehouse.toFile)
    }
  }

  private def createSparkSession(warehouse: String): SparkSession = {
    val conf = new SparkConf()
      .setMaster("local[1]")
      .setAppName("VeloxBackendProcessLifecycleSuite")
      .set("spark.plugins", "org.apache.gluten.GlutenPlugin")
      .set("spark.executor.instances", "1")
      .set("spark.executor.cores", "1")
      .set("spark.executor.memory", "1g")
      .set("spark.executor.memoryOverhead", "512m")
      .set("spark.memory.offHeap.enabled", "true")
      .set("spark.memory.offHeap.size", "1g")
      .set("spark.shuffle.manager", "org.apache.spark.shuffle.sort.ColumnarShuffleManager")
      .set("spark.sql.adaptive.enabled", "false")
      .set("spark.sql.shuffle.partitions", "1")
      .set("spark.sql.warehouse.dir", warehouse)
      .set("spark.ui.enabled", "false")
      .set(GlutenConfig.GLUTEN_UI_ENABLED.key, "false")
    SparkSession.builder().config(conf).getOrCreate()
  }

  private def validateNativePlanning(spark: SparkSession): Unit = {
    try {
      val plan = spark.sql("SELECT array(1, 2) AS value").queryExecution.executedPlan
      assert(
        plan.find {
          case _: ProjectExecTransformer | _: MppNativeQueryExec => true
          case _ => false
        }.isDefined,
        plan.treeString)
    } finally {
      spark.stop()
      SparkSession.clearActiveSession()
      SparkSession.clearDefaultSession()
    }
  }
}
