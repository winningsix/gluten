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

import org.apache.spark.SparkConf

import java.io.File

/**
 * Run the five TPC-H queries whose Gluten MPP plan matches Presto (Q6, Q12, Q16, Q17, Q18) against
 * an external 1TB TPC-H float parquet dataset, using the Plan-C MppStrategy and the cross-cut
 * plan-shape parity flags. Goal: prove the matching queries actually execute end-to-end via
 * MppNativeQueryRDD's JNI path.
 *
 * Dataset directory is taken from the system property `gluten.tpch.externalDataDir` (default
 * `/data/tpch/sf1k_v2_float`). Each TPC-H table is expected at
 * `${externalDataDir}/[table]/`.
 *
 * This suite extends [[VeloxTPCHTableSupport]] directly (NOT [[VeloxTPCHSuite]]) so it does not
 * inherit the 22 TPC-H tests; we only register the five matching queries here.
 */
class VeloxTPCHFloatSF1KSuite extends VeloxTPCHTableSupport {

  protected val externalDataDir: String =
    sys.props.getOrElse("gluten.tpch.externalDataDir", "/data/tpch/sf1k_v2_float")

  override protected def createTPCHNotNullTables(): Unit = {
    TPCHTableDataFrames = TPCHTables
      .map(_.name)
      .map {
        table =>
          val tablePath = new File(externalDataDir, table).getAbsolutePath
          val tableDF = spark.read.format(fileFormat).load(tablePath)
          tableDF.createOrReplaceTempView(table)
          (table, tableDF)
      }
      .toMap
  }

  override protected def sparkConf: SparkConf = {
    super.sparkConf
      // 1TB-scale resources; parent default is too small.
      .set("spark.sql.shuffle.partitions", "16")
      .set("spark.memory.offHeap.size", "16g")
      .set("spark.sql.adaptive.enabled", "false")
      // Plan-C MppStrategy: intercept queries at planner level and route through MPP.
      .set("spark.gluten.mpp.enabled", "true")
      .set("spark.gluten.mpp.strategy.enabled", "true")
      // Cross-cut plan-shape parity (re-applied inside MppNativeQueryExec).
      .set("spark.gluten.mpp.singlePartitionSort", "true")
      .set("spark.gluten.mpp.removeRedundantShuffle", "true")
      .set("spark.gluten.mpp.parallelSortSplit", "true")
      .set("spark.gluten.mpp.fuseBroadcastBuilds", "true")
      // Run on cudf (GPU) where supported.
      .set("spark.gluten.sql.columnar.cudf", "true")
      .set("spark.gluten.sql.columnar.backend.velox.cudf.enabled", "true")
      .set("spark.gluten.sql.columnar.backend.velox.cudf.enableTableScan", "true")
      // Dump every MppNativeQueryExec plan for offline diagnosis if the run fails.
      .set("spark.gluten.mpp.substraitDumpDir", "/opt/gluten/mpp-dumps-tpch-sf1k")
  }

  test("TPC-H q6") {
    runTPCHQuery(6, tpchQueries, queriesResults, compareResult = false, noFallBack = true) {
      _ => ()
    }
  }

  test("TPC-H q12") {
    runTPCHQuery(12, tpchQueries, queriesResults, compareResult = false, noFallBack = true) {
      _ => ()
    }
  }

  test("TPC-H q16") {
    runTPCHQuery(16, tpchQueries, queriesResults, compareResult = false, noFallBack = true) {
      _ => ()
    }
  }

  test("TPC-H q17") {
    runTPCHQuery(17, tpchQueries, queriesResults, compareResult = false, noFallBack = true) {
      _ => ()
    }
  }

  test("TPC-H q18") {
    runTPCHQuery(18, tpchQueries, queriesResults, compareResult = false, noFallBack = true) {
      _ => ()
    }
  }
}
