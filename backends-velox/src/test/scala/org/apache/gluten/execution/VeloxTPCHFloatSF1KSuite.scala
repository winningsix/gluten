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

import org.scalatest.concurrent.TimeLimits
import org.scalatest.time.{Seconds, Span}

import java.io.File

/**
 * Run the five TPC-H queries whose Gluten MPP plan matches Presto (Q6, Q12, Q16, Q17, Q18) against
 * an external 1TB TPC-H float parquet dataset, using the Plan-C MppStrategy and the cross-cut
 * plan-shape parity flags. Goal: prove the matching queries actually execute end-to-end via
 * MppNativeQueryRDD's JNI path.
 *
 * Dataset directory is taken from the system property `gluten.tpch.externalDataDir` (default
 * `/data/tpch/sf1k_v2_float`). Each TPC-H table is expected at `${externalDataDir}/[table]/`.
 *
 * This suite extends [[VeloxTPCHTableSupport]] directly (NOT [[VeloxTPCHSuite]]) so it does not
 * inherit the 22 TPC-H tests; we only register the five matching queries here.
 */
class VeloxTPCHFloatSF1KSuite extends VeloxTPCHTableSupport with TimeLimits {

  // Per-query hard cap. The test path runs df.collect() twice (once inside
  // runTPCHQuery for the compareResult=false branch, once inside dumpRows),
  // so each query effectively pays double the single-collect wall clock.
  // SF1K Q6 single-collect ~15s, so double = ~30s; 60s gives headroom for
  // the heavier queries (Q12/Q17/Q18). Anything past 60s is genuinely hung.
  private val perQueryTimeout = Span(60, Seconds)

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
      // Disable cudf JIT-fused expressions: NVRTC fails to compile EQUAL on Q12's
      // filter (`cudf::ast::operator_functor<EQUAL, true>::operator() no instance
      // matches`), which produces a Spark task retry that masks as success but
      // burns ~25s/query. AST/standalone-cudf path handles the same expressions
      // without the JIT compile step. Re-enable per-query if a workload needs it.
      .set("spark.gluten.sql.columnar.backend.velox.cudf.jit_expression_enabled", "false")
      // Disable cudf AST expression evaluator: hits "AST expression was provided
      // non-matching operand types" on Q17 and "like expects 2 inputs (3 vs. 2)"
      // on Q18. Standalone cudf-function path runs the same filters/projects
      // without going through the AST builder.
      .set("spark.gluten.sql.columnar.backend.velox.cudf.ast_expression_enabled", "false")
      // Dump every MppNativeQueryExec plan for offline diagnosis if the run fails.
      .set("spark.gluten.mpp.substraitDumpDir", "/opt/gluten/mpp-dumps-tpch-sf1k")
  }

  // Print actual rows for offline diff vs Presto-GPU SF1K reference (post-ANALYZE).
  // compareResult=false because we don't ship a q*.out reference in this suite's
  // resources; the [MPP-RESULT] tag lets us grep run logs deterministically.
  private def dumpRows(qid: Int, df: org.apache.spark.sql.DataFrame): Unit = {
    val rows = df.collect()
    // scalastyle:off println
    println(s"[MPP-RESULT] Q$qid count=${rows.length} schema=${df.schema.simpleString}")
    rows.take(10).foreach(r => println(s"[MPP-RESULT] Q$qid row: ${r.mkString("|")}"))
    if (rows.length > 10) {
      rows.takeRight(2).foreach(r => println(s"[MPP-RESULT] Q$qid tail: ${r.mkString("|")}"))
    }
    // scalastyle:on println
  }

  test("TPC-H q6") {
    failAfter(perQueryTimeout) {
      runTPCHQuery(6, tpchQueries, queriesResults, compareResult = false, noFallBack = true) {
        df => dumpRows(6, df)
      }
    }
  }

  test("TPC-H q12") {
    failAfter(perQueryTimeout) {
      runTPCHQuery(12, tpchQueries, queriesResults, compareResult = false, noFallBack = true) {
        df => dumpRows(12, df)
      }
    }
  }

  test("TPC-H q16") {
    failAfter(perQueryTimeout) {
      runTPCHQuery(16, tpchQueries, queriesResults, compareResult = false, noFallBack = true) {
        df => dumpRows(16, df)
      }
    }
  }

  test("TPC-H q17") {
    failAfter(perQueryTimeout) {
      runTPCHQuery(17, tpchQueries, queriesResults, compareResult = false, noFallBack = true) {
        df => dumpRows(17, df)
      }
    }
  }

  test("TPC-H q18") {
    failAfter(perQueryTimeout) {
      runTPCHQuery(18, tpchQueries, queriesResults, compareResult = false, noFallBack = true) {
        df => dumpRows(18, df)
      }
    }
  }
}
