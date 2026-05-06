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

  private val q4ExistsLineitemDedupKey = "spark.gluten.mpp.q4ExistsLineitemDedup"

  // Per-query hard cap. Presto-GPU SF1K runs the full 22 in ~66s = ~3s/query;
  // measured Q6 wall-clock through Spark-Gluten + multi-fragment + UCX
  // tear-down is ~32s (Spark path overhead, not query execution). 45s
  // gives correct queries headroom to land green; 22-query sweep capped
  // at ~16 min worst case until we shave the Spark-side overhead.
  private val perQueryTimeout = Span(
    sys.props
      .get("gluten.tpch.perQueryTimeoutSeconds")
      .flatMap(s => scala.util.Try(s.toInt).toOption)
      .getOrElse(45),
    Seconds)

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
      // 1TB-scale resources; parent default is too small. 32g (was 16g) so the
      // 22-query sweep doesn't OOM on heavy joins like Q1 where 16-replica
      // PartitionedOutput each holds ~512MB. Long-term fix: switch to
      // 1-task-per-worker model so memory is shared across drivers.
      .set("spark.sql.shuffle.partitions", "16")
      .set("spark.memory.offHeap.size", "32g")
      .set("spark.sql.adaptive.enabled", "false")
      // BHJ via explicit BROADCAST(t) hints (see tpchSQL override). CBO is
      // unreachable here because parquet temp views report defaultSizeInBytes
      // = Long.MaxValue and full ANALYZE FOR ALL COLUMNS on SF1K lineitem
      // takes >30 min in surefire local mode. Hints sidestep CBO entirely.
      // Bump autoBroadcastJoinThreshold past Spark's default 10MB so the
      // hinted broadcast tables (part ~30MB, supplier ~1.4MB, customer
      // ~300MB) clear the threshold even without stats.
      .set("spark.sql.autoBroadcastJoinThreshold", (2L * 1024 * 1024 * 1024).toString)
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
      // Keep AST expression evaluator ENABLED. Earlier we disabled it to dodge
      // Q17/Q18 cuDF-AST builder errors ("like expects 2 inputs", "startswith
      // unsupported"), but disabling it also unregisters the AST evaluator
      // entirely (registerAstEvaluator is gated on this config in ToCudf.cpp:344),
      // which means basic arithmetic like multiply(a, subtract(1, b)) -- TPC-H
      // Q1's classic l_extendedprice * (1 - l_discount) -- has no evaluator at
      // all and falls back to CPU. The Q17/Q18 issues are separate and need
      // narrower handling (per-expression check, not whole-AST disable).
      .set("spark.gluten.sql.columnar.backend.velox.cudf.ast_expression_enabled", "true")
      // Dump every MppNativeQueryExec plan for offline diagnosis if the run fails.
      .set("spark.gluten.mpp.substraitDumpDir", "/opt/gluten/mpp-dumps-tpch-sf1k")
  }

  // Per-query BROADCAST hint mapping for Track 1 (Phase 3): force Spark to
  // pick BroadcastHashJoin so Phase 1+2 fused-broadcast iterator transport
  // (commit f95dec3b3) actually gets exercised. Mirrors the REPLICATED side
  // Presto's CBO picks; sized so the build relation fits under
  // autoBroadcastJoinThreshold=2GB for the SF1K_v2_float dataset:
  //   part      ~30MB    (Q14, Q16, Q17)
  //   supplier  ~1.4MB   (Q16)
  //   customer  ~300MB   (Q18)
  // Q12's outer join sides (orders ~150GB, lineitem ~720GB) are too large
  // for any broadcast at SF1K; left without a hint until we reshape that
  // query separately.
  private val broadcastHintByQuery: Map[Int, String] = Map(
    14 -> "/*+ BROADCAST(part) */",
    16 -> "/*+ BROADCAST(part, supplier) */",
    17 -> "/*+ BROADCAST(part) */",
    18 -> "/*+ BROADCAST(customer) */",
    19 -> "/*+ BROADCAST(part) */"
  )

  private val q4ExistsLineitemDedupSQL: String =
    """
      |select
      |  o.o_orderpriority,
      |  sum(case when l.late_lineitem_count > 0 then 1 else 0 end) as order_count
      |from
      |  orders o
      |join (
      |  select
      |    l_orderkey,
      |    count(*) as late_lineitem_count
      |  from
      |    lineitem
      |  where
      |    l_commitdate < l_receiptdate
      |  group by
      |    l_orderkey
      |) l
      |  on l.l_orderkey = o.o_orderkey
      |where
      |  o.o_orderdate >= date '1993-07-01'
      |  and o.o_orderdate < date '1993-07-01' + interval '3' month
      |group by
      |  o.o_orderpriority
      |order by
      |  o.o_orderpriority
      |""".stripMargin

  private def q4ExistsLineitemDedupEnabled: Boolean =
    sys.props
      .get(q4ExistsLineitemDedupKey)
      .exists(_.trim.equalsIgnoreCase("true"))

  override protected def tpchSQL(queryNum: Int, tpchQueries: String): String = {
    val raw =
      if (queryNum == 4 && q4ExistsLineitemDedupEnabled) {
        q4ExistsLineitemDedupSQL
      } else {
        super.tpchSQL(queryNum, tpchQueries)
      }
    broadcastHintByQuery.get(queryNum) match {
      case Some(hint) =>
        // Inject hint after the OUTER `select` keyword (case-insensitive,
        // first match only). Spark requires hints between SELECT and the
        // first projection; replaceFirst with a word-boundary regex keeps
        // subqueries' inner SELECTs untouched.
        raw.replaceFirst("(?i)\\bselect\\b", s"select $hint")
      case None => raw
    }
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

  // Run all 22 TPC-H queries. noFallBack=false so non-MPP-eligible queries
  // surface as their actual failure mode (not as a generic fallback test
  // failure). The 60s per-query failAfter contains hangs.
  // Optional sys-prop filter: -Dgluten.tpch.onlyQuery=6 registers only Q6.
  // Comma-separated values are also supported, for example:
  // -Dgluten.tpch.onlyQuery=1,2,3,12,18.
  // Used by per-query JVM-isolation runs to bypass cuDF abort-path memory
  // leaks that survive across tests in a single JVM (RC: failed task leaves
  // ~128MB stuck in MemoryPool, JVM teardown asserts on reservedBytes != 0).
  // Defaults to all 22 when unset.
  private val onlyQueries: Option[Seq[Int]] =
    sys.props
      .get("gluten.tpch.onlyQuery")
      .map {
        _.split(",").toSeq
          .flatMap(s => scala.util.Try(s.trim.toInt).toOption)
          .distinct
          .sorted
      }
      .filter(_.nonEmpty)

  private val queriesToRun: Seq[Int] = onlyQueries.getOrElse(1 to 22)

  queriesToRun.foreach {
    qid =>
      test(s"TPC-H q$qid") {
        failAfter(perQueryTimeout) {
          val previewDf = spark.sql(tpchSQL(qid, tpchQueries))
          logWarning(
            s"VeloxTPCHFloatSF1KSuite: Q$qid PRE-COLLECT optimizedPlan with stats:\n" +
              previewDf.queryExecution.stringWithStats)
          runTPCHQuery(qid, tpchQueries, queriesResults, compareResult = false, noFallBack = false)(
            df => dumpRows(qid, df))
        }
      }
  }
}
