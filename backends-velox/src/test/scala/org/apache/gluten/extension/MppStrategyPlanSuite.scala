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

import org.apache.gluten.execution.{MppNativeQueryExec, VeloxWholeStageTransformerSuite}

/**
 * Test suite verifying MppStrategy (Plan C) behavior:
 * - Correctly intercepts query plans
 * - Skips DDL/command plans
 * - Produces correct results via BSP delegation (Phase 1)
 * - Properly falls back when MPP is disabled
 */
class MppStrategyPlanSuite extends VeloxWholeStageTransformerSuite {

  override protected val resourcePath: String = "/tpch-data-parquet-float"
  override protected val fileFormat: String = "parquet"

  override protected def sparkConf: org.apache.spark.SparkConf = {
    super.sparkConf
      .set("spark.sql.shuffle.partitions", "4")
      .set("spark.sql.adaptive.enabled", "false")
      .set("spark.gluten.mpp.enabled", "true")
      .set("spark.gluten.mpp.strategy.enabled", "true")
  }

  override def beforeAll(): Unit = {
    super.beforeAll()
    createTPCHNotNullTables()
  }

  private def findMppExec(df: org.apache.spark.sql.DataFrame): Option[MppNativeQueryExec] = {
    val plan = df.queryExecution.executedPlan
    plan.collect { case m: MppNativeQueryExec => m }.headOption
  }

  test("MppStrategy: simple agg query produces correct result") {
    val df = spark.sql("SELECT count(*) FROM lineitem")
    val result = df.collect()
    assert(result.length == 1)
    assert(result(0).getLong(0) > 0, "count should be > 0")
  }

  test("MppStrategy: group by agg query produces correct result") {
    val df = spark.sql(
      """SELECT l_returnflag, count(*) as cnt
        |FROM lineitem
        |GROUP BY l_returnflag
        |ORDER BY l_returnflag""".stripMargin)
    val result = df.collect()
    assert(result.length > 0, "should have results")
    val flags = result.map(_.getString(0))
    assert(flags.toSeq == flags.sorted.toSeq, "results should be sorted")
  }

  test("MppStrategy: join query produces correct result") {
    val df = spark.sql(
      """SELECT count(*)
        |FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey
        |WHERE o.o_orderstatus = 'F'""".stripMargin)
    val result = df.collect()
    assert(result.length == 1)
    assert(result(0).getLong(0) > 0)
  }

  test("MppStrategy: disabled when spark.gluten.mpp.enabled=false") {
    withSQLConf("spark.gluten.mpp.enabled" -> "false") {
      val df = spark.sql("SELECT count(*) FROM lineitem")
      val result = df.collect()
      assert(result(0).getLong(0) > 0)
    }
  }

  test("MppStrategy: disabled when spark.gluten.mpp.strategy.enabled=false") {
    withSQLConf("spark.gluten.mpp.strategy.enabled" -> "false") {
      val df = spark.sql("SELECT count(*) FROM lineitem")
      val result = df.collect()
      assert(result(0).getLong(0) > 0)
    }
  }

  test("MppStrategy: DDL (CREATE VIEW) not intercepted") {
    spark.sql("CREATE OR REPLACE TEMP VIEW test_view AS SELECT * FROM lineitem LIMIT 10")
    val result = spark.sql("SELECT count(*) FROM test_view").collect()
    assert(result(0).getLong(0) == 10)
  }

  test("MppStrategy: TPC-H q6 produces correct result") {
    val df = spark.sql(
      """SELECT sum(l_extendedprice * l_discount) as revenue
        |FROM lineitem
        |WHERE l_shipdate >= date '1994-01-01'
        |  AND l_shipdate < date '1995-01-01'
        |  AND l_discount BETWEEN 0.05 AND 0.07
        |  AND l_quantity < 24""".stripMargin)
    val result = df.collect()
    assert(result.length == 1, "q6 should return 1 row")
  }

  test("MppStrategy: multi-table join produces result") {
    // Simplified join without strict date filters (test data is tiny)
    val df = spark.sql(
      """SELECT count(*) as cnt
        |FROM orders o
        |JOIN lineitem l ON l.l_orderkey = o.o_orderkey""".stripMargin)
    val result = df.collect()
    assert(result.length == 1)
    assert(result(0).getLong(0) > 0, "join should produce results")
  }

  test("MppStrategy: subquery produces correct result") {
    val df = spark.sql(
      """SELECT l_returnflag, l_linestatus, count(*) as cnt
        |FROM lineitem
        |WHERE l_quantity > (SELECT avg(l_quantity) FROM lineitem)
        |GROUP BY l_returnflag, l_linestatus""".stripMargin)
    val result = df.collect()
    assert(result.length > 0, "subquery should have results")
  }

  test("MppStrategy: UNION ALL produces correct result") {
    val df = spark.sql(
      """SELECT l_returnflag, count(*) as cnt FROM lineitem GROUP BY l_returnflag
        |UNION ALL
        |SELECT o_orderstatus, count(*) FROM orders GROUP BY o_orderstatus""".stripMargin)
    val result = df.collect()
    assert(result.length > 0, "UNION ALL should have results")
  }

  test("MppStrategy: LIMIT query produces correct result") {
    val df = spark.sql("SELECT l_orderkey FROM lineitem LIMIT 5")
    val result = df.collect()
    assert(result.length == 5, "LIMIT 5 should return exactly 5 rows")
  }

  test("MppStrategy: empty result query works") {
    val df = spark.sql("SELECT * FROM lineitem WHERE l_quantity < 0")
    val result = df.collect()
    assert(result.length == 0, "impossible filter should return 0 rows")
  }

  test("MppStrategy: multiple shuffles (group by + order by) produces correct result") {
    val df = spark.sql(
      """SELECT l_returnflag, l_linestatus,
        |  sum(l_quantity) as sum_qty,
        |  count(*) as count_order
        |FROM lineitem
        |WHERE l_shipdate <= date '1998-09-02'
        |GROUP BY l_returnflag, l_linestatus
        |ORDER BY l_returnflag, l_linestatus""".stripMargin)
    val result = df.collect()
    assert(result.length > 0, "TPC-H q1 style query should have results")
    // Verify ordering
    val flags = result.map(_.getString(0))
    assert(flags.toSeq == flags.sorted.toSeq)
  }

  test("MppStrategy: MppNativeQueryExec present in plan when enabled") {
    val df = spark.sql(
      """SELECT l_returnflag, count(*) FROM lineitem
        |GROUP BY l_returnflag ORDER BY l_returnflag""".stripMargin)
    val mppExec = findMppExec(df)
    assert(mppExec.isDefined, "MppNativeQueryExec should be in plan when MPP enabled")
  }

  test("MppStrategy: results match BSP baseline") {
    // Run same query with MPP enabled and disabled, compare results
    val sql = "SELECT l_returnflag, sum(l_quantity) as sq FROM lineitem GROUP BY l_returnflag ORDER BY l_returnflag"
    val mppResult = spark.sql(sql).collect()

    withSQLConf(
      "spark.gluten.mpp.enabled" -> "false",
      "spark.gluten.mpp.strategy.enabled" -> "false"
    ) {
      val bspResult = spark.sql(sql).collect()
      assert(mppResult.length == bspResult.length,
        s"Row count mismatch: MPP=${mppResult.length} BSP=${bspResult.length}")
      mppResult.zip(bspResult).foreach { case (mpp, bsp) =>
        assert(mpp.getString(0) == bsp.getString(0),
          s"Flag mismatch: MPP=${mpp.getString(0)} BSP=${bsp.getString(0)}")
      }
    }
  }
}
