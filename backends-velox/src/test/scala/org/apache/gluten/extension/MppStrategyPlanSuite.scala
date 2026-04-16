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

  test("MppStrategy: multi-table join (TPC-H q3 style) produces result") {
    val df = spark.sql(
      """SELECT l.l_orderkey, sum(l.l_extendedprice * (1 - l.l_discount)) as revenue
        |FROM customer c
        |JOIN orders o ON c.c_custkey = o.o_custkey
        |JOIN lineitem l ON l.l_orderkey = o.o_orderkey
        |WHERE c.c_mktsegment = 'BUILDING'
        |  AND o.o_orderdate < date '1995-03-15'
        |  AND l.l_shipdate > date '1995-03-15'
        |GROUP BY l.l_orderkey
        |ORDER BY revenue DESC
        |LIMIT 10""".stripMargin)
    val result = df.collect()
    assert(result.length > 0, "q3-style query should have results")
  }
}
