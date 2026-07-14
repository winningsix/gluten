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

import org.apache.gluten.execution.{MppNativeQueryExec, MppPreparedChildExec, VeloxWholeStageTransformerSuite, WholeStageTransformer}

import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.exchange.ShuffleExchangeLike

/**
 * Test suite verifying MppStrategy (Plan C) behavior:
 *   - Correctly intercepts query plans
 *   - Skips DDL/command plans
 *   - Produces correct results through dynamic native MPP execution
 *   - Properly falls back when MPP is disabled
 *   - Preserves prepare-time child isolation while exposing the plan used for native extraction
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
      .set("spark.gluten.mpp.substraitDumpDir", "/tmp/gluten-mpp-strategy-dumps")
      .set("spark.gluten.sql.columnar.cudf", "true")
      .set("spark.gluten.sql.columnar.backend.velox.cudf.enabled", "true")
      .set("spark.gluten.sql.columnar.backend.velox.cudf.enableTableScan", "true")
  }

  override def beforeAll(): Unit = {
    super.beforeAll()
    createTPCHNotNullTables()
  }

  private def findMppExec(df: org.apache.spark.sql.DataFrame): Option[MppNativeQueryExec] = {
    val plan = df.queryExecution.executedPlan
    plan.collect { case m: MppNativeQueryExec => m }.headOption
  }

  /**
   * Walk a physical plan tree and extract fragment boundaries.
   *
   * Fragments are WholeStageTransformer subtrees separated by ShuffleExchangeLike boundaries. This
   * test-only structural preview walks the same prepared child that Plan C passes into its native
   * fragment extraction path.
   *
   * @return
   *   (wholeStageTransformers, shuffleExchanges) found in the plan tree
   */
  private def extractFragmentsFromPhysicalPlan(
      plan: SparkPlan): (Seq[WholeStageTransformer], Seq[ShuffleExchangeLike]) = {
    val fragments = scala.collection.mutable.ArrayBuffer[WholeStageTransformer]()
    val exchanges = scala.collection.mutable.ArrayBuffer[ShuffleExchangeLike]()

    def walk(node: SparkPlan): Unit = {
      node match {
        case wst: WholeStageTransformer => fragments += wst
        case ex: ShuffleExchangeLike => exchanges += ex
        case _ =>
      }
      node.children.foreach(walk)
    }

    walk(plan)
    (fragments.toSeq, exchanges.toSeq)
  }

  test("MppStrategy: simple agg query produces correct result") {
    val df = spark.sql("SELECT count(*) FROM lineitem")
    val result = df.collect()
    assert(result.length == 1)
    assert(result(0).getLong(0) > 0, "count should be > 0")
  }

  test("MppStrategy: group by agg query produces correct result") {
    val df = spark.sql("""SELECT l_returnflag, count(*) as cnt
                         |FROM lineitem
                         |GROUP BY l_returnflag
                         |ORDER BY l_returnflag""".stripMargin)
    val result = df.collect()
    assert(result.length > 0, "should have results")
    val flags = result.map(_.getString(0))
    assert(flags.toSeq == flags.sorted.toSeq, "results should be sorted")
  }

  test("MppStrategy: join query produces correct result") {
    val df = spark.sql("""SELECT count(*)
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
    val df = spark.sql("""SELECT sum(l_extendedprice * l_discount) as revenue
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
    val df = spark.sql("""SELECT count(*) as cnt
                         |FROM orders o
                         |JOIN lineitem l ON l.l_orderkey = o.o_orderkey""".stripMargin)
    val result = df.collect()
    assert(result.length == 1)
    assert(result(0).getLong(0) > 0, "join should produce results")
  }

  test("MppStrategy: subquery produces correct result") {
    val df = spark.sql("""SELECT l_returnflag, l_linestatus, count(*) as cnt
                         |FROM lineitem
                         |WHERE l_quantity > (SELECT avg(l_quantity) FROM lineitem)
                         |GROUP BY l_returnflag, l_linestatus""".stripMargin)
    val result = df.collect()
    assert(result.length > 0, "subquery should have results")
  }

  test("MppStrategy: UNION ALL produces correct result") {
    val df =
      spark.sql("""SELECT l_returnflag, count(*) as cnt FROM lineitem GROUP BY l_returnflag
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
    val df = spark.sql("""SELECT l_returnflag, l_linestatus,
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
    val df = spark.sql("""SELECT l_returnflag, count(*) FROM lineitem
                         |GROUP BY l_returnflag ORDER BY l_returnflag""".stripMargin)
    val mppExec = findMppExec(df)
    assert(mppExec.isDefined, "MppNativeQueryExec should be in plan when MPP enabled")
  }

  test("MppStrategy: results match BSP baseline") {
    // Run same query with MPP enabled and disabled, compare results
    val sql =
      "SELECT l_returnflag, sum(l_quantity) as sq FROM lineitem " +
        "GROUP BY l_returnflag ORDER BY l_returnflag"
    val mppResult = spark.sql(sql).collect()

    withSQLConf(
      "spark.gluten.mpp.enabled" -> "false",
      "spark.gluten.mpp.strategy.enabled" -> "false"
    ) {
      val bspResult = spark.sql(sql).collect()
      assert(
        mppResult.length == bspResult.length,
        s"Row count mismatch: MPP=${mppResult.length} BSP=${bspResult.length}")
      mppResult.zip(bspResult).foreach {
        case (mpp, bsp) =>
          assert(
            mpp.getString(0) == bsp.getString(0),
            s"Flag mismatch: MPP=${mpp.getString(0)} BSP=${bsp.getString(0)}")
      }
    }
  }

  // ============================================================
  // Plan C prepared-child and dynamic-extraction tests.
  // ============================================================

  test("MppNativeQueryExec child has ShuffleExchangeLike nodes (shuffle proof)") {
    // A GROUP BY + ORDER BY query forces at least one shuffle exchange.
    // This verifies the child plan retains exchange nodes (Plan D: wrap, don't replace).
    val df = spark.sql("""SELECT l_returnflag, count(*) as cnt
                         |FROM lineitem
                         |GROUP BY l_returnflag
                         |ORDER BY l_returnflag""".stripMargin)
    val mppExec = findMppExec(df)
    assert(mppExec.isDefined, "MppNativeQueryExec should be in plan")

    val childPlan = mppExec.get.preparedChildForTests
    val exchangeNodes = childPlan.collect { case ex: ShuffleExchangeLike => ex }
    // Plan D wraps the original plan; the child tree must still contain exchanges.
    // If this fails, the child was incorrectly stripped of its exchange nodes.
    assert(
      exchangeNodes.nonEmpty,
      s"Child plan should contain ShuffleExchangeLike nodes for GROUP BY + ORDER BY query. " +
        s"Child plan tree:\n${childPlan.treeString}"
    )
  }

  test("MppNativeQueryExec extraction plan has WholeStageTransformer fragments") {
    // A query with GROUP BY + ORDER BY should produce at least 2 WholeStageTransformer
    // nodes (one for the scan+agg stage, one for the final sort stage).
    // This is the prerequisite for Phase 2: each WST becomes a native fragment.
    val df = spark.sql("""SELECT l_returnflag, l_linestatus,
                         |  sum(l_quantity) as sum_qty,
                         |  count(*) as count_order
                         |FROM lineitem
                         |WHERE l_shipdate <= date '1998-09-02'
                         |GROUP BY l_returnflag, l_linestatus
                         |ORDER BY l_returnflag, l_linestatus""".stripMargin)
    val mppExec = findMppExec(df)
    assert(mppExec.isDefined, "MppNativeQueryExec should be in plan")

    val childPlan = mppExec.get.fragmentExtractionPlanForTests
    val wstNodes = childPlan.collect { case wst: WholeStageTransformer => wst }
    // Phase 2 requirement: >= 2 fragments means there is at least one exchange boundary
    // that can be converted to a streaming GPU exchange.
    assert(
      wstNodes.size >= 2,
      s"Child plan should have >= 2 WholeStageTransformer nodes (found ${wstNodes.size}). " +
        s"Child plan tree:\n${childPlan.treeString}")
  }

  test("Fragment extraction preview finds exchange boundaries") {
    // Use the extractFragmentsFromPhysicalPlan helper (Phase 2 prototype) to walk the
    // child plan and identify fragment boundaries at ShuffleExchangeLike nodes.
    val df = spark.sql("""SELECT l_returnflag, count(*) as cnt
                         |FROM lineitem
                         |GROUP BY l_returnflag
                         |ORDER BY l_returnflag""".stripMargin)
    val mppExec = findMppExec(df)
    assert(mppExec.isDefined, "MppNativeQueryExec should be in plan")

    val childPlan = mppExec.get.fragmentExtractionPlanForTests
    val (fragments, exchanges) = extractFragmentsFromPhysicalPlan(childPlan)

    // With GROUP BY + ORDER BY we expect:
    //   >= 2 WholeStageTransformer fragments (scan+agg, final sort)
    //   >= 1 ShuffleExchangeLike boundary
    assert(
      fragments.size >= 2,
      s"Expected >= 2 fragments, got ${fragments.size}. " +
        s"Child plan:\n${childPlan.treeString}")
    assert(
      exchanges.nonEmpty,
      s"Expected >= 1 exchange boundary, got ${exchanges.size}. " +
        s"Child plan:\n${childPlan.treeString}")

    // Phase 2 invariant: #exchanges should be #fragments - 1 (linear pipeline)
    // or >= #fragments - 1 (DAG with fan-in). Log for documentation.
    logInfo(
      s"Fragment extraction: ${fragments.size} fragments, ${exchanges.size} exchanges. " +
        s"Fragment stageIds: ${fragments.map(_.stageId).mkString(", ")}")
  }

  test("Each extraction-plan WholeStageTransformer can generate Substrait") {
    // For each WholeStageTransformer in the child plan, verify that Substrait generation
    // succeeds. This is what Phase 2 will do for each fragment before submitting to JNI.
    val df = spark.sql("""SELECT l_returnflag, count(*) as cnt
                         |FROM lineitem
                         |GROUP BY l_returnflag
                         |ORDER BY l_returnflag""".stripMargin)
    val mppExec = findMppExec(df)
    assert(mppExec.isDefined, "MppNativeQueryExec should be in plan")

    val childPlan = mppExec.get.fragmentExtractionPlanForTests
    val wstNodes = childPlan.collect { case wst: WholeStageTransformer => wst }
    assert(wstNodes.nonEmpty, "Should have at least one WholeStageTransformer")

    wstNodes.foreach {
      wst =>
        // doWholeStageTransform() generates a full Substrait plan for this stage.
        // This is the same call that WholeStageTransformer.doExecuteColumnar() makes.
        val wsCtx = wst.doWholeStageTransform()
        assert(wsCtx != null, s"doWholeStageTransform() returned null for stage ${wst.stageId}")
        assert(wsCtx.root != null, s"Substrait PlanNode is null for stage ${wst.stageId}")

        // Verify the plan can be serialized to bytes (what JNI expects)
        val planBytes = wsCtx.root.toProtobuf.toByteArray
        assert(
          planBytes.length > 0,
          s"Substrait plan bytes should be non-empty for stage ${wst.stageId}")

        logInfo(s"Stage ${wst.stageId}: Substrait plan = ${planBytes.length} bytes")
    }
  }

  test("MPP metrics are reported") {
    // Run a Plan C query and check that dynamic extraction updates the public MPP metrics.
    val df = spark.sql("""SELECT l_returnflag, count(*) as cnt
                         |FROM lineitem
                         |GROUP BY l_returnflag
                         |ORDER BY l_returnflag""".stripMargin)
    // Force execution so metrics are populated
    df.collect()

    val mppExec = findMppExec(df)
    assert(mppExec.isDefined, "MppNativeQueryExec should be in plan")

    val metricsMap = mppExec.get.metrics
    // Verify metric keys exist (defined in MppNativeQueryExec)
    assert(metricsMap.contains("numFragments"), "Should have numFragments metric")
    assert(metricsMap.contains("numExchanges"), "Should have numExchanges metric")
    assert(metricsMap.contains("outputRows"), "Should have outputRows metric")
    assert(metricsMap.contains("totalQueryTimeMs"), "Should have totalQueryTimeMs metric")

    val numFragments = metricsMap("numFragments").value
    val numExchanges = metricsMap("numExchanges").value
    assert(
      numFragments >= 2,
      s"Plan C should report at least two dynamically extracted fragments, got $numFragments")
    assert(
      numExchanges >= 1,
      s"Plan C should report at least one dynamically extracted exchange, got $numExchanges")
  }

  test("Plan C defers extraction while keeping the prepared child opaque to Spark") {
    val df = spark.sql("""SELECT l_returnflag, count(*) as cnt
                         |FROM lineitem
                         |GROUP BY l_returnflag
                         |ORDER BY l_returnflag""".stripMargin)
    val mppExec = findMppExec(df)
    assert(mppExec.isDefined, "MppNativeQueryExec should be in plan")

    val planC = mppExec.get
    assert(
      planC.fragments.forall(_.rootOperator == null) && planC.exchanges.isEmpty,
      "Plan C should defer concrete fragment extraction until execution")
    assert(
      planC.child.isInstanceOf[MppPreparedChildExec],
      s"Plan C should hide its prepared child, got ${planC.child.getClass.getSimpleName}")
    assert(planC.child.children.isEmpty, "The prepare-time wrapper must remain a Spark leaf")

    val preparedChild = planC.preparedChildForTests
    assert(
      !preparedChild.isInstanceOf[MppPreparedChildExec],
      "The execution-visible child should be unwrapped")
    assert(
      preparedChild.collect { case exchange: ShuffleExchangeLike => exchange }.nonEmpty,
      "The execution-visible child should retain exchange boundaries:\n" +
        preparedChild.treeString)
  }
}
