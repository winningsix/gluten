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

import org.apache.gluten.extension.columnar.RewriteExistenceJoinRhsDedup

import org.apache.spark.sql.{QueryTest, Row}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.classic.ClassicDataset
import org.apache.spark.sql.test.SharedSparkSession

import java.sql.Date

class MergeExistenceJoinSummariesSuite extends QueryTest with SharedSparkSession {
  import testImplicits._

  private val enabledKey =
    "spark.gluten.sql.optimizer.mergeExistenceJoinSummaries.enabled"

  override protected def sparkConf: org.apache.spark.SparkConf = {
    val conf = super.sparkConf
    conf.getAll.collect { case (key, "null") => key }.foreach(conf.remove)
    conf
      .set("spark.master", "local[2]")
      .set("spark.sql.adaptive.enabled", "false")
  }

  private def date(value: String): Date = Date.valueOf(value)

  private def createTestViews(): Unit = {
    Seq(
      (10L, "Supplier#10", 1),
      (20L, "Supplier#20", 1),
      (30L, "Supplier#30", 2))
      .toDF("s_suppkey", "s_name", "s_nationkey")
      .createOrReplaceTempView("merge_supplier")
    Seq(
      (1, "SAUDI ARABIA"),
      (2, "GERMANY"))
      .toDF("n_nationkey", "n_name")
      .createOrReplaceTempView("merge_nation")
    Seq(
      (1L, "F"),
      (2L, "F"),
      (3L, "F"),
      (4L, "O"),
      (5L, "F"))
      .toDF("o_orderkey", "o_orderstatus")
      .createOrReplaceTempView("merge_orders")
    Seq(
      // Two late rows for supplier 10 and one on-time row for another supplier: count both.
      (1L, 10L, date("1996-01-01"), date("1996-01-02")),
      (1L, 10L, date("1996-01-01"), date("1996-01-03")),
      (1L, 20L, date("1996-01-03"), date("1996-01-03")),
      // Two different late suppliers: rejected by NOT EXISTS.
      (2L, 10L, date("1996-01-01"), date("1996-01-02")),
      (2L, 20L, date("1996-01-01"), date("1996-01-02")),
      // No other supplier: rejected by EXISTS.
      (3L, 10L, date("1996-01-01"), date("1996-01-02")),
      // Correct existence shape, but the order is not final.
      (4L, 10L, date("1996-01-01"), date("1996-01-02")),
      (4L, 20L, date("1996-01-02"), date("1996-01-02")),
      // Supplier 20 is the sole late supplier.
      (5L, 20L, date("1996-01-01"), date("1996-01-02")),
      (5L, 10L, date("1996-01-02"), date("1996-01-02")))
      .toDF("l_orderkey", "l_suppkey", "l_commitdate", "l_receiptdate")
      .createOrReplaceTempView("merge_lineitem")
  }

  private val query =
    """
      |select s_name, count(*) as numwait
      |from merge_supplier, merge_lineitem l1, merge_orders, merge_nation
      |where s_suppkey = l1.l_suppkey
      |  and o_orderkey = l1.l_orderkey
      |  and o_orderstatus = 'F'
      |  and l1.l_receiptdate > l1.l_commitdate
      |  and exists (
      |    select * from merge_lineitem l2
      |    where l2.l_orderkey = l1.l_orderkey
      |      and l2.l_suppkey <> l1.l_suppkey)
      |  and not exists (
      |    select * from merge_lineitem l3
      |    where l3.l_orderkey = l1.l_orderkey
      |      and l3.l_suppkey <> l1.l_suppkey
      |      and l3.l_receiptdate > l3.l_commitdate)
      |  and s_nationkey = n_nationkey
      |  and n_name = 'SAUDI ARABIA'
      |group by s_name
      |""".stripMargin

  private def lineitemLeafCount(plan: LogicalPlan): Int = {
    plan.collect {
      case leaf
          if leaf.children.isEmpty &&
            leaf.output.exists(_.name == "l_orderkey") &&
            leaf.output.exists(_.name == "l_suppkey") =>
        leaf
    }.size
  }

  test("merges paired unfiltered and filtered existence summaries") {
    createTestViews()
    val original = withSQLConf(enabledKey -> "false") {
      spark.sql(query).queryExecution.optimizedPlan
    }
    val summarized = RewriteExistenceJoinRhsDedup(spark).apply(original)
    val rewritten = withSQLConf(enabledKey -> "true") {
      MergeExistenceJoinSummaries(spark).apply(summarized)
    }

    checkAnswer(
      ClassicDataset.ofRows(spark, rewritten),
      Seq(Row("Supplier#10", 2L), Row("Supplier#20", 1L)))
    assert(lineitemLeafCount(summarized) == 3, summarized.treeString)
    assert(lineitemLeafCount(rewritten) == 2, rewritten.treeString)

    val rewrittenAgain = withSQLConf(enabledKey -> "true") {
      MergeExistenceJoinSummaries(spark).apply(rewritten)
    }
    assert(lineitemLeafCount(rewrittenAgain) == 2, rewrittenAgain.treeString)
  }
}
