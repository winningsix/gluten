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

import org.apache.spark.sql.{QueryTest, Row}
import org.apache.spark.sql.catalyst.expressions.ScalarSubquery
import org.apache.spark.sql.catalyst.plans.logical.{LocalRelation, LogicalPlan, Window}
import org.apache.spark.sql.classic.ClassicDataset
import org.apache.spark.sql.test.SharedSparkSession

class RewriteSelfMaxScalarToDenseRankSuite extends QueryTest with SharedSparkSession {

  override protected def sparkConf: org.apache.spark.SparkConf = {
    val conf = super.sparkConf
    conf.getAll.collect { case (key, "null") => key }.foreach(conf.remove)
    conf
      .set("spark.master", "local[2]")
      .set("spark.sql.adaptive.enabled", "false")
      .set("spark.gluten.mpp.enabled", "true")
  }

  private val query =
    """
      |with revenue (supplier_no, total_revenue) as (
      |  select l_suppkey, sum(l_extendedprice * (1 - l_discount))
      |  from self_max_lineitem
      |  group by l_suppkey
      |)
      |select s_suppkey, s_name, total_revenue
      |from self_max_supplier, revenue
      |where s_suppkey = supplier_no
      |  and total_revenue = (select max(total_revenue) from revenue)
      |order by s_suppkey
      |""".stripMargin

  private def rewrittenPlan(): LogicalPlan = {
    RewriteSelfMaxScalarToDenseRank(spark).apply(spark.sql(query).queryExecution.optimizedPlan)
  }

  private def containsScalarSubquery(plan: LogicalPlan): Boolean = {
    plan.exists(node => node.expressions.exists(_.exists(_.isInstanceOf[ScalarSubquery])))
  }

  test("reuses the aggregate and preserves all maximum ties") {
    spark
      .sql("""
             |select * from values
             |  (cast(1 as bigint), 's1'),
             |  (cast(2 as bigint), 's2'),
             |  (cast(3 as bigint), 's3')
             |as self_max_supplier(s_suppkey, s_name)
             |""".stripMargin)
      .createOrReplaceTempView("self_max_supplier")
    spark
      .sql("""
             |select * from values
             |  (cast(1 as bigint), 10.0D, 0.0D),
             |  (cast(2 as bigint), 10.0D, 0.0D),
             |  (cast(3 as bigint), 5.0D, 0.0D)
             |as self_max_lineitem(l_suppkey, l_extendedprice, l_discount)
             |""".stripMargin)
      .createOrReplaceTempView("self_max_lineitem")

    val rewritten = rewrittenPlan()
    checkAnswer(
      ClassicDataset.ofRows(spark, rewritten),
      Seq(Row(1L, "s1", 10.0d), Row(2L, "s2", 10.0d)))
    assert(rewritten.exists(_.isInstanceOf[Window]), rewritten.treeString)
    assert(!containsScalarSubquery(rewritten), rewritten.treeString)
  }

  test("a null grouping key can win scalar MAX but remains excluded outside the aggregate") {
    spark
      .sql("""
             |select * from values
             |  (cast(1 as bigint), 's1'),
             |  (cast(2 as bigint), 's2')
             |as self_max_supplier(s_suppkey, s_name)
             |""".stripMargin)
      .createOrReplaceTempView("self_max_supplier")
    spark
      .sql("""
             |select * from values
             |  (cast(1 as bigint), 10.0D, 0.0D),
             |  (cast(2 as bigint), 5.0D, 0.0D),
             |  (cast(null as bigint), 20.0D, 0.0D)
             |as self_max_lineitem(l_suppkey, l_extendedprice, l_discount)
             |""".stripMargin)
      .createOrReplaceTempView("self_max_lineitem")

    val original = spark.sql(query)
    val rewritten = ClassicDataset.ofRows(spark, rewrittenPlan())
    checkAnswer(original, Seq.empty)
    checkAnswer(rewritten, Seq.empty)
  }

  test("does not introduce a global dense-rank window on a streaming plan") {
    spark
      .sql("""
             |select * from values
             |  (cast(1 as bigint), 's1'),
             |  (cast(2 as bigint), 's2')
             |as self_max_supplier(s_suppkey, s_name)
             |""".stripMargin)
      .createOrReplaceTempView("self_max_supplier")
    spark
      .sql("""
             |select * from values
             |  (cast(1 as bigint), 10.0D, 0.0D),
             |  (cast(2 as bigint), 5.0D, 0.0D)
             |as self_max_lineitem(l_suppkey, l_extendedprice, l_discount)
             |""".stripMargin)
      .createOrReplaceTempView("self_max_lineitem")

    val batchPlan = spark.sql(query).queryExecution.optimizedPlan
    val streamingPlan = batchPlan.transformDown {
      case relation: LocalRelation => relation.copy(isStreaming = true)
    }
    assert(streamingPlan.isStreaming)
    val unchanged = RewriteSelfMaxScalarToDenseRank(spark).apply(streamingPlan)
    assert(unchanged.fastEquals(streamingPlan), s"Streaming plan must remain unchanged:\n$unchanged")
    assert(containsScalarSubquery(unchanged), unchanged.treeString)
    assert(!unchanged.exists(_.isInstanceOf[Window]), unchanged.treeString)
  }
}
