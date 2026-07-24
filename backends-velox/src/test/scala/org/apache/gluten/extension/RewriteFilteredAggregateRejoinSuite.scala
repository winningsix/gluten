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
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.classic.ClassicDataset
import org.apache.spark.sql.test.SharedSparkSession

class RewriteFilteredAggregateRejoinSuite extends QueryTest with SharedSparkSession {
  import testImplicits._

  private val enabledKey = "spark.gluten.sql.optimizer.reuseFilteredAggregate.enabled"

  override protected def sparkConf: org.apache.spark.SparkConf = {
    val conf = super.sparkConf
    conf.getAll.collect { case (key, "null") => key }.foreach(conf.remove)
    conf
      .set("spark.master", "local[2]")
      .set("spark.sql.adaptive.enabled", "false")
  }

  private def createTestViews(): Unit = {
    Seq((1L, 10L), (2L, 20L), (3L, 30L))
      .toDF("o_orderkey", "o_custkey")
      .createOrReplaceTempView("reuse_orders")
    Seq((10L, "alice"), (20L, "bob"), (30L, "carol"))
      .toDF("c_custkey", "c_name")
      .createOrReplaceTempView("reuse_customer")
    Seq(
      (1L, 200.0),
      (1L, 150.0),
      (2L, 100.0),
      (2L, 50.0),
      (3L, 301.0),
      (3L, null.asInstanceOf[java.lang.Double]))
      .toDF("l_orderkey", "l_quantity")
      .createOrReplaceTempView("reuse_lineitem")
  }

  private def query(extraInnerPredicate: String = ""): String = {
    s"""
       |select c_name, c_custkey, o_orderkey, sum(l_quantity)
       |from reuse_customer, reuse_orders, reuse_lineitem
       |where o_orderkey in (
       |  select l_orderkey
       |  from reuse_lineitem
       |  ${if (extraInnerPredicate.isEmpty) "" else s"where $extraInnerPredicate"}
       |  group by l_orderkey
       |  having sum(l_quantity) > 300
       |)
       |and c_custkey = o_custkey
       |and o_orderkey = l_orderkey
       |group by c_name, c_custkey, o_orderkey
       |order by o_orderkey
       |""".stripMargin
  }

  private def optimizedWithoutRule(sqlText: String): LogicalPlan = {
    withSQLConf(enabledKey -> "false") {
      spark.sql(sqlText).queryExecution.optimizedPlan
    }
  }

  private def factLeafCount(plan: LogicalPlan): Int = {
    plan.collect {
      case leaf
          if leaf.children.isEmpty &&
            leaf.output.exists(_.name == "l_orderkey") &&
            leaf.output.exists(_.name == "l_quantity") =>
        leaf
    }.size
  }

  test("reuses the qualifying aggregate and preserves Q18 semantics") {
    createTestViews()
    val originalPlan = optimizedWithoutRule(query())
    val rewrittenPlan = withSQLConf(enabledKey -> "true") {
      RewriteFilteredAggregateRejoin(spark).apply(originalPlan)
    }

    checkAnswer(
      ClassicDataset.ofRows(spark, rewrittenPlan),
      Seq(Row("alice", 10L, 1L, 350.0), Row("carol", 30L, 3L, 301.0)))
    assert(factLeafCount(originalPlan) == 2, originalPlan.treeString)
    assert(factLeafCount(rewrittenPlan) == 1, rewrittenPlan.treeString)

    val rewrittenAgain = withSQLConf(enabledKey -> "true") {
      RewriteFilteredAggregateRejoin(spark).apply(rewrittenPlan)
    }
    assert(factLeafCount(rewrittenAgain) == 1, rewrittenAgain.treeString)
  }

  test("does not reuse an aggregate over a differently filtered fact branch") {
    createTestViews()
    val originalPlan = optimizedWithoutRule(query("l_quantity >= 100"))
    val rewrittenPlan = withSQLConf(enabledKey -> "true") {
      RewriteFilteredAggregateRejoin(spark).apply(originalPlan)
    }

    assert(factLeafCount(originalPlan) == 2, originalPlan.treeString)
    assert(factLeafCount(rewrittenPlan) == 2, rewrittenPlan.treeString)
    checkAnswer(
      ClassicDataset.ofRows(spark, rewrittenPlan),
      ClassicDataset.ofRows(spark, originalPlan))
  }
}
