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
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Window}
import org.apache.spark.sql.classic.ClassicDataset
import org.apache.spark.sql.test.SharedSparkSession

class ReuseGroupedAggregateForScalarMaxSuite extends QueryTest with SharedSparkSession {
  import testImplicits._

  private val enabledKey =
    "spark.gluten.sql.optimizer.reuseGroupedAggregateForScalarMax.enabled"

  override protected def sparkConf: org.apache.spark.SparkConf = {
    val conf = super.sparkConf
    conf.getAll.collect { case (key, "null") => key }.foreach(conf.remove)
    conf
      .set("spark.master", "local[2]")
      .set("spark.sql.adaptive.enabled", "false")
  }

  private def createTestView(): Unit = {
    Seq(
      (10L, 11.0d),
      (10L, 9.0d),
      (20L, 20.0d),
      (30L, 3.0d))
      .toDF("supplier_no", "amount")
      .createOrReplaceTempView("reuse_grouped_revenue")
  }

  private val query =
    """
      |with revenue0 as (
      |  select supplier_no, sum(amount) as total_revenue
      |  from reuse_grouped_revenue
      |  group by supplier_no)
      |select supplier_no, total_revenue
      |from revenue0
      |where total_revenue = (select max(total_revenue) from revenue0)
      |order by supplier_no
      |""".stripMargin

  private def revenueLeafCount(plan: LogicalPlan): Int = {
    plan.collect {
      case leaf
          if leaf.children.isEmpty &&
            leaf.output.exists(_.name == "supplier_no") &&
            leaf.output.exists(_.name == "amount") =>
        leaf
    }.size
  }

  test("reuses one grouped sum for scalar max and preserves tied maxima") {
    createTestView()
    val original = withSQLConf(enabledKey -> "false") {
      spark.sql(query).queryExecution.optimizedPlan
    }
    val rewritten = withSQLConf(enabledKey -> "true") {
      ReuseGroupedAggregateForScalarMax(spark).apply(original)
    }

    checkAnswer(
      ClassicDataset.ofRows(spark, rewritten),
      Seq(Row(10L, 20.0d), Row(20L, 20.0d)))
    assert(revenueLeafCount(original) == 2, original.treeString)
    assert(revenueLeafCount(rewritten) == 1, rewritten.treeString)
    assert(rewritten.exists(_.isInstanceOf[Window]), rewritten.treeString)

    val rewrittenAgain = withSQLConf(enabledKey -> "true") {
      ReuseGroupedAggregateForScalarMax(spark).apply(rewritten)
    }
    assert(revenueLeafCount(rewrittenAgain) == 1, rewrittenAgain.treeString)
  }
}
