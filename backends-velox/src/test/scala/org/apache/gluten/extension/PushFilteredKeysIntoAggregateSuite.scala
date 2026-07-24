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
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Join, LogicalPlan}
import org.apache.spark.sql.classic.ClassicDataset
import org.apache.spark.sql.test.SharedSparkSession

class PushFilteredKeysIntoAggregateSuite extends QueryTest with SharedSparkSession {
  import testImplicits._

  private val enabledKey =
    "spark.gluten.sql.optimizer.pushFilteredKeysIntoAggregate.enabled"

  override protected def sparkConf: org.apache.spark.SparkConf = {
    val conf = super.sparkConf
    conf.getAll.collect { case (key, "null") => key }.foreach(conf.remove)
    conf
      .set("spark.master", "local[2]")
      .set("spark.sql.adaptive.enabled", "false")
  }

  private def createTestViews(): Unit = {
    Seq(
      (1L, "Brand#23", "MED BOX"),
      // A duplicate matching dimension row must continue to duplicate outer results.
      (1L, "Brand#23", "MED BOX"),
      (2L, "Brand#23", "MED BOX"),
      (3L, "Brand#12", "SM BOX"))
      .toDF("p_partkey", "p_brand", "p_container")
      .createOrReplaceTempView("push_part")
    Seq(
      (1L, 10.0, 70.0),
      (1L, 100.0, 700.0),
      (2L, 5.0, 35.0),
      (2L, 15.0, 105.0),
      (3L, 1.0, 7.0),
      (3L, 100.0, 700.0))
      .toDF("l_partkey", "l_quantity", "l_extendedprice")
      .createOrReplaceTempView("push_lineitem")
  }

  private def query(selectivePredicate: String): String = {
    s"""
       |select sum(l_extendedprice) / 7.0 as avg_yearly
       |from push_lineitem, push_part
       |where p_partkey = l_partkey
       |${if (selectivePredicate.isEmpty) "" else s"and $selectivePredicate"}
       |and l_quantity < (
       |  select 0.2 * avg(inner_lineitem.l_quantity)
       |  from push_lineitem inner_lineitem
       |  where inner_lineitem.l_partkey = push_part.p_partkey
       |)
       |""".stripMargin
  }

  private def optimizedWithoutRule(sqlText: String): LogicalPlan = {
    withSQLConf(enabledKey -> "false") {
      spark.sql(sqlText).queryExecution.optimizedPlan
    }
  }

  private def dimensionLeafCount(plan: LogicalPlan): Int = {
    plan.collect {
      case leaf
          if leaf.children.isEmpty &&
            leaf.output.exists(_.name == "p_partkey") &&
            leaf.output.exists(_.name == "p_brand") =>
        leaf
    }.size
  }

  test("pushes distinct selective keys into a correlated aggregate") {
    createTestViews()
    val originalPlan =
      optimizedWithoutRule(query("p_brand = 'Brand#23' and p_container = 'MED BOX'"))
    val rewrittenPlan = withSQLConf(enabledKey -> "true") {
      PushFilteredKeysIntoAggregate(spark).apply(originalPlan)
    }

    checkAnswer(ClassicDataset.ofRows(spark, rewrittenPlan), Seq(Row(20.0)))
    assert(dimensionLeafCount(originalPlan) == 1, originalPlan.treeString)
    assert(dimensionLeafCount(rewrittenPlan) == 2, rewrittenPlan.treeString)
    assert(
      rewrittenPlan.collect {
        case aggregate: Aggregate if aggregate.child.isInstanceOf[Join] => aggregate
      }.nonEmpty,
      rewrittenPlan.treeString)

    val rewrittenAgain = withSQLConf(enabledKey -> "true") {
      PushFilteredKeysIntoAggregate(spark).apply(rewrittenPlan)
    }
    assert(dimensionLeafCount(rewrittenAgain) == 2, rewrittenAgain.treeString)
  }

  test("does not duplicate an unfiltered dimension below the aggregate") {
    createTestViews()
    val originalPlan = optimizedWithoutRule(query(""))
    val rewrittenPlan = withSQLConf(enabledKey -> "true") {
      PushFilteredKeysIntoAggregate(spark).apply(originalPlan)
    }

    assert(dimensionLeafCount(originalPlan) == 1, originalPlan.treeString)
    assert(dimensionLeafCount(rewrittenPlan) == 1, rewrittenPlan.treeString)
    checkAnswer(
      ClassicDataset.ofRows(spark, rewrittenPlan),
      ClassicDataset.ofRows(spark, originalPlan))
  }
}
