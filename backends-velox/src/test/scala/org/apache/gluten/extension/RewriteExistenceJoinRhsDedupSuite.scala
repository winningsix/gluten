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

import org.apache.gluten.config.GlutenConfig
import org.apache.gluten.extension.columnar.RewriteExistenceJoinRhsDedup

import org.apache.spark.sql.{QueryTest, Row}
import org.apache.spark.sql.catalyst.plans.{LeftAnti, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Join, LogicalPlan}
import org.apache.spark.sql.classic.ClassicDataset
import org.apache.spark.sql.test.SharedSparkSession

class RewriteExistenceJoinRhsDedupSuite extends QueryTest with SharedSparkSession {
  import testImplicits._

  override protected def sparkConf: org.apache.spark.SparkConf = {
    val conf = super.sparkConf
    conf.getAll.collect { case (key, "null") => key }.foreach(conf.remove)
    conf
      .set("spark.master", "local[2]")
      .set("spark.executor.cores", "2")
      .set("spark.executor.memory", "1g")
      .set("spark.executor.memoryOverhead", "512m")
      .set("spark.sql.adaptive.enabled", "false")
      .set(GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key, "true")
  }

  private def createTestViews(): Unit = {
    Seq((1, 10, "a"), (1, 20, "b"), (2, 10, "c"), (3, 10, "d"))
      .toDF("id", "supp", "payload")
      .createOrReplaceTempView("exist_l")

    Seq((1, 20, "Y"), (1, 20, "Y"), (1, 30, "Y"), (2, 10, "Y"), (3, 20, "N"))
      .toDF("id", "supp", "flag")
      .createOrReplaceTempView("exist_r")
  }

  private def hasOptimizedExistenceAggregate(plan: LogicalPlan): Boolean = {
    plan.collect {
      case join @ Join(_, _, LeftSemi | LeftAnti, _, _)
          if join.right.exists {
            case Aggregate(groupingExpressions, aggregateExpressions, _) =>
              groupingExpressions.length == 1 && aggregateExpressions.length == 3
            case _ => false
          } =>
        true
    }.nonEmpty
  }

  private def optimizedExistenceAggregateCount(plan: LogicalPlan): Int = {
    plan.collect {
      case join @ Join(_, _, LeftSemi | LeftAnti, _, _)
          if join.right.exists {
            case Aggregate(groupingExpressions, aggregateExpressions, _) =>
              groupingExpressions.length == 1 && aggregateExpressions.length == 3
            case _ => false
          } =>
        true
    }.size
  }

  private def rewrite(plan: LogicalPlan): LogicalPlan = {
    RewriteExistenceJoinRhsDedup(spark).apply(plan)
  }

  test("deduplicates RHS of correlated EXISTS after Spark decorrelation") {
    createTestViews()
    val df = spark.sql("""
        |select id, supp
        |from exist_l l
        |where exists (
        |  select 1
        |  from exist_r r
        |  where r.id = l.id
        |    and r.supp <> l.supp
        |    and r.flag = 'Y'
        |    and l.payload is not null
        |)
        |order by id, supp
        |""".stripMargin)
    val rewrittenPlan = rewrite(df.queryExecution.optimizedPlan)
    val rewrittenDf = ClassicDataset.ofRows(spark, rewrittenPlan)

    checkAnswer(rewrittenDf, Seq(Row(1, 10), Row(1, 20)))
    assert(
      hasOptimizedExistenceAggregate(rewrittenPlan),
      s"Expected RHS summary Aggregate below existence join:\n${rewrittenPlan.treeString}")
    val repeatedRewritePlan = rewrite(rewrittenPlan)
    assert(
      optimizedExistenceAggregateCount(repeatedRewritePlan) ==
        optimizedExistenceAggregateCount(rewrittenPlan),
      s"Expected existence summary rewrite to be idempotent:\n${repeatedRewritePlan.treeString}")
  }

  test("rewrites correlated EXISTS before Spark predicate subquery rewrite") {
    createTestViews()
    val df = spark.sql("""
        |select id, supp
        |from exist_l l
        |where exists (
        |  select 1
        |  from exist_r r
        |  where r.id = l.id
        |    and r.supp <> l.supp
        |    and r.flag = 'Y'
        |)
        |order by id, supp
        |""".stripMargin)
    val rewrittenAnalyzedPlan = rewrite(df.queryExecution.analyzed)
    val rewrittenDf = ClassicDataset.ofRows(spark, rewrittenAnalyzedPlan)
    val optimizedPlan = rewrittenDf.queryExecution.optimizedPlan

    checkAnswer(rewrittenDf, Seq(Row(1, 10), Row(1, 20)))
    assert(
      rewrittenAnalyzedPlan.treeString.contains("_existence_min_supp"),
      s"Expected analyzed EXISTS subquery to be summarized:\n${rewrittenAnalyzedPlan.treeString}")
    assert(
      hasOptimizedExistenceAggregate(optimizedPlan),
      s"Expected Spark predicate subquery rewrite to preserve RHS summary:\n$optimizedPlan")
  }

  test("summarized NOT EXISTS remains semantically equivalent when disabled") {
    createTestViews()
    val sql =
      """
        |select id, supp
        |from exist_l l
        |where not exists (
        |  select 1
        |  from exist_r r
        |  where r.id = l.id
        |    and r.supp <> l.supp
        |    and r.flag = 'Y'
        |)
        |order by id, supp
        |""".stripMargin

    val original = spark.sql(sql)
    val rewrittenPlan = rewrite(original.queryExecution.optimizedPlan)
    val rewrittenDf = ClassicDataset.ofRows(spark, rewrittenPlan)
    var disabledPlan = original.queryExecution.optimizedPlan
    withSQLConf(GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key -> "false") {
      disabledPlan = rewrite(spark.sql(sql).queryExecution.optimizedPlan)
    }

    checkAnswer(rewrittenDf, ClassicDataset.ofRows(spark, disabledPlan))
    assert(
      hasOptimizedExistenceAggregate(rewrittenPlan),
      s"Expected RHS summary Aggregate below NOT EXISTS anti join:\n${rewrittenPlan.treeString}")
    assert(
      !hasOptimizedExistenceAggregate(disabledPlan),
      s"Expected disabled rule to leave the existence join unchanged:\n${disabledPlan.treeString}")
  }
}
