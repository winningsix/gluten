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
import org.apache.spark.sql.catalyst.expressions.{Alias, And, AttributeReference, EqualTo}
import org.apache.spark.sql.catalyst.expressions.IsNotNull
import org.apache.spark.sql.catalyst.expressions.aggregate.Min
import org.apache.spark.sql.catalyst.plans.{Inner, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, JoinHint, LocalRelation}
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Project}
import org.apache.spark.sql.classic.ClassicDataset
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.LongType

class RewriteLargeLeftSemiToInnerDistinctSuite extends QueryTest with SharedSparkSession {

  private val enabledKey = "spark.gluten.mpp.rewriteLargeLeftSemiToInnerDistinct"
  private val multiplierKey = "spark.gluten.mpp.leftSemiDistinctThresholdMultiplier"

  override protected def sparkConf: org.apache.spark.SparkConf = {
    val conf = super.sparkConf
    conf.getAll.collect { case (key, "null") => key }.foreach(conf.remove)
    conf
      .set("spark.master", "local[2]")
      .set("spark.executor.cores", "2")
      .set("spark.executor.memory", "1g")
      .set("spark.executor.memoryOverhead", "512m")
      .set("spark.sql.adaptive.enabled", "false")
  }

  private def withRule(body: RewriteLargeLeftSemiToInnerDistinct => LogicalPlan): LogicalPlan = {
    val original = spark.experimental.extraOptimizations
    var result: LogicalPlan = null
    try {
      withSQLConf(
        enabledKey -> "true",
        multiplierKey -> "2",
        "spark.sql.autoBroadcastJoinThreshold" -> "-1") {
        result = body(RewriteLargeLeftSemiToInnerDistinct(spark))
      }
      result
    } finally {
      spark.experimental.extraOptimizations = original
    }
  }

  private def joins(
      plan: LogicalPlan,
      joinType: org.apache.spark.sql.catalyst.plans.JoinType): Int =
    plan.collect { case Join(_, _, current, _, _) if current == joinType => true }.size

  test("already unique paired state becomes inner without another distinct aggregate") {
    val leftOrder = AttributeReference("l_orderkey", LongType)()
    val leftSupp = AttributeReference("l_suppkey", LongType)()
    val stateOrder = AttributeReference("state_orderkey", LongType)()
    val stateSupp = AttributeReference("state_suppkey", LongType)()
    val left = LocalRelation(Seq(leftOrder, leftSupp))
    val stateInput = LocalRelation(Seq(stateOrder, stateSupp))
    val delayedMin = Alias(Min(stateSupp).toAggregateExpression(), "delayed_min")()
    val state = Aggregate(Seq(stateOrder), Seq(stateOrder, delayedMin), stateInput)
    val eligibleState = Project(
      Seq(stateOrder, delayedMin.toAttribute),
      Filter(IsNotNull(delayedMin.toAttribute), state))
    val condition = And(EqualTo(leftOrder, stateOrder), EqualTo(leftSupp, delayedMin.toAttribute))
    val semi = Join(left, eligibleState, LeftSemi, Some(condition), JoinHint.NONE)

    val rewritten = withRule(_.apply(semi))

    assert(rewritten.output == left.output)
    assert(joins(rewritten, LeftSemi) == 0)
    assert(joins(rewritten, Inner) == 1)
    assert(
      rewritten.collect { case aggregate: Aggregate => aggregate }.size == 1,
      s"the paired state aggregate must not be wrapped in another DISTINCT:\n$rewritten")
  }

  test("non-unique existence RHS still receives distinct before inner conversion") {
    val leftKey = AttributeReference("left_key", LongType)()
    val rightKey = AttributeReference("right_key", LongType)()
    val left = LocalRelation(Seq(leftKey))
    val right = LocalRelation(Seq(rightKey))
    val semi = Join(left, right, LeftSemi, Some(EqualTo(leftKey, rightKey)), JoinHint.NONE)

    val rewritten = withRule(_.apply(semi))

    assert(rewritten.output == left.output)
    assert(joins(rewritten, Inner) == 1)
    assert(
      rewritten.collect { case aggregate: Aggregate => aggregate }.size == 1,
      s"ordinary non-unique RHS still needs DISTINCT:\n$rewritten")
  }

  test("non-unique rewrite preserves LeftSemi bag and NULL semantics") {
    import testImplicits._

    withTempView("semi_left", "semi_right") {
      Seq[(java.lang.Long, String)](
        (1L, "one-a"),
        (1L, "one-b"),
        (2L, "two"),
        (3L, "three"),
        (null, "null-key")
      ).toDF("left_key", "payload").createOrReplaceTempView("semi_left")
      Seq[java.lang.Long](1L, 1L, 3L, null)
        .toDF("right_key")
        .createOrReplaceTempView("semi_right")

      val sqlText =
        """
          |SELECT l.left_key, l.payload
          |FROM semi_left l LEFT SEMI JOIN semi_right r
          |  ON l.left_key = r.right_key
          |ORDER BY l.payload
          |""".stripMargin
      var baselineRows = Seq.empty[Row]
      var originalPlan: LogicalPlan = null
      withSQLConf(enabledKey -> "false") {
        val baseline = spark.sql(sqlText)
        baselineRows = baseline.collect().toSeq
        originalPlan = baseline.queryExecution.optimizedPlan
      }

      val rewritten = withRule(_.apply(originalPlan))

      assert(rewritten.output == originalPlan.output)
      checkAnswer(ClassicDataset.ofRows(spark, rewritten), baselineRows)
      assert(baselineRows == Seq(Row(1L, "one-a"), Row(1L, "one-b"), Row(3L, "three")))
      val rewrittenAgain = withRule(_.apply(rewritten))
      assert(rewrittenAgain.fastEquals(rewritten), rewrittenAgain.treeString)
    }
  }

  test("grouping wider than the equi keys is not treated as already unique") {
    val leftOrder = AttributeReference("left_orderkey", LongType)()
    val stateOrder = AttributeReference("state_orderkey", LongType)()
    val stateSupp = AttributeReference("state_suppkey", LongType)()
    val left = LocalRelation(Seq(leftOrder))
    val stateInput = LocalRelation(Seq(stateOrder, stateSupp))
    val state = Aggregate(Seq(stateOrder, stateSupp), Seq(stateOrder, stateSupp), stateInput)
    val semi = Join(left, state, LeftSemi, Some(EqualTo(leftOrder, stateOrder)), JoinHint.NONE)

    val rewritten = withRule(_.apply(semi))

    assert(rewritten.output == left.output)
    assert(
      rewritten.collect { case aggregate: Aggregate => aggregate }.size == 2,
      s"GROUP BY(orderkey, suppkey) is not unique for a join on orderkey alone:\n$rewritten")
  }
}
