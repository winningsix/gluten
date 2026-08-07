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
import org.apache.spark.sql.catalyst.expressions.{Attribute, LessThan, Literal, Rand}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftOuter, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, BROADCAST, Filter, HintInfo, Join}
import org.apache.spark.sql.catalyst.plans.logical.{JoinHint, LogicalPlan}
import org.apache.spark.sql.classic.ClassicDataset
import org.apache.spark.sql.test.SharedSparkSession

class PushSelectiveDimensionFilterIntoAggregateSuite extends QueryTest with SharedSparkSession {

  private val enabledKey = "spark.gluten.mpp.pushSelectiveDimensionFilterIntoAggregate"
  private val ratioKey =
    "spark.gluten.mpp.pushSelectiveDimensionFilterIntoAggregate.maxDimensionScanRatio"
  private val minImprovementRatioKey =
    "spark.gluten.mpp.pushSelectiveDimensionFilterIntoAggregate.minImprovementRatio"
  private val maxBuildBytesKey = "spark.gluten.mpp.factProbeBroadcastHint.maxBuildBytes"
  private val partitionsKey = "spark.gluten.mpp.multiExecutor.numPartitions"

  override protected def sparkConf: org.apache.spark.SparkConf = {
    // The backend Maven profile forwards optional spark.* properties to the forked JVM.  Maven
    // renders unset values as the literal string "null", which Spark then tries to parse as typed
    // configuration during SparkContext/SQLConf initialization.  Ignore only those absent values;
    // explicitly supplied test properties remain intact.
    sys.props.iterator
      .collect { case (key, value) if key.startsWith("spark.") && value == "null" => key }
      .toSeq
      .foreach(sys.props.remove)
    super.sparkConf
      .set("spark.master", "local[2]")
      .set("spark.executor.instances", "1")
      .set("spark.executor.cores", "2")
      .set("spark.executor.memory", "1g")
      .set("spark.executor.memoryOverhead", "512m")
      .set("spark.driver.maxResultSize", "1g")
      .set("spark.sql.files.maxPartitionBytes", "128m")
      .set("spark.sql.files.openCostInBytes", "4m")
      .set("spark.sql.files.minPartitionNum", "1")
      .set("spark.sql.shuffle.partitions", "2")
      .set("spark.sql.adaptive.enabled", "false")
      .set("spark.gluten.mpp.enabled", "true")
      // LocalRelation statistics are not representative of parquet scan widths.  Production uses
      // the conservative 0.25 default; these tests separately exercise the cost guard.
      .set(ratioKey, "100")
  }

  private def q17Query(
      dimensionPredicate: String = "p_brand = 'Brand#23' and p_container = 'MED BOX'",
      aggregateTable: String = "q17_lineitem",
      dimensionTable: String = "q17_part"): String = {
    s"""
       |select sum(l_extendedprice) / 7.0 as avg_yearly
       |from q17_lineitem, $dimensionTable
       |where p_partkey = l_partkey
       |  and ($dimensionPredicate)
       |  and l_quantity < (
       |    select 0.2 * avg(l_quantity)
       |    from $aggregateTable
       |    where l_partkey = p_partkey)
       |""".stripMargin
  }

  private val query = q17Query()

  override def beforeAll(): Unit = {
    super.beforeAll()
    org.apache.logging.log4j.core.config.Configurator.setLevel(
      classOf[PushSelectiveDimensionFilterIntoAggregate].getName,
      org.apache.logging.log4j.Level.INFO)
    spark
      .sql("""
             |select * from values
             |  (cast(1 as bigint), 'Brand#23', 'MED BOX'),
             |  (cast(1 as bigint), 'Brand#23', 'MED BOX'),
             |  (cast(2 as bigint), 'Brand#11', 'MED BOX'),
             |  (cast(3 as bigint), 'Brand#23', 'MED BOX'),
             |  (cast(null as bigint), 'Brand#23', 'MED BOX')
             |as q17_part(p_partkey, p_brand, p_container)
             |""".stripMargin)
      .createOrReplaceTempView("q17_part")
    spark
      .sql("""
             |select * from values
             |  (cast(1 as bigint), 1.0D, 70.0D),
             |  (cast(1 as bigint), 10.0D, 700.0D),
             |  (cast(2 as bigint), 1.0D, 140.0D),
             |  (cast(2 as bigint), 10.0D, 1400.0D),
             |  (cast(3 as bigint), 5.0D, 210.0D),
             |  (cast(null as bigint), 1.0D, 280.0D)
             |as q17_lineitem(l_partkey, l_quantity, l_extendedprice)
             |""".stripMargin)
      .createOrReplaceTempView("q17_lineitem")
    spark
      .sql("""
             |select * from values
             |  (cast(1 as bigint), 1.0D, 70.0D),
             |  (cast(1 as bigint), 9.0D, 630.0D),
             |  (cast(2 as bigint), 2.0D, 280.0D)
             |as q17_alt_lineitem(l_partkey, l_quantity, l_extendedprice)
             |""".stripMargin)
      .createOrReplaceTempView("q17_alt_lineitem")
    spark
      .sql("""
             |select * from values
             |  (cast(99 as bigint), 'Brand#11', 'SMALL BOX')
             |as q17_part_empty_filtered(p_partkey, p_brand, p_container)
             |""".stripMargin)
      .createOrReplaceTempView("q17_part_empty_filtered")
    spark
      .sql("""
             |select * from values
             |  (cast(99 as bigint), 'Brand#23', 'MED BOX')
             |as q17_part_no_match(p_partkey, p_brand, p_container)
             |""".stripMargin)
      .createOrReplaceTempView("q17_part_no_match")
    spark
      .sql("""
             |select * from values
             |  (cast(1 as bigint), 'Brand#23', 'MED BOX'),
             |  (cast(2 as bigint), 'Brand#23', 'MED BOX'),
             |  (cast(3 as bigint), 'Brand#23', 'MED BOX')
             |as q17_part_all_match(p_partkey, p_brand, p_container)
             |""".stripMargin)
      .createOrReplaceTempView("q17_part_all_match")
  }

  private def originalPlan(sqlText: String = query): LogicalPlan = {
    // Once the rule self-registers, subsequent QueryExecutions in this suite would otherwise be
    // rewritten before a negative test can mutate the candidate shape.
    var original: LogicalPlan = null
    withSQLConf(enabledKey -> "false") {
      original = spark.sql(sqlText).queryExecution.optimizedPlan
    }
    original
  }

  private def applyRule(plan: LogicalPlan): LogicalPlan = {
    PushSelectiveDimensionFilterIntoAggregate(spark).apply(plan)
  }

  private def rewrittenPlan(sqlText: String = query): LogicalPlan = {
    applyRule(originalPlan(sqlText))
  }

  private def semiBelowGroupedAggregate(plan: LogicalPlan): Boolean = {
    plan.exists {
      case aggregate: Aggregate if aggregate.groupingExpressions.size == 1 =>
        aggregate.child match {
          case Join(_, _, LeftSemi, Some(_), _) => true
          case _ => false
        }
      case _ => false
    }
  }

  private def semiCount(plan: LogicalPlan): Int = {
    plan.collect { case Join(_, _, LeftSemi, Some(_), _) => 1 }.size
  }

  private def insertedSemi(plan: LogicalPlan): Option[Join] = {
    plan.collectFirst { case join @ Join(_, _, LeftSemi, Some(_), _) => join }
  }

  private def assertBroadcastSemi(plan: LogicalPlan): Unit = {
    val semi = insertedSemi(plan).getOrElse(fail(plan.treeString))
    assert(
      semi.hint.rightHint.flatMap(_.strategy).contains(BROADCAST),
      s"new semi must explicitly broadcast its right side:\n${plan.treeString}")
    assert(semi.hint.leftHint.isEmpty, plan.treeString)
  }

  private def containsGroupedAggregate(plan: LogicalPlan): Boolean = {
    plan.exists {
      case aggregate: Aggregate if aggregate.groupingExpressions.nonEmpty => true
      case _ => false
    }
  }

  private def mutateCandidateTopJoin(
      plan: LogicalPlan,
      rewrite: Join => LogicalPlan): LogicalPlan = {
    var rewritten = false
    plan.transformDown {
      case join @ Join(left, right, Inner, Some(_), _)
          if !rewritten && (containsGroupedAggregate(left) ^ containsGroupedAggregate(right)) =>
        rewritten = true
        rewrite(join)
    }
  }

  test("pushes a selective dimension into the correlated aggregate and preserves semantics") {
    val original = ClassicDataset.ofRows(spark, originalPlan())
    val rewrittenPlanValue = rewrittenPlan()
    val rewritten = ClassicDataset.ofRows(spark, rewrittenPlanValue)

    // Duplicate dimension keys must still duplicate the final outer result, not the aggregate
    // input.  The left-semi filter is what preserves that distinction.
    checkAnswer(original, Seq(Row(20.0d)))
    checkAnswer(rewritten, Seq(Row(20.0d)))
    assert(semiBelowGroupedAggregate(rewrittenPlanValue), rewrittenPlanValue.treeString)
    assertBroadcastSemi(rewrittenPlanValue)
  }

  test("does not rewrite when disabled") {
    withSQLConf(enabledKey -> "false") {
      val plan = rewrittenPlan()
      assert(!semiBelowGroupedAggregate(plan), plan.treeString)
    }
  }

  test("relative cost guard rejects an expensive duplicated dimension scan") {
    withSQLConf(ratioKey -> "0.000000000001") {
      val plan = rewrittenPlan()
      assert(!semiBelowGroupedAggregate(plan), plan.treeString)
    }
  }

  test("broadcast guard rejects a dimension above or disabled by the Spark threshold") {
    Seq("1", "-1").foreach {
      threshold =>
        withSQLConf("spark.sql.autoBroadcastJoinThreshold" -> threshold) {
          val plan = rewrittenPlan()
          assert(!semiBelowGroupedAggregate(plan), s"threshold=$threshold\n${plan.treeString}")
        }
    }
  }

  test("cost gate can extend the broadcast threshold for two independent literal filters") {
    withSQLConf(
      "spark.sql.autoBroadcastJoinThreshold" -> "1",
      maxBuildBytesKey -> "1g",
      partitionsKey -> "2",
      minImprovementRatioKey -> "1.0") {
      val plan = rewrittenPlan(q17Query(dimensionTable = "q17_part_no_match"))
      assert(semiBelowGroupedAggregate(plan), plan.treeString)
      assertBroadcastSemi(plan)
    }
  }

  test("cost gate above the Spark threshold requires two independent literal filters") {
    withSQLConf(
      "spark.sql.autoBroadcastJoinThreshold" -> "1",
      maxBuildBytesKey -> "1g",
      partitionsKey -> "2",
      minImprovementRatioKey -> "1.0") {
      val plan = rewrittenPlan(q17Query(dimensionPredicate = "p_brand = 'Brand#23'"))
      assert(!semiBelowGroupedAggregate(plan), plan.treeString)
    }
  }

  test("cost gate respects the explicit FLUX build ceiling") {
    withSQLConf(
      "spark.sql.autoBroadcastJoinThreshold" -> "1",
      maxBuildBytesKey -> "2",
      partitionsKey -> "2",
      minImprovementRatioKey -> "1.0") {
      val plan = rewrittenPlan()
      assert(!semiBelowGroupedAggregate(plan), plan.treeString)
    }
  }

  test("relative cost guard rejects non-finite ratios") {
    Seq("NaN", "Infinity", "-Infinity").foreach {
      ratio =>
        withSQLConf(ratioKey -> ratio) {
          val plan = rewrittenPlan()
          assert(!semiBelowGroupedAggregate(plan), s"ratio=$ratio\n${plan.treeString}")
        }
    }
  }

  test("does not rewrite a non-inner top join") {
    val nonInner = mutateCandidateTopJoin(originalPlan(), join => join.copy(joinType = LeftOuter))
    val rewritten = applyRule(nonInner)
    assert(!semiBelowGroupedAggregate(rewritten), rewritten.treeString)
  }

  test("does not cross a nondeterministic aggregate-side wrapper") {
    val nondeterministic = mutateCandidateTopJoin(
      originalPlan(),
      join => {
        val predicate = LessThan(Rand(0L), Literal(0.5d))
        if (containsGroupedAggregate(join.left)) {
          join.copy(left = Filter(predicate, join.left))
        } else {
          join.copy(right = Filter(predicate, join.right))
        }
      }
    )
    val rewritten = applyRule(nondeterministic)
    assert(!semiBelowGroupedAggregate(rewritten), rewritten.treeString)
  }

  test("accepts an existing broadcast hint on the dimension side") {
    var hinted = false
    val withHint = originalPlan().transformUp {
      case join @ Join(left, right, Inner, Some(_), JoinHint.NONE)
          if !hinted && !containsGroupedAggregate(left) && !containsGroupedAggregate(right) &&
            (left.output.exists(_.name == "p_partkey") ||
              right.output.exists(_.name == "p_partkey")) =>
        hinted = true
        val broadcast = HintInfo(strategy = Some(BROADCAST))
        if (right.output.exists(_.name == "p_partkey")) {
          join.copy(hint = JoinHint(None, Some(broadcast)))
        } else {
          join.copy(hint = JoinHint(Some(broadcast), None))
        }
    }
    assert(hinted, withHint.treeString)
    val rewritten = applyRule(withHint)
    assert(semiBelowGroupedAggregate(rewritten), rewritten.treeString)
    assertBroadcastSemi(rewritten)
  }

  test("preserves an existing broadcast hint on the top aggregate side") {
    val broadcast = HintInfo(strategy = Some(BROADCAST))
    val withHint = mutateCandidateTopJoin(
      originalPlan(),
      join =>
        if (containsGroupedAggregate(join.left)) {
          join.copy(hint = JoinHint(Some(broadcast), None))
        } else {
          join.copy(hint = JoinHint(None, Some(broadcast)))
        }
    )
    val rewritten = applyRule(withHint)
    assert(semiBelowGroupedAggregate(rewritten), rewritten.treeString)
    assertBroadcastSemi(rewritten)
  }

  test("preserves an existing broadcast hint on the original fact side") {
    var hinted = false
    val withHint = originalPlan().transformUp {
      case join @ Join(left, right, Inner, Some(_), JoinHint.NONE)
          if !hinted && !containsGroupedAggregate(left) && !containsGroupedAggregate(right) &&
            (left.output.exists(_.name == "p_partkey") ||
              right.output.exists(_.name == "p_partkey")) =>
        hinted = true
        val broadcast = HintInfo(strategy = Some(BROADCAST))
        if (right.output.exists(_.name == "p_partkey")) {
          join.copy(hint = JoinHint(Some(broadcast), None))
        } else {
          join.copy(hint = JoinHint(None, Some(broadcast)))
        }
    }
    assert(hinted, withHint.treeString)
    val rewritten = applyRule(withHint)
    assert(semiBelowGroupedAggregate(rewritten), rewritten.treeString)
    assertBroadcastSemi(rewritten)
  }

  test("does not rewrite a dimension without a literal equality filter") {
    val rewritten = rewrittenPlan(q17Query(dimensionPredicate = "p_brand = p_container"))
    assert(!semiBelowGroupedAggregate(rewritten), rewritten.treeString)
  }

  test("does not rewrite when the outer fact and aggregate fact are different relations") {
    val rewritten = rewrittenPlan(q17Query(aggregateTable = "q17_alt_lineitem"))
    assert(!semiBelowGroupedAggregate(rewritten), rewritten.treeString)
  }

  test("does not rewrite a multi-key aggregate") {
    var changed = false
    val multiKey = originalPlan().transformUp {
      case aggregate: Aggregate if !changed =>
        aggregate.groupingExpressions match {
          case Seq(grouping: Attribute) =>
            aggregate.child.output.find(!_.semanticEquals(grouping)) match {
              case Some(secondGrouping) =>
                changed = true
                aggregate.copy(
                  groupingExpressions = Seq(grouping, secondGrouping),
                  aggregateExpressions = aggregate.aggregateExpressions :+ secondGrouping)
              case None => aggregate
            }
          case _ => aggregate
        }
    }
    assert(changed, multiKey.treeString)
    val rewritten = applyRule(multiKey)
    assert(!semiBelowGroupedAggregate(rewritten), rewritten.treeString)
  }

  test("filtered dimension can be empty without changing the scalar result") {
    val rewritten = rewrittenPlan(q17Query(dimensionTable = "q17_part_empty_filtered"))
    assertBroadcastSemi(rewritten)
    checkAnswer(ClassicDataset.ofRows(spark, rewritten), Seq(Row(null)))
  }

  test("filtered dimension can contain only keys absent from the fact") {
    val rewritten = rewrittenPlan(q17Query(dimensionTable = "q17_part_no_match"))
    assertBroadcastSemi(rewritten)
    checkAnswer(ClassicDataset.ofRows(spark, rewritten), Seq(Row(null)))
  }

  test("filtered dimension can contain every non-null fact key") {
    val rewritten = rewrittenPlan(q17Query(dimensionTable = "q17_part_all_match"))
    assertBroadcastSemi(rewritten)
    checkAnswer(ClassicDataset.ofRows(spark, rewritten), Seq(Row(30.0d)))
  }

  test("fixed-point application and self-registration are idempotent") {
    val original = originalPlan()
    val experimental = spark.experimental
    experimental.synchronized {
      experimental.extraOptimizations = experimental.extraOptimizations.filterNot(
        _.isInstanceOf[PushSelectiveDimensionFilterIntoAggregate])
    }

    // Construction must register the post-subquery pass before the first query is optimized.
    PushSelectiveDimensionFilterIntoAggregate(spark)
    assert(
      spark.experimental.extraOptimizations.count(
        _.isInstanceOf[PushSelectiveDimensionFilterIntoAggregate]) == 1)

    // This is the first optimized query after construction. The rule must already be present in
    // the final optimizer batch so it can see the decorrelated aggregate join.
    val firstOptimizedQuery = spark.sql(query).queryExecution.optimizedPlan
    assertBroadcastSemi(firstOptimizedQuery)

    val once = applyRule(original)
    val twice = applyRule(once)
    assert(semiCount(once) == 1, once.treeString)
    assert(semiCount(twice) == 1, twice.treeString)
    assert(once.sameResult(twice), s"once:\n${once.treeString}\ntwice:\n${twice.treeString}")

    // Constructing additional rule instances must not append duplicate optimizer instances to
    // the same SparkSession.
    PushSelectiveDimensionFilterIntoAggregate(spark).apply(original)
    PushSelectiveDimensionFilterIntoAggregate(spark).apply(original)
    val registrations = spark.experimental.extraOptimizations.count {
      _.isInstanceOf[PushSelectiveDimensionFilterIntoAggregate]
    }
    assert(registrations == 1, spark.experimental.extraOptimizations.mkString("\n"))
  }
}
