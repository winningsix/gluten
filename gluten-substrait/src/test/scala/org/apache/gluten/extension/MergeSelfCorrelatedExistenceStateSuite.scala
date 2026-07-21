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
import org.apache.gluten.extension.columnar.{MergeSelfCorrelatedExistenceState, RegisterMppExistencePostSubqueryRules, RewriteExistenceJoinRhsDedup}

import org.apache.spark.sql.{QueryTest, Row}
import org.apache.spark.sql.catalyst.expressions.{And, AttributeReference, EqualTo, Not}
import org.apache.spark.sql.catalyst.optimizer.ConvertToLocalRelation
import org.apache.spark.sql.catalyst.plans.{ExistenceJoin, LeftAnti, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, BROADCAST, HintInfo, Join, JoinHint, LocalRelation, LogicalPlan}
import org.apache.spark.sql.classic.ClassicDataset
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.IntegerType

class MergeSelfCorrelatedExistenceStateSuite extends QueryTest with SharedSparkSession {
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
      .set(GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key, "false")
      .set(GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key, "false")
  }

  private def withConfigValues[T](pairs: (String, String)*)(body: => T): T = {
    val previous = pairs.map { case (key, _) => key -> spark.conf.getOption(key) }
    pairs.foreach { case (key, value) => spark.conf.set(key, value) }
    try {
      body
    } finally {
      previous.foreach {
        case (key, Some(value)) => spark.conf.set(key, value)
        case (key, None) => spark.conf.unset(key)
      }
    }
  }

  private val canonicalSql =
    """
      |select l1.orderkey, l1.suppkey, l1.payload
      |from paired_lineitem l1
      |where l1.receipt > l1.commit
      |  and exists (
      |    select 1
      |    from paired_lineitem l2
      |    where l2.orderkey = l1.orderkey
      |      and l2.suppkey <> l1.suppkey
      |  )
      |  and not exists (
      |    select 1
      |    from paired_lineitem l3
      |    where l3.orderkey = l1.orderkey
      |      and l3.suppkey <> l1.suppkey
      |      and l3.receipt > l3.commit
      |  )
      |order by l1.orderkey, l1.suppkey, l1.payload
      |""".stripMargin

  private def createCanonicalView(paddingRows: Int = 0): Unit = {
    val canonicalRows = Seq(
      (1, 10, 2, 1, "one-delayed"),
      (1, 20, 1, 1, "other-not-delayed"),
      (2, 10, 2, 1, "first-delayed"),
      (2, 20, 3, 1, "second-delayed"),
      (3, 10, 2, 1, "only-supplier"),
      (3, 10, 2, 1, "duplicate-supplier"),
      (4, 10, 2, 1, "duplicate-a"),
      (4, 10, 2, 1, "duplicate-b"),
      (4, 20, 1, 1, "other-not-delayed")
    )
    // Optional non-candidate rows make the fact/dimension size relationship realistic for the
    // auto-cost test without changing its answer: receipt == commit fails the l1 delayed filter.
    val padding = (0 until paddingRows).map(index => (100 + index, 10, 1, 1, s"padding-$index"))
    (canonicalRows ++ padding)
      .toDF("orderkey", "suppkey", "receipt", "commit", "payload")
      .createOrReplaceTempView("paired_lineitem")
  }

  private def rawOptimizedPlan(sqlText: String): LogicalPlan = {
    withConfigValues(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "false",
      GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key -> "false") {
      spark.sql(sqlText).queryExecution.optimizedPlan
    }
  }

  private def rewrite(plan: LogicalPlan): LogicalPlan = {
    withConfigValues(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "true",
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SOURCE_BYTES.key -> "0",
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SCAN_REDUCTION_RATIO.key -> "1.5"
    ) {
      MergeSelfCorrelatedExistenceState(spark).apply(plan)
    }
  }

  private def rewriteCandidateFirst(plan: LogicalPlan): LogicalPlan = {
    withConfigValues(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "true",
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SOURCE_BYTES.key -> "0",
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SCAN_REDUCTION_RATIO.key -> "1.5",
      GlutenConfig.CANDIDATE_FIRST_EXISTENCE_MODE.key -> "force"
    ) {
      MergeSelfCorrelatedExistenceState(spark).apply(plan)
    }
  }

  private def rewriteCandidateFirstAuto(
      plan: LogicalPlan,
      minCostImprovementRatio: String = "1.15"): LogicalPlan = {
    withConfigValues(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "true",
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SOURCE_BYTES.key -> "0",
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SCAN_REDUCTION_RATIO.key -> "1.5",
      GlutenConfig.CANDIDATE_FIRST_EXISTENCE_MODE.key -> "auto",
      GlutenConfig.CANDIDATE_FIRST_EXISTENCE_MIN_COST_IMPROVEMENT_RATIO.key ->
        minCostImprovementRatio
    ) {
      MergeSelfCorrelatedExistenceState(spark).apply(plan)
    }
  }

  private def pairedStateAggregateCount(plan: LogicalPlan): Int = {
    plan.collect {
      case aggregate: Aggregate
          if aggregate.aggregateExpressions.count(_.name.startsWith("_paired_existence_")) == 4 =>
        aggregate
    }.size
  }

  private def existenceJoinCount(plan: LogicalPlan): Int = {
    plan.collect { case Join(_, _, LeftSemi | LeftAnti, _, _) => true }.size
  }

  private def antiJoinCount(plan: LogicalPlan): Int = {
    plan.collect { case Join(_, _, LeftAnti, _, _) => true }.size
  }

  private def pairedStateSemiJoinCount(plan: LogicalPlan): Int = {
    plan.collect {
      case Join(_, right, LeftSemi, _, _) if pairedStateAggregateCount(right) == 1 => true
    }.size
  }

  private def candidateFirstExistenceJoins(plan: LogicalPlan): Seq[Join] = {
    plan.collect {
      case join @ Join(_, _, ExistenceJoin(exists), _, _)
          if exists.name.startsWith(
            MergeSelfCorrelatedExistenceState.CandidateFirstExistenceAttributePrefix) =>
        join
    }
  }

  test("canonical paired EXISTS and NOT EXISTS use one state aggregate and preserve answers") {
    createCanonicalView()
    val raw = rawOptimizedPlan(canonicalSql)
    assert(existenceJoinCount(raw) == 2, s"Expected raw semi/anti pair:\n${raw.treeString}")

    val rewritten = rewrite(raw)
    val rewrittenDf = ClassicDataset.ofRows(spark, rewritten)
    val baseline = withConfigValues(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "false",
      GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key -> "false") {
      spark.sql(canonicalSql).collect().toSeq
    }

    checkAnswer(rewrittenDf, baseline)
    checkAnswer(
      rewrittenDf,
      Seq(Row(1, 10, "one-delayed"), Row(4, 10, "duplicate-a"), Row(4, 10, "duplicate-b")))
    assert(
      pairedStateAggregateCount(rewritten) == 1,
      s"Expected exactly one merged state aggregate:\n${rewritten.treeString}")
    assert(
      existenceJoinCount(rewritten) == 1 && antiJoinCount(rewritten) == 0 &&
        pairedStateSemiJoinCount(rewritten) == 1,
      s"Expected the raw pair to become one paired-state semi join:\n${rewritten.treeString}"
    )
    val rewrittenAgain = rewrite(rewritten)
    assert(pairedStateAggregateCount(rewrittenAgain) == 1)
    assert(pairedStateSemiJoinCount(rewrittenAgain) == 1)
    assert(rewrittenAgain.output.map(_.exprId) == raw.output.map(_.exprId))
  }

  test("generic existence dedup does not re-aggregate the paired state") {
    createCanonicalView()
    val rewritten = rewrite(rawOptimizedPlan(canonicalSql))
    val afterGenericDedup =
      withConfigValues(GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key -> "true") {
        RewriteExistenceJoinRhsDedup(spark).apply(rewritten)
      }

    assert(
      afterGenericDedup.fastEquals(rewritten),
      s"Paired state must already count as a summarized existence RHS:\n" +
        afterGenericDedup.treeString)
    assert(pairedStateAggregateCount(afterGenericDedup) == 1)
    assert(pairedStateSemiJoinCount(afterGenericDedup) == 1)
  }

  test("forced candidate-first uses private existence columns and preserves answers") {
    createCanonicalView()
    val raw = rawOptimizedPlan(canonicalSql)
    val rewritten = rewriteCandidateFirst(raw)
    val probes = candidateFirstExistenceJoins(rewritten)
    val baseline = withConfigValues(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "false",
      GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key -> "false") {
      spark.sql(canonicalSql).collect().toSeq
    }

    checkAnswer(ClassicDataset.ofRows(spark, rewritten), baseline)
    assert(
      probes.size == 2,
      s"Expected two candidate-first existence probes:\n${rewritten.treeString}")
    assert(existenceJoinCount(rewritten) == 0)
    assert(pairedStateAggregateCount(rewritten) == 0)
    assert(rewritten.output.map(_.exprId) == raw.output.map(_.exprId))

    val afterGenericDedup =
      withConfigValues(GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key -> "true") {
        RewriteExistenceJoinRhsDedup(spark).apply(rewritten)
      }
    assert(
      afterGenericDedup.fastEquals(rewritten),
      s"Private candidate-first probes must bypass generic RHS summarization:\n" +
        afterGenericDedup.treeString)
  }

  test("candidate-first auto without a costable join chain and off retain paired-state") {
    createCanonicalView()
    val raw = rawOptimizedPlan(canonicalSql)
    Seq("auto", "off").foreach {
      mode =>
        val rewritten = withConfigValues(
          GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "true",
          GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SOURCE_BYTES.key -> "0",
          GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SCAN_REDUCTION_RATIO.key -> "1.5",
          GlutenConfig.CANDIDATE_FIRST_EXISTENCE_MODE.key -> mode
        ) {
          MergeSelfCorrelatedExistenceState(spark).apply(raw)
        }
        assert(
          candidateFirstExistenceJoins(rewritten).isEmpty,
          s"Mode $mode must not select candidate-first:\n${rewritten.treeString}")
        assert(
          pairedStateAggregateCount(rewritten) == 1,
          s"Mode $mode must retain the paired-state fallback:\n${rewritten.treeString}")
    }
  }

  test("forced and cost-gated candidate-first absorb selective inner joins") {
    createCanonicalView(paddingRows = 16)
    Seq((10, 100, "supplier-10"), (20, 200, "supplier-20"))
      .toDF("s_suppkey", "s_nationkey", "s_name")
      .createOrReplaceTempView("candidate_first_supplier")
    Seq((100, "SAUDI ARABIA"), (200, "GERMANY"))
      .toDF("n_nationkey", "n_name")
      .createOrReplaceTempView("candidate_first_nation")
    Seq((1, "F"), (2, "F"), (3, "O"), (4, "F"))
      .toDF("o_orderkey", "o_status")
      .createOrReplaceTempView("candidate_first_orders")

    val sql =
      """
        |select l1.orderkey, l1.suppkey, l1.payload, s.s_name, n.n_name, o.o_status
        |from paired_lineitem l1
        |join candidate_first_supplier s on s.s_suppkey = l1.suppkey
        |join candidate_first_nation n on n.n_nationkey = s.s_nationkey
        |join candidate_first_orders o on o.o_orderkey = l1.orderkey
        |where n.n_name = 'SAUDI ARABIA'
        |  and o.o_status = 'F'
        |  and l1.receipt > l1.commit
        |  and exists (
        |    select 1 from paired_lineitem l2
        |    where l2.orderkey = l1.orderkey and l2.suppkey <> l1.suppkey)
        |  and not exists (
        |    select 1 from paired_lineitem l3
        |    where l3.orderkey = l1.orderkey and l3.suppkey <> l1.suppkey
        |      and l3.receipt > l3.commit)
        |order by l1.orderkey, l1.suppkey, l1.payload
        |""".stripMargin
    // Keep the literal nation/orders filters visible to the cost model. The normal local-relation
    // optimizer eagerly evaluates them into new LocalRelation rows, which is unlike the Parquet
    // relations used in production and would turn this into a test of the conservative fallback.
    val raw =
      withConfigValues(SQLConf.OPTIMIZER_EXCLUDED_RULES.key -> ConvertToLocalRelation.ruleName) {
        rawOptimizedPlan(sql)
      }
    val rewritten = rewriteCandidateFirst(raw)
    val autoRewritten = rewriteCandidateFirstAuto(raw, minCostImprovementRatio = "1.10")
    val rejectedByCost = rewriteCandidateFirstAuto(raw, minCostImprovementRatio = "100.0")
    val probes = candidateFirstExistenceJoins(rewritten)
    val autoProbes = candidateFirstExistenceJoins(autoRewritten)
    val firstProbe = probes
      .find {
        case Join(_, _, ExistenceJoin(exists), _, _) => exists.name.endsWith("all")
        case _ => false
      }
      .getOrElse(fail(s"Missing all-RHS candidate-first probe:\n${rewritten.treeString}"))
    val candidateNames = firstProbe.left.output.map(_.name).toSet
    val baseline = withConfigValues(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "false",
      GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key -> "false") {
      spark.sql(sql).collect().toSeq
    }

    checkAnswer(ClassicDataset.ofRows(spark, rewritten), baseline)
    checkAnswer(ClassicDataset.ofRows(spark, autoRewritten), baseline)
    assert(probes.size == 2, s"Expected two candidate-first probes:\n${rewritten.treeString}")
    assert(
      autoProbes.size == 2,
      s"Expected cost-gated candidate-first probes:\n${autoRewritten.treeString}")
    assert(
      candidateFirstExistenceJoins(rejectedByCost).isEmpty &&
        pairedStateAggregateCount(rejectedByCost) == 1,
      s"High candidate-first cost threshold must retain paired-state:\n" +
        rejectedByCost.treeString
    )
    assert(
      Set("s_name", "n_name", "o_status").subsetOf(candidateNames),
      s"Supplier, nation, and orders must be inside the candidate. " +
        s"candidateOutput=${candidateNames.toSeq.sorted.mkString(",")}\n${rewritten.treeString}"
    )
  }

  test("NULL compared values and duplicates keep SQL existence semantics") {
    Seq[(Int, java.lang.Integer, Int, Int, String)](
      (1, 10, 2, 1, "nonnull-delayed"),
      (1, null, 1, 1, "null-other"),
      (2, null, 2, 1, "null-candidate"),
      (2, 20, 1, 1, "nonnull-other"),
      (3, 10, 2, 1, "candidate"),
      (3, 20, 1, 1, "other-a"),
      (3, 20, 1, 1, "other-b")
    )
      .toDF("orderkey", "suppkey", "receipt", "commit", "payload")
      .createOrReplaceTempView("paired_lineitem")

    val raw = rawOptimizedPlan(canonicalSql)
    val rewritten = rewrite(raw)
    val baseline = withConfigValues(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "false",
      GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key -> "false") {
      spark.sql(canonicalSql).collect().toSeq
    }
    checkAnswer(ClassicDataset.ofRows(spark, rewritten), baseline)
    checkAnswer(ClassicDataset.ofRows(spark, rewritten), Seq(Row(3, 10, "candidate")))
  }

  test("cost guard rejects a source below the configured minimum") {
    createCanonicalView()
    val raw = rawOptimizedPlan(canonicalSql)
    val unchanged = withConfigValues(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "true",
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SOURCE_BYTES.key -> "1gb") {
      MergeSelfCorrelatedExistenceState(spark).apply(raw)
    }
    assert(pairedStateAggregateCount(unchanged) == 0)
    assert(existenceJoinCount(unchanged) == 2)
  }

  test("explicitly disabled rule leaves the raw semi and anti joins unchanged") {
    createCanonicalView()
    val raw = rawOptimizedPlan(canonicalSql)
    val unchanged = withConfigValues(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "false",
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SOURCE_BYTES.key -> "0") {
      MergeSelfCorrelatedExistenceState(spark).apply(raw)
    }
    assert(pairedStateAggregateCount(unchanged) == 0)
    assert(existenceJoinCount(unchanged) == 2)
  }

  test("same column names from different relations are not treated as a self-correlation") {
    createCanonicalView()
    Seq((1, 20, 1, 1, "different"), (2, 30, 2, 1, "different"))
      .toDF("orderkey", "suppkey", "receipt", "commit", "payload")
      .createOrReplaceTempView("other_lineitem")
    val sql = canonicalSql
      .replace("from paired_lineitem l2", "from other_lineitem l2")
      .replace("from paired_lineitem l3", "from other_lineitem l3")
    val raw = rawOptimizedPlan(sql)
    val unchanged = rewrite(raw)
    assert(pairedStateAggregateCount(unchanged) == 0)
    assert(existenceJoinCount(unchanged) == 2)
  }

  test("different correlation key is rejected") {
    createCanonicalView()
    val sql = canonicalSql.replace("l3.orderkey = l1.orderkey", "l3.suppkey = l1.orderkey")
    val unchanged = rewrite(rawOptimizedPlan(sql))
    assert(pairedStateAggregateCount(unchanged) == 0)
    assert(existenceJoinCount(unchanged) == 2)
  }

  test("an additional mixed residual is rejected") {
    createCanonicalView()
    val sql = canonicalSql.replace(
      "and l2.suppkey <> l1.suppkey",
      "and l2.suppkey <> l1.suppkey and l2.receipt <> l1.receipt")
    val unchanged = rewrite(rawOptimizedPlan(sql))
    assert(pairedStateAggregateCount(unchanged) == 0)
    assert(existenceJoinCount(unchanged) == 2)
  }

  test("join hints are not discarded by the rewrite") {
    createCanonicalView()
    val raw = rawOptimizedPlan(canonicalSql)
    var hinted = false
    val withHint = raw.transformUp {
      case join @ Join(_, _, LeftSemi, _, JoinHint.NONE) if !hinted =>
        hinted = true
        join.copy(hint = JoinHint(None, Some(HintInfo(strategy = Some(BROADCAST)))))
    }
    assert(hinted)
    val unchanged = rewrite(withHint)
    assert(pairedStateAggregateCount(unchanged) == 0)
    assert(existenceJoinCount(unchanged) == 2)
  }

  test("candidate must explicitly imply the delayed predicate") {
    createCanonicalView()
    val sql = canonicalSql.replace("where l1.receipt > l1.commit", "where l1.receipt >= l1.commit")
    val unchanged = rewrite(rawOptimizedPlan(sql))
    assert(pairedStateAggregateCount(unchanged) == 0)
    assert(existenceJoinCount(unchanged) == 2)
  }

  test("a nondeterministic delayed predicate is rejected") {
    createCanonicalView()
    val sql = canonicalSql.replace(
      "and l3.receipt > l3.commit",
      "and l3.receipt > l3.commit and rand() < 0.5")
    val unchanged = rewrite(rawOptimizedPlan(sql))
    assert(pairedStateAggregateCount(unchanged) == 0)
    assert(existenceJoinCount(unchanged) == 2)
  }

  test("streaming inputs are rejected before cost or state reasoning") {
    val l1Order = AttributeReference("orderkey", IntegerType, nullable = false)()
    val l1Supp = AttributeReference("suppkey", IntegerType, nullable = true)()
    val l2Order = AttributeReference("orderkey", IntegerType, nullable = false)()
    val l2Supp = AttributeReference("suppkey", IntegerType, nullable = true)()
    val l3Order = AttributeReference("orderkey", IntegerType, nullable = false)()
    val l3Supp = AttributeReference("suppkey", IntegerType, nullable = true)()
    val l1 = LocalRelation(Seq(l1Order, l1Supp), isStreaming = true)
    val l2 = LocalRelation(Seq(l2Order, l2Supp), isStreaming = true)
    val l3 = LocalRelation(Seq(l3Order, l3Supp), isStreaming = true)
    val semiCondition = And(EqualTo(l1Order, l2Order), Not(EqualTo(l1Supp, l2Supp)))
    val semi = Join(l1, l2, LeftSemi, Some(semiCondition), JoinHint.NONE)
    val antiCondition = And(EqualTo(l1Order, l3Order), Not(EqualTo(l1Supp, l3Supp)))
    val anti = Join(semi, l3, LeftAnti, Some(antiCondition), JoinHint.NONE)

    val unchanged = rewrite(anti)
    assert(unchanged.fastEquals(anti), s"Streaming plan must remain unchanged:\n$unchanged")
  }

  test("post-subquery rules are registered before the first optimizer execution") {
    createCanonicalView()
    val analyzed = withConfigValues(
      GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "false",
      GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key -> "false") {
      spark.sql(canonicalSql).queryExecution.analyzed
    }
    val experimental = spark.experimental
    val original = experimental.extraOptimizations
    try {
      experimental.extraOptimizations = original.filterNot {
        case _: MergeSelfCorrelatedExistenceState => true
        case _: RewriteExistenceJoinRhsDedup => true
        case _ => false
      }
      withConfigValues(
        GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "true",
        GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key -> "true",
        GlutenConfig.CANDIDATE_FIRST_EXISTENCE_MODE.key -> "force",
        GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SOURCE_BYTES.key -> "0",
        SQLConf.OPTIMIZER_EXCLUDED_RULES.key -> ConvertToLocalRelation.ruleName
      ) {
        RegisterMppExistencePostSubqueryRules(spark).apply(analyzed)
      }
      val mergeIndex = experimental.extraOptimizations.indexWhere(
        _.isInstanceOf[MergeSelfCorrelatedExistenceState])
      val dedupIndex =
        experimental.extraOptimizations.indexWhere(_.isInstanceOf[RewriteExistenceJoinRhsDedup])
      assert(mergeIndex >= 0 && dedupIndex >= 0 && mergeIndex < dedupIndex)
      val optimized = withConfigValues(
        GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_ENABLED.key -> "true",
        GlutenConfig.ENABLE_EXISTENCE_JOIN_RHS_DEDUP.key -> "true",
        GlutenConfig.CANDIDATE_FIRST_EXISTENCE_MODE.key -> "force",
        GlutenConfig.MERGE_PAIRED_EXISTENCE_STATE_MIN_SOURCE_BYTES.key -> "0",
        SQLConf.OPTIMIZER_EXCLUDED_RULES.key -> ConvertToLocalRelation.ruleName
      ) {
        spark.sessionState.optimizer.execute(analyzed)
      }
      assert(
        candidateFirstExistenceJoins(optimized).size == 2,
        s"The first optimizer execution must see both candidate-first probes:\n" +
          optimized.treeString)
    } finally {
      experimental.extraOptimizations = original
    }
  }
}
