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
package org.apache.gluten.extension.columnar

import org.apache.gluten.config.GlutenConfig
import org.apache.gluten.sql.shims.SparkShimLoader

import org.apache.spark.sql.{GlutenQueryTest, SQLContext}
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, AttributeReference, AttributeSet}
import org.apache.spark.sql.catalyst.expressions.{EqualTo, Expression, GreaterThan, Literal}
import org.apache.spark.sql.catalyst.expressions.aggregate.Min
import org.apache.spark.sql.catalyst.plans.{ExistenceJoin, FullOuter, Inner, JoinType}
import org.apache.spark.sql.catalyst.plans.{LeftAnti, LeftOuter, LeftSemi, RightOuter}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, BROADCAST, Filter, HintInfo, Join}
import org.apache.spark.sql.catalyst.plans.logical.{JoinHint, LeafNode, LogicalPlan}
import org.apache.spark.sql.catalyst.plans.logical.{Project, UnaryNode}
import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.sources.BaseRelation
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{DateType, DoubleType, IntegerType, LongType}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

class PushSelectiveDimensionChainBeforeFactSuite extends GlutenQueryTest with SharedSparkSession {

  import testImplicits._

  private val confKey = GlutenConfig.PUSH_DIMENSION_CHAIN_BEFORE_FACT_ENABLED.key

  test("can be disabled") {
    val p = q5LikePlan()
    withSQLConf(confKey -> "false") {
      assert(PushSelectiveDimensionChainBeforeFact(spark)(p.plan).fastEquals(p.plan))
    }
  }

  test("self-registers exactly once for the post-CBO optimizer pass") {
    val experimental = spark.experimental
    val before = experimental.extraOptimizations
    try {
      experimental.synchronized {
        experimental.extraOptimizations =
          before.filterNot(_.isInstanceOf[PushSelectiveDimensionChainBeforeFact])
      }

      withSQLConf(confKey -> "true") {
        val rule = PushSelectiveDimensionChainBeforeFact(spark)
        assert(postCboRewriteCount(experimental) == 1)
        assert(postCboHintCount(experimental) == 1)
        val rewriteIndex = experimental.extraOptimizations.indexWhere(
          _.isInstanceOf[PushSelectiveDimensionChainBeforeFact])
        val hintIndex =
          experimental.extraOptimizations.indexWhere(_.isInstanceOf[FluxFactProbeBroadcastHint])
        assert(rewriteIndex >= 0 && hintIndex > rewriteIndex)
        rule(q5LikePlan().plan)
        assert(postCboRewriteCount(experimental) == 1)
        assert(postCboHintCount(experimental) == 1)
      }
    } finally {
      experimental.synchronized {
        experimental.extraOptimizations = before
      }
    }
  }

  test("prunes customer through nation/region before the orders join; supplier stays late") {
    val p = q5LikePlan()
    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = PushSelectiveDimensionChainBeforeFact(spark)(p.plan)
    }
    assert(!rewritten.fastEquals(p.plan), "rule should have reordered the plan")
    assert(rewritten.output == p.plan.output, "rule should preserve the original output order")

    val rewrittenCore = stripTopProject(rewritten)

    // supplier must be the last (top-level right) join, carrying BOTH its high-NDV key and the
    // same-nation residual.
    rewrittenCore match {
      case Join(_, right, Inner, Some(cond), _) =>
        assert(right.eq(p.supplier), s"top-level right should be supplier, got $right")
        val preds = splitAnd(cond)
        assert(
          preds.exists(_.semanticEquals(p.lSuppKeyEqSSuppKey)),
          s"supplier join must keep l_suppkey=s_suppkey, got $cond")
        assert(
          preds.exists(_.semanticEquals(p.cNationKeyEqSNationKey)),
          s"supplier join must carry the same-nation residual c_nationkey=s_nationkey, got $cond")
      case other => fail(s"expected supplier as the outermost join, got:\n$other")
    }

    // customer must be directly joined to (nation JOIN region) via c_nationkey = n_nationkey.
    val custPruned = findCustomerPrune(rewrittenCore)
    assert(
      custPruned.isDefined,
      s"expected customer JOIN (nation JOIN region) subtree in:\n$rewrittenCore")
    custPruned.get match {
      case Join(left, Join(nLeft, nRight, Inner, _, _), Inner, Some(c), _) =>
        assert(left.eq(p.customer))
        assert(nLeft.eq(p.nation) && nRight.eq(p.region))
        assert(
          c.semanticEquals(EqualTo(p.cNationKey, p.nNationKey)) ||
            c.semanticEquals(EqualTo(p.nNationKey, p.cNationKey)))
      case other => fail(s"unexpected custPruned shape: $other")
    }
  }

  test("unwraps attribute-only project around the post-CBO join cluster") {
    val p = q5LikePlan()
    val planWithProject = Project(p.plan.output, p.plan)

    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = PushSelectiveDimensionChainBeforeFact(spark)(planWithProject)
    }

    assert(!rewritten.fastEquals(planWithProject), "rule should look through Project wrappers")
    assert(rewritten.output == planWithProject.output, "rule should preserve projected output")
    assert(
      findCustomerPrune(stripTopProject(rewritten)).isDefined,
      s"expected customer JOIN (nation JOIN region) subtree in:\n$rewritten")
  }

  test("rewrites post-CBO clusters even after broadcast hints were attached") {
    val p = q5LikePlan()
    val hinted = addBroadcastHints(p.plan)

    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = PushSelectiveDimensionChainBeforeFact(spark)(hinted)
    }

    assert(!rewritten.fastEquals(hinted), "rule should ignore existing hints while matching")
    assert(
      findCustomerPrune(stripTopProject(rewritten)).isDefined,
      s"expected customer JOIN (nation JOIN region) subtree in:\n$rewritten")
  }

  test("pushes a filtered seed into a directly joined chain dimension before a large victim") {
    val p = directChainVictimPlan()
    var rewritten: LogicalPlan = null
    var rewrittenAgain: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      val rule = PushSelectiveDimensionChainBeforeFact(spark)
      rewritten = rule(p.plan)
      rewrittenAgain = rule(rewritten)
    }

    assert(!rewritten.fastEquals(p.plan), "rule should reorder the direct chain-victim plan")
    assert(
      rewrittenAgain.fastEquals(rewritten),
      s"direct chain-victim rewrite should be fixed-point idempotent:\n$rewrittenAgain")
    assert(
      findVictimPrunedByChainSeed(rewritten, "l_orderkey", "s_suppkey", "n_nationkey").isDefined,
      s"expected victim JOIN (chain JOIN filtered seed) subtree in:\n$rewritten"
    )
    assert(
      containsPredicate(rewritten, p.lOrderKeyEqOOrderKey),
      s"fact-ward high-NDV predicate must remain:\n$rewritten")
    assert(
      broadcastHintSideForPredicate(rewritten, p.lOrderKeyEqOOrderKey).isEmpty,
      s"fact-ward high-NDV join should stay partitionable:\n$rewritten")
  }

  test("allows a fact-ward literal filter on the chosen victim high-NDV path") {
    val p = directChainVictimPlan(filterOrders = true)

    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = PushSelectiveDimensionChainBeforeFact(spark)(p.plan)
    }

    assert(!rewritten.fastEquals(p.plan), "fact-ward filters should not block the rewrite")
    assert(
      findVictimPrunedByChainSeed(rewritten, "l_orderkey", "s_suppkey", "n_nationkey").isDefined,
      s"expected victim JOIN (chain JOIN filtered seed) subtree in:\n$rewritten"
    )
    assert(
      containsPredicate(rewritten, p.lOrderKeyEqOOrderKey),
      s"fact-ward high-NDV predicate must remain:\n$rewritten")
  }

  test("keeps the original competing-filter guard for a base-range chain without a spine") {
    val p = directChainVictimPlan(filterOrders = true, includeOrderState = true)

    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = PushSelectiveDimensionChainBeforeFact(spark)(p.plan)
    }

    assert(
      rewritten.fastEquals(p.plan),
      "a second selective entrance must retain the pre-cost-rule conservative behavior")
  }

  test("allows a moderate chain dimension when a tiny seed prunes a much larger victim") {
    val p = directChainVictimPlan(chainRows = 50000000L)

    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = PushSelectiveDimensionChainBeforeFact(spark)(p.plan)
    }

    assert(!rewritten.fastEquals(p.plan), "rule should allow moderate prunable chain dimensions")
    assert(
      findVictimPrunedByChainSeed(rewritten, "l_orderkey", "s_suppkey", "n_nationkey").isDefined,
      s"expected victim JOIN (chain JOIN filtered seed) subtree in:\n$rewritten"
    )
  }

  test("large configured chain requires multiple high-NDV fact-ward exits") {
    val singleExit = directChainVictimPlan(chainRows = 300000000L)
    val twoExits = directChainVictimPlan(chainRows = 300000000L, includeOrderState = true)
    val maxRowsKey =
      "spark.gluten.sql.columnar.pushDimensionChainBeforeFact.maxChainRows"
    val maxBytesKey =
      "spark.gluten.sql.columnar.pushDimensionChainBeforeFact.maxChainBytes"

    withSQLConf(confKey -> "true", maxRowsKey -> "1000000000", maxBytesKey -> "64g") {
      val rule = PushSelectiveDimensionChainBeforeFact(spark)
      assert(
        rule(singleExit.plan).fastEquals(singleExit.plan),
        "an extended-range chain with only one downstream exit should retain the CBO plan")
      assert(
        !rule(twoExits.plan).fastEquals(twoExits.plan),
        "multiple high-NDV exits should justify the extended-range chain rewrite")
    }
  }

  test("cost admits a large chain only when multiple exits repay worst-case FLUX broadcast") {
    val singleExit = directChainVictimPlan(chainRows = 300000000L, victimRows = 180000000000L)
    val twoExits = directChainVictimPlan(
      chainRows = 300000000L,
      victimRows = 180000000000L,
      includeOrderState = true)
    val maxRowsKey =
      "spark.gluten.sql.columnar.pushDimensionChainBeforeFact.maxChainRows"
    val maxBytesKey =
      "spark.gluten.sql.columnar.pushDimensionChainBeforeFact.maxChainBytes"

    withSQLConf(
      confKey -> "true",
      maxRowsKey -> "10000000",
      maxBytesKey -> "1g",
      "spark.sql.autoBroadcastJoinThreshold" -> "12g",
      "spark.gluten.mpp.multiExecutor.numPartitions" -> "32"
    ) {
      val rule = PushSelectiveDimensionChainBeforeFact(spark)
      assert(
        rule(singleExit.plan).fastEquals(singleExit.plan),
        "a single fact-ward exit must not open the cost-based large-chain path")
      assert(
        !rule(twoExits.plan).fastEquals(twoExits.plan),
        "two exits over a much larger victim should repay the broadcast-adjusted chain cost")
    }
  }

  test("cost rejects a large chain when broadcast-adjusted benefit is too small") {
    val lowBenefit = directChainVictimPlan(
      chainRows = 300000000L,
      victimRows = 6000000000L,
      includeOrderState = true)
    val maxRowsKey =
      "spark.gluten.sql.columnar.pushDimensionChainBeforeFact.maxChainRows"
    val maxBytesKey =
      "spark.gluten.sql.columnar.pushDimensionChainBeforeFact.maxChainBytes"

    withSQLConf(
      confKey -> "true",
      maxRowsKey -> "10000000",
      maxBytesKey -> "1g",
      "spark.sql.autoBroadcastJoinThreshold" -> "12g",
      "spark.gluten.mpp.multiExecutor.numPartitions" -> "32"
    ) {
      assert(
        PushSelectiveDimensionChainBeforeFact(spark)(lowBenefit.plan).fastEquals(lowBenefit.plan),
        "the old size gate should remain closed when avoided work cannot repay broadcast cost"
      )
    }
  }

  test("cost rejects a large chain whose filtered estimate exceeds broadcast budget") {
    val p = directChainVictimPlan(
      chainRows = 300000000L,
      victimRows = 180000000000L,
      includeOrderState = true)
    val maxRowsKey =
      "spark.gluten.sql.columnar.pushDimensionChainBeforeFact.maxChainRows"
    val maxBytesKey =
      "spark.gluten.sql.columnar.pushDimensionChainBeforeFact.maxChainBytes"

    withSQLConf(
      confKey -> "true",
      maxRowsKey -> "10000000",
      maxBytesKey -> "1g",
      "spark.sql.autoBroadcastJoinThreshold" -> "2g",
      "spark.gluten.mpp.multiExecutor.numPartitions" -> "32"
    ) {
      assert(
        PushSelectiveDimensionChainBeforeFact(spark)(p.plan).fastEquals(p.plan),
        "large-chain cost path must respect Spark's broadcast threshold")
    }
  }

  test("extended cost path keeps competing selective entrances without a probe spine") {
    val p = directChainVictimPlan(
      chainRows = 300000000L,
      victimRows = 180000000000L,
      filterOrders = true,
      includeOrderState = true,
      filterOrderState = true)
    withCanonicalCostConf {
      assert(
        PushSelectiveDimensionChainBeforeFact(spark)(p.plan).fastEquals(p.plan),
        "the Q21 multi-exit exemption must not widen ordinary inner-only clusters")
    }
  }

  test("pushes a Project-wrapped dimension through canonical dedup-inner and anti wrappers") {
    val p = canonicalMixedProbeSpinePlan()
    var rewritten: LogicalPlan = null
    var rewrittenAgain: LogicalPlan = null
    withCanonicalCostConf {
      val rule = PushSelectiveDimensionChainBeforeFact(spark)
      rewritten = rule(p.plan)
      rewrittenAgain = rule(rewritten)
    }

    assert(
      !rewritten.fastEquals(p.plan),
      s"canonical mixed probe spine was not rewritten:\n$rewritten")
    assert(rewritten.output == p.plan.output, "top-level output exprIds/order must be preserved")
    assert(rewrittenAgain.fastEquals(rewritten), "mixed probe-spine rewrite must be fixed point")
    val injectedPrune =
      findVictimPrunedByChainSeed(rewritten, "l_orderkey", "s_suppkey", "n_nationkey").getOrElse(
        fail(s"injected dimension prune missing:\n$rewritten"))
    assert(
      isBelowJoinType(rewritten, injectedPrune, LeftAnti),
      s"supplier/nation must be inside the anti probe, not above it:\n$rewritten")
    assert(
      injectedPrune.asInstanceOf[Join].hint == p.expectedPruneHint,
      s"chain-left broadcast hint must follow dimension to injected right side:\n$rewritten")

    val rewrittenAnti = rewritten
      .collectFirst {
        case join: Join if join.joinType == LeftAnti => join
      }
      .getOrElse(fail(s"rewritten anti join missing:\n$rewritten"))
    assert(rewrittenAnti.condition == p.anti.condition)
    assert(rewrittenAnti.hint == p.anti.hint)
    assert(
      containsJoinWithConditionAndHint(rewrittenAnti.left, p.dedupInner),
      s"dedup-inner type/condition/hint must survive the commute:\n$rewritten")
    assert(
      hasProjectPrefixWithCarriedAttr(rewritten, p.victim.output, p.sName),
      s"attribute-only wrappers must preserve original output order and carry s_name:\n$rewritten"
    )
  }

  test("pushes a Project-wrapped dimension through a paired existence state join") {
    val p = pairedExistenceStateProbeSpinePlan()
    var rewritten: LogicalPlan = null
    var rewrittenAgain: LogicalPlan = null
    withCanonicalCostConf {
      val rule = PushSelectiveDimensionChainBeforeFact(spark)
      rewritten = rule(p.plan)
      rewrittenAgain = rule(rewritten)
    }

    assert(
      !rewritten.fastEquals(p.plan),
      s"paired state probe spine was not rewritten:\n$rewritten")
    assert(rewritten.output == p.plan.output, "top-level output exprIds/order must be preserved")
    assert(rewrittenAgain.fastEquals(rewritten), "paired state rewrite must be fixed point")
    val injectedPrune =
      findVictimPrunedByChainSeed(rewritten, "l_orderkey", "s_suppkey", "n_nationkey")
        .getOrElse(fail(s"injected dimension prune missing:\n$rewritten"))
    assert(
      isBelowJoinType(rewritten, injectedPrune, LeftSemi),
      s"supplier/nation must be inside the paired state join:\n$rewritten")
    assert(
      hasProjectPrefixWithCarriedAttr(rewritten, p.victim.output, p.sName),
      s"attribute-only wrappers must carry s_name through the paired state join:\n$rewritten"
    )
  }

  test("rejects a mixed probe spine with a non-equi semi or anti residual") {
    val p = canonicalMixedProbeSpinePlan(antiResidual = true)
    withCanonicalCostConf {
      assert(
        PushSelectiveDimensionChainBeforeFact(spark)(p.plan).fastEquals(p.plan),
        "non-equi existence residual must block probe-spine commute")
    }
  }

  test("probe-spine guard rejects RHS and conditions that reference a dimension attribute") {
    val p = canonicalMixedProbeSpinePlan(rhsReferencesDimension = true)
    withCanonicalCostConf {
      val rule = PushSelectiveDimensionChainBeforeFact(spark)
      val victimAttr = p.victim.output.find(_.name == "l_suppkey").get
      assert(
        !rule.probeSpineCanCommuteForTesting(p.victim, AttributeSet(Seq(p.sName)), victimAttr),
        "a wrapper RHS that captures a dimension attr must block the commute"
      )
      val clean = canonicalMixedProbeSpinePlan()
      val cleanVictimAttr = clean.victim.output.find(_.name == "l_suppkey").get
      val conditionCapture = clean.anti.copy(
        condition = Some(And(clean.anti.condition.get, EqualTo(clean.sName, clean.sName))))
      val capturedVictim = Project(conditionCapture.output, conditionCapture)
      assert(
        !rule.probeSpineCanCommuteForTesting(
          capturedVictim,
          AttributeSet(Seq(clean.sName)),
          cleanVictimAttr),
        "a wrapper condition that captures a dimension attr must block the commute"
      )
    }
  }

  test("commutes through LeftSemi and ExistenceJoin without changing type or output identities") {
    val existsAttr = AttributeReference("exists", org.apache.spark.sql.types.BooleanType)()
    Seq[JoinType](LeftSemi, ExistenceJoin(existsAttr)).foreach {
      wrapperType =>
        val p = canonicalMixedProbeSpinePlan(firstWrapperType = wrapperType)
        withCanonicalCostConf {
          val rewritten = PushSelectiveDimensionChainBeforeFact(spark)(p.plan)
          assert(!rewritten.fastEquals(p.plan), s"$wrapperType spine should be rewritten")
          assert(rewritten.output == p.plan.output)
          val matching = rewritten
            .collectFirst {
              case join: Join if join.joinType == wrapperType => join
            }
            .getOrElse(fail(s"$wrapperType missing after commute:\n$rewritten"))
          assert(matching.condition == p.dedupInner.condition)
          assert(matching.hint == p.dedupInner.hint)
          wrapperType match {
            case ExistenceJoin(originalExists) =>
              val rewrittenExists = matching.joinType.asInstanceOf[ExistenceJoin].exists
              assert(rewrittenExists.exprId == originalExists.exprId)
            case _ =>
          }
        }
    }
  }

  test("pure-equi probe-spine commute preserves duplicate and NULL bag semantics") {
    val probe = Seq(
      (Some(1L), Some(10L), "p1"),
      (Some(1L), Some(10L), "p1"),
      (Some(2L), Some(20L), "p2"),
      (Some(2L), Some(20L), "p2"),
      (Some(3L), None, "p3"),
      (Some(4L), Some(40L), "p4"),
      (Some(5L), Some(50L), "p5"),
      (Some(6L), None, "p6")
    ).toDF("l_orderkey", "l_suppkey", "payload")
    val rawInnerRhs =
      Seq(Some(1L), Some(1L), Some(2L), Some(3L), Some(4L), Some(5L), Some(6L), None)
        .toDF("i_orderkey")
    val uniqueInnerRhs = rawInnerRhs.groupBy("i_orderkey").count().drop("count")
    val antiRhs = Seq(
      (Some(1L), Some(99L)),
      (Some(1L), Some(99L)),
      (Some(3L), Some(30L)),
      (Some(4L), Some(99L)),
      (None, Some(40L))).toDF("a_orderkey", "a_suppkey")
    val dimension = Seq(
      (Some(10L), "d10a"),
      (Some(10L), "d10b"),
      (Some(20L), "d20a"),
      (Some(20L), "d20b"),
      (None, "d-null"),
      (Some(40L), "d40"),
      (Some(50L), "d50")).toDF("d_suppkey", "d_name")

    def addExistenceWrappers(left: org.apache.spark.sql.DataFrame): org.apache.spark.sql.DataFrame =
      left
        .join(uniqueInnerRhs, left("l_orderkey") === uniqueInnerRhs("i_orderkey"), "inner")
        .join(antiRhs, left("l_orderkey") === antiRhs("a_orderkey"), "left_anti")

    val original = addExistenceWrappers(probe)
      .join(dimension, probe("l_suppkey") === dimension("d_suppkey"), "inner")
      .select("l_orderkey", "l_suppkey", "payload", "d_name")
    val probeWithDimension = probe
      .join(dimension, probe("l_suppkey") === dimension("d_suppkey"), "inner")
    val commuted = addExistenceWrappers(probeWithDimension)
      .select("l_orderkey", "l_suppkey", "payload", "d_name")

    val originalRows = original.collect()
    val commutedRows = commuted.collect()
    def rowBag(rows: Array[org.apache.spark.sql.Row]): Map[org.apache.spark.sql.Row, Int] =
      rows.groupBy(identity).map { case (row, copies) => row -> copies.length }

    assert(originalRows.length == 5, "expected duplicate probe and dimension rows to survive")
    assert(rowBag(originalRows) == rowBag(commutedRows), "commute must preserve the row multiset")
  }

  test("does not count low-NDV probe wrappers as avoided high-NDV work") {
    val p = canonicalMixedProbeSpinePlan(lowNdvWrappers = true)
    withCanonicalCostConf {
      assert(
        PushSelectiveDimensionChainBeforeFact(spark)(p.plan).fastEquals(p.plan),
        "low-NDV state wrappers must not inflate the extended-chain cost benefit")
    }
  }

  test("rejects non-deduplicated inner and every outer-join probe wrapper") {
    val nonDedup = canonicalMixedProbeSpinePlan(deduplicateInnerRhs = false)
    val onlyUniqueOnWiderKey = canonicalMixedProbeSpinePlan(extraInnerGroupingKey = true)
    withCanonicalCostConf {
      assert(PushSelectiveDimensionChainBeforeFact(spark)(nonDedup.plan).fastEquals(nonDedup.plan))
      assert(
        PushSelectiveDimensionChainBeforeFact(spark)(onlyUniqueOnWiderKey.plan)
          .fastEquals(onlyUniqueOnWiderKey.plan),
        "GROUP BY (orderkey, suppkey) must not prove uniqueness for a join on orderkey alone"
      )
      Seq(LeftOuter, RightOuter, FullOuter).foreach {
        joinType =>
          val p = canonicalMixedProbeSpinePlan(outerWrapperType = joinType)
          assert(
            PushSelectiveDimensionChainBeforeFact(spark)(p.plan).fastEquals(p.plan),
            s"$joinType probe wrapper must never commute")
      }
    }
  }

  test("rejects a composite prune key supplied by a probe-wrapper RHS") {
    val p = canonicalMixedProbeSpinePlan(extraPruneKeyFromInnerRhs = true)
    withCanonicalCostConf {
      assert(
        PushSelectiveDimensionChainBeforeFact(spark)(p.plan).fastEquals(p.plan),
        "every injected prune-key reference must be available at the bottom probe")
    }
  }

  test("rejects a mixed probe spine beyond the total unary-and-join depth bound") {
    val p = canonicalMixedProbeSpinePlan(probeUnaryDepth = 8)
    withCanonicalCostConf {
      assert(
        PushSelectiveDimensionChainBeforeFact(spark)(p.plan).fastEquals(p.plan),
        "the mixed probe-spine matcher must stop after its bounded total depth")
    }
  }

  test("does not duplicate the synthesized prune predicate in fixed point") {
    val p = q5LikePlan()
    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      val rule = PushSelectiveDimensionChainBeforeFact(spark)
      rewritten = rule(rule(p.plan))
    }

    assert(
      countPredicate(rewritten, EqualTo(p.cNationKey, p.nNationKey)) == 1,
      s"synthesized prune predicate should appear exactly once:\n$rewritten")
  }

  test("preserves residual predicates while rebuilding the join tree") {
    val p = q5LikePlan()
    val residual = GreaterThan(p.cNationKey, Literal(0))
    val planWithResidual = Filter(residual, p.plan)

    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = PushSelectiveDimensionChainBeforeFact(spark)(planWithResidual)
    }

    assert(!rewritten.fastEquals(planWithResidual), "rule should have reordered the plan")
    assert(containsPredicate(rewritten, residual), s"residual predicate was dropped:\n$rewritten")
  }

  test("narrows star-chain intermediate outputs under a top projection") {
    val p = q5LikePlan()
    val projected = Project(Seq(p.nName, p.lExtPrice, p.lDiscount), p.plan)

    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = PushSelectiveDimensionChainBeforeFact(spark)(projected)
    }

    assert(!rewritten.fastEquals(projected), "rule should have reordered the projected plan")
    assert(rewritten.output == projected.output, "rule should preserve the top projection output")

    val projectOutputs = projectOutputNameSets(rewritten)
    assert(
      projectOutputs.contains(Set("n_nationkey", "n_name")),
      s"expected nation/region keyset to keep only n_nationkey,n_name:\n$rewritten")
    assert(
      projectOutputs.contains(Set("c_custkey", "c_nationkey", "n_name")),
      s"expected pruned customer to keep only c_custkey,c_nationkey,n_name:\n$rewritten")
    val ordersCustomerOutput = Set("o_orderkey", "c_nationkey", "n_name")
    assert(
      projectOutputs.contains(ordersCustomerOutput),
      s"expected orders/customer build to keep only o_orderkey,c_nationkey,n_name:\n$rewritten"
    )
    assert(
      hasLeftBroadcastJoinForProjectedOutput(rewritten, ordersCustomerOutput),
      s"expected pruned-customer/orders join to broadcast the pruned build side:\n$rewritten"
    )
    assert(
      !containsPredicate(rewritten, EqualTo(p.sNationKey, p.nNationKey)),
      s"same-nation predicate should not keep n_nationkey alive after projection:\n$rewritten"
    )
    assert(
      containsPredicate(rewritten, p.cNationKeyEqSNationKey),
      s"supplier residual c_nationkey=s_nationkey should remain:\n$rewritten")
  }

  test("broadcasts a pruned branch into a much larger fact-ward branch") {
    val p = q5LikePlan()

    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = PushSelectiveDimensionChainBeforeFact(spark)(p.plan)
    }

    assert(
      broadcastHintSideForPredicate(rewritten, p.lOrderKeyEqOOrder).contains("left"),
      s"expected pruned customer/orders branch to broadcast into lineitem:\n$rewritten"
    )
  }

  test("does not fire when another independent selective literal filter is present") {
    val p = q5LikePlan()
    val xKey = AttributeReference("x_key", DoubleType)()
    val xType = AttributeReference("x_type", StringType)()
    val filteredLargeBranch =
      Filter(EqualTo(xType, Literal("SPECIAL")), StatRel(Seq(xKey, xType), 200000000L))
    val withSecondSelectiveFilter =
      Join(p.plan, filteredLargeBranch, Inner, Some(EqualTo(p.lExtPrice, xKey)), JoinHint.NONE)

    withSQLConf(confKey -> "true") {
      assert(
        PushSelectiveDimensionChainBeforeFact(spark)(withSecondSelectiveFilter)
          .fastEquals(withSecondSelectiveFilter))
    }
  }

  test("does not fire without a literal-filtered seed dimension") {
    val p = q5LikePlan(filterRegion = false)
    withSQLConf(confKey -> "true") {
      assert(PushSelectiveDimensionChainBeforeFact(spark)(p.plan).fastEquals(p.plan))
    }
  }

  test("does not fire when the victim dimension is small") {
    val p = q5LikePlan(customerRows = 50000L)
    withSQLConf(confKey -> "true") {
      assert(PushSelectiveDimensionChainBeforeFact(spark)(p.plan).fastEquals(p.plan))
    }
  }

  test("falls back to leaf scan size when wrapper row counts are zero") {
    val p = q5LikePlan(sizeOnlyStats = true)
    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = PushSelectiveDimensionChainBeforeFact(spark)(p.plan)
    }

    assert(!rewritten.fastEquals(p.plan), "rule should use scan-size fallback for zero rowCount")
    assert(
      findCustomerPrune(stripTopProject(rewritten)).isDefined,
      s"expected customer JOIN (nation JOIN region) subtree in:\n$rewritten")
  }

  test("fact-probe broadcast hint respects Spark auto broadcast threshold") {
    val oneGb = 1L << 30
    val twoGb = 2L << 30
    val fourGb = 4L << 30

    val leftSmaller = sizedInnerJoin(leftBytes = oneGb, rightBytes = fourGb)
    withSQLConf("spark.sql.autoBroadcastJoinThreshold" -> twoGb.toString) {
      assert(broadcastHintSide(FluxFactProbeBroadcastHint(spark)(leftSmaller)).contains("left"))
    }
    withSQLConf("spark.sql.autoBroadcastJoinThreshold" -> (512L << 20).toString) {
      assert(broadcastHintSide(FluxFactProbeBroadcastHint(spark)(leftSmaller)).isEmpty)
    }

    val rightSmaller = sizedInnerJoin(leftBytes = fourGb, rightBytes = oneGb)
    withSQLConf("spark.sql.autoBroadcastJoinThreshold" -> twoGb.toString) {
      assert(broadcastHintSide(FluxFactProbeBroadcastHint(spark)(rightSmaller)).contains("right"))
    }
    withSQLConf("spark.sql.autoBroadcastJoinThreshold" -> "-1") {
      assert(broadcastHintSide(FluxFactProbeBroadcastHint(spark)(rightSmaller)).isEmpty)
    }
  }

  test("fact-probe broadcast threshold uses Spark plan stats after filtering") {
    val oneGb = 1L << 30
    val twoGb = 2L << 30
    val threeGb = 3L << 30
    val twelveGb = 12L << 30
    val sixtyFourGb = 64L << 30

    val projectedSmallSide = sizedInnerJoin(
      leftBytes = twelveGb,
      rightBytes = sixtyFourGb,
      leftStatsBytes = oneGb,
      rightStatsBytes = sixtyFourGb)
    withSQLConf("spark.sql.autoBroadcastJoinThreshold" -> twoGb.toString) {
      assert(
        broadcastHintSide(FluxFactProbeBroadcastHint(spark)(projectedSmallSide)).contains("left"))
    }

    val statsTooLarge = sizedInnerJoin(
      leftBytes = twelveGb,
      rightBytes = sixtyFourGb,
      leftStatsBytes = threeGb,
      rightStatsBytes = sixtyFourGb)
    withSQLConf("spark.sql.autoBroadcastJoinThreshold" -> twoGb.toString) {
      assert(broadcastHintSide(FluxFactProbeBroadcastHint(spark)(statsTooLarge)).isEmpty)
    }

    val rightProjectedSmallSide = sizedInnerJoin(
      leftBytes = sixtyFourGb,
      rightBytes = twelveGb,
      leftStatsBytes = sixtyFourGb,
      rightStatsBytes = oneGb)
    withSQLConf("spark.sql.autoBroadcastJoinThreshold" -> twoGb.toString) {
      assert(
        broadcastHintSide(FluxFactProbeBroadcastHint(spark)(rightProjectedSmallSide))
          .contains("right"))
    }
  }

  test("fact-probe broadcast hint extends the threshold only when FLUX network cost wins") {
    val twoGb = 2L << 30
    val threeGb = 3L << 30
    val fourGb = 4L << 30
    val fiveGb = 5L << 30
    val eightGb = 8L << 30
    val twelveGb = 12L << 30
    val sixtyFourGb = 64L << 30
    val maxBuildKey = "spark.gluten.mpp.factProbeBroadcastHint.maxBuildBytes"
    val partitionsKey = "spark.gluten.mpp.multiExecutor.numPartitions"

    val networkWin = sizedInnerJoin(
      leftBytes = sixtyFourGb,
      rightBytes = twelveGb,
      leftStatsBytes = sixtyFourGb,
      rightStatsBytes = threeGb)
    withSQLConf(
      "spark.sql.autoBroadcastJoinThreshold" -> twoGb.toString,
      maxBuildKey -> fourGb.toString,
      partitionsKey -> "8") {
      assert(broadcastHintSide(FluxFactProbeBroadcastHint(spark)(networkWin)).contains("right"))
    }

    val networkLoss = sizedInnerJoin(
      leftBytes = sixtyFourGb,
      rightBytes = twelveGb,
      leftStatsBytes = eightGb,
      rightStatsBytes = threeGb)
    withSQLConf(
      "spark.sql.autoBroadcastJoinThreshold" -> twoGb.toString,
      maxBuildKey -> fourGb.toString,
      partitionsKey -> "8") {
      assert(broadcastHintSide(FluxFactProbeBroadcastHint(spark)(networkLoss)).isEmpty)
    }

    val overMemoryCeiling = sizedInnerJoin(
      leftBytes = sixtyFourGb,
      rightBytes = twelveGb,
      leftStatsBytes = sixtyFourGb,
      rightStatsBytes = fiveGb)
    withSQLConf(
      "spark.sql.autoBroadcastJoinThreshold" -> twoGb.toString,
      maxBuildKey -> fourGb.toString,
      partitionsKey -> "8") {
      assert(broadcastHintSide(FluxFactProbeBroadcastHint(spark)(overMemoryCeiling)).isEmpty)
    }
  }

  // ---- helpers ----

  private def postCboRewriteCount(experimental: org.apache.spark.sql.ExperimentalMethods): Int =
    experimental.extraOptimizations.count(_.isInstanceOf[PushSelectiveDimensionChainBeforeFact])

  private def postCboHintCount(experimental: org.apache.spark.sql.ExperimentalMethods): Int =
    experimental.extraOptimizations.count(_.isInstanceOf[FluxFactProbeBroadcastHint])

  private def splitAnd(e: Expression): Seq[Expression] = e match {
    case And(l, r) => splitAnd(l) ++ splitAnd(r)
    case other => Seq(other)
  }

  private def stripTopProject(plan: LogicalPlan): LogicalPlan = plan match {
    case Project(_, child) => child
    case other => other
  }

  private def withCanonicalCostConf(body: => Unit): Unit = {
    withSQLConf(
      confKey -> "true",
      "spark.gluten.sql.columnar.pushDimensionChainBeforeFact.maxChainRows" -> "10000000",
      "spark.gluten.sql.columnar.pushDimensionChainBeforeFact.maxChainBytes" -> "1g",
      "spark.sql.autoBroadcastJoinThreshold" -> "12g",
      "spark.gluten.mpp.multiExecutor.numPartitions" -> "32"
    )(body)
  }

  private def isBelowJoinType(root: LogicalPlan, target: LogicalPlan, joinType: JoinType): Boolean =
    root.exists {
      case join: Join if join.joinType == joinType =>
        join.children.exists(child => child.exists(_ eq target))
      case _ => false
    }

  private def containsJoinWithConditionAndHint(root: LogicalPlan, target: Join): Boolean =
    root.exists {
      case join: Join =>
        join.joinType == target.joinType && join.condition == target.condition &&
        join.hint == target.hint
      case _ => false
    }

  private def hasProjectPrefixWithCarriedAttr(
      root: LogicalPlan,
      original: Seq[Attribute],
      carried: Attribute): Boolean =
    root.exists {
      case Project(projectList, _) if projectList.length > original.length =>
        projectList.take(original.length).map(_.toAttribute) == original &&
        projectList.drop(original.length).exists(_.toAttribute.exprId == carried.exprId)
      case _ => false
    }

  private def containsPredicate(plan: LogicalPlan, target: Expression): Boolean =
    plan.expressions.exists(e => splitAnd(e).exists(_.semanticEquals(target))) ||
      plan.children.exists(containsPredicate(_, target))

  private def countPredicate(plan: LogicalPlan, target: Expression): Int =
    plan.expressions.map(e => splitAnd(e).count(p => samePredicate(p, target))).sum +
      plan.children.map(countPredicate(_, target)).sum

  private def projectOutputNameSets(plan: LogicalPlan): Set[Set[String]] =
    plan.collect { case Project(projectList, _) => projectList.map(_.name).toSet }.toSet

  private def hasLeftBroadcastJoinForProjectedOutput(
      plan: LogicalPlan,
      outputNames: Set[String]): Boolean =
    plan.collect {
      case Project(projectList, Join(_, _, Inner, _, JoinHint(Some(leftHint), None)))
          if projectList.map(_.name).toSet == outputNames &&
            leftHint.strategy.contains(BROADCAST) =>
        true
    }.nonEmpty

  private def addBroadcastHints(plan: LogicalPlan): LogicalPlan =
    plan.transform {
      case join @ Join(_, _, Inner, _, JoinHint.NONE) =>
        join.copy(hint = JoinHint(Some(HintInfo(strategy = Some(BROADCAST))), None))
    }

  private def broadcastHintSide(plan: LogicalPlan): Option[String] = plan match {
    case Join(_, _, Inner, _, JoinHint(leftHint, rightHint)) =>
      if (leftHint.exists(_.strategy.contains(BROADCAST))) {
        Some("left")
      } else if (rightHint.exists(_.strategy.contains(BROADCAST))) {
        Some("right")
      } else {
        None
      }
    case other => fail(s"expected inner join, got:\n$other")
  }

  private def broadcastHintSideForPredicate(plan: LogicalPlan, target: Expression): Option[String] =
    plan
      .collectFirst {
        case Join(_, _, Inner, Some(cond), JoinHint(leftHint, rightHint))
            if splitAnd(cond).exists(p => samePredicate(p, target)) =>
          if (leftHint.exists(_.strategy.contains(BROADCAST))) {
            "left"
          } else if (rightHint.exists(_.strategy.contains(BROADCAST))) {
            "right"
          } else {
            "none"
          }
      }
      .flatMap(side => if (side == "none") None else Some(side))

  private def findVictimPrunedByChainSeed(
      plan: LogicalPlan,
      victimKey: String,
      chainKey: String,
      seedKey: String): Option[LogicalPlan] = plan match {
    case j @ Join(victim, dimension, Inner, _, _)
        if victim.output.exists(_.name == victimKey) &&
          dimension.exists(_.output.exists(_.name == chainKey)) &&
          dimension.exists(_.output.exists(_.name == seedKey)) =>
      Some(j)
    case _ =>
      plan.children.iterator
        .flatMap(findVictimPrunedByChainSeed(_, victimKey, chainKey, seedKey))
        .toList
        .headOption
  }

  private def sizedInnerJoin(leftBytes: Long, rightBytes: Long): LogicalPlan = {
    val leftKey = AttributeReference("left_key", LongType)()
    val rightKey = AttributeReference("right_key", LongType)()
    val left = sizedLogicalRelation(Seq(leftKey), leftBytes)
    val right = sizedLogicalRelation(Seq(rightKey), rightBytes)
    Join(left, right, Inner, Some(EqualTo(leftKey, rightKey)), JoinHint.NONE)
  }

  private def sizedInnerJoin(
      leftBytes: Long,
      rightBytes: Long,
      leftStatsBytes: BigInt,
      rightStatsBytes: BigInt): LogicalPlan = {
    val leftKey = AttributeReference("left_key", LongType)()
    val rightKey = AttributeReference("right_key", LongType)()
    val left = StatsOverride(sizedLogicalRelation(Seq(leftKey), leftBytes), leftStatsBytes)
    val right = StatsOverride(sizedLogicalRelation(Seq(rightKey), rightBytes), rightStatsBytes)
    Join(left, right, Inner, Some(EqualTo(leftKey, rightKey)), JoinHint.NONE)
  }

  private def sizedLogicalRelation(attrs: Seq[AttributeReference], bytes: Long): LogicalPlan = {
    SparkShimLoader.getSparkShims.createLogicalRelation(
      SizedRelation(
        spark.sqlContext,
        StructType(attrs.map(a => StructField(a.name, a.dataType, a.nullable, a.metadata))),
        bytes),
      attrs,
      catalogTable = None,
      isStreaming = false
    )
  }

  private def samePredicate(left: Expression, right: Expression): Boolean =
    left.semanticEquals(right) || ((left, right) match {
      case (EqualTo(ll, lr), EqualTo(rl, rr)) =>
        ll.semanticEquals(rr) && lr.semanticEquals(rl)
      case _ => false
    })

  /** Find the `customer JOIN (nation JOIN region)` subtree anywhere in the tree. */
  private def findCustomerPrune(plan: LogicalPlan): Option[LogicalPlan] = plan match {
    case j @ Join(left, Join(_, _, Inner, _, _), Inner, _, _)
        if left.output.exists(_.name == "c_custkey") =>
      Some(j)
    case _ => plan.children.iterator.flatMap(findCustomerPrune).toList.headOption
  }

  private case class StatRel(rels: Seq[Attribute], rows: Long) extends LeafNode {
    override def output: Seq[Attribute] = rels
    override def computeStats(): Statistics =
      Statistics(sizeInBytes = BigInt(rows) * 16, rowCount = Some(BigInt(rows)))
  }

  private case class SizeOnlyRel(rels: Seq[Attribute], bytes: BigInt) extends LeafNode {
    override def output: Seq[Attribute] = rels
    override def computeStats(): Statistics =
      Statistics(sizeInBytes = bytes, rowCount = Some(BigInt(0)))
  }

  private case class StatsOverride(child: LogicalPlan, bytes: BigInt) extends UnaryNode {
    override def output: Seq[Attribute] = child.output
    override def stats: Statistics = Statistics(sizeInBytes = bytes)
    override protected def withNewChildInternal(newChild: LogicalPlan): LogicalPlan =
      copy(child = newChild)
  }

  private case class SizedRelation(
      sqlContext: SQLContext,
      schema: StructType,
      override val sizeInBytes: Long)
    extends BaseRelation

  private case class Q5LikePlan(
      plan: LogicalPlan,
      customer: LogicalPlan,
      orders: LogicalPlan,
      lineitem: LogicalPlan,
      supplier: LogicalPlan,
      nation: LogicalPlan,
      region: LogicalPlan,
      cNationKey: Attribute,
      nNationKey: Attribute,
      sNationKey: Attribute,
      nName: Attribute,
      lExtPrice: Attribute,
      lDiscount: Attribute,
      lOrderKeyEqOOrder: EqualTo,
      lSuppKeyEqSSuppKey: EqualTo,
      cNationKeyEqSNationKey: EqualTo)

  private case class DirectChainVictimPlan(plan: LogicalPlan, lOrderKeyEqOOrderKey: EqualTo)

  private case class CanonicalMixedProbeSpinePlan(
      plan: LogicalPlan,
      victim: LogicalPlan,
      dedupInner: Join,
      anti: Join,
      sName: Attribute,
      expectedPruneHint: JoinHint)

  private def canonicalMixedProbeSpinePlan(
      antiResidual: Boolean = false,
      rhsReferencesDimension: Boolean = false,
      deduplicateInnerRhs: Boolean = true,
      extraInnerGroupingKey: Boolean = false,
      lowNdvWrappers: Boolean = false,
      probeUnaryDepth: Int = 1,
      extraPruneKeyFromInnerRhs: Boolean = false,
      firstWrapperType: JoinType = Inner,
      outerWrapperType: JoinType = LeftAnti): CanonicalMixedProbeSpinePlan = {
    val sSuppKey = AttributeReference("s_suppkey", LongType)()
    val sName = AttributeReference("s_name", StringType)()
    val sNationKey = AttributeReference("s_nationkey", IntegerType)()
    val nNationKey = AttributeReference("n_nationkey", IntegerType)()
    val nName = AttributeReference("n_name", StringType)()
    val lSuppKey = AttributeReference("l_suppkey", LongType)()
    val lOrderKey = AttributeReference("l_orderkey", LongType)()
    val firstStateOrderKey = AttributeReference("state1_orderkey", LongType)()
    val firstStateSuppKey = AttributeReference("state1_suppkey", LongType)()
    val secondStateOrderKey = AttributeReference("state2_orderkey", LongType)()
    val secondStateSuppKey = AttributeReference("state2_suppkey", LongType)()
    val oOrderKey = AttributeReference("o_orderkey", LongType)()

    val supplier = StatRel(Seq(sSuppKey, sName, sNationKey), 300000000L)
    val nation =
      Filter(EqualTo(nName, Literal("SAUDI ARABIA")), StatRel(Seq(nNationKey, nName), 25L))
    val lineitem = StatRel(Seq(lOrderKey, lSuppKey), 180000000000L)

    val firstStateInput = StatRel(Seq(firstStateOrderKey, firstStateSuppKey), 180000000000L)
    val firstState =
      if (deduplicateInnerRhs) {
        val keys =
          if (lowNdvWrappers) {
            Seq(firstStateSuppKey)
          } else if (extraInnerGroupingKey) {
            Seq(firstStateOrderKey, firstStateSuppKey)
          } else {
            Seq(firstStateOrderKey)
          }
        Aggregate(keys, keys, firstStateInput)
      } else {
        firstStateInput
      }
    val firstCondition =
      if (lowNdvWrappers) EqualTo(lSuppKey, firstStateSuppKey)
      else EqualTo(lOrderKey, firstStateOrderKey)
    val rightBroadcast = JoinHint(None, Some(HintInfo(strategy = Some(BROADCAST))))
    val dedupInner =
      Join(lineitem, firstState, firstWrapperType, Some(firstCondition), rightBroadcast)
    val projectedDedupInner = (0 until probeUnaryDepth).foldLeft(dedupInner: LogicalPlan) {
      case (child, _) => Project(child.output, child)
    }

    val secondStateAttrs =
      Seq(secondStateOrderKey, secondStateSuppKey) ++
        (if (rhsReferencesDimension) Seq(sName) else Seq.empty)
    val secondStateInput = StatRel(secondStateAttrs, 180000000000L)
    val secondGrouping =
      if (antiResidual) {
        Seq(secondStateOrderKey, secondStateSuppKey)
      } else if (lowNdvWrappers) {
        Seq(secondStateSuppKey)
      } else {
        Seq(secondStateOrderKey)
      }
    val secondState = Aggregate(secondGrouping, secondGrouping, secondStateInput)
    val secondEquality =
      if (lowNdvWrappers) EqualTo(lSuppKey, secondStateSuppKey)
      else EqualTo(lOrderKey, secondStateOrderKey)
    val secondCondition: Expression =
      if (antiResidual) And(secondEquality, GreaterThan(lSuppKey, secondStateSuppKey))
      else secondEquality
    val anti = Join(
      projectedDedupInner,
      secondState,
      outerWrapperType,
      Some(secondCondition),
      rightBroadcast)
    val victim = Project(anti.output, anti)

    val chainLeftHint = JoinHint(Some(HintInfo(strategy = Some(BROADCAST))), None)
    val supplierVictimCondition: Expression =
      if (extraPruneKeyFromInnerRhs) {
        And(EqualTo(sSuppKey, lSuppKey), EqualTo(sSuppKey, firstStateOrderKey))
      } else {
        EqualTo(sSuppKey, lSuppKey)
      }
    val supplierVictim =
      Join(supplier, victim, Inner, Some(supplierVictimCondition), chainLeftHint)
    val withOrders = Join(
      supplierVictim,
      StatRel(Seq(oOrderKey), 45000000000L),
      Inner,
      Some(EqualTo(lOrderKey, oOrderKey)),
      JoinHint.NONE)
    val withNation =
      Join(withOrders, nation, Inner, Some(EqualTo(sNationKey, nNationKey)), JoinHint.NONE)

    CanonicalMixedProbeSpinePlan(
      Project(Seq(sName), withNation),
      victim,
      dedupInner,
      anti,
      sName,
      JoinHint(None, chainLeftHint.leftHint))
  }

  private def pairedExistenceStateProbeSpinePlan(): CanonicalMixedProbeSpinePlan = {
    val sSuppKey = AttributeReference("s_suppkey", LongType)()
    val sName = AttributeReference("s_name", StringType)()
    val sNationKey = AttributeReference("s_nationkey", IntegerType)()
    val nNationKey = AttributeReference("n_nationkey", IntegerType)()
    val nName = AttributeReference("n_name", StringType)()
    val lSuppKey = AttributeReference("l_suppkey", LongType)()
    val lOrderKey = AttributeReference("l_orderkey", LongType)()
    val stateOrderKey = AttributeReference("state_orderkey", LongType)()
    val stateSuppKey = AttributeReference("state_suppkey", LongType)()
    val oOrderKey = AttributeReference("o_orderkey", LongType)()

    val supplier = StatRel(Seq(sSuppKey, sName, sNationKey), 300000000L)
    val nation =
      Filter(EqualTo(nName, Literal("SAUDI ARABIA")), StatRel(Seq(nNationKey, nName), 25L))
    val lineitem = StatRel(Seq(lOrderKey, lSuppKey), 180000000000L)
    val stateInput = StatRel(Seq(stateOrderKey, stateSuppKey), 180000000000L)
    val delayedMin = Alias(Min(stateSuppKey).toAggregateExpression(), "delayed_min_suppkey")()
    val state = Aggregate(Seq(stateOrderKey), Seq(stateOrderKey, delayedMin), stateInput)
    val pairedCondition =
      And(EqualTo(lOrderKey, stateOrderKey), EqualTo(lSuppKey, delayedMin.toAttribute))
    val rightBroadcast = JoinHint(None, Some(HintInfo(strategy = Some(BROADCAST))))
    val pairedJoin = Join(lineitem, state, LeftSemi, Some(pairedCondition), rightBroadcast)
    val victim = Project(lineitem.output, pairedJoin)

    val chainLeftHint = JoinHint(Some(HintInfo(strategy = Some(BROADCAST))), None)
    val supplierVictim =
      Join(supplier, victim, Inner, Some(EqualTo(sSuppKey, lSuppKey)), chainLeftHint)
    val withOrders = Join(
      supplierVictim,
      StatRel(Seq(oOrderKey), 45000000000L),
      Inner,
      Some(EqualTo(lOrderKey, oOrderKey)),
      JoinHint.NONE)
    val withNation =
      Join(withOrders, nation, Inner, Some(EqualTo(sNationKey, nNationKey)), JoinHint.NONE)

    CanonicalMixedProbeSpinePlan(
      Project(Seq(sName), withNation),
      victim,
      pairedJoin,
      pairedJoin,
      sName,
      JoinHint(None, chainLeftHint.leftHint))
  }

  private def directChainVictimPlan(
      chainRows: Long = 2000000L,
      victimRows: Long = 6000000000L,
      filterOrders: Boolean = false,
      includeOrderState: Boolean = false,
      filterOrderState: Boolean = false): DirectChainVictimPlan = {
    val sSuppKey = AttributeReference("s_suppkey", LongType)()
    val sNationKey = AttributeReference("s_nationkey", IntegerType)()
    val nNationKey = AttributeReference("n_nationkey", IntegerType)()
    val nName = AttributeReference("n_name", StringType)()
    val lSuppKey = AttributeReference("l_suppkey", LongType)()
    val lOrderKey = AttributeReference("l_orderkey", LongType)()
    val oOrderKey = AttributeReference("o_orderkey", LongType)()
    val oStatus = AttributeReference("o_orderstatus", StringType)()
    val stateOrderKey = AttributeReference("state_orderkey", LongType)()
    val stateSuppKey = AttributeReference("state_suppkey", LongType)()
    val stateType = AttributeReference("state_type", StringType)()

    val supplier = StatRel(Seq(sSuppKey, sNationKey), chainRows)
    val nation =
      Filter(EqualTo(nName, Literal("SAUDI ARABIA")), StatRel(Seq(nNationKey, nName), 25L))
    val lineitem = StatRel(Seq(lSuppKey, lOrderKey), victimRows)
    val orders: LogicalPlan =
      if (filterOrders) {
        Project(
          Seq(oOrderKey),
          Filter(EqualTo(oStatus, Literal("F")), StatRel(Seq(oOrderKey, oStatus), 1500000000L)))
      } else {
        StatRel(Seq(oOrderKey), 1500000000L)
      }

    val sNationEqNNation = EqualTo(sNationKey, nNationKey)
    val sSuppKeyEqLSuppKey = EqualTo(sSuppKey, lSuppKey)
    val lOrderKeyEqOOrderKey = EqualTo(lOrderKey, oOrderKey)

    val supplierLineitem = Join(supplier, lineitem, Inner, Some(sSuppKeyEqLSuppKey), JoinHint.NONE)
    val withOrders =
      Join(supplierLineitem, orders, Inner, Some(lOrderKeyEqOOrderKey), JoinHint.NONE)
    val withOrderState =
      if (includeOrderState) {
        val stateInput = StatRel(Seq(stateOrderKey, stateSuppKey, stateType), victimRows)
        val state: LogicalPlan =
          if (filterOrderState) {
            Filter(EqualTo(stateType, Literal("MATCH")), stateInput)
          } else {
            stateInput
          }
        Join(
          withOrders,
          state,
          Inner,
          Some(And(EqualTo(lOrderKey, stateOrderKey), EqualTo(lSuppKey, stateSuppKey))),
          JoinHint.NONE
        )
      } else {
        withOrders
      }
    val withNation = Join(withOrderState, nation, Inner, Some(sNationEqNNation), JoinHint.NONE)
    DirectChainVictimPlan(withNation, lOrderKeyEqOOrderKey)
  }

  private def q5LikePlan(
      filterRegion: Boolean = true,
      customerRows: Long = 150000000L,
      sizeOnlyStats: Boolean = false): Q5LikePlan = {
    val cCustKey = AttributeReference("c_custkey", LongType)()
    val cNationKey = AttributeReference("c_nationkey", IntegerType)()
    val oOrderKey = AttributeReference("o_orderkey", LongType)()
    val oCustKey = AttributeReference("o_custkey", LongType)()
    val oOrderDate = AttributeReference("o_orderdate", DateType)()
    val lOrderKey = AttributeReference("l_orderkey", LongType)()
    val lSuppKey = AttributeReference("l_suppkey", LongType)()
    val lExtPrice = AttributeReference("l_extendedprice", DoubleType)()
    val lDiscount = AttributeReference("l_discount", DoubleType)()
    val sSuppKey = AttributeReference("s_suppkey", LongType)()
    val sNationKey = AttributeReference("s_nationkey", IntegerType)()
    val nNationKey = AttributeReference("n_nationkey", IntegerType)()
    val nRegionKey = AttributeReference("n_regionkey", IntegerType)()
    val nName = AttributeReference("n_name", StringType)()
    val rRegionKey = AttributeReference("r_regionkey", IntegerType)()
    val rName = AttributeReference("r_name", StringType)()

    def statRel(attrs: Seq[Attribute], rows: Long): LogicalPlan =
      if (sizeOnlyStats) SizeOnlyRel(attrs, BigInt(rows) * 16) else StatRel(attrs, rows)

    val customer = statRel(Seq(cCustKey, cNationKey), customerRows)
    val orders = statRel(Seq(oOrderKey, oCustKey, oOrderDate), 1500000000L)
    val lineitem = statRel(Seq(lOrderKey, lSuppKey, lExtPrice, lDiscount), 6000000000L)
    val supplier = statRel(Seq(sSuppKey, sNationKey), 2000000L)
    val nation = statRel(Seq(nNationKey, nRegionKey, nName), 25L)
    val regionBase = statRel(Seq(rRegionKey, rName), 5L)
    val region: LogicalPlan =
      if (filterRegion) Filter(EqualTo(rName, Literal("ASIA")), regionBase) else regionBase

    val cCustEqOCust = EqualTo(cCustKey, oCustKey)
    val lOrderEqOOrder = EqualTo(lOrderKey, oOrderKey)
    val lSuppEqSSupp = EqualTo(lSuppKey, sSuppKey)
    val cNationEqSNation = EqualTo(cNationKey, sNationKey)
    val sNationEqNNation = EqualTo(sNationKey, nNationKey)
    val nRegionEqRRegion = EqualTo(nRegionKey, rRegionKey)

    // "bad" Spark default order: customer JOIN orders first (full 150M customer), supplier early.
    val j1 = Join(customer, orders, Inner, Some(cCustEqOCust), JoinHint.NONE)
    val j2 = Join(j1, lineitem, Inner, Some(lOrderEqOOrder), JoinHint.NONE)
    val j3 = Join(j2, supplier, Inner, Some(And(lSuppEqSSupp, cNationEqSNation)), JoinHint.NONE)
    val j4 = Join(j3, nation, Inner, Some(sNationEqNNation), JoinHint.NONE)
    val j5 = Join(j4, region, Inner, Some(nRegionEqRRegion), JoinHint.NONE)

    Q5LikePlan(
      j5,
      customer,
      orders,
      lineitem,
      supplier,
      nation,
      region,
      cNationKey,
      nNationKey,
      sNationKey,
      nName,
      lExtPrice,
      lDiscount,
      lOrderEqOOrder,
      lSuppEqSSupp,
      cNationEqSNation
    )
  }
}
