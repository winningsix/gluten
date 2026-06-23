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

import org.apache.spark.sql.GlutenQueryTest
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, EqualTo}
import org.apache.spark.sql.catalyst.expressions.{Expression, Literal}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{BROADCAST, Filter, HintInfo, Join}
import org.apache.spark.sql.catalyst.plans.logical.{JoinHint, LeafNode, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{IntegerType, StringType}

class SelectiveDimensionJoinReorderSuite extends GlutenQueryTest with SharedSparkSession {

  private val confKey = GlutenConfig.SELECTIVE_DIMENSION_JOIN_REORDER_ENABLED.key
  private val q11BroadcastThreshold = (2L << 30).toString

  test("selective dimension join reorder is disabled by default") {
    val testPlan = q11LikePlan(projectNation = true)

    assert(SelectiveDimensionJoinReorder(spark)(testPlan.plan).fastEquals(testPlan.plan))
  }

  test("auto-enables selective dimension join reorder under MPP") {
    val testPlan = q11LikePlan()
    val experimental = spark.experimental
    val before = experimental.extraOptimizations
    try {
      var rewritten: LogicalPlan = null
      withSQLConf(
        "spark.gluten.mpp.enabled" -> "true",
        "spark.sql.autoBroadcastJoinThreshold" -> q11BroadcastThreshold) {
        rewritten = SelectiveDimensionJoinReorder(spark)(testPlan.plan)
      }
      assertReorderedCore(rewritten, testPlan)
    } finally {
      experimental.synchronized {
        experimental.extraOptimizations = before
      }
    }
  }

  test("explicit false disables selective dimension join reorder under MPP") {
    val testPlan = q11LikePlan(projectNation = true)

    withSQLConf(confKey -> "false", "spark.gluten.mpp.enabled" -> "true") {
      assert(SelectiveDimensionJoinReorder(spark)(testPlan.plan).fastEquals(testPlan.plan))
    }
  }

  test("self-registers exactly once for the post-CBO optimizer pass") {
    val experimental = spark.experimental
    val before = experimental.extraOptimizations
    try {
      experimental.synchronized {
        experimental.extraOptimizations = before.filterNot(
          r =>
            r.isInstanceOf[SelectiveDimensionJoinReorder] ||
              r.isInstanceOf[MppFactProbeBroadcastHint]) :+ MppFactProbeBroadcastHint(spark)
      }

      withSQLConf(confKey -> "true") {
        val rule = SelectiveDimensionJoinReorder(spark)
        assert(postCboRewriteCount == 1)
        val rewriteIndex =
          experimental.extraOptimizations.indexWhere(_.isInstanceOf[SelectiveDimensionJoinReorder])
        val hintIndex =
          experimental.extraOptimizations.indexWhere(_.isInstanceOf[MppFactProbeBroadcastHint])
        assert(rewriteIndex >= 0 && hintIndex > rewriteIndex)
        rule(q11LikePlan().plan)
        assert(postCboRewriteCount == 1)
      }
    } finally {
      experimental.synchronized {
        experimental.extraOptimizations = before
      }
    }
  }

  test("reorders Q11-like supplier and nation join before partsupp") {
    val testPlan = q11LikePlan()

    var rewritten: LogicalPlan = null
    withSQLConf(
      confKey -> "true",
      "spark.sql.autoBroadcastJoinThreshold" -> q11BroadcastThreshold) {
      rewritten = SelectiveDimensionJoinReorder(spark)(testPlan.plan)
    }

    rewritten match {
      case Join(
            outerLeft,
            Join(innerLeft, innerRight, Inner, Some(innerCondition), innerHint),
            Inner,
            Some(outerCondition),
            outerHint) =>
        assert(outerLeft.fastEquals(testPlan.partsupp))
        assert(innerLeft.fastEquals(testPlan.supplier))
        assert(innerRight.fastEquals(testPlan.nation))
        assert(innerHint == JoinHint.NONE)
        assert(isRightBroadcastHint(outerHint))
        assert(innerCondition.semanticEquals(testPlan.supplierNationCondition))
        assert(outerCondition.semanticEquals(testPlan.partSuppSupplierCondition))
      case other =>
        fail(s"Unexpected reordered plan:\n$other")
    }
  }

  test("reorders Q11-like plan with projected filtered nation before partsupp") {
    val testPlan = q11LikePlan(projectNation = true)

    var rewritten: LogicalPlan = null
    withSQLConf(
      confKey -> "true",
      "spark.sql.autoBroadcastJoinThreshold" -> q11BroadcastThreshold) {
      rewritten = SelectiveDimensionJoinReorder(spark)(testPlan.plan)
    }

    rewritten match {
      case Join(
            outerLeft,
            Join(innerLeft, innerRight, Inner, Some(innerCondition), innerHint),
            Inner,
            Some(outerCondition),
            outerHint) =>
        assert(outerLeft.fastEquals(testPlan.partsupp))
        assert(innerLeft.fastEquals(testPlan.supplier))
        assert(innerRight.fastEquals(testPlan.nation))
        assert(innerHint == JoinHint.NONE)
        assert(isRightBroadcastHint(outerHint))
        assert(innerCondition.semanticEquals(testPlan.supplierNationCondition))
        assert(outerCondition.semanticEquals(testPlan.partSuppSupplierCondition))
      case other =>
        fail(s"Unexpected reordered plan:\n$other")
    }
  }

  test("reorders Q11-like plan with projected fact-dimension branch") {
    val testPlan = q11LikePlan(projectFactDimension = true)

    var rewritten: LogicalPlan = null
    withSQLConf(
      confKey -> "true",
      "spark.sql.autoBroadcastJoinThreshold" -> q11BroadcastThreshold) {
      rewritten = SelectiveDimensionJoinReorder(spark)(testPlan.plan)
    }

    assert(rewritten.output == testPlan.plan.output)
    rewritten match {
      case Project(projectList, child) =>
        assert(projectList.map(_.toAttribute) == testPlan.plan.output)
        assertReorderedCore(child, testPlan)
      case other =>
        fail(s"Unexpected reordered plan:\n$other")
    }
  }

  test("reorders Q11-like plan after broadcast hints were attached") {
    val testPlan = q11LikePlan(projectFactDimension = true, broadcastHints = true)

    var rewritten: LogicalPlan = null
    withSQLConf(
      confKey -> "true",
      "spark.sql.autoBroadcastJoinThreshold" -> q11BroadcastThreshold) {
      rewritten = SelectiveDimensionJoinReorder(spark)(testPlan.plan)
    }

    assert(rewritten.output == testPlan.plan.output)
    rewritten match {
      case Project(_, child) =>
        assertReorderedCore(child, testPlan)
      case other =>
        fail(s"Unexpected reordered plan:\n$other")
    }
  }

  test("does not broadcast reordered dimension chain above the Spark threshold") {
    val testPlan = q11LikePlan()

    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true", "spark.sql.autoBroadcastJoinThreshold" -> "0") {
      rewritten = SelectiveDimensionJoinReorder(spark)(testPlan.plan)
    }

    rewritten match {
      case Join(_, Join(_, _, Inner, _, _), Inner, _, outerHint) =>
        assert(outerHint == JoinHint.NONE)
      case other =>
        fail(s"Unexpected reordered plan:\n$other")
    }
  }

  test("does not reorder when the dimension is not filtered") {
    val testPlan = q11LikePlan(filterNation = false, projectNation = true)

    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = SelectiveDimensionJoinReorder(spark)(testPlan.plan)
    }

    assert(rewritten.fastEquals(testPlan.plan))
  }

  private case class Q11LikePlan(
      plan: LogicalPlan,
      partsupp: LogicalPlan,
      supplier: LogicalPlan,
      nation: LogicalPlan,
      partSuppSupplierCondition: Expression,
      supplierNationCondition: Expression)

  private def q11LikePlan(
      filterNation: Boolean = true,
      projectNation: Boolean = false,
      projectFactDimension: Boolean = false,
      broadcastHints: Boolean = false): Q11LikePlan = {
    val psSuppKey = AttributeReference("ps_suppkey", IntegerType)()
    val sSuppKey = AttributeReference("s_suppkey", IntegerType)()
    val sNationKey = AttributeReference("s_nationkey", IntegerType)()
    val nNationKey = AttributeReference("n_nationkey", IntegerType)()
    val nName = AttributeReference("n_name", StringType)()

    val partsupp = StatRel(Seq(psSuppKey), rows = 800000000L)
    val supplier = StatRel(Seq(sSuppKey, sNationKey), rows = 10000000L)
    val nationBase = StatRel(Seq(nNationKey, nName), rows = 25L)
    val filteredNation = if (filterNation) {
      Filter(EqualTo(nName, Literal("GERMANY")), nationBase)
    } else {
      nationBase
    }
    val nation = if (projectNation) {
      Project(Seq(nNationKey), filteredNation)
    } else {
      filteredNation
    }

    val partSuppSupplierCondition = EqualTo(psSuppKey, sSuppKey)
    val supplierNationCondition = EqualTo(sNationKey, nNationKey)
    val broadcastHint = HintInfo(strategy = Some(BROADCAST))
    val joinHint = if (broadcastHints) {
      JoinHint(None, Some(broadcastHint))
    } else {
      JoinHint.NONE
    }
    val factDimensionJoin =
      Join(partsupp, supplier, Inner, Some(partSuppSupplierCondition), joinHint)
    val factDimension = if (projectFactDimension) {
      Project(Seq(psSuppKey, sNationKey), factDimensionJoin)
    } else {
      factDimensionJoin
    }
    val plan = Join(factDimension, nation, Inner, Some(supplierNationCondition), joinHint)

    Q11LikePlan(
      plan,
      partsupp,
      supplier,
      nation,
      partSuppSupplierCondition,
      supplierNationCondition)
  }

  private def assertReorderedCore(rewritten: LogicalPlan, testPlan: Q11LikePlan): Unit = {
    rewritten match {
      case Join(
            outerLeft,
            Join(innerLeft, innerRight, Inner, Some(innerCondition), innerHint),
            Inner,
            Some(outerCondition),
            outerHint) =>
        assert(outerLeft.fastEquals(testPlan.partsupp))
        assert(innerLeft.fastEquals(testPlan.supplier))
        assert(innerRight.fastEquals(testPlan.nation))
        assert(innerHint == JoinHint.NONE)
        assert(isRightBroadcastHint(outerHint))
        assert(innerCondition.semanticEquals(testPlan.supplierNationCondition))
        assert(outerCondition.semanticEquals(testPlan.partSuppSupplierCondition))
      case other =>
        fail(s"Unexpected reordered plan:\n$other")
    }
  }

  private def isRightBroadcastHint(hint: JoinHint): Boolean =
    hint.leftHint.isEmpty && hint.rightHint.exists(_.strategy.contains(BROADCAST))

  private case class StatRel(attrs: Seq[Attribute], rows: Long) extends LeafNode {
    override def output: Seq[Attribute] = attrs
    override def computeStats(): Statistics =
      Statistics(sizeInBytes = BigInt(rows) * 16, rowCount = Some(BigInt(rows)))
  }

  private def postCboRewriteCount: Int =
    spark.experimental.extraOptimizations.count(_.isInstanceOf[SelectiveDimensionJoinReorder])
}
