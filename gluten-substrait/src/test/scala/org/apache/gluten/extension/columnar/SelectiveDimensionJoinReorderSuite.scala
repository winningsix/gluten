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
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, EqualTo, Expression, Literal}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{Filter, Join, JoinHint, LocalRelation, LogicalPlan, Project}
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{IntegerType, StringType}

class SelectiveDimensionJoinReorderSuite extends GlutenQueryTest with SharedSparkSession {

  test("selective dimension join reorder is disabled by default") {
    val testPlan = q11LikePlan(projectNation = true)

    assert(SelectiveDimensionJoinReorder(spark)(testPlan.plan).fastEquals(testPlan.plan))
  }

  test("reorders Q11-like supplier and nation join before partsupp") {
    val testPlan = q11LikePlan()

    var rewritten: LogicalPlan = null
    withSQLConf(GlutenConfig.SELECTIVE_DIMENSION_JOIN_REORDER_ENABLED.key -> "true") {
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
        assert(outerHint == JoinHint.NONE)
        assert(innerCondition.semanticEquals(testPlan.supplierNationCondition))
        assert(outerCondition.semanticEquals(testPlan.partSuppSupplierCondition))
      case other =>
        fail(s"Unexpected reordered plan:\n$other")
    }
  }

  test("reorders Q11-like plan with projected filtered nation before partsupp") {
    val testPlan = q11LikePlan(projectNation = true)

    var rewritten: LogicalPlan = null
    withSQLConf(GlutenConfig.SELECTIVE_DIMENSION_JOIN_REORDER_ENABLED.key -> "true") {
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
        assert(outerHint == JoinHint.NONE)
        assert(innerCondition.semanticEquals(testPlan.supplierNationCondition))
        assert(outerCondition.semanticEquals(testPlan.partSuppSupplierCondition))
      case other =>
        fail(s"Unexpected reordered plan:\n$other")
    }
  }

  test("does not reorder when the dimension is not filtered") {
    val testPlan = q11LikePlan(filterNation = false, projectNation = true)

    var rewritten: LogicalPlan = null
    withSQLConf(GlutenConfig.SELECTIVE_DIMENSION_JOIN_REORDER_ENABLED.key -> "true") {
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
      projectNation: Boolean = false): Q11LikePlan = {
    val psSuppKey = AttributeReference("ps_suppkey", IntegerType)()
    val sSuppKey = AttributeReference("s_suppkey", IntegerType)()
    val sNationKey = AttributeReference("s_nationkey", IntegerType)()
    val nNationKey = AttributeReference("n_nationkey", IntegerType)()
    val nName = AttributeReference("n_name", StringType)()

    val partsupp = LocalRelation(psSuppKey)
    val supplier = LocalRelation(sSuppKey, sNationKey)
    val nationBase = LocalRelation(nNationKey, nName)
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
    val plan = Join(
      Join(partsupp, supplier, Inner, Some(partSuppSupplierCondition), JoinHint.NONE),
      nation,
      Inner,
      Some(supplierNationCondition),
      JoinHint.NONE)

    Q11LikePlan(
      plan,
      partsupp,
      supplier,
      nation,
      partSuppSupplierCondition,
      supplierNationCondition)
  }
}
