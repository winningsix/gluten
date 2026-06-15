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

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, AttributeReference}
import org.apache.spark.sql.catalyst.expressions.{EqualTo, GreaterThan, Literal}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Join, JoinHint, LocalRelation}
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Project}
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.IntegerType

class PruneRedundantLeftSemiFiltersSuite extends QueryTest with SharedSparkSession {

  test("removes duplicate left-semi filters connected by an inner equi join") {
    val p = redundantSemiPlan()
    val rewritten = PruneRedundantLeftSemiFilters(spark)(p.plan)

    assert(leftSemiCount(rewritten) == 1, s"expected one semi filter:\n$rewritten")
    assert(rewritten.output == p.plan.output)
  }

  test("looks through attribute-only project around inner-join clusters") {
    val p = redundantSemiPlan(projectWrapsLeftCluster = true)
    val rewritten = PruneRedundantLeftSemiFilters(spark)(p.plan)

    assert(leftSemiCount(rewritten) == 1, s"expected one semi filter:\n$rewritten")
    assert(rewritten.output == p.plan.output)
  }

  test("can be disabled") {
    val p = redundantSemiPlan()
    withSQLConf("spark.gluten.mpp.pruneRedundantLeftSemiFilters" -> "false") {
      val rewritten = PruneRedundantLeftSemiFilters(spark)(p.plan)
      assert(leftSemiCount(rewritten) == 2, s"rule should be disabled:\n$rewritten")
    }
  }

  test("keeps semi filters whose probe keys are not equality-connected") {
    val p = redundantSemiPlan(connectProbeKeys = false)
    val rewritten = PruneRedundantLeftSemiFilters(spark)(p.plan)

    assert(leftSemiCount(rewritten) == 2, s"unconnected keys are not redundant:\n$rewritten")
  }

  test("keeps semi filters with residual predicates") {
    val p = redundantSemiPlan(secondSemiHasResidual = true)
    val rewritten = PruneRedundantLeftSemiFilters(spark)(p.plan)

    assert(leftSemiCount(rewritten) == 2, s"residual semi predicate must be preserved:\n$rewritten")
  }

  test("keeps semi filters with different right sides") {
    val p = redundantSemiPlan(sameRightSide = false)
    val rewritten = PruneRedundantLeftSemiFilters(spark)(p.plan)

    assert(leftSemiCount(rewritten) == 2, s"different RHS plans are not redundant:\n$rewritten")
  }

  private case class PlanParts(plan: LogicalPlan)

  private def redundantSemiPlan(
      connectProbeKeys: Boolean = true,
      secondSemiHasResidual: Boolean = false,
      sameRightSide: Boolean = true,
      projectWrapsLeftCluster: Boolean = false): PlanParts = {
    val orderKey = intAttr("o_key")
    val orderOther = intAttr("o_other")
    val customerKey = intAttr("c_key")
    val lineKey = intAttr("l_key")
    val lineOther = intAttr("l_other")
    val rightKey1 = intAttr("r_key")
    val rightKey2 = if (sameRightSide) intAttr("r_key") else intAttr("r_other_key")

    val orders = LocalRelation(orderKey, orderOther)
    val customer = LocalRelation(customerKey)
    val lineitem = LocalRelation(lineKey, lineOther)
    val right1 = LocalRelation(rightKey1)
    val right2 = LocalRelation(rightKey2)

    val orderSemi =
      Join(orders, right1, LeftSemi, Some(EqualTo(orderKey, rightKey1)), JoinHint.NONE)
    val secondCondition =
      if (secondSemiHasResidual) {
        And(EqualTo(lineKey, rightKey2), GreaterThan(lineOther, Literal(0)))
      } else {
        EqualTo(lineKey, rightKey2)
      }
    val lineSemi = Join(lineitem, right2, LeftSemi, Some(secondCondition), JoinHint.NONE)
    val leftSide =
      if (projectWrapsLeftCluster) {
        val customerJoin =
          Join(customer, orderSemi, Inner, Some(EqualTo(customerKey, orderOther)), JoinHint.NONE)
        Project(orderSemi.output, customerJoin)
      } else {
        orderSemi
      }
    val innerCondition =
      if (connectProbeKeys) EqualTo(orderKey, lineKey) else EqualTo(orderOther, lineOther)

    PlanParts(Join(leftSide, lineSemi, Inner, Some(innerCondition), JoinHint.NONE))
  }

  private def intAttr(name: String): Attribute =
    AttributeReference(name, IntegerType, nullable = false)()

  private def leftSemiCount(plan: LogicalPlan): Int = {
    plan.collect { case Join(_, _, LeftSemi, _, _) => 1 }.sum
  }
}
