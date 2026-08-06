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
import org.apache.spark.sql.catalyst.expressions.{Add, Alias, AttributeReference, EqualTo, GreaterThan, IsNotNull, Literal, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Complete, Max, Sum}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, JoinHint, LocalRelation, LogicalPlan, Project}
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{DoubleType, LongType}

class EliminateRedundantSelfAggregateJoinSuite extends QueryTest with SharedSparkSession {

  test("admits an is-not-null check on the summary join key") {
    val plan = selfAggregateJoin(summaryPredicateOnValue = false)
    val rewritten = EliminateRedundantSelfAggregateJoin(spark)(plan)

    assert(!rewritten.fastEquals(plan), s"expected redundant self join to be removed:\n$plan")
    assert(rewritten.collect { case _: Join => 1 }.isEmpty, s"unexpected join:\n$rewritten")
    assert(rewritten.collect { case _: LocalRelation => 1 }.size == 1)
  }

  test("rejects a value predicate on the summary source") {
    val plan = selfAggregateJoin(summaryPredicateOnValue = true)
    val rewritten = EliminateRedundantSelfAggregateJoin(spark)(plan)

    assert(rewritten.fastEquals(plan), s"value predicate must prevent the rewrite:\n$rewritten")
  }

  test("rejects a compound summary aggregate output") {
    val plan = selfAggregateJoin(compoundSummary = true)
    val rewritten = EliminateRedundantSelfAggregateJoin(spark)(plan)

    assert(rewritten.fastEquals(plan), s"compound summary must prevent the rewrite:\n$rewritten")
  }

  test("rejects another outer aggregate that references the removed value") {
    val plan = selfAggregateJoin(additionalOuterValueAggregate = true)
    val rewritten = EliminateRedundantSelfAggregateJoin(spark)(plan)

    assert(
      rewritten.fastEquals(plan),
      s"another outer value aggregate must prevent the rewrite:\n$rewritten")
  }

  test("rejects grouping by the removed value") {
    val plan = selfAggregateJoin(groupOuterByValue = true)
    val rewritten = EliminateRedundantSelfAggregateJoin(spark)(plan)

    assert(
      rewritten.fastEquals(plan),
      s"grouping by the removed value must prevent the rewrite:\n$rewritten")
  }

  test("rejects a compound Project expression over the removed value") {
    val plan = selfAggregateJoin(compoundProjectedValue = true)
    val rewritten = EliminateRedundantSelfAggregateJoin(spark)(plan)

    assert(
      rewritten.fastEquals(plan),
      s"compound Project RHS consumer must prevent the rewrite:\n$rewritten")
    assert(plan.resolved && rewritten.resolved)
  }

  private def selfAggregateJoin(
      summaryPredicateOnValue: Boolean = false,
      compoundSummary: Boolean = false,
      additionalOuterValueAggregate: Boolean = false,
      groupOuterByValue: Boolean = false,
      compoundProjectedValue: Boolean = false): LogicalPlan = {
    val key = AttributeReference("key", LongType, nullable = true)()
    val value = AttributeReference("value", DoubleType, nullable = true)()
    val source = LocalRelation(key, value)
    val right = source.newInstance()
    val rightKey = right.output.head
    val rightValue = right.output(1)

    val sourcePredicate =
      if (summaryPredicateOnValue) GreaterThan(value, Literal(0.0)) else IsNotNull(key)
    val summaryInput = Filter(sourcePredicate, source)
    val summaryExpression = AggregateExpression(
      Sum(value),
      Complete,
      isDistinct = false,
      filter = None,
      resultId = NamedExpression.newExprId)
    val summaryAlias = Alias(
      if (compoundSummary) Add(summaryExpression, Literal(1.0)) else summaryExpression,
      "summary")()
    val summary = Aggregate(Seq(key), Seq(key, summaryAlias), summaryInput)
    val filteredSummary = Filter(GreaterThan(summaryAlias.toAttribute, Literal(0.0)), summary)
    val joined = Join(filteredSummary, right, Inner, Some(EqualTo(key, rightKey)), JoinHint.NONE)
    val adjustedValue = Alias(Add(rightValue, Literal(1.0)), "adjusted")()
    val projected = Project(
      Seq(rightValue) ++ (if (compoundProjectedValue) Seq(adjustedValue) else Seq.empty),
      joined)
    val outerExpression = AggregateExpression(
      Sum(rightValue),
      Complete,
      isDistinct = false,
      filter = None,
      resultId = NamedExpression.newExprId)
    val outerAggregateExpressions =
      Seq(Alias(outerExpression, "total")()) ++
        (if (additionalOuterValueAggregate) {
           Seq(
             Alias(
               AggregateExpression(
                 Max(rightValue),
                 Complete,
                 isDistinct = false,
                 filter = None,
                 resultId = NamedExpression.newExprId),
               "maximum")())
         } else {
           Seq.empty
         }) ++
        (if (compoundProjectedValue) {
           Seq(
             Alias(
               AggregateExpression(
                 Max(adjustedValue.toAttribute),
                 Complete,
                 isDistinct = false,
                 filter = None,
                 resultId = NamedExpression.newExprId),
               "adjusted_maximum")())
         } else {
           Seq.empty
         })
    Aggregate(
      if (groupOuterByValue) Seq(rightValue) else Seq.empty,
      outerAggregateExpressions,
      projected)
  }
}
