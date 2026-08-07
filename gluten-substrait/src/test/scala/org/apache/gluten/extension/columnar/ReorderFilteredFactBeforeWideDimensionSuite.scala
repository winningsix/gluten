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

import org.apache.spark.sql.{GlutenQueryTest, Row}
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Complete, Count, Sum}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftOuter}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.classic.ClassicDataset
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{DoubleType, LongType, StringType}

class ReorderFilteredFactBeforeWideDimensionSuite extends GlutenQueryTest with SharedSparkSession {

  private val confKey =
    "spark.gluten.sql.columnar.reorderFilteredFactBeforeWideDimension.enabled"

  test("reassociates filtered fact pair before an unfiltered wide dimension") {
    val p = candidatePlan()
    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = ReorderFilteredFactBeforeWideDimension(spark)(p.aggregate)
    }

    assert(!rewritten.fastEquals(p.aggregate), s"expected rewrite:\n$rewritten")
    assert(rewritten.output == p.aggregate.output)
    assert(countPredicate(rewritten, p.wideBridge) == 1)
    assert(countPredicate(rewritten, p.bridgeMeasure) == 1)

    val joins = rewritten.collect { case join: Join => join }
    assert(
      joins.exists(
        join =>
          contains(join, p.bridge) && contains(join, p.measure) &&
            !contains(join, p.wide)),
      s"filtered bridge and measure must join before the wide dimension:\n$rewritten"
    )
    assert(
      !joins.exists(
        join =>
          contains(join, p.bridge) && contains(join, p.wide) &&
            !contains(join, p.measure)),
      s"old wide+bridge intermediate must be gone:\n$rewritten"
    )
  }

  test("rewrite is fixed-point idempotent") {
    val p = candidatePlan()
    var once: LogicalPlan = null
    var twice: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      val rule = ReorderFilteredFactBeforeWideDimension(spark)
      once = rule(p.aggregate)
      twice = rule(once)
    }
    assert(twice.fastEquals(once), s"second application changed plan:\n$twice")
  }

  test("does not reassociate a streaming plan") {
    val p = candidatePlan(streaming = true)
    assert(p.aggregate.isStreaming)
    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = ReorderFilteredFactBeforeWideDimension(spark)(p.aggregate)
    }
    assert(rewritten.fastEquals(p.aggregate), s"Streaming plan must remain unchanged:\n$rewritten")
  }

  test("self-registers once before post-CBO dimension and broadcast-hint rules") {
    val experimental = spark.experimental
    val before = experimental.extraOptimizations
    try {
      experimental.synchronized {
        experimental.extraOptimizations =
          before.filterNot(_.isInstanceOf[ReorderFilteredFactBeforeWideDimension])
      }
      withSQLConf(confKey -> "true") {
        val rule = ReorderFilteredFactBeforeWideDimension(spark)
        rule(candidatePlan().aggregate)
        val rules = experimental.extraOptimizations
        assert(rules.count(_.isInstanceOf[ReorderFilteredFactBeforeWideDimension]) == 1)
        val rewriteIndex =
          rules.indexWhere(_.isInstanceOf[ReorderFilteredFactBeforeWideDimension])
        val hintIndex = rules.indexWhere(_.isInstanceOf[FluxFactProbeBroadcastHint])
        assert(rewriteIndex >= 0 && hintIndex > rewriteIndex)
        val dimensionIndex =
          rules.indexWhere(_.isInstanceOf[PushSelectiveDimensionChainBeforeFact])
        assert(dimensionIndex < 0 || dimensionIndex > rewriteIndex)
      }
    } finally {
      experimental.synchronized {
        experimental.extraOptimizations = before
      }
    }
  }

  test("resolves a deterministic aggregate-input alias to one measure leaf") {
    val p = candidatePlan(aliasedMeasure = true)
    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = ReorderFilteredFactBeforeWideDimension(spark)(p.aggregate)
    }
    assert(!rewritten.fastEquals(p.aggregate), s"expected aliased measure rewrite:\n$rewritten")
  }

  test("preserves a tiny peripheral broadcast hint in a post-CBO shaped cluster") {
    val p = candidatePlan(tinyPeripheralBroadcast = true)
    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = ReorderFilteredFactBeforeWideDimension(spark)(p.aggregate)
    }
    assert(!rewritten.fastEquals(p.aggregate), s"expected rewrite with tiny broadcast:\n$rewritten")
    val hinted = rewritten.collect { case join: Join if join.hint != JoinHint.NONE => join }
    assert(hinted.size == 1, s"expected exactly one preserved broadcast hint:\n$rewritten")
    assert(hinted.head.hint.leftHint.isEmpty)
    assert(hinted.head.hint.rightHint.exists(_.strategy.contains(BROADCAST)))
    assert(hinted.head.right.output.exists(_.name == "tiny_wide_key"))
  }

  test("can be disabled") {
    val p = candidatePlan()
    withSQLConf(confKey -> "false") {
      assert(ReorderFilteredFactBeforeWideDimension(spark)(p.aggregate).fastEquals(p.aggregate))
    }
  }

  test("does not reorder outer joins") {
    val p = candidatePlan(outerWideBridge = true)
    withSQLConf(confKey -> "true") {
      assert(ReorderFilteredFactBeforeWideDimension(spark)(p.aggregate).fastEquals(p.aggregate))
    }
  }

  test("does not reorder non-equi join clusters") {
    val p = candidatePlan(nonEquiBridgeMeasure = true)
    withSQLConf(confKey -> "true") {
      assert(ReorderFilteredFactBeforeWideDimension(spark)(p.aggregate).fastEquals(p.aggregate))
    }
  }

  test("does not reorder hinted joins") {
    val p = candidatePlan(hinted = true)
    withSQLConf(confKey -> "true") {
      assert(ReorderFilteredFactBeforeWideDimension(spark)(p.aggregate).fastEquals(p.aggregate))
    }
  }

  test("does not reorder non-SUM aggregates") {
    val p = candidatePlan(useCount = true)
    withSQLConf(confKey -> "true") {
      assert(ReorderFilteredFactBeforeWideDimension(spark)(p.aggregate).fastEquals(p.aggregate))
    }
  }

  test("does not reorder a nondeterministic SUM input") {
    val p = candidatePlan(nondeterministicSum = true)
    withSQLConf(confKey -> "true") {
      assert(ReorderFilteredFactBeforeWideDimension(spark)(p.aggregate).fastEquals(p.aggregate))
    }
  }

  test("does not reorder when the projected-width cost improvement is too small") {
    val p = candidatePlan(narrowDimension = true)
    withSQLConf(confKey -> "true") {
      assert(ReorderFilteredFactBeforeWideDimension(spark)(p.aggregate).fastEquals(p.aggregate))
    }
  }

  test("inner-join reassociation preserves duplicate and unmatched-key results") {
    import testImplicits._

    val wide = Seq((1L, "a"), (1L, "a-duplicate"), (2L, "b")).toDF("wide_key", "wide_value")
    val bridge = Seq((10L, 1L, true), (11L, 1L, true), (12L, 3L, true)).toDF(
      "measure_key",
      "wide_key",
      "selected")
    val measure = Seq((10L, 2.0, "R"), (10L, 3.0, "R"), (11L, 5.0, "R"), (12L, 7.0, "R")).toDF(
      "measure_key",
      "amount",
      "flag")

    val current = wide
      .join(bridge.filter($"selected"), "wide_key")
      .join(measure.filter($"flag" === "R"), "measure_key")
      .groupBy($"wide_key", $"wide_value")
      .sum("amount")
      .orderBy($"wide_key", $"wide_value")
      .collect()
      .toSeq
    val reassociated = bridge
      .filter($"selected")
      .join(measure.filter($"flag" === "R"), "measure_key")
      .join(wide, "wide_key")
      .groupBy($"wide_key", $"wide_value")
      .sum("amount")
      .orderBy($"wide_key", $"wide_value")
      .collect()
      .toSeq

    assert(current == reassociated)
    assert(current == Seq(Row(1L, "a", 10.0), Row(1L, "a-duplicate", 10.0)))
  }

  test("rule-generated rewritten plan preserves the original row multiset") {
    import testImplicits._

    val wide = spark.sparkContext
      .parallelize(
        Seq(
          (1L, "a", "a2", "a3", "a4", "a5", "a6"),
          (1L, "a-duplicate", "b2", "b3", "b4", "b5", "b6"),
          (2L, "unmatched", "c2", "c3", "c4", "c5", "c6")),
        2)
      .toDF("wide_key", "w1", "w2", "w3", "w4", "w5", "w6")
    val bridge = spark.sparkContext
      .parallelize(Seq((1L, 10L, 1L), (1L, 11L, 1L), (1L, 13L, 0L), (3L, 12L, 1L)), 2)
      .toDF("bridge_wide_key", "bridge_measure_key", "selected")
    val measure = spark.sparkContext
      .parallelize(
        Seq(
          (10L, 2.0, "R"),
          (10L, 3.0, "R"),
          (11L, 5.0, "R"),
          (12L, 7.0, "R"),
          (13L, 100.0, "R"),
          (10L, 200.0, "N")),
        2)
      .toDF("measure_key", "amount", "flag")

    val aggregate = wide
      .join(bridge.filter($"selected" === 1L), $"wide_key" === $"bridge_wide_key")
      .join(measure.filter($"flag" === "R"), $"bridge_measure_key" === $"measure_key")
      .groupBy($"wide_key", $"w1", $"w2", $"w3", $"w4", $"w5", $"w6")
      .agg(org.apache.spark.sql.functions.sum($"amount").as("metric"))
    val original = aggregate.queryExecution.optimizedPlan
    var rewritten: LogicalPlan = null
    withSQLConf(confKey -> "true") {
      rewritten = ReorderFilteredFactBeforeWideDimension(spark)(original)
    }

    assert(!rewritten.fastEquals(original), s"expected rule-generated rewrite:\n$rewritten")
    assert(rewritten.output == original.output)
    val originalRows = ClassicDataset.ofRows(spark, original).collect().toSeq
    val rewrittenRows = ClassicDataset.ofRows(spark, rewritten).collect().toSeq
    assert(rowBag(rewrittenRows) == rowBag(originalRows))
    assert(originalRows.size == 2)
  }

  private case class CandidatePlan(
      aggregate: Aggregate,
      wide: LogicalPlan,
      bridge: LogicalPlan,
      measure: LogicalPlan,
      wideBridge: Expression,
      bridgeMeasure: Expression)

  private def candidatePlan(
      outerWideBridge: Boolean = false,
      nonEquiBridgeMeasure: Boolean = false,
      hinted: Boolean = false,
      useCount: Boolean = false,
      narrowDimension: Boolean = false,
      aliasedMeasure: Boolean = false,
      tinyPeripheralBroadcast: Boolean = false,
      nondeterministicSum: Boolean = false,
      streaming: Boolean = false): CandidatePlan = {
    val wideKey = AttributeReference("wide_key", LongType)()
    val wideStrings = (1 to (if (narrowDimension) 3 else 6)).map {
      index =>
        AttributeReference(s"wide_field_$index", if (narrowDimension) LongType else StringType)()
    }
    val bridgeWideKey = AttributeReference("bridge_wide_key", LongType)()
    val bridgeMeasureKey = AttributeReference("bridge_measure_key", LongType)()
    val bridgeDate = AttributeReference("bridge_date", LongType)()
    val measureKey = AttributeReference("measure_key", LongType)()
    val measureValue = AttributeReference("measure_value", DoubleType)()
    val measureFlag = AttributeReference("measure_flag", StringType)()

    val wide = StatRel(wideKey +: wideStrings, bytes = 320L << 30, streaming = streaming)
    val bridge = Filter(
      GreaterThanOrEqual(bridgeDate, Literal(1L)),
      StatRel(Seq(bridgeWideKey, bridgeMeasureKey, bridgeDate), bytes = 360L << 30))
    val measure = Filter(
      EqualTo(measureFlag, Literal("R")),
      StatRel(Seq(measureKey, measureValue, measureFlag), bytes = 1024L << 30))

    val wideBridge = EqualTo(wideKey, bridgeWideKey)
    val bridgeMeasure: Expression =
      if (nonEquiBridgeMeasure) GreaterThan(bridgeMeasureKey, measureKey)
      else EqualTo(bridgeMeasureKey, measureKey)
    val hint =
      if (hinted) JoinHint(Some(HintInfo(strategy = Some(BROADCAST))), None)
      else JoinHint.NONE
    val wideBridgeJoin =
      Join(wide, bridge, if (outerWideBridge) LeftOuter else Inner, Some(wideBridge), hint)
    val factCurrent = Join(wideBridgeJoin, measure, Inner, Some(bridgeMeasure), JoinHint.NONE)
    val current =
      if (tinyPeripheralBroadcast) {
        val tinyWideKey = AttributeReference("tiny_wide_key", LongType)()
        val tiny = StatRel(Seq(tinyWideKey), bytes = 1L << 20)
        Join(
          factCurrent,
          tiny,
          Inner,
          Some(EqualTo(wideKey, tinyWideKey)),
          JoinHint(None, Some(HintInfo(strategy = Some(BROADCAST)))))
      } else {
        factCurrent
      }

    val grouping = wideKey +: wideStrings
    val (aggregateChild, measureInput) =
      if (aliasedMeasure) {
        val measureAlias = Alias(Multiply(measureValue, Literal(2.0)), "measure_input")()
        (Project(current.output :+ measureAlias, current), measureAlias.toAttribute)
      } else {
        (current, measureValue)
      }
    val function =
      if (useCount) Count(Seq(measureInput))
      else if (nondeterministicSum) Sum(Add(measureInput, Rand(0L)))
      else Sum(measureInput)
    val aggregateExpression = AggregateExpression(
      function,
      Complete,
      isDistinct = false,
      filter = None,
      resultId = NamedExpression.newExprId)
    val aggregate =
      Aggregate(grouping, grouping :+ Alias(aggregateExpression, "metric")(), aggregateChild)
    CandidatePlan(aggregate, wide, bridge, measure, wideBridge, bridgeMeasure)
  }

  private def contains(root: LogicalPlan, target: LogicalPlan): Boolean =
    root.exists(plan => plan.eq(target) || plan.fastEquals(target))

  private def countPredicate(plan: LogicalPlan, target: Expression): Int =
    plan.expressions.map(splitAnd(_).count(_.semanticEquals(target))).sum +
      plan.children.map(countPredicate(_, target)).sum

  private def rowBag(rows: Seq[Row]): Map[Row, Int] =
    rows.groupBy(identity).map { case (row, matches) => row -> matches.size }

  private def splitAnd(expression: Expression): Seq[Expression] = expression match {
    case And(left, right) => splitAnd(left) ++ splitAnd(right)
    case other => Seq(other)
  }

  private case class StatRel(attributes: Seq[Attribute], bytes: BigInt, streaming: Boolean = false)
    extends LeafNode {
    override def output: Seq[Attribute] = attributes
    override def isStreaming: Boolean = streaming
    override def computeStats(): Statistics = Statistics(sizeInBytes = bytes)
  }
}
