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

import org.apache.gluten.execution.MppNativeQueryExec
import org.apache.gluten.utils.LocalTableScanExecCompat

import org.apache.spark.sql.{QueryTest, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Alias, Ascending, AttributeReference, ExprId, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.{IdentityBroadcastMode, RangePartitioning}
import org.apache.spark.sql.execution.{ColumnarBroadcastExchangeExec, ColumnarShuffleExchangeExec, LocalTableScanExec, ProjectExec, ScalarSubquery, SparkPlan, SubqueryExec, UnionExec}
import org.apache.spark.sql.execution.exchange.ReusedExchangeExec
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.IntegerType

class MppSuppressDeadBroadcastsRuleSuite extends QueryTest with SharedSparkSession {

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

  private def broadcast(name: String): ColumnarBroadcastExchangeExec = {
    val attribute = AttributeReference(name, IntegerType, nullable = false)()
    val child = LocalTableScanExecCompat(Seq(attribute), Seq.empty[InternalRow])
    ColumnarBroadcastExchangeExec(IdentityBroadcastMode, child)
  }

  private def mpp(child: SparkPlan): MppNativeQueryExec =
    MppNativeQueryExec(child, fragments = Seq.empty, exchanges = Seq.empty)

  private def range(child: SparkPlan): ColumnarShuffleExchangeExec = {
    ColumnarShuffleExchangeExec(
      outputPartitioning =
        RangePartitioning(Seq(SortOrder(child.output.head, Ascending)), numPartitions = 4),
      child = child,
      projectOutputAttributes = child.output
    )
  }

  test("defers eager prepare but keeps RANGE broadcasts live") {
    val exchange = broadcast("range_key")

    MppSuppressDeadBroadcastsRule()(mpp(range(exchange)))

    assert(!exchange.isMppSuppressed)
    assert(exchange.isMppPrepareDeferred)
  }

  test("defers a direct broadcast during final planning") {
    val exchange = broadcast("broadcast_key")

    MppSuppressDeadBroadcastsRule()(mpp(exchange))

    assert(!exchange.isMppSuppressed)
    assert(exchange.isMppPrepareDeferred)
  }

  test("runtime commit suppresses a direct broadcast after validation") {
    val exchange = broadcast("committed_key")

    mpp(exchange).suppressDeadBroadcastsForNativeMpp(exchange)

    assert(exchange.isMppSuppressed)
  }

  test("runtime commit honors the dead-broadcast diagnostic switch") {
    val exchange = broadcast("diagnostic_key")

    withSQLConf("spark.gluten.mpp.suppressDeadBroadcast" -> "false") {
      mpp(exchange).suppressDeadBroadcastsForNativeMpp(exchange)
    }

    assert(!exchange.isMppSuppressed)
  }

  test("keeps a broadcast shared by RANGE and direct branches live") {
    val exchange = broadcast("shared_key")
    val shared = UnionExec(Seq(range(exchange), exchange))

    MppSuppressDeadBroadcastsRule()(mpp(shared))

    assert(!exchange.isMppSuppressed)
    assert(exchange.isMppPrepareDeferred)
  }

  test("keeps a reused broadcast under RANGE live") {
    val exchange = broadcast("reused_key")
    val reused = ReusedExchangeExec(exchange.output, exchange)

    MppSuppressDeadBroadcastsRule()(mpp(range(reused)))

    assert(!exchange.isMppSuppressed)
    assert(exchange.isMppPrepareDeferred)
  }

  test("keeps a subquery broadcast under RANGE live") {
    val exchange = broadcast("subquery_key")
    val subquery = SubqueryExec("range-subquery", exchange)

    MppSuppressDeadBroadcastsRule()(mpp(range(subquery)))

    assert(!exchange.isMppSuppressed)
    assert(exchange.isMppPrepareDeferred)
  }

  test("defers a broadcast referenced only by an executable subquery expression") {
    val exchange = broadcast("expression_subquery_key")
    val subquery = SubqueryExec("expression-subquery", exchange)
    val project = ProjectExec(
      Seq(Alias(ScalarSubquery(subquery, ExprId(1L)), "scalar_value")()),
      LocalTableScanExec(Seq.empty, Seq(InternalRow.empty)))

    MppSuppressDeadBroadcastsRule()(mpp(project))

    assert(!exchange.isMppSuppressed)
    assert(exchange.isMppPrepareDeferred)
  }

  test("runtime commit suppresses a broadcast referenced only by a subquery expression") {
    val exchange = broadcast("committed_expression_subquery_key")
    val subquery = SubqueryExec("committed-expression-subquery", exchange)
    val project = ProjectExec(
      Seq(Alias(ScalarSubquery(subquery, ExprId(2L)), "scalar_value")()),
      LocalTableScanExec(Seq.empty, Seq(InternalRow.empty)))

    mpp(project).suppressDeadBroadcastsForNativeMpp(project)

    assert(exchange.isMppSuppressed)
  }

  test("uses the thread SQLConf when no SparkSession is active") {
    val exchange = broadcast("thread_conf_key")
    val previous = SparkSession.getActiveSession
    SparkSession.clearActiveSession()
    try {
      MppSuppressDeadBroadcastsRule()(mpp(exchange))
    } finally {
      previous.foreach(SparkSession.setActiveSession)
    }

    assert(exchange.isMppPrepareDeferred)
  }

  test("diagnostic switch keeps eager preparation enabled") {
    val exchange = broadcast("diagnostic_prepare_key")

    withSQLConf("spark.gluten.mpp.suppressDeadBroadcast" -> "false") {
      MppSuppressDeadBroadcastsRule()(mpp(exchange))
    }

    assert(!exchange.isMppSuppressed)
    assert(!exchange.isMppPrepareDeferred)
  }
}
