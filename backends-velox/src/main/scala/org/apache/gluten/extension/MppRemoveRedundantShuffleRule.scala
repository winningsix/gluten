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

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.plans.physical.{HashPartitioning, Partitioning, PartitioningCollection}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.AQEShuffleReadExec
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec

/**
 * Eliminate ShuffleExchangeExec(HashPartitioning) when the child's outputPartitioning already
 * satisfies the same hash distribution. This mirrors Presto's distributed planner behavior of
 * skipping redundant exchanges, reducing fragment count for join-heavy queries (TPC-H Q5/Q7/Q8/Q9
 * etc.).
 *
 * Conservative: only matches when child is HashPartitioning (or PartitioningCollection containing
 * one) with identical key set + numPartitions. Skips AQEShuffleReadExec (could have applied
 * coalesce/skew-split that breaks key alignment).
 *
 * Gated by spark.gluten.mpp.removeRedundantShuffle (default: false).
 */
case class MppRemoveRedundantShuffleRule() extends Rule[SparkPlan] with Logging {
  private val confKey = "spark.gluten.mpp.removeRedundantShuffle"

  override def apply(plan: SparkPlan): SparkPlan = {
    val sess = org.apache.spark.sql.SparkSession.getActiveSession
    val enabled = sess.exists(_.conf.get(confKey, "false").toBoolean)
    logDebug(
      s"MppRemoveRedundantShuffleRule.apply: enabled=$enabled plan=${plan.getClass.getSimpleName}")
    if (!enabled) {
      return plan
    }
    plan.transformUp {
      case s @ ShuffleExchangeExec(req: HashPartitioning, child, _, _)
          if !child.isInstanceOf[AQEShuffleReadExec] &&
            hashSatisfies(child.outputPartitioning, req) =>
        logInfo(
          s"MppRemoveRedundantShuffle: eliminated redundant shuffle, " +
            s"child partitioning ${child.outputPartitioning} already satisfies $req")
        child
    }
  }

  private def hashSatisfies(child: Partitioning, req: HashPartitioning): Boolean = child match {
    case h: HashPartitioning =>
      h.numPartitions == req.numPartitions &&
      h.expressions.length == req.expressions.length &&
      h.expressions.zip(req.expressions).forall { case (a, b) => a.semanticEquals(b) }
    case PartitioningCollection(parts) => parts.exists(p => hashSatisfies(p, req))
    case _ => false
  }
}
