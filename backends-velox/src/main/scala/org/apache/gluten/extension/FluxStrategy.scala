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

import org.apache.gluten.execution.FluxNativeQueryExec

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution.{LeafExecNode, SparkPlan, SparkStrategy}
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeLike, ShuffleExchangeLike}
import org.apache.spark.sql.internal.SQLConf

import java.util.concurrent.atomic.AtomicInteger

/**
 * Plan C: FluxStrategy intercepts the entire logical plan at the strategy level and generates a
 * single FluxNativeQueryExec that executes all fragments concurrently via Velox's
 * FluxQueryCoordinator.
 *
 * This runs BEFORE EnsureRequirements -- no ShuffleExchange nodes are ever created for FLUX
 * queries. The strategy:
 *
 *   1. Checks if FLUX is enabled via `spark.gluten.mpp.enabled` and
 *      `spark.gluten.mpp.strategy.enabled`. 2. Generates a "shadow" physical plan using Spark's
 *      internal planner (with FLUX disabled to avoid recursion). 3. Runs the shadow plan through
 *      `QueryExecution.executedPlan` to get ShuffleExchange boundaries inserted by
 *      EnsureRequirements. 4. Walks the shadow plan to extract fragment boundaries and exchange
 *      specs. 5. Returns a single FluxNativeQueryExec containing all fragments.
 *
 * If the plan contains BroadcastExchange nodes or any unsupported operators, FluxStrategy returns
 * Nil (falls back to Gluten BSP mode).
 *
 * When FluxStrategy claims a plan, it disables AQE for that query because FLUX runs all stages
 * concurrently -- there are no intermediate statistics to observe.
 */
case class FluxStrategy(session: SparkSession) extends SparkStrategy with Logging {

  private val FLUX_ENABLED_KEY = "spark.gluten.mpp.enabled"
  private val FLUX_STRATEGY_KEY = "spark.gluten.mpp.strategy.enabled"
  private val FLUX_ENABLED_DEFAULT = "true"
  // Disable Plan C for now - shadow plan generation returns CommandResultExec.
  // Use Plan D (FluxCollapseRule) which has a working child plan.
  private val FLUX_STRATEGY_DEFAULT = "false"

  /**
   * Guard flag to prevent re-entrant shadow plan generation. When we generate the shadow plan,
   * Spark's planner will call this strategy again. The flag ensures we return Nil during shadow
   * planning.
   */
  @transient
  private var generatingShadowPlan: Boolean = false

  override def apply(plan: LogicalPlan): Seq[SparkPlan] = {
    if (!isFluxStrategyEnabled) {
      logDebug(
        s"FluxStrategy: disabled (" +
          s"flux.enabled=${session.conf.get(FLUX_ENABLED_KEY, FLUX_ENABLED_DEFAULT)}, " +
          s"strategy.enabled=${session.conf.get(FLUX_STRATEGY_KEY, FLUX_STRATEGY_DEFAULT)})")
      return Nil
    }
    if (generatingShadowPlan) return Nil

    logDebug(s"FluxStrategy: apply() called with ${plan.getClass.getSimpleName}")

    if (isTopLevelPlan(plan)) {
      // Unwrap ReturnAnswer - it's just a Spark wrapper, not a real operator.
      // The shadow planner needs the actual query plan, not ReturnAnswer.
      val queryPlan = plan match {
        case ReturnAnswer(child) => child
        case other => other
      }
      // After unwrapping, skip commands (DDL, DML) - only queries should go FLUX
      if (
        queryPlan.isInstanceOf[org.apache.spark.sql.catalyst.plans.logical.Command] ||
        queryPlan.isInstanceOf[CommandResult]
      ) {
        return Nil
      }
      logDebug(
        s"FluxStrategy: top-level plan detected (${plan.getClass.getSimpleName} " +
          s"-> ${queryPlan.getClass.getSimpleName}), attempting FLUX...")
      tryFlux(queryPlan)
        .map {
          exec =>
            logInfo(s"FluxStrategy: *** PLAN C ACTIVE *** returning FluxNativeQueryExec")
            Seq(exec)
        }
        .getOrElse {
          logWarning(s"FluxStrategy: tryFlux returned None, falling back to BSP")
          Nil
        }
    } else {
      logDebug(s"FluxStrategy: not a top-level plan (${plan.getClass.getSimpleName}), skipping")
      Nil
    }
  }

  private def isFluxStrategyEnabled: Boolean = {
    // Both spark.gluten.mpp.enabled and spark.gluten.mpp.strategy.enabled must be true.
    // The first is the global FLUX toggle; the second enables Plan C specifically
    // (vs Plan D which uses FluxCollapseRule).
    val fluxEnabled = session.conf.get(FLUX_ENABLED_KEY, FLUX_ENABLED_DEFAULT).toBoolean
    val strategyEnabled = session.conf.get(FLUX_STRATEGY_KEY, FLUX_STRATEGY_DEFAULT).toBoolean
    fluxEnabled && strategyEnabled
  }

  /**
   * Check whether the logical plan looks like a top-level query root. We avoid intercepting DDL
   * commands, CTAS, or other non-query plans.
   */
  private def isTopLevelPlan(plan: LogicalPlan): Boolean = {
    // Only match ReturnAnswer - this is the outermost wrapper Spark adds for queries.
    // This ensures we create exactly ONE FluxNativeQueryExec per query, not nested ones.
    plan match {
      case _: ReturnAnswer => true
      case _ => false
    }
  }

  /**
   * Attempt to build an FLUX execution plan for the given logical plan. Returns None if the plan
   * cannot be fully handled in FLUX mode.
   */
  private def tryFlux(logicalPlan: LogicalPlan): Option[FluxNativeQueryExec] = {
    logDebug("FluxStrategy: attempting Plan C FLUX transformation")

    // Plan C Approach: Return FluxNativeQueryExec with planLater(logicalPlan) as child.
    // Spark will plan the child normally (including EnsureRequirements inserting
    // ShuffleExchange nodes). Then at execution time, FluxNativeQueryExec examines
    // its child's physical plan to extract fragment/exchange info.
    //
    // This avoids the "cannot transform shuffle node" issue because FluxNativeQueryExec
    // is inserted BEFORE ShuffleExchange nodes exist. Spark adds them to the child
    // plan, and we read them - never replace them.

    // For now, create placeholder fragments - the real extraction happens at execution time
    // from the child physical plan.
    val fragments = Seq(
      NativeFragment(
        id = 0,
        rootOperator = null, // Will be populated at execution time from child plan
        outputAttributes = logicalPlan.output,
        parallelism = session.conf.get("spark.sql.shuffle.partitions", "200").toInt
      ))
    val exchanges = Seq.empty[ExchangeSpec]

    logDebug(
      s"FluxStrategy: *** FLUX MODE (Plan C) *** query=${logicalPlan.getClass.getSimpleName}, " +
        s"fragments will be extracted at execution time from child physical plan")
    exchanges.foreach {
      e =>
        logDebug(
          s"  Exchange ${e.id}: F${e.producerFragmentId} -> F${e.consumerFragmentId} " +
            s"(${e.exchangeType}, ${e.numPartitions} partitions)")
    }

    // Step 4: Disable AQE for this query. FLUX runs all stages concurrently --
    // there are no intermediate shuffle statistics to observe.
    session.conf.set("spark.sql.adaptive.enabled", "false")

    // Step 5: Build FluxNativeQueryExec.
    // We can't use planLater(logicalPlan) because Spark already assigned
    // this node to FluxStrategy. Instead, plan children of the logical plan
    // and collect them. FluxNativeQueryExec wraps the planned children.
    //
    // For a single-child plan (Sort, Aggregate, Project, etc.):
    //   plan the child via planLater, then wrap
    // For multi-child plans (Join): plan each child via planLater
    // Use FluxSchemaOnlyExec as child - it provides schema only, never executes.
    // FluxNativeQueryExec.doExecuteColumnar() generates its own execution plan
    // via JNI/FluxQueryCoordinator, independent of the child.
    // planLater(logicalPlan) - plan the ENTIRE query (Sort/Aggregate/etc.)
    // FluxStrategy claims ReturnAnswer; logicalPlan is ReturnAnswer.child.
    // Since they're different nodes, no recursion conflict.
    // Spark will plan logicalPlan normally -> with ShuffleExchange, Gluten rules, etc.
    val plannedChild = planLater(logicalPlan)

    Some(
      FluxNativeQueryExec(
        child = plannedChild,
        fragments = fragments,
        exchanges = exchanges,
        originalLogicalPlan = logicalPlan
      ))
  }

  /**
   * Generate a "shadow" physical plan by running Spark's internal planner (with FLUX fully
   * disabled) and letting EnsureRequirements insert ShuffleExchange nodes.
   *
   * We use `QueryExecution.executedPlan` which runs the full preparation pipeline including
   * EnsureRequirements.
   *
   * To avoid infinite recursion, we:
   *   - Set the `generatingShadowPlan` flag (prevents re-entrant calls)
   *   - Disable `spark.gluten.mpp.enabled` (prevents Plan D FluxCollapseRule)
   *   - Disable AQE (ensures simple plan without AdaptiveSparkPlanExec)
   */
  private def generateShadowPlan(logicalPlan: LogicalPlan): Option[SparkPlan] = {
    // Save current config values to restore after shadow planning.
    val prevFluxEnabled = session.conf.get(FLUX_ENABLED_KEY, FLUX_ENABLED_DEFAULT)
    val prevStrategyEnabled = session.conf.get(FLUX_STRATEGY_KEY, FLUX_STRATEGY_DEFAULT)
    val prevAqe = session.conf.get("spark.sql.adaptive.enabled", "true")

    generatingShadowPlan = true
    try {
      // Disable FLUX entirely for the shadow plan so neither FluxStrategy (Plan C)
      // nor FluxCollapseRule (Plan D) interfere.
      session.conf.set(FLUX_ENABLED_KEY, "false")
      session.conf.set(FLUX_STRATEGY_KEY, "false")
      // Disable AQE so we get a flat plan with ShuffleExchange nodes instead
      // of AdaptiveSparkPlanExec wrapping.
      session.conf.set("spark.sql.adaptive.enabled", "false")

      // Use Spark's internal QueryExecution to generate the physical plan.
      // executedPlan runs: planner -> prepareForExecution (EnsureRequirements etc.)
      // Approach: use Spark's planner to get physical plan, then manually
      // run preparations. The key is planLater() - Spark strategies can
      // defer child planning. We plan children normally, then examine
      // the resulting physical plan for ShuffleExchange nodes.
      //
      // IMPORTANT: We must use the ORIGINAL planner (not create new QueryExecution)
      // because we're inside a strategy call - creating a new QE causes recursion
      // or short-circuits to CommandResultExec.
      //
      // Instead, use planLater to let Spark plan the children normally,
      // then we intercept and wrap.
      val physicalPlan = planLater(logicalPlan)
      logDebug(s"FluxStrategy: planLater result class=${physicalPlan.getClass.getSimpleName}")
      Some(physicalPlan)
    } finally {
      generatingShadowPlan = false
      session.conf.set(FLUX_ENABLED_KEY, prevFluxEnabled)
      session.conf.set(FLUX_STRATEGY_KEY, prevStrategyEnabled)
      session.conf.set("spark.sql.adaptive.enabled", prevAqe)
    }
  }

  /** Check whether a physical plan tree contains any BroadcastExchange nodes. */
  private def containsBroadcastExchange(plan: SparkPlan): Boolean = {
    plan match {
      case _: BroadcastExchangeLike => true
      case _ => plan.children.exists(containsBroadcastExchange)
    }
  }

  /**
   * Walk the shadow physical plan (which has ShuffleExchange nodes inserted by EnsureRequirements)
   * and extract NativeFragment + ExchangeSpec lists.
   *
   * We match on ShuffleExchangeLike (the Spark trait) rather than a specific implementation because
   * the shadow plan may contain either Spark's ShuffleExchangeExec or Gluten's
   * ColumnarShuffleExchangeExec depending on which columnar rules ran.
   *
   * Each contiguous subtree of non-exchange operators becomes a fragment. Each ShuffleExchangeLike
   * becomes an exchange boundary linking a producer fragment to a consumer fragment.
   */
  private def extractFragmentsFromShadowPlan(
      plan: SparkPlan): (Seq[NativeFragment], Seq[ExchangeSpec]) = {

    val fragments = scala.collection.mutable.ArrayBuffer[NativeFragment]()
    val exchanges = scala.collection.mutable.ArrayBuffer[ExchangeSpec]()
    val fragmentCounter = new AtomicInteger(0)
    val exchangeCounter = new AtomicInteger(0)

    def walk(node: SparkPlan): Int = {
      node match {
        case shuffle: ShuffleExchangeLike =>
          // The child of the exchange belongs to the producer fragment.
          val producerFragmentId = walk(shuffle.child)

          // Create an exchange spec linking producer to consumer.
          val consumerFragmentId = fragmentCounter.getAndIncrement()
          val (exchangeType, partitionKeys) =
            classifyPartitioning(shuffle.outputPartitioning)

          exchanges += ExchangeSpec(
            id = exchangeCounter.getAndIncrement(),
            producerFragmentId = producerFragmentId,
            consumerFragmentId = consumerFragmentId,
            exchangeType = exchangeType,
            numPartitions = shuffle.outputPartitioning.numPartitions,
            partitionKeys = partitionKeys,
            rangeOrdering = shuffle.outputPartitioning match {
              case range: RangePartitioning => range.ordering
              case _ => Seq.empty
            },
            rangeSamplePlan = shuffle.child
          )

          // The consumer fragment is a placeholder that represents the
          // exchange receive side. The parent operator will consume from it.
          fragments += NativeFragment(
            id = consumerFragmentId,
            rootOperator = node,
            outputAttributes = shuffle.output,
            parallelism = shuffle.outputPartitioning.numPartitions
          )

          consumerFragmentId

        case other =>
          // Non-exchange node. Walk all children (exchange children handled above).
          val childFragIds = other.children.map(walk)

          val fragId = fragmentCounter.getAndIncrement()
          val parallelism = inferParallelism(other)
          fragments += NativeFragment(
            id = fragId,
            rootOperator = other,
            outputAttributes = other.output,
            parallelism = parallelism
          )
          fragId
      }
    }

    walk(plan)
    (fragments.toSeq, exchanges.toSeq)
  }

  /** Classify a Spark partitioning into an exchange type string and partition keys. */
  private def classifyPartitioning(partitioning: Partitioning): (String, Seq[Attribute]) = {
    partitioning match {
      case hash: HashPartitioning =>
        val keys = hash.expressions.collect { case attr: Attribute => attr }
        ("HASH", keys)
      case _: RoundRobinPartitioning =>
        ("ROUND_ROBIN", Seq.empty)
      case SinglePartition =>
        ("SINGLE", Seq.empty)
      case _: RangePartitioning =>
        ("RANGE", Seq.empty)
      case _: BroadcastPartitioning =>
        ("BROADCAST", Seq.empty)
      case other =>
        logWarning(s"FluxStrategy: unexpected partitioning type: ${other.getClass.getSimpleName}")
        ("HASH", Seq.empty)
    }
  }

  /** Infer parallelism for a fragment from the operator's output partitioning. */
  private def inferParallelism(plan: SparkPlan): Int = {
    plan.outputPartitioning match {
      case p if p.numPartitions > 0 => p.numPartitions
      case _ =>
        SQLConf.get.getConfString("spark.sql.shuffle.partitions", "200").toInt
    }
  }
}

/**
 * A minimal leaf SparkPlan that provides schema information only. Never executed -- used as the
 * child of FluxNativeQueryExec when the strategy bypasses normal physical planning.
 */
case class FluxSchemaOnlyExec(outputAttributes: Seq[Attribute])
  extends LeafExecNode
  with org.apache.gluten.execution.GlutenPlan {

  override def output: Seq[Attribute] = outputAttributes

  override def batchType(): org.apache.gluten.extension.columnar.transition.Convention.BatchType =
    org.apache.gluten.backendsapi.BackendsApiManager.getSettings.primaryBatchType

  override def rowType0(): org.apache.gluten.extension.columnar.transition.Convention.RowType =
    org.apache.gluten.extension.columnar.transition.Convention.RowType.None

  override def requiredChildConvention()
      : Seq[org.apache.gluten.extension.columnar.transition.ConventionReq] = Nil

  override protected def doExecute(): RDD[InternalRow] = {
    throw new UnsupportedOperationException("FluxSchemaOnlyExec should never be executed.")
  }
}
