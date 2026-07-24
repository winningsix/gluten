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
package org.apache.gluten.execution

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Add, Alias, And, Ascending, Attribute, AttributeReference, CurrentRow, DenseRank, EqualTo, Literal, Murmur3Hash, Rand, Rank, RowFrame, RowNumber, SortOrder, SpecifiedWindowFrame, UnboundedPreceding, WindowExpression, WindowSpecDefinition}
import org.apache.spark.sql.catalyst.plans.physical.HashPartitioning
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.execution.{ColumnarInputAdapter, ColumnarShuffleExchangeExec, InputIteratorTransformer, LeafExecNode, ProjectExec, SparkPlan, UnionExec}
import org.apache.spark.sql.execution.exchange.ENSURE_REQUIREMENTS
import org.apache.spark.sql.execution.window.{GlutenFinal, GlutenPartial}
import org.apache.spark.sql.types.{IntegerType, LongType}

import org.scalatest.funsuite.AnyFunSuite

class MppRankFilterWindowRewriteSuite extends AnyFunSuite {
  private val SentinelTag = TreeNodeTag[String]("mpp.rank-filter-window-rewrite.sentinel")
  private case class TestLeaf(override val output: Seq[Attribute]) extends LeafExecNode {
    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("planner-only test leaf")
  }

  private case class ChildOptions(
      omitPartitionFromChild: Boolean = false,
      includePreProject: Boolean = false,
      includeRandomPreProject: Boolean = false)

  private case class SortOptions(validLocalSort: Boolean = true, extraLocalSortKey: Boolean = false)

  private case class PartialOptions(
      matchingPartial: Boolean = true,
      interveningFilterBeforePartial: Boolean = false,
      mppBoundaryWrappers: Boolean = false,
      hashProjectKind: String = "valid",
      wholeStageExtraFilter: Boolean = false)

  private case class BranchOptions(
      partitioned: Boolean = true,
      rankKind: String = "row_number",
      child: ChildOptions = ChildOptions(),
      sort: SortOptions = SortOptions(),
      partial: PartialOptions = PartialOptions())

  test("rollout follows cuDF by default and accepts an explicit override") {
    assert(!MppRankFilterWindowRewrite.isEnabled(configured = None, cudfEnabled = false))
    assert(MppRankFilterWindowRewrite.isEnabled(configured = None, cudfEnabled = true))
    assert(!MppRankFilterWindowRewrite.isEnabled(configured = Some(false), cudfEnabled = true))
    assert(MppRankFilterWindowRewrite.isEnabled(configured = Some(true), cudfEnabled = false))
  }

  test("an explicit disable leaves the rank-filter plan unchanged") {
    val plan = rankFilterBranch("disabled", includeExchange = false)
    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = false, numPartitions = 4)

    assert(rewritten eq plan)
    assert(stats.rewrittenWindows == 0)
    assert(stats.insertedHashExchanges == 0)
  }

  test("three rank-filter branches preserve Window Filter local Sort and bounded Partial TopN") {
    val plan = UnionExec(
      Seq(
        rankFilterBranch("first", includeExchange = false),
        rankFilterBranch("second", includeExchange = false),
        rankFilterBranch("third", includeExchange = false)))

    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 3)
    assert(stats.insertedHashExchanges == 3)
    assert(rewritten.collect { case _: WindowExecTransformer => 1 }.size == 3)
    assert(rewritten.collect { case _: FilterExecTransformer => 1 }.size == 3)
    val sorts = rewritten.collect { case sort: SortExecTransformer => sort }
    assert(sorts.size == 3)
    assert(sorts.forall(!_.global))
    assertRetainsOnlyPartial(rewritten, expectedCount = 3)
    val exchanges = rewritten.collect { case exchange: ColumnarShuffleExchangeExec => exchange }
    assert(exchanges.size == 3)
    assert(exchanges.forall(_.outputPartitioning.isInstanceOf[HashPartitioning]))
  }

  test("exact rank one retains the semantic Window") {
    val plan =
      rankFilterBranch("rank", includeExchange = false, options = BranchOptions(rankKind = "rank"))
    val originalOutputExprIds = plan.output.map(_.exprId)
    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 1)
    assert(stats.insertedHashExchanges == 1)
    assert(rewritten.collect { case _: WindowExecTransformer => 1 }.size == 1)
    assert(rewritten.collect { case _: SortExecTransformer => 1 }.size == 1)
    assert(rewritten.collect { case _: FilterExecTransformer => 1 }.size == 1)
    assertRetainsOnlyPartial(rewritten)
    assert(rewritten.output.map(_.exprId) == originalOutputExprIds)
  }

  test("rank one conjunct keeps the semantic Window and complete filter") {
    val original =
      rankFilterBranch(
        "rank_conjunct",
        includeExchange = false,
        options = BranchOptions(rankKind = "rank"))
        .asInstanceOf[FilterExecTransformer]
    val window = original.child.asInstanceOf[WindowExecTransformer]
    val rankAttribute = window.windowExpression.head.toAttribute
    val payload = window.child.output.find(_.name == "rank_conjunct_payload").get
    val residual = EqualTo(payload, Literal(7))
    val plan = original.copy(condition = And(EqualTo(rankAttribute, Literal(1)), residual))
    val originalOutputExprIds = plan.output.map(_.exprId)

    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 1)
    assert(stats.insertedHashExchanges == 1)
    assert(rewritten.collect { case _: WindowExecTransformer => 1 }.size == 1)
    assert(rewritten.collect { case _: SortExecTransformer => 1 }.size == 1)
    val filters = rewritten.collect { case filter: FilterExecTransformer => filter }
    assert(filters.size == 1)
    assert(filters.head.condition.semanticEquals(plan.condition))
    assertRetainsOnlyPartial(rewritten)
    assert(rewritten.output.map(_.exprId) == originalOutputExprIds)
  }

  test("semantic Window rewrite resolves deterministic aliases on both sides of the local Sort") {
    val partition = AttributeReference("aliased_partition", LongType)()
    val order = AttributeReference("aliased_order", LongType)()
    val payload = AttributeReference("aliased_payload", IntegerType)()
    val baseRank = Rank(Seq(order))
    val baseOrderSpec = Seq(SortOrder(order, Ascending))
    val scan = TestLeaf(Seq(partition, order, payload))
    val partial = WindowGroupLimitExecTransformer(
      Seq(partition),
      baseOrderSpec,
      baseRank,
      limit = 1,
      GlutenPartial,
      scan)
    val finalGroupLimit = WindowGroupLimitExecTransformer(
      Seq(partition),
      baseOrderSpec,
      baseRank,
      limit = 1,
      GlutenFinal,
      partial)
    val duplicateOrder = Alias(order, "aliased_duplicate_order")()
    val preSortProject =
      ProjectExecTransformer(finalGroupLimit.output :+ duplicateOrder, finalGroupLimit)
    val sort = SortExecTransformer(
      Seq(SortOrder(partition, Ascending), SortOrder(duplicateOrder.toAttribute, Ascending)),
      global = false,
      preSortProject)
    val windowPartition = Alias(partition, "aliased_window_partition")()
    val windowOrder = Alias(duplicateOrder.toAttribute, "aliased_window_order")()
    val windowPayload = Alias(payload, "aliased_window_payload")()
    val postSortProject =
      ProjectExecTransformer(Seq(windowPartition, windowOrder, windowPayload), sort)
    val windowPartitionSpec = Seq(windowPartition.toAttribute)
    val windowOrderSpec = Seq(SortOrder(windowOrder.toAttribute, Ascending))
    val windowSpec = WindowSpecDefinition(
      windowPartitionSpec,
      windowOrderSpec,
      SpecifiedWindowFrame(RowFrame, UnboundedPreceding, CurrentRow))
    val rankAlias =
      Alias(WindowExpression(Rank(Seq(windowOrder.toAttribute)), windowSpec), "aliased_rank")()
    val window =
      WindowExecTransformer(Seq(rankAlias), windowPartitionSpec, windowOrderSpec, postSortProject)
    val residual = EqualTo(windowPayload.toAttribute, Literal(7))
    val plan =
      FilterExecTransformer(And(EqualTo(rankAlias.toAttribute, Literal(1)), residual), window)
    val originalOutputExprIds = plan.output.map(_.exprId)

    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 1)
    assert(stats.insertedHashExchanges == 1)
    assert(rewritten.collect { case _: WindowExecTransformer => 1 }.size == 1)
    assert(rewritten.collect { case _: SortExecTransformer => 1 }.size == 1)
    val filters = rewritten.collect { case filter: FilterExecTransformer => filter }
    assert(filters.size == 1)
    assert(filters.head.condition.semanticEquals(plan.condition))
    assertRetainsOnlyPartial(rewritten)
    assert(rewritten.output.map(_.exprId) == originalOutputExprIds)
  }

  test("a nondeterministic rank one residual retains the semantic Window path") {
    val original =
      rankFilterBranch(
        "rank_nondeterministic",
        includeExchange = false,
        options = BranchOptions(rankKind = "rank"))
        .asInstanceOf[FilterExecTransformer]
    val window = original.child.asInstanceOf[WindowExecTransformer]
    val rankAttribute = window.windowExpression.head.toAttribute
    val plan = original.copy(
      condition = And(EqualTo(rankAttribute, Literal(1)), EqualTo(Rand(42L), Literal(0.5))))

    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 1)
    assert(rewritten.collect { case _: WindowExecTransformer => 1 }.size == 1)
    assertRetainsOnlyPartial(rewritten)
  }

  test("a non-one rank predicate retains the semantic Window path") {
    val original =
      rankFilterBranch(
        "rank_two",
        includeExchange = false,
        options = BranchOptions(rankKind = "rank"))
        .asInstanceOf[FilterExecTransformer]
    val window = original.child.asInstanceOf[WindowExecTransformer]
    val rankAttribute = window.windowExpression.head.toAttribute
    val plan = original.copy(condition = EqualTo(rankAttribute, Literal(2)))

    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 1)
    assert(rewritten.collect { case _: WindowExecTransformer => 1 }.size == 1)
    assertRetainsOnlyPartial(rewritten)
  }

  test("an existing compatible native HASH exchange is retained") {
    val plan = rankFilterBranch("existing", includeExchange = true)
    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 1)
    assert(stats.insertedHashExchanges == 0)
    assert(rewritten.collect { case _: ColumnarShuffleExchangeExec => 1 }.size == 1)
    assertRetainsOnlyPartial(rewritten)
  }

  test("MPP value-stream wrappers and the synthetic hash project are retained") {
    val plan = rankFilterBranch(
      "mpp_wrapped",
      includeExchange = true,
      options = BranchOptions(partial = PartialOptions(mppBoundaryWrappers = true)))
    val originalHashProject = plan.collectFirst {
      case project: ProjectExecTransformer
          if project.projectList.headOption.exists(_.name == "hash_partition_key") =>
        project
    }.get
    val originalExprIds = originalHashProject.projectList.map(_.exprId)
    val originalWindow = plan.collectFirst { case node: WindowExecTransformer => node }.get
    val originalSort = plan.collectFirst { case node: SortExecTransformer => node }.get
    val originalInput = plan.collectFirst { case node: InputIteratorTransformer => node }.get
    val originalAdapter = plan.collectFirst { case node: ColumnarInputAdapter => node }.get
    val originalExchange =
      plan.collectFirst { case node: ColumnarShuffleExchangeExec => node }.get
    val originalWholeStage = plan.collectFirst { case node: WholeStageTransformer => node }.get
    val originalOutputExprIds = plan.output.map(_.exprId)
    val originalExchangeOutputExprIds = originalExchange.output.map(_.exprId)
    Seq(
      "window" -> originalWindow,
      "sort" -> originalSort,
      "input" -> originalInput,
      "adapter" -> originalAdapter,
      "exchange" -> originalExchange,
      "whole_stage" -> originalWholeStage,
      "hash_project" -> originalHashProject
    ).foreach { case (value, node) => node.setTagValue(SentinelTag, value) }
    originalWholeStage.setTagValue(CudfTag.CudfTag, true)
    originalWholeStage.setTagValue(CudfTag.GpuShuffleStageTag, true)
    originalHashProject.setTagValue(CudfTag.CudfTag, true)

    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 1)
    assert(stats.insertedHashExchanges == 0)
    assert(rewritten.collect { case _: InputIteratorTransformer => 1 }.size == 1)
    assert(rewritten.collect { case _: ColumnarInputAdapter => 1 }.size == 1)
    assert(rewritten.collect { case _: ColumnarShuffleExchangeExec => 1 }.size == 1)
    assert(rewritten.collect { case _: WholeStageTransformer => 1 }.size == 1)
    assert(rewritten.collect { case _: SortExecTransformer => 1 }.size == 1)
    assertRetainsOnlyPartial(rewritten)
    val rewrittenHashProject = rewritten.collectFirst {
      case project: ProjectExecTransformer
          if project.projectList.headOption.exists(_.name == "hash_partition_key") =>
        project
    }.get
    assert(rewrittenHashProject.projectList.map(_.exprId) == originalExprIds)
    val rewrittenWindow = rewritten.collectFirst { case node: WindowExecTransformer => node }.get
    val rewrittenSort = rewritten.collectFirst { case node: SortExecTransformer => node }.get
    val rewrittenInput = rewritten.collectFirst { case node: InputIteratorTransformer => node }.get
    val rewrittenAdapter = rewritten.collectFirst { case node: ColumnarInputAdapter => node }.get
    val rewrittenExchange =
      rewritten.collectFirst { case node: ColumnarShuffleExchangeExec => node }.get
    val rewrittenWholeStage =
      rewritten.collectFirst { case node: WholeStageTransformer => node }.get
    Seq(
      "window" -> rewrittenWindow,
      "sort" -> rewrittenSort,
      "input" -> rewrittenInput,
      "adapter" -> rewrittenAdapter,
      "exchange" -> rewrittenExchange,
      "whole_stage" -> rewrittenWholeStage,
      "hash_project" -> rewrittenHashProject
    ).foreach {
      case (value, node) =>
        assert(node.getTagValue(SentinelTag).contains(value))
    }
    assert(rewritten.output.map(_.exprId) == originalOutputExprIds)
    assert(rewrittenExchange.output.map(_.exprId) == originalExchangeOutputExprIds)
    assert(rewrittenWholeStage.output.map(_.exprId) == originalWholeStage.output.map(_.exprId))
    assert(rewrittenWholeStage.stageId == originalWholeStage.stageId)
    assert(rewrittenWholeStage.materializeInput == originalWholeStage.materializeInput)
    assert(!rewrittenWholeStage.wholeStageTransformerContextDefined)
    assert(rewrittenWholeStage.getTagValue(CudfTag.CudfTag).contains(true))
    assert(rewrittenWholeStage.getTagValue(CudfTag.GpuShuffleStageTag).contains(true))
    assert(rewrittenHashProject.getTagValue(CudfTag.CudfTag).contains(true))
    assert(rewritten.collect { case node => node.expressions }.flatten.forall(_.resolved))
  }

  test("only the exact synthetic hash project may bridge an exchange to Partial") {
    Seq("wrong_name", "non_hash", "wrong_keys", "wrong_seed", "reordered_tail").foreach {
      hashProjectKind =>
        val plan = rankFilterBranch(
          s"malformed_$hashProjectKind",
          includeExchange = true,
          options = BranchOptions(
            partial = PartialOptions(mppBoundaryWrappers = true, hashProjectKind = hashProjectKind))
        )

        val (rewritten, stats) =
          MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

        withClue(hashProjectKind) {
          assert(stats.rewrittenWindows == 0)
          assert(stats.insertedHashExchanges == 0)
          assert(rewritten.collect { case _: WindowGroupLimitExecTransformer => 1 }.size == 2)
        }
    }
  }

  test("an extra node inside WholeStage keeps both pruning boundaries") {
    val plan = rankFilterBranch(
      "whole_stage_extra_node",
      includeExchange = true,
      options = BranchOptions(
        partial = PartialOptions(mppBoundaryWrappers = true, wholeStageExtraFilter = true)))

    val (rewritten, stats) =
      MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 0)
    assert(rewritten.collect { case _: WindowGroupLimitExecTransformer => 1 }.size == 2)
  }

  test("a pre-project and its ExprIds survive removal of the Final group limit") {
    val plan = rankFilterBranch(
      "project",
      includeExchange = true,
      options = BranchOptions(child = ChildOptions(includePreProject = true)))
    val originalProject = plan.collectFirst { case project: ProjectExec => project }.get
    val originalExprIds = originalProject.output.map(_.exprId)

    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 1)
    val rewrittenProject = rewritten.collectFirst { case project: ProjectExec => project }.get
    assert(rewrittenProject.output.map(_.exprId) == originalExprIds)
    assertRetainsOnlyPartial(rewritten)
  }

  test("a nondeterministic pre-project keeps both group-limit pruning boundaries") {
    val plan = rankFilterBranch(
      "nondeterministic",
      includeExchange = true,
      options = BranchOptions(child = ChildOptions(includeRandomPreProject = true)))

    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 0)
    assert(stats.insertedHashExchanges == 0)
    assert(rewritten.collect { case _: WindowGroupLimitExecTransformer => 1 }.size == 2)
    val project = rewritten.collectFirst { case project: ProjectExec => project }.get
    assert(project.projectList.exists(!_.deterministic))
  }

  test("a global rank window is not rewritten") {
    val plan = rankFilterBranch(
      "global",
      includeExchange = false,
      options = BranchOptions(partitioned = false))
    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 0)
    assert(rewritten.collect { case _: WindowGroupLimitExecTransformer => 1 }.size == 2)
    assert(rewritten.collect { case _: ColumnarShuffleExchangeExec => 1 }.isEmpty)
  }

  test("dense_rank stays on the existing path until native streaming support is verified") {
    val plan = rankFilterBranch(
      "dense",
      includeExchange = false,
      options = BranchOptions(rankKind = "dense_rank"))
    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 0)
    assert(rewritten.collect { case _: WindowGroupLimitExecTransformer => 1 }.size == 2)
  }

  test("incomplete local ordering stays on the existing path") {
    val plan = rankFilterBranch(
      "bad_sort",
      includeExchange = false,
      options = BranchOptions(sort = SortOptions(validLocalSort = false)))
    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 0)
    assert(rewritten.collect { case _: WindowGroupLimitExecTransformer => 1 }.size == 2)
  }

  test("a local sort with trailing keys is not claimed as the native streaming contract") {
    val plan = rankFilterBranch(
      "extra_sort_key",
      includeExchange = false,
      options = BranchOptions(sort = SortOptions(extraLocalSortKey = true)))
    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 0)
    assert(stats.insertedHashExchanges == 0)
    assert(rewritten.collect { case _: WindowGroupLimitExecTransformer => 1 }.size == 2)
  }

  test("a mismatched Partial group limit leaves the complete pair untouched") {
    val plan = rankFilterBranch(
      "bad_partial",
      includeExchange = false,
      options = BranchOptions(partial = PartialOptions(matchingPartial = false)))
    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 0)
    assert(rewritten.collect { case _: WindowGroupLimitExecTransformer => 1 }.size == 2)
  }

  test("an intervening Filter prevents removal of a non-adjacent Partial group limit") {
    val plan = rankFilterBranch(
      "intervening_filter",
      includeExchange = false,
      options = BranchOptions(partial = PartialOptions(interveningFilterBeforePartial = true)))
    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 0)
    assert(stats.insertedHashExchanges == 0)
    assert(rewritten.collect { case _: WindowGroupLimitExecTransformer => 1 }.size == 2)
    assert(rewritten.collect { case _: FilterExecTransformer => 1 }.size == 2)
  }

  test("a missing partition attribute does not produce an invalid HASH exchange") {
    val plan = rankFilterBranch(
      "missing",
      includeExchange = false,
      options = BranchOptions(child = ChildOptions(omitPartitionFromChild = true)))
    val (rewritten, stats) = MppRankFilterWindowRewrite(plan, enabled = true, numPartitions = 4)

    assert(stats.rewrittenWindows == 0)
    assert(stats.insertedHashExchanges == 0)
    assert(rewritten.collect { case _: WindowGroupLimitExecTransformer => 1 }.size == 2)
  }

  private def assertRetainsOnlyPartial(plan: SparkPlan, expectedCount: Int = 1): Unit = {
    val groupLimits =
      plan.collect { case groupLimit: WindowGroupLimitExecTransformer => groupLimit }
    assert(groupLimits.size == expectedCount)
    assert(groupLimits.forall(_.mode == GlutenPartial))
  }

  private def rankFilterBranch(
      prefix: String,
      includeExchange: Boolean,
      options: BranchOptions = BranchOptions()): SparkPlan = {
    val partition = AttributeReference(s"${prefix}_partition", LongType)()
    val order = AttributeReference(s"${prefix}_order", LongType)()
    val payload = AttributeReference(s"${prefix}_payload", IntegerType)()
    val rankFunction = options.rankKind match {
      case "row_number" => RowNumber()
      case "rank" => Rank(Seq(order))
      case "dense_rank" => DenseRank(Seq(order))
      case other => fail(s"Unknown test rank function: $other")
    }
    val partitionSpec = if (options.partitioned) Seq(partition) else Seq.empty
    val orderSpec = Seq(SortOrder(order, Ascending))
    val frame = SpecifiedWindowFrame(RowFrame, UnboundedPreceding, CurrentRow)
    val windowSpec = WindowSpecDefinition(partitionSpec, orderSpec, frame)
    val rankAlias = Alias(WindowExpression(rankFunction, windowSpec), s"${prefix}_rank")()
    val childOutput =
      if (options.child.omitPartitionFromChild) Seq(order, payload)
      else Seq(partition, order, payload)
    val scan = TestLeaf(childOutput)
    val partialOrderSpec =
      if (options.partial.matchingPartial) orderSpec else Seq(SortOrder(payload, Ascending))
    val partial = WindowGroupLimitExecTransformer(
      partitionSpec,
      partialOrderSpec,
      rankFunction,
      limit = 1,
      GlutenPartial,
      scan)
    val partialChild =
      if (options.partial.interveningFilterBeforePartial) {
        FilterExecTransformer(EqualTo(payload, Literal(1)), partial)
      } else partial
    val exchangeInput =
      if (options.partial.mppBoundaryWrappers) {
        val hashExpression = options.partial.hashProjectKind match {
          case "valid" | "wrong_name" | "reordered_tail" =>
            new Murmur3Hash(partitionSpec)
          case "wrong_seed" =>
            new Murmur3Hash(partitionSpec, 41)
          case "non_hash" => Add(partitionSpec.head, Literal(1L))
          case "wrong_keys" => new Murmur3Hash(Seq(order))
          case other => fail(s"Unknown hash project kind: $other")
        }
        val hashName =
          if (options.partial.hashProjectKind == "wrong_name") "not_hash_partition_key"
          else "hash_partition_key"
        val passthrough =
          if (options.partial.hashProjectKind == "reordered_tail") partialChild.output.reverse
          else partialChild.output
        ProjectExecTransformer(Alias(hashExpression, hashName)() +: passthrough, partialChild)
      } else partialChild
    val wholeStageInput =
      if (options.partial.wholeStageExtraFilter) {
        FilterExecTransformer(EqualTo(payload, Literal(1)), exchangeInput)
      } else exchangeInput
    val stagedExchangeInput =
      if (options.partial.mppBoundaryWrappers) {
        WholeStageTransformer(wholeStageInput, materializeInput = true)(transformStageId = 17)
      } else exchangeInput
    val finalChild =
      if (includeExchange) {
        val exchange = ColumnarShuffleExchangeExec(
          HashPartitioning(partitionSpec, 4),
          stagedExchangeInput,
          ENSURE_REQUIREMENTS,
          partialChild.output,
          None)
        if (options.partial.mppBoundaryWrappers) {
          InputIteratorTransformer(ColumnarInputAdapter(exchange))
        } else exchange
      } else {
        partialChild
      }
    val finalGroupLimit = WindowGroupLimitExecTransformer(
      partitionSpec,
      orderSpec,
      rankFunction,
      limit = 1,
      GlutenFinal,
      finalChild)
    val sortChild =
      if (options.child.includeRandomPreProject) {
        ProjectExec(
          finalGroupLimit.output :+ Alias(Rand(42L), s"${prefix}_random")(),
          finalGroupLimit)
      } else if (options.child.includePreProject) {
        ProjectExec(finalGroupLimit.output, finalGroupLimit)
      } else finalGroupLimit
    val requiredSortOrder = partitionSpec.map(SortOrder(_, Ascending)) ++ orderSpec
    val sortOrder =
      if (!options.sort.validLocalSort) orderSpec
      else if (options.sort.extraLocalSortKey) requiredSortOrder :+ SortOrder(payload, Ascending)
      else requiredSortOrder
    val sort = SortExecTransformer(sortOrder, global = false, sortChild)
    val window = WindowExecTransformer(Seq(rankAlias), partitionSpec, orderSpec, sort)

    FilterExecTransformer(EqualTo(rankAlias.toAttribute, Literal(1)), window)
  }
}
