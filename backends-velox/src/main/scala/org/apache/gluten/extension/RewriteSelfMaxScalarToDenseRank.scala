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
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, CurrentRow, DenseRank, Descending, EqualTo, Expression, IsNotNull, Literal, NamedExpression, NullsLast, ScalarSubquery, SortOrder, SpecifiedWindowFrame, UnboundedPreceding, WindowExpression, WindowSpecDefinition}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Max}
import org.apache.spark.sql.catalyst.plans.QueryPlan
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, LogicalPlan, Project, Window}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreePattern.FILTER
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.IntegerType

/**
 * Reuse an aggregate relation when a filter asks for that relation's maximum value.
 *
 * A common SQL shape (TPC-H Q15 is the canonical example) is:
 * {{{
 *   WITH r AS (...) SELECT ... FROM r
 *   WHERE r.value = (SELECT max(value) FROM r)
 * }}}
 * Spark inlines both CTE references and computes `r` twice. Native MPP can merge floating-point
 * partial sums in a different arrival order in the two copies. The two mathematically identical
 * values can then differ by a few ULPs, making exact equality nondeterministically return no row.
 *
 * For a self-max scalar this rule applies the exact, tie-preserving equivalence:
 * {{{
 *   value = scalar(max(value))
 *       <=>
 *   dense_rank() over (order by value desc nulls last) = 1 AND value IS NOT NULL
 * }}}
 * The explicit null test preserves SQL MAX/equality semantics for all-null input. Spark turns the
 * rank predicate into Partial/Final WindowGroupLimit, so only local maxima cross the global
 * exchange. The relation is evaluated once and all rows tied for the maximum are retained.
 *
 * The match is deliberately narrow: both sides must be aggregates with equivalent grouping, value
 * expressions and inputs. It is enabled only for MPP and can be disabled with
 * `spark.gluten.mpp.rewriteSelfMaxScalarToDenseRank=false`.
 */
case class RewriteSelfMaxScalarToDenseRank(spark: SparkSession)
  extends Rule[LogicalPlan]
  with Logging {

  private val confKey = "spark.gluten.mpp.rewriteSelfMaxScalarToDenseRank"

  override def apply(plan: LogicalPlan): LogicalPlan = {
    val conf = SQLConf.get
    if (
      plan.isStreaming ||
      !conf.getConfString("spark.gluten.mpp.enabled", "false").toBoolean ||
      !conf.getConfString(confKey, "true").toBoolean
    ) {
      return plan
    }

    plan.transformUpWithPruning(_.containsPattern(FILTER)) {
      case filter @ Filter(condition, outerAggregate: Aggregate) =>
        rewriteFilter(filter, condition, outerAggregate).getOrElse(filter)
    }
  }

  private def rewriteFilter(
      filter: Filter,
      condition: Expression,
      outerAggregate: Aggregate): Option[LogicalPlan] = {
    val conjuncts = splitConjuncts(condition)
    val candidate = conjuncts.iterator
      .flatMap(extractSelfMaxCandidate)
      .collectFirst {
        case (equality, outerValue, innerAggregate, innerValue)
            if outerAggregate.outputSet.contains(outerValue) =>
          equivalentAggregates(outerAggregate, outerValue, innerAggregate, innerValue).map {
            case (reusableAggregate, restoredPredicates) =>
              (equality, outerValue, reusableAggregate, restoredPredicates)
          }
      }
      .flatten
    candidate.map {
      case (equality, outerValue, reusableAggregate, restoredPredicates) =>
        val orderSpec = Seq(SortOrder(outerValue, Descending, NullsLast, Seq.empty))
        val windowSpec = WindowSpecDefinition(
          partitionSpec = Seq.empty,
          orderSpec = orderSpec,
          frameSpecification = SpecifiedWindowFrame(
            org.apache.spark.sql.catalyst.expressions.RowFrame,
            UnboundedPreceding,
            CurrentRow)
        )
        val rankAlias = Alias(
          WindowExpression(DenseRank(orderSpec.map(_.child)), windowSpec),
          "__gluten_self_max_rank")()
        val ranked = Window(Seq(rankAlias), Seq.empty, orderSpec, reusableAggregate)
        val retainedConjuncts = conjuncts.filterNot(_ eq equality)
        val rankPredicate = EqualTo(rankAlias.toAttribute, Literal.create(1, IntegerType))
        val rewrittenCondition = combineConjuncts(
          retainedConjuncts ++ restoredPredicates ++ Seq(IsNotNull(outerValue), rankPredicate))
        val filtered = Filter(rewrittenCondition, ranked)
        logInfo(
          s"RewriteSelfMaxScalarToDenseRank: reused aggregate and replaced self-max scalar " +
            s"equality on ${outerValue.name} with tie-preserving dense_rank=1")
        Project(outerAggregate.output.map(a => a: NamedExpression), filtered)
    }
  }

  /**
   * Catalyst's InferFiltersFromConstraints can push `grouping_key IS NOT NULL` into only the outer
   * copy because that copy later joins on the key. The scalar-max copy then remains over the full
   * input, so the two aggregate children no longer satisfy `sameResult` even though the CTE was
   * originally identical.
   *
   * Remove only those grouping-key null predicates for equivalence checking and for the reused
   * aggregate. Reapply them to the corresponding grouping outputs after ranking. This is exact: the
   * maximum still includes a possible null-key group, just as the original scalar MAX did, while
   * the outer result continues to exclude that group.
   */
  private def normalizeGroupingNullFilters(
      aggregate: Aggregate): Option[(LogicalPlan, Seq[Expression])] = {
    var removedGroupingIndexes = Set.empty[Int]
    val normalizedChild = aggregate.child.transformUp {
      case Filter(filterCondition, child) =>
        val retained = splitConjuncts(filterCondition).filterNot {
          case IsNotNull(candidate) =>
            aggregate.groupingExpressions.zipWithIndex.collectFirst {
              case (grouping, index) if grouping.semanticEquals(candidate) => index
            } match {
              case Some(index) =>
                removedGroupingIndexes += index
                true
              case None => false
            }
          case _ => false
        }
        if (retained.isEmpty) child else Filter(combineConjuncts(retained), child)
    }

    val restoredPredicates = removedGroupingIndexes.toSeq.sorted.map {
      index =>
        groupingOutputAttribute(aggregate, aggregate.groupingExpressions(index))
          .map(IsNotNull)
    }
    if (restoredPredicates.forall(_.isDefined)) {
      Some((normalizedChild, restoredPredicates.flatten))
    } else {
      None
    }
  }

  private def groupingOutputAttribute(
      aggregate: Aggregate,
      grouping: Expression): Option[Attribute] = {
    aggregate.aggregateExpressions.collectFirst {
      case alias: Alias if alias.child.semanticEquals(grouping) => alias.toAttribute
      case named if named.semanticEquals(grouping) => named.toAttribute
    }
  }

  private def extractSelfMaxCandidate(
      expression: Expression): Option[(Expression, Attribute, Aggregate, Attribute)] = {
    expression match {
      case equality @ EqualTo(outerValue: Attribute, scalar: ScalarSubquery) =>
        extractMaxAggregate(scalar.plan).map {
          case (innerAggregate, innerValue) =>
            (equality, outerValue, innerAggregate, innerValue)
        }
      case equality @ EqualTo(scalar: ScalarSubquery, outerValue: Attribute) =>
        extractMaxAggregate(scalar.plan).map {
          case (innerAggregate, innerValue) =>
            (equality, outerValue, innerAggregate, innerValue)
        }
      case _ => None
    }
  }

  private def extractMaxAggregate(plan: LogicalPlan): Option[(Aggregate, Attribute)] = plan match {
    case Aggregate(Seq(), Seq(alias: Alias), child: Aggregate) =>
      alias.child match {
        case aggregate: AggregateExpression if !aggregate.isDistinct && aggregate.filter.isEmpty =>
          aggregate.aggregateFunction match {
            case Max(innerValue: Attribute) => Some((child, innerValue))
            case _ => None
          }
        // Keep the direct form for plans passed to the rule before Spark wraps aggregate
        // functions in AggregateExpression.
        case Max(innerValue: Attribute) => Some((child, innerValue))
        case _ => None
      }
    case Project(_, child) => extractMaxAggregate(child)
    case _ => None
  }

  private def equivalentAggregates(
      outer: Aggregate,
      outerValue: Attribute,
      inner: Aggregate,
      innerValue: Attribute): Option[(Aggregate, Seq[Expression])] = {
    val outerValueExpression = aggregateValueExpression(outer, outerValue)
    val innerValueExpression = aggregateValueExpression(inner, innerValue)
    val normalizedOuter = normalizeGroupingNullFilters(outer)
    val normalizedInner = normalizeGroupingNullFilters(inner)
    val equivalent =
      outer.groupingExpressions.length == inner.groupingExpressions.length &&
        outer.groupingExpressions.zip(inner.groupingExpressions).forall {
          case (left, right) =>
            equivalentExpressions(left, outer.child.output, right, inner.child.output)
        } &&
        outerValueExpression.exists {
          left =>
            innerValueExpression.exists {
              right => equivalentExpressions(left, outer.child.output, right, inner.child.output)
            }
        } &&
        normalizedOuter.exists {
          case (outerChild, _) =>
            normalizedInner.exists { case (innerChild, _) => outerChild.sameResult(innerChild) }
        }
    if (equivalent) {
      normalizedOuter.map {
        case (child, restoredPredicates) =>
          (
            Aggregate(outer.groupingExpressions, outer.aggregateExpressions, child),
            restoredPredicates)
      }
    } else {
      None
    }
  }

  private def equivalentExpressions(
      left: Expression,
      leftInput: Seq[Attribute],
      right: Expression,
      rightInput: Seq[Attribute]): Boolean = {
    QueryPlan.normalizeExpressions(left, leftInput).canonicalized ==
      QueryPlan.normalizeExpressions(right, rightInput).canonicalized
  }

  private def aggregateValueExpression(
      aggregate: Aggregate,
      output: Attribute): Option[Expression] = {
    aggregate.aggregateExpressions.collectFirst {
      case alias: Alias if alias.toAttribute.exprId == output.exprId => alias.child
      case named if named.toAttribute.exprId == output.exprId => named
    }
  }

  private def splitConjuncts(expression: Expression): Seq[Expression] = expression match {
    case And(left, right) => splitConjuncts(left) ++ splitConjuncts(right)
    case other => Seq(other)
  }

  private def combineConjuncts(expressions: Seq[Expression]): Expression = {
    expressions.reduceLeft(And)
  }
}
