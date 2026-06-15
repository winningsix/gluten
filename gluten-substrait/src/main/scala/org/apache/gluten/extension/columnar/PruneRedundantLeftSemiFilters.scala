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

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeSet, EqualTo}
import org.apache.spark.sql.catalyst.expressions.{Expression, PredicateHelper}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Filter, Join, JoinHint}
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Project, SubqueryAlias}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreePattern.JOIN
import org.apache.spark.sql.execution.datasources.LogicalRelation
import org.apache.spark.sql.internal.SQLConf

/**
 * Remove redundant deterministic equi LeftSemi filters inferred through an inner-join equality
 * graph.
 *
 * Spark can derive a second semi filter from constraints after rewriting an IN/EXISTS subquery.
 * Spark physical planning may hide the duplicate with ReusedExchange, but MPP fragment extraction
 * cannot safely share one producer fragment across multiple consumers yet. This rule only removes a
 * semi filter when another semi filter in the same inner-join cluster:
 *   - has an equivalent RHS plan,
 *   - uses the same RHS key positions,
 *   - has left keys proven equal through inner equi-join predicates, and
 *   - has no residual non-equi predicates or user hints.
 */
case class PruneRedundantLeftSemiFilters(spark: SparkSession)
  extends Rule[LogicalPlan]
  with PredicateHelper
  with Logging {

  private val confKey = "spark.gluten.mpp.pruneRedundantLeftSemiFilters"
  private val confDefault = "true"
  private val maxWrapperDepth = 4
  private type KeySignature = (String, String, Boolean)

  override def apply(plan: LogicalPlan): LogicalPlan = {
    registerPostSubqueryPass()

    if (!SQLConf.get.getConfString(confKey, confDefault).toBoolean || !plan.resolved) {
      return plan
    }

    val toDrop = redundantSemiJoinIds(plan)
    if (toDrop.isEmpty) {
      return plan
    }

    plan.transformDownWithPruning(_.containsPattern(JOIN)) {
      case join @ Join(left, _, LeftSemi, _, _) if toDrop(System.identityHashCode(join)) =>
        logDebug(
          "PruneRedundantLeftSemiFilters: removed redundant LeftSemi filter " +
            s"left=${planSummary(left)}")
        left
    }
  }

  private def registerPostSubqueryPass(): Unit = {
    val experimental = spark.experimental
    experimental.synchronized {
      if (!experimental.extraOptimizations.exists(_.isInstanceOf[PruneRedundantLeftSemiFilters])) {
        experimental.extraOptimizations =
          experimental.extraOptimizations :+ PruneRedundantLeftSemiFilters(spark)
        logDebug(
          "PruneRedundantLeftSemiFilters: registered into " +
            "spark.experimental.extraOptimizations for post-RewriteSubquery pass")
      }
    }
  }

  private case class SemiCandidate(
      join: Join,
      left: LogicalPlan,
      right: LogicalPlan,
      leftKeys: Seq[Attribute],
      rightKeyIndexes: Seq[Int],
      rightKeySignatures: Seq[KeySignature]) {
    val id: Int = System.identityHashCode(join)
  }

  private def redundantSemiJoinIds(plan: LogicalPlan): Set[Int] = {
    val drops = scala.collection.mutable.HashSet.empty[Int]

    plan.foreach {
      case root if root.containsPattern(JOIN) =>
        extractInnerJoinCluster(root).foreach {
          case (items, conditions) =>
            val candidates = items.flatMap(extractSemiCandidate)
            if (candidates.length >= 2) {
              val uf = equalityClasses(conditions)
              connectedDuplicateGroups(candidates, uf).foreach {
                group =>
                  val keeper = group.minBy(c => (realScanBytes(c.left), c.id))
                  group.foreach {
                    candidate =>
                      if (candidate.id != keeper.id) {
                        drops += candidate.id
                      }
                  }
              }
            }
        }
      case _ =>
    }

    drops.toSet
  }

  private def extractInnerJoinCluster(
      plan: LogicalPlan): Option[(Seq[LogicalPlan], Seq[Expression])] = {
    val items = scala.collection.mutable.ArrayBuffer.empty[LogicalPlan]
    val conditions = scala.collection.mutable.ArrayBuffer.empty[Expression]

    def collect(node: LogicalPlan, depth: Int): Unit = node match {
      case Join(left, right, Inner, condition, _) =>
        condition.foreach(conditions += _)
        collect(left, depth)
        collect(right, depth)
      case Project(projectList, child)
          if depth < maxWrapperDepth && isAttributeOnlyProject(projectList) &&
            containsInnerJoin(child, depth + 1) =>
        collect(child, depth + 1)
      case Filter(condition, child)
          if depth < maxWrapperDepth && condition.deterministic &&
            containsInnerJoin(child, depth + 1) =>
        conditions += condition
        collect(child, depth + 1)
      case SubqueryAlias(_, child) if depth < maxWrapperDepth =>
        collect(child, depth + 1)
      case other =>
        items += other
    }

    collect(plan, 0)
    if (items.length >= 2 && conditions.nonEmpty) {
      Some(items.toSeq -> conditions.toSeq)
    } else {
      None
    }
  }

  private def containsInnerJoin(plan: LogicalPlan, depth: Int): Boolean = plan match {
    case Join(_, _, Inner, _, _) => true
    case Project(projectList, child)
        if depth < maxWrapperDepth && isAttributeOnlyProject(projectList) =>
      containsInnerJoin(child, depth + 1)
    case Filter(_, child) if depth < maxWrapperDepth =>
      containsInnerJoin(child, depth + 1)
    case SubqueryAlias(_, child) if depth < maxWrapperDepth =>
      containsInnerJoin(child, depth + 1)
    case _ => false
  }

  private def isAttributeOnlyProject(projectList: Seq[Expression]): Boolean = {
    projectList.forall(_.isInstanceOf[Attribute])
  }

  private def extractSemiCandidate(plan: LogicalPlan): Option[SemiCandidate] = {
    plan match {
      case join @ Join(left, right, LeftSemi, Some(condition), hint)
          if condition.deterministic && !hasUserHint(hint) && !join.isStreaming =>
        extractEquiKeys(condition, left.outputSet, right.output).map {
          case (leftKeys, rightKeyIndexes, rightKeySignatures) =>
            SemiCandidate(join, left, right, leftKeys, rightKeyIndexes, rightKeySignatures)
        }
      case _ => None
    }
  }

  private def extractEquiKeys(
      condition: Expression,
      leftOutput: AttributeSet,
      rightOutput: Seq[Attribute]): Option[(Seq[Attribute], Seq[Int], Seq[KeySignature])] = {
    val rightSet = AttributeSet(rightOutput)
    val leftKeys = scala.collection.mutable.ArrayBuffer.empty[Attribute]
    val rightIndexes = scala.collection.mutable.ArrayBuffer.empty[Int]
    val rightKeySignatures = scala.collection.mutable.ArrayBuffer.empty[KeySignature]

    splitConjunctivePredicates(condition).foreach {
      case EqualTo(l: Attribute, r: Attribute) =>
        extractKeyPair(l, r, leftOutput, rightOutput, rightSet) match {
          case Some((leftKey, rightIndex)) =>
            leftKeys += leftKey
            rightIndexes += rightIndex
            rightKeySignatures += keySignature(rightOutput(rightIndex))
          case None => return None
        }
      case _ => return None
    }

    if (leftKeys.nonEmpty) {
      Some((leftKeys.toSeq, rightIndexes.toSeq, rightKeySignatures.toSeq))
    } else {
      None
    }
  }

  private def keySignature(attribute: Attribute): KeySignature = {
    (attribute.name, attribute.dataType.catalogString, attribute.nullable)
  }

  private def extractKeyPair(
      first: Attribute,
      second: Attribute,
      leftOutput: AttributeSet,
      rightOutput: Seq[Attribute],
      rightSet: AttributeSet): Option[(Attribute, Int)] = {
    if (leftOutput.contains(first) && rightSet.contains(second)) {
      rightOutput.indexWhere(_.semanticEquals(second)) match {
        case -1 => None
        case idx => Some(first -> idx)
      }
    } else if (leftOutput.contains(second) && rightSet.contains(first)) {
      rightOutput.indexWhere(_.semanticEquals(first)) match {
        case -1 => None
        case idx => Some(second -> idx)
      }
    } else {
      None
    }
  }

  private def equalityClasses(conditions: Seq[Expression]): EquivClasses = {
    val uf = new EquivClasses
    conditions.flatMap(splitConjunctivePredicates).foreach {
      case EqualTo(l: Attribute, r: Attribute) => uf.union(l, r)
      case _ =>
    }
    uf
  }

  private def connectedDuplicateGroups(
      candidates: Seq[SemiCandidate],
      uf: EquivClasses): Seq[Seq[SemiCandidate]] = {
    val adjacency = scala.collection.mutable.Map.empty[Int, scala.collection.mutable.Set[Int]]
    candidates.indices.foreach(i => adjacency.getOrElseUpdate(i, scala.collection.mutable.Set(i)))

    for {
      i <- candidates.indices
      j <- (i + 1) until candidates.length
      if sameSemiFilter(candidates(i), candidates(j)) &&
        connectedLeftKeys(candidates(i), candidates(j), uf)
    } {
      adjacency(i) += j
      adjacency(j) += i
    }

    val seen = scala.collection.mutable.HashSet.empty[Int]
    candidates.indices.flatMap {
      start =>
        if (seen(start)) {
          None
        } else {
          val component = scala.collection.mutable.ArrayBuffer.empty[Int]
          val stack = scala.collection.mutable.Stack(start)
          seen += start
          while (stack.nonEmpty) {
            val current = stack.pop()
            component += current
            adjacency(current).foreach {
              next =>
                if (!seen(next)) {
                  seen += next
                  stack.push(next)
                }
            }
          }
          if (component.length >= 2) {
            Some(component.map(candidates).toSeq)
          } else {
            None
          }
        }
    }
  }

  private def sameSemiFilter(left: SemiCandidate, right: SemiCandidate): Boolean = {
    left.rightKeyIndexes == right.rightKeyIndexes &&
    left.rightKeySignatures == right.rightKeySignatures &&
    left.right.sameResult(right.right)
  }

  private def connectedLeftKeys(
      left: SemiCandidate,
      right: SemiCandidate,
      uf: EquivClasses): Boolean = {
    left.leftKeys.length == right.leftKeys.length &&
    left.leftKeys.zip(right.leftKeys).forall { case (l, r) => uf.connected(l, r) }
  }

  private def hasUserHint(hint: JoinHint): Boolean = {
    hint.leftHint.exists(_.strategy.isDefined) ||
    hint.rightHint.exists(_.strategy.isDefined)
  }

  private def realScanBytes(plan: LogicalPlan): BigInt = {
    var total = BigInt(0)
    plan.foreach {
      case lr: LogicalRelation => total += BigInt(lr.relation.sizeInBytes)
      case leaf if leaf.children.isEmpty => total += leaf.stats.sizeInBytes
      case _ =>
    }
    if (total > 0) total else plan.stats.sizeInBytes
  }

  private def planSummary(plan: LogicalPlan): String = {
    val attrs = plan.output.map(_.name).take(4).mkString(",")
    s"${plan.nodeName}[$attrs] bytes=${realScanBytes(plan)}"
  }

  final private class EquivClasses {
    private val parent = scala.collection.mutable.HashMap.empty[Long, Long]
    private def touch(a: Attribute): Long = {
      val k = a.exprId.id
      parent.getOrElseUpdate(k, k)
      k
    }
    private def find(x: Long): Long = {
      var r = x
      while (parent.getOrElse(r, r) != r) {
        r = parent(r)
      }
      r
    }
    def union(a: Attribute, b: Attribute): Unit = {
      val ra = find(touch(a))
      val rb = find(touch(b))
      if (ra != rb) {
        parent(rb) = ra
      }
    }
    def connected(a: Attribute, b: Attribute): Boolean = {
      parent.contains(a.exprId.id) && parent.contains(b.exprId.id) &&
      find(a.exprId.id) == find(b.exprId.id)
    }
  }
}
