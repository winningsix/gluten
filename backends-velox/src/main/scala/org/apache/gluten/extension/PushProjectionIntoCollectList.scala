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

import org.apache.gluten.expression.aggregate.VeloxCollectList

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.AggregateExpression
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf

/**
 * Pushes deterministic expressions on fields of an exploded collect_list(struct(...)) into the
 * struct before the aggregate.
 *
 * For example, `collect_list(struct(payload)) -> explode -> hash(payload)` becomes
 * `collect_list(struct(hash(payload))) -> explode`. This is particularly important for Flux: the
 * smaller struct crosses the network exchange and is retained by both PARTIAL and FINAL
 * aggregation. The rewrite is deliberately conservative: it requires one non-null named struct, one
 * collect_list, one explode output, and post-project expressions that depend only on fields of that
 * output. Attribute exprIds at the query boundary are preserved.
 */
case class PushProjectionIntoCollectList(spark: SparkSession)
  extends Rule[LogicalPlan]
  with Logging {

  private val enabledKey = "spark.gluten.mpp.pushProjectionIntoCollectList.enabled"

  override def apply(plan: LogicalPlan): LogicalPlan = {
    val conf = SQLConf.get
    if (
      !conf.getConfString("spark.gluten.mpp.enabled", "false").toBoolean ||
      !conf.getConfString(enabledKey, "false").toBoolean
    ) {
      return plan
    }

    plan.transformUp { case project: Project => rewrite(project).getOrElse(project) }
  }

  private case class CollectShape(
      collectAlias: Alias,
      edgeAttribute: AttributeReference,
      edgeAlias: Alias,
      edgeStruct: CreateNamedStruct)

  private case class PushedField(alias: Alias, expression: Expression)

  private def rewrite(project: Project): Option[LogicalPlan] = {
    val generate = project.child match {
      case value: Generate => value
      case _ => return None
    }
    if (generate.outer || generate.generatorOutput.size != 1) {
      return None
    }

    val collectedAttribute = generate.generator match {
      case Explode(attribute: AttributeReference) => attribute
      case _ => return None
    }
    val generatedAttribute = generate.generatorOutput.head

    val (filter, aggregate) = generate.child match {
      case value @ Filter(_, child: Aggregate) => (Some(value), child)
      case value: Aggregate => (None, value)
      case _ => return None
    }
    val inputProject = aggregate.child match {
      case value: Project => value
      case _ => return None
    }

    val shape = collectShape(aggregate, inputProject, collectedAttribute).getOrElse(return None)
    val pushed = project.projectList.flatMap {
      case alias: Alias if references(alias.child, generatedAttribute) =>
        if (
          !alias.child.deterministic ||
          !alias.child.references.forall(_.exprId == generatedAttribute.exprId)
        ) {
          return None
        }
        val substituted = alias.child.transformUp {
          case field: GetStructField
              if field.child
                .isInstanceOf[AttributeReference] && field.child
                .asInstanceOf[AttributeReference]
                .exprId == generatedAttribute.exprId =>
            shape.edgeStruct.valExprs(field.ordinal)
        }
        if (substituted.references.exists(_.exprId == generatedAttribute.exprId)) {
          return None
        }
        Some(PushedField(alias, substituted))
      case expression if references(expression, generatedAttribute) => return None
      case _ => None
    }

    // Avoid repeatedly rewriting the already-pushed fixed-point result.
    val hasDerivedExpression = pushed.exists {
      case PushedField(alias, _) =>
        alias.child match {
          case field: GetStructField =>
            !field.child
              .isInstanceOf[AttributeReference] || field.child
              .asInstanceOf[AttributeReference]
              .exprId != generatedAttribute.exprId
          case _ => true
        }
    }
    if (pushed.isEmpty || !hasDerivedExpression) {
      return None
    }

    val newStruct = CreateNamedStruct(pushed.flatMap {
      field => Seq(Literal(field.alias.name), field.expression)
    })
    val newEdgeAlias = Alias(newStruct, shape.edgeAlias.name)(
      shape.edgeAlias.exprId,
      shape.edgeAlias.qualifier,
      shape.edgeAlias.explicitMetadata,
      shape.edgeAlias.nonInheritableMetadataKeys)
    val newEdgeAttribute = newEdgeAlias.toAttribute
    val newInputProject = inputProject.copy(projectList = inputProject.projectList.map {
      case alias: Alias if alias.exprId == shape.edgeAlias.exprId => newEdgeAlias
      case expression => expression
    })

    val newAggregate = aggregate.copy(
      aggregateExpressions = aggregate.aggregateExpressions.map {
        expression =>
          expression
            .transformUp {
              case attribute: AttributeReference
                  if attribute.exprId == shape.edgeAttribute.exprId =>
                newEdgeAttribute
            }
            .asInstanceOf[NamedExpression]
      },
      child = newInputProject
    )
    val newCollectedAttribute = newAggregate.output
      .collectFirst {
        case attribute if attribute.exprId == shape.collectAlias.exprId => attribute
      }
      .getOrElse(return None)

    def replaceCollected(expression: Expression): Expression = expression.transformUp {
      case attribute: AttributeReference if attribute.exprId == collectedAttribute.exprId =>
        newCollectedAttribute
    }

    val newGenerateChild = filter match {
      case Some(value) =>
        value.copy(condition = replaceCollected(value.condition), child = newAggregate)
      case None => newAggregate
    }
    val newGenerator = replaceCollected(generate.generator).asInstanceOf[Generator]
    val newGeneratedAttribute = AttributeReference(
      generatedAttribute.name,
      newStruct.dataType,
      generatedAttribute.nullable,
      generatedAttribute.metadata)(generatedAttribute.exprId, generatedAttribute.qualifier)
    val newGenerate = generate.copy(
      generator = newGenerator,
      generatorOutput = Seq(newGeneratedAttribute),
      child = newGenerateChild)

    val pushedOrdinals = pushed.zipWithIndex.map {
      case (field, ordinal) => field.alias.exprId -> ordinal
    }.toMap
    val newProjectList = project.projectList.map {
      case alias: Alias if pushedOrdinals.contains(alias.exprId) =>
        val field =
          GetStructField(newGeneratedAttribute, pushedOrdinals(alias.exprId), Some(alias.name))
        Alias(field, alias.name)(
          alias.exprId,
          alias.qualifier,
          alias.explicitMetadata,
          alias.nonInheritableMetadataKeys)
      case expression => expression
    }

    logInfo(
      s"PushProjectionIntoCollectList: pushed ${pushed.size} fields and changed collected " +
        s"element type from ${shape.edgeStruct.dataType.catalogString} to " +
        s"${newStruct.dataType.catalogString}")
    Some(project.copy(projectList = newProjectList, child = newGenerate))
  }

  private def collectShape(
      aggregate: Aggregate,
      inputProject: Project,
      collectedAttribute: AttributeReference): Option[CollectShape] = {
    val collectAlias = aggregate.aggregateExpressions
      .collectFirst {
        case alias: Alias if alias.exprId == collectedAttribute.exprId => alias
      }
      .getOrElse(return None)
    val edgeAttribute = collectAlias.child match {
      case aggregateExpression: AggregateExpression
          if !aggregateExpression.isDistinct && aggregateExpression.filter.isEmpty =>
        aggregateExpression.aggregateFunction match {
          case collect: VeloxCollectList =>
            collect.child match {
              case attribute: AttributeReference => attribute
              case _ => return None
            }
          case _ => return None
        }
      case _ => return None
    }
    val edgeAlias = inputProject.projectList
      .collectFirst {
        case alias: Alias if alias.exprId == edgeAttribute.exprId => alias
      }
      .getOrElse(return None)
    val edgeStruct = edgeAlias.child match {
      case struct: CreateNamedStruct if !struct.nullable => struct
      case _ => return None
    }
    Some(CollectShape(collectAlias, edgeAttribute, edgeAlias, edgeStruct))
  }

  private def references(expression: Expression, attribute: Attribute): Boolean =
    expression.references.exists(_.exprId == attribute.exprId)
}
