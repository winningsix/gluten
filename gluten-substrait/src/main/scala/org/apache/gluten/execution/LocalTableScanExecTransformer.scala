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

import org.apache.gluten.expression.ConverterUtils
import org.apache.gluten.metrics.MetricsUpdater
import org.apache.gluten.substrait.SubstraitContext
import org.apache.gluten.substrait.expression.{ExpressionBuilder, LiteralNode}
import org.apache.gluten.substrait.rel.{RelBuilder, SplitInfo}
import org.apache.gluten.substrait.rel.LocalFilesNode.ReadFileFormat

import org.apache.spark.Partition
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, GenericInternalRow, SortOrder, UnsafeProjection}
import org.apache.spark.sql.catalyst.plans.physical.{Partitioning, SinglePartition}
import org.apache.spark.sql.execution.{RDDScanExec, SparkPlan}

import scala.collection.JavaConverters._

case class LocalTableScanExecTransformer(output: Seq[Attribute], var rows: Seq[InternalRow])
  extends LeafTransformSupport {

  rows = LocalTableScanExecTransformer.materializeRows(output, rows)

  override def outputPartitioning: Partitioning = SinglePartition

  override def outputOrdering: Seq[SortOrder] = Nil

  override def metricsUpdater(): MetricsUpdater = MetricsUpdater.None

  override def getSplitInfos: Seq[SplitInfo] = Nil

  override def getPartitions: Seq[Partition] = Nil

  override def getPartitionWithReadFileFormats: Seq[(Partition, ReadFileFormat)] = Nil

  override protected def doValidateInternal(): ValidationResult = {
    val context = new SubstraitContext
    val relNode = doTransform(context).root
    doNativeValidation(context, relNode)
  }

  override protected def doTransform(context: SubstraitContext): TransformContext = {
    val typeNodes = ConverterUtils.collectAttributeTypeNodes(output)
    val nameList = ConverterUtils.collectAttributeNamesWithExprId(output)
    val literalRows = rows.map(rowToLiteralNodes).map(_.asJava).asJava
    val readNode = RelBuilder.makeReadRelForVirtualTable(
      typeNodes,
      nameList,
      literalRows,
      context,
      context.nextOperatorId(this.nodeName))
    TransformContext(output, readNode)
  }

  private def rowToLiteralNodes(row: InternalRow): Seq[LiteralNode] = {
    output.zipWithIndex.map {
      case (attr, index) =>
        val typeNode = ConverterUtils.getTypeNode(attr.dataType, attr.nullable)
        val value =
          if (row.isNullAt(index)) {
            null
          } else {
            row.get(index, attr.dataType)
          }
        ExpressionBuilder.makeLiteral(value, typeNode)
    }
  }
}

object LocalTableScanExecTransformer {
  val MaxRows: Int = 1024

  private[execution] def materializeRows(
      output: Seq[Attribute],
      rows: Seq[InternalRow]): Seq[InternalRow] = {
    if (output.isEmpty) {
      rows.map(_ => new GenericInternalRow(0))
    } else {
      val projection = UnsafeProjection.create(output, output)
      rows.map(row => projection(row).copy())
    }
  }

  def supportsOneRowRelation(plan: SparkPlan): Boolean = plan match {
    case scan: RDDScanExec =>
      scan.output.isEmpty && scan.nodeName.contains("OneRowRelation")
    case _ => false
  }

  def oneRowRelation(output: Seq[Attribute]): LocalTableScanExecTransformer =
    LocalTableScanExecTransformer(output, Seq(InternalRow.empty))
}
