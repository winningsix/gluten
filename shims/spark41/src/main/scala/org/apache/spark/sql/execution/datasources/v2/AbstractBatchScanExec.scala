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
package org.apache.spark.sql.execution.datasources.v2

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Expression, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.SinglePartition
import org.apache.spark.sql.connector.catalog.Table
import org.apache.spark.sql.connector.read.{Batch, InputPartition, PartitionReaderFactory, Scan}
import org.apache.spark.sql.execution.EmptyRDDWithPartitions
import org.apache.spark.util.ArrayImplicits._

import java.util.Objects

/**
 * Physical plan node for scanning a batch of data from a data source v2. This mirrors Spark 5's
 * BatchScanExec shape while keeping Gluten's grouped InputPartition view for native scan planning.
 */
abstract class AbstractBatchScanExec(
    output: Seq[AttributeReference],
    @transient scan: Scan,
    val runtimeFilters: Seq[Expression],
    ordering: Option[Seq[SortOrder]] = None,
    @transient table: Table,
    val keyGroupedPartitioning: Option[Seq[Expression]] = None)
  extends DataSourceV2ScanExecBase {

  @transient lazy val batch: Batch = if (scan == null) null else scan.toBatch

  override def equals(other: Any): Boolean = other match {
    case other: AbstractBatchScanExec =>
      this.batch != null && this.batch == other.batch &&
      this.runtimeFilters == other.runtimeFilters &&
      this.keyGroupedPartitioning == other.keyGroupedPartitioning
    case _ =>
      false
  }

  override def hashCode(): Int = Objects.hash(batch, runtimeFilters, keyGroupedPartitioning)

  @transient override lazy val inputPartitions: Seq[InputPartition] =
    batch.planInputPartitions().toImmutableArraySeq

  @transient private lazy val sparkFilteredPartitions: Seq[Option[InputPartition]] =
    PushDownUtils.replanWithRuntimeFilters(
      scan,
      runtimeFilters,
      table,
      output,
      outputPartitioning,
      inputPartitions)

  @transient protected lazy val filteredPartitions: Seq[Seq[InputPartition]] =
    sparkFilteredPartitions.map(_.toSeq)

  override lazy val readerFactory: PartitionReaderFactory = batch.createReaderFactory()

  override lazy val inputRDD: RDD[InternalRow] = {
    val rdd = if (sparkFilteredPartitions.isEmpty && outputPartitioning == SinglePartition) {
      new EmptyRDDWithPartitions(sparkContext, 1)
    } else {
      new DataSourceRDD(
        sparkContext,
        sparkFilteredPartitions,
        readerFactory,
        supportsColumnar,
        customMetrics)
    }
    postDriverMetrics(scan.reportDriverMetrics())
    rdd
  }
}
