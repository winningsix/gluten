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

import org.apache.spark.sql.catalyst.plans.physical.{Partitioning, UnknownPartitioning}
import org.apache.spark.sql.connector.read.InputPartition

import scala.collection.mutable.ArrayBuffer
import scala.util.control.NonFatal

/** Coalesces adjacent V2 input partitions without changing their child order. */
private[execution] object InputPartitionCoalescer {

  def coalesceAdjacentIfSupported(
      partitions: Seq[Seq[InputPartition]],
      targetBytes: Long,
      mppEnabled: Boolean,
      singleTaskMode: Boolean,
      outputPartitioning: Partitioning,
      hasOutputOrdering: Boolean,
      hasKeyGroupedPartitioning: Boolean,
      hasCommonPartitionValues: Boolean,
      applyPartialClustering: Boolean,
      replicatePartitions: Boolean)(
      partitionInfo: InputPartition => Option[(Long, String)]): Seq[Seq[InputPartition]] = {
    if (
      targetBytes <= 0 || !mppEnabled || singleTaskMode ||
      !outputPartitioning.isInstanceOf[UnknownPartitioning] || hasOutputOrdering ||
      hasKeyGroupedPartitioning || hasCommonPartitionValues || applyPartialClustering ||
      replicatePartitions || partitions.size <= 1
    ) {
      return partitions
    }

    val coalesced = ArrayBuffer.empty[Seq[InputPartition]]
    val current = ArrayBuffer.empty[InputPartition]
    var currentBytes = 0L
    var currentMergeKey: Option[String] = None

    def flushCurrent(): Unit = {
      if (current.nonEmpty) {
        coalesced += current.toVector
        current.clear()
        currentBytes = 0L
        currentMergeKey = None
      }
    }

    partitions.foreach {
      group =>
        groupInfo(group, partitionInfo) match {
          case Some((groupBytes, mergeKey)) if groupBytes <= targetBytes =>
            // This form avoids overflowing currentBytes + groupBytes.
            if (
              current.nonEmpty &&
              (currentMergeKey.exists(_ != mergeKey) || groupBytes > targetBytes - currentBytes)
            ) {
              flushCurrent()
            }
            current ++= group
            currentBytes += groupBytes
            currentMergeKey = Some(mergeKey)

          case _ =>
            // Unknown, invalid, empty, exceptional, or oversized groups are barriers. Keeping the
            // original group intact also preserves any source-specific task-group contract.
            flushCurrent()
            coalesced += group
        }
    }
    flushCurrent()

    if (coalesced.size == partitions.size) partitions else coalesced.toVector
  }

  private def groupInfo(
      group: Seq[InputPartition],
      partitionInfo: InputPartition => Option[(Long, String)]): Option[(Long, String)] = {
    if (group.isEmpty) {
      return None
    }

    var total = 0L
    var mergeKey: Option[String] = None
    group.foreach {
      partition =>
        val info = try {
          partitionInfo(partition)
        } catch {
          case NonFatal(_) => None
        }
        info match {
          case Some((value, key))
              if key != null && value >= 0 && value <= Long.MaxValue - total &&
                mergeKey.forall(_ == key) =>
            total += value
            mergeKey = Some(key)
          case _ => return None
        }
    }
    mergeKey.map(key => (total, key))
  }
}
