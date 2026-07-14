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
package org.apache.spark.sql.execution.utils

import org.apache.spark.RangePartitioner
import org.apache.spark.rdd.{PartitionPruningRDD, RDD}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, BoundReference, Descending, NullsFirst, NullsLast, SortOrder, UnsafeProjection, UnsafeRow}
import org.apache.spark.sql.catalyst.expressions.codegen.LazilyGeneratedOrdering
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._

import com.fasterxml.jackson.databind.ObjectMapper

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.hashing.byteswap32

/**
 * Computes Spark-compatible range bounds for an MPP exchange.
 *
 * The input plan remains an RDD of columnar batches. Executors scan its keys to build bounded
 * reservoir samples; only those bounded samples and the final unique boundaries are collected on
 * the driver. This deliberately mirrors Spark's [[RangePartitioner]] algorithm instead of inventing
 * an MPP-specific approximation.
 */
object MppRangeBoundsGenerator {

  case class Result(json: String, boundaryCount: Int) {
    val effectivePartitions: Int = boundaryCount + 1
  }

  private val SupportedTypeNames = Set(
    BooleanType.typeName,
    ByteType.typeName,
    ShortType.typeName,
    IntegerType.typeName,
    LongType.typeName,
    StringType.typeName,
    DateType.typeName,
    TimestampType.typeName,
    "timestamp_ntz"
  )

  def supports(dataType: DataType): Boolean = SupportedTypeNames.contains(dataType.typeName)

  def generate(
      samplePlan: SparkPlan,
      outputAttributes: Seq[Attribute],
      ordering: Seq[SortOrder],
      requestedPartitions: Int): Result = {
    require(samplePlan != null, "MPP RANGE requires a producer plan for bounded sampling")
    require(
      requestedPartitions > 0,
      s"MPP RANGE requires positive partitions: $requestedPartitions")
    require(ordering.nonEmpty, "MPP RANGE requires at least one SortOrder")
    ordering.foreach {
      order =>
        require(
          order.child.isInstanceOf[Attribute],
          s"MPP RANGE D1 only supports attribute sort keys, found ${order.child.sql}")
        require(
          supports(order.dataType),
          s"MPP RANGE D1 does not support sort-key type ${order.dataType.catalogString}")
    }

    val projected: RDD[(InternalRow, Null)] = samplePlan.executeColumnar().mapPartitions {
      batches =>
        val projection = UnsafeProjection.create(ordering.map(_.child), outputAttributes)
        batches.flatMap {
          batch =>
            // RangePartitioner.sketch consumes the full iterator on executors, but retains only a
            // bounded reservoir. Copy the projected key because UnsafeProjection reuses its row.
            ExecUtil
              .convertColumnarToRow(batch)
              .map(row => (projection(row).copy(), null))
        }
    }

    val orderingAttributes = ordering.zipWithIndex.map {
      case (order, index) =>
        order.copy(child = BoundReference(index, order.dataType, order.nullable))
    }
    implicit val keyOrdering: Ordering[InternalRow] =
      new LazilyGeneratedOrdering(orderingAttributes)

    val bounds = determineBounds(
      requestedPartitions,
      projected,
      SQLConf.get.rangeExchangeSampleSizePerPartition)
    Result(encode(ordering, bounds), bounds.length)
  }

  private def determineBounds(
      partitions: Int,
      keyedRdd: RDD[(InternalRow, Null)],
      samplePointsPerPartitionHint: Int)(implicit
      ordering: Ordering[InternalRow]): Array[InternalRow] = {
    if (partitions <= 1 || keyedRdd.partitions.isEmpty) {
      return Array.empty[InternalRow]
    }

    val sampleSize = math.min(samplePointsPerPartitionHint.toDouble * partitions, 1e6)
    val sampleSizePerPartition =
      math.ceil(3.0 * sampleSize / keyedRdd.partitions.length).toInt
    val keys = keyedRdd.map(_._1)
    val (numItems, sketched) = RangePartitioner.sketch(keys, sampleSizePerPartition)
    if (numItems == 0L) {
      return Array.empty[InternalRow]
    }

    val fraction = math.min(sampleSize / math.max(numItems, 1L), 1.0)
    val candidates = ArrayBuffer.empty[(InternalRow, Float)]
    val imbalancedPartitions = mutable.Set.empty[Int]
    sketched.foreach {
      case (partition, count, sample) =>
        if (fraction * count > sampleSizePerPartition) {
          imbalancedPartitions += partition
        } else if (sample.nonEmpty) {
          val weight = (count.toDouble / sample.length).toFloat
          sample.foreach(key => candidates += ((key, weight)))
        }
    }

    if (imbalancedPartitions.nonEmpty) {
      val imbalanced = new PartitionPruningRDD(keys, imbalancedPartitions.contains)
      val seed = byteswap32(-keyedRdd.id - 1)
      val reSampled = imbalanced.sample(withReplacement = false, fraction, seed).collect()
      val weight = (1.0 / fraction).toFloat
      reSampled.foreach(key => candidates += ((key, weight)))
    }

    RangePartitioner.determineBounds(candidates, math.min(partitions, candidates.size))
  }

  private[utils] def encode(ordering: Seq[SortOrder], bounds: Array[InternalRow]): String = {
    val mapper = new ObjectMapper
    val root = mapper.createObjectNode()
    root.put("version", 1)
    val keys = root.putArray("keys")
    ordering.foreach {
      order =>
        val key = keys.addObject()
        key.put("sparkType", order.dataType.catalogString)
        order.direction match {
          case Ascending => key.put("ascending", true)
          case Descending => key.put("ascending", false)
        }
        order.nullOrdering match {
          case NullsFirst => key.put("nullsFirst", true)
          case NullsLast => key.put("nullsFirst", false)
        }
    }

    val encodedBounds = root.putArray("bounds")
    bounds.foreach {
      bound =>
        val row = bound.asInstanceOf[UnsafeRow]
        val encodedRow = encodedBounds.addArray()
        ordering.indices.foreach {
          index =>
            val value = encodedRow.addObject()
            if (row.isNullAt(index)) {
              value.put("isNull", true)
            } else {
              value.put("isNull", false)
              ordering(index).dataType match {
                case BooleanType => value.put("value", row.getBoolean(index))
                case ByteType => value.put("value", row.getByte(index).toInt)
                case ShortType => value.put("value", row.getShort(index).toInt)
                case IntegerType | DateType => value.put("value", row.getInt(index))
                case LongType | TimestampType => value.put("value", row.getLong(index))
                case t if t.typeName == "timestamp_ntz" => value.put("value", row.getLong(index))
                case StringType => value.put("value", row.getUTF8String(index).toString)
                case other =>
                  throw new IllegalArgumentException(
                    s"MPP RANGE D1 cannot encode ${other.catalogString}")
              }
            }
        }
    }
    mapper.writeValueAsString(root)
  }
}
