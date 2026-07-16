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
import org.apache.spark.network.util.JavaUtils
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, BoundReference, Descending, NullsFirst, NullsLast, SortOrder, UnsafeProjection, UnsafeRow}
import org.apache.spark.sql.catalyst.expressions.codegen.LazilyGeneratedOrdering
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._

import com.fasterxml.jackson.databind.ObjectMapper

import java.util.concurrent.{CompletableFuture, ExecutionException}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.control.NonFatal

/**
 * Computes Spark-compatible range bounds for an MPP exchange.
 *
 * Each successful sampling task scans its input partition once and retains a deterministic, bounded
 * priority sample. Only those samples and the final unique boundaries are materialized on the
 * driver. This preserves Spark's range ordering and boundary selection semantics without an
 * optional second pass over imbalanced input partitions.
 *
 * The bounded row and byte shares are currently assigned statically per input partition and are not
 * transferable from empty partitions. A sparse layout therefore fails closed when it cannot supply
 * at least one candidate per requested RANGE peer. A future mergeable global reservoir can make
 * those unused shares transferable without another producer scan.
 */
object MppRangeBoundsGenerator {

  private val MaxSampleRowsKey = "spark.gluten.mpp.rangeSampleMaxRows"
  private val MaxSampleBytesKey = "spark.gluten.mpp.rangeSampleMaxBytes"
  private val DefaultMaxSampleRows = 200000
  private val DefaultMaxSampleBytes = 128L << 20

  // The byte cap covers exact serialized UnsafeRow key payload only. The independent row cap
  // bounds the number of driver objects; neither value is presented as an exact JVM heap estimate.

  /**
   * The count is the number of producer rows observed, not just the retained sample size. It is
   * used as the weight for this partition's sample by [[RangePartitioner.determineBounds]].
   */
  private[utils] case class PartitionSketch(
      count: Long,
      samples: Array[UnsafeRow],
      serializedKeyBytes: Long)

  private case class PrioritizedRow(priority: Long, row: UnsafeRow, serializedKeyBytes: Long)

  implicit private val PrioritizedRowOrdering: Ordering[PrioritizedRow] =
    Ordering.by[PrioritizedRow, Long](_.priority)

  case class Result(json: String, boundaryCount: Int) {
    val effectivePartitions: Int = boundaryCount + 1
  }

  private case class OrderingKey(
      outputOrdinal: Int,
      dataTypeJson: String,
      nullable: Boolean,
      ascending: Boolean,
      nullsFirst: Boolean)

  private case class CacheEntry(
      samplePlan: SparkPlan,
      ordering: Seq[OrderingKey],
      requestedPartitions: Int,
      result: CompletableFuture[Result])

  /**
   * Bounds cache scoped to one MPP query launch. The launch owner supplies its Spark SQL execution
   * id for diagnostics; no entry is shared with a subsequent action/launch. Equivalent concurrent
   * requests join one future, and a failed owner removes the entry before propagating the original
   * failure so a later request is never poisoned.
   */
  final class QueryCache(val queryExecutionId: String) {
    require(queryExecutionId != null && queryExecutionId.nonEmpty)

    private val entries = ArrayBuffer.empty[CacheEntry]

    def getOrCompute(
        samplePlan: SparkPlan,
        outputAttributes: Seq[Attribute],
        ordering: Seq[SortOrder],
        requestedPartitions: Int)(compute: => Result): (Result, Boolean) = {
      val orderingKey = cacheOrderingKey(outputAttributes, ordering)
      var owner = false
      val entry = entries.synchronized {
        entries
          .find(
            cached =>
              cached.requestedPartitions == requestedPartitions &&
                cached.ordering == orderingKey &&
                sameSamplePlan(cached.samplePlan, samplePlan))
          .getOrElse {
            owner = true
            val created = CacheEntry(
              samplePlan,
              orderingKey,
              requestedPartitions,
              new CompletableFuture[Result]())
            entries += created
            created
          }
      }

      if (owner) {
        try {
          val generated = compute
          require(generated != null, "MPP RANGE bounds generator returned null")
          entry.result.complete(generated)
          (generated, false)
        } catch {
          case throwable: Throwable =>
            entries.synchronized {
              // Publish the original failure before removing the key, while holding the same lock.
              // Existing waiters observe this future; a retry cannot install a second generation
              // until the failed generation is complete and no longer discoverable.
              entry.result.completeExceptionally(throwable)
              entries -= entry
            }
            throw throwable
        }
      } else {
        try {
          (entry.result.get(), true)
        } catch {
          case interrupted: InterruptedException =>
            Thread.currentThread().interrupt()
            throw interrupted
          case execution: ExecutionException =>
            throw execution.getCause
        }
      }
    }
  }

  def supports(dataType: DataType): Boolean = dataType match {
    case BooleanType | ByteType | ShortType | IntegerType | LongType | StringType | DateType |
        TimestampType =>
      true
    // Keep older Spark profiles source-compatible: timestamp_ntz may not expose a stable singleton
    // there. Every other accepted type uses exact singleton matching, which deliberately rejects
    // Spark 4 collated StringType instances even though they share the "string" typeName.
    case timestampNtz if timestampNtz.typeName == "timestamp_ntz" => true
    case _ => false
  }

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

    val orderingAttributes = ordering.zipWithIndex.map {
      case (order, index) =>
        order.copy(child = BoundReference(index, order.dataType, order.nullable))
    }
    implicit val keyOrdering: Ordering[InternalRow] =
      new LazilyGeneratedOrdering(orderingAttributes)

    val input = samplePlan.executeColumnar()
    val inputPartitions = input.partitions.length
    if (requestedPartitions <= 1 || inputPartitions == 0) {
      return Result(encode(ordering, Array.empty[InternalRow]), 0)
    }
    val samplePointsPerPartitionHint = SQLConf.get.rangeExchangeSampleSizePerPartition
    val maxSampleRows = positiveIntConf(MaxSampleRowsKey, DefaultMaxSampleRows)
    val maxSampleBytes = positiveBytesConf(MaxSampleBytesKey, DefaultMaxSampleBytes)
    validateGlobalLimits(
      inputPartitions,
      requestedPartitions,
      samplePointsPerPartitionHint,
      maxSampleRows,
      maxSampleBytes)
    val inputRddId = input.id
    val bounds = determineBounds(
      requestedPartitions,
      inputPartitions,
      samplePointsPerPartitionHint,
      maxSampleRows,
      maxSampleBytes,
      input
        .mapPartitionsWithIndex {
          case (partitionId, batches) =>
            val projection = UnsafeProjection.create(ordering.map(_.child), outputAttributes)
            val projectedRows = batches.flatMap {
              batch =>
                ExecUtil
                  .convertColumnarToRow(batch)
                  .map(row => projection(row).asInstanceOf[UnsafeRow])
            }
            val limits = partitionLimits(
              partitionId,
              inputPartitions,
              requestedPartitions,
              samplePointsPerPartitionHint,
              maxSampleRows,
              maxSampleBytes)
            Iterator.single(
              sketchPartition(
                projectedRows,
                limits._1,
                limits._2,
                sampleSeed(inputRddId, partitionId)))
        }
        .collect()
    )
    Result(encode(ordering, bounds), bounds.length)
  }

  private[utils] def determineBounds(
      partitions: Int,
      inputPartitions: Int,
      samplePointsPerPartitionHint: Int,
      maxSampleRows: Int,
      maxSampleBytes: Long,
      sketches: Array[PartitionSketch])(implicit
      ordering: Ordering[InternalRow]): Array[InternalRow] = {
    if (partitions <= 1 || inputPartitions == 0) {
      return Array.empty[InternalRow]
    }

    validateGlobalLimits(
      inputPartitions,
      partitions,
      samplePointsPerPartitionHint,
      maxSampleRows,
      maxSampleBytes)
    require(
      sketches.length == inputPartitions,
      s"MPP RANGE sampling returned ${sketches.length} sketches for $inputPartitions inputs")

    val retainedRows =
      sketches.foldLeft(0L)((sum, sketch) => Math.addExact(sum, sketch.samples.length))
    val retainedBytes = sketches.foldLeft(0L) {
      case (sum, sketch) => Math.addExact(sum, sketch.serializedKeyBytes)
    }
    val targetRows =
      globalSampleRowLimit(inputPartitions, partitions, samplePointsPerPartitionHint, maxSampleRows)
    require(
      retainedRows <= targetRows,
      s"MPP RANGE retained $retainedRows sample rows, exceeding the $targetRows-row limit")
    require(
      retainedBytes <= maxSampleBytes,
      s"MPP RANGE retained $retainedBytes serialized key bytes, exceeding the " +
        s"$maxSampleBytes-byte limit")

    sketches.zipWithIndex.foreach {
      case (sketch, partitionId) if sketch.count > 0L && sketch.samples.isEmpty =>
        throw new IllegalArgumentException(
          s"MPP RANGE input partition $partitionId had ${sketch.count} rows but produced no " +
            "bounded samples")
      case _ =>
    }

    val numItems = sketches.foldLeft(0L)((sum, sketch) => Math.addExact(sum, sketch.count))
    if (numItems == 0L) {
      return Array.empty[InternalRow]
    }
    require(
      numItems < partitions.toLong || retainedRows >= partitions.toLong,
      s"MPP RANGE found $numItems input rows but only $retainedRows bounded samples for " +
        s"$partitions requested partitions; sparse input partitions cannot transfer unused " +
        s"sample budget yet. Increase $MaxSampleRowsKey/$MaxSampleBytesKey or implement a " +
        "mergeable global reservoir"
    )

    val candidates = ArrayBuffer.empty[(InternalRow, Float)]
    sketches.foreach {
      sketch =>
        if (sketch.samples.nonEmpty) {
          val weight = (sketch.count.toDouble / sketch.samples.length).toFloat
          sketch.samples.foreach(key => candidates += ((key, weight)))
        }
    }

    RangePartitioner.determineBounds(candidates, math.min(partitions, candidates.size))
  }

  /**
   * Retains exactly the rows with the smallest deterministic random priorities. The serialized-key
   * byte budget is an admission guard for that uniform row reservoir: it never changes which rows
   * are selected. If the required selected set cannot fit, sampling fails closed.
   */
  private[utils] def sketchPartition(
      rows: Iterator[UnsafeRow],
      maxRows: Int,
      maxSerializedKeyBytes: Long,
      seed: Long): PartitionSketch = {
    require(maxRows > 0, s"MPP RANGE partition sample row limit must be positive: $maxRows")
    require(
      maxSerializedKeyBytes > 0L,
      s"MPP RANGE partition sample byte limit must be positive: $maxSerializedKeyBytes")

    val retained = mutable.PriorityQueue.empty[PrioritizedRow]
    val random = new java.util.Random(seed)
    var count = 0L
    var serializedKeyBytes = 0L

    rows.foreach {
      row =>
        count = Math.addExact(count, 1L)
        // UnsafeRow's backing byte array is exactly the serialized key payload transported to the
        // driver. Account that payload rather than JVM object-size estimates.
        val rowBytes = row.getSizeInBytes.toLong
        val priority = random.nextLong() & Long.MaxValue
        if (retained.size < maxRows) {
          val admittedBytes = Math.addExact(serializedKeyBytes, rowBytes)
          require(
            admittedBytes <= maxSerializedKeyBytes,
            s"MPP RANGE selected reservoir keys require $admittedBytes serialized bytes, " +
              s"exceeding the $maxSerializedKeyBytes-byte partition budget; increase " +
              s"$MaxSampleBytesKey"
          )
          // UnsafeProjection reuses its output row. Copy only selected keys instead of allocating
          // one UnsafeRow for every producer row scanned.
          retained.enqueue(PrioritizedRow(priority, row.copy(), rowBytes))
          serializedKeyBytes = admittedBytes
        } else if (priority < retained.head.priority) {
          val evicted = retained.head
          val replacementBytes = Math.addExact(
            Math.subtractExact(serializedKeyBytes, evicted.serializedKeyBytes),
            rowBytes)
          require(
            replacementBytes <= maxSerializedKeyBytes,
            s"MPP RANGE selected reservoir keys require $replacementBytes serialized bytes, " +
              s"exceeding the $maxSerializedKeyBytes-byte partition budget; increase " +
              s"$MaxSampleBytesKey"
          )
          retained.dequeue()
          retained.enqueue(PrioritizedRow(priority, row.copy(), rowBytes))
          serializedKeyBytes = replacementBytes
        }
    }

    val selected = retained.dequeueAll.reverseIterator.map {
      entry: PrioritizedRow => entry.row
    }.toArray
    PartitionSketch(count, selected, serializedKeyBytes)
  }

  private def partitionLimits(
      partitionId: Int,
      inputPartitions: Int,
      requestedPartitions: Int,
      samplePointsPerPartitionHint: Int,
      maxSampleRows: Int,
      maxSampleBytes: Long): (Int, Long) = {
    validateGlobalLimits(
      inputPartitions,
      requestedPartitions,
      samplePointsPerPartitionHint,
      maxSampleRows,
      maxSampleBytes)
    val totalRows = globalSampleRowLimit(
      inputPartitions,
      requestedPartitions,
      samplePointsPerPartitionHint,
      maxSampleRows)
    (
      fairShare(totalRows.toLong, partitionId, inputPartitions).toInt,
      fairShare(maxSampleBytes, partitionId, inputPartitions))
  }

  private def validateGlobalLimits(
      inputPartitions: Int,
      requestedPartitions: Int,
      samplePointsPerPartitionHint: Int,
      maxSampleRows: Int,
      maxSampleBytes: Long): Unit = {
    require(
      inputPartitions > 0,
      s"MPP RANGE input partition count must be positive: $inputPartitions")
    require(
      requestedPartitions > 0,
      s"MPP RANGE partition count must be positive: $requestedPartitions")
    require(
      samplePointsPerPartitionHint > 0,
      s"MPP RANGE sample-points hint must be positive: $samplePointsPerPartitionHint")
    require(maxSampleRows > 0, s"$MaxSampleRowsKey must be positive: $maxSampleRows")
    require(maxSampleBytes > 0L, s"$MaxSampleBytesKey must be positive: $maxSampleBytes")
    require(
      inputPartitions <= maxSampleRows,
      s"MPP RANGE has $inputPartitions input partitions but $MaxSampleRowsKey=$maxSampleRows; " +
        "at least one bounded sample slot per input partition is required"
    )
    require(
      maxSampleBytes / inputPartitions > 0L,
      s"MPP RANGE has $inputPartitions input partitions but $MaxSampleBytesKey=$maxSampleBytes; " +
        "the bounded byte budget cannot represent one key per input partition"
    )
  }

  private def globalSampleRowLimit(
      inputPartitions: Int,
      requestedPartitions: Int,
      samplePointsPerPartitionHint: Int,
      maxSampleRows: Int): Int = {
    val sparkTarget = math.min(
      Math.multiplyExact(samplePointsPerPartitionHint.toLong, requestedPartitions.toLong),
      1000000L)
    math
      .min(
        maxSampleRows.toLong,
        math.max(inputPartitions.toLong, Math.multiplyExact(3L, sparkTarget)))
      .toInt
  }

  private def fairShare(total: Long, partitionId: Int, partitions: Int): Long = {
    require(partitionId >= 0 && partitionId < partitions)
    total / partitions + (if (partitionId < total % partitions) 1L else 0L)
  }

  private def sampleSeed(rddId: Int, partitionId: Int): Long = {
    var value = (rddId.toLong << 32) ^ partitionId.toLong ^ 0x9e3779b97f4a7c15L
    value ^= value >>> 30
    value *= 0xbf58476d1ce4e5b9L
    value ^= value >>> 27
    value *= 0x94d049bb133111ebL
    value ^ (value >>> 31)
  }

  private def cacheOrderingKey(
      outputAttributes: Seq[Attribute],
      ordering: Seq[SortOrder]): Seq[OrderingKey] = {
    require(ordering.nonEmpty, "MPP RANGE cache requires at least one SortOrder")
    ordering.map {
      order =>
        val attribute = order.child match {
          case attr: Attribute => attr
          case other =>
            throw new IllegalArgumentException(
              s"MPP RANGE cache only supports attribute sort keys, found ${other.sql}")
        }
        val outputOrdinal = outputAttributes.indexWhere(_.exprId == attribute.exprId)
        require(
          outputOrdinal >= 0,
          s"MPP RANGE sort key ${attribute.sql} is absent from producer output")
        OrderingKey(
          outputOrdinal,
          order.dataType.json,
          order.nullable,
          order.direction match {
            case Ascending => true
            case Descending => false
          },
          order.nullOrdering match {
            case NullsFirst => true
            case NullsLast => false
          }
        )
    }
  }

  private def sameSamplePlan(left: SparkPlan, right: SparkPlan): Boolean = {
    (left eq right) || {
      try {
        left.sameResult(right)
      } catch {
        // Canonicalization is an optimization-only cache comparison. An unsupported or malformed
        // canonical form must cause an independent bounds generation, not fail the query.
        case NonFatal(_) => false
      }
    }
  }

  private def positiveIntConf(key: String, defaultValue: Int): Int = {
    val raw = SQLConf.get.getConfString(key, defaultValue.toString)
    val value = raw.toInt
    require(value > 0, s"$key must be positive: $raw")
    value
  }

  private def positiveBytesConf(key: String, defaultValue: Long): Long = {
    val raw = SQLConf.get.getConfString(key, defaultValue.toString)
    val value = JavaUtils.byteStringAsBytes(raw)
    require(value > 0L, s"$key must be positive: $raw")
    value
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
