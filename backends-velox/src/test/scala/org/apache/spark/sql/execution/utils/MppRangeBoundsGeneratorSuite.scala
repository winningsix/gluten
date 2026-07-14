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

import org.apache.spark.SparkFunSuite
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Ascending, AttributeReference, Descending, GenericInternalRow, NullsFirst, NullsLast, SortOrder, UnsafeProjection}
import org.apache.spark.sql.execution.LocalTableScanExec
import org.apache.spark.sql.types.{DecimalType, DoubleType, FloatType, IntegerType, StringType}
import org.apache.spark.unsafe.types.UTF8String

import com.fasterxml.jackson.databind.ObjectMapper

import java.util.concurrent.{Callable, CountDownLatch, Executors, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger

class MppRangeBoundsGeneratorSuite extends SparkFunSuite {

  test("range descriptor preserves mixed directions, null ordering, and boundary values") {
    val intKey = AttributeReference("i", IntegerType, nullable = true)()
    val stringKey = AttributeReference("s", StringType, nullable = true)()
    val projection = UnsafeProjection.create(Seq(intKey, stringKey), Seq(intKey, stringKey))
    val rows: Array[InternalRow] = Array(
      projection(new GenericInternalRow(Array[Any](1, null))).copy(),
      projection(new GenericInternalRow(Array[Any](7, UTF8String.fromString("seven")))).copy())
    val ordering = Seq(
      SortOrder(intKey, Ascending, NullsFirst, Seq.empty),
      SortOrder(stringKey, Descending, NullsLast, Seq.empty))

    val descriptor = new ObjectMapper().readTree(MppRangeBoundsGenerator.encode(ordering, rows))

    assert(descriptor.get("version").asInt() == 1)
    assert(descriptor.get("keys").get(0).get("ascending").asBoolean())
    assert(descriptor.get("keys").get(0).get("nullsFirst").asBoolean())
    assert(!descriptor.get("keys").get(1).get("ascending").asBoolean())
    assert(!descriptor.get("keys").get(1).get("nullsFirst").asBoolean())
    assert(descriptor.get("bounds").size() == 2)
    assert(descriptor.get("bounds").get(0).get(0).get("value").asInt() == 1)
    assert(descriptor.get("bounds").get(0).get(1).get("isNull").asBoolean())
    assert(descriptor.get("bounds").get(1).get(1).get("value").asText() == "seven")
  }

  test("D1 type guard accepts verified keys and rejects unsupported ordering domains") {
    assert(MppRangeBoundsGenerator.supports(IntegerType))
    assert(MppRangeBoundsGenerator.supports(StringType))
    assert(!MppRangeBoundsGenerator.supports(DecimalType(18, 2)))
    assert(!MppRangeBoundsGenerator.supports(FloatType))
    assert(!MppRangeBoundsGenerator.supports(DoubleType))
  }

  test("one-pass priority reservoir enforces serialized row and byte limits deterministically") {
    val key = AttributeReference("key", IntegerType, nullable = false)()
    val projection = UnsafeProjection.create(Seq(key), Seq(key))
    val rows = (0 until 100).map {
      value => projection(new GenericInternalRow(Array[Any](value))).copy()
    }
    val rowBytes = rows.head.getSizeInBytes.toLong

    val byRows = MppRangeBoundsGenerator.sketchPartition(
      rows.iterator,
      maxRows = 7,
      maxSerializedKeyBytes = Long.MaxValue,
      seed = 17L)
    val repeated = MppRangeBoundsGenerator.sketchPartition(
      rows.iterator,
      maxRows = 7,
      maxSerializedKeyBytes = Long.MaxValue,
      seed = 17L)
    assert(byRows.count == 100L)
    assert(byRows.samples.length == 7)
    assert(byRows.serializedKeyBytes == byRows.samples.map(_.getSizeInBytes.toLong).sum)
    assert(byRows.samples.map(_.getInt(0)).sameElements(repeated.samples.map(_.getInt(0))))

    val byBytes = MppRangeBoundsGenerator.sketchPartition(
      rows.iterator,
      maxRows = 100,
      maxSerializedKeyBytes = rowBytes * 3L,
      seed = 19L)
    assert(byBytes.samples.length == 3)
    assert(byBytes.serializedKeyBytes == rowBytes * 3L)

    val oversized = MppRangeBoundsGenerator.sketchPartition(
      rows.iterator,
      maxRows = 100,
      maxSerializedKeyBytes = rowBytes - 1L,
      seed = 23L)
    assert(oversized.samples.isEmpty)
    assert(oversized.oversizedRows == 100L)
  }

  test("weighted skew samples produce monotonic Spark-compatible bounds") {
    val key = AttributeReference("key", IntegerType, nullable = false)()
    val projection = UnsafeProjection.create(Seq(key), Seq(key))
    def row(value: Int) = projection(new GenericInternalRow(Array[Any](value))).copy()

    val heavy = Array(row(0), row(1))
    val light = Array(row(10), row(20))
    val sketches = Array(
      MppRangeBoundsGenerator.PartitionSketch(
        count = 800L,
        samples = heavy,
        serializedKeyBytes = heavy.map(_.getSizeInBytes.toLong).sum,
        oversizedRows = 0L),
      MppRangeBoundsGenerator.PartitionSketch(
        count = 200L,
        samples = light,
        serializedKeyBytes = light.map(_.getSizeInBytes.toLong).sum,
        oversizedRows = 0L))
    implicit val ordering: Ordering[InternalRow] = Ordering.by(_.getInt(0))

    val bounds = MppRangeBoundsGenerator.determineBounds(
      partitions = 4,
      inputPartitions = 2,
      samplePointsPerPartitionHint = 100,
      maxSampleRows = 100,
      maxSampleBytes = 1L << 20,
      sketches = sketches)

    assert(bounds.nonEmpty)
    // The first heavy sample represents 400 rows and crosses the first 250-row target. This
    // specifically verifies that count/sample weighting, rather than unweighted keys, is used.
    assert(bounds.head.getInt(0) == 0)
    assert(bounds.sliding(2).forall(pair => ordering.lt(pair(0), pair(1))))
    assert(bounds.length + 1 <= 4)
  }

  test("driver validation fails closed on oversized keys and collected budget overflow") {
    val key = AttributeReference("key", IntegerType, nullable = false)()
    val projection = UnsafeProjection.create(Seq(key), Seq(key))
    val row = projection(new GenericInternalRow(Array[Any](1))).copy()
    val rowBytes = row.getSizeInBytes.toLong
    implicit val ordering: Ordering[InternalRow] = Ordering.by(_.getInt(0))

    val oversized = MppRangeBoundsGenerator.PartitionSketch(
      count = 1L,
      samples = Array.empty,
      serializedKeyBytes = 0L,
      oversizedRows = 1L)
    val oversizedFailure = intercept[IllegalArgumentException] {
      MppRangeBoundsGenerator.determineBounds(
        partitions = 2,
        inputPartitions = 1,
        samplePointsPerPartitionHint = 1,
        maxSampleRows = 10,
        maxSampleBytes = rowBytes - 1L,
        sketches = Array(oversized))
    }
    assert(oversizedFailure.getMessage.contains("serialized keys"))

    val overflow = MppRangeBoundsGenerator.PartitionSketch(
      count = 2L,
      samples = Array(row, row.copy()),
      serializedKeyBytes = rowBytes * 2L,
      oversizedRows = 0L)
    val byteFailure = intercept[IllegalArgumentException] {
      MppRangeBoundsGenerator.determineBounds(
        partitions = 2,
        inputPartitions = 1,
        samplePointsPerPartitionHint = 1,
        maxSampleRows = 10,
        maxSampleBytes = rowBytes,
        sketches = Array(overflow))
    }
    assert(byteFailure.getMessage.contains("serialized key bytes"))
  }

  test("query cache computes equivalent RANGE bounds once under concurrent access") {
    val key = AttributeReference("key", IntegerType, nullable = false)()
    val plan = LocalTableScanExec(Seq(key), Seq.empty[InternalRow], None)
    val ordering = Seq(SortOrder(key, Ascending, NullsFirst, Seq.empty))
    val cache = new MppRangeBoundsGenerator.QueryCache("execution-1")
    val calls = new AtomicInteger(0)
    val ownerStarted = new CountDownLatch(1)
    val releaseOwner = new CountDownLatch(1)
    val pool = Executors.newFixedThreadPool(2)

    try {
      val first = pool.submit(new Callable[(MppRangeBoundsGenerator.Result, Boolean)] {
        override def call(): (MppRangeBoundsGenerator.Result, Boolean) = {
          cache.getOrCompute(plan, plan.output, ordering, requestedPartitions = 8) {
            calls.incrementAndGet()
            ownerStarted.countDown()
            assert(releaseOwner.await(10, TimeUnit.SECONDS))
            MppRangeBoundsGenerator.Result("first", boundaryCount = 7)
          }
        }
      })
      assert(ownerStarted.await(10, TimeUnit.SECONDS))
      val second = pool.submit(new Callable[(MppRangeBoundsGenerator.Result, Boolean)] {
        override def call(): (MppRangeBoundsGenerator.Result, Boolean) = {
          cache.getOrCompute(plan, plan.output, ordering, requestedPartitions = 8) {
            calls.incrementAndGet()
            MppRangeBoundsGenerator.Result("unexpected", boundaryCount = 0)
          }
        }
      })
      releaseOwner.countDown()

      val results = Seq(first.get(10, TimeUnit.SECONDS), second.get(10, TimeUnit.SECONDS))
      assert(calls.get() == 1)
      assert(results.map(_._1.json).toSet == Set("first"))
      assert(results.map(_._2).toSet == Set(false, true))
    } finally {
      releaseOwner.countDown()
      pool.shutdownNow()
    }
  }

  test("query cache separates ordering and partition count and evicts failures") {
    val key = AttributeReference("key", IntegerType, nullable = false)()
    val plan = LocalTableScanExec(Seq(key), Seq.empty[InternalRow], None)
    val ascending = Seq(SortOrder(key, Ascending, NullsFirst, Seq.empty))
    val descending = Seq(SortOrder(key, Descending, NullsLast, Seq.empty))
    val cache = new MppRangeBoundsGenerator.QueryCache("execution-2")
    val calls = new AtomicInteger(0)

    def result(label: String): MppRangeBoundsGenerator.Result = {
      calls.incrementAndGet()
      MppRangeBoundsGenerator.Result(label, boundaryCount = 1)
    }

    assert(!cache.getOrCompute(plan, plan.output, ascending, 8)(result("a8"))._2)
    assert(cache.getOrCompute(plan, plan.output, ascending, 8)(result("cached"))._2)
    assert(!cache.getOrCompute(plan, plan.output, ascending, 9)(result("a9"))._2)
    assert(!cache.getOrCompute(plan, plan.output, descending, 8)(result("d8"))._2)
    assert(calls.get() == 3)

    val failed = new MppRangeBoundsGenerator.QueryCache("execution-3")
    val failure = intercept[IllegalStateException] {
      failed.getOrCompute(plan, plan.output, ascending, 8) {
        throw new IllegalStateException("sample failed")
      }
    }
    assert(failure.getMessage == "sample failed")
    val retried = failed.getOrCompute(plan, plan.output, ascending, 8)(result("retry"))
    assert(retried._1.json == "retry")
    assert(!retried._2)
  }
}
