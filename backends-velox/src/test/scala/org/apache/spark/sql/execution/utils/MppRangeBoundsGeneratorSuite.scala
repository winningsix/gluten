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

import org.apache.gluten.utils.LocalTableScanExecCompat

import org.apache.spark.{HashPartitioner, SparkConf, SparkContext, SparkFunSuite}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Ascending, AttributeReference, Descending, GenericInternalRow, NullsFirst, NullsLast, SortOrder, UnsafeProjection, UnsafeRow}
import org.apache.spark.sql.types.{DecimalType, DoubleType, FloatType, IntegerType, StringType}
import org.apache.spark.unsafe.types.UTF8String

import com.fasterxml.jackson.databind.ObjectMapper

import java.util.concurrent.{Callable, CountDownLatch, ExecutionException, Executors, TimeUnit}
import java.util.concurrent.atomic.{AtomicInteger, AtomicReference}

class MppRangeBoundsGeneratorSuite extends SparkFunSuite {

  test("RANGE preparation materializes shuffle dependencies in topological order") {
    val conf = new SparkConf(false)
      .setMaster("local[2]")
      .setAppName(getClass.getSimpleName)
      .set("spark.ui.enabled", "false")
    val sc = new SparkContext(conf)
    try {
      val sourceRows = 0 until 12
      val sourceCalls = sc.longAccumulator("range-source-calls")
      val intermediateCalls = sc.longAccumulator("range-intermediate-calls")
      val source = sc.parallelize(sourceRows, 2).map {
        value =>
          sourceCalls.add(1L)
          value -> value
      }
      val firstShuffle = source.partitionBy(new HashPartitioner(2)).map {
        pair =>
          intermediateCalls.add(1L)
          pair
      }
      val secondShuffle = firstShuffle.partitionBy(new HashPartitioner(3))

      val dependencies =
        MppRangeBoundsGenerator.shuffleDependenciesInTopologicalOrder(secondShuffle)
      assert(dependencies.size == 2)
      assert(dependencies.map(_.shuffleId).distinct.size == 2)

      MppRangeBoundsGenerator.materializeShuffleDependencies(secondShuffle)
      assert(sourceCalls.value == sourceRows.size)
      assert(intermediateCalls.value == sourceRows.size)

      assert(secondShuffle.collect().map(_._1).sorted.toSeq == sourceRows)
      assert(sourceCalls.value == sourceRows.size)
      assert(intermediateCalls.value == sourceRows.size)
    } finally {
      sc.stop()
    }
  }

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

  test("D1 type guard rejects Spark 4 collated strings when that API is available") {
    StringType.getClass.getMethods
      .find(
        method =>
          method.getName == "apply" &&
            method.getParameterTypes.sameElements(Array[Class[_]](classOf[String])))
      .foreach {
        applyByCollationName =>
          val collated = applyByCollationName
            .invoke(StringType, "UTF8_LCASE")
            .asInstanceOf[org.apache.spark.sql.types.DataType]
          assert(!MppRangeBoundsGenerator.supports(collated))
      }
  }

  test("one-pass priority reservoir enforces row limits deterministically") {
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

    assert(rowBytes > 0L)
  }

  test("variable-width reservoir matches pure row-priority selection") {
    val key = AttributeReference("key", StringType, nullable = false)()
    val projection = UnsafeProjection.create(Seq(key), Seq(key))
    val values = Seq("a", "medium-key", "x" * 200, "bb", "y" * 80, "tail")
    val rows = values.map {
      value => projection(new GenericInternalRow(Array[Any](UTF8String.fromString(value)))).copy()
    }
    val seed = 137L
    val referenceRandom = new java.util.Random(seed)
    val expected = rows
      .map(row => (referenceRandom.nextLong() & Long.MaxValue, row.getUTF8String(0).toString))
      .sortBy(_._1)
      .take(3)
      .map(_._2)

    val sketch = MppRangeBoundsGenerator.sketchPartition(
      rows.iterator,
      maxRows = 3,
      maxSerializedKeyBytes = rows.map(_.getSizeInBytes.toLong).sum,
      seed = seed)

    assert(sketch.samples.map(_.getUTF8String(0).toString).toSeq == expected)
    assert(sketch.serializedKeyBytes == sketch.samples.map(_.getSizeInBytes.toLong).sum)
  }

  test("byte admission fails only when a selected priority key cannot fit") {
    val key = AttributeReference("key", StringType, nullable = false)()
    val projection = UnsafeProjection.create(Seq(key), Seq(key))
    def row(value: String): UnsafeRow = {
      projection(new GenericInternalRow(Array[Any](UTF8String.fromString(value)))).copy()
    }
    val small = row("s")
    val large = row("z" * 4096)

    val selectedFailure = intercept[IllegalArgumentException] {
      MppRangeBoundsGenerator.sketchPartition(
        Iterator(large),
        maxRows = 1,
        maxSerializedKeyBytes = small.getSizeInBytes.toLong,
        seed = 1L)
    }
    assert(selectedFailure.getMessage.contains("selected reservoir keys"))

    val seedWithUnselectedSecond = (0L until 10000L).find {
      seed =>
        val random = new java.util.Random(seed)
        val first = random.nextLong() & Long.MaxValue
        val second = random.nextLong() & Long.MaxValue
        second > first
    }.get
    val unaffected = MppRangeBoundsGenerator.sketchPartition(
      Iterator(small, large),
      maxRows = 1,
      maxSerializedKeyBytes = small.getSizeInBytes.toLong,
      seed = seedWithUnselectedSecond)
    assert(unaffected.samples.length == 1)
    assert(unaffected.samples.head.getUTF8String(0).toString == "s")
  }

  test("weighted skew samples produce monotonic Spark-compatible bounds") {
    val key = AttributeReference("key", IntegerType, nullable = false)()
    val projection = UnsafeProjection.create(Seq(key), Seq(key))
    def row(value: Int): UnsafeRow =
      projection(new GenericInternalRow(Array[Any](value))).copy()

    val heavy = Array(row(0), row(1))
    val light = Array(row(10), row(20))
    val sketches = Array(
      MppRangeBoundsGenerator.PartitionSketch(
        count = 800L,
        samples = heavy,
        serializedKeyBytes = heavy.map(_.getSizeInBytes.toLong).sum),
      MppRangeBoundsGenerator.PartitionSketch(
        count = 200L,
        samples = light,
        serializedKeyBytes = light.map(_.getSizeInBytes.toLong).sum)
    )
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

  test("driver validation fails closed on collected row and byte budget overflow") {
    val key = AttributeReference("key", IntegerType, nullable = false)()
    val projection = UnsafeProjection.create(Seq(key), Seq(key))
    val row = projection(new GenericInternalRow(Array[Any](1))).copy()
    val rowBytes = row.getSizeInBytes.toLong
    implicit val ordering: Ordering[InternalRow] = Ordering.by(_.getInt(0))

    val rowOverflow = MppRangeBoundsGenerator.PartitionSketch(
      count = 2L,
      samples = Array(row, row.copy()),
      serializedKeyBytes = rowBytes * 2L)
    val rowFailure = intercept[IllegalArgumentException] {
      MppRangeBoundsGenerator.determineBounds(
        partitions = 2,
        inputPartitions = 1,
        samplePointsPerPartitionHint = 1,
        maxSampleRows = 1,
        maxSampleBytes = rowBytes * 2L,
        sketches = Array(rowOverflow))
    }
    assert(rowFailure.getMessage.contains("sample rows"))

    val overflow = MppRangeBoundsGenerator.PartitionSketch(
      count = 2L,
      samples = Array(row, row.copy()),
      serializedKeyBytes = rowBytes * 2L)
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

  test("sparse partition samples fail closed unless the full input is smaller than RANGE width") {
    val key = AttributeReference("key", IntegerType, nullable = false)()
    val projection = UnsafeProjection.create(Seq(key), Seq(key))
    def row(value: Int): UnsafeRow =
      projection(new GenericInternalRow(Array[Any](value))).copy()
    val one = row(1)
    val two = row(2)
    val empty = MppRangeBoundsGenerator.PartitionSketch(
      count = 0L,
      samples = Array.empty,
      serializedKeyBytes = 0L)
    implicit val ordering: Ordering[InternalRow] = Ordering.by(_.getInt(0))

    val sparseLarge = Array(
      MppRangeBoundsGenerator.PartitionSketch(
        count = 1000L,
        samples = Array(one),
        serializedKeyBytes = one.getSizeInBytes.toLong)) ++ Array.fill(15)(empty)
    val sparseFailure = intercept[IllegalArgumentException] {
      MppRangeBoundsGenerator.determineBounds(
        partitions = 8,
        inputPartitions = sparseLarge.length,
        samplePointsPerPartitionHint = 100,
        maxSampleRows = 200,
        maxSampleBytes = 1L << 20,
        sketches = sparseLarge)
    }
    assert(sparseFailure.getMessage.contains("only 1 bounded samples"))
    assert(sparseFailure.getMessage.contains("mergeable global reservoir"))

    val sparseSmall = Array(
      MppRangeBoundsGenerator.PartitionSketch(
        count = 2L,
        samples = Array(one, two),
        serializedKeyBytes = one.getSizeInBytes.toLong + two.getSizeInBytes.toLong)) ++
      Array.fill(15)(empty)
    val smallBounds = MppRangeBoundsGenerator.determineBounds(
      partitions = 8,
      inputPartitions = sparseSmall.length,
      samplePointsPerPartitionHint = 100,
      maxSampleRows = 200,
      maxSampleBytes = 1L << 20,
      sketches = sparseSmall)
    assert(smallBounds.length + 1 <= 2)
    assert(smallBounds.length + 1 <= 8)
  }

  test("query cache computes equivalent RANGE bounds once under concurrent access") {
    val key = AttributeReference("key", IntegerType, nullable = false)()
    val plan = LocalTableScanExecCompat(Seq(key), Seq.empty[InternalRow])
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
    val plan = LocalTableScanExecCompat(Seq(key), Seq.empty[InternalRow])
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

  test("concurrent cache failure reaches all waiters before a clean retry generation") {
    val key = AttributeReference("key", IntegerType, nullable = false)()
    val plan = LocalTableScanExecCompat(Seq(key), Seq.empty[InternalRow])
    val ordering = Seq(SortOrder(key, Ascending, NullsFirst, Seq.empty))
    val cache = new MppRangeBoundsGenerator.QueryCache("execution-failure")
    val failure = new IllegalStateException("concurrent sample failed")
    val calls = new AtomicInteger(0)
    val ownerStarted = new CountDownLatch(1)
    val releaseOwner = new CountDownLatch(1)
    val waiterThread = new AtomicReference[Thread]()
    val pool = Executors.newFixedThreadPool(2)

    try {
      val owner = pool.submit(new Callable[Unit] {
        override def call(): Unit = {
          cache.getOrCompute(plan, plan.output, ordering, requestedPartitions = 8) {
            calls.incrementAndGet()
            ownerStarted.countDown()
            assert(releaseOwner.await(10, TimeUnit.SECONDS))
            throw failure
          }
          ()
        }
      })
      assert(ownerStarted.await(10, TimeUnit.SECONDS))
      val waiter = pool.submit(new Callable[Unit] {
        override def call(): Unit = {
          waiterThread.set(Thread.currentThread())
          cache.getOrCompute(plan, plan.output, ordering, requestedPartitions = 8) {
            calls.incrementAndGet()
            MppRangeBoundsGenerator.Result("unexpected", boundaryCount = 0)
          }
          ()
        }
      })

      val waitDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10L)
      while (
        Option(waiterThread.get()).forall(_.getState != Thread.State.WAITING) &&
        System.nanoTime() < waitDeadline
      ) {
        Thread.`yield`()
      }
      assert(waiterThread.get() != null)
      assert(waiterThread.get().getState == Thread.State.WAITING)
      releaseOwner.countDown()

      val ownerFailure = intercept[ExecutionException](owner.get(10, TimeUnit.SECONDS))
      val waiterFailure = intercept[ExecutionException](waiter.get(10, TimeUnit.SECONDS))
      assert(ownerFailure.getCause eq failure)
      assert(waiterFailure.getCause eq failure)
      assert(calls.get() == 1)

      val retry = cache.getOrCompute(plan, plan.output, ordering, requestedPartitions = 8) {
        calls.incrementAndGet()
        MppRangeBoundsGenerator.Result("retry", boundaryCount = 7)
      }
      assert(retry._1.json == "retry")
      assert(!retry._2)
      assert(calls.get() == 2)
    } finally {
      releaseOwner.countDown()
      pool.shutdownNow()
    }
  }
}
