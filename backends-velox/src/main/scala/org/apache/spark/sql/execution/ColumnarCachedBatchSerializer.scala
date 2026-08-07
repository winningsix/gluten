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
package org.apache.spark.sql.execution

import org.apache.gluten.backendsapi.BackendsApiManager
import org.apache.gluten.columnarbatch.{ColumnarBatches, ColumnarBatchJniWrapper, VeloxColumnarBatches}
import org.apache.gluten.config.{GlutenConfig, VeloxConfig}
import org.apache.gluten.execution.{MppNativeQueryRDD, RowToVeloxColumnarExec, VeloxColumnarToRowExec}
import org.apache.gluten.iterator.Iterators
import org.apache.gluten.memory.arrow.alloc.ArrowBufferAllocators
import org.apache.gluten.runtime.Runtimes
import org.apache.gluten.utils.ArrowAbiUtil
import org.apache.gluten.vectorized.ColumnarBatchSerializerJniWrapper

import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression}
import org.apache.spark.sql.columnar.{CachedBatch, CachedBatchSerializer}
import org.apache.spark.sql.execution.columnar.DefaultCachedBatchSerializer
import org.apache.spark.sql.execution.unsafe.UnsafeByteArray
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.utils.SparkSchemaUtil
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.StorageLevel

import com.esotericsoftware.kryo.{Kryo, Serializer => KryoSerializer}
import com.esotericsoftware.kryo.DefaultSerializer
import com.esotericsoftware.kryo.io.{Input, Output}
import org.apache.arrow.c.ArrowSchema

import java.nio.file.{Files, Path, Paths}
import java.util.UUID
import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable.ArrayBuffer

/**
 * TODO: fix on Spark-4.1 - Documentation
 *
 * If you encounter serialization issues, manually register this class:
 * {{{
 *   spark.kryo.classesToRegister=org.apache.spark.sql.execution.CachedColumnarBatch
 * }}}
 */
@DefaultSerializer(classOf[CachedColumnarBatchKryoSerializer])
case class CachedColumnarBatch(
    override val numRows: Int,
    override val sizeInBytes: Long,
    bytes: Array[Byte],
    unsafeBytes: UnsafeByteArray = null,
    cacheFilePath: String = null)
  extends CachedBatch {}

class CachedColumnarBatchKryoSerializer extends KryoSerializer[CachedColumnarBatch] {
  override def write(kryo: Kryo, output: Output, batch: CachedColumnarBatch): Unit = {
    output.writeInt(batch.numRows)
    output.writeLong(batch.sizeInBytes)
    val isFileBacked = batch.cacheFilePath != null
    output.writeBoolean(isFileBacked)
    if (isFileBacked) {
      output.writeString(batch.cacheFilePath)
      return
    }
    val isOffHeap = batch.unsafeBytes != null
    output.writeBoolean(isOffHeap)
    if (isOffHeap) {
      // UnsafeByteArray is used only for DISK_ONLY pages. Once Kryo has
      // synchronously copied the page into DiskStore's output stream, the
      // native staging allocation has no remaining owner or consumer. Do not
      // wait for finalization: a wide materialization can otherwise retain
      // tens of GiB of already-written pages and exhaust Spark off-heap.
      try {
        batch.unsafeBytes.write(kryo, output)
      } finally {
        batch.unsafeBytes.release()
      }
    } else {
      require(
        batch.bytes != null,
        "The object 'CachedColumnarBatch.bytes' is invalid or malformed to " +
          s"serialize using ${this.getClass.getName}")
      output.writeInt(batch.bytes.length + 1) // +1 to distinguish Kryo.NULL
      output.writeBytes(batch.bytes)
    }
  }

  override def read(
      kryo: Kryo,
      input: Input,
      cls: Class[CachedColumnarBatch]): CachedColumnarBatch = {
    val numRows = input.readInt()
    val sizeInBytes = input.readLong()
    if (input.readBoolean()) {
      return CachedColumnarBatch(numRows, sizeInBytes, null, null, input.readString())
    }
    if (input.readBoolean()) {
      val unsafeBytes = new UnsafeByteArray()
      unsafeBytes.read(kryo, input)
      CachedColumnarBatch(numRows, sizeInBytes, null, unsafeBytes)
    } else {
      val length = input.readInt()
      require(
        length != Kryo.NULL,
        "The object 'CachedColumnarBatch.bytes' is invalid or malformed to " +
          s"deserialize using ${this.getClass.getName}")
      val bytes = new Array[Byte](length - 1) // -1 to restore
      input.readBytes(bytes)
      CachedColumnarBatch(numRows, sizeInBytes, bytes)
    }
  }
}

private object ParquetCacheFilePlacement {
  // One counter per executor JVM is sufficient: page names are UUID based,
  // while this counter only spreads page traffic across configured devices.
  // Starting from an executor-specific offset avoids all executors choosing
  // the same device for their first page after a stage starts.
  private val nextPage = new AtomicLong(0L)

  def nextRootIndex(executorId: String, rootCount: Int): Int = {
    require(rootCount > 0, "Parquet cache requires at least one file root")
    Math.floorMod(executorId.hashCode.toLong + nextPage.getAndIncrement(), rootCount)
  }
}

// format: off
/**
 * Feature:
 * 1. This serializer supports column pruning
 * 2. TODO: support push down filter
 * 3. Super TODO: support store offheap object directly
 *
 * The data transformation pipeline:
 *
 *   - Serializer ColumnarBatch -> CachedColumnarBatch
 *     -> serialize to byte[]
 *
 *   - Deserializer CachedColumnarBatch -> ColumnarBatch
 *     -> deserialize to byte[] to create Velox ColumnarBatch
 *
 *   - Serializer InternalRow -> CachedColumnarBatch (support RowToColumnar)
 *     -> Convert InternalRow to ColumnarBatch
 *     -> Serializer ColumnarBatch -> CachedColumnarBatch
 *
 *   - Serializer InternalRow -> DefaultCachedBatch (unsupport RowToColumnar)
 *     -> Convert InternalRow to DefaultCachedBatch using vanilla Spark serializer
 *
 *   - Deserializer CachedColumnarBatch -> InternalRow (support ColumnarToRow)
 *     -> Deserializer CachedColumnarBatch -> ColumnarBatch
 *     -> Convert ColumnarBatch to InternalRow
 *
 *   - Deserializer DefaultCachedBatch -> InternalRow (unsupport ColumnarToRow)
 *     -> Convert DefaultCachedBatch to InternalRow using vanilla Spark serializer
 */
// format: on
class ColumnarCachedBatchSerializer extends CachedBatchSerializer with Logging {
  private lazy val rowBasedCachedBatchSerializer = new DefaultCachedBatchSerializer
  private def newParquetCacheFile(): Path = {
    val configuredRoots = sys.env
      .get("GLUTEN_CACHE_FILE_DIRECTORIES")
      .toSeq
      .flatMap(_.split(","))
      .map(_.trim)
      .filter(_.nonEmpty)
    val sparkRoots = sys.env
      .get("SPARK_LOCAL_DIRS")
      .toSeq
      .flatMap(_.split(","))
      .map(_.trim)
      .filter(_.nonEmpty)
    val roots = if (configuredRoots.nonEmpty) configuredRoots else sparkRoots
    require(
      roots.nonEmpty,
      "GLUTEN_CACHE_FILE_DIRECTORIES or SPARK_LOCAL_DIRS must name a cache directory")

    val sparkEnv = SparkEnv.get
    val executorId = Option(sparkEnv).map(_.executorId).getOrElse("driver")
    val appId = Option(sparkEnv)
      .map(_.conf.get("spark.app.id", "unknown-app"))
      .getOrElse("unknown-app")
    // Rotate individual standalone Parquet pages instead of pinning an entire
    // executor to one root. This allows scan/decode on one device to overlap
    // cache writes across the remaining NVMe devices and avoids one slow or
    // busy root dominating a materialize task.
    val root = roots(ParquetCacheFilePlacement.nextRootIndex(executorId, roots.size))
    val directory = Paths.get(root, "gluten-parquet-cache", appId, executorId)
    Files.createDirectories(directory)
    directory.resolve(s"page-${UUID.randomUUID()}.parquet")
  }

  private def glutenConf: GlutenConfig = GlutenConfig.get

  private def toStructType(schema: Seq[Attribute]): StructType = {
    StructType(schema.map(a => StructField(a.name, a.dataType, a.nullable, a.metadata)))
  }

  private def validateSchema(schema: Seq[Attribute]): Boolean = {
    val dt = toStructType(schema)
    validateSchema(dt)
  }

  private def validateSchema(schema: StructType): Boolean = {
    val reason = BackendsApiManager.getValidatorApiInstance.doSchemaValidate(schema)
    if (reason.isDefined) {
      logInfo(s"Columnar cache does not support schema $schema, due to ${reason.get}")
      false
    } else {
      true
    }
  }

  override def supportsColumnarInput(schema: Seq[Attribute]): Boolean = {
    glutenConf.enableGluten && validateSchema(schema)
  }

  override def supportsColumnarOutput(schema: StructType): Boolean = {
    glutenConf.enableGluten && validateSchema(schema)
  }

  override def convertInternalRowToCachedBatch(
      input: RDD[InternalRow],
      schema: Seq[Attribute],
      storageLevel: StorageLevel,
      conf: SQLConf): RDD[CachedBatch] = {
    val localSchema = toStructType(schema)
    if (!validateSchema(localSchema)) {
      // we cannot use columnar cache here, as the `RowToColumnar` does not support this schema
      rowBasedCachedBatchSerializer.convertInternalRowToCachedBatch(
        input,
        schema,
        storageLevel,
        conf)
    } else {
      val numRows = conf.columnBatchSize
      val rddColumnarBatch = input.mapPartitions {
        it =>
          RowToVeloxColumnarExec.toColumnarBatchIterator(
            it,
            localSchema,
            numRows,
            VeloxConfig.get.veloxPreferredBatchBytes)
      }
      convertColumnarBatchToCachedBatch(rddColumnarBatch, schema, storageLevel, conf)
    }
  }

  override def convertCachedBatchToInternalRow(
      input: RDD[CachedBatch],
      cacheAttributes: Seq[Attribute],
      selectedAttributes: Seq[Attribute],
      conf: SQLConf): RDD[InternalRow] = {
    if (!validateSchema(cacheAttributes)) {
      // if we do not support this schema, that means we are using row-based serializer,
      // see `convertInternalRowToCachedBatch`, so fallback to vanilla Spark serializer
      rowBasedCachedBatchSerializer.convertCachedBatchToInternalRow(
        input,
        cacheAttributes,
        selectedAttributes,
        conf)
    } else {
      val rddColumnarBatch =
        convertCachedBatchToColumnarBatch(input, cacheAttributes, selectedAttributes, conf)
      rddColumnarBatch.mapPartitions(it => VeloxColumnarToRowExec.toRowIterator(it))
    }
  }

  override def convertColumnarBatchToCachedBatch(
      input: RDD[ColumnarBatch],
      schema: Seq[Attribute],
      storageLevel: StorageLevel,
      conf: SQLConf): RDD[CachedBatch] = {
    // Spark creates the producer RDD before invoking the cache serializer.  Mark this exact input
    // graph as a terminal GPU sink now, while still on the driver.  This lets MPP return a
    // CudfVector directly to the Parquet cache writer without globally changing unrelated actions
    // such as count/aggregate, which still require the normal Velox boundary.
    val markedMppProducers = MppNativeQueryRDD.enableDeviceOutputForTerminalGpuSink(input)
    if (markedMppProducers > 0) {
      logInfo(
        s"Columnar cache fill enabled direct device output for $markedMppProducers MPP producer(s)")
    }
    val parquetBatchesPerPage =
      if (
        storageLevel.useDisk && !storageLevel.useMemory &&
        sys.env.get("GLUTEN_CACHE_FORMAT").contains("parquet")
      ) {
        sys.env
          .get("GLUTEN_CACHE_PARQUET_BATCHES_PER_PAGE")
          .map(_.toInt)
          .filter(_ > 0)
          .getOrElse(8)
      } else {
        1
      }
    val standaloneParquetFiles =
      storageLevel.useDisk && !storageLevel.useMemory &&
        sys.env.get("GLUTEN_CACHE_FORMAT").contains("parquet") &&
        sys.env.get("GLUTEN_FILE_BACKED_TABLE_CACHE").exists(_.toBoolean)
    val standaloneParquetPageBytes = sys.env
      .get("GLUTEN_CACHE_PARQUET_PAGE_BYTES")
      .map(_.toLong)
      .filter(_ > 0)
      .getOrElse(200L << 20)
    val standaloneParquetAppendBytes = sys.env
      .get("GLUTEN_CACHE_PARQUET_APPEND_BYTES")
      .map(_.toLong)
      .filter(_ > 0)
      .getOrElse(200L << 20)
    val asyncStandaloneParquetWrites = standaloneParquetFiles && sys.env
      .get("GLUTEN_CACHE_ASYNC_FILE_WRITES")
      .exists(_.toBoolean)
    val asyncParquetAppendBytes = sys.env
      .get("GLUTEN_CACHE_ASYNC_APPEND_BYTES")
      .map(_.toLong)
      .filter(_ > 0)
      // Keep a single append bounded by the target standalone Parquet file
      // size.  The page-size check runs between appends, so making an append
      // much larger than the page target creates an equally oversized file.
      .getOrElse(200L << 20)
    // Scan/decode and Parquet encode have different optimal batch sizes. A
    // large scan chunk avoids repeated reader setup, but forwarding that
    // boundary directly to the writer can produce undersized or oversized
    // compression launches depending on file boundaries. Treat ENCODE_BYTES
    // as a maximum and adapt the actual writer admission from the compression
    // ratio observed by this cache partition.
    val parquetEncodeBytes = sys.env
      .get("GLUTEN_CACHE_PARQUET_ENCODE_BYTES")
      .map(_.toLong)
      .filter(_ > 0)
      .getOrElse(256L << 20)
    val boundedParquetEncodeBytes =
      Math.min(standaloneParquetAppendBytes, parquetEncodeBytes)
    val maximumParquetAppendBytes = if (asyncStandaloneParquetWrites) {
      Math.min(boundedParquetEncodeBytes, asyncParquetAppendBytes)
    } else {
      boundedParquetEncodeBytes
    }
    input.mapPartitions {
      it =>
        val veloxBatches = it.map {
          /* Native code needs a Velox offloaded batch, making sure to offload
             if heavy batch is encountered */
          batch => VeloxColumnarBatches.ensureVeloxBatch(batch)
        }
        new Iterator[CachedBatch] {
          private val wrapper = ColumnarBatchSerializerJniWrapper.create(
            Runtimes.contextInstance(
              BackendsApiManager.getBackendName,
              "ColumnarCachedBatchSerializer#serialize"))

          // Keep one batch of lookahead so an append byte target is a real
          // admission boundary. The old loop checked the target only before
          // pulling the next batch. A small tail followed by a large GPU scan
          // batch was therefore concatenated and submitted even when the pair
          // greatly exceeded the target. Besides increasing the live set, the
          // overshoot made writer row-group geometry depend on source-file
          // boundaries instead of the writer-local encode target. An
          // individually oversized batch is still admitted so the iterator
          // always makes progress.
          private var pendingParquetBatch: ColumnarBatch = null
          private var observedParquetInputBytes = 0L
          private var observedParquetOutputBytes = 0L
          private val cacheTimingStartNanos = System.nanoTime()
          private val cacheTaskPartition = Option(TaskContext.get())
            .map(_.partitionId())
            .getOrElse(-1)
          private var cacheTimingLogged = false
          private var upstreamExhausted = false
          private var upstreamHasNextNanos = 0L
          private var upstreamNextNanos = 0L
          private var batchSizeNanos = 0L
          private var writerInitNanos = 0L
          private var writerAppendNanos = 0L
          private var writerFinishNanos = 0L
          private var cachePages = 0L
          private var cacheAppends = 0L
          private var cacheInputBatches = 0L
          private var cacheInputRows = 0L
          private var cacheInputBytes = 0L
          private var cacheParquetBytes = 0L

          // A strict compressed-byte target frequently needs a second, small
          // append: the first append lands just below the page boundary and
          // the loop then concatenates and encodes another device table. That
          // is disproportionately expensive for very wide schemas. Admit a
          // modest amount of headroom so steady-state pages normally finish
          // in one append while remaining far below the 600+ MiB files seen
          // when a low-compression schema blindly admits 1 GiB of raw input.
          private val parquetPageAdmissionHeadroom = 1.15d

          // Start conservatively so a low-compression schema does not turn a
          // 200 MiB cache-page target into a 600+ MiB first file. Once one
          // append completes, use the measured compressed/raw ratio. The
          // estimate is cumulative to avoid oscillating on individual source
          // batches and is local to this Spark cache partition.
          private def parquetCompressionRatio: Double = {
            val compressionRatio =
              if (observedParquetInputBytes > 0L && observedParquetOutputBytes > 0L) {
                observedParquetOutputBytes.toDouble / observedParquetInputBytes.toDouble
              } else {
                0.5d
              }
            Math.max(0.01d, Math.min(4.0d, compressionRatio))
          }

          private def parquetAppendTarget(estimatedPageBytes: Long): Long = {
            val remainingOutputBytes =
              Math.max(1L, standaloneParquetPageBytes - estimatedPageBytes)
            val boundedRatio = parquetCompressionRatio
            val estimatedInputBytes =
              Math.ceil(remainingOutputBytes * parquetPageAdmissionHeadroom / boundedRatio).toLong
            Math.max(1L, Math.min(maximumParquetAppendBytes, estimatedInputBytes))
          }

          private def upstreamHasNext: Boolean = {
            if (upstreamExhausted) {
              false
            } else {
              val start = System.nanoTime()
              val result =
                try {
                  veloxBatches.hasNext
                } finally {
                  upstreamHasNextNanos += System.nanoTime() - start
                }
              if (!result) {
                upstreamExhausted = true
              }
              result
            }
          }

          private def hasParquetBatch: Boolean =
            pendingParquetBatch != null || upstreamHasNext

          private def takeParquetBatch(): ColumnarBatch = {
            if (pendingParquetBatch != null) {
              val batch = pendingParquetBatch
              pendingParquetBatch = null
              batch
            } else {
              val start = System.nanoTime()
              try {
                veloxBatches.next()
              } finally {
                upstreamNextNanos += System.nanoTime() - start
              }
            }
          }

          private def logCacheTiming(): Unit = {
            if (!cacheTimingLogged) {
              cacheTimingLogged = true
              val totalNanos = System.nanoTime() - cacheTimingStartNanos
              val measuredNanos = upstreamHasNextNanos + upstreamNextNanos + batchSizeNanos +
                writerInitNanos + writerAppendNanos + writerFinishNanos
              def millis(nanos: Long): Double = nanos.toDouble / 1000000.0d
              logWarning(
                s"[COLUMNAR_CACHE_TIMING] partition=$cacheTaskPartition " +
                  f"totalMs=${millis(totalNanos)}%.3f " +
                  f"upstreamHasNextMs=${millis(upstreamHasNextNanos)}%.3f " +
                  f"upstreamNextMs=${millis(upstreamNextNanos)}%.3f " +
                  f"batchSizeMs=${millis(batchSizeNanos)}%.3f " +
                  f"writerInitMs=${millis(writerInitNanos)}%.3f " +
                  f"writerAppendMs=${millis(writerAppendNanos)}%.3f " +
                  f"writerFinishMs=${millis(writerFinishNanos)}%.3f " +
                  f"scalaResidualMs=${millis(Math.max(0L, totalNanos - measuredNanos))}%.3f " +
                  s"pages=$cachePages appends=$cacheAppends batches=$cacheInputBatches " +
                  s"rows=$cacheInputRows inputBytes=$cacheInputBytes " +
                  s"parquetBytes=$cacheParquetBytes")
            }
          }

          override def hasNext: Boolean = {
            val result = hasParquetBatch
            if (!result && standaloneParquetFiles) {
              logCacheTiming()
            }
            result
          }

          override def next(): CachedBatch = {
            if (standaloneParquetFiles) {
              val file = newParquetCacheFile()
              var writerHandle = 0L
              var pageBytes = 0L
              // chunked_parquet_writer can retain its newest row group until
              // the next write or close(). Consequently sink.bytes_written()
              // is not a safe page admission signal: after the first append
              // it may still be zero, and a second large row group then turns
              // a 200 MiB target into a 400-500 MiB standalone file. Use the
              // compression ratio of completed files for admission instead.
              var estimatedPageBytes = 0L
              var pageInputBytes = 0L
              var pageRows = 0L
              try {
                val writerInitStart = System.nanoTime()
                try {
                  writerHandle = wrapper.initParquetFileWriter(file.toString)
                } finally {
                  writerInitNanos += System.nanoTime() - writerInitStart
                }
                while (estimatedPageBytes < standaloneParquetPageBytes && hasParquetBatch) {
                  // Form one bounded device micro-batch. Native concatenates
                  // it into one Parquet row group, performs one writer append
                  // and one synchronization, then all inputs are released.
                  // The file itself remains open across micro-batches.
                  val batches = new ArrayBuffer[ColumnarBatch]()
                  var appendBytes = 0L
                  var appendComplete = false
                  val appendTargetBytes = parquetAppendTarget(estimatedPageBytes)
                  while (!appendComplete && hasParquetBatch) {
                    val batch = takeParquetBatch()
                    val handle =
                      ColumnarBatches.getNativeHandle(BackendsApiManager.getBackendName, batch)
                    val batchSizeStart = System.nanoTime()
                    val batchBytes =
                      try {
                        ColumnarBatchJniWrapper.numBytes(handle)
                      } finally {
                        batchSizeNanos += System.nanoTime() - batchSizeStart
                      }
                    if (
                      batches.nonEmpty &&
                      appendBytes + batchBytes > appendTargetBytes
                    ) {
                      pendingParquetBatch = batch
                      appendComplete = true
                    } else {
                      ColumnarBatches.retain(batch)
                      batches += batch
                      appendBytes += batchBytes
                      pageRows += batch.numRows()
                      cacheInputBatches += 1
                      appendComplete = appendBytes >= appendTargetBytes
                    }
                  }
                  try {
                    val handles = batches
                      .map(ColumnarBatches.getNativeHandle(BackendsApiManager.getBackendName, _))
                      .toArray
                    // Snapshot the estimate before append: finish() below is
                    // the first point at which the final compressed size of
                    // the current row group is guaranteed to be visible.
                    val estimatedAppendOutputBytes =
                      Math.ceil(appendBytes.toDouble * parquetCompressionRatio).toLong
                    val writerAppendStart = System.nanoTime()
                    try {
                      pageBytes = wrapper.appendParquetFileWriterMany(writerHandle, handles)
                    } finally {
                      writerAppendNanos += System.nanoTime() - writerAppendStart
                    }
                    estimatedPageBytes += Math.max(1L, estimatedAppendOutputBytes)
                    pageInputBytes += appendBytes
                    cacheAppends += 1
                    cacheInputBytes += appendBytes
                  } finally {
                    batches.foreach(_.close())
                  }
                }
                val writerFinishStart = System.nanoTime()
                try {
                  pageBytes = wrapper.finishParquetFileWriter(writerHandle)
                } finally {
                  writerFinishNanos += System.nanoTime() - writerFinishStart
                }
                writerHandle = 0L
                // Update the estimator only from completed standalone files.
                // Unlike bytes_written() between appends, this includes the
                // last buffered row group and the Parquet footer.
                observedParquetInputBytes += pageInputBytes
                observedParquetOutputBytes += pageBytes
                cachePages += 1
                cacheInputRows += pageRows
                cacheParquetBytes += pageBytes
                val cached = CachedColumnarBatch(
                  Math.toIntExact(pageRows),
                  pageBytes,
                  null,
                  null,
                  file.toString)
                if (!hasParquetBatch) {
                  logCacheTiming()
                }
                cached
              } catch {
                case t: Throwable =>
                  if (writerHandle != 0L) {
                    wrapper.close(writerHandle)
                  }
                  Files.deleteIfExists(file)
                  throw t
              }
            } else {
              val batches = new ArrayBuffer[ColumnarBatch](parquetBatchesPerPage)
              while (batches.size < parquetBatchesPerPage && veloxBatches.hasNext) {
                val batch = veloxBatches.next()
                // The upstream iterator recycles its previous payload when it
                // advances. Hold one reference until serializeMany has consumed
                // every table in this cache page.
                ColumnarBatches.retain(batch)
                batches += batch
              }
              val handles = batches
                .map(ColumnarBatches.getNativeHandle(BackendsApiManager.getBackendName, _))
                .toArray
              val unsafeBuffer =
                try {
                  if (handles.length == 1) {
                    wrapper.serialize(handles.head)
                  } else {
                    wrapper.serializeMany(handles)
                  }
                } finally {
                  batches.foreach(_.close())
                }
              val numRows = Math.toIntExact(batches.foldLeft(0L)(_ + _.numRows()))
              if (storageLevel.useDisk && !storageLevel.useMemory) {
                // Keep DISK_ONLY cache pages off heap.  Materializing a wide MPP
                // relation otherwise allocates and zeroes a new JVM byte[] for
                // every native page before DiskStore immediately copies it
                // again.  UnsafeByteArray streams the Arrow buffer through the
                // Spark serializer and also enables direct JNI restore.
                // Spark may use either Kryo or Java serialization for its
                // DiskStore stream. The Kryo wrapper releases explicitly;
                // mark the Java Externalizable path to do the same after its
                // one synchronous write.
                val unsafeBytes = unsafeBuffer.toUnsafeByteArray
                CachedColumnarBatch(
                  numRows,
                  unsafeBytes.size(),
                  null,
                  unsafeBytes.releaseAfterExternalWrite())
              } else {
                val bytes = unsafeBuffer.toByteArray
                CachedColumnarBatch(numRows, bytes.length, bytes)
              }
            }
          }
        }
    }
  }

  override def convertCachedBatchToColumnarBatch(
      input: RDD[CachedBatch],
      cacheAttributes: Seq[Attribute],
      selectedAttributes: Seq[Attribute],
      conf: SQLConf): RDD[ColumnarBatch] = {
    // Count-only cache scans do not need to deserialize any cached payload.  Preserve the
    // CachedBatch row count in a zero-column batch, matching Spark RAPIDS'
    // ParquetCachedBatchSerializer.  Without this fast path, materialize().count() reads and
    // reconstructs every cached column even though the aggregate consumes none of them.
    if (selectedAttributes.isEmpty) {
      return input.map {
        batch =>
          batch match {
            case cached: CachedColumnarBatch =>
              try {
                new ColumnarBatch(Array.empty, cached.numRows)
              } finally {
                // DiskStore has already materialized this page through Java deserialization.
                // A count-only scan consumes only numRows, so release the native page now
                // instead of retaining every page until task completion.
                if (cached.unsafeBytes != null) {
                  cached.unsafeBytes.release()
                }
              }
            case _ => new ColumnarBatch(Array.empty, batch.numRows)
          }
      }
    }
    if (!validateSchema(cacheAttributes)) {
      // if we do not support this schema, that means we are using row-based serializer,
      // see `convertInternalRowToCachedBatch`, so fallback to vanilla Spark serializer
      rowBasedCachedBatchSerializer.convertCachedBatchToColumnarBatch(
        input,
        cacheAttributes,
        selectedAttributes,
        conf)
    } else {
      // Find the ordinals and data types of the requested columns.
      val requestedColumnIndices = selectedAttributes.map {
        a => cacheAttributes.map(_.exprId).indexOf(a.exprId)
      }
      val shouldSelectAttributes = cacheAttributes != selectedAttributes
      val localSchema = toStructType(cacheAttributes)
      val timezoneId = SQLConf.get.sessionLocalTimeZone
      input.mapPartitions {
        it =>
          val runtime = Runtimes.contextInstance(
            BackendsApiManager.getBackendName,
            "ColumnarCachedBatchSerializer#read")
          val jniWrapper = ColumnarBatchSerializerJniWrapper
            .create(runtime)
          // Use Gluten's shaded Arrow bridge. EMR's SparkArrowUtil exposes the same source-level
          // method with an unshaded Arrow return type, which is binary-incompatible with
          // ArrowAbiUtil even though upstream Spark compilation accepts it.
          val schema = SparkSchemaUtil.toArrowSchema(localSchema, timezoneId)
          val arrowAlloc = ArrowBufferAllocators.contextInstance()
          val cSchema = ArrowSchema.allocateNew(arrowAlloc)
          ArrowAbiUtil.exportSchema(arrowAlloc, schema, cSchema)
          val deserializerHandle = jniWrapper
            .init(cSchema.memoryAddress())
          cSchema.close()

          Iterators
            .wrap(new Iterator[ColumnarBatch] {
              override def hasNext: Boolean = it.hasNext

              override def next(): ColumnarBatch = {
                val cachedBatch = it.next().asInstanceOf[CachedColumnarBatch]
                val batchHandle = if (cachedBatch.cacheFilePath != null) {
                  if (shouldSelectAttributes) {
                    jniWrapper.deserializeParquetFileSelected(
                      deserializerHandle,
                      cachedBatch.cacheFilePath,
                      requestedColumnIndices.toArray)
                  } else {
                    jniWrapper.deserializeParquetFile(deserializerHandle, cachedBatch.cacheFilePath)
                  }
                } else if (cachedBatch.unsafeBytes != null) {
                  // Native restore is synchronous: once it returns, packed
                  // data has been copied or Parquet has been decoded to the
                  // GPU. Release the transient DiskStore read buffer now.
                  try {
                    if (shouldSelectAttributes) {
                      jniWrapper.deserializeDirectSelected(
                        deserializerHandle,
                        cachedBatch.unsafeBytes.address(),
                        Math.toIntExact(cachedBatch.unsafeBytes.size()),
                        requestedColumnIndices.toArray)
                    } else {
                      jniWrapper.deserializeDirect(
                        deserializerHandle,
                        cachedBatch.unsafeBytes.address(),
                        Math.toIntExact(cachedBatch.unsafeBytes.size()))
                    }
                  } finally {
                    cachedBatch.unsafeBytes.release()
                  }
                } else {
                  jniWrapper.deserialize(deserializerHandle, cachedBatch.bytes)
                }
                val batch = ColumnarBatches.create(batchHandle)
                if (
                  shouldSelectAttributes && cachedBatch.unsafeBytes == null &&
                  cachedBatch.cacheFilePath == null
                ) {
                  try {
                    ColumnarBatches.select(
                      BackendsApiManager.getBackendName,
                      batch,
                      requestedColumnIndices.toArray)
                  } finally {
                    batch.close()
                  }
                } else {
                  batch
                }
              }
            })
            .protectInvocationFlow()
            .recycleIterator {
              jniWrapper.close(deserializerHandle)
            }
            .recyclePayload(_.close())
            .create()
      }
    }
  }

  override def buildFilter(
      predicates: Seq[Expression],
      cachedAttributes: Seq[Attribute]): (Int, Iterator[CachedBatch]) => Iterator[CachedBatch] = {
    // TODO, support build filter as we did not support collect min/max value for columnar batch
    (_, it) => it
  }
}
