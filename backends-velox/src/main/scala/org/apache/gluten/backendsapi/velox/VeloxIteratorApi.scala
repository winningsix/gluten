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
package org.apache.gluten.backendsapi.velox

import org.apache.gluten.backendsapi.{BackendsApiManager, IteratorApi}
import org.apache.gluten.backendsapi.velox.VeloxIteratorApi.unescapePathName
import org.apache.gluten.config.{GlutenConfig, VeloxConfig}
import org.apache.gluten.execution._
import org.apache.gluten.gpu.GpuMemoryTrackerJniWrapper
import org.apache.gluten.iterator.Iterators
import org.apache.gluten.metrics.{IMetrics, IteratorMetricsJniWrapper}
import org.apache.gluten.sql.shims.SparkShimLoader
import org.apache.gluten.substrait.plan.PlanNode
import org.apache.gluten.substrait.rel.{LocalFilesBuilder, LocalFilesNode, SplitInfo}
import org.apache.gluten.substrait.rel.LocalFilesNode.ReadFileFormat
import org.apache.gluten.vectorized._

import org.apache.spark.{Partition, SparkConf, SparkException, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.shuffle.{NativeUcxShuffleExecution, NativeUcxShuffleReadSpec}
import org.apache.spark.softaffinity.SoftAffinity
import org.apache.spark.sql.catalyst.catalog.ExternalCatalogUtils
import org.apache.spark.sql.catalyst.util.{DateFormatter, TimestampFormatter}
import org.apache.spark.sql.execution.datasources.{FilePartition, PartitionedFile}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.types._
import org.apache.spark.sql.utils.SparkInputMetricsUtil.InputMetricsWrapper
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.SparkDirectoryUtil

import java.lang.{Long => JLong}
import java.nio.charset.StandardCharsets
import java.time.ZoneOffset
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference

import scala.collection.JavaConverters._
import scala.collection.mutable

class VeloxIteratorApi extends IteratorApi with Logging {

  private def setFileSchemaForLocalFiles(
      localFilesNode: LocalFilesNode,
      fileSchema: StructType,
      fileFormat: ReadFileFormat): LocalFilesNode = {
    if (
      ((fileFormat == ReadFileFormat.OrcReadFormat || fileFormat == ReadFileFormat.DwrfReadFormat)
        && !VeloxConfig.get.orcUseColumnNames)
      || (fileFormat == ReadFileFormat.ParquetReadFormat && !VeloxConfig.get.parquetUseColumnNames)
    ) {
      localFilesNode.setFileSchema(fileSchema)
    }

    localFilesNode
  }

  override def genSplitInfo(
      partitionIndex: Int,
      partitions: Seq[Partition],
      partitionSchema: StructType,
      dataSchema: StructType,
      fileFormat: ReadFileFormat,
      metadataColumnNames: Seq[String],
      properties: Map[String, String]): SplitInfo = {
    val filePartitions: Seq[FilePartition] = partitions.map {
      case p: FilePartition => p
      case o =>
        throw new UnsupportedOperationException(
          s"Unsupported input partition: ${o.getClass.getName}")
    }
    val partitionFiles = filePartitions.flatMap(_.files)
    val locations = filePartitions.flatMap(p => SoftAffinity.getFilePartitionLocations(p))
    val (paths, starts, lengths) = getPartitionedFileInfo(partitionFiles).unzip3
    val (fileSizes, modificationTimes) = partitionFiles
      .map(f => SparkShimLoader.getSparkShims.getFileSizeAndModificationTime(f))
      .collect {
        case (Some(size), Some(time)) =>
          (JLong.valueOf(size), JLong.valueOf(time))
      }
      .unzip

    val partitionColumns = getPartitionColumns(partitionSchema, partitionFiles)
    val metadataColumns = partitionFiles
      .map(
        f => SparkShimLoader.getSparkShims.generateMetadataColumns(f, metadataColumnNames).asJava)
    val otherMetadataColumns = partitionFiles
      .map(f => SparkShimLoader.getSparkShims.getOtherConstantMetadataColumnValues(f))

    setFileSchemaForLocalFiles(
      LocalFilesBuilder.makeLocalFiles(
        partitionIndex,
        paths.asJava,
        starts.asJava,
        lengths.asJava,
        fileSizes.asJava,
        modificationTimes.asJava,
        partitionColumns.map(_.asJava).asJava,
        metadataColumns.asJava,
        fileFormat,
        locations.toList.asJava,
        mapAsJavaMap(properties),
        otherMetadataColumns.asJava
      ),
      dataSchema,
      fileFormat
    )
  }

  /** Generate native row partition. */
  override def genPartitions(
      wsCtx: WholeStageTransformContext,
      splitInfos: Seq[Seq[SplitInfo]],
      leaves: Seq[LeafTransformSupport]): Seq[BaseGlutenPartition] = {
    // Only serialize plan once, save lots time when plan is complex.
    val planByteArray = wsCtx.root.toProtobuf.toByteArray

    splitInfos.zipWithIndex.map {
      case (splitInfos, index) =>
        GlutenPartition(
          index,
          planByteArray,
          splitInfos.toArray
        )
    }
  }

  private def getPartitionedFileInfo(
      partitionedFiles: Seq[PartitionedFile]): Seq[(String, JLong, JLong)] = {
    partitionedFiles.map {
      partitionedFile =>
        val path = unescapePathName(partitionedFile.filePath.toString)
        (path, JLong.valueOf(partitionedFile.start), JLong.valueOf(partitionedFile.length))
    }
  }

  private def getPartitionColumns(
      schema: StructType,
      partitionedFiles: Seq[PartitionedFile]): Seq[Map[String, String]] = {
    partitionedFiles.map {
      partitionedFile =>
        val partitionColumn = mutable.Map[String, String]()
        for (i <- 0 until partitionedFile.partitionValues.numFields) {
          val partitionColumnValue = if (partitionedFile.partitionValues.isNullAt(i)) {
            ExternalCatalogUtils.DEFAULT_PARTITION_NAME
          } else {
            val pv = partitionedFile.partitionValues.get(i, schema.fields(i).dataType)
            schema.fields(i).dataType match {
              case _: BinaryType =>
                new String(pv.asInstanceOf[Array[Byte]], StandardCharsets.UTF_8)
              case _: DateType =>
                DateFormatter.apply().format(pv.asInstanceOf[Integer])
              case _: DecimalType =>
                pv.asInstanceOf[Decimal].toJavaBigInteger.toString
              case _: TimestampType =>
                TimestampFormatter
                  .getFractionFormatter(ZoneOffset.UTC)
                  .format(pv.asInstanceOf[java.lang.Long])
              case _ => pv.toString
            }
          }
          partitionColumn += (schema.names(i) -> partitionColumnValue)
        }
        partitionColumn.toMap
    }
  }

  override def injectWriteFilesTempPath(path: String, fileName: String): Unit = {
    NativePlanEvaluator.injectWriteFilesTempPath(path, fileName)
  }

  /** Generate Iterator[ColumnarBatch] for first stage. */
  override def genFirstStageIterator(
      inputPartition: BaseGlutenPartition,
      context: TaskContext,
      pipelineTime: SQLMetric,
      updateInputMetrics: InputMetricsWrapper => Unit,
      updateNativeMetrics: IMetrics => Unit,
      partitionIndex: Int,
      inputIterators: Seq[Iterator[ColumnarBatch]] = Seq(),
      enableCudf: Boolean = false,
      skipOutputToVelox: Boolean = false
  ): Iterator[ColumnarBatch] = {
    assert(
      inputPartition.isInstanceOf[GlutenPartition],
      "Velox backend only accept GlutenPartition.")
    val glutenPartition = inputPartition.asInstanceOf[GlutenPartition]
    val plannedInputBytes = glutenPartition.splitInfos.collect {
      case localFiles: LocalFilesNode =>
        localFiles.getLengths.asScala.foldLeft(0L)(_ + _.longValue())
    }.sum

    val planBuildStart = System.nanoTime()

    if (enableCudf) {
      trySetCurrentTask(context)
    }

    val taskContextBoundInputIterators = bindTaskContextToInputIterators(inputIterators, context)
    val columnarNativeIterators = taskContextBoundInputIterators.map {
      iter => new ColumnarBatchInIterator(BackendsApiManager.getBackendName, iter.asJava)
    }

    val extraConf = nativeUcxExtraConf(
      inputIterators,
      Map(
      GlutenConfig.COLUMNAR_CUDF_ENABLED.key ->
        enableCudf.toString,
      "spark.gluten.sql.columnar.cudf.skipOutputToVelox" ->
        skipOutputToVelox.toString
      ))
      .asJava
    val transKernel = NativePlanEvaluator
      .create(BackendsApiManager.getBackendName, extraConf)

    val splitInfoByteArray = glutenPartition.splitInfos
      .map(splitInfo => splitInfo.toProtobuf.toByteArray)
      .toArray
    val spillDirPath = SparkDirectoryUtil
      .get()
      .namespace("gluten-spill")
      .mkChildDirRoundRobin(UUID.randomUUID.toString)
      .getAbsolutePath
    val resIter: ColumnarBatchOutIterator =
      transKernel.createKernelWithBatchIterator(
        inputPartition.plan,
        if (splitInfoByteArray.nonEmpty) {
          splitInfoByteArray
        } else null,
        if (columnarNativeIterators.nonEmpty) {
          columnarNativeIterators.toArray
        } else null,
        partitionIndex,
        BackendsApiManager.getSparkPlanExecApiInstance
          .rewriteSpillPath(spillDirPath)
      )
    resIter.noMoreSplits()
    val nativeUcxPollerFailure = new AtomicReference[Throwable](null)
    val nativeUcxPollers =
      startNativeUcxExchangePollers(inputIterators, resIter, nativeUcxPollerFailure)

    val tracker =
      org.apache.gluten.metrics.TaskWallTimeTracker.get()
    tracker.planBuildNanos +=
      (System.nanoTime() - planBuildStart)
    val itrMetrics = IteratorMetricsJniWrapper.create()

    Iterators
      .wrap(checkedNativeUcxIterator(resIter.asScala, nativeUcxPollerFailure))
      .protectInvocationFlow()
      .recycleIterator {
        stopNativeUcxExchangePollers(nativeUcxPollers)
        updateNativeMetrics(itrMetrics.fetch(resIter))
        val inputMetrics = context.taskMetrics().inputMetrics
        updateInputMetrics(inputMetrics)
        // A native UCX sink may still be draining when Spark snapshots the task. In that case
        // live Velox scan statistics can be empty even though all file splits were consumed.
        // Preserve the actual split lengths as a conservative Stage "Input Size" fallback.
        if (inputMetrics.bytesRead == 0L && plannedInputBytes > 0L) {
          inputMetrics.bridgeIncBytesRead(plannedInputBytes)
        }
        resIter.close()
        if (enableCudf) {
          tryStopTaskTracking(context)
          tryClearTaskMemory(context)
        }
      }
      .recyclePayload(batch => batch.close())
      .collectLifeMillis(millis => pipelineTime += millis)
      .asInterruptible(context)
      .create()
  }

  // scalastyle:off argcount

  /** Generate Iterator[ColumnarBatch] for final stage. */
  override def genFinalStageIterator(
      context: TaskContext,
      inputIterators: Seq[Iterator[ColumnarBatch]],
      sparkConf: SparkConf,
      rootNode: PlanNode,
      pipelineTime: SQLMetric,
      updateNativeMetrics: IMetrics => Unit,
      partitionIndex: Int,
      materializeInput: Boolean,
      enableCudf: Boolean = false,
      skipOutputToVelox: Boolean = false
  ): Iterator[ColumnarBatch] = {
    val planBuildStart = System.nanoTime()

    if (enableCudf) {
      trySetCurrentTask(context)
    }

    val extraConf = nativeUcxExtraConf(
      inputIterators,
      Map(
      GlutenConfig.COLUMNAR_CUDF_ENABLED.key ->
        enableCudf.toString,
      "spark.gluten.sql.columnar.cudf.skipOutputToVelox" ->
        skipOutputToVelox.toString
      ))
      .asJava
    val transKernel = NativePlanEvaluator
      .create(BackendsApiManager.getBackendName, extraConf)
    val taskContextBoundInputIterators = bindTaskContextToInputIterators(inputIterators, context)
    val columnarNativeIterator =
      taskContextBoundInputIterators.map {
        iter => new ColumnarBatchInIterator(BackendsApiManager.getBackendName, iter.asJava)
      }
    val spillDirPath = SparkDirectoryUtil
      .get()
      .namespace("gluten-spill")
      .mkChildDirRoundRobin(UUID.randomUUID.toString)
      .getAbsolutePath
    val nativeResultIterator =
      transKernel.createKernelWithBatchIterator(
        rootNode.toProtobuf.toByteArray,
        null,
        if (columnarNativeIterator.nonEmpty) {
          columnarNativeIterator.toArray
        } else null,
        partitionIndex,
        BackendsApiManager.getSparkPlanExecApiInstance
          .rewriteSpillPath(spillDirPath)
      )
    nativeResultIterator.noMoreSplits()
    val nativeUcxPollerFailure = new AtomicReference[Throwable](null)
    val nativeUcxPollers =
      startNativeUcxExchangePollers(inputIterators, nativeResultIterator, nativeUcxPollerFailure)

    val tracker =
      org.apache.gluten.metrics.TaskWallTimeTracker.get()
    tracker.planBuildNanos +=
      (System.nanoTime() - planBuildStart)
    val itrMetrics = IteratorMetricsJniWrapper.create()

    Iterators
      .wrap(checkedNativeUcxIterator(nativeResultIterator.asScala, nativeUcxPollerFailure))
      .protectInvocationFlow()
      .recycleIterator {
        stopNativeUcxExchangePollers(nativeUcxPollers)
        updateNativeMetrics(itrMetrics.fetch(nativeResultIterator))
        nativeResultIterator.close()
        if (enableCudf) {
          tryStopTaskTracking(context)
          tryClearTaskMemory(context)
        }
      }
      .recyclePayload(batch => batch.close())
      .collectLifeMillis(millis => pipelineTime += millis)
      .create()
  }
  // scalastyle:on argcount

  private def trySetCurrentTask(context: TaskContext): Unit = {
    try {
      GpuMemoryTrackerJniWrapper.setCurrentTask(context.taskAttemptId())
    } catch {
      case _: UnsatisfiedLinkError => // JNI not loaded, skip
    }
  }

  private def nativeUcxExtraConf(
      inputIterators: Seq[Iterator[ColumnarBatch]],
      base: Map[String, String]): Map[String, String] = {
    val readerStreams = nativeUcxReaderStreams(inputIterators).map(_._1)
    val writerConf =
      NativeUcxShuffleExecution.currentWriterContext
        .map(NativeUcxShuffleExecution.writerExtraConf)
        .getOrElse(Map.empty)
    val readerConf = NativeUcxShuffleExecution.readerExtraConf(readerStreams)
    val merged = base ++ writerConf ++ readerConf
    if (writerConf.nonEmpty || readerConf.nonEmpty) {
      logInfo(
        s"Velox native UCX shuffle conf writer=${writerConf.nonEmpty} " +
          s"readerStreams=${readerStreams.sorted.mkString("[", ",", "]")} " +
          s"inputIterators=${nativeUcxInputSummary(inputIterators)}")
    }
    merged
  }

  private def nativeUcxReaderStreams(
      inputIterators: Seq[Iterator[ColumnarBatch]]): Seq[(Int, NativeUcxShuffleReadSpec)] = {
    inputIterators.zipWithIndex.flatMap {
      case (iterator, streamIdx) =>
        NativeUcxShuffleExecution.readSpec(iterator).map(streamIdx -> _)
    }
  }

  private def nativeUcxInputSummary(
      inputIterators: Seq[Iterator[ColumnarBatch]]): String = {
    inputIterators.zipWithIndex
      .map {
        case (iterator, streamIdx) =>
          val nativeSpec = NativeUcxShuffleExecution
            .readSpec(iterator)
            .map {
              spec =>
                s"native(shuffleId=${spec.shuffleId},reduce=${spec.reducePartitionId}," +
                  s"maps=${spec.expectedMaps},mapRange=[${spec.startMapIndex},${spec.endMapIndex}))"
            }
            .getOrElse("local")
          s"$streamIdx=${iterator.getClass.getName}:$nativeSpec"
      }
      .mkString("[", ",", "]")
  }

  private def bindTaskContextToInputIterators(
      inputIterators: Seq[Iterator[ColumnarBatch]],
      context: TaskContext): Seq[Iterator[ColumnarBatch]] = {
    inputIterators.map { input =>
      new Iterator[ColumnarBatch] {
        override def hasNext: Boolean =
          withCapturedTaskContext(context) {
            input.hasNext
          }

        override def next(): ColumnarBatch =
          withCapturedTaskContext(context) {
            input.next()
          }
      }
    }
  }

  private def withCapturedTaskContext[T](context: TaskContext)(body: => T): T = {
    if (TaskContext.get() == null) {
      TaskContext.withTaskContext(context) {
        body
      }
    } else {
      body
    }
  }

  private def startNativeUcxExchangePollers(
      inputIterators: Seq[Iterator[ColumnarBatch]],
      nativeResultIterator: ColumnarBatchOutIterator,
      failure: AtomicReference[Throwable]): Seq[Thread] = {
    nativeUcxReaderStreams(inputIterators).map {
      case (streamIdx, spec) =>
        val exchangeNodeId = NativeUcxShuffleExecution.exchangeNodeId(streamIdx)
        NativeUcxShuffleExecution.startVeloxExchangeSplitPoller(
          spec,
          streamIdx,
          urls => nativeResultIterator.addUcxExchangeSplits(exchangeNodeId, urls),
          () => nativeResultIterator.noMoreUcxExchangeSplits(exchangeNodeId),
          error => failure.compareAndSet(null, error))
    }
  }

  private def stopNativeUcxExchangePollers(pollers: Seq[Thread]): Unit = {
    pollers.foreach(_.interrupt())
    pollers.foreach {
      poller =>
        if (Thread.currentThread() != poller) {
          try {
            poller.join(1000L)
          } catch {
            case e: InterruptedException =>
              Thread.currentThread().interrupt()
              throw new SparkException("Interrupted stopping Velox UCX exchange split poller", e)
          }
        }
    }
  }

  private def checkedNativeUcxIterator(
      delegate: Iterator[ColumnarBatch],
      failure: AtomicReference[Throwable]): Iterator[ColumnarBatch] = {
    new Iterator[ColumnarBatch] {
      override def hasNext: Boolean = {
        throwIfNativeUcxPollerFailed(failure)
        val result = delegate.hasNext
        throwIfNativeUcxPollerFailed(failure)
        result
      }

      override def next(): ColumnarBatch = {
        throwIfNativeUcxPollerFailed(failure)
        val result = delegate.next()
        throwIfNativeUcxPollerFailed(failure)
        result
      }
    }
  }

  private def throwIfNativeUcxPollerFailed(failure: AtomicReference[Throwable]): Unit = {
    val error = failure.get()
    if (error != null) {
      throw new SparkException("Velox UCX exchange split poller failed", error)
    }
  }

  /** Stop attributing new GPU allocations to this task. */
  private def tryStopTaskTracking(context: TaskContext): Unit = {
    try {
      GpuMemoryTrackerJniWrapper.clearCurrentTask()
    } catch {
      case _: UnsatisfiedLinkError =>
    }
  }

  /** Clean up native tracking data after semaphore release has recorded the peak memory. */
  private def tryClearTaskMemory(context: TaskContext): Unit = {
    try {
      GpuMemoryTrackerJniWrapper.clearTaskMemory(context.taskAttemptId())
    } catch {
      case _: UnsatisfiedLinkError =>
    }
  }
}

object VeloxIteratorApi {
  // lookup table to translate '0' -> 0 ... 'F'/'f' -> 15
  private val unhexDigits = {
    val array = Array.fill[Byte](128)(-1)
    (0 to 9).foreach(i => array('0' + i) = i.toByte)
    (0 to 5).foreach(i => array('A' + i) = (i + 10).toByte)
    (0 to 5).foreach(i => array('a' + i) = (i + 10).toByte)
    array
  }

  def unescapePathName(path: String): String = {
    if (path == null || path.isEmpty) {
      return path
    }
    var plaintextEndIdx = path.indexOf('%')
    val length = path.length
    if (plaintextEndIdx == -1 || plaintextEndIdx + 2 >= length) {
      // fast path, no %xx encoding found then return the string identity
      path
    } else {
      val sb = new java.lang.StringBuilder(length)
      var plaintextStartIdx = 0
      while (plaintextEndIdx != -1 && plaintextEndIdx + 2 < length) {
        if (plaintextEndIdx > plaintextStartIdx) sb.append(path, plaintextStartIdx, plaintextEndIdx)
        val high = path.charAt(plaintextEndIdx + 1)
        if ((high >>> 8) == 0 && unhexDigits(high) != -1) {
          val low = path.charAt(plaintextEndIdx + 2)
          if ((low >>> 8) == 0 && unhexDigits(low) != -1) {
            sb.append((unhexDigits(high) << 4 | unhexDigits(low)).asInstanceOf[Char])
            plaintextStartIdx = plaintextEndIdx + 3
          } else {
            sb.append('%')
            plaintextStartIdx = plaintextEndIdx + 1
          }
        } else {
          sb.append('%')
          plaintextStartIdx = plaintextEndIdx + 1
        }
        plaintextEndIdx = path.indexOf('%', plaintextStartIdx)
      }
      if (plaintextStartIdx < length) {
        sb.append(path, plaintextStartIdx, length)
      }
      sb.toString
    }
  }
}
