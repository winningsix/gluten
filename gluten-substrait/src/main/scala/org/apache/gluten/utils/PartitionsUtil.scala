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
package org.apache.gluten.utils

import org.apache.gluten.config.GlutenConfig
import org.apache.gluten.sql.shims.SparkShimLoader
import org.apache.gluten.utils.PartitionsUtil.regeneratePartition

import org.apache.spark.Partition
import org.apache.spark.internal.Logging
import org.apache.spark.network.util.JavaUtils
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution.datasources.{BucketingUtils, FilePartition, HadoopFsRelation, PartitionDirectory, PartitionedFile}
import org.apache.spark.sql.types.StructType
import org.apache.spark.util.collection.BitSet

import org.apache.hadoop.fs.{FileStatus, Path}

import scala.collection.mutable

case class PartitionsUtil(
    relation: HadoopFsRelation,
    requiredSchema: StructType,
    selectedPartitions: Array[PartitionDirectory],
    output: Seq[Attribute],
    bucketedScan: Boolean,
    optionalBucketSet: Option[BitSet],
    optionalNumCoalescedBuckets: Option[Int],
    disableBucketedScan: Boolean)
  extends Logging {

  def genPartitionSeq(): Seq[Partition] = {
    if (bucketedScan) {
      genBucketedPartitionSeq()
    } else {
      genNonBuckedPartitionSeq()
    }
  }

  private def genNonBuckedPartitionSeq(): Seq[Partition] = {
    val openCostInBytes = relation.sparkSession.sessionState.conf.filesOpenCostInBytes
    val sparkMaxSplitBytes =
      FilePartition.maxSplitBytes(relation.sparkSession, selectedPartitions)
    val partitionFiles = selectedPartitions.flatMap {
      partition =>
        SparkShimLoader.getSparkShims.getFileStatus(partition).map(file => (partition, file))
    }
    val splitPlanning =
      maybeAdjustMppScanSplitBytes(sparkMaxSplitBytes, partitionFiles, openCostInBytes)
    logInfo(
      s"Planning scan with bin packing, max size: ${splitPlanning.partitionMaxSplitBytes} bytes, " +
        s"physical split max size: ${splitPlanning.physicalSplitBytes} bytes, " +
        s"open cost is considered as scanning $openCostInBytes bytes.")

    // Filter files with bucket pruning if possible
    val bucketingEnabled = relation.sparkSession.sessionState.conf.bucketingEnabled
    val shouldProcess: Path => Boolean = optionalBucketSet match {
      case Some(bucketSet) if bucketingEnabled =>
        filePath => {
          BucketingUtils.getBucketId(filePath.getName) match {
            case Some(id) => bucketSet.get(id)
            case None =>
              // Do not prune the file if bucket file name is invalid
              true
          }
        }
      case _ =>
        _ => true
    }

    val splitFiles = partitionFiles
      .flatMap {
        case (partition, file) =>
          // getPath() is very expensive so we only want to call it once in this block:
          val filePath = file._1.getPath
          if (shouldProcess(filePath)) {
            val isSplitable =
              SparkShimLoader.getSparkShims.isFileSplittable(relation, filePath, requiredSchema)
            SparkShimLoader.getSparkShims.splitFiles(
              sparkSession = relation.sparkSession,
              file = file._1,
              filePath = filePath,
              isSplitable = isSplitable,
              maxSplitBytes = splitPlanning.physicalSplitBytes,
              partitionValues = partition.values,
              metadata = file._2
            )
          } else {
            Seq.empty
          }
      }
      .sortBy(_.length)(implicitly[Ordering[Long]].reverse)

    val inputPartitions =
      FilePartition.getFilePartitions(
        relation.sparkSession,
        splitFiles,
        splitPlanning.partitionMaxSplitBytes)

    regeneratePartition(inputPartitions, GlutenConfig.get.smallFileThreshold)
  }

  private def maybeAdjustMppScanSplitBytes(
      sparkMaxSplitBytes: Long,
      partitionFiles: Seq[(PartitionDirectory, (FileStatus, Map[String, Any]))],
      openCostInBytes: Long): PartitionsUtil.MppScanSplitPlanning = {
    val conf = relation.sparkSession.sessionState.conf
    val mppEnabled = conf.getConfString("spark.gluten.mpp.enabled", "false").toBoolean
    val singleTaskMode =
      conf
        .getConfString("spark.gluten.sql.columnar.backend.velox.mpp.singleTaskMode", "false")
        .toBoolean
    val sizeAwareEnabled =
      conf.getConfString("spark.gluten.mpp.scan.sizeAwarePartitioning", "true").toBoolean &&
        !singleTaskMode
    if (!mppEnabled || !sizeAwareEnabled || partitionFiles.isEmpty) {
      return PartitionsUtil.MppScanSplitPlanning(sparkMaxSplitBytes, sparkMaxSplitBytes)
    }

    val targetSplitBytes = optionalBytesConf("spark.gluten.mpp.scan.targetSplitBytes")
    val maxNativeSplitBytes = optionalBytesConf("spark.gluten.mpp.scan.maxNativeSplitBytes")
    val maxWholeFileBytes =
      bytesConf("spark.gluten.mpp.scan.maxWholeFileBytes", conf.filesMaxPartitionBytes.toString)
    val fileSizes = partitionFiles.map(_._2._1.getLen)
    val wholeFileMinFiles =
      intConf(
        "spark.gluten.mpp.scan.wholeFileMinFiles",
        conf.getConfString("spark.gluten.mpp.multiExecutor.numPartitions", "1"))
    val wholeFileFloorEnabled =
      conf.getConfString("spark.gluten.mpp.scan.wholeFileFloor", "true").toBoolean

    val splitPlanning = PartitionsUtil.planMppScanSplitBytes(
      sparkMaxSplitBytes = sparkMaxSplitBytes,
      fileSizes = fileSizes,
      openCostInBytes = openCostInBytes,
      mppEnabled = mppEnabled,
      sizeAwareEnabled = sizeAwareEnabled,
      targetSplitBytes = targetSplitBytes,
      maxWholeFileBytes = maxWholeFileBytes,
      wholeFileMinFiles = wholeFileMinFiles,
      wholeFileFloorEnabled = wholeFileFloorEnabled,
      maxNativeSplitBytes = maxNativeSplitBytes
    )

    if (
      splitPlanning.physicalSplitBytes != sparkMaxSplitBytes ||
      splitPlanning.partitionMaxSplitBytes != sparkMaxSplitBytes
    ) {
      logInfo(
        s"Adjusted MPP scan split bytes from $sparkMaxSplitBytes to " +
          s"physical=${splitPlanning.physicalSplitBytes}, " +
          s"partition=${splitPlanning.partitionMaxSplitBytes} " +
          s"(target=${targetSplitBytes.map(_.toString).getOrElse("<unset>")}, " +
          s"maxNative=${maxNativeSplitBytes.map(_.toString).getOrElse("<unset>")}, " +
          s"wholeFileFloorEnabled=$wholeFileFloorEnabled, largestFile=${fileSizes.max}, " +
          s"maxWholeFile=$maxWholeFileBytes, fileCount=${partitionFiles.size}, " +
          s"wholeFileMinFiles=$wholeFileMinFiles).")
    }
    splitPlanning
  }

  private def optionalBytesConf(key: String): Option[Long] = {
    relation.sparkSession.sessionState.conf.getAllConfs
      .get(key)
      .map(JavaUtils.byteStringAsBytes)
  }

  private def bytesConf(key: String, defaultValue: String): Long = {
    val raw = relation.sparkSession.sessionState.conf.getConfString(key, defaultValue)
    JavaUtils.byteStringAsBytes(raw)
  }

  private def intConf(key: String, defaultValue: String): Int = {
    val raw = relation.sparkSession.sessionState.conf.getConfString(key, defaultValue)
    math.max(1, raw.toInt)
  }

  private def genBucketedPartitionSeq(): Seq[Partition] = {
    val bucketSpec = relation.bucketSpec.get
    logInfo(s"Planning with ${bucketSpec.numBuckets} buckets")
    val filesGroupedToBuckets =
      SparkShimLoader.getSparkShims.filesGroupedToBuckets(selectedPartitions)

    val prunedFilesGroupedToBuckets = if (optionalBucketSet.isDefined) {
      val bucketSet = optionalBucketSet.get
      filesGroupedToBuckets.filter(f => bucketSet.get(f._1))
    } else {
      filesGroupedToBuckets
    }

    optionalNumCoalescedBuckets
      .map {
        numCoalescedBuckets =>
          logInfo(s"Coalescing to $numCoalescedBuckets buckets")
          val coalescedBuckets = prunedFilesGroupedToBuckets.groupBy(_._1 % numCoalescedBuckets)
          Seq.tabulate(numCoalescedBuckets) {
            bucketId =>
              val partitionedFiles = coalescedBuckets
                .get(bucketId)
                .map {
                  _.values.flatten.toArray
                }
                .getOrElse(Array.empty)
              FilePartition(bucketId, partitionedFiles)
          }
      }
      .getOrElse {
        Seq.tabulate(bucketSpec.numBuckets) {
          bucketId =>
            FilePartition(bucketId, prunedFilesGroupedToBuckets.getOrElse(bucketId, Array.empty))
        }
      }
  }

  private def toAttribute(colName: String): Option[Attribute] = {
    output.find(_.name == colName)
  }
}

object PartitionsUtil {
  private[utils] case class MppScanSplitPlanning(
      physicalSplitBytes: Long,
      partitionMaxSplitBytes: Long)

  private[utils] def planMppScanSplitBytes(
      sparkMaxSplitBytes: Long,
      fileSizes: Seq[Long],
      openCostInBytes: Long,
      mppEnabled: Boolean,
      sizeAwareEnabled: Boolean,
      targetSplitBytes: Option[Long],
      maxWholeFileBytes: Long,
      wholeFileMinFiles: Int,
      wholeFileFloorEnabled: Boolean,
      maxNativeSplitBytes: Option[Long]): MppScanSplitPlanning = {
    if (!mppEnabled || !sizeAwareEnabled || fileSizes.isEmpty) {
      return MppScanSplitPlanning(sparkMaxSplitBytes, sparkMaxSplitBytes)
    }

    val largestFileBytes = fileSizes.max
    val wholeFileEligible =
      fileSizes.size >= math.max(1, wholeFileMinFiles) &&
        largestFileBytes > 0 &&
        largestFileBytes <= maxWholeFileBytes
    val wholeFileFloor =
      if (wholeFileFloorEnabled && wholeFileEligible) {
        safeAdd(largestFileBytes, openCostInBytes)
      } else {
        0L
      }
    val targetForPacking =
      targetSplitBytes.filter(_ => maxNativeSplitBytes.isEmpty || wholeFileEligible)
    val partitionMaxSplitBytes =
      maxPositive(Seq(sparkMaxSplitBytes, wholeFileFloor) ++ targetForPacking)

    val uncappedPhysicalSplitBytes =
      maxPositive(Seq(sparkMaxSplitBytes, wholeFileFloor) ++ targetSplitBytes)
    val physicalSplitBytes =
      maxNativeSplitBytes
        .filter(limit => limit > 0 && largestFileBytes > limit)
        .map(limit => math.min(uncappedPhysicalSplitBytes, limit))
        .getOrElse(uncappedPhysicalSplitBytes)

    MppScanSplitPlanning(physicalSplitBytes, partitionMaxSplitBytes)
  }

  private def maxPositive(values: Seq[Long]): Long = {
    values.filter(_ > 0).reduceOption(_ max _).getOrElse(0L)
  }

  private def safeAdd(left: Long, right: Long): Long = {
    if (Long.MaxValue - left < right) Long.MaxValue else left + right
  }

  /**
   * Regenerate the partitions by balancing the number of files per partition and total size per
   * partition.
   */
  def regeneratePartition(
      inputPartitions: Seq[FilePartition],
      smallFileThreshold: Double): Seq[FilePartition] = {

    val partitions = Array.fill(inputPartitions.size)(mutable.ArrayBuffer.empty[PartitionedFile])

    def addToBucket(
        heap: mutable.PriorityQueue[(Long, Int, Int)],
        file: PartitionedFile,
        fileSize: Long): Unit = {
      val (size, numFiles, idx) = heap.dequeue()
      partitions(idx) += file
      heap.enqueue((size + fileSize, numFiles + 1, idx))
    }

    def initializeHeap(
        ordering: Ordering[(Long, Int, Int)]): mutable.PriorityQueue[(Long, Int, Int)] = {
      val heap = mutable.PriorityQueue.empty[(Long, Int, Int)](ordering)
      inputPartitions.indices.foreach(i => heap.enqueue((0L, 0, i)))
      heap
    }

    // Flatten and sort descending by file size.
    val filesSorted: Seq[(PartitionedFile, Long)] =
      inputPartitions
        .flatMap(_.files)
        .map(f => (f, f.length))
        .sortBy(_._2)(Ordering.Long.reverse)

    // First by size, then by number of files.
    val sizeFirstOrdering = Ordering
      .by[(Long, Int, Int), (Long, Int)] { case (size, numFiles, _) => (size, numFiles) }
      .reverse

    if (smallFileThreshold > 0) {
      val smallFileTotalSize = filesSorted.map(_._2).sum * smallFileThreshold
      // First by number of files, then by size.
      val numFirstOrdering = Ordering
        .by[(Long, Int, Int), (Int, Long)] { case (size, numFiles, _) => (numFiles, size) }
        .reverse
      val heapByFileNum = initializeHeap(numFirstOrdering)

      var numSmallFiles = 0
      var smallFileSize = 0L
      // Distribute small files evenly across partitions to achieve load balancing of small files.
      filesSorted.reverseIterator
        .takeWhile(f => f._2 + smallFileSize <= smallFileTotalSize)
        .foreach {
          case (file, fileSize) =>
            addToBucket(heapByFileNum, file, fileSize)
            numSmallFiles += 1
            smallFileSize += fileSize
        }

      val heapByFileSize = mutable.PriorityQueue.empty[(Long, Int, Int)](sizeFirstOrdering)
      // Move buckets from heapByFileNum to heapByFileSize.
      while (heapByFileNum.nonEmpty) {
        heapByFileSize.enqueue(heapByFileNum.dequeue())
      }

      // Finally, enqueue remaining files.
      filesSorted.take(filesSorted.size - numSmallFiles).foreach {
        case (file, fileSize) =>
          addToBucket(heapByFileSize, file, fileSize)
      }
    } else {
      val heapByFileSize = initializeHeap(sizeFirstOrdering)

      filesSorted.foreach {
        case (file, fileSize) =>
          addToBucket(heapByFileSize, file, fileSize)
      }
    }

    partitions.zipWithIndex.map { case (p, idx) => FilePartition(idx, p.toArray) }
  }
}
