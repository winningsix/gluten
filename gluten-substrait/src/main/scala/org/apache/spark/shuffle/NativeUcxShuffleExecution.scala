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
package org.apache.spark.shuffle

import org.apache.spark.{SparkConf, SparkEnv, SparkException, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.vectorized.ColumnarBatch

import java.lang.reflect.Modifier
import java.util.IdentityHashMap
import java.util.concurrent.atomic.AtomicReference

import scala.collection.mutable
import scala.util.control.NonFatal

case class NativeUcxShuffleWriterContext(
    shuffleId: Int,
    mapId: Long,
    attemptId: Long,
    nativeTaskId: String,
    numPartitions: Int,
    partitioning: String,
    startPartitionId: Int,
    dropFirstColumn: Boolean,
    partitionKeyIndices: Seq[Int])

case class NativeUcxShuffleReadSpec(
    shuffleId: Int,
    reducePartitionId: Int,
    startMapIndex: Int,
    endMapIndex: Int,
    expectedMaps: Int,
    initialEndpoints: Seq[UcxShuffleEndpoint],
    taskAttemptId: Long = -1L,
    nativeReaderId: String = "",
    replicated: Boolean = false)

trait NativeUcxShuffleReadMetadata {
  def nativeUcxShuffleReadSpec: NativeUcxShuffleReadSpec
}

class NativeUcxShuffleReadProductIterator[K, C](val spec: NativeUcxShuffleReadSpec)
  extends Iterator[Product2[K, C]]
  with NativeUcxShuffleReadMetadata {
  override def nativeUcxShuffleReadSpec: NativeUcxShuffleReadSpec = spec

  override def hasNext: Boolean = false

  override def next(): Product2[K, C] = Iterator.empty.next()
}

class NativeUcxShuffleReadBatchIterator(val spec: NativeUcxShuffleReadSpec)
  extends Iterator[ColumnarBatch]
  with NativeUcxShuffleReadMetadata {
  override def nativeUcxShuffleReadSpec: NativeUcxShuffleReadSpec = spec

  override def hasNext: Boolean = false

  override def next(): ColumnarBatch = Iterator.empty.next()

  def startVeloxExchangeSplitPoller(
      streamIdx: Int,
      addSplits: Array[String] => Unit,
      noMoreSplits: () => Unit,
      onFailure: Throwable => Unit): Thread = {
    NativeUcxShuffleExecution.startVeloxExchangeSplitPoller(
      spec,
      streamIdx,
      addSplits,
      noMoreSplits,
      onFailure)
  }
}

class NativeUcxShuffleReadMetadataIterator[T](
    delegate: Iterator[T],
    val spec: NativeUcxShuffleReadSpec)
  extends Iterator[T]
  with NativeUcxShuffleReadMetadata {
  override def nativeUcxShuffleReadSpec: NativeUcxShuffleReadSpec = spec

  override def hasNext: Boolean = delegate.hasNext

  override def next(): T = delegate.next()
}

object NativeUcxShuffleExecution extends Logging {
  val EnabledConf: String = "spark.gluten.ucx.shuffle.nativeExchange.enabled"
  val RawHashEnabledConf: String = "spark.gluten.ucx.shuffle.native.rawHash.enabled"
  val WriteEnabledConf: String = "spark.gluten.ucx.shuffle.native.write.enabled"
  val ReadStreamIndicesConf: String = "spark.gluten.ucx.shuffle.native.read.streams"
  val ReplicatedReadStreamIndicesConf: String =
    "spark.gluten.ucx.shuffle.native.read.replicatedStreams"
  val NativeTaskIdConf: String = "spark.gluten.ucx.shuffle.native.taskId"
  val WriteNumPartitionsConf: String = "spark.gluten.ucx.shuffle.native.write.numPartitions"
  val WritePartitioningConf: String = "spark.gluten.ucx.shuffle.native.write.partitioning"
  val WriteStartPartitionIdConf: String =
    "spark.gluten.ucx.shuffle.native.write.startPartitionId"
  val WriteDropFirstColumnConf: String = "spark.gluten.ucx.shuffle.native.write.dropFirstColumn"
  val WritePartitionKeyIndicesConf: String =
    "spark.gluten.ucx.shuffle.native.write.partitionKeyIndices"

  private val MaxReadSpecUnwrapDepth = 8

  private val writerContext = new ThreadLocal[NativeUcxShuffleWriterContext]()
  private val readSpecCapture =
    new ThreadLocal[mutable.ArrayBuffer[NativeUcxShuffleReadSpec]]()

  def enabled(conf: SparkConf): Boolean = conf.getBoolean(EnabledConf, false)

  def writerTaskId(shuffleId: Int, mapId: Long, attemptId: Long): String =
    s"Gluten_UCX_Shuffle_${shuffleId}_Map_${mapId}_Attempt_$attemptId"

  def exchangeNodeId(streamIdx: Int): String = s"ucx_exchange_$streamIdx"

  def currentWriterContext: Option[NativeUcxShuffleWriterContext] =
    Option(writerContext.get())

  def captureReadSpec(spec: NativeUcxShuffleReadSpec): Unit = {
    val buffer = readSpecCapture.get()
    if (buffer != null) {
      buffer += spec
    }
  }

  def withReadSpecCapture[T](f: => T): (T, Seq[NativeUcxShuffleReadSpec]) = {
    val previous = readSpecCapture.get()
    val current = mutable.ArrayBuffer.empty[NativeUcxShuffleReadSpec]
    readSpecCapture.set(current)
    try {
      val result = f
      (result, current.toSeq)
    } finally {
      if (previous == null) {
        readSpecCapture.remove()
      } else {
        readSpecCapture.set(previous)
      }
    }
  }

  def readSpec(iterator: Iterator[_]): Option[NativeUcxShuffleReadSpec] = {
    readSpec(iterator, new IdentityHashMap[AnyRef, java.lang.Boolean](), 0)
  }

  private def readSpec(
      candidate: Any,
      seen: IdentityHashMap[AnyRef, java.lang.Boolean],
      depth: Int): Option[NativeUcxShuffleReadSpec] = {
    if (candidate == null || depth > MaxReadSpecUnwrapDepth) {
      return None
    }

    candidate match {
      case metadata: NativeUcxShuffleReadMetadata =>
        Some(metadata.nativeUcxShuffleReadSpec)
      case iterator: Iterator[_] =>
        val ref = iterator.asInstanceOf[AnyRef]
        if (seen.containsKey(ref)) {
          None
        } else {
          seen.put(ref, java.lang.Boolean.TRUE)
          iteratorFields(iterator.getClass).iterator
            .flatMap {
              field =>
                try {
                  field.setAccessible(true)
                  readSpec(field.get(ref), seen, depth + 1)
                } catch {
                  case NonFatal(_) => None
                }
            }
            .toSeq
            .headOption
        }
      case _ => None
    }
  }

  private def iteratorFields(clazz: Class[_]): Seq[java.lang.reflect.Field] = {
    val fields = scala.collection.mutable.ArrayBuffer[java.lang.reflect.Field]()
    var current = clazz
    while (current != null && current != classOf[Object]) {
      fields ++= current.getDeclaredFields.filterNot {
        field => Modifier.isStatic(field.getModifiers)
      }
      current = current.getSuperclass
    }
    fields.toSeq
  }

  def withWriterContext[T](context: NativeUcxShuffleWriterContext)(f: => T): T = {
    val previous = writerContext.get()
    writerContext.set(context)
    try {
      f
    } finally {
      if (previous == null) {
        writerContext.remove()
      } else {
        writerContext.set(previous)
      }
    }
  }

  def wrapShuffleMapTaskInput(
      handle: UcxColumnarShuffleHandle[_, _, _],
      mapId: Long,
      context: TaskContext)(
      createInput: => Iterator[_]): Iterator[_] = {
    val conf = SparkEnv.get.conf
    if (!enabled(conf)) {
      return createInput
    }

    val columnarDependency =
      handle.dependency.asInstanceOf[ColumnarShuffleDependencyLike]
    val partitioning = columnarDependency.nativePartitioning.getShortName
    val ucxMapId = context.partitionId().toLong
    val attemptId = context.taskAttemptId()
    val writerContext = NativeUcxShuffleWriterContext(
      shuffleId = handle.shuffleId,
      mapId = ucxMapId,
      attemptId = attemptId,
      nativeTaskId = writerTaskId(handle.shuffleId, ucxMapId, attemptId),
      numPartitions = handle.dependency.partitioner.numPartitions,
      partitioning = partitioning,
      startPartitionId = GlutenShuffleUtils.getStartPartitionId(
        columnarDependency.nativePartitioning,
        context.partitionId()),
      dropFirstColumn =
        partitioning == GlutenShuffleUtils.HashPartitioningShortName &&
          columnarDependency.nativePartitioning.getKeyIndices == null,
      partitionKeyIndices =
        Option(columnarDependency.nativePartitioning.getKeyIndices)
          .map(_.toSeq)
          .getOrElse(Seq.empty)
    )
    logInfo(
      s"Installing native UCX shuffle writer context before map iterator creation " +
        s"shuffleId=${handle.shuffleId} ucxMapId=$ucxMapId sparkMapId=$mapId " +
        s"attemptId=$attemptId " +
        s"nativeTaskId=${writerContext.nativeTaskId} " +
        s"partitions=${writerContext.numPartitions} " +
        s"partitioning=$partitioning " +
        s"handleClass=${handle.getClass.getName}")
    val inputIterator = withWriterContext(writerContext) {
      createInput
    }
    new Iterator[Any] {
      override def hasNext: Boolean =
        withWriterContext(writerContext) {
          inputIterator.hasNext
        }

      override def next(): Any =
        withWriterContext(writerContext) {
          inputIterator.next()
        }
    }
  }

  def writerExtraConf(context: NativeUcxShuffleWriterContext): Map[String, String] = {
    Map(
      WriteEnabledConf -> "true",
      NativeTaskIdConf -> context.nativeTaskId,
      WriteNumPartitionsConf -> context.numPartitions.toString,
      WritePartitioningConf -> context.partitioning,
      WriteStartPartitionIdConf -> context.startPartitionId.toString,
      WriteDropFirstColumnConf -> context.dropFirstColumn.toString,
      WritePartitionKeyIndicesConf -> context.partitionKeyIndices.mkString(",")
    )
  }

  def readerExtraConf(
      streamIndices: Seq[Int],
      replicatedStreamIndices: Seq[Int]): Map[String, String] = {
    if (streamIndices.isEmpty) {
      Map.empty
    } else {
      Map(
        ReadStreamIndicesConf -> streamIndices.sorted.mkString(","),
        ReplicatedReadStreamIndicesConf -> replicatedStreamIndices.sorted.mkString(",")
      )
    }
  }

  def startVeloxExchangeSplitPoller(
      spec: NativeUcxShuffleReadSpec,
      streamIdx: Int,
      addSplits: Array[String] => Unit,
      noMoreSplits: () => Unit,
      onFailure: Throwable => Unit): Thread = {
    val exchangeNodeId = NativeUcxShuffleExecution.exchangeNodeId(streamIdx)
    val conf = SparkEnv.get.conf
    val coordinator = UcxShuffleCoordinator.getOrCreate(conf, isDriver = false)
    val timeoutMs =
      conf.getLong(UcxColumnarShuffleManager.ReaderEndpointWaitMsConf, 300000L)
    val pollMs =
      math.max(1L, conf.getLong(UcxColumnarShuffleManager.ReaderEndpointPollMsConf, 10L))
    val seen = new java.util.HashSet[Long]()
    val failure = new AtomicReference[Throwable](null)
    val poller = new Thread(
      new Runnable {
        override def run(): Unit = {
          val startNs = System.nanoTime()
          var noMoreSent = false
          try {
            spec.initialEndpoints.foreach {
              endpoint =>
                if (inMapRange(spec, endpoint.mapId) && seen.add(endpoint.mapId)) {
                  addSplits(Array(remoteTaskUrl(endpoint, spec.reducePartitionId)))
                }
            }
            logInfo(
              s"Started Velox UCX ExchangeNode split poller $exchangeNodeId " +
                s"shuffleId=${spec.shuffleId} reduce=${spec.reducePartitionId} " +
                s"initialSeen=${seen.size}/${spec.expectedMaps} " +
                s"mapRange=[${spec.startMapIndex},${spec.endMapIndex})")
            while (!noMoreSent) {
              val response = coordinator.getAvailableWriters(spec.shuffleId).getOrElse {
                throw new SparkException(
                  s"UCX shuffle ${spec.shuffleId} is not registered or has been aborted")
              }
              val newUrls = response.endpoints.iterator
                .filter { case (mapId, _) => inMapRange(spec, mapId) }
                .toSeq
                .sortBy(_._1)
                .flatMap {
                  case (mapId, endpoint) =>
                    if (seen.add(mapId)) {
                      Some(remoteTaskUrl(endpoint, spec.reducePartitionId))
                    } else {
                      None
                    }
                }
              if (newUrls.nonEmpty) {
                addSplits(newUrls.toArray)
                logInfo(
                  s"Added ${newUrls.size} UCX producer split(s) to Velox ExchangeNode " +
                    s"$exchangeNodeId shuffleId=${spec.shuffleId} " +
                    s"reduce=${spec.reducePartitionId} seen=${seen.size}/${spec.expectedMaps}")
              }
              val completedMaps = completedMapCount(spec, response, seen)
              if (completedMaps >= spec.expectedMaps) {
                noMoreSplits()
                reportNativeReaderState(
                  coordinator,
                  spec,
                  seenMaps = seen.size,
                  completedMaps = completedMaps,
                  noMoreSplits = true,
                  finished = false)
                noMoreSent = true
                logInfo(
                  s"Velox UCX ExchangeNode $exchangeNodeId reached noMoreSplits " +
                    s"shuffleId=${spec.shuffleId} reduce=${spec.reducePartitionId} " +
                    s"knownProducers=${seen.size}/${spec.expectedMaps} " +
                    s"completed=$completedMaps/${spec.expectedMaps}")
              } else {
                val elapsedMs = (System.nanoTime() - startNs) / 1000000L
                if (elapsedMs > timeoutMs) {
                  throw new SparkException(
                    s"Timed out after ${timeoutMs}ms polling UCX producer endpoints for " +
                      s"Velox ExchangeNode $exchangeNodeId shuffleId=${spec.shuffleId} " +
                      s"reduce=${spec.reducePartitionId} " +
                      s"knownProducers=${seen.size}/${spec.expectedMaps} " +
                      s"completed=$completedMaps/${spec.expectedMaps}")
                }
                Thread.sleep(pollMs)
              }
            }
          } catch {
            case e: InterruptedException =>
              Thread.currentThread().interrupt()
            case NonFatal(e) =>
              if (failure.compareAndSet(null, e)) {
                onFailure(e)
              }
          }
        }
      },
      s"velox-ucx-exchange-split-poller-${spec.shuffleId}-${spec.reducePartitionId}-$streamIdx"
    )
    poller.setDaemon(true)
    poller.start()
    poller
  }

  private def reportNativeReaderState(
      coordinator: UcxShuffleCoordinator,
      spec: NativeUcxShuffleReadSpec,
      seenMaps: Int,
      completedMaps: Int,
      noMoreSplits: Boolean,
      finished: Boolean): Unit = {
    val response =
      coordinator.reportNativeReaderState(
        UcxNativeReaderState(
          shuffleId = spec.shuffleId,
          reducePartitionId = spec.reducePartitionId,
          taskAttemptId = spec.taskAttemptId,
          nativeReaderId = spec.nativeReaderId,
          seenMaps = seenMaps,
          completedMaps = completedMaps,
          expectedMaps = spec.expectedMaps,
          noMoreSplits = noMoreSplits,
          finished = finished,
          timestampMs = System.currentTimeMillis()))
    if (!response.accepted) {
      logWarning(
        s"Native UCX reader state rejected by transport coordinator " +
          s"shuffleId=${spec.shuffleId} reduce=${spec.reducePartitionId} " +
          s"taskAttemptId=${spec.taskAttemptId} groupId=${response.groupId} " +
          s"groupState=${response.groupState} " +
          s"reason=${response.reason}")
    }
  }

  private def remoteTaskUrl(endpoint: UcxShuffleEndpoint, reducePartitionId: Int): String = {
    s"http://${endpoint.host}:${endpoint.ucxPort}/v1/task/${endpoint.nativeTaskId}/results/" +
      s"$reducePartitionId"
  }

  private def inMapRange(spec: NativeUcxShuffleReadSpec, mapId: Long): Boolean = {
    if (spec.endMapIndex == Int.MaxValue) {
      mapId >= spec.startMapIndex
    } else {
      mapId >= spec.startMapIndex && mapId < spec.endMapIndex
    }
  }

  private def completedMapCount(
      spec: NativeUcxShuffleReadSpec,
      response: UcxShuffleEndpointResponse,
      seen: java.util.HashSet[Long]): Int = {
    val finishedWithoutEndpoint =
      response.finishedMapIds.count(mapId => inMapRange(spec, mapId) && !seen.contains(mapId))
    seen.size + finishedWithoutEndpoint
  }
}
