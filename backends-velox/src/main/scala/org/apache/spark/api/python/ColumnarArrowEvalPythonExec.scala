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
package org.apache.spark.api.python

import org.apache.gluten.backendsapi.arrow.ArrowBatchTypes.ArrowJavaBatchType
import org.apache.gluten.columnarbatch.ColumnarBatches
import org.apache.gluten.execution.{ValidatablePlan, ValidationResult}
import org.apache.gluten.extension.columnar.transition.{Convention, ConventionReq}
import org.apache.gluten.iterator.Iterators
import org.apache.gluten.memory.arrow.alloc.ArrowBufferAllocators
import org.apache.gluten.utils.{ArrowUtil, PullOutProjectHelper}
import org.apache.gluten.vectorized.ArrowWritableColumnVector

import org.apache.spark.{ContextAwareIterator, SparkEnv, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.execution.{ProjectExec, SparkPlan}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.python.{ArrowEvalPythonExec, BasePythonRunnerShim, EvalPythonExecBase}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.apache.spark.sql.utils.{SparkSchemaUtil, SparkVectorUtil}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}
import org.apache.spark.util.{SparkVersionUtil, Utils}

import org.apache.arrow.vector.{VectorLoader, VectorSchemaRoot, VectorUnloader}
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}

import java.io.{BufferedInputStream, BufferedOutputStream, DataInputStream, DataOutputStream, File, FileInputStream, FileOutputStream, IOException, OutputStream}
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong}

import scala.annotation.nowarn
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

private[python] trait PythonInputBatchSerializationAck {
  def serializationFinished(): Unit
}

private[python] trait ArrowUdfSerializationAckHook {
  def acknowledge(ack: () => Unit): Unit

  def outputWaitingForAck(): Unit
}

private[python] object ArrowUdfSerializationAckHookRegistry {
  @volatile private var hook: ArrowUdfSerializationAckHook = null

  def installForTesting(newHook: ArrowUdfSerializationAckHook): Unit = synchronized {
    require(hook == null, "An Arrow UDF serialization ACK test hook is already installed")
    hook = newHook
  }

  def clearForTesting(): Unit = synchronized {
    hook = null
  }

  def acknowledge(ack: () => Unit): Unit = {
    val current = hook
    if (current == null) {
      ack()
    } else {
      current.acknowledge(ack)
    }
  }

  def outputWaitingForAck(): Unit = {
    val current = hook
    if (current != null) {
      current.outputWaitingForAck()
    }
  }
}

private[python] object ArrowUdfSpillFileTracker {
  private val activeFiles = new ConcurrentHashMap[String, java.lang.Boolean]
  private val createdFiles = new AtomicLong()

  def register(file: File): Unit = {
    val previous = activeFiles.putIfAbsent(file.getAbsolutePath, java.lang.Boolean.TRUE)
    if (previous != null) {
      throw new IllegalStateException(s"Arrow UDF spill path is already active: $file")
    }
    createdFiles.incrementAndGet()
  }

  def markDeleted(file: File): Unit = {
    if (activeFiles.remove(file.getAbsolutePath) == null) {
      throw new IllegalStateException(s"Arrow UDF spill path was not active: $file")
    }
  }

  def activeFileCount: Int = activeFiles.size()

  def createdFileCount: Long = createdFiles.get()
}

private[python] class AckingPythonInputBatch(
    columns: Array[ColumnVector],
    numRows: Int,
    ack: () => Unit)
  extends ColumnarBatch(columns, numRows)
  with PythonInputBatchSerializationAck {
  private val acknowledged = new AtomicBoolean(false)

  override def serializationFinished(): Unit = {
    if (acknowledged.compareAndSet(false, true)) {
      ArrowUdfSerializationAckHookRegistry.acknowledge(ack)
    }
  }
}

class ColumnarArrowPythonRunner(
    funcs: Seq[(ChainedPythonFunctions, Long)],
    evalType: Int,
    argMetas: Array[Array[(Int, Option[String])]],
    schema: StructType,
    timeZoneId: String,
    conf: Map[String, String],
    pythonMetrics: Map[String, SQLMetric])
  extends BasePythonRunnerShim(funcs, evalType, argMetas, pythonMetrics) {

  override val simplifiedTraceback: Boolean = SQLConf.get.pysparkSimplifiedTraceback

  override val bufferSize: Int = SQLConf.get.pandasUDFBufferSize
  require(
    bufferSize >= 4,
    "Pandas execution requires more than 4 bytes. Please set higher buffer. " +
      s"Please change '${SQLConf.PANDAS_UDF_BUFFER_SIZE.key}'.")

  protected def newReaderIterator(
      stream: DataInputStream,
      writer: Writer,
      startTime: Long,
      env: SparkEnv,
      worker: PythonWorker,
      pid: scala.Option[scala.Int],
      releasedOrClosed: AtomicBoolean,
      context: TaskContext): Iterator[ColumnarBatch] = {

    new ReaderIterator(stream, writer, startTime, env, worker, pid, releasedOrClosed, context) {
      private val allocator = ArrowBufferAllocators.contextInstance()

      private var reader: ArrowStreamReader = _
      private var root: VectorSchemaRoot = _

      context.addTaskCompletionListener[Unit] {
        _ =>
          if (reader != null) {
            reader.close(false)
          }
      }

      private var batchLoaded = true

      override protected def read(): ColumnarBatch = {
        if (writer.exception.isDefined) {
          throw writer.exception.get
        }
        try {
          if (reader != null && batchLoaded) {
            batchLoaded = reader.loadNextBatch()
            if (batchLoaded) {
              // The reader reuses its root for the next Python record batch. Materialize every
              // output into an independent root so downstream owns exactly one vector reference.
              val recordBatch = new VectorUnloader(root).getRecordBatch
              ArrowUtil.loadBatch(allocator, recordBatch, root.getSchema)
            } else {
              reader.close(false)
              reader = null
              root = null
              // Reach end of stream. Call `read()` again to read control data.
              read()
            }
          } else {
            stream.readInt() match {
              case SpecialLengths.START_ARROW_STREAM =>
                reader = new ArrowStreamReader(stream, allocator)
                root = reader.getVectorSchemaRoot()
                read()
              case SpecialLengths.TIMING_DATA =>
                handleTimingData()
                read()
              case SpecialLengths.PYTHON_EXCEPTION_THROWN =>
                throw handlePythonException()
              case SpecialLengths.END_OF_DATA_SECTION =>
                handleEndOfDataSection()
                null
            }
          }
        } catch handleException
      }
    }
  }

  override def createNewWriter(
      env: SparkEnv,
      worker: PythonWorker,
      inputIterator: Iterator[ColumnarBatch],
      partitionIndex: Int,
      context: TaskContext): Writer = {
    new Writer(env, worker, inputIterator, partitionIndex, context) {
      override protected def writeCommand(dataOut: DataOutputStream): Unit = {
        // Write config for the worker as a number of key -> value pairs of strings
        dataOut.writeInt(conf.size)
        for ((k, v) <- conf) {
          PythonRDD.writeUTF(k, dataOut)
          PythonRDD.writeUTF(v, dataOut)
        }
        ColumnarArrowPythonRunner.this.writeUdf(dataOut, argMetas)
      }

      // For Spark earlier than 4.0. It overrides the corresponding abstract method
      // in Writer class. We omitted the override keyword for compatibility consideration.
      def writeIteratorToStream(dataOut: DataOutputStream): Unit = {
        writeToStreamHelper(dataOut)
      }

      // For Spark 4.0. It overrides the corresponding abstract method in Writer class.
      // We omitted the override keyword for compatibility consideration.
      def writeNextInputToStream(dataOut: DataOutputStream): Boolean = {
        writeToStreamHelper(dataOut)
      }

      def writeToStreamHelper(dataOut: DataOutputStream): Boolean = {
        if (!inputIterator.hasNext) {
          // See https://issues.apache.org/jira/browse/SPARK-44705:
          // Starting from Spark 4.0, we should return false once the iterator is drained out,
          // otherwise Spark won't stop calling this method repeatedly.
          return false
        }
        var numRows: Long = 0
        val arrowSchema = SparkSchemaUtil.toArrowSchema(schema, timeZoneId)
        val allocator = ArrowBufferAllocators.contextInstance()
        val root = VectorSchemaRoot.create(arrowSchema, allocator)

        Utils.tryWithSafeFinally {
          val loader = new VectorLoader(root)
          val writer = new ArrowStreamWriter(root, null, dataOut)
          writer.start()
          while (inputIterator.hasNext) {
            val nextBatch = inputIterator.next()
            try {
              numRows += nextBatch.numRows

              val cols = (0 until nextBatch.numCols).toList.map(
                i =>
                  nextBatch
                    .column(i)
                    .asInstanceOf[ArrowWritableColumnVector]
                    .getValueVector)
              val nextRecordBatch =
                SparkVectorUtil.toArrowRecordBatch(nextBatch.numRows, cols)
              try {
                loader.load(nextRecordBatch)
                writer.writeBatch()
              } finally {
                nextRecordBatch.close()
              }
            } finally {
              nextBatch match {
                case ack: PythonInputBatchSerializationAck => ack.serializationFinished()
                case _ =>
              }
            }
          }
          // end writes footer to the output stream and doesn't clean any resources.
          // It could throw exception if the output stream is closed, so it should be
          // in the try block.
          writer.end()
          true
        } {
          root.close()
          // allocator can't close now or the data will loss
          // allocator.close()
        }
      }
    }
  }

}

case class ColumnarArrowEvalPythonExec(
    udfs: Seq[PythonUDF],
    resultAttrs: Seq[Attribute],
    child: SparkPlan,
    evalType: Int)
  extends EvalPythonExecBase
  with ValidatablePlan {

  override def batchType(): Convention.BatchType = ArrowJavaBatchType

  override def rowType0(): Convention.RowType = Convention.RowType.None

  override protected def doValidateInternal(): ValidationResult = {
    val (_, inputs) = udfs.map(ColumnarArrowEvalPythonExec.collectFunctions).unzip
    inputs.foreach {
      input =>
        input.foreach {
          case e: AttributeReference if child.output.exists(_.exprId == e.exprId) =>
          // Valid case, continue validation
          case _: AttributeReference =>
            return ValidationResult.failed("Expression Id does not exist for AttributeReference")
          case _ =>
            return ValidationResult.failed("UDF input is not an instance of AttributeReference")
        }
    }
    super.doValidateInternal()
  }

  override def requiredChildConvention(): Seq[ConventionReq] = List(
    ConventionReq.ofBatch(ConventionReq.BatchType.Is(ArrowJavaBatchType)))

  override lazy val metrics = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "output_batches"),
    "numInputRows" -> SQLMetrics.createMetric(sparkContext, "number of input rows"),
    "processTime" -> SQLMetrics.createTimingMetric(sparkContext, "totaltime_arrow_udf"),
    "spillBytes" -> SQLMetrics.createSizeMetric(sparkContext, "Arrow UDF input spill bytes"),
    "spillFiles" -> SQLMetrics.createMetric(sparkContext, "Arrow UDF input spill files")
  )

  private val sessionLocalTimeZone = conf.sessionLocalTimeZone

  private def getPythonRunnerConfMap(conf: SQLConf): Map[String, String] = {
    val timeZoneConf = Seq(SQLConf.SESSION_LOCAL_TIMEZONE.key -> conf.sessionLocalTimeZone)
    val pandasColsByName = Seq(
      SQLConf.PANDAS_GROUPED_MAP_ASSIGN_COLUMNS_BY_NAME.key ->
        conf.pandasGroupedMapAssignColumnsByName.toString)
    val arrowSafeTypeCheck = Seq(
      SQLConf.PANDAS_ARROW_SAFE_TYPE_CONVERSION.key ->
        conf.arrowSafeTypeConversion.toString)
    Map(timeZoneConf.toSeq ++ pandasColsByName.toSeq ++ arrowSafeTypeCheck: _*)
  }

  private val pythonRunnerConf = getPythonRunnerConfMap(conf)

  protected def evaluateColumnar(
      funcs: Seq[(ChainedPythonFunctions, Long)],
      argMetas: Array[Array[(Int, Option[String])]],
      iter: Iterator[ColumnarBatch],
      schema: StructType,
      context: TaskContext): Iterator[ColumnarBatch] = {

    val outputTypes = output.drop(child.output.length).map(_.dataType)

    val columnarBatchIter = new ColumnarArrowPythonRunner(
      funcs,
      evalType,
      argMetas,
      schema,
      sessionLocalTimeZone,
      pythonRunnerConf,
      Map()).compute(iter, context.partitionId(), context)

    columnarBatchIter.map {
      batch =>
        try {
          val actualDataTypes = (0 until batch.numCols()).map(i => batch.column(i).dataType())
          assert(
            outputTypes == actualDataTypes,
            "Invalid schema from arrow_udf: " +
              s"expected ${outputTypes.mkString(", ")}, got ${actualDataTypes.mkString(", ")}")
          batch
        } catch {
          case t: Throwable =>
            try {
              batch.close()
            } catch {
              case cleanupFailure: Throwable => t.addSuppressed(cleanupFailure)
            }
            throw t
        }
    }
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val numOutputRows = longMetric("numOutputRows")
    val numOutputBatches = longMetric("numOutputBatches")
    val numInputRows = longMetric("numInputRows")
    val procTime = longMetric("processTime")
    val spillBytes = longMetric("spillBytes")
    val spillFiles = longMetric("spillFiles")
    val inputRDD = child.executeColumnar()
    inputRDD.mapPartitions {
      iter =>
        val context = TaskContext.get()
        val maxPendingInputBatches = conf
          .getConfString("spark.gluten.sql.columnar.arrowUdf.maxPendingInputBatches", "2")
          .toInt
        require(
          maxPendingInputBatches > 0,
          "spark.gluten.sql.columnar.arrowUdf.maxPendingInputBatches must be positive")
        val maxSpillBytesPerTask = conf
          .getConfString(
            "spark.gluten.sql.columnar.arrowUdf.maxSpillBytesPerTask",
            (8L * 1024 * 1024 * 1024).toString)
          .toLong
        require(
          maxSpillBytesPerTask > 0,
          "spark.gluten.sql.columnar.arrowUdf.maxSpillBytesPerTask must be positive")
        val serializationAckTimeoutMillis = conf
          .getConfString("spark.gluten.sql.columnar.arrowUdf.serializationAckTimeoutMs", "30000")
          .toLong
        require(
          serializationAckTimeoutMillis > 0,
          "spark.gluten.sql.columnar.arrowUdf.serializationAckTimeoutMs must be positive")
        val pendingInputs = new ColumnarArrowEvalPythonExec.SpillablePendingInputQueue(
          maxPendingInputBatches,
          maxSpillBytesPerTask,
          child.schema,
          sessionLocalTimeZone,
          context,
          serializationAckTimeoutMillis,
          spillBytes,
          spillFiles)
        context.addTaskCompletionListener[Unit](_ => pendingInputs.close())
        val (pyFuncs, inputs) = udfs.map(ColumnarArrowEvalPythonExec.collectFunctions).unzip
        // We only write the referred cols by UDFs to python worker. So we need
        // get corresponding offsets
        val allInputs = new ArrayBuffer[Expression]
        val dataTypes = new ArrayBuffer[DataType]
        val originalOffsets = new ArrayBuffer[Int]
        val argMetas: Array[Array[(Int, Option[String])]] = if (SparkVersionUtil.gteSpark40) {
          // Spark 4.0 requires ArgumentMetadata rather than trivial integer-based offset.
          // See https://issues.apache.org/jira/browse/SPARK-44918.
          inputs.map {
            input =>
              input.map {
                e =>
                  val (key, value) = e match {
                    case EvalPythonExecBase.NamedArgumentExpressionShim(key, value) =>
                      (Some(key), value)
                    case _ =>
                      (None, e)
                  }
                  val pair: (Int, Option[String]) = if (allInputs.exists(_.semanticEquals(value))) {
                    allInputs.indexWhere(_.semanticEquals(value)) -> key
                  } else {
                    val offset = child.output.indexWhere(
                      _.exprId.equals(e.asInstanceOf[AttributeReference].exprId))
                    originalOffsets += offset
                    allInputs += value
                    dataTypes += value.dataType
                    (allInputs.length - 1) -> key
                  }
                  pair
              }.toArray
          }.toArray
        } else {
          inputs.map {
            input =>
              input.map {
                e =>
                  val pair: (Int, Option[String]) = if (allInputs.exists(_.semanticEquals(e))) {
                    allInputs.indexWhere(_.semanticEquals(e)) -> None
                  } else {
                    val offset = child.output.indexWhere(
                      _.exprId.equals(e.asInstanceOf[AttributeReference].exprId))
                    originalOffsets += offset
                    allInputs += e
                    dataTypes += e.dataType
                    (allInputs.length - 1) -> None
                  }
                  pair
              }.toArray
          }.toArray
        }
        val schema = StructType(dataTypes.zipWithIndex.map {
          case (dt, i) =>
            StructField(s"_$i", dt)
        }.toSeq)

        @nowarn val contextAwareIterator = new ContextAwareIterator(context, iter)
        val inputBatchIter = new Iterator[ColumnarBatch] {
          override def hasNext: Boolean = {
            val hasNext = contextAwareIterator.hasNext
            if (!hasNext) {
              pendingInputs.markProducerComplete()
            }
            hasNext
          }

          override def next(): ColumnarBatch = {
            val inputCb = contextAwareIterator.next()
            ColumnarBatches.checkLoaded(inputCb)
            // Retain and enqueue before handing the projection to the asynchronous Python writer.
            // The spillable FIFO owns this retained reference until it is either spilled or
            // transferred to the merged output batch. The Python writer cannot advance this child
            // iterator again until it has serialized the returned projection.
            val serializationFinished = pendingInputs.enqueue(inputCb)
            numInputRows += inputCb.numRows
            val colsForEval = new ArrayBuffer[ColumnVector]()
            for (i <- originalOffsets) {
              colsForEval += inputCb.column(i)
            }
            new AckingPythonInputBatch(
              colsForEval.toArray,
              inputCb.numRows(),
              serializationFinished)
          }
        }

        val outputColumnarBatchIterator =
          evaluateColumnar(pyFuncs, argMetas, inputBatchIter, schema, context)
        val res = new Iterator[ColumnarBatch] {
          private var outputComplete = false

          override def hasNext: Boolean = {
            val hasNext = outputColumnarBatchIterator.hasNext
            if (!hasNext && !outputComplete) {
              outputComplete = true
              pendingInputs.verifyCompleteAndDrained()
            }
            hasNext
          }

          override def next(): ColumnarBatch = {
            val outputCb = outputColumnarBatchIterator.next()
            val pending =
              try {
                pendingInputs.dequeue(outputCb.numRows())
              } catch {
                case t: Throwable =>
                  outputCb.close()
                  throw t
              }
            val inputCb = pending.batch
            try {
              val joinedVectors = (0 until inputCb.numCols).toArray.map(
                i => inputCb.column(i)) ++ (0 until outputCb.numCols).toArray.map(
                i => outputCb.column(i))
              // Transfer the queue/materialized input ownership and the independently materialized
              // Python output ownership into the joined batch. Do not close the old containers:
              // ColumnarBatch itself has no separate ownership beyond its column vectors.
              val numRows = inputCb.numRows
              numOutputBatches += 1
              numOutputRows += numRows
              val batch = new ColumnarBatch(joinedVectors, numRows)
              ColumnarBatches.checkLoaded(batch)
              procTime += (System.nanoTime() - pending.enqueuedAtNanos) / 1000000
              batch
            } catch {
              case t: Throwable =>
                ColumnarBatches.release(inputCb)
                outputCb.close()
                throw t
            }
          }
        }
        Iterators
          .wrap(res)
          .recycleIterator {
            pendingInputs.close()
          }
          .recyclePayload(_.close())
          .create()
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): ColumnarArrowEvalPythonExec =
    copy(udfs, resultAttrs, newChild)
}

object ColumnarArrowEvalPythonExec {

  private val SpillLimitConf = "spark.gluten.sql.columnar.arrowUdf.maxSpillBytesPerTask"

  sealed private trait PendingInputEntry {
    def numRows: Int
    def enqueuedAtNanos: Long
  }

  private class RetainedPendingInput(
      val batch: ColumnarBatch,
      val numRows: Int,
      val enqueuedAtNanos: Long)
    extends PendingInputEntry {
    var serialized = false
    var detached = false
    var released = false
  }

  private case class SpilledPendingInput(file: File, numRows: Int, enqueuedAtNanos: Long)
    extends PendingInputEntry

  private case class PendingInput(batch: ColumnarBatch, enqueuedAtNanos: Long)

  /**
   * The Python runner writes input on a background thread while its result iterator is consumed on
   * the task thread. Keep at most `maxRetainedBatches` retained inputs in a synchronized FIFO. When
   * that bound is reached, replace the oldest retained entry in place with a task-local Arrow IPC
   * file before retaining the current input. That oldest entry has already been completely written
   * to Python: the Python writer only requests a later input after `writeBatch` returns.
   *
   * Transfer one FIFO entry only when Python returns a batch with exactly the same row count.
   * Scalar Arrow UDFs are required to preserve record-batch row counts; an unexpected split or
   * coalesce therefore fails explicitly instead of silently joining result columns to wrong rows.
   */
  private class SpillablePendingInputQueue(
      maxRetainedBatches: Int,
      maxSpillBytesPerTask: Long,
      inputSchema: StructType,
      timeZoneId: String,
      context: TaskContext,
      serializationAckTimeoutMillis: Long,
      spillBytesMetric: SQLMetric,
      spillFilesMetric: SQLMetric)
    extends AutoCloseable {
    private val entries = new ArrayBuffer[PendingInputEntry]
    private val arrowSchema = SparkSchemaUtil.toArrowSchema(inputSchema, timeZoneId)
    private var headIndex = 0
    private var retainedCount = 0
    private var spilledBytes = 0L
    private var closed = false
    private var producerComplete = false

    def enqueue(batch: ColumnarBatch): () => Unit = synchronized {
      try {
        checkTaskActive()
        if (retainedCount == maxRetainedBatches) {
          spillOldestRetainedInput()
          checkTaskActive()
        }
        ColumnarBatches.retain(batch)
        val retained = new RetainedPendingInput(batch, batch.numRows(), System.nanoTime())
        var appended = false
        try {
          entries += retained
          retainedCount += 1
          appended = true
        } finally {
          if (!appended) {
            ColumnarBatches.release(batch)
          }
        }
        () => markSerializationFinished(retained)
      } catch {
        case t: Throwable =>
          closeAfterFailure(t)
          throw t
      }
    }

    def markProducerComplete(): Unit = synchronized {
      producerComplete = true
    }

    def dequeue(outputRows: Int): PendingInput = synchronized {
      try {
        if (headIndex == entries.length) {
          throw new IllegalStateException(
            s"Columnar Arrow Python returned an output batch with $outputRows rows " +
              "without a corresponding input batch")
        }
        val entry = entries(headIndex)
        if (entry.numRows != outputRows) {
          removeHead()
          cleanupEntry(entry)
          throw new IllegalStateException(
            s"Columnar Arrow Python changed record-batch row count: " +
              s"input=${entry.numRows} output=$outputRows")
        }
        entry match {
          case retained: RetainedPendingInput =>
            awaitSerializationFinished(retained)
            removeHead()
            retainedCount -= 1
            PendingInput(retained.batch, retained.enqueuedAtNanos)
          case spilled: SpilledPendingInput =>
            removeHead()
            PendingInput(loadSpilledInput(spilled), spilled.enqueuedAtNanos)
        }
      } catch {
        case t: Throwable =>
          closeAfterFailure(t)
          throw t
      }
    }

    def verifyCompleteAndDrained(): Unit = synchronized {
      try {
        if (!producerComplete) {
          throw new IllegalStateException(
            "Columnar Arrow Python output ended before the input writer consumed its partition")
        }
        if (headIndex != entries.length) {
          throw new IllegalStateException(
            s"Columnar Arrow Python output ended with " +
              s"${entries.length - headIndex} unmatched input batch(es)")
        }
      } catch {
        case t: Throwable =>
          closeAfterFailure(t)
          throw t
      }
    }

    override def close(): Unit = synchronized {
      closeInternal()
    }

    private def spillOldestRetainedInput(): Unit = {
      var index = headIndex
      while (
        index < entries.length &&
        !entries(index).isInstanceOf[RetainedPendingInput]
      ) {
        index += 1
      }
      if (index == entries.length) {
        throw new IllegalStateException(
          "Columnar Arrow Python retained-input accounting lost its FIFO entry")
      }
      val retained = entries(index).asInstanceOf[RetainedPendingInput]
      if (!retained.serialized) {
        throw new IllegalStateException(
          "Columnar Arrow Python tried to spill input before its writer serialization ACK")
      }
      // Do not mutate the FIFO or release its retained vectors until the file is complete.
      val spilled = writeSpillFile(retained)
      entries(index) = spilled
      retainedCount -= 1
      releaseRetained(retained)
    }

    private def writeSpillFile(retained: RetainedPendingInput): SpilledPendingInput = {
      checkTaskActive()
      val file = SparkEnv.get.blockManager.diskBlockManager.createTempLocalBlock()._2
      var registered = false
      var success = false
      try {
        ArrowUdfSpillFileTracker.register(file)
        registered = true
        val output = new SpillLimitOutputStream(
          new BufferedOutputStream(new FileOutputStream(file)),
          spilledBytes,
          maxSpillBytesPerTask,
          context)
        Utils.tryWithSafeFinally {
          val root = VectorSchemaRoot.create(arrowSchema, ArrowBufferAllocators.contextInstance())
          Utils.tryWithSafeFinally {
            val loader = new VectorLoader(root)
            val writer = new ArrowStreamWriter(root, null, output)
            writer.start()
            val cols = (0 until retained.batch.numCols()).toList.map(
              i =>
                retained.batch
                  .column(i)
                  .asInstanceOf[ArrowWritableColumnVector]
                  .getValueVector)
            val recordBatch = SparkVectorUtil.toArrowRecordBatch(retained.numRows, cols)
            try {
              loader.load(recordBatch)
              writer.writeBatch()
            } finally {
              recordBatch.close()
            }
            writer.end()
          } {
            root.close()
          }
        } {
          output.close()
        }
        checkTaskActive()
        val bytes = output.bytesWritten
        spilledBytes += bytes
        spillBytesMetric += bytes
        spillFilesMetric += 1
        success = true
        SpilledPendingInput(file, retained.numRows, retained.enqueuedAtNanos)
      } finally {
        if (!success && registered) {
          deleteSpillFile(file)
        }
      }
    }

    /**
     * Materialize vectors into a root independent of the Arrow stream root. All stream resources
     * are closed and the spill file is deleted before this method returns the batch.
     */
    private def loadSpilledInput(spilled: SpilledPendingInput): ColumnarBatch = {
      var input: BufferedInputStream = null
      var reader: ArrowStreamReader = null
      var root: VectorSchemaRoot = null
      var batch: ColumnarBatch = null
      var bodyFailure: Throwable = null
      try {
        input = new BufferedInputStream(new FileInputStream(spilled.file))
        reader = new ArrowStreamReader(input, ArrowBufferAllocators.contextInstance())
        if (!reader.loadNextBatch()) {
          throw new IOException(s"Arrow UDF spill file is empty: ${spilled.file}")
        }
        root = reader.getVectorSchemaRoot()
        if (root.getRowCount != spilled.numRows) {
          throw new IOException(
            s"Arrow UDF spill row count changed: expected=${spilled.numRows} " +
              s"actual=${root.getRowCount}")
        }
        val recordBatch = new VectorUnloader(root).getRecordBatch
        batch =
          ArrowUtil.loadBatch(ArrowBufferAllocators.contextInstance(), recordBatch, root.getSchema)
        if (reader.loadNextBatch()) {
          throw new IOException(s"Arrow UDF spill file has more than one batch: ${spilled.file}")
        }
      } catch {
        case t: Throwable => bodyFailure = t
      }

      val cleanupFailure = cleanupSpillReader(reader, input, spilled.file)
      if (bodyFailure != null) {
        if (batch != null) {
          batch.close()
        }
        if (cleanupFailure != null) {
          bodyFailure.addSuppressed(cleanupFailure)
        }
        throw bodyFailure
      }
      if (cleanupFailure != null) {
        if (batch != null) {
          batch.close()
        }
        throw cleanupFailure
      }
      batch
    }

    private def cleanupSpillReader(
        reader: ArrowStreamReader,
        input: BufferedInputStream,
        file: File): Throwable = {
      var failure: Throwable = null
      failure = runCleanup(failure, if (reader != null) reader.close(false))
      failure = runCleanup(failure, if (input != null) input.close())
      runCleanup(failure, deleteSpillFile(file))
    }

    private def removeHead(): PendingInputEntry = {
      val entry = entries(headIndex)
      headIndex += 1
      if (headIndex > 64 && headIndex * 2 >= entries.length) {
        entries.remove(0, headIndex)
        headIndex = 0
      }
      entry
    }

    private def cleanupEntry(entry: PendingInputEntry): Unit = entry match {
      case retained: RetainedPendingInput =>
        retainedCount -= 1
        if (retained.serialized) {
          releaseRetained(retained)
        } else {
          // The writer still shares these vectors through its projected input batch. Remove the
          // entry from the FIFO but let its serialization ACK perform the final release.
          retained.detached = true
        }
      case spilled: SpilledPendingInput => deleteSpillFile(spilled.file)
    }

    private def markSerializationFinished(retained: RetainedPendingInput): Unit = {
      synchronized {
        if (!retained.serialized) {
          retained.serialized = true
          notifyAll()
          if (retained.detached) {
            releaseRetained(retained)
          }
        }
      }
    }

    private def awaitSerializationFinished(retained: RetainedPendingInput): Unit = {
      if (retained.serialized) {
        return
      }
      ArrowUdfSerializationAckHookRegistry.outputWaitingForAck()
      val deadlineNanos =
        System.nanoTime() + serializationAckTimeoutMillis * 1000L * 1000L
      while (!retained.serialized) {
        checkTaskActive()
        val remainingNanos = deadlineNanos - System.nanoTime()
        if (remainingNanos <= 0) {
          throw new IOException(
            "Timed out waiting for the Columnar Arrow Python input serialization ACK after " +
              s"$serializationAckTimeoutMillis ms")
        }
        val waitMillis = math.max(1L, math.min(50L, remainingNanos / (1000L * 1000L)))
        try {
          wait(waitMillis)
        } catch {
          case interrupted: InterruptedException =>
            Thread.currentThread().interrupt()
            throw new IllegalStateException(
              "Interrupted while waiting for the Columnar Arrow Python input serialization ACK",
              interrupted)
        }
      }
    }

    private def releaseRetained(retained: RetainedPendingInput): Unit = {
      if (!retained.released) {
        retained.released = true
        ColumnarBatches.release(retained.batch)
      }
    }

    private def closeAfterFailure(primary: Throwable): Unit = {
      try {
        closeInternal()
      } catch {
        case cleanupFailure: Throwable => primary.addSuppressed(cleanupFailure)
      }
    }

    private def closeInternal(): Unit = {
      if (closed) {
        return
      }
      closed = true
      var failure: Throwable = null
      while (headIndex < entries.length) {
        val entry = removeHead()
        try {
          cleanupEntry(entry)
        } catch {
          case t: Throwable => failure = appendFailure(failure, t)
        }
      }
      entries.clear()
      headIndex = 0
      retainedCount = 0
      if (failure != null) {
        throw failure
      }
    }

    private def deleteSpillFile(file: File): Unit = {
      var failure: Throwable = null
      try {
        if (file.exists()) {
          Utils.deleteRecursively(file)
        }
      } catch {
        case t: Throwable => failure = t
      }
      if (file.exists()) {
        failure =
          appendFailure(failure, new IOException(s"Failed to delete Arrow UDF spill file: $file"))
      } else {
        try {
          ArrowUdfSpillFileTracker.markDeleted(file)
        } catch {
          case t: Throwable => failure = appendFailure(failure, t)
        }
      }
      if (failure != null) {
        throw failure
      }
    }

    private def runCleanup(current: Throwable, cleanup: => Unit): Throwable = {
      try {
        cleanup
        current
      } catch {
        case t: Throwable => appendFailure(current, t)
      }
    }

    private def appendFailure(current: Throwable, next: Throwable): Throwable = {
      if (current == null) {
        next
      } else {
        current.addSuppressed(next)
        current
      }
    }

    private def checkTaskActive(): Unit = {
      if (closed || context.isInterrupted() || Thread.currentThread().isInterrupted) {
        throw new IllegalStateException(
          "Columnar Arrow Python input FIFO closed or task interrupted while its writer was active")
      }
    }
  }

  private class SpillLimitOutputStream(
      out: OutputStream,
      alreadySpilledBytes: Long,
      maxSpillBytesPerTask: Long,
      context: TaskContext)
    extends OutputStream {
    private var count = 0L

    def bytesWritten: Long = count

    override def write(value: Int): Unit = {
      ensureCapacity(1)
      out.write(value)
      count += 1
    }

    override def write(bytes: Array[Byte], offset: Int, length: Int): Unit = {
      if (length < 0) {
        throw new IndexOutOfBoundsException(s"negative write length: $length")
      }
      ensureCapacity(length.toLong)
      out.write(bytes, offset, length)
      count += length
    }

    override def flush(): Unit = out.flush()

    override def close(): Unit = out.close()

    private def ensureCapacity(additionalBytes: Long): Unit = {
      if (context.isInterrupted()) {
        throw new IOException("Columnar Arrow Python input spill interrupted")
      }
      val remaining = maxSpillBytesPerTask - alreadySpilledBytes - count
      if (additionalBytes > remaining) {
        throw new IOException(
          s"Columnar Arrow Python input spill exceeded $SpillLimitConf=" +
            s"$maxSpillBytesPerTask bytes for one task")
      }
    }
  }

  def collectFunctions(udf: PythonUDF): ((ChainedPythonFunctions, Long), Seq[Expression]) = {
    udf.children match {
      case Seq(u: PythonUDF) =>
        val ((chained, _), children) = collectFunctions(u)
        ((ChainedPythonFunctions(chained.funcs ++ Seq(udf.func)), udf.resultId.id), children)
      case children =>
        // There should be no PythonUDF, or the children can't be evaluated directly.
        assert(!children.exists(_.isInstanceOf[PythonUDF]))
        ((ChainedPythonFunctions(Seq(udf.func)), udf.resultId.id), udf.children)
    }
  }
}

object PullOutArrowEvalPythonPreProjectHelper extends PullOutProjectHelper {

  private def rewriteUDF(
      udf: PythonUDF,
      expressionMap: mutable.HashMap[Expression, NamedExpression]): PythonUDF = {
    udf.children match {
      case Seq(u: PythonUDF) =>
        udf
          .withNewChildren(udf.children.toIndexedSeq.map {
            func => rewriteUDF(func.asInstanceOf[PythonUDF], expressionMap)
          })
          .asInstanceOf[PythonUDF]
      case children =>
        val newUDFChildren = udf.children.map {
          case literal: Literal => literal
          case other => replaceExpressionWithAttribute(other, expressionMap)
        }
        udf.withNewChildren(newUDFChildren).asInstanceOf[PythonUDF]
    }
  }

  def pullOutPreProject(arrowEvalPythonExec: ArrowEvalPythonExec): SparkPlan = {
    // pull out preproject
    val (_, inputs) =
      arrowEvalPythonExec.udfs.map(ColumnarArrowEvalPythonExec.collectFunctions).unzip
    val expressionMap = new mutable.HashMap[Expression, NamedExpression]()
    // flatten all the arguments
    val allInputs = new ArrayBuffer[Expression]
    for (input <- inputs) {
      input.foreach {
        e =>
          if (!allInputs.exists(_.semanticEquals(e))) {
            allInputs += e
            replaceExpressionWithAttribute(e, expressionMap)
          }
      }
    }
    if (!expressionMap.isEmpty) {
      // Need preproject.
      val preProject = ProjectExec(
        eliminateProjectList(arrowEvalPythonExec.child.outputSet, expressionMap.values.toSeq),
        arrowEvalPythonExec.child)
      val newUDFs = arrowEvalPythonExec.udfs.map(f => rewriteUDF(f, expressionMap))
      val newArrowEvalPythonExec = arrowEvalPythonExec.copy(udfs = newUDFs, child = preProject)
      newArrowEvalPythonExec.copyTagsFrom(arrowEvalPythonExec)
      newArrowEvalPythonExec
    } else {
      arrowEvalPythonExec
    }
  }
}
