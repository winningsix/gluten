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
package org.apache.gluten.execution.python

import org.apache.gluten.execution.WholeStageTransformerSuite
import org.apache.gluten.vectorized.ArrowWritableColumnVector

import org.apache.spark.SparkConf
import org.apache.spark.api.python.{ArrowUdfSerializationAckTestSupport, ColumnarArrowEvalPythonExec}
import org.apache.spark.sql.{DataFrame, IntegratedUDFTestUtils}
import org.apache.spark.sql.execution.python.{BatchEvalPythonExec, EvalPythonExecTransformer, UserDefinedPythonFunction}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{DataType, LongType, StringType}
import org.apache.spark.util.SparkVersionUtil

import org.scalatest.time.{Seconds, Span}

import java.io.File
import java.nio.file.Files

import scala.sys.process.Process

class ArrowEvalPythonExecSuite extends WholeStageTransformerSuite {

  import IntegratedUDFTestUtils._
  import testImplicits._

  override protected val resourcePath: String = "/tpch-data-parquet"
  override protected val fileFormat: String = "parquet"
  private val pyarrowTestUDFString =
    newTestScalarPandasUDF(name = "pyarrowUDF", returnType = Some(StringType))
  private val pyarrowTestUDFLong =
    newTestScalarPandasUDF(name = "pyarrowUDF", returnType = Some(LongType))
  private val ordinaryPythonUDFString =
    newTestPythonUDF(name = "ordinaryPythonUDF", returnType = Some(StringType))

  override def sparkConf: SparkConf = {
    super.sparkConf
      .set("spark.sql.shuffle.partitions", "1")
      .set("spark.default.parallelism", "1")
      .set("spark.executor.cores", "1")
  }

  // TODO: fix on spark-4.1
  testWithMaxSparkVersion("arrow_udf test: without projection", "4.0") {
    lazy val base =
      Seq(("1", 1), ("1", 2), ("2", 1), ("2", 2), ("3", 1), ("3", 2), ("0", 1), ("3", 0))
        .toDF("a", "b")
    lazy val expected = Seq(
      ("1", "1"),
      ("1", "1"),
      ("2", "2"),
      ("2", "2"),
      ("3", "3"),
      ("3", "3"),
      ("0", "0"),
      ("3", "3")
    ).toDF("a", "p_a")

    val df2 = base.select("a").withColumn("p_a", pyarrowTestUDFString(base("a")))
    checkSparkPlan[ColumnarArrowEvalPythonExec](df2)
    checkAnswer(df2, expected)
  }

  // TODO: fix on spark-4.1
  testWithMaxSparkVersion("arrow_udf test: with unrelated projection", "4.0") {
    lazy val base =
      Seq(("1", 1), ("1", 2), ("2", 1), ("2", 2), ("3", 1), ("3", 2), ("0", 1), ("3", 0))
        .toDF("a", "b")
    lazy val expected = Seq(
      ("1", 1, "1", 2),
      ("1", 2, "1", 4),
      ("2", 1, "2", 2),
      ("2", 2, "2", 4),
      ("3", 1, "3", 2),
      ("3", 2, "3", 4),
      ("0", 1, "0", 2),
      ("3", 0, "3", 0)
    ).toDF("a", "b", "p_a", "d_b")

    val df =
      base.withColumn("p_a", pyarrowTestUDFString(base("a"))).withColumn("d_b", base("b") * 2)
    checkSparkPlan[ColumnarArrowEvalPythonExec](df)
    checkAnswer(df, expected)
  }

  // TODO: fix on spark-4.1
  testWithMaxSparkVersion("arrow_udf test: with preprojection", "4.0") {
    lazy val base =
      Seq(("1", 1), ("1", 2), ("2", 1), ("2", 2), ("3", 1), ("3", 2), ("0", 1), ("3", 0))
        .toDF("a", "b")
    lazy val expected = Seq(
      ("1", 1, 2, "1", 2),
      ("1", 2, 4, "1", 4),
      ("2", 1, 2, "2", 2),
      ("2", 2, 4, "2", 4),
      ("3", 1, 2, "3", 2),
      ("3", 2, 4, "3", 4),
      ("0", 1, 2, "0", 2),
      ("3", 0, 0, "3", 0)
    ).toDF("a", "b", "d_b", "p_a", "p_b")
    val df = base
      .withColumn("d_b", base("b") * 2)
      .withColumn("p_a", pyarrowTestUDFString(base("a")))
      .withColumn("p_b", pyarrowTestUDFLong(base("b") * 2))
    checkAnswer(df, expected)
  }

  testWithMaxSparkVersion(
    "ordinary scalar UDF with non-null input is normalized to Arrow by engine config",
    "4.0") {
    assume(SparkVersionUtil.gteSpark35, "ordinary scalar Arrow UDF eval type requires Spark 3.5+")
    val rowReference = withSQLConf(
      "spark.sql.execution.pythonUDF.arrow.enabled" -> "false",
      "spark.gluten.mpp.enabled" -> "false") {
      val base = spark.range(0, 28, 1, 3).coalesce(1)
      base
        .select(base("id"), ordinaryPythonUDFString(base("id")).as("decoded"))
        .collect()
        .toSeq
    }
    val vectorStateBefore = ArrowWritableColumnVector.stat()
    val activeSpillFilesBefore = activeArrowUdfSpillFiles
    val createdSpillFilesBefore = createdArrowUdfSpillFiles

    withSQLConf(
      "spark.sql.execution.pythonUDF.arrow.enabled" -> "true",
      "spark.gluten.sql.columnar.arrowUdf" -> "true",
      "spark.gluten.sql.columnar.maxBatchSize" -> "2",
      "spark.gluten.sql.columnar.arrowUdf.maxPendingInputBatches" -> "1",
      "spark.gluten.mpp.enabled" -> "false"
    ) {
      // Coalescing three source partitions into one task preserves three input batches. With a
      // capacity-one pending FIFO this spills older batches instead of waiting for Python output.
      val base = spark.range(0, 28, 1, 3).coalesce(1)
      // Keep id in the final result: spilled input vectors must still be readable after their IPC
      // reader/root/stream have closed and the task-local file has been deleted.
      val df = base.select(base("id"), ordinaryPythonUDFString(base("id")).as("decoded"))
      val plan = df.queryExecution.executedPlan
      val arrowStages = plan.collect { case stage: ColumnarArrowEvalPythonExec => stage }

      assert(arrowStages.nonEmpty, plan.treeString)
      assert(plan.collect { case _: BatchEvalPythonExec => 1 }.isEmpty, plan.treeString)
      failAfter(Span(30, Seconds)) {
        assert(df.collect().toSeq == rowReference)
      }
      assert(
        arrowStages.map(_.metrics("numOutputBatches").value).sum > 1,
        "expected multiple input/output batches through the capacity-one pending FIFO")
      assert(
        arrowStages.map(_.metrics("spillFiles").value).sum > 0,
        "expected the capacity-one pending FIFO to spill at least one input batch")
      assert(
        arrowStages.map(_.metrics("spillBytes").value).sum > 0,
        "expected Arrow IPC spill bytes to be recorded")
      assert(createdArrowUdfSpillFiles > createdSpillFilesBefore)
      assert(activeArrowUdfSpillFiles == activeSpillFilesBefore)
      // A materialized spill has one wrapper owner after the independent reload root is created.
      // Successful result recycling must close it exactly once and return the global vector count
      // to the pre-query state.
      assert(ArrowWritableColumnVector.stat() == vectorStateBefore)
    }
  }

  testWithMaxSparkVersion(
    "ordinary scalar UDF with nullable input preserves row-UDF None semantics",
    "4.0") {
    assume(SparkVersionUtil.gteSpark35, "ordinary scalar Arrow UDF eval type requires Spark 3.5+")
    // Spark's TestPythonUDF is explicitly `None if x is None else str(x)`. Under Spark 4's
    // ordinary Arrow scalar protocol a nullable string can arrive as float NaN instead of None, so
    // changing this row UDF's eval type would change its observable result.
    def nullableInputs: DataFrame = spark
      .range(0, 3, 1, 1)
      .selectExpr(
        "CASE id WHEN 0 THEN CAST(NULL AS STRING) WHEN 1 THEN '' ELSE 'value' END AS payload")
      // Keep the tested Python input as a nullable AttributeReference. The engine deliberately
      // does not broaden its conservative pre-projection contract to arbitrary CASE expressions.
      .repartition(1)
    val rowReference = withSQLConf(
      "spark.sql.execution.pythonUDF.arrow.enabled" -> "false",
      "spark.gluten.mpp.enabled" -> "false") {
      val base = nullableInputs
      base.select(ordinaryPythonUDFString(base("payload")).as("decoded")).collect().toSeq
    }
    assert(rowReference.head.isNullAt(0))

    withSQLConf(
      "spark.sql.execution.pythonUDF.arrow.enabled" -> "true",
      "spark.gluten.sql.columnar.arrowUdf" -> "true",
      "spark.gluten.sql.columnar.arrowUdf.nullPreservingInput" -> "false",
      "spark.gluten.mpp.enabled" -> "false"
    ) {
      val base = nullableInputs
      val df = base.select(ordinaryPythonUDFString(base("payload")).as("decoded"))
      val actual = df.collect().toSeq
      val plan = getExecutedPlan(df)

      assert(!plan.exists(_.isInstanceOf[ColumnarArrowEvalPythonExec]), plan.mkString("\n"))
      assert(
        plan.exists {
          case _: BatchEvalPythonExec | _: EvalPythonExecTransformer => true
          case _ => false
        },
        plan.mkString("\n"))
      assert(actual == rowReference)
    }
  }

  testWithMaxSparkVersion(
    "ordinary scalar Arrow UDF coordinates output that arrives before serialization ACK",
    "4.0") {
    assume(SparkVersionUtil.gteSpark35, "ordinary scalar Arrow UDF eval type requires Spark 3.5+")
    val vectorStateBefore = ArrowWritableColumnVector.stat()
    val activeSpillFilesBefore = activeArrowUdfSpillFiles
    ArrowUdfSerializationAckTestSupport.install()
    try {
      withSQLConf(
        "spark.sql.execution.pythonUDF.arrow.enabled" -> "true",
        "spark.gluten.sql.columnar.arrowUdf" -> "true",
        "spark.gluten.sql.columnar.arrowUdf.serializationAckTimeoutMs" -> "5000",
        "spark.gluten.mpp.enabled" -> "false"
      ) {
        val base = spark.range(0, 2, 1, 1)
        val df = base.select(base("id"), ordinaryPythonUDFString(base("id")).as("decoded"))
        val plan = df.queryExecution.executedPlan
        assert(plan.collect { case _: ColumnarArrowEvalPythonExec => 1 }.nonEmpty, plan.treeString)

        val actual = failAfter(Span(30, Seconds)) {
          df.collect().map(row => row.getLong(0) -> row.getString(1)).toSeq
        }
        assert(actual == Seq(0L -> "0", 1L -> "1"))
        assert(ArrowUdfSerializationAckTestSupport.ackDeferred)
        assert(ArrowUdfSerializationAckTestSupport.outputWaitObserved)
        assert(ArrowUdfSerializationAckTestSupport.actualQueueWaitObserved)
        assert(ArrowUdfSerializationAckTestSupport.ackFromDifferentThread)
        assert(ArrowUdfSerializationAckTestSupport.observedAckDelayMillis >= 50L)
        assert(ArrowUdfSerializationAckTestSupport.asyncFailure == null)
        assert(activeArrowUdfSpillFiles == activeSpillFilesBefore)
        assert(ArrowWritableColumnVector.stat() == vectorStateBefore)
      }
    } finally {
      ArrowUdfSerializationAckTestSupport.clear()
    }
    assert(!ArrowUdfSerializationAckTestSupport.workerAlive)
  }

  testWithMaxSparkVersion("ordinary scalar Arrow UDF fails closed at its task spill cap", "4.0") {
    assume(SparkVersionUtil.gteSpark35, "ordinary scalar Arrow UDF eval type requires Spark 3.5+")
    val vectorStateBefore = ArrowWritableColumnVector.stat()
    val activeSpillFilesBefore = activeArrowUdfSpillFiles
    val createdSpillFilesBefore = createdArrowUdfSpillFiles
    withSQLConf(
      "spark.sql.execution.pythonUDF.arrow.enabled" -> "true",
      "spark.gluten.sql.columnar.arrowUdf" -> "true",
      "spark.gluten.sql.columnar.maxBatchSize" -> "2",
      "spark.gluten.sql.columnar.arrowUdf.maxPendingInputBatches" -> "1",
      // Even an Arrow IPC schema header cannot fit. The bounded output stream must reject before
      // the task-local file grows beyond this limit, then remove the partial file and all retains.
      "spark.gluten.sql.columnar.arrowUdf.maxSpillBytesPerTask" -> "1",
      "spark.gluten.mpp.enabled" -> "false"
    ) {
      val base = spark.range(0, 28, 1, 3).coalesce(1)
      val df = base.select(base("id"), ordinaryPythonUDFString(base("id")).as("decoded"))
      assert(df.queryExecution.executedPlan.collect {
        case _: ColumnarArrowEvalPythonExec => 1
      }.nonEmpty)

      val error = failAfter(Span(30, Seconds)) {
        intercept[Exception] {
          df.collect()
        }
      }
      val messageChain = Iterator
        .iterate[Throwable](error)(_.getCause)
        .takeWhile(_ != null)
        .flatMap(e => Option(e.getMessage))
        .mkString("\n")
      assert(
        messageChain.contains("spark.gluten.sql.columnar.arrowUdf.maxSpillBytesPerTask"),
        messageChain)
      assert(createdArrowUdfSpillFiles > createdSpillFilesBefore)
      assert(activeArrowUdfSpillFiles == activeSpillFilesBefore)
      assert(ArrowWritableColumnVector.stat() == vectorStateBefore)
    }
  }

  testWithMaxSparkVersion("ordinary scalar Arrow UDF cleans spills after Python failure", "4.0") {
    assume(SparkVersionUtil.gteSpark35, "ordinary scalar Arrow UDF eval type requires Spark 3.5+")
    val vectorStateBefore = ArrowWritableColumnVector.stat()
    val activeSpillFilesBefore = activeArrowUdfSpillFiles
    val createdSpillFilesBefore = createdArrowUdfSpillFiles
    withSQLConf(
      "spark.sql.execution.pythonUDF.arrow.enabled" -> "true",
      "spark.gluten.sql.columnar.arrowUdf" -> "true",
      "spark.gluten.sql.columnar.maxBatchSize" -> "2",
      "spark.gluten.sql.columnar.arrowUdf.maxPendingInputBatches" -> "1",
      "spark.gluten.mpp.enabled" -> "false"
    ) {
      val base = spark.range(0, 28, 1, 3).coalesce(1)
      val df = base.select(base("id"), failingOrdinaryPythonUDF(base("id")).as("decoded"))
      val arrowStages = df.queryExecution.executedPlan.collect {
        case stage: ColumnarArrowEvalPythonExec => stage
      }
      assert(arrowStages.nonEmpty, df.queryExecution.executedPlan.treeString)

      val error = failAfter(Span(30, Seconds)) {
        intercept[Exception] {
          df.collect()
        }
      }
      val messageChain = Iterator
        .iterate[Throwable](error)(_.getCause)
        .takeWhile(_ != null)
        .flatMap(e => Option(e.getMessage))
        .mkString("\n")
      assert(messageChain.contains("intentional Arrow UDF failure"), messageChain)
      // Failed-task SQL metrics are not merged into the driver plan. The process-local lifecycle
      // tracker proves this task created spill files, and that exception cleanup deleted all of
      // them before the failed action returned.
      assert(createdArrowUdfSpillFiles > createdSpillFilesBefore)
      assert(activeArrowUdfSpillFiles == activeSpillFilesBefore)
      assert(ArrowWritableColumnVector.stat() == vectorStateBefore)
    }
  }

  testWithMaxSparkVersion(
    "ordinary scalar Arrow UDF closes an invalid-schema output batch",
    "4.0") {
    assume(SparkVersionUtil.gteSpark35, "ordinary scalar Arrow UDF eval type requires Spark 3.5+")
    val vectorStateBefore = ArrowWritableColumnVector.stat()
    val activeSpillFilesBefore = activeArrowUdfSpillFiles
    withSQLConf(
      "spark.sql.execution.pythonUDF.arrow.enabled" -> "true",
      "spark.gluten.sql.columnar.arrowUdf" -> "true",
      "spark.gluten.mpp.enabled" -> "false") {
      val base = spark.range(0, 2, 1, 1)
      val df = base.select(mismatchedOrdinaryPythonUDF(base("id")).as("decoded"))
      val plan = df.queryExecution.executedPlan
      assert(plan.collect { case _: ColumnarArrowEvalPythonExec => 1 }.nonEmpty, plan.treeString)

      val error = failAfter(Span(30, Seconds)) {
        intercept[Exception] {
          df.collect()
        }
      }
      val messageChain = Iterator
        .iterate[Throwable](error)(_.getCause)
        .takeWhile(_ != null)
        .flatMap(e => Option(e.getMessage))
        .mkString("\n")
      assert(messageChain.contains("Invalid schema from arrow_udf"), messageChain)
      assert(activeArrowUdfSpillFiles == activeSpillFilesBefore)
      assert(ArrowWritableColumnVector.stat() == vectorStateBefore)
    }
  }

  testWithMaxSparkVersion(
    "nullable ordinary scalar UDF requires a null-preserving runtime capability",
    "4.0") {
    assume(SparkVersionUtil.gteSpark35, "ordinary scalar Arrow UDF eval type requires Spark 3.5+")
    def nullableInputs: DataFrame = spark
      .range(0, 3, 1, 1)
      .selectExpr(
        "CASE id WHEN 0 THEN CAST(NULL AS STRING) WHEN 1 THEN '' ELSE 'value' END AS payload")
      // Keep the tested Python input as a nullable AttributeReference. The engine deliberately
      // does not broaden its conservative pre-projection contract to arbitrary CASE expressions.
      .repartition(1)
    val rowReference = withSQLConf(
      "spark.sql.execution.pythonUDF.arrow.enabled" -> "false",
      "spark.gluten.mpp.enabled" -> "false") {
      val base = nullableInputs
      base.select(ordinaryPythonUDFString(base("payload")).as("decoded")).collect().toSeq
    }

    withSQLConf(
      "spark.sql.execution.pythonUDF.arrow.enabled" -> "true",
      "spark.gluten.sql.columnar.arrowUdf" -> "true",
      "spark.gluten.sql.columnar.arrowUdf.nullPreservingInput" -> "true",
      "spark.gluten.mpp.enabled" -> "false"
    ) {
      val base = nullableInputs
      val df = base.select(ordinaryPythonUDFString(base("payload")).as("decoded"))
      val actual = df.collect().toSeq
      val plan = getExecutedPlan(df)

      assert(plan.exists(_.isInstanceOf[ColumnarArrowEvalPythonExec]), plan.mkString("\n"))
      assert(!plan.exists(_.isInstanceOf[BatchEvalPythonExec]), plan.mkString("\n"))
      assert(actual == rowReference)
      assert(actual.head.isNullAt(0))
    }
  }

  private def newTestScalarPandasUDF(
      name: String,
      returnType: Option[DataType] = None): TestScalarPandasUDF = {
    if (SparkVersionUtil.gteSpark40) {
      // After https://github.com/apache/spark/pull/42864 which landed in Spark 4.0, the return
      // type of the UDF must be explicitly specified when creating the UDF instance with column
      // expressions as parameter.
      classOf[TestScalarPandasUDF]
        .getConstructor(classOf[String], classOf[Option[DataType]])
        .newInstance(name, returnType)
    } else {
      TestScalarPandasUDF(name)
    }
  }

  private def newTestPythonUDF(name: String, returnType: Option[DataType] = None): TestPythonUDF = {
    if (SparkVersionUtil.gteSpark40) {
      classOf[TestPythonUDF]
        .getConstructor(classOf[String], classOf[Option[DataType]])
        .newInstance(name, returnType)
    } else {
      TestPythonUDF(name)
    }
  }

  private lazy val failingOrdinaryPythonUDF: UserDefinedPythonFunction =
    buildOrdinaryPythonUDF(
      "failingOrdinaryPythonUDF",
      "lambda x: (_ for _ in ()).throw(RuntimeError('intentional Arrow UDF failure')) " +
        "if int(x) >= 20 else str(x)",
      "StringType",
      StringType
    )

  private lazy val mismatchedOrdinaryPythonUDF: UserDefinedPythonFunction =
    buildOrdinaryPythonUDF(
      "mismatchedOrdinaryPythonUDF",
      "lambda x: int(x)",
      "LongType",
      StringType)

  private def buildOrdinaryPythonUDF(
      name: String,
      functionExpression: String,
      pythonReturnType: String,
      declaredReturnType: DataType): UserDefinedPythonFunction = {
    val commandFile = Files.createTempFile(s"gluten-arrow-udf-$name", ".pickle")
    try {
      val escapedPath = commandFile.toAbsolutePath.toString
        .replace("\\", "\\\\")
        .replace("'", "\\'")
      val code =
        s"from pyspark.sql.types import $pythonReturnType; " +
          "from pyspark.serializers import CloudPickleSerializer; " +
          s"f = open('$escapedPath', 'wb'); " +
          s"fn = $functionExpression; " +
          s"f.write(CloudPickleSerializer().dumps((fn, $pythonReturnType()))); f.close()"
      val sparkPythonLibZips = sys.env
        .get("SPARK_HOME")
        .toSeq
        .flatMap(sparkHome => Option(new File(sparkHome, "python/lib").listFiles()).toSeq.flatten)
        .filter(file => file.isFile && file.getName.endsWith(".zip"))
        .map(_.getAbsolutePath)
      val serializerPythonPath =
        (sys.env.get("PYTHONPATH").toSeq ++ sparkPythonLibZips).mkString(File.pathSeparator)
      val exitCode =
        Process(Seq(pythonExec, "-c", code), None, "PYTHONPATH" -> serializerPythonPath).!
      require(exitCode == 0, s"Python UDF serializer exited with $exitCode")

      // Spark exposes these classes as private[spark] at Scala source level. Reuse the public JVM
      // constructors reflectively so this Gluten-package test can change only the pickled command.
      val udfAccessor = ordinaryPythonUDFString.getClass.getDeclaredMethod("udf")
      udfAccessor.setAccessible(true)
      val baseUdf = udfAccessor
        .invoke(ordinaryPythonUDFString)
        .asInstanceOf[UserDefinedPythonFunction]
      val baseFunction = classOf[UserDefinedPythonFunction].getMethod("func").invoke(baseUdf)
      def functionProperty(name: String): AnyRef =
        baseFunction.getClass.getMethod(name).invoke(baseFunction)

      val simplePythonFunctionClass =
        ordinaryPythonUDFString.getClass.getClassLoader
          .loadClass("org.apache.spark.api.python.SimplePythonFunction")
      val simplePythonFunctionConstructor = simplePythonFunctionClass.getConstructors
        .find(
          constructor =>
            constructor.getParameterCount == 7 &&
              constructor.getParameterTypes.head == classOf[Array[Byte]])
        .get
      val function = simplePythonFunctionConstructor.newInstance(
        Files.readAllBytes(commandFile),
        functionProperty("envVars"),
        functionProperty("pythonIncludes"),
        functionProperty("pythonExec"),
        functionProperty("pythonVer"),
        functionProperty("broadcastVars"),
        functionProperty("accumulator")
      )
      val udfConstructor = classOf[UserDefinedPythonFunction].getConstructors
        .find(_.getParameterCount == 5)
        .get
      udfConstructor
        .newInstance(
          name,
          function,
          declaredReturnType,
          Int.box(baseUdf.pythonEvalType),
          Boolean.box(true))
        .asInstanceOf[UserDefinedPythonFunction]
    } finally {
      Files.deleteIfExists(commandFile)
    }
  }

  private lazy val arrowUdfSpillFileTracker: AnyRef = {
    val trackerClass = ordinaryPythonUDFString.getClass.getClassLoader
      .loadClass("org.apache.spark.api.python.ArrowUdfSpillFileTracker$")
    trackerClass.getField("MODULE$").get(null)
  }

  private def activeArrowUdfSpillFiles: Int =
    arrowUdfSpillFileTracker.getClass
      .getMethod("activeFileCount")
      .invoke(arrowUdfSpillFileTracker)
      .asInstanceOf[Int]

  private def createdArrowUdfSpillFiles: Long =
    arrowUdfSpillFileTracker.getClass
      .getMethod("createdFileCount")
      .invoke(arrowUdfSpillFileTracker)
      .asInstanceOf[Long]
}
