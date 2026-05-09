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
package org.apache.gluten.execution

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types._

import com.fasterxml.jackson.databind.{JsonNode, ObjectMapper}
import org.scalatest.concurrent.TimeLimits
import org.scalatest.time.{Seconds, Span}

import java.io.File

import scala.collection.JavaConverters._
import scala.util.Try

/**
 * Full-output correctness harness for SF1K TPC-H runs.
 *
 * This suite intentionally stays separate from [[VeloxTPCHFloatSF1KSuite]], which is still useful
 * as a broad runtime smoke test. When `gluten.tpch.prestoReferenceDir` is set it uses Presto-GPU
 * JSON as the primary oracle; otherwise it preserves the RAPIDS Parquet reference path.
 */
class VeloxTPCHFloatSF1KCompareSuite extends VeloxTPCHTableSupport with TimeLimits {

  private val defaultReferenceRoot = "/raid/pv-cli/spark_output/rapids_sf1000_1GPU_3run_0423"
  private val objectMapper = new ObjectMapper()

  private def nonNullProperty(key: String, defaultValue: String): String = {
    sys.props.get(key).filter(value => value.nonEmpty && value != "null").getOrElse(defaultValue)
  }

  private val perQueryTimeout = Span(
    sys.props
      .get("gluten.tpch.perQueryTimeoutSeconds")
      .flatMap(s => Try(s.toInt).toOption)
      .getOrElse(45),
    Seconds)

  private val absTolerance: Double =
    sys.props
      .get("gluten.tpch.compareAbsTolerance")
      .flatMap(s => Try(s.toDouble).toOption)
      .getOrElse(1e-5d)

  private val relTolerance: Double =
    sys.props
      .get("gluten.tpch.compareRelTolerance")
      .flatMap(s => Try(s.toDouble).toOption)
      .getOrElse(1e-5d)

  private val maxDiffs: Int =
    sys.props
      .get("gluten.tpch.compareMaxDiffs")
      .flatMap(s => Try(s.toInt).toOption)
      .getOrElse(20)

  private val q13InnerDiagnosticEnabled: Boolean =
    sys.props
      .get("gluten.tpch.q13InnerDiagnostic")
      .exists(_.equalsIgnoreCase("true"))

  private val q13InnerDiagnosticOnly: Boolean =
    sys.props
      .get("gluten.tpch.q13InnerDiagnosticOnly")
      .exists(_.equalsIgnoreCase("true"))

  private val q13DiagnosticCustomerStart: Long =
    sys.props
      .get("gluten.tpch.q13DiagnosticCustomerStart")
      .flatMap(s => Try(s.toLong).toOption)
      .getOrElse(1L)

  private val q13DiagnosticCustomerEnd: Long =
    sys.props
      .get("gluten.tpch.q13DiagnosticCustomerEnd")
      .flatMap(s => Try(s.toLong).toOption)
      .getOrElse(500000L)

  private val q13DiagnosticBucketCount: Int =
    sys.props
      .get("gluten.tpch.q13DiagnosticBucketCount")
      .flatMap(s => Try(s.toInt).toOption)
      .filter(_ > 0)
      .getOrElse(32)

  private val referenceOutputRoot: File =
    new File(nonNullProperty("gluten.tpch.referenceOutputRoot", defaultReferenceRoot))

  private val referenceRun: String =
    nonNullProperty("gluten.tpch.referenceRun", "run-1-test")

  private val prestoReferenceDir: Option[File] =
    sys.props
      .get("gluten.tpch.prestoReferenceDir")
      .filter(value => value.nonEmpty && value != "null")
      .map(new File(_))

  private val allowPrestoSampleOnly: Boolean =
    sys.props
      .get("gluten.tpch.allowPrestoSampleOnly")
      .exists(_.equalsIgnoreCase("true"))

  protected val externalDataDir: String =
    nonNullProperty("gluten.tpch.externalDataDir", "/data/tpch/sf1k_v2_float")

  private val mppDumpDir: String =
    sys.props
      .get("spark.gluten.mpp.substraitDumpDir")
      .orElse(sys.env.get("CURSOR_MPP_DUMP_DIR"))
      .filter(_.nonEmpty)
      .getOrElse("/opt/gluten/mpp-dumps-tpch-sf1k")

  override protected def createTPCHNotNullTables(): Unit = {
    TPCHTableDataFrames = TPCHTables
      .map(_.name)
      .map {
        table =>
          val tablePath = new File(externalDataDir, table).getAbsolutePath
          val tableDF = spark.read.format(fileFormat).load(tablePath)
          tableDF.createOrReplaceTempView(table)
          (table, tableDF)
      }
      .toMap
  }

  override protected def sparkConf: SparkConf = {
    val conf = super.sparkConf
      .set("spark.sql.shuffle.partitions", "16")
      .set("spark.memory.offHeap.size", "32g")
      .set("spark.sql.adaptive.enabled", "false")
      .set("spark.sql.autoBroadcastJoinThreshold", (2L * 1024 * 1024 * 1024).toString)
      .set(
        "spark.driver.maxResultSize",
        sys.props
          .get("spark.driver.maxResultSize")
          .filter(v => v.nonEmpty && v != "null")
          .getOrElse("8g"))
      .set("spark.gluten.mpp.enabled", "true")
      .set("spark.gluten.mpp.strategy.enabled", "true")
      .set("spark.gluten.mpp.singlePartitionSort", "true")
      .set("spark.gluten.mpp.removeRedundantShuffle", "true")
      .set("spark.gluten.mpp.parallelSortSplit", "true")
      .set("spark.gluten.mpp.fuseBroadcastBuilds", "true")
      .set("spark.gluten.sql.columnar.cudf", "true")
      .set("spark.gluten.sql.columnar.backend.velox.cudf.enabled", "true")
      .set("spark.gluten.sql.columnar.backend.velox.cudf.enableTableScan", "true")
      .set("spark.gluten.sql.columnar.backend.velox.cudf.jit_expression_enabled", "false")
      .set("spark.gluten.sql.columnar.backend.velox.cudf.ast_expression_enabled", "true")
      .set("spark.gluten.mpp.substraitDumpDir", mppDumpDir)

    Seq(
      "spark.gluten.sql.columnar.libpath",
      "spark.gluten.loadLibFromJar",
      "spark.gluten.mpp.substraitDumpDir",
      "spark.gluten.mpp.localHashExchangeTasks",
      "spark.gluten.mpp.maxDriversPerFragment",
      "spark.gluten.mpp.maxInboundExchangesPerFragment",
      "spark.gluten.mpp.maxHashInboundExchangesPerFragment",
      "spark.gluten.mpp.maxBroadcastInboundExchangesPerFragment",
      "spark.gluten.mpp.q3.replicateOrdersPath",
      "spark.gluten.mpp.fuseBroadcastBuilds",
      "spark.gluten.mpp.normalizeJoinBuildSide",
      "spark.gluten.mpp.q4ExistsLineitemDedup"
    ).foreach {
      key =>
        sys.props
          .get(key)
          .filter(value => value.nonEmpty && value != "null")
          .foreach(conf.set(key, _))
    }
    conf
  }

  private val broadcastHintByQuery: Map[Int, String] = Map(
    14 -> "/*+ BROADCAST(part) */",
    16 -> "/*+ BROADCAST(part, supplier) */",
    17 -> "/*+ BROADCAST(part) */",
    18 -> "/*+ BROADCAST(customer) */",
    19 -> "/*+ BROADCAST(part) */"
  )

  override protected def tpchSQL(queryNum: Int, tpchQueries: String): String = {
    val raw = super.tpchSQL(queryNum, tpchQueries)
    broadcastHintByQuery.get(queryNum) match {
      case Some(hint) => raw.replaceFirst("(?i)\\bselect\\b", s"select $hint")
      case None => raw
    }
  }

  private val queriesToCompare: Seq[Int] =
    if (q13InnerDiagnosticOnly) {
      Seq.empty
    } else {
      sys.props
        .get("gluten.tpch.onlyQuery")
        .map {
          _.split(",").toSeq
            .flatMap(s => Try(s.trim.toInt).toOption)
            .distinct
            .sorted
        }
        .filter(_.nonEmpty)
        .getOrElse {
          if (prestoReferenceDir.isDefined &&
            sys.props.get("gluten.tpch.compareAllQueries").exists(_.equalsIgnoreCase("true"))) {
            (1 to 22).toSeq
          } else {
            Seq(1, 6, 12)
          }
        }
    }

  queriesToCompare.foreach {
    qid =>
      val referenceLabel =
        if (prestoReferenceDir.isDefined) "Presto-GPU JSON reference" else "RAPIDS SF1K reference"
      test(s"TPC-H q$qid matches $referenceLabel") {
        failAfter(perQueryTimeout) {
          val actualDf = spark.sql(tpchSQL(qid, tpchQueries))
          logWarning(
            s"VeloxTPCHFloatSF1KCompareSuite: Q$qid optimizedPlan with stats:\n" +
              actualDf.queryExecution.stringWithStats)

          prestoReferenceDir match {
            case Some(_) => comparePrestoJsonOutput(qid, actualDf)
            case None =>
              val expectedDf = spark.read.parquet(referencePath(qid).getAbsolutePath)
              compareFullOutput(qid, actualDf, expectedDf)
          }
        }
      }
  }

  if (q13InnerDiagnosticEnabled) {
    test("TPC-H q13 inner MPP vs CPU diagnostic") {
      failAfter(perQueryTimeout) {
        compareQ13InnerSubquery()
      }
    }
  }

  private def referencePath(qid: Int): File = {
    val queryDir = if (qid == 15) "query15_part2" else s"query$qid"
    val direct = new File(referenceOutputRoot, queryDir)
    if (direct.isDirectory) {
      direct
    } else {
      val nested = new File(new File(referenceOutputRoot, referenceRun), queryDir)
      assert(
        nested.isDirectory,
        s"Missing RAPIDS reference for Q$qid. Checked ${direct.getAbsolutePath} and " +
          s"${nested.getAbsolutePath}")
      nested
    }
  }

  private case class PrestoReference(path: File, rows: Option[Long], sampleRows: Seq[Row])

  private def prestoReferencePath(qid: Int): File = {
    val dir = prestoReferenceDir.getOrElse(fail("Presto reference directory is not configured"))
    val path = new File(dir, s"Q$qid.result.json")
    assert(path.isFile, s"Missing Presto-GPU JSON reference for Q$qid: ${path.getAbsolutePath}")
    path
  }

  private def loadPrestoReference(qid: Int, schema: StructType): PrestoReference = {
    val path = prestoReferencePath(qid)
    val root =
      try {
        objectMapper.readTree(path)
      } catch {
        case e: Exception =>
          fail(
            s"Failed to parse Presto-GPU JSON reference ${path.getAbsolutePath}: " +
              s"${e.getMessage}")
      }

    assert(root != null && root.isObject, s"Presto-GPU JSON reference is not an object: $path")
    val rows = Option(root.get("rows")).filterNot(_.isNull).map(jsonRowsAsLong(path, _))
    val sampleNode = Option(root.get("sample"))
      .getOrElse(fail(s"Presto-GPU JSON reference missing sample array: $path"))
    assert(sampleNode.isArray, s"Presto-GPU JSON field sample is not an array: $path")

    val sampleRows = sampleNode
      .elements()
      .asScala
      .zipWithIndex
      .map { case (rowNode, index) => rowFromJsonArray(path, index, rowNode, schema) }
      .toSeq

    rows.foreach {
      rowCount =>
        assert(
          sampleRows.size.toLong <= rowCount,
          s"Presto-GPU JSON sample has more rows than rows field for Q$qid: " +
            s"sample=${sampleRows.size}, rows=$rowCount, path=${path.getAbsolutePath}"
        )
    }
    PrestoReference(path, rows, sampleRows)
  }

  private def jsonRowsAsLong(path: File, node: JsonNode): Long = {
    if (node.isIntegralNumber) {
      node.asLong()
    } else {
      Try(node.asText().toLong).getOrElse(
        fail(s"Presto-GPU JSON rows field is not an integer in ${path.getAbsolutePath}: $node"))
    }
  }

  private def rowFromJsonArray(
      path: File,
      rowIndex: Int,
      rowNode: JsonNode,
      schema: StructType): Row = {
    assert(
      rowNode.isArray,
      s"Presto-GPU JSON sample row[$rowIndex] is not an array in ${path.getAbsolutePath}")
    assert(
      rowNode.size() == schema.length,
      s"Presto-GPU JSON sample row[$rowIndex] column count mismatch in ${path.getAbsolutePath}: " +
        s"expected schema columns=${schema.length}, actual=${rowNode.size()}"
    )

    Row.fromSeq(schema.fields.zipWithIndex.map {
      case (field, index) => jsonValueAsSparkValue(path, rowIndex, index, rowNode.get(index), field)
    })
  }

  private def jsonValueAsSparkValue(
      path: File,
      rowIndex: Int,
      columnIndex: Int,
      node: JsonNode,
      field: StructField): Any = {
    if (node == null || node.isNull) {
      null
    } else {
      try {
        field.dataType match {
          case ByteType => new java.math.BigDecimal(jsonScalarText(node)).byteValueExact()
          case ShortType => new java.math.BigDecimal(jsonScalarText(node)).shortValueExact()
          case IntegerType => new java.math.BigDecimal(jsonScalarText(node)).intValueExact()
          case LongType => new java.math.BigDecimal(jsonScalarText(node)).longValueExact()
          case FloatType => jsonScalarText(node).toFloat
          case DoubleType => jsonScalarText(node).toDouble
          case _: DecimalType => new java.math.BigDecimal(jsonScalarText(node))
          case BooleanType => node.asBoolean()
          case DateType => java.sql.Date.valueOf(jsonScalarText(node))
          case TimestampType => java.sql.Timestamp.valueOf(jsonScalarText(node).replace('T', ' '))
          case StringType => jsonScalarText(node)
          case _ => jsonScalarText(node)
        }
      } catch {
        case e: Exception =>
          fail(
            s"Cannot coerce Presto-GPU JSON value for ${path.getAbsolutePath} " +
              s"row[$rowIndex] col[$columnIndex] (${field.name}:${field.dataType.catalogString}) " +
              s"value=$node: ${e.getMessage}")
      }
    }
  }

  private def jsonScalarText(node: JsonNode): String =
    if (node.isTextual) node.asText() else node.toString

  private def comparePrestoJsonOutput(qid: Int, actualDf: DataFrame): Unit = {
    val actualRowsInOrder = actualDf.collect().toSeq
    val reference = loadPrestoReference(qid, actualDf.schema)
    val prestoRowsText = reference.rows.map(_.toString).getOrElse("<missing>")
    val fullCaptured = reference.rows.contains(reference.sampleRows.size.toLong)
    val mode = if (fullCaptured) "FULL" else "SAMPLE_ONLY"

    logWarning(
      s"[PRESTO-JSON] Q$qid reference=${reference.path.getAbsolutePath} " +
        s"prestoRows=$prestoRowsText sampleRows=${reference.sampleRows.size} mode=$mode " +
        s"allowSampleOnly=$allowPrestoSampleOnly")

    reference.rows.foreach {
      expectedRows =>
        assert(
          actualRowsInOrder.size.toLong == expectedRows,
          s"Q$qid row count mismatch vs Presto-GPU JSON ${reference.path.getAbsolutePath}: " +
            s"expected=$expectedRows, actual=${actualRowsInOrder.size}"
        )
    }

    if (fullCaptured) {
      val mismatch = collectedOutputMismatch(
        s"Q$qid Presto-GPU JSON full compare",
        CollectedOutput(actualDf.schema, sortRows(actualRowsInOrder, actualDf.schema)),
        CollectedOutput(actualDf.schema, sortRows(reference.sampleRows, actualDf.schema))
      )
      mismatch.foreach(message => fail(message))
      logWarning(
        s"[PRESTO-JSON] Q$qid FULL_PASS rows=${actualRowsInOrder.size} " +
          s"reference=${reference.path.getAbsolutePath}")
    } else {
      assert(
        reference.sampleRows.size <= actualRowsInOrder.size,
        s"Q$qid Presto-GPU JSON sample has more rows than actual output: " +
          s"sample=${reference.sampleRows.size}, actual=${actualRowsInOrder.size}, " +
          s"reference=${reference.path.getAbsolutePath}"
      )
      val overlap = math.min(reference.sampleRows.size, actualRowsInOrder.size)
      val mismatch = collectedOutputMismatch(
        s"Q$qid Presto-GPU JSON sample prefix compare",
        CollectedOutput(actualDf.schema, actualRowsInOrder.take(overlap)),
        CollectedOutput(actualDf.schema, reference.sampleRows.take(overlap))
      )
      mismatch.foreach(message => fail(message))

      val samplePassMessage =
        s"Q$qid SAMPLE_PASS only: row count matched Presto-GPU rows=$prestoRowsText and " +
          s"sample prefix rows=$overlap matched, but JSON captured " +
          s"${reference.sampleRows.size} of $prestoRowsText rows. This is not a full " +
          s"correctness pass. Capture full Presto rows or set " +
          s"-Dgluten.tpch.allowPrestoSampleOnly=true for diagnostic sample-only runs."
      logWarning(s"[PRESTO-JSON] $samplePassMessage")
      assert(allowPrestoSampleOnly, samplePassMessage)
    }
  }

  private val q13BoundedCustomerPredicate: String =
    s"c_custkey BETWEEN $q13DiagnosticCustomerStart AND $q13DiagnosticCustomerEnd"

  private val q13BoundedOrderPredicate: String =
    s"o_custkey BETWEEN $q13DiagnosticCustomerStart AND $q13DiagnosticCustomerEnd"

  private val q13BoundedInnerSql: String =
    """
      |SELECT
      |  c_custkey,
      |  count(o_orderkey) AS c_count
      |FROM
      |  (
      |    SELECT c_custkey
      |    FROM customer
      |    WHERE
      |""".stripMargin +
      s"      $q13BoundedCustomerPredicate\n" +
      """
        |  ) sampled_customer LEFT OUTER JOIN (
        |    SELECT o_orderkey, o_custkey
        |    FROM orders
        |    WHERE
        |""".stripMargin +
      s"      $q13BoundedOrderPredicate\n" +
      """
        |      AND o_comment NOT LIKE '%special%requests%'
        |  ) sampled_orders ON
        |    c_custkey = o_custkey
        |GROUP BY
        |  c_custkey
        |""".stripMargin

  private val q13BoundedInnerBucketFingerprintSql: String =
    s"""
       |SELECT
       |  pmod(c_custkey, $q13DiagnosticBucketCount) AS sample_bucket,
       |  count(*) AS row_count,
       |  sum(c_count) AS sum_count,
       |  sum(c_count * c_count) AS sum_count_sq,
       |  min(c_count) AS min_count,
       |  max(c_count) AS max_count,
       |  sum(CASE WHEN c_count = 0 THEN 1 ELSE 0 END) AS zero_count_customers,
       |  sum(c_custkey) AS sum_custkey,
       |  sum(c_custkey * (c_count + CAST(1 AS BIGINT))) AS weighted_custkey_count
       |FROM
       |  ($q13BoundedInnerSql) q13_inner
       |GROUP BY
       |  pmod(c_custkey, $q13DiagnosticBucketCount)
       |ORDER BY
       |  sample_bucket
       |""".stripMargin

  private val q13BoundedOuterDistributionSql: String =
    s"""
       |SELECT
       |  c_count,
       |  count(*) AS custdist
       |FROM
       |  ($q13BoundedInnerSql) q13_inner
       |GROUP BY
       |  c_count
       |ORDER BY
       |  custdist DESC,
       |  c_count DESC
       |""".stripMargin

  private case class CollectedOutput(schema: StructType, rows: Seq[Row])

  private def compareQ13InnerSubquery(): Unit = {
    logWarning(
      s"[CURSOR-Q13-INNER] bounded diagnostic customerStart=$q13DiagnosticCustomerStart " +
        s"customerEnd=$q13DiagnosticCustomerEnd bucketCount=$q13DiagnosticBucketCount")

    val mppFingerprint = collectOutput(spark.sql(q13BoundedInnerBucketFingerprintSql))
    val mppDistribution = collectOutput(spark.sql(q13BoundedOuterDistributionSql))

    withSQLConf(
      "spark.gluten.enabled" -> "false",
      "spark.gluten.mpp.enabled" -> "false",
      "spark.gluten.mpp.strategy.enabled" -> "false") {
      val cpuFingerprint = collectOutput(spark.sql(q13BoundedInnerBucketFingerprintSql))
      val cpuDistribution = collectOutput(spark.sql(q13BoundedOuterDistributionSql))

      logDiagnosticOutput("Q13 bounded inner bucket fingerprint MPP", mppFingerprint)
      logDiagnosticOutput("Q13 bounded inner bucket fingerprint CPU", cpuFingerprint)
      logDiagnosticOutput("Q13 bounded outer distribution MPP", mppDistribution)
      logDiagnosticOutput("Q13 bounded outer distribution CPU", cpuDistribution)

      val innerMismatch =
        collectedOutputMismatch(
          "Q13 bounded inner bucket fingerprint MPP vs CPU",
          mppFingerprint,
          cpuFingerprint)
      val outerMismatch =
        collectedOutputMismatch(
          "Q13 bounded outer distribution MPP vs CPU",
          mppDistribution,
          cpuDistribution)
      val classification = (innerMismatch, outerMismatch) match {
        case (Some(_), _) => "INNER_JOIN_FILTER_AGGREGATE"
        case (None, Some(_)) => "OUTER_AGGREGATE"
        case (None, None) => "NO_MISMATCH_IN_BOUNDED_SAMPLE"
      }

      logWarning(s"[CURSOR-Q13-INNER] bounded classification=$classification")

      assert(
        innerMismatch.isEmpty && outerMismatch.isEmpty,
        s"Q13 bounded diagnostic classification=$classification\n" +
          Seq(innerMismatch, outerMismatch).flatten.mkString("\n")
      )
    }
  }

  private def collectOutput(df: DataFrame): CollectedOutput =
    CollectedOutput(df.schema, sortRows(df.collect().toSeq, df.schema))

  private def compareCollectedOutput(
      label: String,
      actual: CollectedOutput,
      expected: CollectedOutput): Unit = {
    collectedOutputMismatch(label, actual, expected).foreach(message => fail(message))
  }

  private def collectedOutputMismatch(
      label: String,
      actual: CollectedOutput,
      expected: CollectedOutput): Option[String] = {
    val schemaMismatch = schemaMismatchMessage(label, actual.schema, expected.schema)
    if (schemaMismatch.isDefined) {
      schemaMismatch
    } else if (actual.rows.size != expected.rows.size) {
      Some(s"$label row count mismatch: expected=${expected.rows.size}, actual=${actual.rows.size}")
    } else {
      val mismatches = actual.rows
        .zip(expected.rows)
        .zipWithIndex
        .collect {
          case ((actualRow, expectedRow), index)
              if !sameRow(actualRow, expectedRow, actual.schema) =>
            s"row[$index]\n  expected=${formatRow(expectedRow)}\n  actual  =${formatRow(actualRow)}"
        }
        .take(maxDiffs)

      if (mismatches.isEmpty) {
        None
      } else {
        Some(
          s"$label mismatch. Showing ${mismatches.size} diff(s), " +
            s"absTolerance=$absTolerance, relTolerance=$relTolerance:\n" +
            mismatches.mkString("\n"))
      }
    }
  }

  private def logDiagnosticOutput(label: String, output: CollectedOutput): Unit = {
    val preview = output.rows.take(maxDiffs).map(formatRow).mkString("; ")
    logWarning(
      s"[CURSOR-Q13-INNER] $label rows=${output.rows.size} " +
        s"schema=${output.schema.simpleString} preview=$preview")
  }

  private def compareFullOutput(qid: Int, actualDf: DataFrame, expectedDf: DataFrame): Unit = {
    compareSchema(s"Q$qid", actualDf.schema, expectedDf.schema)

    val actualRows = sortRows(actualDf.collect().toSeq, actualDf.schema)
    val expectedRows = sortRows(expectedDf.collect().toSeq, expectedDf.schema)
    assert(
      actualRows.size == expectedRows.size,
      s"Q$qid row count mismatch: expected=${expectedRows.size}, actual=${actualRows.size}")

    val mismatches = actualRows
      .zip(expectedRows)
      .zipWithIndex
      .collect {
        case ((actual, expected), index) if !sameRow(actual, expected, actualDf.schema) =>
          s"row[$index]\n  expected=${formatRow(expected)}\n  actual  =${formatRow(actual)}"
      }
      .take(maxDiffs)

    assert(
      mismatches.isEmpty,
      s"Q$qid output mismatch. Showing ${mismatches.size} diff(s), " +
        s"absTolerance=$absTolerance, relTolerance=$relTolerance:\n${mismatches.mkString("\n")}"
    )
  }

  private def compareSchema(
      label: String,
      actualSchema: StructType,
      expectedSchema: StructType): Unit = {
    schemaMismatchMessage(label, actualSchema, expectedSchema).foreach(message => fail(message))
  }

  private def schemaMismatchMessage(
      label: String,
      actualSchema: StructType,
      expectedSchema: StructType): Option[String] = {
    val actualFields = actualSchema.fields.map(f => s"${f.name}:${f.dataType.catalogString}")
    val expectedFields = expectedSchema.fields.map(f => s"${f.name}:${f.dataType.catalogString}")
    if (actualFields.sameElements(expectedFields)) {
      None
    } else {
      Some(
        s"$label schema mismatch:\n  expected=${expectedFields.mkString(", ")}\n  " +
          s"actual  =${actualFields.mkString(", ")}")
    }
  }

  private def sortRows(rows: Seq[Row], schema: StructType): Seq[Row] =
    rows.sortBy(row => schema.indices.map(i => sortValue(row, i, schema)).mkString("|#|"))

  private def sortValue(row: Row, index: Int, schema: StructType): String =
    if (row.isNullAt(index)) {
      "__NULL__"
    } else {
      schema(index).dataType match {
        case DoubleType => f"${row.getDouble(index)}%.12g"
        case FloatType => f"${row.getFloat(index).toDouble}%.12g"
        case _ => row.get(index).toString
      }
    }

  private def sameRow(actual: Row, expected: Row, schema: StructType): Boolean =
    schema.indices.forall(i => sameValue(actual, expected, i, schema))

  private def sameValue(actual: Row, expected: Row, index: Int, schema: StructType): Boolean = {
    val actualNull = actual.isNullAt(index)
    val expectedNull = expected.isNullAt(index)
    if (actualNull || expectedNull) {
      actualNull == expectedNull
    } else {
      schema(index).dataType match {
        case DoubleType => sameDouble(actual.getDouble(index), expected.getDouble(index))
        case FloatType =>
          sameDouble(actual.getFloat(index).toDouble, expected.getFloat(index).toDouble)
        case _ =>
          (actual.get(index), expected.get(index)) match {
            case (left: Array[Byte], right: Array[Byte]) => left.sameElements(right)
            case (left: java.math.BigDecimal, right: java.math.BigDecimal) =>
              left.compareTo(right) == 0
            case (left, right) => left == right
          }
      }
    }
  }

  private def sameDouble(actual: Double, expected: Double): Boolean =
    if (actual.isNaN || expected.isNaN) {
      actual.isNaN && expected.isNaN
    } else if (actual.isInfinity || expected.isInfinity) {
      actual == expected
    } else {
      val diff = Math.abs(actual - expected)
      diff <= absTolerance || diff <= relTolerance * Math.max(Math.abs(actual), Math.abs(expected))
    }

  private def formatRow(row: Row): String =
    row.toSeq
      .map {
        case null => "null"
        case bytes: Array[Byte] => bytes.mkString("[", ",", "]")
        case value => value.toString
      }
      .mkString("[", ", ", "]")
}
