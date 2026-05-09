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
import org.apache.spark.sql.types.{DoubleType, FloatType, StructType}

import org.scalatest.concurrent.TimeLimits
import org.scalatest.time.{Seconds, Span}

import java.io.File

import scala.util.Try

/**
 * Full-output correctness harness for SF1K TPC-H runs.
 *
 * This suite intentionally stays separate from [[VeloxTPCHFloatSF1KSuite]], which is still useful
 * as a broad runtime smoke test. It compares complete query output against Spark RAPIDS Parquet
 * references rather than relying on the truncated [MPP-RESULT] log preview.
 */
class VeloxTPCHFloatSF1KCompareSuite extends VeloxTPCHTableSupport with TimeLimits {

  private val defaultReferenceRoot = "/raid/pv-cli/spark_output/rapids_sf1000_1GPU_3run_0423"
  private val q4ExistsLineitemDedupKey = "spark.gluten.mpp.q4ExistsLineitemDedup"

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

  protected val externalDataDir: String =
    nonNullProperty("gluten.tpch.externalDataDir", "/data/tpch/sf1k_v2_float")

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
      .set("spark.gluten.mpp.substraitDumpDir", "/opt/gluten/mpp-dumps-tpch-sf1k")

    Seq(
      "spark.driver.maxResultSize",
      "spark.gluten.sql.columnar.libpath",
      "spark.gluten.loadLibFromJar",
      q4ExistsLineitemDedupKey).foreach {
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

  private val q4ExistsLineitemDedupSQL: String =
    """
      |select
      |  o.o_orderpriority,
      |  sum(case when l.late_lineitem_count > 0 then 1 else 0 end) as order_count
      |from
      |  orders o
      |join (
      |  select
      |    l_orderkey,
      |    count(*) as late_lineitem_count
      |  from
      |    lineitem
      |  where
      |    l_commitdate < l_receiptdate
      |  group by
      |    l_orderkey
      |) l
      |  on l.l_orderkey = o.o_orderkey
      |where
      |  o.o_orderdate >= date '1993-07-01'
      |  and o.o_orderdate < date '1993-07-01' + interval '3' month
      |group by
      |  o.o_orderpriority
      |order by
      |  o.o_orderpriority
      |""".stripMargin

  private def q4ExistsLineitemDedupEnabled: Boolean =
    sys.props
      .get(q4ExistsLineitemDedupKey)
      .exists(_.trim.equalsIgnoreCase("true"))

  override protected def tpchSQL(queryNum: Int, tpchQueries: String): String = {
    val raw =
      if (queryNum == 4 && q4ExistsLineitemDedupEnabled) {
        q4ExistsLineitemDedupSQL
      } else {
        super.tpchSQL(queryNum, tpchQueries)
      }
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
      test(s"TPC-H q$qid matches RAPIDS SF1K reference") {
        failAfter(perQueryTimeout) {
          val actualDf = spark.sql(tpchSQL(qid, tpchQueries))
          logWarning(
            s"VeloxTPCHFloatSF1KCompareSuite: Q$qid optimizedPlan with stats:\n" +
              actualDf.queryExecution.stringWithStats)

          val expectedDf = spark.read.parquet(referencePath(qid).getAbsolutePath)
          compareFullOutput(qid, actualDf, expectedDf)
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
