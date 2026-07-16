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
package org.apache.gluten.extension.columnar.offload

import org.apache.gluten.backendsapi.BackendsApiManager
import org.apache.gluten.config.GlutenConfig
import org.apache.gluten.execution._
import org.apache.gluten.expression.ExpressionTransformerProvider
import org.apache.gluten.extension.columnar.FallbackTags
import org.apache.gluten.logging.LogLevelUtil
import org.apache.gluten.sql.shims.SparkShimLoader

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference, Cast, Expression, Literal, PythonUDF}
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight, BuildSide}
import org.apache.spark.sql.catalyst.plans.logical.Join
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.RDDScanTransformer
import org.apache.spark.sql.execution.aggregate.{HashAggregateExec, ObjectHashAggregateExec, SortAggregateExec}
import org.apache.spark.sql.execution.datasources.WriteFilesExec
import org.apache.spark.sql.execution.datasources.v2.BatchScanExec
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeExec, ShuffleExchangeExec}
import org.apache.spark.sql.execution.joins._
import org.apache.spark.sql.execution.python.{ArrowEvalPythonExec, BatchEvalPythonExec, EvalPythonExecTransformer}
import org.apache.spark.sql.execution.window.WindowExec
import org.apache.spark.sql.hive.HiveTableScanExecTransformer
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import org.apache.spark.util.SparkReflectionUtil

import scala.collection.mutable.ArrayBuffer
import scala.util.Try

// Exchange transformation.
case class OffloadExchange() extends OffloadSingleNode with LogLevelUtil {
  override def offload(plan: SparkPlan): SparkPlan = plan match {
    case p if FallbackTags.nonEmpty(p) =>
      p
    case s: ShuffleExchangeExec =>
      logDebug(s"Columnar Processing for ${s.getClass} is currently supported.")
      BackendsApiManager.getSparkPlanExecApiInstance.genColumnarShuffleExchange(s)
    case b: BroadcastExchangeExec =>
      val child = b.child
      logDebug(s"Columnar Processing for ${b.getClass} is currently supported.")
      ColumnarBroadcastExchangeExec(b.mode, child)
    case other => other
  }
}

// Join transformation.
case class OffloadJoin() extends OffloadSingleNode with LogLevelUtil {
  override def offload(plan: SparkPlan): SparkPlan = {
    if (FallbackTags.nonEmpty(plan)) {
      logDebug(s"Columnar Processing for ${plan.getClass} is under row guard.")
      return plan
    }
    val result = plan match {
      case plan: ShuffledHashJoinExec =>
        val left = plan.left
        val right = plan.right
        BackendsApiManager.getSparkPlanExecApiInstance
          .genShuffledHashJoinExecTransformer(
            plan.leftKeys,
            plan.rightKeys,
            plan.joinType,
            OffloadJoin.getShjBuildSide(plan),
            plan.condition,
            left,
            right,
            plan.isSkewJoin)
      case plan: SortMergeJoinExec =>
        val left = plan.left
        val right = plan.right
        BackendsApiManager.getSparkPlanExecApiInstance
          .genSortMergeJoinExecTransformer(
            plan.leftKeys,
            plan.rightKeys,
            plan.joinType,
            plan.condition,
            left,
            right,
            plan.isSkewJoin)
      case plan: BroadcastHashJoinExec =>
        val left = plan.left
        val right = plan.right
        BackendsApiManager.getSparkPlanExecApiInstance
          .genBroadcastHashJoinExecTransformer(
            plan.leftKeys,
            plan.rightKeys,
            plan.joinType,
            plan.buildSide,
            plan.condition,
            left,
            right,
            isNullAwareAntiJoin = plan.isNullAwareAntiJoin)
      case plan: CartesianProductExec =>
        val left = plan.left
        val right = plan.right
        BackendsApiManager.getSparkPlanExecApiInstance
          .genCartesianProductExecTransformer(left, right, plan.condition)
      case plan: BroadcastNestedLoopJoinExec =>
        val left = plan.left
        val right = plan.right
        BackendsApiManager.getSparkPlanExecApiInstance
          .genBroadcastNestedLoopJoinExecTransformer(
            left,
            right,
            plan.buildSide,
            plan.joinType,
            plan.condition)
      case other => other
    }
    if (result.ne(plan)) {
      logDebug(s"Columnar Processing for ${plan.getClass} is currently supported.")
    }
    result
  }
}

object OffloadJoin {
  def getShjBuildSide(shj: ShuffledHashJoinExec): BuildSide = {
    val leftBuildable =
      BackendsApiManager.getSettings.supportHashBuildJoinTypeOnLeft(shj.joinType)
    val rightBuildable =
      BackendsApiManager.getSettings.supportHashBuildJoinTypeOnRight(shj.joinType)

    assert(leftBuildable || rightBuildable)

    if (!leftBuildable) {
      return BuildRight
    }
    if (!rightBuildable) {
      return BuildLeft
    }

    // Both left and right are buildable. Find out the better one.
    if (!GlutenConfig.get.shuffledHashJoinOptimizeBuildSide) {
      // User disabled build side re-optimization. Return original build side from vanilla Spark.
      return shj.buildSide
    }
    shj.logicalLink
      .flatMap {
        case join: Join => Some(getOptimalBuildSide(join))
        case _ => None
      }
      .getOrElse {
        // Some shj operators generated in certain Spark tests such as OuterJoinSuite,
        // could possibly have no logical link set.
        shj.buildSide
      }
  }

  def getOptimalBuildSide(join: Join): BuildSide = {
    val leftSize = join.left.stats.sizeInBytes
    val rightSize = join.right.stats.sizeInBytes
    val leftRowCount = join.left.stats.rowCount
    val rightRowCount = join.right.stats.rowCount
    if (leftSize == rightSize && rightRowCount.isDefined && leftRowCount.isDefined) {
      if (rightRowCount.get <= leftRowCount.get) {
        return BuildRight
      }
      return BuildLeft
    }
    if (rightSize <= leftSize) {
      return BuildRight
    }
    BuildLeft
  }
}

// Other transformations.
case class OffloadOthers() extends OffloadSingleNode with LogLevelUtil {
  import OffloadOthers._
  private val replace = new ReplaceSingleNode

  override def offload(plan: SparkPlan): SparkPlan = replace.doReplace(plan)
}

object OffloadOthers {
  val ARROW_SCALAR_NORMALIZATION_REJECTION_TAG: TreeNodeTag[String] =
    TreeNodeTag[String]("gluten.python.arrowScalarNormalization.rejection")
  private val ARROW_SCALAR_NULL_PRESERVING_INPUT_CAPABILITY =
    "spark.gluten.sql.columnar.arrowUdf.nullPreservingInput"

  private[offload] def isConservativeArrowInputExpression(input: Expression): Boolean = {
    if (input.find(_.isInstanceOf[PythonUDF]).isDefined) {
      return false
    }
    input match {
      case _: AttributeReference | _: Literal => true
      // External providers may opt in an application expression only after establishing the same
      // serializer and native-evaluation contract used by their transformer.
      case external if ExpressionTransformerProvider.isSafeArrowPreProjection(external) =>
        true
      // Materialize only simple casts whose leaf is already a safe input. Broader deterministic
      // expressions can be added with explicit serializer-parity tests.
      case cast: Cast => isConservativeArrowInputExpression(cast.child)
      case _ => false
    }
  }

  // Utility to replace single node within transformed Gluten node.
  // Children will be preserved as they are as children of the output node.
  //
  // Do not look up on children on the input node in this rule. Otherwise,
  // it may break RAS which would group all the possible input nodes to
  // search for validate candidates.
  private class ReplaceSingleNode extends LogLevelUtil with Logging {

    def doReplace(p: SparkPlan): SparkPlan = {
      val plan = p
      if (FallbackTags.nonEmpty(plan)) {
        return plan
      }
      val result = plan match {
        case plan: BatchScanExec =>
          ScanTransformerFactory.createBatchScanTransformer(plan)
        case plan: FileSourceScanExec =>
          ScanTransformerFactory.createFileSourceScanTransformer(plan)
        case plan if HiveTableScanExecTransformer.isHiveTableScan(plan) =>
          // TODO: Add DynamicPartitionPruningHiveScanSuite.scala
          HiveTableScanExecTransformer(plan)
        case plan: CoalesceExec =>
          ColumnarCoalesceExec(plan.numPartitions, plan.child)
        case plan: FilterExec =>
          BackendsApiManager.getSparkPlanExecApiInstance
            .genFilterExecTransformer(plan.condition, plan.child)
        case plan: ProjectExec =>
          val columnarChild = plan.child
          ProjectExecTransformer(plan.projectList, columnarChild)
        case plan: HashAggregateExec =>
          HashAggregateExecBaseTransformer.from(plan)
        case plan: SortAggregateExec =>
          HashAggregateExecBaseTransformer.from(plan)
        case plan: ObjectHashAggregateExec =>
          HashAggregateExecBaseTransformer.from(plan)
        case plan: UnionExec =>
          ColumnarUnionExec.from(plan)
        case plan: ExpandExec =>
          val child = plan.child
          ExpandExecTransformer(plan.projections, plan.output, child)
        case plan: WriteFilesExec =>
          val child = plan.child
          val writeTransformer = WriteFilesExecTransformer(
            child,
            plan.fileFormat,
            plan.partitionColumns,
            plan.bucketSpec,
            plan.options,
            plan.staticPartitions)
          ColumnarWriteFilesExec(
            writeTransformer,
            plan.fileFormat,
            plan.partitionColumns,
            plan.bucketSpec,
            plan.options,
            plan.staticPartitions)
        case plan: SortExec =>
          val child = plan.child
          SortExecTransformer(plan.sortOrder, plan.global, child, plan.testSpillFrequency)
        case plan: TakeOrderedAndProjectExec =>
          val child = plan.child
          val (limit, offset) = SparkShimLoader.getSparkShims.getLimitAndOffsetFromTopK(plan)
          TakeOrderedAndProjectExecTransformer(
            limit,
            plan.sortOrder,
            plan.projectList,
            child,
            offset)
        case plan: WindowExec =>
          WindowExecTransformer(
            plan.windowExpression,
            plan.partitionSpec,
            plan.orderSpec,
            plan.child)
        case plan if SparkShimLoader.getSparkShims.isWindowGroupLimitExec(plan) =>
          val windowGroupLimitExecShim =
            SparkShimLoader.getSparkShims.getWindowGroupLimitExecShim(plan)
          BackendsApiManager.getSparkPlanExecApiInstance.genWindowGroupLimitTransformer(
            windowGroupLimitExecShim.partitionSpec,
            windowGroupLimitExecShim.orderSpec,
            windowGroupLimitExecShim.rankLikeFunction,
            windowGroupLimitExecShim.limit,
            windowGroupLimitExecShim.mode,
            windowGroupLimitExecShim.child
          )
        case plan: GlobalLimitExec =>
          val child = plan.child
          val (limit, offset) =
            SparkShimLoader.getSparkShims.getLimitAndOffsetFromGlobalLimit(plan)
          LimitExecTransformer(child, offset, limit)
        case plan: LocalLimitExec =>
          val child = plan.child
          LimitExecTransformer(child, 0L, plan.limit)
        case plan: LocalTableScanExec
            if plan.rows.length <= LocalTableScanExecTransformer.MaxRows =>
          LocalTableScanExecTransformer(plan.output, plan.rows)
        case plan if LocalTableScanExecTransformer.supportsOneRowRelation(plan) =>
          LocalTableScanExecTransformer.oneRowRelation(plan.output)
        case plan: GenerateExec =>
          val child = plan.child
          BackendsApiManager.getSparkPlanExecApiInstance.genGenerateTransformer(
            plan.generator,
            plan.requiredChildOutput,
            plan.outer,
            plan.generatorOutput,
            child)
        case plan: BatchEvalPythonExec =>
          arrowOptimizedScalarUdfs(plan.udfs) match {
            case Some((arrowUdfs, arrowEvalType)) =>
              logInfo(
                s"Offloading ${plan.udfs.size} ordinary scalar Python UDF(s) through the " +
                  "columnar Arrow runner because " +
                  "spark.sql.execution.pythonUDF.arrow.enabled=true")
              createArrowScalarExec(plan, arrowUdfs, arrowEvalType)
            case None =>
              val rowTransformer =
                EvalPythonExecTransformer(plan.udfs, plan.resultAttrs, plan.child)
              arrowScalarNormalizationRejection(plan.udfs).foreach {
                reason =>
                  rowTransformer.setTagValue(ARROW_SCALAR_NORMALIZATION_REJECTION_TAG, reason)
                  logWarning(reason)
                  failStrictMppOnArrowNormalizationRejection(reason)
              }
              rowTransformer
          }
        case plan: ArrowEvalPythonExec =>
          val child = plan.child
          // For ArrowEvalPythonExec, CH supports it through EvalPythonExecTransformer while
          // Velox backend uses ColumnarArrowEvalPythonExec.
          if (
            !BackendsApiManager.getSettings.supportColumnarArrowUdf() ||
            !GlutenConfig.get.enableColumnarArrowUDF
          ) {
            EvalPythonExecTransformer(plan.udfs, plan.resultAttrs, child)
          } else {
            BackendsApiManager.getSparkPlanExecApiInstance.createColumnarArrowEvalPythonExec(
              plan.udfs,
              plan.resultAttrs,
              child,
              plan.evalType)
          }
        case plan: RangeExec =>
          ColumnarRangeBaseExec.from(plan)
        case plan: SampleExec =>
          val child = plan.child
          BackendsApiManager.getSparkPlanExecApiInstance.genSampleExecTransformer(
            plan.lowerBound,
            plan.upperBound,
            plan.withReplacement,
            plan.seed,
            child)
        case plan: RDDScanExec if RDDScanTransformer.isSupportRDDScanExec(plan) =>
          RDDScanTransformer.getRDDScanTransform(plan)
        case p if !p.isInstanceOf[GlutenPlan] =>
          logDebug(s"Transformation for ${p.getClass} is currently not supported.")
          p
        case other => other
      }
      if (result.ne(plan)) {
        logDebug(s"Columnar Processing for ${plan.getClass} is currently supported.")
      }
      result
    }

    /**
     * PySpark fixes an ordinary UDF's eval type when `functions.udf` is called. Applications that
     * define the UDF before constructing their SparkSession therefore cannot observe the session's
     * `spark.sql.execution.pythonUDF.arrow.enabled` setting. Normalize that physical-plan artifact
     * here, but only when the user explicitly enabled Arrow and every function/input/output type is
     * supported by the conservative scalar Arrow contract below.
     *
     * `SQL_ARROW_BATCHED_UDF` was added after some Spark versions that Gluten still cross-builds.
     * Resolve it reflectively so older profiles keep their existing row-UDF path.
     */
    private def arrowOptimizedScalarUdfs(udfs: Seq[PythonUDF]): Option[(Seq[PythonUDF], Int)] = {
      if (
        !SQLConf.get
          .getConfString("spark.sql.execution.pythonUDF.arrow.enabled", "false")
          .toBoolean ||
        !BackendsApiManager.getSettings.supportColumnarArrowUdf() ||
        !GlutenConfig.get.enableColumnarArrowUDF
      ) {
        return None
      }

      pythonScalarEvalTypes
        .filter {
          case (batchedEvalType, _) =>
            udfs.nonEmpty && udfs.forall(isArrowScalarUdf(_, batchedEvalType))
        }
        .map {
          case (_, arrowEvalType) =>
            udfs.map(rewriteScalarUdfEvalType(_, arrowEvalType)) -> arrowEvalType
        }
    }

    private def createArrowScalarExec(
        plan: BatchEvalPythonExec,
        arrowUdfs: Seq[PythonUDF],
        arrowEvalType: Int): SparkPlan = {
      val projectedInputs = new ArrayBuffer[(Expression, Alias)]

      def materializeInput(input: Expression): Expression = input match {
        case attribute: AttributeReference => attribute
        case other =>
          val alias = projectedInputs
            .find { case (existing, _) => existing.semanticEquals(other) }
            .map(_._2)
            .getOrElse {
              val created = Alias(other, s"_gluten_arrow_udf_input_${projectedInputs.size}")()
              projectedInputs += other -> created
              created
            }
          alias.toAttribute
      }

      def rewriteInputs(udf: PythonUDF): PythonUDF = {
        val rewrittenChildren = udf.children.map {
          case nested: PythonUDF => rewriteInputs(nested)
          case input => materializeInput(input)
        }
        udf.copy(children = rewrittenChildren)
      }

      val rewrittenUdfs = arrowUdfs.map(rewriteInputs)
      val arrowChild = if (projectedInputs.isEmpty) {
        plan.child
      } else {
        ProjectExecTransformer(plan.child.output ++ projectedInputs.map(_._2), plan.child)
      }
      val arrowExec =
        BackendsApiManager.getSparkPlanExecApiInstance.createColumnarArrowEvalPythonExec(
          rewrittenUdfs,
          plan.resultAttrs,
          arrowChild,
          arrowEvalType)

      if (projectedInputs.isEmpty) {
        arrowExec
      } else {
        // The pre-project columns are implementation details. Restore BatchEvalPythonExec's
        // original output so callers never observe them above the Arrow boundary.
        ProjectExecTransformer(plan.output, arrowExec)
      }
    }

    private def failStrictMppOnArrowNormalizationRejection(reason: String): Unit = {
      val conf = SQLConf.get
      if (
        conf.getConfString("spark.gluten.mpp.enabled", "false").toBoolean &&
        conf.getConfString("spark.gluten.mpp.failOnFallback", "false").toBoolean
      ) {
        // EvalPythonExecTransformer validation falls back to Spark's original
        // BatchEvalPythonExec, which cannot retain the transformer tag. Fail here while the
        // precise semantic reason is still available.
        throw new IllegalStateException(reason)
      }
    }

    private lazy val pythonScalarEvalTypes: Option[(Int, Int)] = Try {
      // PythonEvalType is private[spark] in some Spark releases even though the JVM methods are
      // public. Reflection keeps this cross-version source outside Spark's private namespace.
      val moduleClass =
        SparkReflectionUtil.classForName("org.apache.spark.api.python.PythonEvalType$")
      val module = moduleClass.getField("MODULE$").get(null)
      val batched = moduleClass.getMethod("SQL_BATCHED_UDF").invoke(module).asInstanceOf[Int]
      val arrow = moduleClass.getMethod("SQL_ARROW_BATCHED_UDF").invoke(module).asInstanceOf[Int]
      batched -> arrow
    }.toOption

    private def arrowScalarNormalizationRejection(udfs: Seq[PythonUDF]): Option[String] = {
      if (
        !SQLConf.get
          .getConfString("spark.sql.execution.pythonUDF.arrow.enabled", "false")
          .toBoolean ||
        !BackendsApiManager.getSettings.supportColumnarArrowUdf() ||
        !GlutenConfig.get.enableColumnarArrowUDF
      ) {
        return None
      }

      pythonScalarEvalTypes.flatMap {
        case (batchedEvalType, _) =>
          udfs.iterator
            .filter(udf => udf.evalType == batchedEvalType && isConservativeArrowType(udf.dataType))
            .flatMap(firstUnsafeArrowInput)
            .toSeq
            .headOption
            .map {
              inputReason =>
                "ordinary Arrow UDF semantic guard: input kept on row execution because changing " +
                  s"its eval type is unsafe ($inputReason). Set " +
                  s"$ARROW_SCALAR_NULL_PRESERVING_INPUT_CAPABILITY=true only when the active " +
                  "Spark Python runtime is verified to preserve Arrow input null validity."
            }
      }
    }

    private def firstUnsafeArrowInput(udf: PythonUDF): Option[String] = {
      udf.children.iterator
        .map {
          case _: PythonUDF =>
            Some("a chained row UDF can produce a nullable intermediate value")
          case input if input.nullable && !nullPreservingArrowScalarInputEnabled =>
            Some(s"nullable input ${input.sql}:${input.dataType.catalogString} may map None to NaN")
          case input if !isNonNullableArrowInputType(input.dataType) =>
            Some(s"input ${input.sql}:${input.dataType.catalogString} lacks a safe Arrow contract")
          case input if !input.deterministic =>
            Some(s"non-deterministic input ${input.sql} cannot be safely pre-projected")
          case input if !isConservativeArrowInputExpression(input) =>
            Some(s"input expression ${input.sql} is outside the conservative Arrow contract")
          case _ => None
        }
        .collectFirst { case Some(reason) => reason }
    }

    private def isArrowScalarUdf(udf: PythonUDF, batchedEvalType: Int): Boolean = {
      udf.evalType == batchedEvalType &&
      isConservativeArrowType(udf.dataType) &&
      udf.children.forall {
        // A chained row UDF may return null even when its original input is non-null. The Arrow
        // scalar protocol can expose that intermediate null as pandas NaN, so do not change the
        // eval type of a chain without a proven null-preserving contract.
        case _: PythonUDF => false
        // Spark 4 ordinary Arrow UDFs can expose a nullable SQL string as float NaN instead of
        // Python None. A row UDF is allowed to distinguish these values (for example,
        // `None if x is None else x.strip()`), hence rewriting a nullable input would silently
        // change semantics. Keep it on Spark's original row-UDF path unless the active runtime
        // explicitly advertises the null-preserving capability.
        case input =>
          (!input.nullable || nullPreservingArrowScalarInputEnabled) &&
          input.deterministic &&
          isNonNullableArrowInputType(input.dataType) &&
          isConservativeArrowInputExpression(input)
      }
    }

    private def nullPreservingArrowScalarInputEnabled: Boolean =
      SQLConf.get
        .getConfString(ARROW_SCALAR_NULL_PRESERVING_INPUT_CAPABILITY, "false")
        .toBoolean

    private def rewriteScalarUdfEvalType(udf: PythonUDF, evalType: Int): PythonUDF = {
      val rewrittenChildren = udf.children.map {
        case nested: PythonUDF => rewriteScalarUdfEvalType(nested, evalType)
        case input => input
      }
      udf.copy(children = rewrittenChildren, evalType = evalType)
    }

    // Keep this deliberately narrower than Arrow's full evolving type matrix. The engine can
    // widen the list with parity tests; unsupported types retain Spark's existing row-UDF path.
    private def isConservativeArrowType(dataType: DataType): Boolean = dataType match {
      case NullType | BooleanType | ByteType | ShortType | IntegerType | LongType | FloatType |
          DoubleType | StringType | BinaryType | DateType | TimestampType =>
        true
      case _: DecimalType => true
      case ArrayType(elementType, _) => isConservativeArrowType(elementType)
      case MapType(keyType, valueType, _) =>
        isConservativeArrowType(keyType) && isConservativeArrowType(valueType)
      case StructType(fields) => fields.forall(field => isConservativeArrowType(field.dataType))
      case _ => false
    }

    private def isNonNullableArrowInputType(dataType: DataType): Boolean = dataType match {
      case ArrayType(elementType, containsNull) =>
        !containsNull && isNonNullableArrowInputType(elementType)
      case MapType(keyType, valueType, valueContainsNull) =>
        !valueContainsNull &&
        isNonNullableArrowInputType(keyType) &&
        isNonNullableArrowInputType(valueType)
      case StructType(fields) =>
        fields.forall(field => !field.nullable && isNonNullableArrowInputType(field.dataType))
      case primitive => isConservativeArrowType(primitive)
    }
  }
}
