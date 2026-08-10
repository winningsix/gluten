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
import org.apache.gluten.execution.ValidatablePlan
import org.apache.gluten.extension.columnar.transition.Convention
import org.apache.gluten.metrics.GlutenTimeMetric
import org.apache.gluten.sql.shims.SparkShimLoader

import org.apache.spark.{broadcast, SparkException}
import org.apache.spark.launcher.SparkLauncher
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.catalyst.plans.physical.{BroadcastMode, BroadcastPartitioning, Partitioning}
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeExec, BroadcastExchangeLike}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.util.SparkFatalException

import java.util.UUID
import java.util.concurrent.{CompletableFuture, TimeoutException, TimeUnit}

import scala.concurrent.Promise
import scala.util.control.NonFatal

case class ColumnarBroadcastExchangeExec(mode: BroadcastMode, child: SparkPlan)
  extends BroadcastExchangeLike
  with ValidatablePlan {

  // Note: "metrics" is made transient to avoid sending driver-side metrics to tasks.
  @transient override lazy val metrics: Map[String, SQLMetric] =
    BackendsApiManager.getMetricsApiInstance.genColumnarBroadcastExchangeMetrics(sparkContext)

  /**
   * Marker check: after RANGE preparation and every fallback-capable validation succeeds,
   * FluxNativeQueryExec tags this exchange dead because Plan C single-task merge has inlined the
   * build subtree into the consumer fragment (see FluxJniWrapper "single-task merge BROADCAST, no
   * LocalPartition wrap"). The driver-side relationFuture collect is dead work in that path, and
   * keeping it leaks ColumnarBatchSerializeResult arrays into driver heap (~5 GB per Q14 iter, OOM
   * by iter 3). Stays false on the BSP fallback path, which still needs executeBroadcast for
   * non-FLUX BroadcastHashJoin.
   *
   * Implemented as a TreeNodeTag (not a transient var on the case class) so the marker survives
   * Catalyst plan transformations (`copy()`, `withNewChildren()`, `transform*()`) -- Spark
   * frequently rebuilds tree nodes during later columnar rules, and a per-instance var would
   * silently reset to false.
   */
  def isFluxSuppressed: Boolean =
    getTagValue(ColumnarBroadcastExchangeExec.FluxSuppressedTag).contains(true)

  /**
   * Skip Spark's eager `prepare()` launch while an enclosing FLUX query is still validating.
   *
   * Unlike [[isFluxSuppressed]], this marker is reversible in behavior: `doExecuteBroadcast` and
   * direct access to [[relationFuture]] still start the broadcast on demand. This lets RANGE
   * sampling and BSP fallback consume a live exchange without also running dead broadcast jobs for
   * queries that commit to native FLUX.
   */
  def isFluxPrepareDeferred: Boolean =
    getTagValue(ColumnarBroadcastExchangeExec.FluxPrepareDeferredTag).contains(true)

  def deferPrepareForFluxNativeExecution(): Boolean = synchronized {
    val newlyMarked = !isFluxPrepareDeferred
    setTagValue(ColumnarBroadcastExchangeExec.FluxPrepareDeferredTag, true)
    newlyMarked
  }

  @transient
  private lazy val promise = Promise[broadcast.Broadcast[Any]]()

  @transient
  lazy val completionFuture: scala.concurrent.Future[broadcast.Broadcast[Any]] =
    promise.future

  @transient
  @volatile
  private var relationFutureRef: java.util.concurrent.Future[broadcast.Broadcast[Any]] = _

  @transient
  override lazy val relationFuture: java.util.concurrent.Future[broadcast.Broadcast[Any]] = {
    val future =
      if (isFluxSuppressed) {
        failedFluxSuppressedFuture()
      } else {
        SQLExecution.withThreadLocalCaptured[broadcast.Broadcast[Any]](
          session,
          BroadcastExchangeExec.executionContext) {
          try {
            SparkShimLoader.getSparkShims
              .setJobDescriptionOrTagForBroadcastExchange(sparkContext, this)
            val relation = GlutenTimeMetric.millis(longMetric("collectTime")) {
              _ =>
                // this created relation ignore HashedRelationBroadcastMode isNullAware, because we
                // cannot get child output rows, then compare the hash key is null, if not null,
                // compare the isNullAware, so gluten will not generate
                // HashedRelationWithAllNullKeys
                // or EmptyHashedRelation, this difference will cause performance regression in some
                // cases.
                // For the above reason, the same implementation can be used for both
                // HashedRelationBroadcastMode as well as IdentityBroadcastMode.
                BackendsApiManager.getSparkPlanExecApiInstance.createBroadcastRelation(
                  mode,
                  child,
                  longMetric("numOutputRows"),
                  longMetric("dataSize"))
            }

            val broadcasted = GlutenTimeMetric.millis(longMetric("broadcastTime")) {
              _ =>
                // Broadcast the relation
                SparkShimLoader.getSparkShims.broadcastInternal(
                  sparkContext,
                  relation.asInstanceOf[Any])
            }

            // Update driver metrics
            val executionId = sparkContext.getLocalProperty(SQLExecution.EXECUTION_ID_KEY)
            SQLMetrics.postDriverMetricUpdates(sparkContext, executionId, metrics.values.toSeq)

            promise.success(broadcasted)
            broadcasted
          } catch {
            // SPARK-24294: To bypass scala bug: https://github.com/scala/bug/issues/9554, we throw
            // SparkFatalException, which is a subclass of Exception. ThreadUtils.awaitResult
            // will catch this exception and re-throw the wrapped fatal throwable.
            case oe: OutOfMemoryError =>
              val ex = new SparkFatalException(
                new OutOfMemoryError(
                  "Not enough memory to build and broadcast the table to all " +
                    "worker nodes. As a workaround, you can either disable broadcast by setting " +
                    s"${SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key} to -1 or increase the spark " +
                    s"driver memory by setting ${SparkLauncher.DRIVER_MEMORY} to a higher value.")
                  .initCause(oe.getCause))
              promise.failure(ex)
              throw ex
            case e if !NonFatal(e) =>
              val ex = new SparkFatalException(e)
              promise.failure(ex)
              throw ex
            case e: Throwable =>
              promise.failure(e)
              throw e
          }
        }
      }
    relationFutureRef = future
    future
  }

  def suppressForFluxNativeExecution(): Boolean = synchronized {
    val newlyMarked = !isFluxSuppressed
    setTagValue(ColumnarBroadcastExchangeExec.FluxSuppressedTag, true)

    val ex = fluxSuppressedException
    promise.tryFailure(ex)
    Option(relationFutureRef).foreach {
      future =>
        if (!future.isDone) {
          SparkShimLoader.getSparkShims.cancelJobGroupForBroadcastExchange(sparkContext, this)
          future.cancel(true)
        }
    }
    newlyMarked
  }

  private def failedFluxSuppressedFuture()
      : java.util.concurrent.Future[broadcast.Broadcast[Any]] = {
    val ex = fluxSuppressedException
    promise.tryFailure(ex)
    val failed = new CompletableFuture[broadcast.Broadcast[Any]]()
    failed.completeExceptionally(ex)
    failed
  }

  private def fluxSuppressedException: IllegalStateException = {
    new IllegalStateException(
      "ColumnarBroadcastExchangeExec is marked fluxSuppressed and will not start " +
        "driver-side relationFuture collection.")
  }

  override val runId: UUID = UUID.randomUUID

  @transient
  private val timeout: Long = SQLConf.get.broadcastTimeout

  override def output: Seq[Attribute] = child.output

  override def outputPartitioning: Partitioning = BroadcastPartitioning(mode)

  override def batchType(): Convention.BatchType = BackendsApiManager.getSettings.primaryBatchType

  override def rowType0(): Convention.RowType = Convention.RowType.None

  override def doCanonicalize(): SparkPlan = {
    val canonicalized =
      BackendsApiManager.getSparkPlanExecApiInstance.doCanonicalizeForBroadcastMode(mode)
    ColumnarBroadcastExchangeExec(canonicalized, child.canonicalized)
  }

  override def doPrepare(): Unit = {
    // Skip the driver-side build-side collect when an enclosing
    // FluxNativeQueryExec has inlined this build subtree into its consumer
    // fragment via single-task merge. See [[isFluxSuppressed]] for the full
    // rationale and FluxNativeQueryExec for the commit point.
    if (isFluxSuppressed || isFluxPrepareDeferred) return
    relationFuture
  }

  override def doExecute(): RDD[InternalRow] = {
    throw new UnsupportedOperationException(
      "ColumnarBroadcastExchange does not support the execute() code path.")
  }

  override protected[sql] def doExecuteBroadcast[T](): broadcast.Broadcast[T] = {
    // A suppressed exchange must never be consumed via doExecuteBroadcast: the
    // build subtree has been inlined into a native FLUX fragment and no one is
    // supposed to await the broadcast variable. Fail loudly so a future
    // re-enable of fused-broadcast paths or an unexpected BSP consumer surface
    // here instead of returning null / a stale promise.
    if (isFluxSuppressed) {
      throw new IllegalStateException(
        "ColumnarBroadcastExchangeExec is marked fluxSuppressed (build inlined " +
          "into FluxNativeQueryExec consumer fragment) and cannot serve " +
          "doExecuteBroadcast.")
    }
    try {
      relationFuture.get(timeout, TimeUnit.SECONDS).asInstanceOf[broadcast.Broadcast[T]]
    } catch {
      case ex: TimeoutException =>
        logError(s"Could not execute broadcast in $timeout secs.", ex)
        if (!relationFuture.isDone) {
          SparkShimLoader.getSparkShims.cancelJobGroupForBroadcastExchange(sparkContext, this)
          relationFuture.cancel(true)
        }
        throw new SparkException(
          s"""
             |Could not execute broadcast in $timeout secs.
             |You can increase the timeout for broadcasts via
             |${SQLConf.BROADCAST_TIMEOUT.key} or disable broadcast join
             |by setting ${SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key} to -1
            """.stripMargin,
          ex
        )
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): ColumnarBroadcastExchangeExec =
    copy(child = newChild)

  // Ported from BroadcastExchangeExec
  override def runtimeStatistics: Statistics = {
    val dataSize = metrics("dataSize").value
    val rowCount = metrics("numOutputRows").value
    Statistics(dataSize, Some(rowCount))
  }
}

object ColumnarBroadcastExchangeExec {

  /**
   * Prevent eager `doPrepare` for a broadcast below FLUX while keeping lazy broadcast execution
   * available to RANGE sampling and BSP fallback.
   */
  val FluxPrepareDeferredTag: TreeNodeTag[Boolean] =
    TreeNodeTag[Boolean]("flux.prepareDeferred")

  /**
   * Plan-tree-attached marker committed by `FluxNativeQueryExec` after fallback-capable validation
   * to indicate this exchange is dead work because the enclosing native fragment has inlined the
   * build subtree into its consumer. See `ColumnarBroadcastExchangeExec.isFluxSuppressed` for full
   * rationale.
   *
   * Using a `TreeNodeTag` (rather than a non-ctor `var` on the case class) means the marker
   * survives `copy()` / `withNewChildren()` / `transform*()` because `TreeNode.copyTagsFrom`
   * carries tags through plan rewrites.
   */
  val FluxSuppressedTag: TreeNodeTag[Boolean] = TreeNodeTag[Boolean]("flux.suppressed")
}
