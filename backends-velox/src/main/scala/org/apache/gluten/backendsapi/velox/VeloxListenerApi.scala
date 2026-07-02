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

import org.apache.gluten.backendsapi.ListenerApi
import org.apache.gluten.backendsapi.arrow.ArrowBatchTypes.{ArrowJavaBatchType, ArrowNativeBatchType}
import org.apache.gluten.config.{GlutenConfig, GlutenCoreConfig, VeloxConfig}
import org.apache.gluten.config.VeloxConfig._
import org.apache.gluten.execution.datasource.GlutenFormatFactory
import org.apache.gluten.expression.UDFMappings
import org.apache.gluten.extension.columnar.transition.Convention
import org.apache.gluten.gpu.GpuMemoryTrackerJniWrapper
import org.apache.gluten.init.NativeBackendInitializer
import org.apache.gluten.jni.{JniLibLoader, JniWorkspace}
import org.apache.gluten.memory.{MemoryUsageRecorder, SimpleMemoryUsageRecorder}
import org.apache.gluten.memory.listener.ReservationListener
import org.apache.gluten.memory.memtarget.MemoryTarget
import org.apache.gluten.monitor.VeloxMemoryProfiler
import org.apache.gluten.mpp.control.GlutenMppDriverService
import org.apache.gluten.mpp.control.GlutenMppExecutorService
import org.apache.gluten.mpp.control.GlutenMppQueryControlListener
import org.apache.gluten.udf.UdfJniWrapper
import org.apache.gluten.utils._

import org.apache.spark.{HdfsConfGenerator, ShuffleDependency, SparkConf, SparkContext}
import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.internal.Logging
import org.apache.spark.memory.GlobalOffHeapMemory
import org.apache.spark.network.util.ByteUnit
import org.apache.spark.shuffle.{ColumnarShuffleDependency, LookupKey, ShuffleManagerRegistry}
import org.apache.spark.shuffle.sort.ColumnarShuffleManager
import org.apache.spark.sql.execution.ColumnarCachedBatchSerializer
import org.apache.spark.sql.execution.datasources.GlutenWriterColumnarRules
import org.apache.spark.sql.execution.datasources.velox.{VeloxParquetWriterInjects, VeloxRowSplitter}
import org.apache.spark.sql.expression.UDFResolver
import org.apache.spark.sql.internal.{GlutenConfigUtil, StaticSQLConf}
import org.apache.spark.sql.internal.SparkConfigUtil._
import org.apache.spark.util.{SparkDirectoryUtil, SparkResourceUtil, SparkShutdownManagerUtil}

import org.apache.commons.lang3.StringUtils

import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

class VeloxListenerApi extends ListenerApi with Logging {
  import VeloxListenerApi._

  override def onDriverStart(sc: SparkContext, pc: PluginContext): Unit = {
    val conf = pc.conf()

    // When the Velox cache is enabled, the Velox file handle cache should also be enabled.
    // Otherwise, a 'reference id not found' error may occur.
    if (
      conf.get(COLUMNAR_VELOX_CACHE_ENABLED) &&
      !conf.get(COLUMNAR_VELOX_FILE_HANDLE_CACHE_ENABLED)
    ) {
      throw new IllegalArgumentException(
        s"${COLUMNAR_VELOX_CACHE_ENABLED.key} and " +
          s"${COLUMNAR_VELOX_FILE_HANDLE_CACHE_ENABLED.key} should be enabled together.")
    }

    if (
      conf.get(COLUMNAR_VELOX_CACHE_ENABLED) &&
      !conf.get(GlutenConfig.GLUTEN_SOFT_AFFINITY_ENABLED)
    ) {
      logWarning(
        s"It's recommened to enable ${GlutenConfig.GLUTEN_SOFT_AFFINITY_ENABLED.key} when " +
          s"${COLUMNAR_VELOX_CACHE_ENABLED.key} is set to get better locality.")
    }

    if (conf.get(COLUMNAR_VELOX_CACHE_ENABLED) && conf.get(LOAD_QUANTUM) > 8 * 1024 * 1024) {
      throw new IllegalArgumentException(
        s"Velox currently only support up to 8MB load quantum size " +
          s"on SSD cache enabled by ${COLUMNAR_VELOX_CACHE_ENABLED.key}, " +
          s"User can set ${LOAD_QUANTUM.key} <= 8MB skip this error.")
    }

    if (conf.contains(DIRECTORY_SIZE_GUESS.key)) {
      logWarning(
        s"${DIRECTORY_SIZE_GUESS.key} is Deprecated " +
          s"replacing it with ${FOOTER_ESTIMATED_SIZE.key} instead.")
    }

    // Generate HDFS client configurations.
    HdfsConfGenerator.addHdfsClientToSparkWorkDirectory(sc)

    // Overhead memory limits.
    val offHeapSize = conf.getSizeAsBytes(GlutenCoreConfig.SPARK_OFFHEAP_SIZE_KEY)
    val desiredOverheadSize = (0.3 * offHeapSize).toLong.max(ByteUnit.MiB.toBytes(384))
    if (!SparkResourceUtil.isMemoryOverheadSet(conf)) {
      // If memory overhead is not set by user, automatically set it according to off-heap settings.
      logInfo(
        s"Memory overhead is not set. Setting it to $desiredOverheadSize automatically." +
          " Gluten doesn't follow Spark's calculation on default value of this option because the" +
          " actual required memory overhead will depend on off-heap usage than on on-heap usage.")
      conf.set(
        GlutenConfig.SPARK_OVERHEAD_SIZE_KEY,
        ByteUnit.BYTE.toMiB(desiredOverheadSize).toString)
    }
    val overheadSize: Long = SparkResourceUtil.getMemoryOverheadSize(conf)
    if (ByteUnit.BYTE.toMiB(overheadSize) < ByteUnit.BYTE.toMiB(desiredOverheadSize)) {
      logWarning(
        s"Memory overhead is set to ${ByteUnit.BYTE.toMiB(overheadSize)}MiB which is smaller than" +
          s" the recommended size ${ByteUnit.BYTE.toMiB(desiredOverheadSize)}MiB." +
          s" This may cause OOM.")
    }
    conf.set(GlutenCoreConfig.COLUMNAR_OVERHEAD_SIZE_IN_BYTES, overheadSize)

    // Sql table cache serializer.
    if (conf.get(GlutenConfig.COLUMNAR_TABLE_CACHE_ENABLED)) {
      conf.set(StaticSQLConf.SPARK_CACHE_SERIALIZER, classOf[ColumnarCachedBatchSerializer].getName)
    }

    // Static initializers for driver.
    if (!driverInitialized.compareAndSet(false, true)) {
      // Make sure we call the static initializers only once.
      logInfo(
        "Skip rerunning static initializers since they are only supposed to run once." +
          " You see this message probably because you are creating a new SparkSession.")
      return
    }

    SparkDirectoryUtil.init(conf)
    initialize(conf, isDriver = true)
    UdfJniWrapper.registerFunctionSignatures()

    if (inLocalMode(conf)) {
      initializeGpuConcurrency(conf)
    }
    GlutenMppDriverService.init(conf)
    GlutenMppDriverService.get().foreach {
      service => sc.addSparkListener(new GlutenMppQueryControlListener(service))
    }
  }

  override def onDriverShutdown(): Unit = {
    GlutenMppDriverService.shutdown()
    try {
      GpuMemoryTrackerJniWrapper.shutdown()
    } catch {
      case _: UnsatisfiedLinkError =>
    }
    shutdown()
  }

  override def onExecutorStart(pc: PluginContext): Unit = {
    val conf = pc.conf()

    // Static initializers for executor.
    if (!executorInitialized.compareAndSet(false, true)) {
      // Make sure we call the static initializers only once.
      logInfo(
        "Skip rerunning static initializers since they are only supposed to run once." +
          " You see this message probably because you are creating a new SparkSession.")
      return
    }
    if (inLocalMode(conf)) {
      // Don't do static initializations from executor side in local mode.
      // Driver already did that.
      logInfo(
        "Gluten is running with Spark local mode. Skip running static initializer for executor.")
      GlutenMppExecutorService.onExecutorStart(pc)
      return
    }

    SparkDirectoryUtil.init(conf)
    initialize(conf, isDriver = false)
    initializeGpuConcurrency(conf)
    addIfNeedMemoryDumpShutdownHook(conf)
    GlutenMppExecutorService.onExecutorStart(pc)
  }

  override def onExecutorShutdown(): Unit = {
    GlutenMppExecutorService.onExecutorShutdown()
    try {
      GpuMemoryTrackerJniWrapper.shutdown()
    } catch {
      case _: UnsatisfiedLinkError =>
    }
    shutdown()
  }

  private def initialize(conf: SparkConf, isDriver: Boolean): Unit = {
    // Sets this configuration only once, since not undoable.
    // DebugInstance should be created first.
    if (conf.get(GlutenConfig.DEBUG_KEEP_JNI_WORKSPACE)) {
      val debugDir = conf.get(GlutenConfig.DEBUG_KEEP_JNI_WORKSPACE_DIR)
      JniWorkspace.enableDebug(debugDir)
    } else {
      JniWorkspace.initializeDefault(
        () =>
          SparkDirectoryUtil.get
            .namespace("jni")
            .mkChildDirRandomly(UUID.randomUUID.toString)
            .getAbsolutePath)
    }

    UDFResolver.resolveUdfConf(conf, isDriver)

    // Do row / batch type initializations.
    Convention.ensureSparkRowAndBatchTypesRegistered()
    ArrowJavaBatchType.ensureRegistered()
    ArrowNativeBatchType.ensureRegistered()
    VeloxBatchType.ensureRegistered()
    VeloxCarrierRowType.ensureRegistered()

    // Register columnar shuffle so can be considered when
    // `org.apache.spark.shuffle.GlutenShuffleManager` is set as Spark shuffle manager.
    ShuffleManagerRegistry
      .get()
      .register(
        new LookupKey {
          override def accepts[K, V, C](dependency: ShuffleDependency[K, V, C]): Boolean = {
            dependency.getClass == classOf[ColumnarShuffleDependency[_, _, _]]
          }
        },
        classOf[ColumnarShuffleManager].getName
      )

    // Set the system properties.
    // Use appending policy for children with the same name in a arrow struct vector.
    System.setProperty("arrow.struct.conflict.policy", "CONFLICT_APPEND")

    // Load supported hive/python/scala udfs
    UDFMappings.loadFromSparkConf(conf)

    // Initial library loader.
    val loader = JniWorkspace.getDefault.libLoader

    // Load shared native libraries the backend libraries depend on.
    SharedLibraryLoaderUtils.load(conf, loader)

    // Load the combined native library (libgluten.so contains both core and
    // backend symbols, including MPP JNI).
    val libPath = conf.get(GlutenConfig.GLUTEN_LIB_PATH)
    if (StringUtils.isBlank(libPath)) {
      val baseLibName = conf.get(GlutenConfig.GLUTEN_LIB_NAME)
      loader.load(s"$platformLibDir/${System.mapLibraryName(baseLibName)}")
    } else {
      // Path based load. Ignore all other loaders.
      JniLibLoader.loadFromPath(libPath)
    }

    // Initial native backend with configurations.
    NativeBackendInitializer
      .forBackend(VeloxBackend.BACKEND_NAME)
      .initialize(newGlobalOffHeapMemoryListener(), parseConf(conf, isDriver))

    // Inject backend-specific implementations to override spark classes.
    GlutenFormatFactory.register(new VeloxParquetWriterInjects)
    GlutenFormatFactory.injectPostRuleFactory(
      session => GlutenWriterColumnarRules.NativeWritePostRule(session))
    GlutenFormatFactory.register(new VeloxRowSplitter())
  }

  private def initializeGpuConcurrency(conf: SparkConf): Unit = {
    val cudfEnabled = conf.getBoolean(GlutenConfig.COLUMNAR_CUDF_ENABLED.key, false)
    if (!cudfEnabled) {
      return
    }

    val gpuMemorySize = conf
      .getOption(CUDF_GPU_MEMORY_SIZE.key)
      .map(_.toLong)
      .getOrElse {
        val memPercent = conf.getInt(CUDF_MEMORY_PERCENT.key, 50)
        val detectedGpuMem =
          try {
            val total = GpuMemoryTrackerJniWrapper.getDeviceMemorySize()
            if (total > 0) {
              logInfo(s"Detected GPU device memory: ${total / (1024 * 1024)}MB")
              total
            } else {
              logWarning("GPU memory detection returned 0, falling back to 16GB estimate")
              16L * 1024 * 1024 * 1024
            }
          } catch {
            case e: UnsatisfiedLinkError =>
              logWarning(
                s"getDeviceMemorySize JNI not available, falling back to 16GB estimate: " +
                  s"${e.getMessage}")
              16L * 1024 * 1024 * 1024
          }
        detectedGpuMem * memPercent / 100
      }

    val gpuBatchBytes =
      conf.getLong(CUDF_GPU_TARGET_BATCH_BYTES.key, CUDF_GPU_TARGET_BATCH_BYTES.defaultValue.get)
    val concurrentTasks = conf.getOption(CUDF_CONCURRENT_GPU_TASKS.key).map(_.toInt)
    val maxConcurrent = concurrentTasks.getOrElse {
      math.max(1, math.min(4, gpuMemorySize / (4 * math.max(gpuBatchBytes, 1)))).toInt
    }

    try {
      GpuMemoryTrackerJniWrapper.setMaxConcurrentGpuTasks(maxConcurrent)
      logInfo(
        s"C++ GpuLock configured: maxConcurrent=$maxConcurrent, " +
          s"gpuMemory=${gpuMemorySize / (1024 * 1024)}MB, " +
          s"batchBytes=${gpuBatchBytes / (1024 * 1024)}MB")
    } catch {
      case e: UnsatisfiedLinkError =>
        logWarning(
          s"setMaxConcurrentGpuTasks JNI not available, using default (serial): ${e.getMessage}")
    }

    try {
      GpuMemoryTrackerJniWrapper.initialize()
      logInfo("Native GpuMemoryTracker initialized")
    } catch {
      case e: UnsatisfiedLinkError =>
        logWarning(s"GpuMemoryTracker JNI not available: ${e.getMessage}")
    }
  }

  private def addIfNeedMemoryDumpShutdownHook(conf: SparkConf): Unit = {
    val memoryDumpOnExit = conf.get(MEMORY_DUMP_ON_EXIT)
    if (memoryDumpOnExit) {
      SparkShutdownManagerUtil.addHook(
        () => {
          logInfo("MemoryDumpOnExit triggered, dumping memory profile.")
          VeloxMemoryProfiler.dump()
          logInfo("MemoryDumpOnExit completed.")
        })
    }
  }

  private def shutdown(): Unit = {
    // TODO shutdown implementation in velox to release resources
  }
}

object VeloxListenerApi {
  // TODO: Implement graceful shutdown and remove these flags.
  //  As spark conf may change when active Spark session is recreated.
  private val driverInitialized: AtomicBoolean = new AtomicBoolean(false)
  private val executorInitialized: AtomicBoolean = new AtomicBoolean(false)
  private val platformLibDir: String = {
    val osName = System.getProperty("os.name") match {
      case n if n.contains("Linux") => "linux"
      case n if n.contains("Mac") => "darwin"
      case _ =>
        // Default to linux
        "linux"
    }
    val arch = System.getProperty("os.arch")
    s"$osName/$arch"
  }

  private def inLocalMode(conf: SparkConf): Boolean = {
    SparkResourceUtil.isLocalMaster(conf)
  }

  private def newGlobalOffHeapMemoryListener(): ReservationListener = {
    new ReservationListener {
      private val recorder: MemoryUsageRecorder = new SimpleMemoryUsageRecorder()
      private val target: MemoryTarget = GlobalOffHeapMemory.target

      override def reserve(size: Long): Long = {
        assert(target.borrow(size) == size)
        recorder.inc(size)
        size
      }

      override def unreserve(size: Long): Long = {
        assert(target.repay(size) == size)
        recorder.inc(-size)
        size
      }

      override def getUsedBytes: Long = {
        recorder.current()
      }
    }
  }

  def parseConf(conf: SparkConf, isDriver: Boolean): Map[String, String] = {
    // Ensure velox conf registered.
    VeloxConfig.get

    var parsed: Map[String, String] = GlutenConfigUtil.parseConfig(conf.getAll.toMap)

    // Workaround for https://github.com/apache/incubator-gluten/issues/7837
    if (isDriver && !inLocalMode(conf)) {
      parsed += (COLUMNAR_VELOX_CACHE_ENABLED.key -> "false")
      // Disable cuDF on driver in cluster mode to prevent GPU/RMM initialization
      // (cudaMallocAsync) on driver nodes that may not have a GPU.
      // GPU init should only happen on executors.
      parsed += (GlutenConfig.COLUMNAR_CUDF_ENABLED.key -> "false")
    }

    parsed
  }
}
