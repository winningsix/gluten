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

#pragma once

#include "config/GlutenConfig.h"

namespace gluten {
// memory
const std::string kSpillStrategy = "spark.gluten.sql.columnar.backend.velox.spillStrategy";
const std::string kSpillStrategyDefaultValue = "auto";
const std::string kSpillThreadNum = "spark.gluten.sql.columnar.backend.velox.spillThreadNum";
const uint32_t kSpillThreadNumDefaultValue = 0;
const std::string kAggregationSpillEnabled = "spark.gluten.sql.columnar.backend.velox.aggregationSpillEnabled";
const std::string kJoinSpillEnabled = "spark.gluten.sql.columnar.backend.velox.joinSpillEnabled";
const std::string kOrderBySpillEnabled = "spark.gluten.sql.columnar.backend.velox.orderBySpillEnabled";
const std::string kWindowSpillEnabled = "spark.gluten.sql.columnar.backend.velox.windowSpillEnabled";

// spill config
// refer to
// https://github.com/facebookincubator/velox/blob/95f3e80e77d046c12fbc79dc529366be402e9c2b/velox/docs/configs.rst#spilling
const std::string kMaxSpillLevel = "spark.gluten.sql.columnar.backend.velox.maxSpillLevel";
const std::string kMaxSpillFileSize = "spark.gluten.sql.columnar.backend.velox.maxSpillFileSize";
const std::string kSpillStartPartitionBit = "spark.gluten.sql.columnar.backend.velox.spillStartPartitionBit";
const std::string kSpillPartitionBits = "spark.gluten.sql.columnar.backend.velox.spillPartitionBits";
const std::string kMaxSpillRunRows = "spark.gluten.sql.columnar.backend.velox.MaxSpillRunRows";
const std::string kMaxSpillBytes = "spark.gluten.sql.columnar.backend.velox.MaxSpillBytes";
const std::string kSpillReadBufferSize = "spark.unsafe.sorter.spill.reader.buffer.size";
const uint64_t kMaxSpillFileSizeDefault = 1L * 1024 * 1024 * 1024;

const std::string kSpillableReservationGrowthPct =
    "spark.gluten.sql.columnar.backend.velox.spillableReservationGrowthPct";
const std::string kSpillPrefixSortEnabled = "spark.gluten.sql.columnar.backend.velox.spillPrefixsortEnabled";
// Whether to compress data spilled. Compression will use spark.io.compression.codec or kSpillCompressionKind.
const std::string kSparkShuffleSpillCompress = "spark.shuffle.spill.compress";
const std::string kCompressionKind = "spark.io.compression.codec";
/// The compression codec to use for spilling. Use kCompressionKind if not set.
const std::string kSpillCompressionKind = "spark.gluten.sql.columnar.backend.velox.spillCompressionCodec";
const std::string kMaxPartialAggregationMemoryRatio =
    "spark.gluten.sql.columnar.backend.velox.maxPartialAggregationMemoryRatio";
const std::string kMaxPartialAggregationMemory = "spark.gluten.sql.columnar.backend.velox.maxPartialAggregationMemory";
const std::string kMaxExtendedPartialAggregationMemoryRatio =
    "spark.gluten.sql.columnar.backend.velox.maxExtendedPartialAggregationMemoryRatio";
const std::string kMaxExtendedPartialAggregationMemory =
    "spark.gluten.sql.columnar.backend.velox.maxExtendedPartialAggregationMemory";
const std::string kAbandonPartialAggregationMinPct =
    "spark.gluten.sql.columnar.backend.velox.abandonPartialAggregationMinPct";
const std::string kAbandonPartialAggregationMinRows =
    "spark.gluten.sql.columnar.backend.velox.abandonPartialAggregationMinRows";

// hashmap build
const std::string kAbandonDedupHashMapMinRows = "spark.gluten.velox.abandonDedupHashMap.minRows";
const std::string kAbandonDedupHashMapMinPct = "spark.gluten.velox.abandonDedupHashMap.minPct";

// execution
const std::string kBloomFilterExpectedNumItems = "spark.gluten.sql.columnar.backend.velox.bloomFilter.expectedNumItems";
const std::string kBloomFilterNumBits = "spark.gluten.sql.columnar.backend.velox.bloomFilter.numBits";
const std::string kBloomFilterMaxNumBits = "spark.gluten.sql.columnar.backend.velox.bloomFilter.maxNumBits";
const std::string kVeloxSplitPreloadPerDriver = "spark.gluten.sql.columnar.backend.velox.SplitPreloadPerDriver";

const std::string kHashProbeDynamicFilterPushdownEnabled =
    "spark.gluten.sql.columnar.backend.velox.hashProbe.dynamicFilterPushdown.enabled";

const std::string kHashProbeBloomFilterPushdownMaxSize =
    "spark.gluten.sql.columnar.backend.velox.hashProbe.bloomFilterPushdown.maxSize";

const std::string kShowTaskMetricsWhenFinished = "spark.gluten.sql.columnar.backend.velox.showTaskMetricsWhenFinished";
const bool kShowTaskMetricsWhenFinishedDefault = false;

const std::string kTaskMetricsToEventLogThreshold =
    "spark.gluten.sql.columnar.backend.velox.taskMetricsToEventLog.threshold";
const int64_t kTaskMetricsToEventLogThresholdDefault = -1;

const std::string kEnableUserExceptionStacktrace =
    "spark.gluten.sql.columnar.backend.velox.enableUserExceptionStacktrace";
const bool kEnableUserExceptionStacktraceDefault = true;

const std::string kEnableSystemExceptionStacktrace =
    "spark.gluten.sql.columnar.backend.velox.enableSystemExceptionStacktrace";
const bool kEnableSystemExceptionStacktraceDefault = true;

const std::string kMemoryUseHugePages = "spark.gluten.sql.columnar.backend.velox.memoryUseHugePages";
const bool kMemoryUseHugePagesDefault = false;

const std::string kVeloxMemInitCapacity = "spark.gluten.sql.columnar.backend.velox.memInitCapacity";
const uint64_t kVeloxMemInitCapacityDefault = 8 << 20;

const std::string kVeloxMemReclaimMaxWaitMs = "spark.gluten.sql.columnar.backend.velox.reclaimMaxWaitMs";
const uint64_t kVeloxMemReclaimMaxWaitMsDefault = 3600000; // 60min

const std::string kHiveConnectorId = "test-hive";
const std::string kVeloxCacheEnabled = "spark.gluten.sql.columnar.backend.velox.cacheEnabled";

const std::string kExprMaxCompiledRegexes = "spark.gluten.sql.columnar.backend.velox.maxCompiledRegexes";

// memory cache
const std::string kVeloxMemCacheSize = "spark.gluten.sql.columnar.backend.velox.memCacheSize";
const uint64_t kVeloxMemCacheSizeDefault = 1073741824; // 1G

// ssd cache
const std::string kVeloxSsdCacheSize = "spark.gluten.sql.columnar.backend.velox.ssdCacheSize";
const uint64_t kVeloxSsdCacheSizeDefault = 1073741824; // 1G
const std::string kVeloxSsdCachePath = "spark.gluten.sql.columnar.backend.velox.ssdCachePath";
const std::string kVeloxSsdCachePathDefault = "/tmp/";
const std::string kVeloxSsdCacheShards = "spark.gluten.sql.columnar.backend.velox.ssdCacheShards";
const uint32_t kVeloxSsdCacheShardsDefault = 1;
const std::string kVeloxSsdCacheIOThreads = "spark.gluten.sql.columnar.backend.velox.ssdCacheIOThreads";
const uint32_t kVeloxSsdCacheIOThreadsDefault = 1;
const std::string kVeloxSsdODirectEnabled = "spark.gluten.sql.columnar.backend.velox.ssdODirect";
const std::string kVeloxSsdCheckpointIntervalBytes =
    "spark.gluten.sql.columnar.backend.velox.ssdCheckpointIntervalBytes";
const std::string kVeloxSsdDisableFileCow = "spark.gluten.sql.columnar.backend.velox.ssdDisableFileCow";
const std::string kVeloxSsdCheckSumEnabled = "spark.gluten.sql.columnar.backend.velox.ssdChecksumEnabled";
const std::string kVeloxSsdCheckSumReadVerificationEnabled =
    "spark.gluten.sql.columnar.backend.velox.ssdChecksumReadVerificationEnabled";

// async
const std::string kVeloxIOThreads = "spark.gluten.sql.columnar.backend.velox.IOThreads";
const uint32_t kVeloxIOThreadsDefault = 0;

// MPP local mode: collapse all fragments into one Velox Task connected via
// LocalPartitionNode (intra-task), bypassing UcxExchange entirely. This is the
// default path for the single-worker/local-task GPU backend. It supports
// SINGLE/HASH/RANGE/ROUND_ROBIN via LocalPartition and BROADCAST by inlining the
// producer plan.
const std::string kMppSingleTaskMode =
    "spark.gluten.sql.columnar.backend.velox.mpp.singleTaskMode";
const bool kMppSingleTaskModeDefault = true;

// Per-fragment UCX output queue high-water mark.  A multi-fragment query can
// have dozens of producers alive at once, so the old fixed 1 GiB allowance per
// fragment can exhaust a GPU long before backpressure engages.
const std::string kMppMaxOutputBufferSize =
    "spark.gluten.sql.columnar.backend.velox.mpp.maxOutputBufferSize";
const uint64_t kMppMaxOutputBufferSizeDefault = 1L << 30;

// Cap on per-task driver count (= local-partition lane count) when single-
// task mode is active. Mirrors IBM's pbench GPU deployment choice
// (velox-testing/.../generate_presto_config.sh sets VCPU_PER_WORKER=2 for
// GPU variant, which becomes task.max-drivers-per-task=2 in
// config_native.properties). The reason: cuDF GPU operators saturate one
// stream pretty effectively, so adding more drivers per task adds
// contention (RMM mutex, cuda runtime) more than parallelism. Same number
// also caps how many partitions a HASH/RANGE LocalPartitionNode emits in
// single-task mode (= consumer pipeline driver count).
const std::string kMppSingleTaskMaxDrivers =
    "spark.gluten.sql.columnar.backend.velox.mpp.singleTaskMaxDrivers";
const int32_t kMppSingleTaskMaxDriversDefault = 2;
const std::string kVeloxAsyncTimeoutOnTaskStopping =
    "spark.gluten.sql.columnar.backend.velox.asyncTimeoutOnTaskStopping";
const int32_t kVeloxAsyncTimeoutOnTaskStoppingDefault = 30000; // 30s

// udf
const std::string kVeloxUdfLibraryPaths = "spark.gluten.sql.columnar.backend.velox.internal.udfLibraryPaths";

// VeloxShuffleReader print flag.
const std::string kVeloxShuffleReaderPrintFlag = "spark.gluten.velox.shuffleReaderPrintFlag";

const std::string kVeloxFileHandleCacheEnabled = "spark.gluten.sql.columnar.backend.velox.fileHandleCacheEnabled";
const bool kVeloxFileHandleCacheEnabledDefault = false;

/* configs for file read in velox*/
const std::string kDirectorySizeGuess = "spark.gluten.sql.columnar.backend.velox.directorySizeGuess";
const std::string kFooterEstimatedSize = "spark.gluten.sql.columnar.backend.velox.footerEstimatedSize";
const std::string kFilePreloadThreshold = "spark.gluten.sql.columnar.backend.velox.filePreloadThreshold";
const std::string kPrefetchRowGroups = "spark.gluten.sql.columnar.backend.velox.prefetchRowGroups";
const std::string kLoadQuantum = "spark.gluten.sql.columnar.backend.velox.loadQuantum";
const std::string kMaxCoalescedDistance = "spark.gluten.sql.columnar.backend.velox.maxCoalescedDistance";
const std::string kMaxCoalescedBytes = "spark.gluten.sql.columnar.backend.velox.maxCoalescedBytes";
const std::string kCachePrefetchMinPct = "spark.gluten.sql.columnar.backend.velox.cachePrefetchMinPct";
const std::string kMemoryPoolCapacityTransferAcrossTasks =
    "spark.gluten.sql.columnar.backend.velox.memoryPoolCapacityTransferAcrossTasks";
const std::string kOrcUseColumnNames = "spark.gluten.sql.columnar.backend.velox.orcUseColumnNames";
const std::string kParquetUseColumnNames = "spark.gluten.sql.columnar.backend.velox.parquetUseColumnNames";

// write fies
const std::string kMaxPartitions = "spark.gluten.sql.columnar.backend.velox.maxPartitionsPerWritersSession";

const std::string kGlogVerboseLevel = "spark.gluten.sql.columnar.backend.velox.glogVerboseLevel";
const uint32_t kGlogVerboseLevelDefault = 0;
const uint32_t kGlogVerboseLevelMaximum = 99;
const std::string kGlogSeverityLevel = "spark.gluten.sql.columnar.backend.velox.glogSeverityLevel";
const uint32_t kGlogSeverityLevelDefault = 1;

// Query trace
/// Enable query tracing flag.
const std::string kQueryTraceEnabled = "spark.gluten.sql.columnar.backend.velox.queryTraceEnabled";
/// Base dir of a query to store tracing data.
const std::string kQueryTraceDir = "spark.gluten.sql.columnar.backend.velox.queryTraceDir";
/// The max trace bytes limit. Tracing is disabled if zero.
const std::string kQueryTraceMaxBytes = "spark.gluten.sql.columnar.backend.velox.queryTraceMaxBytes";
/// The regexp of traced task id. We only enable trace on a task if its id
/// matches.
const std::string kQueryTraceTaskRegExp = "spark.gluten.sql.columnar.backend.velox.queryTraceTaskRegExp";
/// Config used to create operator trace directory. This config is provided to
/// underlying file system and the config is free form. The form should be
/// defined by the underlying file system.
const std::string kOpTraceDirectoryCreateConfig =
    "spark.gluten.sql.columnar.backend.velox.opTraceDirectoryCreateConfig";

// Cudf config.
// GPU RMM memory resource
const std::string kCudfMemoryResource = "spark.gluten.sql.columnar.backend.velox.cudf.memoryResource";
const std::string kCudfMemoryResourceDefault =
    "async"; // Allowed: "cuda", "pool", "async", "arena", "managed", "managed_pool"

// Initial percent of GPU memory to allocate for memory resource for one thread
const std::string kCudfMemoryPercent = "spark.gluten.sql.columnar.backend.velox.cudf.memoryPercent";
const std::string kCudfMemoryPercentDefault = "50";

const std::string kCudfTimestampUnit = "spark.gluten.sql.columnar.backend.velox.cudf.timestampUnit";
const std::string kCudfTimestampUnitDefault = "us";

const std::string kCudfAllowCpuFallback =
    "spark.gluten.sql.columnar.backend.velox.cudf.allow_cpu_fallback";
const std::string kCudfAllowCpuFallbackDefault = "true";

/// Preferred size of batches in bytes to be returned by operators.
const std::string kVeloxPreferredBatchBytes = "spark.gluten.sql.columnar.backend.velox.preferredBatchBytes";

/// cudf
const std::string kCudfEnableTableScan = "spark.gluten.sql.columnar.backend.velox.cudf.enableTableScan";
const bool kCudfEnableTableScanDefault = false;
const std::string kCudfHiveConnectorId = "cudf-hive";
const std::string kCudfIcebergConnectorId = "cudf-iceberg";

// Forward to IBM CudfConfig::kCudfJitExpressionEnabled. When false, NVRTC JIT
// compilation of cudf::ast expressions is skipped and the AST/standalone-cudf
// path runs filters/projects directly, avoiding NVRTC compile failures (e.g.
// `operator_functor<EQUAL,true>::operator() no instance` reported on Q12 SF1K).
const std::string kCudfJitExpressionEnabled =
    "spark.gluten.sql.columnar.backend.velox.cudf.jit_expression_enabled";
const std::string kCudfJitExpressionEnabledDefault = "true";

// Forward to IBM CudfConfig::kCudfAstExpressionEnabled. When false, the cudf
// AST expression evaluator is bypassed; filters/projects run via standalone
// cudf::ast / cudf functions instead. Avoids "AST expression was provided
// non-matching operand types" (Q17 SF1K) and "like expects 2 inputs (3 vs. 2)"
// (Q18 SF1K).
const std::string kCudfAstExpressionEnabled =
    "spark.gluten.sql.columnar.backend.velox.cudf.ast_expression_enabled";
const std::string kCudfAstExpressionEnabledDefault = "true";

// Forward to IBM CudfConfig::kCudfConcatOptimizationEnabled. When true,
// OperatorAdapters inserts a CudfBatchConcat operator before every
// CudfHashAggregation, coalescing upstream batches until their cumulative
// row count reaches kCudfBatchSizeMinThreshold before forwarding to the
// agg. Lets the per-batch concat-with-bufferedResult_ cost be amortized
// across multiple raw input batches (reduces D2D for high-cardinality
// non-converging groupbys such as Q18 lineitem-by-l_orderkey).
const std::string kCudfConcatOptimizationEnabled =
    "spark.gluten.sql.columnar.backend.velox.cudf.concat_optimization_enabled";
const std::string kCudfConcatOptimizationEnabledDefault = "true";

// Enables cuDF's persistent FINAL streaming groupby with a fixed distinct-key
// capacity. Zero keeps the all-GPU levelled aggregation path.
const std::string kCudfGroupbyStreamingMaxDistinctKeys =
    "spark.gluten.sql.columnar.backend.velox.cudf.groupbyStreamingMaxDistinctKeys";
const std::string kCudfGroupbyStreamingMaxDistinctKeysDefault = "0";

const std::string kCudfOrderBySortedRunBytes =
    "spark.gluten.sql.columnar.backend.velox.cudf.orderBySortedRunBytes";
const std::string kCudfOrderBySortedRunBytesDefault = "268435456";

const std::string kCudfOrderByMergeFanIn =
    "spark.gluten.sql.columnar.backend.velox.cudf.orderByMergeFanIn";
const std::string kCudfOrderByMergeFanInDefault = "8";

// Forward to IBM CudfConfig::kCudfBatchSizeMinThreshold. Target minimum row
// count CudfBatchConcat coalesces upstream batches up to. Only used when
// kCudfConcatOptimizationEnabled is true. Default 100000 matches velox's
// internal default but is far too small for the typical 100M-row lineitem
// scan; reasonable values for SF1000 are 100M-1B.
const std::string kCudfBatchSizeMinThreshold =
    "spark.gluten.sql.columnar.backend.velox.cudf.batch_size_min_threshold";
const std::string kCudfBatchSizeMinThresholdDefault = "100000";

// Byte threshold for CudfBatchConcat.  Keep this independent from the row
// guard so a wide GPU batch is forwarded as soon as it reaches the compute
// target even when its row count is small.
const std::string kCudfBatchSizeMinThresholdBytes =
    "spark.gluten.sql.columnar.backend.velox.cudf.batch_size_min_threshold_bytes";

const std::string kCudfGpuTargetBatchRows = "spark.gluten.sql.columnar.backend.velox.cudf.gpuTargetBatchRows";
const std::string kCudfGpuTargetBatchRowsDefault = "1000000";

const std::string kCudfGpuTargetBatchBytes = "spark.gluten.sql.columnar.backend.velox.cudf.gpuTargetBatchBytes";
const std::string kCudfGpuTargetBatchBytesDefault = "2147483648"; // 2 GiB

// Pinned host memory pool size in bytes for fast HtoD/DtoH PCIe transfers.
// Default "0" uses cudf's default (0.5% of device memory, capped at 64 MB).
const std::string kCudfPinnedPoolSize = "spark.gluten.sql.columnar.backend.velox.cudf.pinnedPoolSize";
const std::string kCudfPinnedPoolSizeDefault = "0";

// Host allocations <= this threshold use pinned memory from the pool.
// Default "0" disables (all host allocations use pageable memory).
const std::string kCudfHostAsPinnedThreshold = "spark.gluten.sql.columnar.backend.velox.cudf.hostAsPinnedThreshold";
const std::string kCudfHostAsPinnedThresholdDefault = "0";

// Use cudf::pack to consolidate GPU column buffers into a single contiguous
// device buffer before D2H, reducing cudaMemcpyAsync call count from N (one
// per buffer) to 1.  Set to "false" to revert to the legacy per-buffer path.
const std::string kCudfPackedDtoH = "spark.gluten.sql.columnar.backend.velox.cudf.packedDtoH";
const std::string kCudfPackedDtoHDefault = "true";

// When true, the last GPU operator skips inserting CudfToVelox so CudfVector
// flows directly to VeloxGpuHashShuffleWriter::gpuPartitionAndEvict.
// Set at plan build time for WholeStageTransformers that feed GPU shuffle.
const std::string kCudfSkipOutputToVelox = "spark.gluten.sql.columnar.cudf.skipOutputToVelox";
const bool kCudfSkipOutputToVeloxDefault = false;

const std::string kStaticBackendConfPrefix = "spark.gluten.velox.";
const std::string kDynamicBackendConfPrefix = "spark.gluten.sql.columnar.backend.velox.";

} // namespace gluten
