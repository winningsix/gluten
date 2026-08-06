# Reproducing the TPC-H SF1000 S3 70-second profile

This document describes the performance contract shared by this Spark-Gluten
change and HighPerfDataAccelerator/velox#36. The pair is accepted only when a
fresh build completes one continuous Q1-Q22 application in approximately 70
seconds, with all 22 queries using native MPP execution.

The number is a hot power-run statistic. It is not the elapsed time of the
whole Spark application: run every query once to populate the Velox cache, run
it three more times, and sum the minimum of iterations 2-4 for Q1-Q22. Do not
change the TPC-H SQL.

## Source and build contract

Use the head of this PR with Velox PR #36 checked out as `VELOX_HOME`. The
validated companion Velox revision is
`83eb29bd40bc24cbeffc1f5d8f48655f788087a2`.

The validated software matrix is:

- EMR 8.0.0, Spark 4.0.2, Scala 2.13, Java 17;
- CUDA 13.1 and `sm_120` code generation;
- cuDF 26.06.01 and UCXX 0.50.01;
- release builds with S3 and GPU support enabled;
- AWS SDK for C++ built with `s3`, `identity-management`, and `s3-crt`.

A clean build can use the repository build entry point:

```bash
./dev/buildbundle-veloxbe.sh \
  --build_type=Release \
  --build_tests=OFF \
  --build_velox_tests=OFF \
  --enable_s3=ON \
  --enable_gpu=ON \
  --cuda_arch=120-real \
  --spark_version=4.0 \
  --velox_home=/path/to/velox-pr-36
```

Package both the resulting Spark 4.0/Scala 2.13 Gluten bundle and its matching
`libgluten.so`. Reusing a native library or JAR from another source revision is
not a valid reproduction.

## Runtime topology contract

The accepted AWS topology has one CPU-only driver and one dedicated
four-GPU executor node. Start four long-lived executors on that node, with 16
cores, 8 GiB JVM heap, 40 GiB memory overhead, 32 GiB Spark off-heap memory,
and one primary GPU per executor. The driver requests no GPU.

All four executors must share host IPC and be able to open CUDA IPC handles
exported by their peers while each executor remains pinned to its own primary
GPU. How Kubernetes establishes that device/IPC contract is deployment
specific; a device plugin or custom pod feature may be used. It is part of the
platform contract, not an additional Gluten or Velox source patch. Use
`UCX_TLS=self,sm,tcp,cuda_copy,cuda_ipc`.

Mount executor-local NVMe at `/var/data`. The accepted cache allocation per
executor is 36 GiB Velox memory cache plus 128 GiB Velox SSD cache. Keep the
same executor processes alive for the full Q1-Q22 sequence.

## Application-wide performance configuration

The following settings define the material part of the accepted profile.
Cluster-specific image, IAM, log, and pod-template locations are intentionally
not included.

```properties
spark.dynamicAllocation.enabled=false
spark.executor.instances=4
spark.executor.cores=16
spark.executor.memory=8g
spark.executor.memoryOverhead=40g
spark.executor.resource.gpu.amount=1
spark.task.resource.gpu.amount=1
spark.memory.offHeap.enabled=true
spark.memory.offHeap.size=32g
spark.sql.adaptive.enabled=false
spark.sql.shuffle.partitions=16
spark.sql.files.minPartitionNum=60
spark.sql.files.maxPartitionBytes=512mb
spark.sql.autoBroadcastJoinThreshold=2147483648
spark.sql.cbo.enabled=false
spark.sql.cbo.joinReorder.enabled=false
spark.sql.dynamicPartitionPruning.enabled=false
spark.speculation=false

spark.plugins=org.apache.gluten.GlutenPlugin
spark.shuffle.manager=org.apache.spark.shuffle.GlutenShuffleManager
spark.gluten.enabled=true
spark.gluten.mpp.enabled=true
spark.gluten.mpp.strategy.enabled=true
spark.gluten.mpp.multiExecutor.enabled=true
spark.gluten.mpp.multiExecutor.numPartitions=4
spark.gluten.mpp.controlPlane.endpointRegistry.enabled=true
spark.gluten.mpp.controlPlane.endpointRegistry.awaitMinExecutors=true
spark.gluten.mpp.controlPlane.endpointRegistry.fallbackToProbe=false
spark.gluten.mpp.localHashExchangeTasks=4
spark.gluten.mpp.maxDriversPerFragment=4
spark.gluten.mpp.joinDriversPerFragment=4
spark.gluten.mpp.scanDriversPerFragment=4
spark.gluten.mpp.scan.largeScanDriversPerFragment=4
spark.gluten.mpp.scan.targetSplitBytes=512mb
spark.gluten.mpp.scan.maxNativeSplitBytes=512mb
spark.gluten.mpp.scan.maxWholeFileBytes=512mb
spark.gluten.mpp.scan.sizeAwarePartitioning=true
spark.gluten.mpp.removeRedundantShuffle=true
spark.gluten.mpp.splitFinalAggBeforeJoinHub=true
spark.gluten.mpp.singlePartitionSort=true
spark.gluten.mpp.factProbeBroadcastHint=true
spark.gluten.mpp.normalizeJoinBuildSide=true
spark.gluten.mpp.pushSelectiveDimensionFilterIntoAggregate=true
spark.gluten.mpp.realScanBuildSideDominanceRatio=2
spark.gluten.mpp.replicatedJoinMaxBuildBytes=8g

spark.gluten.sql.columnar.cudf=true
spark.gluten.sql.columnar.cudf.gpuPartition=true
spark.gluten.sql.columnar.forceShuffledHashJoin=true
spark.gluten.sql.optimizer.candidateFirstExistence.mode=restricted-rowid
spark.gluten.sql.optimizer.candidateFirstExistence.pushDimensionChain=true
spark.gluten.sql.columnar.pushDimensionChainBeforeFact.enabled=true
spark.gluten.sql.columnar.reorderFilteredFactBeforeWideDimension.enabled=true
spark.gluten.sql.columnar.selectiveDimensionJoinReorder.enabled=false
spark.gluten.sql.columnar.query.fallback.threshold=-1
spark.gluten.sql.columnar.wholeStage.fallback.threshold=-1

spark.gluten.sql.columnar.backend.velox.IOThreads=64
spark.gluten.sql.columnar.backend.velox.SplitPreloadPerDriver=16
spark.gluten.sql.columnar.backend.velox.SplitPreloadPerTask=128
spark.gluten.sql.columnar.backend.velox.cacheEnabled=true
spark.gluten.sql.columnar.backend.velox.cacheContiguousEntries=true
spark.gluten.sql.columnar.backend.velox.cacheLargestSizeClassPages=2048
spark.gluten.sql.columnar.backend.velox.cachePinnedBytes=38654705664
spark.gluten.sql.columnar.backend.velox.cachePinnedPrewarmBytes=0
spark.gluten.sql.columnar.backend.velox.memCacheSize=38654705664
spark.gluten.sql.columnar.backend.velox.ssdCachePath=/var/data
spark.gluten.sql.columnar.backend.velox.ssdCacheSize=137438953472
spark.gluten.sql.columnar.backend.velox.ssdCacheShards=4
spark.gluten.sql.columnar.backend.velox.ssdCacheIOThreads=8
spark.gluten.sql.columnar.backend.velox.loadQuantum=8388608
spark.gluten.sql.columnar.backend.velox.connector.hive.s3.max-connections=256
spark.gluten.sql.columnar.backend.velox.connector.hive.s3.max-attempts=5
spark.gluten.sql.columnar.backend.velox.connector.hive.s3.retry-mode=adaptive
spark.gluten.sql.columnar.backend.velox.cudf.enableTableScan=true
spark.gluten.sql.columnar.backend.velox.cudf.concurrentGpuTasks=2
spark.gluten.sql.columnar.backend.velox.cudf.memoryPercent=95
spark.gluten.sql.columnar.backend.velox.cudf.memoryResource=async
spark.gluten.sql.columnar.backend.velox.cudf.hive.use_buffered_input=true
spark.gluten.sql.columnar.backend.velox.cudf.hive.prefetchThreads=128
spark.gluten.sql.columnar.backend.velox.cudf.hive.executorSplitPrefetchConcurrency=64
spark.gluten.sql.columnar.backend.velox.cudf.hive.prefetchMaxInFlightBytes=4g
spark.gluten.sql.columnar.backend.velox.cudf.batch_size_min_threshold=16000000
spark.gluten.sql.columnar.backend.velox.cudf.batch_size_max_threshold=32000000
spark.gluten.sql.columnar.backend.velox.cudf.partitioned_output_batch_rows=8000000
spark.gluten.sql.columnar.backend.velox.cudf.partitioned_output_max_batch_rows=8000000
spark.gluten.sql.columnar.backend.velox.cudf.partitioned_output_chunk_rows=0

spark.executorEnv.GLUTEN_S3_HTTP_BACKEND=crt
spark.executorEnv.GLUTEN_CPP_S3_AWS_SDK=1
spark.executorEnv.GLUTEN_CPP_S3_CRT=1
spark.executorEnv.GLUTEN_CPP_S3_SHARDS=8
spark.executorEnv.GLUTEN_CPP_S3_CONCURRENCY_PER_SHARD=16
spark.executorEnv.GLUTEN_CPP_S3_SDK_CONCURRENCY=256
spark.executorEnv.GLUTEN_CPP_S3_SDK_RANGE_BYTES=8388608
spark.executorEnv.GLUTEN_CPP_S3_CRT_DOWNLOAD_WINDOW_BYTES=1073741824
spark.executorEnv.GLUTEN_CPP_S3_CRT_PART_BYTES=8388608
spark.executorEnv.GLUTEN_CPP_S3_CRT_TARGET_GBPS=80
spark.executorEnv.GLUTEN_CPP_S3_MAX_RETRIES=5
spark.executorEnv.GLUTEN_CPP_S3_PINNED_POOL=1
spark.executorEnv.GLUTEN_CPP_S3_PINNED_POOL_MAX_BYTES=4294967296
spark.executorEnv.GLUTEN_CPP_S3_H2D_SLOTS=4
spark.executorEnv.GLUTEN_CPP_S3_HOST_BATCH=1
spark.executorEnv.GLUTEN_CPP_S3_PAGEABLE_HOST_BUFFER=0
spark.executorEnv.GLUTEN_CPP_S3_METADATA_READAHEAD_BYTES=65536
spark.executorEnv.GLUTEN_CPP_S3_SYNTHESIZE_PARQUET_MAGIC=1
spark.executorEnv.GLUTEN_CUDF_CACHE_H2D_INLINE=1
spark.executorEnv.GLUTEN_VELOX_CACHE_NATIVE_PINNED=1
spark.executorEnv.GLUTEN_CPP_S3_LOCAL_CACHE=0
spark.executorEnv.GLUTEN_UCX_MAX_INFLIGHT_RECV_BYTES=67108864
spark.executorEnv.GLUTEN_UCX_STRICT_PARTITION_MEMORY_ADMISSION=1
spark.executorEnv.GLUTEN_UCX_STRICT_PARTITION_HEADROOM_BYTES=1073741824
spark.executorEnv.GLUTEN_UCX_STRICT_PARTITION_TRIM_ASYNC_POOL=1
spark.executorEnv.UCX_CUDA_IPC_CACHE=y
spark.executorEnv.UCX_CUDA_IPC_CACHE_MAX_REGIONS=128
spark.executorEnv.UCX_CUDA_IPC_CACHE_MAX_SIZE=8G
spark.executorEnv.UCX_MAX_RNDV_RAILS=1
spark.executorEnv.UCX_USE_MT_MUTEX=y
```

## Query-scoped configuration

Restore the application-wide values before every query, then apply these
deltas before the query is analyzed and planned. Queries not listed have no
delta.

| Query | Configuration delta |
|---|---|
| Q5 | `spark.gluten.mpp.factProbeBroadcastHint=false` |
| Q7 | `spark.gluten.sql.columnar.backend.velox.cudf.partitioned_output_batch_rows=32000000`; `spark.gluten.sql.columnar.backend.velox.cudf.partitioned_output_max_batch_rows=32000000` |
| Q14 | `spark.gluten.mpp.factProbeBroadcastHint=false`; `spark.gluten.mpp.normalizeJoinBuildSide=false`; `spark.gluten.mpp.pushSelectiveDimensionFilterIntoAggregate=false`; `spark.gluten.mpp.replicatedJoinMaxBuildBytes=512m`; `spark.gluten.sql.columnar.pushDimensionChainBeforeFact.enabled=false`; `spark.gluten.sql.columnar.reorderFilteredFactBeforeWideDimension.enabled=false`; `spark.sql.autoBroadcastJoinThreshold=-1` |
| Q18 | `spark.gluten.mpp.factProbeBroadcastHint=false`; `spark.gluten.mpp.parallelSortSplit=true`; `spark.gluten.mpp.realScanBuildSideDominanceRatio=8`; `spark.gluten.mpp.scan.largeScanDriversPerFragment=2`; `spark.gluten.mpp.scanDriversPerFragment=2`; `spark.gluten.sql.columnar.backend.velox.cudf.batch_size_max_threshold=2147483647`; `spark.gluten.sql.columnar.backend.velox.cudf.groupbyStreamingMaxDistinctKeys=400000000`; `spark.gluten.sql.columnar.backend.velox.cudf.partialIdentityAggregation=true`; `spark.gluten.sql.columnar.backend.velox.cudf.partitioned_output_chunk_rows=1000000`; `spark.gluten.sql.columnar.pushDimensionChainBeforeFact.enabled=false`; `spark.sql.files.maxPartitionBytes=8gb` |
| Q21 | `spark.gluten.mpp.realScanBuildSideDominanceRatio=8`; `spark.gluten.mpp.scanDriversPerFragment=2`; `spark.gluten.sql.columnar.pushDimensionChainBeforeFact.enabled=false` |

The canonical serialization of all 22 effective query profiles has SHA-256
`e9b8fa731608354eec0fba1cd82b225fc184736ddf08070efc7800f28635c2db`.

## Acceptance gates

Record the source revisions and hashes of the JAR and `libgluten.so`, and reject
the run if any executor is replaced or any query falls back from native MPP.
Capture the optimized plan and operator runtime metrics for every selected hot
iteration. The primary gate is 22/22 successful queries and a hot-min sum near
70 seconds; compare plan shapes and TableScan/UcxExchange blocked time before
attributing a miss to S3.
