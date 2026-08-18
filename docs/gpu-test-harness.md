<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements.  See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to You under the Apache License, Version 2.0
  (the "License"); you may not use this file except in compliance with
  the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->
# GPU (cuDF) + Flux Test Harness

This page describes how Gluten tests are tagged and selected for the GPU (cuDF) and
Flux (native MPP) execution paths, so the same test logic can run on CPU and GPU
without duplication.

## Tags

Two ScalaTest tags live under `backends-velox/src/test/java/org/apache/gluten/tags/`:

| Tag | Meaning | Runtime toggle |
|-----|---------|----------------|
| `@GpuTest` | Must run on the GPU (cuDF) backend. | `spark.gluten.sql.columnar.cudf=true` |
| `@FluxTest` | Exercises the native Flux (MPP) path. | `spark.gluten.mpp.enabled=true` |

> Note: the feature is named **Flux** (package `org.apache.gluten.flux`), but the config
> **key is still `spark.gluten.mpp.enabled`**.

## Mixin traits

Suite bases can flip on the GPU / Flux runtime toggles by mixing in a trait from
`backends-velox/src/test/scala/org/apache/gluten/execution/GpuMode.scala`:

- `GpuMode` — sets `spark.gluten.sql.columnar.cudf=true` (+ the Velox cuDF enable keys).
- `FluxMode` — sets `spark.gluten.mpp.enabled=true` (+ `spark.gluten.mpp.strategy.enabled`).

Example: author a suite once, run it on CPU by default and on GPU when selected.

```scala
@org.apache.gluten.tags.GpuTest
class MyFeatureSuite extends VeloxTPCHSuite with GpuMode {
  // test logic identical to the CPU path; sparkConf is augmented by GpuMode
}
```

## Maven selection

The CPU CI runs exclude `GpuTest` and `FluxTest` by default via
`-DtagsToExclude=...,org.apache.gluten.tags.GpuTest,org.apache.gluten.tags.FluxTest`
(see `.github/workflows/velox_backend_*.yml`). GPU runs are executed on the Blossom
Spark-Flux pipeline (GitLab `nvspark/blossom-jenkins`) and select the GPU subset with
`-DtagsToInclude=org.apache.gluten.tags.GpuTest`.

To run only GPU tests locally (requires a GPU and the cuDF native build):

```bash
build/mvn test -Pspark-4.0 -Pscala-2.13 -Pjava-17 -Pbackends-velox \
  -DtagsToInclude=org.apache.gluten.tags.GpuTest \
  -DtagsToExclude=org.apache.spark.tags.ExtendedSQLTest
```

## Build

The cuDF native library must be built with `--enable_gpu=ON`
(`VELOX_ENABLE_CUDF=ON`); see `docs/get-started/VeloxGPU.md` and
`dev/docker-build-cudf.sh`.
