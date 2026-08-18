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

import org.apache.gluten.config.GlutenConfig
import org.apache.gluten.config.VeloxConfig

import org.apache.spark.SparkConf
import org.apache.spark.sql.test.SharedSparkSession

// Note: `spark.gluten.sql.columnar.backend.velox.cudf.enabled` has no typed constant in
// VeloxConfig today (the existing GPU suites set it as a string literal), so we follow the
// same convention here. `CUDF_ENABLE_TABLE_SCAN` is typed. The trait extends
// `SharedSparkSession` so that `super.sparkConf` resolves against the concrete suite base
// (e.g. `VeloxWholeStageTransformerSuite`) when mixed in via `with GpuMode`.

/**
 * A mixin trait that enables the GPU (cuDF) backend for a test suite. Suites extend a
 * `WholeStageTransformerSuite`-based base and add `with GpuMode` to flip on the cuDF runtime
 * toggles, so the same test logic can run in CPU vs GPU mode without duplication.
 *
 * Combine with the `@org.apache.gluten.tags.GpuTest` annotation so the suite is selected only on
 * GPU runners.
 *
 * Runtime toggles enabled here:
 *   - `spark.gluten.sql.columnar.cudf` = true (Velox PlanNodes -> cuDF GPU operators)
 *   - `spark.gluten.sql.columnar.backend.velox.cudf.enabled` = true
 *   - `spark.gluten.sql.columnar.backend.velox.cudf.enableTableScan` = true
 *
 * The Flux (native MPP) path is NOT enabled here; mix in [[FluxMode]] for that.
 */
trait GpuMode extends SharedSparkSession {

  override protected def sparkConf: SparkConf = {
    super.sparkConf
      .set(GlutenConfig.COLUMNAR_CUDF_ENABLED.key, "true")
      .set("spark.gluten.sql.columnar.backend.velox.cudf.enabled", "true")
      .set(VeloxConfig.CUDF_ENABLE_TABLE_SCAN.key, "true")
  }
}

/**
 * A mixin trait that enables the Flux (native MPP) execution path for a test suite. The feature is
 * named "Flux" but the config key is still `spark.gluten.mpp.enabled`. Combine with [[GpuMode]] for
 * GPU + Flux runs, and tag the suite with `@FluxTest`.
 */
trait FluxMode extends SharedSparkSession {

  override protected def sparkConf: SparkConf = {
    super.sparkConf
      .set("spark.gluten.mpp.enabled", "true")
      .set("spark.gluten.mpp.strategy.enabled", "true")
  }
}
