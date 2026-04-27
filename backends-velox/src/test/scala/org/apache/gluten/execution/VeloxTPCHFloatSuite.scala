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

/**
 * TPC-H suite using float (double) data instead of decimal. This avoids Velox decimal precision
 * bugs (a_precision not defined) that cause q18/q20/q21 to fail with the default decimal parquet
 * data.
 */
class VeloxTPCHFloatSuite extends VeloxTPCHSuite {
  override protected val resourcePath: String = "/tpch-data-parquet-float"
  override def subType(): String = "float"
  override def shouldCheckGoldenFiles(): Boolean = false

  override protected def sparkConf: org.apache.spark.SparkConf = {
    super.sparkConf
      // Override shuffle partitions to force shuffle exchanges in the plan.
      // The parent sets it to 1, which lets Spark optimize away shuffles entirely.
      // With 4 partitions, agg/join queries will have ShuffleExchange nodes,
      // giving MppCollapseRule something to absorb.
      .set("spark.sql.shuffle.partitions", "4")
      .set("spark.sql.adaptive.enabled", "false")
      // Plan C (MppStrategy) — intercept at Strategy level
      .set("spark.gluten.mpp.enabled", "true")
      .set("spark.gluten.mpp.strategy.enabled", "true")
      // Dump every MppNativeQueryExec plan for offline comparison with Presto
      .set("spark.gluten.mpp.substraitDumpDir", "/opt/gluten/mpp-dumps-tpch")
      // Plan-shape parity with Presto (opt-in cross-cut rules)
      .set("spark.gluten.mpp.singlePartitionSort", "true")
      .set("spark.gluten.mpp.removeRedundantShuffle", "true")
  }
}
