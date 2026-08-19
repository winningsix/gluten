#!/usr/bin/env python3

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the fixed local query used to validate a relocated full bundle.

The script verifies the smoke-owned Spark configuration and loaded
``libgluten.so``, then requires the exact result and a native cuDF plan marker.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

from artifact_metadata import ArtifactMetadataError, BundleJarInfo, read_bundle_jar

QUERY = "SELECT sum(id) AS total FROM range(0, 32, 1, 1) " "WHERE id % 2 = 0"
EXPECTED_TOTAL = 240
NATIVE_PLAN_RE = re.compile(r"\bCudf[A-Za-z0-9]*Transformer\b")
BUNDLE_ROOT = Path("/opt/gluten-deploy")
LIBGLUTEN = BUNDLE_ROOT / "libs" / "libgluten.so"

PLUGIN_CLASS = "org.apache.gluten.GlutenPlugin"
SHUFFLE_MANAGER_CLASS = "org.apache.spark.shuffle.sort.ColumnarShuffleManager"


def _required_configuration(jar: Path) -> tuple[tuple[str, str], ...]:
    libs = BUNDLE_ROOT / "libs"
    return (
        ("spark.sql.shuffle.partitions", "1"),
        ("spark.sql.adaptive.enabled", "false"),
        ("spark.sql.ansi.enabled", "false"),
        ("spark.ui.enabled", "false"),
        ("spark.plugins", PLUGIN_CLASS),
        ("spark.shuffle.manager", SHUFFLE_MANAGER_CLASS),
        ("spark.memory.offHeap.enabled", "true"),
        ("spark.memory.offHeap.size", "1g"),
        ("spark.gluten.loadLibFromJar", "false"),
        ("spark.gluten.sql.columnar.libpath", str(LIBGLUTEN)),
        ("spark.gluten.sql.columnar.executor.libpath", str(LIBGLUTEN)),
        ("spark.driver.extraClassPath", str(jar)),
        ("spark.executor.extraClassPath", str(jar)),
        ("spark.driver.extraLibraryPath", str(libs)),
        ("spark.executor.extraLibraryPath", str(libs)),
        ("spark.executorEnv.UCX_MODULE_DIR", str(libs / "ucx")),
        ("spark.gluten.sql.columnar.cudf", "true"),
        (
            "spark.gluten.sql.columnar.backend.velox.cudf.enableValidation",
            "false",
        ),
        (
            "spark.gluten.sql.columnar.backend.velox.cudf.allow_cpu_fallback",
            "true",
        ),
        ("spark.gluten.sql.columnar.backend.velox.glogSeverityLevel", "0"),
        ("spark.gluten.sql.debug.cudf", "true"),
    )


def _single_bundle_jar() -> BundleJarInfo:
    jars = sorted(path for path in BUNDLE_ROOT.glob("*.jar") if path.is_file())
    if len(jars) != 1:
        raise AssertionError(
            "runtime bundle does not contain exactly one canonical Gluten bundle JAR"
        )
    try:
        return read_bundle_jar(jars[0])
    except ArtifactMetadataError as error:
        raise AssertionError(str(error)) from error


def _jar_entry_path(value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme not in ("", "file"):
        return None
    raw_path = unquote(parsed.path) if parsed.scheme else value
    return Path(raw_path).resolve(strict=False)


def _assert_runtime_identity(
    jar_info: BundleJarInfo,
    spark_version: str,
    scala_version: str,
    java_version: str,
) -> None:
    if spark_version != jar_info.spark_version:
        raise AssertionError(
            f"expected Spark {jar_info.spark_version}, found {spark_version}"
        )
    scala_match = re.fullmatch(r"([0-9]+[.][0-9]+)(?:[.].*)?", scala_version)
    if scala_match is None or scala_match.group(1) != jar_info.scala_binary:
        raise AssertionError(
            f"expected Scala {jar_info.scala_binary}, found {scala_version}"
        )
    if java_version != "17":
        raise AssertionError(f"expected JDK 17, found {java_version}")


def _assert_configuration(spark, jar_info: BundleJarInfo) -> None:
    context = spark.sparkContext
    scala_version = str(spark._jvm.scala.util.Properties.versionNumberString())
    java_version = str(
        spark._jvm.java.lang.System.getProperty("java.specification.version")
    )
    _assert_runtime_identity(jar_info, spark.version, scala_version, java_version)
    if context.master != "local[1]":
        raise AssertionError(f"expected local[1], found {context.master}")

    conf = context.getConf()
    for key, expected in _required_configuration(jar_info.path):
        actual = conf.get(key, None)
        if actual != expected:
            raise AssertionError(f"smoke-owned Spark setting changed: {key}")

    spark_jars = conf.get("spark.jars", "")
    resolved_jars = {
        resolved
        for value in spark_jars.split(",")
        if value and (resolved := _jar_entry_path(value)) is not None
    }
    if jar_info.path.resolve(strict=True) not in resolved_jars:
        raise AssertionError("the relocated bundle JAR is absent from spark.jars")


def _assert_relocated_libgluten_loaded(spark) -> None:
    expected = LIBGLUTEN.resolve(strict=True)
    driver_pid = int(spark._jvm.java.lang.ProcessHandle.current().pid())
    maps_path = Path(f"/proc/{driver_pid}/maps")
    mapped_paths: set[Path] = set()
    for line in maps_path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        mapped_value = fields[5].removesuffix(" (deleted)")
        mapped_paths.add(Path(mapped_value).resolve(strict=False))
    if expected not in mapped_paths:
        raise AssertionError(
            "the Java driver did not map the relocated bundle libgluten.so"
        )


def run_query() -> None:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("gluten-velox-gpu-runtime-smoke").getOrCreate()
    try:
        jar_info = _single_bundle_jar()
        _assert_configuration(spark, jar_info)
        frame = spark.sql(QUERY)
        rows = frame.collect()
        if len(rows) != 1 or rows[0]["total"] != EXPECTED_TOTAL:
            raise AssertionError(
                f"expected one total={EXPECTED_TOTAL} row, found {rows!r}"
            )
        executed_plan = frame._jdf.queryExecution().executedPlan().toString()
        marker = NATIVE_PLAN_RE.search(executed_plan)
        if marker is None:
            raise AssertionError(
                "executed plan has no Gluten/cuDF native Transformer marker"
            )
        _assert_relocated_libgluten_loaded(spark)

        timezone = spark.conf.get("spark.sql.session.timeZone")
        print(f"Runtime smoke effective session timezone: {timezone}")
        print(f"Runtime smoke native marker: {marker.group(0)}")
        print(f"Runtime smoke result: total={EXPECTED_TOTAL}")
    finally:
        spark.stop()


def main() -> int:
    try:
        run_query()
        return 0
    except Exception as error:  # Spark/Py4J exceptions are expected smoke failures.
        print(f"ERROR: runtime query smoke failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
