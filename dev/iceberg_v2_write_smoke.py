#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import pyarrow.parquet as pq
from pyspark.sql import SparkSession


def command_plan(spark, statement):
    command = spark.sql(statement)
    return command._jdf.queryExecution().executedPlan().toString()


def assert_native_write_plan(name, plan, writer):
    required_nodes = (writer, "MppNativeQuery")
    missing = [node for node in required_nodes if node not in plan]
    if missing:
        raise RuntimeError(f"{name} plan is missing {missing}:\n{plan}")
    if plan.index("MppNativeQuery") < plan.index(writer):
        raise RuntimeError(f"{name} MPP query is outside the Iceberg writer boundary:\n{plan}")
    if "ColumnarToRow" in plan:
        raise RuntimeError(f"{name} plan contains a row boundary:\n{plan}")


def inspect_local_data_files(spark, table, expected_field_ids):
    current_files = spark.sql(
        f"SELECT file_path, record_count FROM {table}.files WHERE content = 0"
    ).collect()
    parquet_writers = set()
    for data_file in current_files:
        parsed_path = urlparse(data_file["file_path"])
        local_path = unquote(
            parsed_path.path if parsed_path.scheme == "file" else data_file["file_path"]
        )
        parquet_file = pq.ParquetFile(local_path)
        parquet_writers.add(parquet_file.metadata.created_by)
        actual_field_ids = {
            int(value) for value in re.findall(r"field_id=(\d+)", str(parquet_file.schema))
        }
        if not expected_field_ids.issubset(actual_field_ids):
            raise RuntimeError(
                f"Iceberg field ID mismatch in {local_path}: "
                f"expected={expected_field_ids}, actual={actual_field_ids}"
            )
    if not parquet_writers or any(
        "cudf" not in writer.lower() for writer in parquet_writers
    ):
        raise RuntimeError(
            f"expected libcudf Parquet files for {table}, "
            f"created_by={sorted(parquet_writers)}"
        )
    return {
        "file_count": len(current_files),
        "record_count": sum(row["record_count"] for row in current_files),
        "parquet_created_by": sorted(parquet_writers),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="v2smoke")
    parser.add_argument("--namespace", default="boundary")
    parser.add_argument("--warehouse", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("Iceberg V2 write boundary smoke").getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    catalog = args.catalog
    namespace = args.namespace
    table = f"{catalog}.{namespace}.v2_boundary"
    static_table = f"{catalog}.{namespace}.v2_static_partition"
    nested_table = f"{catalog}.{namespace}.v2_nested"
    warehouse_path = args.warehouse.removeprefix("file://")
    append_source = (Path(warehouse_path) / "_smoke_input" / "append").as_uri()
    append_again_source = (
        Path(warehouse_path) / "_smoke_input" / "append_again"
    ).as_uri()
    overwrite_source = (Path(warehouse_path) / "_smoke_input" / "overwrite").as_uri()
    static_source = (Path(warehouse_path) / "_smoke_input" / "static").as_uri()
    nested_source = (Path(warehouse_path) / "_smoke_input" / "nested").as_uri()

    spark.conf.set("spark.gluten.enabled", "false")
    spark.conf.set("spark.gluten.mpp.enabled", "false")
    spark.conf.set("spark.gluten.sql.native.writer.enabled", "false")
    spark.range(0, 10000, 1, 16).selectExpr(
        "id",
        "CAST(pmod(id, 4) AS INT) AS p",
        "concat('row-', CAST(id AS STRING)) AS payload",
    ).write.mode("overwrite").parquet(append_source)
    spark.range(0, 500, 1, 16).selectExpr(
        "id + 100000 AS id",
        "1 AS p",
        "concat('replacement-', CAST(id AS STRING)) AS payload",
    ).write.mode("overwrite").parquet(overwrite_source)
    spark.range(0, 400, 1, 16).selectExpr(
        "id + 200000 AS id",
        "CAST(pmod(id, 4) AS INT) AS p",
        "concat('appended-again-', CAST(id AS STRING)) AS payload",
    ).write.mode("overwrite").parquet(append_again_source)
    spark.range(0, 100, 1, 4).selectExpr(
        "id + 300000 AS id",
        "concat('static-', CAST(id AS STRING)) AS payload",
    ).write.mode("overwrite").parquet(static_source)
    spark.range(0, 1000, 1, 16).selectExpr(
        "id",
        "array(concat('tag-', CAST(pmod(id, 7) AS STRING))) AS tags",
        "map('source', concat('row-', CAST(id AS STRING))) AS attrs",
        "named_struct('label', concat('label-', CAST(id AS STRING)), "
        "'score', id * 10) AS detail",
    ).write.mode("overwrite").parquet(nested_source)
    spark.conf.set("spark.gluten.enabled", "true")
    spark.conf.set("spark.gluten.mpp.enabled", "true")

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{namespace}")
    spark.sql(f"DROP TABLE IF EXISTS {table}")
    spark.sql(f"DROP TABLE IF EXISTS {static_table}")
    spark.sql(f"DROP TABLE IF EXISTS {nested_table}")
    spark.sql(
        f"""
        CREATE TABLE {table} (
          id BIGINT,
          p INT,
          payload STRING
        ) USING iceberg
        PARTITIONED BY (p)
        TBLPROPERTIES ('format-version' = '2')
        """
    )
    spark.sql(
        f"""
        CREATE TABLE {static_table} (
          id BIGINT,
          p INT,
          payload STRING
        ) USING iceberg
        PARTITIONED BY (p)
        TBLPROPERTIES ('format-version' = '2')
        """
    )
    spark.sql(
        f"""
        CREATE TABLE {nested_table} (
          id BIGINT,
          tags ARRAY<STRING>,
          attrs MAP<STRING, STRING>,
          detail STRUCT<label: STRING, score: BIGINT>
        ) USING iceberg
        TBLPROPERTIES ('format-version' = '2')
        """
    )

    append_plan = command_plan(
        spark,
        f"""
        INSERT INTO {table}
        SELECT id, p, payload
        FROM parquet.`{append_source}`
        """,
    )

    append_again_plan = command_plan(
        spark,
        f"""
        INSERT INTO {table}
        SELECT id, p, payload
        FROM parquet.`{append_again_source}`
        """,
    )

    overwrite_plan = command_plan(
        spark,
        f"""
        INSERT OVERWRITE {table}
        SELECT id, p, payload
        FROM parquet.`{overwrite_source}`
        """,
    )
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "static")
    static_overwrite_plan = command_plan(
        spark,
        f"""
        INSERT OVERWRITE {static_table} PARTITION (p = 7)
        SELECT id, payload
        FROM parquet.`{static_source}`
        """,
    )
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    nested_append_plan = command_plan(
        spark,
        f"""
        INSERT INTO {nested_table}
        SELECT id, tags, attrs, detail
        FROM parquet.`{nested_source}`
        """,
    )
    assert_native_write_plan("append", append_plan, "VeloxIcebergAppendData")
    assert_native_write_plan(
        "append to existing partitions", append_again_plan, "VeloxIcebergAppendData"
    )
    assert_native_write_plan(
        "dynamic overwrite", overwrite_plan, "VeloxIcebergOverwritePartitionsDynamic"
    )
    assert_native_write_plan(
        "static partition overwrite",
        static_overwrite_plan,
        "VeloxIcebergOverwriteByExpression",
    )
    assert_native_write_plan(
        "nested append", nested_append_plan, "VeloxIcebergAppendData"
    )

    # Validate committed Iceberg state through Spark's reference readers. Iceberg metadata tables
    # are intentionally outside the native write boundary and are not supported by strict MPP.
    spark.conf.set("spark.gluten.enabled", "false")
    spark.conf.set("spark.gluten.mpp.enabled", "false")

    rows_by_partition = {
        str(row["p"]): row["count"]
        for row in spark.sql(
            f"SELECT p, count(*) AS count FROM {table} GROUP BY p ORDER BY p"
        ).collect()
    }
    total_rows = sum(rows_by_partition.values())
    primary_files = inspect_local_data_files(spark, table, {1, 2, 3})
    static_files = inspect_local_data_files(spark, static_table, {1, 2, 3})
    nested_files = inspect_local_data_files(spark, nested_table, set(range(1, 10)))
    static_rows = {
        str(row["p"]): row["count"]
        for row in spark.sql(
            f"SELECT p, count(*) AS count FROM {static_table} GROUP BY p ORDER BY p"
        ).collect()
    }
    nested_stats = spark.sql(
        f"""
        SELECT
          count(*) AS count,
          sum(detail.score) AS score_sum,
          count_if(size(tags) = 1) AS single_tag_rows,
          count_if(attrs['source'] = concat('row-', CAST(id AS STRING))) AS matching_attrs
        FROM {nested_table}
        """
    ).first().asDict()
    snapshots = [
        {
            "operation": row["operation"],
            "summary": dict(row["summary"]),
        }
        for row in spark.sql(
            f"SELECT operation, summary FROM {table}.snapshots ORDER BY committed_at"
        ).collect()
    ]
    format_version = spark.sql(
        f"SHOW TBLPROPERTIES {table} ('format-version')"
    ).first()["value"]

    expected_rows = {"0": 2600, "1": 500, "2": 2600, "3": 2600}
    if rows_by_partition != expected_rows:
        raise RuntimeError(
            f"row count mismatch: expected={expected_rows}, actual={rows_by_partition}"
        )
    if total_rows != primary_files["record_count"]:
        raise RuntimeError(
            "file record count mismatch: "
            f"table={total_rows}, files={primary_files['record_count']}"
        )
    if static_rows != {"7": 100} or static_files["record_count"] != 100:
        raise RuntimeError(
            f"static partition mismatch: rows={static_rows}, files={static_files}"
        )
    expected_nested_stats = {
        "count": 1000,
        "score_sum": 4995000,
        "single_tag_rows": 1000,
        "matching_attrs": 1000,
    }
    if nested_stats != expected_nested_stats or nested_files["record_count"] != 1000:
        raise RuntimeError(
            "nested data mismatch: "
            f"expected={expected_nested_stats}, actual={nested_stats}, files={nested_files}"
        )
    if format_version != "2":
        raise RuntimeError(f"format version mismatch: expected=2, actual={format_version}")
    snapshot_operations = [snapshot["operation"] for snapshot in snapshots]
    if snapshot_operations != ["append", "append", "overwrite"]:
        raise RuntimeError(
            "snapshot operation mismatch: "
            f"expected=['append', 'append', 'overwrite'], actual={snapshot_operations}"
        )

    print(
        "ICEBERG_V2_WRITE_SMOKE="
        + json.dumps(
            {
                "warehouse": args.warehouse,
                "table": table,
                "format_version": int(format_version),
                "rows_by_partition": rows_by_partition,
                "total_rows": total_rows,
                "primary_files": primary_files,
                "static_rows_by_partition": static_rows,
                "static_files": static_files,
                "nested_stats": nested_stats,
                "nested_files": nested_files,
                "snapshots": snapshots,
                "append_plan": append_plan,
                "append_again_plan": append_again_plan,
                "overwrite_plan": overwrite_plan,
                "static_overwrite_plan": static_overwrite_plan,
                "nested_append_plan": nested_append_plan,
            },
            sort_keys=True,
        )
    )
    spark.stop()


if __name__ == "__main__":
    main()
