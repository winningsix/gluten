# Presto-GPU + UCX TPC-H Testing Recipe

End-to-end recipe for running TPC-H benchmarks on the Presto-GPU + UCX shuffle reference image. Used as the **comparison baseline** for the Spark-Gluten FLUX path: same TPC-H queries, same SF1 / SF1K data, same physical plan operators, same exchange transport (UCX).

This document is descriptive — the actual scripts live in `code/velox-testing/presto/` and the launchers (`start_native_gpu_presto.sh`, `setup_benchmark_tables.sh`, `analyze_tables.sh`) wrap the docker compose / Hive metastore steps.

---

## Image stack

Three images are required. The reference pair is **IBM Presto `c77e7e6` + Velox `0ac4367`** — same Velox pin used by the Spark-Gluten Track-A C++ tree, so cuDF behavior is comparable.

| Image | Tag | Size | Contents |
|---|---|---|---|
| Coordinator | `presto-coordinator:ferdinandx` | 1.84 GB | IBM presto `c77e7e6` |
| GPU worker | `presto-native-worker-gpu:ferdinandx` | 16.9 GB | IBM presto `c77e7e6` + velox pin `0ac4367` + UCX/cuDF |
| CPU worker | `presto-native-worker-cpu:ferdinandx` | 11.4 GB | Same sources, GPU=OFF. Used for `ANALYZE TABLE` (which needs aggregations the GPU path doesn't yet implement) |

Source tarballs: `/datasets/presto-cudf-exchange-testing/20260416/presto_coordinator.tar.gz` + `presto_worker.tar.gz` (owner `paiyedun`). Load → retag if you need to rebuild the local image set.

---

## Cluster topology

A multi-worker cluster is required to exercise the UCX data plane. The `cudf.exchange=true` toggle is auto-flipped by `generate_presto_config.sh` only when `-w >= 2`.

| Component | Port (host) | Notes |
|---|---|---|
| Coordinator | `${PRESTO_HOST_PORT:-8080}` | UI / `/v1/query` REST |
| Worker N HTTP | `10000 + N*10` | e.g. worker 0 = 10000, worker 1 = 10010 |
| Worker N UCX | `http_port + 3` | e.g. worker 0 = 10003, worker 1 = 10013 |

### Multi-user safety

The upstream compose files hardcode `container_name: presto-coordinator` and `8080:8080`. On a shared host (others run their own Presto on the same node), this collides. Local-only patches in `code/velox-testing/presto/docker/`:

- `docker-compose.common.yml`: drop `container_name`, change `8080:8080` → `${PRESTO_HOST_PORT:-8080}:8080`
- `docker-compose/template/docker-compose.native-gpu.yml.jinja`: drop `container_name` on both worker variants

**Always launch with `COMPOSE_PROJECT_NAME` set** so `docker compose down` (called inside `stop_presto.sh`) tears down only your stack.

---

## Launch

```bash
cd code/velox-testing/presto/scripts
mkdir -p ../docker/.hive_metastore   # first time only

COMPOSE_PROJECT_NAME=ferdinandx-presto \
PRESTO_HOST_PORT=8180 \
PRESTO_DATA_DIR=/raid/knataraj/datasets/tpch \
  ./start_native_gpu_presto.sh -w 2 -g 0,1
```

Variables:

- `-w 2` = two workers (required for UCX). `-g 0,1` = bind workers to host GPU 0 and GPU 1.
- `PRESTO_DATA_DIR` is the host path that gets bind-mounted as `/data` inside the workers. Hive table DDL points at `file:/data/...`.
- The Hive metastore is **file-based** at `code/velox-testing/presto/docker/.hive_metastore/`. Create the directory first; it persists schema metadata across cluster restarts.

### TPC-H data

Pre-generated under `/raid/knataraj/datasets/tpch/` (world-readable):

- `sf1_v2_float` — 278 MB, lightweight smoke
- `sf1k_v2_float` — 294 GB, 1 TB-scale benchmark
- `sf3k_v2_float`, `sf10k_v2_float` available if needed

`v2_float` = float-typed schema (matches Spark-Gluten SF1K suite).

---

## Schema setup

### Why two clusters

Presto's GPU worker can't run `ANALYZE TABLE` (the aggregation expressions ANALYZE emits aren't all on the GPU evaluator). To produce CBO statistics, **bring up a CPU cluster pointing at the same metastore, run `ANALYZE`, tear it down, restart the GPU cluster.** Both clusters share `.hive_metastore/`, so stats persist across the swap.

### CPU cluster for ANALYZE

```bash
# Stop GPU cluster first
COMPOSE_PROJECT_NAME=ferdinandx-presto ./stop_presto.sh

# Bring up CPU
COMPOSE_PROJECT_NAME=ferdinandx-presto-cpu \
PRESTO_HOST_PORT=8181 \
PRESTO_DATA_DIR=/raid/knataraj/datasets/tpch \
  ./start_native_cpu_presto.sh -w 1

# Run ANALYZE on every table in the schema
./analyze_tables.sh -s tpch_probe

# Stop CPU cluster
COMPOSE_PROJECT_NAME=ferdinandx-presto-cpu ./stop_presto.sh

# Bring GPU back
COMPOSE_PROJECT_NAME=ferdinandx-presto \
PRESTO_HOST_PORT=8180 \
PRESTO_DATA_DIR=/raid/knataraj/datasets/tpch \
  ./start_native_gpu_presto.sh -w 2 -g 0,1
```

Note the **distinct `COMPOSE_PROJECT_NAME` per cluster** so they don't clobber each other.

### Table DDL

`setup_benchmark_tables.sh` registers all 8 TPC-H tables under a schema. There's a known bug (it traps and deletes the generated schema files on exit), so the practical recipe is:

1. Run `setup_benchmark_tables.sh` once to generate schema `.sql` files. Copy them out of `temp-schema-dir` before the trap fires (or regenerate via `benchmark_data_tools/generate_table_schemas.py` if duckdb is available).
2. Manually `CREATE SCHEMA` + loop `CREATE TABLE` via the Presto Python driver, using `file:/data/<dir>/<table>` paths.
3. **Use SQL user `test_user`** so the script's later `DROP/RESET` flows still own the tables.

Avoid `tpch_sf1k` as a schema name on this metastore — there's a polluted instance from early manual probes with a stuck `t` user owner, undroppable via Presto. Use `tpch_probe` instead, or nuke `.hive_metastore/tpch_sf1k/` while the cluster is down.

---

## Configuration that matters for fair comparison

These are auto-set by `generate_presto_config.sh`; documenting them here so the Spark-Gluten side can match.

| Config | Value | Why it matters |
|---|---|---|
| `task.max-drivers-per-task` | `2` | Caps per-task parallelism. Spark-Gluten default is `spark.executor.cores`; for fair comparison set Spark side to 2 too |
| `cudf.exchange` | `true` | Multi-worker auto-flip — the UCX data plane |
| `cudf.exchange.server.port` | `http.port + 3` | Per-worker convention |
| `cudf.intra_node_exchange` | `true` | Required for UCX NVLink loopback on the cudf_exchange 20260212 branch |
| `cudf.partitioned_output_batch_rows` | `100_000_000` | **Hardcoded**: Presto's `PrestoToVeloxQueryConfig.cpp` has a fixed whitelist that doesn't pass this through, so the worker template hardcodes 100M. **Spark-Gluten side `cudfPartitionedOutputBatchRows` (QueryConfig.h) must match 100M** for like-for-like throughput |
| `system-memory-gb` | `2013` | B200 has 192 GB HBM × 8 GPUs ≈ 1.5 TB; this is the system-wide pool |
| `query-memory-gb` | `1912` | Per-query budget |

---

## Running queries

Two flavors; both are wrapped by scripts under `code/velox-testing/presto/scripts/`:

### EXPLAIN (plan capture for diff against Spark-Gluten)

```bash
./run_tpch_explain.sh -s tpch_probe -o /home/nfs/ferdinandx/q2/runs/presto-gpu-ucx-sf1k/plans
```

Produces 22 distributed-plan text files + 22 JSON files. The 22 distributed-plan texts are the **canonical reference** for the Spark-Gluten FLUX fragment graph: stage count, exchange types, operator names per stage.

### TIMED RUN (wall-clock + result CSV)

```bash
./run_tpch_benchmark.sh -s tpch_probe -o /home/nfs/ferdinandx/q2/runs/presto-gpu-ucx-sf1k/runtimes
```

Produces per-query `result.json` (correctness) + `query.json` (per-stage stats from `/v1/query/{id}`) + a `summary.csv` with one wall-time per query.

### Known TPC-H runner gotchas

- **Q15** uses `GROUP BY supplier_no` (an alias). Presto's planner doesn't resolve aliases inside `GROUP BY` for this template; **fix**: replace with `GROUP BY l_suppkey`. The patched template lives in the script's `tpch-templates/`.
- **OOM on builds without ANALYZE**: SF1K Q3 tried to allocate 25.8 GB on the GPU build side because the planner couldn't size the join input. Always run ANALYZE first.

---

## Verifying UCX is actually being used

Look at `Stage` operator trees in `/v1/query/{id}`. UCX-enabled stages will show `UcxExchange` between stage boundaries. Reference Q1 tree:

```
Stage3 (scan):  TableScan → CudfFilterProject → CudfAggregationPARTIAL → cudfPartitionedOutput
Stage2:         UcxExchange → CudfFilterProject → ... → cudfPartitionedOutput
Stage1:         UcxExchange → CudfOrderBy → LocalMerge → ... → cudfPartitionedOutput
Stage0 (out):   UcxExchange → CudfOrderBy → CudfToVelox → PartitionedOutput
```

3× `UcxExchange` across 4 stages = data plane fully on UCX. (No need for upstream PR 27418 — `/v1/query/{id}` already exposes the per-stage Velox operator tree.)

---

## Reference results (Presto-GPU SF1K, post-ANALYZE)

Captured 2026-04-24 under `/home/nfs/ferdinandx/q2/runs/presto-gpu-ucx-sf1k/`:

- **22-query wall-clock total: ~66 s**
- Q1: 2.57 s (4 rows) — the Spark-Gluten Q1 reference target
- Q6: 1.29 s (1 row, revenue ≈ 1.233136e11) — the same scalar value Spark-Gluten FLUX must hit

The SF1 (small) corpus also has 22-query results captured under `presto-gpu-ucx-sf1/` for plan-shape comparison; wall times there are too small (0.05–0.74 s) for throughput conclusions but the **physical plan** is the relevant comparison axis.

---

## Cleaning up

```bash
COMPOSE_PROJECT_NAME=ferdinandx-presto ./stop_presto.sh
```

Hive metastore at `.hive_metastore/` persists. To fully reset (drop all schemas + tables):

```bash
COMPOSE_PROJECT_NAME=ferdinandx-presto ./stop_presto.sh
rm -rf code/velox-testing/presto/docker/.hive_metastore
```

The cluster will come up empty next time; rerun the Schema setup steps to repopulate.

---

## Pointers

- Memory `project_presto_ucx_image.md` — image inventory + local compose patches
- Memory `project_presto_gpu_run_2026_04_24.md` — captured run results and follow-up notes
- Captured results under `/home/nfs/ferdinandx/q2/runs/presto-gpu-ucx-sf1k/` and `presto-gpu-ucx-sf1/`
- Reference IBM Presto `c77e7e6`, Velox pin `0ac4367` (same as Spark-Gluten Track-A baseline)
