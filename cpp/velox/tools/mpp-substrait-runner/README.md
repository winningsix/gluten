# Gluten MPP Dump And Velox-cuDF Replay

This toolchain captures the Spark-Gluten MPP Substrait fragments that are sent
to native code, then loads the dump directly into Velox-cuDF through
`MppQueryCoordinator`. It does not require a Presto plan dump.

Use this when you want to debug the Gluten MPP plan shape or native Velox-cuDF
execution without repeatedly driving the full Spark test stack.

## What Gets Dumped

Enable dumping with Spark conf:

```bash
-Dspark.gluten.mpp.substraitDumpDir=/opt/gluten/mpp-dumps-tpch-sf1k
```

Each `MppNativeQueryExec` writes one subdirectory:

```text
<dump-dir>/<query-id>/
  fragment-0.pb              # raw substrait::Plan bytes for fragment 0
  fragment-1.pb
  ...
  splits-frag0-leaf0.pb      # raw ReadRel.LocalFiles bytes for scan leaves
  splits-frag1-leaf0.pb
  ...
  manifest.json              # fragments, driver counts, splits, exchanges
  exchange-specs.json        # same exchange JSON passed through JNI
  query.sql                  # best-effort Spark logical / child plan text
```

The protobuf files are byte-identical to the plan/split bytes that the JNI MPP
path receives during normal Spark execution.

## Dump From Spark-Gluten

Example using the shared GPU ScalaTest wrapper from the host:

```bash
export CONTAINER=gluten_q2_ibm
export GLUTEN_DIR=/opt/CURSOR-q2-rc-20260506-1-allfix
export LIBGLUTEN_PATH=/opt/CURSOR-q2-rc-20260506-1-allfix/cpp/build-dev-e50eb1eaf/releases/libgluten.so
export HOST_LOG_ROOT=/home/nfs/ferdinandx/q2/runs/sf1k-q5/CURSOR-mpp-dump
export JDK=17
export GPU=0
export PRECOMPILE=false
export SPARK_DRIVER_MAX_RESULT_SIZE=8g
export EXTRA_MVN_ARGS="-Dspark.gluten.mpp.substraitDumpDir=/opt/gluten/mpp-dumps-tpch-sf1k"

QUERY=18 /home/nfs/ferdinandx/q2/runs/CURSOR-scripts/run_gpu_scalatest_query.sh
```

The wrapper prints the host log path. In the log, look for:

```text
MppNativeQueryExec: dumped Substrait plan (...) to /opt/gluten/mpp-dumps-tpch-sf1k/<query-id>
```

If the dump directory is inside Docker and not bind-mounted to the host, copy it
out before sharing:

```bash
docker cp gluten_q2_ibm:/opt/gluten/mpp-dumps-tpch-sf1k \
  /home/nfs/ferdinandx/q2/runs/CURSOR-mpp-dumps-tpch-sf1k
```

## Build The Replay Tool

Build inside the same Spark-Gluten checkout/native build tree that produced the
`libgluten.so` you want to test:

```bash
cd /opt/CURSOR-q2-rc-20260506-1-allfix
cmake --build cpp/build-dev-e50eb1eaf -j8 --target mpp-substrait-runner
```

The binary is written under the build release directory, for example:

```text
cpp/build-dev-e50eb1eaf/releases/mpp-substrait-runner
```

If your build directory is named differently, replace `cpp/build-dev-e50eb1eaf`
with that directory.

## Replay / Load Into Velox-cuDF

Start with plan-only mode. This parses every dump, converts Substrait to Velox
plans, builds fragment/exchange specs, and exits without GPU execution:

```bash
/opt/CURSOR-q2-rc-20260506-1-allfix/cpp/build-dev-e50eb1eaf/releases/mpp-substrait-runner \
  --dump-dir=/opt/gluten/mpp-dumps-tpch-sf1k \
  --query=8 \
  --plan_only \
  --log_level=INFO
```

Run the native replay path on Velox-cuDF:

```bash
CUDA_VISIBLE_DEVICES=0 \
/opt/CURSOR-q2-rc-20260506-1-allfix/cpp/build-dev-e50eb1eaf/releases/mpp-substrait-runner \
  --dump-dir=/opt/gluten/mpp-dumps-tpch-sf1k \
  --query=8 \
  --force_drivers=1 \
  --log_level=INFO
```

Notes:

- `--query` is the dump subdirectory name. Current dumps often use numeric
  Spark query IDs such as `8`, not `q18`.
- Omit `--query` to iterate all subdirectories under `--dump-dir`.
- `--force_drivers=1` is the default and is recommended for single-GPU SF1K
  debugging. Use `--force_drivers=0` only when you want to honor the manifest
  driver counts.
- `--builtin_smoke` runs a small built-in coordinator smoke test with no dump
  files.

## Verify A Dump Exists

```bash
docker exec gluten_q2_ibm bash -lc '
ls -lah /opt/gluten/mpp-dumps-tpch-sf1k
ls -lah /opt/gluten/mpp-dumps-tpch-sf1k/8
'
```

Inspect the topology:

```bash
docker exec gluten_q2_ibm bash -lc '
python3 -m json.tool /opt/gluten/mpp-dumps-tpch-sf1k/8/manifest.json | sed -n "1,120p"
python3 -m json.tool /opt/gluten/mpp-dumps-tpch-sf1k/8/exchange-specs.json
'
```

## Troubleshooting

- If no dump appears, verify `spark.gluten.mpp.enabled=true` and
  `spark.gluten.mpp.substraitDumpDir` is set in the Spark conf that reaches the
  suite. The dump is best-effort; I/O errors are warnings and do not fail the
  query.
- If `mpp-substrait-runner` cannot find a fragment, check that
  `fragment-<id>.pb`, `manifest.json`, and `exchange-specs.json` are in the
  selected query directory.
- If plan-only succeeds but replay OOMs, keep `--force_drivers=1`, run one GPU
  job at a time, and inspect `MppWatchdog` / `FAILED-TASK` logs.
- If exchange source errors appear, check `exchange-specs.json` and the
  manifest for the producer/consumer fragment IDs, exchange node IDs, and
  partitioning kinds (`HASH`, `ROUND_ROBIN`, `SINGLE`, `RANGE`, `BROADCAST`).
