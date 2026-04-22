# mpp-substrait-runner

Standalone C++ binary that replays Gluten-dumped multi-fragment Substrait
plans directly on Velox-cudf via `MppQueryCoordinator`, bypassing
Spark/Gluten/JNI entirely.

Goal: fast iteration loop for debugging TPC-H correctness on the MPP path.
Compile-run-see-result in seconds instead of minutes through the full Spark
stack.

## Build

```bash
cd /home/ferdinandx/mpp/code/spark-gluten
cmake --build cpp/build -j$(nproc) --target mpp-substrait-runner
```

The binary is placed at `cpp/build/releases/mpp-substrait-runner` (same
output directory as the other artifacts).

## Usage

```
mpp-substrait-runner --dump-dir=<path> [--query=<name>] [--verify=<path>]
mpp-substrait-runner --builtin-smoke
```

With no arguments, the binary prints usage and exits cleanly.

### Flags

| Flag | Description |
| --- | --- |
| `--dump-dir` | Root directory containing per-query sub-directories. Each sub-directory has `fragment-{id}.pb` protobufs + a `manifest.json` describing exchanges. |
| `--query` | Run a single query by directory name (e.g. `q06`). If omitted, iterate every sub-directory. |
| `--verify` | Optional path to an `expected.csv` for row-count sanity check. |
| `--log_level` | glog level (default: `WARNING`). |
| `--builtin-smoke` | Run a hand-constructed one-fragment smoke test. Proves the harness, coordinator wiring, memory pool, and GPU exchange registration all link and run — no dump files needed. |

### Expected dump layout (aligning with Task #3)

```
<dump-dir>/<query-name>/
  fragment-0.pb     # raw substrait::Plan protobuf for fragment 0
  fragment-1.pb
  ...
  manifest.json     # schema below
  expected.csv      # optional golden output for --verify
```

`manifest.json`:

```json
{
  "exchanges": [
    {
      "id": 0,
      "producerFragmentId": 0,
      "consumerFragmentId": 1,
      "exchangeNodeId": "n3",
      "numPartitions": 4,
      "partitioningKind": "HASH",
      "partitionKeyIndices": [0]
    }
  ]
}
```

The partitioning kinds match `MppExchangeSpec.partitionType`:
`HASH`, `ROUND_ROBIN`, `SINGLE`, `RANGE`, `BROADCAST`.

## Current status (skeleton, first pass)

- CMake integration works; `--target mpp-substrait-runner` builds the binary.
- `--builtin-smoke` runs a single-fragment `Values → PartitionedOutput` plan
  end-to-end, proving the coordinator create/start/next/drain wiring.
- `--dump-dir` path is stubbed — the loader returns a `TODO` error until
  Task #3 (Gluten dump agent) produces the expected format.

## Smallest unblock from Task #3

The harness needs, per query directory:

1. One `fragment-{id}.pb` per fragment containing a raw `substrait::Plan`
   serialized exactly as `MppJniWrapper` receives it over JNI today.
2. A manifest describing exchanges using the schema above (kind strings must
   match `MppExchangeSpec.partitionType`).
3. Optional scan-split info per fragment (the `splitInfosPerFragArr` JNI
   equivalent) for leaf fragments that read Parquet.

With those three pieces, the harness reuses the same Gluten
`VeloxPlanConverter` + ValueStream-to-Exchange rewrite that
`MppJniWrapper.cc` uses.
