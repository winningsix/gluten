# Task Wall-Time Metrics & Sunburst Visualization

## Overview

This directory contains tools for analyzing Spark task wall-time distribution
in the Gluten-Velox-cuDF stack. The instrumentation measures **where time is
spent** during Spark task execution at two levels of granularity.

## Metrics Collected

### Level 1: Task-Level Breakdown (Java side, `TaskWallTimeTracker`)

Each Spark task records the following wall-time segments (all in nanoseconds)
via a `ThreadLocal` tracker (`TaskWallTimeTracker`):

| Metric | What it measures |
|---|---|
| `taskWallNanos` | Total task wall time (from `onTaskStart` to `logAndReset`) |
| `planBuildNanos` | Time building the native execution plan |
| `nativeHasNextNanos` | Time in `ColumnarBatchOutIterator.hasNext()` (native calls) |
| `nativeNextNanos` | Time in `ColumnarBatchOutIterator.next()` (native calls) |
| `arrowImportNanos` | Time importing Arrow batches to Spark columnar batches |
| `shuffleReadInitNanos` | Time initializing shuffle readers |
| `broadcastBuildNanos` | Time building broadcast relations |
| `shuffleWriterInitNanos` | Time initializing the native shuffle writer |
| `shuffleWriteJniNanos` | Time in `shuffleWriter.write()` JNI calls |
| `shuffleWriteStopNanos` | Time in `shuffleWriter.stop()` (flush + spill) |
| `shuffleWriteMetaNanos` | Time writing shuffle metadata & committing |
| `taskCleanupNanos` | Time in task resource cleanup |

**"Native Execution"** = `nativeHasNextNanos + nativeNextNanos`. This
represents the total time the task thread spends inside native (Velox/cuDF)
code, including all operator execution, data movement, and GPU kernel launches.

A `nativeNestingDepth` counter prevents double-counting when native calls are
nested (e.g., a shuffle-read inside a native `hasNext` call).

These metrics are logged as `[TASK_TIMING]` lines to executor stderr.

### Level 2: Velox Operator Breakdown (C++/Spark SQL metrics)

Each cuDF operator in Velox reports **two** timing metrics via Spark SQL
accumulators:

| Metric type | How it's measured | What it shows |
|---|---|---|
| `time of <op>` (Wall Time) | CPU-side `std::chrono` around the operator's `getOutput()` | Total wall clock time the task thread spends in that operator, including GPU kernel launches, synchronization waits, data prep, and IO |
| `gpu compute time` (GPU Compute) | CUDA events (`cudaEventRecord` before/after GPU kernels) | Actual GPU kernel execution time only |

The difference between Wall Time and GPU Compute reveals CPU-side overhead
(IO, memory allocation, data format conversion, kernel launch latency, sync).

**Operators tracked:**

| Operator class | Display name | Primary metric |
|---|---|---|
| `FileSourceScanExecTransformer` | Scan+Filter | `time of scan and filter` |
| `CudfFilterExecTransformer` | Filter | `time of filter` |
| `CudfProjectExecTransformer` | Project | `time of project` |
| `CudfFlushableHashAggregateExecTransformer` | HashAgg (partial) | `time of aggregation` |
| `CudfRegularHashAggregateExecTransformer` | HashAgg (final) | `time of aggregation` |
| `CudfSortExecTransformer` | Sort | `time of sort` |
| `CudfInputIteratorTransformer` | InputIterator | `time of reducer/operator/broadcast input` |
| `CudfShuffledHashJoinExecTransformer` | ShuffledHashJoin | `time of hash build` + `probe` + projections |
| `CudfBroadcastHashJoinExecTransformer` | BroadcastHashJoin | same as above |
| `TakeOrderedAndProjectExecTransformer` | TopN | `time of sort` |
| `CudfWindowExecTransformer` | Window | `time of window` |
| `CudfExpandExecTransformer` | Expand | `time of expand` |

## Sunburst Chart (`metrics_sunburst.py`)

Generates a nested donut chart with three concentric rings:

- **Inner ring (Level 1)**: Task wall-time breakdown into the categories above
  (Native Execution, Shuffle Write, Plan Build, etc.)
- **Middle ring (Level 2)**: Drill-down of each L1 segment that has sub-metrics.
  Currently, "Native Execution" expands into per-operator wall times.
  Non-drillable L1 segments show as faded pass-through wedges.
- **Outer ring (Level 3)**: GPU Compute vs CPU Overhead split for each L2
  operator. Green = actual GPU kernel time; Amber = CPU-side overhead
  (IO, data prep, sync, kernel launch latency). This reveals how much of an
  operator's wall time is spent on GPU vs CPU work.

Labels are placed directly on wedges (large segments get inline text, small
segments get arrow annotations pointing outward).

See [example.png](example.png) for a sample output (q1+q6+q9+q5+q21, SF100).

### Data Sources

The chart joins two data sources by `taskAttemptId`:

1. **Spark event log** (`SparkListenerTaskEnd` events):
   - `executorRunTime` for baseline comparison
   - SQL accumulator updates for operator-level "time of ..." metrics
2. **Executor stderr** (`[TASK_TIMING]` log lines):
   - All `TaskWallTimeTracker` fields listed above

### Filtering

- Only tasks with `nativeExec > 0` are included (filters out pure metadata
  stages like table loading).
- Slices below `--min-pct` threshold are merged into "Other (small)".

### Usage

```bash
python3 metrics_sunburst.py <event_log> <executor_log> \
    [--output chart.png] \
    [--min-pct 1.0] \
    [--title-suffix '(q1+q6+q9+q5+q21, SF100)']
```

### Example Output

```
=== Level 1 (total wall = 1,277,161.7 ms) ===
  Native Execution                 816090.5 ms  ( 63.9%)
  Shuffle Write (JNI)               91988.8 ms  (  7.2%)
  Shuffle Write (stop)             149418.1 ms  ( 11.7%)
  Shuffle Read Init                 31142.2 ms  (  2.4%)
  Plan Build                        19941.4 ms  (  1.6%)
  Framework / Others               153017.6 ms  ( 12.0%)

=== Level 2: Native Execution breakdown ===
  InputIterator    wall= 329258.7 ms (40.3%)  gpu=     0.0 ms ( 0.0%)
  Scan+Filter      wall= 323823.9 ms (39.7%)  gpu= 51165.6 ms (15.8%)
  Project          wall=  79016.5 ms ( 9.7%)  gpu=  5314.9 ms ( 6.7%)
  ShuffledHashJoin wall=  45874.6 ms ( 5.6%)  gpu= 17759.1 ms (38.7%)
  HashAgg (partial)wall=  17495.5 ms ( 2.1%)  gpu=  6199.9 ms (35.4%)
```

Key observation: Scan+Filter is only 15.8% GPU, meaning 84% of its wall time
is CPU-side IO and data preparation. This is a common pattern in GPU-accelerated
query engines where data loading dominates compute.
