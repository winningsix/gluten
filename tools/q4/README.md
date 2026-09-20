# Q4 measured GPU timeline

Open `finra-local-timeline.html` in a browser. It is self-contained: no CDN,
server, AWS credentials, or live telemetry service is needed. Play/pause,
scrub and change playback speed. The chart is recorded-data replay, not a
live monitor. Blue is cuDF-Spark; green is Flux.

## Evidence boundaries

- Local FINRA160M, two GPUs, original prewarmed ORC/Parquet inputs.
- cuDF-Spark: historical `cudf-primary-160m-h32-warm-r01`, skipMerge, 64 output files.
- Flux: `flux-primary-160m-parallel-fill-candidate-r01-p1`, two output files.
  This is the first candidate repetition, not the two-repetition mean.
- Both produce 160M rows. Different file layouts and dates mean this is not a
  contemporaneous, fully tuned A/B. The chart is not an engine speedup claim.
- True DCGM SM Active and NVML duty use GPU 0/1, equally weighted, in 250 ms
  bins. A missing GPU sample leaves a gap, not an imputed zero.
- Throughput cards are output file bytes / job-body seconds. They are **not**
  instantaneous operator throughput. The storage curve is one explicitly
  selected cgroup device's write-counter delta; it can include other job I/O
  and does not measure logical output progress or network traffic.
- The body excludes the later fsync tail. Neither an ended run nor missing
  measurements are padded with zero.
- There is no EFA/TCP-qualified network comparison in this local evidence.
  Those views remain unavailable until real measurements are imported. EFA
  traffic must not be inferred from generic Ethernet counters; TCP attribution
  requires a verified TCP-only interface or protocol-specific counters.

## Regenerate and test

```sh
python3 tools/q4/build_timeline.py \
  --cudf /path/to/cudf-primary-160m-h32-warm-r01 \
  --flux /path/to/flux-primary-160m-parallel-fill-candidate-r01-p1 \
  --write-device 259:2 \
  --output tools/q4/finra-local-timeline.html
python3 -m unittest discover -s tools/q4 -v
```

Validate the major:minor mapping against the original run before changing
`--write-device`. Do not sum device-mapper and underlying NVMe counters.
The embedded data includes SHA-256 digests of the original inputs.

## Additional measured transports

The JSON under the chart's provenance expander is the import schema:
`schema_version=1`, `title`, `notes`, and `runs`. Each run includes `engine`
(`cuDF-Spark` or `Flux`), `transport` (`local`, `efa`, or `tcp`), `run`,
`body_s`, `files`, `output_gbps`, `mean_sm`, `mean_duty`, `provenance`, and
`metrics`. Series are `[seconds_from_body_start, value_or_null]` pairs.
Metric keys: `sm`, `duty` (percent); `storage_write`, `efa_tx`, `tcp_tx`
(decimal GB/s). Supply TX-only network totals, not TX+RX double counting.
Retain hardware, workload, runtime, sampling and transport proof in provenance.
Never relabel a local run as EFA/TCP. Import replaces the displayed dataset.
