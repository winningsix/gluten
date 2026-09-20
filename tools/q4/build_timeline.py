#!/usr/bin/env python3
"""Build an offline, replayable chart from retained local FINRA measurements.

No AWS calls, no inferred network rates, no synthetic instantaneous throughput.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


def finite(value, maximum):
    try:
        number = float(value)
        return number if math.isfinite(number) and 0 <= number <= maximum else None
    except (TypeError, ValueError):
        return None


def bin_samples(rows, start, end, gpu_ids=(0, 1), width=0.25):
    """Equal GPU weighting; a missing GPU or empty interval stays null."""
    bins = {}
    for timestamp, gpu, value in rows:
        if gpu in gpu_ids and start <= timestamp < end and value is not None:
            index = int((timestamp - start) / (width * 1e9))
            bins.setdefault(index, {}).setdefault(gpu, []).append(value)
    result = []
    for index in range(math.ceil((end - start) / (width * 1e9))):
        group = bins.get(index, {})
        value = None
        if all(gpu in group for gpu in gpu_ids):
            value = sum(sum(group[gpu]) / len(group[gpu]) for gpu in gpu_ids) / len(gpu_ids)
        result.append([round(index * width, 6), None if value is None else round(value, 4)])
    return result


def write_rates(records, start, end, device):
    """One explicit device only: never sum device-mapper and leaf counters."""
    points = []
    for record in records:
        for line in record.get('io.stat', '').splitlines():
            fields = line.split()
            if fields and fields[0] == device:
                counters = dict(item.split('=', 1) for item in fields[1:])
                if 'wbytes' in counters:
                    points.append((record['epoch_ns'], int(counters['wbytes'])))
    rates = []
    for (t0, b0), (t1, b1) in zip(points, points[1:]):
        # Both endpoints must belong to the measured body. No fsync tail leakage.
        if start <= t0 < t1 <= end:
            rate = (b1 - b0) / (t1 - t0) if b1 >= b0 else None
            rates.append([round(((t0 + t1) / 2 - start) / 1e9, 6), rate])
    return rates


def load_run(path, engine, device):
    result = json.loads((path / 'result.json').read_text())
    phase, = [p for p in result['phase_timings'] if p['phase'] == 'primary_join_write']
    start, end = phase['started_epoch_ns'], phase['ended_epoch_ns']
    duration = (end - start) / 1e9
    duty, sm = [], []
    for fields in csv.reader((path / 'telemetry/nvml.csv').read_text().splitlines()):
        if len(fields) >= 3:
            try:
                duty.append((int(fields[0]), int(fields[1]), finite(fields[2], 100)))
            except ValueError:
                pass
    pattern = re.compile(r'^(\d+),\s*GPU\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$')
    for line in (path / 'telemetry/dcgm.csv').read_text().splitlines():
        match = pattern.match(line)
        if match:
            value = finite(match[4], 1)
            sm.append((int(match[1]), int(match[2]), None if value is None else value * 100))
    host = [json.loads(line) for line in (path / 'telemetry/host-io.jsonl').read_text().splitlines()]
    def average(rows):
        groups = [[v for t, g, v in rows if g == gpu and start <= t <= end and v is not None]
                  for gpu in (0, 1)]
        return sum(sum(g) / len(g) for g in groups) / 2 if all(groups) else None
    sources = ['result.json', 'telemetry/nvml.csv', 'telemetry/dcgm.csv', 'telemetry/host-io.jsonl']
    return dict(engine=engine, transport='local', run=path.name, body_s=duration,
                output_bytes=result['output']['bytes'], files=len(result['output']['files']),
                output_gbps=result['output']['bytes'] / duration / 1e9,
                mean_sm=average(sm), mean_duty=average(duty),
                metrics=dict(sm=bin_samples(sm, start, end), duty=bin_samples(duty, start, end),
                             storage_write=write_rates(host, start, end, device),
                             efa_tx=[], tcp_tx=[]),
                provenance={name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in sources})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cudf', required=True, type=Path)
    parser.add_argument('--flux', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--write-device', required=True, help='Validated cgroup major:minor, e.g. 259:2')
    args = parser.parse_args()
    payload = dict(schema_version=1, title='FINRA 160M · cuDF-Spark / Flux',
                   notes='Historical individual shots, not a contemporaneous tuned A/B. '
                         'Original prewarmed inputs; 2 GPUs; Snappy; different output file counts. '
                         'GPU samples: equal-weight GPU 0/1, 250 ms bins, gaps remain null. '
                         'Storage: cgroup ' + args.write_device + ' write-counter deltas, not logical output or network throughput. '
                         'No EFA/TCP transport qualification or network counters in this local dataset. '
                         'Body excludes subsequent fsync. Initialization outside body is not displayed.',
                   runs=[load_run(args.cudf, 'cuDF-Spark', args.write_device),
                         load_run(args.flux, 'Flux', args.write_device)])
    template = Path(__file__).with_name('timeline.html.in').read_text()
    # Prevent data from ending the JSON script element when external labels contain HTML.
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    args.output.write_text(template.replace('__DATA__', data))
    print(json.dumps({r['engine']: {k: r[k] for k in ('run', 'body_s', 'mean_sm', 'mean_duty', 'output_gbps')}
                      for r in payload['runs']}, indent=2))


if __name__ == '__main__':
    main()
