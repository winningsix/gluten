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


def load_spark_stages(path, start, end, engine):
    """Return successful Spark stages intersecting the measured body."""
    stages = {}
    for line in path.read_text(errors='replace').splitlines():
        event = json.loads(line)
        if event.get('Event') not in ('SparkListenerStageSubmitted',
                                      'SparkListenerStageCompleted'):
            continue
        info = event.get('Stage Info', {})
        key = (info.get('Stage ID'), info.get('Stage Attempt ID'))
        if None in key:
            continue
        item = stages.setdefault(key, {})
        for source, target in (('Stage ID', 'id'), ('Stage Attempt ID', 'attempt'),
                               ('Stage Name', 'source_name'), ('Number of Tasks', 'tasks'),
                               ('Submission Time', 'submit_ms'),
                               ('Completion Time', 'complete_ms'),
                               ('Failure Reason', 'failure')):
            if info.get(source) is not None:
                item[target] = info[source]
    roles = {
        'cuDF-Spark': {
            1: 'left scan + GPU shuffle producer',
            2: 'right scan + GPU shuffle producer',
            3: 'shuffle read + symmetric hash join + write',
            4: 'write commit/result',
        },
        'Flux': {1: 'Spark wrapper — native pipeline'},
    }
    result = []
    for item in stages.values():
        if item.get('failure') or 'submit_ms' not in item or 'complete_ms' not in item:
            continue
        raw_start = (item['submit_ms'] * 1_000_000 - start) / 1e9
        raw_end = (item['complete_ms'] * 1_000_000 - start) / 1e9
        if raw_end < 0 or raw_start > (end - start) / 1e9:
            continue
        item.update(start_s=round(max(0, raw_start), 6),
                    end_s=round(min((end - start) / 1e9, raw_end), 6),
                    clipped_start=raw_start < 0, clipped_end=raw_end > (end - start) / 1e9,
                    label=roles.get(engine, {}).get(item['id'], item['source_name']))
        result.append(item)
    return sorted(result, key=lambda item: (item['start_s'], item['id']))


def load_flux_topology(path):
    """Structural topology only; the manifest has no per-fragment wall intervals."""
    manifest = json.loads(path.read_text())
    exchanges = manifest.get('exchanges', [])
    producers = {exchange['producer'] for exchange in exchanges}
    consumers = {exchange['consumer'] for exchange in exchanges}
    fragments = []
    for fragment in manifest.get('fragments', []):
        fragment_id = fragment['id']
        if fragment_id in producers:
            role = 'scan + hash exchange producer'
        elif fragment_id in consumers:
            role = 'exchange consumer + hash join + writer'
        else:
            role = 'native fragment'
        fragments.append(dict(id=fragment_id, parallelism=fragment['parallelism'], role=role))
    return dict(fragments=fragments,
                exchanges=[dict(id=item['id'], producer=item['producer'],
                                consumer=item['consumer'], type=item['type'])
                           for item in exchanges],
                timing='structure_only_inside_spark_wrapper')


def load_run(path, engine, device, eventlog, flux_manifest=None):
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
    provenance = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                  for name in sources}
    provenance['spark_eventlog'] = hashlib.sha256(eventlog.read_bytes()).hexdigest()
    topology = None
    if flux_manifest:
        topology = load_flux_topology(flux_manifest)
        provenance['flux_manifest'] = hashlib.sha256(flux_manifest.read_bytes()).hexdigest()
    return dict(engine=engine, transport='local', run=path.name, body_s=duration,
                output_bytes=result['output']['bytes'], files=len(result['output']['files']),
                output_gbps=result['output']['bytes'] / duration / 1e9,
                mean_sm=average(sm), mean_duty=average(duty),
                spark_stages=load_spark_stages(eventlog, start, end, engine),
                native_topology=topology,
                metrics=dict(sm=bin_samples(sm, start, end), duty=bin_samples(duty, start, end),
                             storage_write=write_rates(host, start, end, device),
                             efa_tx=[], tcp_tx=[]),
                provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cudf', required=True, type=Path)
    parser.add_argument('--flux', required=True, type=Path)
    parser.add_argument('--cudf-eventlog', required=True, type=Path)
    parser.add_argument('--flux-eventlog', required=True, type=Path)
    parser.add_argument('--flux-manifest', type=Path,
                        help='Flux runtime topology manifest; defaults under the Flux run')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--write-device', required=True, help='Validated cgroup major:minor, e.g. 259:2')
    args = parser.parse_args()
    flux_manifest = args.flux_manifest or args.flux / 'mpp-dumps/0/manifest.json'
    payload = dict(schema_version=2, title='FINRA 160M · cuDF-Spark / Flux',
                   notes='Historical individual shots, not a contemporaneous tuned A/B. '
                         'Original prewarmed inputs; 2 GPUs; Snappy; different output file counts. '
                         'GPU samples: equal-weight GPU 0/1, 250 ms bins, gaps remain null. '
                         'Storage: cgroup ' + args.write_device + ' write-counter deltas, not logical output or network throughput. '
                         'No EFA/TCP transport qualification or network counters in this local dataset. '
                         'Body excludes subsequent fsync. Initialization outside body is not displayed. '
                         'Spark stage intervals come from event logs. Flux native fragments are '
                         'structural children of one Spark wrapper stage; per-fragment wall '
                         'intervals were not captured and are not inferred.',
                   runs=[load_run(args.cudf, 'cuDF-Spark', args.write_device,
                                  args.cudf_eventlog),
                         load_run(args.flux, 'Flux', args.write_device,
                                  args.flux_eventlog, flux_manifest)])
    template = Path(__file__).with_name('timeline.html.in').read_text()
    # Prevent data from ending the JSON script element when external labels contain HTML.
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    args.output.write_text(template.replace('__DATA__', data))
    print(json.dumps({r['engine']: {k: r[k] for k in ('run', 'body_s', 'mean_sm', 'mean_duty', 'output_gbps')}
                      for r in payload['runs']}, indent=2))


if __name__ == '__main__':
    main()
