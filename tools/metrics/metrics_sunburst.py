#!/usr/bin/env python3
"""
Generate a 3-ring sunburst chart of task wall-time distribution.

Ring 1 (inner)  = Level 1: Task wall-time breakdown
                  (Native Execution, Shuffle Write, etc.)
Ring 2 (middle) = Level 2: Operator drill-down of Native Execution
                  (Scan+Filter, Project, Join, etc.)
Ring 3 (outer)  = Level 3: GPU Compute vs CPU Overhead split
                  for each L2 operator

Annotations are placed directly on wedges instead of a legend box.

Data sources:
  - Spark event log  (executorRunTime, operator accumulators)
  - Executor stderr  ([TASK_TIMING] custom metrics)

Usage:
  python3 metrics_sunburst.py <event_log> <executor_stderr> \
      [--output chart.png] [--min-pct 1.0] \
      [--title-suffix '(q1+q6, SF100)']
"""
import json
import math
import re
import sys
import os
import argparse
from collections import defaultdict

# ============================================================
# Constants
# ============================================================

SEGMENT_FIELDS = [
    "planBuildNanos", "nativeHasNextNanos", "nativeNextNanos",
    "arrowImportNanos", "shuffleReadInitNanos",
    "broadcastBuildNanos",
    "shuffleWriterInitNanos", "shuffleWriteJniNanos",
    "shuffleWriteStopNanos", "shuffleWriteMetaNanos",
    "taskCleanupNanos",
]

ALL_FIELDS = ["taskWallNanos"] + SEGMENT_FIELDS

TRACKER_RE = re.compile(
    r"\[TASK_TIMING\]\s+"
    r"stageId=(\d+)\s+taskAttemptId=(\d+)\s+"
    + r"\s+".join(rf"{f}=(\d+)" for f in ALL_FIELDS)
)

L1_GROUPS = [
    ("Native Execution",
     ["nativeHasNextNanos", "nativeNextNanos"], True),
    ("Shuffle Write (JNI)",
     ["shuffleWriteJniNanos"], False),
    ("Shuffle Write (stop)",
     ["shuffleWriteStopNanos"], False),
    ("Shuffle Write (init+meta)",
     ["shuffleWriterInitNanos", "shuffleWriteMetaNanos"], False),
    ("Shuffle Read Init",
     ["shuffleReadInitNanos"], False),
    ("Plan Build",
     ["planBuildNanos"], False),
    ("Arrow Import",
     ["arrowImportNanos"], False),
    ("Broadcast Build",
     ["broadcastBuildNanos"], False),
    ("Task Cleanup",
     ["taskCleanupNanos"], False),
]

OPERATOR_PRIMARY_METRIC = {
    "FileSourceScanExecTransformer parquet":
        "time of scan and filter",
    "CudfFilterExecTransformer":
        "time of filter",
    "CudfProjectExecTransformer":
        "time of project",
    "CudfFlushableHashAggregateExecTransformer":
        "time of aggregation",
    "CudfRegularHashAggregateExecTransformer":
        "time of aggregation",
    "CudfSortExecTransformer":
        "time of sort",
    "CudfWriteFilesExecTransformer":
        "time of write",
    "CudfInputIteratorTransformer":
        None,
    "CudfShuffledHashJoinExecTransformer":
        None,
    "CudfBroadcastHashJoinExecTransformer":
        None,
    "TakeOrderedAndProjectExecTransformer":
        "time of sort",
    "CudfWindowExecTransformer":
        "time of window",
    "CudfExpandExecTransformer":
        "time of expand",
    "CudfNestedLoopJoinExecTransformer":
        None,
    "CudfTopNRowNumberExecTransformer":
        "time of sort",
}

GPU_COMPUTE_METRIC = "gpu compute time"

INPUT_ITER_METRICS = {
    "time of reducer input",
    "time of operator input",
    "time of broadcast input",
}

JOIN_METRICS = {
    "time of hash build",
    "time of hash probe",
    "time of postProjection",
    "time of stream preProjection",
    "time to build preProjection",
}

DISPLAY_NAMES = {
    "FileSourceScanExecTransformer parquet": "Scan+Filter",
    "CudfFilterExecTransformer": "Filter",
    "CudfProjectExecTransformer": "Project",
    "CudfFlushableHashAggregateExecTransformer":
        "HashAgg (partial)",
    "CudfRegularHashAggregateExecTransformer":
        "HashAgg (final)",
    "CudfSortExecTransformer": "Sort",
    "CudfWriteFilesExecTransformer": "Write",
    "CudfInputIteratorTransformer": "InputIterator",
    "CudfShuffledHashJoinExecTransformer": "ShuffledHashJoin",
    "CudfBroadcastHashJoinExecTransformer": "BroadcastHashJoin",
    "TakeOrderedAndProjectExecTransformer": "TopN",
    "CudfWindowExecTransformer": "Window",
    "CudfExpandExecTransformer": "Expand",
    "CudfNestedLoopJoinExecTransformer": "NestedLoopJoin",
    "CudfTopNRowNumberExecTransformer": "TopNRowNumber",
}


# ============================================================
# Parsing
# ============================================================

def parse_event_log(path):
    tasks = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or '"SparkListenerTaskEnd"' not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("Event") != "SparkListenerTaskEnd":
                continue
            reason = ev.get(
                "Task End Reason", {}).get("Reason", "")
            if reason != "Success":
                continue
            info = ev["Task Info"]
            task_id = info["Task ID"]
            stage_id = ev["Stage ID"]
            run_time_ms = None
            for acc in info.get("Accumulables", []):
                if acc.get("Name") == \
                        "internal.metrics.executorRunTime":
                    run_time_ms = acc["Update"]
                    break
            if run_time_ms is not None:
                tasks[task_id] = (stage_id, run_time_ms)
    return tasks


def parse_tracker_log(path):
    tasks = {}
    with open(path) as f:
        for line in f:
            m = TRACKER_RE.search(line)
            if not m:
                continue
            stage_id = int(m.group(1))
            task_id = int(m.group(2))
            values = {}
            for i, field in enumerate(ALL_FIELDS):
                values[field] = int(m.group(3 + i))
            values["stageId"] = stage_id
            tasks[task_id] = values
    return tasks


# ============================================================
# Level 1 aggregation
# ============================================================

def aggregate(spark_tasks, tracker_tasks):
    matched = set(spark_tasks.keys()) & set(tracker_tasks.keys())
    totals = defaultdict(int)
    total_wall = 0
    total_exec_run = 0
    n_tasks = 0

    for tid in matched:
        tracker = tracker_tasks[tid]
        native_exec = (tracker["nativeHasNextNanos"]
                       + tracker["nativeNextNanos"])
        if native_exec <= 0:
            continue
        n_tasks += 1
        total_wall += tracker["taskWallNanos"]
        _, run_ms = spark_tasks[tid]
        total_exec_run += run_ms * 1_000_000
        for f in SEGMENT_FIELDS:
            totals[f] += tracker[f]

    return totals, total_wall, total_exec_run, n_tasks


def build_l1_slices(totals, total_wall, min_pct):
    slices = []
    accounted = 0
    for label, fields, _ in L1_GROUPS:
        val = sum(totals[f] for f in fields)
        if val > 0:
            slices.append((label, val))
            accounted += val

    others = max(0, total_wall - accounted)
    if others > 0:
        slices.append(("Framework / Others", others))

    threshold = total_wall * min_pct / 100.0
    main = [(l, v) for l, v in slices if v >= threshold]
    small_sum = sum(v for l, v in slices if v < threshold)
    if small_sum > 0:
        main.append(("Other (small)", small_sum))
    return main


# ============================================================
# Level 2 + Level 3: operator breakdown + GPU/CPU split
# ============================================================

def build_acc_map(event_log_path):
    acc_map = {}
    with open(event_log_path) as f:
        for line in f:
            line = line.strip()
            if 'SparkListenerSQLExecutionStart' not in line:
                continue
            ev = json.loads(line)
            if ev.get('Event') != \
                    'org.apache.spark.sql.execution.ui' \
                    '.SparkListenerSQLExecutionStart':
                continue
            desc = ev.get('description', '')
            if 'query' not in desc.lower():
                continue
            _walk_acc(ev['sparkPlanInfo'], acc_map)
    return acc_map


def _walk_acc(node, acc_map):
    name = node.get('nodeName', '')
    for m in node.get('metrics', []):
        acc_map[m['accumulatorId']] = (
            m['name'], name, m['metricType'])
    for child in node.get('children', []):
        _walk_acc(child, acc_map)


def parse_task_accumulables(event_log_path, acc_ids):
    tasks = {}
    with open(event_log_path) as f:
        for line in f:
            line = line.strip()
            if 'SparkListenerTaskEnd' not in line:
                continue
            ev = json.loads(line)
            if ev.get('Event') != 'SparkListenerTaskEnd':
                continue
            reason = ev.get(
                'Task End Reason', {}).get('Reason', '')
            if reason != 'Success':
                continue
            info = ev['Task Info']
            tid = info['Task ID']
            vals = {}
            for a in info.get('Accumulables', []):
                aid = a['ID']
                if aid in acc_ids:
                    v = a.get('Update', 0)
                    vals[aid] = int(v) if v else 0
            if vals:
                tasks[tid] = vals
    return tasks


def _match_operator(opname):
    """Match an operator name to a known key, return key or None."""
    op_stripped = opname.strip()
    for key in OPERATOR_PRIMARY_METRIC:
        if op_stripped.startswith(key):
            return key
    return None


def aggregate_operators_with_gpu(event_log_path,
                                 tracker_tasks, spark_tasks):
    """Aggregate wall time AND gpu compute time per operator."""
    acc_map = build_acc_map(event_log_path)

    # Collect wall-time accumulators (same as before)
    wall_ids = set()
    # Collect gpu compute time accumulators (new)
    gpu_ids = set()
    # Map gpu acc_id -> display name
    gpu_id_to_display = {}

    for aid, (mname, opname, mtype) in acc_map.items():
        if mtype != 'nsTiming':
            continue
        matched_op = _match_operator(opname)
        if matched_op is None:
            continue

        if mname == GPU_COMPUTE_METRIC:
            gpu_ids.add(aid)
            display = DISPLAY_NAMES.get(matched_op,
                                        opname.strip())
            gpu_id_to_display[aid] = display
            continue

        primary = OPERATOR_PRIMARY_METRIC[matched_op]
        if primary is not None and mname == primary:
            wall_ids.add(aid)
        elif matched_op == 'CudfInputIteratorTransformer' \
                and mname in INPUT_ITER_METRICS:
            wall_ids.add(aid)
        elif matched_op in (
                'CudfShuffledHashJoinExecTransformer',
                'CudfBroadcastHashJoinExecTransformer',
                'CudfNestedLoopJoinExecTransformer') \
                and mname in JOIN_METRICS:
            wall_ids.add(aid)

    all_ids = wall_ids | gpu_ids
    task_accs = parse_task_accumulables(
        event_log_path, all_ids)

    matched = set(spark_tasks.keys()) & \
        set(tracker_tasks.keys()) & set(task_accs.keys())

    op_wall = defaultdict(int)
    op_gpu = defaultdict(int)
    total_native = 0
    n = 0

    for tid in matched:
        tracker = tracker_tasks[tid]
        native = (tracker["nativeHasNextNanos"]
                  + tracker["nativeNextNanos"])
        if native <= 0:
            continue
        n += 1
        total_native += native
        for aid, val in task_accs[tid].items():
            if aid in gpu_ids:
                op_gpu[gpu_id_to_display[aid]] += val
            if aid in wall_ids:
                mname, opname, _ = acc_map[aid]
                display = None
                for key, dname in DISPLAY_NAMES.items():
                    if opname.strip().startswith(key):
                        display = dname
                        break
                if display is None:
                    display = opname.strip()
                op_wall[display] += val

    return op_wall, op_gpu, total_native, n


def build_l2_slices(op_totals, total_native, min_pct):
    slices = []
    accounted = 0
    for label, val in sorted(
            op_totals.items(), key=lambda x: -x[1]):
        if val > 0:
            slices.append((label, val))
            accounted += val

    overhead = max(0, total_native - accounted)
    if overhead > 0:
        slices.append(("Driver/JNI overhead", overhead))

    threshold = total_native * min_pct / 100.0
    main = [(l, v) for l, v in slices if v >= threshold]
    small_sum = sum(v for l, v in slices if v < threshold)
    if small_sum > 0:
        main.append(("Other ops (small)", small_sum))
    return main


def build_l3_map(l2_slices, op_gpu):
    """For each L2 operator, split into GPU Compute + CPU Overhead."""
    l3 = {}
    for label, wall_val in l2_slices:
        gpu_val = op_gpu.get(label, 0)
        if gpu_val > 0 and gpu_val < wall_val:
            cpu_val = wall_val - gpu_val
            l3[label] = [
                (f"GPU", gpu_val),
                (f"CPU", cpu_val),
            ]
        elif gpu_val >= wall_val > 0:
            l3[label] = [(f"GPU", wall_val)]
        else:
            l3[label] = None
    return l3


# ============================================================
# Sunburst drawing with on-wedge annotations
# ============================================================

def _wedge_mid_angle(wedge):
    """Mid-angle of a wedge in radians, measured from 3 o'clock."""
    mid_deg = (wedge.theta1 + wedge.theta2) / 2.0
    return math.radians(mid_deg)


def _polar_to_xy(angle_rad, radius):
    """Convert (angle in radians from East, radius) to (x, y)."""
    return (radius * math.cos(angle_rad),
            radius * math.sin(angle_rad))


def _annotate_wedges(ax, wedges, labels, pcts, ring_mid_r,
                     ring_outer_r, min_angle_deg=8,
                     fontsize=7, fmt_fn=None):
    """Place annotations on or near wedges.

    - Large wedges (> min_angle_deg): text inside the wedge.
    - Small wedges: arrow annotation pointing outward.
    """
    if fmt_fn is None:
        def fmt_fn(label, pct):
            return f"{label}\n{pct:.1f}%"

    placed = []

    for wedge, label, pct in zip(wedges, labels, pcts):
        if not label:
            placed.append((label, False))
            continue
        span = abs(wedge.theta2 - wedge.theta1)
        mid_rad = _wedge_mid_angle(wedge)

        if span >= min_angle_deg:
            x, y = _polar_to_xy(mid_rad, ring_mid_r)
            txt = fmt_fn(label, pct)
            fs = fontsize if span >= 15 else fontsize - 1
            ax.text(x, y, txt,
                    ha='center', va='center',
                    fontsize=fs, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='black', alpha=0.45,
                              edgecolor='none'))
            placed.append((label, True))
        else:
            placed.append((label, False))

    return placed


def _annotate_small_wedges(ax, wedges, labels, pcts, vals,
                           ring_mid_r, base_r,
                           fontsize=6.5, fmt_fn=None,
                           min_angle_deg=8):
    """Draw arrow annotations for small wedges that didn't get
    inline labels. Spreads labels vertically to avoid overlap."""
    if fmt_fn is None:
        def fmt_fn(label, pct, val_ns):
            ms = val_ns / 1e6
            return f"{label}: {ms:,.0f}ms ({pct:.1f}%)"

    smalls = []
    for i, (wedge, label, pct) in enumerate(
            zip(wedges, labels, pcts)):
        if not label:
            continue
        span = abs(wedge.theta2 - wedge.theta1)
        if span < min_angle_deg and pct >= 0.3:
            mid_rad = _wedge_mid_angle(wedge)
            smalls.append((mid_rad, label, pct, vals[i], wedge))

    if not smalls:
        return

    # Sort by angle to space labels vertically
    smalls.sort(key=lambda x: x[0])

    text_r = base_r + 0.30
    min_y_gap = 0.12
    prev_y = None

    for mid_rad, label, pct, val_ns, wedge in smalls:
        wx, wy = _polar_to_xy(mid_rad, ring_mid_r)
        tx, ty = _polar_to_xy(mid_rad, text_r)

        # Push labels apart vertically
        if prev_y is not None and abs(ty - prev_y) < min_y_gap:
            ty = prev_y - min_y_gap
            tx_sign = 1 if tx >= 0 else -1
            tx = tx_sign * max(abs(tx), text_r * 0.7)
        prev_y = ty

        txt = fmt_fn(label, pct, val_ns)
        ax.annotate(
            txt, xy=(wx, wy), xytext=(tx, ty),
            fontsize=fontsize,
            arrowprops=dict(arrowstyle='-',
                            color='gray', lw=0.6),
            ha='left' if tx >= 0 else 'right',
            va='center',
            bbox=dict(boxstyle='round,pad=0.1',
                      facecolor='white', alpha=0.8,
                      edgecolor='gray', linewidth=0.4))


def draw_sunburst_3ring(l1_slices, l2_map, l3_map,
                        total_wall, total_exec_run,
                        n_tasks, output_path,
                        title_extra=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba, to_hex
        import matplotlib.patches as mpatches
    except ImportError:
        print("ERROR: matplotlib not installed.",
              file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(18, 12))

    cmap = plt.get_cmap("tab10")
    l1_colors = [cmap(i % 10) for i in range(len(l1_slices))]

    # ---- Ring 1: Level 1 (inner) ----
    R1_INNER, R1_OUTER = 0.35, 0.60
    r1_vals = [v for _, v in l1_slices]
    r1_labels = [l for l, _ in l1_slices]

    w1, _ = ax.pie(
        r1_vals, radius=R1_OUTER, colors=l1_colors,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=R1_OUTER - R1_INNER,
                        edgecolor='white', linewidth=1.5))

    r1_pcts = [v / total_wall * 100 for v in r1_vals]
    _annotate_wedges(
        ax, w1, r1_labels, r1_pcts,
        ring_mid_r=(R1_INNER + R1_OUTER) / 2,
        ring_outer_r=R1_OUTER,
        min_angle_deg=12, fontsize=7.5)
    _annotate_small_wedges(
        ax, w1, r1_labels, r1_pcts, r1_vals,
        ring_mid_r=(R1_INNER + R1_OUTER) / 2,
        base_r=R1_OUTER, min_angle_deg=12, fontsize=6)

    # ---- Ring 2: Level 2 (middle) ----
    R2_INNER, R2_OUTER = 0.62, 0.90
    r2_vals = []
    r2_colors = []
    r2_labels = []

    # Distinct palette for L2 sub-segments
    l2_cmap = plt.get_cmap("Set2")

    r2_is_passthrough = []

    for i, (l1_label, l1_val) in enumerate(l1_slices):
        base_color = l1_colors[i]
        if l1_label in l2_map and l2_map[l1_label]:
            subs = l2_map[l1_label]
            n_subs = len(subs)
            for j, (sub_label, sub_val) in enumerate(subs):
                r2_vals.append(sub_val)
                r2_labels.append(sub_label)
                r2_colors.append(l2_cmap(j % 8))
                r2_is_passthrough.append(False)
        else:
            r2_vals.append(l1_val)
            r2_labels.append(l1_label)
            r, g, b, _ = to_rgba(base_color)
            r2_colors.append((r, g, b, 0.45))
            r2_is_passthrough.append(True)

    w2, _ = ax.pie(
        r2_vals, radius=R2_OUTER, colors=r2_colors,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=R2_OUTER - R2_INNER,
                        edgecolor='white', linewidth=1.0))

    # Only annotate non-passthrough (drilled-down) segments;
    # passthrough segments already have labels on the inner ring.
    native_total = dict(l1_slices).get("Native Execution", 1)
    r2_pcts = []
    r2_ann_labels = []
    for j, (label, val) in enumerate(zip(r2_labels, r2_vals)):
        if r2_is_passthrough[j]:
            r2_pcts.append(0)
            r2_ann_labels.append("")
        else:
            r2_pcts.append(val / native_total * 100)
            r2_ann_labels.append(label)

    _annotate_wedges(
        ax, w2, r2_ann_labels, r2_pcts,
        ring_mid_r=(R2_INNER + R2_OUTER) / 2,
        ring_outer_r=R2_OUTER,
        min_angle_deg=10, fontsize=6.5)
    _annotate_small_wedges(
        ax, w2, r2_ann_labels, r2_pcts, r2_vals,
        ring_mid_r=(R2_INNER + R2_OUTER) / 2,
        base_r=R2_OUTER, min_angle_deg=10, fontsize=6)

    # ---- Ring 3: Level 3 (outer) - GPU vs CPU ----
    R3_INNER, R3_OUTER = 0.92, 1.15
    r3_vals = []
    r3_colors = []
    r3_labels = []
    r3_parent_labels = []

    GPU_COLOR = (0.2, 0.7, 0.3, 0.85)   # green
    CPU_COLOR = (0.9, 0.55, 0.2, 0.85)   # amber

    for i, (l2_label, l2_val) in enumerate(
            zip(r2_labels, r2_vals)):
        if l3_map and l2_label in l3_map \
                and l3_map[l2_label] is not None:
            for sub_label, sub_val in l3_map[l2_label]:
                r3_vals.append(sub_val)
                r3_labels.append(sub_label)
                r3_parent_labels.append(l2_label)
                if sub_label == "GPU":
                    r3_colors.append(GPU_COLOR)
                else:
                    r3_colors.append(CPU_COLOR)
        else:
            r3_vals.append(l2_val)
            r3_labels.append("")
            r3_parent_labels.append(l2_label)
            c = r2_colors[i]
            r, g, b, _ = to_rgba(c)
            r3_colors.append((r, g, b, 0.25))

    w3, _ = ax.pie(
        r3_vals, radius=R3_OUTER, colors=r3_colors,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=R3_OUTER - R3_INNER,
                        edgecolor='white', linewidth=0.6))

    # Annotate L3 wedges that are large enough
    for wedge, label, parent, val in zip(
            w3, r3_labels, r3_parent_labels, r3_vals):
        if not label:
            continue
        span = abs(wedge.theta2 - wedge.theta1)
        if span < 6:
            continue
        parent_wall = dict(zip(r2_labels, r2_vals)).get(
            parent, 1)
        pct = val / parent_wall * 100 if parent_wall > 0 else 0
        mid_rad = _wedge_mid_angle(wedge)
        x, y = _polar_to_xy(
            mid_rad, (R3_INNER + R3_OUTER) / 2)
        fs = 5.5 if span < 12 else 6
        ax.text(x, y, f"{label}\n{pct:.0f}%",
                ha='center', va='center',
                fontsize=fs, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor='black', alpha=0.4,
                          edgecolor='none'))

    # ---- Compact legend for ring meanings + L3 colors ----
    legend_items = [
        mpatches.Patch(facecolor='gray', alpha=0.5,
                       edgecolor='white'),
        mpatches.Patch(facecolor=GPU_COLOR,
                       edgecolor='white'),
        mpatches.Patch(facecolor=CPU_COLOR,
                       edgecolor='white'),
    ]
    legend_labels = [
        "Inner=L1  |  Middle=L2 operators",
        "Outer: GPU Compute",
        "Outer: CPU Overhead (Wall - GPU)",
    ]
    ax.legend(legend_items, legend_labels,
              loc='lower right',
              bbox_to_anchor=(1.0, 0.0),
              fontsize=8, framealpha=0.85,
              handlelength=1.5, handleheight=1.0,
              title="Ring Legend", title_fontsize=9)

    # ---- Title ----
    wall_ms = total_wall / 1e6
    exec_ms = total_exec_run / 1e6
    title = (
        f"Task Wall-Time Sunburst "
        f"({n_tasks} tasks, nativeExec > 0)"
        f"{title_extra}\n"
        f"Total taskWallNanos: {wall_ms:,.0f} ms  |  "
        f"executorRunTime: {exec_ms:,.0f} ms")
    ax.set_title(title, fontsize=11, pad=20)

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="3-ring sunburst chart of task wall-time "
                    "with GPU/CPU split")
    parser.add_argument("event_log",
                        help="Spark event log file")
    parser.add_argument("executor_log",
                        help="Executor stderr with "
                             "[TASK_TIMING] lines")
    parser.add_argument("--output", "-o",
                        default="metrics_sunburst.png",
                        help="Output image path")
    parser.add_argument("--min-pct", type=float, default=1.0,
                        help="Merge slices below this %% "
                             "threshold")
    parser.add_argument("--title-suffix", default="",
                        help="Extra text for chart title")
    args = parser.parse_args()

    # --- Parse ---
    spark_tasks = parse_event_log(args.event_log)
    tracker_tasks = parse_tracker_log(args.executor_log)
    print(f"Event log tasks: {len(spark_tasks)}")
    print(f"Tracker tasks:   {len(tracker_tasks)}")

    totals, total_wall, total_exec_run, n_tasks = aggregate(
        spark_tasks, tracker_tasks)
    print(f"Matched (nativeExec > 0): {n_tasks}")

    if n_tasks == 0:
        print("No tasks with nativeExec > 0. Nothing to plot.")
        sys.exit(0)

    # --- Level 1 ---
    l1_slices = build_l1_slices(
        totals, total_wall, args.min_pct)

    print(f"\n=== Level 1 (total wall = "
          f"{total_wall / 1e6:,.1f} ms) ===")
    for label, val in l1_slices:
        pct = val / total_wall * 100
        print(f"  {label:30s} {val / 1e6:>10.1f} ms  "
              f"({pct:5.1f}%)")

    # --- Level 2 + GPU compute ---
    op_wall, op_gpu, total_native, n_l2 = \
        aggregate_operators_with_gpu(
            args.event_log, tracker_tasks, spark_tasks)

    print(f"\n=== Level 2: Native Execution breakdown "
          f"({n_l2} tasks) ===")
    print(f"Total nativeExec = {total_native / 1e6:,.1f} ms")
    for label, val in sorted(
            op_wall.items(), key=lambda x: -x[1]):
        gpu = op_gpu.get(label, 0)
        pct = val / total_native * 100
        gpu_pct = gpu / val * 100 if val > 0 else 0
        print(f"  {label:30s} "
              f"wall={val / 1e6:>10.1f} ms ({pct:5.1f}%)  "
              f"gpu={gpu / 1e6:>8.1f} ms "
              f"({gpu_pct:5.1f}% of wall)")

    l2_slices = build_l2_slices(
        op_wall, total_native, args.min_pct)

    # --- Level 3 ---
    l3_map = build_l3_map(l2_slices, op_gpu)

    # Build maps for drawing
    l2_draw_map = {"Native Execution": l2_slices}

    # --- Draw ---
    suffix = (f"\n{args.title_suffix}"
              if args.title_suffix else "")
    draw_sunburst_3ring(
        l1_slices, l2_draw_map, l3_map,
        total_wall, total_exec_run, n_tasks,
        args.output, title_extra=suffix)


if __name__ == "__main__":
    main()
