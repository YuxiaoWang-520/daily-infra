"""Interval accounting in microseconds; no inference of hardware bottlenecks."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

Interval = tuple[float, float]


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timestamps and durations must be finite numbers")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("number exceeds binary64 range") from error
    if not math.isfinite(result):
        raise ValueError("timestamps and durations must be finite numbers")
    if isinstance(value, int) and int(result) != value:
        raise ValueError("integer timestamp cannot be represented exactly as binary64")
    return result


def _clipped(intervals: Iterable[Interval], window: Interval) -> list[Interval]:
    start, end = map(_number, window)
    if end <= start or not math.isfinite(end - start):
        raise ValueError("window must have finite positive duration")
    result = []
    for interval in intervals:
        left, right = map(_number, interval)
        if right < left:
            raise ValueError("interval end precedes its start")
        left, right = max(left, start), min(right, end)
        if right > left:
            result.append((left, right))
    return result


def merge_intervals(
    intervals: Iterable[Interval], window: Interval
) -> tuple[Interval, ...]:
    """Clip to a positive window, then merge overlaps and touching intervals."""
    merged: list[Interval] = []
    for start, end in sorted(_clipped(intervals, window)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def _length(intervals: Iterable[Interval]) -> float:
    return math.fsum(end - start for start, end in intervals)


def overlap_summary(
    compute: Iterable[Interval],
    communication: Iterable[Interval],
    window: Interval,
) -> dict[str, float]:
    """Count selected intervals on one device and one aligned time axis.

    'No selected activity' does not prove GPU idleness. Communication-only time
    is a set difference, not the speedup from hypothetically removing communication.
    """
    compute_raw = _clipped(compute, window)
    comm_raw = _clipped(communication, window)
    comp = merge_intervals(compute_raw, window)
    comm = merge_intervals(comm_raw, window)
    i = j = 0
    intersections = []
    while i < len(comp) and j < len(comm):
        start = max(comp[i][0], comm[j][0])
        end = min(comp[i][1], comm[j][1])
        if end > start:
            intersections.append((start, end))
        if comp[i][1] <= comm[j][1]:
            i += 1
        else:
            j += 1
    compute_us = _length(comp)
    comm_us = _length(comm)
    overlap_us = _length(intersections)
    any_us = _length(merge_intervals((*comp, *comm), window))
    start, end = map(_number, window)
    window_us = end - start
    return {
        "window_us": window_us,
        "compute_us": compute_us,
        "communication_us": comm_us,
        "overlap_us": overlap_us,
        "communication_only_us": comm_us - overlap_us,
        "any_activity_us": any_us,
        "no_selected_activity_us": window_us - any_us,
        "summed_activity_us": _length((*compute_raw, *comm_raw)),
    }


@dataclass(frozen=True)
class TraceEvent:
    name: str
    category: str
    start_us: float
    duration_us: float
    pid: str
    tid: str

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us


def complete_events(trace: Mapping[str, Any]) -> tuple[TraceEvent, ...]:
    """Read only Chrome complete ('X') events. Does not pair B/E or flow records."""
    if not isinstance(trace, Mapping) or not isinstance(trace.get("traceEvents"), list):
        raise TypeError("expected a traceEvents list")
    result = []
    for raw in trace["traceEvents"]:
        if not isinstance(raw, Mapping):
            raise TypeError("trace event must be an object")
        if raw.get("ph") != "X":
            continue
        for key in ("name", "cat"):
            if not isinstance(raw.get(key), str):
                raise TypeError(f"complete event requires string {key}")
        for key in ("pid", "tid"):
            if isinstance(raw.get(key), bool) or not isinstance(
                raw.get(key), (str, int)
            ):
                raise TypeError(f"complete event requires string or integer {key}")
        start = _number(raw.get("ts"))
        duration = _number(raw.get("dur"))
        end = start + duration
        if duration < 0 or not math.isfinite(end):
            raise ValueError("invalid complete event duration")
        result.append(
            TraceEvent(
                raw["name"],
                raw["cat"],
                start,
                duration,
                str(raw["pid"]),
                str(raw["tid"]),
            )
        )
    return tuple(result)


def project_cpu_trace(
    trace: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep real CPU complete events, stripping args/metadata and rebasing time.

    Names are retained: this projection is intended for the public toy workload,
    not a general privacy sanitizer for arbitrary production traces.
    """
    events = [
        e for e in complete_events(trace) if e.category in {"cpu_op", "user_annotation"}
    ]
    origin = min((e.start_us for e in events), default=0.0)
    processes: dict[str, int] = {}
    threads: dict[tuple[str, str], int] = {}
    projected = []
    for event in events:
        relative_start = event.start_us - origin
        if not (
            math.isfinite(relative_start)
            and math.isfinite(relative_start + event.duration_us)
        ):
            raise ValueError("timestamp rebasing exceeds binary64 range")
        pid = processes.setdefault(event.pid, len(processes) + 1)
        tid = threads.setdefault((event.pid, event.tid), len(threads) + 1)
        projected.append(
            {
                "ph": "X",
                "name": event.name,
                "cat": event.category,
                "ts": relative_start,
                "dur": event.duration_us,
                "pid": pid,
                "tid": tid,
            }
        )
    return (
        {
            "displayTimeUnit": "ms",
            "provenance": "real CPU trace; projected, not full raw trace",
            "traceEvents": projected,
        },
        {
            "raw_event_records": len(trace["traceEvents"]),
            "retained_complete_events": len(projected),
            "timestamp_origin_us": origin,
            "durations_preserved": True,
            "dropped": [
                "non-X events",
                "non-CPU categories",
                "args",
                "top-level metadata",
            ],
            "remapped": ["pid", "tid"],
            "names_retained_require_manual_privacy_review": True,
        },
    )


def synthetic_trace() -> dict[str, Any]:
    """Known roles/lanes are authored, not guessed from a real kernel name."""
    rows = [
        ("step", "user_annotation", 0, 15000, 1, 1),
        ("cudaLaunchKernel A", "cuda_runtime", 1000, 200, 1, 1),
        ("cudaLaunchKernel B", "cuda_runtime", 1300, 200, 1, 1),
        ("communication enqueue", "cuda_runtime", 1600, 100, 1, 1),
        ("compute_A", "kernel", 2000, 6000, 2, 101),
        ("compute_B", "kernel", 8000, 3000, 2, 101),
        ("communication", "kernel", 5000, 8000, 2, 102),
    ]
    return {
        "displayTimeUnit": "ms",
        "provenance": "synthetic; not a hardware measurement",
        "traceEvents": [
            {"name": n, "cat": c, "ts": t, "dur": d, "pid": p, "tid": s, "ph": "X"}
            for n, c, t, d, p, s in rows
        ],
    }
