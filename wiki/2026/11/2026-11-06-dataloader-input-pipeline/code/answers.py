"""Ideal stage-capacity and ready-queue models, using milliseconds and bytes."""

import math
from fractions import Fraction

from models import PipelineBudget


def _count(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _duration(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")  # noqa: TRY004 -- exercise contract
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is outside the finite float range") from error
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return result


def pipeline_budget(
    storage_ms: float,
    cpu_ms: float,
    workers: int,
    transfer_ms: float,
    compute_ms: float,
    prefetch_factor: int,
    batch_bytes: int,
    allow_device_overlap: bool = False,
) -> PipelineBudget:
    """Return an optimistic steady-state interval lower bound.

    Shared storage service cannot scale with workers. A worker is occupied by
    both reading and CPU preparation. Copy and compute serialize unless the caller
    explicitly models device overlap. Queue payload assumes one batch-sized
    payload per task slot and excludes all other process/pinning/cache costs.
    """
    for name, value in (
        ("workers", workers),
        ("prefetch_factor", prefetch_factor),
        ("batch_bytes", batch_bytes),
    ):
        _count(value, name)
    if not isinstance(allow_device_overlap, bool):
        raise ValueError("allow_device_overlap must be a bool")  # noqa: TRY004
    storage = _duration(storage_ms, "storage_ms")
    cpu = _duration(cpu_ms, "cpu_ms")
    transfer = _duration(transfer_ms, "transfer_ms")
    compute = _duration(compute_ms, "compute_ms", positive=True)
    device_interval = (
        max(transfer, compute) if allow_device_overlap else transfer + compute
    )
    serial = storage + cpu + transfer + compute
    if not math.isfinite(serial) or not math.isfinite(device_interval):
        raise ValueError("combined durations exceed the finite float range")
    worker_interval = float((Fraction(storage) + Fraction(cpu)) / workers)
    stages = {"storage": storage, "worker_preparation": worker_interval}
    if allow_device_overlap:
        stages.update(transfer=transfer, compute=compute)
    else:
        stages["copy_and_compute"] = device_interval
    interval = max(stages.values())
    slots = workers * prefetch_factor
    return PipelineBudget(
        storage,
        worker_interval,
        device_interval,
        interval,
        serial,
        slots,
        slots * batch_bytes,
        tuple(stage for stage, duration in stages.items() if duration == interval),
    )


def exposed_pause(
    pause_ms: float, ready_batches: int, consumption_interval_ms: float
) -> float:
    """Model an isolated pause with R ready batches and instant recovery.

    A ready batch takes one consumption interval before the next is needed.
    Fractions avoid overflow when a large integer ready count covers the pause.
    Refill delay, partial batches and repeated supply shortfalls are omitted.
    """
    _count(ready_batches, "ready_batches", minimum=0)
    pause = _duration(pause_ms, "pause_ms")
    consumption = _duration(
        consumption_interval_ms, "consumption_interval_ms", positive=True
    )
    remaining = Fraction(pause) - ready_batches * Fraction(consumption)
    return float(max(Fraction(0), remaining))
