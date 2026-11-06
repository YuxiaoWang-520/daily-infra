"""Result schemas for the capacity exercise and the CPU DataLoader fixture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineBudget:
    storage_interval_ms: float
    worker_interval_ms: float
    device_interval_ms: float
    ideal_interval_ms: float
    serial_ms: float
    task_slots: int
    task_payload_bytes: int
    limiting_stages: tuple[str, ...]


@dataclass(frozen=True)
class BatchObservation:
    index: int
    samples: int
    next_wait_ms: float
    consumer_ms: float
    source_wall_ms: float
    decode_ms: float
    transform_ms: float
    worker_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrialReport:
    workers: int
    sample_ids: tuple[int, ...]
    checksums: tuple[float, ...]
    fixture_bytes: int
    iterator_creation_ms: float
    total_iteration_ms: float
    warmup_batches: int
    steady_next_wait_median_ms: float
    batches: tuple[BatchObservation, ...]
