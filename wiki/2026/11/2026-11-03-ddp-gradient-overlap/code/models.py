"""Immutable results for an explicitly simplified communication timeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BucketEvent:
    bucket_index: int
    gradient_indices: tuple[int, ...]
    size_bytes: int
    ready_ms: float
    start_ms: float
    finish_ms: float


@dataclass(frozen=True)
class Schedule:
    events: tuple[BucketEvent, ...]
    backward_ms: float
    finish_ms: float
    tail_ms: float
    serial_ms: float
