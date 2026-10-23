"""Logical stream timelines and a same-current-stream CUDA timing recipe."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

import torch


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and >= 0")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return float(value)


def _stream(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("stream name must be a nonempty string")


@dataclass(frozen=True)
class Task:
    start_ms: float
    end_ms: float


class QueueModel:
    """Ideal independent streams, unlimited resources; all times are synthetic.

    Only submit advances the host by submit_ms; event/wait have zero host cost.
    There is no default-stream special behavior, allocator or device contention.
    """

    def __init__(self, submit_ms: float = 0.1) -> None:
        self.submit_ms = _nonnegative(submit_ms, "submit_ms")
        self.host_ms = 0.0
        self._tails: dict[str, float] = {}

    def submit(self, stream: str, duration_ms: float) -> Task:
        _stream(stream)
        duration = _nonnegative(duration_ms, "duration_ms")
        self.host_ms += self.submit_ms
        start = max(self.host_ms, self._tails.get(stream, 0.0))
        task = Task(start, start + duration)
        self._tails[stream] = task.end_ms
        return task

    def event(self, stream: str) -> float:
        _stream(stream)
        return max(self.host_ms, self._tails.get(stream, 0.0))

    def wait_event(self, stream: str, event_ms: float) -> None:
        _stream(stream)
        event = _nonnegative(event_ms, "event_ms")
        self._tails[stream] = max(self._tails.get(stream, 0.0), event)

    def synchronize(self) -> None:
        self.host_ms = max([self.host_ms, *self._tails.values()])


def benchmark_cuda(
    fn: Callable[[], object], warmup: int = 5, iterations: int = 20, repeats: int = 5
) -> dict[str, list[float] | float]:
    """Batch-average current-stream event intervals, in seconds per call.

    Caller prepares data/current device and correctness checks. fn must enqueue
    work only on the unchanged current stream; no side-stream or end-to-end claim.
    Warmup is excluded. No CPU scalar reads, logging or transfers inside fn.
    GPU path is a supplied recipe, NOT hardware-validated by this CPU-only repo.
    """
    for name, value, minimum in (
        ("warmup", warmup, 0),
        ("iterations", iterations, 1),
        ("repeats", repeats, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not callable(fn):
        raise ValueError("fn must be callable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no timing was performed")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0 / iterations)
    return {"seconds_per_call": samples, "median_seconds": median(samples)}
