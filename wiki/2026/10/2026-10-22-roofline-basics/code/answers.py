"""A small analytical Roofline model, not a GPU benchmark or traffic profiler."""

import math
from dataclasses import dataclass


def _count(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


@dataclass(frozen=True)
class Work:
    """Conventional FLOP count and explicitly assumed memory-layer traffic."""

    flops: int
    bytes_moved: int

    def __post_init__(self) -> None:
        _count(self.flops, "flops")
        _count(self.bytes_moved, "bytes_moved")
        if self.flops and not self.bytes_moved:
            raise ValueError("this model requires positive traffic for arithmetic work")


@dataclass(frozen=True)
class Estimate:
    intensity: float
    ridge_intensity: float
    compute_seconds: float
    memory_seconds: float
    lower_seconds: float
    attainable_tflops: float
    limiter: str


def vector_add_work(n: int, itemsize: int) -> Work:
    """Distinct dense x/y inputs, one new output, one read/write per element."""
    _count(n, "n")
    _count(itemsize, "itemsize", 1)
    return Work(n, 3 * n * itemsize)


def gemm_work(m: int, n: int, k: int, itemsize: int) -> Work:
    """C=A@B, beta=0: optimistic read-A/B-once and write-C-once traffic."""
    for name, value in (("m", m), ("n", n), ("k", k), ("itemsize", itemsize)):
        _count(value, name, 1)
    return Work(2 * m * n * k, (m * k + k * n + m * n) * itemsize)


def estimate(work: Work, peak_tflops: float, bandwidth_gbs: float) -> Estimate:
    """Decimal TFLOP/s and GB/s; lower bound assumes ideal resource overlap."""
    peak = _positive(peak_tflops, "peak_tflops") * 1e12
    bandwidth = _positive(bandwidth_gbs, "bandwidth_gbs") * 1e9
    intensity = work.flops / work.bytes_moved if work.bytes_moved else 0.0
    compute = work.flops / peak
    memory = work.bytes_moved / bandwidth
    if not work.bytes_moved:
        limiter = "empty"
    elif compute == memory:
        limiter = "balanced"
    else:
        limiter = "compute" if compute > memory else "memory"
    return Estimate(
        intensity,
        peak / bandwidth,
        compute,
        memory,
        max(compute, memory),
        min(peak, bandwidth * intensity) / 1e12,
        limiter,
    )


def effective_rates(work: Work, elapsed_s: float) -> dict[str, float]:
    """Effective model-byte rate is not counter-measured DRAM bandwidth."""
    elapsed = _positive(elapsed_s, "elapsed_s")
    return {
        "tflops": work.flops / elapsed / 1e12,
        "effective_gbs": work.bytes_moved / elapsed / 1e9,
    }
