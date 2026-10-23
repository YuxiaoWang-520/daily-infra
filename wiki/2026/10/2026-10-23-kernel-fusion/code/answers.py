"""CPU pipeline semantics and a deliberately hypothetical fusion cost model."""

import math

import torch


def _count(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value: float, name: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


@torch.inference_mode()
def chunked_affine_relu(
    x: torch.Tensor, bias: torch.Tensor, scale: float, chunk_rows: int
) -> torch.Tensor:
    """CPU inference-only y=relu((x+bias)*scale), without modifying inputs.

    One explicit arithmetic buffer per row chunk; full output is still required.
    This runs multiple CPU operations and is NOT GPU kernel fusion or a speed test.
    """
    _count(chunk_rows, "chunk_rows", 1)
    scale = _number(scale, "scale", -math.inf)
    for value in (x, bias):
        if not isinstance(value, torch.Tensor):
            raise ValueError("inputs must be tensors")
        if value.layout != torch.strided or value.device.type != "cpu":
            raise ValueError("inputs must be CPU strided tensors")
        if value.dtype not in (torch.float32, torch.float64):
            raise ValueError("inputs must be float32 or float64")
    if x.ndim != 2 or bias.ndim != 1 or x.shape[1] != bias.shape[0]:
        raise ValueError("x must be (M,N) and bias must be (N,)")
    if x.dtype != bias.dtype:
        raise ValueError("input dtypes must match")
    out = torch.empty(x.shape, dtype=x.dtype)
    for row in range(0, x.shape[0], chunk_rows):
        buffer = x[row : row + chunk_rows] + bias
        buffer.mul_(scale)
        buffer.clamp_min_(0)
        out[row : row + chunk_rows] = buffer
        del buffer
    return out


def pipeline_model(
    m: int,
    n: int,
    itemsize: int,
    bandwidth_gbs: float = 1000.0,
    launch_us: float = 5.0,
    fused_slowdown: float = 1.0,
) -> dict[str, int | float]:
    """Toy time = launches*launch_cost + model_bytes/bandwidth * slowdown.

    Hypothetical three-stage pipeline versus one fused kernel. Bias is counted
    once (unique N elements), scalar scale traffic ignored. No cache/counter data.
    Slowdown only penalizes the fused memory term; it is an assumption, not fitted.
    """
    _count(m, "m")
    _count(n, "n")
    _count(itemsize, "itemsize", 1)
    bandwidth = _number(bandwidth_gbs, "bandwidth_gbs", 0.0)
    if not bandwidth:
        raise ValueError("bandwidth_gbs must be positive")
    launch = _number(launch_us, "launch_us", 0.0) * 1e-6
    slowdown = _number(fused_slowdown, "fused_slowdown", 1.0)
    elements = m * n
    bias = n if elements else 0
    staged_bytes = (6 * elements + bias) * itemsize
    fused_bytes = (2 * elements + bias) * itemsize
    staged_launches, fused_launches = (3, 1) if elements else (0, 0)
    return {
        "staged_bytes": staged_bytes,
        "fused_bytes": fused_bytes,
        "saved_bytes": staged_bytes - fused_bytes,
        "staged_launches": staged_launches,
        "fused_launches": fused_launches,
        "staged_seconds": staged_launches * launch + staged_bytes / (bandwidth * 1e9),
        "fused_seconds": fused_launches * launch
        + fused_bytes / (bandwidth * 1e9) * slowdown,
    }
