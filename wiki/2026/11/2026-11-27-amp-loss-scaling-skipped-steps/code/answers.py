"""Finite-gradient mathematical examples, explicitly evaluated on CPU."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch


@dataclass(frozen=True)
class Quantization:
    gradient: float
    scale: float
    dtype: str
    direct_storage: float
    scaled_storage: float
    recovered: float


def finite_number(value: Real, name: str, *, positive: bool = False) -> float:
    """Reject booleans, non-real values, nonfinite values and invalid scales."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, excluding bool")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is outside the finite float range") from error
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(
            f"{name} must be finite" + (" and positive" if positive else "")
        )
    return number


def quantized_gradient(
    gradient: float, scale: float, *, dtype: torch.dtype = torch.float16
) -> Quantization:
    """Compare casting g with casting S*g before recovering g in float64.

    This is a storage/rounding model, not a GPU backward-kernel simulation.
    Overflow of the scaled value is an intentional possible result.
    """
    gradient = finite_number(gradient, "gradient")
    scale = finite_number(scale, "scale", positive=True)
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("dtype must be float16 or bfloat16")
    original = torch.tensor(gradient, dtype=torch.float64, device="cpu")
    direct = original.to(dtype).item()
    scaled = (original * scale).to(dtype).to(torch.float64)
    return Quantization(
        gradient, scale, str(dtype), direct, scaled.item(), (scaled / scale).item()
    )


def unscale_and_clip(
    scaled_gradients: Sequence[float], scale: float, max_norm: float
) -> list[float]:
    """Return ideal L2 clipping after unscaling; no native epsilon term."""
    scale = finite_number(scale, "scale", positive=True)
    limit = finite_number(max_norm, "max_norm", positive=True)
    values = [finite_number(g, "gradient") / scale for g in scaled_gradients]
    if not values:
        raise ValueError("gradients must be nonempty")
    norm = math.hypot(*values)
    if not math.isfinite(norm):
        raise ValueError("unscaled norm must be finite")
    coefficient = min(1.0, limit / norm) if norm else 1.0
    return [g * coefficient for g in values]


def unscale_accumulated(
    gradients: Sequence[float], scales: Sequence[float], final_scale: float
) -> float:
    """Compute sum(S_i*g_i)/S_final, including the intentionally wrong mixed case."""
    gradients = list(gradients)
    scales = list(scales)
    final_scale = finite_number(final_scale, "final_scale", positive=True)
    if not gradients or len(gradients) != len(scales):
        raise ValueError("gradients and scales must have equal, nonzero lengths")
    terms = [
        finite_number(g, "gradient") * finite_number(s, "scale", positive=True)
        for g, s in zip(gradients, scales, strict=True)
    ]
    if not all(math.isfinite(term) for term in terms):
        raise ValueError("scaled terms must be finite in this analytic example")
    try:
        result = math.fsum(terms) / final_scale
    except OverflowError as error:
        raise ValueError(
            "accumulated gradient is outside the finite float range"
        ) from error
    if not math.isfinite(result):
        raise ValueError("accumulated gradient must be finite")
    return result
