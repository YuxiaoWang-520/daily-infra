"""Small CPU experiments for storage rounding, accumulation and finite checks."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
from fractions import Fraction

import torch

FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def roundtrip(values: Sequence[float], dtype: torch.dtype) -> torch.Tensor:
    """Quantize float64 input to dtype, then expose the stored values in float64."""
    if dtype not in FLOAT_DTYPES:
        raise ValueError("dtype must be a supported real floating format")
    source = torch.tensor(values, dtype=torch.float64)
    if source.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return source.to(dtype).to(torch.float64)


def sequential_sum(
    values: torch.Tensor, accumulator_dtype: torch.dtype
) -> torch.Tensor:
    """Round every sequential addition to accumulator_dtype; CPU teaching only."""
    if accumulator_dtype not in FLOAT_DTYPES:
        raise ValueError("accumulator must use a supported floating dtype")
    if (
        values.device.type != "cpu"
        or values.layout != torch.strided
        or values.ndim != 1
        or not values.is_floating_point()
    ):
        raise ValueError("values must be a dense one-dimensional real CPU tensor")
    accumulator = torch.zeros((), dtype=accumulator_dtype)
    for value in values:
        accumulator = accumulator + value.to(accumulator_dtype)
    return accumulator


@dataclass(frozen=True)
class ErrorReport:
    total: int
    failed: int
    nonfinite: int
    max_absolute: float
    relative_l2: float

    @property
    def passed(self) -> bool:
        return self.failed == 0


@torch.inference_mode()
def compare_finite(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> ErrorReport:
    """Audit small CPU tensors against exact tolerances on their stored values.

    Finite binary values and tolerances are converted to exact integer ratios.
    This intentionally slow reference avoids overflow/rounding in the checker.
    Relative L2 uses exact squared sums and an 80-digit decimal square root;
    final descriptive metrics are floats and can overflow to infinity.
    """
    if any(not math.isfinite(tol) or tol < 0 for tol in (atol, rtol)):
        raise ValueError("tolerances must be finite and non-negative")
    if actual.shape != reference.shape:
        raise ValueError("actual and reference must have the same shape")
    for value in (actual, reference):
        if (
            value.device.type != "cpu"
            or value.layout != torch.strided
            or not value.is_floating_point()
        ):
            raise ValueError("comparison requires dense real floating CPU tensors")
    a = actual.to(torch.float64).reshape(-1)
    r = reference.to(torch.float64).reshape(-1)
    total = a.numel()
    if total == 0:
        return ErrorReport(0, 0, 0, 0.0, 0.0)
    finite = torch.isfinite(a) & torch.isfinite(r)
    count_nonfinite = int((~finite).sum().item())
    absolute_tolerance = Fraction(float(atol))
    relative_tolerance = Fraction(float(rtol))
    errors = []
    squared_reference = Fraction(0)
    failed = count_nonfinite
    for value, target in zip(a[finite].tolist(), r[finite].tolist()):
        value, target = Fraction(value), Fraction(target)
        error = abs(value - target)
        errors.append(error)
        squared_reference += target * target
        failed += error > absolute_tolerance + relative_tolerance * abs(target)
    if count_nonfinite:
        return ErrorReport(total, failed, count_nonfinite, math.inf, math.inf)
    try:
        max_absolute = float(max(errors))
    except OverflowError:
        max_absolute = math.inf
    squared_error = sum((error * error for error in errors), Fraction(0))
    if not squared_reference:
        relative_l2 = math.inf if squared_error else 0.0
    else:
        ratio = squared_error / squared_reference
        with localcontext(Context(prec=80)):
            relative_l2 = float(
                (Decimal(ratio.numerator) / Decimal(ratio.denominator)).sqrt()
            )
    return ErrorReport(total, failed, 0, max_absolute, relative_l2)
