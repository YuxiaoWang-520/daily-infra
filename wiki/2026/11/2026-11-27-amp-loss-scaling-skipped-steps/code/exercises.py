"""Exercise starters. Implement these contracts before reading the answers."""

from collections.abc import Sequence
from typing import Any

import torch

from answers import Quantization


def quantized_gradient(
    gradient: float, scale: float, *, dtype: torch.dtype = torch.float16
) -> Quantization:
    """Compare storage of g and S*g; recover in float64, with explicit CPU tensors."""
    raise NotImplementedError("Exercise 1: validate inputs and compare rounding")


def unscale_and_clip(
    scaled_gradients: Sequence[float], scale: float, max_norm: float
) -> list[float]:
    """Divide by scale, then clip the L2 norm to max_norm; omit native epsilon."""
    raise NotImplementedError("Exercise 1: recover the original gradient units")


def unscale_accumulated(
    gradients: Sequence[float], scales: Sequence[float], final_scale: float
) -> float:
    """Evaluate sum(S_i*g_i)/S_final and explain the constant-scale condition."""
    raise NotImplementedError("Exercise 1: compare constant and mixed scales")


def training_trace(
    faults: Sequence[bool], *, restore_after: int | None = None, init_scale: float = 8
) -> dict[str, Any]:
    """Exercise 2: implement the full CPU loop and return the documented trace.

    Use float32 w=1, loss=w*w/2, ordinary SGD(lr=.1, momentum=.9),
    explicit GradScaler('cpu', growth_interval=2) and a constant LambdaLR.
    Inject Inf before unscale when faults[i] is True. Count actual optimizer
    calls, not step return values or parameter changes. Advance the scheduler
    only for accepted calls; always advance the data-batch counter.
    Restore public state_dict objects and application counters after the given
    completed update boundary. See README.md for output keys and acceptance.
    """
    raise NotImplementedError("Exercise 2: scaler protocol and training progress")
