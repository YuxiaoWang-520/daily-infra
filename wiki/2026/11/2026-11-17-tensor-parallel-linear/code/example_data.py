"""Synthetic, hand-checkable two-layer MLP data."""

import torch


def make_inputs(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, ...]:
    """Return X, logical W1, b1, logical W2, b2 and the target."""
    values = (
        [[1.0, -2.0], [-1.0, 1.0]],
        [[1.0, -1.0, 2.0, 0.0], [0.0, 2.0, -1.0, 1.0]],
        [2.0, 0.0, 1.0, 3.0],
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
        [1.0, -1.0],
        [[1.0, -1.0], [2.0, 1.0]],
    )
    return tuple(torch.tensor(value, dtype=dtype) for value in values)
