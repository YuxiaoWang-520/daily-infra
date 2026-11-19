"""Public synthetic data shared by the answer runner and Gloo check."""

import torch
from torch import Tensor


def synthetic_inputs() -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(1919)
    return tuple(
        torch.randn(8, 2, generator=generator, dtype=torch.float64) for _ in range(3)
    )


def scalar_example() -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.ones(4, 1, dtype=torch.float64),
        torch.arange(1, 5, dtype=torch.float64).log()[:, None],
        torch.arange(10, 50, 10, dtype=torch.float64)[:, None],
    )
