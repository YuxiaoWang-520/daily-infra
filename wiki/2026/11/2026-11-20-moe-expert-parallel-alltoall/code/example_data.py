"""Public synthetic token, router and expert inputs."""

import torch
from torch import Tensor


def example_logits(*, skew: bool = False) -> Tensor:
    probabilities = torch.tensor(
        [
            [0.5, 0.3, 0.15, 0.05],
            [0.1, 0.2, 0.6, 0.1],
            [0.05, 0.15, 0.3, 0.5],
            [0.4, 0.1, 0.35, 0.15],
        ],
        dtype=torch.float64,
    )
    if skew:
        probabilities = torch.tensor(
            [[0.55, 0.3, 0.1, 0.05]], dtype=torch.float64
        ).repeat(4, 1)
    return probabilities.log()


def synthetic_inputs(*, skew: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(2020)
    x = torch.randn(4, 2, generator=generator, dtype=torch.float64)
    w1 = torch.randn(4, 2, 3, generator=generator, dtype=torch.float64) * 0.3
    w2 = torch.randn(4, 3, 2, generator=generator, dtype=torch.float64) * 0.3
    return x, example_logits(skew=skew), w1, w2
