"""Public synthetic inputs and deterministic CPU model for this article."""

import torch
from torch import nn


def make_model() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(3, 5, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(5, 2, dtype=torch.float64),
    )
    with torch.no_grad():
        model[0].weight.copy_(
            torch.linspace(-0.4, 0.5, 15, dtype=torch.float64).reshape(5, 3)
        )
        model[0].bias.copy_(torch.linspace(-0.1, 0.1, 5, dtype=torch.float64))
        model[2].weight.copy_(
            torch.linspace(-0.3, 0.6, 10, dtype=torch.float64).reshape(2, 5)
        )
        model[2].bias.copy_(torch.tensor([0.2, -0.2], dtype=torch.float64))
    return model


def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 10 - 1
    targets = torch.arange(16, dtype=torch.float64).reshape(8, 2) / 7 - 0.5
    return inputs, targets
