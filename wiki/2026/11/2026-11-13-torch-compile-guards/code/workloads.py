"""Small fixed workloads used by tests and the experiment runner."""

import torch


def affine(x: torch.Tensor, scale: float) -> torch.Tensor:
    return (x * scale + 1).relu()


def split_affine(x: torch.Tensor) -> torch.Tensor:
    y = x * 2
    # Explicit internal debugging API, pinned to the recorded PyTorch version.
    torch._dynamo.graph_break()
    return y.relu() + 1


def python_branch(x: torch.Tensor) -> torch.Tensor:
    y = x * 2
    if x.sum() > 0:
        return y + 1
    return y - 1
