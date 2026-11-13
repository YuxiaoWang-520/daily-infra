"""Tensor-based replacements for changing Python scalars and data branches."""

import torch


def tensor_affine(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Compute relu(x*scale+1), using a zero-dimensional scale tensor."""
    return (x * scale + 1).relu()


def tensor_branch(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Use the sign of the whole input sum to choose y+1 or y-1."""
    y = x * scale
    return torch.where(x.sum() > 0, y + 1, y - 1)
