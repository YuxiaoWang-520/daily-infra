"""Exercise interfaces. Full contracts and acceptance criteria are in README.md."""

import torch


def tp_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    tp_size: int,
) -> dict[str, object]:
    """Return preactivations, hidden_shards, partial_outputs and output."""
    raise NotImplementedError


def collective_budget(
    tokens: int,
    hidden: int,
    tp_size: int,
    element_bytes: int,
    *,
    sequence_parallel: bool = False,
    regather_input_for_wgrad: bool = False,
) -> dict[str, int]:
    """Return payload, collective calls, ring rounds and per-rank send/receive bytes."""
    raise NotImplementedError
