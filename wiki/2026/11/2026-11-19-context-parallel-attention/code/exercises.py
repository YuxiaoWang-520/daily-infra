"""Starter contracts. Implement here, then adapt test_answers imports to this file."""

from collections.abc import Iterable

from answers import KVBlock
from torch import Tensor


def streaming_attention(
    q: Tensor,
    blocks: Iterable[KVBlock],
    query_positions: Tensor,
    *,
    causal: bool = True,
) -> tuple[Tensor, Tensor]:
    """Exercise 1: return output and global row logsumexp, including empty masks."""
    raise NotImplementedError("merge maximum, denominator and weighted numerator")


def cp_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cp_size: int,
    *,
    layout: str = "contiguous",
    causal: bool = True,
) -> Tensor:
    """Exercise 2: partition global positions, visit K/V owners, restore row order."""
    raise NotImplementedError("implement contiguous/zigzag layouts and global masks")
