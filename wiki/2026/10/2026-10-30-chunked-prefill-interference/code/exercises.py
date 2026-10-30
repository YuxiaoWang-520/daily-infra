"""Implement the two functions according to code/README.md."""

import torch
from models import Schedule


def chunked_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    raise NotImplementedError(
        "Exercise 1: retain prefix KV and global causal positions"
    )


def schedule_prefill(
    prompt_tokens: int, decode_requests: int, decode_steps: int, token_budget: int
) -> Schedule:
    raise NotImplementedError(
        "Exercise 2: schedule decodes before allocating the remaining budget"
    )
