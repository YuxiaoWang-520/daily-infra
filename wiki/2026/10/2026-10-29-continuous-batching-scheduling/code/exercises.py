"""Implement the two functions; the contracts and examples are in exercises.md."""

from collections.abc import Sequence
from typing import Literal

from models import Admission, Prefill, Request, Simulation


def simulate(
    requests: Sequence[Request],
    capacity: int,
    policy: Literal["static", "continuous"],
) -> Simulation:
    raise NotImplementedError("Exercise 1: refill only at the allowed boundaries")


def admit_prefills(
    waiting: Sequence[Prefill],
    *,
    running_decodes: int,
    max_num_seqs: int,
    token_budget: int,
    free_blocks: int,
    block_size: int,
) -> Admission:
    raise NotImplementedError("Exercise 2: check all budgets before charging them")
