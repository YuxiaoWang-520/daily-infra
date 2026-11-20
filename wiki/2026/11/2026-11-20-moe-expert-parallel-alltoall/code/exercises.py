"""Starter interfaces; Routes and other helpers can be reused from answers."""

from answers import Routes
from torch import Tensor


def route_tokens(
    logits: Tensor, topk: int, *, pre_softmax: bool = False, capacity: int | None = None
) -> Routes:
    """Exercise 1: choose weighted assignments and drop excess expert rows."""
    raise NotImplementedError(
        "define normalization scope and preserve token/expert IDs"
    )


def combine_tokens(expert_output: Tensor, routes: Routes) -> Tensor:
    """Exercise 2: return a weighted sum per original token, including empty routes."""
    raise NotImplementedError("scatter-add, rather than overwrite or average")
