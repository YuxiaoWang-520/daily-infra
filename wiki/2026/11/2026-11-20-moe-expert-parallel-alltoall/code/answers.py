"""CPU MoE model: deterministic routing, sparse dispatch and weighted combine."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Routes:
    token_indices: Tensor
    expert_indices: Tensor
    weights: Tensor
    num_tokens: int
    num_experts: int
    topk: int

    @property
    def expert_counts(self) -> Tensor:
        return torch.bincount(self.expert_indices, minlength=self.num_experts)


@dataclass(frozen=True)
class DispatchPlan:
    order: Tensor
    send_splits: tuple[int, ...]


def _positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _floating(value: Tensor, name: str, ndim: int) -> None:
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def route_tokens(
    logits: Tensor, topk: int, *, pre_softmax: bool = False, capacity: int | None = None
) -> Routes:
    """Select gates; capacity drops lowest weights without renormalization.

    Equal scores prefer lower expert ID, then earlier token. These deterministic
    tie rules are teaching conventions, not GPU topk guarantees. Capacity applies
    to this call's routing batch. The discrete selections are not differentiable.
    """
    _floating(logits, "logits", 2)
    _positive(topk, "topk")
    tokens, experts = logits.shape
    if tokens < 1 or experts < topk:
        raise ValueError("require tokens > 0 and 1 <= topk <= num_experts")
    if capacity is not None and (
        isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0
    ):
        raise ValueError("capacity must be None or a nonnegative integer")
    indices = logits.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    selected = logits.gather(1, indices)
    weights = (
        logits.softmax(dim=-1).gather(1, indices)
        if pre_softmax
        else selected.softmax(dim=-1)
    )
    token_indices = torch.arange(tokens, device=logits.device).repeat_interleave(topk)
    expert_indices = indices.reshape(-1)
    weights = weights.reshape(-1)
    if capacity is not None:
        keep = torch.zeros_like(weights, dtype=torch.bool)
        for expert in range(experts):
            assignments = (expert_indices == expert).nonzero().flatten()
            order = weights[assignments].argsort(descending=True, stable=True)
            keep[assignments[order[:capacity]]] = True
        token_indices, expert_indices, weights = (
            token_indices[keep],
            expert_indices[keep],
            weights[keep],
        )
    return Routes(token_indices, expert_indices, weights, tokens, experts, topk)


def capacity_for(tokens: int, experts: int, topk: int, factor: float) -> int:
    """One routing batch: ceil(factor * tokens * k / E)."""
    for name, value in (("tokens", tokens), ("experts", experts), ("topk", topk)):
        _positive(value, name)
    if (
        topk > experts
        or isinstance(factor, bool)
        or not math.isfinite(factor)
        or factor <= 0
    ):
        raise ValueError(
            "require topk <= experts and a positive finite capacity factor"
        )
    return math.ceil(factor * tokens * topk / experts)


def dispatch_plan(routes: Routes, ep_size: int) -> DispatchPlan:
    """Pack by owner; each rank owns E/p consecutive expert IDs."""
    _positive(ep_size, "ep_size")
    if routes.num_experts % ep_size:
        raise ValueError("num_experts must be divisible by ep_size")
    owners = routes.expert_indices // (routes.num_experts // ep_size)
    return DispatchPlan(
        owners.argsort(stable=True),
        tuple(torch.bincount(owners, minlength=ep_size).tolist()),
    )


def load_balance_loss(logits: Tensor, topk: int, *, coefficient: float = 1.0) -> Tensor:
    """Local-batch Switch-style extension, using counts before capacity dropping."""
    if (
        isinstance(coefficient, bool)
        or not math.isfinite(coefficient)
        or coefficient < 0
    ):
        raise ValueError("coefficient must be finite and nonnegative")
    routes = route_tokens(logits, topk)
    frequencies = routes.expert_counts.to(logits.dtype) / (routes.num_tokens * topk)
    average_probabilities = logits.softmax(dim=-1).mean(dim=0)
    return (
        coefficient * routes.num_experts * (frequencies * average_probabilities).sum()
    )


def evaluate_experts(
    x: Tensor, expert_ids: Tensor, w1: Tensor, w2: Tensor, *, expert_offset: int = 0
) -> Tensor:
    """Compute tanh(X W1[e]) W2[e], including differentiable empty owners.

    This model has no biases, dropout, shared experts, TP or fused GEMM.
    """
    for name, tensor, ndim in (("x", x, 2), ("w1", w1, 3), ("w2", w2, 3)):
        _floating(tensor, name, ndim)
    if (
        w1.shape[0] < 1
        or w1.shape[1] < 1
        or w1.shape[2] < 1
        or w2.shape[2] < 1
        or w1.shape[0] != w2.shape[0]
        or x.shape[1] != w1.shape[1]
        or w1.shape[2] != w2.shape[1]
    ):
        raise ValueError("expert weights must be [E_local,D,H] and [E_local,H,Dout]")
    if any(t.dtype != x.dtype or t.device != x.device for t in (w1, w2)):
        raise TypeError("tokens and weights must share dtype and device")
    if expert_ids.ndim != 1 or expert_ids.numel() != x.shape[0]:
        raise ValueError("provide one expert ID per received row")
    if expert_ids.dtype != torch.int64 or expert_ids.device != x.device:
        raise TypeError("expert IDs must be int64 on the token device")
    if (
        isinstance(expert_offset, bool)
        or not isinstance(expert_offset, int)
        or expert_offset < 0
    ):
        raise ValueError("expert_offset must be a nonnegative integer")
    if bool(
        (
            (expert_ids < expert_offset) | (expert_ids >= expert_offset + w1.shape[0])
        ).any()
    ):
        raise ValueError("expert ID is not owned by this rank")
    zero = (x * 0).sum() + (w1 * 0).sum() + (w2 * 0).sum()
    out = x.new_zeros((x.shape[0], w2.shape[2])) + zero
    for local_id in range(w1.shape[0]):
        rows = (expert_ids == expert_offset + local_id).nonzero().flatten()
        if rows.numel():
            values = torch.tanh(x.index_select(0, rows) @ w1[local_id]) @ w2[local_id]
            out = out.index_copy(0, rows, values)
    return out


def combine_tokens(expert_output: Tensor, routes: Routes) -> Tensor:
    """Weight and add retained assignment outputs at their original token IDs."""
    _floating(expert_output, "expert_output", 2)
    if expert_output.shape[0] != routes.weights.numel() or expert_output.shape[1] < 1:
        raise ValueError("one output row is required per retained assignment")
    if (
        expert_output.dtype != routes.weights.dtype
        or expert_output.device != routes.weights.device
    ):
        raise TypeError("outputs and weights must share dtype and device")
    out = expert_output.new_zeros((routes.num_tokens, expert_output.shape[1]))
    return out.index_add(
        0, routes.token_indices, expert_output * routes.weights[:, None]
    )


def moe_forward(
    x: Tensor,
    logits: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk: int,
    *,
    pre_softmax: bool = False,
    capacity: int | None = None,
) -> tuple[Tensor, Routes]:
    """Packed reference; auxiliary losses are not silently added."""
    routes = route_tokens(logits, topk, pre_softmax=pre_softmax, capacity=capacity)
    _floating(x, "x", 2)
    if (
        x.shape[0] != routes.num_tokens
        or w1.ndim != 3
        or w1.shape[0] != routes.num_experts
    ):
        raise ValueError("token and expert counts must match the router")
    payload = x.index_select(0, routes.token_indices)
    values = evaluate_experts(payload, routes.expert_indices, w1, w2)
    return combine_tokens(values, routes), routes
