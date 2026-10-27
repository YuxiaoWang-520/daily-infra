"""Dense KV byte accounting and a small CPU GQA reference, not a serving kernel."""

import math
from collections.abc import Sequence

import torch


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive Python int")


def kv_cache_bytes(
    lengths: Sequence[int],
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    element_bytes: int,
) -> int:
    """Count dense K and V content without page padding, sharing, or metadata.

    Lengths are already-cached token counts, not output-token budgets. All layers
    share this layout; K and V have the same head dimension and element dtype.
    An empty batch and zero-length sequences contribute zero bytes.
    """
    for name, value in [
        ("layers", layers),
        ("kv_heads", kv_heads),
        ("head_dim", head_dim),
        ("element_bytes", element_bytes),
    ]:
        _positive_int(value, name)
    total_tokens = 0
    for length in lengths:
        if type(length) is not int or length < 0:
            raise ValueError("each cached length must be a nonnegative Python int")
        total_tokens += length
    return 2 * layers * total_tokens * kv_heads * head_dim * element_bytes


def kv_heads_per_rank(kv_heads: int, tp_size: int) -> int:
    """Model the checked vLLM Llama KV-head sharding/replication rule only.

    Query-head divisibility by tp_size must be checked separately. This helper
    does not model pipeline/context parallelism, MLA, or heterogeneous layers.
    """
    _positive_int(kv_heads, "kv_heads")
    _positive_int(tp_size, "tp_size")
    if kv_heads >= tp_size:
        if kv_heads % tp_size:
            raise ValueError("kv_heads must be divisible by tp_size when partitioned")
        return kv_heads // tp_size
    if tp_size % kv_heads:
        raise ValueError("tp_size must be divisible by kv_heads when replicated")
    return 1


def grouped_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_start: int,
) -> torch.Tensor:
    """Attend with contiguous query-head groups over compact, shared K/V.

    Shapes are Q[B,Hq,Q,D], K/V[B,Hkv,T,D], output[B,Hq,Q,D]. Inputs must
    be finite CPU float32/float64 tensors with positive dimensions and Hq % Hkv=0.
    K/V positions start at zero; query rows start at query_start. The explicit
    mask permits keys j <= query_start+i. No dropout, ragged batch, or padding.
    This materializes scores and may copy reshaped Q; it is a semantic reference.
    Inputs are read-only. Arithmetic overflow is rejected rather than accepted.
    """
    tensors = (q, k, v)
    if any(t.ndim != 4 or any(size == 0 for size in t.shape) for t in tensors):
        raise ValueError("q, k, v must be rank-4 tensors with positive dimensions")
    if any(
        t.device.type != "cpu" or t.dtype not in (torch.float32, torch.float64)
        for t in tensors
    ):
        raise ValueError("this teaching reference requires CPU float32 or float64")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, v must have the same dtype")
    batch, query_heads, query_length, dim = q.shape
    if k.shape != v.shape or k.shape[0] != batch or k.shape[3] != dim:
        raise ValueError("K/V layouts must match Q batch and head dimension")
    kv_heads, key_length = k.shape[1:3]
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if (
        type(query_start) is not int
        or query_start < 0
        or query_start + query_length > key_length
    ):
        raise ValueError("query positions must lie within the supplied K/V history")
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError("inputs must be finite")

    group_size = query_heads // kv_heads
    grouped_q = q.reshape(batch, kv_heads, group_size, query_length, dim)
    scores = torch.einsum("bhgqd,bhkd->bhgqk", grouped_q, k) / math.sqrt(dim)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("score computation overflowed")
    query_positions = query_start + torch.arange(query_length, device=q.device)
    keep = (
        torch.arange(key_length, device=q.device)[None, :] <= query_positions[:, None]
    )
    probabilities = torch.softmax(scores.masked_fill(~keep, -torch.inf), dim=-1)
    output = torch.einsum("bhgqk,bhkd->bhgqd", probabilities, v)
    if not bool(torch.isfinite(output).all()):
        raise ValueError("value aggregation overflowed")
    return output.reshape(batch, query_heads, query_length, dim)
