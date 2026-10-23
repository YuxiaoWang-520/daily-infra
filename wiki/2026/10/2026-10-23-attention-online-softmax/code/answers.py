"""Small CPU FP64 references for online normalization and tiled attention.

No dropout, backward, KV cache, batch/head dimension or GPU implementation.
keep_mask=True means participate. Causal convention is local j <= i (top-left).
"""

import math

import torch


def _block(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _input(value: torch.Tensor, ndim: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError("inputs must be tensors")
    if (
        value.device.type != "cpu"
        or value.layout != torch.strided
        or value.ndim != ndim
    ):
        raise ValueError(f"inputs must be CPU strided {ndim}D tensors")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError("inputs must be float32 or float64")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("inputs must be finite")


def _mask(keep: torch.Tensor | None, shape: tuple[int, ...]) -> None:
    if keep is None:
        return
    if not isinstance(keep, torch.Tensor):
        raise ValueError("keep mask must be a tensor")
    if keep.device.type != "cpu" or keep.layout != torch.strided:
        raise ValueError("keep mask must be CPU strided")
    if keep.dtype != torch.bool or tuple(keep.shape) != shape:
        raise ValueError(f"keep mask must be bool with shape {shape}")


@torch.inference_mode()
def online_softmax(
    x: torch.Tensor, block_size: int, keep_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Two-pass CPU FP64 softmax; empty/all-masked vectors return zeros.

    First pass keeps only running max/sum; output probabilities require a second
    pass. Full x and output are still present; no gradient graph is constructed.
    """
    _input(x, 1)
    _block(block_size, "block_size")
    _mask(keep_mask, tuple(x.shape))
    values = x.double()
    maximum = -math.inf
    total = 0.0
    for start in range(0, values.numel(), block_size):
        block = values[start : start + block_size]
        if keep_mask is not None:
            block = block[keep_mask[start : start + block_size]]
        if not block.numel():
            continue
        new_max = max(maximum, block.max().item())
        total = (
            total * math.exp(maximum - new_max)
            + torch.exp(block - new_max).sum().item()
        )
        maximum = new_max
    out = torch.zeros_like(values)
    if not total:
        return out
    for start in range(0, values.numel(), block_size):
        shifted = values[start : start + block_size] - maximum
        if keep_mask is not None:
            shifted.masked_fill_(~keep_mask[start : start + block_size], -math.inf)
        out[start : start + block_size] = torch.exp(shifted) / total
    return out


def _prepare(q, k, v, keep_mask, causal):
    for value in (q, k, v):
        _input(value, 2)
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("input dtypes must match")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0] or not q.shape[1]:
        raise ValueError("Q/K dimensions must match and D>0; K/V lengths must match")
    if type(causal) is not bool:
        raise ValueError("causal must be bool")
    _mask(keep_mask, (q.shape[0], k.shape[0]))
    return q.double(), k.double(), v.double(), 1.0 / math.sqrt(q.shape[1])


def _scores(q, k, scale, qi, kj, keep_mask, causal):
    scores = (q @ k.t()) * scale
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("score calculation overflowed")
    if keep_mask is not None:
        scores.masked_fill_(
            ~keep_mask[qi : qi + q.shape[0], kj : kj + k.shape[0]], -math.inf
        )
    if causal:
        rows = torch.arange(qi, qi + q.shape[0])[:, None]
        cols = torch.arange(kj, kj + k.shape[0])[None, :]
        scores.masked_fill_(cols > rows, -math.inf)
    return scores


@torch.inference_mode()
def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_mask: torch.Tensor | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Quadratic-memory FP64 reference; all-masked rows explicitly return zero."""
    q, k, v, scale = _prepare(q, k, v, keep_mask, causal)
    scores = _scores(q, k, scale, 0, 0, keep_mask, causal)
    weights = torch.zeros_like(scores)
    valid_rows = torch.isfinite(scores).any(dim=1)
    weights[valid_rows] = torch.softmax(scores[valid_rows], dim=1)
    out = weights @ v
    if not bool(torch.isfinite(out).all()):
        raise ValueError("output calculation overflowed")
    return out


@torch.inference_mode()
def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_q: int = 16,
    block_k: int = 16,
    keep_mask: torch.Tensor | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """CPU FP64 online attention, single head, no gradients or dropout.

    Scores are allocated per tile with per-query m/l/u, not as a separate
    persistent full score/probability table. A tile can cover the whole problem;
    its score size is min(block_q,Nq)*min(block_k,Nk) elements.
    Optional caller-provided dense mask itself has quadratic size. Input FP64
    promotion is linear storage; live tile arrays are multiple constant-sized
    temporaries, not an allocator-peak promise. Intermediate overflow is rejected.
    """
    _block(block_q, "block_q")
    _block(block_k, "block_k")
    q, k, v, scale = _prepare(q, k, v, keep_mask, causal)
    out = torch.zeros((q.shape[0], v.shape[1]), dtype=torch.float64)
    for qi in range(0, q.shape[0], block_q):
        qb = q[qi : qi + block_q]
        maximum = torch.full((qb.shape[0],), -math.inf, dtype=torch.float64)
        total = torch.zeros(qb.shape[0], dtype=torch.float64)
        numerator = torch.zeros((qb.shape[0], v.shape[1]), dtype=torch.float64)
        for kj in range(0, k.shape[0], block_k):
            kb, vb = k[kj : kj + block_k], v[kj : kj + block_k]
            scores = _scores(qb, kb, scale, qi, kj, keep_mask, causal)
            new_max = torch.maximum(maximum, scores.max(dim=1).values)
            # Avoid -inf - (-inf) for an as-yet empty/masked row.
            safe_max = torch.where(torch.isfinite(new_max), new_max, 0.0)
            alpha = torch.exp(maximum - safe_max)
            weights = torch.exp(scores - safe_max[:, None])
            total = total * alpha + weights.sum(dim=1)
            numerator = numerator * alpha[:, None] + weights @ vb
            if not bool(torch.isfinite(numerator).all()):
                raise ValueError("unnormalized numerator overflowed")
            maximum = new_max
        denominator = torch.where(total > 0, total, 1.0)
        out[qi : qi + qb.shape[0]] = numerator / denominator[:, None]
    return out
