"""CPU teaching references; neither function is a production engine."""

import math

import torch
from models import Round, Schedule, require_int


def chunked_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Single-head causal Attention with retained prefix KV and absolute rows.

    Inputs are finite CPU float32/float64 tensors: Q,K=[L,D], V=[L,Dv].
    torch.cat models the cache contents; its copies are not an engine design.
    """
    require_int(chunk_size, "chunk_size", 1)
    for tensor in (q, k, v):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError("Q, K and V must be rank-2 tensors")
        if tensor.device.type != "cpu" or tensor.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("this reference accepts CPU float32/float64 tensors only")
        if not torch.isfinite(tensor).all():
            raise ValueError("Q, K and V must contain finite values")
    if q.shape != k.shape or q.shape[0] != v.shape[0]:
        raise ValueError("Q/K must share [L,D]; V must share L")
    if q.shape[0] == 0 or q.shape[1] == 0 or v.shape[1] == 0:
        raise ValueError("sequence and feature dimensions must be positive")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K and V must have the same dtype")

    cached_k, cached_v = k[:0], v[:0]
    outputs = []
    for start in range(0, q.shape[0], chunk_size):
        end = min(start + chunk_size, q.shape[0])
        cached_k = torch.cat((cached_k, k[start:end]), dim=0)
        cached_v = torch.cat((cached_v, v[start:end]), dim=0)
        scores = (q[start:end] @ cached_k.T) / math.sqrt(q.shape[1])
        query_positions = torch.arange(start, end, device=q.device)[:, None]
        key_positions = torch.arange(end, device=q.device)[None, :]
        allowed = key_positions <= query_positions
        probabilities = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
        outputs.append(probabilities @ cached_v)
    return torch.cat(outputs, dim=0)


def schedule_prefill(
    prompt_tokens: int, decode_requests: int, decode_steps: int, token_budget: int
) -> Schedule:
    """Synthetic complete-work comparison with decodes scheduled first.

    All requests are ready at t=0. Each old decode needs decode_steps MORE
    outputs, one per scheduled round; its previous output was at t=0.
    The new prompt stops being tracked at its first output. KV/slots suffice.
    Invented duration: 1 ms/round + 0.5 ms/decode token + 1 ms/prefill token.
    This cost model does not predict GPU latency or piggyback efficiency.
    """
    require_int(prompt_tokens, "prompt_tokens", 1)
    require_int(decode_requests, "decode_requests")
    require_int(decode_steps, "decode_steps", 1)
    require_int(token_budget, "token_budget", 1)
    if token_budget < decode_requests:
        raise ValueError("budget must fit every ongoing decode in this model")

    remaining_prompt = prompt_tokens
    remaining_decode_steps = decode_steps if decode_requests else 0
    decode_outputs: list[list[float]] = [[] for _ in range(decode_requests)]
    rounds = []
    now = 0.0
    first_token_ms = None
    while remaining_prompt or remaining_decode_steps:
        decodes = decode_requests if remaining_decode_steps else 0
        prefill = min(remaining_prompt, token_budget - decodes)
        start = now
        now += 1.0 + 0.5 * decodes + prefill
        remaining_prompt -= prefill
        if decodes:
            remaining_decode_steps -= 1
            for outputs in decode_outputs:
                outputs.append(now)
        if remaining_prompt == 0 and first_token_ms is None:
            first_token_ms = now
        rounds.append(
            Round(start, now, decodes, prefill, prompt_tokens - remaining_prompt)
        )

    assert first_token_ms is not None
    return Schedule(
        tuple(rounds), first_token_ms, tuple(tuple(ts) for ts in decode_outputs)
    )
