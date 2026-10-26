"""Small CPU inference references; no production KV allocator or GPU benchmark."""

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class KVCache:
    """One layer, one head, one sequence: K=(T,d), V=(T,dv)."""

    k: torch.Tensor
    v: torch.Tensor

    @property
    def length(self) -> int:
        return self.k.shape[0]


def _matrix(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.ndim != 2:
        raise ValueError(f"{name} must be a CPU strided matrix")
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")


def _check_cache(cache: KVCache, d: int, dv: int, dtype: torch.dtype) -> None:
    if not isinstance(cache, KVCache):
        raise TypeError("cache must be KVCache")
    _matrix(cache.k, "cache.k")
    _matrix(cache.v, "cache.v")
    if cache.k.shape != (cache.v.shape[0], d) or cache.v.shape[1] != dv:
        raise ValueError("cache lengths or feature widths disagree")
    if cache.k.dtype != dtype or cache.v.dtype != dtype:
        raise TypeError("cache and current tensors must have the same dtype")


@torch.inference_mode()
def cached_attention(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    cache: KVCache | None = None,
) -> tuple[torch.Tensor, KVCache]:
    """Append one consecutive chunk and attend at its absolute causal positions.

    All three inputs have the same number of new positions. No padding, gaps,
    arbitrary mask, sliding window, dropout, batching, or gradient support.
    torch.cat copies the history: useful for correctness, not a fast allocator.
    Inputs and previous cache are never mutated; returned K/V own fresh storage.
    """
    for tensor, name in ((q, "q"), (k_new, "k_new"), (v_new, "v_new")):
        _matrix(tensor, name)
    n, d = q.shape
    if d == 0 or k_new.shape != (n, d) or v_new.shape[0] != n:
        raise ValueError("new chunk lengths or Q/K widths disagree")
    if q.dtype != k_new.dtype or q.dtype != v_new.dtype:
        raise TypeError("Q/K/V must have the same dtype")
    past = 0
    if cache is not None:
        _check_cache(cache, d, v_new.shape[1], q.dtype)
        past = cache.length
    keys = torch.cat(([cache.k] if cache is not None else []) + [k_new], dim=0)
    values = torch.cat(([cache.v] if cache is not None else []) + [v_new], dim=0)
    updated = KVCache(keys, values)
    if n == 0:
        return v_new.new_empty((0, v_new.shape[1])), updated
    scores = (q @ keys.T) / sqrt(d)
    if not torch.isfinite(scores).all():
        raise ValueError("attention scores overflowed")
    # Row i is at position past+i, not at local position i.
    keep = torch.arange(past + n)[None, :] <= (past + torch.arange(n))[:, None]
    scores.masked_fill_(~keep, -torch.inf)
    output = torch.softmax(scores, dim=-1) @ values
    if not torch.isfinite(output).all():
        raise ValueError("attention output overflowed")
    return output, updated


def _integer(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


class TinyDecoder:
    """Fixed random CPU FP64 decoder; no pretrained weights or language quality.

    Single sequence/head, learned absolute positions, pre-norm residual blocks,
    two-layer ReLU FFN. Caches must come from the same unchanged model and prefix;
    shape checks cannot establish that semantic identity. No EOS or training.
    """

    def __init__(
        self,
        vocab: int = 13,
        width: int = 8,
        layers: int = 2,
        max_length: int = 32,
        seed: int = 11,
    ) -> None:
        for name, value in (
            ("vocab", vocab),
            ("width", width),
            ("layers", layers),
            ("max_length", max_length),
        ):
            _integer(value, name, 1)
        _integer(seed, "seed", 0)
        self.vocab, self.width, self.layers, self.max_length = vocab, width, layers, max_length
        generator = torch.Generator().manual_seed(seed)

        def draw(rows: int, columns: int) -> torch.Tensor:
            return torch.randn(rows, columns, generator=generator, dtype=torch.float64) / sqrt(rows)

        self.embedding = draw(vocab, width)
        self.positions = draw(max_length, width)
        self.blocks = tuple(
            (
                draw(width, width),
                draw(width, width),
                draw(width, width),
                draw(width, width),
                draw(width, 2 * width),
                draw(2 * width, width),
            )
            for _ in range(layers)
        )
        self.unembed = draw(width, vocab)

    def _tokens(self, tokens: torch.Tensor, past: int) -> None:
        if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64:
            raise TypeError("token IDs must be an int64 tensor")
        if tokens.device.type != "cpu" or tokens.layout != torch.strided or tokens.ndim != 1:
            raise ValueError("token IDs must be a CPU strided vector")
        if tokens.numel() == 0 or (tokens < 0).any() or (tokens >= self.vocab).any():
            raise ValueError("tokens must be nonempty and within vocabulary")
        if past + tokens.numel() > self.max_length:
            raise ValueError("forwarded positions exceed max_length")

    def _run(
        self,
        tokens: torch.Tensor,
        caches: tuple[KVCache, ...] | None,
        incremental: bool,
    ) -> tuple[torch.Tensor, tuple[KVCache, ...]]:
        past = 0
        if caches is not None:
            if len(caches) != self.layers:
                raise ValueError("one cache is required per decoder layer")
            for cache in caches:
                _check_cache(cache, self.width, self.width, torch.float64)
            past = caches[0].length
            if any(cache.length != past for cache in caches):
                raise ValueError("all layer caches must have the same length")
        self._tokens(tokens, past)
        h = self.embedding[tokens] + self.positions[past : past + tokens.numel()]
        updated = []
        for layer, (wq, wk, wv, wo, w1, w2) in enumerate(self.blocks):
            normalized = F.layer_norm(h, (self.width,))
            q, k, v = normalized @ wq, normalized @ wk, normalized @ wv
            if incremental:
                output, cache = cached_attention(q, k, v, None if caches is None else caches[layer])
            else:
                # Independent square causal SDPA reference; does not call cached_attention.
                output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
                cache = KVCache(k, v)
            updated.append(cache)
            h = h + output @ wo
            h = h + torch.relu(F.layer_norm(h, (self.width,)) @ w1) @ w2
        logits = F.layer_norm(h, (self.width,)) @ self.unembed
        return logits, tuple(updated)

    @torch.inference_mode()
    def full(self, tokens: torch.Tensor) -> tuple[torch.Tensor, tuple[KVCache, ...]]:
        """Recompute all positions from zero with independent CPU SDPA."""
        return self._run(tokens, None, False)

    @torch.inference_mode()
    def cached(
        self,
        new_tokens: torch.Tensor,
        caches: tuple[KVCache, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[KVCache, ...]]:
        """Process only consecutive new tokens; None performs the initial prefill."""
        return self._run(new_tokens, caches, True)


@dataclass(frozen=True)
class GenerationResult:
    tokens: torch.Tensor
    caches: tuple[KVCache, ...] | None
    forward_lengths: tuple[int, ...]


@torch.inference_mode()
def greedy_generate(
    model: TinyDecoder,
    prompt: torch.Tensor,
    steps: int,
    *,
    use_cache: bool = True,
) -> GenerationResult:
    """Generate exactly steps tokens; the final sampled token is not forwarded.

    max_length limits forwarded positions. Returned token count can therefore be
    max_length+1. Fixed-length greedy sampling; no EOS, padding, or beam search.
    """
    _integer(steps, "steps", 0)
    if not isinstance(use_cache, bool):
        raise TypeError("use_cache must be boolean")
    model._tokens(prompt, 0)
    if steps and prompt.numel() + steps - 1 > model.max_length:
        raise ValueError("generation would exceed max_length for forwarded positions")
    tokens, caches, lengths = prompt.clone(), None, []
    for step in range(steps):
        if use_cache:
            new = tokens if step == 0 else tokens[-1:]
            logits, caches = model.cached(new, caches)
            lengths.append(new.numel())
        else:
            logits, _ = model.full(tokens)
            lengths.append(tokens.numel())
        tokens = torch.cat((tokens, logits[-1].argmax().reshape(1)))
    return GenerationResult(tokens, caches, tuple(lengths))
