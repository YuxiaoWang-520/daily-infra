"""CPU teaching answers for tensor-parallel algebra and ring traffic."""

import torch


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def tp_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    tp_size: int,
) -> dict[str, object]:
    """Simulate even feature shards in one process, preserving autograd."""
    _positive_integer("tp_size", tp_size)
    if x.ndim != 2 or w1.ndim != 2 or w2.ndim != 2:
        raise ValueError("X, W1 and W2 must be matrices")
    if x.shape[1] != w1.shape[0] or w1.shape[1] != w2.shape[0]:
        raise ValueError("The logical matrix dimensions must agree")
    if b1.shape != (w1.shape[1],) or b2.shape != (w2.shape[1],):
        raise ValueError("Each bias must match its layer's output dimension")
    if w1.shape[1] == 0 or w1.shape[1] % tp_size:
        raise ValueError(
            "The intermediate features must split into equal nonempty shards"
        )
    preactivations, hidden_shards, partial_outputs = [], [], []
    partitions = zip(
        w1.chunk(tp_size, dim=1),
        b1.chunk(tp_size, dim=0),
        w2.chunk(tp_size, dim=0),
        strict=True,
    )
    for local_w1, local_b1, local_w2 in partitions:
        preactivation = x @ local_w1 + local_b1
        hidden = torch.relu(preactivation)
        preactivations.append(preactivation)
        hidden_shards.append(hidden)
        partial_outputs.append(hidden @ local_w2)
    output = torch.stack(partial_outputs).sum(dim=0) + b2
    return {
        "preactivations": preactivations,
        "hidden_shards": hidden_shards,
        "partial_outputs": partial_outputs,
        "output": output,
    }


def collective_budget(
    tokens: int,
    hidden: int,
    tp_size: int,
    element_bytes: int,
    *,
    sequence_parallel: bool = False,
    regather_input_for_wgrad: bool = False,
) -> dict[str, int]:
    """Count one MLP forward/backward using ideal equally chunked rings.

    Boundaries have tokens*hidden elements and equal forward/gradient dtypes.
    Bias-gradient reductions and work outside this MLP are excluded.
    The optional re-gather models saving only the sequence-sharded X.
    """
    for name, value in (
        ("tokens", tokens),
        ("hidden", hidden),
        ("tp_size", tp_size),
        ("element_bytes", element_bytes),
    ):
        _positive_integer(name, value)
    payload = tokens * hidden * element_bytes
    if payload % tp_size:
        raise ValueError("This equal-ring-chunk model requires divisible payload bytes")
    if sequence_parallel and tokens % tp_size:
        raise ValueError("This sequence-parallel model requires equal token shards")
    if regather_input_for_wgrad and not sequence_parallel:
        raise ValueError("The input re-gather option applies to sequence-parallel mode")
    active = int(tp_size > 1)
    all_reduces = 0 if sequence_parallel else 2 * active
    all_gathers = (
        (2 + int(regather_input_for_wgrad)) * active if sequence_parallel else 0
    )
    reduce_scatters = 2 * active if sequence_parallel else 0
    ring_phases = 2 * all_reduces + all_gathers + reduce_scatters
    sent_bytes = ring_phases * (tp_size - 1) * (payload // tp_size)
    return {
        "payload_bytes": payload,
        "all_reduce_calls": all_reduces,
        "all_gather_calls": all_gathers,
        "reduce_scatter_calls": reduce_scatters,
        "ring_rounds": ring_phases * (tp_size - 1),
        "sent_bytes_per_rank": sent_bytes,
        "received_bytes_per_rank": sent_bytes,
    }
