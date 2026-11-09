"""Two-process CPU/Gloo shard ownership and Adam correctness demonstration."""

import json
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [[1, 0, 1, 0, 1], [0, 1, 0, 1, 1], [1, 1, 0, 0, 1], [0, 0, 1, 1, 1]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0.25, -0.5, 0.75, -0.25], dtype=torch.float64)
    initial = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
    return features, targets, initial


def _gather(local: torch.Tensor) -> torch.Tensor:
    output = torch.empty(local.numel() * 2, dtype=local.dtype)
    dist.all_gather_into_tensor(output, local.contiguous())
    return output


def _optimizer(parameter: torch.nn.Parameter) -> torch.optim.Adam:
    return torch.optim.Adam(
        [parameter], lr=0.01, betas=(0.9, 0.999), eps=1e-8, foreach=False
    )


def _case(rank: int, steps: int, normalization: str) -> dict:
    features, targets, initial = _fixture()
    local_slice = slice(0, 1) if rank == 0 else slice(1, 4)
    local_features, local_targets = features[local_slice], targets[local_slice]
    count = torch.tensor([len(local_targets)], dtype=torch.int64)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    global_samples = int(count.item())
    shard_elements = (initial.numel() + 1) // 2
    padded = torch.zeros(shard_elements * 2, dtype=torch.float64)
    padded[: initial.numel()].copy_(initial)
    local = torch.nn.Parameter(padded.chunk(2)[rank].clone())
    optimizer = _optimizer(local)
    reference = torch.nn.Parameter(initial.clone())
    reference_optimizer = _optimizer(reference)
    trace = []
    first_gradient = None
    reference_gradient = None
    shard_gradient = None
    for _ in range(steps):
        full = _gather(local.detach())
        trace.append("all_gather_parameters")
        weights = full[: initial.numel()].detach().requires_grad_()
        local_sum = 0.5 * (local_features @ weights - local_targets).square().sum()
        divisor = (
            global_samples
            if normalization == "sample_weighted"
            else len(local_targets) * 2
        )
        gradient = torch.autograd.grad(local_sum / divisor, weights)[0]
        padded_gradient = torch.zeros_like(full)
        padded_gradient[: initial.numel()].copy_(gradient)
        shard_gradient = torch.empty_like(local)
        dist.reduce_scatter_tensor(
            shard_gradient, padded_gradient, op=dist.ReduceOp.SUM
        )
        trace.append("reduce_scatter_gradients")
        if first_gradient is None:
            first_gradient = shard_gradient.clone()
        local.grad = shard_gradient
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = 0.5 * (features @ reference - targets).square().mean()
        reference_loss.backward()
        reference_gradient = reference.grad.detach().clone()
        reference_optimizer.step()
        del full, weights, local_sum, gradient, padded_gradient

    # These gathers inspect results. They are separate from the training trace.
    state = optimizer.state[local]
    reference_state = reference_optimizer.state[reference]
    gathered = {
        "parameter": _gather(local.detach()),
        "first_moment": _gather(state["exp_avg"]),
        "second_moment": _gather(state["exp_avg_sq"]),
        "gradient": _gather(shard_gradient),
        "first_gradient": _gather(first_gradient),
    }
    expected = {
        "parameter": reference.detach(),
        "first_moment": reference_state["exp_avg"],
        "second_moment": reference_state["exp_avg_sq"],
        "gradient": reference_gradient,
    }
    errors = {
        f"{name}_error": float((value[: initial.numel()] - expected[name]).abs().max())
        for name, value in gathered.items()
        if name in expected
    }
    padding_zero = all(
        bool(torch.count_nonzero(value[initial.numel() :]) == 0)
        for value in gathered.values()
    )
    return {
        "rank": rank,
        "normalization": normalization,
        "steps": steps,
        "local_samples": len(local_targets),
        "global_samples": global_samples,
        "parameters": initial.numel(),
        "shard_elements": shard_elements,
        "local_parameter_and_moments_elements": local.numel()
        + state["exp_avg"].numel()
        + state["exp_avg_sq"].numel(),
        "local_gradient_elements": shard_gradient.numel(),
        "padding_zero": padding_zero,
        "training_collectives": trace,
        "validation_all_gathers": len(gathered),
        "first_global_gradient": gathered["first_gradient"][: initial.numel()].tolist(),
        "final_parameter": gathered["parameter"][: initial.numel()].tolist(),
        **errors,
    }


def _worker(rank: int, store_uri: str, output_directory: str, steps: int) -> None:
    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            "gloo",
            init_method=store_uri,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=30),
        )
        cases = [_case(rank, steps, mode) for mode in ("sample_weighted", "rank_mean")]
        with (Path(output_directory) / f"rank{rank}.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump({"rank": rank, "cases": cases}, stream, indent=2)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_probe(steps: int = 3) -> dict:
    """Run fixed two-rank training cases with one to ten steps."""
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10:
        raise ValueError("steps must be an integer in [1, 10]")
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("the existing PyTorch build must provide CPU/Gloo")
    with tempfile.TemporaryDirectory(prefix="daily-infra-shards-") as directory:
        root = Path(directory)
        mp.spawn(
            _worker,
            args=((root / "store").as_uri(), directory, steps),
            nprocs=2,
            join=True,
        )
        ranks = [
            json.loads((root / f"rank{rank}.json").read_text()) for rank in range(2)
        ]
    return {
        "backend": "gloo",
        "world_size": 2,
        "dtype": "float64",
        "parameter_schedule": "one gather before forward; retain full parameters through backward",
        "scope": "explicit teaching collectives and local Adam; logical state counts, CPU correctness",
        "ranks": ranks,
    }
