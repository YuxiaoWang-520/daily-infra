"""CPU answers: dependency schedules and globally normalized MSE gradients."""

from collections.abc import Sequence

import torch
from torch import nn


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _costs(values: Sequence[int] | None, stages: int) -> list[int]:
    if values is None:
        return [1] * stages
    if not isinstance(values, Sequence) or len(values) != stages:
        raise ValueError("Provide one positive integer duration per stage")
    result = list(values)
    for value in result:
        _positive_integer("duration", value)
    return result


def simulate_pipeline(
    stages: int,
    microbatches: int,
    schedule: str,
    *,
    forward_ticks: Sequence[int] | None = None,
    backward_ticks: Sequence[int] | None = None,
) -> dict[str, object]:
    """Earliest starts for fixed local orders, with zero communication cost.

    Each stage executes one indivisible F or B at a time. F completion saves
    one microbatch's backward state; B completion releases it. This counter
    excludes parameter, output, transport and backward-workspace allocations.
    The mathematical model permits m < p; some framework schedules do not.
    """
    _positive_integer("stages", stages)
    _positive_integer("microbatches", microbatches)
    if schedule not in {"gpipe", "1f1b"}:
        raise ValueError("schedule must be gpipe or 1f1b")
    forward = _costs(forward_ticks, stages)
    backward = _costs(backward_ticks, stages)
    orders = []
    for rank in range(stages):
        if schedule == "gpipe":
            order = [("F", i) for i in range(microbatches)]
            order += [("B", i) for i in range(microbatches)]
        else:
            warmup = min(stages - rank - 1, microbatches)
            order = [("F", i) for i in range(warmup)]
            for i in range(microbatches - warmup):
                order.extend([("F", warmup + i), ("B", i)])
            order += [("B", i) for i in range(microbatches - warmup, microbatches)]
        orders.append(order)

    positions = [0] * stages
    stage_free = [0] * stages
    live = [0] * stages
    peak = [0] * stages
    finish = {}
    events = []
    while len(events) < 2 * stages * microbatches:
        progressed = False
        for rank, order in enumerate(orders):
            if positions[rank] == len(order):
                continue
            phase, microbatch = order[positions[rank]]
            dependencies = []
            if phase == "F" and rank:
                dependencies.append((rank - 1, "F", microbatch))
            if phase == "B":
                dependencies.append((rank, "F", microbatch))
                if rank < stages - 1:
                    dependencies.append((rank + 1, "B", microbatch))
            if any(key not in finish for key in dependencies):
                continue
            start = max([stage_free[rank], *(finish[key] for key in dependencies)])
            duration = forward[rank] if phase == "F" else backward[rank]
            end = start + duration
            events.append(
                {
                    "stage": rank,
                    "phase": phase,
                    "microbatch": microbatch,
                    "start": start,
                    "end": end,
                }
            )
            finish[rank, phase, microbatch] = end
            stage_free[rank] = end
            positions[rank] += 1
            live[rank] += 1 if phase == "F" else -1
            peak[rank] = max(peak[rank], live[rank])
            progressed = True
        if not progressed:
            raise RuntimeError("The local orders contain a dependency cycle")
    makespan = max(stage_free)
    busy = [microbatches * (f + b) for f, b in zip(forward, backward, strict=True)]
    return {
        "schedule": schedule,
        "stages": stages,
        "microbatches": microbatches,
        "makespan_ticks": makespan,
        "compute_utilization": sum(busy) / (stages * makespan),
        "busy_ticks_per_stage": busy,
        "peak_saved_microbatches": peak,
        "events": sorted(events, key=lambda e: (e["stage"], e["start"])),
    }


def microbatch_backward(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    chunk_sizes: Sequence[int],
) -> dict[str, object]:
    """Clear gradients, then accumulate sum-squared errors / global count.

    Intended for row-independent deterministic models and 2-D tensors.
    Performs no optimizer step; the caller updates parameters once afterward.
    """
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("inputs and targets must be matrices")
    if (
        inputs.shape[0] == 0
        or targets.numel() == 0
        or inputs.shape[0] != targets.shape[0]
    ):
        raise ValueError("Provide a nonempty batch with matching target rows")
    if not isinstance(chunk_sizes, Sequence) or not chunk_sizes:
        raise ValueError("Provide positive chunk sizes covering the batch")
    for size in chunk_sizes:
        _positive_integer("chunk size", size)
    if sum(chunk_sizes) != inputs.shape[0]:
        raise ValueError("Chunk sizes must cover the batch exactly")
    model.zero_grad(set_to_none=True)
    outputs = []
    loss_value = 0.0
    offset = 0
    for size in chunk_sizes:
        output = model(inputs[offset : offset + size])
        target = targets[offset : offset + size]
        if output.shape != target.shape:
            raise ValueError("Model output and target shapes must agree exactly")
        loss = (output - target).square().sum() / targets.numel()
        loss.backward()
        outputs.append(output.detach())
        loss_value += loss.item()
        offset += size
    return {"output": torch.cat(outputs), "loss": loss_value}
