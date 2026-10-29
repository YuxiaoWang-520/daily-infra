"""CPU teaching models, not a reproduction of the vLLM scheduler."""

from collections import deque
from collections.abc import Sequence
from typing import Literal

from models import Admission, Prefill, Request, Simulation, Step, require_int


def simulate(
    requests: Sequence[Request],
    capacity: int,
    policy: Literal["static", "continuous"],
) -> Simulation:
    """Schedule one-query-token iterations, with unlimited token/KV budgets.

    Arrivals at tick t are eligible for [t, t+1); outputs arrive at t+1.
    Ties use input order. Output lengths only determine when requests stop.
    """
    require_int(capacity, "capacity", 1)
    if policy not in ("static", "continuous"):
        raise ValueError("policy must be static or continuous")
    requests = tuple(requests)
    ids = [request.request_id for request in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("request IDs must be unique")

    pending = deque(sorted(requests, key=lambda request: request.arrival_tick))
    ready: deque[Request] = deque()
    active: list[Request] = []
    remaining = {request.request_id: request.output_tokens for request in requests}
    output_ticks: dict[str, list[int]] = {request_id: [] for request_id in ids}
    steps: list[Step] = []
    tick = 0

    while pending or ready or active:
        if not active and not ready and pending:
            tick = max(tick, pending[0].arrival_tick)
        while pending and pending[0].arrival_tick <= tick:
            ready.append(pending.popleft())

        if policy == "continuous" or not active:
            while ready and len(active) < capacity:
                active.append(ready.popleft())

        steps.append(Step(tick, tuple(request.request_id for request in active)))
        for request in active:
            output_ticks[request.request_id].append(tick + 1)
            remaining[request.request_id] -= 1
        active = [request for request in active if remaining[request.request_id] > 0]
        tick += 1

    return Simulation(
        tuple(steps),
        tuple((request_id, tuple(output_ticks[request_id])) for request_id in ids),
    )


def admit_prefills(
    waiting: Sequence[Prefill],
    *,
    running_decodes: int,
    max_num_seqs: int,
    token_budget: int,
    free_blocks: int,
    block_size: int,
) -> Admission:
    """Admit a whole-prefill FCFS prefix after running decodes are secured.

    Each running decode uses one query token. free_blocks is measured AFTER
    securing all running work. No prefix hits, sharing, preemption or chunking.
    Admission allocates prompt blocks only; future decode growth is not reserved.
    """
    require_int(running_decodes, "running_decodes")
    require_int(max_num_seqs, "max_num_seqs", 1)
    require_int(token_budget, "token_budget", 1)
    require_int(free_blocks, "free_blocks")
    require_int(block_size, "block_size", 1)
    if running_decodes > max_num_seqs or running_decodes > token_budget:
        raise ValueError(
            "running decodes must already fit the sequence and token limits"
        )
    waiting = tuple(waiting)
    if len({request.request_id for request in waiting}) != len(waiting):
        raise ValueError("waiting request IDs must be unique")

    admitted: list[str] = []
    remaining_tokens = token_budget - running_decodes
    remaining_blocks = free_blocks
    stopped_request = None
    stop_reason = None
    for request in waiting:
        needed_blocks = (request.prompt_tokens + block_size - 1) // block_size
        if running_decodes + len(admitted) >= max_num_seqs:
            stop_reason = "sequence_limit"
        elif request.prompt_tokens > remaining_tokens:
            stop_reason = "token_budget"
        elif needed_blocks > remaining_blocks:
            stop_reason = "kv_capacity"
        if stop_reason is not None:
            stopped_request = request.request_id
            break

        # Charge budgets only after every admission condition succeeds.
        admitted.append(request.request_id)
        remaining_tokens -= request.prompt_tokens
        remaining_blocks -= needed_blocks

    return Admission(
        admitted=tuple(admitted),
        num_scheduled_requests=running_decodes + len(admitted),
        num_scheduled_tokens=token_budget - remaining_tokens,
        remaining_tokens=remaining_tokens,
        remaining_blocks=remaining_blocks,
        stopped_request=stopped_request,
        stop_reason=stop_reason,
    )
