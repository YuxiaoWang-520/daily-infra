"""Print deterministic scheduling and admission evidence; no wall-clock benchmark."""

import json
from dataclasses import asdict

from answers import admit_prefills, simulate
from models import Prefill, Request


def main() -> None:
    requests = tuple(
        Request(name, 0, count) for name, count in zip("ABCD", (1, 4, 2, 1))
    )
    schedules = {}
    for policy in ("static", "continuous"):
        simulation = simulate(requests, 2, policy)
        outputs = dict(simulation.output_ticks)
        start = min(request.arrival_tick for request in requests)
        end = max(ticks[-1] for ticks in outputs.values())
        num_outputs = sum(len(ticks) for ticks in outputs.values())
        schedules[policy] = dict(
            trace=[asdict(step) for step in simulation.steps],
            output_ticks=outputs,
            observation_start_tick=start,
            observation_end_tick=end,
            output_tokens=num_outputs,
            output_tokens_per_tick=num_outputs / (end - start),
            filled_slot_fraction=num_outputs / (2 * len(simulation.steps)),
            first_output_delay_ticks={
                request.request_id: outputs[request.request_id][0]
                - request.arrival_tick
                for request in requests
            },
        )
    waiting = (Prefill("C", 4), Prefill("D", 3), Prefill("E", 1))
    base = dict(
        running_decodes=2, max_num_seqs=4, token_budget=8, free_blocks=3, block_size=4
    )
    admission = {
        "token_limited": asdict(admit_prefills(waiting, **base)),
        "sequence_limited": asdict(
            admit_prefills(waiting, **(base | {"token_budget": 16}))
        ),
        "kv_limited": asdict(
            admit_prefills((Prefill("C", 5),), **(base | {"free_blocks": 1}))
        ),
    }
    # Synthetic timestamps demonstrate the units, not measured server latency.
    arrival_s = 0.0
    outputs_s = (0.12, 0.16, 0.22)
    gaps_s = tuple(right - left for left, right in zip(outputs_s, outputs_s[1:]))
    metrics = dict(
        synthetic=True,
        output_times_seconds=outputs_s,
        ttft_seconds=outputs_s[0] - arrival_s,
        itl_seconds=gaps_s,
        tpot_seconds=(outputs_s[-1] - outputs_s[0]) / (len(outputs_s) - 1),
        e2e_seconds=outputs_s[-1] - arrival_s,
    )
    assert len(schedules["static"]["trace"]) == 6
    assert len(schedules["continuous"]["trace"]) == 4
    assert admission["token_limited"]["admitted"] == ("C",)
    assert admission["sequence_limited"]["admitted"] == ("C", "D")
    assert admission["kv_limited"]["stop_reason"] == "kv_capacity"
    print(
        json.dumps(
            dict(
                scope="CPU logical ticks and synthetic timestamps only; not GPU throughput or utilization",
                schedules=schedules,
                admission=admission,
                metrics=metrics,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
