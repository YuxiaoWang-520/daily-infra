"""Run numerical checks and print synthetic timelines; no GPU timing."""

import json
from dataclasses import asdict

import torch
from answers import chunked_attention, schedule_prefill
from torch.nn import functional as F


def main() -> None:
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(123)
    q, k, v = (
        torch.randn(8, 4, dtype=torch.float64, generator=generator, device="cpu")
        for _ in range(3)
    )
    expected = F.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None], dropout_p=0.0, is_causal=True
    )[0, 0]
    numerical = []
    for chunk_size in (1, 3, 8, 11):
        actual = chunked_attention(q, k, v, chunk_size)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        numerical.append(
            dict(
                chunk_size=chunk_size,
                max_abs_error=(actual - expected).abs().max().item(),
            )
        )

    uniform_q = uniform_k = torch.zeros(6, 2, dtype=torch.float64, device="cpu")
    values = torch.arange(6, dtype=torch.float64, device="cpu").reshape(6, 1)
    correct = chunked_attention(uniform_q, uniform_k, values, 3)
    wrong = F.scaled_dot_product_attention(
        uniform_q[None, None, 3:],
        uniform_k[None, None],
        values[None, None],
        dropout_p=0.0,
        is_causal=True,
    )[0, 0]
    counterexample = dict(
        global_query_position=3,
        correct_visible_key_positions=[0, 1, 2, 3],
        correct_output=correct[3].item(),
        wrong_local_triangular_output=wrong[0].item(),
    )

    timelines = []
    for budget in (18, 6, 4, 2):
        schedule = schedule_prefill(16, 2, 4, budget)
        times = (0.0,) + schedule.decode_output_times_ms[0]
        max_gap = max(right - left for left, right in zip(times, times[1:]))
        timelines.append(
            dict(
                token_budget=budget,
                prefill_chunks=[r.prefill_tokens for r in schedule.rounds],
                prompt_first_token_ms=schedule.prompt_first_token_ms,
                decode_output_times_ms=schedule.decode_output_times_ms,
                max_decode_gap_ms=max_gap,
                last_tracked_event_ms=schedule.rounds[-1].end_ms,
                rounds=[asdict(r) for r in schedule.rounds],
            )
        )
    print(
        json.dumps(
            dict(
                scope="Attention: real CPU arithmetic. Timelines: invented cost formula, not GPU or serving measurements.",
                environment=dict(
                    torch=torch.__version__,
                    device="cpu",
                    dtype="float64",
                    threads=torch.get_num_threads(),
                ),
                attention=numerical,
                mask_counterexample=counterexample,
                synthetic_timelines=timelines,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
