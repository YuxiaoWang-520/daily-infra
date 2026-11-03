"""Run exact teaching examples; the millisecond values are assumed inputs."""

import json
from dataclasses import asdict

from answers import ddp_loss_scales, global_mean_from_local_means, schedule_buckets


def main() -> None:
    mib = 1024 * 1024
    means = [[1], [29 / 3]]
    counts = [1, 3]
    correct = global_mean_from_local_means(means, counts)[0]
    scales = ddp_loss_scales(counts)
    assert correct == 7.5 and scales == (0.5, 1.5)
    ready = [[2, 4, 6, 8], [2, 4, 6, 8]]
    plans = {}
    for name, buckets in {
        "four_small": [[0], [1], [2], [3]],
        "two_medium": [[0, 1], [2, 3]],
        "one_large": [[0, 1, 2, 3]],
        "blocked_order": [[3], [0], [1], [2]],
    }.items():
        plans[name] = asdict(
            schedule_buckets(ready, [4 * mib] * 4, buckets, 2, 4 * mib)
        )
    plans["straggler"] = asdict(
        schedule_buckets(
            [ready[0], [2, 4, 6, 12]], [4 * mib] * 4, [[0, 1], [2, 3]], 2, 4 * mib
        )
    )
    assert [
        plans[n]["finish_ms"] for n in ("four_small", "two_medium", "one_large")
    ] == [14, 12, 14]
    assert plans["straggler"]["finish_ms"] == 16
    print(
        json.dumps(
            {
                "scope": "Actual CPU arithmetic; all readiness and communication costs are synthetic, not measured latency.",
                "reduction": {
                    "counts": counts,
                    "local_means": means,
                    "unweighted_rank_mean": (1 + 29 / 3) / 2,
                    "global_sample_mean": correct,
                    "local_mean_loss_scales": scales,
                },
                "timelines": plans,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
