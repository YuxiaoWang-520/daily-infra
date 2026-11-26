"""Run both exercises; write synthetic results to the requested output directory."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from answers import ddp_sum_scale, first_fit_decreasing, packed_cost, padded_cost
from toy_model import compare_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    lengths = [8, 2, 7, 3]
    groups = first_fit_decreasing(lengths, 10)
    costs = {
        "original": padded_cost(lengths, 2),
        "bucketed": padded_cost(lengths, 2, order=[0, 2, 3, 1]),
        "fixed_dense_packed": packed_cost(lengths, 10, groups),
    }
    ignored = F.cross_entropy(
        torch.zeros(2, 8, device="cpu"), torch.tensor([-100, -100], device="cpu")
    )
    counter_lengths = [6, 5, 3, 2, 2, 2]
    counter_groups = first_fit_decreasing(counter_lengths, 10)
    feasible_groups = [[0, 3, 4], [1, 2, 5]]
    packed_cost(counter_lengths, 10, feasible_groups)
    sums, counts = [6, 6], [3, 1]
    result = {
        "torch_version": torch.__version__,
        "torch_build_commit": torch.version.git_version,
        "costs_are_arithmetic_models_not_benchmarks": True,
        "exercise_one": {
            "lengths": lengths,
            "capacity": 10,
            "groups": groups,
            "costs": {
                k: {**asdict(c), "utilization": c.utilization} for k, c in costs.items()
            },
            "nonoptimal_ffd_example": {
                "lengths": counter_lengths,
                "heuristic_groups": counter_groups,
                "feasible_groups": feasible_groups,
                "heuristic_pack_count": len(counter_groups),
                "feasible_pack_count": len(feasible_groups),
            },
        },
        "exercise_two": compare_models(),
        "normalization": {
            "local_loss_sums": sums,
            "valid_target_counts": counts,
            "global_token_mean": sum(sums) / sum(counts),
            "mean_of_local_means": sum(s / n for s, n in zip(sums, counts, strict=True))
            / len(sums),
            "scale_before_standard_ddp_average": ddp_sum_scale(counts),
            "all_ignored_library_mean_is_nan": bool(torch.isnan(ignored)),
        },
        "optimized_varlen_gpu_run": "not run: this runner checks CPU reference semantics only",
    }
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
