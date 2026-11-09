"""Write a CPU correctness report to a new explicit output path."""

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from answers import ring_budget, state_budget
from sharded_probe import run_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe = run_probe()
    for rank in probe["ranks"]:
        for case in rank["cases"]:
            if case["normalization"] == "sample_weighted":
                if any(
                    case[key] > 1e-12
                    for key in (
                        "parameter_error",
                        "gradient_error",
                        "first_moment_error",
                        "second_moment_error",
                    )
                ):
                    raise RuntimeError(
                        "the shard update differs from the global reference"
                    )
            elif case["gradient_error"] <= 1e-4 or case["first_moment_error"] <= 1e-4:
                raise RuntimeError(
                    "the unequal-count negative control was not detected"
                )
    report = {
        "observed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "torch_build_git": torch.version.git_version,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
        },
        "state_budgets_billion_parameters_eight_ranks": {
            str(stage): asdict(state_budget(10**9, 8, stage)) for stage in range(4)
        },
        "ideal_ring_eight_elements_four_ranks": asdict(ring_budget(8, 4)),
        "probe": probe,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Saved CPU correctness report to {args.output}")


if __name__ == "__main__":
    main()
