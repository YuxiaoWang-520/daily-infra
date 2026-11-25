"""Run both exercises and the real CPU worker experiment."""

import argparse
import json
from pathlib import Path

import torch
from answers import assigned_shards, batch_indices, rank_indices
from runtime_demo import run_runtime_probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "torch_version": torch.__version__,
        "torch_build_commit": torch.version.git_version,
        "exercise_one": {
            "padded": [rank_indices(10, 3, r, shuffle=False) for r in range(3)],
            "truncated": [
                rank_indices(10, 3, r, shuffle=False, drop_last=True) for r in range(3)
            ],
            "rank_zero_drop_tail": batch_indices([0, 3, 6, 9], 3, drop_last=True),
            "shard_owners": [
                assigned_shards(6, 2, r, 2, w) for r in range(2) for w in range(2)
            ],
        },
        "exercise_two": run_runtime_probes(),
    }
    encoded = json.dumps(result, indent=2) + "\n"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
