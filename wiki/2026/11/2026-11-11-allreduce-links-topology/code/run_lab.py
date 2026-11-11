"""Emit modeled costs and actual CPU/Gloo correctness to a new output path."""

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch
from costs import binomial_tree_cost, ring_cost
from gloo_probe import run_gloo_probe
from ring import ring_allreduce


def _flat(ranks: int):
    return (
        {(r, (r + 1) % ranks): (f"link-{r}",) for r in range(ranks)},
        {f"link-{r}": 50e9 for r in range(ranks)},
    )


def _two_node():
    routes = {}
    bandwidths = {"A-tx": 50e9, "A-rx": 50e9, "B-tx": 50e9, "B-rx": 50e9}
    for source in range(4):
        for destination in range(4):
            if source == destination:
                continue
            if source // 2 == destination // 2:
                resource = f"local-{source}-{destination}"
                routes[source, destination] = (resource,)
                bandwidths[resource] = 200e9
            else:
                routes[source, destination] = (
                    "A-tx" if source < 2 else "B-tx",
                    "A-rx" if destination < 2 else "B-rx",
                )
    return routes, bandwidths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    small, large, alpha = 4096, 256 * 2**20, 5e-6
    models = {}
    for ranks in (4, 8):
        routes, bandwidths = _flat(ranks)
        for label, size in (("4KiB", small), ("256MiB", large)):
            models[f"flat-{ranks}-{label}"] = asdict(
                ring_cost(size, tuple(range(ranks)), routes, bandwidths, alpha)
            )
            models[f"full-block-binomial-tree-{ranks}-{label}"] = {
                "estimated_seconds": binomial_tree_cost(size, ranks, 50e9, alpha)
            }
    routes, bandwidths = _two_node()
    for label, order in (("grouped", (0, 1, 2, 3)), ("interleaved", (0, 2, 1, 3))):
        models[f"two-node-{label}"] = asdict(
            ring_cost(large, order, routes, bandwidths, alpha)
        )
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "modeled_inputs": {
            "payload_bytes": {"small": small, "large": large},
            "alpha_seconds": alpha,
            "flat_bandwidth_bytes_per_second": 50e9,
            "local_bandwidth_bytes_per_second": 200e9,
            "note": "Invented inputs for idealized models, not measured hardware bandwidth.",
        },
        "ring_trace": asdict(
            ring_allreduce([[10 * rank + i for i in range(8)] for rank in range(4)], 4)
        ),
        "modeled_costs": models,
        "actual_gloo_correctness": run_gloo_probe(),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
