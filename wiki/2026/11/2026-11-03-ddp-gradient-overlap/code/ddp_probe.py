"""A real two-process CPU/Gloo correctness probe; no timing or GPU claims."""

import argparse
import json
import logging
import math
import tempfile
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def _model() -> DDP:
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(1)
    return DDP(model)


def _loss(model: DDP, values: list[float]) -> torch.Tensor:
    x = torch.tensor(values, dtype=torch.float64).reshape(-1, 1)
    return 0.5 * model(x).square().mean()


def _worker(rank: int, store_uri: str, output_dir: str) -> None:
    logging.basicConfig(level=logging.INFO, format=f"rank{rank}: %(message)s")
    torch.set_num_threads(1)
    logging.info("initializing CPU Gloo")
    dist.init_process_group(
        "gloo",
        init_method=store_uri,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    results = []
    try:
        check = torch.tensor([rank + 1.0], dtype=torch.float64)
        dist.all_reduce(check)
        assert check.item() == 3
        logging.info("SUM AllReduce passed")
        for name, values, scale, expected in (
            ("equal_counts", [1.0] if rank == 0 else [3.0], 1.0, 5.0),
            (
                "unequal_unweighted",
                [1.0] if rank == 0 else [2.0, 3.0, 4.0],
                1.0,
                16 / 3,
            ),
            (
                "unequal_weighted",
                [1.0] if rank == 0 else [2.0, 3.0, 4.0],
                0.5 if rank == 0 else 1.5,
                7.5,
            ),
        ):
            model = _model()
            logging.info("DDP constructed for %s", name)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            optimizer.zero_grad(set_to_none=True)
            (_loss(model, values) * scale).backward()
            gradient = model.module.weight.grad.item()
            assert math.isclose(gradient, expected, rel_tol=1e-12, abs_tol=1e-12)
            optimizer.step()
            weight = model.module.weight.item()
            assert math.isclose(
                weight, 1 - 0.1 * expected, rel_tol=1e-12, abs_tol=1e-12
            )
            results.append(
                dict(
                    case=name,
                    local_count=len(values),
                    local_mean_loss_scale=scale,
                    gradient=gradient,
                    weight_after_sgd=weight,
                )
            )

        for name, divisor, expected in (
            ("accumulation_scaled", 2, 7.5),
            ("accumulation_unscaled", 1, 15.0),
        ):
            model = _model()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            optimizer.zero_grad(set_to_none=True)
            first_local_grad = None
            for micro, value in enumerate([1.0, 2.0] if rank == 0 else [3.0, 4.0]):
                context = model.no_sync() if micro == 0 else nullcontext()
                with context:
                    (_loss(model, [value]) / divisor).backward()
                if micro == 0:
                    first_local_grad = model.module.weight.grad.item()
                    assert first_local_grad == (1 if rank == 0 else 9) / divisor
            gradient = model.module.weight.grad.item()
            assert math.isclose(gradient, expected, rel_tol=1e-12, abs_tol=1e-12)
            optimizer.step()
            weight = model.module.weight.item()
            assert math.isclose(
                weight, 1 - 0.1 * expected, rel_tol=1e-12, abs_tol=1e-12
            )
            results.append(
                dict(
                    case=name,
                    first_no_sync_local_gradient=first_local_grad,
                    gradient=gradient,
                    weight_after_sgd=weight,
                )
            )
        (Path(output_dir) / f"rank{rank}.json").write_text(
            json.dumps({"rank": rank, "results": results}, indent=2) + "\n"
        )
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the machine-readable report to this explicit path.",
    )
    args = parser.parse_args()
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError(
            "This optional probe requires an existing PyTorch build with CPU Gloo."
        )
    with tempfile.TemporaryDirectory(prefix="wiki-ddp-") as temporary:
        folder = Path(temporary)
        mp.spawn(
            _worker,
            args=((folder / "store").as_uri(), str(folder)),
            nprocs=2,
            join=True,
        )
        ranks = [
            json.loads((folder / f"rank{rank}.json").read_text()) for rank in range(2)
        ]
    for left, right in zip(ranks[0]["results"], ranks[1]["results"]):
        assert left["case"] == right["case"]
        assert math.isclose(
            left["gradient"], right["gradient"], rel_tol=1e-12, abs_tol=1e-12
        )
        assert math.isclose(
            left["weight_after_sgd"],
            right["weight_after_sgd"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    report = {
        "scope": "Actual two-process CPU/Gloo DDP with default communication; no hooks, GPU or performance measurement.",
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "backend": "gloo",
        "world_size": 2,
        "dtype": "float64",
        "model": "Linear(1, 1, bias=False), w=1",
        "loss": "0.5 * mean((w*x)^2)",
        "sgd_lr": 0.1,
        "ranks": ranks,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
