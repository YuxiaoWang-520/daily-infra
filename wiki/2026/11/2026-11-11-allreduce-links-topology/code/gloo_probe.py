"""Check AllReduce SUM semantics on four fresh CPU/Gloo processes.

This does not exercise NCCL, CUDA, physical NIC topology or GPU timing.
"""

import json
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ring import ring_allreduce


def _worker(rank: int, world: int, rendezvous: str, directory: str) -> None:
    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=20),
        )
        values = torch.arange(8, dtype=torch.float64) + 10 * rank
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        expected = 60 + 4 * torch.arange(8, dtype=torch.float64)
        torch.testing.assert_close(values, expected, rtol=0, atol=0)
        report = {
            "rank": rank,
            "sum": values.tolist(),
            "mean": (values / world).tolist(),
            "correct": True,
        }
        with (Path(directory) / f"rank-{rank}.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(report, stream)
            stream.write("\n")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_gloo_probe() -> dict[str, Any]:
    """Spawn four ranks, validate each result and compare with the simulator."""
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("existing PyTorch must provide CPU Gloo")
    with tempfile.TemporaryDirectory(prefix="allreduce-gloo-") as temporary:
        directory = Path(temporary)
        mp.spawn(
            _worker,
            args=(4, (directory / "rendezvous").as_uri(), temporary),
            nprocs=4,
            join=True,
        )
        reports = [
            json.loads((directory / f"rank-{rank}.json").read_text())
            for rank in range(4)
        ]
    simulator = ring_allreduce([[10 * rank + i for i in range(8)] for rank in range(4)])
    for rank, report in enumerate(reports):
        if tuple(report["sum"]) != simulator.outputs[rank]:
            raise AssertionError(f"rank {rank}: backend result differs from exact sum")
    return {
        "backend": "gloo",
        "device": "cpu",
        "dtype": "float64",
        "fresh_processes": 4,
        "ranks": reports,
        "matches_exact_integer_simulator": True,
        "scope": "Same-host collective semantics; no NCCL path or performance measurement.",
    }
