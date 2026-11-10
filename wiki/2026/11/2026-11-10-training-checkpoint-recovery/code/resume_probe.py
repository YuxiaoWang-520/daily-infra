"""A small CPU checkpoint experiment with five genuine process restarts.

The model, dataset, Adam settings and LR schedule are fixed by CONFIG. The
negative controls intentionally omit exactly one restored state component.
They are diagnostic counterexamples, not supported resume policies.
"""

import argparse
import copy
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from checkpoint_io import atomic_save, load_checkpoint
from torch import nn

CONFIG = {
    "fixture": "linear-tanh-dropout-v1",
    "dataset": "synthetic-arange-20x3-v1",
    "batch_size": 2,
    "total_steps": 10,
    "dtype": "float64",
    "adam_lr": 0.03,
    "scheduler_step_size": 3,
    "scheduler_gamma": 0.8,
}
MODES = ("none", "optimizer", "rng", "cursor", "scheduler")


class Trainer:
    """A fixed one-epoch fixture whose cursor counts consumed samples."""

    def __init__(self, seed: int = 2026) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        self.features = torch.arange(60, dtype=torch.float64).reshape(20, 3) / 60
        self.targets = 0.5 * self.features.sum(dim=1, keepdim=True) + 0.1
        self.model = nn.Sequential(
            nn.Linear(3, 4), nn.Tanh(), nn.Dropout(0.4), nn.Linear(4, 1)
        ).to(dtype=torch.float64)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=CONFIG["adam_lr"], foreach=False
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=CONFIG["scheduler_step_size"],
            gamma=CONFIG["scheduler_gamma"],
        )
        self.order = torch.randperm(20, generator=torch.Generator().manual_seed(77))
        self.cursor = 0
        self.completed_steps = 0

    def step(self) -> dict[str, Any]:
        """Run one full update, then commit progress with gradients cleared."""
        if self.completed_steps >= CONFIG["total_steps"] or self.cursor + 2 > 20:
            raise ValueError("fixture has no remaining complete batch")
        ids = self.order[self.cursor : self.cursor + 2]
        inputs = self.features[ids] + 0.03 * random.random()
        self.optimizer.zero_grad(set_to_none=True)
        lr_used = self.optimizer.param_groups[0]["lr"]
        loss = nn.functional.mse_loss(self.model(inputs), self.targets[ids])
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.cursor += 2
        self.completed_steps += 1
        return {"ids": ids.tolist(), "loss": loss.item(), "lr_used": lr_used}

    def snapshot(self) -> dict[str, Any]:
        """Freeze all state at the post-update, zero-gradient boundary."""
        return copy.deepcopy(
            {
                "schema": 1,
                "config": CONFIG,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "order": self.order,
                "cursor": self.cursor,
                "completed_steps": self.completed_steps,
            }
        )

    def outcome(self, trace: list[dict[str, Any]]) -> dict[str, Any]:
        """Collect comparison evidence without consuming random values."""
        states = list(self.optimizer.state.values())
        return {
            "parameters": torch.cat(
                [p.detach().flatten() for p in self.model.parameters()]
            ),
            "moments": torch.cat(
                [
                    state[name].flatten()
                    for state in states
                    for name in ("exp_avg", "exp_avg_sq")
                ]
            ),
            "optimizer_steps": [state["step"].item() for state in states],
            "trace": trace,
            "next_lr": self.optimizer.param_groups[0]["lr"],
            "scheduler_epoch": self.scheduler.last_epoch,
            "cursor": self.cursor,
            "completed_steps": self.completed_steps,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
        }


def validate_snapshot(payload: dict[str, Any]) -> None:
    """Reject checkpoints that cannot describe this fixture's committed state."""
    if type(payload.get("schema")) is not int or payload["schema"] != 1:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("config") != CONFIG:
        raise ValueError("training configuration differs from this fixture")
    steps, cursor = payload.get("completed_steps"), payload.get("cursor")
    if type(steps) is not int or not 0 <= steps <= 10:
        raise ValueError("completed_steps must be an integer in [0, 10]")
    if type(cursor) is not int or cursor != steps * 2:
        raise ValueError("cursor must equal the number of committed consumed samples")
    order = payload.get("order")
    if (
        not isinstance(order, torch.Tensor)
        or order.device.type != "cpu"
        or order.dtype != torch.int64
        or tuple(order.shape) != (20,)
        or not torch.equal(order.sort().values, torch.arange(20))
    ):
        raise ValueError("sample order must be a permutation of [0, 20)")


def restore(payload: dict[str, Any], omit: str = "none") -> Trainer:
    """Reconstruct objects first, restore state, and restore RNG last."""
    if omit not in MODES:
        raise ValueError("unknown omit control")
    validate_snapshot(payload)
    trainer = Trainer(seed=999)
    trainer.model.load_state_dict(payload["model"])
    # The scheduler already exists when optimizer state is loaded, as required
    # by Optimizer.load_state_dict's warning about LR scheduler initialization.
    if omit != "scheduler":
        trainer.scheduler.load_state_dict(payload["scheduler"])
    if omit != "optimizer":
        # load_state_dict may reuse same-device tensors; keep the input frozen.
        trainer.optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
    trainer.order = payload["order"].clone()
    trainer.completed_steps = payload["completed_steps"]
    if omit != "cursor":
        trainer.cursor = payload["cursor"]
    if omit != "rng":
        random.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])
    return trainer


def _child(checkpoint: Path, result: Path, omit: str) -> None:
    torch.set_num_threads(1)
    trainer = restore(load_checkpoint(checkpoint), omit=omit)
    remaining = CONFIG["total_steps"] - trainer.completed_steps
    trace = [trainer.step() for _ in range(remaining)]
    atomic_save(trainer.outcome(trace), result)


def _compare(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    expected_trace, actual_trace = expected["trace"][5:], actual["trace"]
    return {
        "parameter_max_abs_error": (expected["parameters"] - actual["parameters"])
        .abs()
        .max()
        .item(),
        "moment_max_abs_error": (expected["moments"] - actual["moments"])
        .abs()
        .max()
        .item(),
        "loss_max_abs_error": max(
            abs(left["loss"] - right["loss"])
            for left, right in zip(expected_trace, actual_trace, strict=True)
        ),
        "sample_ids_equal": [row["ids"] for row in expected_trace]
        == [row["ids"] for row in actual_trace],
        "rng_equal": torch.equal(expected["torch_rng"], actual["torch_rng"])
        and expected["python_rng"] == actual["python_rng"],
        "lr_equal": expected["next_lr"] == actual["next_lr"]
        and expected["scheduler_epoch"] == actual["scheduler_epoch"]
        and [row["lr_used"] for row in expected_trace]
        == [row["lr_used"] for row in actual_trace],
        "optimizer_steps_equal": expected["optimizer_steps"]
        == actual["optimizer_steps"],
        "completed_steps": actual["completed_steps"],
        "cursor": actual["cursor"],
        "next_lr": actual["next_lr"],
        "scheduler_epoch": actual["scheduler_epoch"],
        "resumed_sample_ids": [row["ids"] for row in actual_trace],
    }


def run_probe() -> dict[str, Any]:
    """Save at step five, discard steps six to eight, restart each policy."""
    python_rng, torch_rng = random.getstate(), torch.get_rng_state()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        baseline = Trainer()
        reference = baseline.outcome([baseline.step() for _ in range(10)])
        producer = Trainer()
        for _ in range(5):
            producer.step()
        with tempfile.TemporaryDirectory(prefix="checkpoint-lab-") as temporary:
            directory = Path(temporary)
            checkpoint = directory / "step-5.pt"
            atomic_save(producer.snapshot(), checkpoint)
            for _ in range(3):
                producer.step()
            discarded_at = producer.completed_steps
            del producer
            comparisons = {}
            for omit in MODES:
                result = directory / f"resumed-{omit}.pt"
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--checkpoint",
                        str(checkpoint),
                        "--result",
                        str(result),
                        "--omit",
                        omit,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                comparisons[omit] = _compare(reference, load_checkpoint(result))
        return {
            "schema": 1,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float64",
            "threads": 1,
            "config": CONFIG.copy(),
            "saved_step": 5,
            "discarded_at_step": discarded_at,
            "resume_steps": 5,
            "fresh_processes": len(MODES),
            "modes": comparisons,
            "scope": "Same runtime, CPU, data, one process and post-update boundary.",
        }
    finally:
        random.setstate(python_rng)
        torch.set_rng_state(torch_rng)
        torch.set_num_threads(threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--omit", choices=MODES, default="none")
    arguments = parser.parse_args()
    _child(arguments.checkpoint, arguments.result, arguments.omit)


if __name__ == "__main__":
    main()
