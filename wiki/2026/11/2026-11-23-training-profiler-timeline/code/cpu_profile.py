"""A real CPU-only scheduled trace of a tiny deterministic training workload."""

import hashlib
import json
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from answers import complete_events, project_cpu_trace


def _initial_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.arange(128, dtype=torch.float64).reshape(16, 8) / 128
    target = torch.arange(64, dtype=torch.float64).reshape(16, 4) / 64
    weight = (
        torch.arange(32, dtype=torch.float64).reshape(8, 4) / 256
    ).requires_grad_()
    return x, target, weight


def _train_step(
    x: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, *, annotate: bool
) -> float:
    def phase(name: str):
        return (
            torch.profiler.record_function("phase::" + name)
            if annotate
            else nullcontext()
        )

    with phase("input"):
        batch = (
            x.clone()
        )  # Synthetic CPU input preparation, not a DataLoader/I/O benchmark.
    with phase("forward"):
        prediction = batch @ weight
    with phase("loss"):
        loss = (prediction - target).square().mean()
    with phase("backward"):
        loss.backward()
    with phase("update"), torch.no_grad():
        weight.add_(weight.grad, alpha=-0.01)
        weight.grad = None
    return loss.item()


def run_cpu_profile(output_dir: Path) -> dict[str, Any]:
    """Run four training steps; export only scheduled active steps 2 and 3.

    Wall time includes profiler transitions/export: it is not a clean training
    throughput benchmark. No CUDA activity is requested or inferred.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    callbacks: list[int] = []
    samples: list[dict[str, Any]] = []
    tables: list[str] = []
    try:
        torch.set_num_threads(1)
        x, target, reference = _initial_state()
        reference_losses = [
            _train_step(x, target, reference, annotate=False) for _ in range(4)
        ]
        x, target, weight = _initial_state()
        with TemporaryDirectory(prefix="public-profiler-toy-") as temporary:
            raw_path = Path(temporary) / "raw.json"

            def trace_ready(prof: torch.profiler.profile) -> None:
                callbacks.append(prof.step_num)
                prof.export_chrome_trace(str(raw_path))
                for event in prof.events():
                    if event.name == "phase::forward":
                        samples.append(
                            {
                                "cpu_total_us": event.cpu_time_total,
                                "self_cpu_us": event.self_cpu_time_total,
                                "cpu_children": len(event.cpu_children),
                            }
                        )
                tables.append(
                    prof.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=12
                    )
                )

            schedule = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)
            start = time.perf_counter()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=schedule,
                on_trace_ready=trace_ready,
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
                with_flops=False,
            ) as prof:
                losses = []
                for _ in range(4):
                    losses.append(_train_step(x, target, weight, annotate=True))
                    prof.step()
            wall_seconds = time.perf_counter() - start
            raw_bytes = raw_path.read_bytes()
            raw = json.loads(raw_bytes)
            projected, projection = project_cpu_trace(raw)
        trace_path = output_dir / "cpu_trace.json"
        trace_path.write_text(
            json.dumps(projected, ensure_ascii=False, indent=2) + "\n"
        )
        (output_dir / "cpu_table.txt").write_text("\n".join(tables) + "\n")
        events = complete_events(projected)
        steps = sorted(
            int(e.name.split("#")[-1])
            for e in events
            if e.name.startswith("ProfilerStep#")
        )
        counts = Counter(e.name for e in events if e.name.startswith("phase::"))
        error = (weight.detach() - reference.detach()).abs().max().item()
        result = {
            "device": "CPU",
            "gpu_executed": False,
            "dtype": "float64",
            "threads": 1,
            "input_shape": [16, 8],
            "weight_shape": [8, 4],
            "steps_run": 4,
            "schedule": {"wait": 1, "warmup": 1, "active": 2, "repeat": 1},
            "profiled_steps": steps,
            "callback_step_numbers": callbacks,
            "phase_counts": dict(sorted(counts.items())),
            "forward_samples": samples,
            "losses": losses,
            "losses_match": losses == reference_losses,
            "weight_max_abs_error": error,
            "capture_wall_seconds": wall_seconds,
            "wall_scope": "four annotated steps + profiler transitions + export; not clean throughput",
            "raw_trace_sha256_before_projection": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_trace_archived": False,
            "projection": projection,
        }
        if steps != [2, 3] or callbacks != [4] or len(samples) != 2:
            raise RuntimeError("unexpected active window or callback count")
        if error != 0 or not result["losses_match"]:
            raise RuntimeError("profiling changed the toy workload's numerical result")
        if any(
            counts["phase::" + phase] != 2
            for phase in ["input", "forward", "loss", "backward", "update"]
        ):
            raise RuntimeError("missing phase annotations in active window")
        return result
    finally:
        torch.set_num_threads(previous_threads)
