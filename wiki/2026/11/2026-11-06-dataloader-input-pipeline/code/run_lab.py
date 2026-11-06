"""Print model inputs and three repetitions of each synthetic CPU pipeline."""

import json
import platform
import statistics
import sys
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
from answers import exposed_pause, pipeline_budget
from loader_probe import run_trial


def main() -> None:
    torch.set_num_threads(1)
    trials = {str(workers): [] for workers in (0, 2)}
    for _ in range(3):
        for workers in (0, 2):
            trial = run_trial(workers)
            expected = tuple(
                float(4 * index**2 + 12 * index + 14) for index in range(40)
            )
            if trial.sample_ids != tuple(range(40)) or trial.checksums != expected:
                raise RuntimeError("sample IDs or transformed payloads changed")
            trials[str(workers)].append(asdict(trial))
    summaries = {}
    for workers, reports in trials.items():
        summaries[workers] = {
            "median_iterator_creation_ms": statistics.median(
                report["iterator_creation_ms"] for report in reports
            ),
            "median_first_next_ms": statistics.median(
                report["batches"][0]["next_wait_ms"] for report in reports
            ),
            "median_of_steady_next_wait_medians_ms": statistics.median(
                report["steady_next_wait_median_ms"] for report in reports
            ),
            "checksum_sum": sum(reports[0]["checksums"]),
            "sample_count": len(reports[0]["sample_ids"]),
            "repetitions": len(reports),
        }
    arguments = {
        "storage_ms": 4,
        "cpu_ms": 16,
        "workers": 2,
        "transfer_ms": 3,
        "compute_ms": 10,
        "prefetch_factor": 2,
        "batch_bytes": 64 * 1024**2,
    }
    report = {
        "observed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "torch_build_git": torch.version.git_version,
            "main_cpu_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "fixture": {
            "samples": 40,
            "batch_size": 4,
            "values_per_record": 4,
            "transform": "FP64 sum of squares",
            "source_pause_per_sample_ms": 5,
            "consumer_pause_per_batch_ms": 20,
            "warmup_batches_excluded": 2,
            "worker_counts": [0, 2],
            "multiprocessing_context": "spawn",
            "prefetch_factor_for_workers": 2,
            "pin_memory": False,
            "persistent_workers": False,
            "in_order": True,
            "repetitions_per_case": 3,
            "cache_state": "tiny JSON files created just before each iteration; OS caches uncontrolled",
        },
        "capacity_model_assumptions": {
            "units": "milliseconds and bytes; supplied model inputs",
            "same_stream": asdict(pipeline_budget(**arguments)),
            "independent_copy_and_compute": asdict(
                pipeline_budget(**arguments, allow_device_overlap=True)
            ),
            "40ms_supply_pause_2_ready_13ms_consumption_exposes_ms": exposed_pause(
                40, 2, 13
            ),
        },
        "cpu_observations": summaries,
        "raw_trials": trials,
        "scope": (
            "Real CPU DataLoader, JSON parsing, tensor transform and spawn worker behavior. "
            "Artificial source/consumer sleeps dominate this tiny fixture. "
            "First-next timing and iterator creation separate from steady-state waits. "
            "No GPU/H2D measurements or cold-storage throughput measurements."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
