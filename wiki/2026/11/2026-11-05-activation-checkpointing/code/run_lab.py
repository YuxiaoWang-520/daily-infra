"""Print reproducible model inputs and CPU correctness observations as JSON."""

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
from answers import choose_plan, estimate_uniform
from torch_probe import TrialResult, run_trial


def _max_error(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> float:
    return max(
        float((left - right).abs().max())
        for left, right in zip(actual, expected, strict=True)
    )


def _summary(result: TrialResult, baseline: TrialResult) -> dict[str, object]:
    return {
        "loss": result.loss,
        "output_max_abs_error": float((result.output - baseline.output).abs().max()),
        "input_gradient_max_abs_error": float(
            (result.input_gradient - baseline.input_gradient).abs().max()
        ),
        "parameter_gradient_max_abs_error": _max_error(
            result.parameter_gradients, baseline.parameter_gradients
        ),
        "updated_parameter_max_abs_error": _max_error(
            result.updated_parameters, baseline.updated_parameters
        ),
        "same_next_cpu_random_sample": torch.equal(
            result.next_random, baseline.next_random
        ),
        "saved_forward_activation_storage_bytes": result.saved_activation_bytes,
        "block_entry_calls": result.forward_entry_calls,
    }


def main() -> None:
    torch.set_num_threads(1)
    plain = run_trial(False)
    preserved = run_trial(True)
    unpreserved = run_trial(True, preserve_rng_state=False)
    selected = choose_plan(16, 1024**2, 10 * 1024**2, 16, 32, 0.2)
    report = {
        "observed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "device": "cpu",
            "dtype": "float64",
            "cpu_threads": torch.get_num_threads(),
            "cpu_rng_state_bytes": torch.get_rng_state().numel()
            * torch.get_rng_state().element_size(),
            "cuda_available": torch.cuda.is_available(),
        },
        "fixture": {
            "blocks": 4,
            "batch_size": 2,
            "width": 8,
            "block": "Linear(8, 8) -> sin -> Dropout(p=0.4)",
            "loss": "output.square().mean()",
            "optimizer": "SGD(lr=0.01)",
            "seed": 2026,
            "mode": "eager, use_reentrant=False, default early stop, no compile",
        },
        "synthetic_uniform_model": {
            "units": "bytes and microseconds; assumed inputs, not measured time",
            "plans": [
                asdict(estimate_uniform(16, size, 1024**2, 16, 32, 0.2))
                for size in (1, 2, 4, 8, 16)
            ],
            "selected_for_10_MiB_activation_budget": asdict(selected)
            if selected
            else None,
        },
        "cpu_correctness_observations": {
            "plain": _summary(plain, plain),
            "checkpoint_preserved": _summary(preserved, plain),
            "checkpoint_unpreserved": _summary(unpreserved, plain),
        },
        "measurement_scope": (
            "unique non-parameter/non-input storage bytes observed by OUTER model-forward "
            "saved-tensor hooks; excludes loss, RNG snapshots, metadata and replay workspaces; "
            "not live peak memory, allocated/reserved bytes, or a throughput benchmark"
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
