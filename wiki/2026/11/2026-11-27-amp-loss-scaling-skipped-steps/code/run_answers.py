"""Emit strict JSON evidence for the CPU examples; never benchmark a GPU."""

import argparse
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from answers import quantized_gradient, unscale_accumulated, unscale_and_clip
from runtime_demo import (
    accumulation_probe,
    autocast_probe,
    clipping_probe,
    protocol_probes,
    training_trace,
)


def json_safe(value: Any) -> Any:
    """Represent intentional nonfinite probe results as strings, not invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    faults = [False, False, True, False, False]
    baseline = training_trace(faults)
    resumed = training_trace(faults, restore_after=3)
    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "torch_build_commit": torch.version.git_version,
            "cuda_available": torch.cuda.is_available(),
        },
        "scope": {
            "device": "cpu",
            "protocol_math": "float32 scalar with synthetic Inf injection",
            "autocast_smoke": "CPU BF16 linear and FP32 MSE loss",
            "gpu_underflow_or_performance_measured": False,
            "restore": "in-memory state_dict at completed update boundaries",
        },
        "quantization": {
            "fp16_small": asdict(quantized_gradient(1e-8, 1024)),
            "fp16_overflow": asdict(quantized_gradient(64, 1024)),
            "bf16_small": asdict(quantized_gradient(1e-8, 1, dtype=torch.bfloat16)),
        },
        "analytic_clipping": {
            "correct": unscale_and_clip([24, 32], 8, 1),
            "wrong": [g / 8 for g in unscale_and_clip([24, 32], 1, 1)],
        },
        "analytic_accumulation": {
            "constant": unscale_accumulated([1, 1], [8, 8], 8),
            "mixed": unscale_accumulated([1, 1], [8, 16], 16),
        },
        "trace": baseline,
        "resume_after_3_equal": resumed == baseline,
        "real_clipping": clipping_probe(),
        "real_accumulation": accumulation_probe(),
        "protocol": protocol_probes(),
        "autocast": autocast_probe(),
    }
    serialized = (
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects previous evidence from accidental replacement.
    with (args.output_dir / "results.json").open("x", encoding="utf-8") as result_file:
        result_file.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
