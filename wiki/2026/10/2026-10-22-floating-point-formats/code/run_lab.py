"""Print only bounded CPU numerical experiments, with strict JSON output."""

import json
import math
from dataclasses import asdict

import torch

from answers import compare_finite, roundtrip, sequential_sum


def main() -> None:
    formats = {}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        info = torch.finfo(dtype)
        stored = roundtrip([1.001, 70000.0, 1e-8], dtype).tolist()
        formats[str(dtype)] = {
            "eps": info.eps,
            "tiny_normal": info.tiny,
            "max": info.max,
            "stored": [
                value if math.isfinite(value) else str(value) for value in stored
            ],
        }
    values = torch.full((1000,), 0.1, dtype=torch.float16)
    reference = values.to(torch.float64).sum()
    low = sequential_sum(values, torch.float16)
    high = sequential_sum(values, torch.float32)
    report = compare_finite(high, reference, atol=0.0, rtol=0.0)
    big, small = torch.tensor(1e8), torch.tensor(1.0)
    output = {
        "torch": torch.__version__,
        "device": "cpu",
        "formats": formats,
        "addition_order": [
            float(((big + small) - big).item()),
            float(((big - big) + small).item()),
        ],
        "accumulation": {
            "ideal_decimal_sum": 100.0,
            "quantized_input_reference": float(reference.item()),
            "sequential_fp16": float(low.item()),
            "sequential_fp32": float(high.item()),
            "fp32_report": {**asdict(report), "passed": report.passed},
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
