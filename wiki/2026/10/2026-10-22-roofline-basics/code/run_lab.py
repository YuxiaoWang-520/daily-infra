"""Print analytical estimates; every time below is synthetic, not measured."""

import json
from dataclasses import asdict

from answers import effective_rates, estimate, gemm_work, vector_add_work


def main() -> None:
    cases = {
        "vector_add_fp32": vector_add_work(2**20, 4),
        "gemm_128_fp32": gemm_work(128, 128, 128, 4),
        "gemm_1024_fp32": gemm_work(1024, 1024, 1024, 4),
    }
    result = {
        "scope": "analytical only; illustrative hardware; no GPU measurements",
        "assumptions": {"peak_tflops": 100, "bandwidth_gbs": 1000},
        "cases": {},
    }
    for name, work in cases.items():
        bound = estimate(work, 100, 1000)
        result["cases"][name] = {"work": asdict(work), "bound": asdict(bound)}
    work = cases["gemm_1024_fp32"]
    lower = estimate(work, 100, 1000).lower_seconds
    result["synthetic_twice_bound"] = effective_rates(work, 2 * lower)
    assert result["synthetic_twice_bound"]["tflops"] == 50.0
    assert result["cases"]["vector_add_fp32"]["bound"]["limiter"] == "memory"
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
