"""Run isolated CPU torch.compile cases and archive FX graphs plus real results."""

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
from counting_backend import CountingBackend
from solutions import tensor_affine, tensor_branch
from torch._dynamo.exc import Unsupported
from workloads import affine, python_branch, split_affine


@dataclass(frozen=True)
class Case:
    name: str
    function: Callable[..., torch.Tensor]
    inputs: tuple[tuple[Any, ...], ...]
    dynamic: bool | None = False
    fullgraph: bool = True
    expect_unsupported: bool = False


def matrix(batch: int = 2, value: float = 1.0) -> torch.Tensor:
    return torch.full((batch, 3), value, dtype=torch.float32, device="cpu")


def scalar(value: float = 2.0) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32, device="cpu")


def run_case(case: Case) -> dict[str, Any]:
    # Each case starts a fresh cache in this single-purpose experiment process.
    torch.compiler.reset()
    backend = CountingBackend()
    compiled = torch.compile(
        case.function, backend=backend, dynamic=case.dynamic, fullgraph=case.fullgraph
    )
    calls = []
    failure = None
    try:
        for args in case.inputs:
            expected = case.function(*args)
            try:
                actual = compiled(*args)
            except Unsupported as error:
                if not case.expect_unsupported:
                    raise
                failure = {"type": type(error).__name__, "message": str(error)}
                break
            if case.expect_unsupported:
                raise AssertionError("Expected fullgraph rejection did not occur")
            torch.testing.assert_close(actual, expected)
            calls.append(
                {
                    "input_shapes": [
                        list(value.shape)
                        for value in args
                        if isinstance(value, torch.Tensor)
                    ],
                    "graphs_total": len(backend.records),
                    "output": actual.tolist(),
                }
            )
        return {
            "name": case.name,
            "dynamic": case.dynamic,
            "fullgraph": case.fullgraph,
            "calls": calls,
            "expected_rejection": failure,
            "graphs": [asdict(record) for record in backend.records],
            "numerical_checks": len(calls),
        }
    finally:
        torch.compiler.reset()


def build_cases() -> list[Case]:
    shapes = tuple((matrix(batch), 2.0) for batch in (2, 4, 5))
    python_scales = tuple((matrix(), value) for value in (2.0, 3.0, 2.0))
    tensor_scales = tuple((matrix(), scalar(value)) for value in (2.0, 3.0, 2.0))
    combined = tuple(
        (matrix(batch, value), scalar(scale))
        for batch, value, scale in ((2, 1, 2), (4, -1, 3), (5, 0, -2))
    )
    return [
        Case("same_metadata", affine, ((matrix(), 2.0),) * 3),
        Case("static_shapes", affine, shapes, dynamic=False),
        Case("automatic_shapes", affine, shapes, dynamic=None),
        Case("dynamic_shapes", affine, shapes, dynamic=True),
        Case("python_scalar", affine, python_scales),
        Case("tensor_scalar", tensor_affine, tensor_scales),
        Case("explicit_break", split_affine, ((matrix(),),) * 2, fullgraph=False),
        Case(
            "fullgraph_break_rejected",
            split_affine,
            ((matrix(),),),
            expect_unsupported=True,
        ),
        Case(
            "python_branch_rejected",
            python_branch,
            ((matrix(),),),
            expect_unsupported=True,
        ),
        Case("tensor_selection", tensor_branch, combined, dynamic=True),
        Case(
            "zero_one_dimensions",
            affine,
            tuple((matrix(batch), 2.0) for batch in (0, 1, 2, 4)),
            dynamic=True,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [run_case(case) for case in build_cases()]
    expected_counts = {
        "same_metadata": [1, 1, 1],
        "static_shapes": [1, 2, 3],
        "automatic_shapes": [1, 2, 2],
        "dynamic_shapes": [1, 1, 1],
        "python_scalar": [1, 2, 2],
        "tensor_scalar": [1, 1, 1],
        "explicit_break": [2, 2],
        "tensor_selection": [1, 1, 1],
        "zero_one_dimensions": [1, 2, 3, 3],
    }
    for result in results:
        counts = [call["graphs_total"] for call in result["calls"]]
        if result["name"] in expected_counts:
            assert counts == expected_counts[result["name"]], result
        else:
            assert result["expected_rejection"] is not None, result
            assert len(result["graphs"]) == 0, result
    code = Path(__file__).resolve().parent
    report = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "cuda_available": torch.cuda.is_available(),
        },
        "backend": "custom counting backend returning GraphModule.forward",
        "scope": "CPU Dynamo/FX capture, cache reuse, graph breaks and guards",
        "optimizing_codegen_executed": False,
        "code_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(code.glob("*.py"))
        },
        "cases": results,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
