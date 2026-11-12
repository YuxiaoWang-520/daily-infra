"""Run CPU storage observations and print analytical queue estimates."""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from launch_model import estimate_replays
from static_buffers import StaticAffine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    buffers = StaticAffine(source, 2, 1)
    pointers = (buffers.input_buffer.data_ptr(), buffers.output_buffer.data_ptr())
    first = buffers.replay()
    saved = first.clone()
    source = torch.full_like(source, 4)
    after_rebind = buffers.replay().clone()
    buffers.copy_input(source)
    second = buffers.replay()
    torch.testing.assert_close(saved, torch.tensor([[1.0, 3.0, 5.0], [7.0, 9.0, 11.0]]))
    torch.testing.assert_close(after_rebind, saved)
    torch.testing.assert_close(second, torch.full_like(source, 9))
    assert pointers == (
        buffers.input_buffer.data_ptr(),
        buffers.output_buffer.data_ptr(),
    )
    assert first is second

    code_dir = Path(__file__).resolve().parent
    result = {
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
        "code_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(code_dir.glob("*.py"))
        },
        "modeled_costs_us": {
            str(count): asdict(estimate_replays([1] * 10, 3, 4, 10000, count))
            for count in (1, 100, 500, 501, 1000)
        },
        "compute_bound_model_us": asdict(
            estimate_replays([100] * 10, 3, 4, 10000, 1000)
        ),
        "cpu_observations": {
            "saved_first_output": saved.tolist(),
            "after_source_rebinding": after_rebind.tolist(),
            "after_input_copy": second.tolist(),
            "borrowed_first_output_after_second_run": first.tolist(),
            "input_and_output_addresses_stable": True,
            "output_object_reused": first is second,
        },
        "validation_scope": "CPU tensor storage and analytical timing model",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
