"""Optional real-CUDA recipe; this path has NOT been run on a GPU here."""

import json

import torch

from answers import benchmark_cuda


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        print(
            json.dumps({"status": "SKIP", "reason": "CUDA unavailable; no GPU timing"})
        )
        return
    torch.manual_seed(0)
    x = torch.randn(2**20, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    out = torch.empty_like(x)

    def operation() -> None:
        torch.add(x, y, out=out)

    result = benchmark_cuda(operation)
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)
    print(
        json.dumps(
            {
                "status": "measured",
                "scope": "current-stream batched hot-input add, preallocated output",
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "stride": list(x.stride()),
                "warmup": 5,
                "iterations_per_sample": 20,
                "samples": 5,
                "result": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
