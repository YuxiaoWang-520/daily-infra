"""Optional CUDA capture/replay correctness demo with no performance measurement."""

import json

import torch


def check_output(actual: torch.Tensor, input_value: int) -> None:
    """Check the fixed CPU float32 fixture after CUDA-to-CPU transfer."""
    expected = torch.full(
        (2, 3), 2 * input_value + 1.0, dtype=torch.float32, device="cpu"
    )
    torch.testing.assert_close(actual, expected)


def run_demo() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {
            "status": "cuda_unavailable",
            "cuda_executed": False,
            "torch": torch.__version__,
            "capture_replay_validation": "pending CUDA device",
        }

    static_input = torch.zeros((2, 3), device="cuda", dtype=torch.float32)
    static_output = torch.empty_like(static_input)
    input_pointer = static_input.data_ptr()
    output_pointer = static_output.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(3):
            torch.mul(static_input, 2, out=static_output)
            static_output.add_(1)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), torch.no_grad():
        torch.mul(static_input, 2, out=static_output)
        static_output.add_(1)
    # Replay runs on the caller stream, ordered after capture-stream preparation.
    torch.cuda.current_stream().wait_stream(stream)

    results = []
    for value in (3, 4):
        source = torch.full_like(static_input, value)
        static_input.copy_(source)
        graph.replay()
        saved = static_output.clone()
        # .cpu() waits for these outputs after capture has ended.
        actual = saved.cpu()
        check_output(actual, value)
        results.append(actual.tolist())
    torch.cuda.synchronize()
    assert static_input.data_ptr() == input_pointer
    assert static_output.data_ptr() == output_pointer
    return {
        "status": "passed",
        "cuda_executed": True,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "outputs": results,
        "input_and_output_addresses_stable": True,
        "performance_measurement": "pending separate benchmark",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))
