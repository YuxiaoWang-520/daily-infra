"""Emit actual CPU algebra and explicitly hypothetical scheduling budgets."""

import copy
import json
import platform

import torch
from answers import microbatch_backward, simulate_pipeline
from example_data import make_batch, make_model


def main() -> None:
    x, target = make_batch()
    model = make_model()
    dense = copy.deepcopy(model)
    expected = dense(x)
    full_loss = (expected - target).square().mean()
    full_loss.backward()
    result = microbatch_backward(model, x, target, [1, 3, 4])
    gradient_error = max(
        (a.grad - b.grad).abs().max().item()
        for a, b in zip(model.parameters(), dense.parameters(), strict=True)
    )
    hypothetical = []
    # Each tick is 0.1 ms here; these costs are assumptions, not measurements.
    for size, f_ticks, b_ticks in [(2, 13, 26), (4, 20, 40), (8, 36, 74)]:
        microbatches = 32 // size
        schedule = simulate_pipeline(
            4,
            microbatches,
            "1f1b",
            forward_ticks=[f_ticks] * 4,
            backward_ticks=[b_ticks] * 4,
        )
        hypothetical.append(
            {
                "microbatch_rows": size,
                "microbatches": microbatches,
                "hypothetical_stage_forward_ms": f_ticks / 10,
                "hypothetical_stage_backward_ms": b_ticks / 10,
                "estimated_step_ms": schedule["makespan_ticks"] / 10,
                "compute_utilization": schedule["compute_utilization"],
                "saved_activation_mib_per_microbatch_assumption": 16 * size,
                "gpipe_peak_saved_activation_mib_per_stage": [microbatches * 16 * size]
                * 4,
                "one_f_one_b_peak_saved_activation_mib_per_stage": [
                    n * 16 * size for n in schedule["peak_saved_microbatches"]
                ],
                "payload_bytes_per_microbatch_per_direction": size * 2048 * 8192 * 2,
                "payload_bytes_per_boundary_per_step_per_direction": 32
                * 2048
                * 8192
                * 2,
                "payload_messages_per_boundary_both_directions": 2 * microbatches,
            }
        )
    output = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_build_commit": torch.version.git_version,
            "os": platform.system(),
            "architecture": platform.machine(),
            "cuda_available": torch.cuda.is_available(),
            "gloo_available": torch.distributed.is_gloo_available(),
        },
        "unit_tick_schedules": {
            name: simulate_pipeline(3, 4, name) for name in ["gpipe", "1f1b"]
        },
        "unequal_microbatch_cpu_validation": {
            "chunk_sizes": [1, 3, 4],
            "output": result["output"].tolist(),
            "loss": result["loss"],
            "full_batch_loss": full_loss.item(),
            "max_parameter_gradient_abs_error": gradient_error,
        },
        "hypothetical_fixed_global_batch_64_dp_2": hypothetical,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
