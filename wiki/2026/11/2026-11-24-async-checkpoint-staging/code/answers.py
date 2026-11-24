"""Integer-millisecond model: one in-flight save, blocking staging, final drain."""

from typing import Any


def simulate_checkpoints(
    num_steps: int,
    compute_ms: int,
    interval_steps: int,
    stage_ms: int,
    write_ms: int,
) -> dict[str, Any]:
    """Wait for the previous upload before staging the next snapshot.

    Compute and staging share the training lane. Writes use a separate lane.
    Durations are fixed assumptions, not performance predictions. No implicit
    checkpoint is taken at a final step that is not a multiple of interval_steps.
    """
    values = {
        "num_steps": num_steps,
        "compute_ms": compute_ms,
        "interval_steps": interval_steps,
        "stage_ms": stage_ms,
        "write_ms": write_ms,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        minimum = 1 if name in {"compute_ms", "interval_steps"} else 0
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    time_ms = upload_end_ms = previous_step = backpressure_ms = 0
    checkpoints = []
    for step in range(interval_steps, num_steps + 1, interval_steps):
        time_ms += (step - previous_step) * compute_ms
        wait_ms = max(0, upload_end_ms - time_ms)
        backpressure_ms += wait_ms
        time_ms += wait_ms
        stage_start_ms = time_ms
        time_ms += stage_ms
        upload_end_ms = time_ms + write_ms
        checkpoints.append(
            {
                "step": step,
                "backpressure_ms": wait_ms,
                "stage_start_ms": stage_start_ms,
                "stage_end_ms": time_ms,
                "write_start_ms": time_ms,
                "write_end_ms": upload_end_ms,
            }
        )
        previous_step = step
    time_ms += (num_steps - previous_step) * compute_ms
    all_saved_ms = max(time_ms, upload_end_ms)
    return {
        "model": "synthetic fixed durations; one in-flight save; blocking staging",
        "training_finished_ms": time_ms,
        "all_saved_ms": all_saved_ms,
        "backpressure_ms": backpressure_ms,
        "final_drain_ms": all_saved_ms - time_ms,
        "sync_total_ms": num_steps * compute_ms
        + len(checkpoints) * (stage_ms + write_ms),
        "checkpoints": checkpoints,
    }
