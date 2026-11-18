"""Two actual CPU/Gloo pipeline stages, with GPipe and synchronous 1F1B."""

import copy
import datetime
import json

import torch
import torch.distributed as dist
from example_data import make_batch, make_model
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe
from torch.distributed.pipelining.schedules import PipelineScheduleSingle


class RecordingStage(PipelineStage):
    """Count completed F/B calls, rather than measuring allocator memory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[str] = []
        self.live_states = 0
        self.peak_states = 0

    def forward_one_chunk(self, fwd_chunk_id: int, *args, **kwargs):
        output = super().forward_one_chunk(fwd_chunk_id, *args, **kwargs)
        self.events.append(f"F{fwd_chunk_id}")
        self.live_states += 1
        self.peak_states = max(self.peak_states, self.live_states)
        return output

    def backward_one_chunk(self, bwd_chunk_id: int, *args, **kwargs):
        result = super().backward_one_chunk(bwd_chunk_id, *args, **kwargs)
        self.events.append(f"B{bwd_chunk_id}")
        self.live_states -= 1
        return result


def run_case(schedule_class: type[PipelineScheduleSingle]) -> dict[str, object]:
    rank = dist.get_rank()
    dense = make_model()
    model = copy.deepcopy(dense[:2] if rank == 0 else dense[2:])
    x, target = make_batch()
    expected = dense(x)
    dense_loss = (expected - target).square().mean()
    dense_loss.backward()
    # Explicit static metadata avoids an extra shape-inference forward call.
    stage = RecordingStage(
        model,
        stage_index=rank,
        num_stages=2,
        device=torch.device("cpu"),
        input_args=torch.empty(
            2, 3 if rank == 0 else 5, dtype=torch.float64, requires_grad=rank > 0
        ),
        output_args=torch.empty(
            2, 5 if rank == 0 else 2, dtype=torch.float64, requires_grad=True
        ),
    )

    def loss_fn(output, chunk_target):
        return (output - chunk_target).square().sum() / target.numel()

    schedule = schedule_class(
        stage, n_microbatches=4, loss_fn=loss_fn, scale_grads=False
    )
    losses = []
    if rank == 0:
        output = schedule.step(x)
        if output is not None:
            raise RuntimeError("Only the last stage returns merged output")
    else:
        output = schedule.step(target=target, losses=losses)
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(sum(losses), dense_loss, rtol=1e-12, atol=1e-12)
    reference = list((dense[:2] if rank == 0 else dense[2:]).parameters())
    grad_error = 0.0
    for parameter, baseline in zip(model.parameters(), reference, strict=True):
        torch.testing.assert_close(
            parameter.grad, baseline.grad, rtol=1e-12, atol=1e-12
        )
        grad_error = max(
            grad_error, (parameter.grad - baseline.grad).abs().max().item()
        )
    # All microbatches have completed before either model's one optimizer update.
    torch.optim.SGD(model.parameters(), lr=0.05).step()
    torch.optim.SGD(dense.parameters(), lr=0.05).step()
    update_error = 0.0
    for parameter, baseline in zip(model.parameters(), reference, strict=True):
        torch.testing.assert_close(parameter, baseline, rtol=1e-12, atol=1e-12)
        update_error = max(update_error, (parameter - baseline).abs().max().item())
    if stage.live_states != 0:
        raise RuntimeError("Unmatched completed forward/backward calls")
    return {
        "schedule": schedule_class.__name__,
        "rank": rank,
        "backend": "gloo",
        "device": "cpu",
        "microbatches": 4,
        "microbatch_rows": 2,
        "loss_normalization": "microbatch SSE / 16 global output elements; scale_grads=False",
        "events": stage.events,
        "peak_unmatched_forward_calls": stage.peak_states,
        "final_unmatched_forward_calls": stage.live_states,
        "max_parameter_gradient_abs_error": grad_error,
        "max_parameter_update_abs_error": update_error,
        "output": output.detach().tolist() if output is not None else None,
        "loss": sum(item.item() for item in losses) if losses else None,
    }


def main() -> None:
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=45))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This fixed example needs exactly two ranks")
        for schedule in [ScheduleGPipe, Schedule1F1B]:
            result = run_case(schedule)
            print("PP_RESULT " + json.dumps(result), flush=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
