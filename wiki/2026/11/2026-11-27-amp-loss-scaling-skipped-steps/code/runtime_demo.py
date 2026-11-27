"""Real CPU AMP protocols; no GPU numeric or performance claim is made."""

import copy
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from answers import finite_number


class CountingSGD(torch.optim.SGD):
    """Count applied calls to this ordinary optimizer, regardless of return value."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        lr: float = 0.1,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr=lr, momentum=momentum)
        self.applied_steps = 0

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        result = super().step(closure)
        self.applied_steps += 1
        return result


class TrainingHarness:
    """A deterministic scalar training loop with a documented scheduler policy."""

    def __init__(self, init_scale: float) -> None:
        self.parameter = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device="cpu")
        )
        self.optimizer = CountingSGD([self.parameter], momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda _: 1.0
        )
        self.scaler = torch.amp.GradScaler(
            "cpu", init_scale=init_scale, growth_interval=2
        )
        self.data_batches = 0
        self.scheduler_steps = 0

    def momentum(self) -> float | None:
        buffer = self.optimizer.state.get(self.parameter, {}).get("momentum_buffer")
        return None if buffer is None else buffer.item()

    def attempt(self, inject_nonfinite: bool) -> dict[str, Any]:
        self.optimizer.zero_grad(set_to_none=True)
        scale_before = self.scaler.get_scale()
        parameter_before, momentum_before = self.parameter.item(), self.momentum()
        # Float32 math isolates the scaler protocol. Inf below is synthetic.
        loss = 0.5 * self.parameter.square()
        self.scaler.scale(loss).backward()
        if inject_nonfinite:
            self.parameter.grad.fill_(math.inf)
        scaled_gradient = self.parameter.grad.item()
        self.scaler.unscale_(self.optimizer)
        unscaled_gradient = self.parameter.grad.item()
        previous_steps = self.optimizer.applied_steps
        result = self.scaler.step(self.optimizer)
        called = self.optimizer.applied_steps > previous_steps
        if called:
            # Application policy: scheduler advances only after an applied update.
            self.scheduler.step()
            self.scheduler_steps += 1
        self.scaler.update()
        self.data_batches += 1
        record = {
            "data_batches": self.data_batches,
            "applied_updates": self.optimizer.applied_steps,
            "scheduler_steps": self.scheduler_steps,
            "injected_nonfinite": inject_nonfinite,
            "scale_before": scale_before,
            "scale_after": self.scaler.get_scale(),
            "growth_tracker": self.scaler.state_dict()["_growth_tracker"],
            "scaled_gradient": scaled_gradient,
            "unscaled_gradient": unscaled_gradient,
            "optimizer_step_called": called,
            "step_return_is_none": result is None,
            "parameter_before": parameter_before,
            "parameter_after": self.parameter.item(),
            "momentum_before": momentum_before,
            "momentum_after": self.momentum(),
        }
        self.optimizer.zero_grad(set_to_none=True)
        return record

    def checkpoint(self) -> dict[str, Any]:
        """In-memory snapshot at an update boundary, including app counters."""
        return copy.deepcopy(
            {
                "parameter": self.parameter.detach().clone(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "data_batches": self.data_batches,
                "applied_updates": self.optimizer.applied_steps,
                "scheduler_steps": self.scheduler_steps,
            }
        )

    @classmethod
    def restore(cls, state: dict[str, Any]) -> "TrainingHarness":
        restored = cls(state["scaler"]["scale"])
        with torch.no_grad():
            restored.parameter.copy_(state["parameter"])
        restored.optimizer.load_state_dict(state["optimizer"])
        restored.scheduler.load_state_dict(state["scheduler"])
        restored.scaler.load_state_dict(state["scaler"])
        restored.data_batches = state["data_batches"]
        restored.optimizer.applied_steps = state["applied_updates"]
        restored.scheduler_steps = state["scheduler_steps"]
        return restored


def training_trace(
    faults: Sequence[bool], *, restore_after: int | None = None, init_scale: float = 8.0
) -> dict[str, Any]:
    """Run CPU GradScaler; optionally reload public state after N attempts."""
    faults = list(faults)
    if any(type(fault) is not bool for fault in faults):
        raise TypeError("faults must contain only booleans")
    if restore_after is not None:
        if isinstance(restore_after, bool) or not isinstance(restore_after, int):
            raise TypeError("restore_after must be an integer or None")
        if not 0 <= restore_after <= len(faults):
            raise ValueError("restore_after must be a valid boundary")
    init_scale = finite_number(init_scale, "init_scale", positive=True)
    represented_scale = torch.tensor(
        init_scale, dtype=torch.float32, device="cpu"
    ).item()
    if not math.isfinite(represented_scale) or represented_scale <= 0:
        raise ValueError("init_scale must be positive and finite in float32")
    harness = TrainingHarness(init_scale)
    records = []
    if restore_after == 0:
        harness = TrainingHarness.restore(harness.checkpoint())
    for cursor, fault in enumerate(faults, 1):
        records.append(harness.attempt(fault))
        if cursor == restore_after:
            harness = TrainingHarness.restore(harness.checkpoint())
    return {
        "records": records,
        "data_batches": harness.data_batches,
        "applied_updates": harness.optimizer.applied_steps,
        "scheduler_steps": harness.scheduler_steps,
        "scheduler_last_epoch": harness.scheduler.last_epoch,
        "final_scaler": harness.scaler.state_dict(),
        "scaler_enabled": harness.scaler.is_enabled(),
    }


def clipping_probe() -> dict[str, list[float]]:
    """Compare actual clip_grad_norm_ before and after unscale_."""
    results = {}
    for label in ("correct", "wrong"):
        parameter = nn.Parameter(
            torch.tensor([1.0, 1.0], dtype=torch.float32, device="cpu")
        )
        optimizer = CountingSGD([parameter])
        scaler = torch.amp.GradScaler("cpu", init_scale=8)
        scaler.scale(
            (
                parameter * torch.tensor([3.0, 4.0], dtype=torch.float32, device="cpu")
            ).sum()
        ).backward()
        if label == "correct":
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([parameter], 1.0)
        else:
            nn.utils.clip_grad_norm_([parameter], 1.0)
            scaler.unscale_(optimizer)
        results[label] = parameter.grad.tolist()
        scaler.step(optimizer)
        scaler.update()
    return results


def accumulation_probe() -> dict[str, float]:
    """Show constant scaling and an intentionally invalid mid-window change."""
    results = {}
    for changed in (False, True):
        parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device="cpu"))
        optimizer = CountingSGD([parameter])
        scaler = torch.amp.GradScaler("cpu", init_scale=8)
        scaler.scale(parameter).backward()
        if changed:
            # This public manual value violates the recipe at this boundary.
            scaler.update(new_scale=16.0)
        scaler.scale(parameter).backward()
        scaler.unscale_(optimizer)
        label = "changed_scale_gradient" if changed else "fixed_scale_gradient"
        results[label] = parameter.grad.item()
        scaler.step(optimizer)
        scaler.update()
    return results


def _finite_case(
    *, dtype: torch.dtype = torch.float32
) -> tuple[nn.Parameter, CountingSGD, torch.amp.GradScaler]:
    parameter = nn.Parameter(torch.tensor(1.0, dtype=dtype, device="cpu"))
    optimizer = CountingSGD([parameter])
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    scaler.scale(parameter.square()).backward()
    return parameter, optimizer, scaler


def protocol_probes() -> dict[str, Any]:
    """Ordinary optimizer state checks, including cached finite flags."""
    results: dict[str, Any] = {}
    _, optimizer, scaler = _finite_case()
    scaler.unscale_(optimizer)
    try:
        scaler.unscale_(optimizer)
    except RuntimeError as error:
        results["double_unscale_error"] = type(error).__name__
    else:
        raise RuntimeError("expected double-unscale rejection")

    _, optimizer, scaler = _finite_case()
    scaler.step(optimizer)
    try:
        scaler.step(optimizer)
    except RuntimeError as error:
        results["double_step_error"] = type(error).__name__
    else:
        raise RuntimeError("expected double-step rejection")

    _, optimizer, scaler = _finite_case(dtype=torch.float16)
    try:
        scaler.unscale_(optimizer)
    except ValueError as error:
        results["half_parameter_unscale_error"] = type(error).__name__
    else:
        raise RuntimeError("expected ordinary FP16-gradient rejection")

    parameters = [
        nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device="cpu"))
        for _ in range(2)
    ]
    optimizers = [CountingSGD([parameter]) for parameter in parameters]
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    scaler.scale(2 * parameters[0] + 3 * parameters[1]).backward()
    parameters[1].grad.fill_(math.inf)
    for optimizer in optimizers:
        scaler.step(optimizer)
    scaler.update()
    results["multiple_optimizers"] = {
        "step_calls": [optimizer.applied_steps for optimizer in optimizers],
        "parameters_after": [parameter.item() for parameter in parameters],
        "scale_after": scaler.get_scale(),
    }

    parameter, optimizer, scaler = _finite_case()
    scaler.unscale_(optimizer)
    # Deliberately corrupt a checked gradient. Ordinary step uses cached flags.
    parameter.grad.fill_(math.inf)
    scaler.step(optimizer)
    scaler.update()
    results["mutation_after_unscale"] = {
        "step_calls": optimizer.applied_steps,
        "parameter_after": parameter.item(),
    }
    return results


def autocast_probe() -> dict[str, Any]:
    """Fixed-data BF16 CPU forward, FP32 loss and FP32 parameter gradient."""
    parameter = nn.Parameter(
        torch.tensor([[0.25, -0.5], [1.0, 0.25]], dtype=torch.float32, device="cpu")
    )
    inputs = torch.tensor([[0.1, 0.2], [0.3, -0.4]], dtype=torch.float32, device="cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = F.linear(inputs, parameter)
        loss = F.mse_loss(output, torch.zeros_like(output))
    loss.backward()
    return {
        "output_dtype": str(output.dtype),
        "loss_dtype": str(loss.dtype),
        "parameter_dtype": str(parameter.dtype),
        "gradient_dtype": str(parameter.grad.dtype),
        "parameter_device": str(parameter.device),
        "loss": loss.item(),
    }
