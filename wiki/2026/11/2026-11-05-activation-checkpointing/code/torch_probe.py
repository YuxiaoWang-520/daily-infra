"""CPU eager experiment: saved forward storages, replay, gradients and RNG."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TrialResult:
    output: torch.Tensor
    loss: float
    input_gradient: torch.Tensor
    parameter_gradients: tuple[torch.Tensor, ...]
    updated_parameters: tuple[torch.Tensor, ...]
    next_random: torch.Tensor
    saved_activation_bytes: int
    forward_entry_calls: tuple[int, ...]


class _Block(nn.Module):
    def __init__(
        self, width: int, dropout_p: float, index: int, calls: list[int]
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, dtype=torch.float64, device="cpu")
        self.dropout_p = dropout_p
        self.index = index
        self.calls = calls

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Diagnostic only: no tensor computation depends on this mutable list.
        # An entry does not imply the whole function finished (early stop).
        self.calls.append(self.index)
        return functional.dropout(
            torch.sin(self.linear(value)), p=self.dropout_p, training=True
        )


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _gradient(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.grad is None:
        raise RuntimeError("the experiment expected a gradient for every leaf tensor")
    return tensor.grad.detach().clone()


def run_trial(
    checkpointed: bool,
    preserve_rng_state: bool = True,
    dropout_p: float = 0.4,
    blocks: int = 4,
    width: int = 8,
    batch_size: int = 2,
    seed: int = 2026,
) -> TrialResult:
    """Run one seeded FP64 CPU SGD step with a fresh identical model.

    Count unique nonzero storages seen by OUTER saved-tensor pack hooks during
    model forward, excluding the input and parameter storages. The loss is
    outside the counting context. Metadata retains no tensors; the pack hook
    returns a detached alias, so it does not clone or inflate storage payload.
    This is neither live peak memory nor allocator memory, and is specific to
    this plain eager checkpoint experiment (not selective checkpointing).
    """
    for name, value in (
        ("checkpointed", checkpointed),
        ("preserve_rng_state", preserve_rng_state),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a bool")  # noqa: TRY004 -- exercise API
    for name, value in (
        ("blocks", blocks),
        ("width", width),
        ("batch_size", batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(dropout_p, bool)
        or not isinstance(dropout_p, (int, float))
        or not 0 <= dropout_p < 1
        or not math.isfinite(dropout_p)
    ):
        raise ValueError("dropout_p must be finite and in [0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")

    # Local generator seeding avoids changing CUDA/MPS generator state.
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.set_rng_state(generator.get_state())
        calls: list[int] = []
        model = nn.ModuleList(
            _Block(width, dropout_p, index, calls) for index in range(blocks)
        )
        inputs = (
            torch.linspace(
                -0.9, 0.9, batch_size * width, dtype=torch.float64, device="cpu"
            )
            .reshape(batch_size, width)
            .requires_grad_()
        )
        parameters = tuple(model.parameters())
        excluded = {_storage_key(tensor) for tensor in (*parameters, inputs)}
        saved: dict[tuple[str, int, int], int] = {}

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            key = _storage_key(tensor)
            if key not in excluded and key[2] > 0:
                saved[key] = key[2]
            return tensor.detach()

        def unpack(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        optimizer = torch.optim.SGD(parameters, lr=0.01)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            value = inputs
            for block in model:
                value = (
                    checkpoint(
                        block,
                        value,
                        use_reentrant=False,
                        preserve_rng_state=preserve_rng_state,
                    )
                    if checkpointed
                    else block(value)
                )
        loss = value.square().mean()
        loss.backward()
        gradients = tuple(_gradient(parameter) for parameter in parameters)
        optimizer.step()
        return TrialResult(
            output=value.detach().clone(),
            loss=float(loss.detach()),
            input_gradient=_gradient(inputs),
            parameter_gradients=gradients,
            updated_parameters=tuple(
                parameter.detach().clone() for parameter in parameters
            ),
            next_random=torch.rand(8, dtype=torch.float64, device="cpu"),
            saved_activation_bytes=sum(saved.values()),
            forward_entry_calls=tuple(calls),
        )
