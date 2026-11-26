"""Actual float64 CPU SDPA; no dropout, varlen kernel, distributed group or benchmark."""

import torch
from answers import loss_sum_and_count, pack_batch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class ToyAttentionLM(nn.Module):
    """One head with learned absolute positions, residual and vocabulary projection."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(31)

        def parameter(*shape: int) -> nn.Parameter:
            return nn.Parameter(
                torch.randn(
                    shape, generator=generator, dtype=torch.float64, device="cpu"
                )
                * 0.2
            )

        self.token_embedding = parameter(8, 4)
        self.position_embedding = parameter(16, 4)
        self.wq = parameter(4, 4)
        self.wk = parameter(4, 4)
        self.wv = parameter(4, 4)
        self.wo = parameter(4, 4)
        self.head = parameter(4, 8)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        allow: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_embedding[tokens] + self.position_embedding[positions]
        q, k, v = [
            (x @ weight)[None, None, :, :] for weight in (self.wq, self.wk, self.wv)
        ]
        with sdpa_kernel(SDPBackend.MATH):
            attention = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allow,
                is_causal=False,
                dropout_p=0.0,
            )[0, 0]
        return (x + attention @ self.wo) @ self.head


def compare_models() -> dict:
    """Compare shared parameters over separate sequences versus one isolated pack."""
    sequences = [[1, 4, 6], [2, 5]]
    reference = ToyAttentionLM()
    pieces, sums, counts = [], [], []
    for sequence in sequences:
        single = pack_batch([sequence])
        logits = reference(single.tokens, single.positions, single.allow)
        total, count = loss_sum_and_count(logits, single.labels)
        pieces.append(logits)
        sums.append(total)
        counts.append(count)
    reference_loss = sum(sums) / sum(counts)
    reference_loss.backward()

    model = ToyAttentionLM()
    batch = pack_batch(sequences)
    logits = model(batch.tokens, batch.positions, batch.allow)
    total, count = loss_sum_and_count(logits, batch.labels)
    loss = total / count
    loss.backward()
    gradients = [
        (left.grad - right.grad).abs().max().item()
        for left, right in zip(reference.parameters(), model.parameters(), strict=True)
    ]
    reference_logits = torch.cat(pieces)
    with torch.no_grad():
        naive_mask = torch.ones_like(batch.allow, device="cpu").tril()
        naive_logits = model(batch.tokens, batch.positions, naive_mask)
        global_positions = torch.arange(len(batch.tokens), device="cpu")
        unreset_logits = model(batch.tokens, global_positions, batch.allow)
        naive_labels = torch.cat([batch.tokens[1:], torch.tensor([-100], device="cpu")])
        naive_sum, naive_count = loss_sum_and_count(logits, naive_labels)
        naive_boundary_loss = (naive_sum / naive_count).item()
    return {
        "environment_scope": "float64 CPU SDPA math backend, one head, no dropout",
        "sequences": sequences,
        "tokens": batch.tokens.tolist(),
        "positions": batch.positions.tolist(),
        "segments": batch.segments.tolist(),
        "labels_at_predictor_positions": batch.labels.tolist(),
        "cu_seqlens": batch.cu_seqlens.tolist(),
        "sdpa_boolean_allow_mask": batch.allow.tolist(),
        "max_logit_abs_error": (logits - reference_logits).abs().max().item(),
        "loss_abs_error": abs(loss.item() - reference_loss.item()),
        "max_parameter_grad_abs_error": max(gradients),
        "valid_target_count": count,
        "token_mean_loss": loss.item(),
        "sequence_mean_loss": (
            sum(s / n for s, n in zip(sums, counts, strict=True)) / len(sums)
        ).item(),
        "naive_causal_first_sequence_logit_error": (
            naive_logits[:3] - reference_logits[:3]
        )
        .abs()
        .max()
        .item(),
        "naive_causal_second_sequence_logit_error": (
            naive_logits[3:] - reference_logits[3:]
        )
        .abs()
        .max()
        .item(),
        "unreset_position_second_sequence_logit_error": (
            unreset_logits[3:] - reference_logits[3:]
        )
        .abs()
        .max()
        .item(),
        "naive_concat_target_count": naive_count,
        "naive_boundary_loss": naive_boundary_loss,
        "limitations": [
            "Not a GPU or optimized varlen execution",
            "No distributed training",
            "No dropout/RoPE/MoE/packing throughput validation",
        ],
    }
