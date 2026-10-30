"""Immutable records for the synthetic latency exercise."""

from dataclasses import dataclass


def require_int(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Round:
    start_ms: float
    end_ms: float
    decode_count: int
    prefill_tokens: int
    computed_prompt_tokens: int


@dataclass(frozen=True)
class Schedule:
    rounds: tuple[Round, ...]
    prompt_first_token_ms: float
    decode_output_times_ms: tuple[tuple[float, ...], ...]
