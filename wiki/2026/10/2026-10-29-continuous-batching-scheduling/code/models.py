"""Small immutable inputs and outputs shared by both exercises."""

from dataclasses import dataclass


def require_int(value: int, name: str, minimum: int = 0) -> None:
    """Reject invalid counts, including booleans masquerading as integers."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def require_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("request_id must be a nonempty string")


@dataclass(frozen=True)
class Request:
    """One-token prompt; one output per scheduled tick; no KV shortage."""

    request_id: str
    arrival_tick: int
    output_tokens: int

    def __post_init__(self) -> None:
        require_id(self.request_id)
        require_int(self.arrival_tick, "arrival_tick")
        require_int(self.output_tokens, "output_tokens", 1)


@dataclass(frozen=True)
class Step:
    """Model execution occupies [start_tick, start_tick + 1)."""

    start_tick: int
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class Simulation:
    steps: tuple[Step, ...]
    output_ticks: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class Prefill:
    """Fresh prompt with no cached prefix, processed whole in this exercise."""

    request_id: str
    prompt_tokens: int

    def __post_init__(self) -> None:
        require_id(self.request_id)
        require_int(self.prompt_tokens, "prompt_tokens", 1)


@dataclass(frozen=True)
class Admission:
    admitted: tuple[str, ...]
    num_scheduled_requests: int
    num_scheduled_tokens: int
    remaining_tokens: int
    remaining_blocks: int
    stopped_request: str | None
    stop_reason: str | None
