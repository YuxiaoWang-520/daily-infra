"""Exact tensor-payload models; these do not simulate a CUDA allocator."""

from collections.abc import Sequence

from models import Event, MemoryPoint, PeakReport, StateBytes


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def estimate_state_bytes(
    numel: int,
    parameter_bytes: int,
    gradient_bytes: int,
    moment_bytes: int,
    master_bytes: int = 0,
    *,
    gradients_live: bool = True,
    state_initialized: bool = True,
    amsgrad: bool = False,
) -> StateBytes:
    """Count a homogeneous, nonaliased Adam group's explicit storage layout.

    Master weights, when requested, exist independently of lazy Adam state.
    Scalar counters, activations, alignment and allocator cache are excluded.
    """
    _integer("numel", numel, 0)
    _integer("parameter_bytes", parameter_bytes, 1)
    _integer("gradient_bytes", gradient_bytes, 1)
    _integer("moment_bytes", moment_bytes, 1)
    _integer("master_bytes", master_bytes, 0)
    for name, flag in (
        ("gradients_live", gradients_live),
        ("state_initialized", state_initialized),
        ("amsgrad", amsgrad),
    ):
        if type(flag) is not bool:
            raise ValueError(f"{name} must be a bool")
    return StateBytes(
        parameters=numel * parameter_bytes,
        gradients=numel * gradient_bytes if gradients_live else 0,
        first_moment=numel * moment_bytes if state_initialized else 0,
        second_moment=numel * moment_bytes if state_initialized else 0,
        master=numel * master_bytes,
        amsgrad_max=numel * moment_bytes if state_initialized and amsgrad else 0,
    )


def replay_lifetimes(events: Sequence[Event]) -> PeakReport:
    """Replay logical references, counting each live storage exactly once.

    Only list/tuple traces are accepted. A reused name receives a fresh storage
    ID, so an alias to an older allocation continues to hold that older storage.
    """
    if not isinstance(events, (list, tuple)):
        raise ValueError("events must be a list or tuple of Event instances")
    names: dict[str, int] = {}
    sizes: dict[int, int] = {}
    references: dict[int, int] = {}
    points: list[MemoryPoint] = []
    next_id = 0
    live_bytes = 0
    peak_bytes = 0
    peak_index = None
    for index, event in enumerate(events):
        if not isinstance(event, Event):
            raise ValueError("each item must be an Event")
        if not isinstance(event.name, str) or not event.name.strip():
            raise ValueError("event name must be a nonempty string")
        if event.op == "alloc":
            _integer("allocation bytes", event.nbytes, 1)
            if event.name in names or event.source is not None:
                raise ValueError("alloc requires a fresh live name and no source")
            storage_id = next_id
            next_id += 1
            names[event.name] = storage_id
            sizes[storage_id] = event.nbytes
            references[storage_id] = 1
            live_bytes += event.nbytes
        elif event.op == "alias":
            _integer("alias bytes", event.nbytes, 0)
            if event.nbytes != 0:
                raise ValueError("alias must not allocate bytes")
            if (
                event.name in names
                or not isinstance(event.source, str)
                or event.source not in names
            ):
                raise ValueError("alias requires a fresh live name and live source")
            storage_id = names[event.source]
            names[event.name] = storage_id
            references[storage_id] += 1
        elif event.op == "drop":
            _integer("drop bytes", event.nbytes, 0)
            if event.nbytes != 0 or event.source is not None or event.name not in names:
                raise ValueError("drop requires an existing live name and no payload")
            storage_id = names.pop(event.name)
            references[storage_id] -= 1
            if references[storage_id] == 0:
                live_bytes -= sizes.pop(storage_id)
                del references[storage_id]
        else:
            raise ValueError("op must be alloc, alias or drop")
        points.append(MemoryPoint(index, live_bytes, len(sizes), tuple(sorted(names))))
        if live_bytes > peak_bytes:
            peak_bytes, peak_index = live_bytes, index
    return PeakReport(peak_bytes, peak_index, live_bytes, tuple(points))
