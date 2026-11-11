"""Synchronous integer ReduceScatter + AllGather on a logical ring.

Each rank starts with a full equal-sized vector, split into P equal chunks.
Messages capture pre-round values before any receiver mutates its local copy.
Python integers model exact arithmetic; element_bytes is a declared logical
element width for traffic accounting, not Python object serialization size.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    phase: str
    step: int
    source: int
    destination: int
    chunk: int
    values: tuple[int, ...]


@dataclass(frozen=True)
class RingResult:
    outputs: tuple[tuple[int, ...], ...]
    reduced_chunks: tuple[tuple[int, ...], ...]
    messages: tuple[Message, ...]
    sent_bytes: tuple[int, ...]


def ring_allreduce(
    inputs: Sequence[Sequence[int]], element_bytes: int = 8
) -> RingResult:
    """Return exact SUMs, phase-one ownership, immutable trace and TX bytes."""
    if type(element_bytes) is not int:
        raise TypeError("element_bytes must be an integer")
    if element_bytes <= 0:
        raise ValueError("element_bytes must be positive")
    ranks = len(inputs)
    if not ranks or not inputs[0]:
        raise ValueError("inputs must contain nonempty vectors")
    count = len(inputs[0])
    if any(len(row) != count for row in inputs):
        raise ValueError("all rank vectors must have the same length")
    if count % ranks:
        raise ValueError("element count must be divisible by rank count")
    if any(type(value) is not int for row in inputs for value in row):
        raise TypeError("this exact-arithmetic fixture requires integer elements")
    width = count // ranks
    chunks = [
        [list(row[index : index + width]) for index in range(0, count, width)]
        for row in inputs
    ]
    trace = []
    sent_bytes = [0] * ranks

    def round_messages(phase: str, step: int) -> list[Message]:
        messages = []
        for rank in range(ranks):
            chunk = (
                (rank - step - 1) % ranks
                if phase == "reduce_scatter"
                else (rank - step) % ranks
            )
            values = tuple(chunks[rank][chunk])
            messages.append(
                Message(phase, step, rank, (rank + 1) % ranks, chunk, values)
            )
            sent_bytes[rank] += len(values) * element_bytes
        return messages

    for step in range(ranks - 1):
        pending = round_messages("reduce_scatter", step)
        for message in pending:
            local = chunks[message.destination][message.chunk]
            chunks[message.destination][message.chunk] = [
                a + b for a, b in zip(local, message.values, strict=True)
            ]
        trace.extend(pending)
    reduced = tuple(tuple(chunks[rank][rank]) for rank in range(ranks))
    for step in range(ranks - 1):
        pending = round_messages("allgather", step)
        for message in pending:
            chunks[message.destination][message.chunk] = list(message.values)
        trace.extend(pending)
    outputs = tuple(tuple(value for chunk in row for value in chunk) for row in chunks)
    return RingResult(outputs, reduced, tuple(trace), tuple(sent_bytes))
