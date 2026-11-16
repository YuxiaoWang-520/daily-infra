"""Emit asserted CPU teaching traces without CUDA allocation or timing claims."""

import datetime
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from allocator_model import Allocator
from model_types import Block, Segment
from snapshot_stats import summarize

MIB = 1024 * 1024


def capture(allocator: Allocator) -> dict:
    snapshot = allocator.snapshot()
    return {
        "totals": asdict(summarize(snapshot)),
        "segments": [asdict(segment) for segment in snapshot],
        "segment_allocations": allocator.segment_allocations,
        "allocation_retries": allocator.allocation_retries,
        "ooms": allocator.ooms,
    }


def fragmentation_case() -> dict:
    allocator = Allocator(16 * MIB, 8 * MIB)
    handles = [allocator.malloc(size * MIB) for size in (6, 2, 6, 2)]
    allocator.free(handles[0])
    allocator.free(handles[2])
    before = capture(allocator)
    assert before["totals"]["inactive_bytes"] == 12 * MIB
    assert before["totals"]["largest_free_bytes"] == 6 * MIB
    try:
        allocator.malloc(7 * MIB)
    except MemoryError as error:
        rejection = type(error).__name__
    else:
        raise AssertionError("The fragmented request must fail")
    released = allocator.release_free_segments()
    assert released == 0
    assert capture(allocator)["segments"] == before["segments"]
    allocator.free(handles[1])
    allocator.malloc(7 * MIB)
    after = capture(allocator)
    assert after["totals"]["allocated_bytes"] == 9 * MIB
    assert after["segment_allocations"] == 2
    return {"before": before, "expected_rejection": rejection, "after": after}


def whole_segment_retry_case() -> dict:
    allocator = Allocator(16 * MIB, 8 * MIB)
    handles = [allocator.malloc(8 * MIB), allocator.malloc(8 * MIB)]
    for handle in handles:
        allocator.free(handle)
    before = capture(allocator)
    allocator.malloc(12 * MIB)
    after = capture(allocator)
    assert before["totals"]["releasable_bytes"] == 16 * MIB
    assert after["totals"]["reserved_bytes"] == 12 * MIB
    assert after["allocation_retries"] == 1
    return {"before": before, "after": after}


def deferred_case() -> dict:
    allocator = Allocator(8 * MIB, 8 * MIB)
    handle = allocator.malloc(8 * MIB)
    allocator.free(handle, deferred=True)
    before = capture(allocator)
    assert before["totals"]["awaiting_bytes"] == 8 * MIB
    assert before["totals"]["inactive_bytes"] == 0
    released = allocator.release_free_segments()
    assert released == 0
    completed = allocator.complete_deferred()
    assert completed == 8 * MIB
    allocator.malloc(8 * MIB)
    after = capture(allocator)
    assert after["segment_allocations"] == 1
    return {"before": before, "after": after}


def cache_reuse_case() -> dict:
    allocator = Allocator(8 * MIB, 8 * MIB)
    iterations = 200
    for _ in range(iterations):
        allocator.free(allocator.malloc(4 * MIB))
    assert allocator.segment_allocations == 1
    released = allocator.release_free_segments()
    assert released == 8 * MIB
    assert allocator.snapshot() == ()
    return {
        "iterations": iterations,
        "model_segment_allocations": allocator.segment_allocations,
        "model_final_segment_releases": int(released > 0),
        "released_bytes": released,
        "final_reserved_bytes": summarize(allocator.snapshot()).reserved_bytes,
        "uncached_reference_allocations": iterations,
        "uncached_reference_frees": iterations,
        "measurement": "Teaching operation counts, without GPU latency measurements",
    }


def same_gap_cases() -> dict:
    cases = {
        "releasable": (Segment(0, (Block(8 * MIB, "inactive"),)),),
        "awaiting": (Segment(0, (Block(8 * MIB, "active_awaiting_free"),)),),
        "trapped": (
            Segment(
                0, (Block(8 * MIB, "active_allocated"), Block(8 * MIB, "inactive"))
            ),
        ),
    }
    result = {}
    for name, segments in cases.items():
        totals = summarize(segments)
        assert totals.reserved_bytes - totals.allocated_bytes == 8 * MIB
        result[name] = asdict(totals)
    return result


def main() -> None:
    code = Path(__file__).parent
    result = {
        "checked_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime_executed": False,
        "unit": "bytes",
        "scope": "CPU single-pool ordinary-segment teaching model",
        "code_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(code.glob("*.py"))
        },
        "fragmentation": fragmentation_case(),
        "whole_segment_retry": whole_segment_retry_case(),
        "deferred": deferred_case(),
        "cache_reuse": cache_reuse_case(),
        "same_reserved_gap": same_gap_cases(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
