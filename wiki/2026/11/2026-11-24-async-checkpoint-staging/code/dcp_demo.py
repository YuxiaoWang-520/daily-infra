"""Real local CPU DCP saves with event-controlled staging and storage boundaries.

Only public synthetic tensors are saved. Temporary checkpoints are loaded
only by their owning invocation. This is neither a distributed benchmark nor
a power-loss durability test.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner
from torch.distributed.checkpoint.staging import AsyncStager, BlockingAsyncStager
from torch.distributed.checkpoint.state_dict_saver import AsyncSaveResponse
from torch.distributed.checkpoint.storage import WriteResult

WAIT_SECONDS = 10
State = dict[str, torch.Tensor]


class ControlledStager(AsyncStager):
    """One-shot CPU stager: return a pending future before making an isolated copy."""

    _synchronize_after_execute = False

    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._copy = BlockingAsyncStager(cache_staged_state_dict=False)
        self._used = False

    def stage(self, state_dict: State) -> Future[State]:
        if self._used:
            raise RuntimeError("the teaching stager supports one request only")
        self._used = True

        def copy_after_gate() -> State:
            self.entered.set()
            if not self.release.wait(WAIT_SECONDS):
                raise TimeoutError("CPU staging gate was not released")
            return self._copy.stage(state_dict)

        return self._executor.submit(copy_after_gate)

    def synchronize_staging(self) -> None:
        raise RuntimeError("this future-based example uses staging_completion.result()")

    def close(self) -> None:
        self.release.set()
        self._executor.shutdown(wait=True)
        self._copy.close()


class GateWriter(dcp.FileSystemWriter):
    """Hold before actual file writes; optionally inject an expected write failure."""

    def __init__(self, path: Path, *, fail: bool) -> None:
        super().__init__(path, sync_files=True, thread_count=1)
        self.entered = Event()
        self.release = Event()
        self.fail = fail

    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> torch.futures.Future[list[WriteResult]]:
        self.entered.set()
        if not self.release.wait(WAIT_SECONDS):
            raise TimeoutError("storage gate was not released")
        if self.fail:
            raise RuntimeError("injected synthetic checkpoint write failure")
        return super().write_data(plan, planner)


def _make_state() -> State:
    return {
        "weight": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "momentum": torch.tensor([0.1, 0.2], dtype=torch.float64),
        "step": torch.tensor(7, dtype=torch.int64),
    }


def _json_state(state: State) -> dict[str, Any]:
    return {key: value.tolist() for key, value in state.items()}


def run_checkpoint_case(
    *, async_staging: bool = True, fail_write: bool = False
) -> dict[str, Any]:
    """Demonstrate staging safety, a pending upload, mutation, and save outcome.

    Training-state mutation occurs only after staging successfully completes.
    Event gates establish ordering; no sleep-based timing assumptions are used.
    """
    if not isinstance(async_staging, bool) or not isinstance(fail_write, bool):
        raise TypeError("async_staging and fail_write must be booleans")
    with TemporaryDirectory(prefix="public-async-checkpoint-") as folder:
        path = Path(folder) / "checkpoint"
        state = _make_state()
        expected = {key: value.clone() for key, value in state.items()}
        shallow_alias = state.copy()["weight"].data_ptr() == state["weight"].data_ptr()
        stager = ControlledStager() if async_staging else BlockingAsyncStager()
        writer = GateWriter(path, fail=fail_write)
        upload: Future | None = None
        try:
            response = dcp.async_save(
                state, storage_writer=writer, async_stager=stager, no_dist=True
            )
            if async_staging:
                if not isinstance(response, AsyncSaveResponse) or not isinstance(
                    stager, ControlledStager
                ):
                    raise RuntimeError("expected separate staging and upload futures")
                upload = response.upload_completion
                if not stager.entered.wait(WAIT_SECONDS):
                    raise TimeoutError("staging did not start")
                after_return = {
                    "staging_complete": response.staging_completion.done(),
                    "upload_done": upload.done(),
                    "metadata_exists": (path / ".metadata").exists(),
                }
                stager.release.set()
                response.staging_completion.result(timeout=WAIT_SECONDS)
            else:
                if not isinstance(response, Future):
                    raise RuntimeError("expected a save Future for blocking staging")
                upload = response
                after_return = {
                    "staging_complete": True,
                    "upload_done": upload.done(),
                    "metadata_exists": (path / ".metadata").exists(),
                }
            if not writer.entered.wait(WAIT_SECONDS):
                raise TimeoutError("upload did not reach the storage gate")
            after_staging = {
                "staging_complete": True,
                "upload_done": upload.done(),
                "metadata_exists": (path / ".metadata").exists(),
            }
            state["weight"].add_(100)
            state["momentum"].add_(10)
            state["step"].add_(1)
            writer.release.set()
            failure_type = None
            try:
                upload.result(timeout=WAIT_SECONDS)
            except CheckpointException as error:
                if not fail_write:
                    raise
                failure_type = type(error).__name__
            if fail_write and failure_type is None:
                raise RuntimeError(
                    "injected write failure did not reach the save future"
                )
            restored = None
            matches = False
            if failure_type is None:
                loaded = {
                    key: torch.zeros_like(value) for key, value in expected.items()
                }
                dcp.load(loaded, checkpoint_id=path, no_dist=True)
                matches = all(
                    torch.equal(loaded[key], expected[key]) for key in expected
                )
                restored = _json_state(loaded)
                if not matches:
                    raise RuntimeError(
                        "restored state differs from the staged snapshot"
                    )
            return {
                "scope": "real local CPU DCP; synthetic state; no distributed/GPU benchmark",
                "return_type": type(response).__name__,
                "async_staging": async_staging,
                "after_api_return": after_return,
                "after_staging": after_staging,
                "upload_done_after_result": upload.done(),
                "failure_type": failure_type,
                "all_saved_successfully": failure_type is None,
                "metadata_after_completion": (path / ".metadata").exists(),
                "shallow_copy_aliases_live_weight": shallow_alias,
                "payload_bytes": sum(
                    t.numel() * t.element_size() for t in expected.values()
                ),
                "live_state_changed": all(
                    not torch.equal(state[key], expected[key]) for key in state
                ),
                "snapshot": _json_state(expected),
                "live": _json_state(state),
                "restored": restored,
                "restored_matches_snapshot": matches,
                "checkpoint_files": sorted(p.name for p in path.iterdir())
                if path.exists()
                else [],
                "temporary_checkpoint_retained": False,
            }
        finally:
            writer.release.set()
            if isinstance(stager, ControlledStager):
                stager.release.set()
            try:
                if upload is not None and not upload.done():
                    upload.result(timeout=WAIT_SECONDS)
            finally:
                stager.close()


class AlreadyFailedStager(AsyncStager):
    """Version probe: return an already-completed failed staging Future."""

    def stage(self, state_dict: State) -> Future[State]:
        future: Future[State] = Future()
        future.set_exception(RuntimeError("injected synthetic staging failure"))
        return future

    def synchronize_staging(self) -> None:
        return None

    def close(self) -> None:
        return None


def probe_failed_staging() -> dict[str, Any]:
    """Record version-specific staging-future reporting; always check upload result."""
    with TemporaryDirectory(prefix="public-failed-stage-") as folder:
        path = Path(folder) / "checkpoint"
        stager = AlreadyFailedStager()
        try:
            response = dcp.async_save(
                _make_state(), checkpoint_id=path, async_stager=stager, no_dist=True
            )
            if not isinstance(response, AsyncSaveResponse):
                raise TypeError(
                    "expected AsyncSaveResponse for a Future-returning stager"
                )
            stage_success = True
            try:
                response.staging_completion.result(timeout=WAIT_SECONDS)
            except RuntimeError:
                stage_success = False
            upload_failure = None
            try:
                response.upload_completion.result(timeout=WAIT_SECONDS)
            except RuntimeError as error:
                upload_failure = type(error).__name__
            if upload_failure is None:
                raise RuntimeError("failed staging must not yield a successful upload")
            return {
                "version_probe_only": True,
                "staging_completion_reported_success": stage_success,
                "upload_done": response.upload_completion.done(),
                "upload_failure_type": upload_failure,
                "metadata_exists": (path / ".metadata").exists(),
                "interpretation": "Never use done() or staging_completion alone as proof of a saved checkpoint",
            }
        finally:
            stager.close()
