"""Hand-computed stage limits and queue coverage, independent of PyTorch timing."""

import importlib
import math
import os
import unittest
from dataclasses import FrozenInstanceError

impl = importlib.import_module(os.environ.get("EXERCISE_IMPL", "answers"))


class BudgetTests(unittest.TestCase):
    def budget(self, **overrides):
        arguments = {
            "storage_ms": 4,
            "cpu_ms": 16,
            "workers": 2,
            "transfer_ms": 3,
            "compute_ms": 10,
            "prefetch_factor": 2,
            "batch_bytes": 64 * 1024**2,
        }
        arguments.update(overrides)
        return impl.pipeline_budget(**arguments)

    def test_same_stream_copy_and_compute_add(self):
        result = self.budget()
        self.assertEqual(
            (
                result.storage_interval_ms,
                result.worker_interval_ms,
                result.device_interval_ms,
            ),
            (4, 10, 13),
        )
        self.assertEqual((result.ideal_interval_ms, result.serial_ms), (13, 33))
        self.assertEqual(result.limiting_stages, ("copy_and_compute",))
        self.assertEqual(
            (result.task_slots, result.task_payload_bytes), (4, 256 * 1024**2)
        )

    def test_explicit_device_overlap_changes_the_lower_bound(self):
        result = self.budget(allow_device_overlap=True)
        self.assertEqual(
            (result.ideal_interval_ms, result.device_interval_ms), (10, 10)
        )
        self.assertEqual(result.limiting_stages, ("worker_preparation", "compute"))

    def test_shared_storage_does_not_scale_with_workers(self):
        result = self.budget(storage_ms=20, workers=32)
        self.assertEqual(result.ideal_interval_ms, 20)
        self.assertEqual(result.limiting_stages, ("storage",))

    def test_cpu_capacity_scales_until_another_stage_limits(self):
        self.assertEqual(self.budget(cpu_ms=24).ideal_interval_ms, 14)
        self.assertEqual(
            self.budget(cpu_ms=32).limiting_stages, ("worker_preparation",)
        )
        self.assertEqual(self.budget(cpu_ms=32, workers=4).ideal_interval_ms, 13)

    def test_transfer_can_limit_even_with_fast_data_loading(self):
        result = self.budget(transfer_ms=14, allow_device_overlap=True)
        self.assertEqual(result.ideal_interval_ms, 14)
        self.assertEqual(result.limiting_stages, ("transfer",))

    def test_all_tied_stages_are_reported(self):
        result = self.budget(storage_ms=13, cpu_ms=13)
        self.assertEqual(
            result.limiting_stages,
            ("storage", "worker_preparation", "copy_and_compute"),
        )

    def test_worker_remains_occupied_during_reads(self):
        result = self.budget(transfer_ms=0, compute_ms=1)
        self.assertEqual(result.ideal_interval_ms, 10)
        self.assertEqual(result.limiting_stages, ("worker_preparation",))

    def test_prefetch_changes_payload_without_creating_capacity(self):
        one, four = self.budget(prefetch_factor=1), self.budget(prefetch_factor=4)
        self.assertEqual(one.ideal_interval_ms, four.ideal_interval_ms)
        self.assertEqual(four.task_payload_bytes, 4 * one.task_payload_bytes)

    def test_large_integer_task_payloads_are_exact(self):
        result = self.budget(workers=10**309, batch_bytes=10**50)
        self.assertEqual(result.task_payload_bytes, 2 * 10**359)
        self.assertEqual(result.ideal_interval_ms, 13)

    def test_result_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.budget().task_slots = 0

    def test_invalid_counts_and_overlap_flag(self):
        for overrides in (
            {"workers": 0},
            {"workers": True},
            {"workers": 1.5},
            {"prefetch_factor": 0},
            {"batch_bytes": -1},
            {"allow_device_overlap": "yes"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.budget(**overrides)

    def test_invalid_and_overflowing_durations(self):
        for key in ("storage_ms", "cpu_ms", "transfer_ms", "compute_ms"):
            for value in (-1, True, math.inf, math.nan, "slow", 10**400):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.budget(**{key: value})
        for overrides in ({"compute_ms": 0}, {"storage_ms": 1e308, "cpu_ms": 1e308}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.budget(**overrides)

    def test_single_worker_combined_overflow_uses_the_contract_error(self):
        with self.assertRaises(ValueError):
            self.budget(storage_ms=1e308, cpu_ms=1e308, workers=1)


class QueueCoverageTests(unittest.TestCase):
    def test_two_ready_batches_cover_twenty_six_ms(self):
        self.assertEqual(impl.exposed_pause(40, 2, 13), 14)

    def test_four_ready_batches_cover_the_pause(self):
        self.assertEqual(impl.exposed_pause(40, 4, 13), 0)

    def test_empty_queue_exposes_the_whole_pause(self):
        self.assertEqual(impl.exposed_pause(40, 0, 13), 40)

    def test_zero_pause_and_exact_boundary(self):
        self.assertEqual(impl.exposed_pause(0, 2, 13), 0)
        self.assertEqual(impl.exposed_pause(26, 2, 13), 0)

    def test_huge_ready_count_does_not_overflow(self):
        self.assertEqual(impl.exposed_pause(40, 10**309, 13), 0)

    def test_fractional_durations(self):
        self.assertAlmostEqual(impl.exposed_pause(0.3, 1, 0.1), 0.2)

    def test_invalid_queue_inputs(self):
        for arguments in (
            (40, -1, 13),
            (40, True, 13),
            (40, 1.5, 13),
            (40, 2, 0),
            (math.inf, 2, 13),
            (40, 2, math.nan),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                impl.exposed_pause(*arguments)


if __name__ == "__main__":
    unittest.main()
