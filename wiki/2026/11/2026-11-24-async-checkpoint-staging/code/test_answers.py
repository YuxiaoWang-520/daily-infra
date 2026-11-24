"""Independent arithmetic expectations and real local DCP behavior checks."""

import unittest

from answers import simulate_checkpoints
from dcp_demo import probe_failed_staging, run_checkpoint_case


class ScheduleTests(unittest.TestCase):
    def test_fast_storage_overlaps_compute(self):
        r = simulate_checkpoints(300, 100, 100, 2000, 6000)
        self.assertEqual(r["training_finished_ms"], 36000)
        self.assertEqual(r["all_saved_ms"], 42000)
        self.assertEqual(r["sync_total_ms"], 54000)
        self.assertEqual(r["backpressure_ms"], 0)
        self.assertEqual(r["final_drain_ms"], 6000)

    def test_slow_storage_creates_backpressure(self):
        r = simulate_checkpoints(300, 100, 100, 2000, 16000)
        self.assertEqual(r["training_finished_ms"], 48000)
        self.assertEqual(r["all_saved_ms"], 64000)
        self.assertEqual(r["backpressure_ms"], 12000)
        self.assertEqual(r["sync_total_ms"], 84000)
        self.assertEqual(
            [x["backpressure_ms"] for x in r["checkpoints"]], [0, 6000, 6000]
        )

    def test_exact_first_and_last_schedule(self):
        r = simulate_checkpoints(300, 100, 100, 2000, 6000)
        self.assertEqual(
            r["checkpoints"][0],
            {
                "step": 100,
                "backpressure_ms": 0,
                "stage_start_ms": 10000,
                "stage_end_ms": 12000,
                "write_start_ms": 12000,
                "write_end_ms": 18000,
            },
        )
        self.assertEqual(r["checkpoints"][-1]["write_end_ms"], 42000)

    def test_no_checkpoints(self):
        r = simulate_checkpoints(9, 3, 10, 7, 11)
        self.assertEqual(r["checkpoints"], [])
        self.assertEqual(r["all_saved_ms"], 27)
        self.assertEqual(r["sync_total_ms"], 27)

    def test_zero_steps(self):
        r = simulate_checkpoints(0, 3, 10, 7, 11)
        self.assertEqual(r["training_finished_ms"], 0)
        self.assertEqual(r["all_saved_ms"], 0)

    def test_partial_tail_can_hide_some_drain(self):
        r = simulate_checkpoints(250, 100, 100, 2000, 6000)
        self.assertEqual(r["training_finished_ms"], 29000)
        self.assertEqual(r["all_saved_ms"], 30000)
        self.assertEqual(r["final_drain_ms"], 1000)

    def test_zero_stage_and_write_cost(self):
        r = simulate_checkpoints(10, 4, 2, 0, 0)
        self.assertEqual(r["all_saved_ms"], 40)
        self.assertEqual(r["backpressure_ms"], 0)

    def test_storage_exactly_matches_compute_interval(self):
        r = simulate_checkpoints(30, 10, 10, 2, 100)
        self.assertEqual(r["backpressure_ms"], 0)
        self.assertEqual(r["all_saved_ms"], 406)

    def test_one_checkpoint_cannot_hide_final_write(self):
        r = simulate_checkpoints(100, 100, 100, 2000, 6000)
        self.assertEqual(r["all_saved_ms"], r["sync_total_ms"])

    def test_integer_arithmetic_remains_exact(self):
        r = simulate_checkpoints(2, 10**30, 1, 7, 9)
        self.assertEqual(r["training_finished_ms"], 2 * 10**30 + 14)
        self.assertEqual(r["all_saved_ms"], 2 * 10**30 + 23)

    def test_reject_nonintegers_and_booleans(self):
        for args in [(True, 1, 1, 1, 1), (1, 1.0, 1, 1, 1), (1, 1, "1", 1, 1)]:
            with self.subTest(args=args), self.assertRaises(TypeError):
                simulate_checkpoints(*args)

    def test_reject_invalid_ranges(self):
        for args in [
            (-1, 1, 1, 1, 1),
            (1, 0, 1, 1, 1),
            (1, 1, 0, 1, 1),
            (1, 1, 1, -1, 1),
            (1, 1, 1, 1, -1),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                simulate_checkpoints(*args)

    def test_small_grid_accounting_and_order(self):
        for write in range(7):
            for stage in range(4):
                r = simulate_checkpoints(9, 2, 2, stage, write)
                self.assertEqual(
                    r["training_finished_ms"], 18 + 4 * stage + r["backpressure_ms"]
                )
                self.assertEqual(
                    r["all_saved_ms"], r["training_finished_ms"] + r["final_drain_ms"]
                )
                self.assertLessEqual(r["all_saved_ms"], r["sync_total_ms"])
                prior_end = 0
                for c in r["checkpoints"]:
                    self.assertGreaterEqual(c["stage_start_ms"], prior_end)
                    self.assertEqual(c["stage_end_ms"], c["write_start_ms"])
                    prior_end = c["write_end_ms"]


class DCPTests(unittest.TestCase):
    def test_blocking_stage_returns_save_future(self):
        r = run_checkpoint_case(async_staging=False)
        self.assertEqual(r["return_type"], "Future")
        self.assertTrue(r["after_api_return"]["staging_complete"])
        self.assertFalse(r["after_staging"]["upload_done"])
        self.assertFalse(r["after_staging"]["metadata_exists"])
        self.assertTrue(r["restored_matches_snapshot"])
        self.assertTrue(r["live_state_changed"])
        self.assertTrue(r["metadata_after_completion"])

    def test_async_stage_has_two_distinct_completion_boundaries(self):
        r = run_checkpoint_case()
        self.assertEqual(r["return_type"], "AsyncSaveResponse")
        self.assertFalse(r["after_api_return"]["staging_complete"])
        self.assertFalse(r["after_api_return"]["upload_done"])
        self.assertTrue(r["after_staging"]["staging_complete"])
        self.assertFalse(r["after_staging"]["upload_done"])
        self.assertFalse(r["after_staging"]["metadata_exists"])
        self.assertTrue(r["restored_matches_snapshot"])
        self.assertTrue(r["all_saved_successfully"])
        self.assertEqual(r["restored"]["step"], 7)

    def test_shallow_dictionary_is_not_an_independent_tensor_snapshot(self):
        r = run_checkpoint_case()
        self.assertTrue(r["shallow_copy_aliases_live_weight"])
        self.assertEqual(r["restored"]["weight"], [1.0, 2.0])
        self.assertEqual(r["live"]["weight"], [101.0, 102.0])
        self.assertEqual(r["payload_bytes"], 40)

    def test_failed_save_is_done_but_not_successful(self):
        r = run_checkpoint_case(fail_write=True)
        self.assertTrue(r["after_staging"]["staging_complete"])
        self.assertTrue(r["upload_done_after_result"])
        self.assertFalse(r["all_saved_successfully"])
        self.assertFalse(r["metadata_after_completion"])
        self.assertEqual(r["failure_type"], "CheckpointException")

    def test_already_failed_staging_propagates_to_upload_future(self):
        r = probe_failed_staging()
        self.assertTrue(r["upload_done"])
        self.assertEqual(r["upload_failure_type"], "RuntimeError")
        self.assertFalse(r["metadata_exists"])
        # staging_completion reporting is recorded as a version probe, not a universal guarantee.
        self.assertIsInstance(r["staging_completion_reported_success"], bool)


if __name__ == "__main__":
    unittest.main()
