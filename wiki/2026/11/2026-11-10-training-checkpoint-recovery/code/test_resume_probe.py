"""Check snapshot isolation, validation and real fresh-process continuation."""

import copy
import random
import unittest

import torch
from resume_probe import Trainer, restore, run_probe


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.python_rng = random.getstate()
        self.torch_rng = torch.get_rng_state()
        self.addCleanup(random.setstate, self.python_rng)
        self.addCleanup(torch.set_rng_state, self.torch_rng)
        self.trainer = Trainer()
        for _ in range(5):
            self.trainer.step()
        self.saved = self.trainer.snapshot()

    def test_snapshot_tracks_committed_boundary(self) -> None:
        self.assertEqual(self.saved["completed_steps"], 5)
        self.assertEqual(self.saved["cursor"], 10)

    def test_snapshot_does_not_alias_live_model_or_optimizer(self) -> None:
        frozen = copy.deepcopy(self.saved)
        self.trainer.step()
        for name, tensor in frozen["model"].items():
            self.assertTrue(torch.equal(tensor, self.saved["model"][name]))
        for key, state in frozen["optimizer"]["state"].items():
            self.assertTrue(
                torch.equal(
                    state["exp_avg"], self.saved["optimizer"]["state"][key]["exp_avg"]
                )
            )
        self.assertTrue(
            any(
                not torch.equal(value, self.trainer.model.state_dict()[name])
                for name, value in self.saved["model"].items()
            )
        )

    def test_restore_reproduces_next_batch_and_loss(self) -> None:
        expected = self.trainer.step()
        resumed = restore(self.saved)
        actual = resumed.step()
        self.assertEqual(expected, actual)

    def test_restored_training_does_not_mutate_input_optimizer_snapshot(self) -> None:
        frozen = copy.deepcopy(self.saved["optimizer"]["state"])
        resumed = restore(self.saved)
        resumed.step()
        for key, state in frozen.items():
            for name in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(
                    torch.equal(
                        state[name], self.saved["optimizer"]["state"][key][name]
                    ),
                    f"input snapshot changed: {key}/{name}",
                )

    def test_repeated_restores_from_same_snapshot_are_equivalent(self) -> None:
        first = restore(self.saved)
        first.step()
        expected = torch.cat([p.detach().flatten() for p in first.model.parameters()])
        second = restore(self.saved)
        second.step()
        actual = torch.cat([p.detach().flatten() for p in second.model.parameters()])
        self.assertTrue(torch.equal(expected, actual))

    def test_rejects_unknown_schema(self) -> None:
        self.saved["schema"] = 99
        with self.assertRaisesRegex(ValueError, "schema"):
            restore(self.saved)

    def test_rejects_inconsistent_consumed_cursor(self) -> None:
        self.saved["cursor"] = 12
        with self.assertRaisesRegex(ValueError, "cursor"):
            restore(self.saved)

    def test_rejects_boolean_progress(self) -> None:
        self.saved["completed_steps"] = True
        with self.assertRaises(ValueError):
            restore(self.saved)

    def test_rejects_duplicate_sample_order(self) -> None:
        self.saved["order"][0] = self.saved["order"][1]
        with self.assertRaisesRegex(ValueError, "permutation"):
            restore(self.saved)

    def test_rejects_changed_training_configuration(self) -> None:
        self.saved["config"]["batch_size"] = 4
        with self.assertRaisesRegex(ValueError, "configuration"):
            restore(self.saved)

    def test_rejects_unknown_negative_control(self) -> None:
        with self.assertRaisesRegex(ValueError, "omit"):
            restore(self.saved, omit="everything")


class FreshProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.python_rng = random.getstate()
        cls.torch_rng = torch.get_rng_state()
        cls.threads = torch.get_num_threads()
        cls.result = run_probe()

    def test_full_restore_matches_every_recorded_state(self) -> None:
        full = self.result["modes"]["none"]
        for field in (
            "parameter_max_abs_error",
            "moment_max_abs_error",
            "loss_max_abs_error",
        ):
            self.assertEqual(full[field], 0.0)
        for field in (
            "sample_ids_equal",
            "rng_equal",
            "lr_equal",
            "optimizer_steps_equal",
        ):
            self.assertTrue(full[field], field)
        self.assertEqual(full["completed_steps"], 10)
        self.assertEqual(full["cursor"], 20)

    def test_runs_five_new_processes_after_losing_three_steps(self) -> None:
        self.assertEqual(self.result["fresh_processes"], 5)
        self.assertEqual(self.result["saved_step"], 5)
        self.assertEqual(self.result["discarded_at_step"], 8)
        self.assertEqual(self.result["resume_steps"], 5)

    def test_omitting_optimizer_changes_parameters(self) -> None:
        result = self.result["modes"]["optimizer"]
        self.assertGreater(result["parameter_max_abs_error"], 1e-10)
        self.assertFalse(result["optimizer_steps_equal"])
        self.assertTrue(result["sample_ids_equal"])

    def test_omitting_rng_changes_loss(self) -> None:
        result = self.result["modes"]["rng"]
        self.assertGreater(result["loss_max_abs_error"], 1e-10)
        self.assertFalse(result["rng_equal"])
        self.assertTrue(result["sample_ids_equal"])

    def test_omitting_cursor_replays_wrong_examples(self) -> None:
        result = self.result["modes"]["cursor"]
        self.assertFalse(result["sample_ids_equal"])
        self.assertEqual(result["cursor"], 10)

    def test_omitting_scheduler_changes_lr(self) -> None:
        result = self.result["modes"]["scheduler"]
        self.assertFalse(result["lr_equal"])
        self.assertGreater(result["parameter_max_abs_error"], 1e-10)

    def test_probe_preserves_callers_rng_and_thread_setting(self) -> None:
        self.assertEqual(random.getstate(), self.python_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), self.torch_rng))
        self.assertEqual(torch.get_num_threads(), self.threads)


if __name__ == "__main__":
    unittest.main()
