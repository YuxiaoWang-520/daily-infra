"""Failure-injection tests for a single-writer checkpoint publication."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from checkpoint_io import atomic_save, load_checkpoint


class CheckpointIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "latest.pt"
        self.old = {"step": 5, "weight": torch.tensor([1.0, 2.0])}

    def assert_old_is_intact(self) -> None:
        actual = load_checkpoint(self.path)
        self.assertEqual(actual["step"], 5)
        self.assertTrue(torch.equal(actual["weight"], self.old["weight"]))
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_roundtrip_tensor_and_primitive_state(self) -> None:
        atomic_save({**self.old, "rng": (3, (1, 2), None)}, self.path)
        self.assert_old_is_intact()
        self.assertEqual(load_checkpoint(self.path)["rng"], (3, (1, 2), None))

    def test_success_replaces_previous_checkpoint(self) -> None:
        atomic_save(self.old, self.path)
        atomic_save({"step": 6}, self.path)
        self.assertEqual(load_checkpoint(self.path), {"step": 6})
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_partial_serialization_failure_preserves_previous_checkpoint(self) -> None:
        atomic_save(self.old, self.path)

        def fail_after_writing(payload, stream) -> None:
            stream.write(b"partial invalid checkpoint")
            raise OSError("injected serialization failure")

        with (
            patch("checkpoint_io.torch.save", side_effect=fail_after_writing),
            self.assertRaisesRegex(OSError, "serialization"),
        ):
            atomic_save({"step": 6}, self.path)
        self.assert_old_is_intact()

    def test_fsync_failure_preserves_previous_checkpoint(self) -> None:
        atomic_save(self.old, self.path)
        with (
            patch("checkpoint_io.os.fsync", side_effect=OSError("injected fsync")),
            self.assertRaisesRegex(OSError, "fsync"),
        ):
            atomic_save({"step": 6}, self.path)
        self.assert_old_is_intact()

    def test_replace_failure_preserves_previous_checkpoint(self) -> None:
        atomic_save(self.old, self.path)
        with (
            patch("checkpoint_io.os.replace", side_effect=OSError("injected rename")),
            self.assertRaisesRegex(OSError, "rename"),
        ):
            atomic_save({"step": 6}, self.path)
        self.assert_old_is_intact()

    def test_first_save_failure_does_not_publish_partial_file(self) -> None:
        with (
            patch("checkpoint_io.torch.save", side_effect=OSError("injected write")),
            self.assertRaises(OSError),
        ):
            atomic_save(self.old, self.path)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_parent_must_already_exist(self) -> None:
        with self.assertRaises(FileNotFoundError):
            atomic_save(self.old, self.root / "missing" / "latest.pt")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_loader_rejects_non_dictionary_root(self) -> None:
        torch.save([1, 2], self.path)
        with self.assertRaisesRegex(TypeError, "dictionary"):
            load_checkpoint(self.path)


if __name__ == "__main__":
    unittest.main()
