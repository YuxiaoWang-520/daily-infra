"""Acceptance tests for the standard-library address-model exercise."""

import subprocess
import sys
import unittest
from pathlib import Path

from layout_answers import is_c_contiguous, pack_logical, storage_index


class AddressModelTests(unittest.TestCase):
    def test_transpose_uses_original_storage(self) -> None:
        self.assertEqual(storage_index((4, 3), (1, 4), (2, 1)), 6)

    def test_slice_offset_is_in_elements(self) -> None:
        self.assertEqual(storage_index((2, 4), (4, 1), (1, 2), offset=4), 10)

    def test_scalar_has_one_address(self) -> None:
        self.assertEqual(storage_index((), (), (), offset=2), 2)
        self.assertEqual(pack_logical([10, 20, 30], (), (), offset=2), [30])

    def test_empty_tensor_has_no_valid_index(self) -> None:
        with self.assertRaises(IndexError):
            storage_index((2, 0), (1, 1), (0, 0))
        self.assertEqual(pack_logical([], (2, 0), (1, 1)), [])

    def test_empty_pack_does_not_pool_unused_indices(self) -> None:
        # Isolate the measurement from caller-owned tracemalloc state.
        script = """
import sys
sys.path.insert(0, sys.argv[1])
from layout_answers import pack_logical
import tracemalloc
tracemalloc.start()
assert pack_logical([], (0, 100_000), (0, 0)) == []
print(tracemalloc.get_traced_memory()[1])
"""
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script, str(Path(__file__).parent)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        peak = int(completed.stdout.strip())
        self.assertLess(peak, 1024 * 1024)

    def test_indices_must_be_in_bounds(self) -> None:
        for index in [(-1, 0), (3, 0), (0, 4)]:
            with self.subTest(index=index), self.assertRaises(IndexError):
                storage_index((3, 4), (4, 1), index)

    def test_invalid_metadata_is_rejected(self) -> None:
        for shape, strides in [((2, 3), (1,)), ((-2,), (1,)), ((2,), (-1,))]:
            with self.subTest(shape=shape, strides=strides):
                with self.assertRaises(ValueError):
                    is_c_contiguous(shape, strides)
        with self.assertRaises(ValueError):
            storage_index((2,), (1,), (0,), offset=-1)
        with self.assertRaises(ValueError):
            storage_index((2,), (1,), ())

    def test_integer_metadata_is_required(self) -> None:
        with self.assertRaises(TypeError):
            is_c_contiguous((2.5,), (1,))

    def test_default_contiguity_cases(self) -> None:
        cases = [
            ((3, 4), (4, 1), True),
            ((4, 3), (1, 4), False),
            ((3, 2), (4, 2), False),
            ((2, 1, 3), (3, 99, 1), True),
            ((4, 3), (0, 1), False),
            ((2, 0, 3), (99, 99, 99), True),
            ((), (), True),
        ]
        for shape, strides, expected in cases:
            with self.subTest(shape=shape, strides=strides):
                self.assertIs(is_c_contiguous(shape, strides), expected)

    def test_pack_follows_logical_order(self) -> None:
        self.assertEqual(
            pack_logical(list(range(12)), (4, 3), (1, 4)),
            [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11],
        )
        self.assertEqual(
            pack_logical(list(range(12)), (3, 2), (4, 1), offset=1),
            [1, 2, 5, 6, 9, 10],
        )

    def test_pack_supports_repeated_reads(self) -> None:
        self.assertEqual(pack_logical([7, 8, 9], (2, 3), (0, 1)), [7, 8, 9] * 2)

    def test_pack_rejects_storage_overrun(self) -> None:
        with self.assertRaises(IndexError):
            pack_logical([1, 2], (3,), (1,))


if __name__ == "__main__":
    unittest.main()
