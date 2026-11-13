"""CPU regression tests for backend observation and capture-preserving rewrites."""

import dataclasses
import json
import unittest

import torch
from counting_backend import CountingBackend
from solutions import tensor_affine, tensor_branch
from torch._dynamo.exc import Unsupported
from workloads import affine, python_branch, split_affine


def matrix(batch: int = 2, value: float = 1.0) -> torch.Tensor:
    return torch.full((batch, 3), value, dtype=torch.float32, device="cpu")


def scalar(value: float = 2.0) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32, device="cpu")


class BackendTests(unittest.TestCase):
    def test_manual_fx_execution(self):
        graph = torch.fx.symbolic_trace(lambda x: (x * 2 + 1).relu())
        backend = CountingBackend()
        x = matrix()
        executable = backend(graph, [x])
        torch.testing.assert_close(executable(x), graph(x))
        self.assertEqual(len(backend.records), 1)
        self.assertEqual(len(backend.records[0].operators), 3)
        self.assertEqual(backend.records[0].placeholders, ("x",))
        self.assertIn("Tensor", backend.records[0].input_kinds[0])

    def test_multiple_graph_records(self):
        backend = CountingBackend()
        first = torch.fx.symbolic_trace(lambda x: x + 1)
        second = torch.fx.symbolic_trace(lambda x: x * 2)
        backend(first, [matrix()])
        backend(second, [matrix()])
        self.assertEqual(len(backend.records), 2)
        self.assertNotEqual(backend.records[0].code, backend.records[1].code)

    def test_records_are_immutable_values(self):
        backend = CountingBackend()
        backend(torch.fx.symbolic_trace(lambda x: x + 1), [matrix()])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            backend.records[0].code = "changed"

    def test_record_is_json_serializable(self):
        backend = CountingBackend()
        backend(torch.fx.symbolic_trace(lambda x: x + 1), [matrix()])
        encoded = json.dumps(dataclasses.asdict(backend.records[0]))
        self.assertIn("operators", encoded)


class CompileBehaviorTests(unittest.TestCase):
    def setUp(self):
        # Reset only this test process's compiler state between independent cases.
        torch.compiler.reset()
        self.backend = CountingBackend()

    def tearDown(self):
        torch.compiler.reset()

    def compile(self, fn=affine, *, dynamic=False, fullgraph=True):
        return torch.compile(
            fn, backend=self.backend, dynamic=dynamic, fullgraph=fullgraph
        )

    def test_same_metadata_reuses_graph(self):
        compiled = self.compile()
        for value in (1, 4, -2):
            x = matrix(value=value)
            torch.testing.assert_close(compiled(x, 2.0), affine(x, 2.0))
            self.assertEqual(len(self.backend.records), 1)

    def test_static_shape_specializations(self):
        compiled = self.compile(dynamic=False)
        counts = []
        for batch in (2, 4, 5):
            x = matrix(batch)
            torch.testing.assert_close(compiled(x, 2.0), affine(x, 2.0))
            counts.append(len(self.backend.records))
        self.assertEqual(counts, [1, 2, 3])

    def test_automatic_dynamic_shapes(self):
        compiled = self.compile(dynamic=None)
        counts = []
        for batch in (2, 4, 5):
            x = matrix(batch)
            torch.testing.assert_close(compiled(x, 2.0), affine(x, 2.0))
            counts.append(len(self.backend.records))
        self.assertEqual(counts, [1, 2, 2])

    def test_dynamic_shapes_upfront(self):
        compiled = self.compile(dynamic=True)
        for batch in (2, 4, 5):
            x = matrix(batch)
            torch.testing.assert_close(compiled(x, 2.0), affine(x, 2.0))
            self.assertEqual(len(self.backend.records), 1)
        self.assertTrue(
            any("SymInt" in kind for kind in self.backend.records[0].input_kinds)
        )

    def test_dtype_guard(self):
        compiled = self.compile()
        compiled(matrix(), 2.0)
        x = matrix().double()
        torch.testing.assert_close(compiled(x, 2.0), affine(x, 2.0))
        self.assertEqual(len(self.backend.records), 2)

    def test_zero_one_dimensions_have_separate_specializations(self):
        compiled = self.compile(dynamic=True)
        counts = []
        for batch in (0, 1, 2, 4):
            x = matrix(batch)
            torch.testing.assert_close(compiled(x, 2.0), affine(x, 2.0))
            counts.append(len(self.backend.records))
        self.assertEqual(counts, [1, 2, 3, 3])

    def test_stride_guard(self):
        compiled = self.compile()
        x = torch.arange(9, dtype=torch.float32, device="cpu").reshape(3, 3)
        compiled(x, 2.0)
        torch.testing.assert_close(compiled(x.t(), 2.0), affine(x.t(), 2.0))
        self.assertEqual(len(self.backend.records), 2)

    def test_python_scalar_value_guard_and_cache_reuse(self):
        compiled = self.compile()
        x = matrix()
        counts = []
        for value in (2.0, 3.0, 2.0):
            torch.testing.assert_close(compiled(x, value), affine(x, value))
            counts.append(len(self.backend.records))
        self.assertEqual(counts, [1, 2, 2])

    def test_tensor_scalar_changes_reuse_graph(self):
        compiled = self.compile(tensor_affine)
        x = matrix()
        for value in (2.0, 3.0, 2.0):
            scale = scalar(value)
            torch.testing.assert_close(compiled(x, scale), affine(x, value))
            self.assertEqual(len(self.backend.records), 1)

    def test_split_default_two_regions_and_cached_repeat(self):
        compiled = self.compile(split_affine, fullgraph=False)
        x = matrix(value=-2)
        for _ in range(2):
            torch.testing.assert_close(compiled(x), (x * 2).relu() + 1)
            self.assertEqual(len(self.backend.records), 2)

    def test_fullgraph_rejects_explicit_break(self):
        compiled = self.compile(split_affine)
        with self.assertRaises(Unsupported) as error:
            compiled(matrix())
        self.assertIn("graph_break", str(error.exception))
        self.assertEqual(len(self.backend.records), 0)

    def test_fullgraph_rejects_python_data_branch(self):
        compiled = self.compile(python_branch)
        with self.assertRaises(Unsupported) as error:
            compiled(matrix())
        self.assertIn("Data-dependent", str(error.exception))
        self.assertEqual(len(self.backend.records), 0)

    def test_tensor_branch_handles_sign_changes(self):
        compiled = self.compile(tensor_branch)
        for value in (1, -1, 0):
            x = matrix(value=value)
            expected = x * 2 + (1 if value > 0 else -1)
            torch.testing.assert_close(compiled(x, scalar()), expected)
            self.assertEqual(len(self.backend.records), 1)

    def test_tensor_branch_with_mixed_input(self):
        compiled = self.compile(tensor_branch)
        x = torch.tensor([[4.0, -2.0, -1.0], [0.0, 0.0, 0.0]], device="cpu")
        torch.testing.assert_close(compiled(x, scalar(3)), x * 3 + 1)

    def test_combined_shape_scale_and_sign_changes(self):
        compiled = self.compile(tensor_branch, dynamic=True)
        for batch, value, scale in ((2, 1, 2), (4, -1, 3), (5, 0, -2)):
            x = matrix(batch, value)
            expected = x * scale + (1 if value > 0 else -1)
            torch.testing.assert_close(compiled(x, scalar(scale)), expected)
            self.assertEqual(len(self.backend.records), 1)

    def test_tensor_affine_float64(self):
        compiled = self.compile(tensor_affine)
        x = matrix().double()
        scale = scalar(-0.5).double()
        torch.testing.assert_close(compiled(x, scale), (x * scale + 1).relu())
        self.assertEqual(compiled(x, scale).dtype, torch.float64)

    def test_original_branch_eager_reference(self):
        for value in (1, -1, 0):
            x = matrix(value=value)
            torch.testing.assert_close(
                python_branch(x), x * 2 + (1 if value > 0 else -1)
            )

    def test_rewrites_with_empty_and_unit_shapes(self):
        for function in (tensor_affine, tensor_branch):
            compiled = self.compile(function, dynamic=True)
            for shape in ((0, 3), (2, 0), (0, 0), (1, 3)):
                x = torch.zeros(shape, dtype=torch.float64, device="cpu")
                scale = torch.tensor(-2.0, dtype=torch.float64, device="cpu")
                expected = (
                    torch.ones_like(x)
                    if function is tensor_affine
                    else -torch.ones_like(x)
                )
                torch.testing.assert_close(compiled(x, scale), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
