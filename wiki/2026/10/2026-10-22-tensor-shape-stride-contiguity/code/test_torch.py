"""Real PyTorch CPU checks for article claims and the head-layout exercise."""

import unittest

import torch

from layout_answers import is_c_contiguous, pack_logical
from torch_answers import merge_heads, split_heads


def same_live_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Diagnostic only: these tests use live, nonempty CPU strided tensors."""
    if a.numel() == 0 or b.numel() == 0:
        raise ValueError("Empty storage pointers are not an aliasing test")
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


class TorchLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.arange(12, dtype=torch.int64).reshape(3, 4)

    def test_transpose_is_a_view_and_shares_mutations(self) -> None:
        t = self.x.t()
        self.assertEqual(t.stride(), (1, 4))
        self.assertTrue(same_live_storage(t, self.x))
        t[2, 1] = 600
        self.assertEqual(self.x[1, 2].item(), 600)

    def test_shape_and_values_do_not_determine_layout(self) -> None:
        y = self.x.t().contiguous().t()
        self.assertTrue(torch.equal(y, self.x))
        self.assertEqual(y.shape, self.x.shape)
        self.assertEqual(y.stride(), (1, 3))
        self.assertFalse(y.is_contiguous())

    def test_noncontiguous_step_slice_can_be_viewed(self) -> None:
        flat = self.x[:, ::2].view(-1)
        self.assertEqual(flat.stride(), (2,))
        self.assertEqual(flat.tolist(), [0, 2, 4, 6, 8, 10])
        self.assertTrue(same_live_storage(flat, self.x))

    def test_crop_cannot_flatten_without_copy(self) -> None:
        crop = self.x[:, 1:3]
        with self.assertRaises(RuntimeError):
            crop.view(-1)
        flat = crop.reshape(-1)
        self.assertEqual(flat.tolist(), [1, 2, 5, 6, 9, 10])
        self.assertFalse(same_live_storage(flat, crop))

    def test_transpose_flatten_has_logical_order(self) -> None:
        with self.assertRaises(RuntimeError):
            self.x.t().view(-1)
        packed = self.x.t().contiguous()
        self.assertEqual(packed.stride(), (3, 1))
        self.assertEqual(
            packed.view(-1).tolist(), [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        )

    def test_contiguous_slice_can_have_nonzero_offset(self) -> None:
        row_slice = self.x[1:]
        self.assertTrue(row_slice.is_contiguous())
        self.assertEqual(row_slice.storage_offset(), 4)
        self.assertEqual(
            row_slice.data_ptr() - self.x.data_ptr(), 4 * self.x.element_size()
        )
        self.assertTrue(same_live_storage(row_slice, self.x))
        self.assertIs(row_slice.contiguous(), row_slice)
        self.assertEqual(
            row_slice.untyped_storage().nbytes(), 12 * self.x.element_size()
        )
        compact = row_slice.clone()
        self.assertEqual(compact.untyped_storage().nbytes(), 8 * self.x.element_size())

    def test_clone_is_not_a_contiguous_conversion(self) -> None:
        clone = self.x.t().clone()
        self.assertEqual(clone.stride(), (1, 4))
        self.assertFalse(clone.is_contiguous())
        self.assertFalse(same_live_storage(clone, self.x))

    def test_channels_last_keeps_logical_dimensions(self) -> None:
        nchw = torch.arange(120).reshape(2, 3, 4, 5)
        nhwc_storage = nchw.contiguous(memory_format=torch.channels_last)
        self.assertEqual(nhwc_storage.shape, nchw.shape)
        self.assertEqual(nhwc_storage.stride(), (60, 1, 15, 3))
        self.assertTrue(torch.equal(nhwc_storage, nchw))
        self.assertFalse(nhwc_storage.is_contiguous())
        self.assertTrue(nhwc_storage.is_contiguous(memory_format=torch.channels_last))

    def test_address_model_matches_torch(self) -> None:
        views = [self.x, self.x.t(), self.x[:, ::2], self.x[:, 1:3], self.x[1:]]
        views += [torch.as_strided(torch.arange(6), (2, 1, 3), (3, 99, 1))]
        views += [torch.arange(3).expand(4, 3), torch.empty(2, 0, 3), torch.tensor(7)]
        for tensor in views:
            with self.subTest(shape=tuple(tensor.shape), stride=tensor.stride()):
                self.assertEqual(
                    is_c_contiguous(tuple(tensor.shape), tensor.stride()),
                    tensor.is_contiguous(),
                )
        for tensor in [self.x, self.x.t(), self.x[:, ::2], self.x[:, 1:3], self.x[1:]]:
            expected = tensor.reshape(-1).tolist()
            self.assertEqual(
                pack_logical(
                    list(range(12)),
                    tuple(tensor.shape),
                    tensor.stride(),
                    tensor.storage_offset(),
                ),
                expected,
            )


class HeadLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.arange(24).reshape(2, 3, 4)

    def test_split_has_correct_coordinates_and_no_copy(self) -> None:
        heads = split_heads(self.x, 2)
        self.assertEqual(tuple(heads.shape), (2, 2, 3, 2))
        self.assertEqual(heads.stride(), (12, 2, 4, 1))
        self.assertTrue(same_live_storage(heads, self.x))
        for b in range(2):
            for h in range(2):
                for s in range(3):
                    for d in range(2):
                        self.assertEqual(
                            heads[b, h, s, d].item(), self.x[b, s, h * 2 + d].item()
                        )

    def test_reshape_cannot_replace_permutation(self) -> None:
        correct = split_heads(self.x, 2)
        wrong = self.x.reshape(2, 2, 3, 2)
        self.assertFalse(torch.equal(correct, wrong))
        self.assertEqual(correct[0, 1, 0, 0].item(), 2)
        self.assertEqual(wrong[0, 1, 0, 0].item(), 6)

    def test_round_trip_can_reuse_storage(self) -> None:
        merged = merge_heads(split_heads(self.x, 2))
        self.assertTrue(torch.equal(merged, self.x))
        self.assertTrue(same_live_storage(merged, self.x))

    def test_materialized_bhsd_requires_copy_on_merge(self) -> None:
        materialized = split_heads(self.x, 2).contiguous()
        self.assertEqual(materialized.stride(), (12, 6, 2, 1))
        merged = merge_heads(materialized)
        self.assertTrue(torch.equal(merged, self.x))
        self.assertFalse(same_live_storage(merged, materialized))

    def test_invalid_contracts_are_rejected(self) -> None:
        for heads in [0, 3, True]:
            with self.subTest(heads=heads), self.assertRaises(ValueError):
                split_heads(self.x, heads)
        with self.assertRaises(ValueError):
            split_heads(self.x.transpose(0, 1), 2)
        with self.assertRaises(ValueError):
            split_heads(self.x[0], 2)
        with self.assertRaises(ValueError):
            split_heads(torch.empty(2, 0, 4), 2)
        with self.assertRaises(ValueError):
            merge_heads(self.x)
        with self.assertRaises(ValueError):
            merge_heads(torch.empty(2, 2, 0, 2))
        quantized = torch.quantize_per_tensor(
            self.x.float(), scale=1.0, zero_point=0, dtype=torch.quint8
        )
        with self.assertRaises(ValueError):
            split_heads(quantized, 2)


if __name__ == "__main__":
    unittest.main()
