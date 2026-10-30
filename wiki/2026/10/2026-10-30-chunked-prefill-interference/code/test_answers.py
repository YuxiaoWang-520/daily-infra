"""Independent SDPA comparison, hand calculations and budget invariants."""

import os
import unittest
from importlib import import_module
from itertools import product

import torch
from torch.nn import functional as F

torch.set_num_threads(1)
implementation = import_module(os.environ.get("EXERCISE_IMPL", "answers"))
chunked_attention = implementation.chunked_attention
schedule_prefill = implementation.schedule_prefill


def full_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None], dropout_p=0.0, is_causal=True
    )[0, 0]


class AttentionTests(unittest.TestCase):
    def inputs(self, length=8, width=4, dtype=torch.float64):
        generator = torch.Generator(device="cpu").manual_seed(123)
        return tuple(
            torch.randn(length, width, dtype=dtype, generator=generator, device="cpu")
            for _ in range(3)
        )

    def test_uniform_attention_has_hand_computed_prefix_means(self) -> None:
        q = k = torch.zeros(8, 2, dtype=torch.float64)
        v = torch.arange(8, dtype=torch.float64).reshape(8, 1)
        result = chunked_attention(q, k, v, 3)
        expected = (torch.arange(8, dtype=torch.float64) / 2).reshape(8, 1)
        torch.testing.assert_close(result, expected, rtol=0, atol=1e-15)

    def test_fp64_matches_full_sdpa_across_chunk_boundaries(self) -> None:
        for length in (1, 2, 7, 17):
            q, k, v = self.inputs(length)
            expected = full_sdpa(q, k, v)
            for chunk_size in (1, 2, 3, length, length + 3):
                with self.subTest(length=length, chunk_size=chunk_size):
                    torch.testing.assert_close(
                        chunked_attention(q, k, v, chunk_size),
                        expected,
                        rtol=1e-12,
                        atol=1e-12,
                    )

    def test_fp32_matches_with_explicit_tolerance(self) -> None:
        q, k, v = self.inputs(11, 5, torch.float32)
        torch.testing.assert_close(
            chunked_attention(q, k, v, 4), full_sdpa(q, k, v), rtol=1e-5, atol=1e-6
        )

    def test_noncontiguous_inputs(self) -> None:
        tensors = tuple(tensor[:, ::2] for tensor in self.inputs(9, 8))
        self.assertTrue(all(not tensor.is_contiguous() for tensor in tensors))
        torch.testing.assert_close(
            chunked_attention(*tensors, 4), full_sdpa(*tensors), rtol=1e-12, atol=1e-12
        )

    def test_value_width_may_differ_from_query_width(self) -> None:
        q, k, _ = self.inputs(6, 5)
        v = torch.arange(18, dtype=torch.float64).reshape(6, 3)
        result = chunked_attention(q, k, v, 2)
        self.assertEqual(result.shape, (6, 3))
        torch.testing.assert_close(result, full_sdpa(q, k, v), rtol=1e-12, atol=1e-12)

    def test_future_values_do_not_leak_to_earlier_queries(self) -> None:
        q = k = torch.zeros(6, 2, dtype=torch.float64)
        v = torch.tensor(
            [[0.0], [2.0], [4.0], [1000.0], [2000.0], [3000.0]], dtype=torch.float64
        )
        result = chunked_attention(q, k, v, 4)
        torch.testing.assert_close(
            result[:3, 0], torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        )

    def test_local_triangular_mask_is_a_real_counterexample(self) -> None:
        q = k = torch.zeros(6, 2, dtype=torch.float64)
        v = torch.arange(6, dtype=torch.float64).reshape(6, 1)
        correct = chunked_attention(q, k, v, 3)[3:]
        wrong_rectangular = full_sdpa(q[3:], k, v)
        wrong_without_prefix = full_sdpa(q[3:], k[3:], v[3:])
        self.assertEqual(correct[0].item(), 1.5)
        self.assertEqual(wrong_rectangular[0].item(), 0.0)
        self.assertEqual(wrong_without_prefix[0].item(), 3.0)

    def test_inputs_are_not_modified(self) -> None:
        tensors = self.inputs()
        saved = tuple(tensor.clone() for tensor in tensors)
        first = chunked_attention(*tensors, 3)
        for tensor, original in zip(tensors, saved):
            torch.testing.assert_close(tensor, original, rtol=0, atol=0)
        torch.testing.assert_close(
            chunked_attention(*tensors, 3), first, rtol=0, atol=0
        )

    def test_explicit_cpu_inputs_ignore_default_device(self) -> None:
        tensors = self.inputs()
        expected = full_sdpa(*tensors)
        with torch.device("meta"):
            result = chunked_attention(*tensors, 3)
        self.assertEqual(result.device.type, "cpu")
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)

    def test_invalid_chunk_sizes(self) -> None:
        for chunk_size in (0, -1, True, 2.5):
            with self.subTest(chunk_size=chunk_size), self.assertRaises(ValueError):
                chunked_attention(*self.inputs(), chunk_size)

    def test_invalid_shapes(self) -> None:
        q, k, v = self.inputs()
        for tensors in (
            (q[0], k, v),
            (q, k[:-1], v),
            (q, k[:, :-1], v),
            (q[:0], k[:0], v[:0]),
            (q[:, :0], k[:, :0], v),
            (q, k, v[:, :0]),
        ):
            with (
                self.subTest(shapes=[x.shape for x in tensors]),
                self.assertRaises(ValueError),
            ):
                chunked_attention(*tensors, 3)

    def test_invalid_dtype_device_or_nonfinite_values(self) -> None:
        q, k, v = self.inputs()
        bad_inputs = (
            (q.float(), k, v),
            (q.half(), k.half(), v.half()),
            (q.long(), k.long(), v.long()),
            (torch.empty(q.shape, device="meta"), k, v),
        )
        for tensors in bad_inputs:
            with (
                self.subTest(dtype=tensors[0].dtype, device=tensors[0].device),
                self.assertRaises(ValueError),
            ):
                chunked_attention(*tensors, 3)
        for index, value in product(
            range(3), (float("nan"), float("inf"), float("-inf"))
        ):
            tensors = [t.clone() for t in (q, k, v)]
            tensors[index][0, 0] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                chunked_attention(*tensors, 3)


class SchedulingTests(unittest.TestCase):
    def test_full_prefill_uses_one_long_mixed_round(self) -> None:
        result = schedule_prefill(16, 2, 4, 18)
        self.assertEqual([r.prefill_tokens for r in result.rounds], [16, 0, 0, 0])
        self.assertEqual(result.prompt_first_token_ms, 18.0)
        self.assertEqual(result.decode_output_times_ms, ((18.0, 20.0, 22.0, 24.0),) * 2)
        self.assertEqual(result.rounds[-1].end_ms, 24.0)

    def test_medium_budget_interleaves_four_chunks(self) -> None:
        result = schedule_prefill(16, 2, 4, 6)
        self.assertEqual([r.prefill_tokens for r in result.rounds], [4, 4, 4, 4])
        self.assertEqual(result.prompt_first_token_ms, 24.0)
        self.assertEqual(result.decode_output_times_ms, ((6.0, 12.0, 18.0, 24.0),) * 2)

    def test_chunk_size_grows_when_decodes_finish(self) -> None:
        result = schedule_prefill(16, 2, 4, 4)
        self.assertEqual([r.prefill_tokens for r in result.rounds], [2, 2, 2, 2, 4, 4])
        self.assertEqual(result.prompt_first_token_ms, 26.0)
        self.assertEqual(result.decode_output_times_ms, ((4.0, 8.0, 12.0, 16.0),) * 2)

    def test_equal_budget_defers_prefill_until_decodes_finish(self) -> None:
        result = schedule_prefill(16, 2, 4, 2)
        self.assertEqual([r.prefill_tokens for r in result.rounds[:4]], [0, 0, 0, 0])
        self.assertEqual(result.decode_output_times_ms, ((2.0, 4.0, 6.0, 8.0),) * 2)
        self.assertEqual(result.prompt_first_token_ms, 32.0)

    def test_no_decode_requests(self) -> None:
        result = schedule_prefill(10, 0, 1, 4)
        self.assertEqual([r.prefill_tokens for r in result.rounds], [4, 4, 2])
        self.assertEqual(result.prompt_first_token_ms, 13.0)
        self.assertEqual(result.decode_output_times_ms, ())

    def test_short_prompt_outputs_at_its_completion_only(self) -> None:
        result = schedule_prefill(1, 2, 4, 6)
        self.assertEqual(result.prompt_first_token_ms, 3.0)
        self.assertEqual(result.rounds[-1].end_ms, 9.0)
        self.assertEqual(
            [r.computed_prompt_tokens for r in result.rounds], [1, 1, 1, 1]
        )

    def test_invalid_or_infeasible_inputs(self) -> None:
        for args in (
            (0, 2, 4, 6),
            (16, -1, 4, 6),
            (16, 2, 0, 6),
            (16, 2, 4, 0),
            (16, 3, 4, 2),
            (True, 2, 4, 6),
            (16, 2.0, 4, 6),
            (16, 2, 4, True),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                schedule_prefill(*args)

    def test_generated_cases_conserve_work_and_global_budget(self) -> None:
        for prompt, decodes, decode_steps in product((1, 4, 9), (0, 1, 2), (1, 3)):
            for budget in (max(decodes, 1), decodes + 1, decodes + 4):
                with self.subTest(
                    prompt=prompt, decodes=decodes, steps=decode_steps, budget=budget
                ):
                    result = schedule_prefill(prompt, decodes, decode_steps, budget)
                    self.assertEqual(
                        sum(r.prefill_tokens for r in result.rounds), prompt
                    )
                    self.assertEqual(
                        sum(r.decode_count for r in result.rounds),
                        decodes * decode_steps,
                    )
                    self.assertTrue(
                        all(
                            0 < r.prefill_tokens + r.decode_count <= budget
                            for r in result.rounds
                        )
                    )
                    self.assertEqual(
                        [len(ts) for ts in result.decode_output_times_ms],
                        [decode_steps] * decodes,
                    )
                    self.assertEqual(
                        result.rounds[-1].end_ms,
                        len(result.rounds) + 0.5 * decodes * decode_steps + prompt,
                    )
                    previous_end = 0.0
                    computed = 0
                    first_prefill_end = None
                    for step in result.rounds:
                        self.assertEqual(step.start_ms, previous_end)
                        self.assertGreater(step.end_ms, step.start_ms)
                        computed += step.prefill_tokens
                        self.assertEqual(step.computed_prompt_tokens, computed)
                        if computed == prompt and first_prefill_end is None:
                            first_prefill_end = step.end_ms
                        previous_end = step.end_ms
                    self.assertEqual(result.prompt_first_token_ms, first_prefill_end)


if __name__ == "__main__":
    unittest.main()
