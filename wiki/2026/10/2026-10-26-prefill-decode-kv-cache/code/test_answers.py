"""Executable contracts for incremental causal attention and generation."""

import unittest

import torch
import torch.nn.functional as F

from answers import KVCache, TinyDecoder, cached_attention, greedy_generate


class AttentionTests(unittest.TestCase):
    def test_one_query_reads_whole_prefix(self):
        old = KVCache(torch.tensor([[1.0], [2.0]]), torch.tensor([[10.0], [20.0]]))
        output, cache = cached_attention(
            torch.zeros(1, 1), torch.tensor([[3.0]]), torch.tensor([[30.0]]), old
        )
        torch.testing.assert_close(output, torch.tensor([[20.0]]))
        torch.testing.assert_close(cache.k, torch.tensor([[1.0], [2.0], [3.0]]))
        torch.testing.assert_close(cache.v, torch.tensor([[10.0], [20.0], [30.0]]))
        self.assertEqual(old.length, 2)
        wrong = F.scaled_dot_product_attention(
            torch.zeros(1, 1), cache.k, cache.v, is_causal=True, dropout_p=0.0
        )
        self.assertEqual(wrong.item(), 10.0)

    def test_two_new_queries_have_different_causal_boundaries(self):
        old = KVCache(torch.zeros(2, 1), torch.tensor([[10.0], [20.0]]))
        output, _ = cached_attention(
            torch.zeros(2, 1), torch.zeros(2, 1), torch.tensor([[30.0], [40.0]]), old
        )
        torch.testing.assert_close(output, torch.tensor([[20.0], [25.0]]))

    def test_chunks_match_independent_sdpa_and_complete_state(self):
        generator = torch.Generator().manual_seed(11)
        q = torch.randn(7, 3, generator=generator, dtype=torch.float64)
        k = torch.randn(7, 3, generator=generator, dtype=torch.float64)
        v = torch.randn(7, 2, generator=generator, dtype=torch.float64)
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        outputs, cache, start = [], None, 0
        for size in (2, 1, 3, 1):
            old = cache
            output, cache = cached_attention(
                q[start : start + size], k[start : start + size], v[start : start + size], cache
            )
            outputs.append(output)
            start += size
            torch.testing.assert_close(cache.k, k[:start], rtol=0, atol=0)
            torch.testing.assert_close(cache.v, v[:start], rtol=0, atol=0)
            if old is not None:
                torch.testing.assert_close(old.k, k[: old.length], rtol=0, atol=0)
                torch.testing.assert_close(old.v, v[: old.length], rtol=0, atol=0)
        torch.testing.assert_close(torch.cat(outputs), expected, rtol=1e-12, atol=1e-12)

    def test_noncontiguous_float32_and_input_ownership(self):
        q = (torch.arange(24.0) / 30).reshape(4, 6)[:, ::2]
        k = (q * 0.5).requires_grad_()
        v = torch.arange(16.0).reshape(4, 4)[:, ::2]
        self.assertFalse(q.is_contiguous())
        output, cache = cached_attention(q, k, v)
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        torch.testing.assert_close(output, expected)
        self.assertFalse(output.requires_grad)
        self.assertFalse(cache.k.requires_grad)
        before = cache.v.clone()
        v.fill_(1000)
        torch.testing.assert_close(cache.v, before, rtol=0, atol=0)

    def test_empty_chunk_preserves_existing_state(self):
        old = KVCache(torch.ones(2, 3), torch.ones(2, 4))
        output, cache = cached_attention(
            torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 4), old
        )
        self.assertEqual(output.shape, (0, 4))
        torch.testing.assert_close(cache.k, old.k)
        torch.testing.assert_close(cache.v, old.v)
        output, cache = cached_attention(torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 4))
        self.assertEqual(output.shape, (0, 4))
        self.assertEqual(cache.length, 0)

    def test_invalid_shapes_dtypes_and_cache_are_rejected(self):
        good = torch.ones(2, 3)
        bad = [
            (good, good[:1], good, None),
            (good, torch.ones(2, 4), good, None),
            (good, good.double(), good, None),
            (good, good, good, KVCache(torch.ones(1, 3), torch.ones(2, 3))),
            (good, good, good, KVCache(torch.ones(1, 4), torch.ones(1, 3))),
            (good, good, good, KVCache(torch.ones(1, 3), torch.ones(1, 4))),
            (good.long(), good.long(), good.long(), None),
            (good[:, 0], good, good, None),
            (torch.empty(2, 0), torch.empty(2, 0), good, None),
        ]
        for args in bad:
            with self.subTest(shapes=[getattr(x, "shape", None) for x in args]):
                with self.assertRaises((TypeError, ValueError)):
                    cached_attention(*args)

    def test_nonfinite_inputs_and_score_overflow_are_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                cached_attention(torch.tensor([[value]]), torch.ones(1, 1), torch.ones(1, 1))
        with self.assertRaisesRegex(ValueError, "scores overflowed"):
            cached_attention(
                torch.tensor([[1e308]], dtype=torch.float64),
                torch.tensor([[1e308]], dtype=torch.float64),
                torch.ones(1, 1, dtype=torch.float64),
            )


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyDecoder(vocab=13, width=8, layers=2, max_length=16, seed=11)
        self.tokens = torch.tensor([2, 5, 1, 8, 4], dtype=torch.int64)

    def test_each_incremental_step_matches_full_recomputation(self):
        caches = None
        for i in range(len(self.tokens)):
            old = caches
            logits, caches = self.model.cached(self.tokens[i : i + 1], caches)
            full_logits, full_caches = self.model.full(self.tokens[: i + 1])
            torch.testing.assert_close(logits[-1], full_logits[-1], rtol=1e-11, atol=1e-11)
            for layer, cache in enumerate(caches):
                torch.testing.assert_close(cache.k, full_caches[layer].k, rtol=1e-11, atol=1e-11)
                torch.testing.assert_close(cache.v, full_caches[layer].v, rtol=1e-11, atol=1e-11)
                if old is not None:
                    torch.testing.assert_close(cache.k[:-1], old[layer].k, rtol=0, atol=0)
                    torch.testing.assert_close(cache.v[:-1], old[layer].v, rtol=0, atol=0)

    def test_chunked_prefill_and_decode_match_all_logits_and_caches(self):
        expected, expected_caches = self.model.full(self.tokens)
        start, caches, pieces = 0, None, []
        for size in (1, 2, 2):
            output, caches = self.model.cached(self.tokens[start : start + size], caches)
            pieces.append(output)
            start += size
        torch.testing.assert_close(torch.cat(pieces), expected, rtol=1e-11, atol=1e-11)
        for actual, wanted in zip(caches, expected_caches):
            torch.testing.assert_close(actual.k, wanted.k, rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(actual.v, wanted.v, rtol=1e-11, atol=1e-11)

    def test_future_tokens_do_not_change_past_hidden_states(self):
        prefix_logits, prefix_cache = self.model.full(self.tokens[:3])
        full_logits, full_cache = self.model.full(self.tokens)
        torch.testing.assert_close(prefix_logits, full_logits[:3], rtol=1e-11, atol=1e-11)
        for before, after in zip(prefix_cache, full_cache):
            torch.testing.assert_close(before.k, after.k[:3], rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(before.v, after.v[:3], rtol=1e-11, atol=1e-11)

    def test_generation_matches_and_cache_lags_last_sampled_token(self):
        prompt = self.tokens[:3]
        cached = greedy_generate(self.model, prompt, 4, use_cache=True)
        full = greedy_generate(self.model, prompt, 4, use_cache=False)
        torch.testing.assert_close(cached.tokens, full.tokens, rtol=0, atol=0)
        self.assertEqual(cached.forward_lengths, (3, 1, 1, 1))
        self.assertEqual(full.forward_lengths, (3, 4, 5, 6))
        self.assertEqual(len(cached.tokens), 7)
        self.assertTrue(all(c.length == 6 for c in cached.caches))
        _, expected_caches = self.model.full(cached.tokens[:-1])
        for actual, expected in zip(cached.caches, expected_caches):
            torch.testing.assert_close(actual.k, expected.k, rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(actual.v, expected.v, rtol=1e-11, atol=1e-11)

    def test_zero_generation_does_no_forward_or_input_aliasing(self):
        result = greedy_generate(self.model, self.tokens, 0)
        self.assertEqual(result.forward_lengths, ())
        self.assertIsNone(result.caches)
        self.assertNotEqual(result.tokens.data_ptr(), self.tokens.data_ptr())
        torch.testing.assert_close(result.tokens, self.tokens)

    def test_context_limit_counts_forwarded_positions(self):
        model = TinyDecoder(vocab=13, width=8, layers=2, max_length=4)
        self.assertEqual(len(greedy_generate(model, self.tokens[:3], 2).tokens), 5)
        with self.assertRaises(ValueError):
            greedy_generate(model, self.tokens[:3], 3)

    def test_bad_tokens_steps_and_layer_states_are_rejected(self):
        for tokens in (
            torch.tensor([], dtype=torch.int64),
            torch.tensor([-1]),
            torch.tensor([13]),
            torch.tensor([1.0]),
            self.tokens[None],
        ):
            with self.assertRaises((TypeError, ValueError)):
                self.model.full(tokens)
        for steps in (-1, True, 1.5):
            with self.assertRaises((TypeError, ValueError)):
                greedy_generate(self.model, self.tokens, steps)
        _, cache = self.model.full(self.tokens[:2])
        with self.assertRaises(ValueError):
            self.model.cached(self.tokens[2:3], cache[:1])
        with self.assertRaises(ValueError):
            self.model.cached(self.tokens[2:3], (cache[0], KVCache(cache[1].k[:1], cache[1].v[:1])))

    def test_inference_only_and_local_seed(self):
        state = torch.get_rng_state().clone()
        model = TinyDecoder(vocab=13, width=8, layers=2, max_length=16, seed=11)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        logits, caches = model.cached(self.tokens)
        self.assertFalse(logits.requires_grad)
        self.assertTrue(all(not c.k.requires_grad and not c.v.requires_grad for c in caches))
        expected, _ = self.model.full(self.tokens)
        torch.testing.assert_close(logits, expected, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
