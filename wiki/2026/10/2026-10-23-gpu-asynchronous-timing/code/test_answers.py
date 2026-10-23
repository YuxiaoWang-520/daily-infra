"""Logical timelines and timer API ordering; no CUDA hardware is exercised."""

import math
import unittest
from unittest.mock import patch

from answers import QueueModel, benchmark_cuda


class QueueTests(unittest.TestCase):
    def test_host_return_precedes_device_completion(self):
        model = QueueModel(0.1)
        task = model.submit("s", 3.0)
        self.assertAlmostEqual(model.host_ms, 0.1)
        self.assertAlmostEqual(task.start_ms, 0.1)
        self.assertAlmostEqual(task.end_ms, 3.1)
        model.synchronize()
        self.assertAlmostEqual(model.host_ms, 3.1)

    def test_same_stream_serializes(self):
        model = QueueModel(0.1)
        first = model.submit("s", 3.0)
        second = model.submit("s", 2.0)
        self.assertEqual(second.start_ms, first.end_ms)
        self.assertAlmostEqual(second.end_ms, 5.1)

    def test_independent_streams_can_overlap_in_model(self):
        model = QueueModel(0.1)
        first = model.submit("a", 3.0)
        second = model.submit("b", 2.0)
        self.assertLess(second.start_ms, first.end_ms)
        model.synchronize()
        self.assertAlmostEqual(model.host_ms, 3.1)

    def test_wait_is_device_dependency_not_host_wait(self):
        model = QueueModel(0.1)
        producer = model.submit("a", 3.0)
        event = model.event("a")
        model.wait_event("b", event)
        self.assertAlmostEqual(model.host_ms, 0.1)
        consumer = model.submit("b", 2.0)
        self.assertEqual(consumer.start_ms, producer.end_ms)

    def test_event_captures_only_prior_work(self):
        model = QueueModel(0.1)
        first = model.submit("a", 3.0)
        event = model.event("a")
        later = model.submit("a", 10.0)
        model.wait_event("b", event)
        consumer = model.submit("b", 1.0)
        self.assertEqual(consumer.start_ms, first.end_ms)
        self.assertLess(consumer.end_ms, later.end_ms)

    def test_empty_and_invalid_model(self):
        model = QueueModel(0.0)
        model.synchronize()
        self.assertEqual(model.host_ms, 0.0)
        for bad in [-1, math.nan, math.inf, True]:
            with self.assertRaises(ValueError):
                QueueModel(bad)
            with self.assertRaises(ValueError):
                model.submit("s", bad)
        with self.assertRaises(ValueError):
            model.submit("", 1.0)


class TimingTests(unittest.TestCase):
    def test_no_cuda_is_an_explicit_error(self):
        calls = []
        with patch("answers.torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                benchmark_cuda(lambda: calls.append("fn"))
        self.assertEqual(calls, [])

    def test_timer_order_warmup_wait_and_millisecond_conversion(self):
        calls = []

        class FakeEvent:
            def __init__(self, enable_timing):
                self.assert_enabled = enable_timing
                self.completed = False

            def record(self):
                calls.append("record")

            def synchronize(self):
                self.completed = True
                calls.append("wait_end")

            def elapsed_time(self, end):
                if not self.assert_enabled or not end.completed:
                    raise AssertionError("timing must be enabled and end complete")
                calls.append("elapsed")
                return 20.0

        with (
            patch("answers.torch.cuda.is_available", return_value=True),
            patch("answers.torch.cuda.Event", FakeEvent),
            patch(
                "answers.torch.cuda.synchronize", lambda: calls.append("device_sync")
            ),
        ):
            result = benchmark_cuda(
                lambda: calls.append("fn"), warmup=2, iterations=4, repeats=2
            )
        self.assertEqual(result["seconds_per_call"], [0.005, 0.005])
        self.assertEqual(result["median_seconds"], 0.005)
        self.assertEqual(calls[:3], ["fn", "fn", "device_sync"])
        batch = ["record"] + ["fn"] * 4 + ["record", "wait_end", "elapsed"]
        self.assertEqual(calls[3:], batch * 2)

    def test_invalid_loop_counts_before_cuda_access(self):
        for options in [{"warmup": -1}, {"iterations": 0}, {"repeats": True}]:
            with self.assertRaises(ValueError):
                benchmark_cuda(lambda: None, **options)


if __name__ == "__main__":
    unittest.main()
