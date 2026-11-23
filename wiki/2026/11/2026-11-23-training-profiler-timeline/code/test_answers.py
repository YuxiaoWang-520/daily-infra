"""Tests check interval semantics and a real scheduled CPU profiler run."""

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from answers import (
    complete_events,
    merge_intervals,
    overlap_summary,
    project_cpu_trace,
    synthetic_trace,
)
from cpu_profile import run_cpu_profile


class IntervalTests(unittest.TestCase):
    def test_merge_nested_adjacent_and_unsorted(self):
        self.assertEqual(
            merge_intervals([(4, 8), (1, 5), (8, 9), (2, 3)], (0, 10)),
            ((1.0, 9.0),),
        )

    def test_clip_both_sides_and_discard_empty(self):
        self.assertEqual(
            merge_intervals([(-2, 2), (8, 12), (3, 3), (11, 12)], (0, 10)),
            ((0.0, 2.0), (8.0, 10.0)),
        )

    def test_empty_inputs(self):
        result = overlap_summary([], [], (0, 10))
        self.assertEqual(result["no_selected_activity_us"], 10)
        self.assertEqual(result["any_activity_us"], 0)

    def test_invalid_windows(self):
        for window in [(1, 1), (2, 1), (0, math.inf), (False, 1)]:
            with (
                self.subTest(window=window),
                self.assertRaises((ValueError, TypeError)),
            ):
                merge_intervals([], window)

    def test_invalid_intervals(self):
        for interval in [(2, 1), (0, math.nan), (True, 2), ("0", 2)]:
            with (
                self.subTest(interval=interval),
                self.assertRaises((ValueError, TypeError)),
            ):
                merge_intervals([interval], (0, 10))

    def test_example_overlap_and_partition(self):
        result = overlap_summary(
            [(2000, 8000), (8000, 11000)], [(5000, 13000)], (0, 15000)
        )
        self.assertEqual(
            result,
            {
                "window_us": 15000.0,
                "compute_us": 9000.0,
                "communication_us": 8000.0,
                "overlap_us": 6000.0,
                "communication_only_us": 2000.0,
                "any_activity_us": 11000.0,
                "no_selected_activity_us": 4000.0,
                "summed_activity_us": 17000.0,
            },
        )

    def test_duplicate_kernels_affect_sum_not_union(self):
        result = overlap_summary([(1, 5), (1, 5)], [(3, 7)], (0, 10))
        self.assertEqual(result["compute_us"], 4)
        self.assertEqual(result["summed_activity_us"], 12)
        self.assertEqual(result["any_activity_us"], 6)

    def test_disjoint_touching_intervals_do_not_overlap(self):
        result = overlap_summary([(0, 4)], [(4, 8)], (0, 10))
        self.assertEqual(result["overlap_us"], 0)
        self.assertEqual(result["communication_only_us"], 4)

    def test_clipping_applies_to_sum_and_union(self):
        result = overlap_summary([(-5, 5)], [(3, 15)], (0, 10))
        self.assertEqual(result["summed_activity_us"], 12)
        self.assertEqual(result["overlap_us"], 2)
        self.assertEqual(result["any_activity_us"], 10)

    def test_translation_invariance(self):
        original = overlap_summary([(1, 6)], [(3, 9)], (0, 10))
        shifted = overlap_summary([(101, 106)], [(103, 109)], (100, 110))
        self.assertEqual(original, shifted)

    def test_reject_lossy_integer_timestamps(self):
        window = (2**53 + 1, 2**53 + 3)
        with self.assertRaises(ValueError):
            overlap_summary([window], [], window)
        with self.assertRaises(ValueError):
            merge_intervals([(0, 10**400)], (0, 10))

    def test_exhaustive_integer_interval_oracle(self):
        # Discrete unit cells give an independent exact oracle for integer endpoints.
        for a in range(5):
            for b in range(a, 6):
                result = overlap_summary([(a, b)], [(2, 5)], (0, 6))
                compute = set(range(a, b))
                comm = set(range(2, 5))
                self.assertEqual(result["overlap_us"], len(compute & comm))
                self.assertEqual(result["any_activity_us"], len(compute | comm))


class TraceTests(unittest.TestCase):
    @staticmethod
    def event(**kwargs):
        return {
            "ph": "X",
            "name": "phase::forward",
            "cat": "user_annotation",
            "ts": 100,
            "dur": 3,
            "pid": 123,
            "tid": 456,
            **kwargs,
        }

    def test_ignore_non_complete_records(self):
        trace = {"traceEvents": [{"ph": "M"}, {"ph": "s"}, self.event()]}
        events = complete_events(trace)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].end_us, 103)

    def test_zero_duration_is_valid(self):
        self.assertEqual(
            complete_events({"traceEvents": [self.event(dur=0)]})[0].end_us, 100
        )

    def test_reject_malformed_complete_events(self):
        for change in [
            {"dur": -1},
            {"ts": math.inf},
            {"dur": True},
            {"name": 42},
            {"ts": 1e308, "dur": 1e308},
        ]:
            with (
                self.subTest(change=change),
                self.assertRaises((ValueError, TypeError)),
            ):
                complete_events({"traceEvents": [self.event(**change)]})

    def test_reject_missing_event_fields_or_wrong_container(self):
        bad = self.event()
        del bad["dur"]
        for trace in [
            {},
            {"traceEvents": {}},
            {"traceEvents": [bad]},
            {"traceEvents": [None]},
        ]:
            with self.subTest(trace=trace), self.assertRaises((ValueError, TypeError)):
                complete_events(trace)

    def test_projection_drops_metadata_args_and_remaps_lanes(self):
        trace = {
            "host": "private-host",
            "traceEvents": [
                {"ph": "M", "args": {"path": "/private/example"}},
                self.event(args={"secret": "must-disappear"}),
                self.event(ts=104, cat="cpu_op", name="aten::mm"),
                self.event(ts=110, cat="kernel", name="synthetic_gpu"),
            ],
        }
        projected, metadata = project_cpu_trace(trace)
        events = projected["traceEvents"]
        self.assertEqual([e["ts"] for e in events], [0, 4])
        self.assertEqual([e["dur"] for e in events], [3, 3])
        self.assertEqual([(e["pid"], e["tid"]) for e in events], [(1, 1), (1, 1)])
        self.assertNotIn("must-disappear", json.dumps(projected))
        self.assertNotIn("private-host", json.dumps(projected))
        self.assertEqual(metadata["retained_complete_events"], 2)

    def test_projection_empty_cpu_selection(self):
        projected, metadata = project_cpu_trace({"traceEvents": []})
        self.assertEqual(projected["traceEvents"], [])
        self.assertEqual(metadata["timestamp_origin_us"], 0)

    def test_projection_preserves_raw_duration_at_large_timestamps(self):
        raw = self.event(ts=1e15, dur=0.01)
        projected, _ = project_cpu_trace({"traceEvents": [raw]})
        self.assertEqual(projected["traceEvents"][0]["dur"], 0.01)

    def test_projection_rejects_rebase_overflow(self):
        for rows in [
            [self.event(ts=-1e308), self.event(ts=1e308)],
            [self.event(ts=-1e308, dur=1), self.event(ts=0, dur=1e308)],
        ]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                project_cpu_trace({"traceEvents": rows})

    def test_synthetic_gpu_is_explicitly_labeled(self):
        trace = synthetic_trace()
        self.assertEqual(trace["provenance"], "synthetic; not a hardware measurement")
        events = complete_events(trace)
        compute = [
            (e.start_us, e.end_us) for e in events if e.name.startswith("compute_")
        ]
        comm = [(e.start_us, e.end_us) for e in events if e.name == "communication"]
        result = overlap_summary(compute, comm, (0, 15000))
        self.assertEqual(result["overlap_us"], 6000)


class RealCPUProfileTests(unittest.TestCase):
    def test_real_schedule_trace_and_training_equivalence(self):
        with TemporaryDirectory() as folder:
            result = run_cpu_profile(Path(folder))
            self.assertEqual(result["profiled_steps"], [2, 3])
            self.assertEqual(result["callback_step_numbers"], [4])
            self.assertEqual(result["losses_match"], True)
            self.assertEqual(result["weight_max_abs_error"], 0)
            self.assertEqual(result["gpu_executed"], False)
            for phase in ["input", "forward", "loss", "backward", "update"]:
                self.assertEqual(result["phase_counts"]["phase::" + phase], 2)
            for event in result["forward_samples"]:
                self.assertGreater(event["cpu_children"], 0)
                self.assertGreaterEqual(event["self_cpu_us"], 0)
                self.assertLessEqual(event["self_cpu_us"], event["cpu_total_us"])
            trace = json.loads((Path(folder) / "cpu_trace.json").read_text())
            self.assertTrue(complete_events(trace))
            self.assertNotIn("args", trace["traceEvents"][0])
            self.assertGreater(result["capture_wall_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
