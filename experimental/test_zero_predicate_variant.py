"""Synthetic-only checks for the isolated source filter guard."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np

from experimental.zero_predicate_variant import (
    Observation, ReadOnlyReplies, ReplayMiss, extracted_runner, filter_pseudoclean_rows,
)
from zeroclean.types import digest


def source_filter(attrs, right_by_attr, funcs, right, evaluate):
    for attr in attrs:
        for idx, val in right_by_attr[attr]:
            passed = sum(evaluate(func, val, attr) == 1 for func in funcs[attr]["clean"])
            if passed / (len(funcs[attr]["clean"]) + 1e-6) < .5:
                right.remove((idx, attr))


class TestFilterGuard(unittest.TestCase):
    def test_empty_predicates_preserve_all_existing_pseudoclean_rows(self):
        right = [(0, "a"), (1, "a"), (1, "b")]
        groups = {"a": [(0, {}), (1, {})], "b": [(1, {})]}
        funcs = {"a": {"clean": []}, "b": {"clean": []}}
        trace = filter_pseudoclean_rows(["a", "b"], groups, funcs, right, lambda *args: self.fail("No predicate should be evaluated"), True)
        self.assertEqual(right, [(0, "a"), (1, "a"), (1, "b")])
        self.assertEqual(trace["a"]["preserved_by_guard"], 2)
        self.assertEqual(trace["b"]["preserved_by_guard"], 1)

    def test_original_empty_predicate_behavior_removes_all(self):
        right = [(0, "a"), (1, "a")]
        trace = filter_pseudoclean_rows(["a"], {"a": [(0, {}), (1, {})]}, {"a": {"clean": []}}, right, lambda *args: 1, False)
        self.assertEqual(right, [])
        self.assertEqual(trace["a"]["removed_rows"], 2)

    def test_nonempty_predicates_reproduce_source_exactly_including_errors(self):
        rng = random.Random(42)
        for _ in range(100):
            count = rng.randint(1, 8)
            funcs = {"a": {"clean": list(range(count))}}
            values = [{"outcomes": [rng.choice([-1, 0, 1]) for _ in range(count)]} for _ in range(20)]
            groups = {"a": list(enumerate(values))}
            expected = [(i, "a") for i in range(20)]
            actual = expected.copy()
            evaluator = lambda func, value, attr: value["outcomes"][func]
            source_filter(["a"], groups, funcs, expected, evaluator)
            trace = filter_pseudoclean_rows(["a"], groups, funcs, actual, evaluator, True)
            self.assertEqual(expected, actual)
            self.assertFalse(trace["a"]["empty_predicate_guard"])

    def test_boundary_retains_source_epsilon_semantics(self):
        # Original epsilon means exactly one of two passes is below .5.
        right = [(0, "a")]
        filter_pseudoclean_rows(["a"], {"a": [(0, {})]}, {"a": {"clean": [1, 0]}}, right, lambda func, *args: func, True)
        self.assertEqual(right, [])

    def test_mixed_attributes_only_empty_condition_changes(self):
        groups = {"empty": [(0, {})], "present": [(0, {})]}
        funcs = {"empty": {"clean": []}, "present": {"clean": ["reject"]}}
        right = [(0, "empty"), (0, "present")]
        filter_pseudoclean_rows(["empty", "present"], groups, funcs, right, lambda *args: 0, True)
        self.assertEqual(right, [(0, "empty")])

    def test_extraction_instruments_one_loop_without_live_module_changes(self):
        from zeroclean import upstream_zeroed
        original = upstream_zeroed.run_source_zeroed
        with tempfile.TemporaryDirectory() as directory:
            runner, report = extracted_runner(Observation(Path(directory), True))
            self.assertEqual(report["replacement_count"], 1)
            self.assertEqual(report["training_observer_count"], 1)
            self.assertEqual(report["model_observer_count"], 1)
            self.assertEqual(report["prefilter_stage_hook_count"], 1)
            self.assertIs(upstream_zeroed.run_source_zeroed, original)
            self.assertIsNot(runner, original)


class TestReadOnlyReplay(unittest.TestCase):
    def test_missing_response_never_creates_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = ReadOnlyReplies(root)
            with self.assertRaisesRegex(ReplayMiss, "stage=synthetic_stage"):
                replay.request("synthetic_stage", "fabricated prompt", {})
            self.assertEqual(list(root.iterdir()), [])

    def test_exact_frozen_response_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("requests", "responses"):
                (root / name).mkdir()
            body = {"stage": "synthetic_stage", "prompt": "fabricated prompt", "metadata": {}, "model": "gpt-6.1-sol", "reasoning_effort": "xhigh"}
            key = digest(body)
            (root / "requests" / f"{key}.json").write_text(json.dumps(body))
            reply = {"request_hash": key, "model": body["model"], "reasoning_effort": body["reasoning_effort"], "response": "fabricated reply"}
            (root / "responses" / f"{key}.json").write_text(json.dumps(reply))
            replay = ReadOnlyReplies(root)
            self.assertEqual(replay.request("synthetic_stage", "fabricated prompt", {}), "fabricated reply")
            self.assertEqual(replay.records[0]["request_hash"], key)


class TestFrozenPrefilterState(unittest.TestCase):
    def test_async_predicate_selection_restored_with_all_other_inputs_identical(self):
        class FabricatedFrame:
            def to_dict(self, orientation):
                return [{"a": "fabricated-value"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "control").mkdir()
            (root / "variant").mkdir()
            control = Observation(root / "control", False)
            arguments = ({"a": {"dirty": [{"a": "injected"}]}}, {"a": [(0, {"a": "fabricated-value"}, 0)]}, [(0, "a")], [], {"a": {}}, {"a": [[0], [0]]}, {(0, 0): {"feature": np.asarray([1., 2.])}}, FabricatedFrame())
            control.prefilter_stage({"a": {"clean": ["source-task-A"], "dirty": []}}, *arguments)
            variant = Observation(root / "variant", True, control.stage_state)
            proposed = {"a": {"clean": ["source-task-B"], "dirty": []}}
            variant.prefilter_stage(proposed, *arguments)
            self.assertEqual(proposed["a"]["clean"], ["source-task-A"])
            self.assertEqual(digest(control.stage_state), digest(variant.stage_state))
            provenance = json.loads((root / "variant/prefilter_state_provenance.json").read_text())
            self.assertTrue(provenance["source_async_predicate_selection_differed"])
            self.assertTrue(provenance["full_prefilter_state_identical_to_control"])
            changed = list(arguments)
            changed[1] = {"a": [(0, {"a": "fabricated-value"}, 1)]}
            with self.assertRaisesRegex(ValueError, "Non-predicate pre-filter source state changed: labels"):
                variant.prefilter_stage(proposed, *changed)


if __name__ == "__main__":
    unittest.main()
