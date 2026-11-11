"""Fictional fixtures only. Never loads a benchmark table or invokes evaluation."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from zeroclean.data import DirtyTable, load_dirty
from zeroclean.types import Prediction
from experimental.unified_loop_v5 import consensus, detection, repair, run_study as runner


def fictional_table(rows=(("e1", "bad"), ("e1", "good"), ("e1", "good")), columns=("entity", "value")):
    return DirtyTable("fictional", tuple(columns), tuple(tuple(row) for row in rows), "fictional-dirty-fingerprint")


def fictional_pattern(value):
    value = re.sub(r"[A-Z]+", "A", value)
    value = re.sub(r"[a-z]+", "a", value)
    return re.sub(r"\d+", "9", value)


class FictionalIndex:
    def __init__(self, table, dependencies=()):
        self.table = table
        self.counts = [Counter(row[j] for row in table.rows) for j in range(table.shape[1])]
        self.patterns = [Counter(fictional_pattern(row[j]) for row in table.rows) for j in range(table.shape[1])]

    def conditional(self, i, j, value, row=None):
        count = self.counts[j].get(value, 0) - int(value == self.table.rows[i][j])
        return max(count, 0) / max(self.table.shape[0] - 1, 1), []

    def vicinity(self, i, j, value, row=None):
        return float(self.counts[j].get(value, 0) - int(value == self.table.rows[i][j]))

    def domain_support(self, i, j, value):
        return math.log1p(max(self.counts[j].get(value, 0) - int(value == self.table.rows[i][j]), 0))


class FictionalSandbox:
    def __init__(self):
        self.validation_batches = []
        self.evaluation_batches = []

    @staticmethod
    def value(rule, row):
        tag = rule.get("test", "false")
        if tag == "true":
            return 1
        if tag == "false":
            return 0
        if tag == "bad":
            return int(row.get("value") == "bad")
        if tag == "forbidden":
            return int(row.get("value") == "forbidden")
        if tag == "runtime":
            return -1 if row.get("value") == "runtime" else 0
        if tag == "errors":
            return -1 if row.get("value") == "E" else 0
        if tag == "pair":
            return int(row.get("a") == "y" and row.get("b") == "y")
        raise AssertionError(f"Unknown fictional rule tag: {tag}")

    def validate_rules(self, rules, rows, max_error_rate=.05):
        self.validation_batches.append(len(rows))
        reports, accepted = [], []
        for rule in rules:
            errors = sum(self.value(rule, row) == -1 for row in rows)
            passed = bool(rows) and errors < max_error_rate * len(rows)
            reports.append({"error_count": errors, "status": "accepted" if passed else "error_rate", "error_samples": []})
            if passed:
                accepted.append(rule)
        return accepted, reports

    def evaluate_rules(self, rules, rows):
        self.evaluation_batches.append(len(rows))
        return [[self.value(rule, row) for rule in rules] for row in rows]


class SpyDetection:
    DetectionConfig = detection.DetectionConfig

    def __init__(self):
        self.calls = []

    def gate_detection(self, table, source, empty, violations, metadata, config=None, *, contextual_low_support=None):
        self.calls.append({"rows": table.rows, "source": source.copy(), "empty": empty.copy(),
                           "violations": violations.copy(), "metadata": dict(metadata),
                           "contextual": contextual_low_support.copy()})
        return {"probabilities": empty.copy(), "mask": empty > .5, "evidence": [],
                "diagnostics": {"retained_positive_evidence": [], "fictional": True}}


def fictional_runtime(*, real_detection=False, real_consensus=False):
    return SimpleNamespace(np=np, DirtyIndex=FictionalIndex, pattern=fictional_pattern,
                           Prediction=Prediction, sandbox=FictionalSandbox(), load_dirty=load_dirty,
                           detection=detection if real_detection else SpyDetection(),
                           consensus=consensus if real_consensus else SimpleNamespace(DEFAULT_CONFIG=consensus.DEFAULT_CONFIG,
                               discover_consensus=lambda table: {"candidates": {}, "diagnostics": {"fictional": True}}),
                           repair=repair)


def probability_body(values=((.1, .9), (.1, .1), (.1, .1)), repairs=()):
    return {"probabilities": [list(row) for row in values], "repairs": list(repairs),
            "metadata": {"decision_threshold": .5, "decision_threshold_operator": ">", "fictional": True}}


class RunnerIntegrationTests(unittest.TestCase):
    def test_seed_changes_cannot_change_detection_inputs(self):
        table, runtime = fictional_table(), fictional_runtime()
        source = empty = np.asarray(probability_body()["probabilities"])
        pass2 = {"rules": [], "dependencies": []}
        first = runner.infer_table(table, source, empty, {"fictional": True}, {}, pass2, {}, runtime)
        second = runner.infer_table(table, source, empty, {"fictional": True}, {(0, 1): "good"}, pass2, {}, runtime)
        np.testing.assert_array_equal(first["mask"], second["mask"])
        np.testing.assert_array_equal(first["probabilities"], second["probabilities"])
        a, b = runtime.detection.calls
        for field in ("source", "empty", "violations", "contextual"):
            np.testing.assert_array_equal(a[field], b[field])
        self.assertEqual(a["rows"], b["rows"])
        self.assertEqual(a["metadata"], b["metadata"])
        self.assertEqual(second["repairs"], {(0, 1): "good"})
        self.assertFalse(second["mask"].flags.writeable)

    def test_combined_row_rollback_keeps_frozen_detection(self):
        table = fictional_table((("x", "x"), ("y", "y"), ("y", "y")), ("a", "b"))
        runtime = fictional_runtime()
        scores = np.asarray([[.9, .9], [.1, .1], [.1, .1]])
        result = runner.infer_table(table, scores, scores, {}, {(0, 0): "y", (0, 1): "y"},
                                    {"rules": [{"id": "pair", "target": ["a", "b"], "test": "pair"}]}, {}, runtime)
        self.assertEqual(result["repairs"], {})
        self.assertEqual(len([d for d in result["decisions"] if d["status"] == "selected_pending_row_guard"]), 2)
        self.assertTrue(result["row_guard"][0]["rolled_back"])
        self.assertTrue(result["row_guard"][0]["new_rule_violation"])
        np.testing.assert_array_equal(result["mask"], scores > .5)
        np.testing.assert_array_equal(result["probabilities"], scores)

    def test_candidate_cap_priority_order_and_late_exact_provenance_merge(self):
        table = fictional_table()
        mask = np.asarray([[False, True], [False, False], [False, False]])
        group = {(0, 1): [{"value": "peer", "origin": "key_source_consensus", "actionable": True,
                           "score": .95, "agreement": 1., "margin": 1., "peer_source_count": 2,
                           "sources": ["fictional-source-A"], "peer_rows": [1, 2], "evidence": {}}]}
        cts = {(0, 1): {"retrieved_row_indices": [2, 1, 0], "candidates": [
            {"value": " first ", "origin": "fixture_cts"}, {"value": "seed", "origin": "fixture_cts"},
            {"value": "over cap", "origin": "fixture_cts"}, {"value": "peer", "origin": "fixture_cts"}]}}
        pool, diagnostics = runner.build_candidate_pool(table, mask, group, {(0, 1): "seed"}, cts)
        self.assertEqual([item["value"] for item in pool[(0, 1)]], ["peer", "seed", " first "])
        self.assertEqual(pool[(0, 1)][0]["origins"], ["key_source_consensus", "cts", "fixture_cts"])
        self.assertEqual(pool[(0, 1)][1]["origins"], ["seed", "cts", "fixture_cts"])
        self.assertEqual(len(pool[(0, 1)][0]["provenance_records"]), 2)
        self.assertEqual(pool[(0, 1)][0]["peer_rows"], [1, 2])
        self.assertTrue(all(item["cts_peer_rows"] == [1, 2] for item in pool[(0, 1)]))
        self.assertEqual(pool[(0, 1)][0]["sources"], ["fictional-source-A"])
        self.assertEqual(diagnostics["distinct_proposals_over_cap"], 1)
        self.assertEqual(diagnostics["exact_duplicate_records_merged"], 2)

    def test_contextual_support_requires_actionable_nonidentity_group_proposal(self):
        table = fictional_table()
        proposals = {(0, 1): [{"value": "good", "origin": "key_source_consensus", "actionable": False}],
                     (1, 1): [{"value": "good", "origin": "key_source_consensus", "actionable": True}],
                     (2, 1): [{"value": "other", "origin": "key_source_consensus", "actionable": True}]}
        result = runner.contextual_mask(table, proposals, np)
        self.assertFalse(result[0, 1])
        self.assertFalse(result[1, 1])
        self.assertTrue(result[2, 1])

    def test_all_true_criteria_excluded_only_from_detection_false_guards_remain(self):
        table, runtime = fictional_table(), fictional_runtime()
        rules = [{"id": "true", "target": "value", "test": "true"},
                 {"id": "false", "target": "value", "test": "forbidden"},
                 {"id": "specific", "target": "value", "test": "bad"}]
        accepted, original, violations, diagnostics = runner.validate_and_evaluate_rules(table, {"rules": rules}, runtime)
        self.assertEqual(len(accepted), 3)
        np.testing.assert_array_equal(violations[:, 1], [True, False, False])
        self.assertEqual(diagnostics["applicability"][0]["detection_status"], "nondiscriminating_all_true")
        self.assertTrue(diagnostics["applicability"][1]["observed_constant_false_guard"])
        pool = {(0, 1): [{"value": "forbidden"}]}
        runner.add_counterfactual_features(table, pool, accepted, original, FictionalIndex(table), runtime)
        self.assertTrue(pool[(0, 1)][0]["new_rule_violation"])

    def test_runtime_counterfactual_abstains(self):
        table, runtime = fictional_table(), fictional_runtime()
        rules, original, *_ = runner.validate_and_evaluate_rules(table, {"rules": [{"id": "runtime", "target": "value", "test": "runtime"}]}, runtime)
        pool = {(0, 1): [{"value": "runtime"}]}
        runner.add_counterfactual_features(table, pool, rules, original, FictionalIndex(table), runtime)
        self.assertTrue(pool[(0, 1)][0]["counterfactual_error"])
        self.assertFalse(pool[(0, 1)][0]["criterion_valid"])

    def test_four_feature_deltas_use_original_dirty_counts(self):
        table, runtime = fictional_table(), fictional_runtime()
        rules, original, *_ = runner.validate_and_evaluate_rules(table, {"rules": [{"id": "bad", "target": "value", "test": "bad"}]}, runtime)
        index = FictionalIndex(table)
        before_counts = [counter.copy() for counter in index.counts]
        pool = {(0, 1): [{"value": "good"}]}
        runner.add_counterfactual_features(table, pool, rules, original, index, runtime)
        candidate = pool[(0, 1)][0]
        np.testing.assert_allclose(candidate["delta_features"], [1., 1., math.tanh(.5), math.tanh(math.log(3) / 4)])
        self.assertEqual(candidate["rule_gain"], 1.)
        self.assertFalse(candidate["new_rule_violation"])
        self.assertEqual(index.counts, before_counts)

    def test_validation_aggregates_batches_before_strict_five_percent_gate(self):
        for errors, accepted_count in ((51, 1), (55, 0)):
            table = fictional_table(tuple(("entity", "E" if i < errors else "good") for i in range(1100)))
            runtime = fictional_runtime()
            accepted, _, _, diagnostics = runner.validate_and_evaluate_rules(table, {"rules": [{"id": "errors", "target": "value", "test": "errors"}]}, runtime)
            self.assertEqual(len(accepted), accepted_count)
            self.assertEqual(diagnostics["validation"][0]["error_count"], errors)
            self.assertEqual(runtime.sandbox.validation_batches, [1000, 100])
            self.assertLessEqual(max(runtime.sandbox.evaluation_batches, default=0), 1000)

    def test_counterfactual_batches_never_exceed_one_thousand(self):
        table = fictional_table(tuple(("entity", "bad") for _ in range(1001)))
        runtime = fictional_runtime()
        rules = [{"id": "false", "target": "value", "test": "false"}]
        pool = {(i, 1): [{"value": "good"}] for i in range(1001)}
        runner.add_counterfactual_features(table, pool, rules, np.zeros((1001, 1), dtype=np.int8), FictionalIndex(table), runtime)
        self.assertEqual(runtime.sandbox.evaluation_batches, [1000, 1])

    def test_format_uncertainty_and_exact_pattern_support(self):
        empty = fictional_table((("entity", ""), ("entity", "")))
        result = runner.candidate_format(empty, 0, 1, "new", FictionalIndex(empty), fictional_pattern)
        self.assertFalse(result["candidate_format_valid"])
        self.assertEqual(result["format_status"], "uncertain_no_observed_dirty_grammar")
        table = fictional_table((("entity", "13 oz"), ("entity", "0013"), ("entity", "17")))
        index = FictionalIndex(table)
        valid = runner.candidate_format(table, 0, 1, "0013", index, fictional_pattern)
        self.assertTrue(valid["candidate_format_valid"])
        self.assertEqual(valid["dirty_pattern_exemplars"], 2)
        self.assertFalse(runner.candidate_format(table, 0, 1, "13\n", index, fictional_pattern)["candidate_format_valid"])

    def test_real_module_integration_on_fictional_multisource_clocks(self):
        rows = (("E1", "A", "09:11"), ("E1", "B", "09:10"), ("E1", "C", "09:10"),
                ("E2", "A", "11:10"), ("E2", "B", "11:10"), ("E2", "C", "11:10"),
                ("E3", "A", "13:10"), ("E3", "B", "13:10"), ("E3", "C", "13:10"))
        table = fictional_table(rows, ("entity_id", "source", "event_time"))
        runtime = fictional_runtime(real_detection=True, real_consensus=True)
        scores = np.full(table.shape, .1)
        scores[0, 2] = .9
        result = runner.infer_table(table, scores, scores, {}, {}, {"rules": []}, {}, runtime)
        self.assertEqual(result["repairs"], {(0, 2): "09:10"})
        candidate = result["pool"][(0, 2)][0]
        self.assertEqual(candidate["peer_rows"], [1, 2])
        self.assertEqual(candidate["sources"], ["B", "C"])
        self.assertTrue(candidate["consensus"]["actionable"])
        self.assertIn("evidence", candidate["consensus"])

    def test_repairs_outside_mask_or_synthesized_values_rejected(self):
        table, runtime = fictional_table(), fictional_runtime()
        mask = np.zeros(table.shape, dtype=bool)
        for repairs in ({(0, 1): "good"}, {(0, 1): "bad"}):
            with self.assertRaises(ValueError):
                runner.combined_row_guard(table, mask, repairs, {}, [], np.zeros((3, 0)), runtime)
        mask[0, 1] = True
        with self.assertRaises(ValueError):
            runner.combined_row_guard(table, mask, {(0, 1): "synthesized"}, {(0, 1): [{"value": "good"}]}, [], np.zeros((3, 0)), runtime)


class CarrierValidationTests(unittest.TestCase):
    def test_empty_repair_map_is_never_consulted(self):
        class PoisonRepairs(dict):
            def __getitem__(self, key):
                if key == "repairs":
                    raise AssertionError("empty repair map was accessed")
                return super().__getitem__(key)

        body = PoisonRepairs(probability_body())
        body["repairs"] = "invalid and deliberately unusable"
        probabilities, _metadata = runner._probabilities_from_body(body, (3, 2), np, label="fictional-empty")
        self.assertEqual(probabilities.shape, (3, 2))

    def test_invalid_predictions_and_non_strict_semantics_are_rejected(self):
        bad_bodies = []
        for value in (float("nan"), float("inf"), -.1, 1.1, "0.9", True):
            body = probability_body()
            body["probabilities"][0][0] = value
            bad_bodies.append(body)
        body = probability_body()
        body["probabilities"].pop()
        bad_bodies.append(body)
        body = probability_body()
        body["metadata"]["decision_threshold_operator"] = ">="
        bad_bodies.append(body)
        body = probability_body()
        body["probabilities"][0][1] = .5
        body["metadata"]["upstream_binary_decisions"] = [[False, True], [False, False], [False, False]]
        bad_bodies.append(body)
        for body in bad_bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                runner._probabilities_from_body(body, (3, 2), np, label="fictional-invalid")

    def test_json_nonfinite_and_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.json"
            for text in ('{"probabilities":NaN}', '{"x":1,"x":2}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    runner._read_json(path)


class FrozenStudyTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("builtins.print")
        patcher.start()
        self.addCleanup(patcher.stop)

    def fixture(self, root):
        root = Path(root)
        sources = [runner.STUDY / name for name in ("run_study.py", "detection.py", "consensus.py", "repair.py", "protocol.md", "runner_design.md", "test_integration.py")]
        sources += [Path("zeroclean") / name for name in ("data.py", "types.py", "statistics.py", "sandbox.py")]
        for path in sources:
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text("FICTIONAL SOURCE BINDING ONLY\n")
        for name in runner.DATASETS:
            paths = {role: root / path for role, path in runner.input_paths(name).items()}
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            paths["dirty"].write_text("entity,value\ne1,bad\ne1,good\ne1,good\n")
            packet = {"table": {"id": name, "sha256": runner._sha256(paths["dirty"])}, "fictional": True}
            paths["fresh_packet"].write_text(json.dumps(packet))
            response = {"rules": [], "dependencies": []}
            paths["fresh_response"].write_text(json.dumps(response))
            artifact = {"schema_version": runner.RULE_ARTIFACT_SCHEMA, "dataset": name,
                        "dirty_sha256": runner._sha256(paths["dirty"]), "model": runner.MODEL,
                        "provenance": {"dirty_only": True, "clean_reference_access": False,
                                       "evaluation_feedback_access": False, "external_metadata_access": False,
                                       "input_packet_sha256": runner._sha256(paths["fresh_packet"]),
                                       "response_sha256": runner._sha256(paths["fresh_response"])}, **response}
            paths["pass2"].write_text(json.dumps(artifact))
            paths["old_pass2"].write_text(json.dumps({"dataset": name, "dirty_sha256": runner._sha256(paths["dirty"]), **response}))
            paths["cts"].write_text(json.dumps({"dataset": name, "cells": []}))
            paths["source"].write_text(json.dumps(probability_body(), indent=1) + "\n")
            paths["sequential"].write_text(json.dumps(probability_body(repairs=[{"row": 0, "column": 1, "value": "good"}]), indent=3) + "\n")
            # An unusable map proves that the empty carrier is not a repair source.
            empty = probability_body(repairs=[{"row": 99999, "column": 99999, "value": "MUST NEVER ENTER POOL"}])
            paths["empty"].write_text(json.dumps(empty))
        return fictional_runtime()

    def test_no_output_or_inference_before_global_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = fictional_runtime()
            with self.assertRaisesRegex(ValueError, "Global protocol freeze"):
                runner._run_study(root, ["hospital"], runtime)
            self.assertFalse((root / runner.OUTPUT).exists())
            self.assertEqual(runtime.detection.calls, [])

    def test_freeze_requires_all_six_and_hash_binds_all_fifty_four_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "all six"):
                runner.freeze_study(root, ["hospital"], runtime)
            self.assertFalse((root / runner.LOCK).exists())
            lock = runner.freeze_study(root, runner.DATASETS, runtime)
            self.assertEqual(set(lock["input_files"]), set(runner.DATASETS))
            self.assertEqual(sum(map(len, lock["input_files"].values())), 54)
            self.assertEqual(lock["method_files"], {"unified_loop_v5": "zeroclean_loop.json"})
            self.assertEqual(lock["source_files"]["experimental/__init__.py"], runner.ABSENT_SOURCE)
            self.assertEqual(lock["source_files"]["experimental/unified_loop_v5/__init__.py"], runner.ABSENT_SOURCE)
            self.assertFalse((root / runner.OUTPUT).exists())
            self.assertEqual(runtime.detection.calls, [])
            self.assertEqual(lock, runner.freeze_study(root, runner.DATASETS, runtime))

    def test_partial_lock_and_changed_sources_or_inputs_are_rejected(self):
        for mutation in ("partial", "source", "input"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runtime = self.fixture(root)
                runner.freeze_study(root, runner.DATASETS, runtime)
                if mutation == "partial":
                    lock = runner._read_json(root / runner.LOCK)
                    lock["datasets"] = ["hospital"]
                    (root / runner.LOCK).write_text(json.dumps(lock))
                elif mutation == "source":
                    (root / runner.STUDY / "repair.py").write_text("FICTIONAL CHANGED SOURCE")
                else:
                    (root / runner.input_paths("movies")["cts"]).write_text('{"cells":[]}')
                with self.assertRaises(ValueError):
                    runner._run_study(root, ["hospital"], runtime)
                self.assertFalse((root / runner.OUTPUT).exists())
                self.assertEqual(runtime.detection.calls, [])

    def test_full_fictional_run_preserves_inputs_and_exact_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.fixture(root)
            lock = runner.freeze_study(root, runner.DATASETS, runtime)
            before = {item["path"]: (root / item["path"]).read_bytes() for files in lock["input_files"].values() for item in files.values()}
            runner._run_study(root, ["hospital"], runtime)
            self.assertFalse((root / runner.OUTPUT / "completion_manifest.json").exists())
            runner._run_study(root, list(runner.DATASETS), runtime)
            complete = runner._read_json(root / runner.OUTPUT / "completion_manifest.json")
            self.assertTrue(complete["six_dataset_inference_complete"])
            self.assertFalse(complete["evaluation_performed"])
            for name in runner.DATASETS:
                for role, filename in (("source", "zeroed.json"), ("sequential", "zeroed_zeroec_reconstruction.json")):
                    self.assertEqual((root / runner.OUTPUT / "predictions" / name / filename).read_bytes(), before[runner.input_paths(name)[role]])
                output = runner._read_json(root / runner.OUTPUT / "predictions" / name / "zeroclean_loop.json")
                self.assertEqual(output["metadata"]["system"], "unified_loop_v5")
                self.assertEqual(output["repairs"], [{"row": 0, "column": 1, "value": "good"}])
                self.assertFalse((root / runner.OUTPUT / "predictions" / name / "seeded_carry.json").exists())
                self.assertFalse((root / runner.OUTPUT / "predictions" / name / "empty_carry.json").exists())
            for path, contents in before.items():
                self.assertEqual((root / path).read_bytes(), contents)
            inference_calls = len(runtime.detection.calls)
            snapshots = {path: path.read_bytes() for path in (root / runner.OUTPUT / "predictions").rglob("*.json")}
            runner._run_study(root, list(runner.DATASETS), runtime)
            self.assertEqual(len(runtime.detection.calls), inference_calls)
            for path, contents in snapshots.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_existing_incomplete_or_changed_completed_prediction_never_overwritten(self):
        for completed in (False, True):
            with self.subTest(completed=completed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runtime = self.fixture(root)
                runner.freeze_study(root, runner.DATASETS, runtime)
                if completed:
                    runner._run_study(root, ["hospital"], runtime)
                path = root / runner.OUTPUT / "predictions/hospital/zeroclean_loop.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("FICTIONAL EXISTING OUTPUT")
                with self.assertRaises(ValueError):
                    runner._run_study(root, ["hospital"], runtime)
                self.assertEqual(path.read_text(), "FICTIONAL EXISTING OUTPUT")

    def test_exclusive_writer_never_replaces_completed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            runner._exclusive_bytes(path, b"original fictional bytes")
            with self.assertRaises(FileExistsError):
                runner._exclusive_bytes(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"original fictional bytes")


class ResourceControlTests(unittest.TestCase):
    def test_environment_cap_and_single_threads(self):
        with mock.patch.dict(runner.os.environ, {}, clear=True), mock.patch.object(runner.resource, "getrlimit", return_value=(-1, -1)), mock.patch.object(runner.resource, "setrlimit") as set_limit:
            runner.bound_environment()
            set_limit.assert_called_once_with(runner.resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT"):
                self.assertEqual(runner.os.environ[name], "1")

    def test_foreground_run_holds_lock_until_all_work_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heavy-experiment.lock"
            events = []

            def work(*_args):
                events.append("work")
                with path.open("a+") as handle:
                    with self.assertRaises(BlockingIOError):
                        runner.fcntl.flock(handle.fileno(), runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)

            with mock.patch.object(runner, "HEAVY_LOCK", path), mock.patch.object(runner, "bound_environment", side_effect=lambda: events.append("bounded")), mock.patch.object(runner, "_run_study", side_effect=work):
                runner.main(["--phase", "run"])
            self.assertEqual(events, ["bounded", "work"])
            with path.open("a+") as handle:
                runner.fcntl.flock(handle.fileno(), runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
                runner.fcntl.flock(handle.fileno(), runner.fcntl.LOCK_UN)


if __name__ == "__main__":
    unittest.main()
