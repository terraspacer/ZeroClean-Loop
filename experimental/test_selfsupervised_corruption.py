"""Synthetic-only checks: no benchmark tables, references, labels, or results."""
from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from zeroclean.data import DirtyTable
from experimental.selfsupervised_corruption import (
    FEATURE_NAMES, LIMITS, OPERATORS, PROTOCOL_PATH, SEED, STAGES,
    ReferenceIndex, SyntheticSplit, assert_separation, digest, fit_synthetic,
    make_synthetic, mutate, partition_rows, preview_decision, run_table, value_stage,
)


def table(rows, columns=None):
    return DirtyTable("synthetic_only", tuple(columns or (f"c{j}" for j in range(len(rows[0])))), tuple(tuple(row) for row in rows), "synthetic-no-real-input")


def relational_table(n=1200):
    # Repeated contexts imply a relation, while the numeric field has valid extremes.
    return table([(f"entity-{i:05}", f"group-{i % 6}", f"family-{i % 6}", str(i + 10)) for i in range(n)])


class TestProtocolAndSeparation(unittest.TestCase):
    def test_protocol_feature_order_and_no_clean_label_claim(self):
        protocol = json.loads(PROTOCOL_PATH.read_text())
        self.assertEqual(protocol["seed"], SEED)
        self.assertEqual(protocol["features"], list(FEATURE_NAMES))
        self.assertIn("NOT a verified clean", protocol["label_meaning"]["0"])
        self.assertIn("not verified clean", " ".join(LIMITS))

    def test_complete_duplicate_rows_and_entity_proxy_stay_together(self):
        rows = [(f"id-{i}", "repeat") for i in range(100)]
        rows.extend([rows[4], ("id-7", "alternate")])
        data = table(rows)
        split = partition_rows(data)
        self.assertEqual(split.row_stage[4], split.row_stage[100])
        self.assertEqual(split.component[4], split.component[100])
        self.assertEqual(split.component[7], split.component[101])
        self.assertIn(0, split.proxy_columns)
        self.assertEqual(split.row_stage, partition_rows(data).row_stage)

    def test_supervised_original_and_changed_value_groups_are_held_apart(self):
        data = relational_table(600)
        partition = partition_rows(data)
        index = ReferenceIndex(data, partition.rows["reference"])
        splits = {stage: make_synthetic(data, partition, index, stage) for stage in STAGES}
        audit = assert_separation(partition, splits)
        self.assertEqual(audit["supervised_query_value_group_overlap"], 0)
        for stage, split in splits.items():
            self.assertTrue(split.x)
            self.assertTrue(split.mutations)
            self.assertTrue(all(value_stage(j, value) == stage for j, value in split.query_values))
            for mutation in split.mutations:
                self.assertEqual(partition.row_stage[mutation.row], stage)
                if mutation.donor_row is not None:
                    self.assertEqual(partition.row_stage[mutation.donor_row], stage)
        repeated = make_synthetic(data, partition, index, "test")
        self.assertEqual(splits["test"].x, repeated.x)
        self.assertEqual(splits["test"].mutations, repeated.mutations)

    def test_audit_detects_deliberate_value_leakage(self):
        data = relational_table(100)
        partition = partition_rows(data)
        blank = lambda: SyntheticSplit([], [], [], [], [], set(), {})
        splits = {stage: blank() for stage in STAGES}
        splits["train"].query_values.add((1, "same-value"))
        splits["test"].query_values.add((1, "same-value"))
        with self.assertRaisesRegex(AssertionError, "query-value leakage"):
            assert_separation(partition, splits)


class TestMutationsAndSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = relational_table()
        cls.partition = partition_rows(cls.data)
        cls.index = ReferenceIndex(cls.data, cls.partition.rows["reference"])

    def test_all_operators_have_changed_output_and_provenance(self):
        seen = set()
        rng = random.Random(SEED)
        for operator in OPERATORS:
            for stage in STAGES:
                if operator in seen:
                    break
                for i in self.partition.rows[stage][:80]:
                    if operator in seen:
                        break
                    for j in range(4):
                        for _ in range(5):
                            mutation, reason = mutate(self.data, self.partition, self.index, i, j, operator, rng)
                            if mutation is None:
                                continue
                            self.assertNotEqual(mutation.value, self.data.rows[i][j])
                            self.assertEqual(mutation.original_hash, digest("value", self.data.rows[i][j]))
                            self.assertEqual(mutation.changed_hash, digest("value", mutation.value))
                            self.assertEqual(reason, "retained")
                            self.assertNotIn("value", mutation.public())
                            if operator == "cross_field_displacement":
                                self.assertNotEqual(j, mutation.donor_column)
                            if operator == "observed_same_column_substitution":
                                self.assertEqual(j, mutation.donor_column)
                            seen.add(operator)
                            break
                        if operator in seen:
                            break
        self.assertEqual(seen, set(OPERATORS))

    def test_degenerate_reversal_and_empty_source_rejected(self):
        data = table([(str(i), "aaaa", "") for i in range(200)])
        partition = partition_rows(data)
        index = ReferenceIndex(data, partition.rows["reference"])
        i = partition.rows["train"][0]
        mutation, reason = mutate(data, partition, index, i, 1, "full_reversal", random.Random(0))
        self.assertIsNone(mutation)
        self.assertEqual(reason, "unchanged")
        mutation, reason = mutate(data, partition, index, i, 2, "character_insertion", random.Random(0))
        self.assertIsNone(mutation)
        self.assertEqual(reason, "empty_source")

    def test_semantic_same_vocabulary_swap_has_context_support(self):
        row = ("new-entity", "group-2", "family-3", "500")
        _, support = self.index.features(row, 2)
        self.assertTrue(support["context_contradiction"])
        self.assertTrue(support["supported"])
        decision, reason = preview_decision("test", row[2], support, .99, .9, True)
        self.assertEqual((decision, reason), ("candidate_unpromoted", "context_contradiction"))

    def test_valid_numeric_extreme_and_unusual_name_alone_abstain(self):
        row = ("legitimate unusual person X Æ", "unknown-group", "unique-family", "999999999999")
        features, support = self.index.features(row, 3)
        self.assertEqual(features[7], 20)
        self.assertFalse(support["supported"])
        self.assertEqual(preview_decision("test", row[3], support, .9999, .9, True), ("abstain", "no_supported_contradiction"))
        _, name_support = self.index.features(row, 0)
        self.assertFalse(name_support["supported"])
        self.assertEqual(preview_decision("test", row[0], name_support, .9999, .9, True)[0], "abstain")

    def test_numeric_format_corruption_supported_but_threshold_still_required(self):
        row = ("unknown-entity", "unknown-group", "unknown-family", "12oops")
        _, support = self.index.features(row, 3)
        self.assertTrue(support["numeric_type_contradiction"])
        self.assertEqual(preview_decision("test", row[3], support, .89, .9, True), ("abstain", "below_fixed_synthetic_threshold"))
        self.assertEqual(preview_decision("test", row[3], support, .99, .9, True)[0], "candidate_unpromoted")

    def test_conflicting_contexts_abstain_from_context_assertion(self):
        rows = [(f"entity-{i}", "keyA", "contextX", "red") for i in range(200)]
        rows += [(f"entity-{i+200}", "keyB", "contextY", "blue") for i in range(200)]
        data = table(rows)
        index = ReferenceIndex(data, list(range(400)))
        _, support = index.features(("unseen", "keyA", "contextY", "purple"), 3)
        self.assertTrue(support["context_conflict"])
        self.assertFalse(support["context_contradiction"])
        self.assertFalse(support["supported"])

    def test_reference_rows_and_empty_values_always_abstain(self):
        _, support = self.index.features(self.data.rows[0], 0)
        self.assertEqual(preview_decision("reference", "bad", support, 1, .9, True)[1], "reference_profile_only")
        self.assertEqual(preview_decision("test", "", support, 1, .9, True)[1], "empty_source_unsupported")


class TestFitAndOutputs(unittest.TestCase):
    def test_small_synthetic_input_abstains_without_fitting(self):
        data = table([(str(i), "x") for i in range(20)])
        with tempfile.TemporaryDirectory(prefix="synthetic-corruption-test-") as directory:
            report = run_table(data, Path(directory))
            self.assertEqual(report["model"]["status"], "abstain_insufficient_synthetic_training_or_calibration")
            self.assertEqual(report["structural_coverage"].get("candidate_cells_unpromoted", 0), 0)
            self.assertEqual(report["input_sha256"], "synthetic-no-real-input")
            self.assertIn("not a probability", report["score_meaning"])
            self.assertTrue((Path(directory) / "synthetic_only.scores.csv").is_file())

    def test_fit_threshold_uses_only_calibration_noisy_baselines(self):
        def split(n, seed):
            rng = np.random.default_rng(seed)
            x = rng.normal(size=(n, len(FEATURE_NAMES)))
            y = [i % 2 for i in range(n)]
            x[:, 0] += np.asarray(y) * 2
            return SyntheticSplit(x.tolist(), y, ["injected" if v else "unchanged_dirty_baseline" for v in y], [], [], set(), {})
        splits = {"train": split(500, 1), "calibration": split(240, 2), "test": split(240, 3)}
        model, threshold, report = fit_synthetic(splits)
        self.assertIsNotNone(model)
        scores = model.predict_proba(np.asarray(splits["calibration"].x))[:, 1]
        baseline = scores[np.asarray(splits["calibration"].y) == 0]
        self.assertEqual(threshold, max(.9, float(np.quantile(baseline, .995, method="higher"))))
        self.assertTrue(report["test_synthetic_only"]["not_benchmark_accuracy"])
        # Deliberately changing test labels cannot change the model or threshold.
        splits["test"].y = [1 - label for label in splits["test"].y]
        second_model, second_threshold, _ = fit_synthetic(splits)
        self.assertEqual(threshold, second_threshold)
        np.testing.assert_array_equal(model[-1].coef_, second_model[-1].coef_)


if __name__ == "__main__":
    unittest.main()
