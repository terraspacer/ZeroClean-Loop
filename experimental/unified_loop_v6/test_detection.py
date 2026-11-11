"""Fictional-only tests. No loaders, cached runs, or benchmark fixtures."""
from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from zeroclean.data import DirtyTable
from experimental.unified_loop_v6.detection import DetectionConfig, _gate_inherited_detection


def fictional_table(values, column="reading"):
    return DirtyTable("fictional", (column,), tuple((value,) for value in values), "fictional-dirty-fingerprint")


def run_gate(table, source=None, inherited=None, rules=None, metadata=None, **kwargs):
    shape = table.shape
    return _gate_inherited_detection(
        table,
        np.full(shape, 0.9) if source is None else source,
        np.full(shape, 0.8) if inherited is None else inherited,
        np.zeros(shape, dtype=bool) if rules is None else rules,
        metadata,
        **kwargs,
    )


def cell_record(result, row):
    records = result["evidence"] + result["diagnostics"]["retained_positive_evidence"]
    return next(record for record in records if record["row"] == row)


class DetectionTests(unittest.TestCase):
    def test_high_source_rate_removes_unsupported_inherited_positives(self):
        table = fictional_table([str(i) for i in range(20)])
        result = run_gate(table)
        self.assertFalse(result["mask"].any())
        np.testing.assert_array_equal(result["probabilities"], np.full(table.shape, 0.49))
        self.assertEqual(len(result["evidence"]), 20)
        self.assertEqual(result["diagnostics"]["columns"][0]["source_metadata"]["status"], "missing_classifier_columns")
        self.assertEqual(result["diagnostics"]["aggregate"]["unknown_fallback_column_count"], 1)

    def test_validated_rule_retains_supported_positive_without_a_repair(self):
        table = fictional_table(["12"] * 20)
        rules = np.zeros(table.shape, dtype=bool)
        rules[7, 0] = True
        result = run_gate(table, rules=rules)
        self.assertEqual(np.flatnonzero(result["mask"][:, 0]).tolist(), [7])
        record = cell_record(result, 7)
        self.assertEqual(record["final_score"], 0.8)
        self.assertFalse(record["dirty_support"]["low_support"])
        self.assertEqual(record["evidence_types"], ["validated_target_specific_rule_violation"])
        self.assertNotIn("repairs", result)

    def test_context_disagreement_retains_common_format_positive(self):
        # A dirty-only peer module has found independent sources agreeing on a
        # different timestamp for this entity. All univariate formats are common.
        table = fictional_table(["08:30"] * 20, "event_time")
        contextual = np.zeros(table.shape, dtype=bool)
        contextual[2, 0] = True
        result = run_gate(table, contextual_low_support=contextual)
        self.assertEqual(np.flatnonzero(result["mask"][:, 0]).tolist(), [2])
        record = cell_record(result, 2)
        self.assertFalse(record["validated_rule_violation"])
        self.assertFalse(record["dirty_support"]["low_support"])
        self.assertEqual(record["evidence_types"], ["low_dirty_entity_peer_support"])
        self.assertEqual(result["diagnostics"]["columns"][0]["retained_by_contextual_low_support_count"], 1)

    def test_context_and_rules_never_promote_inherited_negatives(self):
        table = fictional_table(["10"] * 20)
        inherited = np.full(table.shape, 0.1)
        inherited[0, 0] = 0.5
        inherited[1, 0] = 0.49
        result = run_gate(table, inherited=inherited, rules=np.ones(table.shape, dtype=bool),
                          contextual_low_support=np.ones(table.shape, dtype=bool))
        np.testing.assert_array_equal(result["probabilities"], inherited)
        self.assertFalse(result["mask"].any())
        self.assertEqual(result["evidence"], [])

    def test_supported_unique_identifier_formats_are_not_low_support(self):
        table = fictional_table([f"ENTITY-{i:05d}" for i in range(40)], "entity_id")
        result = run_gate(table)
        self.assertFalse(result["mask"].any())
        for record in result["evidence"]:
            support = record["dirty_support"]
            self.assertEqual(support["value_peer_count"], 0)
            self.assertEqual(support["value_comparison"], "abstain_identifier")
            self.assertEqual(support["format_peer_count"], 39)
            self.assertFalse(support["low_support"])

    def test_numeric_identifier_digit_lengths_can_vary(self):
        table = fictional_table([str(10 ** (i % 7) + i) for i in range(40)], "recordID")
        result = run_gate(table)
        self.assertFalse(result["mask"].any())
        self.assertEqual(cell_record(result, 0)["dirty_support"]["format_signature"], "numeric")
        self.assertEqual(result["diagnostics"]["columns"][0]["dirty_profile"]["schema"]["tokens"], ["id", "record"])

    def test_rare_legitimate_free_text_abstains(self):
        table = fictional_table(["A recurring article"] * 59 + ["The 7th bell: an unusual title!"], "article_title")
        result = run_gate(table)
        self.assertFalse(result["mask"].any())
        support = cell_record(result, 59)["dirty_support"]
        self.assertEqual(support["value_comparison"], "abstain_free_text")
        self.assertEqual(support["format_comparison"], "abstain_free_text")
        self.assertFalse(support["low_support"])

    def test_unstructured_open_lexical_values_abstain(self):
        values = [chr(65 + i // 26) + chr(65 + i % 26) + " meadow" for i in range(40)]
        values[-1] = "An unusual grove!"
        result = run_gate(fictional_table(values, "label"))
        self.assertFalse(result["mask"].any())
        support = cell_record(result, 39)["dirty_support"]
        self.assertEqual(support["format_comparison"], "abstain_unstructured_or_mixed_syntax")
        self.assertEqual(support["value_comparison"], "abstain_high_distinctness")

    def test_low_leave_one_out_format_support_retains_outlier(self):
        table = fictional_table([str(i) for i in range(19)] + ["oops"])
        result = run_gate(table)
        self.assertEqual(np.flatnonzero(result["mask"][:, 0]).tolist(), [19])
        support = cell_record(result, 19)["dirty_support"]
        self.assertEqual(support["nonempty_peers"], 19)
        self.assertEqual(support["format_peer_count"], 0)
        self.assertEqual(support["value_peer_count"], 0)
        self.assertEqual(support["dominant_peer_format_fraction"], 1.0)
        self.assertTrue(support["low_format_support"])

    def test_small_sample_abstains_even_when_format_is_unique(self):
        result = run_gate(fictional_table(["10"] * 7 + ["oops"]))
        self.assertFalse(result["mask"].any())
        self.assertEqual(cell_record(result, 7)["dirty_support"]["format_comparison"], "abstain_insufficient_peers")

    def test_blank_can_have_low_structured_format_support(self):
        result = run_gate(fictional_table(["10"] * 19 + [" "]))
        self.assertEqual(np.flatnonzero(result["mask"][:, 0]).tolist(), [19])
        support = cell_record(result, 19)["dirty_support"]
        self.assertIsNone(support["format_signature"])
        self.assertEqual(support["nonempty_peers"], 19)
        self.assertTrue(support["low_format_support"])

    def test_repeated_core_domain_support_retains_rare_value(self):
        table = fictional_table(["active"] * 60 + ["paused"] * 20 + ["unrecognized"], "status")
        result = run_gate(table)
        self.assertEqual(np.flatnonzero(result["mask"][:, 0]).tolist(), [80])
        support = cell_record(result, 80)["dirty_support"]
        self.assertEqual(support["value_comparison"], "comparable")
        self.assertTrue(support["low_value_support"])
        self.assertFalse(support["low_format_support"])

    def test_core_membership_itself_excludes_the_query(self):
        table = fictional_table(["active"] * 3 + ["paused"] * 3 + ["new"], "status")
        config = {"min_domain_observations": 6, "max_domain_distinct_fraction": 1.0,
                  "identifier_unique_fraction": 1.0}
        result = run_gate(table, config=config)
        self.assertEqual(cell_record(result, 0)["dirty_support"]["peer_repeated_core_fraction"], 0.5)
        self.assertEqual(cell_record(result, 0)["dirty_support"]["value_comparison"], "abstain_weak_repeated_core")
        self.assertEqual(cell_record(result, 6)["dirty_support"]["peer_repeated_core_fraction"], 1.0)
        self.assertTrue(result["mask"][6, 0])

    def test_heterogeneous_structured_formats_abstain(self):
        table = fictional_table(["08:30"] * 10 + ["2026-01-01"] * 10 + ["odd"])
        result = run_gate(table)
        self.assertFalse(result["mask"].any())
        self.assertEqual(cell_record(result, 20)["dirty_support"]["format_comparison"], "abstain_heterogeneous_formats")

    def test_low_source_rate_and_clean_mask_are_unchanged(self):
        table = fictional_table([str(i) for i in range(20)])
        source = np.full(table.shape, 0.1)
        source[:8, 0] = 0.9  # Exactly 0.40 is outside the strict rate trigger.
        inherited = np.linspace(0.01, 0.99, 20).reshape(table.shape)
        result = run_gate(table, source=source, inherited=inherited)
        np.testing.assert_array_equal(result["probabilities"], inherited)
        np.testing.assert_array_equal(result["mask"], inherited > 0.5)
        self.assertFalse(result["diagnostics"]["columns"][0]["gated"])
        self.assertEqual(result["evidence"], [])
        source[8, 0] = 0.9
        self.assertTrue(run_gate(table, source=source)["diagnostics"]["columns"][0]["gated"])

    def test_zero_predicate_metadata_triggers_even_at_low_source_rate(self):
        table = fictional_table(["10"] * 20)
        metadata = {"classifier_columns": {"reading": {"clean_predicate_count_after_filter": 0}}}
        result = run_gate(table, source=np.full(table.shape, 0.1), metadata=metadata)
        column = result["diagnostics"]["columns"][0]
        self.assertEqual(column["gate_triggers"], ["source_empty_predicate_fallback"])
        self.assertTrue(column["source_metadata"]["empty_predicate_fallback"])
        self.assertFalse(result["mask"].any())

    def test_metadata_explicit_flags_descriptions_and_conflicts(self):
        table = fictional_table(["10"] * 20)
        low_source = np.full(table.shape, 0.1)
        cases = [
            ({"empty_predicate_fallback": True}, True, False),
            ({"source_empty_predicate_fallback": True}, True, False),
            ({"source_edge_case": "source zero-predicate filter removed all rows"}, True, False),
            ({"source_edge_case": "nonempty predicate retained", "clean_predicate_count_after_filter": 2}, False, False),
            ({"empty_predicate_fallback": False, "clean_predicate_count_after_filter": 0}, True, True),
        ]
        for info, expected, conflict in cases:
            with self.subTest(info=info):
                result = run_gate(table, source=low_source, metadata={"classifier_columns": {"reading": info}})
                diagnostic = result["diagnostics"]["columns"][0]["source_metadata"]
                self.assertIs(diagnostic["empty_predicate_fallback"], expected)
                self.assertEqual(diagnostic["conflicting_evidence"], conflict)
                self.assertEqual(result["diagnostics"]["columns"][0]["gated"], expected)

    def test_constant_positive_classifier_is_not_proof_of_empty_predicates(self):
        table = fictional_table(["10"] * 20)
        info = {"score_kind": "constant_pseudo_label_class", "classes": [1]}
        metadata = {"classifier_columns": {"reading": info}}
        low_rate = run_gate(table, source=np.full(table.shape, 0.1), metadata=metadata)
        diagnostic = low_rate["diagnostics"]["columns"][0]["source_metadata"]
        self.assertIsNone(diagnostic["empty_predicate_fallback"])
        self.assertTrue(diagnostic["constant_positive_classifier"])
        self.assertFalse(low_rate["diagnostics"]["columns"][0]["gated"])
        high_rate = run_gate(table, metadata=metadata)
        self.assertTrue(high_rate["diagnostics"]["columns"][0]["gated"])

    def test_missing_and_malformed_column_diagnostics_are_explicit(self):
        table = fictional_table(["10"] * 20)
        cases = [({}, "missing_column_diagnostics"),
                 ({"reading": "unavailable"}, "malformed_column_diagnostics"),
                 ({"reading": {"clean_predicate_count_after_filter": "0"}}, "unknown_incomplete_column_diagnostics")]
        for columns, status in cases:
            with self.subTest(columns=columns):
                result = run_gate(table, metadata={"classifier_columns": columns})
                diagnostic = result["diagnostics"]["columns"][0]["source_metadata"]
                self.assertIsNone(diagnostic["empty_predicate_fallback"])
                self.assertEqual(diagnostic["status"], status)
                self.assertTrue(result["diagnostics"]["columns"][0]["gated"])

    def test_scores_mask_json_and_inputs_remain_consistent(self):
        table = fictional_table(["10"] * 19 + ["oops"])
        source = np.full(table.shape, 0.9)
        inherited = np.linspace(0.05, 1.0, 20).reshape(table.shape)
        rules = np.zeros(table.shape, dtype=bool)
        contextual = np.zeros(table.shape, dtype=bool)
        metadata = {"classifier_columns": {"reading": {"clean_predicate_count_after_filter": 0}}}
        snapshot = (source.copy(), inherited.copy(), rules.copy(), contextual.copy(), copy.deepcopy(metadata), table.rows)
        result = run_gate(table, source=source, inherited=inherited, rules=rules, metadata=metadata,
                          contextual_low_support=contextual, config={"demoted_score": 0.5})
        self.assertEqual(result["probabilities"].dtype, np.dtype(float))
        self.assertEqual(result["mask"].dtype, np.dtype(bool))
        np.testing.assert_array_equal(result["mask"], result["probabilities"] > 0.5)
        json.dumps(result["evidence"], allow_nan=False)
        json.dumps(result["diagnostics"], allow_nan=False)
        self.assertFalse(np.shares_memory(result["probabilities"], inherited))
        for actual, expected in zip((source, inherited, rules, contextual), snapshot[:4]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(metadata, snapshot[4])
        self.assertEqual(table.rows, snapshot[5])

    def test_inputs_reject_invalid_shapes_scores_and_nonboolean_evidence(self):
        table = fictional_table(["10"] * 10)
        for probabilities in (np.zeros((2, 2)), np.full(table.shape, np.nan), np.full(table.shape, -0.1), np.full(table.shape, 1.1)):
            with self.subTest(probabilities=probabilities.shape), self.assertRaises(ValueError):
                run_gate(table, inherited=probabilities)
        with self.assertRaises(ValueError):
            run_gate(table, rules=np.ones(table.shape, dtype=int))
        with self.assertRaises(ValueError):
            run_gate(table, contextual_low_support=np.ones(table.shape, dtype=int))
        with self.assertRaises(ValueError):
            run_gate(table, contextual_low_support=np.zeros((1, 1), dtype=bool))
        with self.assertRaises(TypeError):
            run_gate(table, metadata=[])

    def test_configuration_is_fixed_explicit_and_validated(self):
        self.assertEqual(DetectionConfig().source_positive_fraction_threshold, 0.40)
        for kwargs in ({"demoted_score": 0.51}, {"min_comparison_peers": 0},
                       {"min_domain_observations": True}, {"max_rare_peer_fraction": float("nan")},
                       {"max_rare_peer_count": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DetectionConfig(**kwargs)
        with self.assertRaises(TypeError):
            run_gate(fictional_table(["10"]), config={"dataset_threshold": 0.2})

    def test_no_rows_is_a_valid_noop(self):
        table = fictional_table([])
        empty = np.empty(table.shape, dtype=float)
        result = run_gate(table, source=empty, inherited=empty)
        self.assertEqual(result["probabilities"].shape, (0, 1))
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["diagnostics"]["aggregate"]["final_positive_count"], 0)


if __name__ == "__main__":
    unittest.main()
