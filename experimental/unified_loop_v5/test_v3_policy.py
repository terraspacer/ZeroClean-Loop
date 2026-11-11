"""Fictional-only v3 method and artifact-boundary tests; no benchmark reads."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from zeroclean.data import DirtyTable
from experimental.unified_loop_v5 import detection, repair, run_study as runner
from experimental.unified_loop_v5.test_repair import catalog, candidate, entry, mask_for


def table(values, column="reading"):
    return DirtyTable("fictional", (column,), tuple((value,) for value in values), "f" * 64)


def gate(dirty, rules=None, contextual=None):
    scores = np.full(dirty.shape, 0.1)
    return detection.gate_detection(dirty, scores, scores,
                                    np.zeros(dirty.shape, dtype=bool) if rules is None else rules,
                                    {}, contextual_low_support=contextual)


class NegativePromotionTests(unittest.TestCase):
    def test_rule_and_source_context_promote_inherited_negatives(self):
        dirty = table(["normal"] * 40, "article_title")
        rules, contextual = np.zeros(dirty.shape, bool), np.zeros(dirty.shape, bool)
        rules[0, 0], contextual[1, 0] = True, True
        result = gate(dirty, rules, contextual)
        self.assertEqual(np.flatnonzero(result["mask"][:, 0]).tolist(), [0, 1])
        self.assertEqual(result["diagnostics"]["aggregate"]["promoted_inherited_negative_count"], 2)
        self.assertEqual(result["probabilities"][0, 0], .75)

    def test_text_rarity_and_unique_identifier_do_not_promote(self):
        for column in ("article_title", "author_name", "entity_id"):
            dirty = table([f"Unique valid content {i}" for i in range(40)], column)
            self.assertFalse(gate(dirty)["mask"].any())

    def test_blank_title_in_nearly_complete_column_is_promoted(self):
        result = gate(table([""] + [f"Valid unique title {i}" for i in range(50)], "article_title"))
        self.assertEqual(np.flatnonzero(result["mask"][:, 0]).tolist(), [0])
        self.assertIn("missing_in_nearly_complete_column", result["evidence"][0]["evidence_types"])

    def test_common_optional_blanks_are_not_missingness_evidence(self):
        result = gate(table([""] * 20 + [f"Valid unique title {i}" for i in range(30)], "article_title"))
        self.assertFalse(result["mask"].any())

    def test_inferred_numeric_grammar_catches_negative_without_candidate(self):
        dirty = table(["twelve"] + [str(i) for i in range(1, 41)], "volume")
        result = gate(dirty)
        self.assertTrue(result["mask"][0, 0])
        self.assertIn("inferred_type_contradiction", result["evidence"][0]["evidence_types"])
        self.assertNotIn("repairs", result)

    def test_rare_valid_number_is_not_an_error(self):
        dirty = table(["999"] + ["12"] * 40, "volume")
        self.assertFalse(gate(dirty)["mask"].any())

    def test_calendar_grammar_accepts_both_orders_and_rejects_impossible_date(self):
        dirty = table(["2/31/20", "23/2/20"] + ["1/15/20"] * 40, "article_created_at")
        result = gate(dirty)
        self.assertTrue(result["mask"][0, 0])
        self.assertFalse(result["mask"][1, 0])

    def test_serialized_list_grammar_and_unicode_names(self):
        dirty = table(['{"Broken name"', '{"Zoë D.","José P."}'] + ['{"Valid Person"}'] * 40, "author_list")
        result = gate(dirty)
        self.assertTrue(result["mask"][0, 0])
        self.assertFalse(result["mask"][1, 0])

    def test_case_variant_uses_observed_exact_peers(self):
        dirty = table(["ENG"] * 3 + ["eng"] * 45, "article_language")
        result = gate(dirty)
        self.assertTrue(result["mask"][:3].all())
        self.assertFalse(result["mask"][3:].any())
        self.assertIn("dominant_observed_case_space_variant", result["evidence"][0]["evidence_types"])

    def test_control_character_is_intrinsic_even_in_free_text(self):
        dirty = table(["Bad\x01title", "Valid title"], "article_title")
        self.assertEqual(gate(dirty)["mask"][:, 0].tolist(), [True, False])

    def test_dirty_fd_requires_global_support_and_excludes_query(self):
        rows = [("Journal A", "WRONG")] + [("Journal A", "A")] * 39 + [("Journal B", "B")] * 40
        dirty = DirtyTable("fictional", ("journal_title", "journal_abbreviation"), tuple(rows), "f" * 64)
        context, diagnostics = detection.intrinsic_context(dirty, detection.DetectionConfig())
        self.assertIn((0, 1), context)
        self.assertEqual(context[0, 1][0]["peer_count"], 39)
        self.assertEqual(context[0, 1][0]["agreement"], 1.)
        self.assertTrue(diagnostics["qualified_relations"])
        self.assertTrue(gate(dirty)["mask"][0, 1])

    def test_nearly_constant_target_does_not_establish_contextual_fd(self):
        rows = [("Group A", "rare")] + [("Group A", "common")] * 39 + [("Group B", "common")] * 40
        dirty = DirtyTable("fictional", ("context", "article_title"), tuple(rows), "f" * 64)
        context, _ = detection.intrinsic_context(dirty, detection.DetectionConfig())
        self.assertNotIn((0, 1), context)

    def test_same_dataset_name_is_not_required(self):
        dirty = table(["bad"] + [str(i) for i in range(40)], "amount")
        alternate = DirtyTable("completely-unrecognized", dirty.columns, dirty.rows, dirty.fingerprint)
        np.testing.assert_array_equal(gate(dirty)["mask"], gate(alternate)["mask"])

    def test_promotions_are_deterministic_and_do_not_mutate_inputs(self):
        dirty = table(["bad"] + [str(i) for i in range(40)], "amount")
        before = deepcopy(dirty.rows)
        first, second = gate(dirty), gate(dirty)
        np.testing.assert_array_equal(first["probabilities"], second["probabilities"])
        self.assertEqual(first["evidence"], second["evidence"])
        self.assertEqual(dirty.rows, before)
        json.dumps(first["diagnostics"], allow_nan=False)


class CatalogPrecisionTests(unittest.TestCase):
    def test_rule_gain_alone_cannot_unlock_catalog_edit(self):
        dirty = catalog([entry("1", "First", "A"), entry("2", "Second", "B"), entry("3", "Third", "C")])
        result = repair.select_repairs(dirty, mask_for(dirty, (0, 2)), {(0, 2): [candidate("Lager", rule_gain=1., delta_features=[1., 0., 0., 0.])]})
        self.assertEqual(result["repairs"], {})
        self.assertIn("catalog_requires_entity_fd_or_observed_numeric_canonical_form", result["decisions"][0]["candidate_reviews"][0]["rejection_reasons"])

    def test_generic_cts_majority_without_entity_match_is_rejected(self):
        dirty = catalog([entry("1", "First", "A", city="Wrong")] + [entry(str(i), f"Item{i}", str(i), city="Common") for i in range(2, 6)])
        proposal = candidate("Common", origin="cts", cts_peer_rows=[1, 2, 3, 4])
        result = repair.select_repairs(dirty, mask_for(dirty, (0, 7)), {(0, 7): [proposal]})
        review = result["decisions"][0]["candidate_reviews"][0]
        self.assertTrue(review["cts_evidence"]["valid"])
        self.assertEqual(result["repairs"], {})

    def test_unit_stripping_without_exact_observed_format_is_rejected(self):
        dirty = catalog([entry("1", "First", "A", ounces="12.0 oz")] + [entry(str(i), str(i), str(i), ounces="12") for i in range(2, 6)])
        result = repair.select_repairs(dirty, mask_for(dirty, (0, 3)), {(0, 3): [candidate("12.0")]})
        self.assertEqual(result["repairs"], {})

    def test_numeric_equivalent_candidate_must_be_observed_canonical_exact_string(self):
        dirty = catalog([entry("1", "First", "A", ounces="12.0 oz")] + [entry(str(i), str(i), str(i), ounces="12") for i in range(2, 6)])
        result = repair.select_repairs(dirty, mask_for(dirty, (0, 3)), {(0, 3): [candidate("12.0"), candidate("12")]})
        self.assertEqual(result["repairs"], {(0, 3): "12"})
        reviews = {item["value"]: item for item in result["decisions"][0]["candidate_reviews"]}
        self.assertFalse(reviews["12.0"]["eligible"])
        self.assertTrue(reviews["12"]["catalog_gate"]["numeric_canonical_evidence"]["valid"])

    def test_numeric_amount_change_or_tied_exact_forms_abstains(self):
        for values, candidate_value in ((["12"] * 4, "13"), (["12"] * 3 + ["12.0"] * 3, "12")):
            dirty = catalog([entry("1", "First", "A", ounces="12.0 oz")] + [entry(str(i), str(i), str(i), ounces=value) for i, value in enumerate(values, 2)])
            result = repair.select_repairs(dirty, mask_for(dirty, (0, 3)), {(0, 3): [candidate(candidate_value)]})
            self.assertEqual(result["repairs"], {})

    def test_entity_fd_and_exact_peers_still_pass(self):
        dirty = catalog([entry("1", "First", "A", city="Typo"), entry("2", "Second", "A"), entry("3", "Third", "A")])
        result = repair.select_repairs(dirty, mask_for(dirty, (0, 7)), {(0, 7): [candidate("Harbor")]})
        self.assertEqual(result["repairs"], {(0, 7): "Harbor"})


class FreshArtifactTests(unittest.TestCase):
    def artifact(self):
        dirty = table(["bad", "good"])
        body = {"schema_version": runner.RULE_ARTIFACT_SCHEMA, "dataset": dirty.name,
                "dirty_sha256": dirty.fingerprint, "model": {**runner.MODEL, "transport": "fictional"},
                "provenance": {"dirty_only": True, "clean_reference_access": False,
                               "evaluation_feedback_access": False, "external_metadata_access": False,
                               "input_packet_sha256": "a" * 64, "response_sha256": "b" * 64},
                "rules": [{"id": "fictional", "target": "reading", "kind": "single", "description": "fictional rule",
                           "source": "def check(row):\n    return row['reading'] == 'bad'"}], "dependencies": []}
        return dirty, body

    def test_valid_wrapper_preserves_extra_provenance(self):
        dirty, body = self.artifact()
        result = runner.validate_rule_artifact(dirty, body, packet_sha256="a" * 64, response_sha256="b" * 64)
        self.assertEqual(result["model"]["transport"], "fictional")
        self.assertEqual(result["rule_count"], 1)

    def test_identity_provenance_and_exact_hashes_fail_closed(self):
        dirty, original = self.artifact()
        for field, value in (("dataset", "different"), ("dirty_sha256", "c" * 64), ("schema_version", "old")):
            body = deepcopy(original); body[field] = value
            with self.assertRaises(ValueError): runner.validate_rule_artifact(dirty, body)
        for group, key, value in (("model", "model", "other"), ("provenance", "dirty_only", False),
                                  ("provenance", "clean_reference_access", True), ("provenance", "response_sha256", "invalid")):
            body = deepcopy(original); body[group][key] = value
            with self.assertRaises(ValueError): runner.validate_rule_artifact(dirty, body)
        with self.assertRaises(ValueError): runner.validate_rule_artifact(dirty, original, packet_sha256="c" * 64)

    def test_duplicate_ids_and_unresolved_targets_fail_closed(self):
        dirty, body = self.artifact()
        body["rules"].append(dict(body["rules"][0]))
        with self.assertRaises(ValueError): runner.validate_rule_artifact(dirty, body)
        dirty, body = self.artifact(); body["rules"][0]["target"] = "unknown"
        with self.assertRaises(ValueError): runner.validate_rule_artifact(dirty, body)

    def test_malformed_dependency_cannot_bypass_support_gates(self):
        dirty, body = self.artifact()
        body["dependencies"] = [{"determinants": ["reading"], "dependent": "reading", "min_support": 1, "confidence": 2}]
        with self.assertRaises(ValueError): runner.validate_rule_artifact(dirty, body)


class RuleUnionTests(unittest.TestCase):
    artifact = FreshArtifactTests.artifact

    def test_distinct_cached_and_fresh_criteria_survive_id_collision(self):
        dirty, fresh = self.artifact()
        old = deepcopy(fresh)
        old["rules"][0]["source"] = "def check(row):\n    return row['reading'] == 'old-error'"
        old["value_repairs"] = [{"value": "MUST NEVER BECOME A CANDIDATE"}]
        before = deepcopy((old, fresh))
        result = runner.merge_rule_artifacts(dirty, old, fresh)
        self.assertEqual([rule["id"] for rule in result["rules"]], ["cached:fictional", "fresh:fictional"])
        self.assertEqual(result["union_diagnostics"]["union_rule_count"], 2)
        self.assertNotIn("value_repairs", result)
        self.assertEqual((old, fresh), before)

    def test_exact_executable_duplicates_receive_one_vote_with_both_origins(self):
        dirty, fresh = self.artifact()
        old = deepcopy(fresh)
        fresh["rules"][0].update(id="different-id", description="different description")
        result = runner.merge_rule_artifacts(dirty, old, fresh)
        self.assertEqual(len(result["rules"]), 1)
        self.assertEqual(result["rules"][0]["input_rule_origins"],
                         [{"origin": "cached", "id": "fictional"}, {"origin": "fresh", "id": "different-id"}])

    def test_dependencies_are_union_deduplicated_with_determinant_order_ignored(self):
        dirty = SimpleNamespace(name="fictional", fingerprint="f" * 64, columns=("a", "b", "c"))
        dependency = {"determinants": ["a", "b"], "dependent": "c", "min_support": 3, "confidence": .9}
        old = {"dataset": dirty.name, "dirty_sha256": dirty.fingerprint, "rules": [], "dependencies": [dependency]}
        fresh = deepcopy(old)
        fresh["dependencies"][0]["determinants"].reverse()
        result = runner.merge_rule_artifacts(dirty, old, fresh)
        self.assertEqual(result["dependencies"], [dependency])

    def test_cached_identity_and_schema_are_mandatory(self):
        dirty, fresh = self.artifact()
        for key, value in (("dataset", "wrong"), ("dirty_sha256", "a" * 64), ("dependencies", None)):
            old = deepcopy(fresh)
            old[key] = value
            with self.assertRaises(ValueError): runner.merge_rule_artifacts(dirty, old, fresh)


if __name__ == "__main__":
    unittest.main()
