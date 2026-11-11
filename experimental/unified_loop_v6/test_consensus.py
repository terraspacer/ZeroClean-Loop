"""Fictional-only tests. No production data, reference files, or outcomes."""
from dataclasses import dataclass
import json
import unittest

from experimental.unified_loop_v6.consensus import DEFAULT_CONFIG, discover_consensus


@dataclass(frozen=True)
class Table:
    columns: tuple
    rows: tuple

    @property
    def name(self):
        raise AssertionError("Dataset names must not influence intrinsic discovery")


def fixture(query=None, training_count=4, sources=("a", "b", "c", "d")):
    query = query or {"a": "broken", "b": "7:10 a.m.", "c": "7:10 a.m.", "d": "7:10 a.m."}
    rows = [("q", source, query[source], "6:00 p.m.") for source in sources]
    for entity in range(training_count):
        for source in sources:
            rows.append((f"training-{entity}", source, f"{8 + entity}:15 a.m.", "6:00 p.m."))
    return Table(("record_id", "src", "start_time", "end_time"), tuple(rows))


def candidate(table, row=0, col=2):
    return discover_consensus(table)["candidates"].get((row, col), [])


class ConsensusTests(unittest.TestCase):
    def test_schema_and_dirty_group_discovery_without_dataset_name(self):
        original = fixture()
        table = Table(("region",) + original.columns, tuple(("same-region",) + row for row in original.rows))
        result = discover_consensus(table)
        self.assertEqual(result["diagnostics"]["source_column"]["name"], "src")
        self.assertEqual([item["name"] for item in result["diagnostics"]["key_columns"]], ["record_id"])
        self.assertTrue(result["candidates"][(0, 3)][0]["actionable"])
        json.dumps(result["diagnostics"], allow_nan=False)
        json.dumps(result["candidates"][(0, 3)][0], allow_nan=False)

    def test_unique_keys_abstain(self):
        original = fixture()
        table = Table(original.columns, tuple((f"unique-{i}",) + row[1:] for i, row in enumerate(original.rows)))
        result = discover_consensus(table)
        self.assertEqual(result["candidates"], {})
        self.assertEqual(result["diagnostics"]["reason"], "no_repeated_multisource_entity_key")

    def test_nonapplicable_without_source_schema(self):
        original = fixture()
        table = Table(("record_id", "category", "start_time", "end_time"), original.rows)
        result = discover_consensus(table)
        self.assertEqual(result["candidates"], {})
        self.assertEqual(result["diagnostics"]["reason"], "no_schema_supported_multisource_column")

    def test_nonapplicable_without_temporal_values(self):
        table = Table(("record_id", "src", "text"), (("a", "x", "note"), ("a", "y", "other")))
        result = discover_consensus(table)
        self.assertEqual(result["candidates"], {})
        self.assertEqual(result["diagnostics"]["reason"], "no_temporal_columns")

    def test_source_duplicates_cannot_inflate_votes_or_reliability(self):
        base = fixture()
        duplicated = Table(base.columns, base.rows + (base.rows[1],) * 30 + (base.rows[5],) * 30)
        before, after = candidate(base)[0], candidate(duplicated)[0]
        for field in ("value", "score", "peer_source_count", "winning_source_count", "agreement", "margin", "actionable"):
            self.assertEqual(before[field], after[field])
        self.assertEqual(before["evidence"]["source_reliability"], after["evidence"]["source_reliability"])
        self.assertEqual(after["winning_source_count"], 3)

    def test_query_source_excluded_including_all_its_duplicate_rows(self):
        base = fixture()
        extra = ("q", "a", "1:00 a.m.", "6:00 p.m.")
        duplicated = Table(base.columns, base.rows + (extra,) * 25)
        result = candidate(duplicated)[0]
        self.assertEqual(result["value"], "7:10 a.m.")
        self.assertEqual(result["peer_source_count"], 3)
        self.assertTrue(result["actionable"])
        self.assertNotIn("a", result["evidence"]["source_reliability"])
        self.assertTrue(all(item["source"] != "a" for item in result["evidence"]["peer_observations"]))

    def test_conflicting_source_abstains_and_stays_in_denominator(self):
        base = fixture()
        table = Table(base.columns, base.rows + (("q", "b", "9:59 p.m.", "6:00 p.m."),))
        result = candidate(table)[0]
        self.assertEqual(result["winning_source_count"], 2)
        self.assertEqual(result["peer_source_count"], 3)
        self.assertAlmostEqual(result["agreement"], 2 / 3)
        self.assertTrue(result["actionable"])
        self.assertIsNone(next(item["usable_vote"] for item in result["evidence"]["peer_observations"] if item["source"] == "b"))

    def test_malformed_peer_times_never_become_candidates(self):
        table = fixture({"a": "broken", "b": "25:00", "c": "7:10 a.m.", "d": "7:99 a.m."})
        result = candidate(table)[0]
        self.assertEqual(result["value"], "7:10 a.m.")
        self.assertFalse(result["actionable"])
        self.assertEqual(result["winning_source_count"], 1)
        self.assertIn("insufficient_winning_sources", result["evidence"]["abstention_reasons"])
        all_bad = fixture({"a": "broken", "b": "24:00", "c": "0:10 p.m.", "d": "2025-02-30T10:00Z"})
        self.assertEqual(candidate(all_bad), [])

    def test_malformed_iso_offset_fields_never_become_proposals(self):
        for offset in ("+01:60", "-01:60", "+23:99", "+24:00", "-24:00"):
            with self.subTest(offset=offset):
                value = "2030-01-01T08:10" + offset
                table = fixture({"a": "broken", "b": value, "c": value, "d": value})
                result = discover_consensus(table)
                self.assertFalse(any(item["value"] == value for items in result["candidates"].values() for item in items))

    def test_malformed_iso_offset_companion_supplies_no_chronology_evidence(self):
        start = "2030-01-01T08:10+00:00"
        base = fixture({"a": "broken", "b": start, "c": start, "d": start})
        table = Table(base.columns, (base.rows[0][:3] + ("2030-01-01T08:10+01:60",),) + base.rows[1:])
        result = candidate(table)[0]
        checks = result["evidence"]["intrinsic_checks"]
        self.assertTrue(result["actionable"])
        self.assertEqual(checks["temporal_relations"], [])
        self.assertIn("companion_missing_or_malformed", checks["skipped_temporal_relations"])

    def test_unhinted_temporal_discovery_is_exact_duplicate_invariant(self):
        rows = (("E0", "a", "malformed"), ("E0", "b", "08:10"), ("E0", "c", "08:10"),
                ("E1", "a", "malformed"), ("E1", "b", "09:10"), ("E1", "c", "09:10"))
        columns = ("entity_id", "source", "observation")
        base = discover_consensus(Table(columns, rows))
        copies = tuple(row for row in rows if row[1] != "a") * 20
        duplicated = discover_consensus(Table(columns, rows + copies))
        self.assertEqual(base["candidates"], duplicated["candidates"])
        self.assertEqual(base["diagnostics"], duplicated["diagnostics"])
        self.assertEqual(base["diagnostics"]["reason"], "no_temporal_columns")

    def test_unhinted_discovery_preserves_same_values_from_distinct_entities(self):
        rows = tuple((entity, source, "malformed" if source == "a" else "08:10")
                     for entity in ("E0", "E1") for source in ("a", "b", "c", "d", "e"))
        columns = ("entity_id", "source", "observation")
        base = discover_consensus(Table(columns, rows))
        self.assertEqual(base["diagnostics"]["status"], "applicable")
        self.assertEqual(base["diagnostics"]["temporal_columns"][0]["nonempty_count"], 10)
        self.assertEqual(base["diagnostics"]["temporal_columns"][0]["parseable_count"], 8)
        self.assertTrue(base["candidates"][(0, 2)][0]["actionable"])
        duplicated = discover_consensus(Table(columns, rows + (rows[1],) * 30))
        self.assertEqual(base["diagnostics"]["temporal_columns"], duplicated["diagnostics"]["temporal_columns"])
        self.assertEqual(base["candidates"][(0, 2)], duplicated["candidates"][(0, 2)])

    def test_raw_vote_tie_does_not_produce_a_winner(self):
        table = fixture({"a": "broken", "b": "08:00", "c": "08:00", "d": "08:01", "e": "08:01"},
                        sources=("a", "b", "c", "d", "e"))
        result = discover_consensus(table)
        self.assertNotIn((0, 2), result["candidates"])
        self.assertGreater(result["diagnostics"]["proposal_reason_counts"]["no_unique_source_count_and_weight_winner"], 0)

    def test_exact_representation_tie_is_not_normalized_into_consensus(self):
        table = fixture({"a": "broken", "b": "08:00", "c": "8:00 a.m.", "d": "08:00:00"})
        self.assertEqual(candidate(table), [])

    def test_query_entity_cannot_affect_source_reliability(self):
        base = fixture(training_count=7)
        # Source d has poor format evidence only in other entities.
        training = tuple(row[:2] + (("broken" if row[1] == "d" else row[2]), row[3])
                         for row in base.rows[4:])
        first = Table(base.columns, base.rows[:4] + training)
        changed_query = tuple((row[0], row[1], "11:59 p.m." if row[1] != "a" else "different bad value", "01:01")
                              for row in base.rows[:4])
        second = Table(base.columns, changed_query + training)
        before = candidate(first)[0]["evidence"]["source_reliability"]
        after = candidate(second)[0]["evidence"]["source_reliability"]
        self.assertEqual(before, after)
        self.assertLess(before["d"]["score"], before["b"]["score"])
        self.assertEqual(before["d"]["training_entity_counts"]["format"], 7)
        self.assertFalse(before["b"]["calibrated_correctness"])

    def test_query_source_cannot_label_its_peers_training_agreement(self):
        base = fixture(training_count=6)
        changed = Table(base.columns, tuple(row[:2] + (("damaged" if row[1] == "a" and row[0] != "q" else row[2]), row[3])
                                           for row in base.rows))
        before = candidate(base)[0]["evidence"]["source_reliability"]
        after = candidate(changed)[0]["evidence"]["source_reliability"]
        self.assertEqual(before, after)

    def test_observed_exact_strings_and_original_preserved(self):
        value = " 7:05 a.m. "
        table = fixture({"a": "07:05", "b": value, "c": value, "d": value})
        result = candidate(table)[0]
        self.assertEqual(result["value"], value)
        self.assertEqual(result["evidence"]["original_value"], "07:05")
        self.assertNotEqual(result["value"], "07:05:00")
        self.assertTrue(result["actionable"])
        for coordinate, proposals in discover_consensus(table)["candidates"].items():
            self.assertLessEqual(len(proposals), 3)
            self.assertEqual(len({item["value"] for item in proposals}), len(proposals))
            self.assertTrue(all(item["value"] != table.rows[coordinate[0]][coordinate[1]] for item in proposals))

    def test_clock_only_overnight_and_timezone_ordering_not_assumed(self):
        for start, end in (("23:50", "00:20"), ("23:50-08:00", "00:20+02:00")):
            base = fixture({"a": "broken", "b": start, "c": start, "d": start})
            table = Table(base.columns, tuple(row[:3] + (end,) for row in base.rows))
            result = candidate(table)[0]
            checks = result["evidence"]["intrinsic_checks"]
            self.assertTrue(result["actionable"])
            self.assertFalse(checks["ordering_assumed_for_clock_only_values"])
            self.assertEqual(checks["temporal_relations"], [])
            self.assertIn("date_and_explicit_timezone_required_for_ordering", checks["skipped_temporal_relations"])

    def test_naive_dated_timestamps_do_not_assert_shared_timezone(self):
        start, end = "2025-05-03T10:00", "2025-05-03T09:00"
        base = fixture({"a": "broken", "b": start, "c": start, "d": start})
        table = Table(base.columns, tuple(row[:3] + (end,) for row in base.rows))
        result = candidate(table)[0]
        self.assertTrue(result["actionable"])
        self.assertEqual(result["evidence"]["intrinsic_checks"]["temporal_relations"], [])

    def test_unknown_negative_zero_offset_is_not_timezone_evidence(self):
        start, end = "2025-05-03T10:00-00:00", "2025-05-03T09:00+00:00"
        base = fixture({"a": "broken", "b": start, "c": start, "d": start})
        table = Table(base.columns, tuple(row[:3] + (end,) for row in base.rows))
        result = candidate(table)[0]
        self.assertTrue(result["actionable"])
        self.assertEqual(result["evidence"]["intrinsic_checks"]["temporal_relations"], [])

    def test_explicit_offsets_allow_defensible_cross_column_checks(self):
        start, end = "2025-05-03T10:00+02:00", "2025-05-03T09:00+00:00"
        base = fixture({"a": "broken", "b": start, "c": start, "d": start})
        table = Table(base.columns, tuple(row[:3] + (end,) for row in base.rows))
        result = candidate(table)[0]
        self.assertTrue(result["actionable"])
        self.assertTrue(result["evidence"]["intrinsic_checks"]["temporal_relations"][0]["passed"])
        weights = result["evidence"]["source_reliability"]
        # Other entities use clock starts, so they supply no chronology evidence.
        self.assertTrue(all(item["training_entity_counts"]["cross_column_consistency"] == 0 for item in weights.values()))

    def test_candidate_conflicting_with_explicit_query_chronology_is_conditional_only(self):
        start = "2025-05-03T10:00+00:00"
        base = fixture({"a": "broken", "b": start, "c": start, "d": start})
        rows = []
        for index, row in enumerate(base.rows):
            end = "2025-05-03T09:00+00:00" if index == 0 else "2025-05-03T11:00+00:00"
            rows.append(row[:3] + (end,))
        result = candidate(Table(base.columns, tuple(rows)))[0]
        self.assertFalse(result["actionable"])
        self.assertIn("explicit_timestamp_chronology_conflict", result["evidence"]["abstention_reasons"])

    def test_cross_column_consistency_enters_leave_entity_out_reliability(self):
        base = fixture(training_count=5)
        rows = []
        for row in base.rows:
            start = "broken" if row[0] == "q" and row[1] == "a" else "2025-05-03T10:00+00:00"
            end = "2025-05-03T09:00+00:00" if row[0] != "q" and row[1] == "d" else "2025-05-03T11:00+00:00"
            rows.append(row[:2] + (start, end))
        result = candidate(Table(base.columns, tuple(rows)))[0]
        weights = result["evidence"]["source_reliability"]
        self.assertLess(weights["d"]["cross_column_consistency_score"], weights["b"]["cross_column_consistency_score"])
        self.assertEqual(weights["d"]["training_entity_counts"]["cross_column_consistency"], 5)

    def test_composite_key_discovered_when_simple_keys_conflate_events(self):
        rows = []
        for entity in range(2):
            for run in range(2):
                for source in ("a", "b", "c"):
                    value = "bad" if (entity, run, source) == (0, 0, "a") else f"{8 + run}:{10 + entity:02d}"
                    rows.append((source, f"entity-{entity}", f"run-{run}", value))
        table = Table(("source", "record_id", "run", "event_time"), tuple(rows))
        result = discover_consensus(table)
        self.assertEqual([item["name"] for item in result["diagnostics"]["key_columns"]], ["record_id", "run"])
        self.assertTrue(result["candidates"][(0, 3)][0]["actionable"])

    def test_policy_is_fixed_validated_and_evidence_scores_bounded(self):
        with self.assertRaises(TypeError):
            DEFAULT_CONFIG["min_agreement"] = 0
        for config in ({"dataset": "fictional"}, {"min_margin": -1}, {"format_weight": .9}, {"max_key_columns": 3}):
            with self.assertRaises(ValueError):
                discover_consensus(fixture(), config)
        for proposals in discover_consensus(fixture())["candidates"].values():
            for item in proposals:
                self.assertGreaterEqual(item["score"], 0)
                self.assertLessEqual(item["score"], 1)


if __name__ == "__main__":
    unittest.main()
