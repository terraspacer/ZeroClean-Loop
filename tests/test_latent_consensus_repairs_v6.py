"""V6 synthetic tests only; all earlier source and test modules stay immutable."""
import hashlib
import json

import numpy as np
import pytest

from zeroclean.data import DirtyTable
from zeroclean.types import Prediction
from experimental import latent_consensus_mean_repairs_v6 as v6


def table(rows, columns=("entity", "a", "b")):
    rows = tuple(tuple(row) for row in rows)
    return DirtyTable("synthetic", tuple(columns), rows,
        hashlib.sha256(json.dumps([columns, rows]).encode()).hexdigest())


def nodes(weights, second=None):
    result = {}
    for mode, values in zip(v6.MODES, (weights, weights if second is None else second)):
        values = dict(values)
        unknown = 1-sum(values.values())
        assert unknown >= -1e-14
        vector = list(values.values())+[max(0, unknown)]
        result[mode] = {"candidate_strings": list(values), "scenarios": [
            {"omitted_group": None, "selected_model_index": i, "posterior_weights": vector}
            for i in range(17)]}
    return result


def selection(original="9:00 a.m.", candidate="10:00 a.m.", probability=.999998):
    return v6.rank_observed_alternatives(original, nodes({original: 1e-6, candidate: probability}))


def cell(data, row, column, ranking=None, eligible=True):
    return {"row": row, "column_index": column, "column": data.columns[column],
            "original": data.rows[row][column], "eligible": eligible, "selection": ranking,
            "claim_context_id": "synthetic"}


def artifact(data, rules=()):
    return {"dirty_sha256": data.fingerprint, "rules": list(rules), "dependencies": []}


def rule(source, kind="multi", target="a"):
    return {"id": "synthetic_guard", "target": target, "kind": kind,
            "source": source, "description": "Synthetic v6 guard"}


def run(data, cells, rules=(), prior_repairs=None, scores=None):
    prior = Prediction(np.full(data.shape, .1), prior_repairs or {})
    fixed = Prediction(np.full(data.shape, .9) if scores is None else scores)
    evidence = {"dirty_sha256": data.fingerprint, "cells": cells}
    prediction, audit = v6.apply_repair_policy(data, prior, fixed, artifact(data, rules), evidence)
    assert np.array_equal(prediction.probabilities, fixed.probabilities)
    return prediction, audit


def test_v6_shared_raw_ranking_top_three_and_absent_alternatives():
    original = "9:00 a.m."
    ranking = v6.rank_observed_alternatives(original, nodes({original: .01,
        "10:00 a.m.": .4, "11:00 a.m.": .3, "12:00 p.m.": .2, "1:00 p.m.": .08}))
    assert [c["value"] for c in ranking["top3"]] == ["10:00 a.m.", "11:00 a.m.", "12:00 p.m."]
    assert ranking["renormalized"] is False
    incompatible = v6.rank_observed_alternatives(original,
        nodes({original: .1, "10:00 a.m.": .9}, {original: .1, "11:00 a.m.": .9}))
    assert incompatible["top3"] == []
    assert incompatible["residual_unshared_or_unknown_mass"] == pytest.approx(.9)
    assert incompatible["top_gate_blockers"] == ["no_shared_observed_nonoriginal"]
    with pytest.raises(ValueError, match="absent"):
        v6.rank_observed_alternatives("absent", nodes({original: .1, "10:00 a.m.": .9}))


def test_v6_tie_and_margin_gates_do_not_pick_a_lexical_winner():
    original = "9:00 a.m."
    tied = v6.rank_observed_alternatives(original, nodes({original: .08,
        "10:00 a.m.": .45, "11:00 a.m.": .45}))
    assert "top_nonoriginal_not_unique" in tied["top_gate_blockers"]
    assert tied["top3"][0]["value"] == "10:00 a.m."
    admitted = v6.rank_observed_alternatives(original, nodes({original: .1, "10:00 a.m.": .85}))
    assert admitted["top_gate_blockers"] == []
    assert admitted["top_margin"] >= .2


@pytest.mark.parametrize("original,candidate", [
    ("9:30 a.m.", "9:30a"), ("09:30 AM", "9:30 a.m."), ("21:30", "9:30 p.m."),
    ("9:30aDec 1", "9:30 a.m."), ("Dec 1 9:30 a.m.", "Dec 1 10:00 a.m."),
    ("Not Available", "10:00 a.m."), ("Contact Airline", "10:00 a.m."),
    ("9:30 a.m. (EST)", "10:00 a.m. (EST)"), ("9:30 a.m.", "10:00 a.m. (UTC)"),
    ("9:30 a.m. Delayed", "10:00 a.m."), ("9:30 a.m./10:30 a.m.", "11:00 a.m.")])
def test_v6_representation_and_changed_annotation_veto(original, candidate):
    assert not v6.representation_gate(original, candidate)["passed"]


def test_v6_bare_clock_change_is_not_vetoed_by_unrelated_status_peer():
    data = table([("e", "9:00 a.m.", "target"), ("e", "10:00 a.m.", "peer"),
                  ("e", "Not Available", "other peer")])
    rank = v6.rank_observed_alternatives("9:00 a.m.", nodes({"9:00 a.m.": 1e-6,
        "10:00 a.m.": .999998, "Not Available": 0}))
    prediction, audit = run(data, [cell(data, 0, 1, rank)])
    assert prediction.repairs[0, 1] == "10:00 a.m."
    assert audit["trace"][0]["candidate_checks"][0]["representation_gate"]["peer_annotation_veto"] is False


def test_v6_no_lower_candidate_fallback_after_top_annotation_veto():
    original = "9:00 a.m."
    data = table([("e", original, "target"), ("e", "10:00 a.m. (EST)", "peer"),
                  ("e", "10:05 a.m.", "other peer")])
    rank = v6.rank_observed_alternatives(original, nodes({original: .04,
        "10:00 a.m. (EST)": .9, "10:05 a.m.": .05}))
    prediction, audit = run(data, [cell(data, 0, 1, rank)])
    assert prediction.repairs == {}
    checks = audit["trace"][0]["candidate_checks"]
    assert not checks[0]["representation_gate"]["passed"]
    assert "not_top_no_lower_rank_fallback" in checks[1]["blockers"]
    assert all(len(c["delta_features"]) == 4 for c in checks)


def test_v6_individual_parent_multi_guard_blocks_new_repair():
    data = table([("e", "9:00 a.m.", "lock"), ("e", "10:00 a.m.", "open")])
    guard = rule("def check(row):\n    return row['a'] == '10:00 a.m.' and row['b'] == 'lock'")
    prediction, audit = run(data, [cell(data, 0, 1, selection())], [guard])
    assert prediction.repairs == {}
    assert "new_parent_multi_column_violation" in audit["trace"][0]["candidate_checks"][0]["blockers"]


def test_v6_individual_parent_runtime_guard_blocks_new_repair():
    data = table([("e", "9:00 a.m.", "lock"), ("e", "10:00 a.m.", "open")])
    guard = rule("def check(row):\n    if row['a'] == '10:00 a.m.' and row['b'] == 'lock':\n        return int(row['b']) < 0\n    return False")
    prediction, audit = run(data, [cell(data, 0, 1, selection())], [guard])
    assert prediction.repairs == {}
    assert "parent_counterfactual_runtime_failure" in audit["trace"][0]["candidate_checks"][0]["blockers"]


def test_v6_four_delta_confidence_gate_preserves_ed():
    data = table([(f"e{i}", "9:00 a.m." if i < 99 else "10:00 a.m.", "fixed") for i in range(100)])
    rank = v6.rank_observed_alternatives("9:00 a.m.", nodes({"9:00 a.m.": .19, "10:00 a.m.": .8}))
    guard = rule("def check(row):\n    return row['a'] == '10:00 a.m.'", kind="single")
    prediction, audit = run(data, [cell(data, 0, 1, rank)], [guard])
    check = audit["trace"][0]["candidate_checks"][0]
    assert check["delta_features"][0] == -1 and check["delta_features"][1] < -.9
    assert check["correction_confidence"] < .8
    assert "counterfactual_confidence_below_threshold" in check["blockers"]
    assert prediction.repairs == {}


def test_v6_joint_rollback_restores_coherent_parent_and_unsupported_prior():
    data = table([("e0", "9:00 a.m.", "11:00 a.m.", "U"),
                  ("e1", "10:00 a.m.", "11:00 a.m.", "U"),
                  ("e2", "9:00 a.m.", "12:00 p.m.", "U")], ("entity", "a", "b", "c"))
    guard = rule("def check(row):\n    return row['a'] == '10:00 a.m.' and row['b'] == '12:00 p.m.'")
    cells = [cell(data, 0, 1, selection()), cell(data, 0, 2, selection("11:00 a.m.", "12:00 p.m."))]
    prior = {(0, 1): "9:15 a.m.", (0, 3): "preserved unsupported"}
    prediction, audit = run(data, cells, [guard], prior)
    assert prediction.repairs == prior
    assert audit["joint_guard"][0]["rollback_to_coherent_prior"]
    assert all(t["new_choice"] is not None and t["joint_rollback"] for t in audit["trace"])


def test_v6_ed_coherence_removes_incompatible_prior_without_score_normalization():
    data = table([("e", "9:00 a.m.", "keep")])
    scores = np.full(data.shape, .9); scores[0, 1] = .2
    prediction, audit = run(data, [cell(data, 0, 1, None, eligible=False)],
                            prior_repairs={(0, 1): "10:00 a.m.", (0, 2): "prior value"}, scores=scores)
    assert prediction.repairs == {(0, 2): "prior value"}
    assert prediction.probabilities[0, 1] == .2
    assert audit["coherence_rollbacks"][0]["removed_prior_repair"] == "10:00 a.m."


def test_v6_ed_coherence_fails_closed_when_prior_rollback_introduces_violation():
    data = table([("e", "9:00 a.m.", "lock")])
    guard = rule("def check(row):\n    return row['a'] == '9:00 a.m.' and row['b'] == 'lock'")
    prior = Prediction(np.full(data.shape, .9), {(0, 1): "10:00 a.m."})
    scores = np.full(data.shape, .9); scores[0, 1] = .2
    fixed = Prediction(scores.copy())
    with pytest.raises(v6.CoherenceConflict, match="ED-incompatible"):
        v6.apply_repair_policy(data, prior, fixed, artifact(data, [guard]),
            {"dirty_sha256": data.fingerprint, "cells": [cell(data, 0, 1, None, eligible=False)]})
    assert prior.repairs == {(0, 1): "10:00 a.m."}
    assert np.array_equal(fixed.probabilities, scores)


def test_v6_actual_audit_is_native_json_serializable_without_scalar_coercion():
    data = table([("e", "9:00 a.m.", "target"), ("e", "10:00 a.m.", "peer")])
    prediction, audit = run(data, [cell(data, 0, 1, selection())])
    encoded = json.dumps(audit, allow_nan=False)
    assert json.loads(encoded)["trace"][0]["considered_for_new_repair"] is True
    assert json.dumps({"ed_unchanged": np.array_equal(prediction.probabilities, np.full(data.shape, .9))})
