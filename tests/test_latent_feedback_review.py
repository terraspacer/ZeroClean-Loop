"""Synthetic review of the frozen feedback implementation, including known gaps."""
import copy
import gzip
import json
import sys

import numpy as np
import pytest

from experimental import latent_consensus_feedback as feedback
from experimental import validate_latent_consensus_feedback as validator
from zeroclean.data import DirtyTable
from zeroclean.types import Prediction, digest


def toy(rows=None):
    rows = rows or [("e0", "A", "B", "U"), ("e1", "X", "B", "U"),
                    ("e2", "A", "Y", "U"), ("e3", "A", "B", "U")]
    columns = ("entity", "a", "b", "c")
    return DirtyTable("synthetic", columns, tuple(map(tuple, rows)), digest([columns, rows]))


def cell(table, row, column, candidate):
    original = table.rows[row][column]
    return {"row": row, "column_index": column, "column": table.columns[column],
            "original": original, "candidate": candidate,
            "original_weight_interval": [.99, .995] if candidate == original else [0, 1e-6],
            "candidate_weight_lower": .99 if candidate == original else .999999,
            "source_assumptions": {"independent_names": {}, "cluster_near_agreement": {}}}


def rule(source, kind="multi"):
    return {"id": "review_guard", "target": "a", "kind": kind,
            "source": source, "description": "Synthetic safety contract"}


def inputs(table, cells, rules=()):
    return ({"dirty_sha256": table.fingerprint, "rules": list(rules), "dependencies": []},
            {"dirty_sha256": table.fingerprint, "cells": cells})


def assert_trace_consistency(table, prior, prediction, audit):
    for trace in audit["trace"]:
        key = trace["row"], trace["column_index"]
        low, high = trace["original_weight_interval"]
        expected = 1 - low if trace["candidate"] == trace["original"] else 1 - high
        assert prediction.probabilities[key] == expected
        assert trace["final_repair"] == prediction.repairs.get(key)
        assert trace["feedback_ed_positive"] == bool(prediction.probabilities[key] >= .5)
        assert trace["repair_changed"] == (prediction.repairs.get(key) != prior.repairs.get(key))
        assert trace["final_tuple"][table.columns[key[1]]] == prediction.repairs.get(key, table.rows[key[0]][key[1]])


def test_unsupported_cells_and_inputs_survive_supported_same_row_changes():
    table = toy()
    probabilities = np.linspace(.01, .99, np.prod(table.shape)).reshape(table.shape)
    prior = Prediction(probabilities.copy(), {(0, 1): "prior a", (0, 3): "  same-row literal  ",
                                            (2, 2): "", (3, 1): "unreviewed"})
    artifact, evidence = inputs(table, [cell(table, 0, 1, "A"), cell(table, 1, 2, "Y")])
    before = copy.deepcopy((prior, artifact, evidence))
    prediction, audit, _ = feedback.apply_feedback(table, prior, artifact, evidence)
    supported = {(0, 1), (1, 2)}
    for key in np.ndindex(table.shape):
        if key not in supported:
            assert prediction.probabilities[key] == prior.probabilities[key]
            assert prediction.repairs.get(key) == prior.repairs.get(key)
    assert prior.repairs == before[0].repairs
    assert np.array_equal(prior.probabilities, before[0].probabilities)
    assert (artifact, evidence) == before[1:]
    assert_trace_consistency(table, prior, prediction, audit)


def test_absent_evidence_preserves_exact_prior_even_preexisting_guard_failures():
    table = toy()
    prior = Prediction(np.full(table.shape, .37), {(0, 1): "X", (0, 2): "Y"})
    guard = rule("def check(row):\n    return row['a'] == 'X' and row['b'] == 'Y'")
    artifact, evidence = inputs(table, [], [guard])
    prediction, audit, bundle = feedback.apply_feedback(table, prior, artifact, evidence)
    assert prediction.repairs == prior.repairs
    assert np.array_equal(prediction.probabilities, prior.probabilities)
    assert audit["trace"] == audit["joint_guard"] == bundle["cells"] == []


def test_original_valid_pruning_can_retain_prior_repair_when_runtime_blocks_removal():
    # One dirty failure in 25 rows is accepted by the strictly-below-5% gate.
    table = toy([("e0", "bad", "B", "U")] + [(f"e{i}", "1", "B", "U") for i in range(1, 25)])
    prior = Prediction(np.full(table.shape, .9), {(0, 1): "1"})
    guard = rule("def check(row):\n    return int(row['a']) < 0", kind="single")
    artifact, evidence = inputs(table, [cell(table, 0, 1, "bad")], [guard])
    prediction, audit, bundle = feedback.apply_feedback(table, prior, artifact, evidence)
    trace = audit["trace"][0]
    assert len(audit["accepted_rules"]) == 1
    assert prediction.probabilities[0, 1] == pytest.approx(.01)
    assert prediction.repairs[0, 1] == "1"
    assert trace["requested_repair"] is None
    assert "counterfactual_runtime_failure" in trace["individual_gate_blockers"]
    assert not trace["feedback_ed_positive"] and trace["final_repair"] == "1"
    assert bundle["cells"][0]["candidates"] == [{"value": "1", "score": None, "origin": "frozen_prior_repair"}]
    assert_trace_consistency(table, prior, prediction, audit)


@pytest.mark.parametrize("runtime", [False, True])
def test_individual_veto_retains_positive_ed_and_exact_prior_ec(runtime):
    table = toy()
    expression = ("return int(row['a']) < 0" if runtime else "return True")
    guard = rule("def check(row):\n    if row['a'] == 'unsafe' and row['b'] == 'B':\n        "
                 + expression + "\n    return False", kind="single" if runtime else "multi")
    prior = Prediction(np.full(table.shape, .1), {(0, 1): "prior literal"})
    artifact, evidence = inputs(table, [cell(table, 0, 1, "unsafe")], [guard])
    prediction, audit, _ = feedback.apply_feedback(table, prior, artifact, evidence)
    assert prediction.probabilities[0, 1] >= .98
    assert prediction.repairs == prior.repairs
    expected = "counterfactual_runtime_failure" if runtime else "new_multi_column_violation"
    assert expected in audit["trace"][0]["individual_gate_blockers"]
    assert_trace_consistency(table, prior, prediction, audit)


@pytest.mark.parametrize("runtime", [False, True])
def test_joint_veto_restores_all_prior_actions_preserving_unreviewed_and_ed(runtime):
    table = toy()
    ending = "return int(row['a']) < 0" if runtime else "return True"
    guard = rule("def check(row):\n    if row['a'] == 'X' and row['b'] == 'Y':\n        "
                 + ending + "\n    return False", kind="single" if runtime else "multi")
    prior = Prediction(np.full(table.shape, .17), {(0, 1): "P", (0, 3): "  untouched  "})
    artifact, evidence = inputs(table, [cell(table, 0, 1, "X"), cell(table, 0, 2, "Y")], [guard])
    prediction, audit, _ = feedback.apply_feedback(table, prior, artifact, evidence)
    assert all(t["individual_gate_passed"] and t["joint_rollback"] for t in audit["trace"])
    assert prediction.repairs == prior.repairs
    assert all(prediction.probabilities[0, j] >= .98 for j in (1, 2))
    assert prediction.probabilities[0, 3] == .17
    assert audit["joint_guard"][0]["runtime_failure"] is runtime
    assert_trace_consistency(table, prior, prediction, audit)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="Frozen gates compare dirty; final assertion compares prior, so restored constraints raise instead of rollback")
@pytest.mark.parametrize("candidate", ["bad", "also bad"])
def test_feedback_must_rollback_constraints_restored_by_prior_repairs(candidate):
    table = toy([("e0", "bad", "lock", "U"), ("e1", "fixed", "open", "U")])
    guard = rule("def check(row):\n    return row['a'] != 'fixed' and row['b'] == 'lock'")
    prior = Prediction(np.full(table.shape, .9), {(0, 1): "fixed"})
    artifact, evidence = inputs(table, [cell(table, 0, 1, candidate)], [guard])
    prediction, audit, _ = feedback.apply_feedback(table, prior, artifact, evidence)
    assert prediction.repairs == prior.repairs
    assert audit["trace"][0]["joint_rollback"]
    assert_trace_consistency(table, prior, prediction, audit)


def freeze_for_validator(base, table, prior, artifact, evidence, mutate=None):
    """Create a wholly synthetic structural fixture, without calling a model."""
    prediction, audit, bundle = feedback.apply_feedback(table, prior, artifact, evidence)
    if mutate:
        mutate(prediction, audit, bundle)
    frozen = base / "frozen_inputs"
    frozen.mkdir(parents=True)
    prior.save(frozen / "prior_prediction.json")
    prediction.save(base / "prediction.json")
    for filename, body in (("statistical_evidence.json.gz", evidence),
                           ("frozen_inputs/dirty_table.json.gz", {"name": table.name, "columns": table.columns,
                              "rows": table.rows, "fingerprint": table.fingerprint})):
        with gzip.open(base / filename, "wt") as handle:
            json.dump(body, handle)
    audit["variant_prediction_sha256"] = feedback.sha(base / "prediction.json")
    bundle.pop("request_hash")
    bundle["parent_variant_prediction_sha256"] = feedback.sha(base / "prediction.json")
    bundle["request_hash"] = digest(bundle)
    with gzip.open(base / "audit.json.gz", "wt") as handle:
        json.dump(audit, handle)
    (base / "verification_bundle.json").write_text(json.dumps(bundle))
    manifest = {"input_hashes": {}, "frozen_input_hashes": {}, "dirty_sha256": table.fingerprint,
                "frozen_dirty_strings_sha256": feedback.sha(frozen / "dirty_table.json.gz")}
    (base / "pre_run_manifest.json").write_text(json.dumps(manifest))
    outputs = {str(path): feedback.sha(path) for path in base.iterdir() if path.is_file()}
    (base / "output_manifest.json").write_text(json.dumps({"outputs": outputs, "method_status": "synthetic"}))


def test_frozen_validator_accepts_synthetic_complete_structure(tmp_path, monkeypatch):
    table = toy()
    artifact, evidence = inputs(table, [cell(table, 0, 1, "X")])
    freeze_for_validator(tmp_path, table, Prediction(np.full(table.shape, .1)), artifact, evidence)
    monkeypatch.setattr(sys, "argv", ["validator", "--directory", str(tmp_path)])
    validator.main()
    assert json.loads((tmp_path / "validation.json").read_text())["status"] == "passed"


@pytest.mark.parametrize("change", ["unsupported_probability", "unsupported_repair", "prior_restored_guard"])
def test_validator_rejects_structural_tampering_even_when_output_hashes_are_updated(tmp_path, monkeypatch, change):
    table = toy()
    guard = rule("def check(row):\n    return row['a'] != 'fixed' and row['b'] == 'B'")
    artifact, evidence = inputs(table, [cell(table, 0, 1, "fixed")], [guard])
    prior = Prediction(np.full(table.shape, .1), {(0, 1): "fixed"})
    def mutate(prediction, audit, bundle):
        if change == "unsupported_probability":
            prediction.probabilities[1, 3] = .8
        elif change == "unsupported_repair":
            prediction.repairs[1, 3] = "altered"
        else:
            prediction.repairs[0, 1] = "bad"
            audit["trace"][0]["final_repair"] = "bad"
    freeze_for_validator(tmp_path, table, prior, artifact, evidence, mutate)
    monkeypatch.setattr(sys, "argv", ["validator", "--directory", str(tmp_path)])
    with pytest.raises(AssertionError):
        validator.main()
    assert not (tmp_path / "validation.json").exists()


@pytest.mark.xfail(strict=True, reason="Frozen validator binds bundle bytes/coordinates but does not cross-check its alternatives against audit evidence")
def test_validator_should_reject_bundle_alternatives_unrelated_to_frozen_evidence(tmp_path, monkeypatch):
    table = toy()
    artifact, evidence = inputs(table, [cell(table, 0, 1, "X")])
    def mutate(prediction, audit, bundle):
        bundle["cells"][0]["candidates"][0]["value"] = "unrelated alternative"
    freeze_for_validator(tmp_path, table, Prediction(np.full(table.shape, .1)), artifact, evidence, mutate)
    monkeypatch.setattr(sys, "argv", ["validator", "--directory", str(tmp_path)])
    with pytest.raises(AssertionError):
        validator.main()
