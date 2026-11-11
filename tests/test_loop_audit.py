"""Synthetic inference-contract regressions; no benchmark files are loaded."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

import evaluate_e2e
from zeroclean.data import DirtyTable, sha256
from zeroclean.loop import LoopConfig, build_candidates, infer_from_candidates
from zeroclean.optimize import hill_climb, proxy_objective
from zeroclean.statistics import DirtyIndex
from zeroclean.types import Prediction, digest


def synthetic_table():
    return DirtyTable(
        "audit", ("left", "right"), (("old", "steady"), ("other", "still")),
        "synthetic-dirty-sha256",
    )


def request(table=None, *, score=.9, rules=None, cells=None):
    table = table or synthetic_table()
    if cells is None:
        cells = [{
            "row": 0, "column": table.columns[0], "column_index": 0,
            "original": table.rows[0][0], "base_probability": .9,
            "tuple": dict(zip(table.columns, table.rows[0])),
            "original_valid_score": None, "original_rule_violations": 0,
            "candidates": [{
                "value": "new", "score": score, "origin": "synthetic",
                "delta_features": [0., 0., 0., 0.],
                "introduced_violation": False, "counterfactual_error": False,
            }],
        }]
    probabilities = np.full(table.shape, .05)
    for cell in cells:
        probabilities[cell["row"], cell["column_index"]] = cell["base_probability"]
    body = {
        "dataset": table.name, "dirty_sha256": table.fingerprint,
        "model": "gpt-6.1-sol", "reasoning_effort": "xhigh",
        "config": asdict(LoopConfig()), "rules": rules or [],
        "rule_validation": [], "base_probabilities": probabilities.tolist(),
        "cells": cells, "artifact_hash": "synthetic-artifact",
    }
    body["request_hash"] = digest(body)
    return body


def reply(payload, *, cells=None):
    if cells is None:
        cells = [{
            "row": 0, "column": "left", "original_valid_score": .01,
            "selected_value": "new", "confidence": .99,
            "candidates": [{"value": "new", "score": .99}],
        }]
    return {
        "request_hash": payload["request_hash"], "model": payload["model"],
        "reasoning_effort": payload["reasoning_effort"],
        "dirty_sha256": payload["dirty_sha256"], "cells": cells,
    }


def test_candidate_content_hash_is_recomputed():
    payload = request()
    payload["cells"][0]["candidates"][0]["score"] = .99
    with pytest.raises(ValueError, match="hash"):
        infer_from_candidates(synthetic_table(), payload)


def test_proposals_require_dirty_fingerprint(tmp_path):
    table = synthetic_table()
    destination = tmp_path / table.name
    destination.mkdir()
    (destination / "proposals.json").write_text(json.dumps({"cells": []}))
    with pytest.raises(ValueError, match="fingerprint"):
        build_candidates(table, {}, artifact_root=tmp_path)


def test_singleton_supplies_no_own_context_or_domain_support():
    table = DirtyTable("singleton", ("key", "value"), (("a", "unique"),), "singleton-input")
    index = DirtyIndex(table, discover=False)
    assert index.vicinity(0, 1, "unique") == 0
    assert index.domain_support(0, 1, "unique") == 0
    assert index.contextual_candidates(0, 1) == []


def test_counterfactual_key_uses_the_new_group_without_subtracting_a_peer():
    table = DirtyTable(
        "groups", ("key", "value"),
        (("a", "typo"), ("a", "blue"), ("a", "blue"), ("b", "red"), ("b", "red")),
        "group-input",
    )
    index = DirtyIndex(table, [{
        "determinants": ["key"], "dependent": "value", "min_support": 2, "confidence": 1.,
    }], discover=False)
    support, evidence = index.conditional(0, 1, "red", row=("b", "red"))
    assert support == 1
    assert evidence[0]["peers"] == 2
    assert index.conditional(0, 1, "typo")[0] == 0


@pytest.mark.parametrize("field", ["request_hash", "model", "reasoning_effort", "dirty_sha256"])
@pytest.mark.parametrize("missing", [False, True])
def test_verifier_requires_matching_provenance(field, missing):
    payload = request()
    response = reply(payload)
    if missing:
        response.pop(field)
    else:
        response[field] = "foreign-provenance"
    with pytest.raises(ValueError):
        infer_from_candidates(synthetic_table(), payload, response)


def test_optional_verifier_dataset_must_match():
    payload = request()
    response = reply(payload)
    response["dataset"] = "another-dataset"
    with pytest.raises(ValueError):
        infer_from_candidates(synthetic_table(), payload, response)


@pytest.mark.parametrize("coordinate", [
    {"row": 1, "column": "left"},  # Inside the table, outside this request.
    {"row": -1, "column": "left"},
    {"row": 2, "column": "left"},
    {"row": .5, "column": "left"},
    {"row": True, "column": "left"},
    {"row": 0, "column": -1},
    {"row": 0, "column": 2},
    {"row": 0, "column": .5},
    {"row": 0, "column": True},
    {"row": 0, "column": "unknown-column"},
])
def test_verifier_coordinates_are_exact_request_members(coordinate):
    payload = request()
    response = reply(payload)
    response["cells"][0].update(coordinate)
    with pytest.raises(ValueError):
        infer_from_candidates(synthetic_table(), payload, response)


def test_duplicate_verifier_coordinates_are_rejected():
    payload = request()
    response = reply(payload)
    response["cells"].append(deepcopy(response["cells"][0]))
    with pytest.raises(ValueError):
        infer_from_candidates(synthetic_table(), payload, response)


def test_optional_verifier_original_must_match():
    payload = request()
    response = reply(payload)
    response["cells"][0]["original"] = "a different original"
    with pytest.raises(ValueError):
        infer_from_candidates(synthetic_table(), payload, response)


@pytest.mark.parametrize("field", ["original_valid_score", "confidence", "candidate_score"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -.01, 1.01])
def test_verifier_scores_are_finite_unit_values(field, invalid):
    payload = request()
    response = reply(payload)
    cell = response["cells"][0]
    if field == "candidate_score":
        cell["candidates"][0]["score"] = invalid
    else:
        cell[field] = invalid
    with pytest.raises(ValueError):
        infer_from_candidates(synthetic_table(), payload, response)


@pytest.mark.parametrize("field", ["selected_value", "candidate_value"])
def test_verifier_cannot_invent_values(field):
    payload = request()
    response = reply(payload)
    cell = response["cells"][0]
    if field == "selected_value":
        cell["selected_value"] = "never-proposed"
    else:
        cell["candidates"][0]["value"] = "never-proposed"
    with pytest.raises(ValueError):
        infer_from_candidates(synthetic_table(), payload, response)


def test_duplicate_verifier_candidate_values_are_rejected():
    payload = request()
    response = reply(payload)
    candidates = response["cells"][0]["candidates"]
    candidates.append(deepcopy(candidates[0]))
    with pytest.raises(ValueError):
        infer_from_candidates(synthetic_table(), payload, response)


@pytest.mark.parametrize("candidate_level", [False, True])
def test_joint_permission_is_a_boolean(candidate_level):
    payload = request()
    response = reply(payload)
    cell = response["cells"][0]
    target = cell["candidates"][0] if candidate_level else cell
    target["joint_repair_required"] = "false"
    with pytest.raises(ValueError):
        infer_from_candidates(synthetic_table(), payload, response)


def test_unsafe_repair_does_not_erase_verified_detection():
    payload = request()
    payload["cells"][0]["candidates"][0]["introduced_violation"] = True
    payload["request_hash"] = digest({k: v for k, v in payload.items() if k != "request_hash"})
    response = reply(payload)
    response["cells"][0].update(selected_value=None, confidence=0.)
    prediction = infer_from_candidates(synthetic_table(), payload, response)
    assert not prediction.repairs
    assert prediction.probabilities[0, 0] >= .9


@pytest.mark.parametrize("explicit_abstention", [True, False])
def test_null_selection_abstains_but_omitted_selection_can_rescore(explicit_abstention):
    payload = request()
    response = reply(payload)
    cell = response["cells"][0]
    if explicit_abstention:
        cell.update(selected_value=None, confidence=0.)
    else:
        cell.pop("selected_value")
    prediction = infer_from_candidates(synthetic_table(), payload, response)
    assert prediction.repairs == ({} if explicit_abstention else {(0, 0): "new"})
    # Original-invalid evidence survives either decision about a replacement.
    assert prediction.probabilities[0, 0] >= .9


def joint_request(*, runtime_error=False):
    originals = ("1", "1") if runtime_error else ("old", "old")
    replacements = ("0", "0") if runtime_error else ("new", "new")
    table = DirtyTable("audit", ("left", "right"), (originals,), "joint-input")
    cells = []
    for j, value in enumerate(replacements):
        cell = request(table)["cells"][0]
        cell.update(column=table.columns[j], column_index=j, original=originals[j])
        cell["candidates"][0].update(
            value=value, score=.99, introduced_violation=not runtime_error,
        )
        cells.append(cell)
    source = (
        "def check(row):\n    return 1 / (int(row['left']) + int(row['right'])) < 0"
        if runtime_error else
        "def check(row):\n    return row['left'] != row['right']"
    )
    rules = [{
        "id": "joint", "kind": "multi", "target": "left",
        "description": "synthetic tuple consistency", "source": source,
    }]
    payload = request(table, cells=cells, rules=rules)
    response = reply(payload, cells=[{
        "row": 0, "column": table.columns[j], "original_valid_score": .01,
        "selected_value": value, "confidence": .99,
        "candidates": [{"value": value, "score": .99, "joint_repair_required": True}],
    } for j, value in enumerate(replacements)])
    return table, payload, response


def test_candidate_joint_permission_survives_a_consistent_final_tuple():
    table, payload, response = joint_request()
    prediction = infer_from_candidates(table, payload, response)
    assert prediction.repairs == {(0, 0): "new", (0, 1): "new"}
    assert proxy_objective(table, prediction, payload) > 0


def test_final_tuple_runtime_error_rolls_back_all_row_repairs():
    table, payload, response = joint_request(runtime_error=True)
    prediction = infer_from_candidates(table, payload, response)
    assert not prediction.repairs
    # A failed replacement does not negate the independent original-invalid verdict.
    assert (prediction.probabilities >= .9).all()


def test_proxy_uses_accepted_verifier_confidence_during_hill_climb():
    payload = request(score=.70)
    response = reply(payload)
    response["cells"][0].update(confidence=.84)
    response["cells"][0]["candidates"][0]["score"] = .84
    table = synthetic_table()
    _, prediction, search = hill_climb(table, payload, response, max_steps=1)
    assert prediction.repairs == {(0, 0): "new"}
    assert proxy_objective(table, prediction, payload) > 0
    assert search["history"][0]["proxy"] > 0


def test_evaluator_rejects_changed_dirty_input_before_evaluation(tmp_path, monkeypatch):
    table = synthetic_table()
    prediction_path = tmp_path / "predictions" / table.name / "zeroclean_loop.json"
    Prediction(np.full(table.shape, .05), metadata={"dirty_sha256": "older-input"}).save(prediction_path)
    frozen_path = tmp_path / "frozen" / f"{table.name}.json"
    frozen_path.parent.mkdir()
    frozen_path.write_text(json.dumps({
        "dirty_sha256": "older-input", "frozen_before_evaluation": True,
        "prediction_files": {prediction_path.name: sha256(prediction_path)},
    }))
    monkeypatch.setattr(evaluate_e2e, "load_dirty", lambda _: table)

    def must_not_evaluate(*args, **kwargs):
        raise AssertionError("A stale dirty input reached evaluation")

    import zeroclean.evaluation
    monkeypatch.setattr(zeroclean.evaluation, "evaluate", must_not_evaluate)
    with pytest.raises(ValueError, match="(?i)dirty|fingerprint|frozen"):
        evaluate_e2e.evaluate_all([table.name], tmp_path, {})
