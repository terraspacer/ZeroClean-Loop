"""Matched-ablation preparation and freezing use synthetic dirty inputs only."""
import json
from pathlib import Path

import numpy as np
import pytest

import evaluate_e2e
from zeroclean.ablations import (
    COUNTERFACTUAL_FIELDS, prepare_matched_ablations, run_matched_ablations,
)
from zeroclean.data import DirtyTable, sha256
from zeroclean.loop import LoopConfig
from zeroclean.types import Prediction, digest, write_json


def fixture_inputs(tmp_path):
    table = DirtyTable(
        "synthetic", ("value", "context"),
        (("bad", "x"), ("new", "x"), ("new", "x"), ("new", "y")), "synthetic-dirty",
    )
    artifact_root = tmp_path / "llm"
    artifact_path = artifact_root / table.name / "pass2.json"
    artifact = {
        "dirty_sha256": table.fingerprint, "model": "gpt-6.1-sol", "reasoning_effort": "xhigh",
        "rules": [
            {"id": "single", "target": "value", "kind": "single", "description": "Synthetic bad token",
             "source": "def check(row):\n    return row['value'] == 'bad'"},
            {"id": "multi", "target": "value", "kind": "multi", "description": "Synthetic pair",
             "source": "def check(row):\n    return row['value'] == 'bad' and row['context'] == 'x'"},
        ],
        "dependencies": [{"determinants": ["context"], "dependent": "value",
                          "min_support": 2, "confidence": .9}],
        "value_repairs": {"value": {"bad": "new"}},
    }
    write_json(artifact_path, artifact)
    baseline_path = tmp_path / "output" / "predictions" / table.name / "zeroed.json"
    probabilities = np.full(table.shape, .05)
    probabilities[0, 0], probabilities[1, 1] = .61, .62
    Prediction(probabilities, metadata={
        "input_fingerprint": table.fingerprint, "reproduced_upstream": True,
    }).save(baseline_path)
    # An available adaptation must never replace the selected source detector.
    Prediction(np.full(table.shape, .04), metadata={
        "input_fingerprint": table.fingerprint, "reproduced_upstream": False,
    }).save(baseline_path.with_name("adapted_zeroed.json"))
    return table, artifact_root, artifact_path, baseline_path


def prepare_fixture(tmp_path):
    table, artifact_root, artifact_path, baseline_path = fixture_inputs(tmp_path)
    directory = prepare_matched_ablations(
        table, tmp_path / "output", artifact_path, baseline_path, detector="source",
        artifact_root=artifact_root, source_signature="synthetic-source",
    )
    return table, directory, artifact_root, artifact_path, baseline_path


def load_manifest(directory):
    return json.loads((directory / "manifest.json").read_text())


def write_fresh_verifications(directory):
    manifest = load_manifest(directory)
    for variant in manifest["variants"].values():
        payload = json.loads((directory / variant["request_file"]).read_text())
        cells = []
        for cell in payload["cells"]:
            replace_bad = cell["original"] == "bad"
            selected = "new" if replace_bad else cell["original"]
            cells.append({
                "row": cell["row"], "column": cell["column"],
                "original_valid_score": .01 if replace_bad else .99,
                "selected_value": selected, "confidence": .99,
                "candidates": [{"value": c["value"], "score": .99 if c["value"] == selected else .05}
                               for c in cell["candidates"]],
            })
        write_json(directory / variant["verification_file"], {
            "request_hash": payload["request_hash"], "dirty_sha256": payload["dirty_sha256"],
            "model": payload["model"], "reasoning_effort": payload["reasoning_effort"], "cells": cells,
        })


def all_keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(all_keys(item) for item in value))
    return set()


def test_preparation_redacts_counterfactual_input_and_keeps_source_detector(tmp_path):
    table, directory, _, _, baseline_path = prepare_fixture(tmp_path)
    manifest = load_manifest(directory)
    assert directory.name == manifest["protocol_hash"]
    assert len({v["request_hash"] for v in manifest["variants"].values()}) == 4
    source_probability_hash = digest(Prediction.load(baseline_path).probabilities.tolist())
    for name, variant in manifest["variants"].items():
        payload = json.loads((directory / variant["request_file"]).read_text())
        assert payload["detector_provenance"]["probabilities_hash"] == source_probability_hash
        assert payload["detector_provenance"]["prediction_file_sha256"] == sha256(baseline_path)
        assert payload["base_probabilities"][1][1] == .62
        assert payload["request_hash"] == digest({k: v for k, v in payload.items() if k != "request_hash"})
        if name == "without_counterfactual":
            assert not (all_keys(payload) & COUNTERFACTUAL_FIELDS)
            assert payload["config"]["counterfactual"] is False
        else:
            assert "delta_features" in payload["cells"][0]["candidates"][0]
        if name == "without_multi_column":
            assert payload["dependency_count"] == 0
            assert {rule["kind"] for rule in payload["rules"]} == {"single"}
        if name == "without_iterative_verification":
            assert payload["config"]["rounds"] == 1
            assert payload["config"]["iterative_verification"] is True
    assert not (directory / "predictions").exists()


def test_preparation_is_idempotent_and_does_not_touch_legacy_freeze(tmp_path):
    table, directory, artifact_root, artifact_path, baseline_path = prepare_fixture(tmp_path)
    legacy = tmp_path / "output" / "frozen" / f"{table.name}.json"
    write_json(legacy, {"legacy": "must remain unchanged"})
    before = {path: path.read_bytes() for path in directory.rglob("*.json")}
    old_freeze = legacy.read_bytes()
    repeated = prepare_matched_ablations(
        table, tmp_path / "output", artifact_path, baseline_path, detector="source",
        artifact_root=artifact_root, source_signature="synthetic-source",
    )
    assert repeated == directory
    assert all(path.read_bytes() == body for path, body in before.items())
    assert legacy.read_bytes() == old_freeze


def test_missing_any_fresh_verifier_stops_before_inference(tmp_path, monkeypatch):
    table, directory, *_ = prepare_fixture(tmp_path)
    called = []
    monkeypatch.setattr("zeroclean.ablations.infer_from_candidates", lambda *args: called.append(args))
    with pytest.raises(ValueError, match="Fresh.*verification required"):
        run_matched_ablations(table, directory, source_signature="synthetic-source")
    assert called == []
    assert not (directory / "predictions").exists()


@pytest.mark.parametrize("problem", ["full_request_hash", "partial_coverage", "partial_candidate_coverage"])
def test_full_or_partial_verifier_is_not_a_variant_fallback(tmp_path, problem):
    table, directory, *_ = prepare_fixture(tmp_path)
    write_fresh_verifications(directory)
    manifest = load_manifest(directory)
    variant = manifest["variants"]["without_counterfactual"]
    path = directory / variant["verification_file"]
    response = json.loads(path.read_text())
    if problem == "full_request_hash":
        response["request_hash"] = manifest["variants"]["zeroclean_loop"]["request_hash"]
    elif problem == "partial_coverage":
        response["cells"].pop()
    else:
        next(cell for cell in response["cells"] if cell["candidates"])["candidates"].pop()
    write_json(path, response)
    with pytest.raises(ValueError, match="exact candidate request|every requested cell|every requested candidate"):
        run_matched_ablations(table, directory, source_signature="synthetic-source")
    assert not (directory / "predictions").exists()


def test_matched_run_freezes_exact_inputs_and_retains_first_pass_verifier(tmp_path):
    table, directory, *_ = prepare_fixture(tmp_path)
    write_fresh_verifications(directory)
    frozen = run_matched_ablations(table, directory, source_signature="synthetic-source")
    assert len(frozen["prediction_files"]) == 4
    assert frozen["ablation_protocol"]["verification_mode"] == "independent_variant_request"
    for name in frozen["prediction_files"]:
        path = directory / "predictions" / table.name / name
        assert sha256(path) == frozen["prediction_files"][name]
        prediction = Prediction.load(path)
        assert prediction.repairs[0, 0] == "new"
        assert prediction.metadata["verifier_coverage"] == 1
        if name == "without_counterfactual.json":
            assert prediction.metadata["trace"][0]["delta_features"] == [0, 0, 0, 0]
        if name == "without_iterative_verification.json":
            assert prediction.metadata["rounds_executed"] == 1
            assert prediction.metadata["verified_cells"] > 0
    for filename, expected in frozen["inference_inputs"].items():
        assert (sha256(filename) if Path(filename).exists() else None) == expected
    frozen_path = directory / "frozen" / f"{table.name}.json"
    before = frozen_path.read_bytes()
    with pytest.raises(ValueError, match="already frozen"):
        run_matched_ablations(table, directory, source_signature="synthetic-source")
    assert frozen_path.read_bytes() == before


@pytest.mark.parametrize("changed", ["baseline", "new_proposals", "source"])
def test_changed_preparation_inputs_require_new_requests(tmp_path, changed):
    table, directory, artifact_root, _, baseline_path = prepare_fixture(tmp_path)
    signature = "synthetic-source"
    if changed == "baseline":
        baseline_path.write_text(baseline_path.read_text() + "\n")
    elif changed == "new_proposals":
        write_json(artifact_root / table.name / "proposals.json", {"dirty_sha256": table.fingerprint, "cells": []})
    else:
        signature = "changed-source"
    with pytest.raises(ValueError, match="input changed|implementation changed"):
        run_matched_ablations(table, directory, source_signature=signature)


def test_cli_prepare_wrapper_selects_source_prediction(tmp_path, monkeypatch):
    table, _, _, _ = fixture_inputs(tmp_path)
    captured = {}
    def capture(*args, **kwargs):
        captured["args"], captured["kwargs"] = args, kwargs
        return tmp_path / "new-run"
    monkeypatch.setattr("zeroclean.ablations.prepare_matched_ablations", capture)
    evaluate_e2e.prepare_ablations(table, tmp_path / "output", LoopConfig(), detector="source")
    assert captured["args"][3].name == "zeroed.json"
    assert captured["kwargs"]["detector"] == "source"


def test_legacy_rule_ablation_preserves_selected_source_probabilities(tmp_path, monkeypatch):
    table, _, artifact_path, baseline_path = fixture_inputs(tmp_path)
    artifact = json.loads(artifact_path.read_text())
    baseline = Prediction.load(baseline_path)
    payload = {
        "dirty_sha256": table.fingerprint, "artifact_hash": digest(artifact),
        "detector_provenance": {"kind": "source", "prediction_hash": digest({
            "probabilities": baseline.probabilities.tolist(), "metadata": baseline.metadata,
        })},
    }
    write_json(tmp_path / "artifacts" / "candidates" / f"{table.name}.json", payload)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate_e2e, "load_artifact", lambda *args: artifact)
    monkeypatch.setattr(evaluate_e2e, "source_signature", lambda: "synthetic-source")
    captured = {}
    def candidates(_table, _artifact, detector, _config):
        captured["probabilities"] = detector.probabilities.copy()
        return {"cells": [], "rules": [], "base_probabilities": detector.probabilities.tolist()}
    def infer(_table, request, verification, config):
        if not config.multi_column:
            captured["provenance"] = request["detector_provenance"]
        return Prediction(np.full(table.shape, .05))
    monkeypatch.setattr(evaluate_e2e, "build_candidates", candidates)
    monkeypatch.setattr(evaluate_e2e, "infer_from_candidates", infer)
    evaluate_e2e.run(table, tmp_path / "output", LoopConfig())
    assert np.array_equal(captured["probabilities"], baseline.probabilities)
    assert captured["provenance"]["kind"] == "source"
    result = Prediction.load(baseline_path.with_name("without_multi_column.json"))
    assert result.metadata["ablation_protocol"]["matched"] is False


def test_summary_keeps_qualified_reconstruction_separate_from_exact_gate():
    synthetic_runs = {"synthetic": {"zeroed_zeroec_reconstruction": {}}}
    calls = []
    def assess(_runs, baseline_ed="zeroed", baseline_repair="zeroed_zeroec"):
        calls.append((baseline_ed, baseline_repair))
        complete = baseline_repair == "zeroed_zeroec_reconstruction"
        return {"met": complete, "status": "verified" if complete else "unverified_missing_baselines_or_runs"}
    summary = evaluate_e2e.comparison_summary(synthetic_runs, assess)
    assert summary["status"] == "measured_experiment_qualified_source_reconstruction"
    assert summary["official_target_assessment"]["met"] is False
    qualified = summary["qualified_source_reconstruction_comparison"]
    assert qualified["met"] is True and qualified["official_exact_baseline"] is False
    assert qualified["fixed_example_count"] == 0
    assert "historical identity unproven" in qualified["fidelity_caveat"]
    assert ("zeroed", "zeroed_zeroec") in calls
    assert ("zeroed", "zeroed_zeroec_reconstruction") in calls


def test_model_info_hashes_large_source_mask_instead_of_copying_it():
    decisions = [[True, False], [False, True]]
    metadata = {"upstream_binary_decisions": decisions, "trace": [{"private": "large"}],
                "input_fingerprint": "synthetic"}
    result = evaluate_e2e.model_metadata(metadata)
    assert "upstream_binary_decisions" not in result and "trace" not in result
    assert result["upstream_detection_mask"] == {
        "sha256": digest(decisions), "shape": [2, 2], "positive_cells": 2,
    }
    assert result["input_fingerprint"] == "synthetic"
    assert metadata["upstream_binary_decisions"] is decisions
