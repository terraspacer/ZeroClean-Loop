"""Component effects are compared with the frozen first verified pass only."""
import json
from pathlib import Path

import pytest

from scripts import prepare_component_ablations as components
from scripts import prepare_experiment
from test_prepare_experiment import fixture as explicit_inputs
from zeroclean.ablations import COUNTERFACTUAL_FIELDS
from zeroclean.closed_loop import advance_closed_loop, start_closed_loop
from zeroclean.data import sha256
from zeroclean.types import Prediction, write_json
from zeroclean.verification import queue_verification


def judgments(cells):
    return [{"row": cell["row"], "column": cell["column"],
             "selected_value": "good" if cell["original"] == "bad" else cell["original"],
             "original_valid_score": .01 if cell["original"] == "bad" else .99, "confidence": .99,
             "explanation": "Synthetic independent first-pass judgment.",
             "candidates": [{"value": candidate["value"], "score": .99 if candidate["value"] == "good" else .05}
                            for candidate in cell["candidates"]]}
            for cell in cells]


def callback(stage, prompt, metadata):
    evidence = json.loads(prompt.rsplit("\n", 1)[1])
    return json.dumps({"cells": judgments(evidence["cells"])})


def fixture(tmp_path, *, complete=True):
    table, artifact, detection, proposals, output = explicit_inputs(tmp_path)
    body = json.loads(artifact.read_text())
    body["rules"].append({"id": "multi", "kind": "multi", "target": "value", "description": "Synthetic contextual token",
                          "source": "def check(row):\n    return row['value'] == 'bad' and row['group'] == 'G'"})
    write_json(artifact, body)
    body = json.loads(proposals.read_text())
    body["retrieval_method"] = "zeroec_source_cts"
    body["cells"][0]["candidates"][0].update(origin="zeroec_cts_llm", nested={"delta_features": [1, 2, 3, 4], "introduced_violation": False})
    write_json(proposals, body)
    payload = prepare_experiment.prepare_experiment(table, artifact, detection, proposals, output)
    response = tmp_path / "initial_verification.json"
    queue_verification(table, payload, callback, response)
    run = start_closed_loop(table, output, response, artifact, tmp_path / "full_runs", max_rounds=3,
                            source_signature="synthetic-full-source")
    if complete: advance_closed_loop(table, run, source_signature="synthetic-full-source", request=callback)
    return table, run, detection, proposals, tmp_path / "prepared", tmp_path / "component_results"


def all_keys(value):
    if isinstance(value, dict): return set(value) | set().union(*(all_keys(item) for item in value.values()))
    if isinstance(value, list): return set().union(*(all_keys(item) for item in value))
    return set()


def fresh_responses(table, directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for variant in manifest["identity"]["variants"].values():
        payload = json.loads((directory / variant["request_file"]).read_text())
        queue_verification(table, payload, callback, directory / variant["verification_file"])
    return manifest


def test_prepare_redacts_counterfactuals_and_rebuilds_only_multi_column_component(tmp_path):
    table, run, detection, proposals, root, _ = fixture(tmp_path)
    before = {path: path.read_bytes() for path in run.rglob("*.json")}
    directory = components.prepare_components(table, run, detection, proposals, root)
    manifest = json.loads((directory / "manifest.json").read_text())
    identity = manifest["identity"]
    assert identity["full_source_signature"] == "synthetic-full-source"
    assert identity["input_file_hashes"][str(detection.resolve())] == sha256(detection)
    assert identity["input_file_hashes"][str(proposals.resolve())] == sha256(proposals)
    for name, variant in identity["variants"].items():
        payload = json.loads((directory / variant["request_file"]).read_text())
        assert payload["config"]["rounds"] == 1 and payload["config"]["iterative_verification"] is True
        assert payload["component_ablation"]["scope"] == "components_of_first_verified_pass"
        assert payload["component_ablation"]["initial_verifier_reused"] is False
        if "counterfactual" in name:
            assert not (all_keys(payload) & COUNTERFACTUAL_FIELDS)
            assert payload["config"]["counterfactual"] is False
            assert {rule["kind"] for rule in payload["rules"]} == {"single", "multi"}
        else:
            assert payload["dependency_count"] == 0
            assert all(candidate["delta_features"][1] == 0 for cell in payload["cells"] for candidate in cell["candidates"])
            assert {rule["kind"] for rule in payload["rules"]} == {"single"}
            assert payload["proposal_provenance"]["method"] == "zeroec_source_cts"
            assert payload["detector_provenance"]["prediction_file_sha256"] == sha256(detection)
            assert any(candidate["origin"] == "zeroec_cts_llm" for cell in payload["cells"] for candidate in cell["candidates"])
    assert all(path.read_bytes() == body for path, body in before.items())
    assert components.prepare_components(table, run, detection, proposals, root) == directory


def test_run_uses_independent_verifiers_and_exact_full_first_round_copy(tmp_path, monkeypatch):
    table, run, detection, proposals, root, output = fixture(tmp_path)
    directory = components.prepare_components(table, run, detection, proposals, root)
    manifest = fresh_responses(table, directory)
    calls = []
    original = components.infer_from_candidates
    def infer(_table, payload, verification, config):
        calls.append((payload["request_hash"], verification["request_hash"], config))
        return original(_table, payload, verification, config)
    monkeypatch.setattr(components, "infer_from_candidates", infer)
    frozen = components.run_components(table, directory, output)
    assert len(calls) == 2
    assert all(request_hash == response_hash and config.rounds == 1 and config.iterative_verification for request_hash, response_hash, config in calls)
    assert len({request_hash for request_hash, _, _ in calls}) == 2
    initial_path = run / "rounds" / "0001" / "prediction.json"
    copied = output / "predictions" / table.name / components.REFERENCE
    assert copied.read_bytes() == initial_path.read_bytes() != (run / "prediction.json").read_bytes()
    assert sha256(copied) == frozen["prediction_files"][components.REFERENCE]
    assert set(frozen["prediction_files"]) == {components.REFERENCE, *(f"{name}.json" for name in components.VARIANTS)}
    protocol = frozen["component_ablation_protocol"]
    assert protocol["inference_passes"] == 1 and protocol["final_tuple_guard_retained"] is True
    assert protocol["iterative_no_component_experiment_claimed"] is False
    for name in components.VARIANTS:
        prediction = Prediction.load(output / "predictions" / table.name / f"{name}.json")
        assert prediction.metadata["verifier_coverage"] == 1 and prediction.metadata["rounds_executed"] == 1
        if "counterfactual" in name:
            assert all(trace["delta_features"] == [0, 0, 0, 0] for trace in prediction.metadata["trace"])
    assert components.run_components(table, directory, output) == frozen
    assert len(calls) == 2
    assert manifest["identity"]["initial_request_hash"] not in {request_hash for request_hash, _, _ in calls}


@pytest.mark.parametrize("problem", ["missing", "partial_cells", "partial_candidates", "full_reply"])
def test_missing_partial_or_reused_full_reply_stops_before_any_variant_inference(tmp_path, monkeypatch, problem):
    table, run, detection, proposals, root, output = fixture(tmp_path)
    directory = components.prepare_components(table, run, detection, proposals, root)
    manifest = fresh_responses(table, directory)
    variant = manifest["identity"]["variants"]["without_multi_column_first_pass"]
    path = directory / variant["verification_file"]
    body = json.loads(path.read_text())
    if problem == "missing": path.unlink()
    elif problem == "partial_cells": body["cells"].pop(); write_json(path, body)
    elif problem == "partial_candidates":
        next(cell for cell in body["cells"] if cell["candidates"])["candidates"].pop(); write_json(path, body)
    else: path.write_bytes((run / "rounds" / "0001" / "verification.json").read_bytes())
    calls = []
    monkeypatch.setattr(components, "infer_from_candidates", lambda *args: calls.append(args))
    with pytest.raises(ValueError): components.run_components(table, directory, output)
    assert calls == [] and not output.exists()


@pytest.mark.parametrize("which", ["detection", "proposals", "pending_full_run"])
def test_explicit_inputs_must_match_frozen_initial_provenance(tmp_path, which):
    table, run, detection, proposals, root, _ = fixture(tmp_path, complete=which != "pending_full_run")
    if which == "detection":
        body = json.loads(detection.read_text()); body["probabilities"][0][0] = .8; write_json(detection, body)
    elif which == "proposals":
        body = json.loads(proposals.read_text()); body["cells"][0]["candidates"][0]["score"] = .8; write_json(proposals, body)
    with pytest.raises(ValueError): components.prepare_components(table, run, detection, proposals, root)
    assert not root.exists()


@pytest.mark.parametrize("moment", ["prepare", "run"])
def test_input_mutation_cannot_publish_component_artifacts(tmp_path, monkeypatch, moment):
    table, run, detection, proposals, root, output = fixture(tmp_path)
    if moment == "prepare":
        original = components.preparer.prepare_experiment
        def mutate(*args, **kwargs):
            payload = original(*args, **kwargs)
            proposals.write_text(proposals.read_text() + "\n")
            return payload
        monkeypatch.setattr(components.preparer, "prepare_experiment", mutate)
        with pytest.raises(ValueError, match="changed"): components.prepare_components(table, run, detection, proposals, root)
        assert not root.exists()
    else:
        directory = components.prepare_components(table, run, detection, proposals, root)
        manifest = fresh_responses(table, directory)
        reply = directory / next(iter(manifest["identity"]["variants"].values()))["verification_file"]
        original = components.infer_from_candidates
        def mutate(*args, **kwargs):
            prediction = original(*args, **kwargs)
            reply.write_text(reply.read_text() + "\n")
            return prediction
        monkeypatch.setattr(components, "infer_from_candidates", mutate)
        with pytest.raises(ValueError, match="changed"): components.run_components(table, directory, output)
        assert not output.exists()


def test_legacy_combined_semantic_source_identity_is_checked_without_guessing(tmp_path):
    table, run, detection, proposals, _, _ = fixture(tmp_path)
    initial = json.loads((run / "rounds" / "0001" / "request.json").read_text())
    initial["detector_provenance"] = {"kind": "source", "prediction_hash": initial["detector_provenance"]["prediction_hash"]}
    initial.pop("experiment_preparation")
    matched = components._match_explicit_inputs(table, initial, detection, proposals)
    assert matched.metadata["reproduced_upstream"] is True
    initial["detector_provenance"] = {"kind": "source"}
    with pytest.raises(ValueError, match="sufficient source detector provenance"):
        components._match_explicit_inputs(table, initial, detection, proposals)
