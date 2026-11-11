"""Synthetic completed detector freezes; no experiment outputs or real rows read."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import prepare_detector_variant as preparer
from zeroclean.closed_loop import start_closed_loop
from zeroclean.data import DATASETS, DirtyTable, sha256
from zeroclean.types import Prediction, digest, write_json
from zeroclean.verification import queue_verification


def fixture(tmp_path):
    root = tmp_path / "detector_experiment"
    mode = root / "hospital/variant"
    control = root / "hospital/control"
    mode.mkdir(parents=True); control.mkdir(parents=True)
    dirty = tmp_path / "synthetic_dirty.csv"
    dirty.write_text("value,group\nbad,G\ngood,H\nother,G\nvalid,H\n")
    rows = (("bad", "G"), ("good", "H"), ("other", "G"), ("valid", "H"))
    table = DirtyTable("hospital", ("value", "group"), rows, sha256(dirty))
    code = preparer.PROJECT_ROOT
    protocol = root / "protocol.json"
    addendum = root / "execution_addendum.json"
    write_json(protocol, {"protocol_id": preparer.EXPERIMENT})
    write_json(addendum, {"addendum_id": "frozen-prefilter-source-stage-v1", "original_protocol_unchanged": True})
    implementation, source_runner, freezer = (code / name for name in (
        "experimental/zero_predicate_variant.py", "zeroclean/upstream_zeroed.py", "experimental/freeze_zero_predicate_variant.py"))
    probabilities = np.array([[.249, .25], [.5, .85], [.851, .1], [.03, .03]])
    metadata = {"input_fingerprint": table.fingerprint, "name": "zeroed_empty_predicate_variant",
                "baseline": "experimental_unpromoted", "mode": "variant", "fidelity": "minimal_empty_predicate_semantic_variant",
                "reproduced_upstream": False, "empty_predicate_guard_enabled": True, "uses_ground_truth": False,
                "strict_zero_shot": True, "seed": 42, "python_hash_seed": "0", "new_requests_queued": 0,
                "protocol_sha256": sha256(protocol), "experimental_source_sha256": sha256(implementation),
                "execution_addendum_sha256": sha256(addendum), "decision_threshold": .5, "decision_threshold_operator": ">",
                "extraction": {"source_runner_sha256": sha256(source_runner), "live_module_modified": False},
                "upstream_binary_decisions": (probabilities > .5).tolist()}
    prediction_path = mode / "prediction.json"
    Prediction(probabilities.copy(), metadata=metadata).save(prediction_path)
    write_json(mode / "metadata.json", metadata)
    np.savez_compressed(mode / "frozen_scores.npz", scores=probabilities,
                        binary_decisions=probabilities > .5, columns=np.array(table.columns))
    control_path = control / "prediction.json"
    Prediction(np.full(table.shape, .9), metadata={"input_fingerprint": table.fingerprint,
               "mode": "control", "name": "zeroed_source_semantics_control", "reproduced_upstream": True}).save(control_path)
    comparison_path = root / "hospital/comparison.json"
    write_json(comparison_path, {"dataset": table.name, "input_sha256": table.fingerprint, "shape": list(table.shape),
        "status": "experimental_unpromoted", "threshold": .5, "threshold_operator": ">",
        "comparison_target": preparer.COMPARISON_TARGET, "no_benchmark_accuracy_claim": True,
        "full_prefilter_state_identical": True, "unaffected_columns_exact_reproduction": True, "recorded_replies_identical": True})
    training = mode / "synthetic_training.npz"
    np.savez_compressed(training, features=np.array([[1., 2.]]), pseudo_labels=np.array([0]))
    source_hashes = {str(p): sha256(p) for p in (dirty, implementation, source_runner, prediction_path,
                    mode / "metadata.json", mode / "frozen_scores.npz", control_path, comparison_path, training)}
    entry = {"input_path": str(dirty), "input_sha256": table.fingerprint, "shape": list(table.shape),
        "variant_prediction_path": str(prediction_path), "variant_prediction_sha256": sha256(prediction_path),
        "control_prediction_path": str(control_path), "control_prediction_sha256": sha256(control_path),
        "comparison_path": str(comparison_path), "comparison_sha256": sha256(comparison_path),
        "source_runner_sha256": sha256(source_runner), "source_sha256": source_hashes,
        "validation": {key: True for key in preparer.VALIDATION_CHECKS}}
    manifest = {"schema_version": 1, "status": "complete", "experiment_id": preparer.EXPERIMENT,
        "experimental_unpromoted": True, "reproduced_upstream": False, "uses_ground_truth": False, "strict_zero_shot": True,
        "model": preparer.MODEL, "reasoning_effort": preparer.EFFORT, "seed": 42, "python_hash_seed": "0",
        "native_threads": {key: "2" for key in preparer.THREADS}, "threshold": .5, "threshold_operator": ">",
        "comparison_target": preparer.COMPARISON_TARGET, "datasets": {name: deepcopy(entry) for name in DATASETS}}
    for prefix, path in (("implementation", implementation), ("source_runner", source_runner),
                         ("freezer", freezer), ("protocol", protocol), ("execution_addendum", addendum)):
        manifest[prefix + "_path"], manifest[prefix + "_sha256"] = str(path), sha256(path)
    manifest["manifest_hash"] = digest(manifest)
    manifest_path = root / "completion_manifest.json"
    write_json(manifest_path, manifest)
    artifact_path = tmp_path / "criteria.json"
    write_json(artifact_path, {"dataset": table.name, "dirty_sha256": table.fingerprint, "model": preparer.MODEL,
        "reasoning_effort": preparer.EFFORT, "rules": [{"id": "token", "kind": "single", "target": "value",
        "description": "Synthetic dirty token criterion", "source": "def check(row):\n    return row['value'] == 'bad'"}],
        "valid_values": {"value": ["other"]}, "dependencies": []})
    source_path, seed_path = tmp_path / "original_source.json", tmp_path / "source_seed.json"
    original_probabilities = np.full(table.shape, .05); original_probabilities[0, 0] = .9
    Prediction(original_probabilities, metadata={"input_fingerprint": table.fingerprint, "reproduced_upstream": True,
               "upstream_binary_decisions": (original_probabilities > .5).tolist()}).save(source_path)
    write_json(seed_path, {"synthetic": "frozen original seed, never used for new candidate eligibility"})
    inputs = {str(p): sha256(p) for p in (source_path, seed_path)}
    source_mask_hash = digest((original_probabilities > .5).tolist())
    identity = {"experiment_policy": preparer.CTS_POLICY, "dirty_sha256": table.fingerprint,
        "seed_request_hash": digest("source-seed"), "source_input_hashes": inputs,
        "script_sha256": sha256(code / "scripts/queue_cts_v2.py"), "source_decision_mask_hash": source_mask_hash,
        "retrieval_only_mask_hash": source_mask_hash}
    proposals_path = tmp_path / "proposals_cts_v2.json"
    proposals = {"dataset": table.name, "dirty_sha256": table.fingerprint, "model": preparer.MODEL, "reasoning_effort": preparer.EFFORT,
        "experiment_policy": preparer.CTS_POLICY, "experiment_identity": identity, "script_sha256": identity["script_sha256"],
        "seed_request_hash": identity["seed_request_hash"], "source_detection_sha256": sha256(source_path),
        "source_input_hashes": inputs, "source_probabilities_hash": digest(original_probabilities.tolist()),
        "retrieval_method": "zeroec_source_cts", "request_hashes": [digest("original request")],
        "policy_disclosure": {"same_policy_all_datasets": True, "source_predictions_modified": False, "mask_query_target": True},
        "cells": [{"row": 0, "column": "value", "original_valid_score": .01,
                   "candidates": [{"value": "good", "score": .96, "origin": "llm_context",
                       "count_evidence": {"trust": "untrusted_dirty_observations", "query_observation_excluded": True}}]}]}
    proposals["artifact_hash"] = digest(proposals)
    write_json(proposals_path, proposals)
    return table, artifact_path, prediction_path, proposals_path, manifest_path, tmp_path / "prepared/variant.json"


def prepare(data):
    return preparer.prepare_detector_variant(*data)


def resign_manifest(path):
    manifest = json.loads(path.read_text())
    for entry in manifest["datasets"].values():
        for filename in entry["source_sha256"]:
            entry["source_sha256"][filename] = sha256(filename)
        for prefix in ("variant_prediction", "control_prediction", "comparison"):
            entry[prefix + "_sha256"] = sha256(entry[prefix + "_path"])
    manifest["manifest_hash"] = digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
    write_json(path, manifest)


def rewrite_prediction(data, change):
    body = json.loads(data[2].read_text()); change(body)
    write_json(data[2], body)
    write_json(data[2].parent / "metadata.json", body["metadata"])
    resign_manifest(data[4])


def test_new_classifier_eligibility_rules_deltas_and_external_cts_identity_are_preserved(tmp_path):
    data = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in data[1:5]}
    payload = prepare(data)
    cells = {(c["row"], c["column_index"]): c for c in payload["cells"]}
    assert {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)} <= set(cells)
    assert (2, 1) not in cells and (3, 0) not in cells and (3, 1) not in cells
    assert cells[2, 0]["base_probability"] == .12  # Valid-values lowering cannot erase actual positive eligibility.
    assert payload["detector_provenance"]["classifier_positive_cells"] == 2
    assert payload["detector_provenance"]["kind"] == "experimental_empty_predicate_variant"
    assert payload["detector_provenance"]["reproduced_upstream"] is False
    assert payload["experiment_preparation"]["reproduced_upstream"] is False
    raw = Prediction.load(data[2])
    assert payload["detector_provenance"]["classifier_mask_hash"] == digest(raw.metadata["upstream_binary_decisions"])
    assert payload["detector_provenance"]["probabilities_hash"] == digest(raw.probabilities.tolist())
    assert payload["external_cts_provenance"]["source_detection_sha256"] != sha256(data[2])
    assert payload["external_cts_provenance"]["retrieval_rerun"] is False
    assert payload["external_cts_provenance"]["method"] == "frozen_original_source_ctsv2_candidate_evidence"
    assert cells[0, 0]["candidates"][0]["count_evidence"]["query_observation_excluded"]
    assert all(len(c["delta_features"]) == 4 and np.isfinite(c["delta_features"]).all()
               for cell in cells.values() for c in cell["candidates"])
    identity = payload["experiment_preparation"]
    manifest = json.loads(data[4].read_text())
    assert identity["detector_manifest_hash"] == manifest["manifest_hash"]
    assert identity["input_file_hashes"][str(data[4])] == sha256(data[4])
    for filename, wanted in manifest["datasets"][data[0].name]["source_sha256"].items():
        assert identity["input_file_hashes"][filename] == wanted
    assert {str(p): p.read_bytes() for p in data[1:5]} == before
    assert payload["request_hash"] == digest({k: v for k, v in payload.items() if k != "request_hash"})


def test_payload_uses_standard_verification_and_exact_artifact_closed_loop(tmp_path):
    data = fixture(tmp_path)
    payload = prepare(data)
    def synthetic(stage, prompt, metadata):
        evidence = json.loads(prompt.split("\n", 1)[1])
        return json.dumps({"cells": [{"row": c["row"], "column": c["column"], "original_valid_score": .99,
            "selected_value": c["original"], "confidence": .99, "explanation": "Synthetic independent abstention",
            "candidates": [{"value": v["value"], "score": .1} for v in c["candidates"]]} for c in evidence["cells"]]})
    verification_path = tmp_path / "verification.json"
    result = queue_verification(data[0], payload, synthetic, verification_path)
    assert result["request_hash"] == payload["request_hash"]
    run = start_closed_loop(data[0], data[5], verification_path, data[1], tmp_path / "loop", max_rounds=1,
                            source_signature="synthetic-standard-inference")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["identity"]["seed_request_hash"] == payload["request_hash"]
    assert manifest["identity"]["artifact_hash"] == payload["artifact_hash"]


@pytest.mark.parametrize("problem", ["unfinished", "wrong_experiment", "wrong_model", "old_effort", "failed_validation", "missing_dataset",
                                    "manifest_tamper", "source_signature", "score_file", "metadata_sidecar"])
def test_detector_freeze_validation_rejects_incomplete_stale_or_tampered_inputs(tmp_path, problem):
    data = fixture(tmp_path)
    manifest = json.loads(data[4].read_text())
    if problem == "unfinished": manifest["status"] = "training"
    elif problem == "wrong_experiment": manifest["experiment_id"] = "some-other-detector"
    elif problem == "wrong_model": manifest["model"] = "other-model"
    elif problem == "old_effort": manifest["reasoning_effort"] = "low"
    elif problem == "failed_validation": manifest["datasets"]["hospital"]["validation"]["full_prefilter_state_identical"] = False
    elif problem == "missing_dataset": manifest["datasets"].pop("movies")
    elif problem == "source_signature": manifest["source_runner_sha256"] = digest("wrong source")
    elif problem == "score_file":
        path = data[2].parent / "frozen_scores.npz"
        with np.load(path) as frozen:
            values = {k: frozen[k] for k in frozen.files}
        values["binary_decisions"] = ~values["binary_decisions"]
        np.savez_compressed(path, **values)
        resign_manifest(data[4]); manifest = json.loads(data[4].read_text())
    elif problem == "metadata_sidecar":
        sidecar = data[2].parent / "metadata.json"
        value = json.loads(sidecar.read_text()); value["unexpected"] = True; write_json(sidecar, value)
        resign_manifest(data[4]); manifest = json.loads(data[4].read_text())
    if problem != "manifest_tamper":
        manifest["manifest_hash"] = digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
    else: manifest["status"] = "silently changed"
    write_json(data[4], manifest)
    with pytest.raises(ValueError): prepare(data)
    assert not data[5].exists()


@pytest.mark.parametrize("problem", ["source_control", "pretend_upstream", "dirty", "probability_mask", "new_repairs", "wrong_predicate_flag"])
def test_arbitrary_or_control_predictions_cannot_masquerade_as_completed_variant(tmp_path, problem):
    data = fixture(tmp_path)
    def change(body):
        metadata = body["metadata"]
        if problem == "source_control": metadata.update(mode="control", name="zeroed_source_semantics_control", fidelity="source_semantics_replay")
        elif problem == "pretend_upstream": metadata["reproduced_upstream"] = True
        elif problem == "dirty": metadata["input_fingerprint"] = "stale"
        elif problem == "probability_mask": metadata["upstream_binary_decisions"][0][0] = True
        elif problem == "new_repairs": body["repairs"] = [{"row": 0, "column": 0, "value": "replacement"}]
        else: metadata["empty_predicate_guard_enabled"] = False
    rewrite_prediction(data, change)
    with pytest.raises(ValueError): prepare(data)
    assert not data[5].exists()


@pytest.mark.parametrize("problem", ["criteria_model", "criteria_dirty", "proposal_model", "proposal_policy", "proposal_hash", "proposal_dirty", "external_identity"])
def test_explicit_rules_and_ctsv2_are_separately_validated(tmp_path, problem):
    data = fixture(tmp_path)
    path = data[1] if problem.startswith("criteria") else data[3]
    body = json.loads(path.read_text())
    if problem in {"criteria_model", "proposal_model"}: body["model"] = "stale-model"
    elif problem in {"criteria_dirty", "proposal_dirty"}: body["dirty_sha256"] = "other-dirty"
    elif problem == "proposal_policy": body["experiment_policy"] = "some-other-retrieval"
    elif problem == "external_identity": body["source_detection_sha256"] = sha256(data[2])
    else: body["artifact_hash"] = "bad hash"
    if path == data[3] and problem != "proposal_hash": body["artifact_hash"] = digest({k: v for k, v in body.items() if k != "artifact_hash"})
    write_json(path, body)
    with pytest.raises(ValueError): prepare(data)
    assert not data[5].exists()


def test_identical_preparation_reuses_immutable_output_and_differing_inputs_refuse_overwrite(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    payload = prepare(data); before = data[5].read_bytes()
    def forbidden(*args, **kwargs): pytest.fail("Unchanged immutable payload must not be regenerated")
    monkeypatch.setattr(preparer, "build_candidates", forbidden)
    assert prepare(data) == payload and data[5].read_bytes() == before
    data[1].write_text(data[1].read_text() + " ")
    with pytest.raises(ValueError, match="overwrite a differing"): prepare(data)
    assert data[5].read_bytes() == before


@pytest.mark.parametrize("change", ["detector_file", "training_input", "implementation"])
def test_mutation_during_candidate_build_prevents_publication(tmp_path, monkeypatch, change):
    data = fixture(tmp_path)
    code = tmp_path / "synthetic-preparer.py"; code.write_text("original source\n")
    monkeypatch.setattr(preparer, "implementation_files", lambda: [code])
    build = preparer.build_candidates
    def mutate(*args, **kwargs):
        result = build(*args, **kwargs)
        path = data[2] if change == "detector_file" else data[2].parent / "synthetic_training.npz" if change == "training_input" else code
        path.write_bytes(path.read_bytes() + b" ")
        return result
    monkeypatch.setattr(preparer, "build_candidates", mutate)
    with pytest.raises(ValueError, match="changed"): prepare(data)
    assert not data[5].exists()


def test_active_and_frozen_directories_are_protected_and_thresholds_cannot_be_tuned(tmp_path):
    data = fixture(tmp_path)
    for output in (preparer.PROJECT_ROOT / "artifacts/candidates/toy.json",
                   preparer.PROJECT_ROOT / "artifacts/llm/toy/pass2.json", data[4].parent / "new-candidates.json"):
        with pytest.raises(ValueError, match="separate"): preparer.prepare_detector_variant(*data[:-1], output)
    with pytest.raises(ValueError, match="fixed"):
        preparer.prepare_detector_variant(*data, config=preparer.LoopConfig(candidate_low=.2))
