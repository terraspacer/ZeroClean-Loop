"""Explicit immutable candidate preparation uses synthetic dirty inputs only."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import prepare_experiment as preparer
from zeroclean.closed_loop import start_closed_loop
from zeroclean.data import DirtyTable, sha256
from zeroclean.types import Prediction, digest, write_json
from zeroclean.verification import queue_verification


def fixture(tmp_path, *, canonical=False):
    table = DirtyTable("synthetic", ("value", "group"), (("bad", "G"), ("good", "G"), ("good", "G")), "synthetic-input")
    artifact = {"dataset": table.name, "dirty_sha256": table.fingerprint, "model": "gpt-6.1-sol", "reasoning_effort": "xhigh",
                "rules": [{"id": "format", "kind": "single", "target": "value", "description": "Synthetic token rule",
                           "source": "def check(row):\n    return row['value'] == 'bad'"}],
                "dependencies": [{"determinants": ["group"], "dependent": "value", "min_support": 2, "confidence": 1.}]}
    probabilities = np.array([[.7, .1], [.2, .3], [.2, .1]])
    detection = Prediction(probabilities, metadata={"input_fingerprint": table.fingerprint, "reproduced_upstream": True,
                           "upstream_binary_decisions": (probabilities > .5).tolist()})
    proposals = {"dataset": table.name, "dirty_sha256": table.fingerprint, "model": "gpt-6.1-sol", "reasoning_effort": "xhigh",
                 "cells": [{"row": 0, "column": "value", "original_valid_score": .01,
                            "candidates": [{"value": "good", "score": .96, "explanation": "Synthetic dirty peers."}]}]}
    artifact_path, detection_path = tmp_path / "future_rules.json", tmp_path / "zeroed.json"
    proposal_path = (tmp_path / "llm" / table.name / "future_proposals.json") if canonical else tmp_path / "one_off_proposals.json"
    write_json(artifact_path, artifact)
    detection.save(detection_path)
    write_json(proposal_path, proposals)
    return table, artifact_path, detection_path, proposal_path, tmp_path / "experiments" / "future.json"


@pytest.mark.parametrize("canonical", [False, True])
def test_explicit_inputs_and_source_code_are_bound_without_source_mutation(tmp_path, canonical, monkeypatch):
    table, artifact, detection, proposals, output = fixture(tmp_path, canonical=canonical)
    before = {path: path.read_bytes() for path in (artifact, detection, proposals)}
    roots = []
    original = preparer.build_candidates
    def capture(_table, _artifact, _detection, config, artifact_root):
        staged = Path(artifact_root) / table.name / config.proposal_filename
        assert staged.read_bytes() == before[proposals]
        roots.append(Path(artifact_root))
        return original(_table, _artifact, _detection, config, artifact_root)
    monkeypatch.setattr(preparer, "build_candidates", capture)
    payload = preparer.prepare_experiment(table, artifact, detection, proposals, output)
    assert payload["artifact_hash"] == digest(json.loads(artifact.read_text()))
    assert payload["request_hash"] == digest({key: value for key, value in payload.items() if key != "request_hash"})
    assert payload["config"]["proposal_filename"] == proposals.name
    assert payload["detector_provenance"]["kind"] == "source"
    assert payload["detector_provenance"]["prediction_file_sha256"] == sha256(detection)
    identity = payload["experiment_preparation"]
    assert identity["input_file_hashes"] == {str(path.resolve()): sha256(path) for path in before}
    implementation = identity["source_implementation_hashes"]
    assert {Path(path).name for path in implementation} == {"prepare_experiment.py", "loop.py", "statistics.py", "sandbox.py", "types.py", "data.py", "upstream_zeroec.py"}
    assert all(sha256(path) == expected for path, expected in implementation.items())
    assert identity["source_implementation_signature"] == digest(implementation)
    assert all(path.read_bytes() == body for path, body in before.items())
    assert json.loads(output.read_text()) == payload
    if canonical: assert roots == [proposals.parent.parent]
    else: assert roots and not roots[0].exists()


def test_prepared_payload_works_with_verifier_queue_and_exact_artifact_closed_loop(tmp_path):
    table, artifact, detection, proposals, output = fixture(tmp_path)
    payload = preparer.prepare_experiment(table, artifact, detection, proposals, output)
    def request(stage, prompt, metadata):
        evidence = json.loads(prompt.rsplit("\n", 1)[1])
        return json.dumps({"cells": [{
            "row": cell["row"], "column": cell["column"], "original_valid_score": .01 if cell["original"] == "bad" else .99,
            "selected_value": "good" if cell["original"] == "bad" else cell["original"], "confidence": .99,
            "explanation": "Synthetic evidence judgment.",
            "candidates": [{"value": candidate["value"], "score": .99 if candidate["value"] == "good" else .05}
                           for candidate in cell["candidates"]],
        } for cell in evidence["cells"]]})
    response_path = tmp_path / "verification.json"
    response = queue_verification(table, payload, request, response_path)
    assert response["request_hash"] == payload["request_hash"]
    run = start_closed_loop(table, output, response_path, artifact, tmp_path / "closed_loop",
                            max_rounds=1, source_signature="synthetic-implementation")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["identity"]["artifact_hash"] == payload["artifact_hash"]
    assert manifest["identity"]["seed_request_hash"] == payload["request_hash"]


@pytest.mark.parametrize("problem", ["artifact_model", "artifact_dirty", "proposal_effort", "proposal_dirty", "detector_dirty", "adapted_detector", "source_mask"])
def test_invalid_explicit_input_contracts_stop_before_publication(tmp_path, problem):
    table, artifact, detection, proposals, output = fixture(tmp_path)
    if problem.startswith("artifact"):
        path = artifact; body = json.loads(path.read_text())
        body["model" if problem.endswith("model") else "dirty_sha256"] = "mismatch"
    elif problem.startswith("proposal"):
        path = proposals; body = json.loads(path.read_text())
        body["reasoning_effort" if problem.endswith("effort") else "dirty_sha256"] = "mismatch"
    else:
        path = detection; body = json.loads(path.read_text())
        if problem == "detector_dirty": body["metadata"]["input_fingerprint"] = "mismatch"
        elif problem == "adapted_detector": body["metadata"]["reproduced_upstream"] = False
        else: body["metadata"]["upstream_binary_decisions"][0][0] = False
    write_json(path, body)
    with pytest.raises(ValueError):
        preparer.prepare_experiment(table, artifact, detection, proposals, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["input", "implementation"])
def test_mutation_during_build_is_rejected_without_publishing(tmp_path, monkeypatch, change):
    table, artifact, detection, proposals, output = fixture(tmp_path)
    synthetic_code = tmp_path / "synthetic_implementation.py"
    synthetic_code.write_text("original code\n")
    monkeypatch.setattr(preparer, "implementation_files", lambda: [synthetic_code])
    original = preparer.build_candidates
    def mutate(*args, **kwargs):
        payload = original(*args, **kwargs)
        path = artifact if change == "input" else synthetic_code
        path.write_text(path.read_text() + "\n")
        return payload
    monkeypatch.setattr(preparer, "build_candidates", mutate)
    with pytest.raises(ValueError, match="input or implementation changed"):
        preparer.prepare_experiment(table, artifact, detection, proposals, output)
    assert not output.exists()


def test_idempotent_identity_never_overwrites_an_existing_payload(tmp_path, monkeypatch):
    table, artifact, detection, proposals, output = fixture(tmp_path)
    payload = preparer.prepare_experiment(table, artifact, detection, proposals, output)
    before = output.read_bytes()
    def forbidden(*args, **kwargs): raise AssertionError("Identical immutable output needs no regeneration")
    monkeypatch.setattr(preparer, "build_candidates", forbidden)
    assert preparer.prepare_experiment(table, artifact, detection, proposals, output) == payload
    assert output.read_bytes() == before
    proposals.write_text(proposals.read_text() + "\n")
    with pytest.raises(ValueError, match="overwrite a differing"):
        preparer.prepare_experiment(table, artifact, detection, proposals, output)
    assert output.read_bytes() == before


def test_concurrent_different_publication_is_preserved(tmp_path, monkeypatch):
    table, artifact, detection, proposals, output = fixture(tmp_path)
    def racing_link(temporary, destination):
        write_json(destination, {"another": "experiment"})
        raise FileExistsError
    monkeypatch.setattr(preparer.os, "link", racing_link)
    with pytest.raises(ValueError, match="overwrite a differing"):
        preparer.prepare_experiment(table, artifact, detection, proposals, output)
    assert json.loads(output.read_text()) == {"another": "experiment"}
    assert not list(output.parent.glob("*.tmp"))


def test_concurrent_same_identity_with_different_candidate_state_is_rejected(tmp_path, monkeypatch):
    table, artifact, detection, proposals, output = fixture(tmp_path)
    def racing_link(temporary, destination):
        other = json.loads(Path(temporary).read_text())
        other["cells"][0]["candidates"][0]["score"] = .12
        other["request_hash"] = digest({key: value for key, value in other.items() if key != "request_hash"})
        write_json(destination, other)
        raise FileExistsError
    monkeypatch.setattr(preparer.os, "link", racing_link)
    with pytest.raises(ValueError, match="Concurrent.*differs"):
        preparer.prepare_experiment(table, artifact, detection, proposals, output)
    assert json.loads(output.read_text())["cells"][0]["candidates"][0]["score"] == .12


def test_active_candidates_and_pass2_are_reserved_output_locations():
    root = preparer.PROJECT_ROOT
    assert preparer._protected_output(root / "artifacts" / "candidates" / "synthetic.json")
    assert preparer._protected_output(root / "artifacts" / "llm" / "synthetic" / "pass2.json")
    assert not preparer._protected_output(root / "experiments" / "synthetic.json")
