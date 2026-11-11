"""External feedback and tentative context tested on synthetic dirty tables."""
import ast
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import zeroclean.closed_loop as closed_loop
from zeroclean.data import DirtyTable, sha256
from zeroclean.llm import ResponsePending
from zeroclean.loop import LoopConfig, build_candidates
from zeroclean.types import Prediction, digest, write_json


def judgments(cells):
    return [{
        "row": cell["row"], "column": cell["column"], "original_valid_score": .01,
        "selected_value": "B", "confidence": .99,
        "explanation": "Synthetic joint context supports both proposed changes.",
        "joint_repair_required": True,
        "candidates": [{"value": candidate["value"], "score": .99} for candidate in cell["candidates"]],
    } for cell in cells]


def seed(tmp_path, *, max_rounds=3):
    table = DirtyTable("synthetic", ("left", "right"), (("A", "A"), ("B", "B"), ("B", "B")), "dirty-only")
    artifact = {
        "dirty_sha256": table.fingerprint, "model": "gpt-6.1-sol", "reasoning_effort": "xhigh",
        "rules": [{"id": "pair", "target": "left", "kind": "multi", "description": "Synthetic equal pair",
                   "source": "def check(row):\n    return row['left'] != row['right']"}],
        "dependencies": [{"determinants": [other], "dependent": column, "min_support": 2, "confidence": 1.}
                         for column, other in (("left", "right"), ("right", "left"))],
        "value_repairs": {column: {"A": "B"} for column in table.columns},
    }
    payload = build_candidates(table, artifact, config=LoopConfig(), artifact_root=tmp_path / "absent-proposals")
    verification = {key: payload[key] for key in ("request_hash", "dirty_sha256", "model", "reasoning_effort", "dataset")}
    verification["cells"] = judgments(payload["cells"])
    payload_path, response_path, artifact_path = (tmp_path / name for name in ("request.json", "verification.json", "pass2.json"))
    for path, value in ((payload_path, payload), (response_path, verification), (artifact_path, artifact)):
        write_json(path, value)
    directory = closed_loop.start_closed_loop(
        table, payload_path, response_path, artifact_path, tmp_path / "runs",
        max_rounds=max_rounds, source_signature="synthetic-implementation",
    )
    return table, directory, payload, verification, artifact


def response_callback(captured):
    def request(stage, prompt, metadata):
        captured.append((stage, prompt, metadata))
        evidence = json.loads(prompt.rsplit("\n", 1)[1])
        return json.dumps({"cells": judgments(evidence["cells"])})
    return request


def test_joint_tentative_context_recomputes_deltas_with_original_statistics(tmp_path, monkeypatch):
    table, directory, initial, *_ = seed(tmp_path)
    observations = []
    original_index = closed_loop.DirtyIndex
    class ObservedIndex(original_index):
        def __init__(self, index_table, *args, **kwargs):
            observations.append(index_table.rows)
            super().__init__(index_table, *args, **kwargs)
    monkeypatch.setattr(closed_loop, "DirtyIndex", ObservedIndex)
    status = closed_loop.advance_closed_loop(table, directory, source_signature="synthetic-implementation")
    assert status["status"] == "awaiting_verification" and status["external_rounds"] == 1
    assert status["next_round"] == 2
    following = json.loads(Path(status["request_path"]).read_text())
    original = next(cell for cell in initial["cells"] if cell["column"] == "left")
    updated = next(cell for cell in following["cells"] if cell["column"] == "left")
    assert original["candidates"][0]["delta_features"][0] == -1
    assert updated["candidates"][0]["delta_features"][0] == 1
    assert updated["candidates"][0]["delta_features"][1] == 1
    assert updated["candidates"][0]["delta_features"][3] == original["candidates"][0]["delta_features"][3]
    assert updated["tuple"] == {"left": "A", "right": "A"}
    assert updated["tentative_tuple"] == {"left": "B", "right": "B"}
    assert updated["counterfactual_before_tuple"] == {"left": "A", "right": "B"}
    assert updated["context_provenance"]["tentative_tuple"] == "untrusted_model_proposal"
    assert all(rows == table.rows for rows in observations)
    assert observations and table.rows[0] == ("A", "A")
    prediction_path = directory / "rounds" / "0001" / "prediction.json"
    assert following["closed_loop"]["parent_prediction_sha256"] == sha256(prediction_path)
    assert following["closed_loop"]["parent_input_hashes"]
    assert following["request_hash"] != initial["request_hash"]
    assert closed_loop._set_hash(following) == closed_loop._set_hash(initial)
    assert not (directory / "frozen.json").exists()


def test_missing_fresh_response_queues_and_waits_without_convergence(tmp_path):
    table, directory, initial, *_ = seed(tmp_path)
    queued = []
    def pending(stage, prompt, metadata):
        queued.append((stage, prompt, metadata))
        raise ResponsePending("Synthetic fresh response pending")
    status = closed_loop.advance_closed_loop(table, directory, source_signature="synthetic-implementation", request=pending)
    assert status["status"] == "awaiting_verification" and status["external_rounds"] == 1
    assert len(queued) == 1
    assert queued[0][2]["candidate_request_hash"] == status["request_hash"] != initial["request_hash"]
    assert "untrusted" in queued[0][1] and "tentative_tuple" in queued[0][1]
    before = (directory / "rounds" / "0001" / "prediction.json").read_bytes()
    again = closed_loop.advance_closed_loop(table, directory, source_signature="synthetic-implementation")
    assert again["status"] == "awaiting_verification" and again["external_rounds"] == 1
    assert (directory / "rounds" / "0001" / "prediction.json").read_bytes() == before
    assert not (directory / "prediction.json").exists()


def test_stale_seed_verification_cannot_be_reused_for_next_round(tmp_path):
    table, directory, _, verification, _ = seed(tmp_path)
    status = closed_loop.advance_closed_loop(table, directory, source_signature="synthetic-implementation")
    write_json(status["verification_path"], verification)
    with pytest.raises(ValueError, match="exact candidate request"):
        closed_loop.advance_closed_loop(table, directory, source_signature="synthetic-implementation")
    assert not (directory / "frozen.json").exists()


def test_fresh_rounds_converge_and_freeze_actual_external_history(tmp_path):
    table, directory, *_ = seed(tmp_path)
    calls = []
    status = closed_loop.advance_closed_loop(table, directory, source_signature="synthetic-implementation",
                                            request=response_callback(calls))
    assert status["status"] == "complete" and status["external_rounds"] == 3
    assert status["stop_reason"] == "unchanged_repairs_and_probabilities"
    assert len(calls) == 2  # Round one was already externally verified at start.
    assert len({metadata["candidate_request_hash"] for _, _, metadata in calls}) == 2
    prediction = Prediction.load(status["prediction_path"])
    assert prediction.repairs == {(0, 0): "B", (0, 1): "B"}
    assert prediction.metadata["rounds_executed"] == prediction.metadata["external_verification_rounds"] == 3
    assert prediction.metadata["closed_loop"]["iterative_cts_requery"] is False
    assert len({round_["request_hash"] for round_ in prediction.metadata["closed_loop"]["history"]}) == 3
    frozen = json.loads(Path(status["frozen_path"]).read_text())
    for path, expected in frozen["inference_inputs"].items():
        assert sha256(path) == expected
    before = {path: path.read_bytes() for path in directory.rglob("*.json")}
    resumed = closed_loop.advance_closed_loop(table, directory, source_signature="synthetic-implementation")
    assert resumed == status
    assert all(path.read_bytes() == body for path, body in before.items())


def test_max_round_limit_counts_only_completed_external_responses(tmp_path):
    table, directory, *_ = seed(tmp_path, max_rounds=1)
    status = closed_loop.advance_closed_loop(table, directory, source_signature="synthetic-implementation")
    assert status["external_rounds"] == 1 and status["stop_reason"] == "max_rounds"
    assert not (directory / "rounds" / "0002").exists()


def test_new_external_judgment_can_retract_prior_repairs(tmp_path):
    table, directory, *_ = seed(tmp_path)
    contexts = []
    def preserve_original(stage, prompt, metadata):
        evidence = json.loads(prompt.rsplit("\n", 1)[1])
        contexts.append(evidence["cells"][0]["tentative_tuple"])
        return json.dumps({"cells": [{
            "row": cell["row"], "column": cell["column"], "original_valid_score": .99,
            "selected_value": cell["original"], "confidence": .99,
            "explanation": "Synthetic new verification preserves the original.",
            "candidates": [{"value": candidate["value"], "score": .05} for candidate in cell["candidates"]],
        } for cell in evidence["cells"]]})
    status = closed_loop.advance_closed_loop(table, directory, source_signature="synthetic-implementation", request=preserve_original)
    initial = Prediction.load(directory / "rounds" / "0001" / "prediction.json")
    final = Prediction.load(status["prediction_path"])
    assert len(initial.repairs) == 2 and final.repairs == {}
    assert contexts == [{"left": "B", "right": "B"}, {"left": "A", "right": "A"}]
    assert status["external_rounds"] == 3


@pytest.mark.parametrize("change", ["artifact", "candidate_set", "incomplete_verifier", "incomplete_candidate_scores", "implementation"])
def test_corrupted_or_incomplete_feedback_inputs_fail_closed(tmp_path, change):
    table, directory, payload, verification, artifact = seed(tmp_path)
    if change == "implementation":
        with pytest.raises(ValueError, match="implementation changed"):
            closed_loop.advance_closed_loop(table, directory, source_signature="different-implementation")
        return
    if change == "artifact":
        artifact["value_repairs"] = {}
        with pytest.raises(ValueError, match="artifact hash"):
            closed_loop.next_round_payload(
                table, payload, Prediction(np.full(table.shape, .9)), artifact, round_number=2,
                initial_request_hash=payload["request_hash"], initial_candidate_set_hash=closed_loop._set_hash(payload),
                parent_inputs={}, parent_prediction_hash="synthetic", parent_record_hash="synthetic", config=LoopConfig(rounds=1),
            )
    elif change == "candidate_set":
        status = closed_loop.advance_closed_loop(table, directory, source_signature="synthetic-implementation")
        following = json.loads(Path(status["request_path"]).read_text())
        following["cells"][0]["candidates"][0]["value"] = "invented"
        following["request_hash"] = digest({key: value for key, value in following.items() if key != "request_hash"})
        write_json(status["request_path"], following)
        with pytest.raises(ValueError, match="eligibility or values changed|immutable closed-loop"):
            closed_loop.advance_closed_loop(table, directory, source_signature="synthetic-implementation")
    else:
        verification = deepcopy(verification)
        if change == "incomplete_verifier": verification["cells"].pop()
        else: verification["cells"][0]["candidates"].pop()
        with pytest.raises(ValueError, match="every requested cell|every requested candidate"):
            closed_loop._complete_verification(table, payload, verification)


def test_feedback_module_and_cli_never_import_reference_evaluation():
    root = Path(closed_loop.__file__).resolve().parents[1]
    for path in (Path(closed_loop.__file__), root / "scripts" / "run_closed_loop.py"):
        tree = ast.parse(path.read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        assert not any(module.endswith("evaluation") or module == "evaluate_e2e" for module in modules)
