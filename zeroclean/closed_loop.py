"""Resumable external-verifier feedback on explicitly untrusted repair context.

Only original dirty observations populate DirtyIndex. Candidate eligibility and
values stay fixed at preparation: this protocol does not rerun CTS retrieval.
Every completed external round uses one inference pass and a final tuple guard.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np

from .data import sha256
from .llm import ResponsePending
from .loop import LoopConfig, infer_from_candidates, validate_verification
from .sandbox import evaluate_rules
from .statistics import DirtyIndex
from .types import Prediction, digest, write_json
from .verification import queue_verification


PROTOCOL_VERSION = "external-verifier-feedback-v1"


def _write_once(path, value):
    path = Path(path)
    if path.exists():
        if digest(json.loads(path.read_text())) != digest(value):
            raise ValueError(f"Refusing to overwrite immutable closed-loop artifact: {path}")
    else:
        write_json(path, value)


def _prediction_body(prediction):
    return {
        "probabilities": prediction.probabilities.tolist(),
        "repairs": [{"row": i, "column": j, "value": value}
                    for (i, j), value in sorted(prediction.repairs.items())],
        "metadata": prediction.metadata,
    }


def _set_hash(payload):
    return digest([{"row": cell["row"], "column_index": cell["column_index"],
                    "original": cell["original"],
                    "values": [candidate["value"] for candidate in cell["candidates"]]}
                   for cell in payload["cells"]])


def _check_hashes(inputs):
    for filename, expected in inputs.items():
        path = Path(filename)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Closed-loop input changed: {filename}")


def _validate_payload(table, payload):
    if payload.get("dirty_sha256") != table.fingerprint or payload.get("dataset") != table.name:
        raise ValueError("Closed-loop dirty input mismatch")
    expected = digest({key: value for key, value in payload.items() if key != "request_hash"})
    if payload.get("request_hash") != expected:
        raise ValueError("Closed-loop request content hash mismatch")
    Prediction(np.asarray(payload["base_probabilities"])).validate(table.shape)
    seen = set()
    for cell in payload["cells"]:
        i, j = cell.get("row"), cell.get("column_index")
        if type(i) is not int or type(j) is not int or not 0 <= i < table.shape[0] or not 0 <= j < table.shape[1]:
            raise ValueError("Invalid closed-loop candidate coordinate")
        if (i, j) in seen:
            raise ValueError("Duplicate closed-loop candidate coordinate")
        seen.add((i, j))
        if cell.get("column") != table.columns[j] or cell.get("original") != table.rows[i][j]:
            raise ValueError("Closed-loop original coordinate/value mismatch")
        if cell.get("tuple") != dict(zip(table.columns, table.rows[i])):
            raise ValueError("Closed-loop tuple must retain original dirty observations")
        values = []
        for candidate in cell["candidates"]:
            value = candidate.get("value")
            if not isinstance(value, str) or value == cell["original"] or value in values:
                raise ValueError("Invalid or duplicate closed-loop candidate value")
            values.append(value)


def _complete_verification(table, payload, response):
    verified = validate_verification(table, payload, response)
    if len(verified) != len(payload["cells"]):
        raise ValueError("Closed-loop verification must cover every requested cell")
    for cell in payload["cells"]:
        item = verified[cell["row"], cell["column_index"]]
        for field in ("selected_value", "original_valid_score", "confidence"):
            if field not in item or (field != "selected_value" and item[field] is None):
                raise ValueError(f"Incomplete closed-loop verifier {field}")
        if {candidate["value"] for candidate in item.get("candidates", [])} != {
            candidate["value"] for candidate in cell["candidates"]
        }:
            raise ValueError("Closed-loop verification must score every requested candidate")
    return verified


def _validate_artifact(table, payload, artifact):
    if payload.get("artifact_hash") != digest(artifact):
        raise ValueError("Closed-loop pass2 artifact hash mismatch")
    for field in ("dirty_sha256", "model", "reasoning_effort"):
        if artifact.get(field) != payload.get(field):
            raise ValueError(f"Closed-loop pass2 {field} mismatch")


def start_closed_loop(table, payload_path, verification_path, artifact_path, output_root, *,
                      max_rounds=3, source_signature):
    """Snapshot a fully verified seed into an immutable, versioned run directory."""
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer")
    paths = [Path(path).resolve() for path in (payload_path, verification_path, artifact_path)]
    original_hashes = {str(path): sha256(path) for path in paths}
    payload, verification, artifact = (json.loads(path.read_text()) for path in paths)
    _validate_payload(table, payload)
    _validate_artifact(table, payload, artifact)
    _complete_verification(table, payload, verification)
    config = LoopConfig(**payload["config"])
    if not config.counterfactual or not config.iterative_verification:
        raise ValueError("Closed-loop feedback requires counterfactual and external verification enabled")
    config = replace(config, rounds=1, iterative_verification=True)
    identity = {
        "version": PROTOCOL_VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
        "seed_request_hash": payload["request_hash"], "seed_verification_hash": digest(verification),
        "artifact_hash": digest(artifact), "initial_candidate_set_hash": _set_hash(payload),
        "max_rounds": max_rounds, "inference_config": asdict(config),
        "source_signature": source_signature, "seed_input_hashes": original_hashes,
    }
    destination = Path(output_root) / table.name / digest(identity)
    snapshots = {
        "request": ("rounds/0001/request.json", payload),
        "verification": ("rounds/0001/verification.json", verification),
        "artifact": ("pass2.json", artifact),
    }
    snapshot_hashes = {}
    for name, (relative, body) in snapshots.items():
        path = destination / relative
        _write_once(path, body)
        snapshot_hashes[name] = {"file": relative, "sha256": sha256(path)}
    _check_hashes(original_hashes)
    manifest = {"identity": identity, "protocol_hash": digest(identity), "snapshots": snapshot_hashes,
                "candidate_policy": "Fixed initial candidate cells and values; no iterative CTS requery",
                "context_policy": "Prior repairs are untrusted model proposals; statistics use original dirty observations only"}
    manifest["manifest_hash"] = digest(manifest)
    _write_once(destination / "manifest.json", manifest)
    return destination


def next_round_payload(table, payload, prediction, artifact, *, round_number,
                       initial_request_hash, initial_candidate_set_hash, parent_inputs,
                       parent_prediction_hash, parent_record_hash, config):
    """Recompute replacement effects with other tentative row edits held fixed."""
    _validate_payload(table, payload)
    _validate_artifact(table, payload, artifact)
    prediction.validate(table.shape)
    proposed = [list(row) for row in table.rows]
    for (i, j), value in prediction.repairs.items():
        proposed[i][j] = value
    result = deepcopy(payload)
    result.pop("request_hash", None)
    result["config"] = asdict(config)
    rules = result["rules"]
    index = DirtyIndex(table, artifact.get("dependencies", []) if config.multi_column else [],
                       discover=config.multi_column)
    target_rules = {j: [k for k, rule in enumerate(rules) if rule["target"] == column]
                    for j, column in enumerate(table.columns)}
    before_rows = []
    for cell in result["cells"]:
        i, j = cell["row"], cell["column_index"]
        tentative = dict(zip(table.columns, proposed[i]))
        before = dict(tentative)
        before[table.columns[j]] = cell["original"]
        cell["tentative_tuple"] = tentative
        cell["counterfactual_before_tuple"] = before
        cell["context_provenance"] = {
            "tuple": "original_dirty_observation", "tentative_tuple": "untrusted_model_proposal",
            "target_before": "original_dirty_value", "parent_prediction_sha256": parent_prediction_hash,
        }
        before_rows.append(before)
    before_matrix = np.asarray(evaluate_rules(rules, before_rows), dtype=np.int8).reshape(len(before_rows), len(rules))
    jobs = []
    for position, cell in enumerate(result["cells"]):
        cell["original_rule_violations"] = int((before_matrix[position] == 1).sum())
        for candidate in cell["candidates"]:
            after = dict(before_rows[position])
            after[cell["column"]] = candidate["value"]
            jobs.append((position, cell, candidate, after))
    for start in range(0, len(jobs), 1000):
        chunk = jobs[start:start + 1000]
        changed = np.asarray(evaluate_rules(rules, [after for _, _, _, after in chunk]), dtype=np.int8).reshape(len(chunk), len(rules))
        for q, (position, cell, candidate, after) in enumerate(chunk):
            i, j = cell["row"], cell["column_index"]
            before = before_rows[position]
            before_values = tuple(before[column] for column in table.columns)
            after_values = tuple(after[column] for column in table.columns)
            rr = target_rules[j]
            rule_delta = ((before_matrix[position, rr] == 1).sum() - (changed[q, rr] == 1).sum()) / max(len(rr), 1)
            fd_delta = (index.conditional(i, j, candidate["value"], after_values)[0]
                        - index.conditional(i, j, cell["original"], before_values)[0]) if config.multi_column else 0.
            vicinity_delta = math.tanh((index.vicinity(i, j, candidate["value"], after_values)
                                       - index.vicinity(i, j, cell["original"], before_values)) / 4)
            domain_delta = math.tanh((index.domain_support(i, j, candidate["value"])
                                     - index.domain_support(i, j, cell["original"])) / 4)
            candidate["delta_features"] = [float(rule_delta), float(fd_delta), float(vicinity_delta), float(domain_delta)]
            candidate["introduced_violation"] = bool(((before_matrix[position] == 0) & (changed[q] == 1)).any())
            candidate["after_rule_violations"] = int((changed[q] == 1).sum())
            candidate["counterfactual_error"] = bool((changed[q] == -1).any())
    result["dependency_count"] = len(index.dependencies)
    result["closed_loop"] = {
        "version": PROTOCOL_VERSION, "external_round": round_number,
        "initial_request_hash": initial_request_hash, "initial_candidate_set_hash": initial_candidate_set_hash,
        "parent_request_hash": payload["request_hash"], "parent_prediction_sha256": parent_prediction_hash,
        "parent_round_record_sha256": parent_record_hash, "parent_input_hashes": parent_inputs,
        "tentative_context_trust": "untrusted_model_proposals",
        "statistical_observations": "original_dirty_table_only", "iterative_cts_requery": False,
    }
    result["request_hash"] = digest(result)
    if _set_hash(result) != initial_candidate_set_hash:
        raise ValueError("Closed-loop initial candidate eligibility or values changed")
    return result


def _read_manifest(table, directory, source_signature):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    expected = digest({key: value for key, value in manifest.items() if key != "manifest_hash"})
    identity = manifest["identity"]
    if manifest.get("manifest_hash") != expected or manifest.get("protocol_hash") != digest(identity) or directory.name != digest(identity):
        raise ValueError("Closed-loop manifest content hash mismatch")
    if identity.get("version") != PROTOCOL_VERSION or identity.get("dataset") != table.name or identity.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Closed-loop protocol or dirty input mismatch")
    if identity.get("source_signature") != source_signature:
        raise ValueError("Closed-loop implementation changed; start a new run")
    inputs = {str(path): sha256(path)}
    for snapshot in manifest["snapshots"].values():
        inputs[str(directory / snapshot["file"])] = snapshot["sha256"]
    _check_hashes(inputs)
    return manifest, inputs


def advance_closed_loop(table, run_directory, *, source_signature, request=None, batch_size=20):
    """Resume until a fresh verifier response is missing, convergence, or the limit.

    Supply the FileLLMQueue request callback to queue missing judgments. Without
    it, this only exposes the next exact request and returns a waiting status.
    """
    directory = Path(run_directory).resolve()
    manifest, seed_inputs = _read_manifest(table, directory, source_signature)
    identity = manifest["identity"]
    frozen_path = directory / "frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if frozen.get("frozen_hash") != digest({key: value for key, value in frozen.items() if key != "frozen_hash"}):
            raise ValueError("Closed-loop frozen record content hash mismatch")
        _check_hashes(frozen["inference_inputs"])
        _check_hashes({str(directory / "prediction.json"): frozen["prediction_sha256"]})
        return {"status": "complete", "external_rounds": frozen["external_rounds"],
                "stop_reason": frozen["stop_reason"], "run_directory": str(directory),
                "prediction_path": str(directory / "prediction.json"), "frozen_path": str(frozen_path)}
    artifact = json.loads((directory / manifest["snapshots"]["artifact"]["file"]).read_text())
    config = LoopConfig(**identity["inference_config"])
    previous = None
    history = []
    all_inputs = dict(seed_inputs)
    for number in range(1, identity["max_rounds"] + 1):
        round_directory = directory / "rounds" / f"{number:04d}"
        request_path = round_directory / "request.json"
        response_path = round_directory / "verification.json"
        prediction_path = round_directory / "prediction.json"
        record_path = round_directory / "frozen.json"
        payload = json.loads(request_path.read_text())
        _validate_payload(table, payload)
        _validate_artifact(table, payload, artifact)
        if _set_hash(payload) != identity["initial_candidate_set_hash"]:
            raise ValueError("Closed-loop candidate eligibility or values changed")
        if not response_path.exists():
            reason = "A fresh external verifier response is required"
            if request is not None:
                try:
                    queue_verification(table, payload, request, response_path, batch_size)
                except ResponsePending as exc:
                    reason = str(exc)
            if not response_path.exists():
                return {"status": "awaiting_verification", "external_rounds": len(history),
                        "next_round": number, "request_hash": payload["request_hash"],
                        "request_path": str(request_path), "verification_path": str(response_path),
                        "run_directory": str(directory), "reason": reason}
        response = json.loads(response_path.read_text())
        _complete_verification(table, payload, response)
        inputs = dict(seed_inputs, **{str(request_path): sha256(request_path), str(response_path): sha256(response_path)})
        if history:
            previous_record = directory / "rounds" / f"{number - 1:04d}" / "frozen.json"
            inputs[str(previous_record)] = sha256(previous_record)
            parent = payload.get("closed_loop", {})
            if parent.get("external_round") != number or parent.get("parent_prediction_sha256") != history[-1]["prediction_sha256"]:
                raise ValueError("Closed-loop parent prediction chain mismatch")
            if parent.get("parent_request_hash") != history[-1]["request_hash"] or parent.get("parent_round_record_sha256") != sha256(previous_record):
                raise ValueError("Closed-loop parent request/record chain mismatch")
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("record_hash") != digest({key: value for key, value in record.items() if key != "record_hash"}):
                raise ValueError("Closed-loop round record content hash mismatch")
            if record.get("inference_inputs") != inputs or record.get("request_hash") != payload["request_hash"]:
                raise ValueError("Closed-loop completed round inputs changed")
            _check_hashes({str(prediction_path): record["prediction_sha256"]})
            prediction = Prediction.load(prediction_path).validate(table.shape)
        else:
            prediction = infer_from_candidates(table, payload, response, config)
            prediction.metadata["closed_loop"] = {
                "version": PROTOCOL_VERSION, "external_round": number,
                "protocol_hash": manifest["protocol_hash"], "fresh_verification": True,
                "tentative_context_trust": "untrusted_model_proposals", "iterative_cts_requery": False,
            }
            _check_hashes(inputs)
            _write_once(prediction_path, _prediction_body(prediction))
            record = {"external_round": number, "request_hash": payload["request_hash"],
                      "inference_inputs": inputs, "inference_config": asdict(config),
                      "prediction_sha256": sha256(prediction_path),
                      "frozen_before_evaluation": True, "ground_truth_access": False}
            record["record_hash"] = digest(record)
            _write_once(record_path, record)
        _check_hashes(inputs)
        all_inputs.update(inputs)
        all_inputs.update({str(prediction_path): record["prediction_sha256"], str(record_path): sha256(record_path)})
        history.append({"external_round": number, "request_hash": payload["request_hash"],
                        "verification_sha256": sha256(response_path), "prediction_sha256": record["prediction_sha256"],
                        "round_record_sha256": sha256(record_path)})
        stable = previous is not None and prediction.repairs == previous.repairs and np.array_equal(prediction.probabilities, previous.probabilities)
        if stable or number == identity["max_rounds"]:
            reason = "unchanged_repairs_and_probabilities" if stable else "max_rounds"
            prediction.metadata["rounds_executed"] = number
            prediction.metadata["external_verification_rounds"] = number
            prediction.metadata["closed_loop"] = {
                **prediction.metadata["closed_loop"], "external_rounds": number,
                "inner_passes_per_external_round": 1, "stop_reason": reason, "history": history,
                "candidate_policy": manifest["candidate_policy"], "context_policy": manifest["context_policy"],
            }
            _write_once(directory / "prediction.json", _prediction_body(prediction))
            frozen = {"version": PROTOCOL_VERSION, "protocol_hash": manifest["protocol_hash"],
                      "dirty_sha256": table.fingerprint, "external_rounds": number, "stop_reason": reason,
                      "prediction_sha256": sha256(directory / "prediction.json"), "inference_inputs": all_inputs,
                      "frozen_before_evaluation": True, "ground_truth_access": False}
            frozen["frozen_hash"] = digest(frozen)
            _check_hashes(all_inputs)
            _write_once(frozen_path, frozen)
            return {"status": "complete", "external_rounds": number, "stop_reason": reason,
                    "run_directory": str(directory), "prediction_path": str(directory / "prediction.json"),
                    "frozen_path": str(frozen_path)}
        next_payload = next_round_payload(
            table, payload, prediction, artifact, round_number=number + 1,
            initial_request_hash=identity["seed_request_hash"], initial_candidate_set_hash=identity["initial_candidate_set_hash"],
            parent_inputs=inputs, parent_prediction_hash=record["prediction_sha256"],
            parent_record_hash=sha256(record_path), config=config,
        )
        _write_once(directory / "rounds" / f"{number + 1:04d}" / "request.json", next_payload)
        previous = prediction
    raise RuntimeError("Closed-loop round limit was not handled")
