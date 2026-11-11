"""Matched, content-addressed ablations with independent verifier responses.

This module only reads dirty inference inputs. Preparation never queues a model
request, and running never imports reference-backed evaluation code.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from .data import sha256
from .loop import LoopConfig, build_candidates, infer_from_candidates, validate_verification
from .types import Prediction, digest, write_json


PROTOCOL_VERSION = "matched-ablation-v1"
COUNTERFACTUAL_FIELDS = frozenset({
    "delta_features", "delta_dimensions", "introduced_violation",
    "after_rule_violations", "counterfactual_error", "original_rule_violations",
})
DEFINITIONS = {
    "zeroclean_loop": "Full method with external verification and repeated immutable-evidence passes.",
    "without_counterfactual": (
        "Remove supplied single-cell counterfactual features, scoring, and gates; "
        "retain independent verification and final tuple consistency checks."
    ),
    "without_multi_column": (
        "Remove multi-column executable rules and functional dependencies; "
        "retain contextual tuple retrieval and independently verify this request."
    ),
    "without_iterative_verification": (
        "One simultaneous inference pass with independent external verification "
        "and one final tuple consistency check; no repeated passes."
    ),
}


def _rehash(payload):
    payload["request_hash"] = digest({k: v for k, v in payload.items() if k != "request_hash"})
    return payload


def redact_counterfactual_evidence(value):
    """Produce the actual no-counterfactual request, without a hidden full copy."""
    if isinstance(value, dict):
        return {key: redact_counterfactual_evidence(item) for key, item in value.items()
                if key not in COUNTERFACTUAL_FIELDS and key != "request_hash"}
    if isinstance(value, list):
        return [redact_counterfactual_evidence(item) for item in value]
    return deepcopy(value)


def _write_immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if digest(json.loads(path.read_text())) != digest(value):
            raise ValueError(f"Refusing to overwrite immutable ablation artifact: {path}")
        return
    write_json(path, value)


def _file_state(path):
    path = Path(path)
    return sha256(path) if path.exists() else None


def _check_inputs(inputs):
    for filename, expected in inputs.items():
        if _file_state(filename) != expected:
            raise ValueError(f"Matched ablation input changed: {filename}")


def prepare_matched_ablations(table, output_root, artifact_path, baseline_path, *,
                              detector="adapted", config=LoopConfig(),
                              artifact_root="artifacts/llm", source_signature):
    """Write four immutable requests and return their content-addressed directory."""
    if detector not in {"adapted", "source"}:
        raise ValueError("Unknown ablation detector kind")
    if not config.counterfactual or not config.multi_column or not config.iterative_verification:
        raise ValueError("Matched ablations require the enabled full-method configuration")
    if config.rounds < 2:
        raise ValueError("The full method needs at least two passes for the iteration ablation")
    artifact_path, baseline_path = Path(artifact_path).resolve(), Path(baseline_path).resolve()
    proposal_path = (Path(artifact_root) / table.name / config.proposal_filename).resolve()
    inputs = {str(path): _file_state(path) for path in (artifact_path, baseline_path, proposal_path)}
    if inputs[str(artifact_path)] is None or inputs[str(baseline_path)] is None:
        raise ValueError("Prepare matched ablations only after rule and detector artifacts exist")
    artifact = json.loads(artifact_path.read_text())
    for field, expected in (("dirty_sha256", table.fingerprint), ("model", "gpt-6.1-sol"),
                            ("reasoning_effort", "xhigh")):
        if artifact.get(field) != expected:
            raise ValueError(f"Matched ablation rule artifact {field} mismatch")
    baseline = Prediction.load(baseline_path).validate(table.shape)
    fingerprint = baseline.metadata.get("input_fingerprint", baseline.metadata.get("dirty_sha256"))
    if fingerprint != table.fingerprint:
        raise ValueError("Matched ablation detector input fingerprint mismatch")
    if detector == "source" and baseline.metadata.get("reproduced_upstream") is not True:
        raise ValueError("Matched source ablations require reproduced upstream detector probabilities")
    provenance = {
        "kind": detector, "prediction_file": str(baseline_path),
        "prediction_file_sha256": inputs[str(baseline_path)],
        "probabilities_hash": digest(baseline.probabilities.tolist()),
        "metadata_hash": digest(baseline.metadata),
    }
    variants = {
        "zeroclean_loop": config,
        "without_counterfactual": replace(config, counterfactual=False),
        "without_multi_column": replace(config, multi_column=False),
        # Keep verifier use enabled; only the repeated passes are removed.
        "without_iterative_verification": replace(config, rounds=1),
    }
    requests = {}
    for name, variant in variants.items():
        payload = build_candidates(table, artifact, baseline, variant, artifact_root=artifact_root)
        if name == "without_counterfactual":
            payload = redact_counterfactual_evidence(payload)
        payload["detector_provenance"] = deepcopy(provenance)
        payload["ablation_protocol"] = {
            "version": PROTOCOL_VERSION, "variant": name, "definition": DEFINITIONS[name],
            "fresh_verification_required": True,
            "repeated_passes_request_new_llm_judgments": False,
        }
        requests[name] = _rehash(payload)
    _check_inputs(inputs)
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "dataset": table.name,
        "dirty_sha256": table.fingerprint, "source_signature": source_signature,
        "detector_provenance": provenance, "input_files": inputs,
        "variants": {
            name: {
                "config": asdict(variants[name]), "definition": DEFINITIONS[name],
                "request_file": f"requests/{name}.json", "request_hash": payload["request_hash"],
                "verification_file": f"verifications/{name}.json",
                "prediction_file": f"predictions/{table.name}/{name}.json",
            } for name, payload in requests.items()
        },
    }
    manifest["protocol_hash"] = digest(manifest)
    destination = Path(output_root) / "matched_ablations" / table.name / manifest["protocol_hash"]
    for name, payload in requests.items():
        _write_immutable_json(destination / manifest["variants"][name]["request_file"], payload)
    _write_immutable_json(destination / "manifest.json", manifest)
    return destination


def run_matched_ablations(table, prepared_directory, *, source_signature):
    """Require all independent replies, infer with fixed configs, then freeze once."""
    directory = Path(prepared_directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = digest({k: v for k, v in manifest.items() if k != "protocol_hash"})
    if manifest.get("protocol_hash") != expected or directory.name != expected:
        raise ValueError("Matched ablation manifest content hash mismatch")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unknown matched ablation protocol")
    if manifest.get("dirty_sha256") != table.fingerprint or manifest.get("dataset") != table.name:
        raise ValueError("Matched ablation dirty input mismatch")
    if manifest.get("source_signature") != source_signature:
        raise ValueError("Matched ablation implementation changed; prepare new requests")
    frozen_path = directory / "frozen" / f"{table.name}.json"
    if frozen_path.exists():
        raise ValueError("Matched ablation run is already frozen; prepare a new protocol instead")
    if set(manifest.get("variants", {})) != set(DEFINITIONS):
        raise ValueError("Matched ablation manifest must contain the full matched variant set")
    _check_inputs(manifest["input_files"])
    inputs = dict(manifest["input_files"])
    inputs[str(manifest_path)] = sha256(manifest_path)
    pending = []
    for name, variant in manifest["variants"].items():
        path = directory / variant["verification_file"]
        if not path.is_file():
            pending.append(str(path))
    if pending:
        raise ValueError("Fresh matched-ablation verification required before inference: " + ", ".join(pending))
    for variant in manifest["variants"].values():
        path = directory / variant["prediction_file"]
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing matched ablation prediction: {path}")
    loaded = {}
    # Validate every response before producing any predictions. A present but
    # partial response must not silently fall back to unverified proposal scores.
    for name, variant in manifest["variants"].items():
        request_path, response_path = (directory / variant[key] for key in ("request_file", "verification_file"))
        payload, response = (json.loads(path.read_text()) for path in (request_path, response_path))
        content_hash = digest({k: v for k, v in payload.items() if k != "request_hash"})
        if payload.get("request_hash") != content_hash or payload["request_hash"] != variant["request_hash"]:
            raise ValueError(f"Matched ablation request changed: {name}")
        if payload.get("detector_provenance") != manifest["detector_provenance"]:
            raise ValueError(f"Matched ablation detector identity changed: {name}")
        verified = validate_verification(table, payload, response)
        if len(verified) != len(payload["cells"]):
            raise ValueError(f"Fresh verification must cover every requested cell: {name}")
        for cell in payload["cells"]:
            judged = {candidate["value"] for candidate in verified[cell["row"], cell["column_index"]].get("candidates", [])}
            if judged != {candidate["value"] for candidate in cell["candidates"]}:
                raise ValueError(f"Fresh verification must score every requested candidate: {name}")
        inputs.update({str(path): sha256(path) for path in (request_path, response_path)})
        loaded[name] = payload, response, LoopConfig(**variant["config"])
    predictions = {}
    for name, (payload, response, config) in loaded.items():
        prediction = infer_from_candidates(table, payload, response, config)
        prediction.metadata["ablation_protocol"] = {
            **payload["ablation_protocol"], "protocol_hash": manifest["protocol_hash"],
            "verification_mode": "independent_variant_request",
            "verification_sha256": inputs[str(directory / manifest["variants"][name]["verification_file"])],
        }
        predictions[name] = prediction
    _check_inputs(inputs)
    prediction_hashes = {}
    for name, prediction in predictions.items():
        path = directory / manifest["variants"][name]["prediction_file"]
        prediction.save(path)
        prediction_hashes[path.name] = sha256(path)
    _check_inputs(inputs)
    frozen = {
        "dirty_sha256": table.fingerprint, "source_signature": source_signature,
        "ablation_protocol": {"version": PROTOCOL_VERSION, "protocol_hash": manifest["protocol_hash"],
                              "verification_mode": "independent_variant_request"},
        "detector_provenance": manifest["detector_provenance"],
        "prediction_files": prediction_hashes, "inference_inputs": inputs,
        "frozen_before_evaluation": True, "time_unix": time.time(),
    }
    _write_immutable_json(frozen_path, frozen)
    return frozen
