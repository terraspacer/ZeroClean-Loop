"""Standalone, preregistered processing of independently verified frozen evidence.

No fit, LLM queue, reference loader, evaluator or standard closed-loop inference
is used. The source prior is fallback; new EC choices are bounded by frozen v3.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from experimental.latent_consensus_feedback import VERSION as V3, FeedbackPolicy
from experimental.latent_consensus_detection_feedback_v4 import VERSION as V4
from scripts import prepare_feedback_verification as adapter
from zeroclean.data import sha256
from zeroclean.loop import validate_verification
from zeroclean.sandbox import evaluate_rules, validate_rules
from zeroclean.types import Prediction, digest


VERSION = "verified-statistical-feedback-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("verified_statistical_feedback_protocol.md")
PROTOCOL_SHA256 = "82b8d5c9409556570293d5fae8719f2048b5f21c0f17b3ac979aeb69a341bea9"
MODES = {"independent_names", "cluster_near_agreement"}


@dataclass(frozen=True)
class Policy:
    detection_threshold: float = .5
    verification_confidence: float = .8
    candidate_score: float = .8
    runtime_error_rate: float = .05
    null_policy: str = "abstain_from_new_statistical_ec_preserve_prior"


POLICY = Policy()


def prediction_body(prediction):
    return {"probabilities": prediction.probabilities.tolist(),
            "repairs": [{"row": i, "column": j, "value": v} for (i, j), v in sorted(prediction.repairs.items())],
            "metadata": prediction.metadata}


def _probability(value):
    if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Weight/score must be finite and in [0,1]")
    return float(value)


def _interval(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Expected a two-endpoint interval")
    low, high = map(_probability, value)
    if low > high:
        raise ValueError("Interval endpoints reversed")
    return low, high


def _fingerprint(table, prediction):
    prediction.validate(table.shape)
    seen = [prediction.metadata[k] for k in ("dirty_sha256", "dirty_fingerprint", "input_fingerprint") if k in prediction.metadata]
    if not seen or any(v != table.fingerprint for v in seen):
        raise ValueError("Prediction dirty fingerprint mismatch")


def _rows(table, repairs):
    rows = table.records()
    for (i, j), value in repairs.items():
        rows[i][table.columns[j]] = value
    return rows


def _contracts(bundle):
    contracts = bundle.get("frozen_semantic_contracts")
    if not isinstance(contracts, list) or not contracts:
        raise ValueError("Both frozen semantic contracts are required")
    result = {}
    for item in contracts:
        if (not isinstance(item, dict) or not isinstance(item.get("attribute"), str)
                or item.get("source_assumption") not in MODES or not isinstance(item.get("contract"), dict)
                or not item["contract"]):
            raise ValueError("Malformed frozen semantic contract")
        key = item["attribute"], item["source_assumption"]
        if key in result:
            raise ValueError("Duplicate frozen semantic contract")
        result[key] = digest(item)
    return result


def _frozen_conclusion(cell, version):
    """Validate arithmetic; do not reconstruct or refit any frozen posterior."""
    if version == V3:
        trace = cell["evidence"]
        low, high = _interval(trace["original_weight_interval"])
        winner = _probability(trace["candidate_weight_lower"])
        if winner < .98 or set(trace["statistical_evidence"]["source_assumptions"]) != MODES:
            raise ValueError("V3 evidence lacks strict frozen support")
        valid = trace["candidate"] == cell["original"]
        if (valid and low < .98) or (not valid and high > .02 + 1e-10):
            raise ValueError("V3 original interval contradicts strict winner")
        score = 1 - low if valid else 1 - high
        if cell["statistical_error_weight"] != score or trace["feedback_probability"] != score:
            raise ValueError("V3 conservative endpoint mismatch")
        return ("valid" if valid else "invalid"), score
    trace = cell["detection_evidence"]
    if set(trace["source_assumptions"]) != MODES:
        raise ValueError("V4 requires both frozen source assumptions")
    if trace.get("eligible_both_source_assumptions") is not True:
        if trace.get("action") != "retain_unsupported" or trace.get("feedback_score") is not None or trace.get("error_weight_interval") is not None:
            raise ValueError("Unsupported V4 evidence has a fabricated conclusion")
        return "unsupported", None
    intervals = []
    for evidence in trace["source_assumptions"].values():
        if evidence.get("eligibility_blockers") != []:
            raise ValueError("V4 interval has source eligibility blockers")
        intervals.append(_interval(evidence["original_weight_interval"]))
    original = [min(v[0] for v in intervals), max(v[1] for v in intervals)]
    expected = [1 - original[1], 1 - original[0]]
    if trace.get("original_weight_interval") != original or trace.get("error_weight_interval") != expected:
        raise ValueError("V4 source/aggregate interval arithmetic mismatch")
    low, high = _interval(expected)
    action, score, conclusion = (("flag", low, "invalid") if low > .5 else
                                 ("prune", high, "valid") if high < .5 else
                                 ("retain_uncertain", None, "uncertain"))
    if trace.get("action") != action or trace.get("feedback_score") != score:
        raise ValueError("V4 frozen action/endpoint mismatch")
    return conclusion, score


def prepare_payload(table, prior, frozen_v3, artifact, bundle):
    """Normalize v3/v4 into a generic-verifier request, never a LoopConfig seed."""
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Preregistered protocol changed; create a new processor version")
    _fingerprint(table, prior)
    _fingerprint(table, frozen_v3)
    if (frozen_v3.metadata.get("system") != V3 or frozen_v3.metadata.get("statistical_feedback_ablation") is not True
            or digest(frozen_v3.metadata.get("policy")) != digest(asdict(FeedbackPolicy()))):
        raise ValueError("Frozen EC must be the original v3 statistical feedback under its fixed policy")
    if any(prior.probabilities[key] < .5 for key in prior.repairs):
        raise ValueError("Source prior repairs must already be inside its detection mask")
    if (not isinstance(bundle, dict) or bundle.get("request_hash") != digest({k: v for k, v in bundle.items() if k != "request_hash"})
            or bundle.get("experiment") not in (V3, V4) or bundle.get("dataset") != table.name
            or bundle.get("dirty_sha256") != table.fingerprint or bundle.get("independent_verification_status") != "not_run"):
        raise ValueError("Frozen bundle envelope/content mismatch")
    version = bundle["experiment"]
    if version == V3:
        adapter._validate_bundle(bundle)
        if digest(bundle["config"]) != digest(asdict(FeedbackPolicy())):
            raise ValueError("Frozen v3 policy differs from the preregistered source policy")
    contracts = _contracts(bundle)
    if artifact.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Executable artifact dirty fingerprint mismatch")
    accepted, validation = validate_rules(artifact.get("rules", []), table.records(), POLICY.runtime_error_rate)
    rules = [r for r in accepted if r["target"] in table.columns]
    if version == V3 and bundle["rules"] != rules:
        raise ValueError("V3 bundle rules differ from the frozen executable artifact")
    originals, prior_rows, statistical_rows = table.records(), _rows(table, prior.repairs), _rows(table, frozen_v3.repairs)
    normalized, seen = [], set()
    source_cells = bundle.get("cells")
    if not isinstance(source_cells, list):
        raise ValueError("Frozen bundle cells must be a list")
    hidden = {k: deepcopy(v) for k, v in bundle.items() if k not in {"cells", "request_hash"}}
    for source in source_cells:
        if not isinstance(source, dict):
            raise ValueError("Frozen bundle cell must be an object")
        i, j = source.get("row"), source.get("column_index")
        if type(i) is not int or type(j) is not int or not 0 <= i < table.shape[0] or not 0 <= j < table.shape[1] or (i, j) in seen:
            raise ValueError("Invalid/duplicate frozen coordinate")
        seen.add((i, j))
        row = source.get("tuple" if version == V3 else "original_tuple")
        if source.get("column") != table.columns[j] or source.get("original") != table.rows[i][j] or row != originals[i]:
            raise ValueError("Frozen original tuple/coordinate mismatch")
        if source.get("prior_inference_tuple") != prior_rows[i] or source.get("tentative_tuple") != statistical_rows[i]:
            raise ValueError("Frozen prior/statistical tuple differs from explicit predictions")
        trace = source["evidence" if version == V3 else "detection_evidence"]
        if any(trace.get(k) != source[k] for k in ("row", "column", "column_index", "original")) or trace.get("prior_probability") != prior.probabilities[i, j]:
            raise ValueError("Frozen trace differs from source prior/cell")
        if version == V4 and (trace.get("original_tuple") != originals[i] or source.get("inherited_strict_ec_value") != frozen_v3.repairs.get((i, j))):
            raise ValueError("V4 inherited EC/original tuple mismatch")
        conclusion, score = _frozen_conclusion(source, version)
        hashes = []
        for mode in sorted(MODES):
            if (table.columns[j], mode) not in contracts:
                raise ValueError("Feedback target lacks both semantic contracts")
            hashes.append(contracts[table.columns[j], mode])
        cell = deepcopy(source)
        cell["tuple"] = deepcopy(row)
        if version == V4:
            candidates = []
            for value, origin in ((frozen_v3.repairs.get((i, j)), "frozen_v3_strict_ec"), (prior.repairs.get((i, j)), "source_prior_fallback")):
                if value is not None and value != source["original"] and value not in {c["value"] for c in candidates}:
                    candidates.append({"value": value, "score": None, "origin": origin})
            cell["candidates"] = candidates
        cell["statistical_conclusion"] = {"original": conclusion, "error_weight_endpoint": score,
                                           "eligible_new_repair": frozen_v3.repairs.get((i, j))}
        cell["feedback_verification_context"] = {
            "frozen_bundle_context": deepcopy(hidden), "assumption_summary": list(adapter.ASSUMPTIONS),
            "evidence_limitations": list(adapter.LIMITATIONS), "required_semantic_contract_hashes": hashes,
            "statistical_evidence_scope": "Frozen statistical intervals, eligibility summaries and any observations already present in the source bundle. Historical scenario files referenced by path/hash are not expanded here; their hashes do not establish semantic assumptions. Unsupported entity/source hypotheses require rejection or abstention.",
            "response_instructions": "Alongside the usual verifier fields, return statistical_evidence_decision as accept_original_valid, accept_original_invalid, reject or ambiguous, and accepted_semantic_contract_hashes as a list. Accepting a conclusion explicitly affirms BOTH supplied target-attribute source/entity contracts. If those hypotheses are unsupported or ambiguous, reject/abstain and return an empty list. This is independent semantic review, not approval from model weight alone. selected_value:null abstains from NEW statistical repair; it does not invalidate source prior repairs. Only explicit accepted original-valid evidence can remove a prior repair.",
            "scope": "Standalone verified statistical feedback; never standard closed-loop inference",
        }
        normalized.append(cell)
    payload = {"dataset": table.name, "dirty_sha256": table.fingerprint, "model": "gpt-6.1-sol", "reasoning_effort": "xhigh",
               "experiment": VERSION, "source_experiment": version, "standard_closed_loop_compatible": False,
               "config": {"schema": "VerifiedStatisticalFeedbackPolicy, not LoopConfig", **asdict(POLICY)},
               "rules": rules, "rule_validation": validation, "delta_dimensions": bundle.get("delta_dimensions", []),
               "cells": normalized, "protocol_sha256": PROTOCOL_SHA256, "frozen_source_bundle": deepcopy(bundle),
               "source_identity": {"bundle_hash": digest(bundle), "prior_prediction_hash": digest(prediction_body(prior)),
                                   "frozen_v3_prediction_hash": digest(prediction_body(frozen_v3)), "artifact_hash": digest(artifact)}}
    payload["request_hash"] = digest(payload)
    return payload


def _complete(table, payload, verification):
    verified = validate_verification(table, payload, verification)
    if len(verified) != len(payload["cells"]):
        raise ValueError("Independent verification must cover every requested cell")
    for source in payload["cells"]:
        item = verified[source["row"], source["column_index"]]
        for field in ("original_valid_score", "selected_value", "confidence", "explanation"):
            if field not in item or (field != "selected_value" and item[field] is None):
                raise ValueError("Incomplete independent verifier fields")
        if not isinstance(item["explanation"], str):
            raise ValueError("Verifier explanation must be text")
        if {c["value"] for c in item.get("candidates", [])} != {c["value"] for c in source["candidates"]}:
            raise ValueError("Independent verification must score every exact candidate")
        decision = item.get("statistical_evidence_decision", "ambiguous")
        if decision not in {"accept_original_valid", "accept_original_invalid", "reject", "ambiguous"}:
            raise ValueError("Unknown statistical evidence decision")
        hashes = item.get("accepted_semantic_contract_hashes", [])
        if not isinstance(hashes, list) or not all(isinstance(h, str) for h in hashes) or len(set(hashes)) != len(hashes):
            raise ValueError("Malformed accepted semantic-contract hashes")
    return verified


def _matrix(rules, rows):
    return np.asarray(evaluate_rules(rules, rows), dtype=np.int8).reshape(len(rows), len(rules))


def apply_verified_feedback(table, prior, frozen_v3, artifact, payload, verification):
    """Consume complete independent judgments with source-prior safe fallback."""
    _fingerprint(table, prior)
    _fingerprint(table, frozen_v3)
    if (payload.get("request_hash") != digest({k: v for k, v in payload.items() if k != "request_hash"})
            or payload.get("protocol_sha256") != PROTOCOL_SHA256 or sha256(PROTOCOL) != PROTOCOL_SHA256
            or payload.get("config") != {"schema": "VerifiedStatisticalFeedbackPolicy, not LoopConfig", **asdict(POLICY)}
            or payload.get("dataset") != table.name or payload.get("dirty_sha256") != table.fingerprint):
        raise ValueError("Verified feedback request/protocol mismatch")
    expected = {"prior_prediction_hash": digest(prediction_body(prior)), "frozen_v3_prediction_hash": digest(prediction_body(frozen_v3)), "artifact_hash": digest(artifact)}
    if any(payload["source_identity"].get(k) != value for k, value in expected.items()):
        raise ValueError("Verified feedback source prediction/artifact mismatch")
    # A newly rehashed conclusion or candidate list must still be the exact
    # deterministic projection of its frozen source, rather than a new policy.
    rebuilt = prepare_payload(table, prior, frozen_v3, artifact, payload["frozen_source_bundle"])
    projected = {k: v for k, v in payload.items() if k not in {"request_hash", "file_preparation"}}
    if projected != {k: v for k, v in rebuilt.items() if k != "request_hash"}:
        raise ValueError("Verified request differs from its frozen source projection")
    if any(prior.probabilities[key] < .5 for key in prior.repairs):
        raise ValueError("Prior EC outside detection mask")
    verified = _complete(table, payload, verification)
    probabilities, repairs = prior.probabilities.copy(), dict(prior.repairs)
    traces = []
    for cell in payload["cells"]:
        key = cell["row"], cell["column_index"]
        judgment = verified[key]
        decision = judgment.get("statistical_evidence_decision", "ambiguous")
        conclusion = cell["statistical_conclusion"]
        required = cell["feedback_verification_context"]["required_semantic_contract_hashes"]
        hashes = judgment.get("accepted_semantic_contract_hashes", [])
        blockers = []
        if set(hashes) != set(required): blockers.append("semantic_contracts_not_explicitly_supported")
        if judgment["confidence"] < .8: blockers.append("verifier_confidence_below_threshold")
        direction = "valid" if decision == "accept_original_valid" else "invalid" if decision == "accept_original_invalid" else None
        if direction is None: blockers.append("rejected_or_ambiguous_statistical_evidence")
        if direction != conclusion["original"]: blockers.append("verifier_frozen_direction_disagreement")
        if direction == "valid" and judgment["original_valid_score"] <= .5 or direction == "invalid" and judgment["original_valid_score"] >= .5:
            blockers.append("verifier_original_score_not_directional")
        accepted = not blockers
        action = "retain_source_prior"
        if accepted:
            probabilities[key] = _probability(conclusion["error_weight_endpoint"])
            if direction == "valid":
                repairs.pop(key, None)
                action = "explicit_original_valid_remove_repair"
            else:
                eligible, selected = frozen_v3.repairs.get(key), judgment["selected_value"]
                scores = {c["value"]: c["score"] for c in judgment["candidates"]}
                if selected is not None and selected == eligible and scores.get(selected, -1) >= .8:
                    repairs[key] = selected
                    action = "accept_exact_frozen_v3_repair" if selected != prior.repairs.get(key) else "retain_verified_prior_repair"
                else:
                    action = "dirty_with_new_repair_abstention_preserve_prior"
        traces.append({"row": key[0], "column_index": key[1], "column": cell["column"],
                       "prior_probability": float(prior.probabilities[key]), "prior_repair": prior.repairs.get(key),
                       "statistical_conclusion": conclusion, "verifier_decision": decision,
                       "ed_accepted": accepted, "ed_blockers": blockers, "intended_action": action,
                       "intended_probability": float(probabilities[key]), "intended_repair": repairs.get(key),
                       "joint_rollback": False, "coherence_ed_rollback": False})
    rules = payload["rules"]
    multi = np.array([r["kind"] == "multi" for r in rules], dtype=bool)
    changed_rows = sorted({i for i, j in set(repairs) | set(prior.repairs) if repairs.get((i, j)) != prior.repairs.get((i, j))})
    before_rows, proposed_rows = _rows(table, prior.repairs), _rows(table, repairs)
    before = _matrix(rules, [before_rows[i] for i in changed_rows])
    proposed = _matrix(rules, [proposed_rows[i] for i in changed_rows])
    guards, rolled_back = [], set()
    for q, i in enumerate(changed_rows):
        violation = (before[q] == 0) & (proposed[q] == 1) & multi
        runtime = (before[q] != -1) & (proposed[q] == -1)
        rollback = bool(violation.any() or runtime.any())
        if rollback:
            rolled_back.add(i)
            for key in set(repairs) | set(prior.repairs):
                if key[0] == i:
                    if key in prior.repairs: repairs[key] = prior.repairs[key]
                    else: repairs.pop(key, None)
        guards.append({"row": i, "prior_tuple": before_rows[i], "proposed_tuple": proposed_rows[i],
                       "new_multi_rule_ids": [r["id"] for r, bit in zip(rules, violation) if bit],
                       "new_runtime_rule_ids": [r["id"] for r, bit in zip(rules, runtime) if bit], "rollback": rollback})
    for trace in traces:
        key = trace["row"], trace["column_index"]
        trace["joint_rollback"] = key[0] in rolled_back
        if key in repairs and probabilities[key] < .5:
            probabilities[key] = prior.probabilities[key]
            trace["coherence_ed_rollback"] = True
        trace.update(final_probability=float(probabilities[key]), final_repair=repairs.get(key),
                     repair_origin=("source_prior_fallback" if repairs.get(key) == prior.repairs.get(key) and key in repairs else
                                    "independently_accepted_frozen_v3_statistical_change" if key in repairs else "none"))
    final_rows = _rows(table, repairs)
    final = _matrix(rules, [final_rows[i] for i in changed_rows])
    if (((before == 0) & (final == 1) & multi).any() or ((before != -1) & (final == -1)).any()
            or any(probabilities[key] < .5 for key in repairs)):
        raise AssertionError("Verified feedback rollback/coherence invariant failed")
    prediction = Prediction(probabilities, repairs, {
        "system": VERSION, "dirty_sha256": table.fingerprint, "fully_verified_main_method": False,
        "standalone_verified_statistical_feedback": True, "source_experiment": payload["source_experiment"],
        "request_hash": payload["request_hash"], "verification_hash": digest(verification),
        "protocol_sha256": PROTOCOL_SHA256, "policy": asdict(POLICY), "new_model_calls_in_processor": 0,
        "ground_truth_access": False, "refit_performed": False,
        "probability_meaning": "Prior probabilities or independently accepted conditional statistical endpoints; not calibrated correctness",
        "new_statistical_ec_bounded_by_frozen_v3": True, "prior_fallback_may_extend_beyond_v3_map": True,
    }).validate(table.shape)
    return prediction, {"experiment": VERSION, "dirty_sha256": table.fingerprint, "trace": traces,
                        "joint_guard": guards, "accepted_rules": rules, "rule_validation": payload["rule_validation"],
                        "guard_reference": "exact source-prior repaired tuples", "ground_truth_access": False}


def implementation_files():
    return sorted(set([Path(__file__).resolve(), PROTOCOL.resolve(), Path(adapter.__file__).resolve(),
                       PROJECT_ROOT / "scripts/run_verified_statistical_feedback.py", *adapter.implementation_files(),
                       PROJECT_ROOT / "experimental/latent_consensus_detection_feedback_v4.py",
                       PROJECT_ROOT / "experimental/latent_consensus_detection_feedback_v4_protocol.md",
                       PROJECT_ROOT / "zeroclean/sandbox.py",
                       PROJECT_ROOT / "tests/test_verified_statistical_feedback.py"]), key=str)


def prepare_from_files(table, bundle_path, manifest_path, prior_path, v3_path, artifact_path, output_path=None):
    paths = {name: Path(path).resolve() for name, path in (
        ("bundle", bundle_path), ("freeze_manifest", manifest_path), ("prior", prior_path),
        ("frozen_v3", v3_path), ("artifact", artifact_path))}
    hashes = {str(p): sha256(p) for p in paths.values()}
    implementation = {str(p): sha256(p) for p in implementation_files()}
    watched = {**hashes, **implementation}
    bundle = json.loads(paths["bundle"].read_text())
    manifest = json.loads(paths["freeze_manifest"].read_text())
    prior, frozen_v3 = Prediction.load(paths["prior"]), Prediction.load(paths["frozen_v3"])
    artifact = json.loads(paths["artifact"].read_text())
    adapter._check_hashes(watched)
    adapter._manifest_binding(manifest, paths["freeze_manifest"], paths["bundle"], hashes[str(paths["bundle"])])
    historical = bundle.get("input_hashes", {})
    if not isinstance(historical, dict) or hashes[str(paths["prior"])] not in historical.values():
        raise ValueError("Explicit source prior is not bound by frozen bundle provenance")
    if frozen_v3.metadata.get("parent_prediction_sha256") != hashes[str(paths["prior"])]:
        raise ValueError("Explicit prior differs from frozen-v3's recorded parent prediction")
    if bundle.get("experiment") == V3:
        if bundle.get("parent_variant_prediction_sha256") != hashes[str(paths["frozen_v3"])]:
            raise ValueError("Explicit frozen-v3 prediction differs from source bundle parent")
    elif hashes[str(paths["frozen_v3"])] not in historical.values():
        raise ValueError("V4 provenance does not bind the explicit frozen-v3 correction map")
    artifact_history = {**historical, **frozen_v3.metadata.get("input_hashes", {})}
    if hashes[str(paths["artifact"])] not in artifact_history.values():
        raise ValueError("Explicit rule artifact is not bound by original frozen-v3 provenance")
    payload = prepare_payload(table, prior, frozen_v3, artifact, bundle)
    payload["file_preparation"] = {"version": VERSION, "explicit_inputs": {k: str(v) for k, v in paths.items()},
                                   "input_file_hashes": hashes, "implementation_hashes": implementation,
                                   "source_signature": digest(implementation), "protocol_sha256": PROTOCOL_SHA256}
    payload["request_hash"] = digest({k: v for k, v in payload.items() if k != "request_hash"})
    adapter._check_hashes(watched)
    if output_path is not None:
        destination = Path(output_path).resolve()
        if destination in paths.values() or destination.is_relative_to(paths["bundle"].parent):
            raise ValueError("Prepared request must be outside the frozen source directory")
        adapter._publish_once(destination, payload)
        adapter._check_hashes(watched)
    return payload


def _copy_once(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != blob:
                raise ValueError("Refusing to overwrite differing immutable processor output")
    finally:
        temporary.unlink(missing_ok=True)


def verify_output(table, output_root):
    output = Path(output_root).resolve()
    frozen = json.loads((output / "frozen" / f"{table.name}.json").read_text())
    if (frozen.get("export_hash") != digest({k: v for k, v in frozen.items() if k != "export_hash"})
            or frozen.get("dirty_sha256") != table.fingerprint or frozen.get("export_schema") != VERSION):
        raise ValueError("Verified statistical feedback output freeze mismatch")
    adapter._check_hashes(frozen["inference_inputs"])
    adapter._check_hashes({str(output / "predictions" / table.name / name): value
                          for name, value in frozen["prediction_files"].items()})
    adapter._check_hashes({str(output / name): value for name, value in frozen["output_files"].items()})
    return frozen


def apply_from_files(table, request_path, verification_path, output_root):
    request_path, verification_path, output = (Path(p).resolve() for p in (request_path, verification_path, output_root))
    inputs = {str(p): sha256(p) for p in (request_path, verification_path)}
    payload, verification = (json.loads(p.read_text()) for p in (request_path, verification_path))
    preparation = payload.get("file_preparation", {})
    adapter._check_hashes({**inputs, **preparation.get("input_file_hashes", {}), **preparation.get("implementation_hashes", {})})
    paths = preparation["explicit_inputs"]
    if any(output == Path(paths[name]).parent or output.is_relative_to(Path(paths[name]).parent)
           for name in ("bundle", "prior", "frozen_v3", "artifact")):
        raise ValueError("Processor output must be separate from frozen/source input directories")
    rebuilt = prepare_from_files(table, paths["bundle"], paths["freeze_manifest"], paths["prior"], paths["frozen_v3"], paths["artifact"])
    if payload != rebuilt:
        raise ValueError("Prepared verifier request differs from exact frozen source projection")
    _complete(table, payload, verification)
    inputs.update(preparation["input_file_hashes"])
    inputs.update(preparation["implementation_hashes"])
    identity = {"version": VERSION, "request_sha256": inputs[str(request_path)],
                "verification_sha256": inputs[str(verification_path)], "source_signature": preparation["source_signature"],
                "protocol_sha256": PROTOCOL_SHA256, "source_experiment": payload["source_experiment"]}
    freeze_path = output / "frozen" / f"{table.name}.json"
    if freeze_path.exists():
        existing = verify_output(table, output)
        if existing["identity"] != identity:
            raise ValueError("Refusing to overwrite a different completed verified-feedback experiment")
        return existing
    prior, frozen_v3 = Prediction.load(paths["prior"]), Prediction.load(paths["frozen_v3"])
    artifact = json.loads(Path(paths["artifact"]).read_text())
    prediction, audit = apply_verified_feedback(table, prior, frozen_v3, artifact, payload, verification)
    adapter._check_hashes(inputs)
    suffix = "v3" if payload["source_experiment"] == V3 else "v4"
    variant = f"verified_statistical_feedback_{suffix}.json"
    blobs = {f"predictions/{table.name}/{variant}": (json.dumps(prediction_body(prediction), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(),
             f"audit/{table.name}.json": (json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()}
    copies = {}
    for number, (filename, wanted) in enumerate(sorted(inputs.items())):
        blob = Path(filename).read_bytes()
        if hashlib.sha256(blob).hexdigest() != wanted:
            raise ValueError("Processor input changed during immutable copying")
        name = f"inputs/{table.name}/{number:03d}_{Path(filename).name}"
        blobs[name] = blob
        copies[name] = wanted
    hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}
    frozen = {"export_schema": VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
              "identity": identity, "source_signature": preparation["source_signature"], "config": asdict(POLICY),
              "prediction_files": {variant: hashes[f"predictions/{table.name}/{variant}"]},
              "inference_inputs": inputs, "output_files": hashes, "frozen_input_copies": copies,
              "frozen_before_evaluation": True, "ground_truth_access": False, "refit_performed": False,
              "statistical_feedback_scope": "Standalone independently verified frozen-bundle coordinates; source-prior fallback",
              "new_statistical_repairs_only_from_v3": True, "final_map_literal_subset_of_v3_claimed": False,
              "baseline_files_included": False, "standard_closed_loop_compatible": False}
    frozen["export_hash"] = digest(frozen)
    adapter._check_hashes(inputs)
    for name, blob in blobs.items():
        _copy_once(output / name, blob)
    adapter._check_hashes(inputs)
    adapter._publish_once(freeze_path, frozen)
    return verify_output(table, output)
