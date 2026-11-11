#!/usr/bin/env python3
"""Derive an immutable verification-only request from a frozen feedback bundle.

The generic verifier forwards cells, rules, dimensions and config. This adapter
places the complete otherwise-omitted bundle context inside EVERY cell, so even
one-row queue groups see all frozen semantic contracts. It preserves every
original candidate, tuple and evidence record; it does not apply judgments.

The copied config is FeedbackPolicy, NOT LoopConfig. This output must not be
passed to run_closed_loop or infer_from_candidates. A separate preregistered
processor is required before any verifier judgments can alter a prediction.
No dirty table, posterior fit, reference data or model is loaded here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experimental.latent_consensus_feedback import FeedbackPolicy, VERSION as FEEDBACK_VERSION
from zeroclean.data import sha256
from zeroclean.types import digest


VERSION = "frozen-feedback-verifier-surface-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORWARDED = {"dataset", "dirty_sha256", "rules", "delta_dimensions", "config", "cells"}
ASSUMPTIONS = [
    "The frozen entity key plus attribute denotes one exact-string snapshot event; verify that grain and representation.",
    "Entity events and the selected source groups are conditionally independent under each frozen source assumption.",
    "Source log-advantage is stationary and missingness is ignorable under the frozen model.",
    "The finite observed candidate vocabulary, unseen category and disclosed Gaussian orientation priors are appropriate; orientation and identifiability remain assumptions.",
]
LIMITATIONS = [
    "Conditional model weights and their error endpoints are not calibrated correctness probabilities or labels.",
    "Dates, time zones, field meanings, source copying, alternative representations and missing sentinels may remain unresolved; infer from supplied evidence and abstain when ambiguity remains.",
    "Consensus, dirty-derived counterfactual dimensions and generated rules may share systematic errors.",
    "Prior and tentative tuples contain untrusted model proposals; the tuple field alone preserves the original observation.",
    "A veto or rollback can retain an exact prior repair while statistical ED feedback changes independently.",
]


def implementation_files():
    return [Path(__file__).resolve(), *[
        PROJECT_ROOT / name for name in (
            "experimental/latent_consensus_feedback.py", "experimental/latent_consensus_feedback_protocol.md",
            "experimental/latent_consensus.py", "experimental/latent_consensus_v2.py",
            "zeroclean/verification.py", "zeroclean/loop.py", "zeroclean/types.py", "zeroclean/data.py")]]


def _check_hashes(expected):
    for filename, wanted in expected.items():
        if not Path(filename).is_file() or sha256(filename) != wanted:
            raise ValueError(f"Feedback verification input or implementation changed: {filename}")


def _manifest_binding(manifest, manifest_path, bundle_path, bundle_sha):
    entries = manifest.get("artifacts", manifest.get("outputs"))
    if not isinstance(entries, dict):
        raise ValueError("Freeze manifest must contain artifact/output file hashes")
    matches = []
    for filename, wanted in entries.items():
        if not isinstance(filename, str):
            raise ValueError("Invalid frozen artifact filename")
        path = Path(filename)
        candidates = {path.resolve(), (manifest_path.parent / path).resolve()}
        if bundle_path in candidates:
            matches.append(wanted)
    if matches != [bundle_sha]:
        raise ValueError("Freeze manifest does not bind this exact bundle file SHA")


def _validate_bundle(bundle):
    if not isinstance(bundle, dict) or bundle.get("request_hash") != digest({k: v for k, v in bundle.items() if k != "request_hash"}):
        raise ValueError("Corrupted frozen feedback bundle request hash")
    if bundle.get("experiment") != FEEDBACK_VERSION or bundle.get("independent_verification_status") != "not_run":
        raise ValueError("Expected the unverified frozen statistical feedback bundle")
    if bundle.get("model") != "gpt-6.1-sol" or bundle.get("reasoning_effort") != "xhigh":
        raise ValueError("Frozen feedback model/effort mismatch")
    if not all(isinstance(bundle.get(k), str) and bundle[k] for k in ("dataset", "dirty_sha256")):
        raise ValueError("Missing feedback dataset/fingerprint")
    policy = bundle.get("config")
    if not isinstance(policy, dict) or set(policy) != {f.name for f in fields(FeedbackPolicy)}:
        raise ValueError("Bundle config must be the exact FeedbackPolicy schema, not LoopConfig")
    # Validate serialization including non-finite values, without reinterpreting
    # frozen weights or changing any original gate/policy.
    digest(bundle)
    contracts = bundle.get("frozen_semantic_contracts")
    if not isinstance(contracts, list) or not contracts:
        raise ValueError("Frozen semantic contracts are required; none may be silently inferred")
    modes = {}
    for contract in contracts:
        if (not isinstance(contract, dict) or not isinstance(contract.get("attribute"), str)
                or contract.get("source_assumption") not in {"independent_names", "cluster_near_agreement"}
                or not isinstance(contract.get("contract"), dict) or not contract["contract"]):
            raise ValueError("Invalid frozen semantic contract")
        key = contract["attribute"], contract["source_assumption"]
        if key in modes:
            raise ValueError("Duplicate frozen semantic contract")
        modes[key] = contract
    if not isinstance(bundle.get("rules"), list) or not isinstance(bundle.get("delta_dimensions"), list):
        raise ValueError("Frozen rules/delta dimensions must be explicit")
    cells = bundle.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Frozen feedback cells must be a list")
    seen = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Frozen feedback cell must be an object")
        i, j = cell.get("row"), cell.get("column_index")
        if type(i) is not int or type(j) is not int or i < 0 or j < 0 or (i, j) in seen:
            raise ValueError("Invalid/duplicate frozen feedback coordinate")
        seen.add((i, j))
        column, original, row = cell.get("column"), cell.get("original"), cell.get("tuple")
        if (not isinstance(row, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in row.items())
                or not isinstance(column, str) or not isinstance(original, str)
                or j >= len(row) or list(row)[j] != column or row[column] != original):
            raise ValueError("Frozen feedback tuple/coordinate/original mismatch")
        for mode in ("independent_names", "cluster_near_agreement"):
            if (column, mode) not in modes:
                raise ValueError("Every feedback attribute requires both frozen source contracts")
        for field in ("prior_inference_tuple", "tentative_tuple"):
            context = cell.get(field)
            if not isinstance(context, dict) or set(context) != set(row) or not all(isinstance(v, str) for v in context.values()):
                raise ValueError("Missing/malformed frozen model-proposed tuple")
        evidence = cell.get("evidence")
        if not isinstance(evidence, dict) or any(evidence.get(k) != cell[k] for k in ("row", "column", "column_index", "original")):
            raise ValueError("Frozen cell evidence coordinate mismatch")
        if any(evidence.get(source) != cell[target] for source, target in (
                ("original_tuple", "tuple"), ("prior_tuple", "prior_inference_tuple"), ("final_tuple", "tentative_tuple"))):
            raise ValueError("Frozen cell tuple disagrees with its evidence record")
        if not isinstance(evidence.get("candidate"), str):
            raise ValueError("Missing exact statistical candidate in frozen evidence")
        candidates = cell.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Frozen feedback candidates must be a list")
        values = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("Frozen feedback candidate must be an object")
            value, score = candidate.get("value"), candidate.get("score")
            if not isinstance(value, str) or value == original or value in values:
                raise ValueError("Invalid/duplicate frozen feedback candidate")
            if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("Invalid frozen candidate weight")
            values.add(value)
        expected_values = []
        for value in (evidence["candidate"], evidence.get("prior_repair")):
            if value is not None and value != original and value not in expected_values:
                expected_values.append(value)
        if [candidate["value"] for candidate in candidates] != expected_values:
            raise ValueError("Frozen alternatives disagree with statistical/prior evidence")
        if "feedback_verification_context" in cell:
            raise ValueError("Refusing to replace a preexisting feedback verification context")


def _publish_once(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != payload:
                raise ValueError("Refusing to overwrite a different immutable feedback verification request")
    finally:
        temporary.unlink(missing_ok=True)


def prepare_feedback_verification(bundle_path, freeze_manifest_path, output_path):
    bundle_path, freeze_manifest_path, output_path = (
        Path(p).resolve() for p in (bundle_path, freeze_manifest_path, output_path))
    if output_path in (bundle_path, freeze_manifest_path) or output_path.is_relative_to(bundle_path.parent):
        raise ValueError("Derived request must be outside the frozen feedback directory")
    source_hashes = {str(p): sha256(p) for p in (bundle_path, freeze_manifest_path)}
    implementations = {str(p): sha256(p) for p in implementation_files()}
    watched = {**source_hashes, **implementations}
    bundle = json.loads(bundle_path.read_text())
    manifest = json.loads(freeze_manifest_path.read_text())
    _check_hashes(watched)
    _manifest_binding(manifest, freeze_manifest_path, bundle_path, source_hashes[str(bundle_path)])
    _validate_bundle(bundle)
    payload = deepcopy(bundle)
    # Forward all otherwise-hidden fields, including future semantic additions.
    hidden_context = {k: deepcopy(v) for k, v in bundle.items() if k not in FORWARDED and k != "request_hash"}
    identity = {"version": VERSION, "source_bundle": str(bundle_path),
                "source_bundle_sha256": source_hashes[str(bundle_path)],
                "source_bundle_request_hash": bundle["request_hash"],
                "freeze_manifest": str(freeze_manifest_path), "input_file_hashes": source_hashes,
                "implementation_hashes": implementations, "implementation_signature": digest(implementations),
                "scope": "independent_feedback_verification_only", "standard_closed_loop_compatible": False,
                "config_schema": "FeedbackPolicy, not LoopConfig", "source_run_validation_reperformed": False,
                "historical_inputs_revalidated": False, "inference_or_fitting_performed": False,
                "original_cells_hash": digest(bundle["cells"])}
    for target, original in zip(payload["cells"], bundle["cells"]):
        trace = original["evidence"]
        target["feedback_verification_context"] = {
            "frozen_bundle_context": deepcopy(hidden_context), "feedback_policy": deepcopy(bundle["config"]),
            "assumption_summary": list(ASSUMPTIONS), "evidence_limitations": list(LIMITATIONS),
            "verification_scope": "Review every exact candidate and original independently; this request cannot apply judgments or start the standard closed loop.",
            "source_bundle_sha256": source_hashes[str(bundle_path)],
            "source_bundle_request_hash": bundle["request_hash"], "original_cell_hash": digest(original),
            "evidence_summary": {k: deepcopy(trace[k]) for k in (
                "candidate", "prior_probability", "feedback_probability", "original_weight_interval",
                "candidate_weight_lower", "delta_features", "delta_dimensions", "correction_confidence",
                "individual_gate_blockers", "joint_rollback", "prior_repair", "requested_repair", "final_repair") if k in trace},
        }
    payload["feedback_verification_preparation"] = identity
    payload["standard_closed_loop_compatible"] = False
    payload["purpose"] = "Independent verification preparation only; no prediction changes or standard closed-loop eligibility"
    payload["request_hash"] = digest({k: v for k, v in payload.items() if k != "request_hash"})
    _check_hashes(watched)
    _publish_once(output_path, payload)
    _check_hashes(watched)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True, help="Original final_manifest.json or output_manifest.json binding the bundle SHA")
    parser.add_argument("--output", type=Path, required=True, help="New immutable file outside the frozen source directory")
    args = parser.parse_args()
    payload = prepare_feedback_verification(args.bundle, args.freeze_manifest, args.output)
    print(json.dumps({"output": str(args.output), "request_hash": payload["request_hash"],
                      "cells": len(payload["cells"]), "standard_closed_loop_compatible": False}, indent=2))


if __name__ == "__main__":
    main()
