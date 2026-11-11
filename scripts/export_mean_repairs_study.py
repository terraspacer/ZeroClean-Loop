#!/usr/bin/env python3
"""Immutable Flights v6 inference export; never evaluates or reads references.

The new v6 manifest and prior inference-only export are pinned byte identities.
Existing recursive binding and immutable-publish helpers are imported unchanged.
New tests belong in their own module; prior frozen sources/tests are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_statistical_feedback as binding
from zeroclean.data import load_dirty, sha256
from zeroclean.types import Prediction, digest


V6 = Path("experimental/latent_consensus_mean_repairs_v6")
V6_MANIFEST_SHA256 = "56889e90d3d9d92f84bd5562ab776f0869e9b001dd5d13c4049834335881282c"
PREVIOUS = Path("results/latent_consensus_model_average_v5")
PREVIOUS_FREEZE_SHA256 = "f6169c987f6bf1f27f09288309fdfdb519e4c26fef55a53b2e0108eebabb3ec1"
TEST_SOURCE = Path("tests/test_export_mean_repairs_study.py")
COMPARISONS = (
    "latent_consensus_model_average_v5.json",
    "latent_consensus_feedback_v3.json",
    "latent_consensus_detection_feedback_v4.json",
    "source_v2_prior.json",
    "zeroed.json",
    "zeroed_zeroec_reconstruction.json",
)
V6_FILENAME = "latent_consensus_mean_repairs_v6.json"


def comparison_paths(root, frozen):
    if any(name not in frozen["prediction_files"] for name in COMPARISONS):
        raise ValueError("Pinned prior export lacks a predefined comparison")
    return {name: Path(root) / "predictions/flights" / name for name in COMPARISONS}


def bind_flat_prior_inputs(frozen, inputs, seen):
    """Check the pinned prior's expanded closure without re-rooting snapshots.

    Recursive live-manifest validation starts from the pinned v6 root. A prior
    exporter already flattened its original and frozen-copy graph into this
    map. Reopening a copied artifact_manifest as a new owner would incorrectly
    resolve its historical relative paths against the copy's directory.
    """
    for name, expected in frozen["inference_inputs"].items():
        binding.collect_bound_files(name, expected, inputs, seen, follow=False)


def prediction_blob(path, table, inputs):
    path = Path(path)
    prediction = Prediction.load(path).validate(table.shape)
    identities = [prediction.metadata[key] for key in ("dirty_sha256", "input_fingerprint", "dirty_fingerprint")
                  if key in prediction.metadata]
    if not identities or any(value != table.fingerprint for value in identities):
        raise ValueError("Export prediction dirty identity mismatch")
    if any(value != table.rows[i][j] and prediction.probabilities[i, j] < .5
           for (i, j), value in prediction.repairs.items()):
        raise ValueError("Frozen changed-value repair outside detection mask")
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != inputs[str(path.resolve())]:
        raise ValueError("Prediction changed while reading")
    return blob, prediction


def export(output=Path("results/latent_consensus_mean_repairs_v6")):
    output = Path(output)
    table = load_dirty("flights")
    sources = [Path(__file__).resolve(), Path(binding.__file__).resolve(), TEST_SOURCE.resolve()]
    inputs = {str(path): sha256(path) for path in sources}
    seen = set()
    binding.collect_bound_files(V6 / "final_manifest.json", V6_MANIFEST_SHA256, inputs, seen)
    previous_path = PREVIOUS / "frozen/flights.json"
    binding.collect_bound_files(previous_path, PREVIOUS_FREEZE_SHA256, inputs, seen)
    previous = json.loads(previous_path.read_text())
    if previous.get("export_hash") != digest({k: v for k, v in previous.items() if k != "export_hash"}):
        raise ValueError("Pinned prior export content hash mismatch")
    if previous.get("dataset") != "flights" or previous.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Pinned prior export dirty identity mismatch")
    bind_flat_prior_inputs(previous, inputs, seen)
    paths = comparison_paths(PREVIOUS, previous)
    for name, path in paths.items():
        binding.collect_bound_files(path, previous["prediction_files"][name], inputs, seen, follow=False)
    paths[V6_FILENAME] = V6 / "prediction.json"
    validation = json.loads((V6 / "validation.json").read_text())
    if validation.get("status") != "passed" or validation.get("accuracy_evaluator_executed") is not False or validation.get("llm_verification_executed") is not False:
        raise ValueError("Expected structurally validated, independently unverified v6 output")
    if validation.get("prediction_sha256") != inputs[str((V6 / "prediction.json").resolve())]:
        raise ValueError("V6 validation does not bind the pinned prediction")
    protocol = json.loads((V6 / "pre_run_manifest.json").read_text())
    policy = protocol["policy"]
    required_policy = {"ed_threshold": .5, "candidate_weight": .8, "candidate_margin": .2,
        "tie_tolerance": 1e-12, "top_k": 3, "selected_models": 17, "repair_confidence": .8,
        "delta_weights": [2.2, 1.8, .65, .35], "logit_epsilon": 1e-6, "runtime_error_rate": .05}
    if any(policy.get(key) != value for key, value in required_policy.items()):
        raise ValueError("Pinned v6 repair policy differs from the fixed protocol")
    coverage = json.loads((V6 / "counts.json").read_text())
    blobs, predictions = {}, {}
    for name, path in paths.items():
        blobs[name], predictions[name] = prediction_blob(path, table, inputs)
    v6 = predictions[V6_FILENAME]
    mean = predictions["latent_consensus_model_average_v5.json"]
    if v6.probabilities.tobytes() != mean.probabilities.tobytes() or coverage.get("fixed_ed_unchanged") is not True:
        raise ValueError("V6 must retain frozen mean-v5 ED bitwise")
    if any(v6.probabilities[key] < .5 for key in v6.repairs):
        raise ValueError("V6 repair outside its unchanged ED mask")
    if not v6.metadata.get("conditional_hypotheses_only") or v6.metadata.get("new_llm_calls") != 0 or v6.metadata.get("refit_performed") is not False:
        raise ValueError("Unexpected v6 method/provenance status")
    frozen = {
        "export_schema": "mean-model-repairs-statistical-study-v6", "dataset": table.name,
        "dirty_sha256": table.fingerprint, "config": {"threshold": .5},
        "inference_inputs": inputs,
        "prediction_files": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
        "pinned_v6_manifest": {"path": str(V6 / "final_manifest.json"), "sha256": V6_MANIFEST_SHA256},
        "pinned_comparison_export": {"path": str(previous_path), "sha256": PREVIOUS_FREEZE_SHA256,
                                     "export_hash": previous["export_hash"]},
        "frozen_before_evaluation": True, "ground_truth_access": False,
        "accuracy_evaluator_executed": False, "independent_verification_status": "not_run",
        "fully_verified_main_method": False, "conditional_hypotheses_only": True,
        "scope": "Flights only; predefined v6 repair-policy ablation with mean-v5 ED fixed, plus frozen v5/v3/v4/prior and source-baseline comparisons",
        "comparison_set_predefined": list(COMPARISONS),
        "ed_policy": {"unchanged_from": "latent_consensus_model_average_v5.json", "bitwise_equal": True,
                      "source_sha256": previous["prediction_files"]["latent_consensus_model_average_v5.json"]},
        "repair_policy": policy,
        "repair_rationale": "Decompose inherited strict-v3 EC from fixed mean-model ED: start from source-v2 prior repairs; admit only unique high-weight shared observed alternatives with representation preservation and individual/joint counterfactual guards; no lower-ranked fallback or outcome-based selection.",
        "repair_coverage": coverage,
        "frozen_semantic_contracts": v6.metadata.get("frozen_semantic_contracts", []),
        "conditional_weight_caveat": "Equal-weight fitted-scenario posteriors and counterfactual scores are conditional, unverified model weights, not calibrated correctness probabilities or exact Bayesian model marginalization. Event/snapshot identity, source independence, stationarity, missingness, finite vocabularies and raw representation semantics remain hypotheses; no new catalog or snapshot truth is asserted.",
        "qualified_ec_caveat": previous["qualified_ec_caveat"],
        "method_status": "Statistical repair ablation only; no independent verifier acceptance, no evaluation, no promotion, and no completed six-dataset target claim.",
    }
    frozen["export_hash"] = digest(frozen)
    for name, expected in inputs.items():
        if sha256(name) != expected:
            raise ValueError(f"Input changed during export: {name}")
    for name, blob in blobs.items():
        binding.publish_exact(output / "predictions/flights" / name, blob)
    binding.publish_exact(output / "frozen/flights.json",
        (json.dumps(frozen, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode())
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/latent_consensus_mean_repairs_v6"))
    args = parser.parse_args()
    frozen = export(args.output)
    print(json.dumps({"output": str(args.output), "export_hash": frozen["export_hash"],
                      "frozen_sha256": sha256(args.output / "frozen/flights.json")}), flush=True)
