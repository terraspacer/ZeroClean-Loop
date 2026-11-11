#!/usr/bin/env python3
"""Copy both preregistered v5 statistical decisions and frozen comparisons."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_statistical_feedback import collect_bound_files, publish_exact
from zeroclean.data import load_dirty, sha256
from zeroclean.types import Prediction, digest


def main():
    table = load_dirty("flights")
    output = Path("results/latent_consensus_model_average_v5")
    inputs = {str(Path(__file__).resolve()): sha256(__file__)}
    final_path = Path("experimental/latent_consensus_model_average_v5/final_manifest.json")
    collect_bound_files(final_path, "a8f82621832cc8d8b43cda5eba71db21e0f937ca8c80ae962c1d04501afd0bf8", inputs)
    previous = Path("results/latent_consensus_statistical")
    previous_freeze_path = previous / "frozen/flights.json"
    previous_freeze = json.loads(previous_freeze_path.read_text())
    inputs[str(previous_freeze_path.resolve())] = sha256(previous_freeze_path)
    for path, expected in previous_freeze["inference_inputs"].items():
        collect_bound_files(path, expected, inputs, follow=False)
    sources = {}
    for name, expected in previous_freeze["prediction_files"].items():
        path = previous / "predictions/flights" / name
        collect_bound_files(path, expected, inputs, follow=False)
        sources[name] = path
    for variant in ("latent_consensus_model_average_v5", "latent_consensus_expected_f1_proxy_v5"):
        path = Path("experimental") / variant / "prediction.json"
        validation_path = path.parent / "validation.json"
        validation = json.loads(validation_path.read_text())
        if validation["status"] != "passed" or validation["llm_verification_executed"] is not False:
            raise ValueError("Expected structurally validated unverified statistical outputs")
        if validation["prediction_sha256"] != inputs[str(path.resolve())]:
            raise ValueError("Validation does not bind the frozen prediction")
        sources[f"{variant}.json"] = path
    blobs = {}
    for name, path in sources.items():
        p = Prediction.load(path).validate(table.shape)
        identities = [p.metadata[k] for k in ("dirty_sha256", "dirty_fingerprint", "input_fingerprint") if k in p.metadata]
        if not identities or any(v != table.fingerprint for v in identities):
            raise ValueError("Study prediction dirty identity mismatch")
        if any(v != table.rows[i][j] and p.probabilities[i, j] < .5 for (i, j), v in p.repairs.items()):
            raise ValueError("Study repair outside detection mask")
        blobs[name] = path.read_bytes()
        if hashlib.sha256(blobs[name]).hexdigest() != inputs[str(path.resolve())]:
            raise ValueError("Study prediction changed during read")
    frozen = {"export_schema": "model-average-statistical-study-v5", "dataset": table.name,
        "dirty_sha256": table.fingerprint, "config": {"threshold": .5}, "inference_inputs": inputs,
        "prediction_files": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
        "frozen_before_evaluation": True, "ground_truth_access": False,
        "independent_verification_status": "not_run", "fully_verified_main_method": False,
        "scope": "Flights only; both predefined policies, no outcome-based selection; v3 correction map fixed",
        "conditional_weight_caveat": "Equal model weights are not calibrated correctness probabilities; the expected-F1 optimizer is exact only for its additional independent-marginals proxy and ranked decision family on eligible cells, not the shared-entity categorical joint model or full benchmark.",
        "qualified_ec_caveat": previous_freeze["qualified_ec_caveat"]}
    frozen["export_hash"] = digest(frozen)
    for path, expected in inputs.items():
        if sha256(path) != expected:
            raise ValueError(f"Study input changed during export: {path}")
    for name, blob in blobs.items():
        publish_exact(output / "predictions/flights" / name, blob)
    publish_exact(output / "frozen/flights.json", (json.dumps(frozen, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
    print(json.dumps({"output": str(output), "export_hash": frozen["export_hash"], "variants": list(sources)}), flush=True)


if __name__ == "__main__":
    main()
