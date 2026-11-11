#!/usr/bin/env python3
"""Freeze the two preregistered Flights statistical ablations for evaluation.

Copies exact predictions; never fits, verifies with an LLM, or reads references.
These variants are not the fully verified six-dataset method.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeroclean.data import load_dirty, sha256
from zeroclean.types import Prediction, digest


FROZEN_VARIANTS = {
    "latent_consensus_feedback_v3": "cf48dc9a361626927e755c5934c243f8455af8cc47765c1561111f10793da830",
    "latent_consensus_detection_feedback_v4": "b775b8919d5c18c0bc79e13af0c73d53d60ed3e025731344fc971ffd8bb1c6b0",
}
HASH_MAPS = ("artifacts", "outputs", "input_hashes", "frozen_input_hashes", "inference_inputs")


def collect_bound_files(path, expected, inputs, seen=None, *, follow=True):
    """Follow only explicitly named hash maps, never table or candidate content."""
    seen = set() if seen is None else seen
    path = Path(path).resolve()
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen input hash mismatch: {path}")
    key = str(path)
    if key in inputs and inputs[key] != actual:
        raise ValueError(f"Conflicting frozen identities: {path}")
    inputs[key] = actual
    if not follow:
        return
    if key in seen:
        return
    seen.add(key)
    # The pinned pre-run manifests bind both originals and their frozen copies.
    # Prediction metadata repeats those dependencies using snapshot-local paths;
    # its bytes are checked without reinterpreting that duplicated graph.
    if path.suffix != ".json" or "manifest" not in path.name:
        return
    body = json.loads(path.read_text())
    owners = [body]
    if isinstance(body.get("metadata"), dict):
        owners.append(body["metadata"])
    for owner in owners:
        for field in HASH_MAPS:
            entries = owner.get(field, {})
            if not isinstance(entries, dict):
                raise ValueError(f"Invalid provenance map {field}: {path}")
            for filename, fingerprint in entries.items():
                if not isinstance(filename, str) or not isinstance(fingerprint, str) or len(fingerprint) != 64:
                    raise ValueError(f"Invalid bound file identity: {path}")
                child = Path(filename)
                if not child.is_absolute() and not child.exists():
                    child = path.parent / child
                # Snapshot manifests retain their original relative references;
                # the original input manifests above bind that dependency graph.
                # Verify every snapshot byte without re-rooting its graph.
                collect_bound_files(child, fingerprint, inputs, seen,
                                    follow=field != "frozen_input_hashes")
        if "validator_path" in owner:
            collect_bound_files(owner["validator_path"], owner["validator_sha256"], inputs, seen)


def publish_exact(path, blob):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".statistical-export-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != blob:
                raise ValueError(f"Refusing to overwrite differing frozen artifact: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def export(output):
    table = load_dirty("flights")
    inputs = {str(Path(__file__).resolve()): sha256(__file__)}
    sources = {}
    for variant, expected in FROZEN_VARIANTS.items():
        directory = Path("experimental") / variant
        collect_bound_files(directory / "final_manifest.json", expected, inputs)
        validation = json.loads((directory / "validation.json").read_text())
        if validation.get("status") != "passed" or validation.get("llm_verification_executed") is not False:
            raise ValueError("Expected structurally validated, unverified statistical ablation")
        sources[f"{variant}.json"] = directory / "prediction.json"
    baseline_root = Path("results/source_baselines_v1")
    baseline_freeze = baseline_root / "frozen/flights.json"
    # The baseline freeze predates these ablations and is independently measured.
    baseline = json.loads(baseline_freeze.read_text())
    inputs[str(baseline_freeze.resolve())] = sha256(baseline_freeze)
    for name, expected in baseline["inference_inputs"].items():
        collect_bound_files(name, expected, inputs)
    for name in ("zeroed.json", "zeroed_zeroec_reconstruction.json"):
        path = baseline_root / "predictions/flights" / name
        collect_bound_files(path, baseline["prediction_files"][name], inputs)
        sources[name] = path
    prior = Path("experimental/latent_consensus_feedback_v3/frozen_inputs/prior_prediction.json")
    sources["source_v2_prior.json"] = prior
    if str(prior.resolve()) not in inputs:
        raise ValueError("Exact prior prediction was not bound by the frozen experiment")
    blobs = {}
    for name, path in sources.items():
        prediction = Prediction.load(path).validate(table.shape)
        metadata = prediction.metadata
        identifiers = [metadata[k] for k in ("dirty_sha256", "input_fingerprint", "dirty_fingerprint") if k in metadata]
        if not identifiers or any(value != table.fingerprint for value in identifiers):
            raise ValueError("Prediction dirty identity mismatch")
        if any(value != table.rows[i][j] and prediction.probabilities[i, j] < .5
               for (i, j), value in prediction.repairs.items()):
            raise ValueError("Frozen repair outside detection mask")
        blobs[name] = path.read_bytes()
        if hashlib.sha256(blobs[name]).hexdigest() != inputs[str(path.resolve())]:
            raise ValueError("Prediction changed while reading")
    frozen = {
        "export_schema": "frozen-statistical-feedback-v1", "dataset": table.name,
        "dirty_sha256": table.fingerprint, "config": {"threshold": .5},
        "prediction_files": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
        "inference_inputs": inputs, "frozen_before_evaluation": True, "ground_truth_access": False,
        "independent_verification_status": "not_run", "fully_verified_main_method": False,
        "scope": "Flights only; both preregistered statistical variants, without outcome-based selection",
        "qualified_ec_caveat": "Author-recovered ZeroEC system prompt; historical identity unproven; zero fixed examples.",
        "known_frozen_code_limitations": [
            "Synthetic prior-repair rollback corner cases raise instead of safe rollback; actual frozen outputs passed tuple guards.",
            "Original validator does not reject all internally rehashed inconsistent bundles; this export pins original manifest bytes.",
            "Conditional posterior weights depend on unresolved entity, source independence and representation assumptions.",
        ],
    }
    frozen["export_hash"] = digest(frozen)
    for filename, expected in inputs.items():
        if sha256(filename) != expected:
            raise ValueError(f"Input changed during export: {filename}")
    for name, blob in blobs.items():
        publish_exact(output / "predictions/flights" / name, blob)
    publish_exact(output / "frozen/flights.json", (json.dumps(frozen, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/latent_consensus_statistical"))
    args = parser.parse_args()
    result = export(args.output)
    print(json.dumps({"output": str(args.output), "export_hash": result["export_hash"],
                      "prediction_files": list(result["prediction_files"])}), flush=True)
