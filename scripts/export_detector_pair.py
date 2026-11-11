#!/usr/bin/env python3
"""Freeze the minimal detector fix, its paired control, and original baseline."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_detector_variant import _validate_completion
from scripts.export_statistical_feedback import publish_exact
from zeroclean.data import DATASETS, load_dirty, sha256
from zeroclean.types import Prediction, digest


def main():
    manifest_path = Path("experimental/zero_predicate_variant/completion_manifest.json").resolve()
    expected_manifest = "e8299edfb26dc650044f2b60e9284747d4179dbab8c0fc6e59534c5fc2be7e59"
    if sha256(manifest_path) != expected_manifest:
        raise ValueError("Paired detector completion manifest changed")
    manifest = json.loads(manifest_path.read_text())
    output = Path("results/empty_predicate_detector_v1")
    for name in DATASETS:
        table = load_dirty(name)
        entry = manifest["datasets"][name]
        variant_path = Path(entry["variant_prediction_path"])
        variant = Prediction.load(variant_path)
        _, inputs, _ = _validate_completion(table, manifest_path, variant_path, variant)
        inputs[str(manifest_path)] = expected_manifest
        inputs[str(Path(__file__).resolve())] = sha256(__file__)
        inputs[str(Path("scripts/prepare_detector_variant.py").resolve())] = sha256("scripts/prepare_detector_variant.py")
        baseline_root = Path("results/source_baselines_v1")
        baseline_freeze_path = baseline_root / "frozen" / f"{name}.json"
        baseline_freeze = json.loads(baseline_freeze_path.read_text())
        for path, expected in baseline_freeze["inference_inputs"].items():
            inputs[str(Path(path).resolve())] = expected
        inputs[str(baseline_freeze_path.resolve())] = sha256(baseline_freeze_path)
        original_path = baseline_root / "predictions" / name / "zeroed.json"
        inputs[str(original_path.resolve())] = baseline_freeze["prediction_files"]["zeroed.json"]
        sources = {"zeroed_empty_predicate_variant.json": variant_path,
                   "fresh_source_control.json": Path(entry["control_prediction_path"]), "zeroed.json": original_path}
        blobs = {}
        for filename, path in sources.items():
            prediction = Prediction.load(path).validate(table.shape)
            if prediction.repairs:
                raise ValueError("Detector-only experiment contains correction output")
            blobs[filename] = path.read_bytes()
            if hashlib.sha256(blobs[filename]).hexdigest() != inputs[str(path.resolve())]:
                raise ValueError("Detector prediction changed during export")
        frozen = {"export_schema": "empty-predicate-paired-detector-v1", "dataset": name,
            "dirty_sha256": table.fingerprint, "config": {"threshold": .5}, "inference_inputs": inputs,
            "prediction_files": {filename: hashlib.sha256(blob).hexdigest() for filename, blob in blobs.items()},
            "frozen_before_evaluation": True, "ground_truth_access": False,
            "scope": "Standalone detection only; no cleaning-loop or repair claim",
            "comparison_target": manifest["comparison_target"], "limitations": manifest["limitations"],
            "threshold_semantics": "All predictions retain original strict p>.5 stored classifier decisions"}
        frozen["export_hash"] = digest(frozen)
        for path, expected in inputs.items():
            if sha256(path) != expected:
                raise ValueError(f"Paired detector input changed: {path}")
        for filename, blob in blobs.items():
            publish_exact(output / "predictions" / name / filename, blob)
        publish_exact(output / "frozen" / f"{name}.json", (json.dumps(frozen, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
        print(json.dumps({"dataset": name, "export_hash": frozen["export_hash"]}), flush=True)


if __name__ == "__main__":
    main()
