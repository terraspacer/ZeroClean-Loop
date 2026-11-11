#!/usr/bin/env python3
"""Prepare an immutable dirty-only experiment from three explicit input files.

No default candidate or rule artifact is read or replaced. The exact output is
ready for the existing verifier queue and closed-loop driver with --artifact.
"""
from dataclasses import asdict, replace
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeroclean.data import DATASETS, load_dirty, sha256
from zeroclean.loop import LoopConfig, build_candidates
from zeroclean.types import Prediction, digest
from zeroclean.upstream_zeroec import _detector_arrays


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_VERSION = "explicit-input-experiment-v1"


def implementation_files():
    return [Path(__file__).resolve(), *[
        PROJECT_ROOT / "zeroclean" / name for name in
        ("loop.py", "statistics.py", "sandbox.py", "types.py", "data.py", "upstream_zeroec.py")]]


def _check_hashes(expected):
    for filename, fingerprint in expected.items():
        if not Path(filename).is_file() or sha256(filename) != fingerprint:
            raise ValueError(f"Experiment input or implementation changed: {filename}")


def _protected_output(path):
    candidate_root = PROJECT_ROOT / "artifacts" / "candidates"
    rule_root = PROJECT_ROOT / "artifacts" / "llm"
    return path == candidate_root or candidate_root in path.parents or (
        path.name == "pass2.json" and rule_root in path.parents)


def _check_existing(path, identity):
    existing = json.loads(path.read_text())
    if (existing.get("experiment_preparation") != identity
            or existing.get("request_hash") != digest({key: value for key, value in existing.items() if key != "request_hash"})):
        raise ValueError("Refusing to overwrite a differing immutable experiment payload")
    return existing


def _publish_once(path, payload, identity):
    """Publish a complete inode exclusively; even a concurrent file is preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = _check_existing(path, identity)
            if existing != payload:
                raise ValueError("Concurrent immutable experiment output differs from the prepared payload")
            return existing
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def prepare_experiment(table, artifact_path, detection_path, proposals_path, output_path, *, config=LoopConfig()):
    artifact_path, detection_path, proposals_path, output_path = (
        Path(path).resolve() for path in (artifact_path, detection_path, proposals_path, output_path))
    explicit_paths = (artifact_path, detection_path, proposals_path)
    if output_path in explicit_paths or _protected_output(output_path):
        raise ValueError("Experiment output must be a new explicit file outside active candidate/pass2 inputs")
    inputs = {str(path): sha256(path) for path in explicit_paths}
    implementation = {str(path.resolve()): sha256(path) for path in implementation_files()}
    artifact = json.loads(artifact_path.read_text())
    proposals = json.loads(proposals_path.read_text())
    for name, body in (("artifact", artifact), ("proposals", proposals)):
        if body.get("dirty_sha256") != table.fingerprint:
            raise ValueError(f"Experiment {name} dirty input fingerprint mismatch")
        if body.get("model") != "gpt-6.1-sol" or body.get("reasoning_effort") != "xhigh":
            raise ValueError(f"Experiment {name} model/effort mismatch")
        if "dataset" in body and body["dataset"] != table.name:
            raise ValueError(f"Experiment {name} dataset mismatch")
    detection = Prediction.load(detection_path).validate(table.shape)
    fingerprints = [detection.metadata[key] for key in ("input_fingerprint", "dirty_sha256", "dirty_fingerprint")
                    if key in detection.metadata]
    if not fingerprints or any(value != table.fingerprint for value in fingerprints):
        raise ValueError("Experiment source detector dirty input fingerprint mismatch")
    if detection.metadata.get("reproduced_upstream") is not True:
        raise ValueError("Experiment detector must be original-source ZeroED")
    # Preserve the source's exact decision/probability contract; no recalibration.
    _detector_arrays(table, detection)
    config = replace(config, proposal_filename=proposals_path.name)
    identity = {
        "version": PREPARATION_VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
        "explicit_inputs": {"artifact": str(artifact_path), "detection": str(detection_path), "proposals": str(proposals_path)},
        "input_file_hashes": inputs, "source_implementation_hashes": implementation,
        "source_implementation_signature": digest(implementation), "config": asdict(config),
        "source_predictions_modified": False, "ground_truth_access": False,
    }
    expected = {**inputs, **implementation}
    _check_hashes(expected)
    if output_path.exists():
        result = _check_existing(output_path, identity)
        _check_hashes(expected)
        return result
    if proposals_path.parent.name == table.name:
        payload = build_candidates(table, artifact, detection, config, artifact_root=proposals_path.parent.parent)
    else:
        # build_candidates uses a dataset directory. Isolate an exact copy when
        # the explicit file lives elsewhere, without changing any active path.
        with tempfile.TemporaryDirectory(prefix="zeroclean-explicit-proposals-") as temporary:
            staged = Path(temporary) / table.name / proposals_path.name
            staged.parent.mkdir()
            shutil.copyfile(proposals_path, staged)
            if sha256(staged) != inputs[str(proposals_path)]:
                raise ValueError("Explicit proposal input changed while staging")
            payload = build_candidates(table, artifact, detection, config, artifact_root=temporary)
            if sha256(staged) != inputs[str(proposals_path)]:
                raise ValueError("Staged proposal input changed during preparation")
    payload["detector_provenance"] = {
        "kind": "source", "prediction_file": str(detection_path), "prediction_file_sha256": inputs[str(detection_path)],
        "prediction_hash": digest({"probabilities": detection.probabilities.tolist(), "metadata": detection.metadata}),
        "probabilities_hash": digest(detection.probabilities.tolist()), "metadata_hash": digest(detection.metadata),
    }
    payload["experiment_preparation"] = identity
    payload["request_hash"] = digest({key: value for key, value in payload.items() if key != "request_hash"})
    _check_hashes(expected)
    result = _publish_once(output_path, payload, identity)
    _check_hashes(expected)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--detection", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = load_dirty(args.dataset)
    payload = prepare_experiment(table, args.artifact, args.detection, args.proposals, args.output)
    print(json.dumps({"dataset": table.name, "request_hash": payload["request_hash"],
                      "candidate_cells": len(payload["cells"]), "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
