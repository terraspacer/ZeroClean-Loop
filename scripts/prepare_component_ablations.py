#!/usr/bin/env python3
"""Independent component ablations of the first externally verified pass.

The frozen reference is rounds/0001/prediction.json, never the final iterative
prediction. Each variant retains one external verification pass and the final
tuple guard. This does not implement an iterative no-counterfactual experiment.
"""
from dataclasses import asdict, replace
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_closed_loop as full_export
from scripts import prepare_experiment as preparer
from zeroclean import ablations, closed_loop
from zeroclean.data import load_dirty, sha256
from zeroclean.loop import LoopConfig, infer_from_candidates
from zeroclean.types import Prediction, digest


VERSION = "first-verified-pass-components-v1"
REFERENCE = "full_first_verified_round.json"
VARIANTS = {
    "without_counterfactual_first_pass": ablations.DEFINITIONS["without_counterfactual"],
    "without_multi_column_first_pass": ablations.DEFINITIONS["without_multi_column"],
}


def _bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _write_once(path, blob):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(blob); handle.flush(); os.fsync(handle.fileno())
        try: os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != blob: raise ValueError(f"Refusing to overwrite different component artifact: {path}")
    finally: temporary.unlink(missing_ok=True)


def _implementation():
    root = preparer.PROJECT_ROOT
    paths = [*preparer.implementation_files(), Path(__file__).resolve(), Path(full_export.__file__).resolve(),
             root / "zeroclean" / "ablations.py", root / "zeroclean" / "closed_loop.py",
             root / "zeroclean" / "verification.py"]
    return {str(path.resolve()): sha256(path) for path in paths}


def _match_explicit_inputs(table, initial, detection_path, proposals_path):
    detection = Prediction.load(detection_path).validate(table.shape)
    metadata = detection.metadata
    if metadata.get("input_fingerprint") != table.fingerprint or metadata.get("reproduced_upstream") is not True:
        raise ValueError("Component ablations require the matching original-source detector")
    provenance = initial.get("detector_provenance", {})
    if provenance.get("kind") != "source": raise ValueError("Full initial request must declare a source detector")
    candidates = {
        "prediction_file_sha256": sha256(detection_path),
        "prediction_hash": digest({"probabilities": detection.probabilities.tolist(), "metadata": metadata}),
        "probabilities_hash": digest(detection.probabilities.tolist()), "metadata_hash": digest(metadata),
    }
    if not ("prediction_file_sha256" in provenance or "prediction_hash" in provenance
            or {"probabilities_hash", "metadata_hash"} <= provenance.keys()):
        raise ValueError("Full initial request lacks sufficient source detector provenance")
    for key, expected in candidates.items():
        if key in provenance and provenance[key] != expected:
            raise ValueError("Explicit source detector does not match full initial provenance")
    proposals = json.loads(Path(proposals_path).read_text())
    if initial.get("proposal_provenance", {}).get("artifact_hash") != digest(proposals):
        raise ValueError("Explicit proposals do not match full initial provenance")
    preparation = initial.get("experiment_preparation", {})
    for role, explicit in (("detection", detection_path), ("proposals", proposals_path)):
        original_path = preparation.get("explicit_inputs", {}).get(role)
        expected = preparation.get("input_file_hashes", {}).get(original_path)
        if expected is not None and expected != sha256(explicit):
            raise ValueError(f"Explicit {role} bytes do not match frozen preparation provenance")
    if "source_detection_sha256" in proposals and proposals["source_detection_sha256"] != sha256(detection_path):
        raise ValueError("Proposal detector bytes do not match explicit source detector")
    return detection


def prepare_components(table, run_directory, detection_path, proposals_path, output_root):
    run, detection_path, proposals_path = (Path(path).resolve() for path in (run_directory, detection_path, proposals_path))
    validated = full_export.validate_completed_run(table, run)
    initial_path = run / "rounds" / "0001" / "request.json"
    verifier_path = run / "rounds" / "0001" / "verification.json"
    reference_path = run / "rounds" / "0001" / "prediction.json"
    initial = json.loads(initial_path.read_text())
    source_config = LoopConfig(**initial["config"])
    if not (source_config.counterfactual and source_config.multi_column and source_config.iterative_verification):
        raise ValueError("Component reference must be the enabled full first verified pass")
    _match_explicit_inputs(table, initial, detection_path, proposals_path)
    artifact_path = run / validated["manifest"]["snapshots"]["artifact"]["file"]
    artifact = json.loads(artifact_path.read_text())
    if digest(artifact) != initial["artifact_hash"]: raise ValueError("Frozen component artifact mismatch")
    inputs = dict(validated["inputs"])
    inputs.update({str(path): sha256(path) for path in (detection_path, proposals_path)})
    implementation = _implementation()
    inputs.update(implementation)
    config = replace(source_config, rounds=1, iterative_verification=True)
    no_cf = ablations.redact_counterfactual_evidence(initial)
    # Original preparation metadata can contain a nested full-model config.
    # Its hashes remain in the component manifest; do not expose that config
    # inside the redacted request alongside the variant's actual configuration.
    no_cf.pop("experiment_preparation", None)
    no_cf["config"] = asdict(replace(config, counterfactual=False))
    # The explicit preparer stages arbitrary proposal locations and reuses the
    # actual core builder; no second implementation of candidate generation.
    with tempfile.TemporaryDirectory(prefix="zeroclean-components-") as temporary:
        no_mc = preparer.prepare_experiment(table, artifact_path, detection_path, proposals_path,
                                             Path(temporary) / "no_multi.json", config=replace(config, multi_column=False))
    if no_mc["rules"] != [rule for rule in initial["rules"] if rule.get("kind") == "single"]:
        raise ValueError("Single-column validation differed from the frozen full reference")
    payloads = {"without_counterfactual_first_pass": no_cf, "without_multi_column_first_pass": no_mc}
    for name, payload in payloads.items():
        payload["component_ablation"] = {
            "version": VERSION, "variant": name, "scope": "components_of_first_verified_pass",
            "definition": VARIANTS[name], "reference_prediction_sha256": sha256(reference_path),
            "initial_verifier_reused": False, "independent_verification_required": True,
            "iterative_no_component_experiment_claimed": False,
        }
        payload["request_hash"] = digest({key: value for key, value in payload.items() if key != "request_hash"})
    preparer._check_hashes(inputs)
    identity = {
        "version": VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
        "full_run_directory": str(run), "full_protocol_hash": validated["manifest"]["protocol_hash"],
        "full_source_signature": validated["manifest"]["identity"]["source_signature"],
        "source_implementation_signature": digest(implementation), "input_file_hashes": inputs,
        "input_match_policy": "Require recorded source byte hashes or combined semantic identity; proposals must match the initial semantic artifact digest and original byte hash when recorded",
        "initial_request_hash": initial["request_hash"], "initial_request_sha256": sha256(initial_path),
        "initial_verification_sha256": sha256(verifier_path), "full_first_prediction_sha256": sha256(reference_path),
        "reference_source": str(reference_path), "artifact_path": str(artifact_path),
        "detection_path": str(detection_path), "proposals_path": str(proposals_path),
        "variants": {name: {"request_file": f"requests/{name}.json", "request_hash": payload["request_hash"],
                            "request_sha256": hashlib.sha256(_bytes(payload)).hexdigest(),
                            "verification_file": f"verifications/{name}.json", "config": payload["config"],
                            "definition": VARIANTS[name]} for name, payload in payloads.items()},
    }
    manifest = {"identity": identity, "protocol_hash": digest(identity)}
    manifest["manifest_hash"] = digest(manifest)
    destination = Path(output_root).resolve() / table.name / manifest["protocol_hash"]
    if destination == run or run in destination.parents:
        raise ValueError("Prepared component artifacts must stay outside the frozen full run")
    for name, payload in payloads.items(): _write_once(destination / identity["variants"][name]["request_file"], _bytes(payload))
    _write_once(destination / "manifest.json", _bytes(manifest))
    preparer._check_hashes(inputs)
    return destination


def run_components(table, prepared_directory, output_root):
    directory, output = Path(prepared_directory).resolve(), Path(output_root).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = manifest["identity"]
    if (manifest.get("protocol_hash") != digest(identity) or directory.name != manifest["protocol_hash"]
            or manifest.get("manifest_hash") != digest({key: value for key, value in manifest.items() if key != "manifest_hash"})):
        raise ValueError("Component preparation content hash mismatch")
    if identity.get("version") != VERSION or identity.get("dataset") != table.name or identity.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Component protocol or dirty input mismatch")
    if set(identity["variants"]) != set(VARIANTS): raise ValueError("Incomplete component variant set")
    inputs = dict(identity["input_file_hashes"])
    preparer._check_hashes(inputs)
    full_export.validate_completed_run(table, identity["full_run_directory"])
    if output == Path(identity["full_run_directory"]) or Path(identity["full_run_directory"]) in output.parents:
        raise ValueError("Component results must stay outside the frozen full run")
    inputs[str(manifest_path)] = sha256(manifest_path)
    freeze_path = output / "frozen" / f"{table.name}.json"
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text())
        if frozen.get("protocol_hash") != manifest["protocol_hash"]: raise ValueError("Different component experiment already occupies output")
        if frozen.get("freeze_hash") != digest({key: value for key, value in frozen.items() if key != "freeze_hash"}):
            raise ValueError("Component freeze hash mismatch")
        preparer._check_hashes(frozen["inference_inputs"])
        preparer._check_hashes({str(output / "predictions" / table.name / name): expected for name, expected in frozen["prediction_files"].items()})
        return frozen
    loaded = {}
    for name, variant in identity["variants"].items():
        request_path, response_path = directory / variant["request_file"], directory / variant["verification_file"]
        if not response_path.is_file(): raise ValueError(f"Independent component verification required: {response_path}")
        if sha256(request_path) != variant["request_sha256"]: raise ValueError("Component request changed after preparation")
        payload, response = (json.loads(path.read_text()) for path in (request_path, response_path))
        if payload.get("request_hash") != variant["request_hash"] or payload["request_hash"] != digest({key: value for key, value in payload.items() if key != "request_hash"}):
            raise ValueError("Component request content hash mismatch")
        closed_loop._complete_verification(table, payload, response)
        inputs.update({str(path): sha256(path) for path in (request_path, response_path)})
        loaded[name] = payload, response, LoopConfig(**variant["config"])
    predictions = {}
    for name, (payload, response, config) in loaded.items():
        prediction = infer_from_candidates(table, payload, response, config)
        prediction.metadata["component_ablation"] = {**payload["component_ablation"], "protocol_hash": manifest["protocol_hash"],
            "verification_mode": "independent_first_pass_variant", "verification_sha256": inputs[str(directory / identity["variants"][name]["verification_file"])]}
        predictions[f"{name}.json"] = _bytes(closed_loop._prediction_body(prediction))
    predictions[REFERENCE] = Path(identity["reference_source"]).read_bytes()
    if hashlib.sha256(predictions[REFERENCE]).hexdigest() != identity["full_first_prediction_sha256"]:
        raise ValueError("Full first-pass reference changed")
    hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in predictions.items()}
    frozen = {
        "dataset": table.name, "dirty_sha256": table.fingerprint, "protocol_hash": manifest["protocol_hash"],
        "source_signature": identity["source_implementation_signature"], "full_source_signature": identity["full_source_signature"],
        "prediction_files": hashes, "inference_inputs": inputs,
        "component_ablation_protocol": {"version": VERSION, "scope": "components_of_first_verified_pass",
            "full_reference": REFERENCE, "full_initial_request_hash": identity["initial_request_hash"],
            "full_initial_verification_sha256": identity["initial_verification_sha256"],
            "full_reference_prediction_sha256": identity["full_first_prediction_sha256"],
            "independent_variant_verification": True, "inference_passes": 1, "final_tuple_guard_retained": True,
            "iterative_no_component_experiment_claimed": False, "definitions": VARIANTS},
        "frozen_before_evaluation": True, "ground_truth_access": False,
    }
    frozen["freeze_hash"] = digest(frozen)
    preparer._check_hashes(inputs)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".component.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, blob in predictions.items():
            path = output / "predictions" / table.name / name
            if path.exists() and path.read_bytes() != blob: raise ValueError("Different component prediction already occupies output")
        if freeze_path.exists() and freeze_path.read_bytes() != _bytes(frozen): raise ValueError("Different component freeze already occupies output")
        for name, blob in predictions.items(): _write_once(output / "predictions" / table.name / name, blob)
        preparer._check_hashes(inputs)
        _write_once(freeze_path, _bytes(frozen))
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--detection", type=Path)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        if any(path is None for path in (args.run_dir, args.detection, args.proposals)): parser.error("prepare requires --run-dir --detection --proposals")
        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        table = load_dirty(manifest["identity"]["dataset"])
        directory = prepare_components(table, args.run_dir, args.detection, args.proposals, args.output)
        print(json.dumps({"prepared_directory": str(directory)}, indent=2), flush=True)
    else:
        if args.prepared_dir is None: parser.error("run requires --prepared-dir")
        manifest = json.loads((args.prepared_dir / "manifest.json").read_text())
        table = load_dirty(manifest["identity"]["dataset"])
        frozen = run_components(table, args.prepared_dir, args.output)
        print(json.dumps({"dataset": table.name, "protocol_hash": frozen["protocol_hash"], "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__": main()
