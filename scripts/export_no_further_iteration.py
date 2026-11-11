#!/usr/bin/env python3
"""Export final versus first externally verified round into a separate experiment.

The ablation retains initial external verification and the first final-tuple
safety check. It omits further verifier feedback and context/delta recomputation;
it does not claim to isolate those two later mechanisms from one another.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_closed_loop as full_export
from zeroclean.data import load_dirty, sha256
from zeroclean.types import Prediction, digest, write_json


EXPORT_VERSION = "closed-loop-no-further-iteration-export-v1"
VARIANT = "without_further_iteration.json"


def verify_export(table, output_root):
    output = Path(output_root).resolve()
    frozen = json.loads((output / "frozen" / f"{table.name}.json").read_text())
    if frozen.get("export_schema") != EXPORT_VERSION or frozen.get("export_hash") != digest({
            key: value for key, value in frozen.items() if key != "export_hash"}):
        raise ValueError("No-further-iteration export freeze hash mismatch")
    if frozen.get("dirty_sha256") != table.fingerprint:
        raise ValueError("No-further-iteration export dirty input mismatch")
    full_export.closed_loop._check_hashes(frozen["inference_inputs"])
    full_export.closed_loop._check_hashes({str(output / "predictions" / table.name / name): expected
                                          for name, expected in frozen["prediction_files"].items()})
    return frozen


def export_no_further_iteration(table, run_directory, baseline_root, output_root):
    run, baselines, output = (Path(path).resolve() for path in (run_directory, baseline_root, output_root))
    if output == baselines.parent or output == baselines or baselines in output.parents or output == run or run in output.parents:
        raise ValueError("Ablation output must be separate from source predictions and the closed-loop run")
    validated = full_export.validate_completed_run(table, run)
    rounds = validated["frozen"]["external_rounds"]
    if rounds < 2:
        raise ValueError("No-further-iteration comparison requires at least two completed external rounds")
    first_path = run / "rounds" / "0001" / "prediction.json"
    first = Prediction.load(first_path).validate(table.shape)
    first_record_path = run / "rounds" / "0001" / "frozen.json"
    first_record = json.loads(first_record_path.read_text())
    if first_record["prediction_sha256"] != sha256(first_path):
        raise ValueError("First external round prediction differs from its frozen state")
    if first.metadata.get("config", {}).get("iterative_verification") is not True or first.metadata.get("rounds_executed") != 1:
        raise ValueError("First external round must retain its initial verifier and one inference pass")
    sources = {"zeroclean_loop.json": validated["prediction_path"], VARIANT: first_path}
    inputs = dict(validated["inputs"])
    inputs[str(Path(full_export.__file__).resolve())] = sha256(full_export.__file__)
    fidelity = {}
    for name in full_export.BASELINES:
        path = baselines / table.name / name
        if not path.is_file(): raise ValueError(f"Required source baseline is missing: {path}")
        expected = sha256(path)
        prediction = Prediction.load(path).validate(table.shape)
        fingerprints = [prediction.metadata[key] for key in ("input_fingerprint", "dirty_sha256", "dirty_fingerprint")
                        if key in prediction.metadata]
        if not fingerprints or any(value != table.fingerprint for value in fingerprints):
            raise ValueError(f"Source baseline fingerprint mismatch: {path}")
        if name == "zeroed.json" and prediction.metadata.get("reproduced_upstream") is not True:
            raise ValueError("Ablation comparison requires original-source ZeroED")
        if name == "zeroed_zeroec_reconstruction.json" and (
                prediction.metadata.get("baseline") != "zeroed_zeroec_reconstruction"
                or prediction.metadata.get("strict_zero_shot") is not True):
            raise ValueError("Ablation correction baseline must be the zero-example qualified reconstruction")
        if prediction.metadata.get("pending_cells", 0): raise ValueError("Pending source baseline cannot be exported")
        sources[name] = path
        inputs[str(path)] = expected
        fidelity[name] = full_export._baseline_fidelity(prediction)
    blobs = {name: path.read_bytes() for name, path in sources.items()}
    hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}
    if any(hashes[name] != inputs[str(path)] for name, path in sources.items()):
        raise ValueError("An ablation source prediction changed while being read")
    full_export.closed_loop._check_hashes(inputs)
    manifest = validated["manifest"]
    first_directory = run / "rounds" / "0001"
    first_request = json.loads((first_directory / "request.json").read_text())
    ec_profile = fidelity["zeroed_zeroec_reconstruction.json"].get("prompt_profile", "")
    frozen = {
        "export_schema": EXPORT_VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
        "source_signature": manifest["identity"]["source_signature"], "config": validated["prediction"].metadata["config"],
        "prediction_files": hashes, "inference_inputs": inputs,
        "closed_loop": {"run_directory": str(run), "protocol_hash": manifest["protocol_hash"],
                        "external_rounds": rounds, "stop_reason": validated["frozen"]["stop_reason"],
                        "history": validated["history"]},
        "ablation_protocol": {
            "variant": VARIANT.removesuffix(".json"), "source_external_round": 1,
            "definition": "Retain initial external verification, one inference pass and its final tuple safety check; omit all further iterative feedback",
            "initial_verification_retained": True, "new_inference_executed_for_export": False,
            "fixed_initial_candidate_set_hash": manifest["identity"]["initial_candidate_set_hash"],
            "shared_initial_request_hash": first_request["request_hash"],
            "shared_initial_request_sha256": sha256(first_directory / "request.json"),
            "shared_initial_verification_sha256": sha256(first_directory / "verification.json"),
            "first_round_record_sha256": sha256(first_record_path),
            "first_round_prediction_sha256": hashes[VARIANT],
            "full_prediction_sha256": hashes["zeroclean_loop.json"],
            "comparison_limit": "Removes later verifier judgments and recomputed tentative-context deltas together; it does not isolate their individual effects",
        },
        "baseline_fidelity": fidelity, "official_exact_ec_baseline_present": False,
        "qualified_ec_caveat": ("Author-recovered ZeroEC system prompt; historical identity unproven; zero fixed examples."
                               if ec_profile.startswith("author_zerodc") else
                               "ZeroEC prompt reconstruction; historical identity to pinned upstream unproven; zero fixed examples."),
        "frozen_before_evaluation": True, "ground_truth_access": False, "exporter_sha256": sha256(__file__),
    }
    frozen["export_hash"] = digest(frozen)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".export.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = output / "predictions" / table.name
        freeze_path = output / "frozen" / f"{table.name}.json"
        for name, expected in hashes.items():
            path = destination / name
            if path.exists() and sha256(path) != expected:
                raise ValueError(f"Refusing to overwrite a differing ablation prediction: {path}")
        if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
            raise ValueError("Refusing to overwrite a differing ablation experiment freeze")
        full_export.closed_loop._check_hashes(inputs)
        for name, blob in blobs.items():
            path = destination / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".json.tmp")
                temporary.write_bytes(blob)
                temporary.replace(path)
        full_export.closed_loop._check_hashes(inputs)
        if not freeze_path.exists(): write_json(freeze_path, frozen)
    return verify_export(table, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=Path("results/predictions"))
    parser.add_argument("--output", type=Path, default=Path("results/source_no_further_iteration_v1"))
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    table = load_dirty(manifest["identity"]["dataset"])
    frozen = export_no_further_iteration(table, args.run_dir, args.baseline_root, args.output)
    print(json.dumps({"dataset": table.name, "output": str(args.output), "export_hash": frozen["export_hash"],
                      "variant": VARIANT.removesuffix(".json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
