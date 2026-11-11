#!/usr/bin/env python3
"""Export a completed closed loop and qualified source baselines without evaluation.

Implementation lives here so adding an export command does not change the source
signature of in-flight inference runs. Export never resumes incomplete inference.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeroclean import closed_loop
from zeroclean.data import load_dirty, sha256
from zeroclean.types import Prediction, digest, write_json


EXPORT_VERSION = "closed-loop-source-export-v1"
BASELINES = ("zeroed.json", "zeroed_zeroec_reconstruction.json")


def validate_completed_run(table, run_directory):
    """Verify the recorded chain and final state without executing inference."""
    directory = Path(run_directory).resolve()
    frozen_path = directory / "frozen.json"
    if not frozen_path.is_file():
        raise ValueError("A pending closed-loop run cannot be exported")
    manifest = json.loads((directory / "manifest.json").read_text())
    signature = manifest["identity"]["source_signature"]
    # The completed branch only verifies hashes. A missing freeze was rejected
    # above, so this cannot infer, query a model, or create a new round.
    status = closed_loop.advance_closed_loop(table, directory, source_signature=signature)
    manifest, seed_inputs = closed_loop._read_manifest(table, directory, signature)
    identity = manifest["identity"]
    frozen = json.loads(frozen_path.read_text())
    count = frozen.get("external_rounds")
    if type(count) is not int or not 1 <= count <= identity["max_rounds"]:
        raise ValueError("Invalid completed closed-loop round count")
    artifact = json.loads((directory / manifest["snapshots"]["artifact"]["file"]).read_text())
    expected_inputs = dict(seed_inputs)
    history = []
    states = []
    for number in range(1, count + 1):
        round_directory = directory / "rounds" / f"{number:04d}"
        request_path, verification_path, prediction_path, record_path = (
            round_directory / name for name in ("request.json", "verification.json", "prediction.json", "frozen.json")
        )
        payload, response, record = (json.loads(path.read_text()) for path in (request_path, verification_path, record_path))
        closed_loop._validate_payload(table, payload)
        closed_loop._validate_artifact(table, payload, artifact)
        closed_loop._complete_verification(table, payload, response)
        if closed_loop._set_hash(payload) != identity["initial_candidate_set_hash"]:
            raise ValueError("Exported closed-loop candidate set changed")
        if record.get("record_hash") != digest({key: value for key, value in record.items() if key != "record_hash"}):
            raise ValueError("Exported closed-loop round record hash mismatch")
        inputs = dict(seed_inputs, **{str(request_path): sha256(request_path), str(verification_path): sha256(verification_path)})
        if history:
            parent_path = directory / "rounds" / f"{number - 1:04d}" / "frozen.json"
            inputs[str(parent_path)] = sha256(parent_path)
            parent = payload.get("closed_loop", {})
            if (parent.get("external_round") != number
                    or parent.get("parent_request_hash") != history[-1]["request_hash"]
                    or parent.get("parent_prediction_sha256") != history[-1]["prediction_sha256"]
                    or parent.get("parent_round_record_sha256") != sha256(parent_path)):
                raise ValueError("Exported closed-loop parent chain mismatch")
        if (record.get("external_round") != number or record.get("request_hash") != payload["request_hash"]
                or record.get("inference_inputs") != inputs or record.get("prediction_sha256") != sha256(prediction_path)):
            raise ValueError("Exported closed-loop round inputs/prediction mismatch")
        state = Prediction.load(prediction_path).validate(table.shape)
        if state.metadata.get("dirty_sha256") != table.fingerprint or state.metadata.get("request_hash") != payload["request_hash"]:
            raise ValueError("Exported closed-loop round prediction provenance mismatch")
        states.append(state)
        expected_inputs.update(inputs)
        expected_inputs.update({str(prediction_path): sha256(prediction_path), str(record_path): sha256(record_path)})
        history.append({"external_round": number, "request_hash": payload["request_hash"],
                        "verification_sha256": sha256(verification_path), "prediction_sha256": sha256(prediction_path),
                        "round_record_sha256": sha256(record_path)})
    if frozen.get("inference_inputs") != expected_inputs:
        raise ValueError("Completed closed-loop freeze does not bind its complete round chain")
    prediction_path = directory / "prediction.json"
    final = Prediction.load(prediction_path).validate(table.shape)
    provenance = final.metadata.get("closed_loop", {})
    if (final.metadata.get("dirty_sha256") != table.fingerprint or final.metadata.get("external_verification_rounds") != count
            or provenance.get("protocol_hash") != manifest["protocol_hash"] or provenance.get("history") != history
            or final.repairs != states[-1].repairs or not np.array_equal(final.probabilities, states[-1].probabilities)):
        raise ValueError("Final closed-loop prediction does not match its completed chain")
    stable = len(states) > 1 and states[-1].repairs == states[-2].repairs and np.array_equal(states[-1].probabilities, states[-2].probabilities)
    reason = "unchanged_repairs_and_probabilities" if stable else "max_rounds"
    if frozen.get("stop_reason") != reason or (reason == "max_rounds" and count != identity["max_rounds"]):
        raise ValueError("Completed closed-loop stop condition is invalid")
    expected_inputs.update({str(frozen_path): sha256(frozen_path), str(prediction_path): sha256(prediction_path)})
    return {"status": status, "manifest": manifest, "frozen": frozen, "prediction": final,
            "prediction_path": prediction_path, "inputs": expected_inputs, "history": history}


def _baseline_fidelity(prediction):
    fields = ("name", "baseline", "fidelity", "reproduced_upstream", "strict_zero_shot", "uses_ground_truth",
              "prompt_profile", "prompt_provenance", "adaptation_warnings", "source_sha256", "model_sha256",
              "null_sentinel_policy")
    return {field: prediction.metadata[field] for field in fields if field in prediction.metadata}


def verify_exported_experiment(table, output_root):
    """Validate a prior export's source and copied hashes without evaluation."""
    output = Path(output_root).resolve()
    path = output / "frozen" / f"{table.name}.json"
    frozen = json.loads(path.read_text())
    if frozen.get("export_schema") != EXPORT_VERSION or frozen.get("export_hash") != digest({
            key: value for key, value in frozen.items() if key != "export_hash"}):
        raise ValueError("Export freeze content hash mismatch")
    if frozen.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Export dirty input fingerprint mismatch")
    closed_loop._check_hashes(frozen["inference_inputs"])
    closed_loop._check_hashes({str(output / "predictions" / table.name / name): expected
                               for name, expected in frozen["prediction_files"].items()})
    return frozen


def export_closed_loop(table, run_directory, baseline_root, output_root):
    """Copy exactly the final method and two source baselines into a separate root."""
    run = Path(run_directory).resolve()
    baselines = Path(baseline_root).resolve()
    output = Path(output_root).resolve()
    if output == baselines.parent or output == baselines or baselines in output.parents or output == run or run in output.parents:
        raise ValueError("Export output must be separate from source predictions and the closed-loop run")
    validated = validate_completed_run(table, run)
    inputs = dict(validated["inputs"])
    sources = {"zeroclean_loop.json": validated["prediction_path"]}
    fidelity = {}
    for name in BASELINES:
        path = baselines / table.name / name
        if not path.is_file():
            raise ValueError(f"Required source baseline is missing: {path}")
        expected = sha256(path)
        prediction = Prediction.load(path).validate(table.shape)
        fingerprints = [prediction.metadata[key] for key in ("input_fingerprint", "dirty_sha256", "dirty_fingerprint")
                        if key in prediction.metadata]
        if not fingerprints or any(value != table.fingerprint for value in fingerprints):
            raise ValueError(f"Source baseline input fingerprint mismatch: {path}")
        if name == "zeroed.json" and prediction.metadata.get("reproduced_upstream") is not True:
            raise ValueError("zeroed.json must contain reproduced original-source detector predictions")
        if name == "zeroed_zeroec_reconstruction.json" and prediction.metadata.get("baseline") != "zeroed_zeroec_reconstruction":
            raise ValueError("Qualified ZeroEC baseline identity mismatch")
        if name == "zeroed_zeroec_reconstruction.json" and prediction.metadata.get("strict_zero_shot") is not True:
            raise ValueError("Source correction baseline must use zero fixed examples")
        if prediction.metadata.get("pending_cells", 0):
            raise ValueError("A pending source correction baseline cannot be exported")
        sources[name] = path
        inputs[str(path)] = expected
        fidelity[name] = _baseline_fidelity(prediction)
    blobs = {name: path.read_bytes() for name, path in sources.items()}
    hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}
    for name, path in sources.items():
        if hashes[name] != inputs[str(path)]:
            raise ValueError(f"Source input changed while reading export: {path}")
    closed_loop._check_hashes(inputs)
    final = validated["prediction"]
    manifest = validated["manifest"]
    ec_profile = fidelity["zeroed_zeroec_reconstruction.json"].get("prompt_profile", "")
    ec_caveat = ("Author-recovered ZeroEC system prompt; historical identity unproven; zero fixed examples."
                 if ec_profile.startswith("author_zerodc") else
                 "ZeroEC prompt reconstruction; historical identity to pinned upstream unproven; zero fixed examples.")
    frozen = {
        "export_schema": EXPORT_VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
        "source_signature": manifest["identity"]["source_signature"], "config": final.metadata["config"],
        "prediction_files": hashes, "inference_inputs": inputs,
        "closed_loop": {"run_directory": str(run), "protocol_hash": manifest["protocol_hash"],
                        "external_rounds": validated["frozen"]["external_rounds"],
                        "stop_reason": validated["frozen"]["stop_reason"], "history": validated["history"]},
        "baseline_fidelity": fidelity,
        "qualified_ec_caveat": ec_caveat,
        "official_exact_ec_baseline_present": False, "matched_ablations_included": False,
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
                raise ValueError(f"Refusing to overwrite a different exported prediction: {path}")
        if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
            raise ValueError("Refusing to overwrite a different exported experiment freeze")
        closed_loop._check_hashes(inputs)
        for name, blob in blobs.items():
            path = destination / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".json.tmp")
                temporary.write_bytes(blob)
                temporary.replace(path)
        closed_loop._check_hashes(inputs)
        if not freeze_path.exists():
            write_json(freeze_path, frozen)
    return verify_exported_experiment(table, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=Path("results/predictions"))
    parser.add_argument("--output", type=Path, default=Path("results/source_v1"))
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    table = load_dirty(manifest["identity"]["dataset"])
    frozen = export_closed_loop(table, args.run_dir, args.baseline_root, args.output)
    print(json.dumps({"dataset": table.name, "output": str(args.output), "export_hash": frozen["export_hash"],
                      "prediction_files": list(frozen["prediction_files"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
