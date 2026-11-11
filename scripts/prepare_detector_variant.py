#!/usr/bin/env python3
"""Prepare standard-loop candidates from a completed experimental detector.

This does not run the detector, retrieve peers, queue models or evaluate. The
empty-predicate classifier remains an unpromoted variant, never an upstream
reproduction. Existing CTSv2 proposals retain their original-source retrieval
identity while build_candidates uses the explicit NEW classifier for eligibility.
"""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import prepare_experiment as original
from zeroclean.data import DATASETS, load_dirty, sha256
from zeroclean.loop import LoopConfig, build_candidates
from zeroclean.types import Prediction, digest
from zeroclean.upstream_zeroec import _detector_arrays


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = "completed-empty-predicate-detector-candidates-v1"
EXPERIMENT = "zeroed-empty-predicate-preserves-pseudoclean-v1"
CTS_POLICY = "degenerate_retrieval_mask_conditional_profiles_v2"
MODEL, EFFORT = "gpt-6.1-sol", "xhigh"
VALIDATION_CHECKS = {
    "full_prefilter_state_identical", "unaffected_columns_exact_reproduction", "recorded_replies_identical",
    "original_clustering_cache_unchanged", "frozen_scores_match_prediction"}
COMPARISON_TARGET = "fresh paired source-semantics control; not previously frozen source baseline"
THREADS = {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"}


def implementation_files():
    return [Path(__file__).resolve(), *original.implementation_files(),
            PROJECT_ROOT / "experimental/zero_predicate_variant.py",
            PROJECT_ROOT / "experimental/freeze_zero_predicate_variant.py",
            PROJECT_ROOT / "scripts/queue_cts_v2.py"]


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _absolute(path):
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("Frozen manifest paths must be absolute")
    return Path(path).resolve()


def _hash_map(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("Frozen provenance requires an explicit file hash mapping")
    result = {}
    for filename, fingerprint in value.items():
        if not _hash(fingerprint): raise ValueError("Invalid frozen file SHA")
        path = str(_absolute(filename))
        if path in result and result[path] != fingerprint:
            raise ValueError("Conflicting frozen path aliases")
        result[path] = fingerprint
    return result


def _fingerprint(table, prediction):
    prediction.validate(table.shape)
    values = [prediction.metadata[k] for k in ("input_fingerprint", "dirty_sha256", "dirty_fingerprint") if k in prediction.metadata]
    if not values or any(v != table.fingerprint for v in values):
        raise ValueError("Detector dirty input fingerprint mismatch")


def _validate_completion(table, manifest_path, detection_path, detection):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_hash") != digest({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("Detector completion manifest content hash mismatch")
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("status") != "complete" or manifest.get("experiment_id") != EXPERIMENT
            or manifest.get("experimental_unpromoted") is not True or manifest.get("reproduced_upstream") is not False
            or manifest.get("uses_ground_truth") is not False or manifest.get("strict_zero_shot") is not True
            or manifest.get("model") != MODEL or manifest.get("reasoning_effort") != EFFORT
            or manifest.get("seed") != 42 or manifest.get("python_hash_seed") != "0"
            or manifest.get("threshold") != .5 or manifest.get("threshold_operator") != ">"
            or manifest.get("comparison_target") != COMPARISON_TARGET):
        raise ValueError("Expected the completed validated minimal detector experiment")
    threads = manifest.get("native_threads")
    if not isinstance(threads, dict) or set(threads) != THREADS or set(threads.values()) != {"2"}:
        raise ValueError("Detector native-thread execution contract mismatch")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError("The paired six-dataset detector experiment is not complete")
    for entry in datasets.values():
        checks = entry.get("validation", {}) if isinstance(entry, dict) else {}
        if set(checks) != VALIDATION_CHECKS or any(checks[k] is not True for k in VALIDATION_CHECKS):
            raise ValueError("Incomplete/failed paired detector validation")
    entry = datasets[table.name]
    if entry.get("input_sha256") != table.fingerprint or entry.get("shape") != list(table.shape):
        raise ValueError("Detector completion dirty input/shape mismatch")
    expected = _hash_map(entry.get("source_sha256"))
    for prefix in ("implementation", "protocol", "execution_addendum", "source_runner", "freezer"):
        path, wanted = _absolute(manifest.get(prefix + "_path")), manifest.get(prefix + "_sha256")
        if not _hash(wanted): raise ValueError("Missing frozen detector implementation SHA")
        if str(path) in expected and expected[str(path)] != wanted:
            raise ValueError("Detector implementation hash roles disagree")
        expected[str(path)] = wanted
    known = {"implementation": PROJECT_ROOT / "experimental/zero_predicate_variant.py",
             "source_runner": PROJECT_ROOT / "zeroclean/upstream_zeroed.py",
             "freezer": PROJECT_ROOT / "experimental/freeze_zero_predicate_variant.py"}
    for prefix, path in known.items():
        if sha256(path) != manifest[prefix + "_sha256"]:
            raise ValueError("Frozen detector implementation differs from the explicit minimal variant")
    for prefix in ("variant_prediction", "control_prediction", "comparison"):
        path, wanted = _absolute(entry.get(prefix + "_path")), entry.get(prefix + "_sha256")
        if not _hash(wanted) or str(path) not in expected or expected[str(path)] != wanted:
            raise ValueError("Detector completion role is absent from frozen source hashes")
    if _absolute(entry["variant_prediction_path"]) != detection_path:
        raise ValueError("Explicit detector must be the completion manifest's exact VARIANT file")
    if entry.get("source_runner_sha256") != manifest["source_runner_sha256"]:
        raise ValueError("Dataset/root detector source signatures disagree")
    if expected.get(str(_absolute(entry.get("input_path")))) != table.fingerprint:
        raise ValueError("Original dirty input file is not bound by detector freeze")
    original._check_hashes(expected)
    protocol = json.loads(_absolute(manifest["protocol_path"]).read_text())
    addendum = json.loads(_absolute(manifest["execution_addendum_path"]).read_text())
    if (protocol.get("protocol_id") != EXPERIMENT or addendum.get("addendum_id") != "frozen-prefilter-source-stage-v1"
            or addendum.get("original_protocol_unchanged") is not True):
        raise ValueError("Frozen detector protocol/addendum identity mismatch")
    metadata = detection.metadata
    required = {"name": "zeroed_empty_predicate_variant", "baseline": "experimental_unpromoted", "mode": "variant",
                "fidelity": "minimal_empty_predicate_semantic_variant", "reproduced_upstream": False,
                "empty_predicate_guard_enabled": True, "uses_ground_truth": False, "strict_zero_shot": True,
                "seed": 42, "python_hash_seed": "0", "new_requests_queued": 0,
                "decision_threshold": .5, "decision_threshold_operator": ">"}
    if any(metadata.get(k) != v or (type(v) is bool and type(metadata.get(k)) is not bool) for k, v in required.items()):
        raise ValueError("Explicit detector is not the frozen minimal experimental variant")
    if detection.repairs:
        raise ValueError("Detector variant must not contain correction output")
    _fingerprint(table, detection)
    for field, root_field in (("protocol_sha256", "protocol_sha256"), ("experimental_source_sha256", "implementation_sha256"),
                              ("execution_addendum_sha256", "execution_addendum_sha256")):
        if metadata.get(field) != manifest[root_field]:
            raise ValueError("Detector metadata and completion signature disagree")
    extraction = metadata.get("extraction", {})
    if extraction.get("source_runner_sha256") != manifest["source_runner_sha256"] or extraction.get("live_module_modified") is not False:
        raise ValueError("Experimental detector extraction is not isolated from core")
    probabilities, mask = _detector_arrays(table, detection)
    if "upstream_binary_decisions" not in metadata:
        raise ValueError("Frozen classifier decisions are required")
    metadata_path, scores_path = detection_path.parent / "metadata.json", detection_path.parent / "frozen_scores.npz"
    if any(str(p) not in expected for p in (metadata_path, scores_path)):
        raise ValueError("Frozen detector metadata/scores are missing from completion hashes")
    if json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Prediction/metadata sidecar mismatch")
    with np.load(scores_path, allow_pickle=False) as frozen:
        if (frozen["scores"].shape != table.shape or not np.array_equal(frozen["scores"], probabilities)
                or frozen["binary_decisions"].dtype != np.dtype(bool)
                or not np.array_equal(frozen["binary_decisions"], mask.astype(bool))
                or tuple(frozen["columns"].tolist()) != table.columns):
            raise ValueError("Frozen detector scores, decisions or column order mismatch")
    comparison = json.loads(_absolute(entry["comparison_path"]).read_text())
    if (comparison.get("dataset") != table.name or comparison.get("input_sha256") != table.fingerprint
            or comparison.get("shape") != list(table.shape) or comparison.get("status") != "experimental_unpromoted"
            or comparison.get("threshold") != .5 or comparison.get("threshold_operator") != ">"
            or comparison.get("comparison_target") != COMPARISON_TARGET
            or comparison.get("no_benchmark_accuracy_claim") is not True
            or any(comparison.get(k) is not True for k in ("full_prefilter_state_identical", "unaffected_columns_exact_reproduction", "recorded_replies_identical"))):
        raise ValueError("Paired detector comparison is not validated under the declared protocol")
    return manifest, expected, mask.astype(bool)


def _validate_cts(table, proposals):
    if (proposals.get("dataset") != table.name or proposals.get("dirty_sha256") != table.fingerprint
            or proposals.get("model") != MODEL or proposals.get("reasoning_effort") != EFFORT
            or proposals.get("experiment_policy") != CTS_POLICY or proposals.get("retrieval_method") != "zeroec_source_cts"
            or proposals.get("artifact_hash") != digest({k: v for k, v in proposals.items() if k != "artifact_hash"})):
        raise ValueError("Expected an exact completed original-source CTSv2 proposal artifact")
    identity = proposals.get("experiment_identity", {})
    source_inputs = _hash_map(proposals.get("source_input_hashes"))
    if (identity.get("source_input_hashes") != source_inputs or identity.get("experiment_policy") != CTS_POLICY
            or identity.get("dirty_sha256") != table.fingerprint
            or identity.get("seed_request_hash") != proposals.get("seed_request_hash")
            or proposals.get("script_sha256") != identity.get("script_sha256")
            or proposals.get("script_sha256") != sha256(PROJECT_ROOT / "scripts/queue_cts_v2.py")):
        raise ValueError("CTSv2 source experiment identity mismatch")
    original._check_hashes(source_inputs)
    source_sha = proposals.get("source_detection_sha256")
    detector_paths = [Path(p) for p, wanted in source_inputs.items() if wanted == source_sha]
    if not _hash(source_sha) or len(detector_paths) != 1:
        raise ValueError("CTSv2 must bind one exact original-source detector")
    source = Prediction.load(detector_paths[0])
    _fingerprint(table, source)
    if source.metadata.get("reproduced_upstream") is not True or source.repairs:
        raise ValueError("CTSv2 external retrieval must retain ORIGINAL-source detector identity")
    probabilities, mask = _detector_arrays(table, source)
    if (proposals.get("source_probabilities_hash") != digest(probabilities.tolist())
            or identity.get("source_decision_mask_hash") != digest(mask.astype(bool).tolist())):
        raise ValueError("CTSv2 source scores/decisions provenance mismatch")
    disclosure = proposals.get("policy_disclosure", {})
    if (disclosure.get("source_predictions_modified") is not False or disclosure.get("mask_query_target") is not True
            or disclosure.get("same_policy_all_datasets") is not True):
        raise ValueError("CTSv2 policy disclosure does not preserve original source inputs")
    cells, requests = proposals.get("cells"), proposals.get("request_hashes")
    if not isinstance(cells, list) or not isinstance(requests, list) or len(cells) != len(requests) or not all(_hash(k) for k in requests):
        raise ValueError("CTSv2 completion must bind each candidate request")
    return source_inputs


def prepare_detector_variant(table, artifact_path, detection_path, proposals_path, detector_manifest_path, output_path, *, config=LoopConfig()):
    artifact_path, detection_path, proposals_path, detector_manifest_path, output_path = (
        Path(p).resolve() for p in (artifact_path, detection_path, proposals_path, detector_manifest_path, output_path))
    explicit = (artifact_path, detection_path, proposals_path, detector_manifest_path)
    if output_path in explicit or original._protected_output(output_path) or output_path.is_relative_to(detector_manifest_path.parent):
        raise ValueError("Candidate output must be new and separate from active/frozen detector inputs")
    if (config.threshold, config.candidate_low, config.candidate_high, config.top_k) != (.5, .25, .85, 3):
        raise ValueError("Detector experiment uses fixed .5 threshold, [.25,.85] candidate band and top three alternatives")
    inputs = {str(p): sha256(p) for p in explicit}
    implementations = {str(p.resolve()): sha256(p) for p in implementation_files()}
    artifact, proposals = (json.loads(p.read_text()) for p in (artifact_path, proposals_path))
    if (artifact.get("dirty_sha256") != table.fingerprint or artifact.get("model") != MODEL
            or artifact.get("reasoning_effort") != EFFORT or artifact.get("dataset", table.name) != table.name):
        raise ValueError("Explicit executable artifact model/dirty identity mismatch")
    detection = Prediction.load(detection_path)
    manifest, frozen_inputs, mask = _validate_completion(table, detector_manifest_path, detection_path, detection)
    cts_inputs = _validate_cts(table, proposals)
    watched = {**frozen_inputs, **cts_inputs, **inputs, **implementations}
    # Reject colliding hash roles instead of silently preferring one source.
    for mapping in (frozen_inputs, cts_inputs, inputs, implementations):
        if any(watched[p] != wanted for p, wanted in mapping.items()):
            raise ValueError("Explicit/frozen/source implementation hash roles conflict")
    config = replace(config, proposal_filename=proposals_path.name)
    identity = {"version": VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
                "detector_experiment_id": EXPERIMENT, "detector_manifest_hash": manifest["manifest_hash"],
                "explicit_inputs": {"artifact": str(artifact_path), "detection": str(detection_path),
                                    "proposals": str(proposals_path), "detector_manifest": str(detector_manifest_path)},
                "input_file_hashes": watched, "source_implementation_hashes": implementations,
                "source_implementation_signature": digest(implementations), "config": asdict(config),
                "detector_predictions_modified": False, "retrieval_rerun": False, "ground_truth_access": False,
                "reproduced_upstream": False, "experimental_unpromoted": True}
    original._check_hashes(watched)
    if output_path.exists():
        result = original._check_existing(output_path, identity)
        original._check_hashes(watched)
        return result
    with tempfile.TemporaryDirectory(prefix="zeroclean-detector-variant-proposals-") as temporary:
        staged = Path(temporary) / table.name / proposals_path.name
        staged.parent.mkdir()
        shutil.copyfile(proposals_path, staged)
        if sha256(staged) != inputs[str(proposals_path)]: raise ValueError("Explicit CTSv2 proposals changed while staging")
        payload = build_candidates(table, artifact, detection, config, artifact_root=temporary)
        if sha256(staged) != inputs[str(proposals_path)]: raise ValueError("Staged CTSv2 proposals changed during preparation")
    payload["detector_provenance"] = {
        "kind": "experimental_empty_predicate_variant", "experiment_id": EXPERIMENT,
        "reproduced_upstream": False, "experimental_unpromoted": True,
        "prediction_file": str(detection_path), "prediction_file_sha256": inputs[str(detection_path)],
        "prediction_hash": digest({"probabilities": detection.probabilities.tolist(), "metadata": detection.metadata}),
        "probabilities_hash": digest(detection.probabilities.tolist()), "metadata_hash": digest(detection.metadata),
        "classifier_mask_hash": digest(mask.tolist()), "classifier_positive_cells": int(mask.sum()),
        "detector_manifest": str(detector_manifest_path), "detector_manifest_sha256": inputs[str(detector_manifest_path)],
        "detector_manifest_hash": manifest["manifest_hash"],
        "candidate_eligibility": "Actual experimental classifier uncertainty [.25,.85] plus all positives, union existing rule/FD/proposal additions; raw scores/mask unchanged",
        "paired_control_caveat": COMPARISON_TARGET}
    payload["external_cts_provenance"] = {
        "method": "frozen_original_source_ctsv2_candidate_evidence", "retrieval_rerun": False,
        "proposal_file_sha256": inputs[str(proposals_path)], "artifact_hash": proposals["artifact_hash"],
        "experiment_policy": CTS_POLICY, "retrieval_method": proposals["retrieval_method"],
        "source_detection_sha256": proposals["source_detection_sha256"], "source_probabilities_hash": proposals["source_probabilities_hash"],
        "source_input_hashes": cts_inputs, "historical_queue_inputs_revalidated": False,
        "scope": "Original-source retrieval is external candidate evidence; candidate eligibility uses the NEW experimental classifier. No peer retrieval or model request is rerun."}
    payload["experiment_preparation"] = identity
    payload["request_hash"] = digest({k: v for k, v in payload.items() if k != "request_hash"})
    original._check_hashes(watched)
    result = original._publish_once(output_path, payload, identity)
    original._check_hashes(watched)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    for name in ("artifact", "detection", "proposals", "detector-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    payload = prepare_detector_variant(load_dirty(args.dataset), args.artifact, args.detection, args.proposals, args.detector_manifest, args.output)
    print(json.dumps({"output": str(args.output), "request_hash": payload["request_hash"], "candidate_cells": len(payload["cells"]),
                      "detector_experiment": EXPERIMENT, "reproduced_upstream": False}, indent=2))


if __name__ == "__main__":
    main()
