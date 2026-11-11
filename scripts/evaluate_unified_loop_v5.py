#!/usr/bin/env python3
"""Evaluator-only entry point; --execute requires a completed immutable cohort.

Importing this module or requesting --help reads no study data or references.
The existing harness keeps its evaluation-only row diagnostics on disk; this
wrapper emits aggregate status only. It never prepares or regenerates baselines.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
STUDY = Path("experimental/unified_loop_v5")
OUTPUT = Path("results/unified_loop_v5")
DATASETS = ("hospital", "flights", "beers", "rayyan", "billionaire", "movies")
FILES = ("zeroed.json", "zeroed_zeroec_reconstruction.json", "zeroclean_loop.json")
METHOD_FILES = {"unified_loop_v5": "zeroclean_loop.json"}
MODEL = {"model": "gpt-6.1-sol", "reasoning_effort": "xhigh"}
COHORT = "local-unified-loop-v5-20261030"
SCHEMA = "unified-loop-v5-freeze-v1"
HEAVY_LOCK = Path(os.environ.get("ZEROCLEAN_HEAVY_LOCK", "/tmp/zeroclean-heavy-experiment.lock"))
BASELINE_STATUS = Path("results/source_baselines_v1/baseline_preflight.json")
MEMORY_BYTES = 12 * 1024**3
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
ABSENT_SOURCE = "ABSENT:package-initializer"
EXECUTION_LOCK_HANDLE = None


def need(condition, label):
    if not condition:
        raise ValueError(label)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    body = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def reject_constant(_value):
    raise ValueError("nonfinite_json_constant")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_object, parse_constant=reject_constant)


def exclusive_json(path, body):
    """Publish a receipt without replacing an existing result."""
    encoded = (json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def replace_json(path, body):
    """Atomically rewrite an evaluator-owned summary after its first publish."""
    path = Path(path)
    encoded = (json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def input_paths(name):
    return {
        "dirty": f"vendor/ZeroED/data/{name}_error-01.csv",
        "pass2": f"artifacts/llm/unified_loop_v5/{name}/pass2.json",
        "old_pass2": f"artifacts/llm/{name}/pass2.json",
        "fresh_packet": f"artifacts/unified_loop_v5_model/packets/{name}.json",
        "fresh_response": f"artifacts/unified_loop_v5_model/responses/{name}.json",
        "cts": f"artifacts/llm/{name}/proposals_cts_v2.json",
        "source": f"results/source_baselines_v1/predictions/{name}/zeroed.json",
        "sequential": f"results/source_baselines_v1/predictions/{name}/zeroed_zeroec_reconstruction.json",
        "empty": f"results/conservative_seed_v1/predictions/{name}/empty_carry.json",
    }


def source_hashes():
    required = {STUDY / name for name in ("run_study.py", "detection.py", "consensus.py",
                "repair.py", "protocol.md", "runner_design.md", "test_integration.py")}
    need(all((ROOT / path).is_file() for path in required), "missing_frozen_source")
    paths = required | {path.relative_to(ROOT) for pattern in ("*.py", "*.md")
                        for path in (ROOT / STUDY).glob(pattern)}
    paths.update(Path("zeroclean") / name for name in ("data.py", "types.py", "statistics.py", "sandbox.py"))
    initializers = {Path("zeroclean/__init__.py"), Path("experimental/__init__.py"), STUDY / "__init__.py"}
    paths.update(initializers)
    return {str(path): ABSENT_SOURCE if path in initializers and not (ROOT / path).exists()
            else sha256(ROOT / path) for path in sorted(paths)}


def evaluator_hashes():
    paths = {Path(__file__).resolve(), ROOT / "evaluate_e2e.py",
             ROOT / "artifacts/unified_loop_v5_execution/evaluation_plan.md"}
    paths.update((ROOT / "zeroclean").glob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def exact_files(directory, names):
    """Reject partial, extra, nested, or symbolic-link prediction artifacts."""
    directory = Path(directory)
    need(directory.is_dir() and not directory.is_symlink(), "missing_artifact_directory")
    entries = list(directory.iterdir())
    need({path.name for path in entries} == set(names)
         and all(path.is_file() and not path.is_symlink() for path in entries),
         "changed_artifact_inventory")


def verify_completion():
    """Hash every frozen input/output, without parsing data or opening references."""
    output = ROOT / OUTPUT
    lock_path, plan_path = ROOT / STUDY / "protocol_lock.json", ROOT / STUDY / "plan.json"
    completion_path = output / "completion_manifest.json"
    lock, plan, completion = map(read_json, (lock_path, plan_path, completion_path))
    for record in (lock, plan, completion):
        need(record.get("schema_version") == SCHEMA and record.get("datasets") == list(DATASETS)
             and record.get("cohort") == COHORT and record.get("method_files") == METHOD_FILES
             and record.get("ground_truth_access") is False, "invalid_global_freeze")
    need(lock.get("model") == MODEL and plan.get("model") == MODEL
         and lock.get("frozen_before_inference") is True
         and plan.get("evaluation_is_separate") is True and plan.get("output_root") == str(OUTPUT),
         "changed_protocol")
    lock_hash, plan_hash = sha256(lock_path), sha256(plan_path)
    need(lock.get("plan_sha256") == plan_hash
         and plan.get("protocol_sha256") == sha256(ROOT / STUDY / "protocol.md"), "changed_plan")
    sources = source_hashes()
    need(sources == lock.get("source_files"), "changed_inference_sources")
    inputs = {name: {role: {"path": path, "sha256": sha256(ROOT / path)}
                     for role, path in input_paths(name).items()} for name in DATASETS}
    need(inputs == lock.get("input_files")
         and plan.get("allowed_inference_inputs") == {name: input_paths(name) for name in DATASETS},
         "changed_inference_inputs")
    need(completion.get("six_dataset_inference_complete") is True
         and completion.get("evaluation_performed") is False
         and completion.get("protocol_lock_sha256") == lock_hash
         and set(completion.get("dataset_freezes", {})) == set(DATASETS), "incomplete_cohort")
    prediction_root = output / "predictions"
    need(prediction_root.is_dir() and not prediction_root.is_symlink(), "missing_predictions")
    destinations = list(prediction_root.iterdir())
    need({path.name for path in destinations} == set(DATASETS)
         and all(path.is_dir() and not path.is_symlink() for path in destinations),
         "changed_prediction_dataset_set")
    exact_files(output / "frozen", [f"{name}.json" for name in DATASETS])
    exact_files(output / "audit", [f"{name}.jsonl" for name in DATASETS])
    exact_files(output / "diagnostics", [f"{name}.json" for name in DATASETS])
    freezes, predictions, supporting = {}, {}, {}
    for name in DATASETS:
        frozen_path = output / "frozen" / f"{name}.json"
        freezes[name] = sha256(frozen_path)
        need(completion["dataset_freezes"][name] == freezes[name], "changed_dataset_freeze")
        frozen = read_json(frozen_path)
        expected_inputs = {str(ROOT / item["path"]): item["sha256"] for item in inputs[name].values()}
        need(frozen.get("frozen_before_evaluation") is True and frozen.get("ground_truth_access") is False
             and frozen.get("dirty_sha256") == inputs[name]["dirty"]["sha256"]
             and frozen.get("protocol_lock_sha256") == lock_hash and frozen.get("plan_sha256") == plan_hash
             and frozen.get("source_signature") == digest(sources)
             and frozen.get("config") == plan.get("configuration")
             and frozen.get("method_files") == METHOD_FILES and frozen.get("model") == MODEL
             and frozen.get("cohort") == COHORT and frozen.get("inference_inputs") == expected_inputs,
             "inconsistent_dataset_freeze")
        destination = output / "predictions" / name
        exact_files(destination, FILES)
        predictions[name] = {filename: sha256(destination / filename) for filename in FILES}
        need(frozen.get("prediction_files") == predictions[name], "changed_predictions")
        need(predictions[name]["zeroed.json"] == inputs[name]["source"]["sha256"]
             and predictions[name]["zeroed_zeroec_reconstruction.json"] == inputs[name]["sequential"]["sha256"],
             "baseline_copy_changed")
        expected_support = (OUTPUT / "audit" / f"{name}.jsonl", OUTPUT / "diagnostics" / f"{name}.json")
        supporting[name] = {str(path): sha256(ROOT / path) for path in expected_support}
        need(frozen.get("supporting_files") == supporting[name], "changed_supporting_files")
    return {"protocol_lock_sha256": lock_hash, "plan_sha256": plan_hash,
            "completion_manifest_sha256": sha256(completion_path), "dataset_freezes": freezes,
            "source_files": sources, "input_files": inputs,
            "prediction_files": predictions, "supporting_files": supporting}


def dirty_shape(name):
    """Check the dirty CSV shape before the evaluator may open any reference."""
    with (ROOT / input_paths(name)["dirty"]).open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle)
        columns = next(rows)
        need(bool(columns) and len(columns) == len(set(columns)), "invalid_dirty_columns")
        count = 0
        for row in rows:
            need(len(row) == len(columns), "invalid_dirty_shape")
            count += 1
    need(count > 0, "empty_dirty_table")
    return count, len(columns)


def validate_prediction_masks(bindings):
    """Validate all 18 stored strict masks before any clean reference is opened."""
    shapes = {}
    for name in DATASETS:
        expected_shape = dirty_shape(name)
        for filename in FILES:
            body = read_json(ROOT / OUTPUT / "predictions" / name / filename)
            need(isinstance(body, dict) and isinstance(body.get("metadata"), dict), "invalid_prediction_object")
            probabilities, metadata = body["probabilities"], body["metadata"]
            mask = metadata.get("upstream_binary_decisions")
            need(isinstance(probabilities, list) and bool(probabilities)
                 and isinstance(probabilities[0], list) and bool(probabilities[0]), "invalid_probabilities")
            shape = (len(probabilities), len(probabilities[0]))
            need(shape == expected_shape, "prediction_dirty_shape_mismatch")
            need(isinstance(mask, list) and len(mask) == shape[0], "missing_boolean_mask")
            need(float(metadata.get("config", {}).get("threshold", .5)) == .5, "changed_detection_threshold")
            need(metadata.get("decision_threshold", .5) == .5
                 and metadata.get("decision_threshold_operator", ">") == ">", "changed_threshold_policy")
            for scores, decisions in zip(probabilities, mask):
                need(isinstance(scores, list) and isinstance(decisions, list)
                     and len(scores) == len(decisions) == shape[1], "invalid_mask_shape")
                for score, decision in zip(scores, decisions):
                    need(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
                         and type(decision) is bool and decision == (score > .5), "invalid_strict_boolean_mask")
            need(name not in shapes or shapes[name] == shape, "inconsistent_prediction_shapes")
            shapes[name] = shape
            seen = set()
            need(isinstance(body.get("repairs"), list), "invalid_repair_map")
            for repair in body["repairs"]:
                i, j = repair["row"], repair["column"]
                need(type(i) is int and type(j) is int and 0 <= i < shape[0] and 0 <= j < shape[1]
                     and isinstance(repair["value"], str) and (i, j) not in seen and mask[i][j],
                     "invalid_repair_or_mask_containment")
                seen.add((i, j))
            if filename == "zeroclean_loop.json":
                need(metadata.get("system") == "unified_loop_v5"
                     and metadata.get("dataset") == name and metadata.get("method_files") == METHOD_FILES
                     and metadata.get("detection_mask_sha256") == digest(mask)
                     and metadata.get("cohort") == COHORT and metadata.get("model") == MODEL
                     and metadata.get("ground_truth_access") is False
                     and metadata.get("frozen_before_evaluation") is True
                     and metadata.get("input_fingerprint") == bindings["input_files"][name]["dirty"]["sha256"]
                     and metadata.get("protocol_lock_sha256") == bindings["protocol_lock_sha256"]
                     and metadata.get("plan_sha256") == bindings["plan_sha256"], "invalid_unified_method_alias")
    return {name: list(shape) for name, shape in shapes.items()}


def bound_environment():
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    os.environ["OMP_THREAD_LIMIT"], os.environ["OMP_DYNAMIC"] = "1", "FALSE"
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limits = [MEMORY_BYTES] + [value for value in (soft, hard) if value != resource.RLIM_INFINITY]
    ceiling = min(limits)
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))


@contextmanager
def heavy_execution():
    global EXECUTION_LOCK_HANDLE
    need(EXECUTION_LOCK_HANDLE is None, "nested_heavy_execution_refused")
    with HEAVY_LOCK.open("a+", encoding="utf-8") as handle:
        print(json.dumps({"event": "evaluation_queued", "datasets": 6}), flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            bound_environment()  # No scientific imports have occurred.
            EXECUTION_LOCK_HANDLE = handle
            yield
        finally:
            EXECUTION_LOCK_HANDLE = None
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def require_execution_context():
    need(EXECUTION_LOCK_HANDLE is not None and not EXECUTION_LOCK_HANDLE.closed,
         "mandatory_heavy_lock_not_held")
    limits = resource.getrlimit(resource.RLIMIT_AS)
    need(all(limit != resource.RLIM_INFINITY and 0 < limit <= MEMORY_BYTES for limit in limits)
         and all(os.environ.get(name) == "1" for name in THREAD_VARIABLES)
         and os.environ.get("OMP_THREAD_LIMIT") == "1"
         and os.environ.get("OMP_DYNAMIC") == "FALSE", "resource_bounds_not_active")


def require_unscored_output(output):
    for name in ("summary_metrics.json", "evaluation_receipt.json", "evaluation_attempt.json",
                 "evaluation_only_failures", "independent_audit.json", "independent_audit.md"):
        path = output / name
        need(not path.exists() and not path.is_symlink(), "existing_or_partial_evaluation_refused")


def evaluate_once():
    require_execution_context()
    output = ROOT / OUTPUT
    summary_path, receipt_path = output / "summary_metrics.json", output / "evaluation_receipt.json"
    attempt_path = output / "evaluation_attempt.json"
    require_unscored_output(output)
    bindings = verify_completion()
    evaluator_sources = evaluator_hashes()
    baseline_hash = sha256(ROOT / BASELINE_STATUS)
    baseline_status = read_json(ROOT / BASELINE_STATUS)
    shapes = validate_prediction_masks(bindings)
    # evaluate_e2e uses relative default data paths. Keep them rooted here.
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        import evaluate_e2e
    need(verify_completion() == bindings and evaluator_hashes() == evaluator_sources
         and sha256(ROOT / BASELINE_STATUS) == baseline_hash, "pre_evaluation_hash_change")
    require_unscored_output(output)
    attempt = {"schema_version": "unified-loop-v5-evaluation-attempt-v1", "cohort": COHORT,
               "datasets": list(DATASETS), "method_files": list(FILES), "prediction_count": 18,
               "started_utc": datetime.now(timezone.utc).isoformat(), "integrity": bindings,
               "evaluator_sources": evaluator_sources, "prediction_shapes": shapes,
               "labels_used_only_in_evaluation": True, "aggregate_only_reporting": True}
    # A crash or interruption after this point permanently blocks a second score.
    exclusive_json(attempt_path, attempt)
    attempt_hash = sha256(attempt_path)
    receipt = {"schema_version": "unified-loop-v5-evaluation-receipt-v1", "cohort": COHORT,
               "datasets": list(DATASETS), "method_files": list(FILES), "prediction_count": 18,
               "started_utc": attempt["started_utc"], "integrity": bindings,
               "evaluator_sources": evaluator_sources, "prediction_shapes": shapes,
               "evaluation_attempt_sha256": attempt_hash,
               "baseline_status": {"path": str(BASELINE_STATUS), "sha256": baseline_hash},
               "heavy_lock": str(HEAVY_LOCK),
               "address_space_limits_bytes": list(resource.getrlimit(resource.RLIMIT_AS)),
               "numerical_threads": {name: os.environ[name] for name in THREAD_VARIABLES},
               "labels_used_only_in_evaluation": True, "aggregate_only_reporting": True}
    stage = "reference_binding"
    try:
        # First clean-reference access follows every all-six guard and the marker.
        references = {name: sha256(ROOT / "vendor/ZeroED/data" / f"{name}_clean.csv") for name in DATASETS}
        receipt["reference_sha256"] = references
        stage = "evaluate_all"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            evaluate_e2e.evaluate_all(DATASETS, output, baseline_status)
        # Evaluator-only adapted baseline: repair the known source detector
        # sampling/empty-predicate path without exposing labels to inference.
        # These values are comparison-only and are never frozen as inference
        # inputs or mixed into the qualified source comparison.
        from zeroclean.baselines import adapted_zeroed, adapted_zeroed_zeroec
        from zeroclean.data import load_dirty
        from zeroclean.llm import load_artifact
        from zeroclean.evaluation import evaluate
        adapted_metrics = {}
        for name in DATASETS:
            table = load_dirty(name)
            pass1 = load_artifact(table, 1)
            adapted = adapted_zeroed(table, pass1)
            sequential = adapted_zeroed_zeroec(table, adapted, pass1)
            adapted_metrics[name] = {
                "adapted_zeroed": evaluate(table, adapted, threshold=.5)[0],
                "adapted_zeroed_zeroec": evaluate(table, sequential, threshold=.5)[0],
            }
        stage = "post_evaluation_integrity"
        need(verify_completion() == bindings and evaluator_hashes() == evaluator_sources
             and sha256(ROOT / BASELINE_STATUS) == baseline_hash
             and sha256(attempt_path) == attempt_hash, "post_evaluation_hash_change")
        need(references == {name: sha256(ROOT / "vendor/ZeroED/data" / f"{name}_clean.csv")
                            for name in DATASETS}, "reference_changed_during_evaluation")
        summary = read_json(summary_path)
        adapted_macro = {variant: {metric: sum(adapted_metrics[name][variant][metric]["f1"] for name in DATASETS) / len(DATASETS)
                                   for metric in ("detection", "repair")}
                         for variant in ("adapted_zeroed", "adapted_zeroed_zeroec")}
        adapted_assessment = {
            "status": "evaluator_only_adapted_comparison",
            "official_exact_baseline": False,
            "metrics": adapted_metrics,
            "macro_f1": adapted_macro,
            "note": "Bug-fixed adapted ZeroED comparison; labels used only in evaluation.",
        }
        summary["adapted_comparison_only"] = adapted_assessment
        # evaluate_e2e publishes the initial summary before this evaluator-only
        # adapted comparison is available. Replace that evaluator-owned summary
        # atomically rather than using the no-overwrite receipt primitive.
        replace_json(summary_path, summary)
        variants = {Path(filename).stem for filename in FILES}
        need(summary.get("measured_datasets") == list(DATASETS)
             and summary.get("six_dataset_benchmark_complete") is True
             and summary.get("baseline_status") == baseline_status
             and set(summary.get("metrics", {})) == set(DATASETS), "invalid_evaluation_summary")
        for name in DATASETS:
            need(set(summary["metrics"][name]) == variants, "unexpected_scored_variant")
            need(all(result["reference_sha256"] == references[name]
                     for result in summary["metrics"][name].values()), "summary_reference_mismatch")
        official = summary["official_target_assessment"]
        qualified = summary["qualified_source_reconstruction_comparison"]
        need(official["met"] is False and official["status"] != "verified"
             and qualified.get("official_exact_baseline") is False
             and qualified.get("fixed_example_count") == 0, "invalid_exact_baseline_claim")
        outcome = {"status": "complete", "summary_sha256": sha256(summary_path),
                   "official_target_assessment": official, "qualified_target_assessment": qualified}
    except BaseException as exc:
        receipt.update({"completed_utc": datetime.now(timezone.utc).isoformat(),
                        "outcome": {"status": "failed", "stage": stage, "error_type": type(exc).__name__}})
        exclusive_json(receipt_path, receipt)
        raise
    receipt.update({"completed_utc": datetime.now(timezone.utc).isoformat(), "outcome": outcome})
    exclusive_json(receipt_path, receipt)
    print(json.dumps({"event": "evaluation_complete", "datasets": 6, "predictions": 18,
                      "macro_f1": summary["macro_f1"], "official_targets": official,
                      "qualified_targets": qualified}, sort_keys=True, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True,
                        help="Run only after explicit authorization and all-six inference completion")
    parser.parse_args(argv)
    try:
        with heavy_execution():
            evaluate_once()
    except BaseException as exc:
        # Exception messages can contain raw values. Never relay them or tracebacks.
        print(json.dumps({"event": "evaluation_failed", "error_type": type(exc).__name__}), file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
