#!/usr/bin/env python3
"""Evaluator-only direct recount for the fixed unified_loop_v5 study.

This standalone standard-library program never imports the inference runner or
either scoring module. An actual audit requires explicit --run-after-scoring-
complete authorization and the existing six-dataset completion and score files.
Only aggregate findings are published; no cell values or coordinates are logged.
Use --self-test to exercise fictional fixtures without reading study artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import tempfile


DATASETS = ("hospital", "flights", "beers", "rayyan", "billionaire", "movies")
VARIANTS = ("zeroed", "zeroed_zeroec_reconstruction", "zeroclean_loop")
ROOT = Path(__file__).resolve().parents[1]
STUDY = Path("experimental/unified_loop_v5")
OUTPUT = Path("results/unified_loop_v5")
PLAN = STUDY / "plan.json"
PROTOCOL_LOCK = STUDY / "protocol_lock.json"
EVALUATION_PLAN = Path("artifacts/unified_loop_v5_execution/evaluation_plan.md")
HEAVY_LOCK = Path(os.environ.get("ZEROCLEAN_HEAVY_LOCK", "/tmp/zeroclean-heavy-experiment.lock"))
MEMORY_BYTES = 12 * 1024**3
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
HEAVY_CONTEXT_ACTIVE = False
ABSENT_INITIALIZER = "ABSENT:package-initializer"
INITIALIZERS = {Path("zeroclean/__init__.py"), Path("experimental/__init__.py"), STUDY / "__init__.py"}
SCHEMA = "unified-loop-v5-freeze-v1"
COHORT = "local-unified-loop-v5-20261030"
MODEL = {"model": "gpt-6.1-sol", "reasoning_effort": "xhigh"}
METHOD_FILES = {"unified_loop_v5": "zeroclean_loop.json"}
TOLERANCE = 1e-12
UNBOUND = object()
TARGETS = {"ed_mean_absolute_gain": .035, "repair_mean_absolute_gain": .08,
           "flights_ed_f1_strictly_above": .76, "rayyan_ed_f1_strictly_above": .77,
           "ed_improves_each_dataset": True}


class AuditError(Exception):
    """A fixed code, never a data-bearing exception message."""


def require(condition, code):
    if not condition:
        raise AuditError(code)


def reject_constant(_value):
    raise AuditError("nonfinite_json_constant")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_object)
    except AuditError:
        raise
    except Exception:
        raise AuditError("json_read_or_parse_failed") from None


def sha256(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except Exception:
        raise AuditError("hash_read_failed") from None
    return digest.hexdigest()


def canonical_digest(value):
    body = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def read_csv_exact(path):
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader)
            require(bool(columns) and len(columns) == len(set(columns)), "csv_header_invalid")
            rows = list(reader)
        require(all(len(row) == len(columns) for row in rows), "csv_ragged")
        return columns, rows
    except AuditError:
        raise
    except Exception:
        raise AuditError("csv_read_or_parse_failed") from None


def prf(tp, predicted, actual):
    """Count ratios are exact rational numbers until JSON comparison/display."""
    return {"tp": tp, "fp": predicted - tp, "fn": actual - tp,
            "predicted": predicted, "actual": actual,
            "precision": Fraction(tp, predicted) if predicted else Fraction(0),
            "recall": Fraction(tp, actual) if actual else Fraction(0),
            "f1": Fraction(2 * tp, predicted + actual) if predicted + actual else Fraction(0)}


def prediction_parts(body, rows, columns):
    """Validate stored decisions and edits without accepting any reference table."""
    require(isinstance(body, dict), "prediction_object_required")
    metadata = body.get("metadata")
    require(isinstance(metadata, dict), "prediction_metadata_object_required")
    require(type(metadata.get("decision_threshold")) in (int, float)
            and metadata["decision_threshold"] == .5
            and metadata.get("decision_threshold_operator") == ">", "strict_threshold_metadata")
    if isinstance(metadata.get("config"), dict) and "threshold" in metadata["config"]:
        require(type(metadata["config"]["threshold"]) in (int, float)
                and metadata["config"]["threshold"] == .5, "config_threshold")
    probabilities, mask = body.get("probabilities"), metadata.get("upstream_binary_decisions")
    for matrix, label in ((probabilities, "probabilities"), (mask, "stored_mask")):
        require(isinstance(matrix, list) and len(matrix) == rows
                and all(isinstance(row, list) and len(row) == columns for row in matrix), label + "_shape")
    for i in range(rows):
        for j in range(columns):
            probability, decision = probabilities[i][j], mask[i][j]
            require(type(probability) in (int, float) and math.isfinite(probability)
                    and 0 <= probability <= 1, "probability_range_or_type")
            # Preserved source masks may encode Booleans as JSON integers 0/1.
            require(type(decision) in (bool, int) and decision in (0, 1), "stored_mask_boolean")
            require(bool(decision) == (probability > .5), "stored_mask_strict_threshold_mismatch")
    repairs = body.get("repairs")
    require(isinstance(repairs, list), "repairs_list_required")
    seen = set()
    for repair in repairs:
        require(isinstance(repair, dict), "repair_object_required")
        i, j, value = repair.get("row"), repair.get("column"), repair.get("value")
        require(type(i) is int and type(j) is int and 0 <= i < rows and 0 <= j < columns,
                "repair_coordinate_invalid")
        require(isinstance(value, str), "repair_string_required")
        require((i, j) not in seen, "duplicate_repair_coordinate")
        seen.add((i, j))
        require(bool(mask[i][j]), "edit_outside_stored_mask")
    return probabilities, mask, repairs


def recount_prediction(body, dirty, clean):
    """Recount one already frozen output; retain no row-valued findings."""
    require(bool(dirty) and bool(dirty[0]), "empty_benchmark_table")
    rows, columns = len(dirty), len(dirty[0])
    require(len(clean) == rows and all(len(row) == columns for row in dirty + clean), "table_alignment")
    _, mask, repairs = prediction_parts(body, rows, columns)
    original_errors = detection_tp = detection_predicted = 0
    for i in range(rows):
        for j in range(columns):
            originally_dirty = dirty[i][j] != clean[i][j]
            original_errors += originally_dirty
            detection_predicted += bool(mask[i][j])
            detection_tp += bool(mask[i][j]) and originally_dirty
    changed = repair_tp = noops = 0
    for repair in repairs:
        i, j, value = repair["row"], repair["column"], repair["value"]
        if value == dirty[i][j]:
            noops += 1
            continue
        changed += 1
        repair_tp += dirty[i][j] != clean[i][j] and value == clean[i][j]
    return {"detection": prf(detection_tp, detection_predicted, original_errors),
            "repair": prf(repair_tp, changed, original_errors), "rows": rows, "columns": columns,
            "declared_noop_repairs": noops}


def input_paths(dataset):
    """The v3 cohort requires a fresh, v4 study-scoped rule artifact for every dataset."""
    return {"dirty": f"vendor/ZeroED/data/{dataset}_error-01.csv",
            "pass2": f"artifacts/llm/unified_loop_v5/{dataset}/pass2.json",
            "old_pass2": f"artifacts/llm/{dataset}/pass2.json",
            "fresh_packet": f"artifacts/unified_loop_v5_model/packets/{dataset}.json",
            "fresh_response": f"artifacts/unified_loop_v5_model/responses/{dataset}.json",
            "cts": f"artifacts/llm/{dataset}/proposals_cts_v2.json",
            "source": f"results/source_baselines_v1/predictions/{dataset}/zeroed.json",
            "sequential": f"results/source_baselines_v1/predictions/{dataset}/zeroed_zeroec_reconstruction.json",
            "empty": f"results/conservative_seed_v1/predictions/{dataset}/empty_carry.json"}


def rational(value):
    return {"numerator": value.numerator, "denominator": value.denominator, "decimal": float(value)}


def target_checks(ed_gains, repair_gains, flights_f1, rayyan_f1):
    """Prespecified targets, using exact fractions through every comparison."""
    ed_macro_gain = sum(ed_gains.values(), Fraction(0)) / len(DATASETS)
    repair_macro_gain = sum(repair_gains.values(), Fraction(0)) / len(DATASETS)
    checks = {"ed_mean_gain": ed_macro_gain >= Fraction(35, 1000),
              "ed_each_dataset": all(value > 0 for value in ed_gains.values()),
              "flights": flights_f1 > Fraction(76, 100),
              "rayyan": rayyan_f1 > Fraction(77, 100),
              "repair_mean_gain": repair_macro_gain >= Fraction(8, 100)}
    return checks, ed_macro_gain, repair_macro_gain


def safe_reported(value):
    # Never echo an arbitrary string or nested object from a malformed artifact.
    if type(value) in (int, float, bool) or value is None:
        return value if not isinstance(value, float) or math.isfinite(value) else "nonfinite"
    return {"type": type(value).__name__}


class Auditor:
    def __init__(self, root, summary_path=None, *, _fictional=False):
        self.root = Path(root).resolve()
        self.summary_path = Path(summary_path) if summary_path else self.root / OUTPUT / "summary_metrics.json"
        self.mismatches = []
        self.receipts = {}
        self.evaluated = 0
        self.integer_fields = self.ratio_fields = 0
        self.stage = "preflight"
        self.aggregate = {}
        self.fictional = _fictional
        self.resource_verified = False

    def compare(self, category, label, field, expected, reported, ratio=False):
        if ratio:
            self.ratio_fields += 1
            valid = type(reported) in (int, float) and math.isfinite(reported)
            equal = valid and abs(float(expected) - reported) <= TOLERANCE
        else:
            if type(expected) is int:
                self.integer_fields += 1
                equal = type(reported) is int and expected == reported
            elif type(expected) is bool:
                equal = type(reported) is bool and expected == reported
            else:
                equal = expected == reported
        if not equal:
            self.mismatches.append({"category": category, "artifact": label, "field": field,
                                    "expected": float(expected) if isinstance(expected, Fraction) else expected,
                                    "reported": safe_reported(reported)})

    def record_hash(self, relative, expected=UNBOUND, binding="observed"):
        relative = Path(relative)
        require(not relative.is_absolute() and ".." not in relative.parts, "nonrelative_receipt_path")
        path = self.root / relative
        key = str(relative)
        if expected == ABSENT_INITIALIZER:
            require(relative in INITIALIZERS, "invalid_absence_sentinel_path")
            actual = ABSENT_INITIALIZER if not path.exists() else sha256(path)
        else:
            if expected is not UNBOUND:
                require(isinstance(expected, str) and len(expected) == 64
                        and all(c in "0123456789abcdef" for c in expected), "invalid_sha256_binding")
            actual = sha256(path)
        if key in self.receipts:
            self.compare("integrity", key, "stable_hash", self.receipts[key]["sha256"], actual)
            if binding not in self.receipts[key]["bindings"]:
                self.receipts[key]["bindings"].append(binding)
        else:
            self.receipts[key] = {"sha256": actual, "bindings": [binding]}
        if expected is not UNBOUND:
            self.compare("integrity", key, binding, expected, actual)
        return actual

    def frozen_integrity(self):
        self.stage = "frozen_integrity"
        self.record_hash(EVALUATION_PLAN)
        lock_hash = self.record_hash(PROTOCOL_LOCK)
        completion_path = OUTPUT / "completion_manifest.json"
        self.record_hash(completion_path)
        lock, completion = read_json(self.root / PROTOCOL_LOCK), read_json(self.root / completion_path)
        plan_hash = self.record_hash(PLAN, lock.get("plan_sha256"), "protocol_lock.plan_sha256")
        plan = read_json(self.root / PLAN)
        for label, body in (("protocol_lock", lock), ("plan", plan), ("completion_manifest", completion)):
            require(isinstance(body, dict), label + "_object")
            self.compare("integrity", label, "datasets", list(DATASETS), body.get("datasets"))
            for field, expected in (("schema_version", SCHEMA), ("method_files", METHOD_FILES),
                                    ("cohort", COHORT), ("ground_truth_access", False)):
                self.compare("integrity", label, field, expected, body.get(field))
        for label, body in (("protocol_lock", lock), ("plan", plan)):
            self.compare("integrity", label, "model", MODEL, body.get("model"))
        self.compare("integrity", "protocol_lock", "frozen_before_inference", True, lock.get("frozen_before_inference"))
        self.compare("integrity", "plan", "evaluation_is_separate", True, plan.get("evaluation_is_separate"))
        self.compare("integrity", "plan", "output_root", str(OUTPUT), plan.get("output_root"))
        allowlist = {dataset: input_paths(dataset) for dataset in DATASETS}
        self.compare("integrity", "plan", "allowed_inference_inputs", allowlist, plan.get("allowed_inference_inputs"))
        self.compare("integrity", "completion_manifest", "protocol_lock_sha256", lock_hash,
                     completion.get("protocol_lock_sha256"))
        self.compare("integrity", "completion_manifest", "six_dataset_inference_complete", True,
                     completion.get("six_dataset_inference_complete"))
        self.compare("integrity", "completion_manifest", "evaluation_performed", False, completion.get("evaluation_performed"))
        sources, inputs = lock.get("source_files"), lock.get("input_files")
        require(isinstance(sources, dict) and isinstance(inputs, dict), "frozen_manifest_objects")
        required_sources = {STUDY / name for name in ("run_study.py", "detection.py", "consensus.py", "repair.py",
                            "protocol.md", "runner_design.md", "test_integration.py")}
        actual_sources = required_sources | INITIALIZERS | {Path("zeroclean") / name for name in
                          ("data.py", "types.py", "statistics.py", "sandbox.py")}
        actual_sources |= {path.relative_to(self.root) for extension in ("*.py", "*.md")
                           for path in (self.root / STUDY).glob(extension)}
        self.compare("integrity", "protocol_lock", "source_path_set", sorted(str(p) for p in actual_sources), sorted(sources))
        for relative in sorted(actual_sources):
            require(str(relative) in sources, "source_manifest_path_missing")
            self.record_hash(relative, sources[str(relative)], "protocol_lock.source_files")
        self.compare("integrity", "plan", "protocol_sha256", sources.get(str(STUDY / "protocol.md")), plan.get("protocol_sha256"))
        self.compare("integrity", "protocol_lock", "input_dataset_set", sorted(DATASETS), sorted(inputs))
        prediction_root = self.root / OUTPUT / "predictions"
        self.compare("integrity", "prediction_inventory", "root_entry_set", sorted(DATASETS),
                     sorted(path.name for path in prediction_root.iterdir()))
        self.compare("integrity", "prediction_inventory", "dataset_directory_set", sorted(DATASETS),
                     sorted(path.name for path in prediction_root.iterdir() if path.is_dir()))
        self.compare("integrity", "prediction_inventory", "all_json_files",
                     sorted(str(OUTPUT / "predictions" / d / (v + ".json")) for d in DATASETS for v in VARIANTS),
                     sorted(str(path.relative_to(self.root)) for path in prediction_root.rglob("*.json")))
        for kind, suffix in (("frozen", ".json"), ("audit", ".jsonl"), ("diagnostics", ".json")):
            self.compare("integrity", kind + "_inventory", "file_set", sorted(d + suffix for d in DATASETS),
                         sorted(path.name for path in (self.root / OUTPUT / kind).iterdir()))
        for dataset in DATASETS:
            dataset_inputs = inputs.get(dataset)
            require(isinstance(dataset_inputs, dict), "dataset_input_manifest_object")
            self.compare("integrity", dataset, "input_roles", sorted(allowlist[dataset]), sorted(dataset_inputs))
            for role, relative in allowlist[dataset].items():
                item = dataset_inputs.get(role)
                require(isinstance(item, dict), "input_binding_object")
                self.compare("integrity", dataset + "/" + role, "path", relative, item.get("path"))
                self.record_hash(relative, item.get("sha256"), "protocol_lock.input_files")
        freezes = completion.get("dataset_freezes")
        require(isinstance(freezes, dict), "completion_dataset_freezes_object")
        self.compare("integrity", "completion_manifest", "freeze_dataset_set", sorted(DATASETS), sorted(freezes))
        for dataset in DATASETS:
            relative = OUTPUT / "frozen" / f"{dataset}.json"
            self.record_hash(relative, freezes.get(dataset), "completion_manifest.dataset_freezes")
            frozen = read_json(self.root / relative)
            require(isinstance(frozen, dict), "dataset_freeze_object")
            expected_fields = {"protocol_lock_sha256": lock_hash, "plan_sha256": plan_hash,
                               "source_signature": canonical_digest(sources), "dirty_sha256": inputs[dataset]["dirty"]["sha256"],
                               "frozen_before_evaluation": True, "ground_truth_access": False,
                               "method_files": METHOD_FILES, "model": MODEL, "cohort": COHORT,
                               "config": plan.get("configuration"), "design_provenance": plan.get("design_provenance"),
                               "inference_inputs": {str(self.root / item["path"]): item["sha256"]
                                                    for item in inputs[dataset].values()}}
            for field, expected in expected_fields.items():
                self.compare("integrity", str(relative), field, expected, frozen.get(field))
            predictions, support = frozen.get("prediction_files"), frozen.get("supporting_files")
            require(isinstance(predictions, dict) and isinstance(support, dict), "dataset_file_binding_objects")
            self.compare("integrity", dataset, "prediction_file_set", sorted(v + ".json" for v in VARIANTS), sorted(predictions))
            self.compare("integrity", dataset, "actual_prediction_file_set", sorted(v + ".json" for v in VARIANTS),
                         sorted(path.name for path in (self.root / OUTPUT / "predictions" / dataset).iterdir()))
            expected_support = {str(OUTPUT / "audit" / f"{dataset}.jsonl"), str(OUTPUT / "diagnostics" / f"{dataset}.json")}
            self.compare("integrity", dataset, "supporting_file_set", sorted(expected_support), sorted(support))
            for variant in VARIANTS:
                filename = variant + ".json"
                prediction_path = OUTPUT / "predictions" / dataset / filename
                self.record_hash(prediction_path, predictions.get(filename), "dataset_freeze.prediction_files")
                if variant != "zeroclean_loop":
                    role = "source" if variant == "zeroed" else "sequential"
                    self.record_hash(prediction_path, inputs[dataset][role]["sha256"], "byte_preserved_baseline_input")
            for support_path in sorted(expected_support):
                self.record_hash(support_path, support.get(support_path), "dataset_freeze.supporting_files")
        require(not self.mismatches, "frozen_integrity_mismatch_no_references_opened")
        return lock

    def evaluation_receipt(self, lock):
        """Verify scoring completion and bindings without using its calculations."""
        self.stage = "initial_evaluation_receipt"
        relative = OUTPUT / "evaluation_receipt.json"
        self.record_hash(relative)
        receipt = read_json(self.root / relative)
        require(isinstance(receipt, dict), "evaluation_receipt_object")
        expected_fields = {"schema_version": "unified-loop-v5-evaluation-receipt-v1", "cohort": COHORT,
                           "datasets": list(DATASETS), "method_files": [v + ".json" for v in VARIANTS],
                           "prediction_count": 18, "heavy_lock": str(HEAVY_LOCK),
                           "labels_used_only_in_evaluation": True, "aggregate_only_reporting": True}
        for field, expected in expected_fields.items():
            self.compare("evaluation_receipt", str(relative), field, expected, receipt.get(field))
        outcome = receipt.get("outcome")
        require(isinstance(outcome, dict), "evaluation_receipt_outcome_object")
        self.compare("evaluation_receipt", str(relative), "outcome.status", "complete", outcome.get("status"))
        summary_relative = str(self.summary_path.resolve().relative_to(self.root))
        self.compare("evaluation_receipt", str(relative), "outcome.summary_sha256",
                     self.receipts[summary_relative]["sha256"], outcome.get("summary_sha256"))
        expected_integrity = {"protocol_lock_sha256": self.receipts[str(PROTOCOL_LOCK)]["sha256"],
                              "plan_sha256": self.receipts[str(PLAN)]["sha256"],
                              "completion_manifest_sha256": self.receipts[str(OUTPUT / "completion_manifest.json")]["sha256"],
                              "dataset_freezes": {d: self.receipts[str(OUTPUT / "frozen" / (d + ".json"))]["sha256"] for d in DATASETS},
                              "source_files": lock["source_files"], "input_files": lock["input_files"],
                              "prediction_files": {d: {v + ".json": self.receipts[str(OUTPUT / "predictions" / d / (v + ".json"))]["sha256"]
                                                       for v in VARIANTS} for d in DATASETS},
                              "supporting_files": {d: {str(OUTPUT / kind / (d + suffix)): self.receipts[str(OUTPUT / kind / (d + suffix))]["sha256"]
                                                       for kind, suffix in (("audit", ".jsonl"), ("diagnostics", ".json"))} for d in DATASETS}}
        self.compare("evaluation_receipt", str(relative), "integrity", expected_integrity, receipt.get("integrity"))
        evaluator_sources = receipt.get("evaluator_sources")
        require(isinstance(evaluator_sources, dict), "evaluation_receipt_source_hash_object")
        expected_evaluator_paths = {Path("scripts/evaluate_unified_loop_v5.py"), Path("evaluate_e2e.py"), EVALUATION_PLAN}
        expected_evaluator_paths |= {p.relative_to(self.root) for p in (self.root / "zeroclean").glob("*.py")}
        self.compare("evaluation_receipt", str(relative), "evaluator_source_set", sorted(str(p) for p in expected_evaluator_paths),
                     sorted(evaluator_sources))
        for path in sorted(expected_evaluator_paths):
            self.record_hash(path, evaluator_sources.get(str(path)), "evaluation_receipt.evaluator_sources")
        attempt_path = OUTPUT / "evaluation_attempt.json"
        self.record_hash(attempt_path, receipt.get("evaluation_attempt_sha256"), "evaluation_receipt.evaluation_attempt_sha256")
        attempt = read_json(self.root / attempt_path)
        require(isinstance(attempt, dict), "evaluation_attempt_object")
        attempt_fields = {"schema_version", "cohort", "datasets", "method_files", "prediction_count", "started_utc",
                          "integrity", "evaluator_sources", "prediction_shapes", "labels_used_only_in_evaluation",
                          "aggregate_only_reporting"}
        self.compare("evaluation_attempt", str(attempt_path), "field_set", sorted(attempt_fields), sorted(attempt))
        for field, expected in (("schema_version", "unified-loop-v5-evaluation-attempt-v1"), ("cohort", COHORT),
                                ("datasets", list(DATASETS)), ("method_files", [v + ".json" for v in VARIANTS]),
                                ("prediction_count", 18), ("integrity", expected_integrity),
                                ("evaluator_sources", evaluator_sources), ("prediction_shapes", receipt.get("prediction_shapes")),
                                ("labels_used_only_in_evaluation", True), ("aggregate_only_reporting", True)):
            self.compare("evaluation_attempt", str(attempt_path), field, expected, attempt.get(field))
        started = attempt.get("started_utc")
        require(isinstance(started, str), "evaluation_attempt_started_timestamp_type")
        try:
            started_time = datetime.fromisoformat(started)
        except (TypeError, ValueError):
            raise AuditError("evaluation_attempt_started_timestamp_format") from None
        require(started_time.tzinfo is not None, "evaluation_attempt_started_timestamp_timezone")
        self.compare("evaluation_attempt", str(attempt_path), "started_utc", started, receipt.get("started_utc"))
        baseline = receipt.get("baseline_status")
        require(isinstance(baseline, dict), "evaluation_receipt_baseline_hash_object")
        baseline_path = Path("results/source_baselines_v1/baseline_preflight.json")
        self.compare("evaluation_receipt", str(relative), "baseline_status.path", str(baseline_path), baseline.get("path"))
        self.record_hash(baseline_path, baseline.get("sha256"), "evaluation_receipt.baseline_status")
        limits = receipt.get("address_space_limits_bytes")
        require(isinstance(limits, list) and len(limits) == 2
                and all(type(value) is int and 0 < value <= MEMORY_BYTES for value in limits), "evaluation_receipt_address_space_limit")
        threads = receipt.get("numerical_threads")
        expected_threads = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}
        self.compare("evaluation_receipt", str(relative), "numerical_threads", expected_threads, threads)
        require(not self.mismatches, "evaluation_receipt_mismatch_no_references_opened")
        return receipt

    def prediction_preflight(self, lock, receipt):
        """Independently validate every stored output before any label read."""
        self.stage = "all_eighteen_prediction_preflight"
        shapes = receipt.get("prediction_shapes")
        require(isinstance(shapes, dict), "evaluation_receipt_prediction_shape_object")
        self.compare("evaluation_receipt", "prediction_shapes", "dataset_set", sorted(DATASETS), sorted(shapes))
        for dataset in DATASETS:
            columns, dirty = read_csv_exact(self.root / input_paths(dataset)["dirty"])
            require(bool(dirty), "empty_benchmark_table")
            rows, width = len(dirty), len(columns)
            self.compare("evaluation_receipt", dataset, "prediction_shape", [rows, width], shapes.get(dataset))
            for variant in VARIANTS:
                body = read_json(self.root / OUTPUT / "predictions" / dataset / (variant + ".json"))
                prediction_parts(body, rows, width)
                if variant == "zeroclean_loop":
                    metadata = body["metadata"]
                    for field, expected in (("system", "unified_loop_v5"), ("dataset", dataset), ("cohort", COHORT),
                                            ("model", MODEL), ("method_files", METHOD_FILES), ("ground_truth_access", False),
                                            ("frozen_before_evaluation", True),
                                            ("input_fingerprint", lock["input_files"][dataset]["dirty"]["sha256"]),
                                            ("plan_sha256", lock["plan_sha256"]),
                                            ("protocol_lock_sha256", self.receipts[str(PROTOCOL_LOCK)]["sha256"]),
                                            ("detection_mask_sha256", canonical_digest(metadata["upstream_binary_decisions"]))):
                        self.compare("prediction_metadata", dataset, field, expected, metadata.get(field))
            del dirty, body
        require(not self.mismatches, "all_prediction_preflight_mismatch_no_references_opened")

    def reference_bindings(self, receipt):
        """Hash all six evaluator-only references before parsing any reference CSV."""
        self.stage = "all_six_reference_hash_bindings"
        references = receipt.get("reference_sha256")
        require(isinstance(references, dict), "evaluation_receipt_reference_hash_object")
        self.compare("evaluation_receipt", "references", "dataset_set", sorted(DATASETS), sorted(references))
        hashes = {}
        for dataset in DATASETS:
            relative = Path(f"vendor/ZeroED/data/{dataset}_clean.csv")
            hashes[dataset] = self.record_hash(relative, references.get(dataset), "evaluation_receipt.reference_sha256")
        require(not self.mismatches, "reference_hash_mismatch_no_reference_csv_parsed")
        return hashes

    def check_summary(self, summary, recounted):
        self.stage = "summary_comparison"
        require(isinstance(summary, dict), "summary_object_required")
        self.compare("summary", "summary_metrics", "measured_datasets", list(DATASETS), summary.get("measured_datasets"))
        self.compare("summary", "summary_metrics", "six_dataset_benchmark_complete", True, summary.get("six_dataset_benchmark_complete"))
        self.compare("summary", "summary_metrics", "status", "measured_experiment_qualified_source_reconstruction", summary.get("status"))
        metrics = summary.get("metrics")
        require(isinstance(metrics, dict), "summary_metrics_object_required")
        self.compare("summary", "summary_metrics", "dataset_set", sorted(DATASETS), sorted(metrics))
        for dataset in DATASETS:
            dataset_metrics = metrics.get(dataset)
            require(isinstance(dataset_metrics, dict), "summary_dataset_object_required")
            self.compare("summary", dataset, "variant_set", sorted(VARIANTS), sorted(dataset_metrics))
            for variant in VARIANTS:
                reported = dataset_metrics.get(variant)
                require(isinstance(reported, dict), "summary_variant_object_required")
                actual = recounted[dataset][variant]
                label = dataset + "/" + variant
                for task in ("detection", "repair"):
                    require(isinstance(reported.get(task), dict), "summary_task_object_required")
                    for field in ("tp", "fp", "fn", "predicted", "actual", "precision", "recall", "f1"):
                        self.compare("summary", label, task + "." + field, actual[task][field],
                                     reported[task].get(field), ratio=field in ("precision", "recall", "f1"))
                for field in ("rows", "columns", "dirty_sha256", "reference_sha256", "reference_policy"):
                    self.compare("summary", label, field, actual[field], reported.get(field))
        macro = {variant: {task: sum((recounted[d][variant][task]["f1"] for d in DATASETS), Fraction(0)) / 6
                           for task in ("detection", "repair")} for variant in VARIANTS}
        reported_macro = summary.get("macro_f1")
        require(isinstance(reported_macro, dict), "summary_macro_object_required")
        self.compare("summary", "macro_f1", "variant_set", sorted(VARIANTS), sorted(reported_macro))
        for variant in VARIANTS:
            require(isinstance(reported_macro.get(variant), dict), "summary_macro_variant_object_required")
            for task in ("detection", "repair"):
                self.compare("summary", variant, "macro_f1." + task, macro[variant][task], reported_macro[variant].get(task), ratio=True)
        ed_gains = {d: recounted[d]["zeroclean_loop"]["detection"]["f1"] - recounted[d]["zeroed"]["detection"]["f1"] for d in DATASETS}
        repair_gains = {d: recounted[d]["zeroclean_loop"]["repair"]["f1"] - recounted[d]["zeroed_zeroec_reconstruction"]["repair"]["f1"] for d in DATASETS}
        checks, ed_macro_gain, repair_macro_gain = target_checks(ed_gains, repair_gains,
            recounted["flights"]["zeroclean_loop"]["detection"]["f1"],
            recounted["rayyan"]["zeroclean_loop"]["detection"]["f1"])
        qualified = summary.get("qualified_source_reconstruction_comparison")
        require(isinstance(qualified, dict), "summary_qualified_comparison_object_required")
        for field, expected in (("status", "verified"), ("met", all(checks.values())),
                                ("official_exact_baseline", False), ("fixed_example_count", 0), ("targets", TARGETS)):
            self.compare("summary", "qualified_comparison", field, expected, qualified.get(field))
        require(isinstance(qualified.get("checks"), dict), "summary_checks_object_required")
        self.compare("summary", "qualified_comparison", "check_set", sorted(checks), sorted(qualified["checks"]))
        for field, value in checks.items():
            self.compare("summary", "qualified_comparison", "checks." + field, value, qualified["checks"].get(field))
        for field, values in (("ed_gains", ed_gains), ("repair_gains", repair_gains)):
            require(isinstance(qualified.get(field), dict), "summary_gain_object_required")
            self.compare("summary", "qualified_comparison", field + ".dataset_set", sorted(DATASETS), sorted(qualified[field]))
            for dataset, value in values.items():
                self.compare("summary", "qualified_comparison", field + "." + dataset, value, qualified[field].get(dataset), ratio=True)
        for field, value in (("ed_macro_gain", ed_macro_gain), ("repair_macro_gain", repair_macro_gain)):
            self.compare("summary", "qualified_comparison", field, value, qualified.get(field), ratio=True)
        official = summary.get("official_target_assessment")
        require(isinstance(official, dict), "summary_official_assessment_object_required")
        for field, expected in (("met", False), ("status", "unverified_missing_baselines_or_runs"),
                                ("missing", [d + "/zeroed_zeroec" for d in DATASETS]), ("targets", TARGETS)):
            self.compare("summary", "official_target_assessment", field, expected, official.get(field))
        self.compare("summary", "summary_metrics", "targets_met", False, summary.get("targets_met"))
        self.aggregate = {"macro_f1_exact": {v: {task: rational(value) for task, value in tasks.items()} for v, tasks in macro.items()},
                          "independent_qualified_comparison": {"status": "verified", "met": all(checks.values()),
                              "checks": checks, "targets": TARGETS, "ed_macro_gain_exact": rational(ed_macro_gain),
                              "repair_macro_gain_exact": rational(repair_macro_gain),
                              "ed_gains_exact": {d: rational(value) for d, value in ed_gains.items()},
                              "repair_gains_exact": {d: rational(value) for d, value in repair_gains.items()},
                              "official_exact_baseline": False, "fixed_example_count": 0},
                          "official_exact_baseline_assessment": {"status": "unverified_missing_baselines_or_runs", "met": False,
                              "reason": "Faithful exact ZeroEC baseline absent; recovered prompt identity remains unproven."}}

    def run(self):
        error = None
        try:
            if not self.fictional:
                require(HEAVY_CONTEXT_ACTIVE, "exclusive_heavy_context_required")
                limits = resource.getrlimit(resource.RLIMIT_AS)
                require(all(type(value) is int and 0 < value <= MEMORY_BYTES for value in limits),
                        "twelve_gib_address_space_limits_required")
                require(all(os.environ.get(name) == "1" for name in THREAD_VARIABLES + ("OMP_THREAD_LIMIT",))
                        and os.environ.get("OMP_DYNAMIC") == "FALSE", "single_numerical_thread_required")
                self.resource_verified = True
            # A summary's existence is mandatory before the first label read.
            require(self.summary_path.is_file(), "initial_scorer_summary_missing")
            summary_relative = self.summary_path.resolve().relative_to(self.root)
            self.record_hash(summary_relative, binding="initial_scorer_summary")
            lock = self.frozen_integrity()
            receipt = self.evaluation_receipt(lock)
            summary = read_json(self.summary_path)
            self.record_hash(Path("scripts/audit_unified_loop_v5_results.py"), binding="independent_auditor_source")
            self.prediction_preflight(lock, receipt)
            self.stage = "pre_reference_hash_stability"
            for relative, binding in self.receipts.items():
                path = self.root / relative
                actual = ABSENT_INITIALIZER if binding["sha256"] == ABSENT_INITIALIZER and not path.exists() else sha256(path)
                self.compare("integrity", relative, "pre_reference_hash", binding["sha256"], actual)
            require(not self.mismatches, "pre_reference_hash_drift_no_references_opened")
            reference_hashes = self.reference_bindings(receipt)
            recounted = {}
            for dataset in DATASETS:
                self.stage = "recount_" + dataset
                dirty_relative = Path(input_paths(dataset)["dirty"])
                reference_relative = Path(f"vendor/ZeroED/data/{dataset}_clean.csv")
                reference_hash = reference_hashes[dataset]
                columns, dirty = read_csv_exact(self.root / dirty_relative)
                reference_columns, clean = read_csv_exact(self.root / reference_relative)
                require(columns == reference_columns and len(dirty) == len(clean), "reference_alignment")
                require(bool(dirty), "empty_benchmark_table")
                recounted[dataset] = {}
                for variant in VARIANTS:
                    body = read_json(self.root / OUTPUT / "predictions" / dataset / (variant + ".json"))
                    actual = recount_prediction(body, dirty, clean)
                    actual.update(dirty_sha256=lock["input_files"][dataset]["dirty"]["sha256"],
                                  reference_sha256=reference_hash, reference_policy="exact strings; no normalization")
                    recounted[dataset][variant] = actual
                    self.evaluated += 1
                del dirty, clean, body
            self.check_summary(summary, recounted)
            self.stage = "post_recount_hash_stability"
            for relative, receipt in self.receipts.items():
                path = self.root / relative
                actual = ABSENT_INITIALIZER if receipt["sha256"] == ABSENT_INITIALIZER and not path.exists() else sha256(path)
                self.compare("integrity", relative, "post_recount_hash", receipt["sha256"], actual)
            require(self.evaluated == 18, "expected_eighteen_outputs")
        except AuditError as exc:
            error = str(exc)
        except Exception as exc:
            # Avoid exception strings: parser/runtime errors can contain cell data.
            error = "unexpected_" + type(exc).__name__
        if error:
            self.mismatches.append({"category": "audit_execution", "stage": self.stage, "code": error})
        return {"schema_version": "unified-loop-v5-independent-audit-v1",
                "status": "passed" if not self.mismatches and self.evaluated == 18 else "failed",
                "time_utc": datetime.now(timezone.utc).isoformat(), "evaluated_output_count": self.evaluated,
                "expected_output_count": 18, "mismatches": self.mismatches,
                "comparison": {"integer_fields_checked": self.integer_fields, "ratio_fields_checked": self.ratio_fields,
                               "ratio_absolute_tolerance": TOLERANCE, "macro_and_target_arithmetic": "exact rational"},
                "resource_policy": {"exclusive_lock": str(HEAVY_LOCK), "address_space_limit_bytes": MEMORY_BYTES,
                                    "numerical_threads": 1, "execution_context_verified": self.resource_verified},
                "information_boundary": {"evaluator_only": True, "row_values_or_coordinates_returned": False,
                                         "scoring_or_inference_modules_imported": False, "inference_recalculated": False},
                "hash_verification": {"file_or_absence_count": len(self.receipts),
                    "post_recount_stability_checked": self.stage == "post_recount_hash_stability" and error is None,
                    "receipt": self.receipts}, **self.aggregate}


@contextmanager
def heavy_execution():
    global HEAVY_CONTEXT_ACTIVE
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with HEAVY_LOCK.open("a+", encoding="utf-8") as handle:
        print("Waiting for exclusive heavy-experiment lock", flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            for name in THREAD_VARIABLES + ("OMP_THREAD_LIMIT",):
                os.environ[name] = "1"
            os.environ["OMP_DYNAMIC"] = "FALSE"
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min(MEMORY_BYTES, hard) if hard != resource.RLIM_INFINITY else MEMORY_BYTES
            if soft != resource.RLIM_INFINITY:
                limit = min(limit, soft)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            HEAVY_CONTEXT_ACTIVE = True
            print("Exclusive lock acquired; 12-GiB maximum and one numerical thread applied", flush=True)
            yield
        finally:
            HEAVY_CONTEXT_ACTIVE = False
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def markdown_report(result):
    lines = ["# Unified Loop v5 independent aggregate audit", "",
             f"Status: **{result['status']}**. Evaluated outputs: **{result['evaluated_output_count']}/18**.",
             f"Mismatches: **{len(result['mismatches'])}**. Hash receipts: **{result['hash_verification']['file_or_absence_count']}**.", "",
             "Direct standard-library CSV/JSON recount; exact strings; stored strict `p > 0.5` masks; changed edits inside the mask.",
             "Repair recall includes every original error. Wrong edits are FP; unrepaired or wrongly repaired original errors are FN.", ""]
    qualified = result.get("independent_qualified_comparison")
    if qualified:
        lines += ["## Independent qualified comparison", "",
                  f"All five quantitative checks: **{'passed' if qualified['met'] else 'not all passed'}**.", ""]
        for key, passed in qualified["checks"].items():
            lines.append(f"- `{key}`: {'pass' if passed else 'fail'}")
        lines.append("")
        for key in ("ed_macro_gain_exact", "repair_macro_gain_exact"):
            value = qualified[key]
            lines.append(f"- `{key}`: {value['numerator']}/{value['denominator']} = {value['decimal']:.12f}")
        lines += ["", "Exact macro F1 values and per-dataset aggregate gains are recorded as rational numbers in the JSON audit.", ""]
    lines += ["Official exact ZeroEC success remains **unverified**: the faithful exact baseline is absent and recovered prompt identity remains unproven.",
              "The available sequential reconstruction is a qualified comparison, with zero fixed examples.",
              "The study follows prior aggregate feedback and is not a previously untouched benchmark validation.", "",
              "No row values, coordinates, labeled examples, or failure records are returned. Historical study artifacts are unchanged.", ""]
    if result["mismatches"]:
        lines += ["## Aggregate mismatch details", "", "```json", json.dumps(result["mismatches"], indent=2, sort_keys=True), "```", ""]
    lines += ["The full path/hash and package-initializer absence receipt is in `independent_audit.json`.", ""]
    return "\n".join(lines)


def publish_exclusive(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def self_test():
    """Fictional values only; never look at the workspace or acquire its lock."""
    checks_run = ["exact_detection_counts", "exact_repair_counts", "noop_edits_excluded"]
    dirty = [["01", "A", ""], ["x", "b", "q"]]
    clean = [["1", "A", ""], ["y", "B", "q"]]
    body = {"probabilities": [[.51, .9, .5], [.8, 0, .7]],
            "metadata": {"decision_threshold": .5, "decision_threshold_operator": ">",
                         "upstream_binary_decisions": [[True, 1, False], [True, False, True]]},
            "repairs": [{"row": 0, "column": 0, "value": "1"}, {"row": 0, "column": 1, "value": "Z"},
                        {"row": 1, "column": 0, "value": "wrong"}, {"row": 1, "column": 2, "value": "q"}]}
    result = recount_prediction(body, dirty, clean)
    require(result["detection"] == {"tp": 2, "fp": 2, "fn": 1, "predicted": 4, "actual": 3,
            "precision": Fraction(1, 2), "recall": Fraction(2, 3), "f1": Fraction(4, 7)}, "fictional_detection_counts")
    require(result["repair"] == {"tp": 1, "fp": 2, "fn": 2, "predicted": 3, "actual": 3,
            "precision": Fraction(1, 3), "recall": Fraction(1, 3), "f1": Fraction(1, 3)}, "fictional_repair_counts")
    require(result["declared_noop_repairs"] == 1, "fictional_noop_not_tp")
    for mutation, expected in (("boundary", "stored_mask_strict_threshold_mismatch"),
                               ("outside", "edit_outside_stored_mask"),
                               ("duplicate", "duplicate_repair_coordinate"),
                               ("null", "repair_string_required")):
        copy = json.loads(json.dumps(body))
        if mutation == "boundary":
            copy["metadata"]["upstream_binary_decisions"][0][2] = True
        elif mutation == "outside":
            copy["repairs"].append({"row": 1, "column": 1, "value": "B"})
        elif mutation == "duplicate":
            copy["repairs"].append(copy["repairs"][0])
        else:
            copy["repairs"][0]["value"] = None
        try:
            recount_prediction(copy, dirty, clean)
        except AuditError as exc:
            require(str(exc) == expected, "fictional_wrong_rejection")
        else:
            raise AuditError("fictional_missing_rejection")
        checks_run.append("reject_" + mutation)
    require(prf(0, 0, 0)["f1"] == 0, "fictional_zero_denominators")
    checks_run.append("zero_denominators")
    ed_gains = {dataset: Fraction(35, 1000) for dataset in DATASETS}
    repair_gains = {dataset: Fraction(8, 100) for dataset in DATASETS}
    targets, _, _ = target_checks(ed_gains, repair_gains, Fraction(76, 100), Fraction(77, 100))
    require(targets == {"ed_mean_gain": True, "ed_each_dataset": True, "flights": False,
                        "rayyan": False, "repair_mean_gain": True}, "fictional_exact_target_boundaries")
    epsilon = Fraction(1, 10**20)
    targets, _, _ = target_checks(ed_gains, repair_gains, Fraction(76, 100) + epsilon, Fraction(77, 100) + epsilon)
    require(all(targets.values()), "fictional_target_strict_above")
    targets, _, _ = target_checks({d: value - epsilon for d, value in ed_gains.items()},
        {d: value - epsilon for d, value in repair_gains.items()}, Fraction(1), Fraction(1))
    require(not targets["ed_mean_gain"] and not targets["repair_mean_gain"], "fictional_target_below_mean")
    targets, _, _ = target_checks({d: Fraction(0) if index == 0 else Fraction(42, 1000)
                                  for index, d in enumerate(DATASETS)}, repair_gains, Fraction(1), Fraction(1))
    require(targets["ed_mean_gain"] and not targets["ed_each_dataset"], "fictional_each_dataset_strict")
    checks_run += ["exact_target_boundaries", "strict_targets_above_by_rational_epsilon",
                   "mean_targets_below_by_rational_epsilon", "each_dataset_strict_improvement"]
    with tempfile.TemporaryDirectory(prefix="unified-v4-auditor-fictional-") as directory:
        path = Path(directory) / "fictional.csv"
        path.write_bytes(b'\xef\xbb\xbfkey,value,empty\r\n01,"a\r\nb",\r\n')
        columns, rows = read_csv_exact(path)
        require(columns == ["key", "value", "empty"] and rows == [["01", "a\r\nb", ""]], "fictional_csv_exactness")
    checks_run.append("csv_exactness")
    checks_run += fictional_integration(body, dirty, clean)
    print(json.dumps({"status": "passed", "fictional_check_count": len(checks_run), "fictional_checks": checks_run,
                      "real_study_reads": False, "real_reference_reads": False}, sort_keys=True))


def fictional_integration(prediction, dirty, clean):
    """An entire invented cohort tests the gates, 18-output comparison and drift."""
    with tempfile.TemporaryDirectory(prefix="unified-v4-auditor-cohort-fictional-") as directory:
        root = Path(directory)

        def save(path, value):
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(json_bytes(value))

        def text(path, value="fictional source\n"):
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(value, encoding="utf-8")

        def table(path, rows):
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["first", "second", "third"])
                writer.writerows(rows)

        def no_reference_csv_audit():
            original_reader = globals()["read_csv_exact"]
            parsed_references = []

            def guarded_reader(path):
                if Path(path).name.endswith("_clean.csv"):
                    parsed_references.append(str(path))
                return original_reader(path)

            globals()["read_csv_exact"] = guarded_reader
            try:
                outcome = Auditor(root, _fictional=True).run()
            finally:
                globals()["read_csv_exact"] = original_reader
            require(not parsed_references, "fictional_gate_parsed_reference_csv")
            require(outcome["status"] == "failed" and outcome["evaluated_output_count"] == 0,
                    "fictional_gate_failed_to_stop")
            return outcome

        study_names = ("run_study.py", "detection.py", "consensus.py", "repair.py", "protocol.md", "runner_design.md", "test_integration.py")
        source_paths = {STUDY / name for name in study_names}
        source_paths |= {Path("zeroclean") / name for name in ("data.py", "types.py", "statistics.py", "sandbox.py")}
        for path in source_paths | {EVALUATION_PLAN, Path("scripts/evaluate_unified_loop_v5.py"),
                                    Path("scripts/audit_unified_loop_v5_results.py"), Path("evaluate_e2e.py")}:
            text(path)
        sources = {str(path): sha256(root / path) for path in source_paths}
        sources.update({str(path): ABSENT_INITIALIZER for path in INITIALIZERS})
        inputs = {}
        for dataset in DATASETS:
            paths = input_paths(dataset)
            table(paths["dirty"], dirty)
            table(Path(f"vendor/ZeroED/data/{dataset}_clean.csv"), clean)
            for role in ("pass2", "old_pass2", "fresh_packet", "fresh_response", "cts", "empty"):
                save(paths[role], {"fictional": True})
            for role in ("source", "sequential"):
                save(paths[role], prediction)
            inputs[dataset] = {role: {"path": path, "sha256": sha256(root / path)} for role, path in paths.items()}
        common = {"schema_version": SCHEMA, "datasets": list(DATASETS), "method_files": METHOD_FILES,
                  "model": MODEL, "cohort": COHORT, "ground_truth_access": False}
        plan = {**common, "configuration": {"fictional": True}, "design_provenance": {"fictional": True},
                "evaluation_is_separate": True, "output_root": str(OUTPUT),
                "allowed_inference_inputs": {d: input_paths(d) for d in DATASETS},
                "protocol_sha256": sources[str(STUDY / "protocol.md")]}
        save(PLAN, plan)
        lock = {**common, "plan_sha256": sha256(root / PLAN), "source_files": sources,
                "input_files": inputs, "frozen_before_inference": True}
        save(PROTOCOL_LOCK, lock)
        lock_hash, plan_hash = sha256(root / PROTOCOL_LOCK), sha256(root / PLAN)
        freezes, predictions, supporting = {}, {}, {}
        for dataset in DATASETS:
            predictions[dataset] = {}
            for variant in VARIANTS:
                body = json.loads(json.dumps(prediction))
                if variant == "zeroclean_loop":
                    body["metadata"].update(system="unified_loop_v5", dataset=dataset, cohort=COHORT, model=MODEL,
                        method_files=METHOD_FILES, ground_truth_access=False, frozen_before_evaluation=True,
                        input_fingerprint=inputs[dataset]["dirty"]["sha256"], plan_sha256=plan_hash,
                        protocol_lock_sha256=lock_hash,
                        detection_mask_sha256=canonical_digest(body["metadata"]["upstream_binary_decisions"]))
                path = OUTPUT / "predictions" / dataset / (variant + ".json")
                save(path, body)
                predictions[dataset][variant + ".json"] = sha256(root / path)
            supporting[dataset] = {}
            for kind, suffix in (("audit", ".jsonl"), ("diagnostics", ".json")):
                path = OUTPUT / kind / (dataset + suffix)
                text(path, "{}\n")
                supporting[dataset][str(path)] = sha256(root / path)
            frozen = {"dirty_sha256": inputs[dataset]["dirty"]["sha256"], "prediction_files": predictions[dataset],
                      "supporting_files": supporting[dataset], "protocol_lock_sha256": lock_hash, "plan_sha256": plan_hash,
                      "source_signature": canonical_digest(sources), "config": plan["configuration"],
                      "design_provenance": plan["design_provenance"], "method_files": METHOD_FILES,
                      "model": MODEL, "cohort": COHORT, "frozen_before_evaluation": True, "ground_truth_access": False,
                      "inference_inputs": {str(root / item["path"]): item["sha256"] for item in inputs[dataset].values()}}
            path = OUTPUT / "frozen" / (dataset + ".json")
            save(path, frozen)
            freezes[dataset] = sha256(root / path)
        completion = {**common, "protocol_lock_sha256": lock_hash, "dataset_freezes": freezes,
                      "six_dataset_inference_complete": True, "evaluation_performed": False}
        save(OUTPUT / "completion_manifest.json", completion)
        known = {"detection": {"tp": 2, "fp": 2, "fn": 1, "predicted": 4, "actual": 3,
                  "precision": .5, "recall": 2 / 3, "f1": 4 / 7},
                 "repair": {"tp": 1, "fp": 2, "fn": 2, "predicted": 3, "actual": 3,
                  "precision": 1 / 3, "recall": 1 / 3, "f1": 1 / 3}, "rows": 2, "columns": 3,
                 "reference_policy": "exact strings; no normalization"}
        official = {"met": False, "status": "unverified_missing_baselines_or_runs",
                    "missing": [d + "/zeroed_zeroec" for d in DATASETS], "targets": TARGETS}
        qualified = {"met": False, "status": "verified", "targets": TARGETS,
                     "checks": {key: False for key in ("ed_mean_gain", "ed_each_dataset", "flights", "rayyan", "repair_mean_gain")},
                     "ed_gains": {d: 0 for d in DATASETS}, "repair_gains": {d: 0 for d in DATASETS},
                     "ed_macro_gain": 0, "repair_macro_gain": 0, "official_exact_baseline": False, "fixed_example_count": 0}
        references = {d: sha256(root / f"vendor/ZeroED/data/{d}_clean.csv") for d in DATASETS}
        summary = {"status": "measured_experiment_qualified_source_reconstruction", "targets_met": False,
                   "measured_datasets": list(DATASETS), "six_dataset_benchmark_complete": True,
                   "macro_f1": {v: {"detection": 4 / 7, "repair": 1 / 3} for v in VARIANTS},
                   "official_target_assessment": official, "qualified_source_reconstruction_comparison": qualified,
                   "metrics": {d: {v: {**known, "dirty_sha256": inputs[d]["dirty"]["sha256"],
                                       "reference_sha256": references[d]} for v in VARIANTS} for d in DATASETS}}
        summary_path = OUTPUT / "summary_metrics.json"
        save(summary_path, summary)
        baseline_path = Path("results/source_baselines_v1/baseline_preflight.json")
        save(baseline_path, {"fictional": True})
        evaluator_paths = {Path("scripts/evaluate_unified_loop_v5.py"), Path("evaluate_e2e.py"), EVALUATION_PLAN}
        evaluator_paths |= {path for path in source_paths if path.parent == Path("zeroclean")}
        receipt = {"schema_version": "unified-loop-v5-evaluation-receipt-v1", "cohort": COHORT,
                   "datasets": list(DATASETS), "method_files": [v + ".json" for v in VARIANTS], "prediction_count": 18,
                   "integrity": {"protocol_lock_sha256": lock_hash, "plan_sha256": plan_hash,
                       "completion_manifest_sha256": sha256(root / OUTPUT / "completion_manifest.json"),
                       "dataset_freezes": freezes, "source_files": sources, "input_files": inputs,
                       "prediction_files": predictions, "supporting_files": supporting},
                   "evaluator_sources": {str(path): sha256(root / path) for path in evaluator_paths},
                   "prediction_shapes": {d: [2, 3] for d in DATASETS}, "reference_sha256": references,
                   "baseline_status": {"path": str(baseline_path), "sha256": sha256(root / baseline_path)},
                   "heavy_lock": str(HEAVY_LOCK), "address_space_limits_bytes": [MEMORY_BYTES, MEMORY_BYTES],
                   "numerical_threads": {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                       "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
                   "labels_used_only_in_evaluation": True, "aggregate_only_reporting": True,
                   "outcome": {"status": "complete", "summary_sha256": sha256(root / summary_path)}}
        started = "2026-10-28T12:00:00+00:00"
        attempt = {"schema_version": "unified-loop-v5-evaluation-attempt-v1", "cohort": COHORT,
                   "datasets": list(DATASETS), "method_files": [v + ".json" for v in VARIANTS], "prediction_count": 18,
                   "started_utc": started, "integrity": receipt["integrity"], "evaluator_sources": receipt["evaluator_sources"],
                   "prediction_shapes": receipt["prediction_shapes"], "labels_used_only_in_evaluation": True,
                   "aggregate_only_reporting": True}
        save(OUTPUT / "evaluation_attempt.json", attempt)
        receipt.update(started_utc=started, evaluation_attempt_sha256=sha256(root / OUTPUT / "evaluation_attempt.json"))
        save(OUTPUT / "evaluation_receipt.json", receipt)
        unguarded = Auditor(root).run()
        require(unguarded["status"] == "failed" and unguarded["evaluated_output_count"] == 0
                and not unguarded["hash_verification"]["receipt"]
                and any(item.get("code") == "exclusive_heavy_context_required" for item in unguarded["mismatches"]),
                "fictional_missing_heavy_context_rejected")
        outcome = Auditor(root, _fictional=True).run()
        require(outcome["status"] == "passed" and outcome["evaluated_output_count"] == 18 and not outcome["mismatches"],
                "fictional_complete_cohort_audit")
        receipt["outcome"]["status"] = "incomplete"
        save(OUTPUT / "evaluation_receipt.json", receipt)
        gated = no_reference_csv_audit()
        require(not any(path.endswith("_clean.csv") for path in gated["hash_verification"]["receipt"]),
                "fictional_unsuccessful_receipt_opened_reference")
        receipt["outcome"]["status"] = "complete"
        save(OUTPUT / "evaluation_receipt.json", receipt)
        (root / OUTPUT / "evaluation_attempt.json").unlink()
        no_reference_csv_audit()
        save(OUTPUT / "evaluation_attempt.json", attempt)
        extra = OUTPUT / "predictions" / "extra" / "extra.json"
        save(extra, {"fictional": True})
        no_reference_csv_audit()
        (root / extra).unlink()
        (root / extra.parent).rmdir()
        expected_reference = receipt["reference_sha256"]["movies"]
        receipt["reference_sha256"]["movies"] = "0" * 64
        save(OUTPUT / "evaluation_receipt.json", receipt)
        no_reference_csv_audit()
        receipt["reference_sha256"]["movies"] = expected_reference
        save(OUTPUT / "evaluation_receipt.json", receipt)
        summary["metrics"]["hospital"]["zeroed"]["detection"] = {**known["detection"], "tp": 99}
        save(summary_path, summary)
        receipt["outcome"]["summary_sha256"] = sha256(root / summary_path)
        save(OUTPUT / "evaluation_receipt.json", receipt)
        outcome = Auditor(root, _fictional=True).run()
        require(outcome["status"] == "failed" and outcome["evaluated_output_count"] == 18
                and any(item.get("field") == "detection.tp" for item in outcome["mismatches"]), "fictional_count_mismatch_detected")
        text(OUTPUT / "predictions" / "hospital" / "zeroed.json", "tampered fictional bytes")
        outcome = Auditor(root, _fictional=True).run()
        require(outcome["status"] == "failed" and outcome["evaluated_output_count"] == 0
                and any(item.get("category") == "integrity" for item in outcome["mismatches"]), "fictional_hash_drift_gates_labels")
        return ["missing_heavy_context_gates_artifact_reads", "complete_fictional_eighteen_output_cohort",
                "unsuccessful_scorer_receipt_gates_references",
                "missing_attempt_marker_gates_references", "extra_prediction_inventory_gates_references",
                "all_reference_hashes_gate_csv_parsing", "integer_count_mismatch_detected",
                "prediction_hash_drift_gates_references"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--run-after-scoring-complete", action="store_true",
                      help="Use only after the parent confirms all six freezes and initial scoring are complete")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    root = args.root.resolve()
    output = root / OUTPUT
    destinations = [output / "independent_audit.json", output / "independent_audit.md"]
    require(not any(path.exists() for path in destinations), "refusing_to_overwrite_existing_independent_audit")
    with heavy_execution():
        result = Auditor(root, args.summary).run()
        publish_exclusive(destinations[0], json_bytes(result))
        publish_exclusive(destinations[1], markdown_report(result).encode("utf-8"))
    print(json.dumps({"status": result["status"], "evaluated_output_count": result["evaluated_output_count"],
                      "mismatch_count": len(result["mismatches"]), "json": str(destinations[0]), "markdown": str(destinations[1])}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(json.dumps({"status": "failed", "code": str(exc)}, sort_keys=True))
        raise SystemExit(1) from None
