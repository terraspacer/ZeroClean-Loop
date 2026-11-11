#!/usr/bin/env python3
"""Freeze the six-table protocol, then run serialized dirty-only inference.

This module deliberately has no evaluator import and no import-time scientific
dependencies. The CLI applies resource limits before loading inference modules.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from types import SimpleNamespace
import unicodedata


DATASETS = ("hospital", "flights", "beers", "rayyan", "billionaire", "movies")
ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("experimental/unified_loop_v2")
OUTPUT = Path("results/unified_loop_v2")
PLAN = STUDY / "plan.json"
LOCK = STUDY / "protocol_lock.json"
HEAVY_LOCK = Path(os.environ.get("ZEROCLEAN_HEAVY_LOCK", "/tmp/zeroclean-heavy-experiment.lock"))
COHORT = "local-unified-loop-v2-20261028"
MODEL = {"model": "gpt-6.1-sol", "reasoning_effort": "xhigh"}
METHOD_FILES = {"unified_loop_v2": "zeroclean_loop.json"}
SCHEMA_VERSION = "unified-loop-v2-freeze-v1"
PACKAGE_INITIALIZERS = (Path("zeroclean/__init__.py"), Path("experimental/__init__.py"), STUDY / "__init__.py")
ABSENT_SOURCE = "ABSENT:package-initializer"
MEMORY_BYTES = 12 * 1024**3
RUNNER_CONFIG = {
    "rule_batch_rows": 1000,
    "counterfactual_batch_rows": 1000,
    "rule_max_error_rate_exclusive": 0.05,
    "candidate_cap": 3,
    "decision_threshold": 0.5,
    "decision_threshold_operator": ">",
    "vicinity_difference_divisor": 4.0,
    "domain_difference_divisor": 4.0,
    "minimum_dirty_pattern_exemplars": 2,
    "combined_row_guard": "rollback_all_edits_on_new_violation_or_runtime_error",
    "detection_all_true_rule_policy": "exclude_from_detection_keep_in_guard",
    "constant_false_rule_policy": "retain_as_counterfactual_guard",
    "rule_delta_scope": "accepted_rules_targeting_proposed_column",
    "fd_delta": "leave_one_out_candidate_conditional_support_minus_original",
    "format_policy": "same_original_pattern_or_two_dirty_exemplars_no_controls",
    "empty_carrier_fields": ["probabilities", "metadata"],
    "seed_source": "source_sequential_repair_map_only",
    "candidate_order": ["actionable_key_source_consensus", "seed", "cts_existing_order"],
    "cts_witness_provenance": "complete_cell_cts_only_peer_union_before_candidate_filtering",
    "blas_openmp_threads": 1,
    "address_space_bytes": MEMORY_BYTES,
}
PROVENANCE = {
    "design_origin": "user_directed_follow_up_informed_by_prior_aggregate_observations",
    "fresh_blind_experiment": False,
    "row_labels_used_for_design_or_inference": False,
    "evaluation_failure_reports_used": False,
    "external_metadata_or_catalogs_used": False,
    "prior_reviewer_judgments_used": False,
    "existing_cached_proposals_are_noisy_proposals": True,
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def _json_bytes(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _exclusive_bytes(path, body):
    """Publish complete bytes atomically, never replacing an existing path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _exclusive_json(path, body):
    _exclusive_bytes(path, _json_bytes(body))


def _exclusive_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                handle.write(_canonical(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _same_or_create(path, body):
    expected = _json_bytes(body)
    if Path(path).exists():
        if Path(path).read_bytes() != expected:
            raise ValueError(f"Refusing to change immutable artifact: {path}")
    else:
        _exclusive_bytes(path, expected)


def input_paths(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return {
        "dirty": f"vendor/ZeroED/data/{dataset}_error-01.csv",
        "pass2": f"artifacts/llm/{dataset}/pass2.json",
        "cts": f"artifacts/llm/{dataset}/proposals_cts_v2.json",
        "source": f"results/source_baselines_v1/predictions/{dataset}/zeroed.json",
        "sequential": f"results/source_baselines_v1/predictions/{dataset}/zeroed_zeroec_reconstruction.json",
        "empty": f"results/conservative_seed_v1/predictions/{dataset}/empty_carry.json",
    }


def source_manifest(root):
    root = Path(root)
    required = [STUDY / name for name in ("run_study.py", "detection.py", "consensus.py", "repair.py", "protocol.md", "runner_design.md", "test_integration.py")]
    for path in required:
        if not (root / path).is_file():
            raise ValueError(f"Source/protocol must exist before global freeze: {path}")
    paths = set(required)
    paths.update(path.relative_to(root) for extension in ("*.py", "*.md") for path in (root / STUDY).glob(extension))
    paths.update(Path("zeroclean") / name for name in ("data.py", "types.py", "statistics.py", "sandbox.py"))
    paths.update(PACKAGE_INITIALIZERS)
    return {str(path): ABSENT_SOURCE if path in PACKAGE_INITIALIZERS and not (root / path).exists()
            else _sha256(root / path) for path in sorted(paths)}


def input_manifest(root):
    root = Path(root)
    return {name: {role: {"path": path, "sha256": _sha256(root / path)}
                   for role, path in input_paths(name).items()} for name in DATASETS}


def _runtime():
    """Call only after the execution environment has been bounded."""
    import numpy as np
    from zeroclean.data import load_dirty
    from zeroclean.statistics import DirtyIndex, pattern
    from zeroclean.types import Prediction
    from zeroclean import sandbox
    return SimpleNamespace(np=np, load_dirty=load_dirty, DirtyIndex=DirtyIndex, pattern=pattern,
                           Prediction=Prediction, sandbox=sandbox,
                           detection=importlib.import_module("experimental.unified_loop_v2.detection"),
                           consensus=importlib.import_module("experimental.unified_loop_v2.consensus"),
                           repair=importlib.import_module("experimental.unified_loop_v2.repair"))


def configuration_snapshot(runtime):
    detection = runtime.detection.DetectionConfig()
    consensus = dict(runtime.consensus.DEFAULT_CONFIG)
    repair = runtime.repair.DEFAULT_CONFIG
    return json.loads(_canonical({"runner": RUNNER_CONFIG, "detection": asdict(detection),
                                  "consensus": consensus, "repair": asdict(repair) if is_dataclass(repair) else dict(repair)}))


def _require_all_datasets(datasets):
    if len(datasets) != len(DATASETS) or set(datasets) != set(DATASETS):
        raise ValueError("A global freeze must cover all six datasets before any real inference")


def freeze_study(root, datasets=DATASETS, runtime=None):
    """Hash inputs only; do not evaluate rules, construct statistics, or infer."""
    _require_all_datasets(datasets)
    root = Path(root)
    runtime = _runtime() if runtime is None else runtime
    plan = {
        "schema_version": SCHEMA_VERSION, "datasets": list(DATASETS), "method_files": METHOD_FILES,
        "model": MODEL, "cohort": COHORT, "configuration": configuration_snapshot(runtime),
        "design_provenance": PROVENANCE, "ground_truth_access": False,
        "evaluation_is_separate": True, "output_root": str(OUTPUT),
        "allowed_inference_inputs": {name: input_paths(name) for name in DATASETS},
        "protocol_sha256": _sha256(root / STUDY / "protocol.md"),
    }
    sources, inputs = source_manifest(root), input_manifest(root)
    lock = {"schema_version": SCHEMA_VERSION, "datasets": list(DATASETS), "plan_sha256": hashlib.sha256(_json_bytes(plan)).hexdigest(),
            "source_files": sources, "input_files": inputs, "method_files": METHOD_FILES,
            "model": MODEL, "cohort": COHORT, "ground_truth_access": False,
            "frozen_before_inference": True}
    if (root / LOCK).exists():
        if _read_json(root / LOCK) != lock:
            raise ValueError("Existing protocol lock differs; a frozen experiment cannot be changed")
        verify_freeze(root)
        return lock
    if (root / OUTPUT).exists() and any((root / OUTPUT).iterdir()):
        raise ValueError("Refusing initial freeze after study output already exists")
    _same_or_create(root / PLAN, plan)
    _exclusive_json(root / LOCK, lock)
    return verify_freeze(root)


def verify_freeze(root):
    root = Path(root)
    if not (root / LOCK).is_file() or not (root / PLAN).is_file():
        raise ValueError("Global protocol freeze is required before inference or output")
    lock, plan = _read_json(root / LOCK), _read_json(root / PLAN)
    _require_all_datasets(lock.get("datasets", []))
    _require_all_datasets(plan.get("datasets", []))
    if (lock.get("schema_version") != SCHEMA_VERSION or plan.get("schema_version") != SCHEMA_VERSION
            or lock.get("method_files") != METHOD_FILES or plan.get("method_files") != METHOD_FILES
            or lock.get("model") != MODEL or plan.get("model") != MODEL
            or lock.get("cohort") != COHORT or plan.get("cohort") != COHORT
            or lock.get("ground_truth_access") is not False or plan.get("ground_truth_access") is not False
            or lock.get("frozen_before_inference") is not True):
        raise ValueError("Unsupported or changed frozen protocol")
    if _sha256(root / PLAN) != lock.get("plan_sha256"):
        raise ValueError("Frozen plan changed")
    if source_manifest(root) != lock.get("source_files"):
        raise ValueError("Frozen source/protocol files changed")
    if input_manifest(root) != lock.get("input_files"):
        raise ValueError("Frozen inference inputs changed")
    if plan.get("allowed_inference_inputs") != {name: input_paths(name) for name in DATASETS}:
        raise ValueError("Frozen input allowlist changed")
    return lock


def _probabilities_from_body(body, shape, np, *, label):
    if not isinstance(body, dict) or not isinstance(body.get("metadata"), dict):
        raise ValueError(f"{label}: prediction and metadata must be objects")
    raw = body.get("probabilities")
    if (not isinstance(raw, list) or len(raw) != shape[0]
            or any(not isinstance(row, list) or len(row) != shape[1] for row in raw)
            or any(type(value) not in (int, float) for row in raw for value in row)):
        raise ValueError(f"{label}: malformed probability matrix")
    probabilities = np.asarray(raw, dtype=float).reshape(shape)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{label}: probabilities must be finite and in [0,1]")
    metadata = body["metadata"]
    if type(metadata.get("decision_threshold")) not in (int, float) or metadata["decision_threshold"] != 0.5 or metadata.get("decision_threshold_operator") != ">":
        raise ValueError(f"{label}: only strict >0.5 decision semantics are allowed")
    if "upstream_binary_decisions" in metadata:
        mask = metadata["upstream_binary_decisions"]
        if (not isinstance(mask, list) or len(mask) != shape[0]
                or any(not isinstance(row, list) or len(row) != shape[1] for row in mask)
                or any(type(v) not in (bool, int) or v not in (0, 1) for row in mask for v in row)
                or not np.array_equal(np.asarray(mask, dtype=bool).reshape(shape), probabilities > 0.5)):
            raise ValueError(f"{label}: binary decisions disagree with strict >0.5 probabilities")
    return probabilities, metadata


def load_probability_carrier(path, shape, np):
    # Deliberately never access body['repairs'], including for empty_carry.
    return _probabilities_from_body(_read_json(path), shape, np, label=str(path))


def load_seed_repairs(path, shape, np):
    body = _read_json(path)
    _probabilities_from_body(body, shape, np, label=str(path))
    raw = body.get("repairs")
    if not isinstance(raw, list):
        raise ValueError("Sequential source repairs must be a list")
    result = {}
    for item in raw:
        if (not isinstance(item, dict) or type(item.get("row")) is not int or type(item.get("column")) is not int
                or not 0 <= item["row"] < shape[0] or not 0 <= item["column"] < shape[1]
                or not isinstance(item.get("value"), str)):
            raise ValueError("Invalid sequential source repair coordinate/value")
        coordinate = (item["row"], item["column"])
        if coordinate in result:
            raise ValueError("Duplicate sequential source repair coordinate")
        result[coordinate] = item["value"]
    return result


def _coordinate(table, row, column):
    if isinstance(column, str):
        try:
            column = table.columns.index(column)
        except ValueError as exc:
            raise ValueError("Unknown candidate column") from exc
    if type(row) is not int or type(column) is not int or not 0 <= row < table.shape[0] or not 0 <= column < table.shape[1]:
        raise ValueError("Invalid candidate coordinate")
    return row, column


def _peers(values, query, n):
    if not isinstance(values, (list, tuple)) or any(type(v) is not int or not 0 <= v < n for v in values):
        raise ValueError("Malformed dirty peer-row identifiers")
    return sorted(set(values) - {query})


def load_cts_cells(body, table):
    if not isinstance(body, dict) or not isinstance(body.get("cells"), list):
        raise ValueError("CTS artifact must contain cells")
    result = {}
    for item in body["cells"]:
        if not isinstance(item, dict) or not isinstance(item.get("candidates"), list):
            raise ValueError("Malformed CTS cell")
        coordinate = _coordinate(table, item.get("row"), item.get("column"))
        if coordinate in result:
            raise ValueError("Duplicate CTS cell")
        result[coordinate] = item
    return result


def _rule_targets(rule, columns):
    target = rule.get("target") if isinstance(rule, dict) else None
    raw = [target] if isinstance(target, str) else target if isinstance(target, list) else []
    return [columns.index(name) for name in raw if isinstance(name, str) and name in columns]


def _evaluate_batches(sandbox, rules, records, np, batch_size=1000):
    if not rules:
        return np.zeros((len(records), 0), dtype=np.int8)
    result = np.empty((len(records), len(rules)), dtype=np.int8)
    for start in range(0, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        values = np.asarray(sandbox.evaluate_rules(rules, records[start:stop]))
        if values.shape != (stop - start, len(rules)) or not np.isin(values, (-1, 0, 1)).all():
            raise ValueError("Sandbox returned invalid rule results")
        result[start:stop] = values
    return result


def validate_and_evaluate_rules(table, pass2, runtime):
    rules = pass2.get("rules")
    if not isinstance(rules, list):
        raise ValueError("Cached pass2 must contain a rule list")
    records = table.records()
    errors = [0] * len(rules)
    invalid = [False] * len(rules)
    statuses = [Counter() for _ in rules]
    examples = [[] for _ in rules]
    for start in range(0, len(records), RUNNER_CONFIG["rule_batch_rows"]):
        batch = records[start:start + RUNNER_CONFIG["rule_batch_rows"]]
        _accepted, reports = runtime.sandbox.validate_rules(rules, batch, max_error_rate=0.05)
        if len(reports) != len(rules):
            raise ValueError("Sandbox validation report length mismatch")
        for k, report in enumerate(reports):
            count = report.get("error_count")
            if type(count) is not int or not 0 <= count <= len(batch):
                raise ValueError("Sandbox validation error count is invalid")
            errors[k] += count
            invalid[k] |= report.get("status") == "invalid_rule"
            statuses[k][str(report.get("status"))] += 1
            for example in report.get("error_samples", []):
                if example not in examples[k] and len(examples[k]) < 3:
                    examples[k].append(example)
    accepted, reports = [], []
    for k, rule in enumerate(rules):
        passed = bool(records) and not invalid[k] and errors[k] < 0.05 * len(records)
        metadata = rule if isinstance(rule, dict) else {}
        reports.append({"id": metadata.get("id"), "target": metadata.get("target"),
                        "source_sha256": _digest(metadata.get("source")), "accepted": passed,
                        "error_count": errors[k], "error_rate": errors[k] / len(records) if records else 1.0,
                        "evaluated_rows": len(records) - errors[k], "batch_status_counts": dict(statuses[k]),
                        "error_samples": examples[k]})
        if passed:
            accepted.append(rule)
    matrix = _evaluate_batches(runtime.sandbox, accepted, records, runtime.np, RUNNER_CONFIG["rule_batch_rows"])
    detection = runtime.np.zeros(table.shape, dtype=bool)
    applicability = []
    for k, rule in enumerate(accepted):
        values = matrix[:, k]
        targets = _rule_targets(rule, table.columns)
        valid_count = int(runtime.np.count_nonzero(values >= 0))
        violations = int(runtime.np.count_nonzero(values == 1))
        all_true = valid_count > 0 and violations == valid_count
        eligible = bool(targets) and not all_true
        if eligible:
            for j in targets:
                detection[:, j] |= values == 1
        applicability.append({"id": rule.get("id"), "target_columns": targets,
                              "valid_rows": valid_count, "runtime_error_rows": int(runtime.np.count_nonzero(values == -1)),
                              "violation_rows": violations, "detection_applicable": eligible,
                              "detection_status": "nondiscriminating_all_true" if all_true else "no_resolved_target" if not targets else "target_specific",
                              "observed_constant_false_guard": valid_count > 0 and violations == 0,
                              "counterfactual_guard_applicable": True})
    return accepted, matrix, detection, {"validation": reports, "applicability": applicability,
                                         "accepted_rule_count": len(accepted), "raw_row_count": len(records)}


def _consensus_peers(table, i, candidate, cache=None):
    evidence = candidate.get("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("Consensus evidence must be an object")
    explicit = candidate.get("peer_rows")
    if explicit is not None:
        return _peers(explicit, i, table.shape[0])
    keys, source = evidence.get("key_column_indices"), evidence.get("source_column_index")
    if not isinstance(keys, list) or not keys or any(type(k) is not int or not 0 <= k < table.shape[1] for k in keys) or type(source) is not int or not 0 <= source < table.shape[1]:
        return []
    original = table.rows[i]
    if cache is None:
        cache = {}
    signature = (tuple(keys), source)
    if signature not in cache:
        groups = defaultdict(lambda: defaultdict(list))
        for r, row in enumerate(table.rows):
            groups[tuple(row[k] for k in keys)][row[source]].append(r)
        cache[signature] = groups
    entity = cache[signature].get(tuple(original[k] for k in keys), {})
    return sorted(r for label, indices in entity.items() if label != original[source] for r in indices if r != i)


def contextual_mask(table, consensus_candidates, np):
    mask = np.zeros(table.shape, dtype=bool)
    for coordinate, candidates in consensus_candidates.items():
        i, j = _coordinate(table, *coordinate)
        mask[i, j] = any(isinstance(item, dict) and item.get("actionable") is True
                         and isinstance(item.get("value"), str) and item["value"] != table.rows[i][j]
                         and item.get("origin") == "key_source_consensus" for item in candidates)
    return mask


def _stable_union(left, right):
    seen = {_canonical(value) for value in left}
    for value in right:
        key = _canonical(value)
        if key not in seen:
            left.append(value)
            seen.add(key)


def build_candidate_pool(table, detection_mask, consensus_candidates, seed_repairs, cts_cells):
    """Cap distinct exact strings before feature computation; merge late duplicates."""
    pool, counters = {}, Counter()
    peer_cache = {}
    for i, row in enumerate(detection_mask):
        for j, positive in enumerate(row):
            if not positive:
                continue
            coordinate, original = (i, j), table.rows[i][j]
            ordered = []
            for candidate in consensus_candidates.get(coordinate, []):
                if candidate.get("actionable") is True and candidate.get("origin") == "key_source_consensus":
                    ordered.append(("key_source_consensus", candidate, _consensus_peers(table, i, candidate, peer_cache), {}))
            if coordinate in seed_repairs:
                ordered.append(("seed", {"value": seed_repairs[coordinate]}, [], {"source_baseline": "zeroed_zeroec_reconstruction.json"}))
            cell = cts_cells.get(coordinate, {})
            complete_cts_peers = _peers(cell.get("retrieved_row_indices", []), i, table.shape[0])
            for candidate in cell.get("candidates", []):
                if not isinstance(candidate, dict):
                    raise ValueError("CTS candidates must be objects")
                peers = _peers(cell.get("retrieved_row_indices", []), i, table.shape[0])
                _stable_union(peers, _peers(candidate.get("peer_rows", []), i, table.shape[0]))
                _stable_union(peers, _peers(candidate.get("cts_peer_rows", []), i, table.shape[0]))
                _stable_union(complete_cts_peers, peers)
                ordered.append(("cts", candidate, peers, {"retrieval_provenance": cell.get("retrieval_provenance", {})}))
            merged = {}
            for origin, candidate, peers, context in ordered:
                counters["proposal_records_seen"] += 1
                value = candidate.get("value")
                if value is None:
                    counters["null_abstentions"] += 1
                    continue
                if not isinstance(value, str):
                    raise ValueError("Candidate values must be exact strings or null abstentions")
                if value == original:
                    counters["identity_proposals_omitted"] += 1
                    continue
                if value not in merged and len(merged) >= RUNNER_CONFIG["candidate_cap"]:
                    counters["distinct_proposals_over_cap"] += 1
                    continue
                if value not in merged:
                    merged[value] = {"value": value, "origins": [], "peer_rows": [], "sources": [], "provenance_records": []}
                else:
                    counters["exact_duplicate_records_merged"] += 1
                item = merged[value]
                _stable_union(item["origins"], [origin] + ([candidate["origin"]] if isinstance(candidate.get("origin"), str) else []))
                _stable_union(item["peer_rows"], peers)
                sources = candidate.get("sources", [])
                if not isinstance(sources, list):
                    raise ValueError("Candidate sources must be a list")
                _stable_union(item["sources"], sources)
                if origin == "key_source_consensus":
                    item["consensus"] = json.loads(_canonical(candidate))
                    _stable_union(item["sources"], [record["source"] for record in candidate.get("evidence", {}).get("peer_observations", []) if "source" in record])
                _stable_union(item["provenance_records"], [{"kind": origin, "proposal": candidate, "context": context}])
            if merged:
                for item in merged.values():
                    item["peer_rows"].sort()
                    item["cts_peer_rows"] = sorted(complete_cts_peers)
                pool[coordinate] = list(merged.values())
    counters["candidate_cells"] = len(pool)
    counters["distinct_candidates"] = sum(map(len, pool.values()))
    return pool, dict(counters)


def _safe_string(value):
    return not any(unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in value)


def candidate_format(table, i, j, value, index, pattern):
    original = table.rows[i][j]
    signature, original_signature = pattern(value), pattern(original)
    exemplars = int(index.patterns[j].get(signature, 0))
    grammar = any(v and _safe_string(v) for v in index.counts[j])
    if not _safe_string(value):
        valid, status = False, "invalid_control_or_line_separator"
    elif not grammar or not value:
        valid, status = False, "uncertain_no_observed_dirty_grammar" if not grammar else "uncertain_empty_candidate_grammar"
    elif signature == original_signature and original and _safe_string(original):
        valid, status = True, "same_original_dirty_pattern"
    elif exemplars >= RUNNER_CONFIG["minimum_dirty_pattern_exemplars"]:
        valid, status = True, "repeated_observed_dirty_pattern"
    else:
        valid, status = False, "uncertain_unobserved_or_singleton_dirty_pattern"
    return {"candidate_format_valid": valid, "format_status": status,
            "candidate_pattern": signature, "original_pattern": original_signature,
            "dirty_pattern_exemplars": exemplars}


def add_counterfactual_features(table, pool, rules, original_rules, index, runtime):
    np = runtime.np
    targets = [_rule_targets(rule, table.columns) for rule in rules]
    pending = []

    def flush():
        if not pending:
            return
        records = [dict(zip(table.columns, row)) for _, _, _, row in pending]
        matrix = _evaluate_batches(runtime.sandbox, rules, records, np, RUNNER_CONFIG["counterfactual_batch_rows"])
        for k, (i, j, candidate, counterfactual) in enumerate(pending):
            original, value = table.rows[i][j], candidate["value"]
            before, after = original_rules[i], matrix[k]
            runtime_error = bool(np.any(before == -1) or np.any(after == -1))
            new_violation = bool(np.any((before == 0) & (after == 1)))
            applicable = [r for r, columns in enumerate(targets) if j in columns and before[r] >= 0 and after[r] >= 0]
            changes = [int(before[r]) - int(after[r]) for r in applicable]
            rule_gain = float(sum(changes))
            rule_delta = rule_gain / len(changes) if changes else 0.0
            original_fd, original_fd_evidence = index.conditional(i, j, original)
            candidate_fd, candidate_fd_evidence = index.conditional(i, j, value, counterfactual)
            fd_delta = float(candidate_fd - original_fd)
            vicinity_delta = math.tanh((index.vicinity(i, j, value, counterfactual) - index.vicinity(i, j, original)) / 4.0)
            domain_delta = math.tanh((index.domain_support(i, j, value) - index.domain_support(i, j, original)) / 4.0)
            features = [rule_delta, fd_delta, vicinity_delta, domain_delta]
            if not all(math.isfinite(float(value)) for value in features):
                raise ValueError("Nonfinite dirty-only counterfactual feature")
            candidate.update(delta_features=features, rule_gain=rule_gain, new_rule_violation=new_violation,
                             counterfactual_error=runtime_error, criterion_valid=not runtime_error,
                             rule_criterion_valid=not runtime_error, counterfactual_valid=not runtime_error,
                             rule_evidence={"applicable_rule_count": len(applicable),
                                            "original_violation_count": sum(int(before[r] == 1) for r in applicable),
                                            "counterfactual_violation_count": sum(int(after[r] == 1) for r in applicable)},
                             fd_evidence={"original_support": float(original_fd), "candidate_support": float(candidate_fd),
                                          "original": original_fd_evidence, "candidate": candidate_fd_evidence})
            candidate.update(candidate_format(table, i, j, value, index, runtime.pattern))
        pending.clear()

    for (i, j), candidates in sorted(pool.items()):
        for candidate in candidates:
            counterfactual = list(table.rows[i])
            counterfactual[j] = candidate["value"]
            pending.append((i, j, candidate, counterfactual))
            if len(pending) == RUNNER_CONFIG["counterfactual_batch_rows"]:
                flush()
    flush()
    return pool


def combined_row_guard(table, mask, repairs, pool, rules, original_rules, runtime):
    by_row = defaultdict(dict)
    for coordinate, value in repairs.items():
        i, j = _coordinate(table, *coordinate)
        if not mask[i, j] or not isinstance(value, str) or value == table.rows[i][j]:
            raise ValueError("Repairs must be nonidentity exact strings inside the frozen detection mask")
        if value not in {candidate["value"] for candidate in pool.get((i, j), [])}:
            raise ValueError("Repair selector synthesized a value outside the candidate pool")
        by_row[i][j] = value
    rows = sorted(by_row)
    records = []
    for i in rows:
        row = list(table.rows[i])
        for j, value in by_row[i].items():
            row[j] = value
        records.append(dict(zip(table.columns, row)))
    matrix = _evaluate_batches(runtime.sandbox, rules, records, runtime.np, RUNNER_CONFIG["counterfactual_batch_rows"])
    committed, audit = {}, []
    for k, i in enumerate(rows):
        before, after = original_rules[i], matrix[k]
        error = bool(runtime.np.any(before == -1) or runtime.np.any(after == -1))
        new = bool(runtime.np.any((before == 0) & (after == 1)))
        rollback = error or new
        if not rollback:
            committed.update({(i, j): value for j, value in by_row[i].items()})
        audit.append({"row": i, "columns": sorted(by_row[i]), "rolled_back": rollback,
                      "new_rule_violation": new, "runtime_error": error,
                      "detection_mask_changed": False})
    return committed, audit


def infer_table(table, source_probabilities, empty_probabilities, source_metadata,
                seed_repairs, pass2, cts_cells, runtime, log=lambda *_args, **_kwargs: None):
    """Pure orchestration entry point, also used with fictional test doubles."""
    started = time.monotonic()
    log("validate_rules", dataset=table.name)
    rules, original_rules, violations, rule_diagnostics = validate_and_evaluate_rules(table, pass2, runtime)
    log("discover_consensus", dataset=table.name)
    consensus = runtime.consensus.discover_consensus(table)
    contextual = contextual_mask(table, consensus["candidates"], runtime.np)
    detection = runtime.detection.gate_detection(table, source_probabilities, empty_probabilities,
                                                violations, source_metadata, contextual_low_support=contextual)
    mask = runtime.np.asarray(detection["mask"], dtype=bool).copy()
    probabilities = runtime.np.asarray(detection["probabilities"], dtype=float).copy()
    if probabilities.shape != table.shape or not runtime.np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any() or not runtime.np.array_equal(mask, probabilities > 0.5):
        raise ValueError("Detection output violates probability/mask contract")
    mask_hash, probability_hash = _digest(mask.tolist()), _digest(probabilities.tolist())
    mask.setflags(write=False)
    probabilities.setflags(write=False)
    log("build_candidates", dataset=table.name, detected_cells=int(mask.sum()))
    pool, pool_diagnostics = build_candidate_pool(table, mask, consensus["candidates"], seed_repairs, cts_cells)
    index = runtime.DirtyIndex(table, dependencies=pass2.get("dependencies", []))
    add_counterfactual_features(table, pool, rules, original_rules, index, runtime)
    log("select_repairs", dataset=table.name, candidate_cells=len(pool))
    selected = runtime.repair.select_repairs(table, mask, pool)
    repairs, row_audit = combined_row_guard(table, mask, selected["repairs"], pool, rules, original_rules, runtime)
    if _digest(mask.tolist()) != mask_hash or _digest(probabilities.tolist()) != probability_hash:
        raise ValueError("Repair processing changed frozen detection output")
    diagnostics = {"rules": rule_diagnostics, "consensus": consensus["diagnostics"],
                   "detection": {key: value for key, value in detection["diagnostics"].items() if key != "retained_positive_evidence"},
                   "candidate_pool": pool_diagnostics, "repair": selected["diagnostics"],
                   "row_guard": {"checked_rows": len(row_audit), "rolled_back_rows": sum(item["rolled_back"] for item in row_audit)},
                   "detection_mask_sha256": mask_hash, "probabilities_sha256": probability_hash,
                   "elapsed_local_seconds": time.monotonic() - started, "cohort": COHORT}
    return {"probabilities": probabilities, "mask": mask, "repairs": repairs, "pool": pool,
            "decisions": selected["decisions"], "row_guard": row_audit,
            "detection_evidence": detection["evidence"],
            "retained_detection_evidence": detection["diagnostics"].get("retained_positive_evidence", []),
            "diagnostics": diagnostics}


def _audit_records(result):
    for item in result["detection_evidence"] + result["retained_detection_evidence"]:
        yield {"event": "detection", **item}
    for (i, j), candidates in sorted(result["pool"].items()):
        yield {"event": "candidate_counterfactuals", "row": i, "column": j, "candidates": candidates}
    for item in result["decisions"]:
        yield {"event": "repair_selection", **item}
    for item in result["row_guard"]:
        yield {"event": "combined_row_guard", **item}


def _verify_dataset_output(root, dataset, lock):
    root = Path(root)
    path = root / OUTPUT / "frozen" / f"{dataset}.json"
    frozen = _read_json(path)
    if (frozen.get("protocol_lock_sha256") != _sha256(root / LOCK)
            or frozen.get("dirty_sha256") != lock["input_files"][dataset]["dirty"]["sha256"]
            or frozen.get("frozen_before_evaluation") is not True or frozen.get("ground_truth_access") is not False
            or frozen.get("method_files") != METHOD_FILES or frozen.get("cohort") != COHORT):
        raise ValueError(f"Completed dataset freeze is inconsistent: {dataset}")
    expected_names = {"zeroclean_loop.json", "zeroed.json", "zeroed_zeroec_reconstruction.json"}
    if set(frozen.get("prediction_files", {})) != expected_names:
        raise ValueError(f"Completed prediction set changed: {dataset}")
    expected_inputs = {str(root / item["path"]): item["sha256"] for item in lock["input_files"][dataset].values()}
    if frozen.get("inference_inputs") != expected_inputs:
        raise ValueError(f"Completed inference-input binding changed: {dataset}")
    for filename, expected in frozen["prediction_files"].items():
        if _sha256(root / OUTPUT / "predictions" / dataset / filename) != expected:
            raise ValueError(f"Completed prediction changed: {dataset}/{filename}")
    expected_support = {str(OUTPUT / "audit" / f"{dataset}.jsonl"), str(OUTPUT / "diagnostics" / f"{dataset}.json")}
    if set(frozen.get("supporting_files", {})) != expected_support:
        raise ValueError(f"Completed audit/diagnostics binding changed: {dataset}")
    for path, expected in frozen["supporting_files"].items():
        if _sha256(root / path) != expected:
            raise ValueError(f"Completed audit/diagnostics changed: {path}")
    return frozen


def _run_study(root, datasets, runtime=None):
    """Internal entry: the CLI must hold heavy_execution for its whole call."""
    root = Path(root)
    if not datasets or len(set(datasets)) != len(datasets) or any(name not in DATASETS for name in datasets):
        raise ValueError("Datasets must be unique members of the fixed six-table cohort")
    lock = verify_freeze(root)  # No output or inference before complete validation.
    runtime = _runtime() if runtime is None else runtime
    if configuration_snapshot(runtime) != _read_json(root / PLAN)["configuration"]:
        raise ValueError("Module defaults differ from the frozen configuration")
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)

    def log(event, **details):
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, "cohort": COHORT, **details}
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()

    log("run_started", datasets=list(datasets), protocol_lock_sha256=_sha256(root / LOCK))
    try:
        for name in datasets:
            frozen_path = output / "frozen" / f"{name}.json"
            destination = output / "predictions" / name
            if frozen_path.exists():
                _verify_dataset_output(root, name, lock)
                log("completed_dataset_preserved", dataset=name)
                continue
            occupied = [destination, output / "audit" / f"{name}.jsonl", output / "diagnostics" / f"{name}.json"]
            if any(path.exists() for path in occupied):
                raise ValueError(f"Refusing to overwrite existing or incomplete dataset output: {name}")
            paths = {role: root / path for role, path in input_paths(name).items()}
            log("dataset_started", dataset=name)
            table = runtime.load_dirty(name, root=root / "vendor/ZeroED/data")
            if table.fingerprint != lock["input_files"][name]["dirty"]["sha256"]:
                raise ValueError("Dirty table changed while loading")
            source, source_metadata = load_probability_carrier(paths["source"], table.shape, runtime.np)
            empty, _empty_metadata = load_probability_carrier(paths["empty"], table.shape, runtime.np)
            seeds = load_seed_repairs(paths["sequential"], table.shape, runtime.np)
            pass2, cts = _read_json(paths["pass2"]), _read_json(paths["cts"])
            for label, artifact in (("pass2", pass2), ("cts", cts)):
                if not isinstance(artifact, dict):
                    raise ValueError(f"{label} artifact must be an object")
                for key in ("dirty_sha256", "dirty_fingerprint"):
                    if key in artifact and artifact[key] != table.fingerprint:
                        raise ValueError(f"{label} artifact dirty fingerprint mismatch")
                if "dataset" in artifact and artifact["dataset"] != name:
                    raise ValueError(f"{label} artifact dataset mismatch")
            cells = load_cts_cells(cts, table)
            result = infer_table(table, source, empty, source_metadata, seeds, pass2, cells, runtime, log)
            verify_freeze(root)
            metadata = {"system": "unified_loop_v2", "dataset": name, "cohort": COHORT,
                        "model": MODEL, "config": {"threshold": 0.5, **RUNNER_CONFIG},
                        "decision_threshold": 0.5, "decision_threshold_operator": ">",
                        "probabilities_semantics": "fixed gated empty-state policy scores; not calibrated correctness probabilities",
                        "upstream_binary_decisions": result["mask"].tolist(),
                        "input_fingerprint": table.fingerprint, "protocol_lock_sha256": _sha256(root / LOCK),
                        "plan_sha256": lock["plan_sha256"], "method_files": METHOD_FILES,
                        "ground_truth_access": False, "frozen_before_evaluation": True,
                        "design_provenance": PROVENANCE, "detection_mask_sha256": result["diagnostics"]["detection_mask_sha256"],
                        "diagnostics_file": str(OUTPUT / "diagnostics" / f"{name}.json"),
                        "audit_file": str(OUTPUT / "audit" / f"{name}.jsonl"),
                        "baseline_qualification": "Byte-preserved source ZeroED and qualified ZeroEC reconstruction; original metadata retained in baseline files"}
            prediction = runtime.Prediction(result["probabilities"], result["repairs"], metadata).validate(table.shape)
            body = {"probabilities": prediction.probabilities.tolist(),
                    "repairs": [{"row": i, "column": j, "value": value} for (i, j), value in sorted(prediction.repairs.items())],
                    "metadata": prediction.metadata}
            _exclusive_json(destination / METHOD_FILES["unified_loop_v2"], body)
            for role, filename in (("source", "zeroed.json"), ("sequential", "zeroed_zeroec_reconstruction.json")):
                _exclusive_bytes(destination / filename, paths[role].read_bytes())
                if _sha256(destination / filename) != lock["input_files"][name][role]["sha256"]:
                    raise ValueError("Source baseline copy did not preserve exact bytes")
            audit_path = output / "audit" / f"{name}.jsonl"
            _exclusive_jsonl(audit_path, _audit_records(result))
            diagnostic_path = output / "diagnostics" / f"{name}.json"
            _exclusive_json(diagnostic_path, result["diagnostics"])
            verify_freeze(root)
            frozen = {"dirty_sha256": table.fingerprint, "prediction_files": {path.name: _sha256(path) for path in sorted(destination.glob("*.json"))},
                      "inference_inputs": {str(root / item["path"]): item["sha256"] for item in lock["input_files"][name].values()},
                      "source_signature": _digest(lock["source_files"]), "protocol_lock_sha256": _sha256(root / LOCK),
                      "plan_sha256": lock["plan_sha256"], "config": _read_json(root / PLAN)["configuration"],
                      "method_files": METHOD_FILES, "model": MODEL, "cohort": COHORT,
                      "supporting_files": {str(path.relative_to(root)): _sha256(path) for path in (audit_path, diagnostic_path)},
                      "frozen_before_evaluation": True, "ground_truth_access": False,
                      "design_provenance": PROVENANCE, "time_unix": time.time()}
            _exclusive_json(frozen_path, frozen)
            log("dataset_frozen", dataset=name, detected_cells=int(result["mask"].sum()), repaired_cells=len(result["repairs"]))
        verify_freeze(root)
        if all((output / "frozen" / f"{name}.json").is_file() for name in DATASETS):
            for name in DATASETS:
                _verify_dataset_output(root, name, lock)
            completion = {"schema_version": SCHEMA_VERSION, "cohort": COHORT, "datasets": list(DATASETS),
                          "protocol_lock_sha256": _sha256(root / LOCK), "method_files": METHOD_FILES,
                          "dataset_freezes": {name: _sha256(output / "frozen" / f"{name}.json") for name in DATASETS},
                          "six_dataset_inference_complete": True, "evaluation_performed": False, "ground_truth_access": False}
            _same_or_create(output / "completion_manifest.json", completion)
            log("cohort_complete", dataset_count=len(DATASETS))
        else:
            log("requested_subset_complete", global_completion=False)
    except Exception as exc:
        log("run_failed", error_type=type(exc).__name__, error=str(exc))
        raise


def bound_environment():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    current, hard = resource.getrlimit(resource.RLIMIT_AS)
    ceiling = MEMORY_BYTES if hard == resource.RLIM_INFINITY else min(MEMORY_BYTES, hard)
    if current != resource.RLIM_INFINITY:
        ceiling = min(ceiling, current)
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, hard))


@contextmanager
def heavy_execution():
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with HEAVY_LOCK.open("a+", encoding="utf-8") as handle:
        print(f"[{COHORT}] waiting for exclusive heavy-experiment lock", flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            print(f"[{COHORT}] acquired exclusive heavy-experiment lock", flush=True)
            bound_environment()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("freeze", "run"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if args.phase == "freeze":
        bound_environment()
        lock = freeze_study(ROOT, args.datasets)
        print(json.dumps({"phase": "freeze", "datasets": lock["datasets"], "protocol_lock": str(ROOT / LOCK),
                          "protocol_lock_sha256": _sha256(ROOT / LOCK), "inference_performed": False}, indent=2), flush=True)
    else:
        with heavy_execution():
            _run_study(ROOT, args.datasets)


if __name__ == "__main__":
    main()
