#!/usr/bin/env python3
"""Hold one foreground resource lock across schemas, freeze and all-six inference."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.unified_loop_v2 import run_study as runner


def _need(condition, label):
    if not condition:
        raise ValueError(label)


def schema_preflight(root, runtime):
    """Use actual loaders; never infer, sandbox, build features, score or write."""
    root = Path(root)
    statuses = []
    for dataset in runner.DATASETS:
        stage = "dirty"
        try:
            paths = {role: root / path for role, path in runner.input_paths(dataset).items()}
            table = runtime.load_dirty(dataset, root=root / "vendor/ZeroED/data")
            stage = "source_carrier"
            runner.load_probability_carrier(paths["source"], table.shape, runtime.np)
            stage = "empty_carrier"
            runner.load_probability_carrier(paths["empty"], table.shape, runtime.np)
            stage = "sequential_seed"
            seeds = runner.load_seed_repairs(paths["sequential"], table.shape, runtime.np)
            stage = "cts_schema"
            cts = runner._read_json(paths["cts"])
            cells = runner.load_cts_cells(cts, table)
            candidate_count = 0
            for (i, _j), cell in cells.items():
                runner._peers(cell.get("retrieved_row_indices", []), i, table.shape[0])
                _need(isinstance(cell.get("retrieval_provenance", {}), dict), "cts_context_object")
                _need(isinstance(cell.get("conditional_profiles", []), list), "cts_conditional_profiles")
                for candidate in cell["candidates"]:
                    _need(isinstance(candidate, dict), "cts_candidate_object")
                    _need(candidate.get("value") is None or isinstance(candidate.get("value"), str), "cts_candidate_string_or_null")
                    _need(isinstance(candidate.get("sources", []), list), "cts_candidate_sources")
                    runner._peers(candidate.get("peer_rows", []), i, table.shape[0])
                    runner._peers(candidate.get("cts_peer_rows", []), i, table.shape[0])
                    if "score" in candidate:
                        _need(type(candidate["score"]) in (int, float) and math.isfinite(candidate["score"]), "cts_candidate_score")
                    candidate_count += 1
            stage = "pass2_schema"
            pass2 = runner._read_json(paths["pass2"])
            _need(isinstance(pass2, dict) and isinstance(pass2.get("rules"), list), "pass2_rules_list")
            dependencies = pass2.get("dependencies", [])
            _need(isinstance(dependencies, list), "pass2_dependencies_list")
            for artifact in (pass2, cts):
                for field in ("dirty_sha256", "dirty_fingerprint"):
                    _need(field not in artifact or artifact[field] == table.fingerprint, "artifact_dirty_fingerprint")
                _need("dataset" not in artifact or artifact["dataset"] == dataset, "artifact_dataset")
            unresolved = 0
            for rule in pass2["rules"]:
                _need(isinstance(rule, dict) and isinstance(rule.get("source"), str), "pass2_rule_source")
                target = rule.get("target")
                _need(isinstance(target, str) or isinstance(target, list) and all(isinstance(v, str) for v in target), "pass2_rule_target")
                unresolved += not bool(runner._rule_targets(rule, table.columns))
            for dependency in dependencies:
                _need(isinstance(dependency, dict), "pass2_dependency_object")
                determinants = dependency.get("determinants")
                _need(isinstance(determinants, list) and bool(determinants) and all(isinstance(v, str) for v in determinants), "pass2_dependency_determinants")
                _need(isinstance(dependency.get("dependent"), str), "pass2_dependency_target")
                support, confidence = dependency.get("min_support", 3), dependency.get("confidence", .9)
                _need(type(support) is int and support >= 1, "pass2_dependency_support")
                _need(type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1, "pass2_dependency_confidence")
            status = {"event": "schema_preflight", "dataset": dataset, "status": "pass",
                      "shape": list(table.shape), "seeds": len(seeds), "cts_cells": len(cells),
                      "candidates": candidate_count, "rules": len(pass2["rules"]),
                      "dependencies": len(dependencies), "unresolved_rule_targets": unresolved}
        except Exception as exc:
            # Messages can contain raw values; only a fixed stage/type is emitted.
            status = {"event": "schema_preflight", "dataset": dataset, "status": "fail",
                      "stage": stage, "error_type": type(exc).__name__}
        statuses.append(status)
        print(json.dumps(status, sort_keys=True, allow_nan=False), flush=True)
    summary = {"event": "schema_preflight_complete", "datasets": len(statuses),
               "failures": sum(status["status"] == "fail" for status in statuses),
               "inference_performed": False, "sandbox_executed": False, "output_files_created": False}
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return summary


def run_cohort(root=ROOT):
    print(json.dumps({"event": "cohort_controller_queued", "pid": os.getpid(), "cohort": runner.COHORT}), flush=True)
    with runner.heavy_execution():
        runtime = runner._runtime()  # Scientific imports follow lock and bounds.
        summary = schema_preflight(root, runtime)
        if summary["failures"]:
            raise ValueError("Schema preflight failed; freeze and inference were not attempted")
        runner.freeze_study(root, runner.DATASETS, runtime)
        print(json.dumps({"event": "global_protocol_frozen", "datasets": 6,
                          "protocol_lock_sha256": runner._sha256(Path(root) / runner.LOCK)}, sort_keys=True), flush=True)
        runner._run_study(root, list(runner.DATASETS), runtime)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    run_cohort()


if __name__ == "__main__":
    main()
