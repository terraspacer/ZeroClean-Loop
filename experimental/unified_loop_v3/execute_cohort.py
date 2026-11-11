#!/usr/bin/env python3
"""Preflight, freeze and infer the complete v3 cohort under one heavy lock.

No evaluator is imported. Console statuses contain stage names and counts;
dirty row values and generated rule source remain in the bound input artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experimental.unified_loop_v3 import run_study as runner


def _status(stage, **counts):
    print(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(),
                      "cohort": runner.COHORT, "stage": stage, **counts},
                     sort_keys=True, allow_nan=False), flush=True)


def preflight(root, runtime, status=_status):
    """Validate every real input schema without evaluating rules or inferring."""
    root = Path(root)
    before_sources, before_inputs = runner.source_manifest(root), runner.input_manifest(root)
    totals = {"dataset_count": 0, "input_file_count": 0, "rows": 0,
              "rules": 0, "dependencies": 0, "static_valid_rules": 0,
              "static_rejected_rules": 0, "cts_cells": 0, "seed_repairs": 0}
    for name in runner.DATASETS:
        paths = {role: root / path for role, path in runner.input_paths(name).items()}
        if len(paths) != 9:
            raise ValueError("The frozen v3 policy requires exactly nine input roles")
        fresh_metadata = runner.validate_artifact_files(root, name)
        table = runtime.load_dirty(name, root=root / "vendor/ZeroED/data")
        if table.fingerprint != before_inputs[name]["dirty"]["sha256"]:
            raise ValueError("Dirty input changed during cohort preflight")
        for role in ("source", "empty"):
            runner.load_probability_carrier(paths[role], table.shape, runtime.np)
        seeds = runner.load_seed_repairs(paths["sequential"], table.shape, runtime.np)
        fresh, old, cts = (runner._read_json(paths[role]) for role in ("pass2", "old_pass2", "cts"))
        merged = runner.merge_rule_artifacts(table, old, fresh)
        if not isinstance(cts, dict):
            raise ValueError("CTS artifact must be an object")
        if cts.get("dataset", name) != name:
            raise ValueError("CTS artifact dataset mismatch")
        for key in ("dirty_sha256", "dirty_fingerprint"):
            if key in cts and cts[key] != table.fingerprint:
                raise ValueError("CTS artifact dirty identity mismatch")
        cells = runner.load_cts_cells(cts, table)
        valid = rejected = 0
        for rule in merged["rules"]:
            try:
                runtime.sandbox._validate_source(rule["source"])
            except (ValueError, SyntaxError, TypeError):
                rejected += 1  # Inference still applies full sandbox admission.
            else:
                valid += 1
        counts = {"rows": table.shape[0], "columns": table.shape[1], "input_file_count": len(paths),
                  "rules": len(merged["rules"]), "dependencies": len(merged["dependencies"]),
                  "static_valid_rules": valid, "static_rejected_rules": rejected,
                  "cts_cells": len(cells), "seed_repairs": len(seeds)}
        status("dataset_preflight_complete", dataset=name, **counts)
        totals["dataset_count"] += 1
        for key in totals.keys() - {"dataset_count"}:
            totals[key] += counts[key]
        if fresh_metadata["rule_union"] != merged["union_diagnostics"]:
            raise ValueError("Rule union changed during cohort preflight")
    if before_sources != runner.source_manifest(root) or before_inputs != runner.input_manifest(root):
        raise ValueError("Source or inputs changed during cohort preflight")
    return {"source_files": before_sources, "input_files": before_inputs, "counts": totals}


def execute(root=runner.ROOT):
    """One mandatory lock encloses the complete preflight/freeze/run sequence."""
    stage = "waiting_for_heavy_lock"
    with runner.heavy_execution():
        try:
            root = Path(root)
            runtime = runner._runtime()  # Scientific imports follow resource bounds.
            stage = "cohort_preflight"
            _status(stage, dataset_count=len(runner.DATASETS), input_file_count=54)
            checked = preflight(root, runtime)
            _status("cohort_preflight_complete", **checked["counts"])
            stage = "global_freeze"
            lock = runner.freeze_study(root, runner.DATASETS, runtime)
            if lock["source_files"] != checked["source_files"] or lock["input_files"] != checked["input_files"]:
                raise ValueError("Freeze differs from the validated cohort snapshot")
            _status("global_freeze_complete", dataset_count=len(runner.DATASETS),
                    protocol_lock_sha256=runner._sha256(root / runner.LOCK))
            stage = "serialized_inference"
            runner._run_study(root, runner.DATASETS, runtime)
            complete = runner._read_json(root / runner.OUTPUT / "completion_manifest.json")
            if complete.get("six_dataset_inference_complete") is not True or complete.get("ground_truth_access") is not False:
                raise ValueError("Complete all-six inference manifest is required")
            _status("cohort_complete", dataset_count=len(runner.DATASETS), prediction_file_count=18,
                    evaluation_performed=False)
            return complete
        except Exception as exc:
            _status("controller_failed", failed_stage=stage, error_type=type(exc).__name__)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    execute()


if __name__ == "__main__":
    main()
