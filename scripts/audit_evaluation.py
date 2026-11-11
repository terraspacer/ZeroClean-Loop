#!/usr/bin/env python3
"""Read-only, evaluation-only audit of already frozen benchmark predictions.

Uses only Python's standard library and deliberately imports no project evaluator.
Reference values and failure examples must never be consumed by inference or tuning.
Only the two audit deliverables are written; input artifacts are never modified.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


DATASETS = ("hospital", "flights", "beers", "rayyan", "billionaire", "movies")
SYSTEMS = (
    "adapted_zeroed", "adapted_zeroed_zeroec", "without_counterfactual",
    "without_iterative_verification", "without_multi_column", "zeroclean_loop",
)
METRIC_KEYS = ("tp", "fp", "fn", "predicted", "actual", "precision", "recall", "f1")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, snapshots):
    raw = path.read_bytes()
    snapshots[path] = digest(raw)
    return json.loads(raw)


def read_table(path, snapshots):
    snapshots[path] = digest(path.read_bytes())
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.reader(stream))
    assert records, f"Empty CSV: {path}"
    header, rows = records[0], records[1:]
    assert len(header) == len(set(header)), f"Duplicate header: {path}"
    assert all(len(row) == len(header) for row in rows), f"Ragged CSV: {path}"
    return header, rows


def rates(tp, fp, fn):
    # Direct confusion-count formula, without project PRF helpers.
    return {
        "tp": tp, "fp": fp, "fn": fn, "predicted": tp + fp, "actual": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def recompute(header, dirty, clean, prediction, threshold, sample_limit=50):
    nrows, ncols = len(dirty), len(header)
    probabilities = prediction["probabilities"]
    assert len(probabilities) == nrows, "Probability row mismatch"
    assert all(len(row) == ncols for row in probabilities), "Probability column mismatch"
    assert all(isinstance(p, (int, float)) and not isinstance(p, bool)
               and math.isfinite(p) and 0 <= p <= 1
               for row in probabilities for p in row), "Invalid probability"
    repairs = {}
    noops = Counter()
    for event in prediction["repairs"]:
        i, j, value = event["row"], event["column"], event["value"]
        assert type(i) is int and type(j) is int, "Non-integer repair coordinate"
        assert 0 <= i < nrows and 0 <= j < ncols, "Repair outside table"
        assert isinstance(value, str), "Non-string repair value"
        assert (i, j) not in repairs, "Duplicate repair coordinate"
        assert probabilities[i][j] >= threshold, "Repair absent from detection mask"
        repairs[i, j] = value
        if value == dirty[i][j]:
            noops["dirty" if value != clean[i][j] else "clean"] += 1

    detection = Counter()
    partition = Counter()
    failure_counts = Counter()
    failure_columns = {kind: Counter() for kind in (
        "false_positives", "false_negatives", "wrong_repairs",
        "wrong_repairs_on_clean", "wrong_repairs_on_dirty",
    )}
    samples = {kind: [] for kind in ("false_positives", "false_negatives", "wrong_repairs")}
    for i, row in enumerate(dirty):
        for j, value in enumerate(row):
            reference = clean[i][j]
            actual_error = value != reference
            predicted_error = probabilities[i][j] >= threshold
            corrected = repairs.get((i, j), value)
            changed = corrected != value
            correct_repair = actual_error and changed and corrected == reference
            detection[(actual_error, predicted_error)] += 1
            if actual_error:
                outcome = ("dirty_correct_repair" if correct_repair else
                           "dirty_wrong_repair" if changed else "dirty_unchanged")
            else:
                outcome = "clean_wrong_repair" if changed else "clean_unchanged"
            partition[outcome] += 1
            categories = []
            if predicted_error and not actual_error:
                categories.append("false_positives")
            if actual_error and not predicted_error:
                categories.append("false_negatives")
            if changed and not correct_repair:
                categories.append("wrong_repairs")
                failure_columns["wrong_repairs_on_dirty" if actual_error
                                else "wrong_repairs_on_clean"][header[j]] += 1
            for kind in categories:
                failure_counts[kind] += 1
                failure_columns[kind][header[j]] += 1
                if len(samples[kind]) < sample_limit:
                    samples[kind].append({
                        "row": i, "column": header[j], "dirty": value,
                        "predicted": corrected, "reference": reference,
                        "probability": probabilities[i][j],
                    })

    det = rates(detection[True, True], detection[False, True], detection[True, False])
    rep = rates(partition["dirty_correct_repair"],
                partition["dirty_wrong_repair"] + partition["clean_wrong_repair"],
                partition["dirty_wrong_repair"] + partition["dirty_unchanged"])
    assert sum(partition.values()) == nrows * ncols
    assert det["actual"] == rep["actual"]
    assert rep["predicted"] == sum(v != dirty[i][j] for (i, j), v in repairs.items())
    assert failure_counts["wrong_repairs"] == rep["fp"]
    return {
        "detection": det, "repair": rep, "detection_tn": detection[False, False],
        "cell_outcomes": {key: partition[key] for key in (
            "dirty_correct_repair", "dirty_wrong_repair", "dirty_unchanged",
            "clean_wrong_repair", "clean_unchanged",
        )},
        "repair_events": len(repairs),
        "no_op_repair_events": {key: noops[key] for key in ("clean", "dirty")},
        "failure_counts": dict(failure_counts),
        "failure_columns": {key: dict(value) for key, value in failure_columns.items()},
    }, samples


def semantic_self_check():
    # Covers clean no-op, damaged clean, correct repair, incorrect dirty repair,
    # dirty no-op, clean untouched, dirty untouched, and p == threshold.
    header = list("abcdefg")
    dirty = [list("abcdefg")]
    clean = [list("abCDEfG")]
    prediction = {
        "probabilities": [[.5, .9, .9, .9, .9, .1, .1]],
        "repairs": [
            {"row": 0, "column": i, "value": value}
            for i, value in enumerate(("a", "wrong", "C", "wrong", "e"))
        ],
    }
    result, _ = recompute(header, dirty, clean, prediction, .5)
    assert [result["detection"][key] for key in ("tp", "fp", "fn")] == [3, 2, 1]
    assert [result["repair"][key] for key in ("tp", "fp", "fn")] == [1, 2, 3]
    assert result["no_op_repair_events"] == {"clean": 1, "dirty": 1}
    assert result["cell_outcomes"]["clean_unchanged"] == 2
    assert rates(0, 0, 0)["f1"] == 0.0
    return {"passed": True, "edge_cases": 8}


def compare_value(expected, actual, location, discrepancies):
    if isinstance(expected, float):
        equal = isinstance(actual, (int, float)) and math.isclose(
            expected, actual, rel_tol=0, abs_tol=1e-12)
    else:
        equal = expected == actual
    if not equal:
        discrepancies.append({"location": location, "recomputed": expected, "reported": actual})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    snapshots, discrepancies = {}, []
    summary_path = root / "results/summary_metrics.json"
    summary = read_json(summary_path, snapshots)
    assert set(summary["metrics"]) == set(DATASETS), "Unexpected dataset population"
    results, source_provenance, freeze_checks, sample_checks = {}, {}, {}, {}
    review_samples = {}
    for dataset in DATASETS:
        frozen_path = root / f"results/frozen/{dataset}.json"
        frozen = read_json(frozen_path, snapshots)
        dirty_path = root / f"vendor/ZeroED/data/{dataset}_error-01.csv"
        clean_path = root / f"vendor/ZeroED/data/{dataset}_clean.csv"
        header, dirty = read_table(dirty_path, snapshots)
        clean_header, clean = read_table(clean_path, snapshots)
        assert header == clean_header and len(dirty) == len(clean), "Reference shape mismatch"
        assert set(summary["metrics"][dataset]) == set(SYSTEMS), "Unexpected system population"
        assert set(frozen["prediction_files"]) == {f"{name}.json" for name in SYSTEMS}
        threshold = frozen["config"]["threshold"]
        assert threshold == .5, "Unexpected frozen detection threshold"
        assert frozen["frozen_before_evaluation"] is True, "Missing freeze declaration"
        compare_value(frozen["dirty_sha256"], snapshots[dirty_path],
                      f"{dataset}/frozen_dirty_hash", discrepancies)
        freeze_checks[dataset] = {
            "frozen_before_evaluation_declared": frozen["frozen_before_evaluation"],
            "frozen_time_before_summary_file_mtime": frozen["time_unix"] <= summary_path.stat().st_mtime,
            "dirty_hash_matches": frozen["dirty_sha256"] == snapshots[dirty_path],
            "prediction_hashes_match": {}, "threshold": threshold,
        }
        compare_value(True, freeze_checks[dataset]["frozen_time_before_summary_file_mtime"],
                      f"{dataset}/freeze_chronology", discrepancies)
        source_provenance[dataset] = {
            "rows": len(dirty), "columns": len(header), "cells": len(dirty) * len(header),
            "dirty_sha256": snapshots[dirty_path], "reference_sha256": snapshots[clean_path],
            "frozen_manifest_sha256": snapshots[frozen_path],
            "reference_policy": "Exact CSV strings; no normalization, stripping, case folding, or null coercion",
        }
        results[dataset], sample_checks[dataset] = {}, {}
        for system in SYSTEMS:
            prediction_path = root / f"results/predictions/{dataset}/{system}.json"
            prediction = read_json(prediction_path, snapshots)
            frozen_hash = frozen["prediction_files"][prediction_path.name]
            matches = frozen_hash == snapshots[prediction_path]
            freeze_checks[dataset]["prediction_hashes_match"][system] = matches
            compare_value(frozen_hash, snapshots[prediction_path],
                          f"{dataset}/{system}/frozen_prediction_hash", discrepancies)
            identity_field = ("dirty_sha256" if "dirty_sha256" in prediction["metadata"]
                              else "input_fingerprint")
            compare_value(snapshots[dirty_path], prediction["metadata"].get(identity_field),
                          f"{dataset}/{system}/prediction_dirty_hash", discrepancies)
            measured, samples = recompute(header, dirty, clean, prediction, threshold)
            results[dataset][system] = measured
            reported = summary["metrics"][dataset][system]
            for kind in ("detection", "repair"):
                for key in METRIC_KEYS:
                    compare_value(measured[kind][key], reported[kind].get(key),
                                  f"{dataset}/{system}/{kind}/{key}", discrepancies)
            for key in ("rows", "columns", "dirty_sha256", "reference_sha256"):
                compare_value(source_provenance[dataset][key], reported.get(key),
                              f"{dataset}/{system}/{key}", discrepancies)
            failure_path = root / f"results/evaluation_only_failures/{dataset}/{system}.json"
            failures = read_json(failure_path, snapshots)
            sample_checks[dataset][system] = {}
            for kind, expected in samples.items():
                observed = failures[kind]
                matches = expected == observed
                sample_checks[dataset][system][kind] = {
                    "total_failures": measured["failure_counts"].get(kind, 0),
                    "reported_sample_size": len(observed), "all_samples_match": matches,
                }
                compare_value(True, matches, f"{dataset}/{system}/failure_sample/{kind}", discrepancies)
            if system == "zeroclean_loop":
                review_samples[dataset] = samples

    macro = {}
    for system in SYSTEMS:
        macro[system] = {}
        for kind in ("detection", "repair"):
            value = sum(results[d][system][kind]["f1"] for d in DATASETS) / len(DATASETS)
            macro[system][kind] = value
            compare_value(value, summary["macro_f1"][system][kind],
                          f"macro_f1/{system}/{kind}", discrepancies)

    gains = {
        "ed_gains": {d: results[d]["zeroclean_loop"]["detection"]["f1"] -
                     results[d]["adapted_zeroed"]["detection"]["f1"] for d in DATASETS},
        "repair_gains": {d: results[d]["zeroclean_loop"]["repair"]["f1"] -
                         results[d]["adapted_zeroed_zeroec"]["repair"]["f1"] for d in DATASETS},
    }
    gains["ed_macro_gain"] = sum(gains["ed_gains"].values()) / len(DATASETS)
    gains["repair_macro_gain"] = sum(gains["repair_gains"].values()) / len(DATASETS)
    checks = {
        "ed_mean_gain": gains["ed_macro_gain"] >= .035,
        "ed_each_dataset": all(value > 0 for value in gains["ed_gains"].values()),
        "repair_mean_gain": gains["repair_macro_gain"] >= .08,
        "flights": results["flights"]["zeroclean_loop"]["detection"]["f1"] > .76,
        "rayyan": results["rayyan"]["zeroclean_loop"]["detection"]["f1"] > .77,
    }
    adapted = {"met": all(checks.values()), "checks": checks, **gains}
    reported_adapted = summary["adapted_comparison_only"]
    for key, expected in adapted.items():
        if isinstance(expected, dict):
            for subkey, value in expected.items():
                compare_value(value, reported_adapted[key][subkey],
                              f"adapted_comparison_only/{key}/{subkey}", discrepancies)
        else:
            compare_value(expected, reported_adapted.get(key),
                          f"adapted_comparison_only/{key}", discrepancies)
    expected_targets = {
        "ed_mean_absolute_gain": .035, "repair_mean_absolute_gain": .08,
        "flights_ed_f1_strictly_above": .76, "rayyan_ed_f1_strictly_above": .77,
        "ed_improves_each_dataset": True,
    }
    for target_source in ("adapted_comparison_only", "official_target_assessment"):
        compare_value(expected_targets, summary[target_source]["targets"],
                      f"{target_source}/targets", discrepancies)
    missing = [f"{dataset}/{baseline}" for dataset in DATASETS
               for baseline in ("zeroed", "zeroed_zeroec")
               if baseline not in summary["metrics"][dataset]]
    official = summary["official_target_assessment"]
    compare_value(missing, official.get("missing"), "official_target_assessment/missing", discrepancies)
    compare_value(False, official["met"], "official_target_assessment/met", discrepancies)
    compare_value("unverified_missing_baselines_or_runs", official["status"],
                  "official_target_assessment/status", discrepancies)
    compare_value(False, summary["targets_met"], "targets_met", discrepancies)

    # Rehash every input after all calculations to detect concurrent writes.
    stable = all(digest(path.read_bytes()) == value for path, value in snapshots.items())
    compare_value(True, stable, "input_snapshot_stability", discrepancies)
    total_cells = sum(source["cells"] for source in source_provenance.values())
    semantic_checks = semantic_self_check()
    audit = {
        "audit_scope": "Evaluation only; no inference, label-driven selection, or tuning recommendations",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "readiness": "Share with caveats" if not discrepancies else "Needs revision",
        "completeness": "Complete for all six datasets and all 36 frozen prediction artifacts",
        "independent_implementation": "Python stdlib csv + Counter; no project evaluator imports or helper calls",
        "input_snapshot_stable": stable,
        "summary_sha256": snapshots[summary_path],
        "input_sha256": {str(path.relative_to(root)): value for path, value in snapshots.items()},
        "coverage": {"datasets": len(DATASETS), "systems_per_dataset": len(SYSTEMS),
                     "prediction_runs": len(DATASETS) * len(SYSTEMS),
                     "unique_cells": total_cells, "cell_run_observations": total_cells * len(SYSTEMS),
                     "metric_values_reconciled": len(DATASETS) * len(SYSTEMS) * 2 * len(METRIC_KEYS),
                     "macro_f1_values_reconciled": len(SYSTEMS) * 2},
        "semantics": {
            "detection": "p >= frozen threshold 0.5; truth is dirty != clean",
            "repair_tp": "Changed erroneous cell with corrected value exactly equal to reference",
            "repair_fp": "Any changed cell whose final value is not its reference",
            "repair_fn": "Any originally erroneous cell not repaired exactly",
            "unchanged_clean": "Excluded from repair TP, FP and FN",
            "wrong_repair_on_dirty": "Counts once as repair FP and once as repair FN",
            "wrong_repair_on_clean": "Counts once as repair FP; no repair FN",
            "no_op_repair": "Excluded from predicted repairs; erroneous no-op remains repair FN",
            "macro_f1": "Unweighted arithmetic mean of six per-dataset F1 values",
            "units": "F1 in [0,1]; multiply by 100 for percent; .035 gain = 3.5 percentage points",
        },
        "synthetic_semantic_checks": semantic_checks,
        "source_provenance": source_provenance, "freeze_checks": freeze_checks,
        "metrics": results, "macro_f1": macro,
        "macro_f1_percent": {s: {k: 100 * v for k, v in m.items()} for s, m in macro.items()},
        "adapted_comparison_only": adapted,
        "adapted_gain_percentage_points": {
            "detection": gains["ed_macro_gain"] * 100,
            "repair": gains["repair_macro_gain"] * 100,
        },
        "official_target_assessment": {
            "met": False, "status": "Unverified: original upstream baseline runs are absent",
            "missing": missing, "targets": expected_targets,
        },
        "failure_sample_checks": sample_checks,
        "baseline_caveat": "Comparisons are against explicitly adapted baselines; they do not establish gains against original ZeroED or ZeroEC.",
        "limits": [
            "This numerical audit cannot prove historical label isolation or pretraining non-contamination.",
            "Freeze chronology is a declared flag plus file timestamps; raw prediction and dirty-input hashes are verified.",
            "Frozen source_signature and provenance descriptions are retained evidence, not independently reproduced upstream-baseline runs.",
            "Failure reports contain capped row-major samples; the audit computes exhaustive counts independently.",
        ],
        "discrepancy_count": len(discrepancies), "discrepancies": discrepancies,
    }
    audit_path = root / "results/evaluation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_review(root, audit, review_samples)
    print(json.dumps({"readiness": audit["readiness"], "coverage": audit["coverage"],
                      "discrepancy_count": len(discrepancies), "snapshot_stable": stable,
                      "macro_f1": macro, "adapted_target_checks": checks}, indent=2))
    return 1 if discrepancies else 0


def write_review(root, audit, samples):
    if audit["discrepancies"]:
        # A failing future run must not inherit the successful review's prose.
        text = (
            "# Evaluation-only frozen-results review\n\n"
            "**Needs revision.** Independent recomputation found mismatches. "
            "All input artifacts remain unchanged. The exhaustive recomputed counts "
            "and provenance are retained in `results/evaluation_audit.json`. "
            "Reference comparisons must never be used for inference or tuning.\n\n"
            "```json\n" + json.dumps(audit["discrepancies"], indent=2) + "\n```\n"
        )
        (root / "results/evaluation_only_failures/review.md").write_text(text, encoding="utf-8")
        return
    lines = [
        "# Evaluation-only frozen-results review", "",
        "**Reference firewall:** This file contains evaluation-only reference comparisons. "
        "It must never be used to revise inference, prompts, rules, thresholds, confidence, "
        "candidate selection, or model selection. Examples are descriptive reporting evidence only.", "",
        f"**{audit['readiness']} — {audit['completeness']}.** "
        f"Independent recomputation found {audit['discrepancy_count']} discrepancies. "
        "No benchmark inputs, predictions, harness, or reported metrics were changed.", "",
        "The controlling inputs are the six frozen manifests, 36 prediction artifacts, "
        "the exact dirty/reference CSV pairs, and the recorded summary. "
        "The standalone audit uses stdlib CSV parsing and Counters. "
        "Every cell is checked; sample examples below are capped.", "",
        "## Numerical reconciliation", "",
        "| Dataset | Detection TP / FP / FN | Repair TP / FP / FN | Detection F1 (%) | Repair F1 (%) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        result = audit["metrics"][dataset]["zeroclean_loop"]
        triples = [" / ".join(str(result[kind][key]) for key in ("tp", "fp", "fn"))
                   for kind in ("detection", "repair")]
        lines.append(f"| {dataset} | {triples[0]} | {triples[1]} | "
                     f"{100 * result['detection']['f1']:.6f} | {100 * result['repair']['f1']:.6f} |")
    lines.extend([
        "", "The table shows the full system. All six systems per dataset reconcile; "
        "the machine-readable audit retains every system's counts and cell-outcome partitions.", "",
        "Repair TP requires both an actual change and an originally erroneous cell. "
        "Unchanged clean cells never count as repair successes. A wrong edit on a dirty "
        "cell counts as repair FP and FN; an edit damaging a clean cell counts as repair FP. "
        "Explicit no-op events are excluded from the predicted repair denominator.", "",
        "## Units and claim boundaries", "",
        "Macro F1 is the unweighted mean of the six dataset F1 values. F1 is stored in "
        "[0,1]; percent values multiply by 100. The +3.5 and +8 targets mean absolute "
        "percentage-point gains (+0.035 and +0.08 in stored F1 units), not relative percent gains. "
        "The Flights >76% and Rayyan >77% requirements are strict inequalities.", "",
        f"Against the adapted baselines, the detection macro gain is "
        f"{audit['adapted_gain_percentage_points']['detection']:.6f} percentage points and "
        f"the repair macro gain is {audit['adapted_gain_percentage_points']['repair']:.6f} "
        "percentage points. Both mean-gain checks and the per-dataset detection-improvement "
        "check pass. Flights and Rayyan absolute detection targets fail. "
        "The combined adapted target assessment therefore fails.", "",
        "Original upstream ZeroED / ZeroEC baseline runs are missing for all six datasets. "
        "Official target achievement remains unverified and cannot be asserted. "
        "The adapted comparisons are measured evidence only for those adaptations. "
        "Historical label isolation and pretraining contamination are not provable from these files.", "",
        "## Scoped quality assessment", "",
        "Counts below are scoped inventory counts, not a composite accuracy percentage.", "",
        "| Category | Observed defects | Assessment |",
        "| --- | --- | --- |",
        "| Usefulness and completeness | 0 / 36 | All requested frozen runs and six datasets recomputed; upstream baseline absence is explicit. |",
        "| Analytical clarity | 0 / 3 | Detection, exact repair, and target units are explicit. |",
        "| Visual / interaction consistency | N/A | JSON and Markdown audit; no dashboard or interactive controls. |",
        "", "| Category | Observed defects | Assessment |",
        "| --- | --- | --- |",
        "| Source authority / confidence | 0 / 36 | Raw prediction hashes match frozen manifests and dirty/reference identities match reported metrics. |",
        "| Value accuracy | 0 / 72 | Every detection and repair metric group independently recomputed. |",
        "| Within-chart agreement | N/A | No charts. |",
        "| Complete source details | 0 / 36 | Input hashes, reference policy, dimensions, and confusion partitions are retained. |",
        "| Cross-artifact consistency | 0 / 5 | Frozen inputs, per-run summary, macro means, adapted target checks, and failure samples reconcile. |",
        "| Data-quality controls | 0 / 36 | Dimensions, rectangular CSVs, coordinate uniqueness, string values, probability bounds, detection coverage, and snapshot stability checked. |",
        "| Conclusion support | 0 / 2 | Adapted target failure and missing official baselines are preserved. |",
        "", "## Prioritized problems and proposed fixes", "",
        "1. **Missing original-baseline comparison.** The official target claim lacks twelve "
        "required original-baseline runs. **Fix (Proposed):** retain the unverified qualification "
        "until independently run original baselines supply those comparisons. "
        "This audit offers no inference or tuning recommendation.", "",
        "2. **Absolute target shortfalls.** The measured full-system Flights and Rayyan "
        "detection F1 values fail their specified strict targets. **Fix (Proposed):** "
        "report these as failed requirements; do not describe the overall target as achieved.", "",
        "## Failure evidence for reporting only", "",
        "All existing failure samples were independently matched to the frozen prediction "
        "and exact reference strings. The following examples use zero-based data-row indices. "
        "They are deliberately confined to this evaluation-only document. "
        "No tuning action is suggested or authorized by these observations.", "",
    ])
    for dataset in DATASETS:
        result = audit["metrics"][dataset]["zeroclean_loop"]
        partition = result["cell_outcomes"]
        lines.extend([
            f"### {dataset}", "",
            f"Full-system detection FP: {result['detection']['fp']}; repair FP: "
            f"{result['repair']['fp']}, consisting of {partition['clean_wrong_repair']} "
            f"edits to clean cells and {partition['dirty_wrong_repair']} incorrect edits "
            "to erroneous cells. These are full counts, not sample extrapolations.", "",
        ])
        for kind in ("false_positives", "wrong_repairs"):
            selected = samples[dataset][kind][:3]
            lines.extend([f"{kind.replace('_', ' ').capitalize()} (up to three examples):", ""])
            if selected:
                lines.extend(["```json", json.dumps(selected, indent=2, ensure_ascii=False), "```", ""])
            else:
                lines.extend(["None in the frozen full-system output.", ""])
    lines.extend([
        "## Reproducibility and limits", "",
        "Run `python scripts/audit_evaluation.py` from the repository. "
        "The command writes only `results/evaluation_audit.json` and this review. "
        "The audit never imports the project evaluator. SHA-256 identities and all "
        "per-system numerical results are in the JSON audit.", "",
        "Freeze-before-evaluation is supported by the manifest declaration and file "
        "chronology; the audit does not claim timestamps establish an immutable historical "
        "attestation. The frozen source signature and original-baseline execution are "
        "outside this numerical audit. Failure examples are not inference inputs.", "",
    ])
    (root / "results/evaluation_only_failures/review.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
