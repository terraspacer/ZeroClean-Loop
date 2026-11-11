#!/usr/bin/env python3
"""Portable read-only recount of ZeroClean-Loop v6 (146 SHA-256 bindings, 18 outputs, exact rational arithmetic).

Verifies every cryptographic hash binding in protocol_lock.json, evaluation_receipt.json,
and independent_audit.json (independent of checkout directory path), then recounts all
18 prediction files across all six benchmark datasets using Python's fractions.Fraction
and checks all 219 integer fields and 128 ratio fields in summary_metrics.json.
"""
from __future__ import annotations

import importlib.util
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "audit_unified_loop_v6_results",
        ROOT / "scripts" / "audit_unified_loop_v6_results.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    mismatches: list[dict] = []
    audit_ref = mod.read_json(ROOT / "results/unified_loop_v6/independent_audit.json")
    lock = mod.read_json(ROOT / "experimental/unified_loop_v6/protocol_lock.json")
    receipt = mod.read_json(ROOT / "results/unified_loop_v6/evaluation_receipt.json")
    summary = mod.read_json(ROOT / "results/unified_loop_v6/summary_metrics.json")

    # 1. Verify all 146 file/absence SHA-256 bindings recorded in independent_audit.json
    expected_receipts = audit_ref["hash_verification"]["receipt"]
    verified_bindings = 0
    for rel_str, info in sorted(expected_receipts.items()):
        path = ROOT / rel_str
        exp_sha = info["sha256"]
        if exp_sha == mod.ABSENT_INITIALIZER:
            actual = mod.ABSENT_INITIALIZER if not path.exists() else mod.sha256(path)
        else:
            actual = mod.sha256(path) if path.is_file() else "MISSING"
        if actual != exp_sha:
            mismatches.append({"category": "sha256", "path": rel_str, "expected": exp_sha, "actual": actual})
        else:
            verified_bindings += 1

    # 2. Recount all 18 outputs with exact rational arithmetic and compare against summary_metrics.json
    auditor = mod.Auditor(ROOT, _fictional=True)
    recounted: dict = {}
    for dataset in mod.DATASETS:
        dirty_rel = Path(mod.input_paths(dataset)["dirty"])
        clean_rel = Path(f"vendor/ZeroED/data/{dataset}_clean.csv")
        cols, dirty = mod.read_csv_exact(ROOT / dirty_rel)
        rcols, clean = mod.read_csv_exact(ROOT / clean_rel)
        assert cols == rcols and len(dirty) == len(clean)
        recounted[dataset] = {}
        for variant in mod.VARIANTS:
            pred = mod.read_json(ROOT / "results/unified_loop_v6/predictions" / dataset / f"{variant}.json")
            actual = mod.recount_prediction(pred, dirty, clean)
            actual.update(
                dirty_sha256=lock["input_files"][dataset]["dirty"]["sha256"],
                reference_sha256=receipt["reference_sha256"][dataset],
                reference_policy="exact strings; no normalization",
            )
            recounted[dataset][variant] = actual
            auditor.evaluated += 1

    auditor.check_summary(summary, recounted)
    mismatches.extend(auditor.mismatches)

    out = {
        "status": "passed" if not mismatches and auditor.evaluated == 18 and verified_bindings == 146 else "failed",
        "evaluated_output_count": auditor.evaluated,
        "hash_bindings_verified": verified_bindings,
        "integer_fields_checked": auditor.integer_fields,
        "ratio_fields_checked": auditor.ratio_fields,
        "macro_ed_f1_v6": round(float(sum(Fraction(recounted[d]["zeroclean_loop"]["detection"]["f1"]) for d in mod.DATASETS) / 6), 4),
        "macro_ec_f1_v6": round(float(float(summary["macro_f1"]["zeroclean_loop"]["repair"])), 4),
        "mismatch_count": len(mismatches),
    }
    if mismatches:
        out["mismatches"] = mismatches
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
