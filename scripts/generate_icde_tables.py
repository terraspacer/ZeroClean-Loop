#!/usr/bin/env python3
"""Generate the manuscript macros, the four tables, and ``paper/paper_facts.json`` from frozen artifacts.

Every table entry and every macro is computed from the frozen evaluator-owned
JSON artifacts under ``results/`` and from the benchmark CSV files. No number
is typed in by hand. ``paper_facts.json`` is the single fact sheet that the
manuscript prose is checked against by ``scripts/check_paper_numbers.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zeroclean.data import read_csv  # noqa: E402

RESULTS = ROOT / "results"
DATA = ROOT / "vendor" / "ZeroED" / "data"
PAPER = ROOT / "paper"

DATASETS = ["hospital", "flights", "beers", "rayyan", "billionaire", "movies"]
TITLE = {d: d.capitalize() for d in DATASETS}
OURS, SRC, SEQ = "zeroclean_loop", "zeroed", "zeroed_zeroec_reconstruction"


def load(path: str) -> dict:
    return json.loads((RESULTS / path).read_text())


def summary(study: str) -> dict:
    return load(f"{study}/summary_metrics.json")


def f4(x: float) -> str:
    return f"{x:.4f}"


def pp(x: float) -> str:
    """Signed percentage-point difference of two ratios."""
    value = 100.0 * x
    sign = "+" if value >= 0 else "-"
    return f"${sign}{abs(value):.2f}$"


def integer(x: int) -> str:
    return f"{x:,}"


def bold_best(values: list[float], cells: list[str]) -> list[str]:
    best = max(values)
    return [rf"\textbf{{{c}}}" if abs(v - best) < 5e-5 else c for v, c in zip(values, cells)]


def main() -> None:
    facts: dict = {"datasets": {}, "methods": {}, "macro": {}, "diagnostics": {}}

    # ---------------------------------------------------------------- data
    for d in DATASETS:
        cols, dirty = read_csv(DATA / f"{d}_error-01.csv")
        _, clean = read_csv(DATA / f"{d}_clean.csv")
        errors = sum(a != b for r1, r2 in zip(dirty, clean) for a, b in zip(r1, r2))
        facts["datasets"][d] = {"rows": len(dirty), "columns": len(cols), "cells": len(dirty) * len(cols),
                                "errors": errors, "error_rate_percent": 100.0 * errors / (len(dirty) * len(cols))}
    _, flights = read_csv(DATA / "flights_error-01.csv")
    facts["datasets"]["flights"]["sources"] = len({row[0] for row in flights})
    facts["datasets"]["flights"]["entities"] = len({row[1] for row in flights})
    facts["datasets"]["total"] = {k: sum(facts["datasets"][d][k] for d in DATASETS) for k in ["rows", "cells", "errors"]}

    # ------------------------------------------------------------- metrics
    studies = {
        "v6": ("unified_loop_v6", OURS), "v5": ("unified_loop_v5", OURS), "v4": ("unified_loop_v4", OURS),
        "v3": ("unified_loop_v3", OURS), "v2": ("unified_loop_v2", OURS),
        "source_zeroed": ("unified_loop_v6", SRC), "sequential": ("unified_loop_v6", SEQ),
        "empty_carry": ("conservative_seed_v1", "empty_carry"), "seeded_carry": ("conservative_seed_v1", "seeded_carry"),
        "source_loop_v1": ("source_v1", OURS), "format_overlay_v1": ("format_overlay_v1", OURS),
        "wo_multi_column": ("components_source_v2", "without_multi_column_first_pass"),
        "wo_counterfactual": ("components_source_v2", "without_counterfactual_first_pass"),
        "full_first_round": ("components_source_v2", "full_first_verified_round"),
    }
    for key, (study, method) in studies.items():
        s = summary(study)
        facts["methods"][key] = {d: s["metrics"][d][method] for d in DATASETS}
        facts["macro"][key] = s["macro_f1"][method]
    adapted = summary("unified_loop_v6")["adapted_comparison_only"]["metrics"]
    for key, method in [("adapted_zeroed", "adapted_zeroed"), ("adapted_sequential", "adapted_zeroed_zeroec")]:
        facts["methods"][key] = {d: adapted[d][method] for d in DATASETS}
        facts["macro"][key] = {t: sum(adapted[d][method][t]["f1"] for d in DATASETS) / 6 for t in ["detection", "repair"]}
    for key in facts["methods"]:
        for task in ["detection", "repair"]:
            for stat in ["precision", "recall"]:
                facts["macro"][key][f"{task}_{stat}"] = sum(facts["methods"][key][d][task][stat] for d in DATASETS) / 6
    facts["macro"]["sequential"]["detection"] = facts["macro"]["source_zeroed"]["detection"]
    published = load("published_baseline_reference.json")["zeroed_ed_reference"]
    facts["published_zeroed_f1"] = {d: published[d]["f1"] for d in DATASETS}

    flight_spec = summary("flight_model_spec_v1")["metrics"]["flights"]
    facts["flights_grouping"] = {k: v["detection"]["f1"] for k, v in flight_spec.items()
                                 if k.startswith(("attribute_near_", "profile_complete_"))}

    # --------------------------------------------------------- diagnostics
    for version in ["unified_loop_v4", "unified_loop_v6"]:
        block = {}
        for d in DATASETS:
            diag = load(f"{version}/diagnostics/{d}.json")
            update = diag.get("counterfactual_detection_update", {"promoted": [], "demoted": []})
            block[d] = {
                "admitted_rules": diag["rules"]["accepted_rule_count"],
                "candidate_cells": diag["candidate_pool"]["candidate_cells"],
                "distinct_candidates": diag["candidate_pool"]["distinct_candidates"],
                "elapsed_seconds": diag["elapsed_local_seconds"],
                "row_guard_checked": diag["row_guard"]["checked_rows"],
                "row_guard_rolled_back": diag["row_guard"]["rolled_back_rows"],
                "detected_cells": diag["repair"]["counts"]["detection_positive_cells"],
                "selected_cells": diag["repair"]["counts"]["selected_cells"],
                "abstained_cells": diag["repair"]["counts"]["abstained_cells"],
                "counterfactual_promoted": len(update["promoted"]),
                "counterfactual_demoted": len(update["demoted"]),
                "gated_columns": diag["detection"]["aggregate"]["gated_column_count"],
            }
            if version == "unified_loop_v6":
                block[d]["generic_structure"] = {k: diag["detection"]["v6_structural_promotion"][k]
                                                 for k in ["candidate_count", "promoted_count", "retained_count"]}
                fallback = diag["detection"]["v6_fallback_structural_promotion"]
                block[d]["fallback_structure"] = {k: fallback[k] for k in ["promoted_count", "retained_count"] if k in fallback}
        block["total"] = {k: sum(block[d][k] for d in DATASETS) for k in
                          ["admitted_rules", "candidate_cells", "distinct_candidates", "elapsed_seconds",
                           "row_guard_checked", "row_guard_rolled_back", "detected_cells", "selected_cells",
                           "abstained_cells", "counterfactual_promoted", "counterfactual_demoted"]}
        facts["diagnostics"][version] = block
    audit = load("unified_loop_v6/independent_audit.json")
    facts["audit"] = {"outputs": audit["evaluated_output_count"], "mismatches": len(audit["mismatches"]),
                      "hash_bindings": audit["hash_verification"]["file_or_absence_count"],
                      "integer_fields": audit["comparison"]["integer_fields_checked"],
                      "ratio_fields": audit["comparison"]["ratio_fields_checked"],
                      "address_space_limit_gib": audit["resource_policy"]["address_space_limit_bytes"] / 2 ** 30,
                      "threads": audit["resource_policy"]["numerical_threads"]}

    # ---------------------------------------- audit-log and prediction recounts
    from collections import Counter
    facts["detection_actions"], facts["repair_by_column"], facts["errors_by_column"] = {}, {}, {}
    for d in DATASETS:
        actions, witnesses = Counter(), Counter()
        with (RESULTS / "unified_loop_v6" / "audit" / f"{d}.jsonl").open() as handle:
            for line in handle:
                rec = json.loads(line)
                if rec.get("event") != "detection":
                    continue
                actions[rec["action"]] += 1
                for kind in set(rec.get("evidence_types") or []):
                    witnesses[kind] += 1
        facts["detection_actions"][d] = {"actions": dict(actions), "witnesses": dict(witnesses)}
        cols, dirty = read_csv(DATA / f"{d}_error-01.csv")
        _, clean = read_csv(DATA / f"{d}_clean.csv")
        errors, blanks = Counter(), Counter()
        for r1, r2 in zip(dirty, clean):
            for j, (a, b) in enumerate(zip(r1, r2)):
                if a != b:
                    errors[cols[j]] += 1
                    blanks[cols[j]] += a == ""
        facts["errors_by_column"][d] = {c: {"errors": errors[c], "blank": blanks[c]} for c in errors}
        facts["errors_by_column"][d]["_total_blank"] = sum(blanks.values())
        repairs = load(f"unified_loop_v6/predictions/{d}/{OURS}.json")["repairs"]
        correct, wrong = Counter(), Counter()
        for r in repairs:
            (correct if clean[r["row"]][r["column"]] == r["value"] else wrong)[cols[r["column"]]] += 1
        facts["repair_by_column"][d] = {c: {"correct": correct[c], "wrong": wrong[c]} for c in set(correct) | set(wrong)}
    facts["billionaire_rejections"] = {}
    for version in ["unified_loop_v3", "unified_loop_v4", "unified_loop_v6"]:
        reasons = Counter()
        with (RESULTS / version / "audit" / "billionaire.jsonl").open() as handle:
            for line in handle:
                rec = json.loads(line)
                if rec.get("event") == "repair_selection":
                    for review in rec.get("candidate_reviews", []):
                        for reason in review.get("rejection_reasons", []):
                            reasons[reason] += 1
        diag = load(f"{version}/diagnostics/billionaire.json")
        facts["billionaire_rejections"][version] = {
            "reasons": dict(reasons), "admitted_rules": diag["rules"]["accepted_rule_count"],
            "selected_cells": diag["repair"]["counts"]["selected_cells"],
            "abstained_cells": diag["repair"]["counts"]["abstained_cells"],
            "row_guard_checked": diag["row_guard"]["checked_rows"],
            "row_guard_rolled_back": diag["row_guard"]["rolled_back_rows"]}

    M, MAC, DG = facts["methods"], facts["macro"], facts["diagnostics"]["unified_loop_v6"]
    header = ["% Auto-generated by scripts/generate_icde_tables.py from frozen evaluator-owned JSON artifacts.",
              "% Do not edit by hand."]
    files: dict[str, list[str]] = {}
    out: list[str] = list(header)
    files["generated_macros.tex"] = out

    def macro(name: str, value: str) -> None:
        out.append(rf"\newcommand{{\{name}}}{{{value}}}")

    macro("VsixMacroED", f4(MAC["v6"]["detection"]))
    macro("VsixMacroRepair", f4(MAC["v6"]["repair"]))
    macro("VfourMacroED", f4(MAC["v4"]["detection"]))
    macro("VfourMacroRepair", f4(MAC["v4"]["repair"]))
    macro("SourceMacroED", f4(MAC["source_zeroed"]["detection"]))
    macro("SeqMacroRepair", f4(MAC["sequential"]["repair"]))
    gains = {
        "VsixEDGain": MAC["v6"]["detection"] - MAC["source_zeroed"]["detection"],
        "VsixRepairGain": MAC["v6"]["repair"] - MAC["sequential"]["repair"],
        "VfourEDGain": MAC["v4"]["detection"] - MAC["source_zeroed"]["detection"],
        "VfourRepairGain": MAC["v4"]["repair"] - MAC["sequential"]["repair"],
    }
    for name, value in gains.items():
        macro(name, f"{100 * value:.2f}~pp")
    facts["macro_gains_pp"] = {k: 100 * v for k, v in gains.items()}
    out = list(header)
    files["generated_table_datasets.tex"] = out

    # ------------------------------------------------------------- Table I
    out += [r"\begin{table}[t]", r"\centering",
            r"\caption{Benchmark Datasets and ZeroClean-Loop v6 Execution Profile: Tuples $|R|$, Attributes $|C|$, True Errors $|\mathcal{E}^{\star}|$, Admitted Rules $|\mathcal{Q}_{\mathrm{adm}}|$, Distinct Candidates $|\mathcal{P}_{\mathrm{cand}}|$, Flagged Cells $N_{\mathrm{pos}}$, and Inference Time $t_{\mathrm{inf}}$.}",
            r"\label{tab:datasets}", r"\footnotesize", r"\setlength{\tabcolsep}{3.6pt}",
            r"\begin{tabular}{lrrrrrrr}", r"\toprule",
            r"\textbf{Dataset} & $|R|$ & $|C|$ & $|\mathcal{E}^{\star}|$ & $|\mathcal{Q}_{\mathrm{adm}}|$ & $|\mathcal{P}_{\mathrm{cand}}|$ & $N_{\mathrm{pos}}$ & $t_{\mathrm{inf}}$ (s) \\",
            r"\midrule"]
    for d in DATASETS:
        ds, g = facts["datasets"][d], DG[d]
        assert g["detected_cells"] == M["v6"][d]["detection"]["predicted"]
        out.append(f"{TITLE[d]} & {integer(ds['rows'])} & {ds['columns']} & {integer(ds['errors'])} & {g['admitted_rules']} & "
                   f"{integer(g['distinct_candidates'])} & {integer(g['detected_cells'])} & {g['elapsed_seconds']:.2f} \\\\")
    tot = facts["datasets"]["total"]
    out += [r"\midrule",
            f"\\textbf{{Total}} & {integer(tot['rows'])} & {sum(facts['datasets'][d]['columns'] for d in DATASETS)} & {integer(tot['errors'])} & "
            f"{DG['total']['admitted_rules']} & {integer(DG['total']['distinct_candidates'])} & {integer(DG['total']['detected_cells'])} & "
            f"{DG['total']['elapsed_seconds']:.2f} \\\\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    # ------------------------------------------------------------ Table II
    out = list(header)
    files["generated_table_benchmark.tex"] = out
    out += [r"\begin{table*}[t]", r"\centering",
            r"\caption{Parallel Six-Dataset Comparison of Error Detection and Error Correction. Pub.\ Ref Is the Uncontrolled Published ZeroED Score~\cite{ni2025zeroed}; All Other Columns Are Measured Under the Same Exact-String Evaluator. The Last Row Gives the Macro Gain in Percentage Points (pp) Over Src ED (Detection) and Seq EC (Repair). Bold Marks the Best Measured $F_1$ per Row and Task.}",
            r"\label{tab:main_benchmark}", r"\footnotesize", r"\setlength{\tabcolsep}{4.2pt}",
            r"\begin{tabular}{l rrrrrrr rrrrrrr}", r"\toprule",
            r"& \multicolumn{7}{c}{\textbf{Error Detection (ED)}} & \multicolumn{7}{c}{\textbf{Error Correction (EC)}} \\",
            r"\cmidrule(lr){2-8} \cmidrule(lr){9-15}",
            r"& Pub.\ Ref & Src ED & Adp ED & v4 & \multicolumn{3}{c}{v6} & Adp EC & Seq EC & Seed EC & v4 & \multicolumn{3}{c}{v6} \\",
            r"\cmidrule(lr){6-8} \cmidrule(lr){13-15}",
            r"\textbf{Dataset} & $F_1^{\mathrm{ED}}$ & $F_1^{\mathrm{ED}}$ & $F_1^{\mathrm{ED}}$ & $F_1^{\mathrm{ED}}$ & $P^{\mathrm{ED}}$ & $R^{\mathrm{ED}}$ & $F_1^{\mathrm{ED}}$ & $F_1^{\mathrm{EC}}$ & $F_1^{\mathrm{EC}}$ & $F_1^{\mathrm{EC}}$ & $F_1^{\mathrm{EC}}$ & $P^{\mathrm{EC}}$ & $R^{\mathrm{EC}}$ & $F_1^{\mathrm{EC}}$ \\",
            r"\midrule"]

    def row(label: str, get) -> str:
        ed = [get("source_zeroed", "detection", "f1"), get("adapted_zeroed", "detection", "f1"),
              get("v4", "detection", "f1"), get("v6", "detection", "f1")]
        ec = [get("adapted_sequential", "repair", "f1"), get("sequential", "repair", "f1"),
              get("seeded_carry", "repair", "f1"), get("v4", "repair", "f1"), get("v6", "repair", "f1")]
        edc, ecc = bold_best(ed, [f4(v) for v in ed]), bold_best(ec, [f4(v) for v in ec])
        cells = [get("published", "detection", "f1"), edc[0], edc[1], edc[2],
                 f4(get("v6", "detection", "precision")), f4(get("v6", "detection", "recall")), edc[3],
                 ecc[0], ecc[1], ecc[2], ecc[3],
                 f4(get("v6", "repair", "precision")), f4(get("v6", "repair", "recall")), ecc[4]]
        return f"{label} & " + " & ".join(cells) + r" \\"

    for d in DATASETS:
        def get(method, task, stat, d=d):
            if method == "published":
                return f"{facts['published_zeroed_f1'][d]:.3f}"
            return M[method][d][task][stat]
        out.append(row(TITLE[d], get))

    def get_macro(method, task, stat):
        if method == "published":
            return f"{sum(facts['published_zeroed_f1'].values()) / 6:.3f}"
        return MAC[method][task] if stat == "f1" else MAC[method][f"{task}_{stat}"]
    out += [r"\midrule", row(r"\textbf{Macro Mean}", get_macro)]
    base_ed, base_ec = MAC["source_zeroed"]["detection"], MAC["sequential"]["repair"]
    gain_cells = ["--", "--", pp(MAC["adapted_zeroed"]["detection"] - base_ed), pp(MAC["v4"]["detection"] - base_ed), "--", "--",
                  pp(MAC["v6"]["detection"] - base_ed), pp(MAC["adapted_sequential"]["repair"] - base_ec), "--",
                  pp(MAC["seeded_carry"]["repair"] - base_ec), pp(MAC["v4"]["repair"] - base_ec), "--", "--",
                  pp(MAC["v6"]["repair"] - base_ec)]
    out += [r"\textbf{Gain (pp)} & " + " & ".join(gain_cells) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    # ----------------------------------------------------------- Table III
    out = list(header)
    files["generated_table_trajectory.tex"] = out
    cohorts = [("Source and Sequential", "sequential"), ("Source Loop v1", "source_loop_v1"),
               ("Format Overlay v1", "format_overlay_v1"), ("Empty Carry", "empty_carry"), ("Seeded Carry", "seeded_carry"),
               ("ZeroClean-Loop v2", "v2"), ("ZeroClean-Loop v3", "v3"), ("ZeroClean-Loop v4", "v4"),
               ("ZeroClean-Loop v5", "v5"), ("ZeroClean-Loop v6", "v6")]
    facts["positive_repair_datasets"] = {}
    columns = [[], [], [], []]
    wins = []
    for _, key in cohorts:
        det_key = "source_zeroed" if key == "sequential" else key
        columns[0].append(MAC[det_key]["detection"])
        columns[1].append(MAC[key]["repair"])
        columns[2].append(M[det_key]["flights"]["detection"]["f1"])
        columns[3].append(M[det_key]["rayyan"]["detection"]["f1"])
        positive = sum(M[key][d]["repair"]["f1"] > M["sequential"][d]["repair"]["f1"] + 1e-12 for d in DATASETS)
        facts["positive_repair_datasets"][key] = positive
        wins.append("--" if key == "sequential" else f"{positive}/6")
    cells = [bold_best(col, [f4(v) for v in col]) for col in columns]
    out += [r"\begin{table}[t]", r"\centering",
            r"\caption{Progressive Evolution Across Frozen Study Cohorts. Flig and Rayy Give the Detection Score $F_1^{\mathrm{ED}}$ on Flights and Rayyan; Pos.\ EC Counts the Datasets on Which the Repair Score $F_1^{\mathrm{EC}}$ Exceeds the Sequential Reconstruction.}",
            r"\label{tab:trajectory}", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"\textbf{Study Cohort} & $\bar{F}_1^{\mathrm{ED}}$ & $\bar{F}_1^{\mathrm{EC}}$ & Flig & Rayy & Pos.\ EC \\",
            r"\midrule"]
    for k, (label, _) in enumerate(cohorts):
        out.append(f"{label} & {cells[0][k]} & {cells[1][k]} & {cells[2][k]} & {cells[3][k]} & {wins[k]} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    # ------------------------------------------------------------ Table IV
    out = list(header)
    files["generated_table_ablation.tex"] = out
    arms = ["wo_multi_column", "wo_counterfactual", "full_first_round"]
    out += [r"\begin{table}[t]", r"\centering",
            r"\caption{Component Ablation of the First Verification Pass on Frozen Inputs: Without Multi-Column Rules and FDs (w/o MC), Without Counterfactual Features (w/o CF), and With Both (Full).}",
            r"\label{tab:ablation}", r"\footnotesize", r"\setlength{\tabcolsep}{4.6pt}",
            r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"& \multicolumn{2}{c}{\textbf{w/o MC}} & \multicolumn{2}{c}{\textbf{w/o CF}} & \multicolumn{2}{c}{\textbf{Full}} \\",
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
            r"\textbf{Dataset} & $F_1^{\mathrm{ED}}$ & $F_1^{\mathrm{EC}}$ & $F_1^{\mathrm{ED}}$ & $F_1^{\mathrm{EC}}$ & $F_1^{\mathrm{ED}}$ & $F_1^{\mathrm{EC}}$ \\",
            r"\midrule"]

    def ablation_row(label, det, rep):
        dc, rc = bold_best(det, [f4(v) for v in det]), bold_best(rep, [f4(v) for v in rep])
        return f"{label} & " + " & ".join(f"{a} & {b}" for a, b in zip(dc, rc)) + r" \\"

    for d in DATASETS:
        out.append(ablation_row(TITLE[d], [M[a][d]["detection"]["f1"] for a in arms], [M[a][d]["repair"]["f1"] for a in arms]))
    out += [r"\midrule", ablation_row(r"\textbf{Macro Mean}", [MAC[a]["detection"] for a in arms], [MAC[a]["repair"] for a in arms]),
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    for name, lines in files.items():
        (PAPER / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (PAPER / "paper_facts.json").write_text(json.dumps(facts, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", ", ".join(files), "and paper_facts.json")


if __name__ == "__main__":
    main()
