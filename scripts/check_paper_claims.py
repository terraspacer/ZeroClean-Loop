#!/usr/bin/env python3
"""Mechanical checks of ``paper/main.tex`` against the frozen artifacts and the editorial rules.

1. Every numeral in the abstract, the introduction, Section V, and the conclusion must equal a value in
   ``paper/paper_facts.json`` or ``paper/figures/figure_data.json`` (both generated from frozen artifacts),
   or a derived value that this script recomputes from the artifacts, or a documented protocol constant.
2. Editorial rules: no hedging words in the manuscript, the generated tables, or the text printed in
   any figure PDF; a "where" clause after every display equation; no thousands separators inside
   math mode; and a space between a number and its unit.

Exit status 0 means that every check passes.  The script reads artifacts only; it runs no experiment.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zeroclean.data import read_csv  # noqa: E402

PAPER, RESULTS, DATA = ROOT / "paper", ROOT / "results", ROOT / "vendor" / "ZeroED" / "data"
DATASETS = ["hospital", "flights", "beers", "rayyan", "billionaire", "movies"]
HEDGES = ["may", "might", "probably", "should", "could", "likely"]


def load(path):
    return json.loads(Path(path).read_text())


def flatten(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            flatten(value, out)
    elif isinstance(obj, list):
        for value in obj:
            flatten(value, out)


def forms(value):
    """Printed forms under which a measured value is allowed to appear."""
    result = set()
    if isinstance(value, int) or float(value).is_integer():
        result.add(str(int(value)))
    value = float(value)
    for scale in (1.0, 100.0):
        for digits in (1, 2, 3, 4):
            result.add(f"{abs(value) * scale:.{digits}f}")
    return result


def recount():
    """Derived values, recomputed from the artifacts. Returns {label: value}."""
    facts = load(PAPER / "paper_facts.json")
    figure = load(PAPER / "figures" / "figure_data.json")
    M, MAC = facts["methods"], facts["macro"]
    derived = {}

    def det(method, d):
        return M[method][d]["detection"]

    def rep(method, d):
        return M[method][d]["repair"]

    derived["source flags"] = sum(det("source_zeroed", d)["predicted"] for d in DATASETS)
    derived["source false positives"] = sum(det("source_zeroed", d)["fp"] for d in DATASETS)
    derived["v6 flags"] = sum(det("v6", d)["predicted"] for d in DATASETS)
    derived["v6 false positives"] = sum(det("v6", d)["fp"] for d in DATASETS)
    for d in DATASETS:
        derived[f"ED gain v6-source {d}"] = 100 * (det("v6", d)["f1"] - det("source_zeroed", d)["f1"])
        derived[f"ED gain v6-v4 {d}"] = 100 * (det("v6", d)["f1"] - det("v4", d)["f1"])
        derived[f"EC gain v6-seq {d}"] = 100 * (rep("v6", d)["f1"] - rep("sequential", d)["f1"])
        derived[f"EC gain v6-v4 {d}"] = 100 * (rep("v6", d)["f1"] - rep("v4", d)["f1"])
    ec_gains = [derived[f"EC gain v6-seq {d}"] for d in DATASETS]
    derived["EC gain sum"] = sum(ec_gains)
    derived["EC gain mean without flights"] = (sum(ec_gains) - derived["EC gain v6-seq flights"]) / 5
    derived["macro EC v6-seeded"] = 100 * (MAC["v6"]["repair"] - MAC["seeded_carry"]["repair"])
    derived["macro EC v6-v4"] = 100 * (MAC["v6"]["repair"] - MAC["v4"]["repair"])
    derived["published macro"] = sum(facts["published_zeroed_f1"].values()) / 6

    fates = figure["fig_diagnostics"]["cell_fates"]
    total = lambda key: sum(fates[d][key] for d in DATASETS)  # noqa: E731
    derived["dropped clean"], derived["dropped error"] = total("dropped_clean"), total("dropped_error")
    derived["dropped"] = derived["dropped clean"] + derived["dropped error"]
    derived["dropped clean share"] = 100 * derived["dropped clean"] / derived["dropped"]
    derived["billionaire dropped"] = fates["billionaire"]["dropped_clean"] + fates["billionaire"]["dropped_error"]
    for key in ("correct_edit", "wrong_edit", "no_edit"):
        derived[key], derived[key + " share"] = total(key), 100 * total(key) / total("flagged")
    derived["beers wrong share"] = 100 * fates["beers"]["wrong_edit"] / rep("v6", "beers")["predicted"]
    derived["beers extra correct"] = rep("v6", "beers")["tp"] - rep("v4", "beers")["tp"]
    derived["beers extra wrong"] = rep("v6", "beers")["fp"] - rep("v4", "beers")["fp"]

    for arm, name in (("wo_multi_column", "MC"), ("wo_counterfactual", "CF")):
        derived[f"ablation ED drop {name}"] = 100 * (MAC["full_first_round"]["detection"] - MAC[arm]["detection"])
        derived[f"ablation EC drop {name}"] = 100 * (MAC["full_first_round"]["repair"] - MAC[arm]["repair"])
    derived["ablation Billionaire EC drop MC"] = 100 * (rep("full_first_round", "billionaire")["f1"] - rep("wo_multi_column", "billionaire")["f1"])
    derived["ablation max other ED difference"] = 100 * max(
        abs(det("full_first_round", d)["f1"] - det(arm, d)["f1"])
        for d in DATASETS if d != "flights" for arm in ("wo_multi_column", "wo_counterfactual"))

    g = facts["flights_grouping"]
    settings = ["lambda0p0", "lambda0p5", "lambda0p9", "selectedEB", "equal"]
    gaps = [100 * (g[f"profile_complete_{s}"] - g[f"attribute_near_{s}"]) for s in settings]
    derived["grouping gap min"], derived["grouping gap max"] = min(gaps), max(gaps)
    learned = [100 * (g[f"{grp}_equal"] - g[f"{grp}_{s}"]) for grp in ("attribute_near", "profile_complete") for s in settings[:4]]
    derived["equal control margin min"], derived["equal control margin max"] = min(learned), max(learned)
    derived["v6 below best control"] = 100 * (g["profile_complete_equal"] - det("v6", "flights")["f1"])

    # ---- recounts from predictions, tables, and audit logs ---------------------------------
    tables = {d: (read_csv(DATA / f"{d}_error-01.csv"), read_csv(DATA / f"{d}_clean.csv")[1]) for d in DATASETS}

    def mask(study, d):
        probabilities = load(RESULTS / study / "predictions" / d / "zeroclean_loop.json")["probabilities"]
        return [[p > 0.5 for p in row] for row in probabilities]

    (cols, dirty), clean = tables["rayyan"]
    m4, m6 = mask("unified_loop_v4", "rayyan"), mask("unified_loop_v6", "rayyan")
    only4 = [(i, j) for i in range(len(dirty)) for j in range(len(cols)) if m4[i][j] and not m6[i][j]]
    derived["rayyan v4-only flags"] = len(only4)
    derived["rayyan v4-only empty"] = sum(dirty[i][j] == "" for i, j in only4)
    derived["rayyan v4-only true errors"] = sum(dirty[i][j] != clean[i][j] for i, j in only4)
    for name, count in Counter(cols[j] for _, j in only4).items():
        derived[f"rayyan v4-only {name}"] = count
    date = cols.index("article_jcreated_at")
    for label, m in (("v4", m4), ("v6", m6)):
        missed = [i for i in range(len(dirty)) if dirty[i][date] != clean[i][date] and not m[i][date]]
        assert all(dirty[i][date] != "" for i in missed)
        derived[f"rayyan date false negatives {label}"] = len(missed)
    derived["rayyan empty errors"] = facts["errors_by_column"]["rayyan"]["_total_blank"]
    source_mask = [[p > 0.5 for p in row] for row in load(RESULTS / "unified_loop_v6" / "predictions" / "rayyan" / "zeroed.json")["probabilities"]]
    for name in ("journal_issn", "article_pagination"):
        derived[f"rayyan source positive fraction {name}"] = sum(row[cols.index(name)] for row in source_mask) / len(source_mask)

    guard = Counter()
    for d in DATASETS:
        (cols, dirty), clean = tables[d]
        repairs = {(r["row"], r["column"]): r["value"] for r in load(RESULTS / "unified_loop_v6" / "predictions" / d / "zeroclean_loop.json")["repairs"]}
        with (RESULTS / "unified_loop_v6" / "audit" / f"{d}.jsonl").open() as handle:
            for line in handle:
                rec = json.loads(line)
                event = rec.get("event")
                if event == "detection":
                    error = dirty[rec["row"]][rec["column"]] != clean[rec["row"]][rec["column"]]
                    if rec["action"] == "promote_v6_generic_structure":
                        derived[f"G promoted {d}"] = derived.get(f"G promoted {d}", 0) + 1
                        derived[f"G promoted true {d}"] = derived.get(f"G promoted true {d}", 0) + int(error)
                    if rec["action"] == "retain_v6_fallback_structure":
                        derived[f"B retained {d}"] = derived.get(f"B retained {d}", 0) + 1
                        derived[f"B retained true {d}"] = derived.get(f"B retained true {d}", 0) + int(error)
                    if "independent_source_consensus_contradiction" in (rec.get("evidence_types") or []):
                        derived[f"K {d}"] = derived.get(f"K {d}", 0) + 1
                        kind = "promoted" if rec["action"].startswith("promote") else "retained"
                        derived[f"K {kind} {d}"] = derived.get(f"K {kind} {d}", 0) + 1
                elif event == "combined_row_guard":
                    guard["rows"] += 1
                    guard["rolled back"] += rec["rolled_back"]
                    if rec["new_rule_violation"] and not rec["rolled_back"]:
                        guard["committed with violation"] += 1
                        guard[f"committed with violation {d}"] += 1
                        for j in rec["columns"]:
                            guard["edits in such rows"] += 1
                            guard["correct edits in such rows"] += repairs[(rec["row"], j)] == clean[rec["row"]][j]
        if d == "beers":
            wrong = Counter()
            for (i, j), value in repairs.items():
                if value != clean[i][j]:
                    wrong[cols[j]] += 1
                    if cols[j] == "ounces" and dirty[i][j] == clean[i][j]:
                        wrong["ounces on clean cell"] += 1
                    if cols[j] == "abv" and dirty[i][j].endswith("%") and value == dirty[i][j].rstrip("%"):
                        wrong["abv percent stripped"] += 1
            derived["beers wrong ounces"], derived["beers wrong abv"] = wrong["ounces"], wrong["abv"]
            derived["beers wrong two attributes"] = wrong["ounces"] + wrong["abv"]
            derived["beers ounces on clean"], derived["beers abv stripped"] = wrong["ounces on clean cell"], wrong["abv percent stripped"]
        if d == "movies":
            by_column = facts["repair_by_column"]["movies"]
            derived["movies correct year genre duration"] = sum(by_column[c]["correct"] for c in ("First Shown Year", "Genre", "Duration"))
    for key, value in guard.items():
        derived[f"row guard {key}"] = value
    v4_diag = facts["diagnostics"]["unified_loop_v4"]
    derived["v4 row guard rows"] = sum(v4_diag[d]["row_guard_checked"] for d in DATASETS)
    derived["v4 row guard rolled back"] = sum(v4_diag[d]["row_guard_rolled_back"] for d in DATASETS)
    derived["candidate cells"] = sum(facts["diagnostics"]["unified_loop_v6"][d]["candidate_cells"] for d in DATASETS)
    derived["counterfactual changes"] = sum(facts["diagnostics"]["unified_loop_v6"][d][k] for d in DATASETS for k in ("counterfactual_promoted", "counterfactual_demoted"))
    relation = load(RESULTS / "unified_loop_v6" / "diagnostics" / "hospital.json")["detection"]["intrinsic_context"]["qualified_relations"][0]
    assert relation["determinants"] == ["ProviderNumber"] and relation["dependent"] == "HospitalName"
    for key in ("repeated_groups", "observations", "global_purity", "marginal_purity"):
        derived[f"hospital FD {key}"] = relation[key]
    inputs = load(RESULTS / "unified_loop_v6" / "diagnostics" / "flights.json")["consensus"]["key_discovery"]
    derived["flights entities"] = inputs["entity_count"]
    return facts, figure, derived


# Constants of the protocol and of the running example that are not measurements.
CONSTANTS = {
    "16": "4 x 4 cells displayed in Fig. 1", "54": "9 pinned inputs x 6 datasets", "38": "Flights sources",
    "98": "attribute-near agreement threshold (protocol)", "0.5": "pooling strength (protocol)",
    "0.9": "pooling strength (protocol)", "0.80": "coverage gate", "1.0000": "source recall on Flights",
    "0.50": "threshold", "0.49": "demotion score", "0.75": "promotion score", "1.00": "trace value",
    "27": "sources of the example flight",
}
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def example_sentences(example):
    """The sentences of the introduction that describe Fig. 1, rebuilt from the drawn data."""
    o, c = example["open_loop"], example["closed_loop"]
    assert o["flagged"] == o["shown"] and o["correct"] == 0 and c["flagged"] == c["flagged_true"]
    return [
        rf"contains {example['errors_shown']} erroneous cells among the {example['time_values_shown']} time values",
        rf"the detector flags all {o['shown']} displayed cells, including the source names, and the repairer edits {WORDS[o['edits']]} cells",
        "and none equals the reference value",
        rf"flags {c['flagged']} of the {o['shown']} displayed cells, all of them erroneous, and commits {c['edits']} edits of which {c['correct']} equal the reference value",
        rf"{WORDS[c['missed']]} erroneous cell of the fourth report remains undetected",
        rf"{WORDS[len(example['columns'])]} time attributes in {WORDS[len(example['sources'])]} of the {example['entity_source_count']} source reports",
    ]


def strip_tex(text):
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\begin\{(equation|algorithm|figure\*?)\}.*?\\end\{\1\}", lambda m: m.group(0) if "caption" in m.group(0) and "equation" not in m.group(1) else " ", text, flags=re.S)
    text = re.sub(r"\\(ref|eqref|cite|label|input|includegraphics(\[[^\]]*\])?|texttt|bibliographystyle|bibliography|IEEEtriggeratref)\{[^{}]*\}", " ", text)
    for literal in ("Qwen2.5-72b", "gpt-6.1-sol", "Python 3.12.3", "256-bit", "SHA-256", "AA-404-MIA-MCO"):
        text = text.replace(literal, " ")
    return text


def main():
    tex = Path(sys.argv[1] if len(sys.argv) > 1 else PAPER / "main.tex").read_text(encoding="utf-8")
    problems = []

    # ---- editorial rules -----------------------------------------------------------------
    sources = {"main.tex": tex}
    for path in sorted(PAPER.glob("generated_*.tex")) + sorted((PAPER / "figures").glob("fig_*.tex")):
        sources[path.name] = path.read_text(encoding="utf-8")
    # the data plots are drawn by matplotlib and have no TeX source, so the text that is
    # actually printed in every figure PDF is read back and checked as well
    figure_pdfs = sorted((PAPER / "figures").glob("fig_*.pdf"))
    for path in figure_pdfs:
        printed = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True).stdout
        if not printed.strip():
            problems.append(f"{path.name}: no extractable text (fonts are not embedded as text)")
        sources[path.name] = printed
    for name, text in sources.items():
        body = re.sub(r"(?<!\\)%.*", "", text)
        for word in HEDGES:
            for match in re.finditer(rf"\b{word}\b", body, flags=re.I):
                problems.append(f"{name}: hedging word '{match.group(0)}' near '{body[max(0, match.start() - 40):match.end() + 20]!r}'")
    equations = re.findall(r"\\end\{equation\}\s*(\S+)", tex)
    for following in equations:
        if following != "where":
            problems.append(f"equation not followed by a where clause (next token: {following})")
    for segment in re.sub(r"\\\$", "", tex).split("$")[1::2]:
        if re.search(r"\d,\d{3}", segment):
            problems.append(f"thousands separator inside math mode: ${segment[:60]}$")
    for match in re.finditer(r"\d\s?\\%|\d (pp|GiB|s)\b|\d(pp|GiB)\b", re.sub(r"\\texttt\{[^{}]*\}", "", tex)):
        problems.append(f"number and unit without a protected space: {match.group(0)}")

    # ---- numeric claims ---------------------------------------------------------------
    facts, figure, derived = recount()
    values = []
    flatten(facts, values)
    flatten(figure, values)
    allowed = set(CONSTANTS)
    for value in values + list(derived.values()):
        allowed |= forms(value)
    start, related, conclusion = tex.index(r"\section{Experimental Evaluation}"), tex.index(r"\section{Related Work}"), tex.index(r"\section{Conclusion}")
    preliminaries = tex.index(r"\section{Preliminaries and Protocol Boundary}")
    checked = strip_tex(tex[tex.index(r"\begin{abstract}"):preliminaries] + tex[start:related] + tex[conclusion:])
    tokens = re.findall(r"(?<![\w.\\{^_])\d[\d,]*(?:\.\d+)?", checked)
    unmatched = Counter()
    for token in tokens:
        plain = token.replace(",", "").rstrip(".")
        if plain in allowed or (plain.isdigit() and int(plain) <= 12):
            continue
        unmatched[token] += 1
    for token, count in sorted(unmatched.items()):
        problems.append(f"numeral without a matching artifact value: {token} (x{count})")
    # the counts quoted for the running example are small integers, so they are checked as sentences
    flat = re.sub(r"\s+", " ", re.sub(r"Fig\.~\\ref\{fig:example\}\([abc]\)", "", tex))
    sentences = example_sentences(figure["fig_example"])
    for sentence in sentences:
        if sentence not in flat:
            problems.append(f"running example: the text does not state '{sentence}'")

    print(f"equations with where clauses: {len(equations)}")
    print(f"figure PDFs whose printed text is checked: {len(figure_pdfs)}")
    print(f"numerals checked in abstract, introduction, Section V, and conclusion: {len(tokens)}")
    print(f"statements about the running example matched against the drawn data: {len(sentences)}")
    print(f"derived values recomputed from artifacts: {len(derived)}")
    for label in sorted(derived):
        value = derived[label]
        print(f"  {label}: {value:.4f}" if isinstance(value, float) else f"  {label}: {value}")
    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for problem in problems:
            print("  -", problem)
        return 1
    print("\nPASSED: all numeric claims match the artifacts and all editorial checks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
