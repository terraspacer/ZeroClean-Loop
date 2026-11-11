"""Unpromoted dirty-only synthetic-corruption detector; never benchmark accuracy.

Run from the repository root with:
  python -m experimental.selfsupervised_corruption --run
  python -m unittest experimental.test_selfsupervised_corruption

Only ``load_dirty`` reads real input tables. The adjacent predeclared protocol
defines the fixed experiment. Synthetic labels mean unchanged versus changed,
not correct versus incorrect. No automatic repair or live promotion is exposed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any
import unicodedata

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from zeroclean.data import DATASETS, DirtyTable, load_dirty


SEED = 20261026
OUTPUT = Path(__file__).with_name("selfsupervised_preview")
PROTOCOL_PATH = OUTPUT / "protocol.json"
STAGES = ("train", "calibration", "test")
OPERATORS = (
    "character_insertion", "character_deletion", "adjacent_transposition",
    "full_reversal", "observed_same_column_substitution", "cross_field_displacement",
)
FEATURE_NAMES = (
    "log1p_reference_value_count", "log1p_reference_pattern_count", "reference_pattern_share",
    "character_bigram_surprisal", "robust_length_deviation_capped_20",
    "numeric_parse_indicator", "reference_numeric_share", "robust_numeric_deviation_capped_20",
    "numeric_type_contradiction", "unknown_value_indicator", "digit_fraction", "alpha_fraction",
    "punctuation_fraction", "whitespace_fraction", "log1p_reliable_context_count",
    "reliable_context_contradiction_fraction", "reliable_context_agreement_fraction",
    "max_context_purity", "log1p_max_context_peers",
)
SCORE_MEANING = (
    "Injected-perturbation versus unchanged-dirty discrimination score; "
    "not a probability of a true data error; experimental and unpromoted."
)
LIMITS = (
    "Unchanged dirty baselines are noisy, not verified clean labels.",
    "Changed strings may remain valid or accidentally repair preexisting dirt.",
    "No benchmark accuracy, natural-error calibration, or exact repair correctness is known.",
    "Generic repeated high-cardinality values are imperfect entity proxies.",
    "Reference dirty support can be shared across held-out query-value partitions; "
    "this is a transductive dirty-support experiment, not unseen-table generalization.",
    "Reference dirt, systematic errors, sparse contexts, and semantic ambiguity can defeat this detector.",
)


def digest(*parts: Any) -> str:
    value = json.dumps((SEED, *parts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def bucket(namespace: str, *parts: Any) -> float:
    return int(digest(namespace, *parts)[:16], 16) / 2**64


def value_stage(column: int, value: str) -> str:
    value_hash = bucket("query-value", column, value)
    return "train" if value_hash < .5 else "calibration" if value_hash < .75 else "test"


def pattern(value: str) -> str:
    """Compress Unicode class runs while retaining literal separators."""
    tokens = []
    for char in value:
        token = "D" if char.isdecimal() else "U" if char.isupper() else "L" if char.islower() else "A" if char.isalpha() else char
        if not tokens or token != tokens[-1] or token not in {"D", "U", "L", "A"}:
            tokens.append(token)
    return "".join(tokens)


def finite_number(value: str) -> float | None:
    try:
        number = Decimal(value.strip())
        result = float(number)
        return result if number.is_finite() and math.isfinite(result) else None
    except (InvalidOperation, ValueError, OverflowError):
        return None


def median_mad(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(v - median) for v in values))
    return median, max(mad * 1.4826, 1.0)


@dataclass
class Partition:
    rows: dict[str, list[int]]
    row_stage: list[str]
    component: list[int]
    proxy_columns: list[int]
    component_sizes: list[int]


def partition_rows(table: DirtyTable) -> Partition:
    """Keep duplicate rows and observed entity proxies together, then hash groups."""
    n, m = table.shape
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    duplicate = {}
    for i, row in enumerate(table.rows):
        if row in duplicate:
            union(i, duplicate[row])
        else:
            duplicate[row] = i
    proxy_columns = []
    for j in range(m):
        counts = Counter(row[j] for row in table.rows if row[j])
        nonempty = sum(counts.values())
        if nonempty >= .8 * n and nonempty and len(counts) / nonempty >= .8:
            proxy_columns.append(j)
            first = {}
            for i, row in enumerate(table.rows):
                if not row[j]:
                    continue
                if row[j] in first:
                    union(i, first[row[j]])
                else:
                    first[row[j]] = i
    components = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)
    rows = {name: [] for name in ("reference", *STAGES)}
    stages, component_ids = [""] * n, [0] * n
    for component_id, indices in sorted(components.items()):
        # Sorted exact row hashes make the assignment insensitive to row order.
        key = sorted(digest("row", table.rows[i]) for i in indices)
        fraction = bucket("row-component", key)
        stage = "reference" if fraction < .4 else "train" if fraction < .7 else "calibration" if fraction < .85 else "test"
        rows[stage].extend(indices)
        for i in indices:
            stages[i], component_ids[i] = stage, component_id
    return Partition(rows, stages, component_ids, proxy_columns, sorted((len(v) for v in components.values()), reverse=True))


class ReferenceIndex:
    """Profiles use independent reference row components, never query labels."""
    def __init__(self, table: DirtyTable, reference_rows: list[int]):
        self.table, self.reference_rows = table, tuple(reference_rows)
        self.n, self.m = len(reference_rows), len(table.columns)
        self.profiles = []
        self.contexts: dict[tuple[int, int], dict[str, tuple[str, int, float]]] = {}
        self.contexts_by_target = defaultdict(list)
        for j in range(self.m):
            values = [table.rows[i][j] for i in reference_rows]
            counts = Counter(values)
            patterns = Counter(pattern(v) for v in values)
            bigrams = Counter(pair for v in values for pair in zip("^" + v, v + "$"))
            numbers = [number for v in values if (number := finite_number(v)) is not None]
            self.profiles.append({
                "counts": counts, "patterns": patterns, "bigrams": bigrams,
                "bigram_total": sum(bigrams.values()),
                "length": median_mad([len(v) for v in values]), "numeric": median_mad(numbers),
                "numeric_share": len(numbers) / max(len(values), 1),
                "alphabet": tuple(sorted(set("".join(values))))[:128],
            })
            selected = sorted((k for k in range(self.m) if k != j), key=lambda k: digest("context-column", k))[:32]
            for k in selected:
                groups = defaultdict(Counter)
                for i in reference_rows:
                    row = table.rows[i]
                    if row[k] and row[j]:
                        groups[row[k]][row[j]] += 1
                reliable = {}
                for key, counter in groups.items():
                    total = sum(counter.values())
                    mode, count = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0]
                    purity = count / total
                    if total >= 5 and purity >= .95:
                        reliable[key] = (mode, total, purity)
                self.contexts[j, k] = reliable
                self.contexts_by_target[j].append((k, reliable))

    def features(self, row: tuple[str, ...], j: int, value: str | None = None) -> tuple[list[float], dict]:
        value = row[j] if value is None else value
        profile = self.profiles[j]
        value_count = profile["counts"].get(value, 0)
        pattern_count = profile["patterns"].get(pattern(value), 0)
        bigram_count = profile["bigrams"]
        denominator = profile["bigram_total"] + max(len(bigram_count), 1)
        pairs = list(zip("^" + value, value + "$"))
        surprisal = sum(-math.log((bigram_count.get(pair, 0) + 1) / max(denominator, 1)) for pair in pairs) / max(len(pairs), 1)
        length_median, length_scale = profile["length"]
        number = finite_number(value)
        numeric_median, numeric_scale = profile["numeric"]
        numeric_deviation = min(20, abs(number - numeric_median) / numeric_scale) if number is not None else 0
        numeric_contradiction = bool(value and number is None and profile["numeric_share"] >= .98 and self.n >= 100)
        evidence = []
        for context, groups in self.contexts_by_target[j]:
            if row[context] in groups:
                mode, peers, purity = groups[row[context]]
                evidence.append((mode, peers, purity, context))
        modes = {entry[0] for entry in evidence}
        unambiguous = len(modes) == 1
        violations = sum(value != mode for mode, *_ in evidence) if unambiguous else 0
        agreements = sum(value == mode for mode, *_ in evidence) if unambiguous else 0
        reliable_contexts = len(evidence) if unambiguous else 0
        context_contradiction = bool(unambiguous and evidence and violations == len(evidence))
        length = max(len(value), 1)
        vector = [
            math.log1p(value_count), math.log1p(pattern_count), pattern_count / max(self.n, 1),
            surprisal, min(20, abs(len(value) - length_median) / length_scale),
            float(number is not None), profile["numeric_share"], numeric_deviation,
            float(numeric_contradiction), float(value_count == 0),
            sum(c.isdecimal() for c in value) / length, sum(c.isalpha() for c in value) / length,
            sum(unicodedata.category(c).startswith(("P", "S")) for c in value) / length,
            sum(c.isspace() for c in value) / length,
            math.log1p(reliable_contexts), violations / max(reliable_contexts, 1),
            agreements / max(reliable_contexts, 1),
            max((entry[2] for entry in evidence), default=0) if unambiguous else 0,
            math.log1p(max((entry[1] for entry in evidence), default=0)) if unambiguous else 0,
        ]
        support = {
            "supported": context_contradiction or numeric_contradiction,
            "context_contradiction": context_contradiction,
            "numeric_type_contradiction": numeric_contradiction,
            "context_conflict": bool(evidence and not unambiguous),
            "reliable_contexts": reliable_contexts,
            "reference_rows": self.n,
        }
        return vector, support


@dataclass(frozen=True)
class Mutation:
    row: int
    column: int
    stage: str
    operator: str
    original_hash: str
    changed_hash: str
    donor_row: int | None
    donor_column: int | None
    value: str

    def public(self) -> dict:
        result = asdict(self)
        # Exact mutated strings are unnecessary for score interpretation or leakage auditing.
        del result["value"]
        result["label"] = 1
        result["label_meaning"] = "deliberately changed; semantic invalidity unknown"
        return result


def mutate(
    table: DirtyTable, partition: Partition, index: ReferenceIndex,
    i: int, j: int, operator: str, rng: random.Random,
) -> tuple[Mutation | None, str]:
    original, stage = table.rows[i][j], partition.row_stage[i]
    if stage not in STAGES:
        return None, "reference_source"
    if not original:
        return None, "empty_source"
    if len(original) > 512:
        return None, "source_over_512"
    donor_row = donor_column = None
    if operator == "character_insertion":
        alphabet = index.profiles[j]["alphabet"]
        if not alphabet:
            return None, "no_alphabet"
        position = rng.randrange(len(original) + 1)
        changed = original[:position] + rng.choice(alphabet) + original[position:]
    elif operator == "character_deletion":
        position = rng.randrange(len(original))
        changed = original[:position] + original[position + 1:]
    elif operator == "adjacent_transposition":
        if len(original) < 2:
            return None, "too_short_transposition"
        position = rng.randrange(len(original) - 1)
        changed = original[:position] + original[position + 1] + original[position] + original[position + 2:]
    elif operator == "full_reversal":
        changed = original[::-1]
    elif operator in ("observed_same_column_substitution", "cross_field_displacement"):
        donor_row = rng.choice(partition.rows[stage])
        if operator == "cross_field_displacement":
            alternatives = [k for k in range(len(table.columns)) if k != j]
            if not alternatives:
                return None, "no_other_column"
            donor_column = rng.choice(alternatives)
        else:
            donor_column = j
        changed = table.rows[donor_row][donor_column]
    else:
        raise ValueError(f"Unknown operator: {operator}")
    if changed == original:
        return None, "unchanged"
    if not changed:
        return None, "empty_output"
    if len(changed) > 512:
        return None, "output_over_512"
    if value_stage(j, original) != stage or value_stage(j, changed) != stage:
        return None, "query_value_gate"
    return Mutation(i, j, stage, operator, digest("value", original), digest("value", changed), donor_row, donor_column, changed), "retained"


@dataclass
class SyntheticSplit:
    x: list[list[float]]
    y: list[int]
    operators: list[str]
    source_cells: list[tuple[int, int]]
    mutations: list[Mutation]
    query_values: set[tuple[int, str]]
    rejection_counts: Counter


def make_synthetic(table: DirtyTable, partition: Partition, index: ReferenceIndex, stage: str) -> SyntheticSplit:
    rng = random.Random(int(digest("sampling", table.name, stage)[:16], 16))
    selected, eligible_count = [], 0
    # Reservoir sample bounds fitting cost without selecting by apparent dirtiness.
    for i in partition.rows[stage]:
        for j, value in enumerate(table.rows[i]):
            if not value or len(value) > 512 or value_stage(j, value) != stage:
                continue
            eligible_count += 1
            if len(selected) < 10000:
                selected.append((i, j))
            else:
                slot = rng.randrange(eligible_count)
                if slot < 10000:
                    selected[slot] = (i, j)
    selected.sort()
    result = SyntheticSplit([], [], [], [], [], set(), Counter())
    for i, j in selected:
        original = table.rows[i][j]
        feature, _ = index.features(table.rows[i], j)
        result.x.append(feature)
        result.y.append(0)
        result.operators.append("unchanged_dirty_baseline")
        result.source_cells.append((i, j))
        result.query_values.add((j, original))
        for _ in range(24):
            operator = rng.choice(OPERATORS)
            mutation, reason = mutate(table, partition, index, i, j, operator, rng)
            result.rejection_counts[f"{operator}:{reason}"] += 1
            if mutation is not None:
                feature, _ = index.features(table.rows[i], j, mutation.value)
                result.x.append(feature)
                result.y.append(1)
                result.operators.append(operator)
                result.source_cells.append((i, j))
                result.mutations.append(mutation)
                result.query_values.add((j, mutation.value))
                break
        else:
            result.rejection_counts["all_24_attempts_rejected"] += 1
    return result


def assert_separation(partition: Partition, splits: dict[str, SyntheticSplit]) -> dict:
    for a, first in enumerate(STAGES):
        first_components = {partition.component[i] for i in partition.rows[first]}
        reference_components = {partition.component[i] for i in partition.rows["reference"]}
        assert first_components.isdisjoint(reference_components), "reference entity-proxy leakage"
        for second in STAGES[a + 1:]:
            assert first_components.isdisjoint(partition.component[i] for i in partition.rows[second]), "row/entity-proxy leakage"
            assert splits[first].query_values.isdisjoint(splits[second].query_values), "query-value leakage"
    return {
        "complete_row_component_overlap": 0, "supervised_query_value_group_overlap": 0,
        "reference_component_overlap_with_supervised": 0,
        "entity_proxy_only": True, "reference_raw_value_support_may_be_shared": True,
    }


def synthetic_diagnostics(labels: list[int], scores: np.ndarray, operators: list[str], threshold: float) -> dict:
    labels_array = np.asarray(labels)
    bins = []
    for k in range(5):
        lower, upper = k / 5, (k + 1) / 5
        selected = (scores >= lower) & ((scores <= upper) if k == 4 else (scores < upper))
        count = int(selected.sum())
        bins.append({
            "lower": lower, "upper": upper, "count": count,
            "mean_score": float(scores[selected].mean()) if count else None,
            "injected_label_fraction": float(labels_array[selected].mean()) if count else None,
        })
    by_operator = {}
    for operator in sorted(set(operators)):
        selected = np.asarray([candidate == operator for candidate in operators])
        by_operator[operator] = {
            "count": int(selected.sum()), "mean_synthetic_score": float(scores[selected].mean()),
            "threshold_exceedance_fraction": float((scores[selected] >= threshold).mean()),
        }
    return {
        "labels": "unchanged dirty=0; injected perturbation=1; semantic correctness unobserved",
        "brier_against_perturbation_label": float(np.mean((scores - labels_array) ** 2)),
        "reliability_bins_synthetic_labels_only": bins,
        "by_operator": by_operator,
        "not_benchmark_accuracy": True,
    }


def fit_synthetic(splits: dict[str, SyntheticSplit]) -> tuple[Any, float | None, dict]:
    counts = {stage: Counter(split.y) for stage, split in splits.items()}
    report = {"label_counts": {stage: {str(label): count for label, count in counter.items()} for stage, counter in counts.items()}}
    if any(counts["train"][label] < 200 for label in (0, 1)) or any(counts["calibration"][label] < 100 for label in (0, 1)):
        report["status"] = "abstain_insufficient_synthetic_training_or_calibration"
        return None, None, report
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=500, random_state=SEED))
    model.fit(np.asarray(splits["train"].x), np.asarray(splits["train"].y))
    calibration_scores = model.predict_proba(np.asarray(splits["calibration"].x))[:, 1]
    unchanged = calibration_scores[np.asarray(splits["calibration"].y) == 0]
    threshold = max(.9, float(np.quantile(unchanged, .995, method="higher")))
    report.update({
        "status": "fit_unpromoted_synthetic_model", "threshold": threshold,
        "threshold_policy": "max(.90, upper empirical .995 quantile of calibration unchanged-dirty scores)",
        "score_meaning": SCORE_MEANING,
        "standardized_feature_coefficients": dict(zip(FEATURE_NAMES, model[-1].coef_[0].tolist())),
        "intercept": float(model[-1].intercept_[0]),
        "calibration_synthetic_only": synthetic_diagnostics(splits["calibration"].y, calibration_scores, splits["calibration"].operators, threshold),
    })
    if all(counts["test"][label] >= 100 for label in (0, 1)):
        test_scores = model.predict_proba(np.asarray(splits["test"].x))[:, 1]
        report["test_synthetic_only"] = synthetic_diagnostics(splits["test"].y, test_scores, splits["test"].operators, threshold)
    else:
        report["test_synthetic_only"] = {"status": "insufficient_heldout_synthetic_labels"}
    return model, threshold, report


def preview_decision(stage: str, value: str, support: dict, score: float | None, threshold: float | None, model_available: bool) -> tuple[str, str]:
    if stage == "reference":
        return "abstain", "reference_profile_only"
    if not value:
        return "abstain", "empty_source_unsupported"
    if len(value) > 512:
        return "abstain", "source_over_512"
    if support["reference_rows"] < 100:
        return "abstain", "insufficient_reference_rows"
    if not model_available:
        return "abstain", "insufficient_synthetic_training_or_calibration"
    if not support["supported"]:
        return "abstain", "conflicting_contexts" if support["context_conflict"] else "no_supported_contradiction"
    if score is None or threshold is None or score < threshold:
        return "abstain", "below_fixed_synthetic_threshold"
    return "candidate_unpromoted", "context_contradiction" if support["context_contradiction"] else "numeric_type_contradiction"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_table(table: DirtyTable, output: Path) -> dict:
    partition = partition_rows(table)
    index = ReferenceIndex(table, partition.rows["reference"])
    splits = {stage: make_synthetic(table, partition, index, stage) for stage in STAGES}
    separation = assert_separation(partition, splits)
    model, threshold, fit_report = fit_synthetic(splits)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{table.name}.synthetic_provenance.jsonl").open("w", encoding="utf-8") as stream:
        for stage in STAGES:
            split = splits[stage]
            for mutation in split.mutations:
                stream.write(json.dumps(mutation.public(), sort_keys=True) + "\n")
            for (i, j), label in zip(split.source_cells, split.y):
                if label == 0:
                    stream.write(json.dumps({
                        "stage": stage, "row": i, "column": j, "label": 0,
                        "original_hash": digest("value", table.rows[i][j]),
                        "label_meaning": "unchanged raw dirty observation; NOT a verified clean label",
                    }, sort_keys=True) + "\n")
    decisions, structural = Counter(), Counter()
    fields = ["row", "column", "column_name", "row_partition", "value_sha256", "synthetic_score", "threshold", "decision", "reason", "context_contradiction", "numeric_type_contradiction", "reliable_contexts", "reference_rows", "status"]
    with (output / f"{table.name}.scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(table.rows):
            feature_rows, supports = zip(*(index.features(row, j) for j in range(len(row))))
            scores = model.predict_proba(np.asarray(feature_rows))[:, 1] if model is not None and partition.row_stage[i] != "reference" else [None] * len(row)
            for j, value in enumerate(row):
                support, score = supports[j], scores[j]
                stage = partition.row_stage[i]
                decision, reason = preview_decision(stage, value, support, score, threshold, model is not None)
                decisions[reason] += 1
                if stage != "reference" and value and len(value) <= 512 and index.n >= 100:
                    structural["basic_eligible_cells"] += 1
                    if support["supported"]:
                        structural["supported_contradiction_cells"] += 1
                if decision == "candidate_unpromoted":
                    structural["candidate_cells_unpromoted"] += 1
                writer.writerow({
                    "row": i, "column": j, "column_name": table.columns[j], "row_partition": stage,
                    "value_sha256": digest("value", value),
                    "synthetic_score": "" if score is None else float(score),
                    "threshold": "" if threshold is None else threshold,
                    "decision": decision, "reason": reason,
                    "context_contradiction": support["context_contradiction"],
                    "numeric_type_contradiction": support["numeric_type_contradiction"],
                    "reliable_contexts": support["reliable_contexts"], "reference_rows": index.n,
                    "status": "experimental_unpromoted_not_error_probability",
                })
    n, m = table.shape
    result = {
        "dataset": table.name, "status": "experimental_unpromoted", "input_sha256": table.fingerprint,
        "shape": {"rows": n, "columns": m, "cells": n * m}, "score_meaning": SCORE_MEANING,
        "partitions": {stage: len(rows) for stage, rows in partition.rows.items()},
        "entity_proxy_columns": [table.columns[j] for j in partition.proxy_columns],
        "row_component_count": len(partition.component_sizes), "largest_component_rows": max(partition.component_sizes, default=0),
        "separation": separation, "structural_coverage": dict(structural), "decision_reasons": dict(decisions),
        "synthetic_mutation_accounting": {stage: dict(split.rejection_counts) for stage, split in splits.items()},
        "synthetic_supervised_value_groups": {stage: len(split.query_values) for stage, split in splits.items()},
        "model": fit_report, "limitations": LIMITS,
    }
    write_json(output / f"{table.name}.report.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run the predeclared protocol on all six raw dirty tables")
    args = parser.parse_args(argv)
    if not args.run:
        parser.print_help()
        return 0
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["protocol_id"] != "dirty-only-synthetic-corruption-v1" or protocol["seed"] != SEED or protocol["features"] != list(FEATURE_NAMES):
        raise ValueError("Code/protocol mismatch; do not run a silently changed experiment")
    metadata = {
        "status": "experimental_unpromoted", "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": SEED, "score_meaning": SCORE_MEANING, "limitations": LIMITS,
        "libraries": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "input_access_policy": "Only zeroclean.data.load_dirty; no reference/evaluation/prediction artifacts read",
    }
    # A manifest is committed to disk before any real dirty table is loaded.
    write_json(OUTPUT / "run_manifest.json", metadata)
    summaries = []
    for name in DATASETS:
        table = load_dirty(name)
        result = run_table(table, OUTPUT)
        summaries.append({
            "dataset": name, "input_sha256": table.fingerprint, "shape": result["shape"],
            "partitions": result["partitions"], "structural_coverage": result["structural_coverage"],
            "decision_reasons": result["decision_reasons"], "model_status": result["model"]["status"],
        })
        print(json.dumps(summaries[-1], sort_keys=True), flush=True)
    metadata["datasets"] = summaries
    write_json(OUTPUT / "summary.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
