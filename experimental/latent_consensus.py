"""Dirty-only categorical source consensus, isolated from the live cleaning loop.

This is a reviewable experiment, not an automatic repair policy. The scalar
source model is a constrained one-coin simplification of Dawid--Skene, not a
full confusion-matrix model. No fitted probability is a measured correctness
rate. See latent_consensus_protocol.md for assumptions and the frozen protocol.

Only ``DirtyTable`` and ``load_dirty`` are imported from the application. The
module never reads references, error masks, past judgments, or baseline output.
``python -m experimental.latent_consensus --output NEW_PATH`` writes a structural
profile, with exclusive creation. Fitting is deliberately an explicit Python
API requiring a semantic contract; it never changes the supplied table.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np

from zeroclean.data import DATASETS, DirtyTable, load_dirty

VERSION = "latent-consensus-v1"
Entity = tuple[str, ...]
Claims = dict[Entity, dict[str, str]]


@dataclass(frozen=True)
class Contract:
    entity_columns: tuple[str, ...]
    source_column: str
    value_column: str
    context_columns: tuple[str, ...] = ()
    single_valued: bool = False
    independent_sources: bool = False
    stationary_reliability: bool = False
    better_than_half: bool = False
    ignorable_missingness: bool = False
    evidence: str = ""
    hypothetical: bool = False

    def blockers(self) -> list[str]:
        flags = ("single_valued", "independent_sources", "stationary_reliability",
                 "better_than_half", "ignorable_missingness")
        reasons = [f"unestablished_contract:{f}" for f in flags if not getattr(self, f)]
        if not self.evidence.strip():
            reasons.append("missing_contract_evidence")
        return reasons


@dataclass(frozen=True)
class Parameters:
    seed: int = 1701
    source_assumption: str = "cluster_near_agreement"
    folds: int = 5
    min_train_entities: int = 30
    min_source_items: int = 20
    min_pair_overlap: int = 12
    min_conflicting_items: int = 10
    min_item_sources: int = 4
    dependency_agreement: float = .98
    unseen_prior: float = .10
    unseen_sensitivity: tuple[float, ...] = (.05, .10, .20)
    prior_alpha: float = 2.0
    prior_beta: float = 2.0
    reliability_floor: float = .5001
    reliability_ceiling: float = .995
    starts: tuple[float, ...] = (.55, .75, .95)
    bootstrap_replicates: int = 12
    max_iter: int = 200
    tolerance: float = 1e-7
    min_conditional_probability: float = .98
    min_leave_group_out_probability: float = .95
    min_margin: float = .20

    def __post_init__(self) -> None:
        if self.source_assumption not in {"cluster_near_agreement", "independent_names"}:
            raise ValueError("Unknown source assumption")
        if self.folds < 2 or self.min_train_entities < 3 or self.min_source_items < 1:
            raise ValueError("Invalid fold or training support parameters")
        if self.min_pair_overlap < 2 or self.min_conflicting_items < 1 or self.min_item_sources < 4:
            raise ValueError("Require overlap, conflicts, and four decision views")
        probabilities = (self.unseen_prior, *self.unseen_sensitivity,
                         self.min_conditional_probability,
                         self.min_leave_group_out_probability,
                         self.dependency_agreement, *self.starts)
        if any(not 0 < x < 1 for x in probabilities):
            raise ValueError("Probabilities must be in (0, 1)")
        if not .5 < self.reliability_floor < self.reliability_ceiling < 1:
            raise ValueError("Reliability constraints require a positive orientation")
        if self.prior_alpha <= 1 or self.prior_beta <= 1:
            raise ValueError("Use an interior Beta penalty")
        if self.bootstrap_replicates < 1 or self.max_iter < 1 or self.tolerance <= 0:
            raise ValueError("Invalid optimization or bootstrap parameters")
        if not 0 <= self.min_margin < 1:
            raise ValueError("Invalid margin")


def _provenance(table: DirtyTable, parameters: Parameters) -> dict[str, Any]:
    return {
        "experiment": VERSION,
        "dataset": table.name,
        "dirty_fingerprint": table.fingerprint,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": asdict(parameters),
        "access_scope": "supplied dirty table only; no evaluation inputs",
        "probability_meaning": "conditional model weights, not empirical correctness",
        "row_index_base": 0,
    }


def _indices(table: DirtyTable, contract: Contract) -> tuple[list[int], int, int]:
    names = (*contract.entity_columns, *contract.context_columns,
             contract.source_column, contract.value_column)
    if not contract.entity_columns or len(set(names)) != len(names):
        raise ValueError("Entity, context, source, and value roles must be distinct")
    missing = set(names) - set(table.columns)
    if missing:
        raise ValueError(f"Unknown columns: {sorted(missing)}")
    return ([table.columns.index(c) for c in (*contract.entity_columns, *contract.context_columns)],
            table.columns.index(contract.source_column), table.columns.index(contract.value_column))


def _extract(table: DirtyTable, contract: Contract) -> tuple[Claims, dict[str, Any]]:
    key_indices, source_index, value_index = _indices(table, contract)
    buckets: dict[tuple[Entity, str], list[tuple[int, str]]] = defaultdict(list)
    skipped = Counter()
    for row_index, row in enumerate(table.rows):
        key = tuple(row[i] for i in key_indices)
        source, value = row[source_index], row[value_index]
        if any(not v.strip() for v in key):
            skipped["missing_entity_or_context"] += 1
        elif not source.strip():
            skipped["missing_source"] += 1
        elif not value.strip():
            skipped["missing_value"] += 1
        else:
            buckets[key, source].append((row_index, value))
    claims: Claims = defaultdict(dict)
    conflicts, duplicate_extra = [], 0
    for (key, source), observations in sorted(buckets.items()):
        values = {value for _, value in observations}
        duplicate_extra += len(observations) - len(values)
        if len(values) != 1:
            conflicts.append({"entity": list(key), "source": source,
                              "rows": [r for r, _ in observations],
                              "values": sorted(values)})
        else:
            claims[key][source] = next(iter(values))
    return dict(claims), {
        "skipped_rows": dict(skipped),
        "duplicate_claim_extra_rows": duplicate_extra,
        "source_entity_conflicts": conflicts,
        "source_entity_pairs": len(buckets),
        "usable_source_entity_pairs": sum(map(len, claims.values())),
    }


def _groups(claims: Claims, parameters: Parameters) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Conservative near-agreement clusters; not a proof of copying."""
    by_source: dict[str, dict[Entity, str]] = defaultdict(dict)
    for key, observations in claims.items():
        for source, value in observations.items():
            by_source[source][key] = value
    parent = {source: source for source in by_source}

    def root(source: str) -> str:
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    evidence = []
    for a, b in combinations(sorted(by_source), 2):
        common = by_source[a].keys() & by_source[b].keys()
        if len(common) < parameters.min_pair_overlap:
            continue
        agreement = sum(by_source[a][k] == by_source[b][k] for k in common) / len(common)
        if agreement >= parameters.dependency_agreement:
            ra, rb = root(a), root(b)
            if parameters.source_assumption == "cluster_near_agreement":
                parent[max(ra, rb)] = min(ra, rb)
            evidence.append({"source_a": a, "source_b": b, "overlap": len(common),
                             "agreement": agreement})
    return {s: root(s) for s in sorted(parent)}, evidence


def _collapse(claims: Claims, mapping: dict[str, str]) -> tuple[Claims, int]:
    result: Claims = {}
    conflicts = 0
    for key, observations in claims.items():
        grouped: dict[str, set[str]] = defaultdict(set)
        for source, value in observations.items():
            if source in mapping:
                grouped[mapping[source]].add(value)
        conflicts += sum(len(v) != 1 for v in grouped.values())
        result[key] = {g: next(iter(v)) for g, v in grouped.items() if len(v) == 1}
    return result, conflicts


def profile_claims(table: DirtyTable, contract: Contract,
                   parameters: Parameters | None = None) -> dict[str, Any]:
    parameters = parameters or Parameters()
    claims, diagnostics = _extract(table, contract)
    mapping, evidence = _groups(claims, parameters)
    collapsed, conflicts = _collapse(claims, mapping)
    return {
        **_provenance(table, parameters), "contract": asdict(contract),
        "contract_blockers": contract.blockers(), **diagnostics,
        "entities": len(claims), "raw_sources": len(mapping),
        "source_groups": {g: [s for s in mapping if mapping[s] == g]
                          for g in sorted(set(mapping.values()))},
        "potential_dependency_pairs": evidence,
        "dependency_warning": "Agreement can reflect correct independent sources; grouping is a conservative stress test.",
        "group_entity_conflicts_excluded": conflicts,
        "raw_entity_coverage": dict(sorted(Counter(map(len, claims.values())).items())),
        "grouped_entity_coverage": dict(sorted(Counter(map(len, collapsed.values())).items())),
        "conflicting_entities_after_grouping": sum(len(set(o.values())) > 1 for o in collapsed.values()),
        "fit_performed": False,
    }


def profile_table(table: DirtyTable, parameters: Parameters | None = None) -> dict[str, Any]:
    """Schema-neutral repetition profile; source header hints are not contracts."""
    parameters = parameters or Parameters()
    columns = []
    source_hints = []
    for j, column in enumerate(table.columns):
        counts = Counter(row[j] for row in table.rows if row[j].strip())
        nonempty = sum(counts.values())
        columns.append({"column": column, "nonempty": nonempty,
                        "missing": len(table.rows) - nonempty,
                        "distinct_nonempty": len(counts),
                        "repeated_nonempty_rows": sum(n for n in counts.values() if n > 1),
                        "delimiter_rows": sum(any(d in row[j] for d in (",", ";", "|")) for row in table.rows)})
        if column.casefold() in {"src", "source", "source_id", "annotator", "publisher"}:
            source_hints.append(column)
    return {**_provenance(table, parameters), "shape": table.shape,
            "columns": columns, "exact_duplicate_rows": len(table.rows) - len(set(table.rows)),
            "source_header_hints": source_hints, "fit_performed": False,
            "status": "structural_only; entity identity, source independence, time scope and value cardinality require a contract"}


def _fold(key: Entity, entity_width: int, parameters: Parameters) -> int:
    # All contexts of the same base entity share a fold.
    payload = json.dumps([parameters.seed, key[:entity_width]], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % parameters.folds


def _training_subset(claims: Claims, parameters: Parameters,
                     entity_width: int) -> tuple[Claims, list[str], list[str]]:
    """Prune under-supported views, then require connected three-view overlap."""
    current = claims
    while True:
        counts = Counter(s for obs in current.values() if len(obs) >= 3 for s in obs)
        supported = {s for s, n in counts.items() if n >= parameters.min_source_items}
        trimmed = {k: {s: v for s, v in obs.items() if s in supported}
                   for k, obs in current.items()}
        trimmed = {k: obs for k, obs in trimmed.items() if len(obs) >= 3}
        if trimmed == current:
            break
        current = trimmed
    sources = sorted({s for obs in current.values() for s in obs})
    reasons = []
    if len({k[:entity_width] for k in current}) < parameters.min_train_entities:
        reasons.append("insufficient_training_entities")
    if len(sources) < 3:
        reasons.append("fewer_than_three_training_views")
    if sum(len(set(o.values())) > 1 for o in current.values()) < parameters.min_conflicting_items:
        reasons.append("insufficient_training_disagreement")
    if not sources:
        return current, sources, reasons
    pair_counts, triple_counts = Counter(), Counter()
    for observations in current.values():
        pair_counts.update(combinations(sorted(observations), 2))
        triple_counts.update(combinations(sorted(observations), 3))
    triangle_sources = {s for triple, n in triple_counts.items()
                        if n >= parameters.min_pair_overlap for s in triple}
    if set(sources) - triangle_sources:
        reasons.append("unsupported_three_view_overlap")
    adjacency: dict[str, set[str]] = defaultdict(set)
    for (a, b), n in pair_counts.items():
        if n >= parameters.min_pair_overlap:
            adjacency[a].add(b)
            adjacency[b].add(a)
    reached, pending = set(), [sources[0]]
    while pending:
        s = pending.pop()
        if s not in reached:
            reached.add(s)
            pending.extend(adjacency[s] - reached)
    if reached != set(sources):
        reasons.append("disconnected_source_overlap")
    return current, sources, reasons


def _posterior(observations: dict[str, str], reliability: dict[str, float],
               unseen_prior: float, candidates: tuple[str, ...] | None = None) -> tuple[tuple[str, ...], np.ndarray]:
    candidates = candidates if candidates is not None else tuple(sorted(set(observations.values())))
    k = len(candidates)
    if not k:
        return candidates, np.array([1.0])
    scores = np.log(np.array([(1 - unseen_prior) / k] * k + [unseen_prior]))
    for source, value in observations.items():
        q = reliability[source]
        likelihood = np.full(k + 1, (1 - q) / k)
        likelihood[candidates.index(value)] = q
        scores += np.log(likelihood)
    scores -= scores.max()
    weights = np.exp(scores)
    return candidates, weights / weights.sum()


def _fit(claims: Claims, sources: list[str], parameters: Parameters, unseen: float,
         start: float, multiplicity: dict[Entity, int] | None = None) -> dict[str, Any]:
    q = {s: start for s in sources}
    converged = False
    for iteration in range(parameters.max_iter):
        successes = {s: parameters.prior_alpha - 1 for s in sources}
        totals = {s: parameters.prior_alpha + parameters.prior_beta - 2 for s in sources}
        for key, observations in claims.items():
            count = 1 if multiplicity is None else multiplicity.get(key, 0)
            if not count:
                continue
            values, posterior = _posterior(observations, q, unseen)
            for source, value in observations.items():
                successes[source] += count * posterior[values.index(value)]
                totals[source] += count
        updated = {s: float(np.clip(successes[s] / totals[s], parameters.reliability_floor,
                                    parameters.reliability_ceiling)) for s in sources}
        delta = max(abs(updated[s] - q[s]) for s in sources)
        q = updated
        if delta <= parameters.tolerance:
            converged = True
            break
    return {"reliability": q, "unseen_prior": unseen, "iterations": iteration + 1,
            "converged": converged,
            "orientation_boundary_sources": [s for s in sources if q[s] <= parameters.reliability_floor + 1e-5],
            "ceiling_sources": [s for s in sources if q[s] >= parameters.reliability_ceiling - 1e-5]}


def _ensemble(claims: Claims, sources: list[str], parameters: Parameters, fold: int,
              entity_width: int) -> list[dict[str, Any]]:
    models = [_fit(claims, sources, parameters, parameters.unseen_prior, start)
              for start in parameters.starts]
    models.extend(_fit(claims, sources, parameters, unseen, .75)
                  for unseen in parameters.unseen_sensitivity if unseen != parameters.unseen_prior)
    rng = np.random.default_rng(parameters.seed + fold)
    # Resample base entities, keeping all contexts together.
    clusters: dict[Entity, list[Entity]] = defaultdict(list)
    for key in claims:
        clusters[key[:entity_width]].append(key)
    entities = sorted(clusters)
    for _ in range(parameters.bootstrap_replicates):
        sampled = Counter(int(x) for x in rng.integers(0, len(entities), size=len(entities)))
        weights = {k: count for index, count in sampled.items() for k in clusters[entities[index]]}
        models.append(_fit(claims, sources, parameters, parameters.unseen_prior, .75, weights))
    return models


def _decision(observations: dict[str, str], models: list[dict[str, Any]],
              parameters: Parameters) -> dict[str, Any]:
    reasons = []
    if len(observations) < parameters.min_item_sources:
        return {"status": "abstain", "reasons": ["insufficient_independent_group_support"]}
    candidates = tuple(sorted(set(observations.values())))
    posteriors = [_posterior(observations, m["reliability"], m["unseen_prior"], candidates)[1] for m in models]
    winners = [int(p.argmax()) for p in posteriors]
    if len(set(winners)) != 1:
        reasons.append("winner_unstable_across_sensitivity_models")
    winner = winners[0]
    if winner == len(candidates):
        reasons.append("unobserved_truth_state_wins")
    lower = min(float(p[winner]) for p in posteriors)
    margin = min(float(p[winner] - max(np.delete(p, winner))) for p in posteriors)
    if lower < parameters.min_conditional_probability:
        reasons.append("conditional_weight_below_threshold")
    if margin < parameters.min_margin:
        reasons.append("ambiguous_margin")
    leave_out_lower = 1.0
    for source in observations:
        remaining = {s: v for s, v in observations.items() if s != source}
        for model in models:
            _, posterior = _posterior(remaining, model["reliability"], model["unseen_prior"], candidates)
            leave_out_lower = min(leave_out_lower, float(posterior[winner]))
            if int(posterior.argmax()) != winner:
                reasons.append("winner_changes_without_one_group")
    if leave_out_lower < parameters.min_leave_group_out_probability:
        reasons.append("leave_group_out_weight_below_threshold")
    return {"status": "abstain" if reasons else "supported_candidate",
            "reasons": sorted(set(reasons)), "candidate": candidates[winner] if winner < len(candidates) else None,
            "candidate_strings": list(candidates), "conditional_weight_min": lower,
            "conditional_margin_min": margin, "leave_group_out_weight_min": leave_out_lower,
            "independent_groups_assumed": len(observations),
            "unobserved_state_weight_max": max(float(p[-1]) for p in posteriors)}


def run_experiment(table: DirtyTable, contract: Contract,
                   parameters: Parameters | None = None) -> dict[str, Any]:
    """Return cross-fitted preview records; never apply or export a repaired table."""
    parameters = parameters or Parameters()
    claims, extraction = _extract(table, contract)
    key_indices, source_index, value_index = _indices(table, contract)
    result = {**_provenance(table, parameters), "contract": asdict(contract),
              "contract_blockers": contract.blockers(), "extraction": extraction,
              "folds": [], "decisions": [], "fit_performed": False}
    entity_width = len(contract.entity_columns)
    entity_decisions = {}
    for fold in range(parameters.folds):
        heldout = {k: o for k, o in claims.items() if _fold(k, entity_width, parameters) == fold}
        if not heldout:
            continue
        training = {k: o for k, o in claims.items() if _fold(k, entity_width, parameters) != fold}
        mapping, dependency = _groups(training, parameters)
        grouped_training, group_conflicts = _collapse(training, mapping)
        training, sources, blockers = _training_subset(grouped_training, parameters, entity_width)
        blockers = contract.blockers() + blockers
        info: dict[str, Any] = {"fold": fold, "training_entities": len(training),
                                "heldout_entities": len(heldout), "training_groups": sources,
                                "potential_dependency_pairs": dependency, "source_group_mapping": mapping,
                                "group_entity_conflicts_excluded": group_conflicts, "blockers": blockers}
        models = []
        if not blockers:
            models = _ensemble(training, sources, parameters, fold, entity_width)
            result["fit_performed"] = True
            info["models"] = models
            if any(not m["converged"] for m in models):
                blockers.append("optimization_did_not_converge")
            if any(len(m["orientation_boundary_sources"]) * 2 >= len(sources) for m in models):
                blockers.append("orientation_boundary_degeneracy")
            if any(len(m["ceiling_sources"]) * 2 >= len(sources) for m in models):
                blockers.append("reliability_ceiling_degeneracy")
            info["reliability_sensitivity_ranges"] = {
                s: [min(m["reliability"][s] for m in models), max(m["reliability"][s] for m in models)]
                for s in sources}
        result["folds"].append(info)
        grouped_heldout, heldout_conflicts = _collapse(heldout, mapping)
        info["heldout_group_conflicts_excluded"] = heldout_conflicts
        for key in heldout:
            observations = {s: v for s, v in grouped_heldout[key].items() if s in sources}
            decision = ({"status": "abstain", "reasons": list(blockers)} if blockers
                        else _decision(observations, models, parameters))
            decision.update({"fold": fold, "trained_sources": sorted(
                                s for s in heldout[key] if s in mapping and mapping[s] in sources),
                             "active_source_groups": sorted(observations)})
            entity_decisions[key] = decision, mapping
    conflicting_rows = {r for conflict in extraction["source_entity_conflicts"] for r in conflict["rows"]}
    for row_index, row in enumerate(table.rows):
        key = tuple(row[i] for i in key_indices)
        value, source = row[value_index], row[source_index]
        decision, mapping = entity_decisions.get(key, ({"status": "abstain", "reasons": ["no_usable_entity_claims"]}, {}))
        decision = dict(decision)
        if not value.strip():
            decision.update(status="abstain", reasons=["missing_value_not_imputed"])
        elif row_index in conflicting_rows:
            decision.update(status="abstain", reasons=["source_entity_has_multiple_values"])
        elif source not in mapping or mapping[source] not in decision.get("active_source_groups", []):
            decision.update(status="abstain", reasons=["source_or_group_not_supported"] + decision["reasons"])
        elif decision["status"] == "supported_candidate":
            decision["status"] = "consistent" if value == decision["candidate"] else "propose"
        result["decisions"].append({**decision, "row": row_index, "column": contract.value_column,
                                    "entity": list(key), "source": source, "original": value})
    result["status_counts"] = dict(Counter(d["status"] for d in result["decisions"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output", type=Path, required=True, help="New JSON path; never overwrite")
    args = parser.parse_args()
    profiles = [profile_table(load_dirty(name)) for name in args.datasets]
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"experiment": VERSION, "profiles": profiles}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
