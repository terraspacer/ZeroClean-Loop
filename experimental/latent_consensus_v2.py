"""Preregistered log-advantage sensitivity; v1 remains unchanged.

q(s, item) = exp(theta_s) / (exp(theta_s) + K_item), for K observed
candidate strings plus one additional unobserved category. theta may be
negative. A Gaussian penalty supplies a disclosed, soft orientation preference,
not identifiability. All prior centers/starts enter candidate sensitivity gates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from zeroclean.data import DirtyTable
from experimental import latent_consensus as v1

VERSION = "latent-consensus-log-advantage-v2"


@dataclass(frozen=True)
class SkillContract:
    entity_columns: tuple[str, ...]
    source_column: str
    value_column: str
    context_columns: tuple[str, ...] = ()
    single_valued: bool = False
    independent_sources: bool = False
    stationary_reliability: bool = False
    ignorable_missingness: bool = False
    orientation_prior_disclosed: bool = False
    evidence: str = ""
    hypothetical: bool = True

    def blockers(self) -> list[str]:
        flags = ("single_valued", "independent_sources", "stationary_reliability",
                 "ignorable_missingness", "orientation_prior_disclosed")
        reasons = [f"unestablished_contract:{f}" for f in flags if not getattr(self, f)]
        if not self.evidence.strip():
            reasons.append("missing_contract_evidence")
        return reasons


@dataclass(frozen=True)
class SkillParameters:
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
    prior_center: float = float(np.log(2))
    prior_center_sensitivity: tuple[float, ...] = (-float(np.log(2)), 0.0, float(np.log(2)))
    prior_sd: float = 2.0
    skill_bound: float = 6.0
    initial_skills: tuple[float, ...] = (-2.0, 0.0, 2.0)
    bootstrap_replicates: int = 12
    max_iter: int = 200
    tolerance: float = 1e-7
    min_conditional_probability: float = .98
    min_leave_group_out_probability: float = .95
    min_margin: float = .20

    def __post_init__(self) -> None:
        common = {f.name for f in fields(v1.Parameters)} & set(asdict(self))
        v1.Parameters(**{name: getattr(self, name) for name in common})
        if self.prior_sd <= 0 or self.skill_bound <= 0:
            raise ValueError("Gaussian scale and numerical skill bound must be positive")
        if not self.initial_skills or any(abs(x) > self.skill_bound for x in self.initial_skills):
            raise ValueError("Invalid skill initializations")
        if self.prior_center not in self.prior_center_sensitivity:
            raise ValueError("Sensitivity centers must include the baseline prior center")


def _provenance(table: DirtyTable, parameters: SkillParameters) -> dict[str, Any]:
    return {"experiment": VERSION, "dataset": table.name, "dirty_fingerprint": table.fingerprint,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "v1_helpers_sha256": hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest(),
            "parameters": asdict(parameters), "row_index_base": 0,
            "probability_meaning": "conditional model weights, not empirical correctness",
            "orientation": "Soft Gaussian penalty; no source or global sign constraint; nonidentifiability remains possible",
            "access_scope": "supplied dirty table only; no evaluation inputs"}


def match_probability(skill: float, other_categories: int) -> float:
    if other_categories < 1:
        raise ValueError("At least one competing category is required")
    return float(1 / (1 + np.exp(np.log(other_categories) - skill)))


def _posterior(observations: dict[str, str], skills: dict[str, float], unseen_prior: float,
               candidates: tuple[str, ...] | None = None) -> tuple[tuple[str, ...], np.ndarray]:
    candidates = candidates if candidates is not None else tuple(sorted(set(observations.values())))
    if not candidates:
        return (), np.array([1.0])
    scores = np.log(np.array([(1 - unseen_prior) / len(candidates)] * len(candidates) + [unseen_prior]))
    for source, value in observations.items():
        scores[candidates.index(value)] += skills[source]
    scores -= scores.max()
    weights = np.exp(scores)
    return candidates, weights / weights.sum()


def _solve_skill(expected_matches: float, cardinalities: Counter, center: float,
                 parameters: SkillParameters) -> float:
    """Unique maximizer of the concave, Gaussian-penalized M-step."""
    ks = np.array(list(cardinalities), dtype=float)
    counts = np.array(list(cardinalities.values()), dtype=float)
    log_ks = np.log(ks)
    precision = 1 / parameters.prior_sd ** 2

    def derivative(skill: float) -> float:
        q = 1 / (1 + np.exp(log_ks - skill))
        return float(expected_matches - np.dot(counts, q) - (skill - center) * precision)

    low, high = -parameters.skill_bound, parameters.skill_bound
    if derivative(low) <= 0:
        return low
    if derivative(high) >= 0:
        return high
    for _ in range(42):
        middle = (low + high) / 2
        if derivative(middle) > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _fit(claims: v1.Claims, sources: list[str], parameters: SkillParameters,
         unseen: float, start: float, center: float,
         multiplicity: dict[v1.Entity, int] | None = None) -> dict[str, Any]:
    source_index = {s: i for i, s in enumerate(sources)}
    items = []
    cardinalities = [Counter() for _ in sources]
    for key, observations in claims.items():
        count = 1 if multiplicity is None else multiplicity.get(key, 0)
        if not count:
            continue
        candidates = tuple(sorted(set(observations.values())))
        k = len(candidates)
        indices = np.array([source_index[s] for s in observations], dtype=int)
        labels = np.array([candidates.index(v) for v in observations.values()], dtype=int)
        log_prior = np.log(np.array([(1 - unseen) / k] * k + [unseen]))
        items.append((indices, labels, log_prior, k, count))
        for source in indices:
            cardinalities[source][k] += count
    theta = np.full(len(sources), start, dtype=float)
    objective_trace = []

    def objective(skill: np.ndarray) -> float:
        value = -.5 * float(np.sum(((skill - center) / parameters.prior_sd) ** 2))
        for indices, labels, log_prior, k, count in items:
            scores = log_prior.copy()
            np.add.at(scores, labels, skill[indices])
            maximum = float(scores.max())
            normalizer = maximum + np.log(np.exp(scores - maximum).sum())
            value += count * (normalizer - np.logaddexp(skill[indices], np.log(k)).sum())
        return float(value)

    objective_trace.append(objective(theta))
    converged = False
    for iteration in range(parameters.max_iter):
        expected = np.zeros(len(sources))
        for indices, labels, log_prior, _, count in items:
            scores = log_prior.copy()
            np.add.at(scores, labels, theta[indices])
            scores -= scores.max()
            posterior = np.exp(scores)
            posterior /= posterior.sum()
            np.add.at(expected, indices, count * posterior[labels])
        updated = np.array([_solve_skill(expected[i], cardinalities[i], center, parameters)
                            for i in range(len(sources))])
        delta = float(np.max(np.abs(updated - theta)))
        theta = updated
        objective_trace.append(objective(theta))
        if delta <= parameters.tolerance:
            converged = True
            break
    return {"skill": dict(zip(sources, map(float, theta))), "unseen_prior": unseen,
            "prior_center": center, "initial_skill": start,
            "converged": converged, "iterations": iteration + 1,
            "objective_trace": objective_trace,
            "boundary_sources": [s for s, x in zip(sources, theta) if abs(x) >= parameters.skill_bound - 1e-5],
            "negative_skill_sources": [s for s, x in zip(sources, theta) if x < 0],
            "q_by_observed_candidate_count": {
                str(k): {s: match_probability(float(x), k) for s, x in zip(sources, theta)}
                for k in sorted({k for c in cardinalities for k in c})}}


def _ensemble(claims: v1.Claims, sources: list[str], parameters: SkillParameters,
              fold: int, entity_width: int) -> list[dict[str, Any]]:
    models = [_fit(claims, sources, parameters, parameters.unseen_prior, start, center)
              for center in parameters.prior_center_sensitivity for start in parameters.initial_skills]
    models.extend(_fit(claims, sources, parameters, unseen, 2.0, parameters.prior_center)
                  for unseen in parameters.unseen_sensitivity if unseen != parameters.unseen_prior)
    clusters: dict[v1.Entity, list[v1.Entity]] = defaultdict(list)
    for key in claims:
        clusters[key[:entity_width]].append(key)
    entities = sorted(clusters)
    rng = np.random.default_rng(parameters.seed + fold)
    for _ in range(parameters.bootstrap_replicates):
        sampled = Counter(int(x) for x in rng.integers(0, len(entities), size=len(entities)))
        weights = {key: count for index, count in sampled.items() for key in clusters[entities[index]]}
        models.append(_fit(claims, sources, parameters, parameters.unseen_prior, 2.0,
                           parameters.prior_center, weights))
    return models


def _decision(observations: dict[str, str], models: list[dict[str, Any]],
              parameters: SkillParameters) -> dict[str, Any]:
    if len(observations) < parameters.min_item_sources:
        return {"status": "abstain", "reasons": ["insufficient_independent_group_support"]}
    candidates = tuple(sorted(set(observations.values())))
    posteriors = [_posterior(observations, m["skill"], m["unseen_prior"], candidates)[1] for m in models]
    winners = [int(p.argmax()) for p in posteriors]
    winner = winners[0]
    reasons = []
    if len(set(winners)) != 1:
        reasons.append("winner_depends_on_orientation_or_sensitivity")
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
            _, posterior = _posterior(remaining, model["skill"], model["unseen_prior"], candidates)
            leave_out_lower = min(leave_out_lower, float(posterior[winner]))
            if int(posterior.argmax()) != winner:
                reasons.append("winner_changes_without_one_group")
    if leave_out_lower < parameters.min_leave_group_out_probability:
        reasons.append("leave_group_out_weight_below_threshold")
    return {"status": "abstain" if reasons else "supported_candidate", "reasons": sorted(set(reasons)),
            "candidate": candidates[winner] if winner < len(candidates) else None,
            "candidate_strings": list(candidates), "conditional_weight_min": lower,
            "conditional_margin_min": margin, "leave_group_out_weight_min": leave_out_lower,
            "independent_groups_assumed": len(observations),
            "unobserved_state_weight_max": max(float(p[-1]) for p in posteriors),
            "orientation_sensitivity_winners": sorted(set(None if w == len(candidates) else candidates[w]
                                                          for w in winners), key=lambda x: (x is None, x or ""))}


def run_experiment(table: DirtyTable, contract: SkillContract,
                   parameters: SkillParameters | None = None) -> dict[str, Any]:
    parameters = parameters or SkillParameters()
    claims, extraction = v1._extract(table, contract)
    key_indices, source_index, value_index = v1._indices(table, contract)
    result = {**_provenance(table, parameters), "contract": asdict(contract),
              "contract_blockers": contract.blockers(), "extraction": extraction,
              "folds": [], "decisions": [], "fit_performed": False}
    entity_width = len(contract.entity_columns)
    entity_decisions = {}
    for fold in range(parameters.folds):
        heldout = {k: o for k, o in claims.items() if v1._fold(k, entity_width, parameters) == fold}
        if not heldout:
            continue
        training = {k: o for k, o in claims.items() if v1._fold(k, entity_width, parameters) != fold}
        mapping, dependency = v1._groups(training, parameters)
        grouped_training, conflicts = v1._collapse(training, mapping)
        training, sources, blockers = v1._training_subset(grouped_training, parameters, entity_width)
        blockers = contract.blockers() + blockers
        info: dict[str, Any] = {"fold": fold, "training_entities": len(training),
                                "heldout_entities": len(heldout), "training_groups": sources,
                                "potential_dependency_pairs": dependency,
                                "source_group_mapping": mapping,
                                "group_entity_conflicts_excluded": conflicts, "blockers": blockers}
        models = []
        if not blockers:
            models = _ensemble(training, sources, parameters, fold, entity_width)
            result["fit_performed"] = True
            info["models"] = models
            if any(not m["converged"] for m in models):
                blockers.append("optimization_did_not_converge")
            if any(len(m["boundary_sources"]) * 2 >= len(sources) for m in models):
                blockers.append("numerical_skill_boundary_degeneracy")
            info["skill_sensitivity_ranges"] = {
                s: [min(m["skill"][s] for m in models), max(m["skill"][s] for m in models)] for s in sources}
            info["sign_ambiguous_sources"] = [s for s in sources
                                               if min(m["skill"][s] for m in models) < 0 < max(m["skill"][s] for m in models)]
        grouped_heldout, conflicts = v1._collapse(heldout, mapping)
        info["heldout_group_conflicts_excluded"] = conflicts
        result["folds"].append(info)
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
        source, value = row[source_index], row[value_index]
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
