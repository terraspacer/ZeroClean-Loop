"""V3 numerical protocol: maximum-objective start per sensitivity scenario.

The categorical log-advantage likelihood, priors, source assumptions, and
candidate acceptance gates are unchanged from v2. All stationary fits are
archived, but only a deterministic best converged start represents a fixed
prior/bootstrap scenario. Different scenarios remain separate sensitivity fits.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from zeroclean.data import DirtyTable
from experimental import latent_consensus as v1
from experimental import latent_consensus_v2 as v2

VERSION = "latent-consensus-map-selection-v3"
SkillParameters = v2.SkillParameters
SkillContract = v2.SkillContract


def _provenance(table: DirtyTable, parameters: SkillParameters) -> dict[str, Any]:
    result = v2._provenance(table, parameters)
    result.update(experiment=VERSION,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  v2_helpers_sha256=hashlib.sha256(Path(v2.__file__).read_bytes()).hexdigest(),
                  optimization_selection="Exact maximum penalized objective among finite converged starts; exact ties use start order")
    return result


def _select_fit(fits: list[dict[str, Any]]) -> dict[str, Any]:
    """Selection depends only on objective and convergence, never candidates."""
    eligible = [(index, fit) for index, fit in enumerate(fits)
                if fit["converged"] and np.isfinite(fit["objective_trace"][-1])]
    finite = [(index, fit) for index, fit in enumerate(fits)
              if np.isfinite(fit["objective_trace"][-1])]
    if not finite:
        raise ValueError("All start objectives are nonfinite")
    index, winner = max(eligible or finite, key=lambda pair: (pair[1]["objective_trace"][-1], -pair[0]))
    result = dict(winner)
    if not eligible:
        result["converged"] = False
    result["optimization"] = {
        "selected_start_index": index, "eligible_converged_starts": len(eligible),
        "all_start_fits": fits,
        "objective_spread": max(f["objective_trace"][-1] for _, f in finite) - min(f["objective_trace"][-1] for _, f in finite),
        "selection": "maximum finite converged penalized objective; exact tie resolves to earliest declared start"}
    return result


def _ensemble(claims: v1.Claims, sources: list[str], parameters: SkillParameters,
              fold: int, entity_width: int) -> list[dict[str, Any]]:
    def fit_scenario(unseen: float, center: float, weights=None):
        return _select_fit([v2._fit(claims, sources, parameters, unseen, start, center, weights)
                            for start in parameters.initial_skills])

    models = [fit_scenario(parameters.unseen_prior, center)
              for center in parameters.prior_center_sensitivity]
    models.extend(fit_scenario(unseen, parameters.prior_center)
                  for unseen in parameters.unseen_sensitivity if unseen != parameters.unseen_prior)
    clusters: dict[v1.Entity, list[v1.Entity]] = defaultdict(list)
    for key in claims:
        clusters[key[:entity_width]].append(key)
    entities = sorted(clusters)
    rng = np.random.default_rng(parameters.seed + fold)
    for _ in range(parameters.bootstrap_replicates):
        sampled = Counter(int(x) for x in rng.integers(0, len(entities), size=len(entities)))
        weights = {key: count for index, count in sampled.items() for key in clusters[entities[index]]}
        models.append(fit_scenario(parameters.unseen_prior, parameters.prior_center, weights))
    return models


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
                blockers.append("no_converged_start_for_sensitivity_scenario")
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
                        else v2._decision(observations, models, parameters))
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
