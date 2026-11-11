"""Frozen equal-weight ED sensitivity and a separate independent-marginals F1 proxy."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import scipy
from scipy.signal import lfilter

from zeroclean.data import DirtyTable
from zeroclean.types import Prediction, digest

VERSION = "latent-consensus-equal-model-average-v5"
PROXY_VERSION = "latent-consensus-independent-marginals-f1-proxy-v5"
V4 = Path("experimental/latent_consensus_detection_feedback_v4")
V3_EC = Path("experimental/latent_consensus_feedback_v3")
EXPECTED_V4_MANIFEST = "b775b8919d5c18c0bc79e13af0c73d53d60ed3e025731344fc971ffd8bb1c6b0"
EXPECTED_EC = "911b11611c136c5c1dcdf9e37d5d3ed4ffdf6fb2c8f414661c0a4efb78b68dd4"
EXPECTED_PRIOR = "312f76d5be26f50b9ddf80190d1339b8297b2cda620be5c27c2bdbe6abae42c8"
FROZEN_V4_TEST = Path("tests/test_latent_consensus.py")
EXPECTED_V4_TEST = "40da4ed6f319142323c63ce3abeb9613f4cb18a2c48400ba771f1ca7ba8cd2c5"
MODES = ("independent_names", "cluster_near_agreement")


@dataclass(frozen=True)
class MeanPolicy:
    threshold: float = .5
    selected_models_per_assumption: int = 17
    source_assumption_weight: float = .5
    equality_decision: str = "positive at exact .5"
    repair_gate: str = "retain exact positive frozen v3 feedback ED score if a fixed repair would be outside detection"


@dataclass(frozen=True)
class ProxyPolicy:
    objective: str = "E[F1] only under an additional independent-marginals Bernoulli proxy over eligible cells"
    mandatory_repairs: bool = True
    probability_order_tie: str = "ascending row then column"
    utility_tie_tolerance: float = 1e-12
    negative_roundoff_tolerance: float = 1e-10
    mass_tolerance: float = 1e-10
    division_tolerance: float = 1e-8
    prediction_encoding: str = "0/1 decision scores on eligible cells; raw conditional means archived separately"
    empty_empty_f1: float = 0.0


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gzip(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_new(path: Path, body: Any):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def write_gzip_new(path: Path, body: Any):
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def aggregate_original(cell, contexts, policy: MeanPolicy = MeanPolicy()):
    """Uniformly average full scenarios; LOO scenario counts never alter weights."""
    if not cell["eligible_both_source_assumptions"]:
        if cell["error_weight_interval"] is not None:
            raise ValueError("Unsupported frozen original has an interval")
        return {"eligible": False, "mean_error_weight": None, "mode_evidence": {},
                "v4_action": cell["action"], "v4_error_weight_interval": None,
                "combined_leave_group_out_error_range": None}
    mode_evidence = {}
    for mode in MODES:
        item = cell["source_assumptions"][mode]
        if item["eligibility_blockers"]:
            raise ValueError("Eligible frozen cell has source blockers")
        context = contexts[item["context_id"]]
        if context["fold_blockers"] or context["source_assumption"] != mode:
            raise ValueError("Frozen context is ineligible")
        if cell["original"] not in context["candidate_strings"]:
            raise ValueError("No invented score for an absent original")
        index = context["candidate_strings"].index(cell["original"])
        buckets = {}
        for scenario in context["scenarios"]:
            omitted = scenario["omitted_group"]
            model = scenario["selected_model_index"]
            values = buckets.setdefault(omitted, {})
            if model in values:
                raise ValueError("Duplicated frozen model/scenario")
            weight = float(scenario["posterior_weights"][index])
            if not np.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError("Invalid frozen original weight")
            values[model] = weight
        if set(buckets) != {None, *context["grouped_claims"]}:
            raise ValueError("Frozen full/LOO scenario coverage changed")
        means = {}
        for omitted, values in buckets.items():
            if set(values) != set(range(policy.selected_models_per_assumption)):
                raise ValueError("Frozen selected-model count changed")
            means[omitted] = float(np.mean([values[k] for k in range(policy.selected_models_per_assumption)]))
        mode_evidence[mode] = {"context_id": item["context_id"],
            "full_original_weights": [buckets[None][k] for k in range(policy.selected_models_per_assumption)],
            "full_original_mean": means[None], "full_error_mean": 1-means[None],
            "leave_group_out_error_means": {str(g): 1-p for g, p in means.items() if g is not None},
            "each_full_model_weight": policy.source_assumption_weight / policy.selected_models_per_assumption}
    error = policy.source_assumption_weight * sum(v["full_error_mean"] for v in mode_evidence.values())
    diagnostics = [error]
    for mode in MODES:
        other = next(m for m in MODES if m != mode)
        diagnostics.extend(policy.source_assumption_weight * (omitted + mode_evidence[other]["full_error_mean"])
                           for omitted in mode_evidence[mode]["leave_group_out_error_means"].values())
    return {"eligible": True, "mean_error_weight": float(error), "mode_evidence": mode_evidence,
            "v4_action": cell["action"], "v4_error_weight_interval": cell["error_weight_interval"],
            "combined_leave_group_out_error_range": [min(diagnostics), max(diagnostics)],
            "leave_group_out_mean_decision_changes": any((p >= policy.threshold) != (error >= policy.threshold) for p in diagnostics)}


def archive_claim_contexts(v4_evidence, contexts):
    """Presentation-only raw claims: never add votes or change any posterior."""
    grouped = {}
    for cell in v4_evidence["cells"]:
        context_id = digest({"column": cell["column"], "entity": cell["entity"]})
        record = grouped.setdefault(context_id, {"claim_context_id": context_id,
            "attribute": cell["column"], "entity": cell["entity"], "raw_claims": [],
            "source_assumptions": {}, "conditional_hypotheses_only": True,
            "warning": "No external snapshot/catalog evidence; source grouping does not prove copying or independent truth."})
        record["raw_claims"].append({"row": cell["row"], "source": cell["source"],
            "value": cell["original"], "blank_or_whitespace": not cell["original"].strip()})
        for mode in MODES:
            node_id = cell["source_assumptions"][mode]["context_id"]
            if node_id is None:
                continue
            node = contexts[node_id]
            record["source_assumptions"][mode] = {"frozen_context_id": node_id, "fold": node["fold"],
                "source_to_group": node["source_mapping"], "trained_groups": node["trained_groups"],
                "active_group_claims": node["grouped_claims"], "fold_blockers": node["fold_blockers"],
                "scenario_linkage": {"selected_model_indices": sorted({s["selected_model_index"] for s in node.get("scenarios", [])}),
                    "omission_groups": sorted({s["omitted_group"] for s in node.get("scenarios", []) if s["omitted_group"] is not None}),
                    "full_and_leave_out_posterior_context_id": node_id}}
    for record in grouped.values():
        claims = defaultdict(list)
        for claim in record["raw_claims"]:
            claims[claim["source"]].append(claim)
        source_conflicts, duplicate_extras = [], 0
        for source, rows in claims.items():
            counts = Counter(r["value"] for r in rows if r["value"].strip())
            duplicate_extras += sum(n-1 for n in counts.values())
            if len(counts) > 1:
                source_conflicts.append({"source": source, "values": sorted(counts),
                    "rows": [r["row"] for r in rows]})
        record["raw_source_conflicts"] = source_conflicts
        record["duplicate_nonempty_claim_extra_rows"] = duplicate_extras
        conflicted = {c["source"] for c in source_conflicts}
        for mode, view in record["source_assumptions"].items():
            raw_group_values = defaultdict(set)
            exclusions = []
            for claim in record["raw_claims"]:
                source, value = claim["source"], claim["value"]
                reasons = []
                group = view["source_to_group"].get(source)
                if not value.strip(): reasons.append("blank_or_whitespace")
                if source in conflicted: reasons.append("conflicted_raw_source")
                if group is None: reasons.append("source_absent_frozen_training_mapping")
                elif group not in view["trained_groups"]: reasons.append("group_not_trained")
                elif group not in view["active_group_claims"]: reasons.append("group_inactive_or_conflicted")
                if group is not None and value.strip() and source not in conflicted:
                    raw_group_values[group].add(value)
                if reasons:
                    exclusions.append({"row": claim["row"], "source": source,
                        "value": value, "frozen_group": group, "reasons": reasons})
            view["raw_group_value_sets"] = {g: sorted(values) for g, values in raw_group_values.items()}
            view["group_value_conflicts"] = {g: values for g, values in view["raw_group_value_sets"].items() if len(values)>1}
            view["claim_exclusions"] = exclusions
    return grouped


def poisson_binomial(probabilities, policy: ProxyPolicy = ProxyPolicy()):
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid proxy marginal weights")
    coefficients = np.array([1.0])
    for value in p:
        updated = np.zeros(len(coefficients)+1)
        updated[:-1] += (1-value)*coefficients
        updated[1:] += value*coefficients
        coefficients = updated
    if abs(float(coefficients.sum())-1) > policy.mass_tolerance:
        raise ArithmeticError("Poisson-binomial mass check failed")
    return coefficients


def leave_one_distribution(total, probability, policy: ProxyPolicy = ProxyPolicy()):
    n = len(total)-1
    p = float(probability)
    if p == 0:
        result = total[:-1].copy()
    elif p == 1:
        result = total[1:].copy()
    elif p <= .5:
        result = lfilter([1/(1-p)], [1, p/(1-p)], total)[:n]
    else:
        result = lfilter([1/p], [1, (1-p)/p], total[::-1])[:n][::-1]
    if not np.isfinite(result).all() or result.min(initial=0) < -policy.negative_roundoff_tolerance:
        raise ArithmeticError("Synthetic-division coefficient check failed")
    clipped = int((result < 0).sum())
    result = np.maximum(result, 0)  # Explicit, preregistered tiny-roundoff correction.
    mass_error = abs(float(result.sum())-1)
    rebuilt = np.zeros(n+1)
    rebuilt[:-1] += (1-p)*result
    rebuilt[1:] += p*result
    residual = float(np.max(np.abs(rebuilt-total)))
    if mass_error > policy.division_tolerance or residual > policy.division_tolerance:
        raise ArithmeticError("Synthetic-division mass/residual check failed")
    return result, {"tiny_negative_coefficients_clipped": clipped,
                    "mass_error": mass_error, "reconstruction_residual": residual}


def conditional_f1_for_set(probabilities, selected, policy: ProxyPolicy = ProxyPolicy()):
    """Exact expectation under the independent-marginals proxy, up to roundoff."""
    p = np.asarray(probabilities, dtype=float)
    selected = sorted(set(selected))
    if not selected:
        return policy.empty_empty_f1
    total = poisson_binomial(p, policy)
    joint = np.zeros(len(p)+1)
    for index in selected:
        leave_one, _ = leave_one_distribution(total, p[index], policy)
        joint[1:] += p[index]*leave_one
    return float(2*np.dot(joint[1:], 1/(len(selected)+np.arange(1, len(p)+1))))


def optimize_independent_marginals_proxy(probabilities, coordinates, mandatory=(),
                                        policy: ProxyPolicy = ProxyPolicy()):
    p = np.asarray(probabilities, dtype=float)
    n = len(p)
    if len(coordinates) != n or len(set(map(tuple, coordinates))) != n:
        raise ValueError("Invalid proxy coordinates")
    required = set(mandatory)
    if any(type(i) is not int or not 0 <= i < n for i in required):
        raise ValueError("Invalid mandatory proxy index")
    total = poisson_binomial(p, policy)
    free = sorted((i for i in range(n) if i not in required), key=lambda i: (-p[i], tuple(coordinates[i])))
    joint = np.zeros(n+1)
    diagnostics = {"tiny_negative_coefficients_clipped": 0, "maximum_mass_error": 0.0,
                   "maximum_reconstruction_residual": 0.0, "total_mass_error": abs(float(total.sum())-1)}

    def add(index):
        distribution, check = leave_one_distribution(total, p[index], policy)
        joint[1:] += p[index]*distribution
        diagnostics["tiny_negative_coefficients_clipped"] += check["tiny_negative_coefficients_clipped"]
        diagnostics["maximum_mass_error"] = max(diagnostics["maximum_mass_error"], check["mass_error"])
        diagnostics["maximum_reconstruction_residual"] = max(diagnostics["maximum_reconstruction_residual"], check["reconstruction_residual"])

    for index in sorted(required):
        add(index)
    counts = np.arange(1, n+1)

    def utility(k):
        result = policy.empty_empty_f1 if k == 0 else float(2*np.dot(joint[1:], 1/(k+counts)))
        if not np.isfinite(result) or not -policy.division_tolerance <= result <= 1+policy.division_tolerance:
            raise ArithmeticError("Invalid independent-marginals expected utility")
        return result

    best_free, best_value = 0, utility(len(required))
    path = [{"positive_count": len(required), "optional_positive_count": 0,
             "independent_marginals_proxy_expected_f1": best_value}]
    for size, index in enumerate(free, start=1):
        add(index)
        value = utility(len(required)+size)
        path.append({"positive_count": len(required)+size, "optional_positive_count": size,
                     "independent_marginals_proxy_expected_f1": value})
    maximum = max(entry["independent_marginals_proxy_expected_f1"] for entry in path)
    chosen = next(entry for entry in path
                  if maximum-entry["independent_marginals_proxy_expected_f1"] <= policy.utility_tie_tolerance)
    best_free = chosen["optional_positive_count"]
    best_value = chosen["independent_marginals_proxy_expected_f1"]
    selected = sorted(required | set(free[:best_free]))
    before = float(p[free[best_free-1]]) if best_free else None
    after = float(p[free[best_free]]) if best_free < len(free) else None
    cutoff = (before+after)/2 if before is not None and after is not None else (1.0 if before is None else 0.0)
    return {"selected_indices": selected, "mandatory_indices": sorted(required), "free_order": free,
            "selected_optional_count": best_free, "selected_positive_count": len(selected),
            "selected_independent_marginals_proxy_expected_f1": best_value,
            "maximum_independent_marginals_proxy_expected_f1": maximum,
            "utility_path": path, "numerical_diagnostics": diagnostics,
            "threshold_description": {"probability_cutoff": cutoff, "last_selected_free_probability": before,
                "first_unselected_free_probability": after,
                "coordinate_tie_cutoff": list(coordinates[free[best_free-1]]) if best_free and before == after else None,
                "mandatory_repair_exceptions": True, "selected_rank_set_is_authoritative": True},
            "optimality_scope": "Declared constrained ranked family under independent-marginals proxy only; not fitted categorical joint or benchmark"}


def _check_repair_prior(fixed, threshold=.5):
    for key in fixed.repairs:
        if fixed.probabilities[key] < threshold:
            raise ValueError("Frozen v3 repair has no positive frozen v3 ED score; no silent normalization permitted")


def _repair_coherence(probabilities, fixed, threshold=.5):
    gates = {}
    for key in fixed.repairs:
        if probabilities[key] < threshold:
            gates[key] = {"candidate_score": float(probabilities[key]),
                          "retained_frozen_v3_ed_score": float(fixed.probabilities[key]),
                          "reason": "frozen_v3_repair_consistency_gate"}
            probabilities[key] = fixed.probabilities[key]
    if any(probabilities[key] < threshold for key in fixed.repairs):
        raise AssertionError("A fixed repair lies outside detection")
    return gates


def _validate_inference_inputs(table, prior, fixed, evidence):
    prior.validate(table.shape); fixed.validate(table.shape)
    if evidence["dirty_sha256"] != table.fingerprint:
        raise ValueError("Stale mean evidence")
    for prediction in (prior, fixed):
        if prediction.metadata.get("dirty_sha256", table.fingerprint) != table.fingerprint:
            raise ValueError("Stale prediction fingerprint")
    seen = set()
    for cell in evidence["cells"]:
        i, j = cell["row"], cell["column_index"]
        if type(i) is not int or type(j) is not int or not 0 <= i < table.shape[0] or not 0 <= j < table.shape[1] or (i, j) in seen:
            raise ValueError("Invalid/duplicate mean coordinate")
        seen.add((i, j))
        if cell["original"] != table.rows[i][j] or cell["column"] != table.columns[j]:
            raise ValueError("Stale mean original/column")
        point = cell["mean_evidence"]["mean_error_weight"]
        if cell["mean_evidence"]["eligible"]:
            if point is None or not np.isfinite(point) or not 0 <= point <= 1:
                raise ValueError("Invalid modeled mean error weight")
        elif point is not None:
            raise ValueError("Unsupported cell has an invented mean score")


def apply_mean(table, prior, fixed, evidence, policy: MeanPolicy = MeanPolicy()):
    _validate_inference_inputs(table, prior, fixed, evidence)
    _check_repair_prior(fixed, policy.threshold)
    probability = prior.probabilities.copy()
    for cell in evidence["cells"]:
        if cell["mean_evidence"]["eligible"]:
            probability[cell["row"], cell["column_index"]] = cell["mean_evidence"]["mean_error_weight"]
    gates = _repair_coherence(probability, fixed, policy.threshold)
    trace = []
    for cell in evidence["cells"]:
        key = cell["row"], cell["column_index"]
        trace.append({**cell, "prior_score": float(prior.probabilities[key]), "final_score": float(probability[key]),
                      "repair_consistency_gate": gates.get(key), "inherited_repair": fixed.repairs.get(key)})
    return Prediction(probability, dict(fixed.repairs), {
        "system": VERSION, "dirty_sha256": table.fingerprint, "policy": asdict(policy),
        "fully_verified_main_method": False, "new_llm_calls": 0, "refit_performed": False,
        "conditional_hypotheses_only": True, "no_external_catalog_or_snapshot_evidence": True,
        "frozen_semantic_contracts": evidence.get("frozen_semantic_contracts", []),
        "score_meaning": "equal-weight conditional scenario average except explicit frozen-v3 repair-coherence fallback; not calibrated correctness or exact Bayesian marginalization"}).validate(table.shape), {"trace": trace, "repair_gates": [{"row": i, "column": j, **v} for (i, j), v in sorted(gates.items())]}


def apply_proxy(table, prior, fixed, evidence, mean_policy: MeanPolicy = MeanPolicy(),
                policy: ProxyPolicy = ProxyPolicy()):
    _validate_inference_inputs(table, prior, fixed, evidence)
    _check_repair_prior(fixed, mean_policy.threshold)
    modeled = [c for c in evidence["cells"] if c["mean_evidence"]["eligible"]]
    coordinates = [(c["row"], c["column_index"]) for c in modeled]
    p = [c["mean_evidence"]["mean_error_weight"] for c in modeled]
    mandatory = [i for i, key in enumerate(coordinates) if key in fixed.repairs]
    solution = optimize_independent_marginals_proxy(p, coordinates, mandatory, policy)
    selected = set(solution["selected_indices"])
    probability = prior.probabilities.copy()
    for index, key in enumerate(coordinates):
        probability[key] = float(index in selected)
    gates = _repair_coherence(probability, fixed, mean_policy.threshold)
    trace = []
    modeled_index = {key: index for index, key in enumerate(coordinates)}
    for cell in evidence["cells"]:
        key = cell["row"], cell["column_index"]
        trace.append({**cell, "prior_score": float(prior.probabilities[key]), "final_score": float(probability[key]),
                      "proxy_modeled_index": modeled_index.get(key), "repair_consistency_gate": gates.get(key),
                      "inherited_repair": fixed.repairs.get(key)})
    return Prediction(probability, dict(fixed.repairs), {
        "system": PROXY_VERSION, "dirty_sha256": table.fingerprint, "policy": asdict(policy),
        "fully_verified_main_method": False, "new_llm_calls": 0, "refit_performed": False,
        "conditional_hypotheses_only": True, "no_external_catalog_or_snapshot_evidence": True,
        "frozen_semantic_contracts": evidence.get("frozen_semantic_contracts", []),
        "score_meaning": policy.prediction_encoding,
        "additional_assumption": "independent-marginals Bernoulli proxy; ignores shared-entity categorical dependence",
        "optimality_scope": solution["optimality_scope"]}).validate(table.shape), {"trace": trace, "solution": solution,
            "repair_gates": [{"row": i, "column": j, **v} for (i, j), v in sorted(gates.items())]}


def _prediction_body(prediction):
    return {"probabilities": prediction.probabilities.tolist(),
            "repairs": [{"row": i, "column": j, "value": value} for (i, j), value in sorted(prediction.repairs.items())],
            "metadata": prediction.metadata}


def _bundle(table, prior, prediction, audit, evidence_path, evidence_sha, claim_contexts, claim_contexts_sha):
    cells = []
    for trace in audit["trace"]:
        i, j = trace["row"], trace["column_index"]
        if prediction.probabilities[i, j] != prior.probabilities[i, j]:
            cells.append({"row": i, "column": table.columns[j], "column_index": j,
                          "original": table.rows[i][j], "original_tuple": dict(zip(table.columns, table.rows[i])),
                          "ed_evidence": trace, "inherited_strict_ec_value": prediction.repairs.get((i, j)),
                          "raw_dirty_claim_context": claim_contexts[trace["claim_context_id"]],
                          "independent_verification_status": "not_run"})
    return {"experiment": prediction.metadata["system"], "dataset": table.name, "dirty_sha256": table.fingerprint,
            "purpose": "Prepared independent ED review only; no LLM call or new correction permission",
            "mean_evidence_path": str(evidence_path), "mean_evidence_sha256": evidence_sha,
            "claim_contexts_sha256": claim_contexts_sha,
            "conditional_hypotheses_only": True,
            "independent_verification_status": "not_run", "cells": cells}


def _save_output(directory, table, prior, prediction, audit, evidence_path, input_hashes, claim_contexts, claim_contexts_sha):
    prediction.metadata.update(input_hashes=input_hashes, inherited_ec_sha256=EXPECTED_EC,
        mean_evidence_sha256=sha(evidence_path), independent_verification_status="not_run")
    write_new(directory / "prediction.json", _prediction_body(prediction))
    audit.update(prediction_sha256=sha(directory / "prediction.json"), mean_evidence_sha256=sha(evidence_path))
    write_gzip_new(directory / "audit.json.gz", audit)
    bundle = _bundle(table, prior, prediction, audit, evidence_path, sha(evidence_path), claim_contexts, claim_contexts_sha)
    bundle["parent_prediction_sha256"] = sha(directory / "prediction.json")
    bundle["request_hash"] = digest(bundle)
    write_new(directory / "verification_bundle.json", bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-output", type=Path, default=Path("experimental/latent_consensus_model_average_v5"))
    parser.add_argument("--proxy-output", type=Path, default=Path("experimental/latent_consensus_expected_f1_proxy_v5"))
    args = parser.parse_args()
    for path in (args.mean_output, args.proxy_output):
        if not path.resolve().is_relative_to(Path("experimental").resolve()):
            raise ValueError("Output must remain under experimental/")
    if args.mean_output.resolve() == args.proxy_output.resolve():
        raise ValueError("Mean and proxy artifacts must be separate")
    if sha(V4 / "final_manifest.json") != EXPECTED_V4_MANIFEST or sha(V3_EC / "prediction.json") != EXPECTED_EC:
        raise ValueError("Frozen v3/v4 input changed")
    if sha(FROZEN_V4_TEST) != EXPECTED_V4_TEST:
        raise ValueError("Live frozen-v4 test binding changed; v5 tests must use their separate module")
    v4_manifest = json.loads((V4 / "final_manifest.json").read_text())
    for name, wanted in v4_manifest["artifacts"].items():
        if sha(Path(name)) != wanted:
            raise ValueError("Frozen v4 artifact changed")
    prior_path = V4 / "frozen_inputs/loop_prior.json"
    if sha(prior_path) != EXPECTED_PRIOR:
        raise ValueError("Exact frozen source-v2 loop prior changed")
    source_paths = [Path(__file__), Path(__file__).with_name("latent_consensus_model_average_v5_protocol.md"),
        Path(__file__).with_name("latent_consensus_expected_f1_v5_protocol.md"), Path("tests/test_latent_consensus_v5.py"), FROZEN_V4_TEST,
        Path("zeroclean/data.py"), Path("zeroclean/types.py")]
    inputs = [prior_path, V3_EC / "prediction.json", V4 / "final_manifest.json", V4 / "posterior_contexts.json.gz",
        V4 / "detection_evidence.json.gz", V4 / "frozen_inputs/dirty_table.json.gz", *source_paths]
    input_hashes = {str(path): sha(path) for path in inputs}
    args.mean_output.mkdir(parents=True, exist_ok=False)
    args.proxy_output.mkdir(parents=True, exist_ok=False)
    frozen = args.mean_output / "frozen_inputs"
    frozen.mkdir()
    copies = [(path, frozen / f"input_{i}_{path.name}") for i, path in enumerate(inputs)]
    frozen_paths, frozen_hashes = {}, {}
    for source, destination in copies:
        shutil.copyfile(source, destination)
        frozen_paths[str(source)] = destination
        frozen_hashes[str(destination)] = input_hashes[str(source)]
        if sha(destination) != input_hashes[str(source)]:
            raise ValueError("Input changed while freezing")
    common_manifest = {"dirty_prior_sha256": EXPECTED_PRIOR, "inherited_ec_sha256": EXPECTED_EC,
        "input_hashes": input_hashes, "frozen_input_hashes": frozen_hashes,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "no_refit": True, "new_llm_calls": 0, "accuracy_evaluator_executed": False}
    write_new(args.mean_output / "pre_run_manifest.json", {**common_manifest, "experiment": VERSION,
        "policy": asdict(MeanPolicy()), "separate_proxy_protocol_frozen": True})
    write_new(args.proxy_output / "pre_run_manifest.json", {**common_manifest, "experiment": PROXY_VERSION,
        "policy": asdict(ProxyPolicy()), "mean_artifact_directory": str(args.mean_output),
        "not_actual_categorical_joint_expectation": True})
    dirty = read_gzip(frozen_paths[str(V4 / "frozen_inputs/dirty_table.json.gz")])
    table = DirtyTable(dirty["name"], tuple(dirty["columns"]), tuple(tuple(r) for r in dirty["rows"]), dirty["fingerprint"])
    prior = Prediction.load(frozen_paths[str(prior_path)]).validate(table.shape)
    fixed = Prediction.load(frozen_paths[str(V3_EC / "prediction.json")]).validate(table.shape)
    _check_repair_prior(fixed)
    contexts = read_gzip(frozen_paths[str(V4 / "posterior_contexts.json.gz")])
    v4_evidence = read_gzip(frozen_paths[str(V4 / "detection_evidence.json.gz")])
    if v4_evidence["dirty_sha256"] != table.fingerprint:
        raise ValueError("Frozen evidence dirty fingerprint mismatch")
    claim_contexts = archive_claim_contexts(v4_evidence, contexts)
    claim_context_path = args.mean_output / "claim_contexts.json.gz"
    write_gzip_new(claim_context_path, claim_contexts)
    mean_cells = []
    for cell in v4_evidence["cells"]:
        i, j = cell["row"], cell["column_index"]
        if table.rows[i][j] != cell["original"]:
            raise ValueError("Stale original string")
        mean_cells.append({"row": i, "column": cell["column"], "column_index": j,
            "original": cell["original"], "entity": cell["entity"], "source": cell["source"],
            "claim_context_id": digest({"column": cell["column"], "entity": cell["entity"]}),
            "mean_evidence": aggregate_original(cell, contexts)})
    evidence = {"experiment": VERSION, "dirty_sha256": table.fingerprint, "policy": asdict(MeanPolicy()),
        "frozen_semantic_contracts": v4_evidence["frozen_semantic_contracts"],
        "input_hashes": input_hashes, "cells": mean_cells,
        "claim_contexts_path": str(claim_context_path), "claim_contexts_sha256": sha(claim_context_path),
        "interpretation": "Scenario-averaged conditional weights, not calibrated truth or exact Bayesian model marginalization"}
    evidence_path = args.mean_output / "mean_evidence.json.gz"
    write_gzip_new(evidence_path, evidence)
    mean, mean_audit = apply_mean(table, prior, fixed, evidence)
    _save_output(args.mean_output, table, prior, mean, mean_audit, evidence_path, input_hashes, claim_contexts, sha(claim_context_path))
    try:
        proxy, proxy_audit = apply_proxy(table, prior, fixed, evidence)
    except (ArithmeticError, ValueError) as exc:
        write_new(args.proxy_output / "numerical_failure.json", {"status": "optional proxy stopped",
            "exception": type(exc).__name__, "reason": str(exc), "policy_unchanged": True,
            "mean_prediction_sha256": sha(args.mean_output / "prediction.json")})
        raise
    _save_output(args.proxy_output, table, prior, proxy, proxy_audit, evidence_path, input_hashes, claim_contexts, sha(claim_context_path))
    coverage = {"interpretation": "Inference coverage only; no truth labels or accuracy evaluation",
        "eligible_cells": sum(c["mean_evidence"]["eligible"] for c in mean_cells),
        "unsupported_cells": sum(not c["mean_evidence"]["eligible"] for c in mean_cells),
        "mean_vs_v4_action_coverage": dict(Counter(("mean_positive" if c["mean_evidence"]["mean_error_weight"] >= .5 else "mean_negative") + "__" + c["mean_evidence"]["v4_action"]
            for c in mean_cells if c["mean_evidence"]["eligible"])),
        "mean_leave_group_out_diagnostic_changes": sum(bool(c["mean_evidence"].get("leave_group_out_mean_decision_changes")) for c in mean_cells),
        "mean_repair_consistency_fallbacks": len(mean_audit["repair_gates"]),
        "proxy_modeled_positive_count": proxy_audit["solution"]["selected_positive_count"],
        "proxy_mandatory_modeled_repairs": len(proxy_audit["solution"]["mandatory_indices"]),
        "proxy_repair_consistency_fallbacks": len(proxy_audit["repair_gates"]),
        "fixed_correction_count": len(fixed.repairs)}
    write_new(args.mean_output / "coverage.json", coverage)
    for path, wanted in input_hashes.items():
        if sha(Path(path)) != wanted:
            raise ValueError("Frozen input changed during execution")
    for directory in (args.mean_output, args.proxy_output):
        names = ["pre_run_manifest.json", "prediction.json", "audit.json.gz", "verification_bundle.json"]
        if directory == args.mean_output:
            names.extend(["mean_evidence.json.gz", "claim_contexts.json.gz", "coverage.json"])
        write_new(directory / "output_manifest.json", {"artifacts": {str(directory / name): sha(directory / name) for name in names},
            "accuracy_evaluator_executed": False, "independent_verification_status": "not_run"})
        print(str(directory / "prediction.json") + " " + sha(directory / "prediction.json"), flush=True)


if __name__ == "__main__":
    main()
