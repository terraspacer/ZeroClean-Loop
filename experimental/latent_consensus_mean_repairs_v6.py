"""Mean-model repair ablation with immutable v5 ED; no refit, verifier or evaluator."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

from zeroclean.data import DirtyTable
from zeroclean.sandbox import evaluate_rules, validate_rules
from zeroclean.statistics import DirtyIndex
from zeroclean.types import Prediction, digest
from experimental import latent_consensus_feedback as frozen_cf

VERSION = "latent-consensus-mean-repairs-v6-statistical-ablation"
V5 = Path("experimental/latent_consensus_model_average_v5")
V4 = Path("experimental/latent_consensus_detection_feedback_v4")
EXPECTED_V5_MANIFEST = "a8f82621832cc8d8b43cda5eba71db21e0f937ca8c80ae962c1d04501afd0bf8"
EXPECTED_ED = "071e430048267ee1d053a78d29ccbf7e645c9618bd45ac66b0cac4ebf9bffe08"
EXPECTED_PRIOR = "312f76d5be26f50b9ddf80190d1339b8297b2cda620be5c27c2bdbe6abae42c8"
RULES = Path("artifacts/llm/flights/pass2.json")
EXPECTED_RULES = "e031188e337da807cb7054d5a54866b5c69f0645d4becdd0392b0f85433d18ea"
MODES = ("independent_names", "cluster_near_agreement")


@dataclass(frozen=True)
class RepairPolicy:
    ed_threshold: float = .5
    candidate_weight: float = .8
    candidate_margin: float = .2
    tie_tolerance: float = 1e-12
    top_k: int = 3
    selected_models: int = 17
    repair_confidence: float = .8
    delta_weights: tuple[float, ...] = (2.2, 1.8, .65, .35)
    logit_epsilon: float = 1e-6
    runtime_error_rate: float = .05
    representation_policy: str = "changed original/candidate must be bare clocks with different wall-clock minutes"
    prior_coherence_policy: str = "remove ED-incompatible prior repairs; fail closed if this introduces a parent violation/runtime failure"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_new(path, body):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def write_gzip_new(path, body):
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


_TWELVE = re.compile(r"(?P<h>0?[1-9]|1[0-2]):(?P<m>[0-5][0-9])\s*(?P<ap>[ap])(?:\.?m\.?)?", re.I)
_TWENTY_FOUR = re.compile(r"(?P<h>[01]?[0-9]|2[0-3]):(?P<m>[0-5][0-9])")
_FRAGMENT = re.compile(r"(?<![0-9])(?:0?[1-9]|1[0-2]):[0-5][0-9]\s*[ap](?:\.?m\.?)?|(?<![0-9])(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?![0-9])", re.I)


def clock_representation(value):
    stripped = value.strip()
    match = _TWELVE.fullmatch(stripped)
    if match:
        minutes = (int(match["h"]) % 12)*60 + int(match["m"]) + (720 if match["ap"].lower() == "p" else 0)
        return {"raw": value, "bare": True, "minutes": minutes, "syntax": "12_hour", "annotations": None}
    match = _TWENTY_FOUR.fullmatch(stripped)
    if match:
        return {"raw": value, "bare": True, "minutes": int(match["h"])*60+int(match["m"]),
                "syntax": "24_hour", "annotations": None}
    fragments = list(_FRAGMENT.finditer(stripped))
    minutes = None
    if len(fragments) == 1:
        fragment = fragments[0]
        parsed = clock_representation(fragment.group())
        minutes = parsed["minutes"]
        annotation = {"prefix": stripped[:fragment.start()], "suffix": stripped[fragment.end():]}
    else:
        annotation = {"uninterpreted_value": stripped, "clock_fragment_count": len(fragments)}
    return {"raw": value, "bare": False, "minutes": minutes,
            "syntax": "annotated_or_uninterpreted", "annotations": annotation}


def representation_gate(original, candidate):
    before, after = clock_representation(original), clock_representation(candidate)
    reasons = []
    if before["minutes"] is not None and before["minutes"] == after["minutes"]:
        reasons.append("same_clock_representation_or_annotation_change")
    if not before["bare"] or not after["bare"]:
        reasons.append("changed_value_has_annotation_status_or_uninterpreted_context")
    return {"passed": not reasons, "blockers": reasons, "original": before, "candidate": after,
            "peer_annotation_veto": False}


def rank_observed_alternatives(original, nodes, policy: RepairPolicy = RepairPolicy()):
    means, vectors, unknowns = {}, {}, {}
    for mode in MODES:
        node = nodes[mode]
        vocab = node["candidate_strings"]
        if original not in vocab:
            raise ValueError("Eligible original absent from frozen vocabulary")
        full = {}
        for scenario in node["scenarios"]:
            if scenario["omitted_group"] is None:
                index = scenario["selected_model_index"]
                if index in full:
                    raise ValueError("Duplicate frozen full-fit scenario")
                full[index] = scenario["posterior_weights"]
        if set(full) != set(range(policy.selected_models)):
            raise ValueError("Frozen full-fit coverage changed")
        matrix = np.array([full[i] for i in range(policy.selected_models)], dtype=float)
        if matrix.shape != (policy.selected_models, len(vocab)+1) or not np.isfinite(matrix).all() or ((matrix<0)|(matrix>1)).any() or not np.allclose(matrix.sum(axis=1), 1, rtol=0, atol=1e-12):
            raise ValueError("Invalid frozen categorical posterior")
        vectors[mode] = {value: matrix[:, k].tolist() for k, value in enumerate(vocab)}
        means[mode] = {value: float(matrix[:, k].mean()) for k, value in enumerate(vocab)}
        unknowns[mode] = float(matrix[:, -1].mean())
    shared = sorted(set(means[MODES[0]]) & set(means[MODES[1]]))
    weights = {value: .5*sum(means[mode][value] for mode in MODES) for value in shared}
    residual = .5*sum(unknowns[mode]+sum(p for value, p in means[mode].items() if value not in shared) for mode in MODES)
    if not np.isclose(sum(weights.values())+residual, 1, rtol=0, atol=1e-12):
        raise ValueError("Shared/residual mean mass is not conserved")
    alternatives = sorted((value for value in shared if value != original), key=lambda value: (-weights[value], value))
    top = alternatives[0] if alternatives else None
    second_nonoriginal = weights[alternatives[1]] if len(alternatives)>1 else None
    tied = bool(top is not None and second_nonoriginal is not None and weights[top]-second_nonoriginal <= policy.tie_tolerance)
    competitor = max([residual, *(p for value, p in weights.items() if value != top)]) if top is not None else None
    margin = weights[top]-competitor if top is not None else None
    blockers = []
    if top is None: blockers.append("no_shared_observed_nonoriginal")
    elif weights[top] < policy.candidate_weight: blockers.append("mean_candidate_weight_below_threshold")
    if tied: blockers.append("top_nonoriginal_not_unique")
    if margin is not None and margin < policy.candidate_margin: blockers.append("mean_margin_below_threshold")
    candidates = [{"rank": rank, "value": value, "mean_weight": weights[value],
                   "per_assumption_mean_weight": {mode: means[mode][value] for mode in MODES},
                   "full_fit_weights": {mode: vectors[mode][value] for mode in MODES}}
                  for rank, value in enumerate(alternatives[:policy.top_k], start=1)]
    return {"original_mean_weight": weights[original], "shared_observed_weights": weights,
            "residual_unshared_or_unknown_mass": residual,
            "nonshared_observed_values": {mode: sorted(set(means[mode])-set(shared)) for mode in MODES},
            "top_nonoriginal": top, "top_margin": margin, "largest_competing_mass": competitor,
            "top_gate_blockers": blockers, "top3": candidates, "renormalized": False}


def matrix(rules, records):
    if not records:
        return np.empty((0, len(rules)), dtype=np.int8)
    chunks = [np.asarray(evaluate_rules(rules, records[start:start+1000]), dtype=np.int8).reshape(len(records[start:start+1000]), len(rules))
              for start in range(0, len(records), 1000)]
    return np.concatenate(chunks, axis=0)


def repaired_rows(table, repairs):
    rows = [list(row) for row in table.rows]
    for (i, j), value in repairs.items(): rows[i][j] = value
    return rows


class CoherenceConflict(ValueError):
    pass


def apply_repair_policy(table, prior, fixed_ed, artifact, evidence, policy: RepairPolicy = RepairPolicy()):
    prior.validate(table.shape); fixed_ed.validate(table.shape)
    if evidence["dirty_sha256"] != table.fingerprint or artifact.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Stale repair evidence/rules")
    records = table.records()
    accepted, validation = validate_rules(artifact.get("rules", []), records, policy.runtime_error_rate)
    rules = [r for r in accepted if r["target"] in table.columns]
    multi = np.array([r["kind"] == "multi" for r in rules], dtype=bool)
    original_matrix = matrix(rules, records)
    parent_rows = repaired_rows(table, prior.repairs)
    parent_matrix = matrix(rules, [dict(zip(table.columns, row)) for row in parent_rows])
    coherent = dict(prior.repairs)
    coherence = []
    for key, value in sorted(prior.repairs.items()):
        if fixed_ed.probabilities[key] < policy.ed_threshold:
            coherent.pop(key)
            coherence.append({"row": key[0], "column_index": key[1], "column": table.columns[key[1]],
                "original": table.rows[key[0]][key[1]], "removed_prior_repair": value,
                "fixed_v5_ed_score": float(fixed_ed.probabilities[key]), "action": "rollback_to_dirty_original"})
    coherent_rows = repaired_rows(table, coherent)
    coherent_matrix = matrix(rules, [dict(zip(table.columns, row)) for row in coherent_rows])
    if (((parent_matrix == 0) & (coherent_matrix == 1) & multi).any()
            or ((parent_matrix != -1) & (coherent_matrix == -1)).any()):
        raise CoherenceConflict("Removing an ED-incompatible prior repair introduces a parent-relative multi/runtime failure")
    index = DirtyIndex(table, artifact.get("dependencies", []), discover=True)
    traces, jobs, seen = [], [], set()
    for cell in evidence["cells"]:
        i, j = cell["row"], cell["column_index"]
        if type(i) is not int or type(j) is not int or not 0 <= i < table.shape[0] or not 0 <= j < table.shape[1] or (i, j) in seen:
            raise ValueError("Invalid/duplicate repair coordinate")
        seen.add((i, j))
        if cell["column"] != table.columns[j] or cell["original"] != table.rows[i][j]:
            raise ValueError("Stale repair original/column")
        eligible = bool(cell["eligible"])
        considered = bool(eligible and fixed_ed.probabilities[i, j] >= policy.ed_threshold)
        if not eligible and cell["selection"] is not None:
            raise ValueError("Unsupported cell has fabricated candidate scores")
        trace = {**cell, "fixed_ed_score": float(fixed_ed.probabilities[i, j]),
                 "considered_for_new_repair": considered, "prior_repair": prior.repairs.get((i, j)),
                 "coherent_parent_repair": coherent.get((i, j)), "new_choice": None,
                 "original_tuple": records[i], "source_prior_tuple": dict(zip(table.columns, parent_rows[i])),
                 "coherent_parent_tuple": dict(zip(table.columns, coherent_rows[i])), "candidate_checks": []}
        traces.append(trace)
        if not considered:
            continue
        for candidate in cell["selection"]["top3"]:
            value = candidate["value"]
            dirty_after = list(table.rows[i]); dirty_after[j] = value
            parent_after = list(coherent_rows[i]); parent_after[j] = value
            check = {**candidate, "representation_gate": representation_gate(cell["original"], value),
                     "dirty_counterfactual_tuple": dict(zip(table.columns, dirty_after)),
                     "parent_counterfactual_tuple": dict(zip(table.columns, parent_after))}
            trace["candidate_checks"].append(check)
            jobs.append((trace, check, dirty_after, parent_after))
    dirty_after_matrix = matrix(rules, [dict(zip(table.columns, after)) for _, _, after, _ in jobs])
    parent_after_matrix = matrix(rules, [dict(zip(table.columns, after)) for _, _, _, after in jobs])
    repairs = dict(coherent)
    for q, (trace, check, dirty_after, _) in enumerate(jobs):
        i, j = trace["row"], trace["column_index"]
        delta = frozen_cf.counterfactual_dimensions(index, i, j, check["value"], dirty_after,
                                                    original_matrix[i], dirty_after_matrix[q], rules)
        confidence = frozen_cf._confidence(check["mean_weight"], delta, policy)
        introduced = (coherent_matrix[i] == 0) & (parent_after_matrix[q] == 1) & multi
        blockers = list(trace["selection"]["top_gate_blockers"])
        if check["rank"] != 1: blockers.append("not_top_no_lower_rank_fallback")
        blockers.extend(check["representation_gate"]["blockers"])
        if confidence < policy.repair_confidence: blockers.append("counterfactual_confidence_below_threshold")
        if introduced.any(): blockers.append("new_parent_multi_column_violation")
        if (parent_after_matrix[q] == -1).any(): blockers.append("parent_counterfactual_runtime_failure")
        if (dirty_after_matrix[q] == -1).any(): blockers.append("dirty_counterfactual_runtime_failure")
        check.update(delta_dimensions=frozen_cf.DIMENSIONS, delta_features=delta,
                     weighted_delta=float(np.dot(policy.delta_weights, delta)), correction_confidence=confidence,
                     parent_rule_outcomes={r["id"]: int(v) for r, v in zip(rules, coherent_matrix[i])},
                     candidate_parent_rule_outcomes={r["id"]: int(v) for r, v in zip(rules, parent_after_matrix[q])},
                     dirty_rule_outcomes={r["id"]: int(v) for r, v in zip(rules, original_matrix[i])},
                     candidate_dirty_rule_outcomes={r["id"]: int(v) for r, v in zip(rules, dirty_after_matrix[q])},
                     blockers=sorted(set(blockers)), passed=not blockers)
        if not blockers:
            repairs[i, j] = check["value"]
            trace["new_choice"] = check["value"]
    changed_rows = sorted({i for (i, j), value in repairs.items() if value != coherent.get((i, j))})
    proposed_rows = repaired_rows(table, repairs)
    proposed_matrix = matrix(rules, [dict(zip(table.columns, proposed_rows[i])) for i in changed_rows])
    joint, rolled = [], set()
    for q, i in enumerate(changed_rows):
        new_multi = (coherent_matrix[i] == 0) & (proposed_matrix[q] == 1) & multi
        runtime = bool((proposed_matrix[q] == -1).any())
        rollback = bool(new_multi.any() or runtime)
        if rollback:
            rolled.add(i)
            for key in [key for key in repairs if key[0] == i]: repairs.pop(key)
            repairs.update({key: value for key, value in coherent.items() if key[0] == i})
        joint.append({"row": i, "coherent_parent_tuple": dict(zip(table.columns, coherent_rows[i])),
                      "proposed_tuple": dict(zip(table.columns, proposed_rows[i])),
                      "new_multi_rule_ids": [r["id"] for r, bit in zip(rules, new_multi) if bit],
                      "runtime_failure": runtime, "rollback_to_coherent_prior": rollback})
    final_rows = repaired_rows(table, repairs)
    final_matrix = matrix(rules, [dict(zip(table.columns, row)) for row in final_rows])
    for before in (parent_matrix, coherent_matrix):
        if (((before == 0) & (final_matrix == 1) & multi).any()
                or ((before != -1) & (final_matrix == -1)).any()):
            raise AssertionError("Final parent-relative guard failed")
    if any(fixed_ed.probabilities[key] < policy.ed_threshold for key in repairs):
        raise AssertionError("Final repair lies outside unchanged ED")
    for trace in traces:
        key = trace["row"], trace["column_index"]
        trace.update(final_repair=repairs.get(key), joint_rollback=key[0] in rolled,
                     final_tuple=dict(zip(table.columns, final_rows[key[0]])))
    prediction = Prediction(fixed_ed.probabilities.copy(), repairs, {"system": VERSION,
        "dirty_sha256": table.fingerprint, "policy": asdict(policy), "ed_bitwise_unchanged": True,
        "conditional_hypotheses_only": True, "no_external_catalog_or_snapshot_evidence": True,
        "fully_verified_main_method": False, "new_llm_calls": 0, "refit_performed": False,
        "probability_meaning": "Exact frozen v5 mean ED scores; no calibration claim"}).validate(table.shape)
    return prediction, {"trace": traces, "coherence_rollbacks": coherence, "joint_guard": joint,
        "accepted_rules": rules, "rule_validation": validation, "original_rule_matrix": original_matrix.tolist(),
        "source_parent_rule_matrix": parent_matrix.tolist(), "coherent_parent_rule_matrix": coherent_matrix.tolist(),
        "final_rule_matrix": final_matrix.tolist(), "ed_bitwise_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experimental/latent_consensus_mean_repairs_v6"))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("experimental").resolve()):
        raise ValueError("Output must remain under experimental/")
    if sha(V5 / "final_manifest.json") != EXPECTED_V5_MANIFEST or sha(V5 / "prediction.json") != EXPECTED_ED or sha(RULES) != EXPECTED_RULES:
        raise ValueError("Frozen v5/rules changed")
    closure = json.loads((V5 / "final_manifest.json").read_text())["artifacts"]
    for name, wanted in closure.items():
        if sha(name) != wanted: raise ValueError("Frozen v5 artifact changed")
    prior_path = V4 / "frozen_inputs/loop_prior.json"
    if sha(prior_path) != EXPECTED_PRIOR: raise ValueError("Frozen source prior changed")
    sources = [Path(__file__), Path(__file__).with_name("latent_consensus_mean_repairs_v6_protocol.md"),
        Path("tests/test_latent_consensus_repairs_v6.py"), Path(frozen_cf.__file__),
        Path("zeroclean/data.py"), Path("zeroclean/types.py"), Path("zeroclean/statistics.py"), Path("zeroclean/sandbox.py")]
    inputs = [prior_path, RULES, V5 / "prediction.json", V5 / "final_manifest.json", V5 / "mean_evidence.json.gz",
        V5 / "claim_contexts.json.gz", V4 / "posterior_contexts.json.gz", V4 / "frozen_inputs/dirty_table.json.gz", *sources]
    hashes = {str(p): sha(p) for p in inputs}
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output / "frozen_inputs"; frozen.mkdir()
    copies, paths = {}, {}
    for i, source in enumerate(inputs):
        destination = frozen / f"input_{i}_{source.name}"
        shutil.copyfile(source, destination); paths[str(source)] = destination; copies[str(destination)] = hashes[str(source)]
        if sha(destination) != hashes[str(source)]: raise ValueError("Input changed during freezing")
    policy = RepairPolicy()
    write_new(args.output / "pre_run_manifest.json", {"experiment": VERSION, "policy": asdict(policy),
        "input_hashes": hashes, "frozen_input_hashes": copies, "fixed_ed_sha256": EXPECTED_ED,
        "source_prior_sha256": EXPECTED_PRIOR, "new_llm_calls": 0, "refit_performed": False,
        "accuracy_evaluator_executed": False, "independent_verification_status": "not_run"})
    dirty = read_gzip(paths[str(V4 / "frozen_inputs/dirty_table.json.gz")])
    table = DirtyTable(dirty["name"], tuple(dirty["columns"]), tuple(tuple(r) for r in dirty["rows"]), dirty["fingerprint"])
    prior = Prediction.load(paths[str(prior_path)]).validate(table.shape)
    fixed_ed = Prediction.load(paths[str(V5 / "prediction.json")]).validate(table.shape)
    artifact = json.loads(paths[str(RULES)].read_text())
    mean = read_gzip(paths[str(V5 / "mean_evidence.json.gz")])
    contexts = read_gzip(paths[str(V4 / "posterior_contexts.json.gz")])
    claim_contexts = read_gzip(paths[str(V5 / "claim_contexts.json.gz")])
    choices, cache = [], {}
    for cell in mean["cells"]:
        point = cell["mean_evidence"]
        selection = None
        if point["eligible"]:
            nodes = {mode: contexts[point["mode_evidence"][mode]["context_id"]] for mode in MODES}
            selection = rank_observed_alternatives(cell["original"], nodes, policy)
            if not np.isclose(selection["original_mean_weight"], 1-point["mean_error_weight"], rtol=0, atol=1e-14):
                raise ValueError("Candidate means differ from frozen v5 averaging")
        choices.append({"row": cell["row"], "column": cell["column"], "column_index": cell["column_index"],
            "original": cell["original"], "claim_context_id": cell["claim_context_id"],
            "eligible": point["eligible"], "selection": selection})
    evidence = {"experiment": VERSION, "dirty_sha256": table.fingerprint, "cells": choices,
        "frozen_semantic_contracts": mean["frozen_semantic_contracts"], "conditional_hypotheses_only": True,
        "probability_meaning": "Mean conditional weights, not calibrated correctness", "input_hashes": hashes}
    write_gzip_new(args.output / "candidate_evidence.json.gz", evidence)
    try:
        prediction, audit = apply_repair_policy(table, prior, fixed_ed, artifact, evidence, policy)
    except CoherenceConflict as exc:
        write_new(args.output / "coherence_failure.json", {"status": "no valid prediction",
            "reason": str(exc), "policy_unchanged": True, "fixed_ed_sha256": EXPECTED_ED})
        raise
    prediction.metadata.update(input_hashes=hashes, fixed_ed_sha256=EXPECTED_ED, source_prior_sha256=EXPECTED_PRIOR,
        frozen_semantic_contracts=mean["frozen_semantic_contracts"], independent_verification_status="not_run")
    write_new(args.output / "prediction.json", {"probabilities": prediction.probabilities.tolist(),
        "repairs": [{"row": i, "column": j, "value": value} for (i, j), value in sorted(prediction.repairs.items())],
        "metadata": prediction.metadata})
    audit.update(prediction_sha256=sha(args.output / "prediction.json"), input_hashes=hashes)
    write_gzip_new(args.output / "audit.json.gz", audit)
    removed = {(c["row"], c["column_index"]): c for c in audit["coherence_rollbacks"]}
    bundle_cells = []
    for trace in audit["trace"]:
        key = trace["row"], trace["column_index"]
        if trace["considered_for_new_repair"] or key in removed:
            bundle_cells.append({"row": key[0], "column": trace["column"], "column_index": key[1],
                "original": trace["original"], "original_tuple": trace["original_tuple"],
                "raw_dirty_claim_context": claim_contexts[trace["claim_context_id"]],
                "repair_evidence": trace, "coherence_rollback": removed.get(key),
                "independent_verification_status": "not_run"})
    bundle = {"experiment": VERSION, "dirty_sha256": table.fingerprint, "cells": bundle_cells,
        "conditional_hypotheses_only": True, "frozen_semantic_contracts": mean["frozen_semantic_contracts"],
        "parent_prediction_sha256": sha(args.output / "prediction.json"), "fixed_ed_sha256": EXPECTED_ED,
        "input_hashes": hashes, "independent_verification_status": "not_run"}
    bundle["request_hash"] = digest(bundle)
    write_new(args.output / "verification_bundle.json", bundle)
    write_new(args.output / "counts.json", {"interpretation": "Inference counts only; no truth labels/accuracy evaluation",
        "considered_cells": sum(t["considered_for_new_repair"] for t in audit["trace"]),
        "source_prior_repairs": len(prior.repairs), "coherence_removed_prior_repairs": len(audit["coherence_rollbacks"]),
        "individually_admitted_top_choices": sum(t["new_choice"] is not None for t in audit["trace"]),
        "joint_rollback_rows": sum(r["rollback_to_coherent_prior"] for r in audit["joint_guard"]),
        "final_repairs": len(prediction.repairs), "fixed_ed_unchanged": np.array_equal(prediction.probabilities, fixed_ed.probabilities)})
    for name, wanted in hashes.items():
        if sha(name) != wanted: raise ValueError("Frozen input changed during execution")
    names = ["pre_run_manifest.json", "candidate_evidence.json.gz", "prediction.json", "audit.json.gz", "verification_bundle.json", "counts.json"]
    write_new(args.output / "output_manifest.json", {"artifacts": {str(args.output / name): sha(args.output / name) for name in names},
        "accuracy_evaluator_executed": False, "independent_verification_status": "not_run"})
    print(str(args.output / "prediction.json")+" "+sha(args.output / "prediction.json"))


if __name__ == "__main__":
    main()
