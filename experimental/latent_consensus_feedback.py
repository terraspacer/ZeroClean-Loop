"""Statistical feedback ablation on immutable v3 evidence; no fitting or LLM calls."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from zeroclean.data import DirtyTable, load_dirty
from zeroclean.sandbox import evaluate_rules, validate_rules
from zeroclean.statistics import DirtyIndex
from zeroclean.types import Prediction, digest
from experimental import latent_consensus as preparation
from experimental import latent_consensus_v2 as categorical

VERSION = "latent-consensus-feedback-v3-statistical-ablation"
PRIOR = Path("artifacts/closed_loop_v2/flights/4e0f4d0f82ada5f10a82bd9fb57a613dd8bf28110ef7ca66633ea38717c31247/prediction.json")
RULES = Path("artifacts/llm/flights/pass2.json")
PREVIEW = Path("experimental/latent_consensus_flights_preview_v3")
EXPECTED_PRIOR_SHA256 = "312f76d5be26f50b9ddf80190d1339b8297b2cda620be5c27c2bdbe6abae42c8"
EXPECTED_RULES_SHA256 = "e031188e337da807cb7054d5a54866b5c69f0645d4becdd0392b0f85433d18ea"
DIMENSIONS = ["rule_satisfaction", "fd_support", "nmi_vicinity", "domain_support"]


@dataclass(frozen=True)
class FeedbackPolicy:
    ed_threshold: float = .5
    repair_confidence: float = .8
    winner_weight: float = .98
    winner_margin: float = .20
    leave_group_out_weight: float = .95
    delta_weights: tuple[float, ...] = (2.2, 1.8, .65, .35)
    logit_epsilon: float = 1e-6
    runtime_error_rate: float = .05
    ec_failure_fallback: str = "exact_prior_repair_state"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, body: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def write_gzip_new(path: Path, body: Any) -> None:
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def original_weight_trace(observations, models, original, candidate):
    candidates = tuple(sorted(set(observations.values())))
    if original not in candidates or candidate not in candidates:
        raise ValueError("Original/candidate absent from active frozen vocabulary")
    original_weights, winner_weights = [], []
    for model in models:
        if not model["converged"]:
            raise ValueError("Unconverged selected frozen model")
        _, posterior = categorical._posterior(observations, model["skill"], model["unseen_prior"], candidates)
        if int(posterior.argmax()) != candidates.index(candidate):
            raise ValueError("Frozen winner is not shared by selected fits")
        original_weights.append(float(posterior[candidates.index(original)]))
        winner_weights.append(float(posterior[candidates.index(candidate)]))
    return {"original_weights": original_weights, "winner_weights": winner_weights,
            "original_weight_interval": [min(original_weights), max(original_weights)],
            "candidate_weight_lower": min(winner_weights),
            "grouped_claims": observations, "candidate_strings": list(candidates)}


def collect_frozen_evidence(table: DirtyTable, preview: Path, policy: FeedbackPolicy):
    """Read only named v3 files; reconstruct weights without calling a fitter."""
    manifest = json.loads((preview / "pre_fit_manifest.json").read_text())
    summary = json.loads((preview / "summary.json").read_text())
    checksums = json.loads((preview / "artifact_manifest.json").read_text())["artifacts"]
    if manifest["dirty_fingerprint"] != table.fingerprint or summary["manifest"] != manifest:
        raise ValueError("Frozen preview dirty fingerprint/manifest mismatch")
    helper_paths = {"v1_helpers_frozen.py": Path(preparation.__file__),
                    "v2_helpers_frozen.py": Path(categorical.__file__)}
    for filename, path in helper_paths.items():
        if sha(path) != manifest["frozen_sha256"][filename]:
            raise ValueError("Posterior reconstruction helper differs from frozen v3")
    accessed = {str(preview / name): sha(preview / name)
                for name in ("pre_fit_manifest.json", "summary.json", "artifact_manifest.json")}
    cells = []
    contracts = []
    for attribute in summary["attributes"]:
        column = attribute["attribute"]
        j = table.columns.index(column)
        roles = categorical.SkillContract(tuple(manifest["entity_columns"]), manifest["source_column"],
                                           column, tuple(manifest["context_columns"]))
        claims, _ = preparation._extract(table, roles)
        mode_decisions, mode_contexts = {}, {}
        for mode in ("independent_names", "cluster_near_agreement"):
            filename = attribute["source_assumptions"][mode]["artifact"]
            if Path(filename).name != filename:
                raise ValueError("Preview artifact path must be local")
            path = preview / filename
            if sha(path) != checksums[filename]:
                raise ValueError("Frozen preview file hash mismatch")
            accessed[str(path)] = sha(path)
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                result = json.load(stream)
            if result["dirty_fingerprint"] != table.fingerprint or result["code_sha256"] != manifest["frozen_sha256"]["model_frozen.py"]:
                raise ValueError("Frozen model provenance mismatch")
            if result["parameters"]["source_assumption"] != mode:
                raise ValueError("Frozen source assumption mismatch")
            if (result["parameters"]["min_conditional_probability"], result["parameters"]["min_margin"],
                result["parameters"]["min_leave_group_out_probability"]) != (
                    policy.winner_weight, policy.winner_margin, policy.leave_group_out_weight):
                raise ValueError("Frozen gate thresholds differ from feedback policy")
            contracts.append({"attribute": column, "source_assumption": mode, "contract": result["contract"]})
            decisions = {}
            for d in result["decisions"]:
                row = d["row"]
                if type(row) is not int or not 0 <= row < table.shape[0] or d["column"] != column or d["original"] != table.rows[row][j]:
                    raise ValueError("Stale frozen evidence coordinate/string")
                if row in decisions:
                    raise ValueError("Duplicate frozen evidence coordinate")
                decisions[row] = d
            if len(decisions) != table.shape[0]:
                raise ValueError("Frozen evidence row coverage mismatch")
            contexts = {}
            for fold in result["folds"]:
                keys = {tuple(d["entity"]) for d in decisions.values() if d.get("fold") == fold["fold"]}
                grouped, _ = preparation._collapse({k: claims[k] for k in keys if k in claims}, fold["source_group_mapping"])
                for key, obs in grouped.items():
                    contexts[key] = ({s: v for s, v in obs.items() if s in fold["training_groups"]}, fold)
            mode_decisions[mode], mode_contexts[mode] = decisions, contexts
        for row in range(table.shape[0]):
            a, b = (mode_decisions[m][row] for m in ("independent_names", "cluster_near_agreement"))
            if a["status"] not in {"consistent", "propose"} or b["status"] not in {"consistent", "propose"} or a["candidate"] != b["candidate"]:
                continue
            original, candidate = table.rows[row][j], a["candidate"]
            traces = {}
            for mode, decision in (("independent_names", a), ("cluster_near_agreement", b)):
                if decision["reasons"] or decision["conditional_weight_min"] < policy.winner_weight or decision["conditional_margin_min"] < policy.winner_margin or decision["leave_group_out_weight_min"] < policy.leave_group_out_weight:
                    raise ValueError("Supported frozen evidence fails its stated gate")
                observations, fold = mode_contexts[mode][tuple(decision["entity"])]
                if fold["blockers"]:
                    raise ValueError("Supported evidence belongs to a blocked frozen fold")
                trace = original_weight_trace(observations, fold["models"], original, candidate)
                if abs(trace["candidate_weight_lower"] - decision["conditional_weight_min"]) > 1e-10:
                    raise ValueError("Reconstructed winner weight differs from frozen evidence")
                trace.update(fold=fold["fold"], active_groups=decision["active_source_groups"],
                             frozen_winner_margin=decision["conditional_margin_min"],
                             frozen_leave_group_out_weight=decision["leave_group_out_weight_min"])
                traces[mode] = trace
            cells.append({"row": row, "column": column, "column_index": j, "original": original,
                          "candidate": candidate, "entity": a["entity"], "source": a["source"],
                          "original_weight_interval": [min(t["original_weight_interval"][0] for t in traces.values()),
                                                       max(t["original_weight_interval"][1] for t in traces.values())],
                          "candidate_weight_lower": min(t["candidate_weight_lower"] for t in traces.values()),
                          "source_assumptions": traces})
    return {"experiment": VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
            "probability_meaning": "conditional model weights, not calibrated correctness",
            "frozen_semantic_contracts": contracts, "input_hashes": accessed,
            "cells": sorted(cells, key=lambda d: (d["row"], d["column_index"]))}


def counterfactual_dimensions(index, i, j, value, after, original_rules, changed_rules, rules):
    target = [k for k, rule in enumerate(rules) if rule["target"] == index.columns[j]]
    before_bad = int((original_rules[target] == 1).sum()) if target else 0
    after_bad = int((changed_rules[target] == 1).sum()) if target else 0
    original = index.data[i, j]
    return [float((before_bad - after_bad) / max(len(target), 1)),
            float(index.conditional(i, j, value, after)[0] - index.conditional(i, j, original)[0]),
            float(math.tanh((index.vicinity(i, j, value, after) - index.vicinity(i, j, original)) / 4)),
            float(math.tanh((index.domain_support(i, j, value) - index.domain_support(i, j, original)) / 4))]


def _rule_matrix(rules, records):
    return np.asarray(evaluate_rules(rules, records), dtype=np.int8).reshape(len(records), len(rules))


def _confidence(weight, delta, policy):
    q = max(policy.logit_epsilon, min(1 - policy.logit_epsilon, weight))
    log_odds = math.log(q / (1 - q)) + float(np.dot(policy.delta_weights, delta))
    return 1 / (1 + math.exp(-max(-30, min(30, log_odds))))


def apply_feedback(table: DirtyTable, prior: Prediction, artifact: dict, evidence: dict,
                   policy: FeedbackPolicy = FeedbackPolicy()):
    prior.validate(table.shape)
    if evidence["dirty_sha256"] != table.fingerprint:
        raise ValueError("Stale statistical evidence")
    if artifact.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Stale pass2 artifact")
    records = table.records()
    accepted, validation = validate_rules(artifact.get("rules", []), records, policy.runtime_error_rate)
    rules = [r for r in accepted if r["target"] in table.columns]
    original_matrix = _rule_matrix(rules, records)
    multi = np.array([r["kind"] == "multi" for r in rules], dtype=bool)
    index = DirtyIndex(table, artifact.get("dependencies", []), discover=True)
    prior_rows = [list(row) for row in table.rows]
    for (i, j), value in prior.repairs.items():
        prior_rows[i][j] = value
    prior_matrix = _rule_matrix(rules, [dict(zip(table.columns, row)) for row in prior_rows])
    probability = prior.probabilities.copy()
    repairs = dict(prior.repairs)
    traces, jobs, seen = [], [], set()
    for cell in evidence["cells"]:
        i, j = cell["row"], cell["column_index"]
        if type(i) is not int or type(j) is not int or not 0 <= i < table.shape[0] or not 0 <= j < table.shape[1] or (i, j) in seen:
            raise ValueError("Invalid/duplicate feedback coordinate")
        seen.add((i, j))
        if cell["column"] != table.columns[j] or cell["original"] != table.rows[i][j] or not isinstance(cell["candidate"], str):
            raise ValueError("Stale feedback original/column")
        if set(cell["source_assumptions"]) != {"independent_names", "cluster_near_agreement"}:
            raise ValueError("Feedback requires both source assumptions")
        low, high = map(float, cell["original_weight_interval"])
        winner = float(cell["candidate_weight_lower"])
        if not 0 <= low <= high <= 1 or not policy.winner_weight <= winner <= 1:
            raise ValueError("Invalid conditional weight interval")
        original_supported = cell["candidate"] == cell["original"]
        if (original_supported and low < policy.winner_weight) or (not original_supported and high > 1 - policy.winner_weight + 1e-10):
            raise ValueError("Original-weight interval conflicts with supported winner")
        error = 1 - low if original_supported else 1 - high
        probability[i, j] = error
        after = list(table.rows[i])
        after[j] = cell["candidate"]
        trace = {"row": i, "column": table.columns[j], "column_index": j,
                 "original": cell["original"], "candidate": cell["candidate"],
                 "prior_probability": float(prior.probabilities[i, j]), "feedback_probability": error,
                 "original_weight_interval": [low, high], "candidate_weight_lower": winner,
                 "ed_interval_endpoint": "upper_error" if original_supported else "lower_error",
                 "prior_ed_positive": bool(prior.probabilities[i, j] >= policy.ed_threshold),
                 "feedback_ed_positive": bool(error >= policy.ed_threshold),
                 "prior_repair": prior.repairs.get((i, j)),
                 "requested_repair": None if original_supported else cell["candidate"],
                 "original_tuple": records[i], "prior_tuple": dict(zip(table.columns, prior_rows[i])),
                 "individual_candidate_tuple": dict(zip(table.columns, after)),
                 "statistical_evidence": cell, "joint_rollback": False}
        traces.append(trace)
        if cell["candidate"] == cell["original"]:
            jobs.append((trace, after, original_matrix[i]))
        else:
            jobs.append((trace, after, None))
    to_evaluate = [(trace, after) for trace, after, matrix in jobs if matrix is None]
    changed = _rule_matrix(rules, [dict(zip(table.columns, after)) for _, after in to_evaluate]) if to_evaluate else np.empty((0, len(rules)), dtype=np.int8)
    cursor = 0
    for trace, after, matrix in jobs:
        i, j = trace["row"], trace["column_index"]
        if matrix is None:
            matrix = changed[cursor]
            cursor += 1
        delta = counterfactual_dimensions(index, i, j, trace["candidate"], after, original_matrix[i], matrix, rules)
        confidence = _confidence(trace["candidate_weight_lower"], delta, policy)
        introduced = ((original_matrix[i] == 0) & (matrix == 1) & multi)
        reasons = []
        if introduced.any():
            reasons.append("new_multi_column_violation")
        if (matrix == -1).any():
            reasons.append("counterfactual_runtime_failure")
        if confidence < policy.repair_confidence:
            reasons.append("counterfactual_confidence_below_threshold")
        trace.update(delta_features=delta, delta_dimensions=DIMENSIONS,
                     weighted_delta=float(np.dot(policy.delta_weights, delta)),
                     correction_confidence=confidence,
                     original_rule_outcomes={r["id"]: int(v) for r, v in zip(rules, original_matrix[i])},
                     candidate_rule_outcomes={r["id"]: int(v) for r, v in zip(rules, matrix)},
                     individual_gate_blockers=reasons, individual_gate_passed=not reasons)
        if not reasons:
            if trace["requested_repair"] is None:
                repairs.pop((i, j), None)
            else:
                repairs[i, j] = trace["requested_repair"]
    feedback_keys = {(t["row"], t["column_index"]) for t in traces}
    changed_rows = sorted({i for i, j in feedback_keys if repairs.get((i, j)) != prior.repairs.get((i, j))})
    proposed_rows = [list(row) for row in prior_rows]
    for i, j in feedback_keys:
        proposed_rows[i][j] = repairs.get((i, j), table.rows[i][j])
    proposed_matrix = _rule_matrix(rules, [dict(zip(table.columns, proposed_rows[i])) for i in changed_rows]) if changed_rows else np.empty((0, len(rules)), dtype=np.int8)
    joint = []
    rolled_back = set()
    for q, i in enumerate(changed_rows):
        new_multi = ((original_matrix[i] == 0) & (proposed_matrix[q] == 1) & multi)
        runtime = bool((proposed_matrix[q] == -1).any())
        rollback = bool(new_multi.any() or runtime)
        if rollback:
            rolled_back.add(i)
            for key in feedback_keys:
                if key[0] == i:
                    if key in prior.repairs:
                        repairs[key] = prior.repairs[key]
                    else:
                        repairs.pop(key, None)
        joint.append({"row": i, "original_tuple": records[i],
                      "prior_tuple": dict(zip(table.columns, prior_rows[i])),
                      "proposed_tuple": dict(zip(table.columns, proposed_rows[i])),
                      "new_multi_rule_ids": [r["id"] for r, bit in zip(rules, new_multi) if bit],
                      "runtime_failure": runtime, "rollback_to_prior": rollback,
                      "prior_runtime_failure": bool((prior_matrix[i] == -1).any()),
                      "prior_new_multi_rule_ids": [r["id"] for r, bit in zip(rules, (original_matrix[i] == 0) & (prior_matrix[i] == 1) & multi) if bit]})
    final_rows = [list(row) for row in table.rows]
    for (i, j), value in repairs.items():
        final_rows[i][j] = value
    final_matrix = _rule_matrix(rules, [dict(zip(table.columns, final_rows[i])) for i in changed_rows]) if changed_rows else np.empty((0, len(rules)), dtype=np.int8)
    for q, i in enumerate(changed_rows):
        if (((prior_matrix[i] == 0) & (final_matrix[q] == 1) & multi).any()
                or ((prior_matrix[i] != -1) & (final_matrix[q] == -1)).any()):
            raise AssertionError("Joint rollback failed to preserve the prior guard state")
    bundle_cells = []
    for trace in traces:
        i, j = trace["row"], trace["column_index"]
        trace.update(final_repair=repairs.get((i, j)), joint_rollback=i in rolled_back,
                     final_tuple=dict(zip(table.columns, final_rows[i])),
                     probability_changed=bool(probability[i, j] != prior.probabilities[i, j]),
                     ed_decision_changed=trace["prior_ed_positive"] != trace["feedback_ed_positive"],
                     repair_changed=repairs.get((i, j)) != prior.repairs.get((i, j)))
        if trace["probability_changed"] or trace["repair_changed"]:
            candidates = []
            for value in (trace["candidate"], trace["prior_repair"]):
                if value is not None and value != trace["original"] and value not in {c["value"] for c in candidates}:
                    candidates.append({"value": value, "score": trace["candidate_weight_lower"] if value == trace["candidate"] else None,
                                       "origin": "frozen_statistical_consensus" if value == trace["candidate"] else "frozen_prior_repair"})
            bundle_cells.append({"row": i, "column": table.columns[j], "column_index": j,
                                 "original": trace["original"], "tuple": records[i],
                                 "prior_inference_tuple": trace["prior_tuple"], "tentative_tuple": trace["final_tuple"],
                                 "candidates": candidates, "base_probability": trace["prior_probability"],
                                 "statistical_error_weight": trace["feedback_probability"],
                                 "original_valid_weight_interval": trace["original_weight_interval"],
                                 "evidence": trace, "independent_verification_status": "not_run"})
    metadata = {"system": VERSION, "dirty_sha256": table.fingerprint,
                "statistical_feedback_ablation": True, "fully_verified_main_method": False,
                "independent_verification_status": "not_run", "new_llm_calls": 0,
                "ground_truth_access": False, "policy": asdict(policy),
                "probability_meaning": "conditional error-weight endpoints, not calibrated correctness",
                "frozen_semantic_contracts": evidence.get("frozen_semantic_contracts", []),
                "input_hashes": evidence.get("input_hashes", {})}
    prediction = Prediction(probability, repairs, metadata).validate(table.shape)
    audit = {"experiment": VERSION, "dirty_sha256": table.fingerprint,
             "rule_validation": validation, "accepted_rules": rules, "delta_dimensions": DIMENSIONS,
             "dirty_rule_matrix": original_matrix.tolist(), "prior_rule_matrix": prior_matrix.tolist(),
             "trace": traces, "joint_guard": joint,
             "guard_assertion": "No new multi-column violation or runtime failure relative to prior repaired tuples after rollback",
             "evidence_sha256": digest(evidence), "refit_performed": False,
             "independent_verification_status": "not_run"}
    bundle = {"experiment": VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
              "model": "gpt-6.1-sol", "reasoning_effort": "xhigh",
              "purpose": "Prepared for later independent verification; never submitted in this ablation",
              "independent_verification_status": "not_run", "rules": rules,
              "delta_dimensions": DIMENSIONS, "config": asdict(policy),
              "input_hashes": evidence.get("input_hashes", {}),
              "frozen_semantic_contracts": evidence.get("frozen_semantic_contracts", []),
              "cells": bundle_cells}
    bundle["request_hash"] = digest(bundle)
    return prediction, audit, bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experimental/latent_consensus_feedback_v3"))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("experimental").resolve()):
        raise ValueError("Experimental output must remain under experimental/")
    policy = FeedbackPolicy()
    table = load_dirty("flights")
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or sha(RULES) != EXPECTED_RULES_SHA256:
        raise ValueError("Authorized frozen prediction/rules changed since initial inspection")
    prior = Prediction.load(PRIOR).validate(table.shape)
    artifact = json.loads(RULES.read_text())
    if prior.metadata.get("dirty_sha256") != table.fingerprint or artifact.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Prior/rules dirty fingerprint mismatch")
    preview_manifest = json.loads((PREVIEW / "pre_fit_manifest.json").read_text())
    preview_summary = json.loads((PREVIEW / "summary.json").read_text())
    preview_files = ["pre_fit_manifest.json", "summary.json", "artifact_manifest.json"]
    preview_files.extend(meta["artifact"] for a in preview_summary["attributes"] for meta in a["source_assumptions"].values())
    if any(Path(name).name != name for name in preview_files):
        raise ValueError("Frozen preview files must be local names")
    source_paths = [Path(__file__), Path(__file__).with_name("latent_consensus_feedback_protocol.md"),
                    Path(preparation.__file__), Path(categorical.__file__), Path("zeroclean/data.py"),
                    Path("zeroclean/types.py"), Path("zeroclean/statistics.py"), Path("zeroclean/sandbox.py")]
    input_hashes = {str(path): sha(path) for path in [PRIOR, RULES, *(PREVIEW / name for name in preview_files), *source_paths]}
    manifest = {"experiment": VERSION, "dirty_sha256": table.fingerprint,
                "dirty_loader": "zeroclean.data.load_dirty('flights')", "shape": table.shape,
                "input_hashes": input_hashes, "policy": asdict(policy),
                "semantic_hypothesis": preview_manifest["hypothesis"],
                "no_refit": True, "new_llm_verification": False, "evaluator_executed": False,
                "method_status": "Statistical feedback ablation atop frozen loop; not fully verified main method or completed target result"}
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output / "frozen_inputs"
    frozen.mkdir()
    copied = {}
    for filename, source in (("prior_prediction.json", PRIOR), ("pass2.json", RULES)):
        shutil.copyfile(source, frozen / filename)
        copied[str(frozen / filename)] = input_hashes[str(source)]
    frozen_preview = frozen / "preview"
    frozen_preview.mkdir()
    for name in preview_files:
        shutil.copyfile(PREVIEW / name, frozen_preview / name)
        copied[str(frozen_preview / name)] = input_hashes[str(PREVIEW / name)]
    for index, source in enumerate(source_paths):
        destination = frozen / f"source_{index}_{source.name}"
        shutil.copyfile(source, destination)
        copied[str(destination)] = input_hashes[str(source)]
    if any(sha(Path(path)) != wanted for path, wanted in copied.items()):
        raise ValueError("Input changed while freezing")
    write_gzip_new(frozen / "dirty_table.json.gz", {"name": table.name, "columns": table.columns,
                                                   "rows": table.rows, "fingerprint": table.fingerprint})
    manifest["frozen_input_hashes"] = copied
    manifest["frozen_dirty_strings_sha256"] = sha(frozen / "dirty_table.json.gz")
    write_new(args.output / "pre_run_manifest.json", manifest)
    prior = Prediction.load(frozen / "prior_prediction.json").validate(table.shape)
    artifact = json.loads((frozen / "pass2.json").read_text())
    evidence = collect_frozen_evidence(table, frozen_preview, policy)
    evidence["input_hashes"].update(input_hashes)
    evidence["input_hashes"][str(args.output / "pre_run_manifest.json")] = sha(args.output / "pre_run_manifest.json")
    write_gzip_new(args.output / "statistical_evidence.json.gz", evidence)
    prediction, audit, bundle = apply_feedback(table, prior, artifact, evidence, policy)
    prediction.metadata.update(parent_prediction_path=str(PRIOR), parent_prediction_sha256=sha(PRIOR),
                               pre_run_manifest_sha256=sha(args.output / "pre_run_manifest.json"),
                               verification_bundle="verification_bundle.json")
    write_new(args.output / "prediction.json", {"probabilities": prediction.probabilities.tolist(),
              "repairs": [{"row": i, "column": j, "value": value} for (i, j), value in sorted(prediction.repairs.items())],
              "metadata": prediction.metadata})
    audit["variant_prediction_sha256"] = sha(args.output / "prediction.json")
    bundle.pop("request_hash")
    bundle["parent_variant_prediction_path"] = str(args.output / "prediction.json")
    bundle["parent_variant_prediction_sha256"] = sha(args.output / "prediction.json")
    bundle["request_hash"] = digest(bundle)
    write_gzip_new(args.output / "audit.json.gz", audit)
    write_new(args.output / "verification_bundle.json", bundle)
    write_new(args.output / "output_manifest.json", {"experiment": VERSION,
              "outputs": {str(path): sha(path) for path in (args.output / "pre_run_manifest.json", args.output / "prediction.json",
                           args.output / "statistical_evidence.json.gz", args.output / "audit.json.gz", args.output / "verification_bundle.json")},
              "method_status": manifest["method_status"], "evaluator_executed": False, "independent_verification_status": "not_run"})
    for path in (args.output / "prediction.json", args.output / "output_manifest.json", args.output / "verification_bundle.json"):
        print(str(path) + " " + sha(path), flush=True)


if __name__ == "__main__":
    main()
