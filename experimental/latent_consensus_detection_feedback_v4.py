"""Detection/repair uncertainty split from frozen v3 fits; no refit or LLM calls."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from zeroclean.data import DirtyTable, load_dirty
from zeroclean.types import Prediction, digest
from experimental import latent_consensus as preparation
from experimental import latent_consensus_v2 as categorical

VERSION = "latent-consensus-detection-feedback-v4-statistical-ablation"
MODES = ("independent_names", "cluster_near_agreement")
PRIOR = Path("artifacts/closed_loop_v2/flights/4e0f4d0f82ada5f10a82bd9fb57a613dd8bf28110ef7ca66633ea38717c31247/prediction.json")
PREVIEW = Path("experimental/latent_consensus_flights_preview_v3")
FIXED_EC = Path("experimental/latent_consensus_feedback_v3")
EXPECTED_PRIOR_SHA = "312f76d5be26f50b9ddf80190d1339b8297b2cda620be5c27c2bdbe6abae42c8"
EXPECTED_EC_SHA = "911b11611c136c5c1dcdf9e37d5d3ed4ffdf6fb2c8f414661c0a4efb78b68dd4"
EXPECTED_EC_MANIFEST_SHA = "cf48dc9a361626927e755c5934c243f8455af8cc47765c1561111f10793da830"


@dataclass(frozen=True)
class DetectionPolicy:
    threshold: float = .5
    min_active_groups: int = 4
    min_leave_out_groups: int = 3
    selected_fits: int = 17
    vocabulary: str = "freeze observed full-item vocabulary before leave-one-group-out"
    uncertain_action: str = "retain_exact_frozen_loop_prior"
    correction_action: str = "copy_exact_frozen_v3_feedback_repair_map"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, body: Any):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def write_gzip_new(path: Path, body: Any):
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def classify_error_interval(interval, policy: DetectionPolicy = DetectionPolicy()):
    low, high = map(float, interval)
    if not np.isfinite([low, high]).all() or not 0 <= low <= high <= 1:
        raise ValueError("Invalid conditional error interval")
    if low > policy.threshold:
        return "flag", low
    if high < policy.threshold:
        return "prune", high
    return "retain_uncertain", None


def source_eligibility(original, source, mapping, trained_groups, observations,
                       *, conflicted=False, complete_key=True,
                       policy: DetectionPolicy = DetectionPolicy()):
    reasons = []
    if not original.strip():
        reasons.append("missing_original")
    if not source.strip():
        reasons.append("missing_source")
    if not complete_key:
        reasons.append("missing_entity_or_context")
    if conflicted:
        reasons.append("conflicted_source_entity_observation")
    if len(observations) < policy.min_active_groups:
        reasons.append("insufficient_active_trained_groups")
    if len(observations) - 1 < policy.min_leave_out_groups:
        reasons.append("insufficient_leave_group_out_support")
    if source not in mapping:
        reasons.append("source_absent_frozen_training_map")
    elif mapping[source] not in trained_groups:
        reasons.append("source_group_not_trained")
    elif mapping[source] not in observations:
        reasons.append("source_group_inactive_or_conflicted")
    elif observations[mapping[source]] != original:
        reasons.append("source_group_claim_differs_from_original")
    if original not in observations.values():
        reasons.append("original_absent_frozen_observation_vocabulary")
    return reasons


def posterior_scenarios(observations, models, policy: DetectionPolicy = DetectionPolicy()):
    if len(observations) < policy.min_active_groups or len(observations) - 1 < policy.min_leave_out_groups:
        raise ValueError("Insufficient full/leave-out support")
    candidates = tuple(sorted(set(observations.values())))
    if not candidates:
        raise ValueError("Empty frozen observation vocabulary")
    scenarios = []
    for index, model in enumerate(models):
        if not model["converged"]:
            raise ValueError("Unconverged frozen selected fit")
        if any(s not in model["skill"] for s in observations):
            raise ValueError("Observed group has no frozen skill")
        for omitted in (None, *sorted(observations)):
            remaining = {s: v for s, v in observations.items() if s != omitted}
            _, weights = categorical._posterior(remaining, model["skill"], model["unseen_prior"], candidates)
            if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1):
                raise ValueError("Invalid frozen conditional weights")
            scenarios.append({"selected_model_index": index, "omitted_group": omitted,
                              "posterior_weights": weights.tolist()})
    return {"candidate_strings": list(candidates), "unobserved_category_index": len(candidates),
            "grouped_claims": observations, "scenarios": scenarios}


def interval_for_original(context, original):
    candidates = context["candidate_strings"]
    if original not in candidates:
        raise ValueError("Cannot score an original absent from the frozen vocabulary")
    index = candidates.index(original)
    weights = [s["posterior_weights"][index] for s in context["scenarios"]]
    if not weights:
        raise ValueError("No eligible frozen sensitivity scenarios")
    minimum, maximum = int(np.argmin(weights)), int(np.argmax(weights))
    return {"original_weight_interval": [weights[minimum], weights[maximum]],
            "error_weight_interval": [1 - weights[maximum], 1 - weights[minimum]],
            "original_candidate_index": index,
            "minimum_original_scenario_index": minimum,
            "maximum_original_scenario_index": maximum}


def collect_detection_evidence(table: DirtyTable, preview: Path,
                               policy: DetectionPolicy = DetectionPolicy()):
    manifest = json.loads((preview / "pre_fit_manifest.json").read_text())
    summary = json.loads((preview / "summary.json").read_text())
    checksums = json.loads((preview / "artifact_manifest.json").read_text())["artifacts"]
    if manifest["dirty_fingerprint"] != table.fingerprint or summary["manifest"] != manifest:
        raise ValueError("Frozen preview manifest/fingerprint mismatch")
    if manifest["parameters"]["min_item_sources"] != policy.min_active_groups:
        raise ValueError("Frozen source support gate changed")
    for filename, path in (("v1_helpers_frozen.py", Path(preparation.__file__)),
                           ("v2_helpers_frozen.py", Path(categorical.__file__))):
        if sha(path) != manifest["frozen_sha256"][filename]:
            raise ValueError("Posterior helper differs from frozen v3")
    contexts, cells, contracts = {}, [], []
    for attribute in summary["attributes"]:
        column = attribute["attribute"]
        j = table.columns.index(column)
        roles = categorical.SkillContract(tuple(manifest["entity_columns"]), manifest["source_column"],
                                           column, tuple(manifest["context_columns"]))
        key_indices, source_index, _ = preparation._indices(table, roles)
        claims, extraction = preparation._extract(table, roles)
        conflicts = {r for conflict in extraction["source_entity_conflicts"] for r in conflict["rows"]}
        mode_contexts = {}
        for mode in MODES:
            filename = attribute["source_assumptions"][mode]["artifact"]
            if Path(filename).name != filename or sha(preview / filename) != checksums[filename]:
                raise ValueError("Frozen preview artifact hash/path mismatch")
            with gzip.open(preview / filename, "rt", encoding="utf-8") as stream:
                result = json.load(stream)
            if result["dirty_fingerprint"] != table.fingerprint or result["code_sha256"] != manifest["frozen_sha256"]["model_frozen.py"]:
                raise ValueError("Frozen fit provenance mismatch")
            if result["parameters"]["source_assumption"] != mode:
                raise ValueError("Frozen source assumption mismatch")
            contracts.append({"attribute": column, "source_assumption": mode, "contract": result["contract"]})
            decisions = {}
            for d in result["decisions"]:
                i = d["row"]
                if type(i) is not int or not 0 <= i < table.shape[0] or d["column"] != column or d["original"] != table.rows[i][j] or i in decisions:
                    raise ValueError("Stale/duplicate frozen row coordinate")
                decisions[i] = d
            if len(decisions) != table.shape[0]:
                raise ValueError("Incomplete frozen row coverage")
            by_key = {}
            for fold in result["folds"]:
                keys = {tuple(d["entity"]) for d in decisions.values() if d.get("fold") == fold["fold"]}
                grouped, _ = preparation._collapse({k: claims[k] for k in keys if k in claims}, fold["source_group_mapping"])
                blockers = list(fold["blockers"])
                if len(fold.get("models", [])) != policy.selected_fits:
                    blockers.append("unexpected_selected_fit_count")
                elif any(not m["converged"] for m in fold["models"]):
                    blockers.append("unconverged_frozen_selected_fit")
                for key, obs in grouped.items():
                    obs = {s: v for s, v in obs.items() if s in fold["training_groups"]}
                    context_id = digest({"attribute": column, "source_assumption": mode,
                                         "fold": fold["fold"], "entity": key})
                    context = {"context_id": context_id, "attribute": column,
                               "source_assumption": mode, "fold": fold["fold"], "entity": list(key),
                               "frozen_artifact": filename, "source_mapping": fold["source_group_mapping"],
                               "trained_groups": fold["training_groups"], "grouped_claims": obs,
                               "fold_blockers": blockers, "scenarios": []}
                    if not blockers and len(obs) >= policy.min_active_groups and len(obs) - 1 >= policy.min_leave_out_groups:
                        context.update(posterior_scenarios(obs, fold["models"], policy))
                    contexts[context_id] = context
                    by_key[key] = context_id
            mode_contexts[mode] = by_key
        for i, row in enumerate(table.rows):
            key = tuple(row[k] for k in key_indices)
            original, source = row[j], row[source_index]
            mode_evidence = {}
            eligible = True
            for mode in MODES:
                context_id = mode_contexts[mode].get(key)
                if context_id is None:
                    mode_evidence[mode] = {"context_id": None, "eligibility_blockers": ["no_frozen_item_context"],
                                           "original_weight_interval": None}
                    eligible = False
                    continue
                context = contexts[context_id]
                reasons = list(context["fold_blockers"]) + source_eligibility(
                    original, source, context["source_mapping"], context["trained_groups"],
                    context["grouped_claims"], conflicted=i in conflicts,
                    complete_key=all(v.strip() for v in key), policy=policy)
                if reasons:
                    mode_evidence[mode] = {"context_id": context_id, "eligibility_blockers": sorted(set(reasons)),
                                           "original_weight_interval": None}
                    eligible = False
                else:
                    mode_evidence[mode] = {"context_id": context_id, "eligibility_blockers": [],
                                           **interval_for_original(context, original)}
            cell = {"row": i, "column": column, "column_index": j, "original": original,
                    "entity": list(key), "source": source, "source_assumptions": mode_evidence,
                    "eligible_both_source_assumptions": eligible,
                    "original_weight_interval": None, "error_weight_interval": None,
                    "action": "retain_unsupported", "feedback_score": None}
            if eligible:
                low = min(v["original_weight_interval"][0] for v in mode_evidence.values())
                high = max(v["original_weight_interval"][1] for v in mode_evidence.values())
                error = [1 - high, 1 - low]
                action, score = classify_error_interval(error, policy)
                cell.update(original_weight_interval=[low, high], error_weight_interval=error,
                            action=action, feedback_score=score)
            cells.append(cell)
    return {"experiment": VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
            "policy": asdict(policy), "frozen_semantic_contracts": contracts,
            "probability_meaning": "conditional sensitivity weights, not calibrated correctness",
            "cells": sorted(cells, key=lambda c: (c["row"], c["column_index"]))}, contexts


def apply_detection_feedback(table: DirtyTable, prior: Prediction, fixed_correction: Prediction,
                             evidence: dict, policy: DetectionPolicy = DetectionPolicy()):
    prior.validate(table.shape)
    fixed_correction.validate(table.shape)
    if evidence["dirty_sha256"] != table.fingerprint:
        raise ValueError("Stale detection evidence")
    probability = prior.probabilities.copy()
    traces, seen = [], set()
    prior_rows = [list(row) for row in table.rows]
    final_rows = [list(row) for row in table.rows]
    for (i, j), value in prior.repairs.items():
        prior_rows[i][j] = value
    for (i, j), value in fixed_correction.repairs.items():
        final_rows[i][j] = value
    bundle_cells = []
    for cell in evidence["cells"]:
        i, j = cell["row"], cell["column_index"]
        if type(i) is not int or type(j) is not int or not 0 <= i < table.shape[0] or not 0 <= j < table.shape[1] or (i, j) in seen:
            raise ValueError("Invalid/duplicate detection coordinate")
        seen.add((i, j))
        if cell["column"] != table.columns[j] or cell["original"] != table.rows[i][j]:
            raise ValueError("Stale detection original/column")
        if set(cell["source_assumptions"]) != set(MODES):
            raise ValueError("Both frozen source assumptions are required")
        if cell["eligible_both_source_assumptions"]:
            mode_intervals = []
            for mode in MODES:
                item = cell["source_assumptions"][mode]
                if item["eligibility_blockers"] or item["original_weight_interval"] is None:
                    raise ValueError("Ineligible source assumption has a fabricated interval")
                low, high = map(float, item["original_weight_interval"])
                if not 0 <= low <= high <= 1:
                    raise ValueError("Invalid original-weight interval")
                mode_intervals.append((low, high))
            expected_original = [min(v[0] for v in mode_intervals), max(v[1] for v in mode_intervals)]
            expected_error = [1 - expected_original[1], 1 - expected_original[0]]
            if expected_original != cell["original_weight_interval"] or expected_error != cell["error_weight_interval"]:
                raise ValueError("Original/error intervals differ from source evidence")
            action, score = classify_error_interval(cell["error_weight_interval"], policy)
            if action != cell["action"] or score != cell["feedback_score"]:
                raise ValueError("Detection action differs from frozen interval policy")
            if score is not None:
                probability[i, j] = score
        elif cell["action"] != "retain_unsupported" or cell["feedback_score"] is not None or cell["error_weight_interval"] is not None:
            raise ValueError("Unsupported original has a fabricated score")
        trace = {**cell, "prior_probability": float(prior.probabilities[i, j]),
                 "final_probability": float(probability[i, j]),
                 "prior_ed_positive": bool(prior.probabilities[i, j] >= policy.threshold),
                 "final_ed_positive": bool(probability[i, j] >= policy.threshold),
                 "probability_changed": bool(prior.probabilities[i, j] != probability[i, j]),
                 "inherited_ec_repair": fixed_correction.repairs.get((i, j)),
                 "original_tuple": dict(zip(table.columns, table.rows[i]))}
        traces.append(trace)
        if trace["probability_changed"]:
            bundle_cells.append({"row": i, "column": table.columns[j], "column_index": j,
                                 "original": cell["original"], "original_tuple": trace["original_tuple"],
                                 "prior_inference_tuple": dict(zip(table.columns, prior_rows[i])),
                                 "tentative_tuple": dict(zip(table.columns, final_rows[i])),
                                 "detection_evidence": trace,
                                 "inherited_strict_ec_value": fixed_correction.repairs.get((i, j)),
                                 "correction_permission": "no changes beyond frozen v3 feedback repair map",
                                 "independent_verification_status": "not_run"})
    repairs = dict(fixed_correction.repairs)
    metadata = {"system": VERSION, "dirty_sha256": table.fingerprint,
                "statistical_detection_feedback_ablation": True, "fully_verified_main_method": False,
                "independent_verification_status": "not_run", "new_llm_calls": 0,
                "refit_performed": False, "ground_truth_access": False, "policy": asdict(policy),
                "probability_meaning": "conditional sensitivity endpoints, not calibrated correctness",
                "frozen_semantic_contracts": evidence.get("frozen_semantic_contracts", []),
                "ec_policy": "Exact inherited v3 feedback correction map; detection generates no repair"}
    prediction = Prediction(probability, repairs, metadata).validate(table.shape)
    audit = {"experiment": VERSION, "dirty_sha256": table.fingerprint, "trace": traces,
             "evidence_sha256": digest(evidence), "refit_performed": False,
             "correction_map_unchanged_from_v3": repairs == fixed_correction.repairs,
             "independent_verification_status": "not_run"}
    bundle = {"experiment": VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
              "purpose": "Prepared for later independent detection review; no verification executed",
              "independent_verification_status": "not_run", "cells": bundle_cells,
              "frozen_semantic_contracts": evidence.get("frozen_semantic_contracts", [])}
    return prediction, audit, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experimental/latent_consensus_detection_feedback_v4"))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("experimental").resolve()):
        raise ValueError("Output must remain under experimental/")
    if sha(PRIOR) != EXPECTED_PRIOR_SHA or sha(FIXED_EC / "prediction.json") != EXPECTED_EC_SHA or sha(FIXED_EC / "final_manifest.json") != EXPECTED_EC_MANIFEST_SHA:
        raise ValueError("Authorized frozen prior/v3 correction changed")
    ec_manifest = json.loads((FIXED_EC / "final_manifest.json").read_text())
    for path, wanted in ec_manifest["artifacts"].items():
        if sha(Path(path)) != wanted:
            raise ValueError("Frozen v3 feedback artifact changed")
    table = load_dirty("flights")
    prior = Prediction.load(PRIOR).validate(table.shape)
    fixed_ec = Prediction.load(FIXED_EC / "prediction.json").validate(table.shape)
    if prior.metadata.get("dirty_sha256") != table.fingerprint or fixed_ec.metadata.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Frozen prior/EC dirty fingerprint mismatch")
    policy = DetectionPolicy()
    preview_summary = json.loads((PREVIEW / "summary.json").read_text())
    preview_names = ["pre_fit_manifest.json", "summary.json", "artifact_manifest.json"]
    preview_names.extend(m["artifact"] for a in preview_summary["attributes"] for m in a["source_assumptions"].values())
    if any(Path(name).name != name for name in preview_names):
        raise ValueError("Preview names must be local")
    sources = [Path(__file__), Path(__file__).with_name("latent_consensus_detection_feedback_v4_protocol.md"),
               Path(preparation.__file__), Path(categorical.__file__), Path("zeroclean/data.py"),
               Path("zeroclean/types.py"), Path("tests/test_latent_consensus.py")]
    input_paths = [PRIOR, FIXED_EC / "prediction.json", FIXED_EC / "final_manifest.json",
                   FIXED_EC / "validation.json", *(PREVIEW / name for name in preview_names), *sources]
    input_hashes = {str(path): sha(path) for path in input_paths}
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output / "frozen_inputs"
    (frozen / "preview").mkdir(parents=True)
    copies = [(PRIOR, frozen / "loop_prior.json"), (FIXED_EC / "prediction.json", frozen / "v3_ec_prediction.json"),
              (FIXED_EC / "final_manifest.json", frozen / "v3_ec_final_manifest.json"),
              (FIXED_EC / "validation.json", frozen / "v3_ec_validation.json")]
    copies.extend((PREVIEW / name, frozen / "preview" / name) for name in preview_names)
    copies.extend((path, frozen / f"source_{i}_{path.name}") for i, path in enumerate(sources))
    frozen_hashes = {}
    for source, destination in copies:
        shutil.copyfile(source, destination)
        frozen_hashes[str(destination)] = input_hashes[str(source)]
        if sha(destination) != input_hashes[str(source)]:
            raise ValueError("Input changed during freezing")
    write_gzip_new(frozen / "dirty_table.json.gz", {"name": table.name, "columns": table.columns,
                                                   "rows": table.rows, "fingerprint": table.fingerprint})
    manifest = {"experiment": VERSION, "dirty_sha256": table.fingerprint, "shape": table.shape,
                "policy": asdict(policy), "input_hashes": input_hashes, "frozen_input_hashes": frozen_hashes,
                "frozen_dirty_strings_sha256": sha(frozen / "dirty_table.json.gz"),
                "inherited_ec_prediction_sha256": EXPECTED_EC_SHA,
                "no_refit": True, "new_llm_verification": False, "accuracy_evaluator_executed": False,
                "method_status": "Unpromoted statistical detection ablation; correction inherited exactly from frozen v3 feedback"}
    write_new(args.output / "pre_run_manifest.json", manifest)
    evidence, contexts = collect_detection_evidence(table, frozen / "preview", policy)
    write_gzip_new(args.output / "posterior_contexts.json.gz", contexts)
    evidence["posterior_contexts_sha256"] = sha(args.output / "posterior_contexts.json.gz")
    evidence["input_hashes"] = input_hashes
    write_gzip_new(args.output / "detection_evidence.json.gz", evidence)
    prior = Prediction.load(frozen / "loop_prior.json").validate(table.shape)
    fixed_ec = Prediction.load(frozen / "v3_ec_prediction.json").validate(table.shape)
    prediction, audit, bundle = apply_detection_feedback(table, prior, fixed_ec, evidence, policy)
    prediction.metadata.update(input_hashes=input_hashes, inherited_ec_prediction_sha256=EXPECTED_EC_SHA,
                               pre_run_manifest_sha256=sha(args.output / "pre_run_manifest.json"),
                               posterior_contexts_sha256=sha(args.output / "posterior_contexts.json.gz"))
    write_new(args.output / "prediction.json", {"probabilities": prediction.probabilities.tolist(),
              "repairs": [{"row": i, "column": j, "value": value} for (i, j), value in sorted(prediction.repairs.items())],
              "metadata": prediction.metadata})
    audit["prediction_sha256"] = sha(args.output / "prediction.json")
    audit["posterior_contexts_sha256"] = sha(args.output / "posterior_contexts.json.gz")
    write_gzip_new(args.output / "audit.json.gz", audit)
    bundle.update(parent_variant_prediction_sha256=sha(args.output / "prediction.json"),
                  posterior_contexts_path=str(args.output / "posterior_contexts.json.gz"),
                  posterior_contexts_sha256=sha(args.output / "posterior_contexts.json.gz"), input_hashes=input_hashes)
    bundle["request_hash"] = digest(bundle)
    write_new(args.output / "verification_bundle.json", bundle)
    for path, wanted in input_hashes.items():
        if sha(Path(path)) != wanted:
            raise ValueError("A frozen input changed during execution")
    paths = [args.output / name for name in ("prediction.json", "pre_run_manifest.json", "posterior_contexts.json.gz",
             "detection_evidence.json.gz", "audit.json.gz", "verification_bundle.json")]
    write_new(args.output / "output_manifest.json", {"artifacts": {str(p): sha(p) for p in paths},
              "method_status": manifest["method_status"], "accuracy_evaluator_executed": False,
              "independent_verification_status": "not_run"})
    for path in (args.output / "prediction.json", args.output / "output_manifest.json", args.output / "verification_bundle.json"):
        print(str(path) + " " + sha(path), flush=True)


if __name__ == "__main__":
    main()
