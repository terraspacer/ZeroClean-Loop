"""Structural validation of frozen v4 detection sensitivity; no accuracy evaluator."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from zeroclean.data import DirtyTable
from zeroclean.types import Prediction, digest
from experimental import latent_consensus as preparation
from experimental import latent_consensus_v2 as categorical
from experimental.latent_consensus_detection_feedback_v4 import MODES, sha, write_new


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("experimental/latent_consensus_detection_feedback_v4"))
    args = parser.parse_args()
    base = args.directory
    manifest = json.loads((base / "pre_run_manifest.json").read_text())
    outputs = json.loads((base / "output_manifest.json").read_text())
    for group in (manifest["input_hashes"], manifest["frozen_input_hashes"], outputs["artifacts"]):
        for path, wanted in group.items():
            assert sha(Path(path)) == wanted, path
    fixed_manifest = json.loads((base / "frozen_inputs/v3_ec_final_manifest.json").read_text())
    for path, wanted in fixed_manifest["artifacts"].items():
        assert sha(Path(path)) == wanted, path
    v3_pre_run = json.loads(Path("experimental/latent_consensus_feedback_v3/pre_run_manifest.json").read_text())
    for path, wanted in v3_pre_run["input_hashes"].items():
        assert sha(Path(path)) == wanted, path
    dirty = read_gzip(base / "frozen_inputs/dirty_table.json.gz")
    assert sha(base / "frozen_inputs/dirty_table.json.gz") == manifest["frozen_dirty_strings_sha256"]
    table = DirtyTable(dirty["name"], tuple(dirty["columns"]), tuple(tuple(r) for r in dirty["rows"]), dirty["fingerprint"])
    prior = Prediction.load(base / "frozen_inputs/loop_prior.json").validate(table.shape)
    fixed = Prediction.load(base / "frozen_inputs/v3_ec_prediction.json").validate(table.shape)
    prediction = Prediction.load(base / "prediction.json").validate(table.shape)
    assert prediction.repairs == fixed.repairs
    assert prediction.metadata["new_llm_calls"] == 0 and prediction.metadata["refit_performed"] is False
    assert prediction.metadata["fully_verified_main_method"] is False
    contexts = read_gzip(base / "posterior_contexts.json.gz")
    evidence = read_gzip(base / "detection_evidence.json.gz")
    audit = read_gzip(base / "audit.json.gz")
    bundle = json.loads((base / "verification_bundle.json").read_text())
    assert digest(evidence) == audit["evidence_sha256"]
    assert audit["prediction_sha256"] == sha(base / "prediction.json")
    assert evidence["posterior_contexts_sha256"] == sha(base / "posterior_contexts.json.gz")
    assert bundle["request_hash"] == digest({k: v for k, v in bundle.items() if k != "request_hash"})
    assert bundle["parent_variant_prediction_sha256"] == sha(base / "prediction.json")
    assert bundle["independent_verification_status"] == "not_run"
    preview = base / "frozen_inputs/preview"
    preview_manifest = json.loads((preview / "pre_fit_manifest.json").read_text())
    preview_summary = json.loads((preview / "summary.json").read_text())
    fold_lookup, claims_lookup, conflicts_lookup = {}, {}, {}
    for attribute in preview_summary["attributes"]:
        column = attribute["attribute"]
        roles = categorical.SkillContract(tuple(preview_manifest["entity_columns"]),
            preview_manifest["source_column"], column, tuple(preview_manifest["context_columns"]))
        claims, extraction = preparation._extract(table, roles)
        claims_lookup[column] = claims
        conflicts_lookup[column] = {r for c in extraction["source_entity_conflicts"] for r in c["rows"]}
        for mode in MODES:
            name = attribute["source_assumptions"][mode]["artifact"]
            frozen = read_gzip(preview / name)
            for fold in frozen["folds"]:
                fold_lookup[name, fold["fold"]] = {
                    "source_group_mapping": fold["source_group_mapping"],
                    "training_groups": fold["training_groups"], "blockers": fold["blockers"],
                    "models": [{"skill": m["skill"], "unseen_prior": m["unseen_prior"], "converged": m["converged"]}
                               for m in fold.get("models", [])]}
    for context_id, context in contexts.items():
        assert context["context_id"] == context_id
        fold = fold_lookup[context["frozen_artifact"], context["fold"]]
        assert context["source_mapping"] == fold["source_group_mapping"]
        assert context["trained_groups"] == fold["training_groups"]
        key = tuple(context["entity"])
        raw = claims_lookup[context["attribute"]][key]
        grouped, _ = preparation._collapse({key: raw}, context["source_mapping"])
        obs = {s: v for s, v in grouped[key].items() if s in context["trained_groups"]}
        assert obs == context["grouped_claims"]
        if not context["scenarios"]:
            continue
        assert len(obs) >= 4 and len(obs) - 1 >= 3
        assert not fold["blockers"] and len(fold["models"]) == 17
        candidates = tuple(context["candidate_strings"])
        assert candidates == tuple(sorted(set(obs.values())))
        assert context["unobserved_category_index"] == len(candidates)
        assert len(context["scenarios"]) == 17 * (len(obs) + 1)
        scenario_keys = set()
        for scenario in context["scenarios"]:
            index, omitted = scenario["selected_model_index"], scenario["omitted_group"]
            assert (index, omitted) not in scenario_keys
            scenario_keys.add((index, omitted))
            assert omitted is None or omitted in obs
            remaining = {s: v for s, v in obs.items() if s != omitted}
            model = fold["models"][index]
            assert model["converged"]
            _, expected = categorical._posterior(remaining, model["skill"], model["unseen_prior"], candidates)
            assert np.allclose(expected, scenario["posterior_weights"], rtol=0, atol=1e-14)
        assert scenario_keys == {(i, omitted) for i in range(17) for omitted in (None, *obs)}
    traces = {(t["row"], t["column_index"]): t for t in audit["trace"]}
    assert len(traces) == len(audit["trace"]) == len(evidence["cells"])
    visited, changed = set(), set()
    for cell in evidence["cells"]:
        i, j = cell["row"], cell["column_index"]
        visited.add((i, j))
        trace = traces[i, j]
        assert cell["original"] == table.rows[i][j]
        assert trace["original_tuple"] == dict(zip(table.columns, table.rows[i]))
        expected = prior.probabilities[i, j]
        if not cell["eligible_both_source_assumptions"]:
            assert cell["error_weight_interval"] is None and cell["feedback_score"] is None
            assert cell["action"] == "retain_unsupported"
        else:
            lows, highs = [], []
            for mode in MODES:
                item = cell["source_assumptions"][mode]
                assert not item["eligibility_blockers"]
                context = contexts[item["context_id"]]
                assert not context["fold_blockers"] and context["source_assumption"] == mode
                original, source = cell["original"], cell["source"]
                assert original.strip() and source.strip()
                assert i not in conflicts_lookup[cell["column"]]
                group = context["source_mapping"][source]
                assert group in context["trained_groups"]
                assert context["grouped_claims"][group] == original
                index = context["candidate_strings"].index(original)
                values = [s["posterior_weights"][index] for s in context["scenarios"]]
                interval = [min(values), max(values)]
                assert item["original_weight_interval"] == interval
                lows.append(interval[0]); highs.append(interval[1])
            original = [min(lows), max(highs)]
            error = [1-original[1], 1-original[0]]
            assert cell["original_weight_interval"] == original and cell["error_weight_interval"] == error
            if error[0] > .5:
                assert cell["action"] == "flag"; expected = error[0]
            elif error[1] < .5:
                assert cell["action"] == "prune"; expected = error[1]
            else:
                assert cell["action"] == "retain_uncertain" and cell["feedback_score"] is None
        assert prediction.probabilities[i, j] == expected
        assert trace["final_probability"] == expected
        assert trace["inherited_ec_repair"] == fixed.repairs.get((i, j))
        if expected != prior.probabilities[i, j]:
            changed.add((i, j))
    mask = np.ones(table.shape, dtype=bool)
    for i, j in visited:
        mask[i, j] = False
    assert np.array_equal(prediction.probabilities[mask], prior.probabilities[mask])
    assert {(c["row"], c["column_index"]) for c in bundle["cells"]} == changed
    validation = {"kind": "structural, frozen-posterior and uncertainty-policy validation; no accuracy evaluator",
                  "status": "passed", "validator_sha256": sha(Path(__file__)),
                  "prediction_sha256": sha(base / "prediction.json"),
                  "exact_v3_correction_map_preserved": True, "v3_artifact_and_source_hashes_preserved": True,
                  "all_scenarios_recomputed_from_frozen_fits": True, "unsupported_and_uncertain_prior_preserved": True,
                  "no_absent_original_scores": True, "strict_half_threshold_verified": True,
                  "no_refit": True, "llm_verification_executed": False, "accuracy_evaluator_executed": False}
    write_new(base / "validation.json", validation)
    paths = [base / name for name in ("prediction.json", "pre_run_manifest.json", "output_manifest.json",
             "posterior_contexts.json.gz", "detection_evidence.json.gz", "audit.json.gz", "verification_bundle.json", "validation.json")]
    write_new(base / "final_manifest.json", {"artifacts": {str(p): sha(p) for p in paths},
              "validator_path": str(Path(__file__)), "validator_sha256": sha(Path(__file__)),
              "method_status": outputs["method_status"], "accuracy_evaluator_executed": False,
              "independent_verification_status": "not_run"})
    for path in (base / "prediction.json", base / "final_manifest.json", base / "verification_bundle.json"):
        print(str(path) + " " + sha(path))


if __name__ == "__main__":
    main()
