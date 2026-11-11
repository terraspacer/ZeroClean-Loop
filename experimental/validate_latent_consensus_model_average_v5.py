"""Structural and mathematical v5 validation; no benchmark accuracy evaluator."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from zeroclean.data import DirtyTable
from zeroclean.types import Prediction, digest
from experimental import latent_consensus_model_average_v5 as v5


def check_hashes(mapping):
    for name, wanted in mapping.items():
        assert v5.sha(Path(name)) == wanted, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean", type=Path, default=Path("experimental/latent_consensus_model_average_v5"))
    parser.add_argument("--proxy", type=Path, default=Path("experimental/latent_consensus_expected_f1_proxy_v5"))
    args = parser.parse_args()
    manifests = [json.loads((base / "pre_run_manifest.json").read_text()) for base in (args.mean, args.proxy)]
    assert manifests[0]["input_hashes"] == manifests[1]["input_hashes"]
    for base, manifest in zip((args.mean, args.proxy), manifests):
        check_hashes(manifest["input_hashes"])
        check_hashes(manifest["frozen_input_hashes"])
        check_hashes(json.loads((base / "output_manifest.json").read_text())["artifacts"])
    assert v5.sha(v5.FROZEN_V4_TEST) == v5.EXPECTED_V4_TEST
    for old in (Path("experimental/latent_consensus_feedback_v3"), v5.V4):
        old_manifest = json.loads((old / "final_manifest.json").read_text())
        check_hashes(old_manifest["artifacts"])
        pre = json.loads((old / "pre_run_manifest.json").read_text())
        check_hashes(pre["input_hashes"])
        check_hashes(pre["frozen_input_hashes"])
    evidence = v5.read_gzip(args.mean / "mean_evidence.json.gz")
    claim_contexts = v5.read_gzip(args.mean / "claim_contexts.json.gz")
    assert evidence["claim_contexts_sha256"] == v5.sha(args.mean / "claim_contexts.json.gz")
    context_path = v5.V4 / "posterior_contexts.json.gz"
    contexts = v5.read_gzip(context_path)
    frozen_evidence = v5.read_gzip(v5.V4 / "detection_evidence.json.gz")
    dirty = v5.read_gzip(v5.V4 / "frozen_inputs/dirty_table.json.gz")
    table = DirtyTable(dirty["name"], tuple(dirty["columns"]), tuple(tuple(r) for r in dirty["rows"]), dirty["fingerprint"])
    prior = Prediction.load(v5.V4 / "frozen_inputs/loop_prior.json").validate(table.shape)
    fixed = Prediction.load(v5.V3_EC / "prediction.json").validate(table.shape)
    frozen_cells = {(c["row"], c["column_index"]): c for c in frozen_evidence["cells"]}
    raw_claims = defaultdict(list)
    for cell in frozen_evidence["cells"]:
        context_id = digest({"column": cell["column"], "entity": cell["entity"]})
        raw_claims[context_id].append({"row": cell["row"], "source": cell["source"],
            "value": cell["original"], "blank_or_whitespace": not cell["original"].strip()})
    assert set(raw_claims) == set(claim_contexts)
    for context_id, record in claim_contexts.items():
        assert record["raw_claims"] == raw_claims[context_id]
        assert record["conditional_hypotheses_only"] is True
        j = table.columns.index(record["attribute"])
        for claim in record["raw_claims"]:
            assert claim["value"] == table.rows[claim["row"]][j]
            assert claim["source"] == table.rows[claim["row"]][table.columns.index("src")]
        for mode, view in record["source_assumptions"].items():
            node = contexts[view["frozen_context_id"]]
            assert view["source_to_group"] == node["source_mapping"]
            assert view["trained_groups"] == node["trained_groups"]
            assert view["active_group_claims"] == node["grouped_claims"]
            assert view["scenario_linkage"]["selected_model_indices"] == sorted({s["selected_model_index"] for s in node["scenarios"]})
    modeled, coordinates, means = [], [], {}
    for cell in evidence["cells"]:
        key = cell["row"], cell["column_index"]
        frozen = frozen_cells[key]
        point = cell["mean_evidence"]
        assert cell["original"] == table.rows[key[0]][key[1]]
        assert point["eligible"] == frozen["eligible_both_source_assumptions"]
        assert point["v4_action"] == frozen["action"]
        assert point["v4_error_weight_interval"] == frozen["error_weight_interval"]
        if not point["eligible"]:
            assert point["mean_error_weight"] is None
            continue
        mode_means, all_diagnostics = {}, []
        for mode in v5.MODES:
            node = contexts[frozen["source_assumptions"][mode]["context_id"]]
            index = node["candidate_strings"].index(cell["original"])
            buckets = defaultdict(dict)
            for scenario in node["scenarios"]:
                buckets[scenario["omitted_group"]][scenario["selected_model_index"]] = scenario["posterior_weights"][index]
            assert all(set(weights) == set(range(17)) for weights in buckets.values())
            full = [buckets[None][i] for i in range(17)]
            mode_means[mode] = float(np.mean(full))
            stored = point["mode_evidence"][mode]
            assert stored["full_original_weights"] == full
            assert stored["full_original_mean"] == mode_means[mode]
            assert stored["each_full_model_weight"] == .5/17
            for group, weights in buckets.items():
                if group is not None:
                    assert stored["leave_group_out_error_means"][group] == 1-float(np.mean([weights[i] for i in range(17)]))
        expected = .5*sum(1-p for p in mode_means.values())
        assert point["mean_error_weight"] == expected
        all_diagnostics.append(expected)
        for mode in v5.MODES:
            other = next(m for m in v5.MODES if m != mode)
            all_diagnostics.extend(.5*(value+(1-mode_means[other]))
                for value in point["mode_evidence"][mode]["leave_group_out_error_means"].values())
        assert point["combined_leave_group_out_error_range"] == [min(all_diagnostics), max(all_diagnostics)]
        assert point["leave_group_out_mean_decision_changes"] == any((p>=.5)!=(expected>=.5) for p in all_diagnostics)
        modeled.append(cell); coordinates.append(key); means[key] = expected
    # Discard the large frozen posterior matrices before parsing embedded verifier contexts.
    del contexts, frozen_evidence, frozen_cells
    predictions = [Prediction.load(base / "prediction.json").validate(table.shape) for base in (args.mean, args.proxy)]
    audits = [v5.read_gzip(base / "audit.json.gz") for base in (args.mean, args.proxy)]
    solution = audits[1]["solution"]
    p = np.array([means[key] for key in coordinates])
    required = {i for i, key in enumerate(coordinates) if key in fixed.repairs}
    selected = set(solution["selected_indices"])
    assert required == set(solution["mandatory_indices"]) and required <= selected
    free = sorted((i for i in range(len(p)) if i not in required), key=lambda i: (-p[i], coordinates[i]))
    assert solution["free_order"] == free
    assert selected == required | set(free[:solution["selected_optional_count"]])
    utility_path = solution["utility_path"]
    maximum = max(entry["independent_marginals_proxy_expected_f1"] for entry in utility_path)
    first = next(entry for entry in utility_path if maximum-entry["independent_marginals_proxy_expected_f1"] <= 1e-12)
    assert first["optional_positive_count"] == solution["selected_optional_count"]
    assert solution["selected_positive_count"] == len(selected)
    if selected:
        # Independent mathematical check using selected/unselected convolutions,
        # rather than the optimizer's leave-one-polynomial accumulation.
        select_distribution = v5.poisson_binomial(p[sorted(selected)])
        others = sorted(set(range(len(p)))-selected)
        other_distribution = v5.poisson_binomial(p[others])
        ranks = np.arange(len(other_distribution))
        value = sum(float(select_distribution[t])*2*t*float(np.dot(other_distribution, 1/(len(selected)+t+ranks)))
                    for t in range(1, len(select_distribution)))
    else:
        value = 0.0
    assert abs(value-solution["selected_independent_marginals_proxy_expected_f1"]) <= 1e-8
    expected_mean = prior.probabilities.copy()
    expected_proxy = prior.probabilities.copy()
    for index, key in enumerate(coordinates):
        expected_mean[key] = p[index]
        expected_proxy[key] = float(index in selected)
    for expected in (expected_mean, expected_proxy):
        for key in fixed.repairs:
            assert fixed.probabilities[key] >= .5
            if expected[key] < .5:
                expected[key] = fixed.probabilities[key]
    for base, prediction, audit, expected in zip((args.mean, args.proxy), predictions, audits, (expected_mean, expected_proxy)):
        assert np.array_equal(prediction.probabilities, expected)
        assert prediction.repairs == fixed.repairs
        assert all(prediction.probabilities[key] >= .5 for key in fixed.repairs)
        assert prediction.metadata["new_llm_calls"] == 0 and prediction.metadata["refit_performed"] is False
        assert prediction.metadata["conditional_hypotheses_only"] is True
        assert prediction.metadata["fully_verified_main_method"] is False
        assert audit["prediction_sha256"] == v5.sha(base / "prediction.json")
        bundle = json.loads((base / "verification_bundle.json").read_text())
        assert bundle["request_hash"] == digest({k: v for k, v in bundle.items() if k != "request_hash"})
        assert bundle["parent_prediction_sha256"] == v5.sha(base / "prediction.json")
        assert bundle["conditional_hypotheses_only"] is True
        changed = set(map(tuple, np.argwhere(prediction.probabilities != prior.probabilities)))
        assert {(c["row"], c["column_index"]) for c in bundle["cells"]} == changed
        for cell in bundle["cells"]:
            trace = cell["ed_evidence"]
            assert cell["raw_dirty_claim_context"] == claim_contexts[trace["claim_context_id"]]
            assert cell["original"] == table.rows[cell["row"]][cell["column_index"]]
        del bundle
        validation = {"kind": "structural provenance, averaging and independent-marginals mathematics only",
            "status": "passed", "validator_sha256": v5.sha(Path(__file__)),
            "prediction_sha256": v5.sha(base / "prediction.json"),
            "v3_v4_live_and_frozen_bindings_preserved": True,
            "frozen_v4_test_sha256": v5.sha(v5.FROZEN_V4_TEST),
            "equal_model_weights_verified": True, "raw_claim_contexts_embedded": True,
            "exact_v3_repairs_and_detection_consistency_verified": True,
            "proxy_joint_expectation_is_only_independent_marginals_approximation": True,
            "selected_proxy_utility_checked_by_independent_convolution": True,
            "accuracy_evaluator_executed": False, "llm_verification_executed": False}
        v5.write_new(base / "validation.json", validation)
    artifacts = {}
    for base in (args.mean, args.proxy):
        artifacts.update(json.loads((base / "output_manifest.json").read_text())["artifacts"])
        for name in ("output_manifest.json", "validation.json"):
            artifacts[str(base / name)] = v5.sha(base / name)
    final = {"artifacts": artifacts, "validator_path": str(Path(__file__)),
        "validator_sha256": v5.sha(Path(__file__)), "accuracy_evaluator_executed": False,
        "independent_verification_status": "not_run",
        "optional_policy_scope": "Independent-marginals proxy on modeled eligible subset; not actual shared-entity joint or benchmark"}
    v5.write_new(args.mean / "final_manifest.json", final)
    for path in (args.mean / "prediction.json", args.proxy / "prediction.json", args.mean / "final_manifest.json", args.mean / "coverage.json"):
        print(str(path)+" "+v5.sha(path))


if __name__ == "__main__":
    main()
