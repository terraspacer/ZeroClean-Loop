"""Structural v6 validation of frozen ED, candidate means and parent guards; no evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zeroclean.data import DirtyTable
from zeroclean.statistics import DirtyIndex
from zeroclean.types import Prediction, digest
from experimental import latent_consensus_mean_repairs_v6 as v6


def hashes(mapping):
    for path, wanted in mapping.items():
        assert v6.sha(path) == wanted, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("experimental/latent_consensus_mean_repairs_v6"))
    args = parser.parse_args()
    base = args.directory
    manifest = json.loads((base / "pre_run_manifest.json").read_text())
    hashes(manifest["input_hashes"]); hashes(manifest["frozen_input_hashes"])
    output = json.loads((base / "output_manifest.json").read_text()); hashes(output["artifacts"])
    for directory in (Path("experimental/latent_consensus_feedback_v3"), v6.V4,
                      v6.V5, Path("experimental/latent_consensus_expected_f1_proxy_v5")):
        pre = json.loads((directory / "pre_run_manifest.json").read_text())
        hashes(pre["input_hashes"]); hashes(pre["frozen_input_hashes"])
    hashes(json.loads((v6.V5 / "final_manifest.json").read_text())["artifacts"])
    dirty = v6.read_gzip(v6.V4 / "frozen_inputs/dirty_table.json.gz")
    table = DirtyTable(dirty["name"], tuple(dirty["columns"]), tuple(tuple(r) for r in dirty["rows"]), dirty["fingerprint"])
    prior = Prediction.load(v6.V4 / "frozen_inputs/loop_prior.json").validate(table.shape)
    fixed = Prediction.load(v6.V5 / "prediction.json").validate(table.shape)
    prediction = Prediction.load(base / "prediction.json").validate(table.shape)
    assert prediction.probabilities.tobytes() == fixed.probabilities.tobytes()
    assert all(fixed.probabilities[key] >= .5 for key in prediction.repairs)
    mean = v6.read_gzip(v6.V5 / "mean_evidence.json.gz")
    mean_cells = {(c["row"], c["column_index"]): c for c in mean["cells"]}
    contexts = v6.read_gzip(v6.V4 / "posterior_contexts.json.gz")
    claims = v6.read_gzip(v6.V5 / "claim_contexts.json.gz")
    evidence = v6.read_gzip(base / "candidate_evidence.json.gz")
    audit = v6.read_gzip(base / "audit.json.gz")
    assert audit["prediction_sha256"] == v6.sha(base / "prediction.json")
    assert len(evidence["cells"]) == len(mean_cells)
    policy = v6.RepairPolicy()
    traces = {(t["row"], t["column_index"]): t for t in audit["trace"]}
    for cell in evidence["cells"]:
        key = cell["row"], cell["column_index"]
        frozen = mean_cells[key]
        assert cell["original"] == table.rows[key[0]][key[1]]
        assert cell["eligible"] == frozen["mean_evidence"]["eligible"]
        if cell["eligible"]:
            nodes = {m: contexts[frozen["mean_evidence"]["mode_evidence"][m]["context_id"]] for m in v6.MODES}
            expected = v6.rank_observed_alternatives(cell["original"], nodes)
            assert expected == cell["selection"]
            assert len(expected["top3"]) <= 3
            assert len({c["value"] for c in expected["top3"]}) == len(expected["top3"])
            assert cell["original"] not in {c["value"] for c in expected["top3"]}
            assert np.isclose(expected["original_mean_weight"], 1-frozen["mean_evidence"]["mean_error_weight"], rtol=0, atol=1e-14)
        else:
            assert cell["selection"] is None
        trace = traces[key]
        assert trace["considered_for_new_repair"] == bool(cell["eligible"] and fixed.probabilities[key] >= .5)
        assert trace["fixed_ed_score"] == fixed.probabilities[key]
    del contexts
    rules = audit["accepted_rules"]
    artifact = json.loads(v6.RULES.read_text())
    accepted, _ = v6.validate_rules(artifact["rules"], table.records(), policy.runtime_error_rate)
    assert rules == [r for r in accepted if r["target"] in table.columns]
    multi = np.array([r["kind"] == "multi" for r in rules], dtype=bool)
    original_matrix = v6.matrix(rules, table.records())
    parent_rows = v6.repaired_rows(table, prior.repairs)
    parent_matrix = v6.matrix(rules, [dict(zip(table.columns, row)) for row in parent_rows])
    removed = {key for key in prior.repairs if fixed.probabilities[key] < .5}
    assert removed == {(r["row"], r["column_index"]) for r in audit["coherence_rollbacks"]}
    coherent = {key: value for key, value in prior.repairs.items() if key not in removed}
    coherent_rows = v6.repaired_rows(table, coherent)
    coherent_matrix = v6.matrix(rules, [dict(zip(table.columns, row)) for row in coherent_rows])
    assert not (((parent_matrix == 0) & (coherent_matrix == 1) & multi).any() or ((parent_matrix != -1) & (coherent_matrix == -1)).any())
    assert original_matrix.tolist() == audit["original_rule_matrix"]
    assert parent_matrix.tolist() == audit["source_parent_rule_matrix"]
    assert coherent_matrix.tolist() == audit["coherent_parent_rule_matrix"]
    index = DirtyIndex(table, artifact.get("dependencies", []), discover=True)
    jobs = [(t, check) for t in audit["trace"] for check in t["candidate_checks"]]
    dirty_changed = v6.matrix(rules, [check["dirty_counterfactual_tuple"] for _, check in jobs])
    parent_changed = v6.matrix(rules, [check["parent_counterfactual_tuple"] for _, check in jobs])
    proposed = dict(coherent)
    for q, (trace, check) in enumerate(jobs):
        i, j = trace["row"], trace["column_index"]
        dirty_after = list(table.rows[i]); dirty_after[j] = check["value"]
        parent_after = list(coherent_rows[i]); parent_after[j] = check["value"]
        assert check["dirty_counterfactual_tuple"] == dict(zip(table.columns, dirty_after))
        assert check["parent_counterfactual_tuple"] == dict(zip(table.columns, parent_after))
        assert check["representation_gate"] == v6.representation_gate(trace["original"], check["value"])
        delta = v6.frozen_cf.counterfactual_dimensions(index, i, j, check["value"], dirty_after,
                                                     original_matrix[i], dirty_changed[q], rules)
        assert delta == check["delta_features"] and len(delta) == 4
        confidence = v6.frozen_cf._confidence(check["mean_weight"], delta, policy)
        assert confidence == check["correction_confidence"]
        blockers = list(trace["selection"]["top_gate_blockers"])
        if check["rank"] != 1: blockers.append("not_top_no_lower_rank_fallback")
        blockers.extend(check["representation_gate"]["blockers"])
        if confidence < .8: blockers.append("counterfactual_confidence_below_threshold")
        if ((coherent_matrix[i] == 0) & (parent_changed[q] == 1) & multi).any(): blockers.append("new_parent_multi_column_violation")
        if (parent_changed[q] == -1).any(): blockers.append("parent_counterfactual_runtime_failure")
        if (dirty_changed[q] == -1).any(): blockers.append("dirty_counterfactual_runtime_failure")
        assert sorted(set(blockers)) == check["blockers"]
        assert check["passed"] == (not blockers)
        if not blockers:
            assert check["rank"] == 1 and trace["new_choice"] == check["value"]
            proposed[i, j] = check["value"]
    proposed_rows = v6.repaired_rows(table, proposed)
    changed_rows = sorted({i for (i, j), value in proposed.items() if value != coherent.get((i, j))})
    joint_matrix = v6.matrix(rules, [dict(zip(table.columns, proposed_rows[i])) for i in changed_rows])
    expected = dict(proposed)
    joint_lookup = {row["row"]: row for row in audit["joint_guard"]}
    assert set(joint_lookup) == set(changed_rows)
    for q, i in enumerate(changed_rows):
        rollback = bool(((coherent_matrix[i] == 0) & (joint_matrix[q] == 1) & multi).any() or (joint_matrix[q] == -1).any())
        assert joint_lookup[i]["rollback_to_coherent_prior"] == rollback
        if rollback:
            for key in [key for key in expected if key[0] == i]: expected.pop(key)
            expected.update({key: value for key, value in coherent.items() if key[0] == i})
    assert expected == prediction.repairs
    final_rows = v6.repaired_rows(table, prediction.repairs)
    final_matrix = v6.matrix(rules, [dict(zip(table.columns, row)) for row in final_rows])
    assert final_matrix.tolist() == audit["final_rule_matrix"]
    for before in (parent_matrix, coherent_matrix):
        assert not ((before == 0) & (final_matrix == 1) & multi).any()
        assert not ((before != -1) & (final_matrix == -1)).any()
    bundle = json.loads((base / "verification_bundle.json").read_text())
    assert bundle["request_hash"] == digest({k: v for k, v in bundle.items() if k != "request_hash"})
    assert bundle["parent_prediction_sha256"] == v6.sha(base / "prediction.json")
    expected_bundle = {key for key, trace in traces.items() if trace["considered_for_new_repair"]} | removed
    assert {(c["row"], c["column_index"]) for c in bundle["cells"]} == expected_bundle
    for cell in bundle["cells"]:
        key = cell["row"], cell["column_index"]
        trace = traces[key]
        assert cell["raw_dirty_claim_context"] == claims[trace["claim_context_id"]]
        assert cell["repair_evidence"] == trace
    assert prediction.metadata["new_llm_calls"] == 0 and prediction.metadata["refit_performed"] is False
    validation = {"kind": "structural mean-repair, representation and guard validation; no accuracy evaluator",
        "status": "passed", "validator_sha256": v6.sha(Path(__file__)),
        "prediction_sha256": v6.sha(base / "prediction.json"), "fixed_v5_ed_bitwise_equal": True,
        "v3_v4_v5_live_and_frozen_bindings_preserved": True, "eligibility_not_expanded": True,
        "shared_observed_top3_and_equal_weights_verified": True, "representation_gate_verified": True,
        "all_four_dirty_counterfactual_dimensions_verified": True, "parent_and_joint_guards_verified": True,
        "repairs_within_unchanged_detection": True, "raw_source_claim_context_embedded": True,
        "accuracy_evaluator_executed": False, "llm_verification_executed": False}
    v6.write_new(base / "validation.json", validation)
    artifacts = dict(output["artifacts"])
    for name in ("output_manifest.json", "validation.json"): artifacts[str(base / name)] = v6.sha(base / name)
    v6.write_new(base / "final_manifest.json", {"artifacts": artifacts, "validator_path": str(Path(__file__)),
        "validator_sha256": v6.sha(Path(__file__)), "fixed_ed_sha256": v6.EXPECTED_ED,
        "accuracy_evaluator_executed": False, "independent_verification_status": "not_run"})
    for path in (base / "prediction.json", base / "final_manifest.json", base / "verification_bundle.json", base / "counts.json"):
        print(str(path)+" "+v6.sha(path))


if __name__ == "__main__":
    main()
