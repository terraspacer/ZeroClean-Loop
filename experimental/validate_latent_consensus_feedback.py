"""Read-only structural/provenance/guard validation; never an accuracy evaluator."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from zeroclean.data import DirtyTable
from zeroclean.sandbox import evaluate_rules
from zeroclean.types import Prediction, digest
from experimental.latent_consensus_feedback import sha, write_new


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("experimental/latent_consensus_feedback_v3"))
    args = parser.parse_args()
    base = args.directory
    manifest = json.loads((base / "pre_run_manifest.json").read_text())
    outputs = json.loads((base / "output_manifest.json").read_text())
    for name, wanted in manifest["input_hashes"].items():
        assert sha(Path(name)) == wanted, name
    for name, wanted in manifest["frozen_input_hashes"].items():
        assert sha(Path(name)) == wanted, name
    for name, wanted in outputs["outputs"].items():
        assert sha(Path(name)) == wanted, name
    dirty = read_gzip(base / "frozen_inputs/dirty_table.json.gz")
    assert sha(base / "frozen_inputs/dirty_table.json.gz") == manifest["frozen_dirty_strings_sha256"]
    table = DirtyTable(dirty["name"], tuple(dirty["columns"]), tuple(tuple(r) for r in dirty["rows"]), dirty["fingerprint"])
    assert table.fingerprint == manifest["dirty_sha256"]
    prior = Prediction.load(base / "frozen_inputs/prior_prediction.json").validate(table.shape)
    prediction = Prediction.load(base / "prediction.json").validate(table.shape)
    evidence = read_gzip(base / "statistical_evidence.json.gz")
    audit = read_gzip(base / "audit.json.gz")
    bundle = json.loads((base / "verification_bundle.json").read_text())
    assert digest(evidence) == audit["evidence_sha256"]
    assert audit["variant_prediction_sha256"] == sha(base / "prediction.json")
    assert bundle["parent_variant_prediction_sha256"] == sha(base / "prediction.json")
    assert bundle["request_hash"] == digest({k: v for k, v in bundle.items() if k != "request_hash"})
    assert prediction.metadata["new_llm_calls"] == 0
    assert prediction.metadata["fully_verified_main_method"] is False
    assert bundle["independent_verification_status"] == "not_run"
    assert audit["refit_performed"] is False
    keys = {(c["row"], c["column_index"]) for c in evidence["cells"]}
    assert len(keys) == len(evidence["cells"])
    mask = np.ones(table.shape, dtype=bool)
    for i, j in keys:
        mask[i, j] = False
    assert np.array_equal(prediction.probabilities[mask], prior.probabilities[mask])
    for key in (set(prior.repairs) | set(prediction.repairs)) - keys:
        assert prior.repairs.get(key) == prediction.repairs.get(key)
    changed = set()
    for trace in audit["trace"]:
        i, j = trace["row"], trace["column_index"]
        assert trace["original"] == table.rows[i][j]
        assert trace["original_tuple"] == dict(zip(table.columns, table.rows[i]))
        assert len(trace["delta_features"]) == 4 and np.isfinite(trace["delta_features"]).all()
        low, high = trace["original_weight_interval"]
        expected = 1 - low if trace["candidate"] == trace["original"] else 1 - high
        assert prediction.probabilities[i, j] == expected
        assert trace["final_repair"] == prediction.repairs.get((i, j))
        if trace["individual_gate_blockers"] or trace["joint_rollback"]:
            assert prediction.repairs.get((i, j)) == prior.repairs.get((i, j))
        if trace["probability_changed"] or trace["repair_changed"]:
            changed.add((i, j))
    assert {(c["row"], c["column_index"]) for c in bundle["cells"]} == changed
    for cell in bundle["cells"]:
        assert cell["original"] == table.rows[cell["row"]][cell["column_index"]]
    rows = [list(row) for row in table.rows]
    for (i, j), value in prediction.repairs.items():
        rows[i][j] = value
    rules = audit["accepted_rules"]
    final = np.asarray(evaluate_rules(rules, [dict(zip(table.columns, row)) for row in rows]), dtype=np.int8).reshape(len(rows), len(rules))
    parent = np.asarray(audit["prior_rule_matrix"], dtype=np.int8)
    multi = np.array([r["kind"] == "multi" for r in rules], dtype=bool)
    assert not ((parent == 0) & (final == 1) & multi).any()
    assert not ((parent != -1) & (final == -1)).any()
    validation = {"kind": "structural provenance and counterfactual guard checks; no accuracy evaluation",
                  "status": "passed", "validator_sha256": sha(Path(__file__)),
                  "prediction_sha256": sha(base / "prediction.json"),
                  "output_manifest_sha256": sha(base / "output_manifest.json"),
                  "frozen_inputs_verified": True, "unsupported_inference_preserved": True,
                  "all_original_strings_preserved": True, "posterior_endpoint_feedback_verified": True,
                  "independent_full_tuple_guard_verified": True, "verification_bundle_binding_verified": True,
                  "llm_verification_executed": False, "accuracy_evaluator_executed": False}
    write_new(base / "validation.json", validation)
    paths = [base / name for name in ("prediction.json", "pre_run_manifest.json", "output_manifest.json",
             "statistical_evidence.json.gz", "audit.json.gz", "verification_bundle.json", "validation.json")]
    write_new(base / "final_manifest.json", {"artifacts": {str(p): sha(p) for p in paths},
              "validator_path": str(Path(__file__)), "validator_sha256": sha(Path(__file__)),
              "method_status": outputs["method_status"], "independent_verification_status": "not_run",
              "accuracy_evaluator_executed": False})
    for path in (base / "prediction.json", base / "final_manifest.json", base / "verification_bundle.json"):
        print(str(path) + " " + sha(path))


if __name__ == "__main__":
    main()
