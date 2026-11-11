"""Synthetic adversarial tests; no benchmark files, labels or predictions read."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experimental import format_overlay as overlay
from zeroclean.data import DirtyTable
from zeroclean.types import Prediction, digest


class FlatDirtyIndex:
    def __init__(self, table, dependencies, discover):
        self.table = table
        self.dependencies = []

    def conditional(self, i, j, value, row):
        return 0., [{"observations": "raw_dirty_only", "raw_original": self.table.rows[i][j]}]

    def vicinity(self, i, j, value, row):
        return 0.

    def domain_support(self, i, j, value):
        return 0.


def evaluator(rules, records):
    matrix = []
    for row in records:
        result = []
        for rule in rules:
            if rule["id"] == "rA":
                result.append(int(row["length_cm"].endswith(" cm")))
            elif rule["id"] == "rB":
                result.append(int(row["role"] == "old"))
            elif rule["id"] == "joint":
                result.append(int(row["length_cm"] == "100" and row["role"] == "new"))
            elif rule["id"] == "runtime":
                result.append(-1 if row["length_cm"] == "100" else 0)
            else:
                raise AssertionError(rule["id"])
        matrix.append(result)
    return matrix


def validator(rules, records, max_error_rate):
    assert max_error_rate == .05
    return rules, [{"id": r["id"], "accepted": True, "error_rate": 0.} for r in rules]


DEPS = {"rule_validator": validator, "rule_evaluator": evaluator, "index_factory": FlatDirtyIndex}


def fixture(*, scalar_columns=("length_cm",), bundle=False, rules_extra=(), parent_repair=None):
    table = DirtyTable("synthetic", ("length_cm", "role"),
                       (("100 cm", "old"), ("100", "new"), ("200", "other")), "synthetic_dirty_hash")
    probabilities = np.zeros(table.shape) + .3
    repairs = {}
    if parent_repair is not None:
        repairs[(0, 0)] = parent_repair
        probabilities[0, 0] = .8
    parent = Prediction(probabilities, repairs, {"secret_prior_judgment": "must_not_be_sent"})
    rules = [{"id": "rA", "contract_id": "CA", "target": "length_cm", "kind": "single", "description": "Synthetic length rendering",
              "source": 'def check(row):\n    return row["length_cm"].endswith(" cm")'},
             {"id": "rB", "contract_id": "CB", "target": "role", "kind": "single", "description": "Synthetic role rendering",
              "source": 'def check(row):\n    return row["role"] == "old"'}]
    for rid in rules_extra:
        rules.append({"id": rid, "contract_id": "CJ", "target": "length_cm", "kind": "multi", "description": "Synthetic whole row interaction",
                      "source": 'def check(row):\n    return row["length_cm"] == "100" and row["role"] == "new"'})
    pass4 = {"dataset": table.name, "dirty_sha256": table.fingerprint, "rules": rules, "dependencies": []}
    contracts = {"dataset": table.name, "dirty_sha256": table.fingerprint,
                 "uniform_protocol_sha256": overlay.PROTOCOL_SHA256,
                 "columns": [{"contract_id": "CA", "column": "length_cm", "checker_ids": ["rA"]},
                             {"contract_id": "CB", "column": "role", "checker_ids": ["rB"]},
                             {"contract_id": "CJ", "column": "length_cm", "checker_ids": list(rules_extra)}]}
    proposals = {"dataset": table.name, "dirty_sha256": table.fingerprint, "max_candidates": 3, "cells": []}
    for column in scalar_columns:
        j = table.columns.index(column)
        source = table.rows[0][j]
        candidate = {"contract_id": "CA" if j == 0 else "CB", "transform_id": "normalize",
                     "value": "100" if j == 0 else "new", "score": .999,
                     "preservation": {"author_claim": "lossless"}, "explanation": "Exact synthetic rendering"}
        candidate["candidate_id"] = overlay.scalar_identity(table.name, 0, column, source, candidate)
        proposals["cells"].append({"row": 0, "column": column, "column_index": j,
                                    "source_value": source, "source_value_sha256": overlay.text_hash(source),
                                    "original_valid_score": .001, "candidates": [candidate],
                                    "application_precondition": {"parent_value_must_equal": source}})
    bundles = {"bundles": []}
    if bundle:
        authored = {"row": 0, "contract_id": "CA", "transform_id": "joint_normalize",
                    "source_values": {"length_cm": "100 cm", "role": "old"},
                    "updates": [{"column": "length_cm", "value": "100"}, {"column": "role", "value": "new"}],
                    "application_mode": "atomic_all_or_nothing", "requires_independent_review": True,
                    "application_precondition": {"parent_values_must_equal": {"length_cm": "100 cm", "role": "old"}}}
        authored["bundle_id"] = overlay.bundle_identity(table.name, authored)
        bundles["bundles"].append(authored)
    request = overlay.prepare_request(table, parent, pass4, contracts, proposals, bundles,
                                      bindings={"author_manifest_sha256": "manifest-hash"}, **DEPS)
    return table, parent, pass4, contracts, proposals, bundles, request


def review(request, *, select=True):
    reply = {key: request[key] for key in ("request_hash", "protocol_sha256", "artifact_schema_sha256")}
    reply.update(author_manifest_sha256=request["bindings"]["author_manifest_sha256"], model="gpt-6.1-sol",
                 reasoning_effort="xhigh", default_for_unreviewed="abstain_preserve_parent", benchmark_data_accessed=False,
                 contracts=[], scalar_candidates=[], atomic_bundles=[], cells=[])
    for contract in request["contracts"]:
        reply["contracts"].append({"contract_id": contract["contract_id"], "decision": "approve",
                                    "rationale": "Independent synthetic contract support",
                                    "approved_transform_ids": ["normalize", "joint_normalize"]})
    for cell in request["cells"]:
        for candidate in cell["candidates"]:
            reply["scalar_candidates"].append({"candidate_id": candidate["candidate_id"], "decision": "approve",
                                               "exact_value": candidate["author_candidate"]["value"],
                                               "verified_confidence": .95, "rationale": "Independent exact conversion"})
        selected = cell["candidates"][0] if select and cell["candidates"] else None
        reply["cells"].append({"cell_id": cell["cell_id"], "original_valid_score": .05,
                               "decision_confidence": .95, "selected_candidate_id": selected["candidate_id"] if selected else None,
                               "selected_value": selected["author_candidate"]["value"] if selected else None,
                               "invalidity_supported": True, "invalidity_rule_ids": ["rA" if cell["column"] == "length_cm" else "rB"],
                               "retract_parent_repair": False, "retraction_supported": False,
                               "rationale": "Independent original representation assessment"})
    for bundle in request["atomic_bundles"]:
        reply["atomic_bundles"].append({"bundle_id": bundle["bundle_id"], "decision": "approve",
                                        "rationale": "Independent exact full bundle", "atomicity_confirmed": True,
                                        "exact_updates": bundle["author_bundle"]["updates"], "verified_confidence": .95,
                                        "original_valid_scores": {"length_cm": .05, "role": .05},
                                        "decision_confidences": {"length_cm": .95, "role": .95},
                                        "invalidity_supported_by_column": {"length_cm": True, "role": True},
                                        "invalidity_rule_ids_by_column": {"length_cm": ["rA"], "role": ["rB"]}})
    return reply


def apply(state, reply):
    table, parent, *_, request = state
    return overlay.apply_overlay(table, parent, request, reply, rule_evaluator=evaluator)


def unchanged(parent, result):
    assert np.array_equal(parent.probabilities, result.probabilities)
    assert parent.repairs == result.repairs


def test_exact_approved_scalar_uses_four_dirty_only_deltas():
    state = fixture()
    request = state[-1]
    candidate = request["cells"][0]["candidates"][0]
    evidence = candidate["evidence"]
    assert len(evidence["delta_features"]) == 4
    assert evidence["delta_features"] == [1., 0., 0., 0.]
    assert evidence["raw_dirty_count_evidence"][0]["before_leave_one_out_value_count"] == 0
    assert evidence["raw_dirty_count_evidence"][0]["after_leave_one_out_value_count"] == 1
    assert len(evidence["single_column_evidence"]) == 2
    result = apply(state, review(request))
    assert result.repairs == {(0, 0): "100"}
    assert result.probabilities[0, 0] == .9
    assert "secret_prior_judgment" not in json.dumps(request)


@pytest.mark.parametrize("collection", ["contracts", "scalar_candidates", "cells"])
def test_missing_decision_fails_closed(collection):
    state = fixture()
    reply = review(state[-1])
    reply[collection].pop()
    with pytest.raises(ValueError, match="coverage"):
        apply(state, reply)


@pytest.mark.parametrize("field", ["request_hash", "protocol_sha256", "artifact_schema_sha256", "author_manifest_sha256"])
def test_stale_review_hash_rejected(field):
    state = fixture()
    reply = review(state[-1])
    reply[field] = "stale"
    with pytest.raises(ValueError):
        apply(state, reply)


@pytest.mark.parametrize("field", ["original_valid_score", "decision_confidence"])
def test_explicit_unknown_preserves_parent_even_with_candidate_approval(field):
    state = fixture(parent_repair="99")
    reply = review(state[-1])
    reply["cells"][0][field] = None
    unchanged(state[1], apply(state, reply))


def test_supported_invalidity_flags_ED_without_authorizing_repair():
    state = fixture()
    reply = review(state[-1], select=False)
    result = apply(state, reply)
    assert result.probabilities[0, 0] == .9
    assert result.repairs == {}


def test_units_need_independent_exact_approval_and_confidence():
    state = fixture()
    reply = review(state[-1])
    reply["scalar_candidates"][0]["decision"] = "abstain"
    reply["scalar_candidates"][0]["rationale"] = "Unit conversion semantics remain ambiguous"
    assert apply(state, reply).repairs == {}
    reply["scalar_candidates"][0]["decision"] = "approve"
    reply["scalar_candidates"][0]["verified_confidence"] = .79
    assert apply(state, reply).repairs == {}


def test_wrong_exact_scalar_value_or_ID_rejected():
    state = fixture()
    reply = review(state[-1])
    reply["scalar_candidates"][0]["exact_value"] = "1"
    with pytest.raises(ValueError, match="exact frozen"):
        apply(state, reply)
    reply = review(state[-1])
    reply["cells"][0]["selected_candidate_id"] = "invented"
    with pytest.raises(ValueError, match="ID/value"):
        apply(state, reply)


def test_exact_parent_precondition_blocks_scalar_and_preserves_inherited_repair():
    state = fixture(parent_repair="99")
    result = apply(state, review(state[-1]))
    assert result.repairs == {(0, 0): "99"}
    assert result.probabilities[0, 0] == .9  # independent raw invalidity can still flag ED


def test_current_parent_state_mismatch_rejected():
    state = fixture()
    state[1].probabilities[0, 0] = .7
    with pytest.raises(ValueError, match="parent binding"):
        apply(state, review(state[-1]))


def test_request_internal_rehash_does_not_replace_frozen_replay():
    state = fixture()
    request = deepcopy(state[-1])
    request["cells"][0]["candidates"][0]["evidence"]["delta_features"][0] = 1000.
    overlay.seal(request)
    with pytest.raises(ValueError, match="authoritative frozen-input replay"):
        overlay.validate_frozen_replay(*state[:6], request, expected_bindings=state[-1]["bindings"], **DEPS)


@pytest.mark.parametrize("key", ["author_manifest_sha256", "author_agent_id"])
def test_rehashed_request_cannot_replace_authoritative_identity_binding(key):
    state = fixture()
    request = deepcopy(state[-1])
    request["bindings"][key] = "forged-author-identity"
    overlay.seal(request)
    with pytest.raises(ValueError, match="authoritative frozen inputs"):
        overlay.validate_frozen_replay(*state[:6], request, expected_bindings=state[-1]["bindings"], **DEPS)


def test_candidate_content_hash_binds_unit_payload():
    state = fixture()
    proposals = deepcopy(state[4])
    proposals["cells"][0]["candidates"][0]["value"] = "1"
    with pytest.raises(ValueError, match="exact scalar"):
        overlay.prepare_request(*state[:4], proposals, state[5], **DEPS)


def test_atomic_bundle_applies_all_updates_without_flattening():
    state = fixture(scalar_columns=(), bundle=True)
    result = apply(state, review(state[-1]))
    assert result.repairs == {(0, 0): "100", (0, 1): "new"}
    assert np.all(result.probabilities[0] == .9)


def test_atomic_bundle_parent_mismatch_preserves_full_parent_tuple():
    state = fixture(scalar_columns=(), bundle=True, parent_repair="99")
    unchanged(state[1], apply(state, review(state[-1])))


def test_atomic_unknown_field_or_missing_atomicity_preserves_bundle():
    state = fixture(scalar_columns=(), bundle=True)
    reply = review(state[-1])
    reply["atomic_bundles"][0]["original_valid_scores"]["role"] = None
    unchanged(state[1], apply(state, reply))
    reply = review(state[-1])
    reply["atomic_bundles"][0]["atomicity_confirmed"] = False
    unchanged(state[1], apply(state, reply))


def test_scalar_bundle_overlap_fail_closed_and_order_independent():
    state = fixture(bundle=True)
    reply = review(state[-1])
    unchanged(state[1], apply(state, reply))
    reply["contracts"].reverse()
    reply["scalar_candidates"].reverse()
    unchanged(state[1], apply(state, reply))


def test_bundle_can_consume_ED_only_cell_decisions_without_false_overlap():
    state = fixture(scalar_columns=("length_cm", "role"), bundle=True)
    result = apply(state, review(state[-1], select=False))
    assert result.repairs == {(0, 0): "100", (0, 1): "new"}


def test_bundle_unchanged_context_ED_decision_is_not_silently_consumed():
    state = list(fixture(scalar_columns=("length_cm", "role"), bundle=True))
    bundle = state[5]["bundles"][0]
    bundle["updates"] = bundle["updates"][:1]
    bundle["bundle_id"] = overlay.bundle_identity(state[0].name, bundle)
    state[-1] = overlay.prepare_request(*state[:6], bindings={"author_manifest_sha256": "manifest-hash"}, **DEPS)
    unchanged(state[1], apply(state, review(state[-1], select=False)))


@pytest.mark.parametrize("extra", [("joint",), ("runtime",)])
def test_combined_whole_row_guard_rolls_back_both_ED_and_repairs(extra):
    state = fixture(scalar_columns=("length_cm", "role"), rules_extra=extra)
    reply = review(state[-1])
    # Individual safe replacements can become unsafe only in combination.
    result = apply(state, reply)
    unchanged(state[1], result)
    assert result.metadata["whole_row_rollbacks"][0]["restored"] == "exact_parent_row_probabilities_and_repairs"


def test_valid_original_requires_explicit_supported_parent_retraction():
    state = fixture(parent_repair="99")
    reply = review(state[-1], select=False)
    item = reply["cells"][0]
    item.update(original_valid_score=.99, selected_value="100 cm")
    unchanged(state[1], apply(state, reply))
    item.update(retract_parent_repair=True, retraction_supported=True)
    # Retraction introduces the approved format violation and thus rolls back
    # its ED pruning too, retaining exact parent coherence.
    unchanged(state[1], apply(state, reply))


def test_valid_raw_original_prunes_ED_without_mapping_score_to_probability():
    state = fixture()
    reply = review(state[-1], select=False)
    reply["cells"][0].update(original_valid_score=.91)
    result = apply(state, reply)
    assert result.probabilities[0, 0] == .1
    assert result.repairs == {}


def test_explicit_supported_retraction_restores_raw_and_prunes_ED_coherently():
    state = fixture(parent_repair="99")
    reply = review(state[-1], select=False)
    reply["contracts"][0]["decision"] = "reject"  # reviewer finds the format contract inapplicable
    reply["cells"][0].update(original_valid_score=.99, selected_value="100 cm",
                              retract_parent_repair=True, retraction_supported=True)
    result = apply(state, reply)
    assert result.repairs == {}
    assert result.probabilities[0, 0] == .1


def test_real_sandbox_and_raw_DirtyIndex_integration_uses_only_synthetic_inputs():
    state = fixture()
    table, parent, pass4, contracts, proposals, bundles, _ = state
    request = overlay.prepare_request(table, parent, pass4, contracts, proposals, bundles,
                                      bindings={"author_manifest_sha256": "manifest-hash"})
    assert all(item["accepted"] for item in request["rule_validation"])
    assert request["cells"][0]["candidates"][0]["evidence"]["delta_features"][0] == 1.
    result = overlay.apply_overlay(table, parent, request, review(request))
    assert result.repairs == {(0, 0): "100"}


def test_mixed_subscript_and_get_checker_references_both_columns():
    state = fixture(scalar_columns=("role",))
    table, parent, pass4, contracts, proposals, bundles, _ = state
    pass4["rules"][0]["source"] = 'def check(row):\n    return row["length_cm"] == "100 cm" and row.get("role") == "old"'
    request = overlay.prepare_request(table, parent, pass4, contracts, proposals, bundles,
                                      bindings={"author_manifest_sha256": "manifest-hash"})
    assert set(request["rule_scopes"]["rA"]) == {"length_cm", "role"}
    evidence = request["cells"][0]["candidates"][0]["evidence"]
    assert "rA" in evidence["relevant_rule_ids"]
    assert any(item["rule_id"] == "rA" for item in evidence["multi_column_evidence"])
    reply = review(request)
    reply["cells"][0]["invalidity_rule_ids"] = ["rA"]
    result = overlay.apply_overlay(table, parent, request, reply)
    assert result.repairs == {(0, 1): "new"}


def test_alias_or_dynamic_mapping_access_never_narrows_mixed_rule_scope():
    for source in ('def check(row):\n    alias = row\n    return row["length_cm"] == alias.get("role")',
                   'def check(row):\n    return row["length_cm"] == row.get(str(1))'):
        assert set(overlay._rule_columns({"source": source}, ("length_cm", "role"))) == {"length_cm", "role"}


def test_unsupported_invalidity_never_flags_or_repairs():
    state = fixture()
    reply = review(state[-1])
    reply["cells"][0]["invalidity_rule_ids"] = ["nonexistent"]
    unchanged(state[1], apply(state, reply))


def test_rejected_contract_cannot_supply_invalidity_or_repair():
    state = fixture()
    reply = review(state[-1])
    reply["contracts"][0]["decision"] = "reject"
    unchanged(state[1], apply(state, reply))


def test_ED_only_prune_without_any_approved_rule_restores_exact_parent():
    state = fixture()
    reply = review(state[-1], select=False)
    reply["cells"][0]["original_valid_score"] = .99
    for contract in reply["contracts"]:
        contract["decision"] = "reject"
    result = apply(state, reply)
    unchanged(state[1], result)
    assert result.metadata["whole_row_rollbacks"]


def test_ED_only_change_with_runtime_unknown_restores_exact_parent():
    state = fixture()
    reply = review(state[-1], select=False)
    reply["cells"][0]["original_valid_score"] = .99
    def unknown_evaluator(rules, records):
        matrix = evaluator(rules, records)
        for row in matrix:
            row[0] = -1
        return matrix
    result = overlay.apply_overlay(state[0], state[1], state[-1], reply, rule_evaluator=unknown_evaluator)
    unchanged(state[1], result)
    assert result.metadata["whole_row_rollbacks"][0]["runtime_unknown"]


def test_parent_repair_outside_ED_is_rejected():
    state = fixture(parent_repair="99")
    state[1].probabilities[0, 0] = .1
    with pytest.raises(ValueError, match="ED mask"):
        overlay.prepare_request(*state[:6], **DEPS)


def test_write_once_refuses_overwrite(tmp_path):
    path = tmp_path / "artifact.json"
    overlay.write_once(path, {"value": 1})
    with pytest.raises(FileExistsError):
        overlay.write_once(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


@pytest.mark.parametrize("path", ["../outside.json", "/tmp/outside.json"])
def test_prepared_manifest_path_escape_rejected_before_outside_read(tmp_path, monkeypatch, path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    overlay.write_once(prepared / "manifest.json", overlay.seal({"version": overlay.VERSION, "stage": "prepared",
                       "output_sha256": {path: "unused"}}, "manifest_hash"))
    reads = []
    original = overlay._read_bytes
    def track_reads(filename):
        reads.append(str(filename))
        return original(filename)
    monkeypatch.setattr(overlay, "_read_bytes", track_reads)
    with pytest.raises(ValueError, match="escapes"):
        overlay.apply_cli(prepared, tmp_path / "reviews", "/root/independent", tmp_path / "output")
    assert reads == [str(prepared / "manifest.json")]


def test_symlink_snapshot_and_live_input_paths_rejected_without_target_read(tmp_path):
    target = tmp_path / "outside.json"
    target.write_text("private synthetic data")
    root = tmp_path / "prepared"
    root.mkdir()
    (root / "link.json").symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        overlay.confined_path(root, "link.json")
    with pytest.raises(ValueError, match="Symlink"):
        overlay.read_json(root / "link.json")


def test_independent_provenance_binds_exact_review_file():
    request = fixture()[-1]
    request["bindings"]["author_agent_id"] = "/root/criteria_contrastive_review"
    raw = b"exact review bytes"
    provenance = {"reviewer_id": "/root/independent", "designated_by": "/root",
                  "reviewer_role": "independent_format_contract_verifier", "request_hash": request["request_hash"],
                  "verification_file_sha256": overlay.sha256_bytes(raw), "model": "gpt-6.1-sol",
                  "reasoning_effort": "xhigh", "benchmark_data_accessed": False}
    overlay.validate_verifier_provenance(request, raw, provenance, "/root/independent")
    with pytest.raises(ValueError, match="provenance"):
        overlay.validate_verifier_provenance(request, raw + b"stale", provenance, "/root/independent")
    provenance["reviewer_id"] = "/root/criteria_contrastive_review"
    with pytest.raises(ValueError, match="provenance"):
        overlay.validate_verifier_provenance(request, raw, provenance, provenance["reviewer_id"])


def test_six_dataset_CLI_snapshots_review_replay_and_no_overwrite(tmp_path, monkeypatch):
    """Exercise complete file protocol with synthetic tables and synthetic parents."""
    author = tmp_path / "author"
    author.mkdir()
    original_root = overlay.AUTHOR_ROOT
    for name in ("protocol.json", "artifact_schema.json"):
        (author / name).write_bytes((original_root / name).read_bytes())
    monkeypatch.setattr(overlay, "AUTHOR_ROOT", author)
    tables, paths = {}, {}
    for dataset in overlay.DATASETS:
        base = fixture()
        table = DirtyTable(dataset, base[0].columns, base[0].rows, base[0].fingerprint)
        tables[dataset] = table
        parent = base[1]
        parent.metadata["dirty_sha256"] = table.fingerprint
        path = tmp_path / "parents" / dataset / "parent.json"
        overlay.write_once(path, {**overlay.prediction_body(parent), "metadata": parent.metadata})
        paths[dataset] = path
        pass4, contracts, proposals, bundles = deepcopy(base[2:6])
        pass4["dataset"] = dataset
        contracts["dataset"] = dataset
        proposals["dataset"] = dataset
        for cell in proposals["cells"]:
            for candidate in cell["candidates"]:
                candidate["candidate_id"] = overlay.scalar_identity(dataset, cell["row"], cell["column"], cell["source_value"], candidate)
        bodies = {"pass4.json": pass4, "column_contracts.json": contracts, "proposals.json": proposals,
                  "joint_conversion_candidates.json": bundles, "runtime_validation.json": {"reports": []},
                  "sampling_provenance.json": {"synthetic": True}}
        for name, body in bodies.items():
            overlay.write_once(author / dataset / name, body)
    files = {str(path.relative_to(author)): overlay.sha256_bytes(path.read_bytes())
             for path in author.rglob("*") if path.is_file()}
    overlay.write_once(author / "manifest.json", {"files": files})
    overlay.write_once(author / "independent_review_request.json", {
        "author_manifest_sha256": overlay.sha256_bytes((author / "manifest.json").read_bytes())})
    monkeypatch.setattr(overlay, "load_dirty", lambda dataset: tables[dataset])
    monkeypatch.setattr(overlay, "_parent_path", lambda dataset: paths[dataset])
    prepared = tmp_path / "prepared"
    result = overlay.prepare_cli(prepared)
    assert result["application_status"] == "not_run"
    assert set(result["request_hashes"]) == set(overlay.DATASETS)
    reviews = tmp_path / "reviews"
    for dataset in overlay.DATASETS:
        request = overlay.read_json(prepared / dataset / "verification_request.json")
        assert "secret_prior_judgment" not in json.dumps(request)
        assert "secret_prior_judgment" in (prepared / dataset / "provenance/parent_prediction.json").read_text()
        review_path = reviews / dataset / "verification.json"
        overlay.write_once(review_path, review(request))
        overlay.write_once(reviews / dataset / "provenance.json", {
            "reviewer_id": "/root/independent", "reviewer_role": "independent_format_contract_verifier",
            "designated_by": "/root", "request_hash": request["request_hash"],
            "verification_file_sha256": overlay.sha256_bytes(review_path.read_bytes()),
            "model": "gpt-6.1-sol", "reasoning_effort": "xhigh", "benchmark_data_accessed": False})
    applied = tmp_path / "applied"
    overlay.apply_cli(prepared, reviews, "/root/independent", applied)
    output_manifest = overlay.read_json(applied / "manifest.json")
    assert output_manifest["evaluation_status"] == "not_run"
    for dataset in overlay.DATASETS:
        prediction = Prediction.load(applied / dataset / "prediction.json")
        assert prediction.repairs == {(0, 0): "100"}
    with pytest.raises(ValueError, match="overwrite"):
        overlay.prepare_cli(prepared)
    with pytest.raises(ValueError, match="overwrite"):
        overlay.apply_cli(prepared, reviews, "/root/independent", applied)
    # Source snapshots bind the authoritative originals, independently of the
    # prepared manifest's self-rehashed output map.
    manifest_path = prepared / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    for relative, pattern in ((f"{overlay.DATASETS[0]}/provenance/author/pass4.json", "author snapshot"),
                              ("provenance/author/manifest.json", "author snapshot"),
                              (f"{overlay.DATASETS[0]}/provenance/parent_prediction.json", "parent snapshot")):
        snapshot = prepared / relative
        original_snapshot = snapshot.read_bytes()
        snapshot.write_text('{"forged_snapshot": true}')
        forged_manifest = overlay.read_json(manifest_path)
        forged_manifest["output_sha256"][relative] = overlay.sha256_bytes(snapshot.read_bytes())
        overlay.seal(forged_manifest, "manifest_hash")
        manifest_path.write_text(json.dumps(forged_manifest))
        with pytest.raises(ValueError, match=pattern):
            overlay.apply_cli(prepared, reviews, "/root/independent", tmp_path / ("snapshot-forged-" + str(len(relative))))
        snapshot.write_bytes(original_snapshot)
        manifest_path.write_bytes(original_manifest)
    # Self-consistent rewrites of request+preparation hashes cannot substitute
    # a false author manifest or author identity into independent provenance.
    request_path = prepared / overlay.DATASETS[0] / "verification_request.json"
    original_request = request_path.read_bytes()
    forged = overlay.read_json(request_path)
    forged["bindings"]["author_manifest_sha256"] = "invented-author-manifest"
    forged["bindings"]["author_agent_id"] = "/root/somebody-else"
    overlay.seal(forged)
    request_path.write_text(json.dumps(forged))
    prepared_manifest = overlay.read_json(manifest_path)
    prepared_manifest["requests"][overlay.DATASETS[0]] = forged["request_hash"]
    prepared_manifest["output_sha256"][str(request_path.relative_to(prepared))] = overlay.sha256_bytes(request_path.read_bytes())
    overlay.seal(prepared_manifest, "manifest_hash")
    manifest_path.write_text(json.dumps(prepared_manifest))
    with pytest.raises(ValueError, match="authoritative frozen inputs"):
        overlay.apply_cli(prepared, reviews, "/root/independent", tmp_path / "forged-output")
    request_path.write_bytes(original_request)
    manifest_path.write_bytes(original_manifest)
    # A current parent file change is detected even if its parsed state is equal.
    paths[overlay.DATASETS[0]].write_text(paths[overlay.DATASETS[0]].read_text() + "\n")
    with pytest.raises(ValueError, match="current parent"):
        overlay.apply_cli(prepared, reviews, "/root/independent", tmp_path / "stale-output")
