"""Preregistered-processor tests use synthetic strings and frozen-schema fixtures."""
from dataclasses import asdict
import copy
import json

import numpy as np
import pytest

from experimental import verified_statistical_feedback as processor
from experimental.latent_consensus_feedback import FeedbackPolicy
from zeroclean.data import DirtyTable, sha256
from zeroclean.llm import ResponsePending
from zeroclean.types import Prediction, digest
from zeroclean.verification import queue_verification


def guard(source, kind="multi"):
    return {"id": "synthetic_guard", "target": "a", "kind": kind,
            "source": source, "description": "Synthetic prior-preservation rule"}


def fixture(version="v3", specs=None, prior_repairs=None, fixed_repairs=None, rows=None, rules=()):
    rows = tuple(rows or (("e0", "A", "B", "U"), ("e1", "X", "B", "U"),
                          ("e2", "A", "Y", "U"), ("e3", "A", "B", "U")))
    table = DirtyTable("synthetic", ("entity", "a", "b", "c"), rows, digest(rows))
    specs = specs or [(0, 1, "X")]
    prior_repairs = dict(prior_repairs or {})
    probabilities = np.full(table.shape, .13)
    for key in prior_repairs:
        probabilities[key] = .9
    prior = Prediction(probabilities, prior_repairs, {"dirty_sha256": table.fingerprint})
    if fixed_repairs is None:
        fixed_repairs = dict(prior_repairs)
        for i, j, value in specs:
            if value == rows[i][j]: fixed_repairs.pop((i, j), None)
            elif value not in {"uncertain", "boundary", "unsupported"}: fixed_repairs[i, j] = value
    fixed = Prediction(probabilities.copy(), dict(fixed_repairs), {
        "dirty_sha256": table.fingerprint, "system": processor.V3,
        "statistical_feedback_ablation": True, "policy": asdict(FeedbackPolicy())})
    original_rows = table.records()
    prior_rows, statistical_rows = processor._rows(table, prior.repairs), processor._rows(table, fixed.repairs)
    artifact = {"dirty_sha256": table.fingerprint, "rules": list(rules), "dependencies": []}
    cells = []
    for i, j, value in specs:
        original = rows[i][j]
        if version == "v3":
            interval = [.99, .995] if value == original else [0, 1e-6]
            endpoint = 1 - interval[0] if value == original else 1 - interval[1]
            evidence = {"row": i, "column_index": j, "column": table.columns[j], "original": original,
                        "candidate": value, "source_assumptions": {mode: {} for mode in processor.MODES}}
            trace = {**evidence, "statistical_evidence": evidence, "original_weight_interval": interval,
                     "candidate_weight_lower": .99 if value == original else .999999,
                     "feedback_probability": endpoint, "prior_probability": float(prior.probabilities[i, j]),
                     "prior_repair": prior.repairs.get((i, j)), "final_repair": fixed.repairs.get((i, j)),
                     "requested_repair": None if value == original else value,
                     "original_tuple": original_rows[i], "prior_tuple": prior_rows[i], "final_tuple": statistical_rows[i],
                     "individual_gate_blockers": [], "joint_rollback": False, "delta_features": [0.] * 4}
            alternatives = []
            for candidate in (value, prior.repairs.get((i, j))):
                if candidate is not None and candidate != original and candidate not in {c["value"] for c in alternatives}:
                    alternatives.append({"value": candidate, "score": .999999 if candidate == value else None})
            cells.append({"row": i, "column_index": j, "column": table.columns[j], "original": original,
                          "tuple": original_rows[i], "prior_inference_tuple": prior_rows[i], "tentative_tuple": statistical_rows[i],
                          "candidates": alternatives, "evidence": trace, "statistical_error_weight": endpoint})
        else:
            interval = ([.4, .6] if value == "uncertain" else [.5, .9] if value == "boundary" else
                        [.8, .9] if value == original else [.1, .2])
            error = [1 - interval[1], 1 - interval[0]]
            conclusion = "prune" if error[1] < .5 else "flag" if error[0] > .5 else "retain_uncertain"
            endpoint = error[1] if conclusion == "prune" else error[0] if conclusion == "flag" else None
            trace = {"row": i, "column_index": j, "column": table.columns[j], "original": original,
                     "source_assumptions": {mode: {"eligibility_blockers": [], "original_weight_interval": interval} for mode in processor.MODES},
                     "eligible_both_source_assumptions": True, "original_weight_interval": interval,
                     "error_weight_interval": error, "action": conclusion, "feedback_score": endpoint,
                     "prior_probability": float(prior.probabilities[i, j]), "original_tuple": original_rows[i]}
            if value == "unsupported":
                trace.update(eligible_both_source_assumptions=False, original_weight_interval=None,
                             error_weight_interval=None, action="retain_unsupported", feedback_score=None)
            cells.append({"row": i, "column_index": j, "column": table.columns[j], "original": original,
                          "original_tuple": original_rows[i], "prior_inference_tuple": prior_rows[i], "tentative_tuple": statistical_rows[i],
                          "detection_evidence": trace, "inherited_strict_ec_value": fixed.repairs.get((i, j)),
                          "correction_permission": "no changes beyond frozen v3 feedback repair map"})
    contracts = [{"attribute": name, "source_assumption": mode,
                  "contract": {"hypothetical": True, "single_valued": True, "evidence": f"synthetic {name} {mode}"}}
                 for name in ("a", "b", "c") for mode in sorted(processor.MODES)]
    bundle = {"experiment": processor.V3 if version == "v3" else processor.V4, "dataset": table.name,
              "dirty_sha256": table.fingerprint, "independent_verification_status": "not_run", "cells": cells,
              "frozen_semantic_contracts": contracts, "future_semantic_hypothesis": "Every group must see this exact assertion"}
    if version == "v3":
        bundle.update(model="gpt-6.1-sol", reasoning_effort="xhigh", config=asdict(FeedbackPolicy()),
                      rules=list(rules), delta_dimensions=["rule_satisfaction", "fd_support", "nmi_vicinity", "domain_support"])
    bundle = json.loads(json.dumps(bundle))
    bundle["request_hash"] = digest(bundle)
    return table, prior, fixed, artifact, bundle


def request_and_reply(data):
    table, prior, fixed, artifact, bundle = data
    payload = processor.prepare_payload(*data)
    reply = {k: payload[k] for k in ("request_hash", "model", "reasoning_effort", "dataset", "dirty_sha256")}
    reply["cells"] = []
    for cell in payload["cells"]:
        conclusion = cell["statistical_conclusion"]["original"]
        reply["cells"].append({"row": cell["row"], "column": cell["column"], "confidence": .9,
            "original_valid_score": .9 if conclusion == "valid" else .1 if conclusion == "invalid" else .5,
            "selected_value": cell["statistical_conclusion"]["eligible_new_repair"] if conclusion == "invalid" else None,
            "explanation": "Synthetic explicit semantic review, not model weight alone",
            "statistical_evidence_decision": "accept_original_" + conclusion if conclusion in {"valid", "invalid"} else "ambiguous",
            "accepted_semantic_contract_hashes": cell["feedback_verification_context"]["required_semantic_contract_hashes"],
            "candidates": [{"value": c["value"], "score": .9} for c in cell["candidates"]]})
    return payload, reply


def apply(data, payload, reply):
    return processor.apply_verified_feedback(*data[:4], payload, reply)


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_accepted_invalid_exact_repair_preserves_all_unsupported_cells_and_inputs(version):
    data = fixture(version, prior_repairs={(1, 3): " literal "})
    before = copy.deepcopy(data)
    payload, reply = request_and_reply(data)
    prediction, audit = apply(data, payload, reply)
    assert prediction.repairs == {(1, 3): " literal ", (0, 1): "X"}
    assert prediction.probabilities[0, 1] > .5
    for key in np.ndindex(data[0].shape):
        if key != (0, 1): assert prediction.probabilities[key] == data[1].probabilities[key]
    assert np.array_equal(data[1].probabilities, before[1].probabilities)
    assert data[1].repairs == before[1].repairs
    assert data[3:] == before[3:]
    assert audit["trace"][0]["repair_origin"] == "independently_accepted_frozen_v3_statistical_change"


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_original_valid_explicitly_removes_prior_repair(version):
    data = fixture(version, specs=[(0, 1, "A")], prior_repairs={(0, 1): "prior"})
    payload, reply = request_and_reply(data)
    prediction, _ = apply(data, payload, reply)
    assert prediction.repairs == {} and prediction.probabilities[0, 1] < .5


@pytest.mark.parametrize("change", ["reject", "ambiguous", "missing_verdict", "missing_contracts", "one_contract",
                                  "unknown_contract", "low_confidence", "score_boundary", "opposite_score", "opposite_verdict"])
def test_nonexplicit_or_conflicting_acceptance_restores_exact_prior_ed_and_ec(change):
    data = fixture(prior_repairs={(0, 1): "prior"})
    payload, reply = request_and_reply(data)
    cell = reply["cells"][0]
    if change in {"reject", "ambiguous"}: cell["statistical_evidence_decision"] = change
    elif change == "missing_verdict": cell.pop("statistical_evidence_decision")
    elif change == "missing_contracts": cell.pop("accepted_semantic_contract_hashes")
    elif change == "one_contract": cell["accepted_semantic_contract_hashes"] = cell["accepted_semantic_contract_hashes"][:1]
    elif change == "unknown_contract": cell["accepted_semantic_contract_hashes"] = ["unknown"]
    elif change == "low_confidence": cell["confidence"] = .799999
    elif change == "score_boundary": cell["original_valid_score"] = .5
    elif change == "opposite_score": cell["original_valid_score"] = .9
    else: cell["statistical_evidence_decision"] = "accept_original_valid"
    prediction, audit = apply(data, payload, reply)
    assert prediction.repairs == data[1].repairs
    assert np.array_equal(prediction.probabilities, data[1].probabilities)
    assert not audit["trace"][0]["ed_accepted"]


@pytest.mark.parametrize("prior_repair", [None, "prior exact"])
def test_explicit_null_abstains_only_from_new_ec_and_preserves_prior(prior_repair):
    data = fixture(prior_repairs={} if prior_repair is None else {(0, 1): prior_repair})
    payload, reply = request_and_reply(data)
    reply["cells"][0]["selected_value"] = None
    prediction, audit = apply(data, payload, reply)
    assert prediction.probabilities[0, 1] > .5
    assert prediction.repairs == data[1].repairs
    assert audit["trace"][0]["intended_action"] == "dirty_with_new_repair_abstention_preserve_prior"


def test_ec_score_threshold_does_not_discard_accepted_detection():
    data = fixture(prior_repairs={(0, 1): "prior"})
    payload, reply = request_and_reply(data)
    reply["cells"][0]["candidates"][0]["score"] = .799999
    prediction, _ = apply(data, payload, reply)
    assert prediction.probabilities[0, 1] > .98 and prediction.repairs == data[1].repairs
    reply["cells"][0]["candidates"][0]["score"] = .8
    reply["cells"][0]["confidence"] = .8
    prediction, _ = apply(data, payload, reply)
    assert prediction.repairs[0, 1] == "X"


@pytest.mark.parametrize("version", ["v3", "v4"])
@pytest.mark.parametrize("runtime", [False, True])
def test_simultaneous_multi_or_runtime_veto_rolls_back_to_actual_prior(version, runtime):
    source = "def check(row):\n    if row['a'] == 'X' and row['b'] == 'Y':\n        " + ("return int(row['a']) < 0" if runtime else "return True") + "\n    return False"
    data = fixture(version, specs=[(0, 1, "X"), (0, 2, "Y")],
                   prior_repairs={(0, 1): "P", (0, 3): "untouched"}, rules=[guard(source, "single" if runtime else "multi")])
    payload, reply = request_and_reply(data)
    prediction, audit = apply(data, payload, reply)
    assert prediction.repairs == data[1].repairs
    assert all(prediction.probabilities[0, j] > .5 for j in (1, 2))
    assert all(t["joint_rollback"] for t in audit["trace"])
    assert audit["joint_guard"][0]["rollback"]


@pytest.mark.parametrize("candidate", ["bad", "also bad"])
def test_constraints_restored_by_prior_get_safe_rollback_and_ed_ec_coherence(candidate):
    data = fixture(specs=[(0, 1, candidate)], rows=(("e0", "bad", "lock", "U"), ("e1", "fixed", "open", "U")),
                   prior_repairs={(0, 1): "fixed"}, rules=[guard("def check(row):\n    return row['a'] != 'fixed' and row['b'] == 'lock'")])
    payload, reply = request_and_reply(data)
    prediction, audit = apply(data, payload, reply)
    assert prediction.repairs == data[1].repairs
    assert audit["trace"][0]["joint_rollback"]
    if candidate == "bad":
        assert prediction.probabilities[0, 1] == .9
        assert audit["trace"][0]["coherence_ed_rollback"]
    else:
        assert prediction.probabilities[0, 1] > .98
    assert all(prediction.probabilities[key] >= .5 for key in prediction.repairs)


def test_runtime_constraint_restored_by_prior_rolls_back_original_valid_removal():
    rows = [("e0", "bad", "B", "U")] + [(f"e{i}", "1", "B", "U") for i in range(1, 25)]
    data = fixture(specs=[(0, 1, "bad")], rows=rows, prior_repairs={(0, 1): "1"},
                   rules=[guard("def check(row):\n    return int(row['a']) < 0", "single")])
    payload, reply = request_and_reply(data)
    prediction, audit = apply(data, payload, reply)
    assert prediction.repairs == data[1].repairs and prediction.probabilities[0, 1] == .9
    assert audit["joint_guard"][0]["new_runtime_rule_ids"] == ["synthetic_guard"]
    assert audit["trace"][0]["coherence_ed_rollback"]


@pytest.mark.parametrize("conclusion", ["uncertain", "boundary", "unsupported"])
def test_v4_uncertainty_or_strict_boundary_cannot_be_overruled_by_verifier(conclusion):
    data = fixture("v4", specs=[(0, 1, conclusion)], fixed_repairs={})
    payload, reply = request_and_reply(data)
    reply["cells"][0].update(statistical_evidence_decision="accept_original_invalid", original_valid_score=.1)
    prediction, _ = apply(data, payload, reply)
    assert np.array_equal(prediction.probabilities, data[1].probabilities)
    assert prediction.repairs == {}


def test_v4_detection_does_not_create_ec_when_no_frozen_v3_repair_exists():
    data = fixture("v4", fixed_repairs={})
    payload, reply = request_and_reply(data)
    assert payload["cells"][0]["candidates"] == []
    prediction, _ = apply(data, payload, reply)
    assert prediction.probabilities[0, 1] > .5 and prediction.repairs == {}


def test_frozen_v3_repairs_outside_original_bundle_coverage_are_not_silently_applied():
    data = fixture("v4", fixed_repairs={(0, 1): "X", (2, 2): "unreviewed v3 value"})
    payload, reply = request_and_reply(data)
    prediction, _ = apply(data, payload, reply)
    assert prediction.repairs == {(0, 1): "X"}
    assert prediction.probabilities[2, 2] == data[1].probabilities[2, 2]


def test_prior_fallback_is_not_claimed_to_be_literal_subset_of_frozen_v3():
    data = fixture("v4", specs=[(0, 1, "A")], prior_repairs={(0, 1): "source prior"}, fixed_repairs={})
    payload, reply = request_and_reply(data)
    reply["cells"][0]["statistical_evidence_decision"] = "reject"
    prediction, audit = apply(data, payload, reply)
    assert prediction.repairs == {(0, 1): "source prior"}
    assert audit["trace"][0]["repair_origin"] == "source_prior_fallback"
    assert prediction.metadata["prior_fallback_may_extend_beyond_v3_map"]


@pytest.mark.parametrize("field,value", [("confidence", float("nan")), ("confidence", 1.01), ("original_valid_score", -.01),
                                        ("confidence", True), ("confidence", None)])
def test_malformed_verifier_probabilities_reject_entire_application(field, value):
    data = fixture()
    payload, reply = request_and_reply(data)
    reply["cells"][0][field] = value
    with pytest.raises(ValueError): apply(data, payload, reply)


@pytest.mark.parametrize("change", ["missing_cell", "missing_candidate", "missing_selected", "invented_candidate", "old_request"])
def test_partial_or_stale_independent_judgments_are_never_fallback_completion(change):
    data = fixture()
    payload, reply = request_and_reply(data)
    if change == "missing_cell": reply["cells"] = []
    elif change == "missing_candidate": reply["cells"][0]["candidates"] = []
    elif change == "missing_selected": reply["cells"][0].pop("selected_value")
    elif change == "invented_candidate": reply["cells"][0]["selected_value"] = "not frozen"
    else: reply["request_hash"] = "stale"
    with pytest.raises(ValueError): apply(data, payload, reply)


def test_rehashed_projection_tamper_cannot_change_frozen_conclusion():
    data = fixture()
    payload, reply = request_and_reply(data)
    payload["cells"][0]["statistical_conclusion"]["error_weight_endpoint"] = .77
    payload["request_hash"] = digest({k: v for k, v in payload.items() if k != "request_hash"})
    reply["request_hash"] = payload["request_hash"]
    with pytest.raises(ValueError, match="projection"): apply(data, payload, reply)


def test_source_prior_repairs_outside_detection_mask_are_rejected_not_silently_rewritten():
    data = fixture(prior_repairs={(0, 1): "source prior"})
    data[1].probabilities[0, 1] = .1
    with pytest.raises(ValueError, match="detection mask"): processor.prepare_payload(*data)


@pytest.mark.parametrize("change", ["interval_nan", "interval_reversed", "v4_aggregate", "v4_blocker", "changed_v3_policy"])
def test_frozen_interval_policy_and_eligibility_mutation_are_rejected(change):
    data = fixture("v3" if change == "changed_v3_policy" else "v4")
    bundle = data[4]
    if change == "changed_v3_policy": bundle["config"]["repair_confidence"] = .7
    else:
        trace = bundle["cells"][0]["detection_evidence"]
        first = trace["source_assumptions"][sorted(processor.MODES)[0]]
        if change == "interval_nan": first["original_weight_interval"] = [0, float("nan")]
        elif change == "interval_reversed": first["original_weight_interval"] = [.8, .2]
        elif change == "v4_aggregate": trace["error_weight_interval"] = [.8, .99]
        else: first["eligibility_blockers"] = ["untrained_source"]
    if change == "interval_nan":
        with pytest.raises(ValueError): processor.prepare_payload(*data)
    else:
        bundle["request_hash"] = digest({k: v for k, v in bundle.items() if k != "request_hash"})
        with pytest.raises(ValueError): processor.prepare_payload(*data)


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_generic_verifier_surface_preserves_all_assumptions_and_extra_response_fields(version, tmp_path):
    data = fixture(version, specs=[(0, 1, "X"), (1, 2, "Y")])
    payload, reply = request_and_reply(data)
    seen = []
    def synthetic(stage, prompt, metadata):
        visible = json.loads(prompt.split("\n", 1)[1]); seen.append(visible)
        for cell in visible["cells"]:
            context = cell["feedback_verification_context"]
            assert context["frozen_bundle_context"]["frozen_semantic_contracts"] == data[4]["frozen_semantic_contracts"]
            assert context["frozen_bundle_context"]["future_semantic_hypothesis"] == data[4]["future_semantic_hypothesis"]
            assert "statistical_evidence_decision" in context["response_instructions"]
        keys = {(c["row"], c["column"]) for c in visible["cells"]}
        return json.dumps({"cells": [r for r in reply["cells"] if (r["row"], r["column"]) in keys]})
    completed = queue_verification(data[0], payload, synthetic, tmp_path / "verification.json", batch_size=1)
    assert len(seen) == 2
    assert completed["cells"][0]["statistical_evidence_decision"] == "accept_original_invalid"
    prediction, _ = apply(data, payload, completed)
    assert prediction.repairs == data[2].repairs


def file_fixture(tmp_path, version="v3"):
    data = fixture(version)
    table, prior, fixed, artifact, bundle = data
    source = tmp_path / "source"
    source.mkdir()
    paths = {k: source / f"{k}.json" for k in ("prior", "v3", "artifact", "bundle", "manifest")}
    paths["artifact"].write_text(json.dumps(artifact))
    fixed.metadata["input_hashes"] = {str(paths["artifact"]): sha256(paths["artifact"])}
    prior.save(paths["prior"])
    fixed.metadata["parent_prediction_sha256"] = sha256(paths["prior"])
    fixed.save(paths["v3"])
    bundle["input_hashes"] = {str(paths[k]): sha256(paths[k]) for k in ("prior", "v3", "artifact")}
    bundle["parent_variant_prediction_sha256"] = sha256(paths["v3"]) if version == "v3" else digest("v4-statistical-output")
    bundle["request_hash"] = digest({k: v for k, v in bundle.items() if k != "request_hash"})
    paths["bundle"].write_text(json.dumps(bundle))
    paths["manifest"].write_text(json.dumps({"artifacts": {str(paths["bundle"]): sha256(paths["bundle"])}}))
    return data, paths, tmp_path / "prepared/request.json", tmp_path / "result"


def prepare_files(data, paths, request):
    return processor.prepare_from_files(data[0], paths["bundle"], paths["manifest"], paths["prior"], paths["v3"], paths["artifact"], request)


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_immutable_file_application_freezes_exact_inputs_prediction_and_audit(version, tmp_path, monkeypatch):
    data, paths, request, output = file_fixture(tmp_path, version)
    original = {str(p): p.read_bytes() for p in paths.values()}
    prepared = prepare_files(data, paths, request)
    assert prepare_files(data, paths, request) == prepared
    _, reply = request_and_reply(data)
    reply["request_hash"] = prepared["request_hash"]
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps(reply))
    frozen = processor.apply_from_files(data[0], request, verification, output)
    name = f"verified_statistical_feedback_{version}.json"
    prediction_path = output / "predictions" / data[0].name / name
    assert frozen["prediction_files"] == {name: sha256(prediction_path)}
    assert processor.verify_output(data[0], output) == frozen
    assert Prediction.load(prediction_path).repairs == data[2].repairs
    for relative, wanted in frozen["frozen_input_copies"].items(): assert sha256(output / relative) == wanted
    assert {str(p): p.read_bytes() for p in paths.values()} == original
    def forbidden(*args, **kwargs): pytest.fail("Completed experiment must verify/reuse, never reapply")
    monkeypatch.setattr(processor, "apply_verified_feedback", forbidden)
    assert processor.apply_from_files(data[0], request, verification, output) == frozen


@pytest.mark.parametrize("tamper", ["source_sha", "partial_reply", "different_completion", "output_sha"])
def test_file_application_rejects_mutation_and_partial_outputs(tmp_path, tamper):
    data, paths, request, output = file_fixture(tmp_path)
    prepared = prepare_files(data, paths, request)
    _, reply = request_and_reply(data)
    reply["request_hash"] = prepared["request_hash"]
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps(reply))
    if tamper in {"different_completion", "output_sha"}:
        frozen = processor.apply_from_files(data[0], request, verification, output)
        if tamper == "output_sha":
            filename = next(iter(frozen["prediction_files"]))
            (output / "predictions" / data[0].name / filename).write_text("{}")
        else:
            reply["cells"][0]["statistical_evidence_decision"] = "reject"
            verification.write_text(json.dumps(reply))
    elif tamper == "source_sha": paths["prior"].write_text(paths["prior"].read_text() + " ")
    else:
        reply["cells"] = []
        verification.write_text(json.dumps(reply))
    with pytest.raises(ValueError): processor.apply_from_files(data[0], request, verification, output)
    if tamper in {"source_sha", "partial_reply"}: assert not output.exists()


@pytest.mark.parametrize("change", ["wrong_parent", "wrong_v3_map", "wrong_artifact"])
def test_explicit_file_roles_must_match_frozen_source_provenance(tmp_path, change):
    data, paths, request, _ = file_fixture(tmp_path)
    if change == "wrong_parent":
        prior = Prediction.load(paths["prior"])
        prior.probabilities[3, 3] = .17
        prior.save(paths["prior"])
        # Merely appearing as some historical hash cannot establish parent role.
        data[4]["input_hashes"]["unrelated_extra_input"] = sha256(paths["prior"])
    elif change == "wrong_v3_map":
        fixed = Prediction.load(paths["v3"])
        fixed.repairs[2, 2] = "new unreviewed correction"
        fixed.save(paths["v3"])
    else:
        paths["artifact"].write_text(json.dumps({**data[3], "unapproved": True}))
    if change == "wrong_parent":
        bundle = data[4]
        bundle["request_hash"] = digest({k: v for k, v in bundle.items() if k != "request_hash"})
        paths["bundle"].write_text(json.dumps(bundle))
        paths["manifest"].write_text(json.dumps({"artifacts": {str(paths["bundle"]): sha256(paths["bundle"])}}))
    with pytest.raises(ValueError): prepare_files(data, paths, request)
    assert not request.exists()
