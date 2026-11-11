"""Synthetic adaptation tests: no model calls, fitting, tables or evaluation IO."""
from dataclasses import asdict
import copy
import json

import numpy as np
import pytest

from experimental import latent_consensus_feedback as feedback
from scripts import prepare_feedback_verification as adapter
from zeroclean.data import DirtyTable, sha256
from zeroclean.llm import ResponsePending
from zeroclean.loop import LoopConfig
from zeroclean.types import Prediction, digest
from zeroclean.verification import queue_verification


def fixture(tmp_path):
    rows = (("e0", "A", "B"), ("e1", "A", "B"), ("e2", "X", "Y"))
    table = DirtyTable("synthetic", ("entity", "a", "b"), rows, digest(rows))
    cells = []
    for i, j, value in ((0, 1, "X"), (0, 2, "Y"), (1, 1, "A")):
        same = value == rows[i][j]
        cells.append({"row": i, "column_index": j, "column": table.columns[j],
                      "original": rows[i][j], "candidate": value,
                      "original_weight_interval": [.99, .995] if same else [0, 1e-6],
                      "candidate_weight_lower": .99 if same else .999999,
                      "source_assumptions": {"independent_names": {}, "cluster_near_agreement": {}}})
    contracts = [{"attribute": column, "source_assumption": mode,
                  "contract": {"entity_columns": ["entity"], "source_column": "source",
                               "single_valued": True, "hypothetical": True,
                               "evidence": f"synthetic-assumption-{column}-{mode}"}}
                 for column in ("a", "b") for mode in ("independent_names", "cluster_near_agreement")]
    artifact = {"dirty_sha256": table.fingerprint, "rules": [], "dependencies": []}
    evidence = {"dirty_sha256": table.fingerprint, "cells": cells, "frozen_semantic_contracts": contracts,
                "input_hashes": {"historical-missing-input": digest("original-frozen-input")}}
    _, _, bundle = feedback.apply_feedback(table, Prediction(np.full(table.shape, .7)), artifact, evidence)
    bundle.pop("request_hash")
    bundle["parent_variant_prediction_sha256"] = digest("synthetic-prior-prediction")
    bundle["future_semantic_assumption"] = {"sentinel": "ALL-GROUPS-MUST-SEE-THIS"}
    bundle["request_hash"] = digest(bundle)
    bundle = json.loads(json.dumps(bundle))
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    source, manifest = frozen / "verification_bundle.json", frozen / "final_manifest.json"
    source.write_text(json.dumps(bundle))
    manifest.write_text(json.dumps({"artifacts": {str(source): sha256(source)}}))
    return table, bundle, source, manifest, tmp_path / "derived" / "request.json"


def rewrite_source(source, manifest, bundle):
    bundle["request_hash"] = digest({k: v for k, v in bundle.items() if k != "request_hash"})
    source.write_text(json.dumps(bundle))
    manifest.write_text(json.dumps({"artifacts": {str(source): sha256(source)}}))


def test_exact_immutable_derivation_preserves_all_frozen_evidence(tmp_path):
    _, bundle, source, manifest, output = fixture(tmp_path)
    original_bytes = source.read_bytes(), manifest.read_bytes()
    payload = adapter.prepare_feedback_verification(source, manifest, output)
    assert payload["request_hash"] != bundle["request_hash"]
    assert payload["request_hash"] == digest({k: v for k, v in payload.items() if k != "request_hash"})
    assert payload["config"] == bundle["config"] == json.loads(json.dumps(asdict(feedback.FeedbackPolicy())))
    assert payload["rules"] == bundle["rules"]
    assert payload["delta_dimensions"] == bundle["delta_dimensions"]
    for derived, original in zip(payload["cells"], bundle["cells"]):
        restored = copy.deepcopy(derived)
        context = restored.pop("feedback_verification_context")
        assert restored == original
        assert context["original_cell_hash"] == digest(original)
        assert context["frozen_bundle_context"]["frozen_semantic_contracts"] == bundle["frozen_semantic_contracts"]
        assert context["source_bundle_sha256"] == sha256(source)
        assert context["evidence_summary"]["final_repair"] == original["evidence"]["final_repair"]
    identity = payload["feedback_verification_preparation"]
    assert identity["input_file_hashes"] == {str(source): sha256(source), str(manifest): sha256(manifest)}
    assert identity["original_cells_hash"] == digest(bundle["cells"])
    assert not identity["source_run_validation_reperformed"] and not identity["historical_inputs_revalidated"]
    assert (source.read_bytes(), manifest.read_bytes()) == original_bytes
    output_bytes = output.read_bytes()
    assert adapter.prepare_feedback_verification(source, manifest, output) == payload
    assert output.read_bytes() == output_bytes


def test_every_generic_queue_group_sees_every_contract_and_all_hidden_semantics(tmp_path):
    table, bundle, source, manifest, output = fixture(tmp_path)
    payload = adapter.prepare_feedback_verification(source, manifest, output)
    seen = []
    def synthetic_reply(stage, prompt, metadata):
        evidence = json.loads(prompt.split("\n", 1)[1])
        seen.append(evidence)
        for cell in evidence["cells"]:
            context = cell["feedback_verification_context"]
            assert context["frozen_bundle_context"]["frozen_semantic_contracts"] == bundle["frozen_semantic_contracts"]
            assert context["frozen_bundle_context"]["future_semantic_assumption"] == bundle["future_semantic_assumption"]
            assert context["feedback_policy"] == bundle["config"]
            assert "not calibrated correctness" in " ".join(context["evidence_limitations"])
        return json.dumps({"cells": [{"row": c["row"], "column": c["column"],
                         "original_valid_score": .4, "selected_value": None, "confidence": .5,
                         "explanation": "Synthetic abstention: frozen semantics remain hypothetical.",
                         "candidates": [{"value": v["value"], "score": .5} for v in c["candidates"]]}
                        for c in evidence["cells"]]})
    verification = queue_verification(table, payload, synthetic_reply, tmp_path / "synthetic-verification.json", batch_size=1)
    assert [len(group["cells"]) for group in seen] == [2, 1]
    assert len(verification["cells"]) == len(bundle["cells"])
    assert verification["request_hash"] == payload["request_hash"]


def test_pending_queue_does_not_create_independent_judgments_or_change_source(tmp_path):
    table, _, source, manifest, output = fixture(tmp_path)
    payload = adapter.prepare_feedback_verification(source, manifest, output)
    original = source.read_bytes(), output.read_bytes()
    seen = []
    def pending(stage, prompt, metadata):
        seen.append(json.loads(prompt.split("\n", 1)[1]))
        raise ResponsePending("synthetic missing reply")
    verification = tmp_path / "verification.json"
    with pytest.raises(ResponsePending):
        queue_verification(table, payload, pending, verification, batch_size=1)
    assert len(seen) == 2 and not verification.exists()
    assert (source.read_bytes(), output.read_bytes()) == original


def test_unadapted_generic_queue_omits_top_level_frozen_contracts(tmp_path):
    table, bundle, _, _, _ = fixture(tmp_path)
    seen = []
    def pending(stage, prompt, metadata):
        visible = json.loads(prompt.split("\n", 1)[1])
        seen.append(visible)
        assert "frozen_semantic_contracts" not in visible
        assert "future_semantic_assumption" not in visible
        assert all("feedback_verification_context" not in c for c in visible["cells"])
        raise ResponsePending("synthetic omitted context")
    with pytest.raises(ResponsePending):
        queue_verification(table, bundle, pending, tmp_path / "unadapted.json", batch_size=1)
    assert len(seen) == 2


def test_feedback_request_is_explicitly_not_a_standard_loop_seed(tmp_path):
    _, _, source, manifest, output = fixture(tmp_path)
    payload = adapter.prepare_feedback_verification(source, manifest, output)
    assert payload["standard_closed_loop_compatible"] is False
    assert payload["feedback_verification_preparation"]["config_schema"] == "FeedbackPolicy, not LoopConfig"
    assert "base_probabilities" not in payload
    with pytest.raises(TypeError):
        LoopConfig(**payload["config"])


def test_preparation_never_loads_table_fits_or_selects_repairs(tmp_path, monkeypatch):
    _, _, source, manifest, output = fixture(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("The adapter may only prepare exact frozen evidence")
    monkeypatch.setattr(feedback, "apply_feedback", forbidden)
    monkeypatch.setattr(feedback.categorical, "_fit", forbidden)
    monkeypatch.setattr("zeroclean.data.load_dirty", forbidden)
    monkeypatch.setattr("zeroclean.verification.queue_verification", forbidden)
    adapter.prepare_feedback_verification(source, manifest, output)
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("change", ["self_hash", "manifest_hash", "missing_contracts", "one_source_contract",
                                   "coordinate", "duplicate_candidate", "loop_config", "wrong_effort",
                                   "unrelated_candidate", "evidence_tuple"])
def test_inconsistent_source_rejected_without_output(tmp_path, change):
    _, bundle, source, manifest, output = fixture(tmp_path)
    if change == "self_hash":
        bundle["purpose"] = "mutated"
        source.write_text(json.dumps(bundle))
        manifest.write_text(json.dumps({"artifacts": {str(source): sha256(source)}}))
    elif change == "manifest_hash":
        manifest.write_text(json.dumps({"artifacts": {str(source): "wrong"}}))
    else:
        if change == "missing_contracts":
            bundle.pop("frozen_semantic_contracts")
        elif change == "one_source_contract":
            bundle["frozen_semantic_contracts"].pop()
        elif change == "coordinate":
            bundle["cells"][0]["column_index"] = True
        elif change == "duplicate_candidate":
            bundle["cells"][0]["candidates"] *= 2
        elif change == "loop_config":
            bundle["config"] = asdict(LoopConfig())
        elif change == "wrong_effort":
            bundle["reasoning_effort"] = "low"
        elif change == "unrelated_candidate":
            bundle["cells"][0]["candidates"][0]["value"] = "unrelated string"
        elif change == "evidence_tuple":
            bundle["cells"][0]["tentative_tuple"]["a"] = "unrelated string"
        rewrite_source(source, manifest, bundle)
    with pytest.raises(ValueError):
        adapter.prepare_feedback_verification(source, manifest, output)
    assert not output.exists()


def test_no_overwrite_of_frozen_bundle_or_differing_derived_artifact(tmp_path):
    _, _, source, manifest, output = fixture(tmp_path)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="outside"):
        adapter.prepare_feedback_verification(source, manifest, source)
    assert source.read_bytes() == original
    output.parent.mkdir()
    output.write_text('{"other":"immutable experiment"}')
    original_output = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        adapter.prepare_feedback_verification(source, manifest, output)
    assert output.read_bytes() == original_output


def test_mid_preparation_source_mutation_rejected_before_publication(tmp_path, monkeypatch):
    _, _, source, manifest, output = fixture(tmp_path)
    validate = adapter._validate_bundle
    def mutate(bundle):
        validate(bundle)
        source.write_bytes(source.read_bytes() + b" ")
    monkeypatch.setattr(adapter, "_validate_bundle", mutate)
    with pytest.raises(ValueError, match="changed"):
        adapter.prepare_feedback_verification(source, manifest, output)
    assert not output.exists()


def test_implementation_signature_cannot_silently_rebind_existing_output(tmp_path, monkeypatch):
    _, _, source, manifest, output = fixture(tmp_path)
    synthetic_implementation = tmp_path / "synthetic-helper.py"
    synthetic_implementation.write_text("VERSION = 1\n")
    monkeypatch.setattr(adapter, "implementation_files", lambda: [synthetic_implementation])
    payload = adapter.prepare_feedback_verification(source, manifest, output)
    original = output.read_bytes()
    assert payload["feedback_verification_preparation"]["implementation_hashes"] == {
        str(synthetic_implementation): sha256(synthetic_implementation)}
    synthetic_implementation.write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="overwrite"):
        adapter.prepare_feedback_verification(source, manifest, output)
    assert output.read_bytes() == original
