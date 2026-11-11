"""Retrieval-only CTS-v2 policy tests; synthetic tables, embeddings and responses."""
import ast
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts import queue_cts_v2 as v2
from zeroclean.data import DirtyTable, sha256
from zeroclean.llm import ResponsePending
from zeroclean.loop import LoopConfig
from zeroclean.types import Prediction, digest, write_json
import zeroclean.upstream_zeroec as source


def table():
    return DirtyTable("synthetic", ("target", "group", "key"), (
        ("typo", "G", "q0"), ("fixed", "G", "q1"), ("fixed", "G", "q1"),
        ("other", "G", "q3"), ("solo", "H", "q4"), ("another", "H", "q5"),
        ("isolated", "I", "q6"),
    ), "synthetic-dirty")


def fixture_files(tmp_path, cells=None):
    dirty = table()
    coordinates = cells or [(0, 0), (4, 1), (1, 0), (6, 2)]
    seed_cells = [{"row": i, "column_index": j, "column": dirty.columns[j], "original": dirty.rows[i][j],
                   "tuple": dict(zip(dirty.columns, dirty.rows[i])), "candidates": [], "base_probability": .9}
                  for i, j in coordinates]
    payload = {"dataset": dirty.name, "dirty_sha256": dirty.fingerprint, "model": v2.MODEL,
               "reasoning_effort": v2.EFFORT, "config": asdict(LoopConfig()),
               "base_probabilities": np.full(dirty.shape, .9).tolist(), "cells": seed_cells}
    payload["request_hash"] = digest(payload)
    probabilities = np.ones(dirty.shape)
    probabilities[:, 1] = .9  # Nondegenerate classifier's all-positive decisions are retained.
    probabilities[:, 2] = 0.
    metadata = {"input_fingerprint": dirty.fingerprint, "reproduced_upstream": True,
                "classifier_columns": {"target": {"classes": [1]}, "group": {"classes": [0, 1]},
                                       "key": {"classes": [0]}},
                "upstream_binary_decisions": (probabilities > .5).tolist()}
    payload_path, detection_path = tmp_path / "source_seed.json", tmp_path / "zeroed.json"
    write_json(payload_path, payload)
    Prediction(probabilities, metadata=metadata).save(detection_path)
    return dirty, payload, payload_path, detection_path


@pytest.fixture
def tiny_source_cts(monkeypatch):
    pandas = pytest.importorskip("pandas")
    def helpers(_):
        return {"format_row_2": lambda row, header, mask: json.dumps({column: value for column, value, bad in zip(header, row, mask) if not bad}),
                "format_row": lambda row, header: json.dumps({column: row[column] for column in header}),
                "sort_dicts": lambda rows, *keys: sorted(rows, key=lambda row: tuple(row[key] for key in keys))}
    monkeypatch.setattr(source, "_dirty_frame", lambda dirty: pandas.DataFrame(dirty.rows, columns=dirty.columns))
    monkeypatch.setattr(source, "_source_helpers", helpers)
    monkeypatch.setattr(source, "_file_hash", lambda _: "synthetic-source-hash")
    monkeypatch.setattr(source, "_embedding_matrix", lambda dirty, frame, embedding, out: (
        np.arange(np.prod(dirty.shape) * 2, dtype=np.float32).reshape(*dirty.shape, 2),
        {"synthetic_model": "synthetic-embedding-hash"}))
    monkeypatch.setattr(source, "normalized_mutual_information", lambda frame, column: np.ones(frame.shape[1]))


def response(captured):
    def request(stage, prompt, metadata):
        evidence = json.loads(prompt.rsplit("\n", 1)[1])
        captured.append((stage, prompt, metadata, evidence))
        original = evidence["dirty_tuple"][evidence["target_column"]]
        replacement = {"target": "fixed" if original != "fixed" else "other",
                       "group": "G" if original != "G" else "H",
                       "key": "q0" if original != "q0" else "q3"}[evidence["target_column"]]
        return json.dumps({"original_valid_score": .1, "candidates": [
            {"value": replacement, "score": .8, "explanation": "Synthetic observed context; support remains fallible."}]})
    return request


def run(tmp_path, dirty, payload_path, detection_path, callback):
    return v2.queue_cts_v2(dirty, payload_path, detection_path, callback,
                           output_path=tmp_path / "proposals_cts_v2.json",
                           status_root=tmp_path / "status", cache_root=tmp_path / "cache")


@pytest.mark.parametrize("one_class", [1, 1.0])
def test_retrieval_mask_relaxes_only_declared_one_class_columns(one_class):
    dirty = DirtyTable("synthetic", ("one", "zero", "mixed", "unknown"),
                       (("a", "b", "c", "d"), ("e", "f", "g", "h")), "fingerprint")
    probabilities = np.array([[1., 0., .9, .8], [1., 0., .1, .9]])
    prediction = Prediction(probabilities.copy(), metadata={"input_fingerprint": dirty.fingerprint,
        "classifier_columns": {"one": {"classes": [one_class]}, "zero": {"classes": [0.]}, "mixed": {"classes": [0, 1]}},
        "upstream_binary_decisions": (probabilities > .5).tolist()})
    before_metadata = json.loads(json.dumps(prediction.metadata))
    mask, original, relaxed = v2.retrieval_only_mask(dirty, prediction)
    assert mask.tolist() == [[False, False, True, True], [False, False, False, True]]
    assert np.array_equal(original, probabilities > .5)
    assert np.array_equal(prediction.probabilities, probabilities) and prediction.metadata == before_metadata
    assert {item["column"] for item in relaxed} == {"one", "zero"}


def test_actual_cts_keeps_degenerate_peer_values_and_nondegenerate_mask(tmp_path, tiny_source_cts):
    dirty, payload, seed_path, detection_path = fixture_files(tmp_path, [(0, 0)])
    before_seed, before_source = seed_path.read_bytes(), detection_path.read_bytes()
    captured = []
    artifact = run(tmp_path, dirty, seed_path, detection_path, response(captured))
    stage, prompt, metadata, evidence = captured[0]
    assert stage == v2.STAGE and metadata["experiment_policy"] == v2.POLICY
    context = evidence["masked_source_context"]
    assert context["provenance"]["mask_query_target"] is True
    assert context["excluded_query_row"] is True and 0 not in context["indices"]
    assert all("target" in peer and "group" not in peer for peer in context["tuples"])
    assert all("group" in peer["tuple"] and peer["trust"] == "untrusted_dirty_observation"
               for peer in evidence["observed_dirty_peers"])
    assert "row grain" in prompt and "never proof" in prompt
    assert "Infer supported entity/context relationships and functional dependencies" in prompt
    assert "Rank the best-supported exact repairs" in prompt
    assert "when ambiguity remains, lower confidence or abstain" in prompt
    assert "Column names and observed formatting do not establish" not in prompt
    candidate = artifact["cells"][0]["candidates"][0]
    conditional = next(item for item in candidate["count_evidence"]["conditional_profiles"] if item["conditioning_column"] == "group")
    assert conditional["candidate_value_count"] == 2
    assert conditional["candidate_distinct_peer_tuple_count"] == 1
    assert artifact["retrieval_method"] == "zeroec_source_cts"
    assert artifact["policy_disclosure"]["source_predictions_modified"] is False
    assert seed_path.read_bytes() == before_seed and detection_path.read_bytes() == before_source
    assert [(cell["row"], cell["column"]) for cell in artifact["cells"]] == [(0, "target")]


def test_conditional_profiles_exclude_query_and_expose_duplicates_entropy():
    dirty = table()
    profiles = v2.ConditionalProfiles(dirty, 0, [(0, 1.), (1, .8), (2, .6)])
    values = profiles.profiles(0)
    group = next(item for item in values if item["conditioning_column"] == "group")
    assert group["peer_row_count"] == 3 and group["distinct_peer_tuple_count"] == 2
    assert group["duplicate_peer_rows"] == 1 and group["original_value_count"] == 0
    assert group["competing_target_values"] == 2
    assert group["top_observed_values"] == [
        {"value": "fixed", "count": 2, "distinct_peer_tuple_count": 1},
        {"value": "other", "count": 1, "distinct_peer_tuple_count": 1},
    ]
    assert math.isclose(group["entropy_bits"], -(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3))
    # The second exact duplicate remains a peer when one query occurrence is removed.
    duplicate_query = profiles.candidate_evidence(1, "fixed", [0, 2, 3])
    conditioned = next(item for item in duplicate_query["conditional_profiles"] if item["conditioning_column"] == "group")
    assert conditioned["candidate_value_count"] == 1 and conditioned["original_value_count"] == 1
    assert conditioned["candidate_distinct_peer_tuple_count"] == 1


def test_singleton_and_high_cardinality_groups_are_explicitly_weak_or_unsupported():
    dirty = table()
    profiles = v2.ConditionalProfiles(dirty, 0, [(1, .8), (2, .9)])
    singleton = next(item for item in profiles.profiles(4) if item["conditioning_column"] == "group")
    assert singleton["peer_row_count"] == 1 and singleton["evidence_status"] == "weak_singleton_peer_tuple"
    unsupported = next(item for item in profiles.profiles(6) if item["conditioning_column"] == "group")
    assert unsupported["peer_row_count"] == 0 and unsupported["evidence_status"] == "unsupported_no_peer_observations"
    high_cardinality = next(item for item in profiles.profiles(0) if item["conditioning_column"] == "key")
    assert high_cardinality["conditioning_cardinality_fraction"] == 5 / 6
    assert high_cardinality["conditioning_cardinality"] == 5 and high_cardinality["cardinality_population_rows"] == 6
    assert high_cardinality["peer_row_count"] == 0
    assert "no cutoff" in high_cardinality["cardinality_caution"]
    empty = DirtyTable("empty", ("target", "condition"), (("x", ""), ("y", "")), "empty")
    assert v2.ConditionalProfiles(empty, 0, [(1, 1.)]).profiles(0)[0]["evidence_status"] == "unsupported_empty_conditioning_value"


def test_column_traversal_restores_seed_cell_and_exact_request_hash_order(tmp_path, tiny_source_cts, monkeypatch):
    dirty, payload, seed_path, detection_path = fixture_files(tmp_path)
    original_cts = v2.SourceCTS
    instances = []
    def bounded(*args, **kwargs):
        assert kwargs["max_cached_columns"] == 2 and kwargs["exclude_query_row"] is True
        instance = original_cts(*args, **kwargs)
        instances.append(instance)
        return instance
    monkeypatch.setattr(v2, "SourceCTS", bounded)
    captured = []
    artifact = run(tmp_path, dirty, seed_path, detection_path, response(captured))
    assert [(metadata["row"], metadata["column"]) for _, _, metadata, _ in captured] == [
        (0, "target"), (1, "target"), (4, "group"), (6, "key")]
    identities = {(metadata["row"], metadata["column"]): digest({
        "stage": stage, "prompt": prompt, "metadata": metadata, "model": v2.MODEL, "reasoning_effort": v2.EFFORT})
        for stage, prompt, metadata, _ in captured}
    expected = [(cell["row"], cell["column"]) for cell in payload["cells"]]
    assert [(cell["row"], cell["column"]) for cell in artifact["cells"]] == expected
    assert artifact["request_hashes"] == [identities[key] for key in expected]
    assert artifact["seed_payload_sha256"] == sha256(seed_path)
    assert artifact["source_detection_sha256"] == sha256(detection_path)
    assert len(instances[0].column_states) == 2


def test_missing_responses_queue_every_cell_but_publish_no_completed_artifact(tmp_path, tiny_source_cts):
    dirty, payload, seed_path, detection_path = fixture_files(tmp_path)
    ids = []
    def pending(stage, prompt, metadata):
        ids.append((metadata["row"], metadata["column"]))
        raise ResponsePending("Synthetic pending response")
    with pytest.raises(ResponsePending, match="CTS-v2 requests"):
        run(tmp_path, dirty, seed_path, detection_path, pending)
    assert len(ids) == len(payload["cells"])
    assert not (tmp_path / "proposals_cts_v2.json").exists()
    status = json.loads(next((tmp_path / "status").rglob("status.json")).read_text())
    assert status["completed_cells"] == 0 and status["pending_cells"] == len(payload["cells"])
    assert len(status["attempted_requests"]) == len(payload["cells"])


def test_completed_artifact_is_immutable_and_source_changes_require_a_new_path(tmp_path, tiny_source_cts, monkeypatch):
    dirty, _, seed_path, detection_path = fixture_files(tmp_path)
    artifact = run(tmp_path, dirty, seed_path, detection_path, response([]))
    artifact_path = tmp_path / "proposals_cts_v2.json"
    before = artifact_path.read_bytes()
    def forbidden(*args, **kwargs): raise AssertionError("Completed artifact should reuse exact input identity")
    monkeypatch.setattr(v2, "SourceCTS", forbidden)
    assert run(tmp_path, dirty, seed_path, detection_path, forbidden) == artifact
    detection_path.write_text(detection_path.read_text() + "\n")
    with pytest.raises(ValueError, match="overwrite a different completed"):
        run(tmp_path, dirty, seed_path, detection_path, forbidden)
    assert artifact_path.read_bytes() == before


def test_new_stage_cannot_collide_with_original_cts_queue(tmp_path, tiny_source_cts):
    dirty, _, seed_path, detection_path = fixture_files(tmp_path, [(0, 0)])
    captured = []
    artifact = run(tmp_path, dirty, seed_path, detection_path, response(captured))
    stage, prompt, metadata, _ = captured[0]
    original_stage_hash = digest({"stage": "loop_cts_top3", "prompt": prompt, "metadata": metadata,
                                  "model": v2.MODEL, "reasoning_effort": v2.EFFORT})
    assert stage == "loop_cts_top3_v2" and artifact["request_hashes"][0] != original_stage_hash


def test_retry_artifact_records_successful_exact_request_identity(tmp_path, tiny_source_cts):
    dirty, _, seed_path, detection_path = fixture_files(tmp_path, [(0, 0)])
    captured, attempts = [], []
    valid = response(captured)
    def retry(stage, prompt, metadata):
        attempts.append(metadata["attempt"])
        if metadata["attempt"] == 0:
            return '{"original_valid_score":NaN,"candidates":[]}'
        return valid(stage, prompt, metadata)
    artifact = run(tmp_path, dirty, seed_path, detection_path, retry)
    assert attempts == [0, 1]
    stage, prompt, metadata, _ = captured[0]
    assert artifact["request_hashes"] == [digest({"stage": stage, "prompt": prompt, "metadata": metadata,
                                                 "model": v2.MODEL, "reasoning_effort": v2.EFFORT})]
    status = json.loads(next((tmp_path / "status").rglob("status.json")).read_text())
    assert len(status["response_errors"]) == 1 and len(status["attempted_requests"]) == 2


def test_script_imports_no_reference_evaluation_or_development_archive():
    modules = []
    for node in ast.walk(ast.parse(Path(v2.__file__).read_text())):
        if isinstance(node, ast.Import): modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): modules.append(node.module or "")
    assert not any(name.endswith("evaluation") or name == "evaluate_e2e" for name in modules)
