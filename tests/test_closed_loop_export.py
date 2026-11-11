"""Safe evaluation export uses only synthetic predictions and dirty input."""
import ast
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import export_closed_loop as exporter
from zeroclean.closed_loop import advance_closed_loop, start_closed_loop
from zeroclean.data import DirtyTable, sha256
from zeroclean.loop import LoopConfig
from zeroclean.types import Prediction, digest, write_json


def inputs(tmp_path, *, complete=True):
    table = DirtyTable("synthetic", ("value",), (("bad",),), "synthetic-fingerprint")
    artifact = {"dataset": table.name, "dirty_sha256": table.fingerprint,
                "model": "gpt-6.1-sol", "reasoning_effort": "xhigh", "rules": [], "dependencies": []}
    payload = {"dataset": table.name, "dirty_sha256": table.fingerprint,
               "model": artifact["model"], "reasoning_effort": artifact["reasoning_effort"],
               "config": asdict(LoopConfig()), "artifact_hash": digest(artifact),
               "rules": [], "rule_validation": [], "base_probabilities": [[.9]],
               "cells": [{"row": 0, "column": "value", "column_index": 0, "original": "bad",
                          "tuple": {"value": "bad"}, "base_probability": .9,
                          "candidates": [{"value": "good", "score": .99, "delta_features": [0, 0, 0, 0],
                                          "introduced_violation": False, "counterfactual_error": False}]}]}
    payload["request_hash"] = digest(payload)
    response = {key: payload[key] for key in ("dataset", "dirty_sha256", "model", "reasoning_effort", "request_hash")}
    response["cells"] = [{"row": 0, "column": "value", "selected_value": "good", "confidence": .99,
                          "original_valid_score": .01, "candidates": [{"value": "good", "score": .99}]}]
    paths = [tmp_path / name for name in ("request.json", "verification.json", "pass2.json")]
    for path, body in zip(paths, (payload, response, artifact)):
        write_json(path, body)
    run = start_closed_loop(table, *paths, tmp_path / "runs", max_rounds=1 if complete else 3,
                            source_signature="synthetic-source")
    if complete:
        advance_closed_loop(table, run, source_signature="synthetic-source")
    baseline_root = tmp_path / "development" / "predictions"
    baseline_directory = baseline_root / table.name
    common = {"uses_ground_truth": False, "strict_zero_shot": True}
    Prediction(np.array([[.9]]), metadata={
        **common, "name": "zeroed_upstream", "baseline": "zeroed_upstream",
        "fidelity": "upstream_function_extraction", "reproduced_upstream": True,
        "input_fingerprint": table.fingerprint,
    }).save(baseline_directory / "zeroed.json")
    Prediction(np.array([[.9]]), {(0, 0): "good"}, metadata={
        **common, "name": "zeroed_zeroec_reconstruction", "baseline": "zeroed_zeroec_reconstruction",
        "fidelity": "qualified_source_reconstruction", "reproduced_upstream": False,
        "dirty_fingerprint": table.fingerprint, "pending_cells": 0,
        "prompt_profile": "author_zerodc_29cde9e0dbcd",
        "prompt_provenance": {"historical_identity": "unproven"},
    }).save(baseline_directory / "zeroed_zeroec_reconstruction.json")
    write_json(baseline_directory / "without_counterfactual.json", {"legacy": "never export"})
    return table, run, baseline_root, tmp_path / "source_v1"


def test_pending_run_is_rejected_without_inference_or_output(tmp_path, monkeypatch):
    table, run, baseline_root, output = inputs(tmp_path, complete=False)
    calls = []
    monkeypatch.setattr(exporter.closed_loop, "advance_closed_loop", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="pending.*cannot be exported"):
        exporter.export_closed_loop(table, run, baseline_root, output)
    assert calls == [] and not output.exists()


def test_export_is_byte_exact_separate_and_evaluator_compatible(tmp_path, monkeypatch):
    table, run, baseline_root, output = inputs(tmp_path)
    source_before = {path: path.read_bytes() for root in (run, baseline_root) for path in root.rglob("*.json")}
    def forbidden(*args, **kwargs):
        raise AssertionError("Export must not execute inference or a model")
    monkeypatch.setattr(exporter.closed_loop, "infer_from_candidates", forbidden)
    monkeypatch.setattr(exporter.closed_loop, "queue_verification", forbidden)
    frozen = exporter.export_closed_loop(table, run, baseline_root, output)
    sources = {"zeroclean_loop.json": run / "prediction.json", **{
        name: baseline_root / table.name / name for name in exporter.BASELINES}}
    assert set(frozen["prediction_files"]) == set(sources)
    assert frozen["dirty_sha256"] == table.fingerprint and frozen["frozen_before_evaluation"] is True
    assert frozen["closed_loop"]["external_rounds"] == 1
    assert len(frozen["closed_loop"]["history"]) == 1
    assert frozen["baseline_fidelity"]["zeroed.json"]["reproduced_upstream"] is True
    assert "historical identity unproven" in frozen["qualified_ec_caveat"]
    assert frozen["official_exact_ec_baseline_present"] is False
    assert frozen["matched_ablations_included"] is False
    for name, source in sources.items():
        copied = output / "predictions" / table.name / name
        assert copied.read_bytes() == source.read_bytes()
        assert copied.stat().st_ino != source.stat().st_ino and not copied.is_symlink()
        assert frozen["prediction_files"][name] == sha256(copied)
        assert frozen["inference_inputs"][str(source.resolve())] == sha256(source)
    assert not (output / "predictions" / table.name / "without_counterfactual.json").exists()
    assert all(path.read_bytes() == body for path, body in source_before.items())
    before = {path: path.read_bytes() for path in output.rglob("*.json")}
    assert exporter.export_closed_loop(table, run, baseline_root, output) == frozen
    assert exporter.verify_exported_experiment(table, output) == frozen
    assert all(path.read_bytes() == body for path, body in before.items())


def test_source_byte_mutation_invalidates_existing_export(tmp_path):
    table, run, baseline_root, output = inputs(tmp_path)
    exporter.export_closed_loop(table, run, baseline_root, output)
    source = baseline_root / table.name / "zeroed.json"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="input changed"):
        exporter.verify_exported_experiment(table, output)


def test_changed_closed_loop_source_hash_rejects_export(tmp_path):
    table, run, baseline_root, output = inputs(tmp_path)
    source = run / "rounds" / "0001" / "request.json"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="input changed"):
        exporter.export_closed_loop(table, run, baseline_root, output)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["missing_ec", "wrong_fingerprint", "adapted_detector", "pending_ec", "fixed_examples"])
def test_required_source_baselines_fail_closed(tmp_path, problem):
    table, run, baseline_root, output = inputs(tmp_path)
    directory = baseline_root / table.name
    if problem == "missing_ec":
        (directory / "zeroed_zeroec_reconstruction.json").unlink()
    else:
        name = "zeroed_zeroec_reconstruction.json" if problem in ("pending_ec", "fixed_examples") else "zeroed.json"
        path = directory / name
        body = json.loads(path.read_text())
        if problem == "wrong_fingerprint": body["metadata"]["input_fingerprint"] = "another-input"
        elif problem == "adapted_detector": body["metadata"]["reproduced_upstream"] = False
        elif problem == "pending_ec": body["metadata"]["pending_cells"] = 1
        else: body["metadata"]["strict_zero_shot"] = False
        write_json(path, body)
    with pytest.raises(ValueError, match="baseline is missing|fingerprint mismatch|original-source detector|pending source correction|zero fixed examples"):
        exporter.export_closed_loop(table, run, baseline_root, output)
    assert not output.exists()


def test_export_refuses_different_output_and_source_archive_root(tmp_path):
    table, run, baseline_root, output = inputs(tmp_path)
    conflicting = output / "predictions" / table.name / "zeroclean_loop.json"
    write_json(conflicting, {"different": "experiment"})
    before = conflicting.read_bytes()
    with pytest.raises(ValueError, match="overwrite a different"):
        exporter.export_closed_loop(table, run, baseline_root, output)
    assert conflicting.read_bytes() == before and not (output / "frozen").exists()
    with pytest.raises(ValueError, match="must be separate"):
        exporter.export_closed_loop(table, run, baseline_root, baseline_root.parent)


def test_export_script_has_no_evaluation_or_reference_imports():
    modules = []
    for node in ast.walk(ast.parse(Path(exporter.__file__).read_text())):
        if isinstance(node, ast.Import): modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): modules.append(node.module or "")
    assert not any(name.endswith("evaluation") or name == "evaluate_e2e" for name in modules)
