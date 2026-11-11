"""First external round is a genuine frozen no-further-feedback ablation."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import export_no_further_iteration as ablation
from test_closed_loop import seed, response_callback
from zeroclean.closed_loop import advance_closed_loop
from zeroclean.data import sha256
from zeroclean.types import Prediction, write_json


def fixture(tmp_path, *, complete=True, max_rounds=3):
    table, run, *_ = seed(tmp_path, max_rounds=max_rounds)
    if complete:
        advance_closed_loop(table, run, source_signature="synthetic-implementation", request=response_callback([]))
    baselines = tmp_path / "development" / "predictions"
    Prediction(np.full(table.shape, .9), metadata={
        "input_fingerprint": table.fingerprint, "reproduced_upstream": True,
        "baseline": "zeroed_upstream", "fidelity": "upstream_function_extraction",
        "strict_zero_shot": True, "uses_ground_truth": False,
    }).save(baselines / table.name / "zeroed.json")
    Prediction(np.full(table.shape, .9), metadata={
        "dirty_fingerprint": table.fingerprint, "reproduced_upstream": False,
        "baseline": "zeroed_zeroec_reconstruction", "fidelity": "qualified_source_reconstruction",
        "strict_zero_shot": True, "uses_ground_truth": False, "prompt_profile": "author_zerodc_29cde9e0dbcd",
    }).save(baselines / table.name / "zeroed_zeroec_reconstruction.json")
    return table, run, baselines, tmp_path / "no_further_iteration"


def test_first_round_exact_state_and_bytes_with_initial_verification_retained(tmp_path, monkeypatch):
    table, run, baselines, output = fixture(tmp_path)
    first_path = run / "rounds" / "0001" / "prediction.json"
    first = Prediction.load(first_path)
    final = Prediction.load(run / "prediction.json")
    assert not np.array_equal(first.probabilities, final.probabilities)
    before = {path: path.read_bytes() for root in (run, baselines) for path in root.rglob("*.json")}
    def forbidden(*args, **kwargs): raise AssertionError("Ablation export must not run inference or models")
    monkeypatch.setattr(ablation.full_export.closed_loop, "infer_from_candidates", forbidden)
    monkeypatch.setattr(ablation.full_export.closed_loop, "queue_verification", forbidden)
    frozen = ablation.export_no_further_iteration(table, run, baselines, output)
    expected_sources = {"zeroclean_loop.json": run / "prediction.json", ablation.VARIANT: first_path,
                        **{name: baselines / table.name / name for name in ablation.full_export.BASELINES}}
    assert set(frozen["prediction_files"]) == set(expected_sources)
    for name, source in expected_sources.items():
        copied = output / "predictions" / table.name / name
        assert copied.read_bytes() == source.read_bytes()
        assert copied.stat().st_ino != source.stat().st_ino and not copied.is_symlink()
        assert sha256(copied) == frozen["prediction_files"][name]
        assert frozen["inference_inputs"][str(source.resolve())] == sha256(source)
    copied_first = Prediction.load(output / "predictions" / table.name / ablation.VARIANT)
    assert copied_first.repairs == first.repairs and np.array_equal(copied_first.probabilities, first.probabilities)
    assert copied_first.metadata["verified_cells"] > 0 and copied_first.metadata["verifier_coverage"] == 1
    protocol = frozen["ablation_protocol"]
    assert protocol["source_external_round"] == 1 and protocol["initial_verification_retained"] is True
    assert protocol["new_inference_executed_for_export"] is False
    assert protocol["shared_initial_verification_sha256"] == sha256(run / "rounds" / "0001" / "verification.json")
    assert protocol["first_round_prediction_sha256"] == sha256(first_path)
    assert "together" in protocol["comparison_limit"]
    assert frozen["official_exact_ec_baseline_present"] is False
    assert frozen["closed_loop"]["external_rounds"] == 3
    assert all(path.read_bytes() == body for path, body in before.items())
    freeze_bytes = (output / "frozen" / f"{table.name}.json").read_bytes()
    assert ablation.export_no_further_iteration(table, run, baselines, output) == frozen
    assert ablation.verify_export(table, output) == frozen
    assert (output / "frozen" / f"{table.name}.json").read_bytes() == freeze_bytes


def test_pending_run_is_rejected_without_new_inference(tmp_path, monkeypatch):
    table, run, baselines, output = fixture(tmp_path, complete=False)
    calls = []
    monkeypatch.setattr(ablation.full_export.closed_loop, "advance_closed_loop", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="pending.*cannot be exported"):
        ablation.export_no_further_iteration(table, run, baselines, output)
    assert calls == [] and not output.exists()


@pytest.mark.parametrize("source", ["first_prediction", "initial_verifier", "final_prediction"])
def test_tampered_source_state_is_rejected_before_export(tmp_path, source):
    table, run, baselines, output = fixture(tmp_path)
    path = {"first_prediction": run / "rounds" / "0001" / "prediction.json",
            "initial_verifier": run / "rounds" / "0001" / "verification.json",
            "final_prediction": run / "prediction.json"}[source]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="input changed"):
        ablation.export_no_further_iteration(table, run, baselines, output)
    assert not output.exists()


def test_single_round_full_run_cannot_claim_a_further_feedback_ablation(tmp_path):
    table, run, baselines, output = fixture(tmp_path, max_rounds=1)
    with pytest.raises(ValueError, match="at least two completed external rounds"):
        ablation.export_no_further_iteration(table, run, baselines, output)
    assert not output.exists()


def test_source_mutation_and_conflicting_output_are_detected(tmp_path):
    table, run, baselines, output = fixture(tmp_path)
    frozen = ablation.export_no_further_iteration(table, run, baselines, output)
    first_copy = output / "predictions" / table.name / ablation.VARIANT
    first_copy.write_text(first_copy.read_text() + "\n")
    with pytest.raises(ValueError, match="input changed"):
        ablation.verify_export(table, output)
    with pytest.raises(ValueError, match="overwrite a differing"):
        ablation.export_no_further_iteration(table, run, baselines, output)
    assert (output / "frozen" / f"{table.name}.json").exists()
    assert frozen["ablation_protocol"]["first_round_prediction_sha256"] == sha256(run / "rounds" / "0001" / "prediction.json")


def test_new_exporter_has_no_reference_evaluation_imports():
    modules = []
    for node in ast.walk(ast.parse(Path(ablation.__file__).read_text())):
        if isinstance(node, ast.Import): modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): modules.append(node.module or "")
    assert not any(module.endswith("evaluation") or module == "evaluate_e2e" for module in modules)
