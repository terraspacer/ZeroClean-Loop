"""Synthetic-only byte/provenance/export adversarial regression tests."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experimental import format_overlay as adapter
from scripts import export_format_overlay as exporter
from zeroclean.data import DirtyTable
from zeroclean.types import Prediction

import test_format_overlay as synthetic


def _write(path, body):
    raw = body if isinstance(body, bytes) else exporter._json_bytes(body)
    adapter._snapshot_once(path, raw)
    return raw


@pytest.fixture
def state(tmp_path, monkeypatch):
    author, parent_root, baselines = (tmp_path / name for name in ("author", "source_v2", "source_baselines_v1"))
    monkeypatch.setattr(exporter, "AUTHOR_ROOT", author)
    monkeypatch.setattr(exporter, "PARENT_ROOT", parent_root)
    monkeypatch.setattr(exporter, "BASELINE_ROOT", baselines)
    for name in ("protocol.json", "artifact_schema.json"):
        _write(author / name, (adapter.AUTHOR_ROOT / name).read_bytes())
    tables, parents, artifacts, parent_bytes = {}, {}, {}, {}
    for dataset in exporter.DATASETS:
        base = synthetic.fixture()
        table = DirtyTable(dataset, base[0].columns, base[0].rows, base[0].fingerprint)
        tables[dataset] = table
        parent = base[1]
        parent.metadata["dirty_sha256"] = table.fingerprint
        parents[dataset] = parent
        parent_bytes[dataset] = _write(parent_root / "predictions" / dataset / "zeroclean_loop.json",
                                        {**adapter.prediction_body(parent), "metadata": parent.metadata})
        pass4, contracts, proposals, bundles = deepcopy(base[2:6])
        for body in (pass4, contracts, proposals):
            body["dataset"] = dataset
        for cell in proposals["cells"]:
            for candidate in cell["candidates"]:
                candidate["candidate_id"] = adapter.scalar_identity(dataset, cell["row"], cell["column"], cell["source_value"], candidate)
        artifacts[dataset] = (pass4, contracts, proposals, bundles)
        bodies = dict(zip(("pass4.json", "column_contracts.json", "proposals.json", "joint_conversion_candidates.json"), artifacts[dataset]))
        bodies.update({"runtime_validation.json": {"reports": []}, "sampling_provenance.json": {"synthetic": True}})
        for name, body in bodies.items():
            _write(author / dataset / name, body)
    author_files = {str(path.relative_to(author)): adapter.sha256_bytes(path.read_bytes()) for path in author.rglob("*") if path.is_file()}
    author_manifest = _write(author / "manifest.json", {"files": author_files})
    author_manifest_sha = adapter.sha256_bytes(author_manifest)
    _write(author / "independent_review_request.json", {"author_manifest_sha256": author_manifest_sha})
    common = {"author_manifest_sha256": author_manifest_sha,
              "author_files": {str(path): adapter.sha256_bytes(path.read_bytes()) for path in author.rglob("*") if path.is_file()},
              "implementation_files": adapter.implementation_hashes(), "author_agent_id": "/root/criteria_contrastive_review"}
    prepared_blobs = {}
    root_blobs = {f"provenance/author/{name}": (author / name).read_bytes() for name in exporter.ROOT_AUTHOR_FILES}
    prepared_blobs.update(root_blobs)
    applied_blobs, requests = dict(root_blobs), {}
    for dataset in exporter.DATASETS:
        bindings = deepcopy(common)
        bindings["parent_file"] = {"path": str(parent_root / "predictions" / dataset / "zeroclean_loop.json"),
                                  "sha256": adapter.sha256_bytes(parent_bytes[dataset])}
        request = adapter.prepare_request(tables[dataset], parents[dataset], *artifacts[dataset], bindings=bindings, **synthetic.DEPS)
        reply = synthetic.review(request)
        review_bytes = exporter._json_bytes(reply)
        provenance = {"reviewer_id": exporter.REVIEWER_ID, "reviewer_role": "independent_format_contract_verifier",
                      "designated_by": "/root", "request_hash": request["request_hash"],
                      "verification_file_sha256": adapter.sha256_bytes(review_bytes), "model": "gpt-6.1-sol",
                      "reasoning_effort": "xhigh", "benchmark_data_accessed": False}
        prediction = adapter.apply_overlay(tables[dataset], parents[dataset], request, reply, rule_evaluator=synthetic.evaluator)
        prediction.metadata["independent_verifier_provenance"] = provenance
        requests[dataset] = request["request_hash"]
        common_blobs = {f"{dataset}/verification_request.json": exporter._json_bytes(request),
                        f"{dataset}/provenance/parent_prediction.json": parent_bytes[dataset],
                        **{f"{dataset}/provenance/author/{name}": (author / dataset / name).read_bytes() for name in adapter.AUTHOR_DATASET_FILES}}
        prepared_blobs.update(common_blobs)
        applied_blobs.update(common_blobs)
        applied_blobs.update({f"{dataset}/prediction.json": exporter._json_bytes({**adapter.prediction_body(prediction), "metadata": prediction.metadata}),
                              f"{dataset}/verification.json": review_bytes,
                              f"{dataset}/verification_provenance.json": exporter._json_bytes(provenance)})
    preparation = adapter.seal({"version": adapter.VERSION, "stage": "prepared", "bindings": common, "requests": requests,
                    "output_sha256": {name: adapter.sha256_bytes(raw) for name, raw in prepared_blobs.items()}}, "manifest_hash")
    preparation_bytes = exporter._json_bytes(preparation)
    applied_blobs["provenance/preparation_manifest.json"] = preparation_bytes
    applied = tmp_path / "applied"
    for name, raw in applied_blobs.items():
        _write(applied / name, raw)
    manifest = adapter.seal({"version": adapter.VERSION, "stage": "applied", "datasets": list(exporter.DATASETS),
                "bindings": common, "prepared_manifest_sha256": adapter.sha256_bytes(preparation_bytes),
                "output_sha256": {name: adapter.sha256_bytes(raw) for name, raw in applied_blobs.items()},
                "benchmark_data_accessed": False, "evaluation_status": "not_run"}, "manifest_hash")
    manifest_raw = _write(applied / "manifest.json", manifest)
    baseline_hashes = {}
    for dataset in exporter.DATASETS:
        table = tables[dataset]
        baseline_fidelity = {}
        hashes = {}
        for name in exporter.BASELINES:
            is_ed = name == "zeroed.json"
            metadata = {"dirty_sha256": table.fingerprint, "uses_ground_truth": False, "strict_zero_shot": True,
                        "reproduced_upstream": is_ed, "baseline": "zeroed_upstream" if is_ed else "zeroed_zeroec_reconstruction",
                        "fidelity": "upstream_function_extraction" if is_ed else "author_source_reconstruction"}
            prediction = Prediction(np.full(table.shape, .9), {} if is_ed else {(0, 0): "100"}, metadata)
            raw = _write(baselines / "predictions" / dataset / name, {**adapter.prediction_body(prediction), "metadata": metadata})
            hashes[name] = adapter.sha256_bytes(raw)
            baseline_fidelity[name] = {key: value for key, value in metadata.items() if key != "dirty_sha256"}
        freeze = adapter.seal({"export_schema": "source-baselines-only-v1", "dataset": dataset,
                  "dirty_sha256": table.fingerprint, "prediction_files": hashes, "baseline_fidelity": baseline_fidelity,
                  "qualified_ec_caveat": "Qualified reconstruction; historical identity unproven; zero fixed examples.",
                  "official_exact_ec_baseline_present": False, "ground_truth_access": False, "frozen_before_evaluation": True}, "export_hash")
        raw = _write(baselines / "frozen" / f"{dataset}.json", freeze)
        baseline_hashes[dataset] = adapter.sha256_bytes(raw)
    monkeypatch.setattr(exporter, "BASELINE_FREEZE_SHA256", baseline_hashes)
    monkeypatch.setattr(exporter, "load_dirty", lambda dataset: tables[dataset])
    return SimpleNamespace(applied=applied, pin=adapter.sha256_bytes(manifest_raw), author=author, parent=parent_root,
                           baselines=baselines, tables=tables, output=tmp_path / "export")


def _reseal_applied(state):
    manifest_path = state.applied / "manifest.json"
    body = json.loads(manifest_path.read_bytes())
    body["output_sha256"] = {name: adapter.sha256_bytes((state.applied / name).read_bytes()) for name in body["output_sha256"]}
    adapter.seal(body, "manifest_hash")
    manifest_path.write_bytes(exporter._json_bytes(body))
    return adapter.sha256_bytes(manifest_path.read_bytes())


def test_export_exact_bytes_all_six_baselines_parent_and_idempotence(state, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Exporter must not execute inference, sandbox or review")
    for name in ("apply_overlay", "prepare_request", "validate_frozen_replay", "evaluate_rules", "validate_rules", "load_author_inputs"):
        monkeypatch.setattr(adapter, name, forbidden)
    before = {path: path.read_bytes() for root in (state.applied, state.author, state.parent, state.baselines)
              for path in root.rglob("*") if path.is_file()}
    manifest = exporter.export(state.applied, state.pin, state.output)
    assert manifest["evaluation_status"] == "not_run"
    assert manifest["inference_reexecuted"] is False and manifest["verification_reexecuted"] is False
    assert manifest["official_exact_ec_baseline_present"] is False
    for dataset in exporter.DATASETS:
        directory = state.output / "predictions" / dataset
        sources = {"zeroclean_loop.json": state.applied / dataset / "prediction.json",
                   "source_v2_parent.json": state.parent / "predictions" / dataset / "zeroclean_loop.json",
                   **{name: state.baselines / "predictions" / dataset / name for name in exporter.BASELINES}}
        assert set(path.name for path in directory.iterdir()) == set(sources)
        frozen = json.loads((state.output / "frozen" / f"{dataset}.json").read_bytes())
        assert frozen["method_metadata"]["method_kind"] == "format_contract_parent_overlay"
        assert frozen["official_exact_ec_baseline_present"] is False
        assert "historical identity unproven" in frozen["qualified_ec_caveat"]
        for name, source in sources.items():
            assert (directory / name).read_bytes() == source.read_bytes()
            assert (directory / name).stat().st_ino != source.stat().st_ino
            assert frozen["prediction_files"][name] == adapter.sha256_bytes(source.read_bytes())
    snapshot = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in state.output.rglob("*") if path.is_file()}
    assert exporter.export(state.applied, state.pin, state.output) == manifest
    assert all(path.read_bytes() == raw and path.stat().st_mtime_ns == mtime for path, (raw, mtime) in snapshot.items())
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize("pin", [None, "", "stale", "0" * 64])
def test_caller_manifest_pin_required_exact_and_no_output(state, pin):
    with pytest.raises(ValueError, match="SHA256|hash mismatch"):
        exporter.export(state.applied, pin, state.output)
    assert not state.output.exists()


@pytest.mark.parametrize("relative", ["hospital/prediction.json", "hospital/verification.json",
                                     "hospital/verification_provenance.json", "hospital/verification_request.json",
                                     "hospital/provenance/author/pass4.json", "provenance/preparation_manifest.json"])
def test_stale_applied_snapshot_hash_rejected(state, relative):
    path = state.applied / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        exporter.export(state.applied, state.pin, state.output)
    assert not state.output.exists()


@pytest.mark.parametrize("role", ["author", "parent", "baseline", "baseline_freeze"])
def test_stale_author_parent_or_original_baseline_source_rejected(state, role):
    paths = {"author": state.author / "hospital/pass4.json",
             "parent": state.parent / "predictions/hospital/zeroclean_loop.json",
             "baseline": state.baselines / "predictions/hospital/zeroed.json",
             "baseline_freeze": state.baselines / "frozen/hospital.json"}
    path = paths[role]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch|parent snapshot"):
        exporter.export(state.applied, state.pin, state.output)
    assert not state.output.exists()


def test_newly_pinned_prediction_with_wrong_fingerprint_still_rejected(state):
    path = state.applied / "hospital/prediction.json"
    body = json.loads(path.read_bytes())
    body["metadata"]["dirty_sha256"] = "different-dirty"
    path.write_bytes(exporter._json_bytes(body))
    with pytest.raises(ValueError, match="fingerprint"):
        exporter.export(state.applied, _reseal_applied(state), state.output)


def test_newly_pinned_review_with_stale_provenance_still_rejected(state):
    path = state.applied / "hospital/verification.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="provenance"):
        exporter.export(state.applied, _reseal_applied(state), state.output)


def test_newly_pinned_author_snapshot_cannot_replace_authoritative_source(state):
    path = state.applied / "hospital/provenance/author/pass4.json"
    path.write_bytes(b'{"forged":true}')
    with pytest.raises(ValueError, match="author snapshot"):
        exporter.export(state.applied, _reseal_applied(state), state.output)


def test_newly_pinned_prediction_request_mismatch_rejected(state):
    path = state.applied / "hospital/prediction.json"
    body = json.loads(path.read_bytes())
    body["metadata"]["request_hash"] = "another-request"
    path.write_bytes(exporter._json_bytes(body))
    with pytest.raises(ValueError, match="prediction provenance"):
        exporter.export(state.applied, _reseal_applied(state), state.output)


def test_changed_export_or_partial_output_is_never_overwritten(state):
    exporter.export(state.applied, state.pin, state.output)
    path = state.output / "predictions/hospital/zeroclean_loop.json"
    path.write_bytes(b'{"unrelated":true}')
    with pytest.raises(ValueError, match="overwrite"):
        exporter.export(state.applied, state.pin, state.output)
    assert path.read_bytes() == b'{"unrelated":true}'


def test_manifest_traversal_and_symlink_inputs_rejected(state):
    manifest_path = state.applied / "manifest.json"
    body = json.loads(manifest_path.read_bytes())
    body["output_sha256"]["../outside.json"] = "0" * 64
    adapter.seal(body, "manifest_hash")
    manifest_path.write_bytes(exporter._json_bytes(body))
    with pytest.raises(ValueError, match="declared output"):
        exporter.export(state.applied, adapter.sha256_bytes(manifest_path.read_bytes()), state.output)
    link = state.applied.parent / "applied-link"
    link.symlink_to(state.applied, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        exporter.export(link, state.pin, state.output)


def test_exporter_imports_no_evaluation_and_never_resumes_inference():
    tree = ast.parse(Path(exporter.__file__).read_text())
    modules = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.append(node.func.attr)
    assert not any("evaluation" in module for module in modules)
    assert not set(calls).intersection({"apply_overlay", "prepare_request", "validate_frozen_replay", "evaluate_rules", "validate_rules"})
