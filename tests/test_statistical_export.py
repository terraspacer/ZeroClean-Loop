"""Provenance failures must stop an export before benchmark evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("statistical_export", Path(__file__).parents[1] / "scripts/export_statistical_feedback.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recursively_bound_manifest_detects_changed_leaf(tmp_path):
    leaf = tmp_path / "evidence.txt"
    leaf.write_text("original")
    pre = tmp_path / "pre_run_manifest.json"
    pre.write_text(json.dumps({"input_hashes": {str(leaf): fingerprint(leaf)}}))
    final = tmp_path / "final_manifest.json"
    final.write_text(json.dumps({"artifacts": {str(pre): fingerprint(pre)}}))
    expected = fingerprint(final)
    inputs = {}
    exporter.collect_bound_files(final, expected, inputs)
    assert set(inputs) == {str(leaf), str(pre), str(final)}
    leaf.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        exporter.collect_bound_files(final, expected, {})


def test_rehashed_manifest_rejected_by_original_identity(tmp_path):
    final = tmp_path / "final_manifest.json"
    final.write_text('{"artifacts": {}}')
    expected = fingerprint(final)
    final.write_text('{"artifacts": {}, "modified": true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        exporter.collect_bound_files(final, expected, {})


def test_local_manifest_paths_are_bound(tmp_path):
    leaf = tmp_path / "model_frozen.py"
    leaf.write_text("frozen_model = True")
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text(json.dumps({"artifacts": {leaf.name: fingerprint(leaf)}}))
    inputs = {}
    exporter.collect_bound_files(manifest, fingerprint(manifest), inputs)
    assert inputs[str(leaf)] == fingerprint(leaf)


def test_publish_preserves_existing_different_prediction(tmp_path):
    path = tmp_path / "prediction.json"
    exporter.publish_exact(path, b"frozen")
    exporter.publish_exact(path, b"frozen")
    with pytest.raises(ValueError, match="overwrite"):
        exporter.publish_exact(path, b"different")
    assert path.read_bytes() == b"frozen"
