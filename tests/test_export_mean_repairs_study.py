"""Synthetic-only immutable export checks; no benchmark results or labels are read."""
import hashlib
import json

import numpy as np
import pytest

from scripts import export_mean_repairs_study as exporter
from zeroclean.data import DirtyTable
from zeroclean.types import Prediction


def test_exact_v6_pin_and_predefined_comparison_scope():
    assert exporter.V6_MANIFEST_SHA256 == "56889e90d3d9d92f84bd5562ab776f0869e9b001dd5d13c4049834335881282c"
    assert exporter.COMPARISONS == ("latent_consensus_model_average_v5.json", "latent_consensus_feedback_v3.json",
        "latent_consensus_detection_feedback_v4.json", "source_v2_prior.json", "zeroed.json", "zeroed_zeroec_reconstruction.json")
    with pytest.raises(ValueError, match="lacks"):
        exporter.comparison_paths("unused", {"prediction_files": {}})


def test_recursive_binding_retains_live_and_snapshot_source_identities(tmp_path):
    source = tmp_path / "source.py"; source.write_text("immutable source\n")
    snapshot = tmp_path / "source_frozen.py"; snapshot.write_bytes(source.read_bytes())
    child = tmp_path / "child_manifest.json"
    child.write_text(json.dumps({"input_hashes": {str(source): exporter.sha256(source)},
                                 "frozen_input_hashes": {str(snapshot): exporter.sha256(snapshot)}}))
    parent = tmp_path / "parent_manifest.json"
    parent.write_text(json.dumps({"artifacts": {str(child): exporter.sha256(child)}}))
    inputs = {}
    exporter.binding.collect_bound_files(parent, exporter.sha256(parent), inputs)
    assert inputs[str(source.resolve())] == inputs[str(snapshot.resolve())]
    assert str(child.resolve()) in inputs
    source.write_text("changed live source\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        exporter.binding.collect_bound_files(parent, exporter.sha256(parent), {})


def test_prediction_blob_keeps_exact_bytes_and_checks_dirty_mask(tmp_path):
    table = DirtyTable("synthetic", ("x",), (("original",),), "synthetic-dirty-identity")
    path = tmp_path / "prediction.json"
    prediction = Prediction(np.array([[.9]]), {(0, 0): "candidate"}, {"dirty_sha256": table.fingerprint})
    prediction.save(path)
    original = path.read_bytes()
    inputs = {str(path.resolve()): hashlib.sha256(original).hexdigest()}
    blob, loaded = exporter.prediction_blob(path, table, inputs)
    assert blob == original and loaded.repairs == prediction.repairs
    prediction.metadata["dirty_sha256"] = "different"
    prediction.save(path)
    with pytest.raises(ValueError, match="identity"):
        exporter.prediction_blob(path, table, {str(path.resolve()): exporter.sha256(path)})
    prediction.metadata["dirty_sha256"] = table.fingerprint
    prediction.probabilities[0, 0] = .2
    prediction.save(path)
    with pytest.raises(ValueError, match="outside detection"):
        exporter.prediction_blob(path, table, {str(path.resolve()): exporter.sha256(path)})


def test_existing_immutable_publisher_is_idempotent_and_refuses_differing_bytes(tmp_path):
    path = tmp_path / "frozen.json"
    exporter.binding.publish_exact(path, b'{"frozen":true}\n')
    exporter.binding.publish_exact(path, b'{"frozen":true}\n')
    with pytest.raises(ValueError, match="overwrite"):
        exporter.binding.publish_exact(path, b'{"frozen":false}\n')
    assert path.read_bytes() == b'{"frozen":true}\n'


def test_flat_prior_closure_does_not_reroot_snapshot_relative_children(tmp_path):
    original = tmp_path / "original"; original.mkdir()
    snapshot = tmp_path / "snapshot"; snapshot.mkdir()
    leaf = original / "model_frozen.py"; leaf.write_text("frozen code\n")
    manifest = original / "artifact_manifest.json"
    manifest.write_text(json.dumps({"artifacts": {"model_frozen.py": exporter.sha256(leaf)}}))
    copied = snapshot / "artifact_manifest.json"; copied.write_bytes(manifest.read_bytes())
    frozen = {"inference_inputs": {str(manifest): exporter.sha256(manifest),
        str(leaf): exporter.sha256(leaf), str(copied): exporter.sha256(copied)}}
    inputs = {}
    exporter.bind_flat_prior_inputs(frozen, inputs, set())
    assert len(inputs) == 3
    assert not (snapshot / "model_frozen.py").exists()
    leaf.write_text("changed code\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        exporter.bind_flat_prior_inputs(frozen, {}, set())
