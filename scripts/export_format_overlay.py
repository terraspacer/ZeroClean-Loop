#!/usr/bin/env python3
"""Export a pinned completed format overlay and byte-identical source baselines.

Only structural/hash validation runs here: no sandbox, inference, model calls,
review execution or benchmark evaluation. Exact repeated exports write nothing.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experimental import format_overlay as adapter
from zeroclean.data import DATASETS, load_dirty
from zeroclean.types import Prediction, digest


PROJECT = Path(__file__).resolve().parents[1]
AUTHOR_ROOT = PROJECT / "experimental/format_contracts_v1"
PARENT_ROOT = PROJECT / "results/source_v2"
BASELINE_ROOT = PROJECT / "results/source_baselines_v1"
DEFAULT_OUTPUT = PROJECT / "results/format_overlay_v1"
VERSION = "format-contract-parent-overlay-export-v1"
BASELINES = ("zeroed.json", "zeroed_zeroec_reconstruction.json")
REVIEWER_ID = "/root/format_contracts_verifier"
ROOT_AUTHOR_FILES = ("protocol.json", "artifact_schema.json", "manifest.json", "independent_review_request.json")
BASELINE_FREEZE_SHA256 = {
    "hospital": "63d9b78a40105452960a7f9b97d1bbb3dbd90a509fe26000e2952cbbd887dc61",
    "flights": "94adf72302983b812d1c22f0b499a147a51bf2b9dcf71f2cca18de63a9f14a28",
    "beers": "ac856c57a805c837b3b7e9d6733cc1f0cdb88488c9922f9f6477fa9f8179fcf7",
    "rayyan": "672f082c61c6564031e632fdc48d4ee8d86f2c8d0116e57f9e582449f1fc5048",
    "billionaire": "210a634a0a304055cb40845471cb886132b4f5f368ced71ebb770245723f0f83",
    "movies": "5bf4e497e4a1f86f98352e321b9c2f49da20233b29f8f4f6932071006f660c7d",
}


def _json_bytes(body):
    return (json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _hash(raw):
    return adapter.sha256_bytes(raw)


def _read(path, inputs, expected=None):
    path = adapter._no_symlinks(path)
    raw = adapter._read_bytes(path)
    got = _hash(raw)
    if expected is not None and expected != got:
        raise ValueError("Frozen export input hash mismatch: " + str(path))
    if str(path) in inputs and inputs[str(path)] != got:
        raise ValueError("Frozen export input changed during validation")
    inputs[str(path)] = got
    return raw


def _prediction(raw, table):
    body = json.loads(raw)
    entries = body.get("repairs")
    if not isinstance(entries, list):
        raise ValueError("Prediction repair list missing")
    coordinates = []
    for entry in entries:
        i, j = entry.get("row"), entry.get("column")
        if type(i) is not int or type(j) is not int:
            raise ValueError("Prediction coordinates must be exact integers")
        coordinates.append((i, j))
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Duplicate prediction repair coordinate")
    result = Prediction(np.asarray(body["probabilities"]),
                        {(entry["row"], entry["column"]): entry["value"] for entry in entries},
                        body.get("metadata", {})).validate(table.shape)
    fingerprints = [result.metadata[key] for key in ("dirty_sha256", "input_fingerprint", "dirty_fingerprint")
                    if key in result.metadata]
    if not fingerprints or any(value != table.fingerprint for value in fingerprints):
        raise ValueError("Prediction dirty fingerprint mismatch")
    adapter.parent_rows(table, result)
    return result


def _applied_paths():
    names = {f"{dataset}/{name}" for dataset in DATASETS for name in
             ("prediction.json", "verification_request.json", "verification.json", "verification_provenance.json",
              "provenance/parent_prediction.json")}
    names |= {f"{dataset}/provenance/author/{name}" for dataset in DATASETS for name in adapter.AUTHOR_DATASET_FILES}
    names |= {f"provenance/author/{name}" for name in ROOT_AUTHOR_FILES}
    names.add("provenance/preparation_manifest.json")
    return names


def _prepared_paths():
    names = {f"{dataset}/{name}" for dataset in DATASETS for name in
             ("verification_request.json", "provenance/parent_prediction.json")}
    names |= {f"{dataset}/provenance/author/{name}" for dataset in DATASETS for name in adapter.AUTHOR_DATASET_FILES}
    names |= {f"provenance/author/{name}" for name in ROOT_AUTHOR_FILES}
    return names


def _validate_common(common, snapshots, inputs):
    if common.get("author_agent_id") != "/root/criteria_contrastive_review":
        raise ValueError("Applied author identity mismatch")
    implementation = adapter.implementation_hashes()
    if common.get("implementation_files") != implementation:
        raise ValueError("Applied adapter implementation hash mismatch")
    for path, expected in implementation.items():
        _read(path, inputs, expected)
    source_names = [*ROOT_AUTHOR_FILES, *[f"{dataset}/{name}" for dataset in DATASETS for name in adapter.AUTHOR_DATASET_FILES]]
    expected_paths = {str(AUTHOR_ROOT / name) for name in source_names}
    if set(common.get("author_files", {})) != expected_paths:
        raise ValueError("Applied source file bindings differ from declared author files")
    for name in source_names:
        path = AUTHOR_ROOT / name
        expected = common["author_files"][str(path)]
        raw = _read(path, inputs, expected)
        if "/" in name:
            dataset, filename = name.split("/", 1)
            relative = f"{dataset}/provenance/author/{filename}"
        else:
            relative = f"provenance/author/{name}"
        if snapshots[relative] != raw:
            raise ValueError("Applied author snapshot differs from authoritative source")
    if (common["author_manifest_sha256"] != common["author_files"][str(AUTHOR_ROOT / "manifest.json")]
            or common["author_files"][str(AUTHOR_ROOT / "protocol.json")] != adapter.PROTOCOL_SHA256
            or common["author_files"][str(AUTHOR_ROOT / "artifact_schema.json")] != adapter.SCHEMA_SHA256):
        raise ValueError("Applied author manifest/protocol/schema identity mismatch")
    author_manifest = json.loads(snapshots["provenance/author/manifest.json"])
    author_files = adapter._manifest_files(author_manifest)
    for name in source_names:
        if name in ("manifest.json", "independent_review_request.json"):
            continue
        expected = common["author_files"][str(AUTHOR_ROOT / name)]
        if author_files.get(name, author_files.get(str(AUTHOR_ROOT / name))) != expected:
            raise ValueError("Author manifest does not bind a required source artifact")
    dispatch = json.loads(snapshots["provenance/author/independent_review_request.json"])
    if dispatch.get("author_manifest_sha256", dispatch.get("manifest_sha256")) != common["author_manifest_sha256"]:
        raise ValueError("Author dispatch manifest binding mismatch")


def _source_request_checks(table, parent, request, source):
    """Check exact authored object membership without recomputing evidence."""
    for name in ("pass4.json", "column_contracts.json", "proposals.json"):
        body = source[name]
        if body.get("dataset") != table.name or body.get("dirty_sha256") != table.fingerprint:
            raise ValueError("Author source dirty fingerprint/dataset mismatch")
    if (request.get("column_contracts") != source["column_contracts.json"]
            or source["column_contracts.json"].get("uniform_protocol_sha256") != adapter.PROTOCOL_SHA256
            or request.get("all_authored_rules") != source["pass4.json"].get("rules", [])
            or request.get("dependencies") != source["pass4.json"].get("dependencies", [])):
        raise ValueError("Request rules/contracts differ from authored source")
    authored_rules = {rule["id"]: rule for rule in source["pass4.json"].get("rules", [])}
    if (request.get("rule_ids") != [rule["id"] for rule in request.get("rules", [])]
            or any(authored_rules.get(rule["id"]) != rule for rule in request.get("rules", []))):
        raise ValueError("Request accepted checker differs from exact authored source")
    authored_cells = source["proposals.json"].get("cells", [])
    if len(authored_cells) != len(request["cells"]):
        raise ValueError("Request scalar coverage differs from authored source")
    rows = adapter.parent_rows(table, parent)
    for authored, cell in zip(authored_cells, request["cells"]):
        i, j = adapter._coordinate(table, authored.get("row"), authored.get("column"))
        raw = table.rows[i][j]
        if (cell.get("author_cell") != authored or cell.get("author_cell_sha256") != digest(authored)
                or authored.get("column_index") != j or authored.get("source_value") != raw
                or authored.get("source_value_sha256") != adapter.text_hash(raw)
                or cell.get("raw_dirty_original") != raw or cell.get("exact_parent_value") != rows[i][j]
                or cell.get("row") != i or cell.get("column_index") != j or cell.get("column") != table.columns[j]):
            raise ValueError("Request scalar coordinate/original/parent differs from source")
        candidates = authored.get("candidates", [])
        if len(candidates) != len(cell["candidates"]) or len(candidates) > 3:
            raise ValueError("Request exact scalar alternative coverage mismatch")
        for candidate, item in zip(candidates, cell["candidates"]):
            if (item.get("author_candidate") != candidate or item.get("author_candidate_sha256") != digest(candidate)
                    or item.get("candidate_id") != candidate.get("candidate_id")
                    or candidate.get("candidate_id") != adapter.scalar_identity(table.name, i, table.columns[j], raw, candidate)):
                raise ValueError("Request scalar identity differs from exact authored alternative")
    authored_bundles = source["joint_conversion_candidates.json"].get("bundles", [])
    if len(authored_bundles) != len(request["atomic_bundles"]):
        raise ValueError("Request atomic coverage differs from authored source")
    for authored, item in zip(authored_bundles, request["atomic_bundles"]):
        if (item.get("author_bundle") != authored or item.get("author_bundle_sha256") != digest(authored)
                or item.get("bundle_id") != authored.get("bundle_id")
                or authored.get("bundle_id") != adapter.bundle_identity(table.name, authored)):
            raise ValueError("Request atomic identity differs from exact authored bundle")


def validate_applied(directory, manifest_sha256, tables, inputs):
    """Validate completed byte/source/review chain; never recompute decisions."""
    directory = adapter._no_symlinks(directory).resolve()
    if (not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in manifest_sha256)):
        raise ValueError("Exact caller-supplied applied manifest SHA256 is required")
    manifest_raw = _read(directory / "manifest.json", inputs, manifest_sha256)
    manifest = json.loads(manifest_raw)
    adapter.check_seal(manifest, "manifest_hash")
    if (manifest.get("version") != adapter.VERSION or manifest.get("stage") != "applied"
            or manifest.get("datasets") != list(DATASETS) or manifest.get("benchmark_data_accessed") is not False
            or manifest.get("evaluation_status") != "not_run"):
        raise ValueError("A completed unevaluated six-dataset applied overlay is required")
    output_hashes = manifest.get("output_sha256", {})
    if set(output_hashes) != _applied_paths():
        raise ValueError("Applied manifest must bind exactly the declared output snapshots")
    paths = {name: adapter.confined_path(directory, name) for name in output_hashes}
    snapshots = {name: _read(path, inputs, output_hashes[name]) for name, path in paths.items()}
    common = manifest.get("bindings", {})
    _validate_common(common, snapshots, inputs)
    preparation_raw = snapshots["provenance/preparation_manifest.json"]
    if _hash(preparation_raw) != manifest.get("prepared_manifest_sha256"):
        raise ValueError("Preparation manifest snapshot hash mismatch")
    preparation = json.loads(preparation_raw)
    adapter.check_seal(preparation, "manifest_hash")
    if (preparation.get("version") != adapter.VERSION or preparation.get("stage") != "prepared"
            or preparation.get("bindings") != common
            or set(preparation.get("requests", {})) != set(DATASETS)
            or set(preparation.get("output_sha256", {})) != _prepared_paths()):
        raise ValueError("Preparation source/request snapshot chain mismatch")
    for name, expected in preparation["output_sha256"].items():
        if _hash(snapshots[name]) != expected:
            raise ValueError("Applied snapshot differs from authoritative prepared snapshot")
    predictions, parents, requests = {}, {}, {}
    for dataset in DATASETS:
        table = tables[dataset]
        parent_path = PARENT_ROOT / "predictions" / dataset / "zeroclean_loop.json"
        parent_raw = _read(parent_path, inputs)
        if snapshots[f"{dataset}/provenance/parent_prediction.json"] != parent_raw:
            raise ValueError("Applied parent snapshot differs from current source_v2 parent")
        parent = _prediction(parent_raw, table)
        request = json.loads(snapshots[f"{dataset}/verification_request.json"])
        expected_bindings = deepcopy(common)
        expected_bindings["parent_file"] = {"path": str(parent_path), "sha256": _hash(parent_raw)}
        if request.get("bindings") != expected_bindings:
            raise ValueError("Request authoritative source/parent bindings mismatch")
        adapter.validate_request(table, parent, request)
        if request["request_hash"] != preparation["requests"][dataset]:
            raise ValueError("Request identity differs from prepared manifest")
        source_objects = {"pass4": "pass4.json", "column_contracts": "column_contracts.json",
                          "proposals": "proposals.json", "joint_candidates": "joint_conversion_candidates.json"}
        wanted = {key: digest(json.loads(snapshots[f"{dataset}/provenance/author/{name}"]))
                  for key, name in source_objects.items()}
        if request.get("author_artifacts_sha256") != wanted:
            raise ValueError("Request author artifacts differ from exact source snapshots")
        source = {name: json.loads(snapshots[f"{dataset}/provenance/author/{name}"])
                  for name in adapter.AUTHOR_DATASET_FILES}
        _source_request_checks(table, parent, request, source)
        review_raw = snapshots[f"{dataset}/verification.json"]
        review, provenance = json.loads(review_raw), json.loads(snapshots[f"{dataset}/verification_provenance.json"])
        # These helpers only validate schemas, hashes and exact coverage. They do
        # not execute sandbox rules, verification/model calls, or application.
        adapter.validate_verifier_provenance(request, review_raw, provenance, REVIEWER_ID)
        adapter.complete_verification(table, parent, request, review)
        prediction_raw = snapshots[f"{dataset}/prediction.json"]
        prediction = _prediction(prediction_raw, table)
        metadata = prediction.metadata
        if (metadata.get("system") != adapter.VERSION or metadata.get("dataset") != dataset
                or metadata.get("request_hash") != request["request_hash"]
                or metadata.get("review_sha256") != digest(review)
                or metadata.get("parent_snapshot_sha256") != request["parent_snapshot_sha256"]
                or metadata.get("policy") != adapter.policy_body()
                or metadata.get("independent_verifier_provenance") != provenance
                or metadata.get("benchmark_data_accessed") is not False
                or metadata.get("evaluation_status") != "not_run"):
            raise ValueError("Applied prediction provenance differs from exact request/review/parent")
        predictions[dataset], parents[dataset], requests[dataset] = prediction_raw, parent_raw, request
    return manifest, manifest_raw, snapshots, predictions, parents, requests


def _baselines(dataset, table, inputs):
    frozen_path = BASELINE_ROOT / "frozen" / f"{dataset}.json"
    raw = _read(frozen_path, inputs, BASELINE_FREEZE_SHA256[dataset])
    frozen = json.loads(raw)
    adapter.check_seal(frozen, "export_hash")
    if (frozen.get("dataset") != dataset or frozen.get("dirty_sha256") != table.fingerprint
            or frozen.get("export_schema") != "source-baselines-only-v1"
            or frozen.get("official_exact_ec_baseline_present") is not False
            or frozen.get("ground_truth_access") is not False
            or frozen.get("frozen_before_evaluation") is not True
            or set(frozen.get("prediction_files", {})) != set(BASELINES)
            or not isinstance(frozen.get("qualified_ec_caveat"), str) or not frozen["qualified_ec_caveat"].strip()):
        raise ValueError("Qualified source baseline freeze/fingerprint mismatch")
    blobs = {}
    for name in BASELINES:
        path = BASELINE_ROOT / "predictions" / dataset / name
        blob = _read(path, inputs, frozen["prediction_files"][name])
        prediction = _prediction(blob, table)
        meta = prediction.metadata
        if (meta.get("uses_ground_truth") is not False or meta.get("strict_zero_shot") is not True
                or meta.get("pending_cells", 0)):
            raise ValueError("Source baseline must be complete, dirty-only and use zero fixed examples")
        if name == "zeroed.json" and (meta.get("reproduced_upstream") is not True or prediction.repairs):
            raise ValueError("Original-source ED baseline identity mismatch")
        if name == "zeroed_zeroec_reconstruction.json" and (meta.get("baseline") != "zeroed_zeroec_reconstruction"
                                                           or meta.get("reproduced_upstream") is not False):
            raise ValueError("Qualified EC reconstruction identity mismatch")
        recorded = frozen.get("baseline_fidelity", {}).get(name)
        if not isinstance(recorded, dict) or any(meta.get(key) != value for key, value in recorded.items()):
            raise ValueError("Baseline fidelity metadata differs from original source freeze")
        blobs[name] = blob
    return frozen, raw, blobs


def _check_existing(output, blobs):
    actual = set()
    for path in output.rglob("*"):
        adapter._no_symlinks(path)
        if path.is_file():
            actual.add(str(path.relative_to(output)))
        elif not path.is_dir():
            raise ValueError("Unexpected nonregular export output")
    if actual != set(blobs):
        raise ValueError("Refusing to overwrite or complete a different export directory")
    for name, wanted in blobs.items():
        if adapter._read_bytes(adapter.confined_path(output, name)) != wanted:
            raise ValueError("Refusing to overwrite a different exported artifact")


def export(applied_directory, applied_manifest_sha256, output=DEFAULT_OUTPUT):
    applied = adapter._no_symlinks(applied_directory).resolve()
    output = adapter._no_symlinks(output).resolve()
    for source in (applied, AUTHOR_ROOT.resolve(), PARENT_ROOT.resolve(), BASELINE_ROOT.resolve()):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Export output must be separate from frozen source directories")
    inputs = {}
    tables = {dataset: load_dirty(dataset) for dataset in DATASETS}
    manifest, manifest_raw, snapshots, predictions, parents, requests = validate_applied(applied, applied_manifest_sha256, tables, inputs)
    _read(Path(__file__), inputs)
    _read(PROJECT / "tests/test_format_overlay_export.py", inputs)
    blobs = {"provenance/applied/manifest.json": manifest_raw,
             **{"provenance/applied/" + name: raw for name, raw in snapshots.items()}}
    freezes = {}
    for dataset in DATASETS:
        baseline_freeze, baseline_raw, baseline_predictions = _baselines(dataset, tables[dataset], inputs)
        dataset_blobs = {"zeroclean_loop.json": predictions[dataset], "source_v2_parent.json": parents[dataset], **baseline_predictions}
        for name, raw in dataset_blobs.items():
            blobs[f"predictions/{dataset}/{name}"] = raw
        blobs[f"provenance/source_baselines/{dataset}.json"] = baseline_raw
        frozen = {"export_schema": VERSION, "dataset": dataset, "dirty_sha256": tables[dataset].fingerprint,
                  "config": {"threshold": .5}, "prediction_files": {name: _hash(raw) for name, raw in dataset_blobs.items()},
                  "method_metadata": {"method_kind": "format_contract_parent_overlay", "primary_prediction": "zeroclean_loop.json",
                                      "parent_prediction": "source_v2_parent.json", "exact_applied_prediction_bytes": True},
                  "applied_manifest_sha256": applied_manifest_sha256,
                  "request_hash": requests[dataset]["request_hash"],
                  "baseline_source_frozen_sha256": _hash(baseline_raw),
                  "baseline_fidelity": baseline_freeze["baseline_fidelity"],
                  "qualified_ec_caveat": baseline_freeze["qualified_ec_caveat"],
                  "official_exact_ec_baseline_present": False, "matched_ablations_included": False,
                  "frozen_before_evaluation": True, "ground_truth_access": False,
                  "independent_verifier": REVIEWER_ID, "independent_verification_status": "complete",
                  "inference_reexecuted": False, "verification_reexecuted": False, "evaluation_status": "not_run"}
        freezes[dataset] = frozen
    # Each per-dataset freeze binds the complete common export chain, including
    # all six inputs. No arbitrary original baseline inference_inputs are read.
    for dataset, frozen in freezes.items():
        frozen["inference_inputs"] = dict(inputs)
        adapter.seal(frozen, "export_hash")
        blobs[f"frozen/{dataset}.json"] = _json_bytes(frozen)
    export_manifest = adapter.seal({"export_schema": VERSION, "datasets": list(DATASETS),
                      "applied_manifest_sha256": applied_manifest_sha256, "inference_inputs": inputs,
                      "output_sha256": {name: _hash(raw) for name, raw in blobs.items()},
                      "official_exact_ec_baseline_present": False, "ground_truth_access": False,
                      "inference_reexecuted": False, "verification_reexecuted": False,
                      "evaluation_status": "not_run"}, "export_hash")
    blobs["manifest.json"] = _json_bytes(export_manifest)
    adapter.check_file_hashes(inputs)
    if output.exists():
        _check_existing(output, blobs)
        adapter.check_file_hashes(inputs)
        return export_manifest
    output.mkdir(parents=True)
    for name, raw in blobs.items():
        adapter._snapshot_once(adapter.confined_path(output, name), raw)
    adapter.check_file_hashes(inputs)
    return export_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--applied", type=Path, required=True)
    parser.add_argument("--applied-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest = export(args.applied, args.applied_manifest_sha256, args.output)
    print(json.dumps({"output": str(args.output), "datasets": manifest["datasets"],
                      "export_hash": manifest["export_hash"], "evaluation_status": "not_run"}))


if __name__ == "__main__":
    main()
