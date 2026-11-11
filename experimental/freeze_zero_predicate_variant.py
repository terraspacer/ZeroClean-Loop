"""Validate and atomically freeze this experiment's completed paired outputs.

Reads only this experiment's outputs, explicitly named source files, known raw
dirty inputs, and the exact recorded-reply/cache paths already used by the run.
No benchmark labels, metrics, masks, or older model predictions are consulted.
"""
from __future__ import annotations

import json
import importlib.metadata
from pathlib import Path
import sys

import numpy as np

from zeroclean.data import DATASETS, sha256
from zeroclean.types import digest, write_json
from experimental.zero_predicate_variant import OUTPUT, PROTOCOL, ROOT, array_hash


def read(path):
    return json.loads(Path(path).read_text())


def freeze() -> dict:
    run = read(OUTPUT / "run_manifest.json")
    progress = read(OUTPUT / "progress.json")
    implementation = ROOT / "experimental" / "zero_predicate_variant.py"
    source_runner = ROOT / "zeroclean" / "upstream_zeroed.py"
    addendum = OUTPUT / "execution_addendum.json"
    assert progress["stage"] == "complete" and progress["complete_datasets"] == list(DATASETS)
    assert sha256(implementation) == run["experimental_source_sha256"]
    assert sha256(PROTOCOL) == run["protocol_sha256"]
    assert sha256(addendum) == run["execution_addendum_sha256"]
    assert run["seed"] == 42 and run["python_hash_seed"] == "0"
    assert set(run["native_threads"].values()) == {"2"}
    manifest = {
        "schema_version": 1, "status": "complete",
        "experiment_id": "zeroed-empty-predicate-preserves-pseudoclean-v1",
        "experimental_unpromoted": True, "reproduced_upstream": False,
        "uses_ground_truth": False, "strict_zero_shot": True,
        "model": "gpt-6.1-sol", "reasoning_effort": "xhigh",
        "python_executable": sys.executable, "python_version": sys.version,
        "libraries_at_freeze": {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scikit-learn", "tqdm")},
        "implementation_path": str(implementation.resolve()), "implementation_sha256": sha256(implementation),
        "protocol_path": str(PROTOCOL.resolve()), "protocol_sha256": sha256(PROTOCOL),
        "execution_addendum_path": str(addendum.resolve()), "execution_addendum_sha256": sha256(addendum),
        "source_runner_path": str(source_runner.resolve()), "source_runner_sha256": sha256(source_runner),
        "freezer_path": str(Path(__file__).resolve()), "freezer_sha256": sha256(__file__),
        "seed": 42, "python_hash_seed": "0", "native_threads": run["native_threads"],
        "threshold": .5, "threshold_operator": ">",
        "training_capture_semantics": "Training NPZ features preserve source numeric values after float64 normalization for hashing; native source matrix dtype is not retained in these observer files. Actual fitting receives original source x unchanged. Exact scores, pseudo-labels/counts, source builders, and pre-filter cached-feature hashes are frozen separately.",
        "comparison_target": "fresh paired source-semantics control; not previously frozen source baseline",
        "limitations": [
            "Source asynchronous predicate selection can differ from older frozen source baselines.",
            "Propagated pseudo-clean training counts include any duplicate examples produced by source label propagation.",
            "Unaffected-column reproduction is vacuous when all columns hit the guard; counts identify those datasets.",
            "Pseudo-label correctness and changed prediction correctness remain unverified; no benchmark evaluation ran.",
        ],
        "datasets": {},
    }
    for name in DATASETS:
        directory = OUTPUT / name
        comparison_path = directory / "comparison.json"
        comparison = read(comparison_path)
        assert comparison["dataset"] == name
        for field in ("full_prefilter_state_identical", "unaffected_columns_exact_reproduction", "recorded_replies_identical"):
            assert comparison[field] is True
        source_hashes = {str(implementation.resolve()): sha256(implementation), str(source_runner.resolve()): sha256(source_runner)}
        predictions, states, replies = {}, {}, {}
        original_cache_hashes = []
        for mode in ("control", "variant"):
            mode_directory = directory / mode
            prediction_path = mode_directory / "prediction.json"
            prediction = read(prediction_path)
            metadata = read(mode_directory / "metadata.json")
            assert prediction["metadata"] == metadata
            assert metadata["input_fingerprint"] == comparison["input_sha256"]
            assert metadata["mode"] == mode and metadata["uses_ground_truth"] is False
            assert metadata["seed"] == 42 and metadata["python_hash_seed"] == "0"
            assert metadata["protocol_sha256"] == manifest["protocol_sha256"]
            assert metadata["execution_addendum_sha256"] == manifest["execution_addendum_sha256"]
            assert metadata["experimental_source_sha256"] == manifest["implementation_sha256"]
            assert metadata["extraction"]["source_runner_sha256"] == manifest["source_runner_sha256"]
            assert metadata["extraction"]["live_module_modified"] is False
            assert metadata["new_requests_queued"] == 0
            assert metadata["clustering_cache_validation"]["compatible"] is True
            if mode == "variant":
                assert metadata["reproduced_upstream"] is False
                assert metadata["fidelity"] == "minimal_empty_predicate_semantic_variant"
            with np.load(mode_directory / "frozen_scores.npz", allow_pickle=False) as frozen:
                assert frozen["scores"].shape == tuple(comparison["shape"])
                assert frozen["columns"].tolist() == list(comparison["columns"])
                np.testing.assert_array_equal(frozen["scores"], np.asarray(prediction["probabilities"]))
                np.testing.assert_array_equal(frozen["binary_decisions"], frozen["scores"] > .5)
                np.testing.assert_array_equal(frozen["binary_decisions"], np.asarray(metadata["upstream_binary_decisions"]))
            for filename in (
                "prediction.json", "metadata.json", "frozen_scores.npz", "prefilter_state.json",
                "prefilter_state_provenance.json", "training_trace.json", "model_trace.json",
                "filter_trace.json", "recorded_reply_provenance.json", "cache_provenance.json", "preflight.json",
            ):
                path = mode_directory / filename
                source_hashes[str(path.resolve())] = sha256(path)
            for trace in read(mode_directory / "training_trace.json").values():
                path = mode_directory / trace["training_file"]
                source_hashes[str(path.resolve())] = sha256(path)
                with np.load(path, allow_pickle=False) as training:
                    assert array_hash(training["features"]) == trace["feature_sha256"]
                    assert array_hash(training["pseudo_labels"]) == trace["pseudo_label_sha256"]
                    assert list(training["features"].shape) == trace["feature_shape"]
            for filename, expected in metadata["source_sha256"].items():
                path = ROOT / "vendor" / "ZeroED" / filename
                assert sha256(path) == expected
                source_hashes[str(path.resolve())] = expected
            preflight = read(mode_directory / "preflight.json")
            prepared = preflight["fasttext_prepared_provenance"]
            model_path = Path(preflight["fasttext_path"]).resolve()
            assert sha256(model_path) == prepared["model_sha256"]
            source_hashes[str(model_path)] = prepared["model_sha256"]
            prepared_manifest = Path(prepared["manifest_path"]).resolve()
            source_hashes[str(prepared_manifest)] = sha256(prepared_manifest)
            states[mode] = read(mode_directory / "prefilter_state.json")
            state_provenance = read(mode_directory / "prefilter_state_provenance.json")
            assert digest(states[mode]) == state_provenance["full_state_sha256"]
            replies[mode] = read(mode_directory / "recorded_reply_provenance.json")
            for reply in replies[mode]:
                for kind, hash_field in (("requests", "request_sha256"), ("responses", "response_sha256")):
                    path = ROOT / "artifacts" / "queue" / kind / (reply["request_hash"] + ".json")
                    assert sha256(path) == reply[hash_field]
                    source_hashes[str(path.resolve())] = reply[hash_field]
            cache = read(mode_directory / "cache_provenance.json")
            assert cache["original_unchanged"] is True and sha256(cache["original_path"]) == cache["original_sha256"]
            original_cache_hashes.append(cache["original_sha256"])
            source_hashes[str(Path(cache["original_path"]).resolve())] = cache["original_sha256"]
            predictions[mode] = prediction
        assert original_cache_hashes[0] == original_cache_hashes[1]
        assert digest(states["control"]) == digest(states["variant"])
        assert replies["control"] == replies["variant"]
        control_scores = np.asarray(predictions["control"]["probabilities"])
        variant_scores = np.asarray(predictions["variant"]["probabilities"])
        control_training = read(directory / "control" / "training_trace.json")
        variant_training = read(directory / "variant" / "training_trace.json")
        for j, attr in enumerate(comparison["columns"]):
            trace = comparison["columns"][attr]
            if not trace["condition_altered"]:
                assert control_training[attr]["feature_sha256"] == variant_training[attr]["feature_sha256"]
                assert control_training[attr]["pseudo_label_sha256"] == variant_training[attr]["pseudo_label_sha256"]
                np.testing.assert_array_equal(control_scores[:, j], variant_scores[:, j])
        dirty_path = ROOT / "vendor" / "ZeroED" / "data" / f"{name}_error-01.csv"
        assert sha256(dirty_path) == comparison["input_sha256"]
        source_hashes[str(dirty_path.resolve())] = comparison["input_sha256"]
        source_hashes[str(comparison_path.resolve())] = sha256(comparison_path)
        manifest["datasets"][name] = {
            "input_path": str(dirty_path.resolve()), "input_sha256": comparison["input_sha256"],
            "shape": comparison["shape"], "altered_columns": comparison["altered_columns"],
            "altered_column_training_counts": {
                attr: {"preserved_source_pseudoclean_examples": comparison["columns"][attr]["guard_preserved_rows"],
                       "control": comparison["columns"][attr]["control_training_label_counts"],
                       "variant": comparison["columns"][attr]["variant_training_label_counts"]}
                for attr in comparison["altered_columns"]
            },
            "altered_column_count": len(comparison["altered_columns"]),
            "unaffected_column_count": sum(not column["condition_altered"] for column in comparison["columns"].values()),
            "unaffected_reproduction_is_vacuous": all(column["condition_altered"] for column in comparison["columns"].values()),
            "variant_prediction_path": str((directory / "variant" / "prediction.json").resolve()),
            "variant_prediction_sha256": sha256(directory / "variant" / "prediction.json"),
            "control_prediction_path": str((directory / "control" / "prediction.json").resolve()),
            "control_prediction_sha256": sha256(directory / "control" / "prediction.json"),
            "comparison_path": str(comparison_path.resolve()), "comparison_sha256": sha256(comparison_path),
            "source_runner_sha256": manifest["source_runner_sha256"], "source_sha256": source_hashes,
            "validation": {
                "full_prefilter_state_identical": True, "unaffected_columns_exact_reproduction": True,
                "recorded_replies_identical": True, "original_clustering_cache_unchanged": True,
                "frozen_scores_match_prediction": True,
            },
        }
    manifest["manifest_hash"] = digest(manifest)
    write_json(OUTPUT / "completion_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    result = freeze()
    print(json.dumps({"status": result["status"], "manifest_hash": result["manifest_hash"],
                      "datasets": list(result["datasets"])}, sort_keys=True))
