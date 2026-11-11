"""Isolated ZeroED empty-predicate semantic fix with paired source replay.

No core edits, new model requests, benchmark labels, or benchmark evaluation.
Run with .venv/bin/python and PYTHONHASHSEED=0. All native thread counts are
fixed at two by the runner before NumPy/scikit-learn are imported.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import threading

# These defaults apply before this standalone module imports numeric libraries.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "2"

import numpy as np

from zeroclean.data import DATASETS, load_dirty, sha256
from zeroclean.types import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experimental" / "zero_predicate_variant"
PROTOCOL = OUTPUT / "protocol.json"
SEED = 42
MODEL = "gpt-6.1-sol"
EFFORT = "xhigh"


class ReplayMiss(RuntimeError):
    pass


class ReadOnlyReplies:
    """Read only the exact content-addressed source request and reply files."""
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.records = []
        self.lock = threading.Lock()

    def request(self, stage: str, prompt: str, metadata: dict) -> str:
        body = {"stage": stage, "prompt": prompt, "metadata": metadata,
                "model": MODEL, "reasoning_effort": EFFORT}
        key = digest(body)
        request_path = self.root / "requests" / f"{key}.json"
        response_path = self.root / "responses" / f"{key}.json"
        if not request_path.is_file() or not response_path.is_file():
            raise ReplayMiss(f"No exact frozen reply at stage={stage}, hash={key}; no request queued")
        request = json.loads(request_path.read_text())
        response = json.loads(response_path.read_text())
        if request != body or digest(request) != key:
            raise ValueError(f"Source request provenance mismatch: {key}")
        if (response.get("request_hash") != key or response.get("model") != MODEL
                or response.get("reasoning_effort") != EFFORT or not isinstance(response.get("response"), str)):
            raise ValueError(f"Source reply provenance mismatch: {key}")
        with self.lock:
            self.records.append({"stage": stage, "request_hash": key,
                                 "request_sha256": sha256(request_path), "response_sha256": sha256(response_path)})
        return response["response"]


def filter_pseudoclean_rows(attrs, right_by_attr, funcs, right, evaluate, preserve_empty=False):
    """Original second filter, with exactly one optional unavailable-evidence guard.

    This deliberately preserves original evaluation order and removal behavior
    for nonempty predicate lists. It does not claim the pseudo-labels are clean.
    """
    trace = {}
    for attr in attrs:
        predicate_count = len(funcs[attr]["clean"])
        before = sum(a == attr for _, a in right)
        skip = preserve_empty and predicate_count == 0
        if not skip:
            for idx, val in right_by_attr[attr]:
                passed = sum(evaluate(func, val, attr) == 1 for func in funcs[attr]["clean"])
                if passed / (len(funcs[attr]["clean"]) + 1e-6) < .5:
                    right.remove((idx, attr))
        after = sum(a == attr for _, a in right)
        trace[attr] = {
            "predicate_count_at_row_filter": predicate_count,
            "propagated_pseudoclean_before_row_filter": before,
            "propagated_pseudoclean_after_row_filter": after,
            "removed_rows": before - after,
            "empty_predicate_guard": skip,
            "preserved_by_guard": before if skip else 0,
        }
    return trace


def array_hash(array) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps({"dtype": value.dtype.str, "shape": value.shape}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


class Observation:
    def __init__(self, output: Path, preserve_empty: bool, frozen_stage: dict | None = None):
        self.output, self.preserve_empty = output, preserve_empty
        self.frozen_stage = frozen_stage
        self.stage_state = None
        self.filters = {}
        self.training = {}
        self.model = {}

    def prefilter_stage(self, funcs, generated, labels, right, wrong, related, clusters, feature_all, dirty):
        state = {
            "funcs": copy.deepcopy(dict(funcs)), "generated": copy.deepcopy(dict(generated)),
            "labels": copy.deepcopy(labels), "right": copy.deepcopy(right), "wrong": copy.deepcopy(wrong),
            "related_sha256": digest(related), "clusters_sha256": digest(clusters),
            "dirty_records_sha256": digest(dirty.to_dict("records")),
            "feature_mapping_sha256": hashlib.sha256(pickle.dumps(dict(feature_all), protocol=5)).hexdigest(),
        }
        raw_predicates_hash = digest(state["funcs"])
        restored = False
        if self.frozen_stage is not None:
            for name in state:
                if name != "funcs" and digest(state[name]) != digest(self.frozen_stage[name]):
                    raise ValueError(f"Non-predicate pre-filter source state changed: {name}")
            funcs.clear()
            funcs.update(copy.deepcopy(self.frozen_stage["funcs"]))
            state["funcs"] = copy.deepcopy(dict(funcs))
            restored = True
            if digest(state) != digest(self.frozen_stage):
                raise ValueError("Full restored pre-filter state does not match paired source control")
        self.stage_state = state
        write_json(self.output / "prefilter_state.json", state)
        write_json(self.output / "prefilter_state_provenance.json", {
            "full_state_sha256": digest(state), "raw_source_predicates_sha256": raw_predicates_hash,
            "effective_predicates_sha256": digest(state["funcs"]),
            "restored_paired_control_predicates": restored,
            "source_async_predicate_selection_differed": raw_predicates_hash != digest(state["funcs"]),
            "full_prefilter_state_identical_to_control": True if restored else None,
        })

    def filter(self, attrs, right_by_attr, funcs, right, evaluate):
        self.filters = filter_pseudoclean_rows(attrs, right_by_attr, funcs, right, evaluate, self.preserve_empty)
        write_json(self.output / "filter_trace.json", self.filters)

    def training_input(self, attr, x, y, funcs, labels, generated, right, wrong):
        features, targets = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64)
        identity = hashlib.sha256(attr.encode()).hexdigest()[:16]
        directory = self.output / "training"
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / f"{identity}.npz", features=features, pseudo_labels=targets)
        result = {
            "feature_shape": list(features.shape), "feature_sha256": array_hash(features),
            "pseudo_label_sha256": array_hash(targets),
            "pseudo_label_counts": {str(k): int(np.count_nonzero(targets == k)) for k in (0, 1)},
            "representative_pseudo_label_counts": {str(k): sum(item[2] == k for item in labels[attr]) for k in (0, 1)},
            "representative_pseudo_labels_sha256": digest(labels[attr]),
            "conditioned_predicates_sha256": digest(funcs[attr]["clean"]),
            "conditioned_predicate_count": len(funcs[attr]["clean"]),
            "synthetic_error_count": len(generated[attr]["dirty"]),
            "synthetic_errors_sha256": digest(generated[attr]["dirty"]),
            "propagated_pseudoclean_count": sum(a == attr for _, a in right),
            "propagated_pseudoerror_count": sum(a == attr for _, a in wrong),
            "training_file": f"training/{identity}.npz",
            "label_meaning": "Unverified source pseudo-labels: 0=clean judgment, 1=error judgment/synthetic example",
        }
        self.training[attr] = result
        write_json(self.output / "training_trace.json", self.training)

    def fitted_model(self, attr, model):
        self.model[attr] = {"available": model is not None}
        if model is not None:
            self.model[attr].update({
                "classes": np.asarray(model.classes_).tolist(), "iterations": int(model.n_iter_),
                "coefficient_hashes": [array_hash(coef) for coef in model.coefs_],
                "intercept_hashes": [array_hash(value) for value in model.intercepts_],
            })
        write_json(self.output / "model_trace.json", self.model)


FILTER_SOURCE = '''for attr in attrs:
    for idx, val in right_by_attr[attr]:
        passed = sum(main["handle_func_exec"](func, val, attr) == 1 for func in funcs[attr]["clean"])
        if passed / (len(funcs[attr]["clean"]) + 1e-6) < .5:
            right.remove((idx, attr))
'''


def extracted_runner(observation: Observation):
    """Compile one source function in a copied namespace; never mutate a module."""
    from zeroclean import upstream_zeroed
    tree = ast.parse(inspect.getsource(upstream_zeroed.run_source_zeroed))
    function = tree.body[0]
    function.name = "run_isolated_source_zeroed"
    expected_filter = ast.dump(ast.parse(FILTER_SOURCE).body[0], include_attributes=False)
    replaced = training_hooks = model_hooks = stage_hooks = 0

    class Rewrite(ast.NodeTransformer):
        def visit_Expr(self, node):
            nonlocal stage_hooks
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "predicate_sandbox"
                    and call.func.attr == "add_records" and call.args and isinstance(call.args[0], ast.ListComp)):
                # The unique source call immediately after all synthetic errors exist.
                stage_hooks += 1
                hook = ast.parse('_experimental_observation.prefilter_stage(funcs, generated, labels, right, wrong, related, clusters, feature_all, dirty)').body[0]
                return [node, hook]
            return self.generic_visit(node)

        def visit_For(self, node):
            nonlocal replaced
            if ast.dump(node, include_attributes=False) == expected_filter:
                replaced += 1
                return ast.parse('_experimental_observation.filter(attrs, right_by_attr, funcs, right, main["handle_func_exec"])').body[0]
            return self.generic_visit(node)

        def visit_Assign(self, node):
            nonlocal training_hooks, model_hooks
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Subscript) and isinstance(call.func.value, ast.Name) and call.func.value.id == "main" and isinstance(call.func.slice, ast.Constant):
                if call.func.slice.value == "process_attr_train_feat":
                    training_hooks += 1
                    hook = ast.parse('_experimental_observation.training_input(attr, x, y, funcs, labels, generated, right, wrong)').body[0]
                    return [node, hook]
                if call.func.slice.value == "train_model":
                    model_hooks += 1
                    hook = ast.parse('_experimental_observation.fitted_model(attr, model)').body[0]
                    return [node, hook]
            return self.generic_visit(node)

    tree = Rewrite().visit(tree)
    if (replaced, training_hooks, model_hooks, stage_hooks) != (1, 1, 1, 1):
        raise ValueError(f"Source runner structure changed: {(replaced, training_hooks, model_hooks, stage_hooks)}")
    ast.fix_missing_locations(tree)
    namespace = dict(vars(upstream_zeroed))
    namespace["_experimental_observation"] = observation
    exec(compile(tree, str(Path(upstream_zeroed.__file__)), "exec"), namespace)
    return namespace[function.name], {
        "source_runner_sha256": sha256(upstream_zeroed.__file__),
        "extracted_ast_sha256": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        "replacement_count": replaced, "training_observer_count": training_hooks, "model_observer_count": model_hooks,
        "prefilter_stage_hook_count": stage_hooks,
        "live_module_modified": False,
    }


def clone_exact_cache(source: Path, destination: Path) -> dict:
    """Copy only the known clustering stage file, never whole artifact trees."""
    original_hash = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(destination) != original_hash:
        raise ValueError("Clustering cache copy mismatch")
    return {"original_path": str(source.resolve()), "original_sha256": original_hash,
            "isolated_path": str(destination.resolve())}


def run_mode(table, name: str, output: Path, source_cache: Path, queue: Path, source: Path, model: Path, preflight: dict, frozen_stage: dict | None = None):
    output.mkdir(parents=True, exist_ok=True)
    cache = clone_exact_cache(source_cache, output / "run_cache" / "clustering.pkl")
    replies = ReadOnlyReplies(queue)
    observation = Observation(output, preserve_empty=name == "variant", frozen_stage=frozen_stage)
    runner, extraction = extracted_runner(observation)
    write_json(output / "preflight.json", preflight)
    try:
        prediction = runner(table, replies.request, output, source, model, SEED, preflight)
    except ReplayMiss as error:
        write_json(output / "replay_miss.json", {"status": "hard_stop", "reason": str(error), "new_requests_queued": 0})
        raise
    prediction.validate(table.shape)
    # The code emits the source metadata only into this isolated mode directory.
    prediction.metadata.update({
        "name": "zeroed_empty_predicate_variant" if name == "variant" else "zeroed_source_semantics_control",
        "baseline": "experimental_unpromoted", "mode": name,
        "fidelity": "minimal_empty_predicate_semantic_variant" if name == "variant" else "source_semantics_replay",
        "reproduced_upstream": False if name == "variant" else prediction.metadata["reproduced_upstream"],
        "empty_predicate_guard_enabled": name == "variant", "uses_ground_truth": False,
        "protocol_sha256": sha256(PROTOCOL), "experimental_source_sha256": sha256(__file__),
        "execution_addendum_sha256": sha256(OUTPUT / "execution_addendum.json"),
        "extraction": extraction, "new_requests_queued": 0,
    })
    prediction.save(output / "prediction.json")
    write_json(output / "metadata.json", prediction.metadata)
    np.savez_compressed(output / "frozen_scores.npz", scores=prediction.probabilities,
                        binary_decisions=prediction.probabilities > .5, columns=np.asarray(table.columns))
    records = sorted(replies.records, key=lambda record: (record["stage"], record["request_hash"]))
    write_json(output / "recorded_reply_provenance.json", records)
    if sha256(source_cache) != cache["original_sha256"]:
        raise ValueError("Original source clustering cache changed during isolated run")
    write_json(output / "cache_provenance.json", {**cache, "original_unchanged": True})
    return prediction, observation, records, cache


def compare_modes(table, control, variant, control_trace, variant_trace, control_replies, variant_replies):
    if control_replies != variant_replies:
        raise ValueError("Source reply provenance differs between paired runs")
    if digest(control_trace.stage_state) != digest(variant_trace.stage_state):
        raise ValueError("Paired full pre-filter source state mismatch")
    columns = {}
    altered = []
    for j, attr in enumerate(table.columns):
        c, v = control_trace.training[attr], variant_trace.training[attr]
        filter_trace = variant_trace.filters[attr]
        condition_altered = filter_trace["preserved_by_guard"] > 0
        same_training = c["feature_sha256"] == v["feature_sha256"] and c["pseudo_label_sha256"] == v["pseudo_label_sha256"]
        exact_scores = np.array_equal(control.probabilities[:, j], variant.probabilities[:, j])
        exact_decisions = np.array_equal(control.probabilities[:, j] > .5, variant.probabilities[:, j] > .5)
        for field in ("representative_pseudo_labels_sha256", "conditioned_predicates_sha256", "synthetic_errors_sha256"):
            if c[field] != v[field]:
                raise ValueError(f"Unexpected paired stage-input change in {attr}: {field}")
        if not condition_altered and (not same_training or not exact_scores or not exact_decisions):
            raise ValueError(f"Unaffected column failed exact numerical reproduction: {attr}")
        if condition_altered:
            altered.append(attr)
        columns[attr] = {
            "guard_preserved_rows": filter_trace["preserved_by_guard"],
            "condition_altered": condition_altered, "training_identical": same_training,
            "control_training_label_counts": c["pseudo_label_counts"], "variant_training_label_counts": v["pseudo_label_counts"],
            "exact_scores_equal": exact_scores, "exact_binary_decisions_equal": exact_decisions,
            "changed_score_cells": int(np.count_nonzero(control.probabilities[:, j] != variant.probabilities[:, j])),
            "changed_binary_cells": int(np.count_nonzero((control.probabilities[:, j] > .5) != (variant.probabilities[:, j] > .5))),
        }
    return {"dataset": table.name, "input_sha256": table.fingerprint, "shape": list(table.shape),
            "altered_columns": altered, "columns": columns, "recorded_replies_identical": True,
            "unaffected_columns_exact_reproduction": True, "no_benchmark_accuracy_claim": True,
            "full_prefilter_state_identical": True,
            "comparison_target": "fresh paired source-semantics control; not previously frozen source baseline",
            "threshold": .5, "threshold_operator": ">", "status": "experimental_unpromoted"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        parser.print_help()
        return 0
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("Run with PYTHONHASHSEED=0 to reproduce source set ordering")
    from zeroclean.baselines import preflight_zeroed
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["protocol_id"] != "zeroed-empty-predicate-preserves-pseudoclean-v1":
        raise ValueError("Unexpected predeclared protocol")
    source = (ROOT / "vendor" / "ZeroED").resolve()
    summaries = []
    write_json(OUTPUT / "run_manifest.json", {
        "protocol_sha256": sha256(PROTOCOL), "experimental_source_sha256": sha256(__file__),
        "execution_addendum_sha256": sha256(OUTPUT / "execution_addendum.json"),
        "seed": SEED, "python_hash_seed": os.environ["PYTHONHASHSEED"],
        "native_threads": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "stage_cache_root": str(ROOT / "artifacts" / "upstream_zeroed"),
        "queue_root": str(ROOT / "artifacts" / "queue"), "no_benchmark_evaluation": True,
    })
    for dataset in args.datasets:
        table = load_dirty(dataset, ROOT / "vendor" / "ZeroED" / "data")
        model = (ROOT / "artifacts" / "models" / "reduced" / f"cc.en.{table.shape[1]}.bin").resolve()
        preflight = preflight_zeroed(source, model)
        if not preflight["ready"]:
            raise RuntimeError(preflight["blockers"])
        source_cache = ROOT / "artifacts" / "upstream_zeroed" / dataset / "run_cache" / "clustering.pkl"
        queue = ROOT / "artifacts" / "queue"
        dataset_output = OUTPUT / dataset
        write_json(OUTPUT / "progress.json", {"dataset": dataset, "stage": "source_control_started", "complete_datasets": [s["dataset"] for s in summaries]})
        control, control_trace, control_replies, _ = run_mode(table, "control", dataset_output / "control", source_cache, queue, source, model, preflight)
        write_json(OUTPUT / "progress.json", {"dataset": dataset, "stage": "variant_started", "complete_datasets": [s["dataset"] for s in summaries]})
        variant, variant_trace, variant_replies, _ = run_mode(table, "variant", dataset_output / "variant", source_cache, queue, source, model, preflight, frozen_stage=control_trace.stage_state)
        summary = compare_modes(table, control, variant, control_trace, variant_trace, control_replies, variant_replies)
        write_json(dataset_output / "comparison.json", summary)
        summaries.append(summary)
        write_json(OUTPUT / "summary.json", summaries)
        print(json.dumps({"dataset": dataset, "altered_columns": summary["altered_columns"],
                          "unaffected_columns_exact_reproduction": True}, sort_keys=True), flush=True)
    write_json(OUTPUT / "progress.json", {"stage": "complete", "complete_datasets": [s["dataset"] for s in summaries]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
