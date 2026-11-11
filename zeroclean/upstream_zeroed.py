"""Source-extracted ZeroED runner: no upstream entrypoint or evaluation imports.

This optional path requires original fastText resources. Its LLM callback is the
only generation transport; utility.py's network client is never loaded.
"""
from __future__ import annotations

import ast
import collections
import contextlib
import concurrent.futures
import csv
from datetime import datetime
import io
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import struct
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import numpy as np


def _definitions(path: Path, namespace: dict, excluded: set[str] | None = None) -> dict:
    """Compile definitions only. In particular, never execute __main__ blocks."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    excluded = excluded or set()
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name not in excluded]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _ProbabilityCapture:
    """Observe the source prediction feature matrix without changing decisions.

    ``make_predictions`` only invokes ``predict``. The original fitted model
    therefore receives exactly the source's X once for its hard decisions and
    once for scores. No fitting, label propagation, or feature logic changes.
    """

    def __init__(self, model):
        self.model = model
        self.probabilities = None
        self.decisions = None
        self.score_kind = None

    def predict(self, features):
        decisions = np.asarray(self.model.predict(features))
        classes = np.asarray(self.model.classes_)
        if classes.ndim != 1 or not len(classes) or not np.isin(classes, [0, 1]).all():
            raise ValueError("Source classifier must have binary pseudo-label classes")
        if len(classes) == 1:
            # sklearn can emit two sigmoid columns even when classes_ has one
            # entry. Its actual prediction is the sole fitted class throughout.
            probabilities = np.full(len(decisions), float(classes[0]))
            self.score_kind = "constant_pseudo_label_class"
        else:
            scores = np.asarray(self.model.predict_proba(features), dtype=float)
            if scores.shape != (len(decisions), len(classes)):
                raise ValueError("Source classifier probability shape mismatch")
            probabilities = scores[:, int(np.flatnonzero(classes == 1)[0])].copy()
            self.score_kind = "mlp_class_1_predict_proba"
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Invalid source classifier probabilities")
        # sklearn's binary sigmoid inverse transform uses a strict threshold.
        # Preserve and verify the actual predict output, including exact .5.
        if not np.array_equal(probabilities > .5, decisions.astype(bool)):
            raise ValueError("Source classifier scores do not reproduce predict at p(error)>0.5")
        self.probabilities, self.decisions = probabilities, decisions.copy()
        return decisions


def _source_dirty_frames(dirty_path: Path):
    """Retain both distinct string representations used by upstream code."""
    import pandas as pd
    main_frame = pd.read_csv(dirty_path, dtype=str).fillna("nan")
    feature_frame = pd.read_csv(dirty_path).astype(str).fillna("nan")
    return main_frame, feature_frame


PREDICATE_CACHE_CONTRACT = "native_row_errors_dual_parser_incremental_v2"


def _predicate_diagnostics(sandbox):
    return {"rejections": dict(sandbox.rejections), "runtime_errors": list(sandbox.runtime_errors),
            "evaluation_errors": sandbox.evaluation_errors,
            "evaluation_batches": sandbox.evaluation_batches, "evaluated_rows": sandbox.evaluated_rows}


def _restore_predicate_diagnostics(sandbox, report):
    sandbox.rejections.update(report["rejections"])
    sandbox.runtime_errors.extend(report["runtime_errors"])
    for name in ("evaluation_errors", "evaluation_batches", "evaluated_rows"):
        setattr(sandbox, name, getattr(sandbox, name) + report[name])


def _refresh_cached_predicate_features(candidate, feature_dirty, attrs, related, preliminary, sandbox):
    """Migrate old caches without fastText/reclustering when geometry is equal.

    Only the target-metadata precheck changed outputs. Recompute the source's
    preliminary predicate components, including its outer-attr argument for
    related tuples, then compare their independently scaled feature geometry.
    A constant -1 becoming constant 1 still scales to zero and needs no new
    clustering. All raw outputs and diagnostics are refreshed for provenance.
    """
    from sklearn.preprocessing import MinMaxScaler
    replacements = {}
    changed = False
    for j, attr in enumerate(attrs):
        expected = [[] for _ in range(len(feature_dirty))]
        for context_attr in [attr] + list(related[attr]):
            columns = [context_attr] + list(related[context_attr])
            functions = preliminary[context_attr]["clean"]
            records = feature_dirty.loc[:, columns].to_dict("records")
            for i, record in enumerate(records):
                expected[i].extend(sandbox.evaluate(code, record, attr) for code in functions)
        new = np.asarray(expected, dtype=float)
        try:
            old = np.asarray([candidate["features"][(i, j)]["pre_funcs_feat"]
                              for i in range(len(feature_dirty))], dtype=float)
        except (KeyError, TypeError, ValueError):
            return {"compatible": False, "reason": "missing or malformed cached predicate features"}
        if old.shape != new.shape:
            return {"compatible": False, "reason": "predicate feature shape changed"}
        if not np.array_equal(old, new):
            changed = True
            if new.shape[1] and not np.array_equal(MinMaxScaler().fit_transform(old),
                                                   MinMaxScaler().fit_transform(new)):
                return {"compatible": False, "reason": "normalized predicate geometry changed"}
        for i, row in enumerate(expected):
            replacements[(i, j)] = np.asarray(row, dtype=np.int64)
    for coordinate, values in replacements.items():
        candidate["features"][coordinate]["pre_funcs_feat"] = values
    return {"compatible": True, "raw_predicates_changed": changed,
            "reason": "source predicate geometry verified without reclustering"}


def run_source_zeroed(table, request: Callable[[str, str, dict], str], out: Path,
                       source: Path, fasttext_path: Path, seed: int, preflight: dict):
    import fasttext
    import fasttext.util
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
    from sklearn.metrics import (mutual_info_score, pairwise_distances,
                                 pairwise_distances_argmin_min)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    from tqdm import tqdm
    from .types import Prediction
    from .llm import ResponsePending
    from .upstream_sandbox import UpstreamPredicateSandbox, distribution_result

    out = out.resolve()
    work = out / "source_work"
    (work / "data").mkdir(parents=True, exist_ok=True)
    model_link = work / "cc.en.300.bin"
    if model_link.is_symlink() and model_link.resolve() != fasttext_path.resolve():
        model_link.unlink()
    if not model_link.exists():
        model_link.symlink_to(fasttext_path.resolve())
    with (work / "data" / f"{table.name}_error-01.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(table.columns)
        writer.writerows(table.rows)
    # Reproduce upstream main.py's string/null handling for its in-memory table.
    dirty, feature_dirty = _source_dirty_frames(work / "data" / f"{table.name}_error-01.csv")
    attrs = list(dirty.columns)
    random.seed(seed)
    np.random.seed(seed)
    original_reduce = fasttext.util.reduce_model
    def cached_reduce(model, dimension):
        # A checkpoint produced by reducing the original model once already
        # contains the exact vectors required by the source algorithm.
        if model.get_dimension() == dimension:
            return model
        return original_reduce(model, dimension)
    fasttext_runtime = SimpleNamespace(load_model=fasttext.load_model,
                                      util=SimpleNamespace(reduce_model=cached_reduce))
    stage = {"name": "initialization"}
    pending: list[ResponsePending] = []
    def llm(prompt, api_use=False):
        return request(stage["name"], str(prompt), {
            "dataset": table.name, "dirty_fingerprint": table.fingerprint,
            "baseline": "zeroed_upstream", "source_sha256": preflight["source_sha256"],
        })

    def stage_call(function, *args):
        """Collect independent requests before yielding the pending batch."""
        try:
            return function(*args)
        except ResponsePending as error:
            pending.append(error)
            return None

    def finish_stage():
        if pending:
            raise ResponsePending(f"{len(pending)} independent {stage['name']} request groups pending; "
                                  f"all available requests were enqueued. First: {pending[0]}")

    shared = {
        "__name__": "zeroclean_extracted_zeroed", "ast": ast, "collections": collections,
        "defaultdict": collections.defaultdict, "Counter": collections.Counter,
        "csv": csv, "datetime": datetime, "Dict": Dict, "List": List,
        "concurrent": __import__("concurrent"), "ThreadPoolExecutor": concurrent.futures.ThreadPoolExecutor,
        "as_completed": concurrent.futures.as_completed, "fasttext": fasttext_runtime, "io": io,
        "json": json, "logging": logging, "np": np, "os": os, "pd": pd,
        "pickle": pickle, "random": random, "re": re, "shutil": shutil,
        "time": __import__("time"), "tqdm": tqdm, "get_ans_from_llm": llm,
        "AgglomerativeClustering": AgglomerativeClustering, "DBSCAN": DBSCAN,
        "KMeans": KMeans, "MLPClassifier": MLPClassifier, "MinMaxScaler": MinMaxScaler,
        "mutual_info_score": mutual_info_score, "pairwise_distances": pairwise_distances,
        "pairwise_distances_argmin_min": pairwise_distances_argmin_min,
        "max_workers": 128, "funcs_with_errors": set(),
    }
    utility = _definitions(source / "utility.py", dict(shared), {"get_ans_from_llm"})
    for name in ("default_dict_of_lists", "default_list", "split_list_to_sublists"):
        shared[name] = utility[name]
    shared["rag_query"] = lambda query, documents, GPT_USE=True: llm(f"Question: {query}\n\n Guidelines:{documents}")
    shared["query_base"] = lambda query, GPT_USE=True: llm(query)
    prompts = _definitions(source / "prompt_gen.py", dict(shared))
    shared.update({name: value for name, value in prompts.items() if callable(value)
                   and getattr(value, "__module__", None) == shared["__name__"]})
    if "guide_gen_prompt" not in shared:
        shared["guide_gen_prompt"] = prompts["kb_gen_prompt"]
    rel = _definitions(source / "get_rel_attrs.py", dict(shared, combinations=__import__("itertools").combinations))
    distribution = _definitions(source / "distri_analys.py", dict(shared))
    feature = _definitions(source / "feature.py", dict(shared))
    # main.py's functions retain this separate global namespace. Evaluation
    # functions and clean-table relation setup are intentionally not compiled.
    main = _definitions(source / "main.py", dict(shared),
                        {"process_related_attr", "measure_llm_label", "load_config"})
    predicate_sandbox = UpstreamPredicateSandbox(dirty.to_dict("records"))
    # feature.py deliberately infers numeric types before astype(str), unlike
    # main.py's dtype=str. Prime both exact representations (e.g. 12/12.0)
    # without canonicalizing either; both must retain their predicate result.
    predicate_sandbox.add_records(feature_dirty.to_dict("records"))
    feature["handle_func_exec"] = predicate_sandbox.evaluate
    main["handle_func_exec"] = predicate_sandbox.evaluate
    distribution_errors = []
    directories = {"guide_directory": out / "guide", "error_checking_res_directory": out / "error_checking",
                   "funcs_directory": out / "funcs", "funcs_pre_directory": out / "funcs_pre",
                   "err_gen_directory": out / "err_gen"}
    for path in directories.values():
        path.mkdir(exist_ok=True)
    log = logging.getLogger("zeroclean.source_zeroed")
    main.update({name: str(path) for name, path in directories.items()})
    main.update({"dataset": table.name, "dirty_csv": dirty, "all_attrs": attrs,
                 "API_USE": True, "GUIDE_USE": True, "err_check_val_num_per_query": 20,
                 "logger": log, "resp_path": str(out), "para_file": io.StringIO()})
    related = rel["cal_strong_res_column_nmi"](rel["cal_all_column_nmi"](dirty), rel_top=2)
    # Preserve default-dict access for uncorrelated/constant columns.
    related = {attr: dict(related[attr]) for attr in attrs}
    main["related_attrs_dict"] = related
    (out / "related_attrs_dict.json").write_text(json.dumps(related, indent=2) + "\n")

    with _working_directory(work):
        stage["name"] = "preliminary_functions"
        pre = collections.defaultdict(utility["default_dict_of_lists"])
        for attr in attrs:
            result = stage_call(main["gen_clean_funcs"], attr, dirty,
                                str(directories["funcs_pre_directory"]), related, log)
            if result is not None:
                pre.update(result)
        finish_stage()
        stage["name"] = "clustering"
        cache_dir = out / "run_cache"
        cache_dir.mkdir(exist_ok=True)
        model_stat = fasttext_path.stat()
        cache_key = hashlib.sha256(json.dumps({
            "dirty": table.fingerprint, "sources": preflight["source_sha256"], "seed": seed,
            "related": related, "preliminary_predicates": dict(pre),
            "model": {"path": str(fasttext_path.resolve()), "size": model_stat.st_size,
                      "mtime_ns": model_stat.st_mtime_ns},
            "representative_method": "RANDOM", "representative_fraction": "5%",
        }, sort_keys=True).encode()).hexdigest()
        cache_path = cache_dir / "clustering.pkl"
        cached = None
        cache_validation = {"compatible": False, "reason": "no matching clustering cache"}
        if cache_path.is_file():
            with cache_path.open("rb") as handle:
                candidate = pickle.load(handle)
            if candidate.get("key") == cache_key:
                if candidate.get("predicate_execution_contract") == PREDICATE_CACHE_CONTRACT and "predicate_diagnostics" in candidate:
                    cached = candidate
                    _restore_predicate_diagnostics(predicate_sandbox, candidate["predicate_diagnostics"])
                    cache_validation = {"compatible": True, "reason": "current predicate contract and cached diagnostics restored"}
                else:
                    cache_validation = _refresh_cached_predicate_features(
                        candidate, feature_dirty, attrs, related, pre, predicate_sandbox)
                    if cache_validation["compatible"]:
                        cached = candidate
                        candidate["predicate_execution_contract"] = PREDICATE_CACHE_CONTRACT
                        candidate["predicate_diagnostics"] = _predicate_diagnostics(predicate_sandbox)
                        with cache_path.with_suffix(".tmp").open("wb") as handle:
                            pickle.dump(candidate, handle)
                        cache_path.with_suffix(".tmp").replace(cache_path)
        if cached is not None:
            clusters, centers, feature_all = cached["clusters"], cached["centers"], cached["features"]
            random.setstate(cached["python_rng_state"])
            np.random.set_state(cached["numpy_rng_state"])
        else:
            clusters, centers = {}, {}
            feature_all = {}
            for j, attr in enumerate(attrs):
                _, representative, groups, _, feats = feature["cluster"](
                    table.name, "RANDOM", "5%", j, related, pre, str(out))
                clusters[attr] = [list(map(int, representative))] + groups
                centers[attr] = [dirty.loc[i, [attr] + list(related[attr])].to_dict() for i in representative]
                feature_all.update(feats)
            cached = {"key": cache_key, "clusters": clusters, "centers": centers,
                      "features": feature_all, "python_rng_state": random.getstate(),
                      "numpy_rng_state": np.random.get_state(),
                      "predicate_execution_contract": PREDICATE_CACHE_CONTRACT,
                      "predicate_diagnostics": _predicate_diagnostics(predicate_sandbox)}
            with cache_path.with_suffix(".tmp").open("wb") as handle:
                pickle.dump(cached, handle)
            cache_path.with_suffix(".tmp").replace(cache_path)
        main.update({"cluster_index_dict": clusters, "center_value_dict": centers})
        (out / "cluster_index_dict.json").write_text(json.dumps(clusters, indent=2) + "\n")
        (out / "center_value_dict.json").write_text(json.dumps(centers, indent=2) + "\n")
        with (out / "cluster_feat_dict.pkl").open("wb") as handle:
            # The source-extracted default factory has no importable module;
            # every cell key exists, so serialize the identical plain mapping.
            pickle.dump(dict(feature_all), handle)

        stage["name"] = "distribution_functions"
        analyzer = distribution["LLMDataDistrAnalyzer"](dirty)
        def confined_distribution(code, attr):
            result = distribution_result(code, dirty.to_dict("records"), attr)
            if result.startswith("Error executing function:"):
                distribution_errors.append({"attribute": attr, "error": result})
            return result
        analyzer.execute_function = confined_distribution
        distr = {}
        distribution_dir = out / "distri_analys"
        distribution_dir.mkdir(exist_ok=True)
        distribution_prompts = {attr: analyzer.generate_llm_prompt(attr)[0] for attr in attrs}
        distribution_responses = {}
        for attr in attrs:
            response = stage_call(llm, distribution_prompts[attr])
            if response is not None:
                distribution_responses[attr] = response
        finish_stage()
        for attr in attrs:
            prompt, response = distribution_prompts[attr], distribution_responses[attr]
            distr[attr] = analyzer.analyze_data(attr, response, str(distribution_dir / f"ori_distri_analys_{attr}.txt"))
            (distribution_dir / f"prompt_distri_analys_{attr}.txt").write_text(prompt)
            (distribution_dir / f"distri_analys_{attr}.txt").write_text(distr[attr])
        stage["name"] = "guidelines"
        guides = {}
        for attr in attrs:
            prompt = shared["guide_gen_prompt"](attr, table.name, clusters[attr][0], dirty, distr[attr])
            response = stage_call(llm, prompt)
            if response is None:
                continue
            guides[attr] = response
            (directories["guide_directory"] / f"prompt_{attr}.txt").write_text(prompt)
            (directories["guide_directory"] / f"guide_{attr}.txt").write_text(guides[attr])
        finish_stage()
        main["guide_content"] = guides
        stage["name"] = "representative_labels"
        for attr in attrs:
            stage_call(main["task_det_initial"], attr, str(directories["error_checking_res_directory"]))
        finish_stage()
        main["center_index_value_label_dict"] = {}
        labels = main["extract_llm_label_res"](attrs, str(directories["error_checking_res_directory"]), clusters, centers)
        main["center_index_value_label_dict"] = labels
        wrong, right = main["label_prop"](str(out), None, None, clusters, labels)
        stage["name"] = "error_conditioned_functions"
        funcs = collections.defaultdict(utility["default_dict_of_lists"])
        generated = collections.defaultdict(utility["default_dict_of_lists"])
        for attr in attrs:
            result = stage_call(main["task_func_gen"], attr, generated)
            if result is not None:
                funcs.update(result)
        finish_stage()
        stage["name"] = "synthetic_errors"
        for attr in attrs:
            stage_call(main["task_gen_err_data"], attr, dirty, labels, related, generated)
        finish_stage()
        predicate_sandbox.add_records([value for attr in attrs for value in generated[attr]["dirty"]])

        # The three filtering loops below reproduce main.py lines 954-990,
        # including upstream removal-while-iterating behavior.
        right_by_attr = collections.defaultdict(list)
        for idx, attr in right:
            right_by_attr[attr].append((idx, dirty.loc[idx, [attr]+list(related[attr])].to_dict()))
        clean_before_filter = {attr: len(right_by_attr[attr]) for attr in attrs}
        for attr in attrs:
            for func in funcs[attr]["clean"]:
                if not right_by_attr[attr]:
                    continue
                passed = sum(main["handle_func_exec"](func, val, attr) == 1 for _, val in right_by_attr[attr])
                if passed / len(right_by_attr[attr]) < .5:
                    funcs[attr]["clean"].remove(func)
        for attr in attrs:
            for idx, val in right_by_attr[attr]:
                passed = sum(main["handle_func_exec"](func, val, attr) == 1 for func in funcs[attr]["clean"])
                if passed / (len(funcs[attr]["clean"]) + 1e-6) < .5:
                    right.remove((idx, attr))
        for attr in attrs:
            good_values = [dirty.loc[idx, [attr]+list(related[attr])].to_dict()
                           for idx, _, label in labels[attr] if label == 0]
            if good_values:
                funcs[attr]["clean"] = [func for func in funcs[attr]["clean"]
                    if sum(main["handle_func_exec"](func, val, attr) == 1 for val in good_values) / len(good_values) >= .5]
        stage["name"] = "local_mlp"
        probabilities = np.zeros((len(dirty), len(attrs)), dtype=float)
        binary_decisions = np.zeros((len(dirty), len(attrs)), dtype=bool)
        classifier_columns = {}
        for j, attr in enumerate(attrs):
            _, x, y = main["process_attr_train_feat"](attr, dirty, right, wrong, related,
                                                      generated, funcs, feature_all, str(out))
            label_counts = {str(label): int(np.count_nonzero(np.asarray(y) == label)) for label in (0, 1)}
            training_trace = {"training_label_counts": label_counts,
                              "representative_label_counts": {str(label): sum(item[2] == label for item in labels[attr]) for label in (0, 1)},
                              "clean_predicate_count_after_filter": len(funcs[attr]["clean"]),
                              "propagated_clean_before_filter": clean_before_filter[attr],
                              "propagated_clean_after_filter": sum(a == attr for _, a in right)}
            if training_trace["representative_label_counts"]["1"] == 0 and not funcs[attr]["clean"] and clean_before_filter[attr] and not label_counts["0"] and label_counts["1"]:
                training_trace["source_edge_case"] = "No error-labeled representatives yield no conditioned predicates; source zero-predicate filter removes clean rows, leaving synthetic label-1 training"
            _, model, *_ = main["train_model"](attr, x, y, 5000)
            if model is not None:
                capture = _ProbabilityCapture(model)
                cells = main["make_predictions"](j, attr, dirty, {attr: capture}, related,
                                                   funcs, feature_all, str(out))
                for idx, _ in cells:
                    binary_decisions[idx, j] = True
                if capture.probabilities is None or not np.array_equal(capture.decisions.astype(bool), binary_decisions[:, j]):
                    raise ValueError("Source prediction observer did not preserve the returned decision cells")
                probabilities[:, j] = capture.probabilities
                classifier_columns[attr] = {"score_kind": capture.score_kind,
                                            "classes": np.asarray(model.classes_).tolist(), **training_trace}
            else:
                classifier_columns[attr] = {"score_kind": "no_training_features_source_predicts_clean",
                                            "classes": [], **training_trace}
    metadata: dict[str, Any] = {
        "name": "zeroed_upstream", "baseline": "zeroed_upstream", "fidelity": "upstream_function_extraction",
        "reproduced_upstream": not predicate_sandbox.rejections and not distribution_errors,
        "uses_ground_truth": False, "strict_zero_shot": True,
        "input_fingerprint": table.fingerprint, "seed": seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "source_sha256": preflight["source_sha256"],
        "fasttext_prepared_provenance": preflight.get("fasttext_prepared_provenance"),
        "runtime_source_repairs": preflight["runtime_source_repairs"],
        "execution_changes": ["Evaluation and clean-table loading omitted",
                              "Independent attribute stages scheduled sequentially",
                              "Independent prompts enqueued in batches; distribution prompts built before response execution",
                              "Fresh LLM transport supplied by explicit callback",
                              "Generated predicates and distribution functions confined in OS sandbox workers",
                              "Predicate cache primed with both upstream CSV parser views; incremental fills evaluate only new normalized tuples",
                              "An already-reduced fastText checkpoint skips a second PCA rotation",
                              "Prediction-only observer exposes class-1 predict_proba while retaining upstream model.predict decisions",
                              "Python and NumPy seeded for reproducibility"],
        "sandbox_predicate_rejections": predicate_sandbox.rejections,
        "sandbox_predicate_evaluation_errors": predicate_sandbox.evaluation_errors,
        "sandbox_predicate_runtime_errors": predicate_sandbox.runtime_errors,
        "sandbox_predicate_cache": {"evaluation_batches": predicate_sandbox.evaluation_batches,
                                     "evaluated_rows": predicate_sandbox.evaluated_rows},
        "clustering_cache_validation": cache_validation,
        "distribution_execution_errors": distribution_errors,
        "probabilities_semantics": "Uncalibrated MLPClassifier.predict_proba for pseudo-label class 1; single fitted classes map to constant 0 or 1",
        "upstream_binary_decisions": binary_decisions.tolist(),
        "decision_threshold": .5,
        "decision_threshold_operator": ">",
        "classifier_columns": classifier_columns,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return Prediction(probabilities=probabilities, repairs={}, metadata=metadata)
