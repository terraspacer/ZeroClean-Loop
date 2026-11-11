"""Dirty-only baseline adapters, with explicit reproduction/fidelity status.

The runnable ``adapted_*`` methods and ``run_reconstructed_zeroed_zeroec`` must
never be reported as reproductions of ZeroEC. ``run_exact_zeroed`` uses upstream
function bodies and fails closed when required resources are absent.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .data import DirtyTable
    from .types import Prediction


class BaselineUnavailable(RuntimeError):
    """An upstream baseline cannot run faithfully with the available resources."""


def _source_manifest(source_dir: str | Path, names: tuple[str, ...]) -> dict[str, str]:
    root = Path(source_dir)
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names if (root / name).is_file()}


def preflight_zeroed(source_dir: str | Path = "vendor/ZeroED",
                     fasttext_path: str | Path | None = None) -> dict[str, Any]:
    """Inspect source/resources only; never inspect a dataset or label file."""
    root = Path(source_dir)
    required = ("main.py", "feature.py", "utility.py", "prompt_gen.py",
                "get_rel_attrs.py", "distri_analys.py")
    missing_sources = [name for name in required if not (root / name).is_file()]
    missing_dependencies = [name for name in ("numpy", "pandas", "sklearn", "fasttext", "tqdm")
                            if importlib.util.find_spec(name) is None]
    model = Path(fasttext_path) if fasttext_path else root / "cc.en.300.bin"
    issues: list[str] = []
    if missing_sources:
        issues.append("Missing upstream source: " + ", ".join(missing_sources))
    if missing_dependencies:
        issues.append("Missing runtime modules: " + ", ".join(missing_dependencies))
    if not model.is_file():
        issues.append("Missing original fastText model cc.en.300.bin")
    provenance = None
    if model.is_file():
        from .fasttext_resources import prepared_fasttext_provenance
        try:
            provenance = prepared_fasttext_provenance(model)
        except (ValueError, KeyError, TypeError) as error:
            issues.append(f"Invalid prepared fastText provenance: {error}")
    repairs: list[str] = []
    prompt_file = root / "prompt_gen.py"
    if prompt_file.is_file():
        defs = {node.name for node in ast.parse(prompt_file.read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if "guide_gen_prompt" not in defs and "kb_gen_prompt" in defs:
            repairs.append("Alias upstream kb_gen_prompt as guide_gen_prompt (upstream import defect)")
    return {"baseline": "zeroed_upstream", "ready": not issues, "blockers": issues,
            "missing_dependencies": missing_dependencies, "fasttext_path": str(model.resolve()),
            "fasttext_prepared_provenance": provenance,
            "source_sha256": _source_manifest(root, required), "runtime_source_repairs": repairs,
            "inference_inputs": ["dirty_table", "fresh_dirty_only_llm_responses", "fasttext_model"],
            "uses_ground_truth": False, "cached_results_reused": False}


def preflight_zeroec(source_dir: str | Path = "vendor/ZeroEC",
                     embedding_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(source_dir)
    required = ("correction_zero_shot.py", "prompt_templates/SystemMessage_EC.txt",
                "prompt_templates/HumanMessage_EC.txt", "prompt_templates/general_examples_EC.txt")
    missing = [name for name in required if not (root / name).is_file()]
    deps = [name for name in ("numpy", "pandas", "sklearn", "fast_sentence_transformers",
                              "langchain", "langchain_openai", "langchain_core",
                              "langchain_community", "faiss")
            if importlib.util.find_spec(name) is None]
    issues = (["Missing upstream sources/templates: " + ", ".join(missing)] if missing else [])
    if deps:
        issues.append("Missing runtime modules: " + ", ".join(deps))
    embedding = Path(embedding_path) if embedding_path else root / "all-MiniLM-L6-v2"
    if not embedding.is_dir() or not any((embedding / filename).is_file()
                                        for filename in ("model.safetensors", "pytorch_model.bin")):
        issues.append("Missing original all-MiniLM-L6-v2 embedding checkpoint")
    from .upstream_zeroec import reconstruction_preflight
    return {"baseline": "zeroec_upstream_zero_shot", "ready": not issues, "blockers": issues,
            "source_sha256": _source_manifest(root, required), "missing_dependencies": deps,
            "embedding_path": str(embedding.resolve()),
            "uses_ground_truth": False, "entrypoint": "correction_zero_shot.py",
            "excluded_entrypoint": "correction.py (human-repair/clean-data supervision path)",
            "reconstruction": reconstruction_preflight(root, embedding)}


def run_reconstructed_zeroed_zeroec(table: DirtyTable, detection: Prediction,
                                    request: Callable[[str, str, dict], str],
                                    output_dir: str | Path,
                                    source_dir: str | Path = "vendor/ZeroEC",
                                    embedding_path: str | Path | None = None,
                                    use_general_examples: bool = True,
                                    prompt_profile: str = "committed_zeroec",
                                    recovered_prompt_dir: str | Path | None = None) -> Prediction:
    """Run the source-derived sequential correction path with declared repairs.

    ZeroEC's named zero-shot prompt files were not committed. This explicitly
    selected reconstruction uses its other committed correction templates;
    it can never claim an exact upstream ZeroEC reproduction. Detection must
    come from the source ZeroED runner; prior repairs are ignored. All detected
    cells are queued independently, and detection scores remain unchanged.

    ``use_general_examples=False`` omits fixed answer examples for a stricter
    zero-example policy and is an additional declared prompt change.
    ``prompt_profile='author_zerodc_29cde9e0dbcd'`` explicitly selects hash-bound
    author-source candidates. Their historical identity remains unproven.
    """
    from .upstream_zeroec import reconstruction_preflight, run_source_zeroec
    source = Path(source_dir).resolve()
    embedding = Path(embedding_path).resolve() if embedding_path else source / "all-MiniLM-L6-v2"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    status = reconstruction_preflight(source, embedding, use_general_examples,
                                       prompt_profile, recovered_prompt_dir)
    (out / "preflight.json").write_text(json.dumps(status, indent=2) + "\n")
    if detection.metadata.get("baseline", detection.metadata.get("name")) != "zeroed_upstream":
        raise ValueError("Sequential source ZeroEC reconstruction requires source ZeroED detection")
    detector_fingerprint = detection.metadata.get("input_fingerprint")
    if detector_fingerprint is not None and detector_fingerprint != table.fingerprint:
        raise ValueError("Source ZeroED detection belongs to a different dirty input")
    if not status["ready"]:
        raise BaselineUnavailable("; ".join(status["blockers"]))
    return run_source_zeroec(table, detection, request, out.resolve(), source, embedding,
                             status, use_general_examples, prompt_profile, recovered_prompt_dir)


def run_exact_zeroed(table: DirtyTable, request: Callable[[str, str, dict], str],
                     output_dir: str | Path, fasttext_path: str | Path | None = None,
                     source_dir: str | Path = "vendor/ZeroED", seed: int = 42) -> Prediction:
    """Run source-extracted upstream functions without main.py's label/eval code.

    ``request(stage, prompt, metadata)`` must return the original response text.
    A file queue may raise its pending-request exception; this runner does not
    invent a response or silently substitute an embedding model.
    """
    status = preflight_zeroed(source_dir, fasttext_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(json.dumps(status, indent=2) + "\n")
    if not status["ready"]:
        raise BaselineUnavailable("; ".join(status["blockers"]))
    prepared = status.get("fasttext_prepared_provenance")
    if prepared and prepared["dimension"] != len(table.columns):
        raise BaselineUnavailable("Prepared fastText dimension must equal the dirty table's column count")
    from .upstream_zeroed import run_source_zeroed
    return run_source_zeroed(table, request, out, Path(source_dir).resolve(),
                             Path(status["fasttext_path"]), seed, status)


def queue_zeroed_preliminary(table: DirtyTable, request: Callable[[str, str, dict], str],
                             output_dir: str | Path, source_dir: str | Path = "vendor/ZeroED",
                             seed: int = 42) -> dict:
    """Enqueue the entire original first LLM stage before fastText is available.

    Responses use exactly the same request bodies as ``run_exact_zeroed``;
    replaying there consumes these responses without changing its prompts.
    No generated source is executed by this preparation step.
    """
    import csv
    import io
    import itertools
    import random
    import re
    from collections import defaultdict
    import logging
    import os
    import pandas as pd
    from .llm import ResponsePending
    from .upstream_zeroed import _definitions
    source = Path(source_dir).resolve()
    preflight = preflight_zeroed(source)
    out = Path(output_dir)
    (out / "funcs_pre").mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(table.columns)
    writer.writerows(table.rows)
    stream.seek(0)
    dirty = pd.read_csv(stream, dtype=str).fillna("nan")
    random.seed(seed)
    from sklearn.metrics import mutual_info_score
    shared = {"__name__": "zeroclean_preliminary_source", "np": np, "pd": pd,
              "re": re, "ast": ast, "random": random, "os": os, "defaultdict": defaultdict,
              "combinations": itertools.combinations, "mutual_info_score": mutual_info_score}
    related_source = _definitions(source / "get_rel_attrs.py", dict(shared))
    related_raw = related_source["cal_strong_res_column_nmi"](
        related_source["cal_all_column_nmi"](dirty), rel_top=2)
    related = {attr: dict(related_raw[attr]) for attr in table.columns}
    prompts = _definitions(source / "prompt_gen.py", dict(shared))
    shared["pre_func_prompt"] = prompts["pre_func_prompt"]
    shared["API_USE"] = True
    shared["get_ans_from_llm"] = lambda prompt, api_use=True: request(
        "preliminary_functions", str(prompt), {"dataset": table.name,
        "dirty_fingerprint": table.fingerprint, "baseline": "zeroed_upstream",
        "source_sha256": preflight["source_sha256"]})
    main = _definitions(source / "main.py", shared,
                        {"process_related_attr", "measure_llm_label", "load_config"})
    missing = []
    ready = []
    for attr in table.columns:
        try:
            main["gen_clean_funcs"](attr, dirty, str(out / "funcs_pre"), related,
                                     logging.getLogger("zeroed_preliminary_queue"))
            ready.append(attr)
        except ResponsePending:
            missing.append(attr)
    report = {"stage": "preliminary_functions", "dataset": table.name,
              "dirty_fingerprint": table.fingerprint, "pending_columns": missing,
              "ready_columns": ready, "source_sha256": preflight["source_sha256"],
              "generated_code_executed": False}
    (out / "preliminary_queue.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _column_index(table: DirtyTable, value: Any) -> int | None:
    if isinstance(value, int) and 0 <= value < len(table.columns):
        return value
    try:
        return table.columns.index(str(value))
    except ValueError:
        return None


def _rule_outputs(table: DirtyTable, artifacts: dict) -> tuple[list[dict], np.ndarray]:
    from .sandbox import evaluate_rules
    rules = list(artifacts.get("rules", []))
    if not rules:
        return [], np.empty((len(table.rows), 0), dtype=np.int8)
    rows = [dict(zip(table.columns, row)) for row in table.rows]
    values = np.asarray(evaluate_rules(rules, rows), dtype=np.int8)
    if values.shape != (len(rows), len(rules)):
        raise ValueError("Rule evaluator returned an incompatible matrix")
    return rules, values


def _character_features(values: list[str], context: list[str]) -> np.ndarray:
    """Explicit replacement for fastText; these are not upstream embeddings."""
    from collections import Counter
    import re
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import MinMaxScaler
    counts = Counter(values)
    patterns = [re.sub(r"\d", "D", re.sub(r"[A-Za-z]", "L", value)) for value in values]
    pattern_counts = Counter(patterns)
    numeric = []
    for value, pattern in zip(values, patterns):
        n = max(len(value), 1)
        numeric.append([len(value), sum(c.isdigit() for c in value) / n,
                        sum(c.isalpha() for c in value) / n,
                        sum(c.isspace() for c in value) / n,
                        counts[value] / max(len(values), 1),
                        pattern_counts[pattern] / max(len(values), 1),
                        float(value.strip().casefold() in {"", "nan", "null", "none", "n/a", "na"})])
    hashed = HashingVectorizer(analyzer="char", ngram_range=(2, 4), n_features=32,
                              alternate_sign=False, norm="l2").transform(context).toarray()
    return np.column_stack([MinMaxScaler().fit_transform(numeric), hashed]).astype(np.float32)


def adapted_zeroed(table: DirtyTable, artifacts: dict | None = None,
                   seed: int = 42, max_iter: int = 500) -> Prediction:
    """Single-pass, source-informed ZeroED adaptation with declared substitutions.

    Representative pseudo-labels come exclusively from pass-1 dirty-only rules.
    A fixed 5% sample is propagated by nearest feature distance; per-column MLPs
    learn those pseudo-labels. No validation labels or pass-2 artifacts are used.
    """
    from sklearn.metrics import pairwise_distances_argmin_min
    from sklearn.neural_network import MLPClassifier
    from .types import Prediction
    artifacts = artifacts or {}
    n, m = len(table.rows), len(table.columns)
    probabilities = np.zeros((n, m), dtype=np.float64)
    rules, outputs = _rule_outputs(table, artifacts)
    diagnostics: dict[str, Any] = {}
    if n:
        for j, column in enumerate(table.columns):
            target_rule_indices = [k for k, rule in enumerate(rules)
                                   if _column_index(table, rule.get("target", rule.get("column"))) == j]
            if not target_rule_indices:
                diagnostics[column] = {"status": "no_pass1_rules", "representatives": 0}
                continue
            matrix = outputs[:, target_rule_indices]
            valid = matrix >= 0
            dirty = matrix == 1
            # The upstream representative decision is an LLM classification. A
            # pass-1 rule hit is the explicitly declared replacement here.
            representative_labels = dirty.any(axis=1).astype(np.int8)
            values = [row[j] for row in table.rows]
            context = [" | ".join(row) for row in table.rows]
            x = _character_features(values, context)
            x = np.column_stack([x, dirty.astype(np.float32), (~valid).astype(np.float32)])
            k = min(max(1, int(n * .05)), len(set(values)))
            centers = np.random.RandomState(0).choice(n, k, replace=False)
            nearest, _ = pairwise_distances_argmin_min(x, x[centers])
            y = representative_labels[centers][nearest]
            classes = np.unique(y)
            if len(classes) == 1:
                probabilities[:, j] = float(classes[0])
                fit_status = "constant_pseudo_label"
                iterations = 0
            else:
                d = x.shape[1]
                model = MLPClassifier(hidden_layer_sizes=(2*d, d), activation="relu", solver="adam",
                                      max_iter=max_iter, random_state=seed, n_iter_no_change=10)
                model.fit(x, y)
                probabilities[:, j] = model.predict_proba(x)[:, list(model.classes_).index(1)]
                fit_status = "fitted"
                iterations = int(model.n_iter_)
            diagnostics[column] = {"status": fit_status, "representatives": int(k),
                                   "pseudo_error_count": int(y.sum()),
                                   "pass1_rule_count": len(target_rule_indices), "iterations": iterations}
    metadata = {
        "name": "adapted_zeroed", "baseline": "adapted_zeroed", "fidelity": "source_informed_adaptation",
        "reproduced_upstream": False, "strict_zero_shot": True, "uses_ground_truth": False,
        "input_fingerprint": table.fingerprint, "seed": seed, "representative_fraction": .05,
        "artifact_sha256": hashlib.sha256(json.dumps(artifacts, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "source_sha256": _source_manifest("vendor/ZeroED", ("main.py", "feature.py")),
        "adaptations": [
            "Character n-gram hashing and string/frequency features replace cc.en.300.bin fastText features",
            "Pass-1 dirty-only LLM rules replace preliminary predicates, distribution/guideline prompts and representative-label prompts",
            "No separate synthetic-error generation or upstream three-step predicate filtering",
            f"MLP max_iter={max_iter}; upstream uses 5000",
            "All row columns provide character context; upstream selects top-2 dirty-data NMI related attributes",
        ],
        "columns": diagnostics, "consumed_artifacts": ["pass1.rules"],
    }
    return Prediction(probabilities=probabilities, repairs={}, metadata=metadata)


def adapted_zeroed_zeroec(table: DirtyTable, detection: Prediction,
                          artifacts: dict | None = None) -> Prediction:
    """Naive detect-then-correct adaptation, restricted to pass-1 proposals.

    Values proposed from dirty-only context are applied only to detected cells.
    This deliberately omits ZeroClean-Loop feedback and its second-pass rules.
    It does not claim upstream ZeroEC MiniLM/FAISS retrieval was reproduced.
    """
    from .types import Prediction
    artifacts = artifacts or {}
    proposals = artifacts.get("value_repairs", {})
    repairs: dict[tuple[int, int], str] = {}
    for column, mapping in proposals.items():
        j = _column_index(table, column)
        if j is None or not isinstance(mapping, dict):
            continue
        for i, row in enumerate(table.rows):
            if detection.probabilities[i, j] >= .5 and row[j] in mapping:
                proposed = mapping[row[j]]
                if isinstance(proposed, dict):
                    proposed = proposed.get("value", proposed.get("new_value"))
                if proposed is not None and str(proposed) != row[j]:
                    repairs[(i, j)] = str(proposed)
    for proposal in artifacts.get("repairs", []):
        j = _column_index(table, proposal.get("column"))
        i = proposal.get("row")
        if j is not None and isinstance(i, int) and 0 <= i < len(table.rows):
            if detection.probabilities[i, j] >= .5 and "value" in proposal:
                value = str(proposal["value"])
                if value != table.rows[i][j]:
                    repairs[(i, j)] = value
    metadata = dict(detection.metadata)
    metadata.update({"name": "adapted_zeroed_zeroec", "baseline": "adapted_zeroed_zeroec",
                     "reproduced_upstream": False,
                     "correction_adaptations": [
                         "Pass-1 dirty-only LLM value proposals replace per-cell MiniLM/FAISS retrieval and ZeroEC prompts",
                         "Only cells detected at probability >= 0.5 are eligible for correction",
                     ], "consumed_artifacts": ["pass1.rules", "pass1.value_repairs", "pass1.repairs"]})
    return Prediction(probabilities=np.array(detection.probabilities, copy=True), repairs=repairs,
                      metadata=metadata)
