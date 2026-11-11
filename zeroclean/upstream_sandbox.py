"""Confine upstream-generated predicates and distribution functions.

Source extraction is trusted repository code. Generated code is executed only in
OS-confined subprocesses, never in the inference interpreter.
"""
from __future__ import annotations
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import re


def _predicate_batch(rule, rows):
    """One confined batch with diagnostics, retaining native missing-key errors.

    The source's attr can be absent from a related tuple. Task-level target
    metadata must not preempt the actual predicate's row-level behavior; AST
    validation and OS confinement remain enabled in the shared executor.
    """
    from .sandbox import _execute, _normalize_rows
    columns, diagnostics = _execute([rule], _normalize_rows(rows), require_target=False)
    return columns[0], diagnostics[0]


def predicate_source(function_code: str, attr: str) -> str:
    """Adapt the upstream (row, attr) signature without changing its body logic."""
    tree = ast.parse(function_code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError("Expected one upstream predicate")
    function = functions[0]
    if len(function.args.args) != 2 or function.args.defaults or function.decorator_list:
        raise ValueError("Expected upstream predicate(row, attr)")
    row_name, attr_name = [argument.arg for argument in function.args.args]
    import_aliases = {}
    class Convert(ast.NodeTransformer):
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name not in {"re", "math", "datetime"} or alias.asname not in (None, alias.name):
                    raise ValueError(f"Unsupported predicate import: {alias.name}")
            return None
        def visit_ImportFrom(self, node):
            if node.module != "datetime" or any(alias.name != "datetime" for alias in node.names):
                raise ValueError(f"Unsupported predicate import: {node.module}")
            for alias in node.names:
                import_aliases[alias.asname or alias.name] = "datetime"
            return None
        def visit_Name(self, node):
            if node.id == row_name:
                return ast.copy_location(ast.Name(id="row", ctx=node.ctx), node)
            if node.id in import_aliases and isinstance(node.ctx, ast.Load):
                return ast.copy_location(ast.Attribute(value=ast.Name(id="datetime", ctx=ast.Load()),
                                                       attr="datetime", ctx=ast.Load()), node)
            return node
        def visit_Return(self, node):
            value = self.visit(node.value) if node.value is not None else ast.Constant(None)
            return ast.copy_location(ast.Return(value=ast.Call(func=ast.Name(id="bool", ctx=ast.Load()),
                                                               args=[value], keywords=[])), node)
    converter = Convert()
    body = []
    for node in function.body:
        changed = converter.visit(node)
        if changed is not None:
            body.append(changed)
    body.insert(0, ast.Assign(targets=[ast.Name(id=attr_name, ctx=ast.Store())], value=ast.Constant(attr)))
    body.append(ast.Return(value=ast.Constant(False)))
    converted = ast.FunctionDef(name="check", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="row")],
                                kwonlyargs=[], kw_defaults=[], defaults=[]), body=body, decorator_list=[])
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[converted], type_ignores=[])))


class UpstreamPredicateSandbox:
    def __init__(self, dirty_records):
        self.dirty_records = list(dirty_records)
        self.extra_records = []
        self._record_keys = {json.dumps({name: str(value) for name, value in record.items()}, sort_keys=True)
                             for record in self.dirty_records}
        self.cache = {}
        self._consumed_extra = {}
        self.rejections = {}
        self.evaluation_errors = 0
        self.runtime_errors = []
        self.evaluation_batches = 0
        self.evaluated_rows = 0
        self._cache_lock = threading.RLock()

    def add_records(self, records):
        with self._cache_lock:
            for record in records:
                if not isinstance(record, dict):
                    continue
                normalized = {name: str(value) for name, value in record.items()}
                key = json.dumps(normalized, sort_keys=True)
                if key not in self._record_keys:
                    self._record_keys.add(key)
                    self.extra_records.append(normalized)

    def evaluate(self, code, value, attr):
        # Source feature extraction calls this concurrently from up to 128
        # workers. One fill evaluates every distinct dirty tuple, so racing
        # misses must not launch duplicate OS sandbox batches. RLock also
        # permits newly generated tuples to extend the cache under this lock.
        with self._cache_lock:
            return self._evaluate_locked(code, value, attr)

    def _evaluate_locked(self, code, value, attr):
        keys = tuple(value.keys())
        key = (code, attr, keys)
        normalized_value = {name: str(v) for name, v in value.items()}
        row_key = json.dumps(normalized_value, sort_keys=True)
        if key in self.cache and row_key in self.cache[key]:
            return self.cache[key][row_key]
        if key not in self.cache:
            self.cache[key] = {}
            candidates = self.dirty_records + self.extra_records
        else:
            # New synthetic records are added together before local training.
            # Process that pending batch once per rule/key shape, retaining all
            # existing results. A previously unseen single query adds one row.
            self.add_records([value])
            candidates = self.extra_records[self._consumed_extra[key]:]
        rows = [{name: str(row[name]) for name in keys if name in row} for row in candidates]
        rows.append(normalized_value)
        distinct = {json.dumps(row, sort_keys=True): row for row in rows}
        distinct = {serialized: row for serialized, row in distinct.items() if serialized not in self.cache[key]}
        self._consumed_extra[key] = len(self.extra_records)
        rows = list(distinct.values())
        if rows:
            try:
                source = predicate_source(code, attr)
                rule = {"id": hashlib.sha256((code + attr).encode()).hexdigest(), "target": attr,
                        "kind": "multi", "description": "Upstream clean predicate; true means clean", "source": source}
                values, diagnostic = _predicate_batch(rule, rows)
                if len(values) != len(rows):
                    raise ValueError("Predicate sandbox returned an incomplete row batch")
                self.evaluation_batches += 1
                self.evaluated_rows += len(rows)
                self.cache[key].update(dict(zip(distinct, values)))
                errors = sum(value == -1 for value in values)
                self.evaluation_errors += errors
                status = diagnostic.get("status", "unavailable")
                samples = diagnostic.get("error_samples", [])
                environment_error = any(re.search(
                    r": (?:PermissionError|MemoryError|OSError|ImportError|ModuleNotFoundError|SystemError):", sample)
                    for sample in samples)
                if status != "ok" or environment_error:
                    self.rejections[hashlib.sha256(code.encode()).hexdigest()] = diagnostic.get("reason") or status
                elif errors:
                    # A supported predicate can raise exactly the same native
                    # KeyError/ValueError/etc. in upstream handle_func_exec.
                    # Retain its -1 and evidence without claiming an adapter
                    # policy change solely because every row happened to fail.
                    self.runtime_errors.append({"code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                                                "attribute": attr, "status": status,
                                                "error_count": errors, "evaluated_rows": len(rows),
                                                "reason": diagnostic.get("reason", ""),
                                                "error_samples": samples})
            except (ValueError, SyntaxError, TypeError) as error:
                self.rejections[hashlib.sha256(code.encode()).hexdigest()] = str(error)
                self.cache[key].update({serialized: -1 for serialized in distinct})
        return self.cache[key][row_key]


def distribution_result(source: str, records: list[dict], attr: str) -> str:
    """Return an upstream-style error string when confinement rejects execution."""
    payload = {"source": source, "records": records, "attr": attr}
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        completed = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--distribution-worker"],
                                   input=json.dumps(payload), capture_output=True, text=True,
                                   timeout=35, env=env)
        if completed.returncode:
            return f"Error executing function: sandbox worker failed ({completed.returncode})"
        result = json.loads(completed.stdout)
        return result.get("result", "Error executing function: missing sandbox result")
    except (subprocess.TimeoutExpired, ValueError, OSError) as error:
        return f"Error executing function: {type(error).__name__}: {error}"


def _distribution_worker():
    # Load all runtime code and construct the dirty dataframe before denying IO.
    import collections
    import datetime
    import math
    import re
    import resource
    import signal
    import statistics
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sandbox
    payload = json.load(sys.stdin)
    frame = pd.DataFrame(payload["records"], dtype=str)
    if len(frame.columns):
        # Preload common pandas dispatch/parsing helpers before confinement.
        sample = frame.iloc[:, 0]
        sample.value_counts(); sample.str.len(); sample.describe(); frame.sample(min(1, len(frame)))
    tree = ast.parse(payload["source"])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or len(tree.body) != 1:
        raise ValueError("Expected one distribution-analysis function")
    allowed = {"numpy": np, "pandas": pd, "re": re, "math": math,
               "collections": collections, "datetime": datetime, "statistics": statistics}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or "__" in node.attr):
            raise ValueError("Private attribute forbidden")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or "__" in node.id):
            raise ValueError("Private name forbidden")
        if isinstance(node, ast.Import) and any(alias.name not in allowed for alias in node.names):
            raise ValueError("Unsupported distribution import")
        if isinstance(node, ast.ImportFrom) and node.module not in allowed:
            raise ValueError("Unsupported distribution import")
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in allowed:
            raise ImportError("Import unavailable in distribution sandbox")
        return allowed[name]
    builtins = {name: getattr(__import__("builtins"), name) for name in
                ("abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len",
                 "list", "map", "max", "min", "range", "reversed", "round", "set", "sorted", "str", "sum",
                 "tuple", "zip", "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "ZeroDivisionError")}
    builtins["__import__"] = safe_import
    environment = {"__builtins__": builtins, "pd": pd, "np": np, "re": re,
                   "math": math, "collections": collections, "Counter": collections.Counter,
                   "datetime": datetime.datetime}
    seccomp = sandbox._load_seccomp()
    resource.setrlimit(resource.RLIMIT_CPU, (31, 31))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (3, 3))
    signal.signal(signal.SIGALRM, sandbox._timeout_handler)
    signal.alarm(30)
    sandbox._install_seccomp(seccomp)
    try:
        exec(compile(tree, "<generated-distribution>", "exec"), environment)
        result = environment[functions[0].name](frame, payload["attr"])
        result = str(result)
    except BaseException as error:
        result = f"Error executing function: {type(error).__name__}: {error}"
    print(json.dumps({"result": result}))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["--distribution-worker"]:
        raise SystemExit("Internal worker only")
    try:
        raise SystemExit(_distribution_worker())
    except Exception as error:
        print(json.dumps({"result": f"Error executing function: {type(error).__name__}: {error}"}))
