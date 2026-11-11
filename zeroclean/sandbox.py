"""Validate and isolate generated violation rules.

``validate_rules(rules, rows, max_error_rate=.05)`` returns ``(accepted,
report)``. A rule is accepted only when its fraction of failed evaluations is
strictly below the threshold. ``evaluate_rules(rules, rows)`` returns a
row-major Python list: 1 is a violation, 0 is valid, and -1 is abstention.

The only permitted source is ``def check(row): ...``. ``row`` is a read-only
mapping of strings. Imports are forbidden; restricted ``re``, ``math``, and
``datetime`` namespaces and ordinary collection/conversion builtins are
predefined. A result must be an actual bool. Generated code never runs in the
calling interpreter. The worker requires a Linux seccomp syscall allowlist;
bubblewrap adds namespace isolation when the host supports it. An unavailable
OS sandbox fails closed, producing abstentions rather than running the code.
"""

from __future__ import annotations

import ast
import ctypes
import ctypes.util
import datetime as _datetime
import errno
import functools
import json
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace


_RULE_TIMEOUT_SECONDS = 1.0
_BATCH_TIMEOUT_SECONDS = 30.0
_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_MAX_SOURCE_BYTES = 32_000
_MAX_AST_NODES = 4_000

_BUILTINS = {
    name: value
    for name, value in {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float,
        "frozenset": frozenset, "int": int, "isinstance": isinstance,
        "len": len, "list": list, "map": map, "max": max, "min": min,
        "range": range, "reversed": reversed, "round": round, "set": set,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
        "Exception": Exception, "ValueError": ValueError,
        "TypeError": TypeError, "KeyError": KeyError, "IndexError": IndexError,
        "OverflowError": OverflowError, "ZeroDivisionError": ZeroDivisionError,
        "ArithmeticError": ArithmeticError, "AssertionError": AssertionError,
    }.items()
}
_EXCEPTIONS = {
    name for name, value in _BUILTINS.items()
    if isinstance(value, type) and issubclass(value, Exception)
}
_RE_NAMES = {
    "fullmatch", "search", "match", "sub", "subn", "findall", "finditer",
    "split", "escape", "compile", "IGNORECASE", "MULTILINE", "DOTALL",
    "ASCII", "VERBOSE", "UNICODE", "I", "M", "S", "A", "X", "U",
}
_MATH_NAMES = {
    "isfinite", "isinf", "isnan", "isclose", "fabs", "floor", "ceil",
    "trunc", "sqrt", "log", "log10", "log2", "log1p", "exp", "expm1",
    "copysign", "fmod", "remainder", "pow", "prod", "fsum", "hypot",
    "sin", "cos", "tan", "degrees", "radians", "pi", "e", "tau",
    "inf", "nan",
}
_DATETIME_NAMES = {
    "datetime", "date", "timedelta", "strptime", "fromisoformat",
}
_ATTRIBUTES = _RE_NAMES | _MATH_NAMES | _DATETIME_NAMES | {
    # str and read-only mapping operations; format/format_map are deliberately
    # absent because their field syntax can traverse arbitrary attributes.
    "get", "keys", "values", "items", "copy", "strip", "lstrip", "rstrip",
    "lower", "upper", "casefold", "capitalize", "title", "swapcase",
    "startswith", "endswith", "replace", "join", "rsplit", "splitlines",
    "partition", "rpartition", "removeprefix", "removesuffix", "find",
    "rfind", "index", "rindex", "count", "isdigit", "isdecimal",
    "isnumeric", "isalpha", "isalnum", "isspace", "isascii", "islower",
    "isupper", "istitle", "zfill", "center", "ljust", "rjust",
    # Local collection operations and regex match/compiled-pattern operations.
    "append", "extend", "add", "discard", "union", "intersection",
    "difference", "issubset", "issuperset", "isdisjoint", "group", "groups",
    "groupdict", "start", "end", "span", "pattern",
    # Deterministic date parsing and arithmetic (no now/today/timezone lookup).
    "year", "month", "day", "hour", "minute", "second", "microsecond",
    "days", "seconds", "total_seconds", "weekday", "isoweekday",
    "isoformat", "strftime", "time",
}
_NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Expr,
    ast.Assign, ast.AugAssign, ast.If, ast.For, ast.While, ast.Break,
    ast.Continue, ast.Pass, ast.Try, ast.ExceptHandler, ast.Raise, ast.Assert,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.Attribute, ast.Subscript,
    ast.Slice, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Call, ast.keyword,
    ast.Starred, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.UAdd,
    ast.USub, ast.Invert, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.LShift, ast.RShift, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.IfExp,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
}


def _validate_source(source: str) -> None:
    if len(source.encode("utf-8")) > _MAX_SOURCE_BYTES:
        raise ValueError("source exceeds the size limit")
    tree = ast.parse(source, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("source must contain only def check(row)")
    function = tree.body[0]
    args = function.args
    if (
        function.name != "check" or function.decorator_list
        or function.returns is not None or getattr(function, "type_params", [])
        or len(args.args) != 1 or args.args[0].arg != "row"
        or args.args[0].annotation is not None or args.posonlyargs
        or args.kwonlyargs or args.vararg or args.kwarg
        or args.defaults or args.kw_defaults
    ):
        raise ValueError("function signature must be exactly def check(row)")
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        raise ValueError("source exceeds the AST size limit")
    local_names = {
        node.id for node in nodes
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    local_names.update(
        node.name for node in nodes
        if isinstance(node, ast.ExceptHandler) and node.name
    )
    known_names = local_names | set(_BUILTINS) | {"row", "re", "math", "datetime"}
    for node in nodes:
        if type(node) not in _NODES:
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise ValueError("nested function definitions are forbidden")
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or "__" in node.id or node.id == "check":
                raise ValueError(f"forbidden name: {node.id}")
            if isinstance(node.ctx, ast.Load) and node.id not in known_names:
                raise ValueError(f"unknown or forbidden name: {node.id}")
        if isinstance(node, ast.Attribute):
            if not isinstance(node.ctx, ast.Load) or node.attr not in _ATTRIBUTES:
                raise ValueError(f"forbidden attribute: {node.attr}")
        if isinstance(node, ast.keyword) and node.arg is not None:
            if node.arg.startswith("_") or "__" in node.arg:
                raise ValueError("private keyword arguments are forbidden")
        if isinstance(node, ast.comprehension) and node.is_async:
            raise ValueError("async comprehensions are forbidden")
        if isinstance(node, ast.Try) and node.finalbody:
            raise ValueError("finally blocks can suppress worker timeouts")
        if isinstance(node, ast.ExceptHandler):
            exception_types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            if not all(isinstance(item, ast.Name) and item.id in _EXCEPTIONS for item in exception_types):
                raise ValueError("except must name an allowed Exception type")
            if node.name and (node.name.startswith("_") or "__" in node.name):
                raise ValueError("private exception names are forbidden")
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Attribute) for target in node.targets):
                raise ValueError("attribute assignment is forbidden")
    # Parsing alone permits invalid break/return placement; compilation checks
    # these cases without evaluating any generated expression.
    compile(tree, "<generated-rule>", "exec")


def _rule_error(rule: object) -> str | None:
    if not isinstance(rule, dict):
        return "rule must be a JSON object"
    for field in ("id", "target", "source", "description"):
        if not isinstance(rule.get(field), str):
            return f"{field} must be a string"
    if not rule["id"] or not rule["target"]:
        return "id and target must be nonempty"
    if rule.get("kind") not in ("single", "multi"):
        return "kind must be single or multi"
    try:
        _validate_source(rule["source"])
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError) as exc:
        return f"{type(exc).__name__}: {str(exc)[:240]}"
    return None


def _normalize_rows(rows: list[dict]) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or any(
            type(key) is not str or type(value) is not str
            for key, value in row.items()
        ):
            raise TypeError("rows must contain dictionaries of str -> str")
        normalized.append(dict(row))
    return normalized


@functools.lru_cache(maxsize=1)
def _bubblewrap_prefix() -> tuple[str, ...]:
    """Use bwrap only after probing the actual host namespace permissions."""
    executable = shutil.which("bwrap")
    if not executable:
        return ()
    prefix = (
        executable, "--unshare-all", "--die-with-parent", "--new-session",
        "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev", "--",
    )
    try:
        result = subprocess.run(
            [*prefix, os.path.realpath(sys.executable), "-I", "-S", "-c", "pass"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=2, check=False,
        )
        return prefix if result.returncode == 0 else ()
    except (OSError, subprocess.TimeoutExpired):
        return ()


def _load_seccomp() -> ctypes.CDLL:
    if sys.platform != "linux":
        raise RuntimeError("Linux seccomp is required")
    library = ctypes.util.find_library("seccomp")
    if not library:
        raise RuntimeError("libseccomp is unavailable")
    return ctypes.CDLL(library, use_errno=True)


def _install_seccomp(seccomp: ctypes.CDLL | None = None) -> None:
    """Deny all syscalls except the interpreter's in-memory runtime needs.

    In particular, no open/stat/access, socket/connect, clone/fork/exec, ptrace,
    filesystem mutation, or resource-limit changes survive this filter.
    Existing worker descriptors are only its stdin/stdout/stderr pipes.
    """
    if seccomp is None:
        seccomp = _load_seccomp()
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.restype = None
    deny = 0x00050000 | errno.EPERM  # SCMP_ACT_ERRNO(EPERM)
    allow = 0x7FFF0000  # SCMP_ACT_ALLOW
    context = seccomp.seccomp_init(deny)
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        for name in (
            "read", "write", "close", "fstat", "fstat64", "lseek",
            "brk", "mmap", "mmap2", "mprotect", "munmap", "mremap", "madvise",
            "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "sigaltstack",
            "setitimer", "getitimer", "clock_gettime", "clock_gettime64",
            "gettimeofday", "time", "futex", "futex_time64", "sched_yield",
            "getpid", "gettid", "getrandom", "restart_syscall", "exit", "exit_group",
        ):
            number = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number >= 0 and seccomp.seccomp_rule_add(context, allow, number, 0) != 0:
                raise RuntimeError(f"cannot allow syscall {name}")
        if seccomp.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)


class _RuleTimeout(BaseException):
    pass


def _timeout_handler(_signum: int, _frame: object) -> None:
    raise _RuleTimeout()


def _worker_main() -> int:
    """Private subprocess entry point. Never execute rules before confinement."""
    try:
        payload = json.load(sys.stdin)
        # Preload strptime's locale/regex helpers before denying filesystem IO.
        _datetime.datetime.strptime("2000-01-01", "%Y-%m-%d")
        rows = [MappingProxyType(row) for row in payload["rows"]]
        namespaces = {
            "re": SimpleNamespace(**{name: getattr(re, name) for name in _RE_NAMES}),
            "math": SimpleNamespace(**{name: getattr(math, name) for name in _MATH_NAMES}),
            "datetime": SimpleNamespace(
                datetime=_datetime.datetime, date=_datetime.date,
                timedelta=_datetime.timedelta, strptime=_datetime.datetime.strptime,
                fromisoformat=_datetime.datetime.fromisoformat,
            ),
        }
        # Library discovery/loading needs filesystem descriptors. Complete it
        # before lowering NOFILE, then install the preloaded filter below.
        seccomp = _load_seccomp()
        # Set hard limits as well as soft limits; generated code cannot increase
        # them, even in the event of an AST validation mistake.
        for which, value in (
            (resource.RLIMIT_AS, payload["memory_limit"]),
            (resource.RLIMIT_CPU, math.ceil(payload["batch_timeout"]) + 1),
            (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_CORE, 0),
            (resource.RLIMIT_NPROC, 0), (resource.RLIMIT_NOFILE, 3),
        ):
            resource.setrlimit(which, (value, value))
        signal.signal(signal.SIGALRM, _timeout_handler)
        _install_seccomp(seccomp)
        print(json.dumps({"sandbox_ready": True}), flush=True)
    except BaseException as exc:
        print(json.dumps({"fatal": f"sandbox unavailable: {type(exc).__name__}: {str(exc)[:200]}"}), flush=True)
        return 2

    for item in payload["rules"]:
        values = [-1] * len(rows)
        samples = []
        status = "ok"
        position = 0
        try:
            signal.setitimer(signal.ITIMER_REAL, payload["rule_timeout"])
            # Repeat validation inside the confined worker, so invoking its
            # private CLI directly cannot bypass the language restriction.
            _validate_source(item["source"])
            environment = {"__builtins__": dict(_BUILTINS), **namespaces}
            exec(compile(item["source"], "<generated-rule>", "exec"), environment)
            check = environment["check"]
            for position, row in enumerate(rows):
                try:
                    value = check(row)
                    if type(value) is not bool:
                        raise TypeError("check must return bool")
                    values[position] = int(value)
                except Exception as exc:
                    if len(samples) < 3:
                        samples.append(f"row {position}: {type(exc).__name__}: {str(exc)[:160]}")
        except _RuleTimeout:
            status = "timeout"
            samples.append(f"rule exceeded {payload['rule_timeout']:g}s wall limit")
        except BaseException as exc:
            status = "runtime_error"
            samples.append(f"{type(exc).__name__}: {str(exc)[:160]}")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        print(json.dumps({
            "index": item["index"], "values": values, "status": status,
            "error_samples": samples[:3],
        }), flush=True)
    return 0


def _execute(rules: list[dict], rows: list[dict], *, require_target: bool = True) -> tuple[list[list[int]], list[dict]]:
    n_rows = len(rows)
    results = [[-1] * n_rows for _ in rules]
    diagnostics = []
    pending = []
    for index, rule in enumerate(rules):
        error = _rule_error(rule)
        if require_target and error is None and rows and any(rule["target"] not in row for row in rows):
            error = "target column is absent from one or more rows"
        diagnostics.append({"status": "invalid_rule" if error else "unavailable", "reason": error or "worker did not evaluate rule", "error_samples": []})
        if error is None:
            pending.append({"index": index, "source": rule["source"]})
    if not pending or not rows:
        return results, diagnostics
    payload = json.dumps({
        "rules": pending, "rows": rows, "rule_timeout": _RULE_TIMEOUT_SECONDS,
        "batch_timeout": _BATCH_TIMEOUT_SECONDS, "memory_limit": _MEMORY_LIMIT_BYTES,
    }, ensure_ascii=True)
    command = [
        # A clean environment, safe-path mode, and disabled site loading give
        # an isolated stdlib interpreter while permitting a fixed hash seed.
        # -I would ignore PYTHONHASHSEED and randomize set iteration per run.
        *_bubblewrap_prefix(), os.path.realpath(sys.executable), "-P", "-S",
        str(Path(__file__).resolve()), "--worker",
    ]
    output = ""
    failure = "sandbox worker failed"
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, close_fds=True,
            env={"PATH": "/usr/bin:/bin", "TZ": "UTC", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"},
            start_new_session=True,
        )
        try:
            output, _stderr = process.communicate(payload, timeout=_BATCH_TIMEOUT_SECONDS)
            failure = f"sandbox worker exited with status {process.returncode}"
        except subprocess.TimeoutExpired:
            process.kill()
            output, _stderr = process.communicate()
            failure = "sandbox worker exceeded batch wall limit"
    except (OSError, ValueError) as exc:
        failure = f"sandbox unavailable: {type(exc).__name__}: {str(exc)[:200]}"
    ready = False
    completed = set()
    pending_indices = {item["index"] for item in pending}
    for line in output.splitlines():
        try:
            message = json.loads(line)
        except (ValueError, TypeError):
            continue
        if message.get("sandbox_ready") is True:
            ready = True
        elif "fatal" in message:
            failure = message["fatal"]
        elif ready and type(message.get("index")) is int:
            index = message["index"]
            values = message.get("values")
            if (
                index in pending_indices
                and isinstance(values, list) and len(values) == n_rows
                and all(type(value) is int and value in (-1, 0, 1) for value in values)
            ):
                results[index] = values
                diagnostics[index] = {
                    "status": message.get("status", "runtime_error"),
                    "reason": "; ".join(message.get("error_samples", [])),
                    "error_samples": message.get("error_samples", []),
                }
                completed.add(index)
    for item in pending:
        if item["index"] not in completed:
            diagnostics[item["index"]]["reason"] = failure
    return results, diagnostics


def evaluate_rules(rules: list[dict], rows: list[dict]) -> list[list[int]]:
    """Evaluate in one isolated worker; return row-major {-1, 0, 1} values.

    Invalid rules, exceptions, non-bool returns, timeouts, and unavailable OS
    confinement all abstain. Input rows must contain only string keys/values.
    """
    normalized = _normalize_rows(rows)
    columns, _ = _execute(rules, normalized)
    return [[column[index] for column in columns] for index in range(len(rows))]


def validate_rules(
    rules: list[dict], rows: list[dict], max_error_rate: float = .05,
) -> tuple[list[dict], list[dict]]:
    """Return accepted rules and per-rule reliability reports.

    Exactly 5% failures is rejected at the default threshold. Empty validation
    samples reject every rule because reliability has not been measured.
    ``evaluated_rows`` counts successful bool results; failed rows abstain.
    """
    if not isinstance(max_error_rate, (int, float)) or not math.isfinite(max_error_rate) or not 0 < max_error_rate <= 1:
        raise ValueError("max_error_rate must be finite and in (0, 1]")
    normalized = _normalize_rows(rows)
    columns, diagnostics = _execute(rules, normalized)
    accepted = []
    report = []
    for rule, values, diagnostic in zip(rules, columns, diagnostics):
        errors = values.count(-1)
        error_rate = errors / len(rows) if rows else 1.0
        passed = bool(rows) and diagnostic["status"] != "invalid_rule" and errors < max_error_rate * len(rows)
        if passed:
            accepted.append(rule)
        metadata = rule if isinstance(rule, dict) else {}
        report.append({
            "id": metadata.get("id"), "target": metadata.get("target"),
            "kind": metadata.get("kind"), "accepted": passed,
            "status": "accepted" if passed else diagnostic["status"] if diagnostic["status"] != "ok" else "error_rate",
            "error_count": errors, "error_rate": error_rate,
            "evaluated_rows": len(rows) - errors,
            "reason": "no validation rows" if not rows else diagnostic["reason"] or ("" if passed else "error rate must be strictly below threshold"),
            "error_samples": diagnostic["error_samples"],
        })
    return accepted, report


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("This module is an internal sandbox worker.")
    raise SystemExit(_worker_main())
