"""Behavioral and containment tests for generated Python rules."""

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from zeroclean import sandbox


def rule(source, *, ident="r", target="x", kind="single"):
    return {
        "id": ident, "target": target, "kind": kind,
        "description": "test rule", "source": textwrap.dedent(source).strip(),
    }


def test_matrix_orientation_and_boolean_contract():
    rules = [
        rule('def check(row):\n return not row["x"].strip()'),
        rule('def check(row):\n return row.get("x") == "abc"', ident="other"),
        rule('def check(row):\n return 1', ident="wrong_type"),
    ]
    assert sandbox.evaluate_rules(rules, [{"x": "abc"}, {"x": ""}]) == [
        [0, 1, -1], [1, 0, -1],
    ]


def test_predefined_regex_math_datetime_and_collection_operations():
    rules = [
        rule('''
            def check(row):
                return re.fullmatch(r"[A-Z]{2}\\d{2}", row["x"]) is None
        '''),
        rule('''
            def check(row):
                try:
                    number = float(re.sub(r"[, ]", "", row["x"]))
                    return not math.isfinite(number) or number < 0
                except (ValueError, OverflowError) as problem:
                    return bool(str(problem))
        ''', ident="numeric"),
        rule('''
            def check(row):
                try:
                    parsed = datetime.strptime(row["x"], "%Y-%m-%d")
                    return parsed.year < 2000
                except ValueError:
                    return True
        ''', ident="date"),
        rule('''
            def check(row):
                parts = row.get("x", "").split(",")
                return any(not item.strip() for item in parts) or len(set(parts)) != len(parts)
        ''', ident="collection"),
    ]
    rows = [{"x": "AB12"}, {"x": "nan"}, {"x": "2025-02-01"}, {"x": "a,a"}]
    values = sandbox.evaluate_rules(rules, rows)
    assert values == [[0, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    accepted, report = sandbox.validate_rules(rules, rows)
    assert accepted == rules
    assert all(item["error_count"] == 0 for item in report)


def test_strict_error_threshold_and_errors_abstain():
    candidate = rule('def check(row):\n return float(row["x"]) < 0')
    exactly_five_percent = [{"x": "1"}] * 19 + [{"x": "bad"}]
    accepted, report = sandbox.validate_rules([candidate], exactly_five_percent)
    assert accepted == []
    assert report[0]["error_rate"] == .05
    assert report[0]["error_count"] == 1
    assert report[0]["evaluated_rows"] == 19
    assert not report[0]["accepted"]
    below_five_percent = exactly_five_percent + [{"x": "1"}]
    accepted, report = sandbox.validate_rules([candidate], below_five_percent)
    assert accepted == [candidate]
    assert report[0]["accepted"]
    assert sandbox.evaluate_rules([candidate], [{"x": "bad"}]) == [[-1]]


@pytest.mark.parametrize("source", [
    'import os\ndef check(row):\n return False',
    'def check(row):\n import re\n return False',
    'def check(row):\n return row.__class__ is dict',
    'def check(row):\n return getattr(row, "items")() is None',
    'def check(row):\n return open("/etc/passwd").read() == ""',
    'def check(row):\n return eval(row["x"])',
    'def check(row):\n return globals() is None',
    'def check(row):\n return "{0.__class__}".format(row) == ""',
    '@str\ndef check(row):\n return False',
    'def check(row=print("executed")):\n return False',
    'def check(row):\n def nested():\n  return False\n return nested()',
    'def check(row):\n try:\n  return False\n except:\n  return True',
    'def check(row):\n try:\n  return False\n finally:\n  while True:\n   pass',
    'def check(row):\n re.search = str\n return False',
    'def check(row):\n global re\n return False',
    'def check(row):\n return check(row)',
    'def check(row):\n break\n return False',
])
def test_unsafe_or_invalid_syntax_never_runs(source):
    candidate = rule(source)
    accepted, report = sandbox.validate_rules([candidate], [{"x": "a"}])
    assert accepted == []
    assert report[0]["status"] == "invalid_rule"
    assert report[0]["reason"]
    assert sandbox.evaluate_rules([candidate], [{"x": "a"}]) == [[-1]]


def test_rule_timeout_is_separate_and_later_rules_still_execute(monkeypatch):
    monkeypatch.setattr(sandbox, "_RULE_TIMEOUT_SECONDS", .05)
    candidates = [
        rule('def check(row):\n while True:\n  pass', ident="infinite"),
        rule('def check(row):\n return row["x"] == "bad"', ident="good"),
    ]
    start = time.monotonic()
    values = sandbox.evaluate_rules(candidates, [{"x": "bad"}, {"x": "ok"}])
    assert values == [[-1, 1], [-1, 0]]
    assert time.monotonic() - start < 5
    accepted, report = sandbox.validate_rules(candidates, [{"x": "bad"}])
    assert accepted == [candidates[1]]
    assert report[0]["status"] == "timeout"


def test_rows_are_immutable_and_rules_do_not_share_mutation():
    rows = [{"x": "original"}]
    candidates = [
        rule('def check(row):\n row["x"] = "changed"\n return False'),
        rule('def check(row):\n return row["x"] == "original"', ident="later"),
    ]
    assert sandbox.evaluate_rules(candidates, rows) == [[-1, 1]]
    assert rows == [{"x": "original"}]


def test_memory_limit_abstains_without_stopping_the_next_rule():
    candidates = [
        rule('def check(row):\n blob = "x" * (1024 ** 3)\n return len(blob) == 0', ident="memory"),
        rule('def check(row):\n return False', ident="later"),
    ]
    assert sandbox.evaluate_rules(candidates, [{"x": "a"}]) == [[-1, 0]]


def test_bad_rule_does_not_poison_valid_rules():
    candidates = [
        rule('def check(row):\n return 1 / 0 == 0'),
        rule('def check(row):\n return', ident="none"),
        rule('def check(row):\n return row["x"] == "ok"', ident="valid"),
    ]
    accepted, report = sandbox.validate_rules(candidates, [{"x": "ok"}])
    assert accepted == [candidates[2]]
    assert [item["error_count"] for item in report] == [1, 1, 0]


def test_empty_inputs_and_schema_validation():
    candidate = rule('def check(row):\n return False')
    assert sandbox.evaluate_rules([], [{"x": "v"}]) == [[]]
    assert sandbox.evaluate_rules([candidate], []) == []
    accepted, report = sandbox.validate_rules([candidate], [])
    assert accepted == []
    assert report[0]["reason"] == "no validation rows"
    assert sandbox.evaluate_rules([candidate], [{"other": "v"}]) == [[-1]]
    for malformed in ({}, None, {**candidate, "kind": "unsupported"}):
        accepted, report = sandbox.validate_rules([malformed], [{"x": "a"}])
        assert not accepted
        assert report[0]["status"] == "invalid_rule"
    with pytest.raises(TypeError):
        sandbox.evaluate_rules([candidate], [{"x": 1}])
    with pytest.raises(ValueError):
        sandbox.validate_rules([candidate], [{"x": "a"}], max_error_rate=float("nan"))


def test_unavailable_sandbox_fails_closed(monkeypatch):
    monkeypatch.setattr(sandbox, "_bubblewrap_prefix", lambda: ())

    def unavailable(*args, **kwargs):
        raise OSError("sandbox host unavailable")

    monkeypatch.setattr(sandbox.subprocess, "Popen", unavailable)
    candidate = rule('def check(row):\n return False')
    assert sandbox.evaluate_rules([candidate], [{"x": "a"}]) == [[-1]]
    accepted, report = sandbox.validate_rules([candidate], [{"x": "a"}])
    assert accepted == []
    assert "unavailable" in report[0]["reason"]


def test_seccomp_blocks_file_network_and_process_syscalls():
    # Probe the actual OS boundary in a sacrificial interpreter, independently
    # of the AST. Every forbidden syscall must fail even for trusted Python.
    module_path = str(Path(sandbox.__file__).resolve())
    source = textwrap.dedent(f'''
        import errno, json, os, runpy, socket
        namespace = runpy.run_path({module_path!r}, run_name="seccomp_probe")
        namespace["_install_seccomp"]()
        denied = []
        for action in (
            lambda: open("/etc/passwd", "rb"),
            lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM),
            os.fork,
        ):
            try:
                result = action()
                if isinstance(result, int) and result == 0:
                    os._exit(90)
                denied.append(False)
            except OSError as exc:
                denied.append(exc.errno == errno.EPERM)
        print(json.dumps(denied))
    ''')
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-c", source], capture_output=True,
        text=True, timeout=5, check=True,
    )
    assert json.loads(process.stdout) == [True, True, True]


def test_worker_never_evaluates_source_if_seccomp_cannot_load():
    module_path = str(Path(sandbox.__file__).resolve())
    source = textwrap.dedent(f'''
        import runpy
        namespace = runpy.run_path({module_path!r}, run_name="unavailable_probe")
        def unavailable(*args):
            raise RuntimeError("seccomp disabled by test")
        namespace["_worker_main"].__globals__["_install_seccomp"] = unavailable
        raise SystemExit(namespace["_worker_main"]())
    ''')
    payload = {
        "rows": [{"x": "a"}],
        "rules": [{"index": 0, "source": 'def check(row):\n return False'}],
        "memory_limit": sandbox._MEMORY_LIMIT_BYTES,
        "batch_timeout": 5, "rule_timeout": 1,
    }
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-c", source], input=json.dumps(payload),
        capture_output=True, text=True, timeout=5,
    )
    assert process.returncode == 2
    message = json.loads(process.stdout)
    assert "seccomp disabled by test" in message["fatal"]
    assert "values" not in message


def test_many_rules_share_one_worker(monkeypatch):
    monkeypatch.setattr(sandbox, "_bubblewrap_prefix", lambda: ())
    real_popen = sandbox.subprocess.Popen
    launches = []

    def counted(*args, **kwargs):
        launches.append(args[0])
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(sandbox.subprocess, "Popen", counted)
    candidates = [rule('def check(row):\n return len(row["x"]) == 0', ident=str(index)) for index in range(30)]
    rows = [{"x": "a"}] * 1000
    values = sandbox.evaluate_rules(candidates, rows)
    assert len(launches) == 1
    assert len(values) == 1000
    assert all(value == 0 for value in values[0])
    assert len(values[0]) == 30


def test_hash_based_collection_iteration_is_reproducible():
    candidate = rule('''
        def check(row):
            options = set(row["x"].split(","))
            return list(options)[0] == "alpha"
    ''')
    rows = [{"x": "alpha,beta,gamma,delta,epsilon"}]
    expected = sandbox.evaluate_rules([candidate], rows)
    assert expected[0][0] in (0, 1)
    assert all(sandbox.evaluate_rules([candidate], rows) == expected for _ in range(4))
