"""Hash-bound dirty-only format-contract review and atomic parent overlays.

Importing performs no IO. Author scores never become error probabilities.
The CLI reads only frozen author inputs, load_dirty, and the named source_v2
parent; it never evaluates benchmark correctness or invokes core inference.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from zeroclean.data import DATASETS, load_dirty
from zeroclean.sandbox import evaluate_rules, validate_rules
from zeroclean.statistics import DirtyIndex, pattern
from zeroclean.types import Prediction, digest


VERSION = "format_overlay_v1"
PROJECT = Path(__file__).resolve().parents[1]
AUTHOR_ROOT = PROJECT / "experimental/format_contracts_v1"
PROTOCOL_SHA256 = "1f87f7d0f57c862f16929fbec90c84396046254ed833ec2d5b820a6ad631a177"
SCHEMA_SHA256 = "16a5b26ecd5c6448e47c26a0f95303b98725d73314aa85e00f548c0ff1cb1ecc"
DIMENSIONS = ["rule_satisfaction", "fd_support", "nmi_vicinity", "domain_support"]


@dataclass(frozen=True)
class OverlayPolicy:
    original_valid_min: float = .9
    original_invalid_max: float = .1
    decision_confidence_min: float = .8
    verified_repair_confidence_min: float = .8
    detection_threshold: float = .5
    valid_detection_cap: float = .1
    invalid_detection_floor: float = .9
    delta_weights: tuple[float, ...] = (2.2, 1.8, .65, .35)
    delta_adjusted_log_support_min: float = math.log(4.)
    overlap_policy: str = "preserve_all_overlapping_selected_action_footprints"
    final_guard: str = "approved_rules_parent_relative_whole_row_exact_ED_EC_rollback"
    statistical_observations: str = "raw_dirty_only_leave_one_out"


def policy_body():
    return json.loads(json.dumps(asdict(OverlayPolicy())))


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def text_hash(value):
    return sha256_bytes(value.encode("utf-8"))


def author_identity(body):
    """Author ID serialization deliberately differs from zeroclean.digest."""
    return sha256_bytes(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode("utf-8"))


def scalar_identity(dataset, row, column, source_value, candidate):
    return author_identity({"dataset": dataset, "row": row, "column": column,
                            "source_value": source_value, **{k: candidate[k] for k in
                            ("contract_id", "transform_id", "value")}})


def bundle_identity(dataset, bundle):
    return author_identity({"dataset": dataset, **{k: bundle[k] for k in
                            ("row", "contract_id", "transform_id", "source_values", "updates", "application_mode")}})


def read_json(path):
    return json.loads(_read_bytes(path))


def _no_symlinks(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlink input/output paths are forbidden: " + str(path))
    return path


def _read_bytes(path):
    path = _no_symlinks(path)
    if not path.is_file():
        raise ValueError("Expected a regular frozen input file: " + str(path))
    return path.read_bytes()


def confined_path(root, relative):
    """Validate an untrusted manifest name before opening any file."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("Invalid prepared snapshot path")
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Prepared snapshot path escapes its directory")
    root = _no_symlinks(root).resolve()
    candidate = _no_symlinks(root / name)
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("Prepared snapshot path escapes its directory")
    return candidate


def write_once(path, body):
    path = _no_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def seal(body, field="request_hash"):
    body.pop(field, None)
    body[field] = digest(body)
    return body


def check_seal(body, field="request_hash"):
    if body.get(field) != digest({k: v for k, v in body.items() if k != field}):
        raise ValueError("Content hash mismatch: " + field)


def prediction_body(parent):
    return {"probabilities": parent.probabilities.tolist(),
            "repairs": [{"row": i, "column": j, "value": v}
                        for (i, j), v in sorted(parent.repairs.items())]}


def parent_rows(table, parent):
    parent.validate(table.shape)
    rows = [list(row) for row in table.rows]
    for (i, j), value in parent.repairs.items():
        if parent.probabilities[i, j] < OverlayPolicy().detection_threshold:
            raise ValueError("Parent repair lies outside its ED mask")
        rows[i][j] = value
    return rows


def _entries(body, key):
    entries = body if isinstance(body, list) else body.get(key, [])
    if not isinstance(entries, list):
        raise ValueError("Expected list: " + key)
    return entries


def _contracts(body):
    if isinstance(body, list):
        entries = body
    else:
        entries = body.get("contracts", body.get("columns", []))
    flattened = []
    for entry in entries:
        if "contract_id" in entry:
            flattened.append(entry)
        flattened.extend(entry.get("contracts", []))
    result = {}
    for entry in flattened:
        cid = entry.get("contract_id")
        if not isinstance(cid, str) or not cid or cid in result:
            raise ValueError("Missing or duplicate contract ID")
        result[cid] = deepcopy(entry)
    return result


def _coordinate(table, row, column):
    if type(row) is not int or not 0 <= row < table.shape[0] or column not in table.columns:
        raise ValueError("Invalid authored coordinate")
    return row, table.columns.index(column)


def _score(value, *, nullable=False):
    if value is None and nullable:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Scores must be finite numbers in [0,1] or explicit null")
    return float(value)


def _matrix(evaluator, rules, records):
    matrix = np.asarray(evaluator(rules, records), dtype=np.int8).reshape(len(records), len(rules))
    if not np.isin(matrix, [-1, 0, 1]).all():
        raise ValueError("Invalid sandbox result")
    return matrix


def _rule_columns(rule, columns):
    """Identify literal row references; unknown/dynamic scope stays multi-column."""
    found = set()
    try:
        tree = ast.parse(rule.get("source", ""))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or node.id != "row":
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.Subscript) and parent.value is node:
                key = parent.slice
            elif isinstance(parent, ast.Attribute) and parent.value is node and parent.attr == "get":
                call = parents.get(parent)
                if not isinstance(call, ast.Call) or call.func is not parent or not call.args:
                    return list(columns)
                key = call.args[0]
            else:
                # Aliases, iteration, dynamic mapping methods and helper calls
                # cannot safely narrow a checker to its literal subscript uses.
                return list(columns)
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return list(columns)
            found.add(key.value)
    except (SyntaxError, TypeError):
        return list(columns)
    return sorted(found) if found else list(columns)


def _rule_contract(rule, contracts):
    cid = rule.get("contract_id")
    if cid in contracts:
        return cid
    matches = [cid for cid, contract in contracts.items()
               if rule.get("id") in contract.get("checker_ids", [])]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("Every rule must identify exactly one declared format contract")


def prepare_request(table, parent, pass4, column_contracts, proposals, joint_candidates,
                    *, bindings=None, rule_validator=validate_rules,
                    rule_evaluator=evaluate_rules, index_factory=DirtyIndex):
    """Pure preparation; injection points exist only for synthetic verification.

    Exact originals, current parent state, frozen author objects, contexts and
    raw-dirty-only counterfactuals are all in the content-addressed request.
    """
    if pass4.get("dataset") != table.name or pass4.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Pass4 does not bind this dirty input")
    if proposals.get("dataset") != table.name or proposals.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Proposals do not bind this dirty input")
    if (not isinstance(column_contracts, dict) or column_contracts.get("dataset") != table.name
            or column_contracts.get("dirty_sha256") != table.fingerprint
            or column_contracts.get("uniform_protocol_sha256") != PROTOCOL_SHA256
            or {entry.get("column") for entry in column_contracts.get("columns", [])} != set(table.columns)):
        raise ValueError("Column contracts must bind this dirty input/protocol and cover every column")
    if proposals.get("max_candidates", 3) != 3:
        raise ValueError("Author must retain fixed maximum of three scalar candidates")
    contracts = _contracts(column_contracts)
    prior = parent_rows(table, parent)
    prior_records = [dict(zip(table.columns, row)) for row in prior]
    all_rules = pass4.get("rules", [])
    ids = [rule.get("id") for rule in all_rules]
    if len(set(ids)) != len(ids) or any(not isinstance(x, str) for x in ids):
        raise ValueError("Missing or duplicate checker ID")
    rule_contracts = {rule["id"]: _rule_contract(rule, contracts) for rule in all_rules}
    rules, validation = rule_validator(all_rules, table.records(), max_error_rate=.05)
    rule_ids = [rule["id"] for rule in rules]
    scopes = {rule["id"]: _rule_columns(rule, table.columns) for rule in rules}
    raw_matrix = _matrix(rule_evaluator, rules, table.records())
    parent_matrix = _matrix(rule_evaluator, rules, prior_records)
    index = index_factory(table, pass4.get("dependencies", []), discover=True)
    raw_counts = [Counter(row[j] for row in table.rows) for j in range(table.shape[1])]
    raw_patterns = [Counter(pattern(row[j]) for row in table.rows) for j in range(table.shape[1])]
    # The sandbox batches contexts; never launch a process per cell/candidate.
    def row_key(record):
        return tuple(record[column] for column in table.columns)
    result_cache = {row_key(record): raw_matrix[i] for i, record in enumerate(table.records())}
    result_cache.update({row_key(record): parent_matrix[i] for i, record in enumerate(prior_records)})
    queued = {}
    for authored in _entries(proposals, "cells"):
        i, j = _coordinate(table, authored.get("row"), authored.get("column"))
        before = dict(prior_records[i])
        before[table.columns[j]] = table.rows[i][j]
        queued[row_key(before)] = before
        for candidate in authored.get("candidates", []):
            after = dict(before)
            value = candidate.get("value")
            if not isinstance(value, str):
                raise ValueError("Scalar candidate must contain an exact string")
            after[table.columns[j]] = value
            queued[row_key(after)] = after
    for authored in _entries(joint_candidates, "bundles"):
        i = authored.get("row")
        if type(i) is not int or not 0 <= i < table.shape[0]:
            raise ValueError("Invalid atomic bundle row")
        if not isinstance(authored.get("source_values"), dict):
            raise ValueError("Missing atomic source tuple")
        before = dict(prior_records[i])
        before.update(authored["source_values"])
        after = dict(prior_records[i])
        for update in authored.get("updates", []):
            after[update.get("column")] = update.get("value")
        queued[row_key(before)] = before
        queued[row_key(after)] = after
    pending = [record for key, record in queued.items() if key not in result_cache]
    for start in range(0, len(pending), 1000):
        chunk = pending[start:start+1000]
        matrix = _matrix(rule_evaluator, rules, chunk)
        result_cache.update({row_key(record): matrix[k] for k, record in enumerate(chunk)})
    seen_cells, seen_candidates, seen_bundles = set(), set(), set()
    cells, bundles = [], []

    def context(i):
        return {"raw_dirty_tuple": dict(zip(table.columns, table.rows[i])),
                "exact_parent_tuple": prior_records[i],
                "raw_dirty_tuple_sha256": digest(table.rows[i]),
                "exact_parent_tuple_sha256": digest(prior_records[i]),
                "raw_rule_results": raw_matrix[i].tolist(),
                "parent_rule_results": parent_matrix[i].tolist()}

    def evidence(i, sources, updates):
        before, after = deepcopy(prior_records[i]), deepcopy(prior_records[i])
        before.update(sources)
        after.update(updates)
        bm, am = result_cache[row_key(before)], result_cache[row_key(after)]
        touched = set(updates)
        relevant = [k for k, rule in enumerate(rules)
                    if rule.get("target") in touched or touched.intersection(scopes[rule["id"]])]
        rule_delta = (int((bm[relevant] == 1).sum()) - int((am[relevant] == 1).sum())) / max(len(relevant), 1)
        btuple = tuple(before[c] for c in table.columns)
        atuple = tuple(after[c] for c in table.columns)
        fd, vicinity, domain = [], [], []
        per_field = []
        for column, value in sorted(updates.items()):
            j = table.columns.index(column)
            raw = table.rows[i][j]
            bs, be = index.conditional(i, j, before[column], btuple)
            av, ae = index.conditional(i, j, value, atuple)
            fd.append(av - bs)
            vicinity.append(math.tanh((index.vicinity(i, j, value, atuple) - index.vicinity(i, j, before[column], btuple)) / 4))
            domain.append(math.tanh((index.domain_support(i, j, value) - index.domain_support(i, j, before[column])) / 4))
            counts, shapes = raw_counts[j], raw_patterns[j]
            per_field.append({"column": column, "raw_dirty_original": raw,
                              "before_value": before[column], "after_value": value,
                              "before_leave_one_out_value_count": counts[before[column]] - int(before[column] == raw),
                              "after_leave_one_out_value_count": counts[value] - int(value == raw),
                              "before_leave_one_out_pattern_count": shapes[pattern(before[column])] - int(pattern(before[column]) == pattern(raw)),
                              "after_leave_one_out_pattern_count": shapes[pattern(value)] - int(pattern(value) == pattern(raw)),
                              "before_fd_evidence": be, "after_fd_evidence": ae})
        dimensions = [float(rule_delta), *[float(sum(xs) / max(len(xs), 1)) for xs in (fd, vicinity, domain)]]
        details = [{"rule_id": rule["id"], "contract_id": rule_contracts[rule["id"]],
                    "scope": scopes[rule["id"]], "before": int(bm[k]), "after": int(am[k]),
                    "parent": int(parent_matrix[i, k])} for k, rule in enumerate(rules)]
        return {"before_tuple": before, "after_tuple": after,
                "before_tuple_sha256": digest(before), "after_tuple_sha256": digest(after),
                "before_rule_results": bm.tolist(), "after_rule_results": am.tolist(),
                "single_column_evidence": [x for x in details if len(x["scope"]) == 1],
                "multi_column_evidence": [x for x in details if len(x["scope"]) != 1],
                "delta_features": dimensions,
                "relevant_rule_ids": [rules[k]["id"] for k in relevant],
                "weighted_delta_support": sum(a*b for a, b in zip(OverlayPolicy().delta_weights, dimensions)),
                "raw_dirty_count_evidence": per_field,
                "statistics_source": "original_dirty_only_never_parent_repairs"}

    for authored in _entries(proposals, "cells"):
        i, j = _coordinate(table, authored.get("row"), authored.get("column"))
        if (i, j) in seen_cells or type(authored.get("column_index")) is not int or authored.get("column_index") != j:
            raise ValueError("Duplicate scalar cell or column-index mismatch")
        seen_cells.add((i, j))
        raw = table.rows[i][j]
        if (authored.get("source_value") != raw or authored.get("source_value_sha256") != text_hash(raw)
                or authored.get("application_precondition", {}).get("parent_value_must_equal") != raw):
            raise ValueError("Scalar raw value/hash/precondition mismatch")
        candidates = authored.get("candidates", [])
        if len(candidates) > 3:
            raise ValueError("More than three scalar alternatives")
        values = set()
        cell_id = digest({"dataset": table.name, "row": i, "column": table.columns[j], "original": raw})
        cell = {"cell_id": cell_id, "row": i, "column_index": j, "column": table.columns[j],
                "raw_dirty_original": raw, "exact_parent_value": prior[i][j],
                "parent_probability": float(parent.probabilities[i, j]),
                "parent_repair": parent.repairs.get((i, j)), "context": context(i),
                "author_cell": deepcopy(authored), "author_cell_sha256": digest(authored), "candidates": []}
        for candidate in candidates:
            cid, value = candidate.get("candidate_id"), candidate.get("value")
            if (not isinstance(cid, str) or not cid or cid in seen_candidates or not isinstance(value, str)
                    or value in values or value == raw or candidate.get("contract_id") not in contracts
                    or not isinstance(candidate.get("transform_id"), str)
                    or cid != scalar_identity(table.name, i, table.columns[j], raw, candidate)):
                raise ValueError("Invalid exact scalar candidate/contract/transform")
            seen_candidates.add(cid)
            values.add(value)
            cell["candidates"].append({"candidate_id": cid, "author_candidate": deepcopy(candidate),
                                       "author_candidate_sha256": digest(candidate),
                                       "parent_precondition_satisfied": prior[i][j] == raw,
                                       "evidence": evidence(i, {table.columns[j]: raw}, {table.columns[j]: value})})
        cells.append(cell)
    for authored in _entries(joint_candidates, "bundles"):
        bid, i = authored.get("bundle_id"), authored.get("row")
        sources = authored.get("source_values")
        updates_list = authored.get("updates", [])
        if (not isinstance(bid, str) or not bid or bid in seen_bundles or bid in seen_candidates
                or not isinstance(sources, dict) or not sources or not updates_list
                or authored.get("contract_id") not in contracts or not isinstance(authored.get("transform_id"), str)
                or authored.get("application_mode") != "atomic_all_or_nothing"
                or authored.get("application_precondition", {}).get("parent_values_must_equal") != sources):
            raise ValueError("Invalid atomic bundle/preconditions")
        seen_bundles.add(bid)
        for column, value in sources.items():
            _, j = _coordinate(table, i, column)
            if not isinstance(value, str) or table.rows[i][j] != value:
                raise ValueError("Atomic bundle source does not match original dirty tuple")
        updates = {}
        for update in updates_list:
            column, value = update.get("column"), update.get("value")
            if column not in sources or column in updates or not isinstance(value, str):
                raise ValueError("Atomic update absent from complete source preconditions or duplicate")
            updates[column] = value
        if list(updates) != sorted(updates, key=table.columns.index):
            raise ValueError("Atomic updates must retain declared table column order")
        if not any(updates[column] != sources[column] for column in updates):
            raise ValueError("Empty atomic conversion")
        if bid != bundle_identity(table.name, authored):
            raise ValueError("Atomic bundle content ID mismatch")
        bundles.append({"bundle_id": bid, "row": i, "context": context(i),
                        "author_bundle": deepcopy(authored), "author_bundle_sha256": digest(authored),
                        "parent_precondition_satisfied": all(prior_records[i][c] == v for c, v in sources.items()),
                        "evidence": evidence(i, sources, updates)})
    return seal({"version": VERSION, "dataset": table.name, "dirty_sha256": table.fingerprint,
                 "columns": list(table.columns), "shape": list(table.shape), "policy": policy_body(),
                 "protocol_sha256": PROTOCOL_SHA256, "artifact_schema_sha256": SCHEMA_SHA256,
                 "bindings": deepcopy(bindings or {}), "parent_snapshot": prediction_body(parent),
                 "parent_snapshot_sha256": digest(prediction_body(parent)),
                 "author_artifacts_sha256": {"pass4": digest(pass4), "column_contracts": digest(column_contracts),
                                              "proposals": digest(proposals), "joint_candidates": digest(joint_candidates)},
                 "contracts": list(contracts.values()), "column_contracts": deepcopy(column_contracts),
                 "rules": deepcopy(rules), "all_authored_rules": deepcopy(all_rules), "rule_validation": validation,
                 "rule_contracts": rule_contracts, "rule_ids": rule_ids, "rule_scopes": scopes,
                 "dependencies": deepcopy(pass4.get("dependencies", [])), "dependency_count": len(index.dependencies),
                 "delta_dimensions": DIMENSIONS, "cells": cells, "atomic_bundles": bundles,
                 "author_scores_are_uncalibrated": True, "parent_is_untrusted_not_labels": True,
                 "benchmark_data_accessed": False, "evaluation_status": "not_run",
                 "review_requirements": REVIEW_REQUIREMENTS})


REVIEW_REQUIREMENTS = {
    "coverage": "Exactly one decision for every contract, scalar candidate, atomic bundle and cell; no extra IDs",
    "bindings": "Echo request_hash, protocol_sha256, artifact_schema_sha256 and author_manifest_sha256",
    "contract": "contract_id, decision approve|abstain|reject, rationale, approved_transform_ids",
    "scalar_candidate": "candidate_id, decision, rationale, exact_value, verified_confidence (number or null)",
    "atomic_bundle": "bundle_id, decision, rationale, exact_updates, atomicity_confirmed, verified_confidence (number or null)",
    "cell": "cell_id, original_valid_score (number or null), decision_confidence (number or null), selected_candidate_id (ID or null), selected_value (exact string or null), invalidity_supported, invalidity_rule_ids, retract_parent_repair, retraction_supported, rationale",
    "bundle_selection": "Approved exact bundle is an explicit atomic selection; every changed field needs independently supported original invalidity via its cell review or original_valid_scores/decision_confidences/invalidity_rule_ids_by_column in bundle decision",
    "unknown": "Null validity/confidence or abstain preserves exact parent; explicit invalidity alone never authorizes repair",
    "scores": "All scores are uncalibrated judgments, never fitted or calibrated error probabilities",
    "independence": "Root-designated reviewer must differ from author; no benchmark/reference/evaluation access",
    "default_for_unreviewed": "abstain_preserve_parent",
}


def validate_request(table, parent, request):
    check_seal(request)
    if (request.get("version") != VERSION or request.get("dataset") != table.name
            or request.get("dirty_sha256") != table.fingerprint
            or request.get("shape") != list(table.shape) or request.get("columns") != list(table.columns)
            or request.get("policy") != policy_body() or request.get("delta_dimensions") != DIMENSIONS
            or request.get("protocol_sha256") != PROTOCOL_SHA256
            or request.get("artifact_schema_sha256") != SCHEMA_SHA256
            or request.get("benchmark_data_accessed") is not False
            or request.get("parent_snapshot_sha256") != digest(prediction_body(parent))
            or request.get("parent_snapshot") != prediction_body(parent)):
        raise ValueError("Request policy/dirty/parent binding mismatch")
    parent_rows(table, parent)


def validate_frozen_replay(table, parent, pass4, contracts, proposals, bundles,
                           request, *, expected_bindings, **dependencies):
    """Recompute from authoritative snapshots; a self-rehashed request is insufficient."""
    validate_request(table, parent, request)
    if request.get("bindings") != expected_bindings:
        raise ValueError("Request bindings differ from authoritative frozen inputs")
    expected = prepare_request(table, parent, pass4, contracts, proposals, bundles,
                               bindings=expected_bindings, **dependencies)
    if request != expected:
        raise ValueError("Request differs from authoritative frozen-input replay")


def _decision_map(entries, key, expected):
    if not isinstance(entries, list):
        raise ValueError("Review decision list missing: " + key)
    result = {}
    for entry in entries:
        identity = entry.get(key)
        if identity not in expected or identity in result:
            raise ValueError("Unknown or duplicate review ID: " + key)
        if entry.get("decision") not in ("approve", "abstain", "reject"):
            raise ValueError("Unknown review decision")
        if not isinstance(entry.get("rationale"), str) or not entry["rationale"].strip():
            raise ValueError("Every review decision requires a rationale")
        result[identity] = entry
    if set(result) != set(expected):
        raise ValueError("Incomplete independent review coverage: " + key)
    return result


def complete_verification(table, parent, request, reply):
    validate_request(table, parent, request)
    if any(reply.get(key) != request.get(key) for key in
           ("request_hash", "protocol_sha256", "artifact_schema_sha256")):
        raise ValueError("Independent review hash binding mismatch")
    manifest_hash = request["bindings"].get("author_manifest_sha256")
    if (reply.get("author_manifest_sha256") != manifest_hash
            or reply.get("model") != "gpt-6.1-sol" or reply.get("reasoning_effort") != "xhigh"
            or reply.get("benchmark_data_accessed") is not False
            or reply.get("default_for_unreviewed") != "abstain_preserve_parent"):
        raise ValueError("Independent review provenance/policy mismatch")
    contracts = _decision_map(reply.get("contracts"), "contract_id", {x["contract_id"] for x in request["contracts"]})
    candidates = {x["candidate_id"]: x for cell in request["cells"] for x in cell["candidates"]}
    scalar = _decision_map(reply.get("scalar_candidates"), "candidate_id", candidates)
    bundled = {x["bundle_id"]: x for x in request["atomic_bundles"]}
    bundles = _decision_map(reply.get("atomic_bundles"), "bundle_id", bundled)
    for item in contracts.values():
        if (not isinstance(item.get("approved_transform_ids"), list)
                or any(not isinstance(x, str) for x in item["approved_transform_ids"])
                or len(set(item["approved_transform_ids"])) != len(item["approved_transform_ids"])):
            raise ValueError("Contract must explicitly enumerate unique approved transforms")
    for cid, item in scalar.items():
        if item.get("exact_value") != candidates[cid]["author_candidate"]["value"]:
            raise ValueError("Independent scalar output differs from exact frozen alternative")
        _score(item.get("verified_confidence"), nullable=True)
        if "verified_confidence" not in item:
            raise ValueError("Missing explicit candidate verified confidence")
    for bid, item in bundles.items():
        if item.get("exact_updates") != bundled[bid]["author_bundle"]["updates"]:
            raise ValueError("Independent bundle differs from exact full frozen updates")
        if type(item.get("atomicity_confirmed")) is not bool:
            raise ValueError("Missing explicit atomicity decision")
        _score(item.get("verified_confidence"), nullable=True)
        if "verified_confidence" not in item:
            raise ValueError("Missing explicit bundle verified confidence")
    expected_cells = {cell["cell_id"]: cell for cell in request["cells"]}
    cells = {}
    if not isinstance(reply.get("cells"), list):
        raise ValueError("Missing independent cell validity decisions")
    for item in reply["cells"]:
        cid = item.get("cell_id")
        if cid not in expected_cells or cid in cells:
            raise ValueError("Unknown/duplicate independent cell ID")
        for key in ("original_valid_score", "decision_confidence", "selected_candidate_id", "selected_value",
                    "invalidity_supported", "invalidity_rule_ids", "retract_parent_repair", "retraction_supported", "rationale"):
            if key not in item:
                raise ValueError("Missing explicit independent cell field: " + key)
        _score(item["original_valid_score"], nullable=True)
        _score(item["decision_confidence"], nullable=True)
        if any(type(item[key]) is not bool for key in ("invalidity_supported", "retract_parent_repair", "retraction_supported")):
            raise ValueError("Cell support/retraction fields must be explicit booleans")
        if not isinstance(item["rationale"], str) or not item["rationale"].strip() or not isinstance(item["invalidity_rule_ids"], list):
            raise ValueError("Missing cell rationale/rule evidence")
        cell = expected_cells[cid]
        selected = item["selected_candidate_id"]
        alternatives = {x["candidate_id"]: x for x in cell["candidates"]}
        if selected is None:
            if item["selected_value"] not in (None, cell["raw_dirty_original"]):
                raise ValueError("Unidentified selected scalar value")
        elif selected not in alternatives or item["selected_value"] != alternatives[selected]["author_candidate"]["value"]:
            raise ValueError("Independent scalar ID/value mismatch")
        if item["original_valid_score"] is not None and item["original_valid_score"] >= OverlayPolicy().original_valid_min and selected is not None:
            raise ValueError("Original-valid and nonoriginal-repair decisions conflict")
        cells[cid] = item
    if set(cells) != set(expected_cells):
        raise ValueError("Incomplete independent cell validity coverage")
    return contracts, scalar, bundles, cells


def _contract_approved(authored, contracts):
    contract = contracts[authored["contract_id"]]
    return contract["decision"] == "approve" and authored["transform_id"] in contract["approved_transform_ids"]


def _supported_invalidity(request, column, validity, confidence, supported, rule_ids,
                          results, contracts):
    if (validity is None or confidence is None or validity > OverlayPolicy().original_invalid_max
            or confidence < OverlayPolicy().decision_confidence_min or supported is not True
            or not isinstance(rule_ids, list) or not rule_ids):
        return False
    accepted = {rule["id"]: (k, rule) for k, rule in enumerate(request["rules"])}
    for rid in rule_ids:
        if rid not in accepted:
            return False
        k, rule = accepted[rid]
        if (contracts[request["rule_contracts"][rid]]["decision"] != "approve"
                or results[k] != 1 or column not in request["rule_scopes"][rid]):
            return False
    return True


def _repair_gate(confidence, evidence):
    if confidence is None or confidence < OverlayPolicy().verified_repair_confidence_min:
        return False
    # This is an uncalibrated support gate, explicitly not a probability update.
    clipped = min(max(confidence, 1e-9), 1 - 1e-9)
    support = math.log(clipped / (1 - clipped)) + evidence["weighted_delta_support"]
    return support >= OverlayPolicy().delta_adjusted_log_support_min - 1e-12


def _reviewed_evidence(request, evidence, contracts):
    """Rejected/abstained contracts never contribute positive or negative votes."""
    result = deepcopy(evidence)
    positions = [k for k, rid in enumerate(request["rule_ids"])
                 if rid in evidence["relevant_rule_ids"]
                 and contracts[request["rule_contracts"][rid]]["decision"] == "approve"]
    before, after = evidence["before_rule_results"], evidence["after_rule_results"]
    result["delta_features"][0] = (sum(before[k] == 1 for k in positions) - sum(after[k] == 1 for k in positions)) / max(len(positions), 1)
    result["weighted_delta_support"] = sum(w*d for w, d in zip(OverlayPolicy().delta_weights, result["delta_features"]))
    return result


def apply_overlay(table, parent, request, reply, *, rule_evaluator=evaluate_rules):
    """Apply exact approved choices, preserving parent on uncertainty or overlap.

    The CLI additionally checks external provenance and replays frozen author
    files before this pure processor. All tentative operations are simultaneous.
    """
    contracts, scalar_reviews, bundle_reviews, cell_reviews = complete_verification(table, parent, request, reply)
    prior = parent_rows(table, parent)
    policy = OverlayPolicy()
    operations, trace = [], []
    cell_lookup = {(x["row"], x["column"]): x for x in request["cells"]}
    for cell in request["cells"]:
        i, j = cell["row"], cell["column_index"]
        item = cell_reviews[cell["cell_id"]]
        record = {"id": cell["cell_id"], "kind": "scalar_cell", "row": i, "column": cell["column"],
                  "action": "preserve_parent", "reason": "unknown_or_insufficient_independent_support"}
        trace.append(record)
        validity, decision_confidence = item["original_valid_score"], item["decision_confidence"]
        if validity is None or decision_confidence is None or decision_confidence < policy.decision_confidence_min:
            continue
        footprint = {(i, cell["column"])}
        if validity >= policy.original_valid_min:
            inherited = (i, j) in parent.repairs and prior[i][j] != cell["raw_dirty_original"]
            if inherited and not (item["retract_parent_repair"] and item["retraction_supported"]
                                  and item["selected_value"] == cell["raw_dirty_original"]):
                record["reason"] = "valid_raw_original_does_not_implicitly_retract_inherited_repair"
                continue
            operations.append({"footprint": footprint, "updates": {cell["column"]: cell["raw_dirty_original"]},
                               "row": i, "mode": "valid_prune", "trace": record})
            continue
        invalid = _supported_invalidity(request, cell["column"], validity, decision_confidence,
                                       item["invalidity_supported"], item["invalidity_rule_ids"],
                                       cell["context"]["raw_rule_results"], contracts)
        if not invalid:
            continue
        op = {"footprint": footprint, "updates": {}, "row": i, "mode": "invalid_flag", "trace": record}
        selected = item["selected_candidate_id"]
        if selected is not None:
            candidate = next(x for x in cell["candidates"] if x["candidate_id"] == selected)
            reviewed = scalar_reviews[selected]
            authored = candidate["author_candidate"]
            if (reviewed["decision"] == "approve" and candidate["parent_precondition_satisfied"]
                    and _contract_approved(authored, contracts)
                    and _repair_gate(reviewed["verified_confidence"], _reviewed_evidence(request, candidate["evidence"], contracts))):
                op["updates"] = {cell["column"]: authored["value"]}
                op["mode"] = "exact_scalar_repair"
        operations.append(op)
    for bundle in request["atomic_bundles"]:
        authored, item = bundle["author_bundle"], bundle_reviews[bundle["bundle_id"]]
        i = authored["row"]
        record = {"id": bundle["bundle_id"], "kind": "atomic_bundle", "row": i,
                  "action": "preserve_parent", "reason": "unknown_or_insufficient_independent_bundle_support"}
        trace.append(record)
        if (item["decision"] != "approve" or not item["atomicity_confirmed"]
                or not bundle["parent_precondition_satisfied"] or not _contract_approved(authored, contracts)
                or not _repair_gate(item["verified_confidence"], _reviewed_evidence(request, bundle["evidence"], contracts))):
            continue
        updates = {x["column"]: x["value"] for x in authored["updates"]}
        changed = {c: v for c, v in updates.items() if v != authored["source_values"][c]}
        supported = True
        for column in changed:
            cell = cell_lookup.get((i, column))
            if cell:
                decision = cell_reviews[cell["cell_id"]]
                validity, confidence = decision["original_valid_score"], decision["decision_confidence"]
                support, rids = decision["invalidity_supported"], decision["invalidity_rule_ids"]
            else:
                validity = _score(item.get("original_valid_scores", {}).get(column), nullable=True)
                confidence = _score(item.get("decision_confidences", {}).get(column), nullable=True)
                support = item.get("invalidity_supported_by_column", {}).get(column)
                rids = item.get("invalidity_rule_ids_by_column", {}).get(column)
            supported &= _supported_invalidity(request, column, validity, confidence, support, rids,
                                                bundle["context"]["raw_rule_results"], contracts)
        if supported:
            operations.append({"footprint": {(i, c) for c in authored["source_values"]},
                               "updates": updates, "row": i, "mode": "exact_atomic_bundle", "trace": record})
    # ED-only decisions on cells participating in a selected bundle are consumed
    # by that bundle; independent selected repairs/prunes still conflict.
    bundle_fields = {(op["row"], column) for op in operations if op["mode"] == "exact_atomic_bundle"
                     for column, value in op["updates"].items()
                     if value != prior[op["row"]][table.columns.index(column)]}
    operations = [op for op in operations if not (op["mode"] == "invalid_flag" and op["footprint"] <= bundle_fields)]
    conflicts = {a for a, op in enumerate(operations) for b, other in enumerate(operations)
                 if a != b and op["footprint"].intersection(other["footprint"])}
    probabilities, repairs = parent.probabilities.copy(), dict(parent.repairs)
    for position, op in enumerate(operations):
        record, i = op["trace"], op["row"]
        if position in conflicts:
            record["reason"] = "selected_action_footprint_overlap_fail_closed"
            continue
        if op["mode"] == "valid_prune":
            for _, column in op["footprint"]:
                j = table.columns.index(column)
                probabilities[i, j] = min(probabilities[i, j], policy.valid_detection_cap)
                repairs.pop((i, j), None)
        else:
            fields = set(op["updates"]) if op["mode"] == "exact_atomic_bundle" else {c for _, c in op["footprint"]}
            for column in fields:
                j = table.columns.index(column)
                if op["mode"] == "exact_atomic_bundle" and op["updates"][column] == prior[i][j]:
                    continue
                probabilities[i, j] = max(probabilities[i, j], policy.invalid_detection_floor)
            for column, value in op["updates"].items():
                j = table.columns.index(column)
                if value == table.rows[i][j]:
                    repairs.pop((i, j), None)
                else:
                    repairs[i, j] = value
        record.update(action=op["mode"], reason="explicit_independent_decision_passed_fixed_gates")
    proposed = [list(row) for row in table.rows]
    for (i, j), value in repairs.items():
        proposed[i][j] = value
    repair_changed_rows = {key[0] for key in set(repairs) | set(parent.repairs)
                           if repairs.get(key) != parent.repairs.get(key)}
    modified = [i for i in range(table.shape[0])
                if proposed[i] != prior[i] or not np.array_equal(probabilities[i], parent.probabilities[i])
                or i in repair_changed_rows]
    approved_rules = [rule for rule in request["rules"] if contracts[request["rule_contracts"][rule["id"]]]["decision"] == "approve"]
    rollbacks = []
    if modified:
        before = _matrix(rule_evaluator, approved_rules, [dict(zip(table.columns, prior[i])) for i in modified])
        after = _matrix(rule_evaluator, approved_rules, [dict(zip(table.columns, proposed[i])) for i in modified])
        for q, i in enumerate(modified):
            introduced = bool(((before[q] == 0) & (after[q] == 1)).any())
            unknown = bool((before[q] == -1).any() or (after[q] == -1).any())
            if introduced or unknown or not approved_rules:
                probabilities[i, :] = parent.probabilities[i, :]
                repairs = {key: value for key, value in repairs.items() if key[0] != i}
                repairs.update({key: value for key, value in parent.repairs.items() if key[0] == i})
                rollbacks.append({"row": i, "introduced_violation": introduced, "runtime_unknown": unknown,
                                  "restored": "exact_parent_row_probabilities_and_repairs"})
                for record in trace:
                    if record["row"] == i and record["action"] != "preserve_parent":
                        record.update(action="preserve_parent", reason="combined_whole_row_rollback")
    if any(probabilities[i, j] < policy.detection_threshold for i, j in repairs):
        raise ValueError("Overlay repair outside coherent ED mask")
    return Prediction(probabilities, repairs, {"system": VERSION, "dataset": table.name,
                      "dirty_sha256": table.fingerprint, "request_hash": request["request_hash"],
                      "review_sha256": digest(reply), "parent_snapshot_sha256": request["parent_snapshot_sha256"],
                      "policy": policy_body(), "trace": trace, "whole_row_rollbacks": rollbacks,
                      "benchmark_data_accessed": False, "evaluation_status": "not_run"}).validate(table.shape)


AUTHOR_DATASET_FILES = ("pass4.json", "column_contracts.json", "proposals.json",
                        "joint_conversion_candidates.json", "runtime_validation.json",
                        "sampling_provenance.json")


def implementation_hashes():
    files = [Path(__file__).resolve(), PROJECT / "scripts/run_format_overlay.py",
             PROJECT / "experimental/format_overlay_protocol.md",
             PROJECT / "tests/test_format_overlay.py",
             *[PROJECT / ("zeroclean/" + name + ".py") for name in ("data", "types", "sandbox", "statistics")]]
    return {str(path): sha256_bytes(_read_bytes(path)) for path in files}


def check_file_hashes(bindings):
    for filename, expected in bindings.items():
        path = Path(filename)
        if sha256_bytes(_read_bytes(path)) != expected:
            raise ValueError("Frozen file changed: " + filename)


def _manifest_files(manifest):
    entries = manifest.get("files", manifest.get("artifacts", {}))
    result = {}
    if isinstance(entries, dict):
        for name, entry in entries.items():
            result[name] = entry.get("sha256") if isinstance(entry, dict) else entry
    elif isinstance(entries, list):
        for entry in entries:
            name = entry.get("path", entry.get("file"))
            if name in result:
                raise ValueError("Duplicate frozen manifest path")
            result[name] = entry.get("sha256")
    else:
        raise ValueError("Author manifest has no frozen file map")
    return result


def load_author_inputs(author_root=None):
    """Read only declared contract artifacts after a complete six-dataset freeze.

    Profile/benchmark summaries are deliberately not opened. Their hashes may
    occur opaquely in the author manifest. The review request is outside the
    author's core manifest to avoid a manifest/request hash cycle.
    """
    root = _no_symlinks(AUTHOR_ROOT if author_root is None else author_root).resolve()
    if root != AUTHOR_ROOT.resolve():
        raise ValueError("CLI author input root is fixed")
    manifest_path = root / "manifest.json"
    manifest_bytes = _read_bytes(manifest_path)
    manifest = json.loads(manifest_bytes)
    files = _manifest_files(manifest)
    expected = ["protocol.json", "artifact_schema.json", *[
        f"{dataset}/{name}" for dataset in DATASETS for name in AUTHOR_DATASET_FILES]]
    bindings = {}
    for relative in expected:
        path = root / relative
        value = _read_bytes(path)
        wanted = files.get(relative, files.get(str(path)))
        if not isinstance(wanted, str) or sha256_bytes(value) != wanted:
            raise ValueError("Missing/stale author manifest binding: " + relative)
        bindings[str(path)] = wanted
    if (bindings[str(root / "protocol.json")] != PROTOCOL_SHA256
            or bindings[str(root / "artifact_schema.json")] != SCHEMA_SHA256):
        raise ValueError("Pinned preregistered author protocol/schema changed")
    review_path = root / "independent_review_request.json"
    review_bytes = _read_bytes(review_path)
    review = json.loads(review_bytes)
    manifest_hash = sha256_bytes(manifest_bytes)
    if review.get("author_manifest_sha256", review.get("manifest_sha256")) != manifest_hash:
        raise ValueError("Author review dispatch does not bind frozen manifest")
    bindings[str(manifest_path)] = manifest_hash
    bindings[str(review_path)] = sha256_bytes(review_bytes)
    artifacts = {dataset: {name: read_json(root / dataset / name) for name in AUTHOR_DATASET_FILES}
                 for dataset in DATASETS}
    return artifacts, {"author_manifest_sha256": manifest_hash, "author_files": bindings,
                       "implementation_files": implementation_hashes(),
                       "author_agent_id": "/root/criteria_contrastive_review"}


def _parent_path(dataset):
    return PROJECT / "results/source_v2/predictions" / dataset / "zeroclean_loop.json"


def _load_parent(dataset, table):
    path = _parent_path(dataset)
    raw = _read_bytes(path)
    body = json.loads(raw)
    # Detect duplicate repairs before Prediction.load could silently collapse them.
    repairs = body.get("repairs", [])
    coordinates = [(x["row"], x["column"]) for x in repairs]
    if any(type(i) is not int or type(j) is not int for i, j in coordinates):
        raise ValueError("Parent repair coordinates must be exact integers")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Parent has duplicate repair coordinates")
    parent = Prediction(np.asarray(body["probabilities"]),
                        {(x["row"], x["column"]): x["value"] for x in repairs}, body.get("metadata", {})).validate(table.shape)
    if parent.metadata.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Parent does not bind exact dirty input")
    parent_rows(table, parent)
    return parent, raw


def _snapshot_once(path, raw):
    path = _no_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def prepare_cli(destination):
    artifacts, common = load_author_inputs()
    destination = _no_symlinks(destination).resolve()
    if destination.exists():
        raise ValueError("Refusing to overwrite existing preparation directory")
    prepared = []
    # Finish input validation before creating any output directory.
    for dataset in DATASETS:
        table = load_dirty(dataset)
        parent, parent_bytes = _load_parent(dataset, table)
        bindings = deepcopy(common)
        bindings["parent_file"] = {"path": str(_parent_path(dataset)), "sha256": sha256_bytes(parent_bytes)}
        source = artifacts[dataset]
        request = prepare_request(table, parent, source["pass4.json"], source["column_contracts.json"],
                                  source["proposals.json"], source["joint_conversion_candidates.json"], bindings=bindings)
        prepared.append((dataset, request, parent_bytes))
    check_file_hashes(common["author_files"])
    check_file_hashes(common["implementation_files"])
    check_file_hashes({request["bindings"]["parent_file"]["path"]: request["bindings"]["parent_file"]["sha256"]
                       for _, request, _ in prepared})
    destination.mkdir(parents=True)
    hashes, requests = {}, {}
    for dataset, request, parent_bytes in prepared:
        folder = destination / dataset
        write_once(folder / "verification_request.json", request)
        # Full parent metadata is provenance only: the independent request uses
        # values/probabilities without trace, rationale, or earlier judgments.
        _snapshot_once(folder / "provenance/parent_prediction.json", parent_bytes)
        for name in AUTHOR_DATASET_FILES:
            _snapshot_once(folder / "provenance/author" / name, _read_bytes(AUTHOR_ROOT / dataset / name))
        requests[dataset] = request["request_hash"]
    for relative in ("protocol.json", "artifact_schema.json", "manifest.json", "independent_review_request.json"):
        _snapshot_once(destination / "provenance/author" / relative, _read_bytes(AUTHOR_ROOT / relative))
    check_file_hashes(common["author_files"])
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(destination))] = sha256_bytes(_read_bytes(path))
    write_once(destination / "manifest.json", seal({"version": VERSION, "stage": "prepared",
               "requests": requests, "output_sha256": hashes, "bindings": common,
               "independent_review_status": "pending", "application_status": "not_run",
               "benchmark_data_accessed": False}, "manifest_hash"))
    return {"destination": str(destination), "datasets": list(DATASETS), "request_hashes": requests,
            "application_status": "not_run", "evaluation_status": "not_run"}


def validate_verifier_provenance(request, review_bytes, provenance, reviewer_id):
    if (not isinstance(reviewer_id, str) or not reviewer_id
            or reviewer_id in ("/root", request["bindings"].get("author_agent_id"))
            or provenance.get("reviewer_id") != reviewer_id or provenance.get("designated_by") != "/root"
            or provenance.get("reviewer_role") != "independent_format_contract_verifier"
            or provenance.get("request_hash") != request["request_hash"]
            or provenance.get("verification_file_sha256") != sha256_bytes(review_bytes)
            or provenance.get("model") != "gpt-6.1-sol" or provenance.get("reasoning_effort") != "xhigh"
            or provenance.get("benchmark_data_accessed") is not False):
        raise ValueError("Root-designated independent verifier provenance mismatch")


def apply_cli(prepared_directory, review_directory, reviewer_id, destination):
    prepared, review_root = _no_symlinks(prepared_directory).resolve(), _no_symlinks(review_directory).resolve()
    destination = _no_symlinks(destination).resolve()
    if destination.exists():
        raise ValueError("Refusing to overwrite existing overlay directory")
    prepared_manifest_bytes = _read_bytes(prepared / "manifest.json")
    manifest = json.loads(prepared_manifest_bytes)
    check_seal(manifest, "manifest_hash")
    if manifest.get("version") != VERSION or manifest.get("stage") != "prepared":
        raise ValueError("Wrong preparation manifest")
    prepared_hashes = {str(confined_path(prepared, name)): wanted for name, wanted in manifest["output_sha256"].items()}
    expected_paths = {f"{dataset}/verification_request.json" for dataset in DATASETS}
    expected_paths |= {f"{dataset}/provenance/parent_prediction.json" for dataset in DATASETS}
    expected_paths |= {f"{dataset}/provenance/author/{name}" for dataset in DATASETS for name in AUTHOR_DATASET_FILES}
    expected_paths |= {f"provenance/author/{name}" for name in ("protocol.json", "artifact_schema.json", "manifest.json", "independent_review_request.json")}
    if set(manifest["output_sha256"]) != expected_paths:
        raise ValueError("Preparation manifest must cover exactly the declared snapshots")
    check_file_hashes(prepared_hashes)
    artifacts, common = load_author_inputs()
    if common != manifest["bindings"]:
        raise ValueError("Author/implementation freeze differs from prepared snapshot")
    authoritative_snapshots = {}
    for dataset in DATASETS:
        for name in AUTHOR_DATASET_FILES:
            authoritative_snapshots[str(prepared / dataset / "provenance/author" / name)] = common["author_files"][str(AUTHOR_ROOT / dataset / name)]
    for name in ("protocol.json", "artifact_schema.json", "manifest.json", "independent_review_request.json"):
        authoritative_snapshots[str(prepared / "provenance/author" / name)] = common["author_files"][str(AUTHOR_ROOT / name)]
    verified_author_snapshots = {}
    for filename, wanted in authoritative_snapshots.items():
        raw = _read_bytes(filename)
        if sha256_bytes(raw) != wanted:
            raise ValueError("Prepared author snapshot differs from authoritative frozen source")
        verified_author_snapshots[filename] = raw
    results = []
    for dataset in DATASETS:
        table = load_dirty(dataset)
        parent, parent_bytes = _load_parent(dataset, table)
        request = read_json(prepared / dataset / "verification_request.json")
        parent_binding = request["bindings"]["parent_file"]
        if (request["request_hash"] != manifest["requests"][dataset]
                or parent_binding != {"path": str(_parent_path(dataset)), "sha256": sha256_bytes(parent_bytes)}):
            raise ValueError("Stale request or current parent file mismatch")
        if _read_bytes(prepared / dataset / "provenance/parent_prediction.json") != parent_bytes:
            raise ValueError("Prepared parent snapshot differs from authoritative current parent")
        expected_bindings = deepcopy(common)
        expected_bindings["parent_file"] = {"path": str(_parent_path(dataset)), "sha256": sha256_bytes(parent_bytes)}
        source = artifacts[dataset]
        validate_frozen_replay(table, parent, source["pass4.json"], source["column_contracts.json"],
                               source["proposals.json"], source["joint_conversion_candidates.json"], request,
                               expected_bindings=expected_bindings)
        review_path, provenance_path = review_root / dataset / "verification.json", review_root / dataset / "provenance.json"
        review_bytes, provenance_bytes = _read_bytes(review_path), _read_bytes(provenance_path)
        provenance = json.loads(provenance_bytes)
        validate_verifier_provenance(request, review_bytes, provenance, reviewer_id)
        result = apply_overlay(table, parent, request, json.loads(review_bytes))
        result.metadata["independent_verifier_provenance"] = provenance
        results.append((dataset, request, result, review_bytes, provenance_bytes, parent_bytes))
    check_file_hashes(common["author_files"])
    check_file_hashes(common["implementation_files"])
    check_file_hashes({request["bindings"]["parent_file"]["path"]: request["bindings"]["parent_file"]["sha256"]
                       for _, request, *_ in results})
    destination.mkdir(parents=True)
    for dataset, request, result, review_bytes, provenance_bytes, parent_bytes in results:
        folder = destination / dataset
        write_once(folder / "prediction.json", {**prediction_body(result), "metadata": result.metadata})
        write_once(folder / "verification_request.json", request)
        _snapshot_once(folder / "verification.json", review_bytes)
        _snapshot_once(folder / "verification_provenance.json", provenance_bytes)
        _snapshot_once(folder / "provenance/parent_prediction.json", parent_bytes)
        for name in AUTHOR_DATASET_FILES:
            _snapshot_once(folder / "provenance/author" / name, verified_author_snapshots[str(prepared / dataset / "provenance/author" / name)])
    for name in ("protocol.json", "artifact_schema.json", "manifest.json", "independent_review_request.json"):
        _snapshot_once(destination / "provenance/author" / name, verified_author_snapshots[str(prepared / "provenance/author" / name)])
    _snapshot_once(destination / "provenance/preparation_manifest.json", prepared_manifest_bytes)
    hashes = {str(path.relative_to(destination)): sha256_bytes(_read_bytes(path))
              for path in sorted(destination.rglob("*")) if path.is_file()}
    write_once(destination / "manifest.json", seal({"version": VERSION, "stage": "applied",
               "datasets": list(DATASETS), "output_sha256": hashes, "bindings": common,
               "prepared_manifest_sha256": sha256_bytes(prepared_manifest_bytes),
               "benchmark_data_accessed": False, "evaluation_status": "not_run"}, "manifest_hash"))
    return {"destination": str(destination), "datasets": list(DATASETS), "evaluation_status": "not_run"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="After author freeze, prepare six hash-bound independent requests only")
    prepare.add_argument("--output", type=Path, required=True)
    apply = sub.add_parser("apply", help="Explicit later application after full independent review; never evaluates")
    apply.add_argument("--prepared", type=Path, required=True)
    apply.add_argument("--reviews", type=Path, required=True)
    apply.add_argument("--reviewer-id", required=True)
    apply.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_cli(args.output) if args.command == "prepare" else apply_cli(args.prepared, args.reviews, args.reviewer_id, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
