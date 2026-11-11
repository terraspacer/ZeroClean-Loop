#!/usr/bin/env python3
"""Validate completed conservative-seed artifacts and export a typed interface.

No evaluator, inference, model, retrieval, or sandbox module is imported or run.
The builder is read-only until all 12 arms have three complete frozen rounds.
Export requires the exact completion-index byte SHA supplied by the root caller.
"""
from __future__ import annotations

import argparse
import ctypes
from copy import deepcopy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import tempfile

DATASETS = ("hospital", "flights", "beers", "rayyan", "billionaire", "movies")
ARMS = ("seeded_carry", "empty_carry")
PROTOCOL = "conservative-seed-v1"
EXPECTED_PROTOCOL_HASH = "2b243fcf5bb07899c17ea3c053f235371a869aad9ff786d93479483eaa7e890e"
INDEX_SCHEMA = "conservative-seed-completion-index-v1"
ADAPTER = "conservative-seed-evaluation-interface-v1"
AREA = "experimental/conservative_seed_export_v1"
EXPERIMENT = "experimental/conservative_seed_v1"
INDEX_PATH = f"{AREA}/completion_index.json"
LOCK_PATH = f"{EXPERIMENT}/protocol_lock.json"
PREPARATION_PATH = f"{EXPERIMENT}/preparation_index.json"
NOTE_PATH = f"{AREA}/evaluation_interface_note.md"
INTERFACE_FILES = ["scripts/export_conservative_seed.py", "tests/test_conservative_seed_export.py",
                   NOTE_PATH, f"{AREA}/completion_index.schema.json"]
SOURCE_NAMES = ("zeroed", "zeroed_zeroec_reconstruction")
MODEL, EFFORT = "gpt-6.1-sol", "xhigh"
HASH_RE = re.compile(r"^[a-f0-9]{64}$")
CELL_RE = re.compile(r"^(0|[1-9][0-9]*):(0|[1-9][0-9]*)$")


def digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_bytes(body):
    return (json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_pairs, parse_constant=nonfinite_json)


def unique_pairs(items):
    body = {}
    for key, value in items:
        if key in body:
            raise ValueError(f"Duplicate JSON key: {key}")
        body[key] = value
    return body


def nonfinite_json(value):
    raise ValueError(f"Nonfinite JSON: {value}")


def parse_json_text(value):
    return json.loads(value, object_pairs_hook=unique_pairs, parse_constant=nonfinite_json)


def fields(body, required, optional=()):
    if not isinstance(body, dict) or not set(required) <= set(body) or set(body) - set(required) - set(optional):
        raise ValueError(f"Unexpected object fields; required={sorted(required)}")


def check_hash(body, key):
    if body.get(key) != digest({name: value for name, value in body.items() if name != key}):
        raise ValueError(f"Invalid {key}")


def hash_string(value):
    if type(value) is not str or not HASH_RE.fullmatch(value):
        raise ValueError("Invalid SHA-256 string")
    return value


def score(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected finite score in [0,1]")
    return float(value)


def same(body, expected, label):
    if digest(body) != digest(expected):
        raise ValueError(f"{label} mismatch")


def cell_id(value, shape):
    if type(value) is not str or not CELL_RE.fullmatch(value):
        raise ValueError("Invalid cell ID")
    i, j = map(int, value.split(":"))
    if i >= shape[0] or j >= shape[1]:
        raise ValueError("Cell ID out of bounds")
    return i, j


def candidate_id(i, j, value):
    return digest({"row": i, "column_index": j, "value": value})


def source_paths(name):
    return [f"results/source_baselines_v1/frozen/{name}.json",
            f"results/source_baselines_v1/predictions/{name}/zeroed.json",
            f"results/source_baselines_v1/predictions/{name}/zeroed_zeroec_reconstruction.json",
            f"artifacts/llm/{name}/pass2.json", f"artifacts/llm/{name}/proposals_cts_v2.json",
            f"vendor/ZeroED/data/{name}_error-01.csv"]


LOCKED_SOURCES = [f"{EXPERIMENT}/{name}" for name in
                  ("runner.py", "protocol_draft.md", "schema_sketch.json", "tests_plan.md", "worker_contract.md", "test_runner.py")] + [
    f"zeroclean/{name}.py" for name in ("data", "types", "loop", "statistics", "sandbox")]


@dataclass(frozen=True)
class Authority:
    root: Path
    protocol_hash: str = EXPECTED_PROTOCOL_HASH

    def path(self, relative):
        if type(relative) is not str or not relative or "\\" in relative:
            raise ValueError("Invalid artifact path")
        part = Path(relative)
        if relative == "." or part.is_absolute() or any(item in (".", "..") for item in part.parts) or part.as_posix() != relative:
            raise ValueError("Artifact path escape or noncanonical path")
        root = self.root.resolve()
        path = root / part
        if path.resolve() != path:
            raise ValueError("Symlink/path escape is forbidden")
        for ancestor in (path, *path.parents):
            if ancestor == root:
                break
            if ancestor.is_symlink():
                raise ValueError("Symlink artifact is forbidden")
        return path

    def relative(self, absolute):
        if type(absolute) is not str:
            raise ValueError("Expected exact absolute artifact path")
        try:
            relative = Path(absolute).relative_to(self.root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("Absolute artifact path escape") from error
        if str(self.path(relative)) != absolute:
            raise ValueError("Noncanonical absolute artifact path")
        return relative


class Store:
    def __init__(self, authority, expected=None):
        self.authority = authority
        self.expected = expected
        self.files = {}

    def file(self, relative):
        path = self.authority.path(relative)
        if not path.is_file():
            raise ValueError(f"Missing complete artifact: {relative}")
        actual = sha256(path)
        if self.expected is not None and self.expected.get(relative) != actual:
            raise ValueError(f"Completion-index byte binding mismatch: {relative}")
        if relative in self.files and self.files[relative] != actual:
            raise ValueError("Input changed during validation")
        self.files[relative] = actual
        return path

    def read(self, relative):
        path = self.file(relative)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.files[relative]:
            raise ValueError("Input changed while reading bound JSON")
        return parse_json_text(data.decode("utf-8"))

    def recheck(self):
        for relative, expected in self.files.items():
            if sha256(self.authority.path(relative)) != expected:
                raise ValueError("Bound input changed during validation/publication")

    def absolute_map(self, relative_paths):
        return {str(self.authority.path(path)): self.files[path] for path in relative_paths}


def prediction(body, shape, mask_kind):
    fields(body, ("probabilities", "repairs", "metadata"))
    probabilities = body["probabilities"]
    if not isinstance(probabilities, list) or len(probabilities) != shape[0]:
        raise ValueError("Prediction row shape mismatch")
    for row in probabilities:
        if not isinstance(row, list) or len(row) != shape[1]:
            raise ValueError("Prediction column shape mismatch")
        for value in row:
            score(value)
    metadata = body["metadata"]
    if not isinstance(metadata, dict) or metadata.get("decision_threshold") != .5 or metadata.get("decision_threshold_operator") != ">":
        raise ValueError("Strict >0.5 declaration required")
    mask = metadata.get("upstream_binary_decisions")
    if not isinstance(mask, list) or len(mask) != shape[0]:
        raise ValueError("Mask row shape mismatch")
    for i, row in enumerate(mask):
        if not isinstance(row, list) or len(row) != shape[1]:
            raise ValueError("Mask column shape mismatch")
        for j, bit in enumerate(row):
            if mask_kind == "integer" and (type(bit) is not int or bit not in (0, 1)):
                raise ValueError("Raw experimental mask must contain integer 0/1 only")
            if mask_kind == "boolean" and type(bit) is not bool:
                raise ValueError("Source/export mask must be boolean")
            if bool(bit) != (probabilities[i][j] > .5):
                raise ValueError("Mask disagrees with strict probability >0.5")
    repairs = {}
    if not isinstance(body["repairs"], list):
        raise ValueError("Repairs must be a list")
    for item in body["repairs"]:
        fields(item, ("row", "column", "value"))
        i, j = item["row"], item["column"]
        if type(i) is not int or type(j) is not int or not 0 <= i < shape[0] or not 0 <= j < shape[1] or type(item["value"]) is not str or (i, j) in repairs:
            raise ValueError("Invalid or duplicate exact repair")
        if not bool(mask[i][j]):
            raise ValueError("Repair outside strict detection mask")
        repairs[i, j] = item["value"]
    return repairs


def float64_bytes(probabilities):
    return b"".join(struct.pack("<d", float(value)) for row in probabilities for value in row)


def normalize_prediction(raw, *, shape, raw_sha, snapshot_path, completion_sha):
    prediction(raw, shape, "integer")
    if "evaluation_interface_adapter" in raw["metadata"] or not isinstance(raw["metadata"].get("repair_statuses"), dict):
        raise ValueError("Expected unadapted raw repair-status metadata")
    normalized = deepcopy(raw)
    normalized["metadata"]["upstream_binary_decisions"] = [[bool(bit) for bit in row] for row in raw["metadata"]["upstream_binary_decisions"]]
    normalized["metadata"]["repair_statuses"] = {
        "representation": "bound_raw_snapshot_pointer", "snapshot_path": snapshot_path,
        "snapshot_sha256": raw_sha, "field_path": ["metadata", "repair_statuses"],
    }
    normalized["metadata"]["evaluation_interface_adapter"] = adapter_provenance(raw_sha, snapshot_path, completion_sha)
    reconcile_normalization(raw, normalized, shape, raw_sha, snapshot_path, completion_sha)
    return normalized


def adapter_provenance(raw_sha, snapshot_path, completion_sha):
    hash_string(raw_sha)
    hash_string(completion_sha)
    return {
        "version": ADAPTER, "raw_prediction_sha256": raw_sha, "raw_prediction_snapshot": snapshot_path,
        "completion_index_sha256": completion_sha,
        "transformations": ["metadata.upstream_binary_decisions: integer 0/1 to boolean",
                            "metadata.repair_statuses: full field to exact raw-snapshot pointer"],
        "probabilities_repairs_and_strict_logical_mask_unchanged": True,
        "selection_or_evaluation_performed": False,
    }


def reconcile_normalization(raw, normalized, shape, raw_sha, snapshot_path, completion_sha):
    prediction(raw, shape, "integer")
    prediction(normalized, shape, "boolean")
    if float64_bytes(raw["probabilities"]) != float64_bytes(normalized["probabilities"]) or raw["repairs"] != normalized["repairs"]:
        raise ValueError("Normalization changed float64 probabilities or exact repair list")
    restored = deepcopy(normalized)
    pointer = restored["metadata"].get("repair_statuses")
    expected_pointer = {"representation": "bound_raw_snapshot_pointer", "snapshot_path": snapshot_path,
                        "snapshot_sha256": raw_sha, "field_path": ["metadata", "repair_statuses"]}
    same(pointer, expected_pointer, "Trace pointer")
    adapter = restored["metadata"].pop("evaluation_interface_adapter", None)
    same(adapter, adapter_provenance(raw_sha, snapshot_path, completion_sha), "Adapter provenance")
    restored["metadata"]["repair_statuses"] = deepcopy(raw["metadata"]["repair_statuses"])
    restored["metadata"]["upstream_binary_decisions"] = [[int(bit) for bit in row] for row in restored["metadata"]["upstream_binary_decisions"]]
    same(restored, raw, "Only the two declared metadata fields may change")


def assessment(item, original=False):
    fields(item, ("disposition", "confidence", "exact_value_score", "explanation"))
    if item["disposition"] not in ("supported", "rejected", "unknown") or type(item["explanation"]) is not str:
        raise ValueError("Invalid recorded assessment")
    confidence = score(item["confidence"])
    value = item["exact_value_score"]
    if value is not None:
        score(value)
    if item["disposition"] != "unknown":
        if value is None or (item["disposition"] == "supported" and value < (.9 if original else .8)) or (item["disposition"] == "rejected" and value > (.1 if original else .2)):
            raise ValueError("Disposition/score inconsistency")
    return item["disposition"] != "unknown" and confidence >= (.9 if original else .8)


def expand_response(request, response, batch_hashes):
    required = ("protocol", "dataset", "arm", "external_round", "request_hash", "dirty_sha256", "candidate_universe_sha256", "model", "reasoning_effort", "fresh_session_id", "fork_turns", "reviewed_batch_hashes", "joint_plans")
    fields(response, required, ("cells", "groups"))
    for key in required[:9]:
        if response[key] != request[key] or (key == "external_round" and type(response[key]) is not int):
            raise ValueError("Recorded verifier response binding mismatch")
    if response["fork_turns"] != "none" or type(response["fresh_session_id"]) is not str or not response["fresh_session_id"] or response["reviewed_batch_hashes"] != batch_hashes:
        raise ValueError("Fresh worker/batch coverage mismatch")
    if ("cells" in response) == ("groups" in response):
        raise ValueError("Exactly one recorded response format is required")
    source = {cell["cell_id"]: cell for cell in request["cells"]}
    replies = response.get("cells")
    if "groups" in response:
        if not isinstance(response["groups"], list):
            raise ValueError("Invalid recorded groups")
        replies = []
        for group in response["groups"]:
            fields(group, ("members", "original_assessment", "selection", "candidate_assessments", "default_candidate_assessment", "joint_plan_id", "applicability_explanation"))
            if not isinstance(group["members"], list) or not group["members"] or not isinstance(group["candidate_assessments"], dict) or type(group["applicability_explanation"]) is not str or not group["applicability_explanation"]:
                raise ValueError("Invalid honest group coverage")
            assessment(group["original_assessment"], True)
            assessment(group["default_candidate_assessment"])
            used = set()
            for key in group["members"]:
                if type(key) is not str or key not in source:
                    raise ValueError("Unknown grouped member")
                cell = source[key]
                selection = deepcopy(group["selection"])
                if selection.get("kind") == "candidate":
                    selected = [candidate for candidate in cell["candidates"] if candidate["value"] == selection.get("value")]
                    if len(selected) != 1:
                        raise ValueError("Grouped selection not in exact alternatives")
                    selection["candidate_id"] = selected[0]["candidate_id"]
                candidates = []
                for candidate in cell["candidates"]:
                    value = candidate["value"]
                    if value in group["candidate_assessments"]:
                        used.add(value)
                    candidates.append({"candidate_id": candidate["candidate_id"], "value": value,
                                       "assessment": deepcopy(group["candidate_assessments"].get(value, group["default_candidate_assessment"]))})
                replies.append({"cell_id": key, "original": cell["original"], "original_assessment": deepcopy(group["original_assessment"]),
                                "selection": selection, "candidates": candidates, "joint_plan_id": group["joint_plan_id"]})
            if used != set(group["candidate_assessments"]):
                raise ValueError("Unrequested grouped candidate claim")
    if not isinstance(replies, list):
        raise ValueError("Missing recorded cells")
    verified = {}
    for item in replies:
        fields(item, ("cell_id", "original", "original_assessment", "selection", "candidates", "joint_plan_id"))
        key = item["cell_id"]
        if type(key) is not str or key not in source or key in verified or item["original"] != source[key]["original"]:
            raise ValueError("Unknown/duplicate/stale response cell")
        assessment(item["original_assessment"], True)
        allowed = {candidate["candidate_id"]: candidate["value"] for candidate in source[key]["candidates"]}
        candidates = {}
        if not isinstance(item["candidates"], list):
            raise ValueError("Missing candidate judgments")
        for candidate in item["candidates"]:
            fields(candidate, ("candidate_id", "value", "assessment"))
            cid = candidate["candidate_id"]
            if type(cid) is not str or cid not in allowed or cid in candidates or candidate["value"] != allowed[cid]:
                raise ValueError("Candidate judgment mismatch")
            assessment(candidate["assessment"])
            candidates[cid] = candidate["assessment"]
        if set(candidates) != set(allowed):
            raise ValueError("Incomplete candidate judgments")
        selection = item["selection"]
        kind = selection.get("kind") if isinstance(selection, dict) else None
        fields(selection, ("kind", "value", "confidence", "candidate_id") if kind == "candidate" else ("kind", "value", "confidence"))
        score(selection["confidence"])
        if kind == "candidate":
            cid = selection["candidate_id"]
            if type(cid) is not str or cid not in allowed or selection["value"] != allowed[cid] or candidates[cid]["disposition"] != "supported" or item["original_assessment"]["disposition"] == "supported":
                raise ValueError("Unjustified or contradictory recorded selection")
        elif kind == "original":
            if selection["value"] != item["original"] or item["original_assessment"]["disposition"] != "supported" or not assessment(item["original_assessment"], True):
                raise ValueError("Unjustified exact-original selection")
        elif kind != "abstain" or selection["value"] is not None:
            raise ValueError("Abstention must be explicit null")
        verified[key] = item
    if set(verified) != set(source):
        raise ValueError("Incomplete recorded verifier coverage")
    joints = {}
    if not isinstance(response["joint_plans"], list):
        raise ValueError("Joint plans must be a list")
    for joint in response["joint_plans"]:
        fields(joint, ("plan_id", "row", "members", "confidence", "disposition", "explanation"))
        jid = joint["plan_id"]
        if type(jid) is not str or not jid or jid in joints or type(joint["row"]) is not int or not isinstance(joint["members"], list) or not joint["members"] or joint["disposition"] not in ("supported", "rejected", "unknown") or type(joint["explanation"]) is not str:
            raise ValueError("Malformed recorded joint plan")
        score(joint["confidence"])
        members = set()
        for member in joint["members"]:
            fields(member, ("cell_id", "candidate_id", "value"))
            key = member["cell_id"]
            if type(key) is not str or key not in source or key in members or source[key]["row"] != joint["row"] or verified[key]["joint_plan_id"] != jid:
                raise ValueError("Joint membership mismatch")
            members.add(key)
            allowed = {candidate["candidate_id"]: candidate["value"] for candidate in source[key]["candidates"]}
            if member["candidate_id"] not in allowed or member["value"] != allowed[member["candidate_id"]]:
                raise ValueError("Joint alternative mismatch")
        joints[jid] = joint
    for key, item in verified.items():
        jid = item["joint_plan_id"]
        if jid is not None and (type(jid) is not str or jid not in joints or key not in {member["cell_id"] for member in joints[jid]["members"]}):
            raise ValueError("Unbound joint reference")
    return verified, joints


def effect_record(effect, rule_count):
    required = ("delta_features", "introduced_violation", "runtime_error", "before_rule_results", "after_rule_results", "raw_value_count", "original_raw_value_count", "raw_conditional_evidence_before", "raw_conditional_evidence_after")
    fields(effect, required)
    if not isinstance(effect["delta_features"], list) or len(effect["delta_features"]) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) or not -1 <= x <= 1 for x in effect["delta_features"]):
        raise ValueError("Malformed captured counterfactual delta")
    before, after = effect["before_rule_results"], effect["after_rule_results"]
    for values in (before, after):
        if not isinstance(values, list) or len(values) != rule_count or any(type(value) is not int or value not in (-1, 0, 1) for value in values):
            raise ValueError("Malformed captured rule results")
    if type(effect["introduced_violation"]) is not bool or effect["introduced_violation"] != any(a == 0 and b == 1 for a, b in zip(before, after)) or type(effect["runtime_error"]) is not bool or effect["runtime_error"] != (-1 in before or -1 in after):
        raise ValueError("Captured guard flags disagree with captured rule results")
    for key in ("raw_value_count", "original_raw_value_count"):
        if type(effect[key]) is not int or effect[key] < 0:
            raise ValueError("Invalid captured raw count")
    for key in ("raw_conditional_evidence_before", "raw_conditional_evidence_after"):
        if not isinstance(effect[key], list):
            raise ValueError("Invalid raw evidence list")


class Validator:
    def __init__(self, authority, expected_files=None):
        self.authority = authority
        self.store = Store(authority, expected_files)
        self.sessions = set()
        self.entries = []
        self.exports = []
        self.rule_evidence = {}
        self.effect_evidence = {}

    def captured_rules(self, name, row, values, results):
        """Compare recorded evaluations of identical tuples, without execution."""
        key = name, row, digest(values)
        recorded = tuple(results)
        if key in self.rule_evidence and self.rule_evidence[key] != recorded:
            raise ValueError("Captured rule results disagree for an identical row tuple")
        self.rule_evidence[key] = recorded

    def captured_effect(self, name, cell, candidate, effect, columns, before):
        self.captured_rules(name, cell["row"], [before[column] for column in columns], effect["before_rule_results"])
        after = dict(before)
        after[cell["column"]] = candidate["value"]
        self.captured_rules(name, cell["row"], [after[column] for column in columns], effect["after_rule_results"])
        key = name, cell["row"], digest([cell["column_index"], candidate["value"], [before[column] for column in columns]])
        recorded = digest(effect)
        if key in self.effect_evidence and self.effect_evidence[key] != recorded:
            raise ValueError("Captured identical counterfactual evidence mismatch")
        self.effect_evidence[key] = recorded

    def lock(self):
        lock = self.store.read(LOCK_PATH)
        check_hash(lock, "protocol_hash")
        if lock.get("protocol_hash") != self.authority.protocol_hash or lock.get("protocol") != PROTOCOL or lock.get("ground_truth_access") is not False or lock.get("status") != "frozen_before_new_predictions":
            raise ValueError("Fixed protocol authority mismatch")
        expected_sources = {str(self.authority.path(path)) for path in LOCKED_SOURCES}
        expected_inputs = {str(self.authority.path(path)) for name in DATASETS for path in source_paths(name)}
        if set(lock.get("source_files", {})) != expected_sources or set(lock.get("coordinated_input_files", {})) != expected_inputs:
            raise ValueError("Protocol source/input coverage mismatch")
        for mapping in (lock["source_files"], lock["coordinated_input_files"]):
            for absolute, expected in mapping.items():
                relative = self.authority.relative(absolute)
                self.store.file(relative)
                if self.store.files[relative] != hash_string(expected):
                    raise ValueError("Pinned live source/input bytes changed")
        # These sources belong to the new interface, not the old inference lock.
        # The root-supplied completion-index SHA binds their actual bytes.
        for relative in INTERFACE_FILES:
            self.store.file(relative)
        config = lock["config"]
        if config.get("protocol") != PROTOCOL or config.get("external_rounds") != 3 or config.get("candidate_cap") != 3 or config.get("detector_operator") != ">" or config.get("detector_threshold") != .5 or config.get("model") != MODEL or config.get("reasoning_effort") != EFFORT or config.get("reretrieval") is not False:
            raise ValueError("Unexpected paired protocol parameters")
        return lock

    def dirty(self, name):
        path = self.store.file(source_paths(name)[5])
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows or not rows[0] or len(set(rows[0])) != len(rows[0]) or any(len(row) != len(rows[0]) for row in rows[1:]):
            raise ValueError("Invalid original dirty CSV schema")
        return rows[0], rows[1:]

    def source_pair(self, name, shape, dirty_sha):
        frozen_path, ed_path, ec_path, _, _, _ = source_paths(name)
        frozen = self.store.read(frozen_path)
        check_hash(frozen, "export_hash")
        if frozen.get("dataset") != name or frozen.get("dirty_sha256") != dirty_sha or frozen.get("frozen_before_evaluation") is not True or frozen.get("ground_truth_access") is not False or frozen.get("export_schema") != "source-baselines-only-v1" or frozen.get("official_exact_ec_baseline_present") is not False:
            raise ValueError("Source baseline freeze mismatch")
        ed, ec = self.store.read(ed_path), self.store.read(ec_path)
        if frozen.get("prediction_files") != {"zeroed.json": self.store.files[ed_path], "zeroed_zeroec_reconstruction.json": self.store.files[ec_path]}:
            raise ValueError("Exact original baseline byte binding mismatch")
        for system, body in (("zeroed_upstream", ed), ("zeroed_zeroec_reconstruction", ec)):
            prediction(body, shape, "boolean")
            metadata = body["metadata"]
            if metadata.get("baseline") != system or metadata.get("uses_ground_truth") is not False or metadata.get("strict_zero_shot") is not True or metadata.get("input_fingerprint", metadata.get("dirty_fingerprint")) != dirty_sha:
                raise ValueError("Original baseline provenance mismatch")
        if ed["repairs"] or ed["probabilities"] != ec["probabilities"] or ed["metadata"]["upstream_binary_decisions"] != ec["metadata"]["upstream_binary_decisions"]:
            raise ValueError("ED→EC must retain original source detection")
        if ed["metadata"].get("fidelity") != "upstream_function_extraction" or ed["metadata"].get("reproduced_upstream") is not True:
            raise ValueError("Exact original-source ED fidelity is required")
        if ec["metadata"].get("fidelity") != "author_source_reconstruction" or ec["metadata"].get("reproduced_upstream") is not False or ec["metadata"].get("pending_cells") != 0 or ec["metadata"].get("completed_cells") != ec["metadata"].get("detected_cells"):
            raise ValueError("Qualified EC source baseline is incomplete or mislabeled")
        return ed, ec

    def plan(self, name, base, lock, columns, rows, ed, ec):
        plan_path, manifest_path = f"{base}/plan.json", f"{base}/manifest.json"
        plan, manifest = self.store.read(plan_path), self.store.read(manifest_path)
        fields(plan, ("protocol", "dataset", "dirty_sha256", "columns", "shape", "raw_probabilities", "original_source_binary_decisions", "cells", "candidate_universe_sha256", "rules", "rule_validation", "dependencies", "raw_statistics_sha256", "config", "source_fidelity", "identity_source_repairs", "ground_truth_access", "plan_hash"))
        check_hash(plan, "plan_hash")
        check_hash(manifest, "manifest_hash")
        fields(manifest, ("protocol", "dataset", "plan_hash", "protocol_hash", "protocol_lock_path", "input_files", "plan_sha256", "arms", "ground_truth_access", "manifest_hash"))
        if base != f"{EXPERIMENT}/runs/{name}/{plan['plan_hash']}" or manifest["plan_hash"] != plan["plan_hash"] or manifest["plan_sha256"] != self.store.files[plan_path] or manifest["protocol"] != PROTOCOL or manifest["dataset"] != name or manifest["protocol_hash"] != lock["protocol_hash"] or manifest["protocol_lock_path"] != str(self.authority.path(LOCK_PATH)) or manifest["arms"] != list(ARMS) or manifest["ground_truth_access"] is not False:
            raise ValueError("Shared plan/manifest identity mismatch")
        expected_inputs = self.store.absolute_map(source_paths(name) + [LOCK_PATH])
        same(manifest["input_files"], expected_inputs, "Exact per-dataset input map")
        shape = (len(rows), len(columns))
        if plan.get("protocol") != PROTOCOL or plan.get("dataset") != name or plan.get("dirty_sha256") != self.store.files[source_paths(name)[5]] or plan.get("columns") != columns or plan.get("shape") != list(shape) or plan.get("config") != lock["config"] or plan.get("ground_truth_access") is not False:
            raise ValueError("Plan/config/original table mismatch")
        same(plan.get("raw_probabilities"), ed["probabilities"], "Plan original probabilities")
        bits = [[int(bit) for bit in row] for row in ed["metadata"]["upstream_binary_decisions"]]
        same(plan.get("original_source_binary_decisions"), bits, "Original upstream decisions")
        hash_string(plan.get("raw_statistics_sha256"))
        pass2 = self.store.read(source_paths(name)[3])
        proposals = self.store.read(source_paths(name)[4])
        if pass2.get("dirty_sha256") != plan["dirty_sha256"] or proposals.get("dirty_sha256") != plan["dirty_sha256"] or proposals.get("dataset") != name or proposals.get("retrieval_method") != "zeroec_source_cts":
            raise ValueError("Fixed source artifact provenance mismatch")
        for artifact in (pass2, proposals):
            if artifact.get("model") != MODEL or artifact.get("reasoning_effort") != EFFORT:
                raise ValueError("Fixed source model provenance mismatch")
        same(plan["dependencies"], pass2.get("dependencies", []), "Fixed dependency declarations")
        same(plan["source_fidelity"], {"ed": ed["metadata"].get("fidelity"), "ec": ec["metadata"].get("fidelity"),
             "ec_prompt_provenance": ec["metadata"].get("prompt_provenance"), "null_sentinel_policy": ec["metadata"].get("null_sentinel_policy")}, "Unchanged source fidelity")
        if self.store.files[source_paths(name)[4]] != lock["config"].get("source_cts_proposal_byte_hashes", {}).get(name):
            raise ValueError("Prescribed source-v2 CTS bytes changed")
        allowed_rules = pass2.get("rules", [])
        cursor = 0
        for rule in plan["rules"]:
            while cursor < len(allowed_rules) and allowed_rules[cursor] != rule:
                cursor += 1
            if cursor == len(allowed_rules) or rule.get("target") not in columns:
                raise ValueError("Accepted rules are not a source-ordered subset of fixed pass2")
            cursor += 1
        supplied = {}
        for cell in proposals["cells"]:
            i, column = cell.get("row"), cell.get("column")
            if type(i) is not int or not 0 <= i < shape[0] or (type(column) is not str and type(column) is not int):
                raise ValueError("Invalid original CTS coordinate")
            j = columns.index(column) if type(column) is str and column in columns else column
            if type(j) is not int or not 0 <= j < shape[1] or f"{i}:{j}" in supplied:
                raise ValueError("Duplicate/unknown original CTS coordinate")
            supplied[f"{i}:{j}"] = cell
        seeds = {f"{r['row']}:{r['column']}": r["value"] for r in ec["repairs"]}
        identities = [{"row": cell_id(key, shape)[0], "column_index": cell_id(key, shape)[1], "value": value}
                      for key, value in sorted(seeds.items(), key=lambda item: cell_id(item[0], shape))
                      if value == rows[cell_id(key, shape)[0]][cell_id(key, shape)[1]]]
        same(plan["identity_source_repairs"], identities, "Source identity-repair audit")
        required = {f"{i}:{j}" for i, row in enumerate(ed["probabilities"]) for j, value in enumerate(row) if value > .5 or .25 <= value <= .85}
        required.update(supplied)
        required.update(key for key, value in seeds.items() if value != rows[cell_id(key, shape)[0]][cell_id(key, shape)[1]])
        for j, column in enumerate(columns):
            mapping = pass2.get("value_repairs", {}).get(column, {})
            required.update(f"{i}:{j}" for i, row in enumerate(rows) if row[j] in mapping)
        cells, ordered = {}, []
        for cell in plan["cells"]:
            fields(cell, ("cell_id", "row", "column_index", "column", "original", "tuple", "retrieved_dirty_peers", "raw_conditional_profiles", "candidates", "private_seed_candidate_id"))
            key = cell["cell_id"]
            i, j = cell_id(key, shape)
            if key in cells or cell.get("row") != i or type(cell.get("row")) is not int or cell.get("column_index") != j or type(cell.get("column_index")) is not int or cell.get("column") != columns[j] or cell.get("original") != rows[i][j] or cell.get("tuple") != dict(zip(columns, rows[i])):
                raise ValueError("Plan original cell alignment mismatch")
            values = []
            provenance = {}
            seed = seeds.get(key)
            if seed is not None and seed != rows[i][j]:
                values.append(seed)
                provenance[seed] = {"private_origin": "naive_source_ec", "private_origins": ["naive_source_ec"],
                                    "evidence_explanation": "Untrusted proposed alternative; assess independently"}
            for candidate in supplied.get(key, {}).get("candidates", []):
                if type(candidate.get("value")) is not str:
                    raise ValueError("Invalid source CTS string")
                if candidate["value"] != rows[i][j] and candidate["value"] not in values and len(values) < 3:
                    values.append(candidate["value"])
                    provenance[candidate["value"]] = {"private_origin": "fixed_source_v2_cts", "private_origins": ["fixed_source_v2_cts"],
                                                        "evidence_explanation": candidate["explanation"]}
                elif candidate["value"] in provenance:
                    record = provenance[candidate["value"]]
                    if "fixed_source_v2_cts" not in record["private_origins"]:
                        record["private_origins"].append("fixed_source_v2_cts")
                    record["evidence_explanation"] = candidate["explanation"]
            if [candidate["value"] for candidate in cell["candidates"]] != values or len(values) > 3:
                raise ValueError("Hard-K3 prior+fixed-CTS candidate universe mismatch")
            for candidate in cell["candidates"]:
                fields(candidate, ("candidate_id", "value", "private_origin", "private_origins", "evidence_explanation", "raw_effect"))
                if candidate["candidate_id"] != candidate_id(i, j, candidate["value"]):
                    raise ValueError("Exact candidate ID/value mismatch")
                same({key: candidate[key] for key in provenance[candidate["value"]]}, provenance[candidate["value"]], "Exact source candidate provenance/evidence")
                effect_record(candidate["raw_effect"], len(plan["rules"]))
                self.captured_effect(name, cell, candidate, candidate["raw_effect"], columns, cell["tuple"])
            expected_seed = candidate_id(i, j, seed) if seed is not None and seed != rows[i][j] else None
            if cell.get("private_seed_candidate_id") != expected_seed:
                raise ValueError("Naive seed identity mismatch")
            peers = supplied.get(key, {}).get("retrieved_row_indices", [])
            if any(type(r) is not int or not 0 <= r < shape[0] for r in peers):
                raise ValueError("Raw CTS peer index mismatch")
            same(cell.get("retrieved_dirty_peers"), [{"row": r, "tuple": dict(zip(columns, rows[r])), "trust": "raw_dirty_observation_not_clean_reference"} for r in peers], "Raw peer tuples")
            same(cell.get("raw_conditional_profiles"), supplied.get(key, {}).get("conditional_profiles", []), "Raw fixed profiles")
            cells[key] = cell
            ordered.append((i, j))
        if ordered != sorted(ordered) or not required <= set(cells):
            raise ValueError("Candidate-cell order/complete source eligibility mismatch")
        # Extra eligibility can arise from executable criteria at preparation.
        # Its original root-sealed request bytes are checked below; no predicate
        # is rerun or replaced with a guessed rule outcome by this exporter.
        if any(not any(rule.get("target") == cells[key]["column"] for rule in plan["rules"]) for key in set(cells) - required):
            raise ValueError("Unexplained extra candidate eligibility")
        universe = [{key: cell[key] for key in ("cell_id", "row", "column_index", "original", "candidates")} for cell in plan["cells"]]
        if plan["candidate_universe_sha256"] != digest(universe):
            raise ValueError("Candidate universe content hash mismatch")
        return plan, manifest, cells

    def state(self, state, plan, cells, name, arm, number, shape, rows):
        fields(state, ("protocol", "dataset", "arm", "external_round", "plan_hash", "probabilities", "active", "cells", "events", "guard_records", "state_hash"))
        check_hash(state, "state_hash")
        if state["protocol"] != PROTOCOL or state["dataset"] != name or state["arm"] != arm or type(state["external_round"]) is not int or state["external_round"] != number or state["plan_hash"] != plan["plan_hash"] or not isinstance(state["active"], dict) or not isinstance(state["cells"], dict) or set(state["cells"]) != set(cells) or not set(state["active"]) <= set(cells) or not isinstance(state["events"], list) or not isinstance(state["guard_records"], list):
            raise ValueError("State identity/coverage mismatch")
        temporary = {"probabilities": state["probabilities"], "repairs": [], "metadata": {"decision_threshold": .5, "decision_threshold_operator": ">", "upstream_binary_decisions": [[int(value > .5) for value in row] for row in state["probabilities"]]}}
        prediction(temporary, shape, "integer")
        for key, cell in cells.items():
            i, j = cell["row"], cell["column_index"]
            status = state["cells"][key]
            fields(status, ("detector_bit", "latest_decisive_original_request", "emission_status", "support_status"))
            if type(status["detector_bit"]) is not bool or status["detector_bit"] != (state["probabilities"][i][j] > .5):
                raise ValueError("State detector bit/score mismatch")
            active = state["active"].get(key)
            allowed = {candidate["candidate_id"]: candidate["value"] for candidate in cell["candidates"]}
            if active is not None and (type(active) is not str or active not in allowed):
                raise ValueError("Unknown active hypothesis")
            expected = "emitted" if active and status["detector_bit"] else "held_detector_negative" if active else None
            if (expected and status["emission_status"] != expected) or (not active and status["emission_status"] not in ("inactive", "guard_blocked")) or status["support_status"] not in ("carried_unresolved", "freshly_supported", "not_active") or (not active and status["support_status"] != "not_active"):
                raise ValueError("State lifecycle/emission mismatch")
        for guard in state["guard_records"]:
            fields(guard, ("row", "round", "reason", "raw_rule_results", "proposed_rule_results", "blocked", "vacuous", "proposed_values"))
            if type(guard["row"]) is not int or not 0 <= guard["row"] < shape[0] or type(guard["round"]) is not int or not 0 <= guard["round"] <= number:
                raise ValueError("Invalid recorded row guard identity")
            before, after = guard["raw_rule_results"], guard["proposed_rule_results"]
            for values in (before, after):
                if not isinstance(values, list) or len(values) != len(plan["rules"]) or any(type(value) is not int or value not in (-1, 0, 1) for value in values):
                    raise ValueError("Invalid recorded guard matrix")
            blocked = any(a == 0 and b == 1 for a, b in zip(before, after)) or -1 in before or -1 in after
            if type(guard["blocked"]) is not bool or guard["blocked"] != blocked or type(guard["vacuous"]) is not bool or guard["vacuous"] != (not plan["rules"]):
                raise ValueError("Recorded complete-row guard result mismatch")
            if not isinstance(guard["proposed_values"], dict):
                raise ValueError("Malformed complete-row proposal")
            for key, value in guard["proposed_values"].items():
                if key not in cells or cells[key]["row"] != guard["row"] or value not in {candidate["value"] for candidate in cells[key]["candidates"]}:
                    raise ValueError("Unbound guarded proposal")
            proposed = rows[guard["row"]].copy()
            for key, value in guard["proposed_values"].items():
                proposed[cells[key]["column_index"]] = value
            self.captured_rules(name, guard["row"], rows[guard["row"]], before)
            self.captured_rules(name, guard["row"], proposed, after)
        if number == 0:
            same(state["probabilities"], plan["raw_probabilities"], "Initial source probabilities")
            if any(status["latest_decisive_original_request"] is not None for status in state["cells"].values()):
                raise ValueError("Initial state contains a fabricated verifier assessment")
            if arm == "empty_carry" and state["active"]:
                raise ValueError("Empty arm was seeded")
            for key, cid in state["active"].items():
                if arm != "seeded_carry" or cid != cells[key]["private_seed_candidate_id"]:
                    raise ValueError("Initial proposal did not come from naive seed")
            if arm == "empty_carry" and state["guard_records"]:
                raise ValueError("Empty arm has unexplained initial row guards")
            if arm == "seeded_carry":
                seeds = {key: cell["private_seed_candidate_id"] for key, cell in cells.items() if cell["private_seed_candidate_id"] is not None}
                guards = {guard["row"]: guard for guard in state["guard_records"]}
                expected_values = {}
                for key, cid in seeds.items():
                    cell = cells[key]
                    if state["cells"][key]["detector_bit"]:
                        value = next(c["value"] for c in cell["candidates"] if c["candidate_id"] == cid)
                        expected_values.setdefault(cell["row"], {})[key] = value
                if len(guards) != len(state["guard_records"]) or set(guards) != set(expected_values):
                    raise ValueError("Initial complete-row guard coverage mismatch")
                for row, guard in guards.items():
                    if guard["round"] != 0 or guard["reason"] != "seed_initialization" or guard["proposed_values"] != expected_values[row]:
                        raise ValueError("Initial seed row proposal/guard mismatch")
                    raw_results = next(c["raw_effect"]["before_rule_results"] for cell in cells.values() if cell["row"] == row for c in cell["candidates"])
                    if guard["raw_rule_results"] != raw_results:
                        raise ValueError("Initial guard changed captured original rule results")
                expected_active = {key: cid for key, cid in seeds.items() if not guards.get(cells[key]["row"], {}).get("blocked", False)}
                same(state["active"], expected_active, "Initial seed carry/whole-row rollback")
            expected_events = [dict(repair, round=0, event="identity_source_repair_ignored") for repair in plan["identity_source_repairs"]]
            if arm == "seeded_carry":
                expected_events.extend({"round": 0, "cell_id": key, "event": "noisy_seed_initialized", "candidate_id": cell["private_seed_candidate_id"]} for key, cell in cells.items() if cell["private_seed_candidate_id"] is not None)
                expected_events.extend({"round": 0, "cell_id": key, "event": "whole_row_guard_blocked", "candidate_id": cid} for row in sorted(guards) if guards[row]["blocked"] for key, cid in seeds.items() if cells[key]["row"] == row)
            same(state["events"], expected_events, "Initial hypothesis ledger")
        return state

    def request(self, request, plan, cells, previous, name, arm, number, columns, rows):
        fields(request, ("protocol", "dataset", "arm", "external_round", "model", "reasoning_effort", "dirty_sha256", "plan_hash", "candidate_universe_sha256", "raw_statistics_sha256", "parent_state_hash", "cells", "rules", "config", "delta_dimensions", "batch_cells", "worker_policy", "request_hash"))
        check_hash(request, "request_hash")
        for key, expected in {"protocol": PROTOCOL, "dataset": name, "arm": arm, "external_round": number,
                              "model": MODEL, "reasoning_effort": EFFORT, "dirty_sha256": plan["dirty_sha256"],
                              "plan_hash": plan["plan_hash"], "candidate_universe_sha256": plan["candidate_universe_sha256"],
                              "raw_statistics_sha256": plan["raw_statistics_sha256"], "parent_state_hash": previous["state_hash"],
                              "rules": plan["rules"], "config": plan["config"], "worker_policy": plan["config"]["worker_policy"],
                              "delta_dimensions": ["rule_satisfaction", "fd_support", "nmi_vicinity", "domain_support"]}.items():
            same(request.get(key), expected, f"Recorded request {key}")
        if type(request.get("external_round")) is not int or not isinstance(request.get("cells"), list) or [cell.get("cell_id") for cell in request["cells"]] != list(cells):
            raise ValueError("Request exact cell coverage mismatch")
        proposed = [row.copy() for row in rows]
        for key, cid in previous["active"].items():
            if previous["cells"][key]["detector_bit"]:
                cell = cells[key]
                proposed[cell["row"]][cell["column_index"]] = next(candidate["value"] for candidate in cell["candidates"] if candidate["candidate_id"] == cid)
        by_row = {}
        for current in request["cells"]:
            fields(current, ("cell_id", "row", "column_index", "column", "original", "tuple", "retrieved_dirty_peers", "raw_conditional_profiles", "tentative_tuple", "counterfactual_before_tuple", "active_candidate_id", "tentative_context_trust", "candidates"))
            key = current["cell_id"]
            cell = cells[key]
            i, j = cell["row"], cell["column_index"]
            expected = {field: cell[field] for field in ("cell_id", "row", "column_index", "column", "original", "tuple", "retrieved_dirty_peers", "raw_conditional_profiles")}
            before = proposed[i].copy()
            before[j] = rows[i][j]
            expected.update(tentative_tuple=dict(zip(columns, proposed[i])), counterfactual_before_tuple=dict(zip(columns, before)),
                            active_candidate_id=previous["active"].get(key), tentative_context_trust="untrusted_model_proposals")
            for field, value in expected.items():
                same(current.get(field), value, f"Request original/tentative field {field}")
            ranked = list(cell["candidates"])
            random.Random(digest([42, name, key, number])).shuffle(ranked)
            if not isinstance(current.get("candidates"), list) or len(current["candidates"]) != len(ranked):
                raise ValueError("Request candidate coverage mismatch")
            for actual, source in zip(current["candidates"], ranked):
                fields(actual, ("candidate_id", "value", "evidence_explanation", "raw_effect", "effect"))
                for field in ("candidate_id", "value", "evidence_explanation", "raw_effect"):
                    same(actual[field], source[field], "Exact shared candidate evidence")
                effect_record(actual["effect"], len(plan["rules"]))
                self.captured_effect(name, cell, source, actual["effect"], columns, current["counterfactual_before_tuple"])
            by_row.setdefault(i, []).append(key)
        cap = 200 if name == "movies" else 20
        expected_groups, group = [], []
        for i in sorted(by_row):
            if group and len(group) + len(by_row[i]) > cap:
                expected_groups.append(group)
                group = []
            group.extend(by_row[i])
        if group:
            expected_groups.append(group)
        same(request.get("batch_cells"), expected_groups, "Fixed complete-row grouping")
        return request

    def round(self, base, plan, manifest, cells, previous, name, arm, number, columns, rows):
        directory = f"{base}/{arm}"
        round_dir = f"{directory}/rounds/{number:04d}"
        request_path, response_path = f"{round_dir}/request.json", f"{round_dir}/verification.json"
        request = self.request(self.store.read(request_path), plan, cells, previous, name, arm, number, columns, rows)
        batches, batch_paths = [], []
        expected_batch_names = {f"{i:04d}.json" for i in range(len(request["batch_cells"]))}
        batch_dir = self.authority.path(f"{round_dir}/batches")
        actual_batch_names = {path.name for path in batch_dir.iterdir()} if batch_dir.exists() else set()
        if actual_batch_names != expected_batch_names:
            raise ValueError("Exact batch file coverage mismatch")
        requested = {cell["cell_id"]: cell for cell in request["cells"]}
        marker = "Exact batch evidence follows:\n"
        for i, keys in enumerate(request["batch_cells"]):
            path = f"{round_dir}/batches/{i:04d}.json"
            batch = self.store.read(path)
            fields(batch, ("batch", "cell_ids", "prompt", "request_hash", "model", "reasoning_effort", "batch_hash"))
            check_hash(batch, "batch_hash")
            if type(batch["batch"]) is not int or batch["batch"] != i or batch["cell_ids"] != keys or batch["request_hash"] != request["request_hash"] or batch["model"] != MODEL or batch["reasoning_effort"] != EFFORT or type(batch["prompt"]) is not str or batch["prompt"].count(marker) != 1:
                raise ValueError("Exact batch envelope mismatch")
            preamble, evidence = batch["prompt"].split(marker)
            if not preamble or (hasattr(self, "prompt_preamble") and self.prompt_preamble != preamble):
                raise ValueError("Verifier prompt policy changed across paired rounds")
            self.prompt_preamble = preamble
            expected = {field: request[field] for field in ("protocol", "dataset", "arm", "external_round", "dirty_sha256", "request_hash", "rules", "delta_dimensions")}
            expected["cells"] = [requested[key] for key in keys]
            same(parse_json_text(evidence), expected, "Exact prompt evidence projection")
            batches.append(batch)
            batch_paths.append(path)
        worker = self.store.read(f"{round_dir}/worker_request.json")
        expected_worker = {"protocol": PROTOCOL, "dataset": name, "arm": arm, "external_round": number,
                           "model": MODEL, "reasoning_effort": EFFORT, "fork_turns": "none",
                           "request_path": str(self.authority.path(request_path)),
                           "batch_paths": [str(self.authority.path(path)) for path in batch_paths],
                           "response_path": str(self.authority.path(response_path)), "contract_path": str(self.authority.path(f"{EXPERIMENT}/worker_contract.md")),
                           "policy": plan["config"]["worker_policy"], "no_previous_verdicts": True,
                           "same_model_error_correlation": True, "dispatch_by_runner": False,
                           "dispatch_policy": "root_coordination_required_before_dispatch"}
        same(worker, expected_worker, "Exact worker request")
        response = self.store.read(response_path)
        verified, joints = expand_response(request, response, [batch["batch_hash"] for batch in batches])
        session = response["fresh_session_id"]
        if session in self.sessions:
            raise ValueError("Worker session reused across dataset/arm/round")
        self.sessions.add(session)
        absolute_round = str(self.authority.path(round_dir))
        owner = {"fresh_session_id": session, "request_hash": request["request_hash"], "round_directory": absolute_round,
                 "response_sha256": self.store.files[response_path]}
        for path in (f"{EXPERIMENT}/runs/worker_sessions/{digest(session)}.json", f"{EXPERIMENT}/runs/round_sessions/{digest(absolute_round)}.json"):
            same(self.store.read(path), owner, "Independent worker/round reservation")
        state_path, prediction_path = f"{round_dir}/state.json", f"{round_dir}/prediction.json"
        state = self.state(self.store.read(state_path), plan, cells, name, arm, number, (len(rows), len(columns)), rows)
        self.transition_checks(previous, state, request, verified, joints, cells)
        raw = self.store.read(prediction_path)
        self.prediction_state(raw, state, plan, name, arm, columns, rows)
        input_paths = [request_path, response_path, f"{directory}/initial_state.json", f"{base}/manifest.json"] + batch_paths
        if number > 1:
            input_paths.append(f"{directory}/rounds/{number-1:04d}/record.json")
        record_path = f"{round_dir}/record.json"
        record = self.store.read(record_path)
        expected_record = {"protocol": PROTOCOL, "external_round": number, "request_hash": request["request_hash"],
                           "parent_state_hash": previous["state_hash"], "state_hash": state["state_hash"],
                           "input_files": self.store.absolute_map(input_paths), "state_sha256": self.store.files[state_path],
                           "prediction_sha256": self.store.files[prediction_path], "fresh_session_id": session,
                           "response_format": "grouped" if "groups" in response else "direct",
                           "ground_truth_access": False, "frozen_before_evaluation": True}
        expected_record["record_hash"] = digest(expected_record)
        same(record, expected_record, "Complete immutable round record")
        history = {"round": number, "record_sha256": self.store.files[record_path], "request_hash": request["request_hash"],
                   "state_hash": state["state_hash"], "fresh_session_id": session}
        return state, raw, history

    def transition_checks(self, previous, state, request, verified, joints, cells):
        if state["events"][:len(previous["events"])] != previous["events"] or state["guard_records"][:len(previous["guard_records"])] != previous["guard_records"]:
            raise ValueError("Recorded event/guard ledger prefix was rewritten")
        events = state["events"][len(previous["events"]):]
        main = [event for event in events if event.get("request_hash") == request["request_hash"]]
        if len(main) != len(cells) or {event.get("cell_id") for event in main} != set(cells):
            raise ValueError("Missing exact per-cell recorded transition")
        guards = state["guard_records"][len(previous["guard_records"]):]
        by_event = {event["cell_id"]: event for event in main}
        before_guard, selected = {}, set()
        for key, cell in cells.items():
            item, event = verified[key], by_event[key]
            fields(event, ("round", "cell_id", "request_hash", "prior_candidate_id", "original_assessment", "candidate_assessments", "selection", "event", "next_candidate_id"))
            i, j = cell["row"], cell["column_index"]
            for field, expected in (("round", request["external_round"]), ("prior_candidate_id", previous["active"].get(key)),
                                    ("original_assessment", item["original_assessment"]), ("candidate_assessments", item["candidates"]), ("selection", item["selection"])):
                same(event.get(field), expected, "Recorded authorization event")
            old_score = previous["probabilities"][i][j]
            original = item["original_assessment"]
            decisive = assessment(original, True)
            expected_score = min(old_score, .1) if decisive and original["disposition"] == "supported" else max(old_score, .9) if decisive else old_score
            if state["probabilities"][i][j] != expected_score:
                raise ValueError("Recorded detector update lacks exact-original authorization")
            expected_latest = request["request_hash"] if decisive else previous["cells"][key]["latest_decisive_original_request"]
            if state["cells"][key]["latest_decisive_original_request"] != expected_latest:
                raise ValueError("Original-validity provenance changed without authorization")
            active = previous["active"].get(key)
            if decisive and original["disposition"] == "supported":
                active = None
                expected_event = "retracted_exact_original_supported"
            else:
                selection = item["selection"]
                assessments = {candidate["candidate_id"]: candidate["assessment"] for candidate in item["candidates"]}
                if selection["kind"] == "candidate" and selection["confidence"] >= .8 and assessment(assessments[selection["candidate_id"]]):
                    active = selection["candidate_id"]
                    selected.add(key)
                    expected_event = "explicit_supported_selection"
                elif active and assessments[active]["disposition"] == "rejected" and assessment(assessments[active]):
                    active = None
                    expected_event = "retracted_explicit_candidate_contradiction"
                else:
                    expected_event = "carried_epistemic_abstention_or_insufficient_authorization"
            if event.get("next_candidate_id") != active or event.get("event") != expected_event:
                raise ValueError("Recorded candidate action lacks explicit authorization")
            if active:
                before_guard[key] = active
        invalid_joint_rows = set()
        requested = {cell["cell_id"]: cell for cell in request["cells"]}
        for key in selected:
            cell = requested[key]
            candidate = next(c for c in cell["candidates"] if c["candidate_id"] == before_guard[key])
            if candidate["raw_effect"]["introduced_violation"]:
                joint = joints.get(verified[key]["joint_plan_id"])
                expected_members = {k: cid for k, cid in before_guard.items() if cells[k]["row"] == cell["row"] and state["cells"][k]["detector_bit"]}
                members = {member["cell_id"]: member["candidate_id"] for member in joint["members"]} if joint else {}
                if not joint or joint["disposition"] != "supported" or joint["confidence"] < .8 or members != expected_members:
                    invalid_joint_rows.add(cell["row"])
        proposed_rows = {}
        for key, cid in before_guard.items():
            cell = cells[key]
            if cell["row"] not in invalid_joint_rows and state["cells"][key]["detector_bit"]:
                proposed_rows.setdefault(cell["row"], {})[key] = next(c["value"] for c in cell["candidates"] if c["candidate_id"] == cid)
        if [guard["row"] for guard in guards] != sorted(proposed_rows):
            raise ValueError("Complete-row guard coverage/order mismatch")
        for guard in guards:
            row = guard["row"]
            if guard["round"] != request["external_round"] or guard["reason"] != "post_transition" or guard["proposed_values"] != proposed_rows[row]:
                raise ValueError("Captured full-row guard did not cover the exact proposed tuple")
            raw_results = next(c["raw_effect"]["before_rule_results"] for cell in cells.values() if cell["row"] == row for c in cell["candidates"])
            if guard["raw_rule_results"] != raw_results:
                raise ValueError("Captured full-row guard changed original rule evidence")
        rule_blocked = {guard["row"] for guard in guards if guard["blocked"]}
        blocked_rows = invalid_joint_rows | rule_blocked
        extra = [event for event in events if event not in main]
        expected_extra = []
        for key, cid in before_guard.items():
            row = cells[key]["row"]
            if row in blocked_rows:
                expected_extra.append({"round": request["external_round"], "cell_id": key,
                                       "event": "missing_complete_joint_authorization" if row in invalid_joint_rows else "whole_row_guard_blocked", "candidate_id": cid})
        for event in extra:
            fields(event, ("round", "cell_id", "event", "candidate_id"))
        if sorted(map(digest, extra)) != sorted(map(digest, expected_extra)):
            raise ValueError("Rollback ledger lacks exact joint/row guard justification")
        for key, cell in cells.items():
            active = before_guard.get(key)
            if cell["row"] in blocked_rows:
                active = None
            if state["active"].get(key) != active:
                raise ValueError("Applied hypothesis differs from authorized recorded state")
            expected_status = "freshly_supported" if active and key in selected else "carried_unresolved" if active else "not_active"
            if state["cells"][key]["support_status"] != expected_status:
                raise ValueError("Repair support status changed without explicit authorization")
        eligible = {(cell["row"], cell["column_index"]) for cell in cells.values()}
        for i, row in enumerate(state["probabilities"]):
            for j, value in enumerate(row):
                if (i, j) not in eligible and value != previous["probabilities"][i][j]:
                    raise ValueError("Unrequested detector score was changed")

    def prediction_state(self, raw, state, plan, name, arm, columns, rows):
        repairs = prediction(raw, (len(rows), len(columns)), "integer")
        same(raw["probabilities"], state["probabilities"], "Raw prediction/state probability matrix")
        expected_repairs = {}
        cells = {cell["cell_id"]: cell for cell in plan["cells"]}
        for key, cid in state["active"].items():
            cell = cells[key]
            if state["cells"][key]["detector_bit"]:
                value = next(candidate["value"] for candidate in cell["candidates"] if candidate["candidate_id"] == cid)
                if value == cell["original"]:
                    raise ValueError("Experimental prediction contains identity repair")
                expected_repairs[cell["row"], cell["column_index"]] = value
        if repairs != expected_repairs:
            raise ValueError("Exact raw repair map does not match final emitted state")
        metadata = raw["metadata"]
        for key, expected in {"system": PROTOCOL, "dataset": name, "arm": arm, "dirty_sha256": plan["dirty_sha256"],
                              "plan_hash": plan["plan_hash"], "protocol_hash": self.authority.protocol_hash,
                              "state_hash": state["state_hash"], "external_verification_rounds": state["external_round"],
                              "original_source_binary_decisions": plan["original_source_binary_decisions"],
                              "repair_statuses": state["cells"], "source_fidelity": plan["source_fidelity"],
                              "raw_statistics_sha256": plan["raw_statistics_sha256"], "same_model_error_correlation": True,
                              "ground_truth_access": False, "frozen_before_evaluation": True}.items():
            same(metadata.get(key), expected, f"Raw export metadata {key}")
        if "NOT original" not in metadata.get("upstream_binary_decisions_semantics", "") or "evaluation_interface_adapter" in metadata:
            raise ValueError("Raw experimental mask must remain explicitly distinguished from upstream source decisions")

    def validate(self):
        lock = self.lock()
        prepared = self.store.read(PREPARATION_PATH)
        check_hash(prepared, "index_hash")
        if prepared.get("protocol") != PROTOCOL or prepared.get("protocol_hash") != self.authority.protocol_hash or not isinstance(prepared.get("entries"), list):
            raise ValueError("Original preparation seal mismatch")
        entries = {}
        for entry in prepared["entries"]:
            key = entry.get("dataset"), entry.get("arm")
            if key in entries or key[0] not in DATASETS or key[1] not in ARMS:
                raise ValueError("Duplicate/unknown prepared arm")
            entries[key] = entry
        if set(entries) != {(name, arm) for name in DATASETS for arm in ARMS}:
            raise ValueError("Exact all-six both-arm preparation coverage required")
        for name in DATASETS:
            columns, rows = self.dirty(name)
            shape = len(rows), len(columns)
            dirty_sha = self.store.files[source_paths(name)[5]]
            ed, ec = self.source_pair(name, shape, dirty_sha)
            directories = [self.authority.relative(entries[name, arm]["run_directory"]) for arm in ARMS]
            bases = [directory.rsplit("/", 1)[0] for directory in directories]
            if bases[0] != bases[1] or any(directory != f"{bases[0]}/{arm}" for directory, arm in zip(directories, ARMS)):
                raise ValueError("Paired arms do not share one exact plan")
            base = bases[0]
            plan, manifest, cells = self.plan(name, base, lock, columns, rows, ed, ec)
            for arm, directory in zip(ARMS, directories):
                prepared_entry = entries[name, arm]
                initial_path = f"{directory}/initial_state.json"
                previous = self.state(self.store.read(initial_path), plan, cells, name, arm, 0, shape, rows)
                history = []
                for number in range(1, 4):
                    previous, raw, item = self.round(base, plan, manifest, cells, previous, name, arm, number, columns, rows)
                    history.append(item)
                    if number == 1:
                        req = f"{directory}/rounds/0001/request.json"
                        if prepared_entry.get("request_sha256") != self.store.files[req] or prepared_entry.get("response_binding", {}).get("request_hash") != item["request_hash"] or prepared_entry.get("response_binding", {}).get("candidate_universe_sha256") != plan["candidate_universe_sha256"]:
                            raise ValueError("Original root-sealed initial request changed")
                final_state_path, final_prediction_path, frozen_path = f"{directory}/state.json", f"{directory}/prediction.json", f"{directory}/frozen.json"
                final_state, final_prediction, frozen = self.store.read(final_state_path), self.store.read(final_prediction_path), self.store.read(frozen_path)
                same(final_state, previous, "Final/round3 state snapshot")
                same(final_prediction, raw, "Final/round3 raw prediction snapshot")
                # The frozen runner writes identical JSON bodies through the same
                # serializer. Require final snapshot bytes as well as semantics.
                if self.store.files[final_state_path] != self.store.files[f"{directory}/rounds/0003/state.json"] or self.store.files[final_prediction_path] != self.store.files[f"{directory}/rounds/0003/prediction.json"]:
                    raise ValueError("Final/round3 snapshot bytes differ")
                expected = {"protocol": PROTOCOL, "dataset": name, "arm": arm, "external_rounds": 3,
                            "stop_reason": "fixed_three_rounds", "history": history, "manifest_hash": manifest["manifest_hash"],
                            "protocol_hash": self.authority.protocol_hash, "state_sha256": self.store.files[final_state_path],
                            "prediction_sha256": self.store.files[final_prediction_path], "ground_truth_access": False,
                            "frozen_before_evaluation": True, "export_ready": True}
                expected["frozen_hash"] = digest(expected)
                same(frozen, expected, "Complete final freeze")
                self.entries.append({"dataset": name, "arm": arm, "run_directory": directory, "plan_path": f"{base}/plan.json",
                                     "manifest_path": f"{base}/manifest.json", "frozen_path": frozen_path,
                                     "state_path": final_state_path, "prediction_path": final_prediction_path,
                                     "completed_external_rounds": 3, "plan_hash": plan["plan_hash"],
                                     "candidate_universe_sha256": plan["candidate_universe_sha256"]})
                self.exports.append({"dataset": name, "arm": arm, "directory": directory, "prediction_path": final_prediction_path,
                                     "shape": shape, "dirty_sha256": dirty_sha})
        if len(self.sessions) != 36:
            raise ValueError("Exactly 36 distinct fresh worker sessions are required")
        if self.store.expected is not None and self.store.files != self.store.expected:
            raise ValueError("Completion-index file map has missing, substituted or unconsumed paths")
        return self


def completion_body(validation):
    body = {"schema_version": INDEX_SCHEMA, "protocol": PROTOCOL, "protocol_hash": validation.authority.protocol_hash,
            "status": "all_six_both_arms_three_complete_rounds", "datasets": list(DATASETS), "arms": list(ARMS),
            "preparation_index_path": PREPARATION_PATH, "protocol_lock_path": LOCK_PATH,
            "entries": validation.entries, "files": dict(sorted(validation.store.files.items())),
            "completed_round_records": 36, "distinct_fresh_worker_sessions": 36,
            "evaluation_performed": False, "model_dispatch_performed": False}
    body["index_hash"] = digest(body)
    return body


def write_immutable(path, data, *, before_publish=None):
    path = Path(path)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite existing artifact: {path}")
        if before_publish is not None:
            before_publish()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if before_publish is not None:
            before_publish()
        # Publish the fully written inode atomically and never replace a name.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise ValueError(f"Refusing to overwrite existing artifact: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_directory(stage, destination):
    """Linux atomic directory publication with RENAME_NOREPLACE."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise ValueError("Atomic no-replace directory publication is unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
        raise ValueError(f"Atomic export publication refused: errno={ctypes.get_errno()}")


def build_completion_index(authority):
    validation = Validator(authority).validate()
    path = authority.path(INDEX_PATH)
    write_immutable(path, json_bytes(completion_body(validation)), before_publish=validation.store.recheck)
    return {"completion_index_path": str(path), "completion_index_sha256": sha256(path), "completed_arms": 12,
            "completed_round_records": 36, "evaluation_performed": False}


def validate_completion_index(authority, supplied_sha):
    hash_string(supplied_sha)
    path = authority.path(INDEX_PATH)
    if not path.is_file() or sha256(path) != supplied_sha:
        raise ValueError("Root-supplied completion-index byte SHA mismatch")
    body = read_json(path)
    check_hash(body, "index_hash")
    if body.get("schema_version") != INDEX_SCHEMA or not isinstance(body.get("files"), dict):
        raise ValueError("Unsupported completion-index schema")
    for relative, expected in body["files"].items():
        authority.path(relative)
        hash_string(expected)
    validation = Validator(authority, body["files"]).validate()
    same(body, completion_body(validation), "Exact all-12 completion index")
    return validation


def export(authority, supplied_sha):
    validation = validate_completion_index(authority, supplied_sha)
    root = authority.path("results/conservative_seed_v1")
    sources, generated = {}, {}
    source = lambda destination, relative: sources.__setitem__(destination, relative)
    source("provenance/completion_index.json", INDEX_PATH)
    source("provenance/protocol_lock.json", LOCK_PATH)
    source("evaluation_interface_note.md", NOTE_PATH)
    for name in DATASETS:
        for method in SOURCE_NAMES:
            relative = f"results/source_baselines_v1/predictions/{name}/{method}.json"
            source(f"predictions/{name}/{method}.json", relative)
            source(f"raw/source_baselines/{name}/{method}.json", relative)
        source(f"raw/source_baselines/{name}/frozen.json", source_paths(name)[0])
    for item in validation.exports:
        name, arm, directory = item["dataset"], item["arm"], item["directory"]
        relative = item["prediction_path"]
        snapshot = f"raw/predictions/{name}/{arm}.json"
        source(snapshot, relative)
        raw = read_json(authority.path(relative))
        normalized = normalize_prediction(raw, shape=item["shape"], raw_sha=validation.store.files[relative], snapshot_path=snapshot, completion_sha=supplied_sha)
        generated[f"predictions/{name}/{arm}.json"] = json_bytes(normalized)
        for state in ("initial_state.json", "state.json"):
            source(f"provenance/states/{name}/{arm}/{state}", f"{directory}/{state}")
        for number in range(1, 4):
            for filename in ("state.json", "record.json"):
                source(f"provenance/states/{name}/{arm}/rounds/{number:04d}/{filename}", f"{directory}/rounds/{number:04d}/{filename}")
        source(f"provenance/states/{name}/{arm}/frozen.json", f"{directory}/frozen.json")
    expected_files = {destination: sha256(authority.path(relative)) for destination, relative in sources.items()}
    expected_files.update({destination: hashlib.sha256(data).hexdigest() for destination, data in generated.items()})
    for name in DATASETS:
        methods = (*SOURCE_NAMES, *ARMS)
        inputs = {str(authority.path(relative)): expected for relative, expected in validation.store.files.items()
                  if relative in LOCKED_SOURCES or relative in INTERFACE_FILES or relative == LOCK_PATH or relative == PREPARATION_PATH or relative in source_paths(name)
                  or relative.startswith(f"{EXPERIMENT}/runs/{name}/") or "/worker_sessions/" in relative or "/round_sessions/" in relative}
        frozen = {"dataset": name, "dirty_sha256": validation.store.files[source_paths(name)[5]],
                  "prediction_files": {f"{method}.json": expected_files[f"predictions/{name}/{method}.json"] for method in methods},
                  "inference_inputs": inputs, "protocol": PROTOCOL, "protocol_hash": authority.protocol_hash,
                  "completion_index_sha256": supplied_sha, "evaluation_interface_adapter": ADAPTER,
                  "frozen_before_evaluation": True, "ground_truth_access": False,
                  "official_exact_ec_baseline_present": False,
                  "qualified_ec_caveat": "Author-source reconstruction; historical identity to omitted pinned ZeroEC prompts remains unproven; zero fixed examples.",
                  "methods": list(methods), "selection_policy": "both fixed paired arms; no outcome-based winner"}
        generated[f"frozen/{name}.json"] = json_bytes(frozen)
        expected_files[f"frozen/{name}.json"] = hashlib.sha256(generated[f"frozen/{name}.json"]).hexdigest()
    manifest = {"schema_version": ADAPTER, "protocol": PROTOCOL, "protocol_hash": authority.protocol_hash,
                "completion_index_sha256": supplied_sha, "methods": [*SOURCE_NAMES, *ARMS],
                "datasets": list(DATASETS), "files": dict(sorted(expected_files.items())),
                "source_inputs": validation.store.files, "interface_only_transformations": ["integer mask to boolean", "repair_statuses to bound raw snapshot pointer"],
                "original_source_baselines_byte_exact": True, "raw_predictions_and_all_state_ledgers_retained": True,
                "evaluation_performed": False, "model_dispatch_performed": False, "winner_selected": False}
    manifest["export_hash"] = digest(manifest)
    manifest_bytes = json_bytes(manifest)
    if root.exists():
        manifest_path = root / "export_manifest.json"
        if root.is_symlink() or not root.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("Existing export differs; refusing overwrite")
        expected_paths = set(expected_files) | {"export_manifest.json"}
        expected_directories = {parent.as_posix() for relative in expected_paths for parent in Path(relative).parents if parent != Path(".")}
        actual_files, actual_directories = set(), set()
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise ValueError("Existing export contains a forbidden filesystem entry")
            (actual_files if path.is_file() else actual_directories).add(relative)
        if actual_files != expected_paths or actual_directories != expected_directories:
            raise ValueError("Existing export path coverage changed")
        for destination, expected in expected_files.items():
            path = root / destination
            if path.is_symlink() or sha256(path) != expected:
                raise ValueError("Existing export bytes changed")
        validation.store.recheck()
        if sha256(authority.path(INDEX_PATH)) != supplied_sha:
            raise ValueError("Root completion index changed during export")
        return {"status": "already_exported", "output_root": str(root), "export_hash": manifest["export_hash"]}
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for destination, relative in sources.items():
            target = stage / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(authority.path(relative), target)
        for destination, data in generated.items():
            target = stage / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for destination, expected in expected_files.items():
            if sha256(stage / destination) != expected:
                raise ValueError("Snapshot copy/normalized export byte mismatch")
        (stage / "export_manifest.json").write_bytes(manifest_bytes)
        validation.store.recheck()
        if sha256(authority.path(INDEX_PATH)) != supplied_sha:
            raise ValueError("Root completion index changed during export")
        if root.exists():
            raise ValueError("Output appeared during export; refusing overwrite")
        publish_directory(stage, root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"status": "exported", "output_root": str(root), "export_hash": manifest["export_hash"],
            "evaluation_performed": False, "model_dispatch_performed": False}


def main(argv=None, *, authority=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-completion-index", help="Read-only validation; writes an index only after all 12 arms are complete")
    validate = sub.add_parser("validate")
    validate.add_argument("--completion-index-sha256", required=True)
    output = sub.add_parser("export")
    output.add_argument("--completion-index-sha256", required=True)
    args = parser.parse_args(argv)
    authority = authority or Authority(Path(__file__).resolve().parents[1])
    if args.command == "build-completion-index":
        result = build_completion_index(authority)
    elif args.command == "validate":
        validated = validate_completion_index(authority, args.completion_index_sha256)
        result = {"status": "valid", "completed_arms": len(validated.entries), "completed_round_records": 36, "evaluation_performed": False}
    else:
        result = export(authority, args.completion_index_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
