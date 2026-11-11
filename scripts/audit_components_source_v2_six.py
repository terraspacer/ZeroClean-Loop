#!/usr/bin/env python3
"""Evaluator-only audit of the complete first-verified-pass component study.

Reads immutable artifacts, independently checks transport coverage/redaction,
then evaluates raw CSV strings. No inference/exporter implementation is imported.
"""

import ast
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import audit_six_dataset_source as a


NAME = "components_source_v2"
ROOT = a.REPO / "results" / NAME
OUTPUT = a.REPO / "results/evaluation_audits/components_source_v2_six.json"
INPUT_SNAPSHOTS = a.REPO / "results/evaluation_audits/components_source_v2_six_inputs"
QUARANTINE = a.REPO / "results/evaluation_only_failures/audit_components_source_v2_six"
VARIANTS = ("without_counterfactual_first_pass", "without_multi_column_first_pass")
CF_FIELDS = frozenset(("delta_features", "delta_dimensions", "introduced_violation",
                       "after_rule_violations", "counterfactual_error", "original_rule_violations"))
TRANSPORT = a.REPO / "artifacts/queue"


def snapshot_exact(source, target):
    raw = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError("Refusing to replace a different historical audit input snapshot")
    else:
        with target.open("xb") as stream:
            stream.write(raw)
    a.compare(a.sha(raw), a.hashfile(target), f"snapshot/{a.short(target)}")
    a.compare(a.sha(raw), a.hashfile(source), f"snapshot/{a.short(source)}")
    return {"path": a.short(target), "sha256": a.sha(raw), "bytes": len(raw)}


def redaction_matches(source, redacted, root=True):
    """Compare recursively without allocating a second copy of a large request."""
    if isinstance(source, dict):
        if not isinstance(redacted, dict):
            return False
        ignored = CF_FIELDS | {"request_hash"}
        if root:
            ignored |= {"experiment_preparation", "config", "component_ablation"}
        expected_keys = set(source) - ignored
        actual_keys = set(redacted) - ({"config", "component_ablation", "request_hash"} if root else set())
        return expected_keys == actual_keys and all(redaction_matches(source[k], redacted[k], False) for k in expected_keys)
    if isinstance(source, list):
        return isinstance(redacted, list) and len(source) == len(redacted) and all(redaction_matches(x, y, False) for x, y in zip(source, redacted))
    return source == redacted


def prohibited_key_count(obj):
    if isinstance(obj, dict):
        return sum(k in CF_FIELDS for k in obj) + sum(prohibited_key_count(x) for x in obj.values())
    if isinstance(obj, list):
        return sum(prohibited_key_count(x) for x in obj)
    return 0


def batch_groups(cells, limit):
    by_row = {}
    for cell in cells:
        by_row.setdefault(cell["row"], []).append(cell)
    groups, pending = [], []
    for row in sorted(by_row):
        whole_tuple = by_row[row]
        if pending and len(pending) + len(whole_tuple) > limit:
            groups.append(pending)
            pending = []
        pending.extend(whole_tuple)
    if pending:
        groups.append(pending)
    return groups


def parse_reply(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    body = json.loads(text)
    assert isinstance(body, dict) and isinstance(body.get("cells"), list)
    return body["cells"]


def cell_map(cells, header, *, request=False):
    indexed = {}
    for cell in cells:
        i = cell["row"]
        j = cell["column_index"] if request else cell["column"]
        j = header.index(j) if isinstance(j, str) else j
        assert type(i) is int and type(j) is int and (i, j) not in indexed
        indexed[i, j] = cell
    return indexed


def validate_cells(header, dirty, requested, replies):
    """Check every coordinate, original, selected value and candidate score."""
    source, response = cell_map(requested, header, request=True), cell_map(replies, header)
    assert set(source) == set(response)

    def probability(value):
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1

    candidate_count = 0
    for (i, j), cell in source.items():
        assert 0 <= i < len(dirty) and 0 <= j < len(header)
        assert cell["column"] == header[j] and cell["original"] == dirty[i][j]
        assert cell["tuple"] == dict(zip(header, dirty[i]))
        candidates = [c["value"] for c in cell["candidates"]]
        assert all(type(v) is str and v != cell["original"] for v in candidates)
        assert len(candidates) == len(set(candidates))
        item = response[i, j]
        if "original" in item:
            assert item["original"] == cell["original"]
        for key in ("original_valid_score", "confidence"):
            assert probability(item[key])
        assert "selected_value" in item
        selected = item["selected_value"]
        assert selected is None or type(selected) is str and selected in set(candidates) | {cell["original"]}
        assert isinstance(item["explanation"], str)
        if "joint_repair_required" in item:
            assert type(item["joint_repair_required"]) is bool
        scored = item["candidates"]
        assert len(scored) == len(candidates)
        assert {c["value"] for c in scored} == set(candidates)
        for candidate in scored:
            assert probability(candidate["score"])
            if "joint_repair_required" in candidate:
                assert type(candidate["joint_repair_required"]) is bool
        candidate_count += len(candidates)
    return {"requested_cells": len(source), "replied_cells": len(response), "requested_alternatives": candidate_count,
            "scored_alternatives": sum(len(c["candidates"]) for c in response.values()),
            "missing_cells": 0, "extra_cells": 0, "duplicate_cells": 0, "missing_alternatives": 0,
            "invalid_probabilities": 0, "out_of_request_selections": 0}


def transport_prefix():
    path = a.REPO / "zeroclean/verification.py"
    a.hashfile(path)
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "queue_verification")
    assignment = next(n for n in ast.walk(function) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "prompt" for t in n.targets))
    assert isinstance(assignment.value, ast.BinOp) and isinstance(assignment.value.left, ast.Constant)
    return assignment.value.left.value


def check_transport(dataset, variant, payload, combined, header, dirty, prefix):
    limit = 200 if dataset == "movies" else 20
    groups = batch_groups(payload["cells"], limit)
    queue_hashes = combined["queue_request_hashes"]
    a.compare(len(groups), len(queue_hashes), f"transport/{dataset}/{variant}/group_count")
    a.compare(len(queue_hashes), len(set(queue_hashes)), f"transport/{dataset}/{variant}/unique_queue_hashes")
    assert len(groups) == len(queue_hashes)
    all_replies, attempts = [], Counter()
    for group_number, (cells, request_hash) in enumerate(zip(groups, queue_hashes)):
        request_path = TRANSPORT / "requests" / f"{request_hash}.json"
        response_path = TRANSPORT / "responses" / f"{request_hash}.json"
        queued, received = a.read_json(request_path), a.read_json(response_path)
        a.HASH_CHECKS["content_addressed_transport_requests"] += 1
        a.HASH_CHECKS["transport_response_files"] += 1
        a.compare(request_hash, a.canonical(queued), f"transport/{dataset}/{variant}/{group_number}/request_content_hash")
        a.compare({"stage", "prompt", "metadata", "model", "reasoning_effort"}, set(queued), f"transport/{dataset}/{variant}/{group_number}/request_fields")
        a.compare("loop_verification", queued["stage"], f"transport/{dataset}/{variant}/{group_number}/stage")
        for key in ("model", "reasoning_effort"):
            a.compare(payload[key], queued[key], f"transport/{dataset}/{variant}/{group_number}/{key}")
            a.compare(payload[key], received[key], f"transport/{dataset}/{variant}/{group_number}/reply_{key}")
        a.compare(request_hash, received["request_hash"], f"transport/{dataset}/{variant}/{group_number}/reply_hash")
        metadata = queued["metadata"]
        expected = {"dataset": dataset, "dirty_fingerprint": payload["dirty_sha256"], "system": "zeroclean_loop",
                    "stage_contract": "counterfactual_verification", "candidate_request_hash": payload["request_hash"],
                    "group": group_number, "cells": [[c["row"], c["column_index"]] for c in cells],
                    "attempt": metadata["attempt"]}
        assert type(metadata["attempt"]) is int and 0 <= metadata["attempt"] < 3
        a.compare(True, metadata == expected, f"transport/{dataset}/{variant}/{group_number}/exact_variant_metadata")
        attempts[metadata["attempt"]] += 1
        assert queued["prompt"].startswith(prefix)
        evidence = json.loads(queued["prompt"][len(prefix):])
        expected_evidence = {k: payload[k] for k in ("dataset", "dirty_sha256", "rules", "delta_dimensions", "config") if k in payload}
        expected_evidence["cells"] = cells
        a.compare(True, evidence == expected_evidence, f"transport/{dataset}/{variant}/{group_number}/exact_redacted_variant_evidence")
        if variant == VARIANTS[0]:
            a.compare(0, prohibited_key_count(evidence), f"transport/{dataset}/{variant}/{group_number}/counterfactual_field_count")
        reply = parse_reply(received["response"])
        validate_cells(header, dirty, cells, reply)
        all_replies.extend(reply)
    a.compare(True, all_replies == combined["cells"], f"transport/{dataset}/{variant}/exact_assembled_reply")
    return {"batch_size": limit, "tuple_cells_kept_together": True, "groups": len(groups),
            "queue_request_count": len(queue_hashes), "request_hashes_unique": len(queue_hashes) == len(set(queue_hashes)),
            "exact_variant_evidence_verified": True, "exact_assembled_reply_match": all_replies == combined["cells"],
            "accepted_attempt_counts": {str(k): v for k, v in sorted(attempts.items())},
            "request_files_sha256_and_content_hashes_recorded": True, "response_files_sha256_recorded": True}


def component_protocols(frozen):
    prefix, outcomes = transport_prefix(), {}
    globally_used = set()
    for dataset in a.SIX:
        manifest = frozen[NAME, dataset]
        protocol = manifest["component_ablation_protocol"]
        for key, expected in (("scope", "components_of_first_verified_pass"), ("inference_passes", 1),
                              ("independent_variant_verification", True), ("final_tuple_guard_retained", True),
                              ("iterative_no_component_experiment_claimed", False)):
            a.compare(expected, protocol[key], f"components/{dataset}/{key}")
        prepared_paths = [a.pathof(k) for k in manifest["inference_inputs"]
                          if "/artifacts/components_source_v2/" in k and k.endswith("/manifest.json")]
        assert len(prepared_paths) == 1
        prepared_path = prepared_paths[0]
        prepared = a.read_json(prepared_path)
        identity = prepared["identity"]
        a.compare(manifest["protocol_hash"], prepared["protocol_hash"], f"components/{dataset}/preparation_protocol")
        a.compare(set(VARIANTS), set(identity["variants"]), f"components/{dataset}/variant_set")
        full_run = Path(identity["full_run_directory"])
        initial = a.read_json(full_run / "rounds/0001/request.json")
        full_reply = a.read_json(full_run / "rounds/0001/verification.json")
        full_source = full_run / "rounds/0001/prediction.json"
        full_reference = ROOT / "predictions" / dataset / "full_first_verified_round.json"
        a.compare(a.hashfile(full_source), a.hashfile(full_reference), f"components/{dataset}/exact_first_verified_reference")
        a.compare(protocol["full_reference_prediction_sha256"], a.hashfile(full_source), f"components/{dataset}/reference_binding")
        a.compare(protocol["full_initial_request_hash"], initial["request_hash"], f"components/{dataset}/initial_request_binding")
        a.compare(protocol["full_initial_verification_sha256"], a.hashfile(full_run / "rounds/0001/verification.json"), f"components/{dataset}/initial_reply_binding")
        a.compare(initial["request_hash"], a.canonical({k: v for k, v in initial.items() if k != "request_hash"}), f"components/{dataset}/initial_request_content_hash")
        header, dirty = a.raw_table(a.REPO / "vendor/ZeroED/data" / f"{dataset}_error-01.csv")
        full_coverage = validate_cells(header, dirty, initial["cells"], full_reply["cells"])
        variant_results, previous_hashes = {}, set(full_reply.get("queue_request_hashes", []))
        for variant in VARIANTS:
            definition = identity["variants"][variant]
            request_path = prepared_path.parent / definition["request_file"]
            reply_path = prepared_path.parent / definition["verification_file"]
            payload, combined = a.read_json(request_path), a.read_json(reply_path)
            status = a.read_json(reply_path.with_suffix(".status.json"))
            a.compare(definition["request_sha256"], a.hashfile(request_path), f"components/{dataset}/{variant}/prepared_request_bytes")
            a.compare(definition["request_hash"], payload["request_hash"], f"components/{dataset}/{variant}/prepared_request_hash")
            a.compare(payload["request_hash"], a.canonical({k: v for k, v in payload.items() if k != "request_hash"}), f"components/{dataset}/{variant}/request_content_hash")
            a.compare(True, payload["request_hash"] != initial["request_hash"], f"components/{dataset}/{variant}/fresh_variant_request")
            for key in ("request_hash", "model", "reasoning_effort", "dirty_sha256", "dataset"):
                a.compare(payload[key], combined[key], f"components/{dataset}/{variant}/reply_envelope/{key}")
            a.compare(dataset, payload["dataset"], f"components/{dataset}/{variant}/dataset")
            a.compare(manifest["dirty_sha256"], payload["dirty_sha256"], f"components/{dataset}/{variant}/dirty_identity")
            config = payload["config"]
            for key, expected in (("rounds", 1), ("iterative_verification", True),
                                  ("counterfactual", variant != VARIANTS[0]), ("multi_column", variant != VARIANTS[1])):
                a.compare(expected, config[key], f"components/{dataset}/{variant}/config/{key}")
            component_meta = payload["component_ablation"]
            for key, expected in (("variant", variant), ("scope", "components_of_first_verified_pass"),
                                  ("initial_verifier_reused", False), ("independent_verification_required", True),
                                  ("iterative_no_component_experiment_claimed", False)):
                a.compare(expected, component_meta[key], f"components/{dataset}/{variant}/metadata/{key}")
            a.compare(a.hashfile(full_source), component_meta["reference_prediction_sha256"], f"components/{dataset}/{variant}/metadata/reference_hash")
            if variant == VARIANTS[0]:
                redacted = redaction_matches(initial, payload)
                field_count = prohibited_key_count(payload)
                a.compare(True, redacted, f"components/{dataset}/{variant}/exact_recursive_redaction")
                a.compare(0, field_count, f"components/{dataset}/{variant}/forbidden_fields")
                a.compare(False, "experiment_preparation" in payload, f"components/{dataset}/{variant}/nested_full_config_absent")
                redaction = {"exact_recursive_redaction_of_full_request": redacted, "prohibited_counterfactual_field_count": field_count,
                             "nested_full_preparation_config_absent": "experiment_preparation" not in payload}
            else:
                single_rules = [r for r in initial["rules"] if r.get("kind") == "single"]
                a.compare(True, payload["rules"] == single_rules, f"components/{dataset}/{variant}/exact_single_column_rules")
                a.compare(0, payload["dependency_count"], f"components/{dataset}/{variant}/dependency_count")
                fd_nonzero = sum(c.get("delta_features", [0, 0])[1] != 0 for cell in payload["cells"] for c in cell["candidates"])
                a.compare(0, fd_nonzero, f"components/{dataset}/{variant}/nonzero_fd_delta_count")
                prep = payload.get("experiment_preparation", {})
                a.compare(False, prep.get("config", {}).get("multi_column"), f"components/{dataset}/{variant}/nested_multi_column_config")
                redaction = {"exact_full_reference_single_column_rule_subset": payload["rules"] == single_rules,
                             "multi_column_rule_count": sum(r.get("kind") != "single" for r in payload["rules"]),
                             "dependency_count": payload["dependency_count"], "nonzero_fd_delta_count": fd_nonzero}
            coverage = validate_cells(header, dirty, payload["cells"], combined["cells"])
            queue_hashes = set(combined["queue_request_hashes"])
            a.compare(0, len(queue_hashes & previous_hashes), f"components/{dataset}/{variant}/no_full_or_other_variant_queue_reuse")
            a.compare(0, len(queue_hashes & globally_used), f"components/{dataset}/{variant}/no_other_dataset_queue_reuse")
            transport = check_transport(dataset, variant, payload, combined, header, dirty, prefix)
            for key, expected in (("candidate_request_hash", payload["request_hash"]), ("groups", transport["groups"]),
                                  ("pending_groups", 0), ("requested_cells", len(payload["cells"])), ("completed_cells", len(payload["cells"]))):
                a.compare(expected, status[key], f"components/{dataset}/{variant}/status/{key}")
            prediction = a.read_json(ROOT / "predictions" / dataset / f"{variant}.json")
            meta = prediction["metadata"]["component_ablation"]
            a.compare(payload["request_hash"], prediction["metadata"]["request_hash"], f"components/{dataset}/{variant}/prediction_request_binding")
            a.compare(a.hashfile(reply_path), meta["verification_sha256"], f"components/{dataset}/{variant}/prediction_reply_binding")
            a.compare("independent_first_pass_variant", meta["verification_mode"], f"components/{dataset}/{variant}/prediction_mode")
            a.compare(manifest["protocol_hash"], meta["protocol_hash"], f"components/{dataset}/{variant}/prediction_protocol_binding")
            variant_results[variant] = {"coverage": coverage, "redaction": redaction, "transport": transport,
                "fresh_request_distinct_from_full_reference": payload["request_hash"] != initial["request_hash"],
                "full_reference_or_other_variant_queue_reuse_count": len(queue_hashes & previous_hashes),
                "other_dataset_queue_reuse_count": len(queue_hashes & globally_used),
                "request_hash": payload["request_hash"], "request_sha256": a.hashfile(request_path),
                "reply_sha256": a.hashfile(reply_path), "pending_groups": status["pending_groups"],
                "transport_retry_error_records": len(status["response_errors"])}
            previous_hashes.update(queue_hashes)
            globally_used.update(queue_hashes)
            print(json.dumps({"component_protocol_finished": dataset, "variant": variant,
                              "groups": transport["groups"], "requested_cells": coverage["requested_cells"]}), flush=True)
        outcomes[dataset] = {"scope": "components_of_first_verified_pass", "full_reference": "rounds/0001/prediction.json",
                             "full_reference_coverage": full_coverage, "variants": variant_results}
    return {"scope": "components_of_first_verified_pass", "inference_passes": 1,
            "full_reference_is_first_externally_verified_round": True,
            "iterative_no_component_experiment_claimed": False,
            "transport_batch_sizes": {d: 200 if d == "movies" else 20 for d in a.SIX},
            "unique_variant_queue_requests_checked": len(globally_used), "datasets": outcomes}


def main():
    started = datetime.now(timezone.utc).isoformat()
    a.ROOTS, a.QUARANTINE = {NAME: a.SIX}, QUARANTINE
    # Exact copies are made before any benchmark reference table is read.
    summary_snapshot = snapshot_exact(ROOT / "summary_metrics.json", INPUT_SNAPSHOTS / "summary_metrics.json")
    freeze_snapshots = {d: snapshot_exact(ROOT / "frozen" / f"{d}.json", INPUT_SNAPSHOTS / "frozen" / f"{d}.json") for d in a.SIX}
    for path in (Path(__file__), Path(a.__file__)):
        a.hashfile(path)
    selfcheck = a.semantic_checks()
    frozen, summaries, counts = a.preflight(a.ROOTS)
    # Evaluate against the preserved historical summary, not a later mutable pointer.
    historical_summary = a.read_json(INPUT_SNAPSHOTS / "summary_metrics.json")
    a.compare(True, historical_summary == summaries[NAME], "historical_summary/exact_original_content")
    summaries[NAME] = historical_summary
    protocols = component_protocols(frozen)
    prelabel_discrepancies = len(a.DISCREPANCIES)
    print(json.dumps({"all_prelabel_hash_and_transport_checks_complete": True, "discrepancy_count": prelabel_discrepancies}), flush=True)
    outputs, dimensions = a.audit_metrics(frozen, summaries)
    changed = []
    for path, expected in list(a.SNAPSHOTS.items()):
        actual = a.hashfile(path, refresh=True)
        if actual != expected:
            changed.append(a.short(path))
            a.compare(expected, actual, f"{a.short(path)}/post_audit_hash")
    report = {
        "audit_schema": "independent-components-source-v2-six-v1", "evaluation_only": True,
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_input_snapshots": {"summary": summary_snapshot, "frozen_manifests": freeze_snapshots,
                                      "snapshotted_before_any_reference_read": True},
        "method": {"implementation": "Python standard library plus separate independent evaluator-only helper; no project evaluator/inference/exporter imports or execution",
                   "reference_policy": "Exact raw CSV strings; no normalization or null coercion",
                   "repair_policy": "TP requires dirty cell changed exactly to reference; wrong dirty edit counts FP and FN; unchanged clean earns no TP",
                   "detection_policy": "Declared threshold/operator with stored-mask precedence; component requests use >=0.5",
                   "aggregation": "Unweighted macro F1 over all six datasets",
                   "examples_quarantined": a.short(QUARANTINE), "no_inference_or_tuning": True},
        "semantic_self_checks": selfcheck,
        "prelabel_integrity": {"all_prediction_inference_and_transport_hashes_checked_before_reference_read": True,
                               "discrepancy_count": prelabel_discrepancies, "root_counts": counts,
                               "hash_checks": dict(a.HASH_CHECKS), "unique_lineage_records_checked": len(a.LINEAGE_CHECKED)},
        "dimensions": dimensions, "experiments": outputs, "component_protocol_checks": protocols,
        "scope_qualification": "Six-dataset components of the first verified pass only; full reference is the first externally verified round. This does not establish full iterative no-counterfactual or no-multi-column effects or official baseline targets.",
        "historical_four_dataset_audit": "The earlier components_source_v2 entry in six_dataset_source_v1_and_components.json remains a historical four-dataset audit; this separate snapshot audits the completed six-dataset study.",
        "read_input_integrity": {"unique_files_rehashed_after_audit": len(a.SNAPSHOTS), "changed_files": changed, "all_stable": not changed},
        "discrepancy_count": len(a.DISCREPANCIES), "discrepancies": a.DISCREPANCIES,
        "input_sha256": {a.short(p): h for p, h in sorted(a.SNAPSHOTS.items(), key=lambda x: str(x[0]))},
        "auditor_script_sha256": a.hashfile(Path(__file__), refresh=True),
        "independent_evaluator_helper_sha256": a.hashfile(Path(a.__file__), refresh=True),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=lambda x: sorted(x) if isinstance(x, set) else str(x)) + "\n")
    print(json.dumps({"output": a.short(OUTPUT), "discrepancy_count": len(a.DISCREPANCIES), "all_inputs_stable": not changed,
                      "prediction_runs": counts[NAME]["prediction_files"], "declared_inference_entries": counts[NAME]["inference_input_entries"],
                      "unique_variant_transport_requests": protocols["unique_variant_queue_requests_checked"]}), flush=True)


if __name__ == "__main__":
    main()
