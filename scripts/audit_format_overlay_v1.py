#!/usr/bin/env python3
"""Evaluator-only immutable-provenance and raw-string scoring audit.

Does not import or execute the adapter, exporter, reviewer, sandbox or inference.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import audit_six_dataset_source as a
import audit_components_source_v2_six as t
import audit_criteria_and_verified_overlays as graph
import audit_detector_pair_and_model_average as data

NAME = "format_overlay_v1"
ROOT = a.REPO / "results" / NAME
APPLIED = a.REPO / "artifacts/format_overlay_v1_applied"
PREPARED = a.REPO / "artifacts/format_overlay_v1_prepared"
REVIEWER = a.REPO / "artifacts/format_contracts_v1_verification"
AUTHOR = a.REPO / "experimental/format_contracts_v1"
OWNED = a.REPO / "results/evaluation_audits/format_overlay_v1_inputs"
OUTPUT = a.REPO / "results/evaluation_audits/format_overlay_v1.json"
QUARANTINE = a.REPO / "results/evaluation_only_failures/audit_format_overlay_v1"
PINS = {
    APPLIED / "manifest.json": "b60373536f1ddefaf26de2aebb5c1f8c135f08c8044a073a69a6a52680a70a44",
    REVIEWER / "manifest.json": "3bc9f051649affa62eddb5b95491f6695a42bba63e3b5826858e7b59677e02a3",
    AUTHOR / "manifest.json": "9d160112ad6d8047437014a5dccd80c79d4588dc182a04c8e0fdf43501a9fca8",
}
SYSTEMS = {"zeroclean_loop.json", "source_v2_parent.json", "zeroed.json", "zeroed_zeroec_reconstruction.json"}
REVIEWER_ID = "/root/format_contracts_verifier"
EXPECTED_POLICY = {"original_valid_min": .9, "original_invalid_max": .1, "decision_confidence_min": .8,
    "verified_repair_confidence_min": .8, "detection_threshold": .5, "valid_detection_cap": .1,
    "invalid_detection_floor": .9, "delta_weights": [2.2, 1.8, .65, .35],
    "delta_adjusted_log_support_min": math.log(4), "overlap_policy": "preserve_all_overlapping_selected_action_footprints",
    "final_guard": "approved_rules_parent_relative_whole_row_exact_ED_EC_rollback", "statistical_observations": "raw_dirty_only_leave_one_out"}
ERRORS = Counter()


def ok(value, category):
    ERRORS[category] += not bool(value)


def check(path, expected):
    path = a.pathof(path)
    ok(not path.is_symlink(), "bound_paths_are_regular_files")
    a.check_hash(path, expected, "format_bound_input/" + a.short(path), "format_additional_bound_entries")


def sealed(body, field):
    ok(body[field] == a.canonical({k: v for k, v in body.items() if k != field}), "declared_object_content_seals")


def inventory(path, field, seal=None):
    body = a.read_json(path)
    if seal:
        sealed(body, seal)
    for rel, fingerprint in body[field].items():
        relative = Path(rel)
        ok(not relative.is_absolute() and ".." not in relative.parts, "inventory_path_confinement")
        check(path.parent / rel, fingerprint)
    return body


def compact(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def numeric_or_null(value):
    return value is None or (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1)


def review_structure(dataset, request, reply, coverage):
    sections = (("contracts", "contract_id"), ("scalar_candidates", "candidate_id"), ("atomic_bundles", "bundle_id"), ("cells", "cell_id"))
    expected = {"contracts": {r["contract_id"]: r for r in request["contracts"]},
        "scalar_candidates": {r["candidate_id"]: r for c in request["cells"] for r in c["candidates"]},
        "atomic_bundles": {r["bundle_id"]: r for r in request["atomic_bundles"]},
        "cells": {r["cell_id"]: r for r in request["cells"]}}
    observed_groups = defaultdict(list)
    for section, key in sections:
        actual = {r[key]: r for r in reply[section]}
        ok(len(actual) == len(reply[section]) and set(actual) == set(expected[section]), "exact_unique_review_id_coverage")
        ok(coverage["exact_ids_sha256"][section] == a.canonical(sorted(actual)), "grouped_exact_id_set_hashes")
        for record in reply[section]:
            rationale = record.get("rationale")
            ok(isinstance(rationale, str) and bool(rationale.strip()), "nonempty_review_rationales")
            prefix = rationale.split("]")[0] + "]"
            observed_groups[section + ":" + prefix].append(record[key])
            if section != "cells":
                ok(record["decision"] in ("approve", "abstain", "reject"), "explicit_decision_vocabulary")
            if section == "contracts":
                transforms = record["approved_transform_ids"]
                ok(isinstance(transforms, list) and all(type(s) is str for s in transforms) and len(transforms) == len(set(transforms)), "contract_transform_approval_schema")
            elif section == "scalar_candidates":
                ok(record["exact_value"] == expected[section][record[key]]["author_candidate"]["value"] and numeric_or_null(record["verified_confidence"]), "exact_scalar_alternatives_and_confidence")
            elif section == "atomic_bundles":
                ok(record["exact_updates"] == expected[section][record[key]]["author_bundle"]["updates"] and type(record["atomicity_confirmed"]) is bool and numeric_or_null(record["verified_confidence"]), "exact_ordered_atomic_bundles_and_confidence")
            else:
                ok(all(numeric_or_null(record[k]) for k in ("original_valid_score", "decision_confidence")), "cell_support_scores")
                ok(all(type(record[k]) is bool for k in ("invalidity_supported", "retract_parent_repair", "retraction_supported")) and isinstance(record["invalidity_rule_ids"], list), "cell_support_typed_schema")
                cell = expected[section][record[key]]
                alternatives = {x["candidate_id"]: x for x in cell["candidates"]}
                choice = record["selected_candidate_id"]
                ok((choice is None and record["selected_value"] in (None, cell["raw_dirty_original"])) or (choice in alternatives and record["selected_value"] == alternatives[choice]["author_candidate"]["value"]), "selected_scalar_id_exact_value")
                ok(not (record["original_valid_score"] is not None and record["original_valid_score"] >= .9 and choice is not None), "valid_original_no_conflicting_scalar_selection")
    ok(dict(observed_groups) == coverage["id_coverage_groups"], "grouped_policy_exact_expansion_coverage")
    counts = {section: len(reply[section]) for section, _ in sections}
    counts.update({"contract_decisions": dict(Counter(r["decision"] for r in reply["contracts"])),
        "candidate_decisions": dict(Counter(r["decision"] for r in reply["scalar_candidates"])),
        "bundle_decisions": dict(Counter(r["decision"] for r in reply["atomic_bundles"])),
        "original_valid": sum(c["original_valid_score"] is not None and c["original_valid_score"] >= .9 for c in reply["cells"]),
        "original_invalid": sum(c["original_valid_score"] is not None and c["original_valid_score"] <= .1 for c in reply["cells"]),
        "original_unresolved": sum(c["original_valid_score"] is None for c in reply["cells"]),
        "selected_scalar_candidates": sum(c["selected_candidate_id"] is not None for c in reply["cells"]),
        "parent_retractions": sum(c["retract_parent_repair"] for c in reply["cells"])})
    ok(counts == coverage["counts"], "review_count_reconciliation")
    counts["coverage_groups"] = len(observed_groups)
    return counts


def provenance(frozen):
    for path, fingerprint in PINS.items():
        check(path, fingerprint)
    author = inventory(AUTHOR / "manifest.json", "files")
    reviewed = inventory(REVIEWER / "manifest.json", "files_sha256")
    applied = inventory(APPLIED / "manifest.json", "output_sha256", "manifest_hash")
    prepared = inventory(PREPARED / "manifest.json", "output_sha256", "manifest_hash")
    exported = inventory(ROOT / "manifest.json", "output_sha256", "export_hash")
    ok(applied["datasets"] == list(a.SIX) and set(prepared["requests"]) == set(a.SIX) and exported["datasets"] == list(a.SIX) and author["datasets"] == list(a.SIX), "all_six_declared_dataset_scope")
    check(PREPARED / "manifest.json", applied["prepared_manifest_sha256"])
    check(APPLIED / "provenance/preparation_manifest.json", applied["prepared_manifest_sha256"])
    ok(prepared["bindings"] == applied["bindings"], "preparation_application_bindings_unchanged")
    for rel, fingerprint in prepared["output_sha256"].items():
        check(APPLIED / rel, fingerprint)
    for kind in ("author_files", "implementation_files"):
        for path, fingerprint in applied["bindings"][kind].items():
            check(path, fingerprint)
    for path, fingerprint in exported["inference_inputs"].items():
        check(path, fingerprint)
    ok(applied["bindings"]["author_manifest_sha256"] == PINS[AUTHOR / "manifest.json"] == reviewed["author_manifest_sha256"], "author_application_reviewer_manifest_binding")
    ok(reviewed["reviewer_id"] == REVIEWER_ID and reviewed["model"] == "gpt-6.1-sol" and reviewed["reasoning_effort"] == "xhigh", "independent_reviewer_model_role")
    for field in ("benchmark_data_accessed", "application_performed", "evaluation_performed"):
        ok(reviewed[field] is False, "reviewer_declared_source_scope")
    coverage = a.read_json(REVIEWER / "grouped_coverage.json")
    check(REVIEWER / "write_review.py", coverage["expansion_script_sha256"])
    ok(coverage["reviewer_id"] == REVIEWER_ID and coverage["designated_by"] == "/root" and coverage["model"] == "gpt-6.1-sol" and coverage["reasoning_effort"] == "xhigh" and coverage["benchmark_data_accessed"] is False and coverage["author_manifest_sha256"] == PINS[AUTHOR / "manifest.json"], "grouped_judgment_provenance")
    ok("No claim" in coverage["authorship"] and "per-cell API" in coverage["authorship"] and "no per-cell API claim" in reviewed["authorship"], "grouped_judgment_no_per_cell_api_claim")
    ok(coverage["source_policy"]["excluded_sources_read"] == [] and coverage["source_policy"]["clean_reference_error_masks_metrics_prior_verdicts_accessed"] is False, "reviewer_attested_input_restrictions")
    validation = a.read_json(REVIEWER / "validation_only.json")
    protocol = a.read_json(AUTHOR / "protocol.json")
    dispatch = a.read_json(AUTHOR / "independent_review_request.json")
    ok(dispatch["author_manifest_sha256"] == PINS[AUTHOR / "manifest.json"] and dispatch["protocol_sha256"] == author["protocol_sha256"] and protocol["model"] == "gpt-6.1-sol" and protocol["reasoning_effort"] == "xhigh", "author_dispatch_protocol_binding")
    aggregate = Counter()
    groups_by_dataset, parent_comparisons, baseline_checks = {}, {}, {}
    for dataset in a.SIX:
        freeze = frozen[NAME, dataset]
        ok(set(freeze["prediction_files"]) == SYSTEMS and freeze["method_metadata"]["exact_applied_prediction_bytes"] is True and freeze["matched_ablations_included"] is False, "fixed_four_variants_and_scope")
        ok(freeze["applied_manifest_sha256"] == PINS[APPLIED / "manifest.json"] and freeze["independent_verifier"] == REVIEWER_ID and freeze["independent_verification_status"] == "complete", "export_applied_reviewer_binding")
        for flag in ("inference_reexecuted", "verification_reexecuted", "official_exact_ec_baseline_present"):
            ok(freeze[flag] is False, "export_scope_flags")
        request_path, reply_path = APPLIED / dataset / "verification_request.json", APPLIED / dataset / "verification.json"
        request, reply = a.read_json(request_path), a.read_json(reply_path)
        provenance_path = APPLIED / dataset / "verification_provenance.json"
        sidecar = a.read_json(provenance_path)
        sealed(request, "request_hash")
        ok(request["request_hash"] == freeze["request_hash"] == prepared["requests"][dataset] == reviewed["request_hashes"][dataset], "six_way_request_identity")
        ok(request["policy"] == EXPECTED_POLICY and request["benchmark_data_accessed"] is False and request["author_scores_are_uncalibrated"] is True and request["parent_is_untrusted_not_labels"] is True, "fixed_policy_and_uncalibrated_support")
        header, dirty = a.raw_table(a.REPO / "vendor/ZeroED/data" / (dataset + "_error-01.csv"))
        ok(request["dirty_sha256"] == freeze["dirty_sha256"] and request["columns"] == header and request["shape"] == [len(dirty), len(header)], "request_dirty_shape_binding")
        for key in ("request_hash", "protocol_sha256", "artifact_schema_sha256"):
            ok(reply[key] == request[key], "review_echoed_hash_bindings")
        ok(reply["author_manifest_sha256"] == PINS[AUTHOR / "manifest.json"] and reply["model"] == "gpt-6.1-sol" and reply["reasoning_effort"] == "xhigh" and reply["benchmark_data_accessed"] is False and reply["default_for_unreviewed"] == "abstain_preserve_parent", "review_response_provenance_policy")
        ok(sidecar["reviewer_id"] == REVIEWER_ID and sidecar["reviewer_role"] == "independent_format_contract_verifier" and sidecar["designated_by"] == "/root" and sidecar["model"] == "gpt-6.1-sol" and sidecar["reasoning_effort"] == "xhigh" and sidecar["benchmark_data_accessed"] is False and sidecar["request_hash"] == request["request_hash"], "reviewer_sidecar_model_and_request")
        check(reply_path, sidecar["verification_file_sha256"])
        check(REVIEWER / "grouped_coverage.json", sidecar["grouped_coverage_sha256"])
        check(reply_path, reviewed["files_sha256"][dataset + "/verification.json"])
        check(provenance_path, reviewed["files_sha256"][dataset + "/provenance.json"])
        group = coverage["datasets"][dataset]
        check(request_path, group["request_file_sha256"])
        check(reply_path, group["verification_file_sha256"])
        counts = review_structure(dataset, request, reply, group)
        groups_by_dataset[dataset] = counts
        aggregate.update({k: v for k, v in counts.items() if type(v) is int})
        vr = validation["results"][dataset]
        ok(vr["structural_validation"] == "passed" and vr["review_sha256"] == a.hashfile(reply_path) and vr["provenance_sha256"] == a.hashfile(provenance_path), "authenticated_structural_validation_receipt")
        parent_path = a.REPO / "results/source_v2/predictions" / dataset / "zeroclean_loop.json"
        check(parent_path, request["bindings"]["parent_file"]["sha256"])
        check(APPLIED / dataset / "provenance/parent_prediction.json", a.hashfile(parent_path))
        check(ROOT / "predictions" / dataset / "source_v2_parent.json", a.hashfile(parent_path))
        check(ROOT / "predictions" / dataset / "zeroclean_loop.json", a.hashfile(APPLIED / dataset / "prediction.json"))
        parent = a.read_json(parent_path)
        pred = a.read_json(ROOT / "predictions" / dataset / "zeroclean_loop.json")
        ok(request["parent_snapshot"] == {"probabilities": parent["probabilities"], "repairs": parent["repairs"]} and request["parent_snapshot_sha256"] == a.canonical(request["parent_snapshot"]), "parent_metadata_free_snapshot_exact")
        ok(pred["metadata"]["request_hash"] == request["request_hash"] and pred["metadata"]["review_sha256"] == a.canonical(reply) and pred["metadata"]["parent_snapshot_sha256"] == request["parent_snapshot_sha256"] and pred["metadata"]["independent_verifier_provenance"] == sidecar and pred["metadata"]["benchmark_data_accessed"] is False, "applied_prediction_request_reply_provenance")
        # All comparisons below are unlabeled identities/counts only.
        parent_comparisons[dataset] = graph.exact_state_difference(pred, parent)
        baseline_root = a.REPO / "results/source_baselines_v1"
        check(baseline_root / "frozen" / (dataset + ".json"), freeze["baseline_source_frozen_sha256"])
        source_freeze = a.read_json(baseline_root / "frozen" / (dataset + ".json"))
        for name in ("zeroed.json", "zeroed_zeroec_reconstruction.json"):
            check(ROOT / "predictions" / dataset / name, source_freeze["prediction_files"][name])
            check(baseline_root / "predictions" / dataset / name, source_freeze["prediction_files"][name])
        ec = a.read_json(ROOT / "predictions" / dataset / "zeroed_zeroec_reconstruction.json")["metadata"]
        ok(ec["strict_zero_shot"] is True and ec["reproduced_upstream"] is False and ec["pending_cells"] == 0 and ec["detected_cells"] == ec["completed_cells"] and ec["prompt_provenance"]["historical_identity_established"] is False and ec["prompt_provenance"]["uses_general_examples"] is False, "qualified_source_ec_fidelity_and_completion")
        baseline_checks[dataset] = {"both_baselines_byte_exact": True, "parent_byte_exact": True, "official_exact_ec_present": False, "fixed_example_count": 0}
        # Validate raw bindings and authored identities, without reapplying gates.
        for cell in request["cells"]:
            i, j = cell["row"], cell["column_index"]
            ok(cell["raw_dirty_original"] == dirty[i][j] and cell["column"] == header[j], "request_raw_original_binding")
            for candidate in cell["candidates"]:
                authored = candidate["author_candidate"]
                expected_id = compact({"dataset": dataset, "row": i, "column": header[j], "source_value": dirty[i][j], **{k: authored[k] for k in ("contract_id", "transform_id", "value")}})
                ok(candidate["candidate_id"] == expected_id and candidate["author_candidate_sha256"] == a.canonical(authored), "exact_authored_scalar_identity")
        for bundle in request["atomic_bundles"]:
            authored = bundle["author_bundle"]
            expected_id = compact({"dataset": dataset, **{k: authored[k] for k in ("row", "contract_id", "transform_id", "source_values", "updates", "application_mode")}})
            ok(bundle["bundle_id"] == expected_id and bundle["author_bundle_sha256"] == a.canonical(authored), "exact_authored_bundle_identity")
    for key, value in reviewed["total_counts"].items():
        ok(aggregate[key] == value, "all_six_reviewer_total_reconciliation")
    for category, value in ERRORS.items():
        a.compare(0, value, "format_structure/" + category)
    return {"manifest_sha256": {a.short(p): h for p, h in PINS.items()}, "manifest_file_counts": {"author": len(author["files"]), "reviewer": len(reviewed["files_sha256"]), "prepared": len(prepared["output_sha256"]), "applied": len(applied["output_sha256"]), "exported": len(exported["output_sha256"])},
        "reviewer_id": REVIEWER_ID, "model": "gpt-6.1-sol", "reasoning_effort": "xhigh", "review_counts_by_dataset": groups_by_dataset, "review_counts_all_six": dict(aggregate),
        "grouped_authorship_qualification": "Independent model-agent grouped judgments with deterministic exact-ID expansion. Coverage, byte hashes and designated-role sidecars verified; no claim of separate per-cell model/API calls, correctness calibration, or cryptographically proven human/agent identity.",
        "parent_comparison_aggregates": parent_comparisons, "source_baseline_checks": baseline_checks,
        "structural_discrepancy_counts": dict(ERRORS), "gate_reconciliation_scope": "This evaluator checks immutable identities, complete typed review coverage, raw bindings and exact exported parent/baseline comparisons. Actual application gate/rule replay is delegated to the separate label-blind adapter audit; no adapter, rule engine, reapplication or inference is executed here.",
        "original_invalidity_is_not_automatic_repair": "Frozen protocol requires separate approved exact scalar/bundle selection, approved contracts/transforms, parent preconditions, delta gates and whole-row guard; invalidity evidence alone provides no repair authority.",
        "matched_component_ablations_present": False, "source_ec_qualification": "Author-prompt reconstruction with unproven historical paper-prompt identity and zero fixed examples."}


def main():
    started = datetime.now(timezone.utc).isoformat()
    inputs = [ROOT / "summary_metrics.json", *[ROOT / "frozen" / (d + ".json") for d in a.SIX], ROOT / "manifest.json", PREPARED / "manifest.json", *PINS]
    snapshots = {a.short(p): t.snapshot_exact(p, OWNED / p.relative_to(a.REPO)) for p in inputs}
    for module in (a, t, graph, data):
        a.hashfile(Path(module.__file__))
    a.hashfile(Path(__file__))
    data.ROOTS = {NAME: a.SIX}
    data.SNAPSHOT_DIR = OWNED / "results"
    data.QUARANTINE = QUARANTINE
    selfchecks = a.semantic_checks()
    frozen, summaries, counts = data.preflight()
    checked = provenance(frozen)
    prelabel = len(a.DISCREPANCIES)
    print(json.dumps({"all_prelabel_integrity_checks_complete": True, "discrepancy_count": prelabel}), flush=True)
    assert not any("format_bound_input/" in d["location"] or "frozen_hash" in d["location"] or "/inference/" in d["location"] or "content_hash" in d["location"] for d in a.DISCREPANCIES), "Bound source mismatch; reference access blocked"
    experiments, dimensions = data.evaluate_roots(frozen, summaries)
    experiments[NAME]["scope"] = "All-six format-contract overlay with independently grouped review; immutable source_v2 parent and qualified source baselines. Exact-string benchmark evaluation only. Matched component ablations absent."
    experiments[NAME]["six_dataset_complete"] = True
    experiments[NAME]["primary_six_dataset_selection_changed_by_this_audit"] = False
    changed = []
    for path, expected in list(a.SNAPSHOTS.items()):
        actual = a.hashfile(path, refresh=True)
        if actual != expected:
            changed.append(a.short(path))
            a.compare(expected, actual, "post_audit_hash/" + a.short(path))
    report = {"audit_schema": "independent-format-overlay-v1", "evaluation_only": True, "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "historical_input_snapshots": snapshots, "summary_and_freeze_snapshots_precede_reference_read": True,
        "method": {"implementation": "Standard library and evaluator-owned helpers; no source adapter, exporter, model, rule engine or inference execution", "reference_policy": "Exact raw CSV strings; no normalization or null coercion", "detection_policy": "Stored source mask precedence; otherwise fixed declared threshold/operator, default >= 0.5", "repair_policy": "Dirty changed cell exactly equals reference for TP; wrong dirty edit is both FP and FN; unchanged clean earns no TP", "examples_quarantined": a.short(QUARANTINE), "no_inference_or_tuning": True},
        "semantic_self_checks": selfchecks, "prelabel_integrity": {"all_declared_hashes_verified_before_reference_read": True, "discrepancy_count": prelabel, "root_counts": counts, "hash_checks": dict(a.HASH_CHECKS)},
        "format_provenance_and_comparisons": checked, "dimensions": dimensions, "experiments": experiments,
        "benchmark_denominator": {"datasets": len(a.SIX), "cells": sum(v["cells"] for v in dimensions.values()),
            "dirty_cells": sum(experiments[NAME]["metrics"][d]["zeroclean_loop"]["detection"]["actual"] for d in a.SIX)},
        "read_input_integrity": {"unique_files_rehashed_after_audit": len(a.SNAPSHOTS), "changed_files": changed, "all_stable": not changed},
        "discrepancy_count": len(a.DISCREPANCIES), "discrepancies": a.DISCREPANCIES,
        "input_sha256": {a.short(p): h for p, h in sorted(a.SNAPSHOTS.items(), key=lambda x: str(x[0]))},
        "auditor_script_sha256": a.hashfile(Path(__file__), refresh=True), "independent_helper_sha256": {a.short(Path(m.__file__)): a.hashfile(Path(m.__file__), refresh=True) for m in (a, t, graph, data)}}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=lambda x: sorted(x) if isinstance(x, set) else str(x)) + "\n")
    print(json.dumps({"output": a.short(OUTPUT), "discrepancy_count": len(a.DISCREPANCIES), "prediction_runs": counts[NAME]["prediction_files"], "all_inputs_stable": not changed}), flush=True)


if __name__ == "__main__":
    main()
