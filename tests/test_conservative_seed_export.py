"""Invented-only interface/export tests. No benchmark or evaluator is opened."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import struct

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_conservative_seed.py"
SPEC = importlib.util.spec_from_file_location("conservative_seed_export_tests", SCRIPT)
e = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = e
SPEC.loader.exec_module(e)


def put(root, relative, body):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(e.json_bytes(body) if isinstance(body, (dict, list)) else body)
    return path


def hashed(body, key):
    body = deepcopy(body)
    body[key] = e.digest(body)
    return body


def unknown():
    return {"disposition": "unknown", "confidence": .8, "exact_value_score": .5,
            "explanation": "Invented-only unresolved evidence."}


def effect():
    return {"delta_features": [0., 0., 0., 0.], "introduced_violation": False, "runtime_error": False,
            "before_rule_results": [], "after_rule_results": [], "raw_value_count": 0,
            "original_raw_value_count": 2, "raw_conditional_evidence_before": [], "raw_conditional_evidence_after": []}


@pytest.fixture
def completed(tmp_path):
    root = tmp_path / "invented-repository"
    root.mkdir()
    columns = ["value", "src"]
    rows = [["A", "invented-source-one"], ["A", "invented-source-two"]]
    probabilities = [[.5, -0.0], [.9, .05]]
    original_bits = [[False, False], [True, False]]
    sources = {}
    for relative in e.LOCKED_SOURCES:
        put(root, relative, f"Invented source fixture only: {relative}\n".encode())
    for relative in e.INTERFACE_FILES:
        put(root, relative, f"Distinct invented interface fixture only: {relative}\n".encode())
    for name in e.DATASETS:
        frozen_path, ed_path, ec_path, pass2_path, cts_path, dirty_path = e.source_paths(name)
        put(root, dirty_path, b"value,src\nA,invented-source-one\nA,invented-source-two\n")
        dirty_sha = e.sha256(root / dirty_path)
        shared = {"uses_ground_truth": False, "decision_threshold": .5, "decision_threshold_operator": ">",
                  "upstream_binary_decisions": original_bits, "strict_zero_shot": True}
        ed = {"probabilities": probabilities, "repairs": [],
              "metadata": dict(shared, baseline="zeroed_upstream", fidelity="upstream_function_extraction",
                               input_fingerprint=dirty_sha, reproduced_upstream=True)}
        ec = {"probabilities": probabilities, "repairs": [{"row": 1, "column": 0, "value": "B"}],
              "metadata": dict(shared, baseline="zeroed_zeroec_reconstruction", fidelity="author_source_reconstruction",
                               dirty_fingerprint=dirty_sha, reproduced_upstream=False, pending_cells=0,
                               completed_cells=1, detected_cells=1,
                               prompt_provenance={"historical_identity_established": False}, null_sentinel_policy="Invented raw sentinel contract")}
        pass2 = {"dirty_sha256": dirty_sha, "model": e.MODEL, "reasoning_effort": e.EFFORT, "rules": [], "dependencies": []}
        proposals = {"dataset": name, "dirty_sha256": dirty_sha, "model": e.MODEL, "reasoning_effort": e.EFFORT,
                     "retrieval_method": "zeroec_source_cts", "cells": [
                         {"row": i, "column": "value", "candidates": [{"value": "C", "score": .7, "explanation": "Fallible invented peer claim."}],
                          "retrieved_row_indices": [1-i], "conditional_profiles": []} for i in (0, 1)]}
        for relative, body in ((ed_path, ed), (ec_path, ec), (pass2_path, pass2), (cts_path, proposals)):
            put(root, relative, body)
        frozen = {"dataset": name, "dirty_sha256": dirty_sha, "frozen_before_evaluation": True,
                  "ground_truth_access": False, "export_schema": "source-baselines-only-v1",
                  "official_exact_ec_baseline_present": False,
                  "prediction_files": {"zeroed.json": e.sha256(root / ed_path), "zeroed_zeroec_reconstruction.json": e.sha256(root / ec_path)}}
        put(root, frozen_path, hashed(frozen, "export_hash"))
        sources[name] = ed, ec, proposals, dirty_sha
    config = {"protocol": e.PROTOCOL, "external_rounds": 3, "candidate_cap": 3,
              "detector_operator": ">", "detector_threshold": .5, "model": e.MODEL, "reasoning_effort": e.EFFORT,
              "reretrieval": False, "worker_policy": "one_fresh_fork_none_worker_per_dataset_arm_round",
              "source_cts_proposal_byte_hashes": {name: e.sha256(root / e.source_paths(name)[4]) for name in e.DATASETS}}
    lock = hashed({"protocol": e.PROTOCOL, "config": config,
                   "source_files": {str(root / path): e.sha256(root / path) for path in e.LOCKED_SOURCES},
                   "coordinated_input_files": {str(root / path): e.sha256(root / path) for name in e.DATASETS for path in e.source_paths(name)},
                   "ground_truth_access": False, "status": "frozen_before_new_predictions"}, "protocol_hash")
    put(root, e.LOCK_PATH, lock)
    authority = e.Authority(root, lock["protocol_hash"])
    prepared_entries, directories = [], {}
    for name in e.DATASETS:
        ed, ec, proposals, dirty_sha = sources[name]
        cells = []
        for i in (0, 1):
            candidates = []
            for value, origin in (([("B", "naive_source_ec")] if i == 1 else []) + [("C", "fixed_source_v2_cts")]):
                candidates.append({"candidate_id": e.candidate_id(i, 0, value), "value": value,
                                   "private_origin": origin, "private_origins": [origin],
                                   "evidence_explanation": "Untrusted proposed alternative; assess independently" if value == "B" else "Fallible invented peer claim.",
                                   "raw_effect": effect()})
            cells.append({"cell_id": f"{i}:0", "row": i, "column_index": 0, "column": "value", "original": "A",
                          "tuple": dict(zip(columns, rows[i])), "retrieved_dirty_peers": [{"row": 1-i, "tuple": dict(zip(columns, rows[1-i])), "trust": "raw_dirty_observation_not_clean_reference"}],
                          "raw_conditional_profiles": [], "candidates": candidates,
                          "private_seed_candidate_id": e.candidate_id(i, 0, "B") if i == 1 else None})
        plan = {"protocol": e.PROTOCOL, "dataset": name, "dirty_sha256": dirty_sha,
                "columns": columns, "shape": [2, 2], "raw_probabilities": probabilities,
                "original_source_binary_decisions": [[int(bit) for bit in row] for row in original_bits],
                "cells": cells, "candidate_universe_sha256": e.digest([{key: c[key] for key in ("cell_id", "row", "column_index", "original", "candidates")} for c in cells]),
                "rules": [], "rule_validation": [], "dependencies": [], "raw_statistics_sha256": e.digest([name, "invented raw counts"]),
                "config": config, "source_fidelity": {"ed": ed["metadata"]["fidelity"], "ec": ec["metadata"]["fidelity"],
                    "ec_prompt_provenance": ec["metadata"]["prompt_provenance"], "null_sentinel_policy": ec["metadata"]["null_sentinel_policy"]},
                "identity_source_repairs": [], "ground_truth_access": False}
        plan = hashed(plan, "plan_hash")
        base = f"{e.EXPERIMENT}/runs/{name}/{plan['plan_hash']}"
        put(root, f"{base}/plan.json", plan)
        manifest = hashed({"protocol": e.PROTOCOL, "dataset": name, "plan_hash": plan["plan_hash"],
                           "protocol_hash": lock["protocol_hash"], "protocol_lock_path": str(root / e.LOCK_PATH),
                           "input_files": {str(root / path): e.sha256(root / path) for path in e.source_paths(name) + [e.LOCK_PATH]},
                           "plan_sha256": e.sha256(root / f"{base}/plan.json"), "arms": list(e.ARMS), "ground_truth_access": False}, "manifest_hash")
        put(root, f"{base}/manifest.json", manifest)
        for arm in e.ARMS:
            directory = f"{base}/{arm}"
            directories[name, arm] = directory
            active = {"1:0": e.candidate_id(1, 0, "B")} if arm == "seeded_carry" else {}
            statuses = {key: {"detector_bit": i == 1, "latest_decisive_original_request": None,
                              "emission_status": "emitted" if key in active else "inactive",
                              "support_status": "carried_unresolved" if key in active else "not_active"}
                        for i, key in enumerate(("0:0", "1:0"))}
            guard = {"row": 1, "round": 0, "reason": "seed_initialization", "raw_rule_results": [],
                     "proposed_rule_results": [], "blocked": False, "vacuous": True, "proposed_values": {"1:0": "B"}}
            state = hashed({"protocol": e.PROTOCOL, "dataset": name, "arm": arm, "external_round": 0,
                            "plan_hash": plan["plan_hash"], "probabilities": probabilities, "active": active, "cells": statuses,
                            "events": [{"round": 0, "cell_id": "1:0", "event": "noisy_seed_initialized", "candidate_id": active["1:0"]}] if active else [],
                            "guard_records": [guard] if active else []}, "state_hash")
            put(root, f"{directory}/initial_state.json", state)
            history = []
            for number in range(1, 4):
                round_dir = f"{directory}/rounds/{number:04d}"
                projected = []
                for cell in cells:
                    item = {field: deepcopy(cell[field]) for field in ("cell_id", "row", "column_index", "column", "original", "tuple", "retrieved_dirty_peers", "raw_conditional_profiles")}
                    tentative = deepcopy(cell["tuple"])
                    if cell["cell_id"] in active:
                        tentative["value"] = "B"
                    item.update(tentative_tuple=tentative, counterfactual_before_tuple=deepcopy(cell["tuple"]),
                                active_candidate_id=active.get(cell["cell_id"]), tentative_context_trust="untrusted_model_proposals")
                    item["candidates"] = [{key: deepcopy(c[key]) for key in ("candidate_id", "value", "evidence_explanation", "raw_effect")} | {"effect": effect()} for c in cell["candidates"]]
                    random.Random(e.digest([42, name, cell["cell_id"], number])).shuffle(item["candidates"])
                    projected.append(item)
                request = hashed({"protocol": e.PROTOCOL, "dataset": name, "arm": arm, "external_round": number,
                                  "model": e.MODEL, "reasoning_effort": e.EFFORT, "dirty_sha256": dirty_sha,
                                  "plan_hash": plan["plan_hash"], "candidate_universe_sha256": plan["candidate_universe_sha256"],
                                  "raw_statistics_sha256": plan["raw_statistics_sha256"], "parent_state_hash": state["state_hash"],
                                  "cells": projected, "rules": [], "config": config,
                                  "delta_dimensions": ["rule_satisfaction", "fd_support", "nmi_vicinity", "domain_support"],
                                  "batch_cells": [["0:0", "1:0"]], "worker_policy": config["worker_policy"]}, "request_hash")
                put(root, f"{round_dir}/request.json", request)
                evidence = {field: request[field] for field in ("protocol", "dataset", "arm", "external_round", "dirty_sha256", "request_hash", "rules", "delta_dimensions")}
                evidence["cells"] = projected
                batch = hashed({"batch": 0, "cell_ids": ["0:0", "1:0"], "prompt": "Fixed invented policy. Exact batch evidence follows:\n" + json.dumps(evidence, ensure_ascii=False, allow_nan=False),
                                "request_hash": request["request_hash"], "model": e.MODEL, "reasoning_effort": e.EFFORT}, "batch_hash")
                batch_path = f"{round_dir}/batches/0000.json"
                put(root, batch_path, batch)
                worker = {"protocol": e.PROTOCOL, "dataset": name, "arm": arm, "external_round": number,
                          "model": e.MODEL, "reasoning_effort": e.EFFORT, "fork_turns": "none",
                          "request_path": str(root / f"{round_dir}/request.json"), "batch_paths": [str(root / batch_path)],
                          "response_path": str(root / f"{round_dir}/verification.json"),
                          "contract_path": str(root / f"{e.EXPERIMENT}/worker_contract.md"), "policy": config["worker_policy"],
                          "no_previous_verdicts": True, "same_model_error_correlation": True, "dispatch_by_runner": False,
                          "dispatch_policy": "root_coordination_required_before_dispatch"}
                put(root, f"{round_dir}/worker_request.json", worker)
                response = {key: request[key] for key in ("protocol", "dataset", "arm", "external_round", "request_hash", "dirty_sha256", "candidate_universe_sha256", "model", "reasoning_effort")}
                response.update(fresh_session_id=f"invented-{name}-{arm}-{number}", fork_turns="none", reviewed_batch_hashes=[batch["batch_hash"]], joint_plans=[], cells=[])
                for cell in projected:
                    response["cells"].append({"cell_id": cell["cell_id"], "original": cell["original"], "original_assessment": unknown(),
                                               "selection": {"kind": "abstain", "value": None, "confidence": .8},
                                               "candidates": [{"candidate_id": c["candidate_id"], "value": c["value"], "assessment": unknown()} for c in cell["candidates"]], "joint_plan_id": None})
                response_path = f"{round_dir}/verification.json"
                put(root, response_path, response)
                next_state = deepcopy(state)
                next_state.pop("state_hash")
                next_state["external_round"] = number
                for item in response["cells"]:
                    key = item["cell_id"]
                    next_state["events"].append({"round": number, "cell_id": key, "request_hash": request["request_hash"],
                        "prior_candidate_id": active.get(key), "original_assessment": item["original_assessment"],
                        "candidate_assessments": item["candidates"], "selection": item["selection"],
                        "event": "carried_epistemic_abstention_or_insufficient_authorization", "next_candidate_id": active.get(key)})
                if active:
                    next_state["guard_records"].append(dict(guard, round=number, reason="post_transition"))
                next_state = hashed(next_state, "state_hash")
                put(root, f"{round_dir}/state.json", next_state)
                raw = {"probabilities": probabilities, "repairs": [{"row": 1, "column": 0, "value": "B"}] if active else [],
                       "metadata": {"system": e.PROTOCOL, "dataset": name, "arm": arm, "dirty_sha256": dirty_sha,
                                    "plan_hash": plan["plan_hash"], "protocol_hash": lock["protocol_hash"], "state_hash": next_state["state_hash"],
                                    "external_verification_rounds": number, "decision_threshold": .5, "decision_threshold_operator": ">",
                                    "probabilities_semantics": "Uncalibrated invented scores", "upstream_binary_decisions": [[int(x) for x in row] for row in original_bits],
                                    "upstream_binary_decisions_semantics": "Experimental legacy field; NOT original upstream detector bits",
                                    "original_source_binary_decisions": plan["original_source_binary_decisions"], "source_fidelity": plan["source_fidelity"],
                                    "repair_statuses": next_state["cells"], "raw_statistics_sha256": plan["raw_statistics_sha256"],
                                    "same_model_error_correlation": True, "ground_truth_access": False, "frozen_before_evaluation": True}}
                put(root, f"{round_dir}/prediction.json", raw)
                inputs = [f"{round_dir}/request.json", response_path, f"{directory}/initial_state.json", f"{base}/manifest.json", batch_path]
                if number > 1:
                    inputs.append(f"{directory}/rounds/{number-1:04d}/record.json")
                record = hashed({"protocol": e.PROTOCOL, "external_round": number, "request_hash": request["request_hash"],
                                 "parent_state_hash": state["state_hash"], "state_hash": next_state["state_hash"],
                                 "input_files": {str(root / path): e.sha256(root / path) for path in inputs},
                                 "state_sha256": e.sha256(root / f"{round_dir}/state.json"), "prediction_sha256": e.sha256(root / f"{round_dir}/prediction.json"),
                                 "fresh_session_id": response["fresh_session_id"], "response_format": "direct", "ground_truth_access": False,
                                 "frozen_before_evaluation": True}, "record_hash")
                put(root, f"{round_dir}/record.json", record)
                owner = {"fresh_session_id": response["fresh_session_id"], "request_hash": request["request_hash"],
                         "round_directory": str(root / round_dir), "response_sha256": e.sha256(root / response_path)}
                put(root, f"{e.EXPERIMENT}/runs/worker_sessions/{e.digest(response['fresh_session_id'])}.json", owner)
                put(root, f"{e.EXPERIMENT}/runs/round_sessions/{e.digest(str(root / round_dir))}.json", owner)
                history.append({"round": number, "record_sha256": e.sha256(root / f"{round_dir}/record.json"), "request_hash": request["request_hash"],
                                "state_hash": next_state["state_hash"], "fresh_session_id": response["fresh_session_id"]})
                if number == 1:
                    prepared_entries.append({"dataset": name, "arm": arm, "run_directory": str(root / directory),
                        "request_sha256": e.sha256(root / f"{round_dir}/request.json"),
                        "response_binding": {"request_hash": request["request_hash"], "candidate_universe_sha256": plan["candidate_universe_sha256"]}})
                state = next_state
            put(root, f"{directory}/state.json", state)
            put(root, f"{directory}/prediction.json", raw)
            frozen = hashed({"protocol": e.PROTOCOL, "dataset": name, "arm": arm, "external_rounds": 3, "stop_reason": "fixed_three_rounds",
                             "history": history, "manifest_hash": manifest["manifest_hash"], "protocol_hash": lock["protocol_hash"],
                             "state_sha256": e.sha256(root / f"{directory}/state.json"), "prediction_sha256": e.sha256(root / f"{directory}/prediction.json"),
                             "ground_truth_access": False, "frozen_before_evaluation": True, "export_ready": True}, "frozen_hash")
            put(root, f"{directory}/frozen.json", frozen)
    put(root, e.PREPARATION_PATH, hashed({"protocol": e.PROTOCOL, "protocol_hash": lock["protocol_hash"], "entries": prepared_entries}, "index_hash"))
    put(root, e.NOTE_PATH, b"Invented-only interface note; no evaluation.\n")
    return authority, directories


def test_complete_builder_and_typed_export_reconcile_exact_raw_bytes(completed):
    authority, directories = completed
    receipt = e.build_completion_index(authority)
    assert receipt["completed_arms"] == 12 and receipt["completed_round_records"] == 36
    supplied = receipt["completion_index_sha256"]
    result = e.export(authority, supplied)
    assert result["status"] == "exported" and result["evaluation_performed"] is False
    output = authority.root / "results/conservative_seed_v1"
    for name in e.DATASETS:
        for method in e.SOURCE_NAMES:
            expected = (authority.root / f"results/source_baselines_v1/predictions/{name}/{method}.json").read_bytes()
            assert (output / f"predictions/{name}/{method}.json").read_bytes() == expected
            assert (output / f"raw/source_baselines/{name}/{method}.json").read_bytes() == expected
        for arm in e.ARMS:
            raw_path = authority.root / f"{directories[name, arm]}/prediction.json"
            raw = e.read_json(raw_path)
            normalized = e.read_json(output / f"predictions/{name}/{arm}.json")
            assert (output / f"raw/predictions/{name}/{arm}.json").read_bytes() == raw_path.read_bytes()
            e.reconcile_normalization(raw, normalized, (2, 2), e.sha256(raw_path), f"raw/predictions/{name}/{arm}.json", supplied)
            assert all(type(bit) is bool for row in normalized["metadata"]["upstream_binary_decisions"] for bit in row)
            assert normalized["metadata"]["upstream_binary_decisions"][0][0] is False  # Exact .5 tie.
            assert e.float64_bytes(raw["probabilities"]) == e.float64_bytes(normalized["probabilities"])
            assert e.float64_bytes(normalized["probabilities"])[8:16] == struct.pack("<d", -0.0)
            assert normalized["repairs"] == raw["repairs"]
            assert normalized["metadata"]["repair_statuses"]["snapshot_sha256"] == e.sha256(raw_path)
            assert (output / f"provenance/states/{name}/{arm}/rounds/0003/state.json").read_bytes() == (authority.root / f"{directories[name, arm]}/rounds/0003/state.json").read_bytes()
    before = {path: e.sha256(path) for path in output.rglob("*") if path.is_file()}
    assert e.export(authority, supplied)["status"] == "already_exported"
    assert {path: e.sha256(path) for path in before} == before


@pytest.mark.parametrize("bad", [True, False, 0.0, 1.0, "0", "1", None, 2, -1])
def test_raw_mask_types_fail_closed(bad):
    raw = {"probabilities": [[.5]], "repairs": [], "metadata": {"decision_threshold": .5, "decision_threshold_operator": ">", "upstream_binary_decisions": [[bad]], "repair_statuses": {}}}
    with pytest.raises(ValueError):
        e.normalize_prediction(raw, shape=(1, 1), raw_sha="a"*64, snapshot_path="raw.json", completion_sha="b"*64)


def test_encoding_only_bridge_preserves_empty_string_and_detects_probability_or_mask_changes():
    raw = {"probabilities": [[.5, .5000000000000001]], "repairs": [{"row": 0, "column": 1, "value": ""}],
           "metadata": {"decision_threshold": .5, "decision_threshold_operator": ">", "upstream_binary_decisions": [[0, 1]], "repair_statuses": {"0:1": {"status": "invented"}}}}
    normalized = e.normalize_prediction(raw, shape=(1, 2), raw_sha="a"*64, snapshot_path="raw.json", completion_sha="b"*64)
    assert normalized["repairs"][0]["value"] == ""
    bad = deepcopy(normalized)
    bad["probabilities"][0][1] = .9
    with pytest.raises(ValueError, match="float64"):
        e.reconcile_normalization(raw, bad, (1, 2), "a"*64, "raw.json", "b"*64)
    bad = deepcopy(raw)
    bad["metadata"]["upstream_binary_decisions"][0][0] = 1
    with pytest.raises(ValueError, match="strict"):
        e.normalize_prediction(bad, shape=(1, 2), raw_sha="a"*64, snapshot_path="raw.json", completion_sha="b"*64)
    bad = deepcopy(normalized)
    bad["metadata"]["repair_statuses"]["field_path"] = ["metadata", "some_other_field"]
    with pytest.raises(ValueError, match="Trace pointer"):
        e.reconcile_normalization(raw, bad, (1, 2), "a"*64, "raw.json", "b"*64)


@pytest.mark.parametrize("change", ["missing_arm", "missing_round", "bad_mask", "tamper_record", "tamper_request", "tamper_session", "source_input", "path_escape"])
def test_builder_fails_closed_before_any_index_or_export(completed, change):
    authority, directories = completed
    directory = directories["hospital", "seeded_carry"]
    if change == "missing_arm":
        (authority.root / f"{directory}/frozen.json").unlink()
    elif change == "missing_round":
        (authority.root / f"{directory}/rounds/0002/verification.json").unlink()
    elif change == "bad_mask":
        path = authority.root / f"{directory}/prediction.json"
        body = e.read_json(path);body["metadata"]["upstream_binary_decisions"][0][0] = 1;put(authority.root, path.relative_to(authority.root).as_posix(), body)
    elif change == "tamper_record":
        path = authority.root / f"{directory}/rounds/0001/record.json"
        body = e.read_json(path);body.pop("record_hash");body["parent_state_hash"] = "a"*64;put(authority.root, path.relative_to(authority.root).as_posix(), hashed(body,"record_hash"))
    elif change == "tamper_request":
        path = authority.root / f"{directory}/rounds/0001/request.json"
        body = e.read_json(path);body.pop("request_hash");body["candidate_universe_sha256"] = "a"*64;put(authority.root, path.relative_to(authority.root).as_posix(), hashed(body,"request_hash"))
    elif change == "tamper_session":
        path = authority.root / f"{e.EXPERIMENT}/runs/worker_sessions/{e.digest('invented-hospital-seeded_carry-1')}.json"
        body = e.read_json(path);body["response_sha256"] = "a"*64;put(authority.root, path.relative_to(authority.root).as_posix(), body)
    elif change == "source_input":
        path = authority.root / e.source_paths("hospital")[3]
        path.write_text('{"altered":true}')
    else:
        path = authority.root / e.PREPARATION_PATH
        body = e.read_json(path);body.pop("index_hash");body["entries"][0]["run_directory"] = str(authority.root / "../outside");put(authority.root,e.PREPARATION_PATH,hashed(body,"index_hash"))
    with pytest.raises((ValueError, KeyError)):
        e.build_completion_index(authority)
    assert not (authority.root / e.INDEX_PATH).exists()
    assert not (authority.root / "results/conservative_seed_v1").exists()


def test_root_sha_is_required_and_flat_map_coverage_is_exact(completed):
    authority, _ = completed
    receipt = e.build_completion_index(authority)
    with pytest.raises(ValueError, match="Root-supplied"):
        e.export(authority, "a"*64)
    path = authority.root / e.INDEX_PATH
    body = e.read_json(path);body.pop("index_hash");body["files"].pop(next(iter(body["files"])))
    put(authority.root,e.INDEX_PATH,hashed(body,"index_hash"))
    with pytest.raises(ValueError, match="byte binding|file map"):
        e.validate_completion_index(authority,e.sha256(path))
    assert not (authority.root / "results/conservative_seed_v1").exists()


def test_existing_differing_output_and_input_symlink_are_rejected(completed):
    authority, _ = completed
    receipt = e.build_completion_index(authority)
    out = authority.root / "results/conservative_seed_v1"
    out.mkdir();(out/"unrelated.txt").write_text("Preserve me")
    with pytest.raises(ValueError, match="refusing overwrite"):
        e.export(authority, receipt["completion_index_sha256"])
    assert (out/"unrelated.txt").read_text() == "Preserve me"
    source = authority.root / e.source_paths("hospital")[5]
    target = source.with_name("invented-target.csv");target.write_bytes(source.read_bytes());source.unlink();source.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        e.validate_completion_index(authority, receipt["completion_index_sha256"])


def test_no_evaluator_or_inference_imports_and_no_real_export_on_import():
    import ast
    tree = ast.parse(SCRIPT.read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom): modules.append(node.module or "")
        elif isinstance(node,ast.Import): modules.extend(item.name for item in node.names)
    assert not any(module.startswith("zeroclean") or module in ("evaluate_e2e","numpy","sklearn") for module in modules)
    for relative in (".", "../escape", "results/../escape", "/absolute", "a\\b"):
        with pytest.raises(ValueError): e.Authority(Path('/synthetic-only')).path(relative)


def test_atomic_index_write_cleans_interrupted_temporary_file_and_never_overwrites(tmp_path, monkeypatch):
    path = tmp_path / "completion_index.json"
    with monkeypatch.context() as patch:
        def interrupted_fsync(_):
            raise KeyboardInterrupt("Invented interruption before publication")
        patch.setattr(e.os, "fsync", interrupted_fsync)
        with pytest.raises(KeyboardInterrupt):
            e.write_immutable(path, b"complete invented index\n")
    assert list(tmp_path.iterdir()) == []
    e.write_immutable(path, b"complete invented index\n")
    e.write_immutable(path, b"complete invented index\n")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        e.write_immutable(path, b"different invented index\n")
    assert path.read_bytes() == b"complete invented index\n"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("winner", [b"same", b"different"])
def test_atomic_index_publish_race_preserves_the_first_complete_writer(tmp_path, monkeypatch, winner):
    path = tmp_path / "completion_index.json"
    original_link = e.os.link
    def racing_link(source, destination):
        Path(destination).write_bytes(winner)
        original_link(source, destination)
    monkeypatch.setattr(e.os, "link", racing_link)
    if winner == b"same":
        e.write_immutable(path, b"same")
    else:
        with pytest.raises(ValueError, match="Refusing to overwrite"):
            e.write_immutable(path, b"same")
    assert path.read_bytes() == winner
    assert list(tmp_path.iterdir()) == [path]


def test_builder_rechecks_all_bound_bytes_after_writing_before_publication(completed, monkeypatch):
    authority, _ = completed
    original_fsync = e.os.fsync
    def changed_after_validation(descriptor):
        original_fsync(descriptor)
        (authority.root / e.INTERFACE_FILES[0]).write_bytes(b"Invented interface changed before publication\n")
    monkeypatch.setattr(e.os, "fsync", changed_after_validation)
    with pytest.raises(ValueError, match="Bound input changed"):
        e.build_completion_index(authority)
    assert not (authority.root / e.INDEX_PATH).exists()
    assert not list((authority.root / e.AREA).glob(".completion_index.json.*"))


def test_all_four_interface_sources_are_bound_separately_and_index_is_idempotent(completed):
    authority, _ = completed
    lock_before = (authority.root / e.LOCK_PATH).read_bytes()
    receipt = e.build_completion_index(authority)
    index_before = (authority.root / e.INDEX_PATH).read_bytes()
    body = e.read_json(authority.root / e.INDEX_PATH)
    assert set(e.INTERFACE_FILES) <= set(body["files"])
    assert e.build_completion_index(authority) == receipt
    assert (authority.root / e.INDEX_PATH).read_bytes() == index_before
    for relative in e.INTERFACE_FILES:
        path = authority.root / relative
        original = path.read_bytes()
        path.write_bytes(original + b"Invented mutation\n")
        with pytest.raises(ValueError, match="byte binding"):
            e.validate_completion_index(authority, receipt["completion_index_sha256"])
        path.write_bytes(original)
    assert (authority.root / e.LOCK_PATH).read_bytes() == lock_before


def test_fresh_worker_session_cannot_be_reused_across_arms(completed):
    authority, directories = completed
    response_path = f"{directories['hospital', 'empty_carry']}/rounds/0001/verification.json"
    body = e.read_json(authority.root / response_path)
    body["fresh_session_id"] = "invented-hospital-seeded_carry-1"
    put(authority.root, response_path, body)
    with pytest.raises(ValueError, match="Worker session reused across dataset/arm/round"):
        e.build_completion_index(authority)
    assert not (authority.root / e.INDEX_PATH).exists()


def test_unknown_completion_map_path_is_rejected_without_opening_it(completed, monkeypatch):
    authority, _ = completed
    e.build_completion_index(authority)
    index_path = authority.root / e.INDEX_PATH
    body = e.read_json(index_path)
    body.pop("index_hash")
    extra = "forbidden/invented_only.txt"
    body["files"][extra] = "a" * 64
    put(authority.root, e.INDEX_PATH, hashed(body, "index_hash"))
    original_file = e.Store.file
    def observed_file(store, relative):
        assert relative != extra, "An unconsumed path must never be opened"
        return original_file(store, relative)
    monkeypatch.setattr(e.Store, "file", observed_file)
    with pytest.raises(ValueError, match="unconsumed paths"):
        e.validate_completion_index(authority, e.sha256(index_path))


@pytest.mark.parametrize("extra", ["empty_directory", "manifest_symlink", "directory_symlink", "fifo"])
def test_existing_export_rejects_every_extra_or_symlink_filesystem_entry(completed, extra):
    authority, _ = completed
    receipt = e.build_completion_index(authority)
    e.export(authority, receipt["completion_index_sha256"])
    output = authority.root / "results/conservative_seed_v1"
    manifest = output / "export_manifest.json"
    manifest_before = manifest.read_bytes()
    if extra == "empty_directory":
        (output / "unexpected_empty_directory").mkdir()
    elif extra == "manifest_symlink":
        target = authority.root / "invented_manifest_copy.json"
        target.write_bytes(manifest_before)
        manifest.unlink()
        manifest.symlink_to(target)
    elif extra == "directory_symlink":
        (output / "unexpected_directory_link").symlink_to(output / "raw", target_is_directory=True)
    else:
        e.os.mkfifo(output / "unexpected_fifo")
    with pytest.raises(ValueError, match="refusing overwrite|forbidden filesystem|path coverage"):
        e.export(authority, receipt["completion_index_sha256"])
    assert manifest.read_bytes() == manifest_before


def test_recorded_guard_must_agree_with_identical_captured_candidate_tuple(tmp_path):
    validator = e.Validator(e.Authority(tmp_path))
    candidate = {"candidate_id": e.candidate_id(0, 0, "B"), "value": "B", "raw_effect": effect()}
    candidate["raw_effect"].update(before_rule_results=[0], after_rule_results=[0])
    cell = {"row": 0, "column_index": 0, "column": "value", "candidates": [candidate]}
    validator.captured_effect("hospital", cell, candidate, candidate["raw_effect"], ["value"], {"value": "A"})
    plan = {"plan_hash": "a" * 64, "rules": [{"target": "value", "code": "Never executed by this interface"}]}
    state = hashed({"protocol": e.PROTOCOL, "dataset": "hospital", "arm": "seeded_carry", "external_round": 1,
                    "plan_hash": plan["plan_hash"], "probabilities": [[.9]], "active": {"0:0": candidate["candidate_id"]},
                    "cells": {"0:0": {"detector_bit": True, "latest_decisive_original_request": None,
                                        "emission_status": "emitted", "support_status": "carried_unresolved"}}, "events": [],
                    "guard_records": [{"row": 0, "round": 1, "reason": "post_transition", "raw_rule_results": [0],
                                       "proposed_rule_results": [1], "blocked": True, "vacuous": False,
                                       "proposed_values": {"0:0": "B"}}]}, "state_hash")
    with pytest.raises(ValueError, match="Captured rule results disagree"):
        validator.state(state, plan, {"0:0": cell}, "hospital", "seeded_carry", 1, (1, 1), [["A"]])


def test_identical_captured_counterfactual_context_cannot_change_evidence(tmp_path):
    validator = e.Validator(e.Authority(tmp_path))
    cell = {"row": 0, "column_index": 0, "column": "value"}
    candidate = {"value": "B"}
    validator.captured_effect("hospital", cell, candidate, effect(), ["value"], {"value": "A"})
    changed = effect()
    changed["raw_value_count"] += 1
    with pytest.raises(ValueError, match="Captured identical counterfactual evidence"):
        validator.captured_effect("hospital", cell, candidate, changed, ["value"], {"value": "A"})


def test_grouped_reply_expands_exactly_and_rejects_candidate_or_member_drift(completed):
    authority, directories = completed
    directory = directories["hospital", "seeded_carry"]
    request = e.read_json(authority.root / f"{directory}/rounds/0001/request.json")
    direct = e.read_json(authority.root / f"{directory}/rounds/0001/verification.json")
    expected, expected_joints = e.expand_response(request, direct, direct["reviewed_batch_hashes"])
    grouped = {key: deepcopy(value) for key, value in direct.items() if key != "cells"}
    grouped["groups"] = [{"members": [cell["cell_id"] for cell in request["cells"]], "original_assessment": unknown(),
                           "selection": {"kind": "abstain", "value": None, "confidence": .8},
                           "candidate_assessments": {"B": unknown(), "C": unknown()}, "default_candidate_assessment": unknown(),
                           "joint_plan_id": None, "applicability_explanation": "Invented shared unresolved evidence for these exact members."}]
    actual, actual_joints = e.expand_response(request, grouped, direct["reviewed_batch_hashes"])
    assert actual == expected and actual_joints == expected_joints
    bad = deepcopy(grouped)
    bad["groups"][0]["candidate_assessments"]["never-requested-invented-value"] = unknown()
    with pytest.raises(ValueError, match="Unrequested grouped candidate"):
        e.expand_response(request, bad, direct["reviewed_batch_hashes"])
    for members in (["0:0"], ["0:0", "0:0", "1:0"]):
        bad = deepcopy(grouped)
        bad["groups"][0]["members"] = members
        bad["groups"][0]["candidate_assessments"] = {}
        with pytest.raises(ValueError, match="Incomplete recorded verifier coverage|duplicate"):
            e.expand_response(request, bad, direct["reviewed_batch_hashes"])


def test_completion_schema_path_names_match_runtime_canonical_restrictions():
    import re
    schema = e.read_json(SCRIPT.parent.parent / e.INTERFACE_FILES[3])
    assert schema["properties"]["files"]["propertyNames"] == {"$ref": "#/$defs/path"}
    pattern = schema["$defs"]["path"]["pattern"]
    for relative in (*e.INTERFACE_FILES, e.PREPARATION_PATH, e.LOCK_PATH):
        assert re.fullmatch(pattern, relative)
    for relative in (".", "..", "../escape", "a/../escape", "a/./escape", "/absolute", "a\\b", "a//b", "a/"):
        assert not re.fullmatch(pattern, relative)


@pytest.mark.parametrize("interruption", [True, False])
def test_export_interruption_or_publication_race_never_exposes_partial_or_overwrites(completed, monkeypatch, interruption):
    authority, _ = completed
    receipt = e.build_completion_index(authority)
    output = authority.root / "results/conservative_seed_v1"
    original_publish = e.publish_directory
    def stopped_publication(stage, destination):
        assert not destination.exists()
        assert (stage / "export_manifest.json").is_file()
        if interruption:
            raise KeyboardInterrupt("Invented interruption before directory publication")
        destination.mkdir()
        (destination / "winner.txt").write_bytes(b"Preserve the first complete destination")
        original_publish(stage, destination)
    monkeypatch.setattr(e, "publish_directory", stopped_publication)
    with pytest.raises(KeyboardInterrupt if interruption else ValueError):
        e.export(authority, receipt["completion_index_sha256"])
    if interruption:
        assert not output.exists()
    else:
        assert {path.name for path in output.iterdir()} == {"winner.txt"}
        assert (output / "winner.txt").read_bytes() == b"Preserve the first complete destination"
    assert not list(output.parent.glob(".conservative_seed_v1.*"))
