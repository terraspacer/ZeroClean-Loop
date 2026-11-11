#!/usr/bin/env python3
"""Evaluator-only audit; never executes the study, optimizer or verifier."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct

import audit_six_dataset_source as a
import audit_components_source_v2_six as t
import audit_criteria_and_verified_overlays as graph
import audit_detector_pair_and_model_average as data

NAME = "flight_model_spec_v1"
ROOT = a.REPO / "results" / NAME
SOURCE = a.REPO / "experimental/flight_model_spec_review"
RUN = SOURCE / "run_v1"
OWNED = a.REPO / "results/evaluation_audits/flight_model_spec_v1_inputs"
OUTPUT = a.REPO / "results/evaluation_audits/flight_model_spec_v1.json"
QUARANTINE = a.REPO / "results/evaluation_only_failures/audit_flight_model_spec_v1"
PINS = {
    ROOT / "artifact_manifest.json": "e79f05714ce2081efb6e3d38ba69cad42d3664b53832d55f18dfbd53d8b513ec",
    ROOT / "frozen/flights.json": "b65e940b167e0f3980fa7a6ba163ce2151a07f195dd2e575fe712a3fbb34dd6d",
    RUN / "artifact_manifest.json": "89e0f86eb668d8e81c1e997b9eb936a03e00414f03e81075d435ac78fadbd2fb",
    SOURCE / "verification_manifest.json": "84d742a6547174b063cb27db03e7cfca339f85695071a9801d879383b5d2607d",
}
ARMS = [g + "_" + s for g in ("attribute_near", "profile_complete")
        for s in ("lambda0p0", "lambda0p5", "lambda0p9", "equal", "selectedEB")] + ["independent_names_lambda0p0"]
ATTRS = ("sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time")
COUNTS = Counter()
ERRORS = Counter()


def ok(condition, category):
    ERRORS[category] += not bool(condition)


def close(x, y, tolerance=1e-12):
    return x is None and y is None or (x is not None and y is not None and
            math.isclose(x, y, rel_tol=0, abs_tol=tolerance))


def check(path, expected):
    path = a.pathof(path)
    a.check_hash(path, expected, "bound_input/" + a.short(path), "study_bound_entries")


def inventory(path, field="files"):
    body = a.read_json(path)
    for rel, expected in body[field].items():
        check(path.parent / rel, expected)
    return body


def preflight():
    summary = a.read_json(OWNED / "results/flight_model_spec_v1/summary_metrics.json")
    ok(summary == a.read_json(ROOT / "summary_metrics.json"), "exact_summary_snapshot")
    ok(summary["measured_datasets"] == ["flights"] and set(summary["metrics"]) == {"flights"} and summary["six_dataset_benchmark_complete"] is False, "dataset_scope")
    freeze = a.read_json(ROOT / "frozen/flights.json")
    # This exporter explicitly uses compact separators; other source exporters
    # use json.dumps defaults. Never change the historical shared helper.
    body = {k: v for k, v in freeze.items() if k != "export_hash"}
    compact_sha = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    a.compare(freeze["export_hash"], compact_sha, "freeze_compact_content_hash")
    ok(freeze["frozen_before_evaluation"] is True and freeze["ground_truth_access"] is False, "pre_evaluation_freeze_flags")
    ok(set(freeze["prediction_files"]) == {s + ".json" for s in summary["metrics"]["flights"]}, "summary_frozen_prediction_set")
    for name, expected in freeze["prediction_files"].items():
        path = ROOT / "predictions/flights" / name
        check(path, expected)
        meta = a.read_json(path)["metadata"]
        origin = a.REPO / "vendor" / ("ZeroEC" if name == "zeroed_zeroec_reconstruction.json" else "ZeroED")
        for filename, fingerprint in meta.get("source_sha256", {}).items():
            source = (a.REPO / "artifacts/recovered_prompts/zerodc_29cde9e0dbcd" / Path(filename).name
                      if filename.startswith("author_recovered/") else origin / filename)
            check(source, fingerprint)
        for filename, fingerprint in meta.get("model_sha256", {}).items():
            check(a.REPO / "vendor/ZeroEC/all-MiniLM-L6-v2" / filename, fingerprint)
        for key in ("ground_truth_access", "uses_ground_truth"):
            if key in meta:
                ok(meta[key] is False, "prediction_declared_label_flags")
    for filename, expected in freeze["inference_inputs"].items():
        check(filename, expected)
    check(a.REPO / "vendor/ZeroED/data/flights_error-01.csv", freeze["dirty_sha256"])
    counts = {NAME: {"prediction_files": len(freeze["prediction_files"]), "inference_input_entries": len(freeze["inference_inputs"])}}
    print(json.dumps({"prelabel_root_hashes_complete": NAME, **counts[NAME]}), flush=True)
    return summary, counts


def fold(entity, seed=1701, n=5):
    raw = json.dumps([seed, [entity]], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % n


def posterior(observations, skills, vocabulary, rho):
    k = len(vocabulary)
    scores = [math.log((1 - rho) / k)] * k + [math.log(rho)]
    for group, value in observations.items():
        scores[vocabulary.index(value)] += skills[group]
    largest = max(scores)
    weights = [math.exp(x - largest) for x in scores]
    total = math.fsum(weights)
    return [x / total for x in weights]


def embargo(prepared, attribute, target):
    members = prepared["attributes"][attribute]["members"]
    banned = set(members[target])
    families = prepared["clone_diagnostics"]["families"]
    while True:
        expanded = set(banned)
        for family in itertools.chain(families.values(), members.values()):
            if banned.intersection(family):
                expanded.update(family)
        if expanded == banned:
            break
        banned = expanded
    return banned, {g for g, ss in members.items() if banned.intersection(ss)}


def claims(rows, entities, attribute, header, mapping, retained=None):
    index = header.index(attribute)
    raw = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row[1] in entities and row[0].strip() and row[index].strip():
            raw[row[1]][row[0]].add(row[index])
    result = {}
    for entity in entities:
        grouped = defaultdict(set)
        for source, values in raw[entity].items():
            if len(values) == 1 and source in mapping:
                grouped[mapping[source]].update(values)
        result[entity] = {g: next(iter(values)) for g, values in grouped.items()
                          if len(values) == 1 and (retained is None or g in retained)}
    return result


def provenance():
    for path, expected in PINS.items():
        check(path, expected)
    exported = inventory(ROOT / "artifact_manifest.json")
    run = inventory(RUN / "artifact_manifest.json")
    verification = inventory(SOURCE / "verification_manifest.json")
    freeze = a.read_json(ROOT / "frozen/flights.json")
    pre = a.read_json(RUN / "pre_fit_manifest.json")
    preexport = inventory(ROOT / "pre_export_manifest.json", "files_before_publication")
    ok(exported["prediction_count"] == 14 and exported["statistical_arm_count"] == 11, "export_counts")
    ok(set(freeze["registered_statistical_arms"]) == set(ARMS), "fixed_eleven_arms")
    ok(set(freeze["prediction_files"]) == {s + ".json" for s in ARMS + ["source_v2_prior", "zeroed", "zeroed_zeroec_reconstruction"]}, "fixed_fourteen_predictions")
    for binding in freeze["input_origin_bindings"]:
        check(binding["source"], binding["expected_sha256"])
        check(ROOT / binding["snapshot"], binding["expected_sha256"])
        ok(a.hashfile(a.pathof(binding["source"])) == a.hashfile(ROOT / binding["snapshot"]), "origin_source_snapshot_equality")
    for path, expected in pre["protected_input_hashes"].items():
        check(path, expected)
    for path, expected in verification["pre_fit_protected_source_hashes"].items():
        check(path, expected)
    check(SOURCE / "study.py", pre["implementation_sha256"])
    check(SOURCE / "test_study.py", pre["test_sha256"])
    check(SOURCE / "protocol.md", pre["protocol_sha256"])
    check(SOURCE / "verify_frozen.py", a.read_json(SOURCE / "verification_v1.json")["verification_code_sha256"])
    check(a.REPO / "tests/test_latent_consensus.py", "40da4ed6f319142323c63ce3abeb9613f4cb18a2c48400ba771f1ca7ba8cd2c5")
    ok(pre["groupings"] == ["attribute_near", "profile_complete"] and pre["lambdas"] == [0, .5, .9], "fixed_grid")
    ok(pre["outer_seed"] == 1701 and pre["inner_seed"] == 1702, "fixed_seeds")
    ok(pre["decision_operator"] == "score >= 0.5" and pre["repairs"] == "zero; ED-only", "fixed_decision_repair_policy")
    ok(freeze["fully_verified_main_method"] is False and freeze["unverified_statistical_ablation"] is True, "statistical_scope_flags")
    ok(verification["verification_passed"] and verification["benchmark_evaluator_or_labels_used"] is False, "verification_receipt_scope")
    ok(preexport["ground_truth_access"] is False and preexport["evaluator_executed"] is False, "export_pre_evaluation_flags")
    access = a.read_json(SOURCE / "access_provenance.json")
    ok(access["protocol_sha256"] == pre["protocol_sha256"] and access["protocol_immutable_before_exposure"] is True, "access_protocol_binding")
    ok(access["inherited_trace_count_displayed"] == 3 and access["benchmark_labels_or_metrics_exposed"] is False, "access_disclosure_counts")
    ok(access["other_researcher_verdicts_read"] is False, "access_disclosure_other_verdicts")
    baseline = a.read_json(a.REPO / "results/source_baselines_v1/frozen/flights.json")
    for key, expected in baseline["inference_inputs"].items():
        check(key, expected)
    for name in ("zeroed", "zeroed_zeroec_reconstruction"):
        check(a.REPO / "results/source_baselines_v1/predictions/flights" / (name + ".json"), freeze["prediction_files"][name + ".json"])
    # Read the original current prior manifest graph, rather than substituting its copies.
    prior_path = a.REPO / "artifacts/closed_loop_v2/flights/4e0f4d0f82ada5f10a82bd9fb57a613dd8bf28110ef7ca66633ea38717c31247/prediction.json"
    check(prior_path, pre["prior_sha256"])
    ok(a.hashfile(prior_path) == a.hashfile(ROOT / "predictions/flights/source_v2_prior.json"), "byte_exact_prior")
    return {"export_file_entries": len(exported["files"]), "source_run_file_entries": len(run["files"]),
            "verification_file_entries": len(verification["files"]), "live_origin_bindings": len(freeze["input_origin_bindings"]),
            "freeze_content_hash_encoding": "SHA-256 of sorted-key UTF-8 JSON with compact comma/colon separators, as declared by this exporter",
            "protected_current_sources_verified": True, "access_provenance_sha256": a.hashfile(SOURCE / "access_provenance.json"),
            "access_qualification": "Protocol-frozen study with disclosed exposure to three inherited dirty-only prior explanation traces; no benchmark labels or metrics disclosed. Trace exclusion is attested and supported by probability-only prior-use code; absence of unrecorded access cannot be proven cryptographically.",
            "fit_verification_manifest_sha256": a.hashfile(SOURCE / "verification_manifest.json"),
            "fit_verification_receipt_authenticated": True}


def prepared_checks(prep, rows, header):
    training = set(prep["training_entities"])
    profiles = defaultdict(dict)
    for row in rows:
        if row[1] in training:
            profiles[row[0]][row[1]] = tuple(row[2:])
    diag = prep["clone_diagnostics"]
    signatures = {s: a.canonical(sorted(p.items())) for s, p in profiles.items()}
    ok(signatures == diag["source_profile_signatures"], "training_only_profile_signatures")
    pair_map = {}
    for record in diag["pairs"]:
        x, y = record["a"], record["b"]
        overlap = profiles[x].keys() & profiles[y].keys()
        equal = sum(profiles[x][e] == profiles[y][e] for e in overlap)
        compatible = len(overlap) >= 12 and equal == len(overlap)
        ok(record["shared_entities"] == len(overlap) and record["equal_four_tuples_including_blanks"] == equal and record["compatible"] == compatible, "complete_profile_pair_arithmetic")
        pair_map[frozenset((x, y))] = compatible
    ok(len(pair_map) == len(profiles) * (len(profiles) - 1) // 2, "all_profile_pairs_retained")
    clusters = [{s} for s in sorted(profiles, key=lambda s: (signatures[s], s))]
    for merge in diag["merges"]:
        options = [(tuple(sorted(signatures[s] for s in x | y)), i, j)
                   for i, x in enumerate(clusters) for j, y in enumerate(clusters) if j > i
                   and all(pair_map[frozenset((u, v))] for u in x for v in y)]
        ok(bool(options), "complete_link_merge_available")
        _, i, j = min(options)
        ok(set(merge["left"]) == clusters[i] and set(merge["right"]) == clusters[j] and set(merge["merged"]) == clusters[i] | clusters[j], "complete_link_merge_order")
        clusters = [x for k, x in enumerate(clusters) if k not in (i, j)] + [clusters[i] | clusters[j]]
    ok({frozenset(v) for v in diag["families"].values()} == {frozenset(v) for v in clusters}, "complete_link_final_partition")
    ok(not any(all(pair_map[frozenset((u, v))] for u in x for v in y) for i, x in enumerate(clusters) for y in clusters[i + 1:]), "complete_link_maximality")
    COUNTS["training_family_records"] += len(clusters)
    ok(prep["clone_mapping"] == {s: g for g, ss in diag["families"].items() for s in ss}, "clone_mapping_membership")
    for attribute, attr in prep["attributes"].items():
        ok(set(attr["training_item_entities"]) <= training, "support_training_only")
        ok(attr["members"] == {g: sorted(s for s, v in attr["mapping"].items() if v == g) for g in set(attr["mapping"].values())}, "group_mapping_membership")
        if prep["grouping"] == "profile_complete":
            ok(attr["mapping"] == prep["clone_mapping"], "profile_grouping_shared_across_attributes")
        if prep["grouping"] == "independent_names":
            ok(all(g == s for s, g in attr["mapping"].items()), "independent_control_singletons")
        active = claims(rows, set(attr["training_item_entities"]), attribute, header, attr["mapping"], set(attr["groups"]))
        support = Counter(g for obs in active.values() for g in obs)
        ok(dict(support) == attr["group_nonblank_training_support"], "recorded_group_support_arithmetic")
    coords, raw_sources, design = prep["coordinates"], prep["raw_sources"], prep["source_membership_design"]
    ok(len(design) == len(coords) and all(len(r) == len(raw_sources) for r in design), "gaussian_membership_dimensions")
    for (attribute, group), vector in zip(coords, design):
        ss = prep["attributes"][attribute]["members"][group]
        expected = [1 / math.sqrt(len(ss)) if s in ss else 0 for s in raw_sources]
        ok(all(close(x, y) for x, y in zip(vector, expected)) and close(math.fsum(x*x for x in vector), 1), "gaussian_normalized_membership_design")


def fit_records(model, primary):
    packs = [(fit["optimization"], fit) for fit in model.get("independent_attribute_fits", {}).values()]
    if "all_start_fits" in model:
        packs.append((model, None))
    for pack, selected in packs:
        starts = pack["all_start_fits"]
        ok([s["initial_skill"] for s in starts] == [-2, 0, 2], "fixed_starts")
        good = [(i, s) for i, s in enumerate(starts) if s["converged"] and math.isfinite(s["objective_trace"][-1])]
        if "selected_start_index" in pack:
            best = max(good, key=lambda z: (z[1]["objective_trace"][-1], -z[0]))[0]
            ok(pack["selected_start_index"] == best, "within_model_start_selection")
            if selected is not None:
                ok(selected["skill"] == starts[best]["skill"], "selected_independent_skills")
        for start in starts:
            trace = start["objective_trace"]
            ok(bool(trace) and all(math.isfinite(x) for x in trace), "fit_trace_finiteness")
            ok(all(y >= x - 1e-7 for x, y in zip(trace, trace[1:])), "fit_trace_monotonicity")
            ok(start["iterations"] <= 200, "fit_iteration_bound")
            if "m_steps" in start:
                ok(all(s["success"] for s in start["m_steps"]) or not start["converged"], "m_step_recorded_success_gate")
            COUNTS["all_archived_start_records"] += 1
            COUNTS["primary_start_records"] += bool(primary)
    if model["lambda"] == "equal":
        ok(all(close(theta, math.log(2)) for skills in model["skill"].values() for theta in skills.values()), "equal_reliability_fixed_contrast")


def scores_check(block, models):
    prep, cases = block["prepared"], block["cases"]
    validation = set(block.get("validation_entities", block.get("heldout_entities", [])))
    for case in cases:
        a_name, target = case["attribute"], case["target_group"]
        vocab = sorted(set(case["conditioning_claims"].values()))
        sources, groups = embargo(prep, a_name, target)
        ok(case["entity"] in validation and vocab == case["conditioning_vocabulary"] and len(case["conditioning_claims"]) >= 3, "case_scope_and_target_excluded_vocabulary")
        ok(sources == set(case["embargo_sources"]) and groups == set(case["embargo_groups"]) and not groups.intersection(case["conditioning_claims"]), "whole_family_embargo_closure")
        target_bucket = vocab.index(case["target_raw_string"]) if case["target_raw_string"] in vocab else len(vocab)
        ok(case["target_bucket"] == target_bucket and case["target_is_other"] == (target_bucket == len(vocab)), "target_other_bucket")
    for name, model in models.items():
        score = block["scores"][name]
        by_item = defaultdict(list)
        ok(len(score["case_values"]) == len(cases), "shared_case_scope_across_arms")
        for case, recorded in zip(cases, score["case_values"]):
            skills = model["skill"][case["attribute"]]
            vocab = case["conditioning_vocabulary"]
            p = posterior(case["conditioning_claims"], skills, vocab, model["rho"])
            advantage = math.exp(skills[case["target_group"]])
            distribution = [(advantage * x + 1 - x) / (advantage + len(vocab)) for x in p]
            probability = distribution[case["target_bucket"]]
            loss = -math.log(probability)
            ok(close(math.fsum(distribution), 1, 1e-10) and close(probability, recorded["predictive_bucket_probability"]) and close(loss, recorded["negative_log_probability"], 1e-10), "predictive_bucket_arithmetic")
            by_item[case["entity"], case["attribute"]].append(loss)
            COUNTS["model_case_probabilities_recalculated"] += 1
        by_entity = defaultdict(list)
        for (entity, _), losses in by_item.items():
            by_entity[entity].append(math.fsum(losses) / len(losses))
        losses = {e: math.fsum(v) / len(v) for e, v in by_entity.items()}
        ok(set(losses) == set(score["entity_losses"]) and all(close(v, score["entity_losses"][e], 1e-10) for e, v in losses.items()), "entity_attribute_balanced_nll")
        mean = math.fsum(losses.values()) / len(losses) if losses else None
        ok(close(mean, score["mean_entity_bucket_nll"], 1e-10) and score["entity_count"] == len(losses) and score["case_count"] == len(cases), "nll_mean_counts")
    COUNTS["predictive_cases"] += len(cases)


def study_checks():
    header, rows = a.raw_table(a.REPO / "vendor/ZeroED/data/flights_error-01.csv")
    entities = {r[1] for r in rows}
    inner = data.load_gzip(RUN / "inner_validation.json.gz")
    outer = data.load_gzip(RUN / "outer_fits_and_scoring.json.gz")
    selection = a.read_json(RUN / "nested_selection.json")
    ok(len(inner) == 30 and len(outer) == 15 and len(selection) == 10, "nested_block_counts")
    for block in inner + outer:
        prep = block["prepared"]
        f = block["outer_fold"]
        train = {e for e in entities if fold(e) != f}
        held = entities - train
        if "inner_fold" in block:
            val = {e for e in train if fold(e, 1702, 3) == block["inner_fold"]}
            expected_train = train - val
            ok(set(block["outer_heldout_entities"]) == held, "outer_heldout_embargo_from_inner")
            models = block["models"]
        else:
            val, expected_train = held, train
            models = {k: v[0] for k, v in block["models_and_sensitivities"].items()}
            if "equal_control" in block:
                models["equal"] = block["equal_control"]
        ok(set(prep["training_entities"]) == expected_train and set(block.get("validation_entities", block.get("heldout_entities", []))) == val and not expected_train.intersection(val), "exact_entity_folds_and_disjointness")
        prepared_checks(prep, rows, header)
        for model in models.values():
            fit_records(model, True)
        if "models_and_sensitivities" in block:
            expected_scenarios = [(math.log(2), .1), (-math.log(2), .1), (0, .1), (math.log(2), .05), (math.log(2), .2)]
            for variants in block["models_and_sensitivities"].values():
                ok([(x["center"], x["rho"]) for x in variants] == expected_scenarios, "fixed_sensitivity_grid")
                for model in variants[1:]:
                    fit_records(model, False)
        scores_check(block, models)
    choices = {}
    for selected in selection:
        f, grouping = selected["outer_fold"], selected["grouping"]
        blocks = [b for b in inner if b["outer_fold"] == f and b["prepared"]["grouping"] == grouping]
        expected_train = {e for e in entities if fold(e) != f}
        ok(len(blocks) == 3 and set(selected["selection_training_entities"]) == expected_train and set(selected["excluded_outer_entities"]) == entities - expected_train, "selection_training_only_within_grouping")
        available = all(b["cases"] and all(set(m["available_attributes"]) == {x for x in ATTRS if not b["prepared"]["attributes"][x]["blockers"]} for m in b["models"].values()) for b in blocks)
        pooled = {}
        for lam in (0., .5, .9):
            loss = {}
            for b in blocks:
                ok(not set(loss).intersection(b["scores"][str(lam)]["entity_losses"]), "inner_validation_entity_unique")
                loss.update(b["scores"][str(lam)]["entity_losses"])
            pooled[str(lam)] = {"entity_losses": loss}
        ok(pooled == selected["inner_entity_losses"], "selected_inner_aggregate_binding")
        if not available:
            expected, reason = 0., "unavailable_inner_fold_scope"
        elif any(not v["entity_losses"] for v in pooled.values()):
            expected, reason = 0., "unavailable_dirty_validation_scope"
        else:
            means = {float(k): math.fsum(v["entity_losses"].values()) / len(v["entity_losses"]) for k, v in pooled.items()}
            minimum = min(means.values())
            expected = next(lam for lam in (0., .5, .9) if means[lam] <= minimum + 1e-8)
            reason = "minimum_inner_entity_bucket_nll"
        ok(selected["selected_lambda"] == expected and selected["reason"] == reason, "nested_lambda_selection_exact")
        choices[f, grouping] = expected
    outer_lookup = {(b["outer_fold"], b["prepared"]["grouping"]): b for b in outer}
    prior = a.read_json(ROOT / "predictions/flights/source_v2_prior.json")["probabilities"]
    for arm in ARMS:
        pred = a.read_json(ROOT / "predictions/flights" / (arm + ".json"))
        ok(pred["repairs"] == [] and pred["metadata"]["ed_only"] is True and pred["metadata"]["decision_operator"] == "score >= 0.5", "all_arms_ed_only_fixed_operator")
        ok(pred["metadata"]["arm"] == arm and pred["metadata"].get("upstream_binary_decisions") is None and pred["metadata"].get("decision_threshold", .5) == .5 and pred["metadata"].get("decision_threshold_operator", ">=") == ">=", "statistical_arm_identity_and_mask_policy")
        context = data.load_gzip(RUN / arm / "raw_claim_contexts_and_sensitivity.json.gz")
        evidence = data.load_gzip(RUN / arm / "cell_evidence.json.gz")
        ok(len(evidence) == len(rows) * 4 and len({(c["row"], c["column_index"]) for c in evidence}) == len(evidence), "complete_unique_cell_records")
        ok(all(struct.pack("<d", p) == struct.pack("<d", prior[i][j]) for i, r in enumerate(pred["probabilities"]) for j, p in enumerate(r) if j < 2), "nonattribute_prior_bitwise_preserved")
        for cid, c in context.items():
            b = outer_lookup[fold(c["entity"]), c["grouping"]]
            prep = b["prepared"]
            attr = prep["attributes"][c["attribute"]]
            lam = choices[b["outer_fold"], c["grouping"]] if arm.endswith("selectedEB") else c["lambda"]
            expected_group = "independent_names" if arm.startswith("independent_names") else "attribute_near" if arm.startswith("attribute_near") else "profile_complete"
            expected_lambda = choices[b["outer_fold"], expected_group] if arm.endswith("selectedEB") else "equal" if arm.endswith("equal") else float(arm.rsplit("lambda", 1)[1].replace("p", "."))
            ok(c["grouping"] == expected_group and c["lambda"] == expected_lambda, "predeclared_arm_model_no_crossgroup_selection")
            variants = [] if lam == "equal" else b["models_and_sensitivities"][str(lam)]
            model = b["equal_control"] if lam == "equal" else variants[0]
            ok(cid == a.canonical([prep["split_id"], c["entity"], c["attribute"]]) and c["context_id"] == cid and c["split_id"] == prep["split_id"] and c["lambda"] == model["lambda"], "context_exact_fold_and_nested_model_binding")
            observed = claims(rows, {c["entity"]}, c["attribute"], header, attr["mapping"], set(attr["groups"]))[c["entity"]]
            vocab = sorted(set(observed.values()))
            ok(c["source_to_group"] == attr["mapping"] and c["group_members"] == attr["members"] and c["trained_groups"] == attr["groups"] and c["active_claims"] == observed and c["candidate_strings"] == vocab and c["support_blockers"] == attr["blockers"], "context_dirty_claims_and_trained_scope")
            expected_rows = {i for i, r in enumerate(rows) if r[1] == c["entity"]}
            ok({r["row"] for r in c["raw_claims"]} == expected_rows and len(c["raw_claims"]) == len(expected_rows), "complete_raw_context_coverage")
            for raw in c["raw_claims"]:
                row = rows[raw["row"]]
                ok(raw["source"] == row[0] and raw["value"] == row[header.index(c["attribute"])] and raw["four_value_tuple"] == dict(zip(ATTRS, row[2:])), "raw_dirty_tuple_binding")
            available = c["attribute"] in model["available_attributes"]
            expected = posterior(observed, model["skill"][c["attribute"]], vocab, model["rho"]) if available and len(observed) >= 4 else None
            ok(c["fit_available"] == available and ((expected is None and c["primary_posterior"] is None) or (expected is not None and len(expected) == len(c["primary_posterior"]) and all(close(x, y) for x, y in zip(expected, c["primary_posterior"])))), "full_context_primary_posterior_arithmetic")
            scenarios = c["sensitivity_scenarios"]
            ok(len(scenarios) == (len(variants) if expected is not None else 0), "sensitivity_scope_no_primary_substitution")
            for scenario, fit in zip(scenarios, variants):
                ok(scenario["center"] == fit["center"] and scenario["rho"] == fit["rho"] and scenario["available"] == (c["attribute"] in fit["available_attributes"]), "sensitivity_model_binding")
                if not scenario["available"]:
                    continue
                skill = fit["skill"][c["attribute"]]
                weights = posterior(observed, skill, vocab, fit["rho"])
                ok(all(close(x, y) for x, y in zip(weights, scenario["posterior_weights"])), "sensitivity_posterior_arithmetic")
                expected_omissions = {frozenset(g for g in embargo(prep, c["attribute"], target)[1] if g in observed) for target in observed}
                ok({frozenset(o["omitted_groups"]) for o in scenario["whole_family_omissions"]} == expected_omissions and len(scenario["whole_family_omissions"]) == len(expected_omissions), "sensitivity_whole_family_omission_scope")
                for omission in scenario["whole_family_omissions"]:
                    remaining = {g: v for g, v in observed.items() if g not in omission["omitted_groups"]}
                    weights = posterior(remaining, skill, vocab, fit["rho"])
                    ok(omission["remaining_groups"] == len(remaining) and all(close(x, y) for x, y in zip(weights, omission["posterior_weights"])), "sensitivity_omission_arithmetic")
                    COUNTS["sensitivity_omission_records"] += 1
            COUNTS["raw_contexts"] += 1
        for c in evidence:
            i, j = c["row"], c["column_index"]
            ctx = context[c["context_id"]]
            b = outer_lookup[fold(ctx["entity"]), ctx["grouping"]]
            lam = ctx["lambda"]
            model = b["equal_control"] if lam == "equal" else b["models_and_sensitivities"][str(lam)][0]
            attr = b["prepared"]["attributes"][ctx["attribute"]]
            original, source = rows[i][j], rows[i][0]
            reasons = []
            if not original.strip(): reasons.append("blank_original")
            if not source.strip(): reasons.append("missing_source")
            if not rows[i][1].strip(): reasons.append("missing_entity_key")
            group = attr["mapping"].get(source)
            skills = model["skill"].get(ctx["attribute"], {})
            observed = ctx["active_claims"]
            if group is None: reasons.append("unseen_source")
            elif group not in skills: reasons.append("untrained_source_group")
            elif group not in observed: reasons.append("inactive_or_conflicted_group")
            elif observed[group] != original: reasons.append("group_original_mismatch")
            if len(observed) < 4: reasons.append("fewer_than_four_active_groups")
            if original not in observed.values(): reasons.append("original_outside_candidate_vocabulary")
            if not ctx["fit_available"]: reasons.append("fit_unavailable")
            if ctx["primary_posterior"] is None and not reasons: reasons.append("context_unavailable")
            eligible = not reasons
            ok(c["column"] == header[j] == ctx["attribute"] and c["original"] == original and c["source"] == source and c["entity"] == rows[i][1] == ctx["entity"], "cell_raw_context_binding")
            ok(c["eligible"] == eligible and c["fallback_reasons"] == sorted(set(reasons)), "eligibility_and_fallback_gates")
            value = pred["probabilities"][i][j]
            ok(value == c["final_probability"], "prediction_evidence_exact_score")
            if eligible:
                score = 1 - ctx["primary_posterior"][ctx["candidate_strings"].index(original)]
                ok(value == score and c["model_error_probability"] == score and c["prior_fallback_probability"] is None, "eligible_model_score")
            else:
                ok(struct.pack("<d", value) == struct.pack("<d", prior[i][j]) and c["model_error_probability"] is None and c["prior_fallback_probability"] == value, "unsupported_fallback_bitwise_exact")
                COUNTS["unsupported_prior_values_exact"] += 1
            scenario_values = [1 - s["posterior_weights"][ctx["candidate_strings"].index(original)] for s in ctx["sensitivity_scenarios"] if eligible and s["available"] and original in ctx["candidate_strings"]]
            expected_range = [min(scenario_values), max(scenario_values)] if scenario_values else None
            ok(expected_range == c["sensitivity_error_range"], "finite_sensitivity_range_arithmetic")
            COUNTS["cell_records"] += 1
        print(json.dumps({"prelabel_arm_evidence_complete": arm, "cell_records": len(evidence)}), flush=True)
    receipt = a.read_json(SOURCE / "verification_v1.json")
    for field, counter in (("predictive_cases_recalculated", "predictive_cases"), ("primary_stationary_fits_checked", "primary_start_records"), ("train_only_complete_link_families_checked", "training_family_records"), ("cell_records_checked", "cell_records"), ("raw_contexts_checked", "raw_contexts"), ("unsupported_prior_values_exact", "unsupported_prior_values_exact")):
        ok(receipt[field] == COUNTS[counter], "authenticated_receipt_aggregate_reconciliation")
    ok(receipt["zero_repairs_all_arms"] is True and receipt["benchmark_evaluator_or_labels_used"] is False, "fit_receipt_scope")
    for category, count in ERRORS.items():
        a.compare(0, count, "study_checks/" + category)
    return {"independent_record_counts": dict(COUNTS), "arithmetic_and_structure_discrepancy_counts": dict(ERRORS),
            "all_eleven_arms_retained": True, "cross_grouping_winner_selected": False,
            "exact_prior_fallback_and_zero_study_repairs_verified": True,
            "verification_scope": "Independent standard-library recalculation of archived posterior/predictive probabilities, balanced losses, within-group nested selection, train-only profile/complete-link records, embargoes, eligibility, fallbacks, sensitivity/omission records and fit-start trace gates. No fit, EM iteration, optimizer, model prediction, source verifier or inference job is rerun. Fitted parameters and immutable-helper behavior are authenticated as frozen source records through pinned code and its verification receipt; parameter estimates and mathematical stationarity are not independently re-estimated or proved here.",
            "limitations": ["Flights only; all six-dataset main-method selections and targets remain unchanged.", "Dirty predictive losses have different scopes across grouping hypotheses and cannot establish benchmark truth, independence, or calibration.", "Computational source verification is not independent semantic verification.", "Sensitivity scenarios are finite diagnostics, not confidence intervals or posterior model samples."]}


def evaluate(summary):
    header, dirty = a.raw_table(a.REPO / "vendor/ZeroED/data/flights_error-01.csv")
    clean_header, reference = a.raw_table(a.REPO / "vendor/ZeroED/data/flights_clean.csv")
    assert header == clean_header and len(dirty) == len(reference)
    dimensions = {"rows": len(dirty), "columns": len(header), "cells": len(dirty) * len(header), "dirty_sha256": a.hashfile(a.REPO / "vendor/ZeroED/data/flights_error-01.csv"), "reference_sha256": a.hashfile(a.REPO / "vendor/ZeroED/data/flights_clean.csv")}
    metrics, sample_checks = {}, {}
    for system in summary["metrics"]["flights"]:
        prediction = a.read_json(ROOT / "predictions/flights" / (system + ".json"))
        metric, samples = a.recompute(header, dirty, reference, prediction)
        metrics[system] = metric
        for kind in ("detection", "repair"):
            for key in a.KEYS:
                a.compare(metric[kind][key], summary["metrics"]["flights"][system][kind][key], "metrics/" + system + "/" + kind + "/" + key)
        for key in ("rows", "columns", "dirty_sha256", "reference_sha256"):
            a.compare(dimensions[key], summary["metrics"]["flights"][system][key], "metrics/" + system + "/" + key)
        recorded = a.read_json(ROOT / "evaluation_only_failures/flights" / (system + ".json"))
        sample_checks[system] = {}
        for kind, examples in samples.items():
            matched = a.compare(True, examples == recorded[kind], "failure_samples/" + system + "/" + kind)
            sample_checks[system][kind] = {"all_samples_match": matched, "sample_size": len(examples), "exhaustive_count": metric["failure_counts"][kind]}
        path = QUARANTINE / "flights" / (system + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"evaluation_only": True, "never_use_for_inference_or_tuning": True, "samples": samples}, ensure_ascii=False, indent=2) + "\n")
    macro = {s: {kind: m[kind]["f1"] for kind in ("detection", "repair")} for s, m in metrics.items()}
    for system, values in macro.items():
        for kind, value in values.items():
            a.compare(value, summary["macro_f1"][system][kind], "macro/" + system + "/" + kind)
    assessments = {}
    for section, baseline in (("official_target_assessment", ("zeroed", "zeroed_zeroec")), ("adapted_comparison_only", ("adapted_zeroed", "adapted_zeroed_zeroec")), ("qualified_source_reconstruction_comparison", ("zeroed", "zeroed_zeroec_reconstruction"))):
        expected = a.target_assessment({"flights": metrics}, baseline)
        for key, value in expected.items():
            a.compare(value, summary[section].get(key), section + "/" + key)
        assessments[section] = expected
    a.compare(False, summary["targets_met"], "no_six_dataset_target_claim")
    return {"dataset_scope": ["flights"], "macro_denominator": 1, "metrics": metrics, "macro_f1": macro, "target_assessments": assessments, "recorded_failure_sample_checks": sample_checks}, dimensions


def main():
    started = datetime.now(timezone.utc).isoformat()
    paths = [ROOT / "summary_metrics.json", ROOT / "frozen/flights.json", ROOT / "artifact_manifest.json", ROOT / "pre_export_manifest.json", RUN / "artifact_manifest.json", RUN / "pre_fit_manifest.json", SOURCE / "verification_manifest.json", SOURCE / "access_provenance.json", SOURCE / "protocol.md", SOURCE / "verification_v1.json"]
    snapshots = {a.short(p): t.snapshot_exact(p, OWNED / p.relative_to(a.REPO)) for p in paths}
    for module in (a, t, graph, data):
        a.hashfile(Path(module.__file__))
    a.hashfile(Path(__file__))
    semantic = a.semantic_checks()
    summary, counts = preflight()
    sources = provenance()
    study = study_checks()
    prelabel_count = len(a.DISCREPANCIES)
    print(json.dumps({"all_prelabel_checks_complete": True, "discrepancy_count": prelabel_count}), flush=True)
    # A hash discrepancy prevents any benchmark-reference read.
    assert not any(d["location"].startswith("bound_input/") or "content_hash" in d["location"] for d in a.DISCREPANCIES), "Bound input hash mismatch; labels remain unopened"
    experiment, dimensions = evaluate(summary)
    changed = []
    for path, expected in list(a.SNAPSHOTS.items()):
        actual = a.hashfile(path, refresh=True)
        if actual != expected:
            changed.append(a.short(path))
            a.compare(expected, actual, "post_audit_hash/" + a.short(path))
    report = {"audit_schema": "independent-flight-model-spec-v1", "evaluation_only": True,
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "historical_input_snapshots": snapshots,
        "summary_and_freeze_snapshots_precede_reference_read": True,
        "method": {"implementation": "Standard library and evaluator-owned helpers; no model, optimizer, rule engine, source verifier or inference execution", "reference_policy": "Exact raw CSV strings; no normalization or null coercion", "detection_policy": "Stored-mask precedence for frozen source baseline; statistical arms declare score >= 0.5, ties positive", "repair_policy": "Dirty cell changed exactly to reference is TP; wrong dirty edit counts FP and FN; unchanged clean never earns TP", "examples_quarantined": a.short(QUARANTINE), "no_inference_or_tuning": True},
        "semantic_self_checks": semantic, "prelabel_integrity": {"all_prediction_and_declared_source_hashes_checked_before_reference_read": True, "discrepancy_count": prelabel_count, "root_counts": counts, "hash_checks": dict(a.HASH_CHECKS)},
        "source_provenance": sources, "study_protocol_and_arithmetic": study, "dimensions": dimensions, "experiment": experiment,
        "source_ec_qualification": "Included EC baseline is an author-prompt reconstruction with unproven paper-prompt identity and zero fixed examples.",
        "read_input_integrity": {"unique_files_rehashed_after_audit": len(a.SNAPSHOTS), "changed_files": changed, "all_stable": not changed},
        "discrepancy_count": len(a.DISCREPANCIES), "discrepancies": a.DISCREPANCIES,
        "input_sha256": {a.short(p): h for p, h in sorted(a.SNAPSHOTS.items(), key=lambda x: str(x[0]))},
        "auditor_script_sha256": a.hashfile(Path(__file__), refresh=True),
        "independent_helper_sha256": {a.short(Path(m.__file__)): a.hashfile(Path(m.__file__), refresh=True) for m in (a, t, graph, data)}}
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=lambda x: sorted(x) if isinstance(x, set) else str(x)) + "\n")
    print(json.dumps({"output": a.short(OUTPUT), "discrepancy_count": len(a.DISCREPANCIES), "all_inputs_stable": not changed, "prediction_runs": counts[NAME]["prediction_files"]}), flush=True)


if __name__ == "__main__":
    main()
