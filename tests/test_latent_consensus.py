"""Synthetic-only checks; this module never loads a benchmark table or reference."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from experimental.latent_consensus import (
    Contract, Parameters, _fold, _posterior, profile_claims, profile_table,
    run_experiment,
)
from zeroclean.data import DirtyTable


def table(rows, columns=("entity", "source", "value")):
    rows = tuple(tuple(row) for row in rows)
    fingerprint = hashlib.sha256(json.dumps([columns, rows]).encode()).hexdigest()
    return DirtyTable("synthetic", tuple(columns), rows, fingerprint)


def contract(**kwargs):
    values = dict(entity_columns=("entity",), source_column="source", value_column="value",
                  single_valued=True, independent_sources=True, stationary_reliability=True,
                  better_than_half=True, ignorable_missingness=True,
                  evidence="Synthetic independent Bernoulli sources with one fixed truth per entity.")
    values.update(kwargs)
    return Contract(**values)


def params(**kwargs):
    values = dict(folds=3, min_train_entities=20, min_source_items=12,
                  min_pair_overlap=8, min_conflicting_items=5,
                  bootstrap_replicates=2, max_iter=150, tolerance=1e-6)
    values.update(kwargs)
    return Parameters(**values)


def independent_table(n=160):
    rng = np.random.default_rng(29)
    rows = []
    for entity in range(n):
        truth, error = f"  Original {entity:04d}  ", f"Different {entity:04d}"
        for source, q in enumerate((.96, .93, .88, .83, .75, .65, .60)):
            rows.append((f"e{entity:04d}", f"s{source}", truth if rng.random() < q else error))
    return table(rows)


def test_requires_explicit_semantic_contract_and_never_fits_from_repetition():
    data = independent_table(45)
    result = run_experiment(data, Contract(("entity",), "source", "value"), params())
    assert result["fit_performed"] is False
    assert result["status_counts"] == {"abstain": len(data.rows)}
    assert "unestablished_contract:independent_sources" in result["contract_blockers"]


def test_source_reliability_is_cross_fitted_and_original_strings_are_preserved():
    data = independent_table()
    result = run_experiment(data, contract(), params())
    assert result["fit_performed"]
    assert result["dirty_fingerprint"] == data.fingerprint
    assert any(d["status"] == "propose" for d in result["decisions"])
    for d in result["decisions"]:
        assert d["original"] == data.rows[d["row"]][2]
        if d["status"] == "propose":
            assert d["candidate"] in d["candidate_strings"]
            assert d["candidate"] != d["original"]
            assert d["conditional_weight_min"] >= params().min_conditional_probability
    for fold in result["folds"]:
        assert not fold["blockers"], fold["blockers"]
        assert all(m["reliability"]["s0"] > m["reliability"]["s6"] for m in fold["models"])
    # Changing an entity's held-out observations cannot change that fold's fit,
    # dependency graph, or bootstrap. It may change other folds' training data.
    target = data.rows[0][0]
    fold_id = _fold((target,), 1, params())
    changed = table([(e, s, "arbitrary changed string" if e == target else v)
                     for e, s, v in data.rows])
    changed_result = run_experiment(changed, contract(), params())
    before = next(f for f in result["folds"] if f["fold"] == fold_id)
    after = next(f for f in changed_result["folds"] if f["fold"] == fold_id)
    assert before["models"] == after["models"]
    assert before["source_group_mapping"] == after["source_group_mapping"]


def test_exact_duplicate_rows_cannot_inflate_votes_or_reliabilities():
    data = independent_table()
    duplicate = table((*data.rows, *data.rows[:100], *data.rows[:100]))
    original_result = run_experiment(data, contract(), params())
    duplicate_result = run_experiment(duplicate, contract(), params())
    assert duplicate_result["extraction"]["duplicate_claim_extra_rows"] == 200
    assert original_result["folds"] == duplicate_result["folds"]
    assert original_result["decisions"] == duplicate_result["decisions"][:len(data.rows)]


def test_same_source_conflict_is_excluded_instead_of_majority_voted():
    data = table([("entity", "same source", "A"), ("entity", "same source", "A"),
                  ("entity", "same source", "B"), ("entity", "other source", "A")])
    result = profile_claims(data, contract(), params())
    assert result["duplicate_claim_extra_rows"] == 1
    assert result["usable_source_entity_pairs"] == 1
    assert result["source_entity_conflicts"][0]["values"] == ["A", "B"]
    decisions = run_experiment(data, contract(), params())["decisions"]
    assert all(d["reasons"] == ["source_entity_has_multiple_values"] for d in decisions[:3])


def test_source_clones_are_one_group_and_force_abstention_when_views_collapse():
    rows = [(str(e), f"clone{s}", f"value{e}") for e in range(60) for s in range(7)]
    data = table(rows)
    profile = profile_claims(data, contract(), params())
    assert len(profile["source_groups"]) == 1
    assert profile["grouped_entity_coverage"] == {1: 60}
    result = run_experiment(data, contract(), params())
    assert result["fit_performed"] is False
    assert result["status_counts"] == {"abstain": len(rows)}
    hypothetical_independence = profile_claims(data, contract(), params(source_assumption="independent_names"))
    assert len(hypothetical_independence["source_groups"]) == 7
    assert hypothetical_independence["potential_dependency_pairs"]


def test_missingness_and_list_cardinality_are_not_silently_repaired():
    data = table([("e", "s", ""), ("e", "s2", "   "), ("e2", "s", "A,B"),
                  ("e3", "s", "0"), ("e4", "s", "N/A"), ("e5", "s", "empty")])
    result = run_experiment(data, contract(single_valued=False), params())
    assert result["extraction"]["skipped_rows"]["missing_value"] == 2
    assert result["extraction"]["usable_source_entity_pairs"] == 4
    assert [d["original"] for d in result["decisions"]] == [r[2] for r in data.rows]
    assert all(d["status"] == "abstain" for d in result["decisions"])
    assert result["decisions"][0]["reasons"] == ["missing_value_not_imputed"]
    assert profile_table(data)["columns"][2]["delimiter_rows"] == 1


def test_time_contexts_share_folds_but_are_distinct_truth_items():
    parameters = params()
    assert _fold(("person", "1996"), 1, parameters) == _fold(("person", "2014"), 1, parameters)
    data = table([("person", "1996", "s", "40"), ("person", "2014", "s", "58")],
                 ("entity", "year", "source", "value"))
    profile = profile_claims(data, contract(context_columns=("year",)), parameters)
    assert profile["entities"] == 2
    assert profile["source_entity_conflicts"] == []
    without_context = profile_claims(data, contract(), parameters)
    assert len(without_context["source_entity_conflicts"]) == 1


def test_unseen_state_ambiguity_and_low_view_support_abstain():
    _, posterior = _posterior({"a": "A", "b": "B"}, {"a": .75, "b": .75}, .1)
    assert posterior[0] == pytest.approx(posterior[1])
    assert posterior[-1] > 0
    data = table([(f"e{e}", f"s{s}", "A" if (e+s) % 3 else "B")
                  for e in range(80) for s in range(2)])
    result = run_experiment(data, contract(), params())
    assert not result["fit_performed"]
    assert result["status_counts"] == {"abstain": len(data.rows)}


def test_invalid_roles_and_parameters_fail_closed():
    with pytest.raises(ValueError, match="distinct"):
        run_experiment(table([]), contract(source_column="entity"), params())
    with pytest.raises(ValueError, match="Unknown columns"):
        run_experiment(table([]), contract(value_column="missing"), params())
    with pytest.raises(ValueError, match="four decision views"):
        replace(params(), min_item_sources=3)


def test_deterministic_synthetic_experiment():
    data = independent_table(85)
    assert run_experiment(data, contract(), params()) == run_experiment(data, contract(), params())


def test_v2_better_than_multiclass_chance_can_be_below_one_half():
    from experimental.latent_consensus_v2 import match_probability
    skill = float(np.log(2))
    assert match_probability(skill, 3) == pytest.approx(.4)
    assert .25 < match_probability(skill, 3) < .5
    assert match_probability(skill, 1) == pytest.approx(2 / 3)
    assert match_probability(skill, 9) == pytest.approx(2 / 11)
    assert match_probability(-skill, 3) < .25


def test_v2_penalized_m_step_allows_adverse_skill_and_solves_score_equation():
    from collections import Counter
    from experimental import latent_consensus_v2 as v2
    parameters = v2.SkillParameters()
    theta = v2._solve_skill(18, Counter({3: 100}), parameters.prior_center, parameters)
    assert theta < 0
    derivative = 18 - 100 * v2.match_probability(theta, 3) - (theta - parameters.prior_center) / parameters.prior_sd ** 2
    assert abs(derivative) < 1e-8


def test_v2_multiclass_mixed_sources_copies_and_monotone_em():
    from experimental import latent_consensus_v2 as v2
    from experimental.latent_consensus import _extract, _groups, _collapse
    rng = np.random.default_rng(915)
    rows = []
    for entity in range(130):
        truth = entity % 3
        last = None
        for source, probability in enumerate((.90, .85, .80, .65, .40, .20, .05)):
            alternatives = [x for x in range(3) if x != truth]
            observed = truth if rng.random() < probability else int(rng.choice(alternatives))
            rows.append((str(entity), f"s{source}", str(observed)))
            last = str(observed)
        rows.append((str(entity), "copy_of_s6", last))
    data = table(rows)
    claims, _ = _extract(data, contract())
    parameters = v2.SkillParameters(bootstrap_replicates=1, max_iter=200, tolerance=1e-6)
    mapping, _ = _groups(claims, parameters)
    assert mapping["s6"] == mapping["copy_of_s6"]
    collapsed, _ = _collapse(claims, mapping)
    sources = sorted({s for o in collapsed.values() for s in o})
    model = v2._fit(collapsed, sources, parameters, .1, 2.0, parameters.prior_center)
    assert model["converged"]
    assert min(np.diff(model["objective_trace"])) >= -1e-8
    assert model["skill"]["s0"] > model["skill"][mapping["s6"]]
    assert model["skill"][mapping["s6"]] < 0
    assert any(q[mapping["s6"]] < .5 for q in model["q_by_observed_candidate_count"].values())
    assert any(len(set(o.values())) == 3 for o in collapsed.values())


def test_v2_orientation_ambiguity_and_unobserved_truth_force_abstention():
    from experimental import latent_consensus_v2 as v2
    observations = {f"s{i}": "A" for i in range(4)}
    positive = {s: 3.0 for s in observations}
    negative = {s: -3.0 for s in observations}
    _, p = v2._posterior(observations, positive, .1)
    _, n = v2._posterior(observations, negative, .1)
    assert p.argmax() == 0
    assert n.argmax() == 1
    models = [{"skill": skill, "unseen_prior": .1} for skill in (positive, negative)]
    decision = v2._decision(observations, models, v2.SkillParameters())
    assert decision["status"] == "abstain"
    assert "winner_depends_on_orientation_or_sensitivity" in decision["reasons"]
    _, absent = v2._posterior({"a": "A", "b": "B", "c": "C"},
                               {"a": -4, "b": -4, "c": -4}, .1)
    assert absent.argmax() == 3


def test_v3_selects_best_converged_objective_in_a_synthetic_multimodal_case():
    from experimental import latent_consensus_v2 as v2
    from experimental import latent_consensus_v3 as v3
    claims = {(str(i),): {f"s{s}": f"value{i}" for s in range(5)} for i in range(55)}
    sources = [f"s{s}" for s in range(5)]
    parameters = v2.SkillParameters(max_iter=200, tolerance=1e-6)
    fits = [v2._fit(claims, sources, parameters, .1, start, parameters.prior_center)
            for start in parameters.initial_skills]
    assert all(f["converged"] for f in fits)
    assert min(min(np.diff(f["objective_trace"])) for f in fits) >= -1e-8
    objectives = [f["objective_trace"][-1] for f in fits]
    assert max(objectives) - min(objectives) > 1.0
    selected = v3._select_fit(fits)
    assert selected["objective_trace"][-1] == max(objectives)
    assert selected["optimization"]["selected_start_index"] == int(np.argmax(objectives))
    assert v3._select_fit([fits[0], fits[0]])["optimization"]["selected_start_index"] == 0
    unconverged = dict(fits[0], converged=False, objective_trace=[max(objectives) + 1000])
    assert v3._select_fit([unconverged, fits[1]])["optimization"]["selected_start_index"] == 1


def feedback_record(data, row, column, candidate):
    original = data.rows[row][column]
    same = candidate == original
    return {"row": row, "column_index": column, "column": data.columns[column],
            "original": original, "candidate": candidate,
            "original_weight_interval": [.99, .995] if same else [0, 1e-6],
            "candidate_weight_lower": .99 if same else .999999,
            "source_assumptions": {"independent_names": {}, "cluster_near_agreement": {}}}


def feedback_inputs(data, cells, rules=()):
    return ({"dirty_sha256": data.fingerprint, "rules": list(rules), "dependencies": []},
            {"dirty_sha256": data.fingerprint, "cells": cells})


def feedback_rule(source, kind="multi", target="a"):
    return {"id": "synthetic_guard", "target": target, "kind": kind,
            "source": source, "description": "Synthetic counterfactual guard"}


def test_feedback_prunes_recovers_and_preserves_unsupported_prior_inference():
    from experimental.latent_consensus_feedback import apply_feedback
    from zeroclean.types import Prediction
    data = table([("e0", "A", "B"), ("e1", "X", "Y"), ("e2", "A", "Y"),
                  ("e3", "X", "B")], ("entity", "a", "b"))
    original_probabilities = np.full(data.shape, .1)
    original_probabilities[0, 1] = .9
    original_probabilities[1, 1] = .7
    prior = Prediction(original_probabilities.copy(), {(0, 1): "X", (1, 1): "prior literal"})
    artifact, evidence = feedback_inputs(data, [feedback_record(data, 0, 1, "A"),
                                              feedback_record(data, 0, 2, "Y")])
    prediction, audit, bundle = apply_feedback(data, prior, artifact, evidence)
    assert prediction.probabilities[0, 1] == pytest.approx(.01)
    assert (0, 1) not in prediction.repairs
    assert prediction.probabilities[0, 2] >= .98
    assert prediction.repairs[0, 2] == "Y"
    assert prediction.probabilities[1, 1] == original_probabilities[1, 1]
    assert prediction.repairs[1, 1] == "prior literal"
    assert np.array_equal(prior.probabilities, original_probabilities)
    assert prior.repairs == {(0, 1): "X", (1, 1): "prior literal"}
    assert all(len(t["delta_features"]) == 4 for t in audit["trace"])
    assert bundle["independent_verification_status"] == "not_run"


def test_feedback_multi_rule_veto_keeps_ed_and_exact_prior_repair_fallback():
    from experimental.latent_consensus_feedback import apply_feedback
    from zeroclean.types import Prediction
    data = table([("e0", "old", "lock"), ("e1", "bad", "open")], ("entity", "a", "b"))
    rule = feedback_rule("def check(row):\n    return row['a'] == 'bad' and row['b'] == 'lock'")
    artifact, evidence = feedback_inputs(data, [feedback_record(data, 0, 1, "bad")], [rule])
    prior = Prediction(np.full(data.shape, .1), {(0, 1): "prior literal"})
    prediction, audit, _ = apply_feedback(data, prior, artifact, evidence)
    assert prediction.probabilities[0, 1] >= .98
    assert prediction.repairs[0, 1] == "prior literal"
    assert "new_multi_column_violation" in audit["trace"][0]["individual_gate_blockers"]


def test_feedback_counterfactual_runtime_veto_keeps_ed():
    from experimental.latent_consensus_feedback import apply_feedback
    from zeroclean.types import Prediction
    data = table([("e0", "1", "lock"), ("e1", "bad", "open")], ("entity", "a", "b"))
    rule = feedback_rule("def check(row):\n    if row['b'] == 'lock':\n        return int(row['a']) < 0\n    return False")
    artifact, evidence = feedback_inputs(data, [feedback_record(data, 0, 1, "bad")], [rule])
    prediction, audit, _ = apply_feedback(data, Prediction(np.full(data.shape, .1)), artifact, evidence)
    assert prediction.probabilities[0, 1] >= .98
    assert (0, 1) not in prediction.repairs
    assert "counterfactual_runtime_failure" in audit["trace"][0]["individual_gate_blockers"]


def test_feedback_joint_interaction_rolls_back_corrections_but_keeps_ed():
    from experimental.latent_consensus_feedback import apply_feedback
    from zeroclean.types import Prediction
    data = table([("e0", "A", "B"), ("e1", "X", "B"), ("e2", "A", "Y"),
                  ("e3", "A", "B")], ("entity", "a", "b"))
    rule = feedback_rule("def check(row):\n    return row['a'] == 'X' and row['b'] == 'Y'")
    artifact, evidence = feedback_inputs(data, [feedback_record(data, 0, 1, "X"),
                                              feedback_record(data, 0, 2, "Y")], [rule])
    prior = Prediction(np.full(data.shape, .1), {(3, 1): "preserved prior"})
    prediction, audit, _ = apply_feedback(data, prior, artifact, evidence)
    assert all(t["individual_gate_passed"] for t in audit["trace"])
    assert all(t["joint_rollback"] for t in audit["trace"])
    assert prediction.repairs == prior.repairs
    assert prediction.probabilities[0, 1] >= .98 and prediction.probabilities[0, 2] >= .98
    assert audit["joint_guard"][0]["rollback_to_prior"]


def test_feedback_reconstructs_conditional_original_weights_without_fitting(monkeypatch):
    from experimental import latent_consensus_feedback as feedback
    from experimental import latent_consensus_v2 as v2
    def forbidden(*args, **kwargs):
        raise AssertionError("A frozen feedback variant must never refit")
    monkeypatch.setattr(v2, "_fit", forbidden)
    observations = {"s0": "candidate", "s1": "candidate", "s2": "candidate", "s3": "original"}
    model = {"skill": {"s0": 4, "s1": 4, "s2": 4, "s3": -4},
             "unseen_prior": .1, "converged": True}
    trace = feedback.original_weight_trace(observations, [model, model], "original", "candidate")
    assert trace["original_weight_interval"][1] <= .02
    assert trace["candidate_weight_lower"] >= .98
    assert trace["original_weights"][0] == trace["original_weights"][1]


def v4_cell_from_contexts(data, row, column, contexts):
    from experimental import latent_consensus_detection_feedback_v4 as v4
    original = data.rows[row][column]
    modes = {mode: {"context_id": mode, "eligibility_blockers": [],
                    **v4.interval_for_original(context, original)}
             for mode, context in contexts.items()}
    low = min(v["original_weight_interval"][0] for v in modes.values())
    high = max(v["original_weight_interval"][1] for v in modes.values())
    error = [1-high, 1-low]
    action, score = v4.classify_error_interval(error)
    return {"row": row, "column_index": column, "column": data.columns[column],
            "original": original, "source_assumptions": modes,
            "eligible_both_source_assumptions": True,
            "original_weight_interval": [low, high], "error_weight_interval": error,
            "action": action, "feedback_score": score}


def test_v4_original_ruled_out_with_two_plausible_repairs_splitting_mass():
    from experimental import latent_consensus_detection_feedback_v4 as v4
    from experimental import latent_consensus_v2 as v2
    from zeroclean.types import Prediction
    obs = {"original_source": "A", "b1": "B", "b2": "B", "c1": "C", "c2": "C"}
    model_b = {"skill": {"original_source": -2, "b1": 1.1, "b2": 1.1, "c1": 1, "c2": 1},
               "unseen_prior": .1, "converged": True}
    model_c = {"skill": {"original_source": -2, "b1": 1, "b2": 1, "c1": 1.1, "c2": 1.1},
               "unseen_prior": .1, "converged": True}
    models = [model_b, model_c]*8+[model_b]
    context = v4.posterior_scenarios(obs, models)
    interval = v4.interval_for_original(context, "A")
    assert interval["error_weight_interval"][0] > .5
    assert v4.classify_error_interval(interval["error_weight_interval"])[0] == "flag"
    _, first = v2._posterior(obs, model_b["skill"], .1)
    _, second = v2._posterior(obs, model_c["skill"], .1)
    assert first.argmax() != second.argmax()
    assert first[1] > .4 and first[2] > .4
    assert max(first) < .98 and max(second) < .98
    assert v2._decision(obs, models, v2.SkillParameters())["status"] == "abstain"
    data = table([("e", "original_source", "A"), ("e", "b1", "B"), ("e", "c1", "C")])
    cell = v4_cell_from_contexts(data, 0, 2, {mode: context for mode in v4.MODES})
    prior = Prediction(np.full(data.shape, .1))
    fixed_ec = Prediction(np.full(data.shape, .1), {})
    prediction, audit, _ = v4.apply_detection_feedback(data, prior, fixed_ec,
        {"dirty_sha256": data.fingerprint, "cells": [cell]})
    assert prediction.probabilities[0, 2] > .5
    assert prediction.repairs == {}
    assert audit["correction_map_unchanged_from_v3"]


def test_v4_uncertain_original_and_exact_half_boundaries_retain_prior():
    from experimental import latent_consensus_detection_feedback_v4 as v4
    from zeroclean.types import Prediction
    obs = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B"}
    model = {"skill": {s: 1.0 for s in obs}, "unseen_prior": .1, "converged": True}
    context = v4.posterior_scenarios(obs, [model]*17)
    interval = v4.interval_for_original(context, "A")["error_weight_interval"]
    assert interval[0] < .5 < interval[1]
    assert v4.classify_error_interval(interval) == ("retain_uncertain", None)
    assert v4.classify_error_interval([.5, .8]) == ("retain_uncertain", None)
    assert v4.classify_error_interval([.2, .5]) == ("retain_uncertain", None)
    data = table([("e", "a1", "A"), ("e", "b1", "B")])
    cell = v4_cell_from_contexts(data, 0, 2, {mode: context for mode in v4.MODES})
    prior = Prediction(np.full(data.shape, .83))
    fixed_ec = Prediction(np.full(data.shape, .1), {(1, 2): "A"})
    prediction, _, _ = v4.apply_detection_feedback(data, prior, fixed_ec,
        {"dirty_sha256": data.fingerprint, "cells": [cell]})
    assert np.array_equal(prediction.probabilities, prior.probabilities)
    assert prediction.repairs == fixed_ec.repairs


def test_v4_invalid_source_missing_or_absent_vocabulary_never_gets_a_score():
    from experimental import latent_consensus_detection_feedback_v4 as v4
    from zeroclean.types import Prediction
    obs = {"g0": "A", "g1": "B", "g2": "C", "g3": "B"}
    mapping = {"s0": "g0", "inactive": "g9", "untrained": "g8"}
    trained = {"g0", "g1", "g2", "g3", "g9"}
    assert not v4.source_eligibility("A", "s0", mapping, trained, obs)
    assert v4.source_eligibility("A", "missing_source", mapping, trained, obs)
    assert v4.source_eligibility("A", "inactive", mapping, trained, obs)
    assert v4.source_eligibility("A", "untrained", mapping, trained, obs)
    assert v4.source_eligibility("A", "s0", mapping, trained, obs, conflicted=True)
    assert v4.source_eligibility("", "s0", mapping, trained, obs)
    assert v4.source_eligibility("absent", "s0", mapping, trained, obs)
    model = {"skill": {s: 1 for s in obs}, "unseen_prior": .1, "converged": True}
    context = v4.posterior_scenarios(obs, [model]*17)
    with pytest.raises(ValueError, match="absent"):
        v4.interval_for_original(context, "absent")
    data = table([("e", "s0", "A")])
    cell = {"row": 0, "column_index": 2, "column": "value", "original": "A",
            "source_assumptions": {m: {"eligibility_blockers": ["invalid_source"], "original_weight_interval": None}
                                   for m in v4.MODES}, "eligible_both_source_assumptions": False,
            "original_weight_interval": None, "error_weight_interval": None,
            "action": "retain_unsupported", "feedback_score": None}
    prior = Prediction(np.full(data.shape, .73))
    prediction, _, _ = v4.apply_detection_feedback(data, prior, Prediction(np.zeros(data.shape)),
        {"dirty_sha256": data.fingerprint, "cells": [cell]})
    assert np.array_equal(prediction.probabilities, prior.probabilities)


def test_feedback_v3_low_counterfactual_confidence_keeps_ed_and_prior_ec():
    from experimental.latent_consensus_feedback import apply_feedback
    from zeroclean.types import Prediction
    rows = [(f"e{i}", "A" if i < 99 else "rare", "fixed") for i in range(100)]
    data = table(rows, ("entity", "a", "b"))
    rule = feedback_rule("def check(row):\n    return row['a'] == 'rare'", kind="single")
    cell = feedback_record(data, 0, 1, "rare")
    cell.update(original_weight_interval=[0, .02], candidate_weight_lower=.98)
    artifact, evidence = feedback_inputs(data, [cell], [rule])
    prior = Prediction(np.full(data.shape, .1), {(0, 1): "prior exact string"})
    prediction, audit, _ = apply_feedback(data, prior, artifact, evidence)
    trace = audit["trace"][0]
    assert trace["delta_features"][0] == -1
    assert trace["delta_features"][1] < -.9
    assert trace["correction_confidence"] < .8
    assert "counterfactual_confidence_below_threshold" in trace["individual_gate_blockers"]
    assert "new_multi_column_violation" not in trace["individual_gate_blockers"]
    assert prediction.repairs == prior.repairs
    assert prediction.probabilities[0, 1] >= .98


def test_feedback_v3_joint_rollback_restores_prior_not_dirty_and_same_row_unsupported():
    from experimental.latent_consensus_feedback import apply_feedback
    from zeroclean.types import Prediction
    data = table([("e0", "A", "B", "U"), ("e1", "X", "B", "U"),
                  ("e2", "A", "Y", "U"), ("e3", "A", "B", "U")], ("entity", "a", "b", "c"))
    rule = feedback_rule("def check(row):\n    return row['a'] == 'X' and row['b'] == 'Y'")
    artifact, evidence = feedback_inputs(data, [feedback_record(data, 0, 1, "X"),
                                              feedback_record(data, 0, 2, "Y")], [rule])
    prior = Prediction(np.full(data.shape, .13), {(0, 1): "P", (0, 3): "Q"})
    prediction, audit, _ = apply_feedback(data, prior, artifact, evidence)
    assert all(t["individual_gate_passed"] and t["joint_rollback"] for t in audit["trace"])
    assert prediction.repairs == {(0, 1): "P", (0, 3): "Q"}
    assert prediction.probabilities[0, 3] == prior.probabilities[0, 3]
    assert prediction.probabilities[0, 1] >= .98 and prediction.probabilities[0, 2] >= .98
