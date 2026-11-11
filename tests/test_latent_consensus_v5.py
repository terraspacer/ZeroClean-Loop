"""V5 synthetic tests; frozen v4 regression module remains byte-exact."""
import hashlib
import json
import numpy as np
import pytest
from zeroclean.data import DirtyTable

def table(rows, columns=("entity", "source", "value")):
    rows = tuple(tuple(row) for row in rows)
    fingerprint = hashlib.sha256(json.dumps([columns, rows]).encode()).hexdigest()
    return DirtyTable("synthetic", tuple(columns), rows, fingerprint)

def v5_average_fixture(full_originals=(.1, .3), group_counts=(4, 8), loo_original=.9):
    from experimental import latent_consensus_model_average_v5 as v5
    contexts = {}
    for mode, original, group_count in zip(v5.MODES, full_originals, group_counts):
        observations = {f"g{i}": "A" if i == 0 else "B" for i in range(group_count)}
        scenarios = []
        for index in range(17):
            for omitted in (None, *observations):
                value = original if omitted is None else loo_original
                scenarios.append({"selected_model_index": index, "omitted_group": omitted,
                                  "posterior_weights": [value, .99-value, .01]})
        contexts[mode] = {"source_assumption": mode, "fold_blockers": [],
                          "candidate_strings": ["A", "B"], "grouped_claims": observations,
                          "scenarios": scenarios}
    cell = {"original": "A", "eligible_both_source_assumptions": True,
            "source_assumptions": {mode: {"context_id": mode, "eligibility_blockers": []} for mode in v5.MODES},
            "action": "retain_uncertain", "error_weight_interval": [.1, .9]}
    return cell, contexts


def test_v5_equal_full_model_weights_do_not_depend_on_loo_or_source_count():
    from experimental import latent_consensus_model_average_v5 as v5
    cell, contexts = v5_average_fixture()
    result = v5.aggregate_original(cell, contexts)
    assert result["mean_error_weight"] == pytest.approx(.8)
    assert result["v4_action"] == "retain_uncertain"
    assert result["leave_group_out_mean_decision_changes"]
    assert result["combined_leave_group_out_error_range"][0] < .5
    for mode in v5.MODES:
        assert result["mode_evidence"][mode]["each_full_model_weight"] == pytest.approx(1/34)
    equal, more_groups = v5_average_fixture(group_counts=(9, 15), loo_original=.01)
    assert v5.aggregate_original(equal, more_groups)["mean_error_weight"] == pytest.approx(.8)


def test_v5_rejects_duplicate_scenarios_and_absent_original_without_invented_scores():
    from experimental import latent_consensus_model_average_v5 as v5
    cell, contexts = v5_average_fixture()
    contexts[v5.MODES[0]]["scenarios"].append(contexts[v5.MODES[0]]["scenarios"][0])
    with pytest.raises(ValueError, match="Duplicated"):
        v5.aggregate_original(cell, contexts)
    cell, contexts = v5_average_fixture()
    cell["original"] = "absent"
    with pytest.raises(ValueError, match="absent"):
        v5.aggregate_original(cell, contexts)
    cell["eligible_both_source_assumptions"] = False
    cell["error_weight_interval"] = None
    assert v5.aggregate_original(cell, contexts)["mean_error_weight"] is None


def v5_evidence(data, weights):
    return {"dirty_sha256": data.fingerprint, "cells": [
        {"row": i, "column": data.columns[j], "column_index": j, "original": data.rows[i][j],
         "mean_evidence": {"eligible": value is not None, "mean_error_weight": value}}
        for (i, j), value in weights.items()]}


def test_v5_explicit_repair_gate_preserves_exact_frozen_score_and_unsupported_prior():
    from experimental import latent_consensus_model_average_v5 as v5
    from zeroclean.types import Prediction
    data = table([("e0", "s0", "A"), ("e1", "s1", "B")])
    prior = Prediction(np.full(data.shape, .17))
    frozen = Prediction(np.full(data.shape, .87), {(0, 2): "B"})
    evidence = v5_evidence(data, {(0, 2): .2, (1, 2): None})
    result, audit = v5.apply_mean(data, prior, frozen, evidence)
    assert result.probabilities[0, 2] == .87
    assert audit["repair_gates"][0]["candidate_score"] == .2
    assert audit["repair_gates"][0]["retained_frozen_v3_ed_score"] == .87
    assert result.probabilities[1, 2] == prior.probabilities[1, 2]
    assert result.repairs == frozen.repairs
    invalid = Prediction(np.full(data.shape, .3), {(0, 2): "B"})
    with pytest.raises(ValueError, match="no silent normalization"):
        v5.apply_mean(data, prior, invalid, evidence)


def test_v5_mean_exact_half_is_positive_and_optional_scores_are_explicit_decisions():
    from experimental import latent_consensus_model_average_v5 as v5
    from zeroclean.types import Prediction
    data = table([("e0", "s0", "A"), ("e1", "s1", "B")])
    prior = Prediction(np.full(data.shape, .11))
    frozen = Prediction(np.full(data.shape, .9), {(1, 2): "A"})
    evidence = v5_evidence(data, {(0, 2): .5, (1, 2): .05})
    mean, _ = v5.apply_mean(data, prior, frozen, evidence)
    assert mean.probabilities[0, 2] == .5
    assert mean.probabilities[0, 2] >= .5
    proxy, audit = v5.apply_proxy(data, prior, frozen, evidence)
    assert proxy.probabilities[1, 2] == 1
    assert proxy.probabilities[0, 2] in (0, 1)
    assert proxy.probabilities[0, 0] == prior.probabilities[0, 0]
    assert proxy.repairs == frozen.repairs
    assert "decision scores" in proxy.metadata["score_meaning"]
    assert audit["solution"]["mandatory_indices"] == [1]


def enumerate_independent_proxy_f1(probabilities, selected):
    from itertools import product
    selected = set(selected)
    result = 0.0
    for labels in product((0, 1), repeat=len(probabilities)):
        weight = float(np.prod([p if y else 1-p for p, y in zip(probabilities, labels)]))
        denominator = len(selected)+sum(labels)
        value = 2*sum(labels[i] for i in selected)/denominator if denominator else 0
        result += weight*value
    return result


def test_v5_proxy_expected_ratio_matches_enumeration_and_differs_from_ratio_of_means():
    from itertools import product
    from experimental import latent_consensus_model_average_v5 as v5
    p = [0, .2, .5, .8, 1]
    for mask in product((0, 1), repeat=len(p)):
        selected = [i for i, included in enumerate(mask) if included]
        assert v5.conditional_f1_for_set(p, selected) == pytest.approx(enumerate_independent_proxy_f1(p, selected), abs=1e-12)
    exact_proxy = v5.conditional_f1_for_set([.4], [0])
    assert exact_proxy == pytest.approx(.4)
    assert exact_proxy != pytest.approx(2*.4/(1+.4))


def test_v5_ranked_proxy_optimizer_matches_exhaustive_constrained_decisions():
    from itertools import product
    from experimental import latent_consensus_model_average_v5 as v5
    p = [.05, .7, .7, .25, 1]
    coords = [(0, 0), (2, 0), (1, 0), (3, 0), (4, 0)]
    mandatory = {0}
    solution = v5.optimize_independent_marginals_proxy(p, coords, mandatory)
    exhaustive = []
    for mask in product((0, 1), repeat=len(p)):
        selected = {i for i, included in enumerate(mask) if included}
        if mandatory <= selected:
            exhaustive.append((enumerate_independent_proxy_f1(p, selected), len(selected-mandatory)))
    maximum = max(v for v, _ in exhaustive)
    assert solution["selected_independent_marginals_proxy_expected_f1"] == pytest.approx(maximum, abs=1e-12)
    assert mandatory <= set(solution["selected_indices"])
    assert solution["free_order"].index(2) < solution["free_order"].index(1)
    assert solution["selected_optional_count"] == min(k for v, k in exhaustive if maximum-v <= 1e-12)


def test_v5_proxy_endpoints_empty_scope_and_shared_entity_dependence_disclaimer():
    from experimental import latent_consensus_model_average_v5 as v5
    p = [0, .5, 1]
    total = v5.poisson_binomial(p)
    for index, value in enumerate(p):
        division, _ = v5.leave_one_distribution(total, value)
        assert division == pytest.approx(v5.poisson_binomial(p[:index]+p[index+1:]), abs=1e-12)
    empty = v5.optimize_independent_marginals_proxy([], [])
    assert empty["selected_indices"] == []
    assert empty["selected_independent_marginals_proxy_expected_f1"] == 0
    assert v5.optimize_independent_marginals_proxy([0, 0], [(0, 0), (0, 1)])["selected_indices"] == []
    assert v5.optimize_independent_marginals_proxy([1, 1], [(0, 0), (0, 1)])["selected_indices"] == [0, 1]
    independent_proxy = v5.conditional_f1_for_set([.5, .5], [0])
    coupled_shared_truth_expectation = .5*(2/3)
    assert independent_proxy == pytest.approx(5/12)
    assert independent_proxy != pytest.approx(coupled_shared_truth_expectation)


def test_v5_verifier_claim_context_preserves_raw_conflicts_status_and_formats():
    from experimental import latent_consensus_model_average_v5 as v5
    raw = [("s1", "A"), ("s1", "B"), ("s2", "Not Available"),
           ("s3", "9:40aDec 1"), ("s4", "9:40 a.m."), ("s3", "9:40aDec 1"), ("s5", "")]
    mapping = {"s1": "g1", "s2": "g2", "s3": "g3", "s4": "g3", "s5": "g5"}
    contexts = {mode: {"fold": 0, "source_mapping": mapping, "trained_groups": list(mapping.values()),
                      "grouped_claims": {"g2": "Not Available"}, "fold_blockers": []}
                for mode in v5.MODES}
    cells = [{"row": i, "column": "value", "entity": ["e"], "source": source,
              "original": value, "source_assumptions": {mode: {"context_id": mode} for mode in v5.MODES}}
             for i, (source, value) in enumerate(raw)]
    archive = v5.archive_claim_contexts({"cells": cells}, contexts)
    record = next(iter(archive.values()))
    assert [(c["source"], c["value"]) for c in record["raw_claims"]] == raw
    assert record["raw_source_conflicts"][0]["values"] == ["A", "B"]
    assert record["duplicate_nonempty_claim_extra_rows"] == 1
    assert record["conditional_hypotheses_only"]
    for mode in v5.MODES:
        view = record["source_assumptions"][mode]
        assert view["group_value_conflicts"]["g3"] == ["9:40 a.m.", "9:40aDec 1"]
        assert view["active_group_claims"] == {"g2": "Not Available"}
        assert any("conflicted_raw_source" in c["reasons"] for c in view["claim_exclusions"])
        assert any("blank_or_whitespace" in c["reasons"] for c in view["claim_exclusions"])


def test_v5_proxy_numerical_ties_choose_fewest_within_global_maximum_tolerance():
    from experimental import latent_consensus_model_average_v5 as v5
    solution = v5.optimize_independent_marginals_proxy([.9e-12, .9e-12], [(0, 0), (0, 1)])
    values = [v["independent_marginals_proxy_expected_f1"] for v in solution["utility_path"]]
    assert values[1] < 1e-12 < values[2]
    assert max(values)-values[1] <= 1e-12
    assert solution["selected_indices"] == [0]
