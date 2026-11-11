# Flights v3: an inspectable experiment with 793 stable proposal previews

V3 produces 793 row-and-attribute proposals that independently pass both source assumptions and all unchanged prior, bootstrap, support, and leave-group-out gates. They remain unpromoted, conditional model outputs. No correctness labels, benchmark references, evaluation results, or live inference changes were used. The other five datasets lack source provenance for this experiment; their [dirty-only structural feasibility](latent_consensus_feasibility.md) remains unchanged.

| Time attribute | Proposed rows: independent names | Proposed rows: conservative clusters | Identical proposal passing both | Raw flights with a common proposal |
| --- | ---: | ---: | ---: | ---: |
| Scheduled departure | 90 | 2 | 2 | 1 |
| Actual departure | 866 | 349 | 343 | 42 |
| Scheduled arrival | 179 | 141 | 126 | 51 |
| Actual arrival | 824 | 328 | 322 | 43 |

These are coverage and model-stability counts, not detected-error counts or estimates of correctness. The 793 proposals refer to row/attribute cells; the flight counts in the last column overlap across attributes. All 116 common scheduled-arrival proposals from v1 appear identically in v3, with ten additional scheduled-arrival previews. V1 and v2 files and artifact hashes were preserved.

## The numerical correction

V2 allowed low and negative source skill but conservatively included every converged start in its uncertainty envelope. Some of those stationary fits had much worse penalized objectives. V3 instead chooses the exact maximum finite converged penalized objective among starts -2, zero, and 2 **separately** for each fixed prior/unseen-state/bootstrap scenario. An exact objective tie selects the earliest declared start. This selection never reads candidates, proposal counts, v1 overlap, or benchmark outcomes.

The [v3 protocol](latent_consensus_v3_protocol.md) was frozen before dirty fitting. The likelihood and priors remain v2's categorical log-advantage model: `q = exp(theta)/(exp(theta)+K)` with Gaussian means -log(2), zero, and log(2), standard deviation 2, and source skill permitted negative. All support, 0.98 candidate-weight, 0.20 margin, 0.95 leave-group-out, source grouping, and missingness rules remain unchanged.

There are 17 selected sensitivity fits per fold: three prior means, two additional unobserved-state priors, and twelve entity-cluster bootstrap samples. Each scenario runs all three starts. Across four attributes, two source assumptions, and five entity folds, the run archives 2,040 stationary fits and 680 selected fits. Four starts did not converge within 200 iterations; every scenario had another converged fit, so no fold lacked an eligible selection. No numerical-boundary rejection occurred.

The selected-start counts are 403 from zero, 256 from 2, and 21 from -2. In 664 of 680 scenarios, the best and runner-up converged objectives differ by at most 1e-8, generally because starts reach the same fitted solution. The smallest checked EM objective increment is about -2.3e-12, consistent with numerical roundoff. Selecting among three starts does not establish a global optimum; close objective ties and all source parameters remain inspectable.

A new synthetic case explicitly creates different stationary orientations, verifies monotone objectives and an objective gap, and checks exact maximum selection, deterministic ties, and exclusion of nonconvergent endpoints. The full focused suite passed 15 tests before the v3 dirty fit. The previous tests cover multiclass chance below one half, negative source skill, copied sources, orientation abstention, entity holdout, duplicate invariance, source conflicts, missingness, and preservation of strings.

## What the sensitivity supports

For rows supported independently under both source assumptions, the winning candidate strings never disagree: 284 rows for scheduled departure, 922 for actual departure, 1,039 for scheduled arrival, and 816 for actual arrival. Common proposals are a subset of these rows whose original strings differ from the common candidate. Different acceptance coverage under the two assumptions remains substantial.

Every common proposal passes all 17 selected sensitivity fits under each source assumption. In each fit it retains the same winner after removing any one active source group. This is a finite stress test, not measured calibration. Removing a group changes that item's likelihood contribution; it does not refit global source parameters. All parameters used for a held-out flight were fitted on other flights. Source groups, support pruning, and bootstrap weights also exclude that flight's observations.

Allowing weak sources resolves v1's lower-bound rejection without asserting all sources are reliable. For example, selected actual-departure fits have 9 to 12 negative-skill sources under independent names and 4 to 5 under clusters. A negative skill describes below-chance reporting in this conditional categorical model; it cannot establish that a real source is wrong, adverse, or independent. Some source signs still change across sensitivity fits, especially actual-arrival clusters, and these unstable conclusions remain visible as abstentions.

## Limits visible in the proposals

A descriptive clock-prefix check finds 35 common proposals with the same 12-hour clock prefix, 745 with different clock prefixes, and 13 from non-clock literals. The latter are six `Not Available` and seven `Contact Airline` values. Version 3 still treats only blank/whitespace as missing, so those 13 are examples of unresolved sentinel semantics, not verified error corrections. Prefix equality can also conceal meaningful date annotations or display information. This check was not used to alter proposals or tune thresholds.

Different clock strings can denote differing observation dates, updates, time zones, event types, display conventions, or erroneous values. The raw table does not identify those explanations. A common model winner under two source assumptions does not distinguish them. Near-agreement clusters can merge independently correct sources or miss partially correlated sources. Shared systematic errors can survive both assumptions.

The adaptive per-item candidate vocabulary and single additional unknown state remain approximations; they are not a complete generative model over unobserved strings. Penalized point fits and finite bootstrap ranges do not integrate all parameter uncertainty. Conditional weights are not empirical correctness rates and provide no false-repair guarantee. A higher penalized objective only identifies a better stationary solution within the disclosed model and starts.

## Reviewable artifacts and next experiment

- [Frozen manifest](latent_consensus_flights_preview_v3/pre_fit_manifest.json) records dirty fingerprint, assumptions, parameters, and hashes of the model, helper modules, runner, and protocol before fitting.
- [Summary and all common proposal strings](latent_consensus_flights_preview_v3/summary.json) retain original value, candidate, row, attribute, source, and raw flight identifier.
- [Sensitivity diagnostics](latent_consensus_flights_preview_v3/sensitivity_diagnostics.json) report source signs, abstention reasons, v1 overlap, objective selection, convergence, and clock-prefix diagnostics.
- [Artifact manifest](latent_consensus_flights_preview_v3/artifact_manifest.json) verifies exact original strings, candidate support in raw dirty observations, frozen hashes, and maximum-objective selection. Compressed per-mode artifacts archive every row decision, every stationary fit, objective trace, and selected-start index.

The supported next step is a separate, frozen comparison of this proposal artifact, while keeping it outside live inference. Source lineage and snapshot/field semantics would strengthen interpretation without requiring cell labels. A typed time and missing-sentinel contract should be specified before any new fit; it should preserve dates and original strings. A held-source predictive experiment needs a candidate vocabulary fixed independently of withheld claims, and would still measure agreement with dirty observations rather than correctness. Any eventual benchmark evaluation must be isolated from this design branch and must not feed threshold changes back into these frozen experiments.
