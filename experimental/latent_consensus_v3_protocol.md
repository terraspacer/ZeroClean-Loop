# Preregistered v3 numerical protocol: select the best penalized objective

Preserve v1 and v2 byte for byte. V3 changes optimization interpretation only. The categorical log-advantage likelihood, Gaussian priors, source hypotheses, preprocessing, and candidate acceptance thresholds remain exactly those of v2. This is an explicitly hypothetical dirty-only investigation, with no benchmark truth, evaluation results, labels, prior external predictions, or desired coverage used for selection.

## Fixed maximum-objective rule

For every fixed training fold, source assumption, Gaussian prior mean, unobserved-state prior, and bootstrap weighting, run all three declared source-skill starts in order -2, zero, 2. Select the finite, converged fit with the exact maximum penalized objective. Break exact objective ties by earlier start order. Do not compare objectives across different priors, bootstrap datasets, or source assumptions. Keep the full objective trace, convergence state, parameters, and selected-start index for every start.

If no start converges for a scenario, retain its diagnostics and reject that fold. A nonconvergent lower-objective start does not invalidate another converged fit. A global optimum is not proven by comparing three starts. Close objective ties are reported as numerical uncertainty; the prescribed exact maximum remains deterministic. All source parameters still lie in [-6, 6] solely for numerical stability, and the preexisting numerical-boundary rejection remains.

Select three fits for the Gaussian means -log(2), zero, and log(2), with standard deviation 2 and unseen prior 0.10. Select two additional fits for unseen priors 0.05 and 0.20 at Gaussian mean log(2). Select twelve entity-cluster bootstrap fits at baseline priors. Each of these 17 scenarios runs three starts, giving 51 stationary fits per fold, with 17 selected fits entering the uncertainty checks. Bootstrap samples and folds retain seed 1701.

## Unchanged acceptance and source sensitivity

Retain at least 30 training entities, 20 items per source group, triple overlap 12, at least 10 conflicting items, and at least four groups for a candidate. Near-agreement grouping still requires overlap 12 and agreement 0.98. All contexts of a base entity share a fold. Source grouping, support, and model fitting remain restricted to training entities.

Require the same exact observed-string winner across all 17 selected sensitivity fits. Keep minimum conditional weight 0.98, margin 0.20, and leave-one-source-group-out conditional weight 0.95 with unchanged winner in every selected fit. Compare independent source names with conservative near-agreement clusters. A final common proposal must pass independently with the identical string under both assumptions. Do not relax these thresholds or select a start using proposal count, agreement with v1, or any benchmark outcome.

## Synthetic validation before real fitting

Before fitting dirty Flights, construct a synthetic population of repeated same-source/entity observations with multiple latent stationary orientations. Run the categorical likelihood from all three starts under one fixed prior, verify monotone conditional objective updates, and verify that the objective values differ by more than numerical roundoff. Check that the selector chooses the exact largest converged objective, handles an exact tie deterministically, and excludes a nonconvergent fit even if its last objective is larger. Candidate strings and coverage must not enter this selection.

Retain the existing 14 synthetic tests for multiclass chance, adverse and correlated sources, EM math, orientation abstention, duplicate invariance, entity holdout, conflicting sources, and string provenance. Only after these checks pass, freeze this protocol, source dependencies, parameters, and dirty fingerprint and run all four Flights time attributes under both source assumptions.

## Interpretation fixed in advance

Selection of a better stationary objective fixes the v2 practice of treating deeply suboptimal modes as equally eligible fitted models. It does not establish truth orientation, source independence, a global optimum, or empirical accuracy. Report objective spreads and all start diagnostics, and compare the selected solutions across prior means, bootstrap fits, unobserved-category priors, source assumptions, and held-out source-group sensitivity. Preserve every abstention.

Dates, snapshot identity, source lineage, exact-string versus typed time semantics, missing literals, and multi-valued attributes remain unresolved. Conditional model weights are not correctness probabilities. Any v3 proposal is an unpromoted next experiment, and no claimed improvement across the six benchmarks follows from this numerical correction.
