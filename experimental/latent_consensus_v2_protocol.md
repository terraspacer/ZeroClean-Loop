# Preregistered v2: categorical source log-advantage

This mathematical sensitivity experiment is separate from frozen v1. Its implementation, protocol, and parameters must be frozen before any real dirty-table fit. First validate the algebra and behavior on synthetic multiclass data with good, weak, adverse, and copied sources. No benchmark labels, masks, results, existing predictions, or external answer catalogs are permitted. No live inference changes are authorized by this experiment.

## Why change the parameterization

Version 1 forces a source's exact-string match probability above one half. In a domain with D categories, random categorical reporting instead matches with probability 1/D. Being more informative than random can therefore coexist with a match probability below one half. A fixed above-one-half lower bound is especially restrictive when an item's candidate cardinality changes.

Let K be the number of observed exact-string candidates, and add one unobserved category as in v1, so D equals K plus one. Give source s a log-advantage theta_s:

```text
P(y = z | theta_s, K) = exp(theta_s) / (exp(theta_s) + K)
P(y = another particular category | theta_s, K) = 1 / (exp(theta_s) + K)
```

The odds of the correct category against a particular incorrect category equal exp(theta_s). Zero skill gives chance 1/(K+1); positive skill exceeds chance; negative skill favors any particular incorrect category over the correct one. For theta equal log(2) and K equal 3, the match probability is 0.4, which exceeds chance 0.25 while remaining below one half. The same theta gives different probabilities as cardinality changes. This parameter is a categorical likelihood contrast, not an empirical correctness estimate.

For fixed item vocabulary, a source's conditional log likelihood is `1[y=z] * theta_s - log(exp(theta_s) + K)`. The E-step is a weighted vote with possibly negative weights: add theta_s to the candidate the source claimed, then normalize with the item prior. The unobserved category receives no direct vote.

Put a Gaussian penalty on each theta_s, with baseline mean log(2) and standard deviation 2. Its mean supplies a soft positive orientation preference; it does not constrain any source, the source average, or the inferred truth sign. The M-step solves:

```text
sum_items weight * (P(z = observed_y | claims) - q_s(K))
    - (theta_s - prior_mean) / prior_sd^2 = 0
```

The derivative of this expression is strictly negative because of the Gaussian term, giving a unique conditional M-step maximizer. This does not make the joint likelihood convex or the full model identifiable. A bounded bisection solver on [-6, 6] supplies numerical stability. The objective and its trace are retained for checking the update.

## Fixed sensitivity protocol

All v1 support, grouping, cross-fitting, candidate, and leave-group-out thresholds remain unchanged: seed 1701, five entity folds, at least 30 training entities, 20 items per group, triple overlap of 12, 10 conflicting items, four decision groups, near-agreement threshold 0.98, candidate conditional weight 0.98 and margin 0.20, and leave-group-out weight 0.95. Empty strings remain missing; other literals and original strings remain untouched. No typed normalization is introduced.

Replace only the per-source positive q restriction and Beta penalty with the above categorical log-advantage and Gaussian penalty. There is no orientation-boundary rejection at q=0.5. Reject a fold if any fit fails the unchanged maximum 200 EM iterations / maximum skill change 1e-7 convergence rule, or if at least half of source skills hit numerical bounds. Do not loosen these criteria to increase coverage.

Run all three starts (-2, 0, 2) at all three Gaussian prior means (-log(2), 0, log(2)), with standard deviation fixed at 2. The negative prior mean is a deliberate orientation stress test. Retain all nine fits in sensitivity decisions without selecting a preferred start from benchmark performance. Add unseen-state priors 0.05 and 0.20 at baseline mean log(2), start 2. Add twelve entity-cluster bootstrap fits at baseline mean, unseen prior 0.10, start 2. These are 23 conditional sensitivity fits per fold. Every fit must agree on a candidate and satisfy the same thresholds before a row is proposed. A soft prior or preferred initialization alone cannot justify a candidate.

Source grouping and all parameters are learned on training entities only; all contexts of an entity share a fold. Compare the same two source assumptions as v1: distinct source names independent, or near-agreement components independent. Final common previews require the exact same proposed string to pass independently in both. Preserve disagreement and abstention counts without claiming measured error rates.

## Synthetic validation before dirty fitting

Check the chance formula at more than two categories and multiple cardinalities, the penalized M-step derivative, and monotonic objective updates. Generate a three-category population with a mixture of high, below-one-half, and adverse match probabilities; include an exact copied source. Verify that weak/adverse sources can have negative fitted skill and that copied sources collapse under the existing grouping test. Do not require any desired number of proposals.

Construct an orientation ambiguity example where reversing skills changes the winning state, including an unobserved-state winner; verify that disagreement across those fits abstains. Preserve the previously tested duplicate invariance, entity-disjoint fitting, source conflicts, temporal scope, missingness, and exact-string provenance through shared v1 preprocessing. Synthetic labels are known by construction only and are never used as a real-table tuning target.

The open-world vocabulary limitation remains: observed candidates are a transductive, data-dependent vocabulary, and one unobserved category does not specify an arbitrary-string generative distribution. Multiclass synthetic inference checks validate algebra and qualitative behavior under this approximation; they cannot prove parameter recovery or real-data calibration.

## Preregistered Flights hypotheses and interpretation

After synthetic validation, fit all four time attributes separately under the same explicitly hypothetical event, snapshot, stationarity, missingness, and source assumptions disclosed in v1. No new source lineage or date semantics is assumed to have been established. A successful fit only demonstrates conditional compatibility with a more flexible likelihood.

Report nonconvergence, negative/near-zero source skills, sign changes across priors/starts, winning-string changes, unobserved-category winners, leave-group-out failures, and changes under source clustering. Compare final stable candidate strings with the preserved v1 previews only after freezing and running v2; do not tune to overlap with v1 or desired coverage. Any conclusion that fails source, orientation, or leave-group-out sensitivity remains unsupported. Do not promote either version to live inference or interpret a model weight as empirical correctness.
