# Preregistered optional v5: independent-marginals proxy expected-F1 decisions

This optional artifact is separate from the uniform fixed-0.5 mean-posterior variant. It uses exactly the same frozen averaged conditional error weights and eligibility. It does not consume correctness labels, benchmark metrics, failure cases, evaluator outputs, or empirical threshold sweeps. Freeze this protocol before either v5 prediction.

## Explicit artificial conditional decision model

Let U contain only cells eligible under both frozen source assumptions. For each i in U, let p_i be the uniformly averaged conditional error weight specified in the mean-aggregation protocol. Define an ADDITIONAL hypothetical product distribution `Y_i independently ~ Bernoulli(p_i)`. This is an **independent-marginals proxy**. Repeated observations of the same entity/attribute share latent truth, so their error indicators are coupled under the actual fitted categorical model. Averaged weights are not calibrated truth probabilities, and this product distribution is not that model's Bayesian joint posterior. No claim of globally optimal decisions under the fitted shared-entity model or full benchmark is permitted.

Optimize the single scalar objective

```text
J(D) = E_product[ 2 * sum_{i in D} Y_i / (|D| + sum_{i in U} Y_i) ],
       D subset U, D contains R,
```

where R is the set of frozen v3 repaired cells lying in U. Define F1(empty, empty)=0. This is the exact expectation of the ratio under the stated product model, NOT the ratio of expected counts and NOT expected benchmark performance. Unsupported cells are excluded from the objective and retain their prior ED state subject to the same explicit repair-consistency gate. The fixed v3 correction map is never modified.

## Analytic ranking and expectation, not an empirical threshold sweep

For a fixed cardinality k, any free selected cell with probability p_j below an unselected free p_i can be exchanged. Conditional on all other labels, the expected F1 improvement is

```text
2 * (p_i - p_j) * E[1 / (k + 1 + T_other)] >= 0.
```

Thus, with R mandatory, the optimal set at each size is R plus the largest remaining probabilities. Sort free cells by descending p_i, breaking equal probabilities by ascending (row, column) coordinate. Consider only these analytically derived nested sets; no observed labels or metric computation is involved.

Let `G(z)=product_i ((1-p_i)+p_i*z)` be the Poisson-binomial generating polynomial for total T. For each item i, obtain the leave-one-item polynomial `H_i(z)=G(z)/((1-p_i)+p_i*z)`. Define `A_D(t)=sum_{i in D} p_i * [z^(t-1)] H_i(z)`. Then

```text
J(D) = 2 * sum_{t=1..|U|} A_D(t) / (|D| + t).
```

Compute G once by coefficient dynamic programming. Compute H_i with stable forward synthetic division when p_i<=0.5 and backward division otherwise, using special exact shifts for p_i=0 or 1. Accumulate A_D along the mandatory-plus-sorted-free path and select the largest computed J. This is mathematical optimization using frozen soft weights; it is not an empirical threshold sweep.

## Fixed numerical and tie policy

Use float64. Check G mass error at most 1e-10. During synthetic division, permit negative coefficient roundoff no smaller than -1e-10, explicitly count and clip those tiny negatives, and require leave-one-item mass error and polynomial reconstruction residual at most 1e-8. Do not otherwise normalize the coefficients. Any numerical check failure stops this optional output without changing the mean-posterior artifact or choosing a different objective.

Treat objective differences at most 1e-12 as numerical ties and choose the set with fewer optional positives. Coordinate order resolves a partial probability tie at the cutoff. Archive every conditional utility along the analytic path, the selected set/cardinality, numerical diagnostics, and the threshold/rank/tie description. A global scalar cutoff can require coordinate tie handling or mandatory-repair exceptions; report these explicitly.

## Prediction encoding and coherence

The optional standard Prediction JSON encodes the selected modeled-cell decisions as 0/1 DECISION SCORES so its usual 0.5 classifier reproduces D. Those entries are not posterior probabilities. Preserve the original averaged p_i separately in shared evidence. Unsupported cells retain the original frozen loop score; if a fixed repair outside U would be negative, use its exact positive frozen v3-feedback ED score under the preregistered coherence gate. Repaired cells in U are mandatory members of D. Assert all repairs remain inside detection and the correction map is exactly v3's.

## Synthetic validation before prediction

For small synthetic probabilities, enumerate every label realization to verify the exact conditional expected-F1 formula and every admissible decision set to verify the mandatory/top-k optimizer. Include p=0/1, equal probabilities, empty scope, mandatory low-probability repairs, and a case where E[F1] differs from F1 of expected counts. Check preservation of unsupported scores, exact v3 repairs, and the declared decision-score semantics. No frozen v3 code is edited to make tests pass.

This optional utility policy is a model-conditional decision experiment, not an optimized empirical F1 result. Its additional product-label assumption and restricted eligible-cell objective must remain explicit in all outputs. No evaluator, LLM verification, or promotion is performed.
