# Preregistered v4: separate detection uncertainty from correction uncertainty

V4 is a separate statistical detection ablation using the SAME frozen v3 categorical fits. It performs no refit, LLM call, benchmark evaluation, or live promotion. It must preserve v3 code and artifacts bitwise. The classifier threshold is fixed at 0.5 by the binary decision definition, not selected using coverage or benchmark outcomes. All reported weights remain conditional model sensitivity weights, not empirically calibrated correctness probabilities.

## Fixed input and correction policy

Use the original dirty Flights table, the exact frozen loop prior already authorized for v3 feedback, the frozen `latent_consensus_flights_preview_v3` fits, and the frozen `latent_consensus_feedback_v3` correction output. Preserve exact bytes/hashes and all original strings. The v4 ED probability grid starts from the original frozen loop prior, not a new run or a refitted detector.

The correction map is copied EXACTLY from the frozen v3 feedback prediction. No correction value, coordinate, strict-common-winner requirement, confidence threshold, counterfactual gate, or joint rollback decision changes in v4. Its v3 provenance and complete audit remain referenced. An ED-positive is permitted to have no repair. A detection-only conclusion never generates a new correction candidate or modifies that fixed correction map.

## Eligibility retained from v3

Both independent-source-name and conservative-cluster assumptions must be eligible. Keep frozen source support, overlap, three-view training, fold convergence, numerical-boundary, and semantic eligibility. Require at least four active trained groups in the full held-out item and at least three after leaving one group out. Use the frozen training-fold mappings and fits; never discover new groups or fit new parameters.

Reject missing/whitespace originals, incomplete entity/context keys, missing source IDs, conflicted same-source/item observations, sources absent from training, inactive/conflicted source groups, and originals absent from either valid frozen observation vocabulary. The raw source's active group claim must equal the original string. Unsupported or ineligible cells retain their exact prior ED probability. Never give an out-of-vocabulary original a made-up zero weight or identify it with the unobserved-category bucket.

Single-snapshot identity, exact-string single-valuedness, conditional source/event independence, stationary source skill, ignorable missingness, finite candidate vocabulary, and Gaussian orientation preferences remain the SAME disclosed hypotheses as v3. Missing sentinels and time/date/zone semantics remain unresolved; only blank/whitespace is treated as missing.

## Posterior sensitivity and binary decision rule

For each eligible item and each source assumption, freeze the observed candidate vocabulary BEFORE leave-one-group-out removal, exactly as v3 did. Keeping a previously observed category during removal does not create a new candidate. These are conditional sensitivity checks and must not be described as held-source predictive calibration.

For every one of v3's 17 selected prior/unobserved-prior/bootstrap fits, calculate the categorical posterior for the full item and after removing each source group. Do this even when the winning alternatives differ. Do not require a 0.98 alternative, a unique winner, the 0.20 repair margin, or 0.95 leave-out repair confidence for detection. Retain all source and fit eligibility checks above.

For the original exact string, let L be the minimum and U the maximum weight across BOTH source assumptions and ALL selected full/leave-group-out scenarios. Conditional error-weight bounds are `[1-U, 1-L]`.

- If the error LOWER bound is strictly greater than 0.5, flag dirty and set the ED score to that lower bound.
- If the error UPPER bound is strictly less than 0.5, prune to clean and set the ED score to that upper bound.
- If the interval straddles or touches 0.5, retain the exact prior ED score.

Record every interval and its minimizing/maximizing scenarios. A category can be very unlikely while two alternative repairs split their mass; that supports a binary detection decision without supporting an exact repair. This is a Bayesian decision distinction within the disclosed model, not evidence of real-data accuracy.

## Synthetic checks before real application

Construct a frozen multiclass example where the original is unlikely in every full/leave-group-out scenario but two alternatives split mass and change winners; assert detection flags while strict v3 correction abstains. Construct an uncertain-original example whose interval crosses 0.5; assert prior preservation. Test strict boundary behavior at 0.5 and reject missing, conflicted, untrained, inactive, and out-of-vocabulary sources/originals without fabricating scores.

Retain and extend the meaningful synthetic regression suite for v3 feedback: individual multi-rule veto, counterfactual runtime veto, low-confidence correction veto using all four dirty-derived dimensions, joint interaction rollback to the actual prior (not merely the dirty row), same-row unsupported-cell preservation, and retention of ED evidence when correction is blocked. Do not edit frozen v3 code to conceal failures; report any failure and stop dependent real application until resolved without rebinding the test target.

## Frozen outputs and later verification

Freeze this policy, code/dependencies, input hashes, and synthetic test source before real application under a new `experimental/latent_consensus_detection_feedback_v4` directory. Save the standard Prediction JSON, the original dirty strings, per-item scenario posterior traces, per-cell eligibility/interval/decision audit, and exact inherited correction provenance. Preserve v3 hashes before and after execution.

Prepare a separate evidence bundle for later independent review of changed ED decisions/probabilities, with exact originals, coordinates, observed vocabularies, interval endpoints, scenario references, prior/final tuples, and inherited correction state. No model is contacted here. Validate output schema, interval arithmetic, unchanged unsupported/uncertain ED cells, exact v3 correction-map equality, hashes, and guard provenance. This is structural validation, not an accuracy evaluator. Final handoff lists only artifact paths and hashes.
