# Preregistered statistical consensus feedback ablation

This is a separate, unverified statistical feedback ablation atop the exact frozen Flights loop prediction. It is not the fully verified main method and is not a completed target result. No refitting, LLM verification, evaluator, benchmark truth, clean table, error mask, or outcome-based selection is permitted. Later independent verification receives a prepared bundle only; this run must not call a model.

## Authorized immutable inputs

Read only the original dirty Flights table through `zeroclean.data.load_dirty('flights')`, the frozen v3 preview, pure source code, and these exact prior inference artifacts:

```text
artifacts/closed_loop_v2/flights/4e0f4d0f82ada5f10a82bd9fb57a613dd8bf28110ef7ca66633ea38717c31247/prediction.json
artifacts/llm/flights/pass2.json
experimental/latent_consensus_flights_preview_v3/
```

Their bytes, all accessed preview files, source dependencies, dirty fingerprint, exact dirty strings, this protocol, and policy parameters must be frozen under a new `experimental/latent_consensus_feedback_v3/` directory before applying feedback. Never overwrite the prior inference or preview. Output probabilities and repairs must retain the standard Prediction JSON schema for a later evaluator after root validation; do not run that evaluator here.

## Compatible statistical evidence, without refitting

Use only row/attribute decisions with status `consistent` or `propose` in BOTH independent-source-name and conservative-cluster v3 files, with the same exact winning string. Both must pass the original support, 0.98 conditional winner-weight, 0.20 margin, and 0.95 leave-source-group-out gates. Unsupported, ambiguous, absent, or incompatible evidence leaves the prior probability and repair unchanged. Verify dirty fingerprints, file hashes, coordinates, original strings, selected model convergence, and stored gate values. Do not invent confidence for consistent rows absent from compatible frozen evidence.

Reconstruct original-string weights from the frozen selected source skills and item priors, using only original dirty claims and the frozen training-fold source mapping. Never call a fitter or update source groups. Across all selected fits and both source assumptions, retain the original-weight interval [L, U] and minimum winning-string weight W. These are conditional model weights; they are not measured correctness probabilities.

For a supported original, set the ED score to `1-L`, the conservative upper endpoint of conditional error weight. For a supported different value, set it to `1-U`, the conservative lower endpoint of conditional error weight. Keep the ED threshold 0.5. A consistent original can thus prune a prior positive; a stable different string can recover a prior negative. Preserve this ED feedback even when an exact repair fails a counterfactual gate or is rolled back. Archive interval endpoints and all underlying model weights.

## Four counterfactual dimensions and fixed correction gate

Build one DirtyIndex on the original dirty table, with the pass-2 dependencies and the existing dirty-only dependency discovery. Never rebuild it from predictions or tentative repairs. Validate pass-2 rules using the existing sandbox and default strict-below-5%-runtime-error acceptance rule; retain the complete acceptance report. Execute accepted rules only in the sandbox. A candidate runtime failure always blocks its repair.

For each compatible candidate, compute the same four dimensions as `zeroclean.loop.build_candidates`, comparing a one-cell candidate against its original dirty tuple and original dirty counts:

1. Rule satisfaction: original minus candidate target-rule violation count, divided by the number of target rules (minimum denominator one).
2. FD support: candidate conditional support in the altered tuple minus original conditional support, with the original observation excluded by DirtyIndex.
3. NMI vicinity: `tanh((candidate_vicinity - original_vicinity)/4)`.
4. Domain support: `tanh((candidate_domain_support - original_domain_support)/4)`.

Use the existing weights [2.2, 1.8, 0.65, 0.35]. Candidate correction confidence is `sigmoid(logit(W) + weighted_delta)`, with logit inputs clipped to [1e-6, 1-1e-6]. Keep repair threshold 0.8. Do not use these features as additional empirical correctness labels.

Block a candidate that introduces any multi-column rule violation relative to the original dirty tuple, produces any accepted-rule runtime failure, or fails correction confidence 0.8. Single-rule changes affect the rule-satisfaction dimension and remain visible; the hard introduction veto concerns multi-column rules. For a supported original, an accepted correction action removes the prior repair; for a supported different string it adds or replaces a repair with that exact string. If an individual gate fails, preserve the exact prior repair state at that coordinate. ED feedback remains independent.

## Simultaneous guard and rollback

Apply individually accepted correction actions simultaneously atop the prior repair map. On every row with a changed repair state, sandbox the complete resulting tuple, including untouched prior repairs. Relative to the original dirty tuple, forbid newly failing multi-column rules or any runtime failure. If the row fails, roll back **all feedback correction actions in that row to the exact prior repair state**, retaining the statistical ED feedback. Record every intended, applied, blocked, and rolled-back action.

The prior inference is immutable and unsupported cells retain it. If a prior row already violates the guard, record that fact; rollback can preserve that preexisting state but must not add any new violation or runtime failure relative to it. After rollback, assert no newly failing multi-rule ID or new runtime failure relative to the prior repaired tuple on any modified row. No rollback uses benchmark outcomes.

## Semantic assumptions and later verification

Retain all v3 hypotheses explicitly: raw flight plus attribute denotes one exact-string snapshot event; base events and source groups are conditionally independent under the chosen source assumption; source log-advantage is stationary; missingness is ignorable; the finite candidate vocabulary and Gaussian orientation priors are appropriate. Dates, time zones, field meanings, source copying, alternate representations, and missing sentinels remain unresolved. The four dirty-derived counterfactual dimensions and generated rules can share systematic errors with consensus.

Prepare a separate hashed candidate/evidence bundle for every coordinate whose probability or repair state changes. Include exact coordinates and originals, original/prior/final tuples, all statistical weights, candidate strings, four counterfactual dimensions, rule outcomes, joint guard decisions, input hashes, and explicit `independent_verification_status='not_run'`. The bundle can later support independent LLM review; producing it is not verification. Do not send it to a model in this variant.

Before the real run, synthetic tests must check positive recovery, consistent-original pruning, unsupported-cell preservation, multi-rule/runtime vetoes, joint interaction rollback, and retention of ED evidence when correction is blocked. Final summary reports only artifact paths and hashes, with no accuracy claims.
