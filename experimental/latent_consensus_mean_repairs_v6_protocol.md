# Preregistered v6: fixed v5 detection with mean-model repair selection

This isolated statistical repair ablation keeps the v5 mean ED array EXACTLY fixed and starts corrections from the same frozen source-v2 loop prior. It decomposes repair policy from model averaging before any v5 benchmark outcomes have been supplied. No refits, new sources, metadata catalogs, labels, evaluation/failure files, LLM calls, or live promotion are permitted. All v3/v4/v5 code, tests and artifacts remain immutable; all new tests live in `tests/test_latent_consensus_repairs_v6.py`.

## Frozen eligibility and conditional candidate weights

Use exactly the same v5 modeled-cell eligibility mask and frozen v3 full-observation posteriors. Unsupported, conflicted, untrained, missing, inactive and out-of-vocabulary cells receive no new repair choices. The ED scores are copied bitwise from the frozen v5 mean prediction; neither a repair decision nor a guard can change them.

For each eligible cell, score only observed raw strings present in BOTH source-assumption vocabularies. An absent string receives no fabricated zero weight. Each source assumption has weight 1/2 and each of its 17 selected full-observation fits has weight 1/17. Do not refit, deduplicate, renormalize shared values, change vocabulary, or mix leave-group-out counts into the mean. Display excluded nonshared raw strings and their aggregate conditional mass separately.

Rank at most three distinct NONORIGINAL shared observed strings by descending mean conditional weight, then raw-string lexical order. Only the unique top nonoriginal value can be chosen; never fall back to the second/third after a veto. Require its mean weight at least 0.8 and its margin at least 0.2 above the largest competing shared value (including original) or residual unknown/unshared mass. Treat weight differences at most 1e-12 as a numerical tie, which abstains. Residual mass is calculated directly from each mode's unobserved bucket plus nonshared categories; no individual out-of-vocabulary probability is invented. These are conditional sensitivity weights, not calibrated correctness probabilities or exact Bayesian model marginalization.

## Representation and semantic preservation

Inspect the exact original and candidate representations ONLY. Do not veto a bare-clock change merely because an unrelated peer has an unavailable/status/missing/dated/annotated claim. Peer claims remain displayed noisy evidence under unchanged fits.

A conservative parser permits only whole-string bare 12-hour clocks with explicit a/p meridiem (periods/spacing/case variants) or whole-string bare 24-hour clocks. Leading/trailing whitespace is ignored for parsing only; output strings are never normalized. A diagnostic parser may expose a single embedded clock, but embedded/annotated strings cannot authorize changes.

If original and candidate denote the same wall-clock minute, abstain on the proposed change even when strings or annotations differ. If either changed representation contains extra dates, statuses, event annotations, timezone text, multiple clocks, or uninterpreted tokens, abstain. Nonclock unavailable/contact literals are not assumed erroneous or silently imputed. A repair can change one bare clock to a genuinely different bare clock using the exact observed candidate string. This policy never introduces/drops annotations, chooses timezone aliases, or asserts snapshot/catalog evidence. Existing multi-column rules veto directly encoded tuple contradictions; no new semantic rule is inferred from peer formatting.

## Coherent source-prior baseline

Copy the source-v2 prior repair map. If an inherited repair has fixed v5 ED score below 0.5, explicitly remove it, restoring the dirty original at that coordinate. Record this coherence rollback; never normalize ED scores or retain a repair outside detection. Other unsupported repairs retain their exact prior state.

Sandbox-check this coherent baseline against the ORIGINAL source-prior repaired tuples. If removing an ED-incompatible repair itself introduces a new multi-column rule violation or new runtime failure, stop without a valid prediction: no safe coherent prior map exists under the fixed constraints. Do not change ED, thresholds, or evidence to resolve it. This fail-closed behavior is predefined.

## Four counterfactual dimensions and individual gate

Build DirtyIndex once from original dirty strings/counts, with frozen pass-2 dependencies and the same dirty-only discovery. Validate generated pass-2 rules in the existing sandbox at the same strict-below-5%-runtime-error acceptance threshold. No predicted value becomes a training observation.

For every displayed top-three candidate, calculate the unchanged four dimensions against the original dirty tuple using the exact v3-feedback feature function: target rule-satisfaction delta, leave-one-out FD-support delta, tanh NMI-vicinity delta divided by four, and tanh domain-support delta divided by four. Use weights [2.2, 1.8, 0.65, 0.35] and correction confidence `sigmoid(logit(mean_weight)+weighted_delta)`, clipping logit inputs to [1e-6,1-1e-6]. Require correction confidence at least 0.8.

Separately evaluate each candidate in its coherent source-prior tuple. Hard-veto a new multi-column rule violation relative to that parent tuple, or any runtime failure in the candidate tuple. A representation, ranking, fixed-ED, confidence, or rule veto preserves the coherent prior repair state; it does not search a lower-ranked value or change ED. Only the top candidate passing every gate adds/replaces an exact raw repair.

## Simultaneous guard and rollback

Apply admissible top choices simultaneously. For any changed row, sandbox the entire resulting tuple against the coherent parent tuple. A newly failing multi-column rule or any runtime failure rolls back ALL new repair choices in that row to the coherent prior map. Initial ED-coherence removals are not undone. Assert the final tuple introduces no new multi-column failure or runtime failure relative to either the coherent parent or the original source-prior parent. Assert every final repair lies inside unchanged v5 ED.

## Tests, freezing, and review evidence

Before real application, test shared-vocabulary ranking, absent alternatives, numerical ties/margins, top-three bounds/no lower fallback, same-clock formatting vetoes, changed-cell annotations/status/timezones, unrelated annotated peers not acting as vetoes, individual confidence/multi/runtime gates, simultaneous rollback, ED-coherence removals/fail-closed conflicts, unsupported preservation, and bitwise fixed ED. Do not edit old test modules to make these pass.

Freeze both protocols/source hashes and exact input bytes before generating output under `experimental/latent_consensus_mean_repairs_v6`. Save the standard prediction, original strings, per-cell mean top-three weights, all four deltas, representation decisions, rule outcomes, coherence/joint traces, and full dirty source-claim contexts embedded in a prepared verifier bundle. No verifier acceptance is required or requested here. All source/event/snapshot/independence/finite-vocabulary hypotheses remain explicitly conditional. Report paths, hashes and inference counts only; do not evaluate or claim accuracy/calibration.
