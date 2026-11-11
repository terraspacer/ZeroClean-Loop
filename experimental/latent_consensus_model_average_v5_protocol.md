# Preregistered v5: equal-weight conditional-posterior averaging

This is an unpromoted statistical ED sensitivity ablation. Use only the original frozen loop prior, frozen v3 correction prediction, and existing v4 posterior contexts/evidence derived from frozen v3 fits. No refitting, new sources, vocabulary changes, LLM calls, labels, evaluation/failure files, or empirical metric information is permitted. Preserve v3/v4 bytes and their input hashes.

## Uniform aggregation before any new prediction

For each cell already eligible under BOTH v4 source assumptions, take the original-string weight in the FULL-observation scenario of each of the 17 selected fits. Give each selected fit weight 1/17 within its source assumption, and give independent source names and conservative clusters weight 1/2 each. Thus each of the 34 full-observation weights has coefficient 1/34. Conditional error weight is one minus that average original weight.

The 17 scenarios are the original three Gaussian-prior centers, two additional unobserved-state priors, and twelve entity-bootstrap fits. They are equally weighted individual scenarios; this deliberately gives the bootstrap family 12/17 of within-assumption weight. Do not deduplicate numerically identical fits, add likelihood factors, rebalance scenario families, or reinterpret the two source assumptions as independent observations. This is a transparent scenario average of conditional weights, not calibrated correctness or exact Bayesian marginalization over models.

Assign that same averaged error score to every eligible cell and use the fixed classifier `score >= 0.5`. An exact 0.5 tie is positive by this declared convention. Preserve the original frozen loop prior for unsupported, conflicted, untrained, missing, inactive, or out-of-vocabulary cells, with the correction-consistency exception below. Never invent an original score absent from a valid frozen vocabulary.

## Leave-group-out and worst-case diagnostics

Keep the observed vocabulary, source mappings, selected fits, full/leave-one-group-out scenarios, and v4 eligibility unchanged. Only full-observation scenarios enter the uniform point estimate. Otherwise, assumptions with more groups would implicitly receive more weight.

For each assumption and omitted source group, average its 17 original/error weights. Report these group-level averages and their range. To form a combined diagnostic for an omission in one assumption, give that omitted-assumption average weight 1/2 and the OTHER assumption's full-observation average weight 1/2. Report the range over these diagnostics plus the full average. These are sensitivity diagnostics, not action vetoes or predictive calibration. Also retain v4's strict worst-case interval/action for an inference-coverage comparison; no truth outcomes enter it.

## Explicit repair consistency and unchanged correction

Copy the frozen v3 feedback correction map exactly. Before producing predictions, require every fixed repair coordinate to have a positive ED score in that same frozen v3 feedback prediction. If this precondition fails, stop and report it without clipping a score, dropping a repair, or changing policy.

After assigning the uniform point estimate or retaining the original loop prior, if a fixed repair would lie outside detection (`score < 0.5`), retain the exact positive ED score from its frozen v3-feedback prediction. Record the candidate score, frozen fallback score, and reason `frozen_v3_repair_consistency_gate`. This is a disclosed coherence constraint, not posterior normalization. It applies equally to eligible and unsupported repaired cells. All other unsupported cells retain the exact original loop score. Assert every final repair is inside detection.

## Freeze, synthetic checks, and output interpretation

Freeze this protocol and the separate optional expected-F1 protocol, code, test source, original strings, and all input hashes BEFORE either prediction. V5 tests live exclusively in the NEW `tests/test_latent_consensus_v5.py`; run only that module for this extension. Preserve the live frozen-v4 `tests/test_latent_consensus.py` byte-exact at SHA-256 `40da4ed6f319142323c63ce3abeb9613f4cb18a2c48400ba771f1ca7ba8cd2c5`, including its original manifest binding. Synthetic tests must prove equal weighting despite different source counts/LOO counts, distinguish an average decision from a worst-case interval, preserve unsupported cells, expose uncertainty diagnostics, and preserve repairs through the explicit coherence gate without silent clipping.

Write a separate standard Prediction artifact, mean-weight evidence, per-cell action/coherence audit, leave-group-out diagnostics, v4-envelope coverage comparison, and a later-verification evidence bundle. Archive every original per-entity/attribute source/value/row claim, raw duplicates and conflicts, frozen source grouping and active group claims/exclusions, and scenario linkage. Embed the relevant raw claim context in each verifier cell; hashes alone are not substantive evidence. Raw literals, formatting, dates, and status/unavailable tokens remain inspectable without asserting unsupported semantic equivalence or snapshot/catalog evidence. This expansion is presentation-only and never adds likelihood votes or changes aggregation weights. The bundle is prepared only; no model or evaluator is called. Output metadata must state statistical ablation, model-conditional weights, no calibration claim, no exact model marginalization, unchanged v3 correction, and no live promotion. Report only paths, hashes, and inference coverage.
