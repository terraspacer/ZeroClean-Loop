# Inference code audit

This audit reviewed `zeroclean/loop.py`, `zeroclean/statistics.py`, `zeroclean/optimize.py`, and `evaluate_e2e.py`, with supporting type and transport code. It used only program sources and synthetic tables. It did not load benchmark clean/reference tables, error masks, evaluation summaries, or failure reports. The regression suite is `tests/test_loop_audit.py`.

## Contracts and findings

| Boundary | Required behavior | Finding reproduced synthetically |
| --- | --- | --- |
| Candidate request | Match the dirty fingerprint and recompute the content hash before inference. Proposal files must carry the dirty fingerprint. | Previously fixed protections are covered against regression. |
| Verifier provenance | Require matching `request_hash`, `model`, `reasoning_effort`, and `dirty_sha256`. Validate `dataset` when supplied. | A matching request hash alone previously admitted foreign or missing provenance. |
| Verifier cells | Accept only unique coordinates in the exact candidate request. Rows and numeric columns must be integers, excluding booleans. An optional original must match. | Unrequested coordinates were silently ignored, fractional coordinates were truncated, and duplicate responses overwrote earlier ones. |
| Verifier values and scores | Candidates must be a unique subset of supplied values; a selection must be the original, a supplied candidate, or null. Scores must be finite numbers in `[0,1]`; joint permissions must be booleans. | Invented values and invalid scores were accepted. In particular, NaN could pass through score transforms as near-certain repair confidence. A string such as `"false"` enabled joint permission. |
| Verifier abstention | Explicit `selected_value: null` suppresses repair while preserving original-invalid evidence. Omitting the field permits candidate rescoring without selecting a value. | Null previously behaved like an omitted selection and could still produce a repair from high proposal confidence. |
| Detection versus repair safety | Rejecting unsafe replacements must preserve independent evidence that the original is invalid. | The all-alternatives-unsafe cap reduced detection to `.45` despite verifier original-validity `.01`. |
| Joint repairs | Honor candidate-level joint permission, then execute checks on the complete changed tuple. Roll back all changes to a tuple that introduces a violation or runtime failure, while retaining independent original-invalid evidence. | Candidate-level permission and runtime rollback already worked; regressions preserve them. A failed replacement must not erase a confident detection. |
| Optimization proxy | Score accepted, finally consistent repairs using the confidence actually used by inference. | The proxy used proposal confidence before verification and penalized individually inconsistent changes even when their joint tuple passed. A `.70` proposal verified at `.84` was removed by hill climbing solely because the stale proxy score was negative. |
| Frozen evaluation | Check the current dirty fingerprint before reaching reference-backed evaluation, as well as hashes of frozen predictions. | Frozen prediction bytes could remain unchanged while their dirty input changed. The test replaces evaluation with a function that must never be reached. |

The statistical index uses dirty observations only. Synthetic checks confirm that a singleton supplies neither its own domain nor contextual support, and a counterfactual determinant change consults the destination peer group without subtracting an unrelated peer. Existing tests cover leave-one-out conditional support for an unchanged determinant. These checks do not establish that a dirty majority is semantically correct.

## Ablation interpretation

`multi_column=False` removes multi-column executable rules and functional dependencies. NMI contextual tuple retrieval remains part of the method; the variant should be described as **without multi-column rules and dependencies**, rather than without all cross-column information.

A fresh candidate request has a different hash and may expose different evidence. Running the rule ablation with no verifier while the full method has one changes both constraints and verification coverage. A matched experiment needs independent verification for the variant request, or a declared transfer policy restricted to judgments whose evidence remains valid. Transferred judgments must be distinguished from independent variant verification in provenance and reporting.

The counterfactual variant currently removes inference-time counterfactual scoring and safety gates while reusing verifier judgments made from a request containing counterfactual features. This is a computational ablation, not evidence that all counterfactual information has been removed. Independent verification on a request without those features is needed for the stronger claim. Likewise, disabling iterative verification removes both external verifier feedback and repeated rounds; it is a combined intervention unless those factors are separated.

## Limits

The objective is dirty-only consistency utility, not accuracy or a calibrated probability estimate. Repeated rounds operate on the same immutable candidate evidence; final tuple checking supplies a consistency guard, not a fresh model judgment. Model/effort headers document declared provenance but do not cryptographically prove which model produced a file. Passing these synthetic regressions establishes the specified code contracts; it does not validate benchmark performance or the semantic correctness of any repair.

Validation: `.venv/bin/python -m pytest -q tests/test_loop_audit.py` — **49 passed** after the inference fixes. The runtime-failure case deliberately expects detection to remain high when the verifier confidently rejects the original, even though every proposed replacement is rolled back.
