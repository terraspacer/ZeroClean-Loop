# FORMAT-CONTRACT verification/application adapter v1

This is a separate, dirty-only adapter. It does not change or import the frozen
catalog wrapper, alter ZeroClean inference, evaluate benchmarks, select a best
iteration, or promote output. Implementation: `experimental/format_overlay.py`;
CLI: `scripts/run_format_overlay.py`. Importing either module performs no IO.

The author protocol SHA256 is
`1f87f7d0f57c862f16929fbec90c84396046254ed833ec2d5b820a6ad631a177`.
The author schema SHA256 is
`16a5b26ecd5c6448e47c26a0f95303b98725d73314aa85e00f548c0ff1cb1ecc`.
The adapter extends the author's review response; it does not rewrite the
author protocol or schema. This protocol and implementation are hashed into
every prepared request. Changing either requires a fresh preparation.

## Fixed policy, established before reading actual inputs

The same policy applies to hospital, flights, beers, rayyan, billionaire and
movies. Original-valid support of at least 0.9 and decision confidence of at
least 0.8 can prune ED. Original-valid support of at most 0.1, decision confidence
of at least 0.8, explicit invalidity support and approved, runtime-accepted
violated rules can flag ED. Unknown/null validity or confidence preserves the
exact parent. These scores describe uncalibrated judgments, never calibrated
error probabilities. Detection uses fixed caps/floors 0.1/0.9, rather than
converting a confidence score into a probability.

A repair additionally needs the independent reviewer's exact approved scalar
selection or full approved atomic bundle, an approved contract and transform,
verified confidence at least 0.8, and exact authored parent preconditions. A
supported invalidity flag alone never authorizes repair. An abstained or rejected
candidate never authorizes repair, even with strong author confidence.

Four dirty-only deltas are recorded in order: rule satisfaction, leave-one-out FD
support, NMI vicinity and domain support. Weights are fixed at
`[2.2, 1.8, 0.65, 0.35]`. Vicinity/domain differences use `tanh(difference/4)`.
The gate is the uncalibrated log support score
`log(verified_confidence/(1-verified_confidence)) + weighted_delta >= log(4)`;
confidence is clipped to `[1e-9,1-1e-9]` only for this calculation. Verified
confidence must independently meet 0.8. This score is not a posterior
probability. Rejected/abstained contracts are excluded from the rule delta at
application. Bundle per-field statistical deltas are averaged, and rule deltas
use the union of applicable rules; bundles are never flattened.

## Preparation and permitted sources

Wait for author `manifest.json` and `independent_review_request.json`. The latter
must bind the manifest's exact byte SHA256 as `author_manifest_sha256` (or
`manifest_sha256`), avoiding a hash cycle. The manifest file map must bind the
protocol, schema and all six datasets' `pass4.json`, `column_contracts.json`,
`proposals.json`, `joint_conversion_candidates.json`, `runtime_validation.json`,
and `sampling_provenance.json`. Supported map forms are
`files: {relative_path: sha256_or_object_with_sha256}` or a list with `path` and
`sha256`. Profile/benchmark summary files are never opened.

The only table loader is `zeroclean.data.load_dirty`. The only prior prediction
read is the explicit untrusted parent
`results/source_v2/predictions/<dataset>/zeroclean_loop.json`. Dirty hashes and
parent metadata dirty hashes must match. Parent repair coordinates must be
unique and every repair must lie inside the parent ED mask.

Run with the repository environment after the complete author freeze:

```sh
.venv/bin/python scripts/run_format_overlay.py prepare --output NEW_DIRECTORY
```

This prepares all six datasets without applying anything. Each dataset request
contains every authored cell and all exact candidates, every atomic bundle,
every column/contract, all generated rules including rejection diagnostics,
runtime-accepted rules, raw originals, exact parent values/probabilities and
whole-row contexts, single/multi-column rule evidence, raw leave-one-out counts,
FD evidence and four deltas. Rule validation uses every dirty row and the
existing strict `< 5%` runtime-error gate. Statistics are built from original
dirty observations only. All author evidence is explicitly fallible.

The independent request omits all parent metadata, traces, rationales and prior
verifier judgments. Full original parent bytes are stored only in
`<dataset>/provenance/parent_prediction.json` with file hashes. Reviewers should
read `verification_request.json`, not the provenance directory. Preparation
also snapshots exact author files. `manifest.json` binds every output file.
All destinations must be new; exclusive writes refuse overwrites.

Author scalar IDs use SHA256 of compact canonical UTF8 JSON (`sort_keys=True`,
`ensure_ascii=False`, separators `(',', ':')`) over dataset, row, column,
source_value, contract_id, transform_id and value. Bundle IDs use the same
serialization over dataset, row, contract_id, transform_id, source_values,
updates and application_mode. Exact authored update list order is retained.
Adapter object hashes, including `request_hash`, use `zeroclean.types.digest`.
Byte file hashes use SHA256. These serializations are explicitly distinct.

## Independent response extension

The root designates a separate `gpt-6.1-sol`, `xhigh` reviewer. Author and root
cannot designate themselves as the independent reviewer. The review may access
only the prepared request and approved dirty-only source context. No benchmark
references, clean tables, masks, evaluation outputs or other workers' verdicts.

Write one `<dataset>/verification.json` for every dataset. All four echoed hashes
are mandatory: `request_hash`, `protocol_sha256`, `artifact_schema_sha256`, and
`author_manifest_sha256`. Also require `model: "gpt-6.1-sol"`,
`reasoning_effort: "xhigh"`, `benchmark_data_accessed: false`, and
`default_for_unreviewed: "abstain_preserve_parent"`.

Provide exact complete coverage, no duplicates and no extra IDs:

- `contracts`: one object per requested `contract_id`, with `decision`
  (`approve`, `abstain`, `reject`), nonempty `rationale` and
  `approved_transform_ids` (explicit unique list, including empty lists).
- `scalar_candidates`: one object per exact `candidate_id`, with `decision`,
  `rationale`, `exact_value` (the exact frozen alternative even on abstention),
  and `verified_confidence` (number in `[0,1]` or explicit null).
- `atomic_bundles`: one object per exact `bundle_id`, with `decision`,
  `rationale`, `exact_updates` (the complete exact ordered authored update
  list), `atomicity_confirmed` (explicit bool) and `verified_confidence`
  (number or explicit null). Approval explicitly selects the full exact bundle.
- `cells`: one object per requested `cell_id`, with `original_valid_score`
  and `decision_confidence` (numbers or explicit null),
  `selected_candidate_id` (exact ID or null), `selected_value` (exact selected
  string or null), `invalidity_supported` (bool), `invalidity_rule_ids` (list),
  `retract_parent_repair` (bool), `retraction_supported` (bool) and `rationale`.

Null selection adds no repair. Explicit supported original invalidity may still
flag ED without identifying a repair. Null original validity/confidence
preserves the parent completely for that cell. The selected candidate ID and
selected exact value must agree; no invented values are accepted. A valid
original and a nonoriginal selected repair are contradictory and rejected.

For bundle fields which also appear in `cells`, those cell validity/support
decisions apply. The cell's scalar selection may be null when the bundle is
selected. Bundle-only changed fields require the following explicit maps in
their atomic decision: `original_valid_scores`, `decision_confidences`,
`invalidity_supported_by_column` and `invalidity_rule_ids_by_column`. Every
changed field must independently meet the original-invalid/confidence gates;
unknown or incomplete field support preserves the entire bundle. Runtime
accepted, approved rules must actually violate the original in the raw context
and reference the claimed field. Rule lists are support, not repair authority.

Each `<dataset>/provenance.json` requires `reviewer_id`,
`reviewer_role: "independent_format_contract_verifier"`, `designated_by: "/root"`,
`request_hash`, `verification_file_sha256` (exact verification bytes),
`model: "gpt-6.1-sol"`, `reasoning_effort: "xhigh"` and
`benchmark_data_accessed: false`.

## Application and fail-closed behavior

Only after independent review is complete and the root explicitly invokes it:

```sh
.venv/bin/python scripts/run_format_overlay.py apply \
  --prepared PREPARED_DIRECTORY --reviews REVIEW_DIRECTORY \
  --reviewer-id ROOT_DESIGNATED_REVIEWER_ID --output NEW_DIRECTORY
```

Application checks all stored byte hashes, the unchanged implementation and
protocol, author manifest, exact current parent bytes, complete reviewer
provenance/coverage and all exact IDs/values. It independently replays every
prepared request from authoritative frozen author files and raw dirty counts;
a manipulated but internally rehashed request is insufficient.
Per-request bindings must equal the freshly validated authoritative author and
implementation bindings plus the exact fixed parent file. They cannot nominate
an alternative author identity or manifest. Prepared manifest paths must be
relative, contain no parent traversal, stay inside the prepared directory, and
cover exactly the declared snapshots. Symlink input/output paths are rejected
before file contents are read.
Every prepared author snapshot is also compared with the authoritative live
author-file hash, and every prepared parent snapshot with the exact current
parent bytes. A self-rehashed preparation manifest cannot substitute provenance
snapshots. Application outputs retain those verified source snapshots too.

Authored scalar preconditions require the current parent value to equal the
original source string. Bundle preconditions require every source field to
match, including unchanged context fields. A mismatch preserves the parent
repair; raw-original invalidity may still independently flag ED. All bundle
updates happen together or none happen.

An independently valid raw original does not silently retract an inherited
repair. Retraction requires `retract_parent_repair: true`,
`retraction_supported: true` and `selected_value` equal to that exact original.
ED-only pruning without an inherited repair remains possible.

Selected scalar repairs, explicit retractions and selected atomic bundles are
assembled simultaneously. Any overlapping source/action footprints preserve
all overlapping operations, regardless of list order. A bundle can consume
ED-only cell decisions for its changed update fields; those are support for its atomic action. Independent
scalar selections remain conflicts even when the final proposed text agrees.

Every row with any changed ED probability or repair-map entry is checked, even
when its rendered tuple text did not change. The combined whole row is checked against the exact parent row under all
runtime-accepted, independently approved rules. Any introduced violation,
unknown runtime result in either before/after, or absence of an approved rule
rolls back the entire row to the exact parent probabilities and repairs. This
includes ED changes on other fields in that row. Repairs outside ED are
rejected. Final output contains snapshots, hashes, decisions and rollback
traces, with evaluation status `not_run`; all existing artifacts are preserved.

Synthetic verification: `.venv/bin/python -m pytest -q tests/test_format_overlay.py`.
