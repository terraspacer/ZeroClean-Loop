# Prespecified repair selector for unified_loop_v2

This policy is fixed before any evaluation. It reads the immutable dirty table,
the already fixed empty-state detection mask, and candidate/feature evidence
prepared by the inference pipeline. It does not access reference data,
evaluation masks, metrics, reports, previous verifier decisions, or history.
There is one method, `unified_loop_v2`, across all schemas.

## Interface and invariants

`select_repairs(table, detection_mask, candidates, config=None)` returns
`repairs`, a map from `(row, column)` to an exact string; JSON-serializable
`decisions`; and JSON-serializable `diagnostics`. The table has `columns` and
`rows` as in `zeroclean.data.DirtyTable`. The mask is a rectangular boolean or
binary array. The selector snapshots the mask and never writes to it, the
table, or candidate objects. Only detection-positive cells can receive an edit.

Candidates are dictionaries with `value`, `origins`, `delta_features`,
`new_rule_violation`, `rule_gain`, `peer_rows`, `cts_peer_rows`, and
`candidate_format_valid`. Feature order is rule satisfaction, FD support,
NMI vicinity, and domain/pattern support. Optional `consensus` must preserve
the full original proposal from `consensus.discover_consensus`, including
`actionable`, `score`, `peer_source_count`, `winning_source_count`, weighted
`agreement`, normalized `margin`, and its nested `evidence`. The runner must
not reduce it to the five original summary fields.
`peer_rows` is general provenance and never supplies CTS votes, even when a
merged candidate's `origins` includes `cts`. The explicit `cts_peer_rows`
contains only raw row IDs supplied by the fixed CTS mechanism. The runner
must preserve the complete per-cell CTS-only union across exact-string
merges and any upstream pruning, for example by attaching that union to
each retained candidate. Consensus-only peer rows cannot enter this field.
An optional truthy `counterfactual_error`, false `criterion_valid`, false
`rule_criterion_valid`, or false `counterfactual_valid` rejects the candidate.
The supplied candidate format decision must be explicitly true. Unknown or
nonfinite feature evidence is rejected rather than guessed.

JSON `null` means abstention and is never a replacement string. `""` remains
an exact, potentially valid string. Values equal to the original are no-ops.
No case folding, number conversion, whitespace normalization, or canonical
spelling is applied to replacement values.

## Fixed candidate budget and local consistency guards

Exact duplicate strings are merged, retaining all origins and distinct peer
rows. Merging never adds consensus scores or source counts. Duplicate feature
evidence is combined conservatively: coordinatewise minimum feature deltas,
minimum rule gain, any new violation/runtime error, and all format/criterion
checks must pass. The nonoriginal string budget is three. Budget priority is
actionable key-source consensus, then seed, then CTS; ties use the supplied
consensus score followed by exact lexicographic string order. A key-source
consensus is actionable only when its supplied action flag is true, at least
two nonquery peer sources and two winning sources are reported, weighted
agreement is at least two-thirds, and normalized margin is at least 0.20
(with the consensus module's 1e-12 boundary tolerance). There is no stricter
repair-only agreement threshold. The selector independently deduplicates
`evidence.peer_observations` by source, excludes `evidence.query_source`, and
reconstructs raw and `evidence.source_reliability[source].score` weighted
votes. The candidate must be the unique winner in both counts, with declared
counts, `evidence.supporting_sources`, agreement, and margin matching those
records. Conflicting duplicate source observations, missing required source
evidence, failed `intrinsic_checks.temporal_checks_passed`, or any failed
`intrinsic_checks.temporal_relations` reject source actionability. Exact
source votes must match unambiguous `exact_values`. Duplicate candidate
records never create another source. A source-only candidate must be
actionable; a merged seed or CTS candidate may still qualify independently
under its own ordinary guards.

Every candidate must have no new rule violation/runtime error, valid format,
no loss of rule satisfaction or FD support beyond 1e-12, and nonnegative
weighted counterfactual delta within 1e-12. The fixed weights are
`(0.4, 0.3, 0.2, 0.1)`. Any available conflicting catalog FD evidence blocks
mutation, including otherwise positive rule-gain candidates.

Outside the catalog safeguard below, a seed passing these guards is eligible
as a **noisy seeded carry** even without FD evidence. It is not labeled
independently verified. Other candidates require actionable source consensus,
strong dirty peer evidence, strict rule gain, or a positive weighted
counterfactual delta. Among eligible candidates, highest weighted delta wins,
then actionable source-consensus priority, seed priority, CTS priority, and
exact string order. A carried seed can therefore be replaced by a better
guarded proposal.

The caller must still evaluate the selected edits together on the whole row.
It rolls back every edit in a row if the combined consistency/format guard
fails, preserving the fixed detection positives. The selector reports this
remaining caller obligation explicitly in diagnostics.

## Catalog applicability, determined only by column roles

Column names are lowercased and stripped of nonalphanumeric characters solely
to identify roles. The safeguard applies exactly when all seven roles exist:

- product ID: `id`, `product_id`, `beer_id`, `item_id`, or `sku`;
- product name: `product_name`, `beer_name`, or `item_name`;
- organization ID: `brewery_id`, `manufacturer_id`, `producer_id`,
  `supplier_id`, or `vendor_id`;
- organization name: the corresponding `*_name` aliases;
- city: `city` or `organization_city`;
- state: `state` or `organization_state`;
- at least one measure: `ounces`, `volume`, `weight`, `abv`, or
  `alcohol_by_volume`.

Role aliases are compared after the stated normalization. If two columns
match the same required role, applicability is ambiguous and an error is
raised; it cannot silently bypass the safeguard. Multiple measure columns
are permitted. The supplied table/dataset name is never inspected. This rule
matches the actual dirty brewery/product column roles while leaving
Hospital/Billionaire-like schemas under ordinary seeded-carry semantics.

Every catalog edit additionally needs at least one of these strict witnesses:

1. **Deterministic unique FD consensus:** at least two other dirty rows match
   a nonempty determinant and all have exactly the candidate target string.
   Allowed determinants are product ID for every other column; organization
   ID and organization name for organization ID/name, city, and state;
   and product name paired with either organization identifier/name for
   targets outside that pair. Any determinant containing the target is
   discarded in its entirety. All
   usable determinants are considered. Any supported mixed group or
   disagreeing unanimous groups causes abstention. No query row is counted.
2. **Literal unit stripping:** the original consists of an exact numeric
   spelling followed by a recognized unit and the candidate is precisely
   that unchanged numeric spelling. Recognized units are `oz`, `oz.`,
   `ounce`, and `ounces` for `ounces`; `%` and `percent` for `abv` or
   `alcohol_by_volume`. Case-insensitive unit recognition does not alter the
   captured numeric string. At least two other nonempty dirty values and
   at least 60% of nonempty dirty peers must already use the bare-number
   grammar. No scaling, rounding, suffix conversion, or stripping an
   unsupported column grammar is allowed.
3. **High-agreement CTS peers:** the union of explicit `cts_peer_rows` for the
   cell, including candidates outside the capped budget and original/no-op
   or null-abstention records, has at
   least three distinct other rows; at least 90% have exactly the candidate
   string; and candidate vote share minus the largest other share is at
   least 0.5. The query, duplicates, invalid row IDs, and source-consensus-only
   peers do not vote. Missing `cts_peer_rows` supplies no CTS witness; there
   is no fallback to general `peer_rows`. A tie never supports an edit.
4. **Strict rule improvement:** supplied per-target rule gain exceeds 1e-12.

With no qualifying witness, detection stays positive and mutation abstains.
Missing FD evidence is not evidence of correctness. Formatting comes only
from exact dirty peer strings or the guarded literal grammar, plus the
caller's dirty-derived format validity. The selector does not synthesize
new repair strings.

## Defaults, validation, and scope

`RepairConfig` is frozen; `DEFAULT_CONFIG` carries the defaults above.
`config` may be that dataclass or a mapping containing its known fields.
Candidate cap must be between one and three; probabilities are bounded,
minimum peer counts are at least two (CTS at least three), tolerance is
nonnegative, and four weights must be finite, nonnegative, and sum to one.
The run configuration must be frozen before evaluation; no result selects
between configurations or arms.

Tests use fictional tables only. They cover the positive-mask gate, immutable
mask/candidate input, exact and empty strings, null abstention, catalog
abstention, FD ties/conflicting determinants, query exclusion, duplicate CTS
peers, noncatalog seed carry, invalid format/new violation/runtime-error
rejection, literal units preserving numeric spelling, fixed candidate cap,
duplicate same-source evidence, shared two-thirds source agreement, declared
winning sources, reconstructed raw/weighted winners, and chronology conflict.
No actual output run or complete project
suite is part of this component's validation.
