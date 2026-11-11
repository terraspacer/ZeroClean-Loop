# Unified loop v6: generic closed-loop dirty-only protocol

Status: implementation and fictional verification only. This document is not a benchmark result. All six tables use one method and one configuration. The implementation must be reviewed and globally frozen before real inference; scoring follows the complete six-table prediction freeze.

## Scope and provenance

This is an explicitly aggregate-informed follow-up, not a fresh blind experiment. The permitted historical observations supplied by the coordinator are v2's Rayyan detection P/R/F1 0.8961/0.3195/0.4711, Flights detection F1 0.7657, macro detection gain +0.2665 and repair gain +0.1243 versus the qualified sequential reconstruction, and Beers repair precision 0.1766 over 2,338 edits versus sequential precision 0.9876 over 241 edits (repair F1 0.1603 versus 0.1558). They motivate recall expansion and stricter catalog abstention. They do not set a per-row decision or establish v3 performance. No clean table, failure row, label, reviewer judgment, external metadata or catalog is available to inference.

The six names are hospital, flights, beers, rayyan, billionaire and movies. Dataset names locate input/output files only. The detection, consensus and repair modules do not dispatch on names. Inherited v2 source is snapshotted inside this directory; no v2 or shared source is changed.

## Inputs and fresh model rules

For every table the frozen allowlist has nine roles: original dirty CSV; source ZeroED carrier; qualified sequential reconstruction used only for repair seed strings; completed empty-state carrier used only for detection scores; cached CTS proposal artifact; new `artifacts/llm/unified_loop_v6/{dataset}/pass2.json`; cached dirty-only `artifacts/llm/{dataset}/pass2.json`; new bounded dirty packet; and exact raw model response. Every table uses the same cached-plus-fresh rule and dependency union; neither origin substitutes for a missing input. Cached valid-value lists, repair strings and sample IDs are ignored. Packets and raw responses are stored under `artifacts/unified_loop_v6_model/{packets,responses}/{dataset}.json`.

Each pass2 wrapper uses `unified-loop-v6-dirty-rules-v1`, exact dataset and dirty SHA-256, the frozen gpt-6.1-sol/xhigh identity, dirty-only provenance flags and packet/response hashes. Its fresh rules/dependencies must equal the raw response body by canonical JSON bytes, preserving JSON type distinctions. Packet identity uses `table.id` and `table.sha256`. Packet dataset/dirty identity and both raw byte hashes are checked before freeze and before inference. Extra transport/model provenance is preserved. Before inference, cached and fresh wrappers must both bind the exact dataset and dirty hash. The stable union places cached rules first and namespaces original rule IDs by origin. Exact target/kind/source duplicates receive one executable vote while retaining both input origins; dependencies deduplicate exact canonical objects after sorting determinant names. Both origins still undergo full sandbox admission. `validate_rule_artifact` rejects duplicate rule IDs, unresolved targets, malformed dependencies and unsupported thresholds. Rule confidence claims do not provide admission evidence.

Every rule has one exact target column, `kind` single or multi, string description and exactly `def check(row): ...` source. True means violation; False means no violation. The existing isolated sandbox statically validates source and runs it on every original dirty row in batches of at most 1,000. A rule is admitted only if its runtime error fraction is strictly below 5%. Runtime errors abstain. All-true rules are excluded from detection but retained as counterfactual guards; all-false guards remain. This protects execution and some degeneracy, not semantic truth of the rule.

## Detection before seeded repair

First apply the snapshotted v2 inherited-positive gate. It demotes unsupported broad-source positives and preserves target-specific rule, comparable dirty-domain/format or independent-source contradiction evidence. Then inspect all original dirty cells, including inherited negatives, for independently usable evidence:

- Sandbox-admitted, target-specific nonglobal rule violation.
- Actionable original-dirty key/source consensus contradiction, with the query source excluded.
- Leave-one-out rarity in a demonstrably repeated compact domain or homogeneous structured grammar, preserving v2's identifier/free-text abstention.
- Missing string in a nearly complete column (at least 30 nonempty peers, at most 2% missing peers).
- Embedded control or replacement character.
- An inferred numeric, calendar-date or serialized-string-list contradiction (at least eight nonempty peers, at least 95% parseable peers). Calendar dates accept multiple literal orders; no ambiguous chronology is asserted. Names/titles are exempt from numeric inference.
- A minority case/space variant of a dominant observed exact string in a compact, nontext, nonidentifier domain (at least five canonical peers, at least 90% exact agreement in the normalized family). Normalization is only evidence; no value is rewritten.
- A contradiction to globally qualified original-dirty single-column FDs. A relation needs at least two repeated groups with different modes, at least eight covered observations, at least 15% table coverage, at least 97% global majority purity and at least 0.10 purity gain over the target's marginal mode. This prevents a nearly constant target from making every determinant look reliable. Per-cell evidence excludes the query row, needs at least three peers and at least 95% agreement. Conflicting reliable modes abstain.

The promotion score is 0.75; demoted score is 0.49; decisions use strictly `score > 0.5`. These are policy scores, not correctness probabilities. Detection never reads seed strings, CTS retrievals, candidate availability or proposed repairs. Freeze the final scores and mask read-only and hash them before constructing the repair candidate pool. Additional positives can remain unrepaired.

## Repair and catalog precision gate

Preserve v2's dirty key/source consensus, including distinct-source votes, query-source exclusion, raw/weighted winner agreement, format/temporal checks and deterministic ordering. Preserve the at-most-three distinct exact-string candidates per cell: actionable consensus, sequential seed, then cached CTS order. No new replacement string is synthesized. All candidates are tested against immutable original-dirty statistics; duplicate provenance is merged conservatively.

Preserve target-rule, FD, format, four-feature consistency checks and combined-row rollback. Noncatalog behavior is inherited. A catalog is recognized by unambiguous product ID/name, organization ID/name, city/state and measure roles; this is schema matching, never a dataset-name condition. For every recognized catalog edit require both:

1. At least two other original dirty rows contain the exact candidate string in that column.
2. Unanimous leave-one-out entity-FD evidence agrees with the candidate without any conflicting eligible entity relation; or a unit-bearing measure preserves its literal numeric amount and the proposal equals the unique observed exact numeric-family mode, with at least three exact peers and at least 95% canonical agreement.

A rule improvement, seed origin, generic CTS majority or unchanged pattern alone cannot pass this gate. Numeric equivalence checks use Decimal only as evidence; the output remains an existing exact proposal. Quantity changes, percentage rescaling, novel formats, approximate string repairs and metadata lookups are outside the method. This stricter gate can reduce repair recall; no precision increase is claimed before scoring.

## Execution and freeze

The global freeze covers all source/protocol/test `.py`/`.md` files in this directory, shared `zeroclean` data/types/statistics/sandbox source and relevant package initializers, all 54 input byte hashes, configuration, provenance, cohort/model identity, and this protocol. A changed source inventory or byte hash invalidates the lock. No inference or output publication precedes the complete six-table freeze. The lock/plan are created exclusively and cannot be overwritten. Existing partial outputs cause failure; completed outputs are verified and preserved.

All heavy local execution must hold the shared `heavy-experiment.lock`, cap both address-space limits at 12 GiB and force numerical libraries to one thread. `execute_cohort.py` holds one lock across all-six schema/static-source preflight, unchanged-input/source checks, global freeze and serialized six-table inference; resource bounds precede scientific imports. Preflight evaluates no rule and constructs no inference statistics. Console statuses contain stages, dataset names and counts only. The phase CLI also requires the complete six-table cohort. A separate cell-verifier expansion was deferred before v3 scoring and is outside this version. It emits immutable prediction, audit, diagnostic and dataset-freeze artifacts, then a complete cohort manifest. Detection hashes are rechecked after repair and combined-row rollback.

Only after the six-table completion manifest is independently verified may a separate scorer access clean references. The inference author does not read row-level evaluator outputs. No parameter selection or method amendment may use v3 scores within this freeze. Any subsequent change is a new version/study. Desired Rayyan F1 above 0.77 is an untested target, not a promised or measured result.

## v6 generic format and counterfactual closure

V5 removes all benchmark-specific schema aliases and metadata token lists. Format evidence is inferred from dominant dirty-table signatures and validated executable rules. No column name is matched against a benchmark list.

Detection starts from inherited positives and uncertain cells in the [0.25, 0.85] interval. Candidate repair proposals are built before the final mask is frozen. For every exact candidate, the runner computes four dirty-only deltas: rule satisfaction, functional-dependency support, vicinity support, and domain/format support. Positive rule/FD deltas with peer agreement promote uncertain or inherited-negative cells; uniformly degrading candidates can demote a positive whose original value has valid dirty support. The resulting mask and probabilities are then frozen and passed to repair selection.

Repair scoring uses weighted empirical rule support. A single low-support rule cannot veto a high-confidence CTS, seed, or consensus proposal; only runtime failure or a sufficiently negative weighted row delta rolls back a row. The v6 evaluator also computes an adapted ZeroED and adapted sequential comparison in its evaluator-only summary block.
