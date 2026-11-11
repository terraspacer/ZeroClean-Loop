# Intrinsic key/source consensus proposals

`discover_consensus(table, config=None)` consumes only string `columns` and
`rows`. It does not access files, `table.name`, references, error masks, detector
judgments, benchmark outcomes, external metadata, or previous study results.
The implementation uses only Python's standard library. No predictions on the
six production dirty tables were generated during this implementation stage.

The return object contains `candidates`, keyed by zero-based `(row, column)`
tuples, and JSON-serializable `diagnostics`. Every candidate has an exact
observed, nonoriginal `value`, bounded heuristic `score`,
`origin="key_source_consensus"`, `actionable`, source counts, weighted
`agreement` and `margin`, source/key/target column names, and inspectable
`evidence`. A consumer must independently detect a cell and require an
actionable proposal before considering repair. This module neither detects
errors nor changes a table. It returns one unique winning alternative at most,
which satisfies the three-candidate cap.

## Fixed shared discovery policy

`POLICY_VERSION` and read-only `DEFAULT_CONFIG` expose the frozen defaults. A
caller can supply an explicit, validated shared configuration; there are no
dataset-specific branches or thresholds.

1. Discover temporal columns through time/date schema tokens and parseable
   values, or an 80% parseable rate in at least three nonempty observations.
   Here an observation is a distinct complete exact dirty row: duplicate copies
   count once, while the same target string in different entities, sources, or
   other dirty contexts counts separately. This duplicate cap applies before
   source/key roles are known. Temporal profile counts use this same unit.
   Supported validation formats are strict 12/24-hour clocks, ISO dates, and ISO
   dated timestamps. AM/PM punctuation and explicit offsets are understood.
   Explicit offset fields are range-checked before parsing (hours 0–23, minutes
   0–59), so parser normalization cannot legitimize malformed offsets such as
   `+01:60`. Parsing never constructs a replacement value.
2. Require a source-like schema name such as `src`, `source`, `data_source`,
   `provider_id`, or `reporter`, with at least three observed source identities.
   Low cardinality alone does not imply a source role.
3. Find a repeated entity key among remaining columns. Require at least two
   repeated multisource entities, at least 50% multisource source/key-slot
   coverage, and at least 80% source/key-slot uniqueness relative to distinct
   temporal observations. Exact duplicate observations cannot increase these
   counts. Identifier-like schema names receive precedence, then the measured
   structural properties determine rank. Ties use column names deterministically.
   A supported single key takes precedence; pairs are considered when single
   columns conflate distinct temporal observations. Empty key/source values
   supply no group. Unsupported tables return empty candidates and an explicit
   nonapplicability reason.

These are structural hypotheses. Repeated identifiers can still conflate
entities, and named sources can be correlated. There is no inference that
distinct source labels establish independent provenance.

## Exact observations and source caps

For a query cell, discard every row bearing its source identity in its entity.
Each remaining source contributes at most one vote for one exact raw string.
Repeated identical rows have no effect. A source with conflicting nonempty
strings abstains even when both strings parse to the same clock value. Missing
values supply no vote; malformed or conflicting nonempty sources remain in
the agreement denominator and cannot increase apparent support by abstaining.
Strings, case, whitespace, leading zeros, and AM/PM representation are preserved.
Semantically equivalent but differently spelled values do not pool votes.

A proposed winner must be the unique winner both by distinct source count and
by reliability-weighted votes. A source-count tie, weighted tie, or disagreement
between those winners yields no proposal. Minority alternatives are not emitted.
The query's original value is never proposed. When the winner equals the query
original, there is no repair proposal.

## Source reliability, without query-entity leakage

Reliability is target-specific, dirty-only evidence. For each source and entity,
cap each of these training signals at one contribution:

- Format: the fraction of distinct nonempty strings passing temporal validation.
- Peer agreement: whether the source's unambiguous valid vote matches a unique
  other-source consensus supported by at least two sources and two-thirds of
  nonempty peers. Both the source being scored and the query source are excluded
  from those training reference votes.
- Cross-column consistency: whether all safely evaluable explicit dated
  start/end relations for that source/entity pass.

All contributions from the entire query entity are removed before calculating
its peer source weights. This includes the entity's other rows and temporal
context. Exact duplicate training rows cannot increase reliability. Each
component has a fixed three-entity prior at 0.5. Components receive fixed weights
0.35 format, 0.50 peer agreement, and 0.15 cross-column consistency; the combined
source weight has a 0.20 floor. Unavailable evidence remains at the neutral
prior. With only three sources, excluding the query source can leave too few
training peers to assess agreement, so that component remains neutral.

These reliability scores are **not calibrated correctness probabilities**.
Dirty peers can agree on a wrong value. Source weights encode conditional
structural evidence and cannot establish ground truth. Evidence includes
component scores, leave-entity-out training counts, and explicit calibration
flags.

## Intrinsic temporal checks and actionability

Only an unambiguous named start/end or departure/arrival schema pair with the
same remaining schema tokens supplies an ordering relation. Compare chronology
only when both strings include a full date and explicit timezone offsets.
Compare actual instants, including offsets and date rollover. Clock-only
values, ISO dates alone, naive dated timestamps, and the unknown `-00:00`
offset convention receive no ordering check.
No rule assumes that arrival follows departure on the displayed local clock,
that scheduled and actual times are ordered, that zones match, or that an
overnight transition occurred. Missing/malformed companions supply no
chronology evidence. Candidate evidence records every evaluated/skipped check.

A proposal is actionable when it has at least two nonquery peer sources, at
least two winning sources, weighted agreement at least two-thirds, normalized
weighted margin at least 0.20, and no evaluable chronology conflict in the query
row. A unique winner below these thresholds can be returned with
`actionable=false` and explicit abstention reasons. A syntactically valid peer
value can therefore be proposed without inventing time or timezone context.
The two-thirds threshold is an explicit distinct-source quorum, fixed before
production predictions or correctness outcomes. Final repair gating should use
this same `actionable` decision plus independent detector acceptance rather
than imposing a second, inconsistent consensus threshold.

The candidate score is `0.7 * agreement + 0.2 * margin +
0.1 * min(winning_sources / 3, 1)`. It is a deterministic support ranking,
bounded in `[0, 1]`, independent of any error label or correctness calibration.

## Fictional verification

Run `python -m unittest experimental.unified_loop_v2.test_consensus -v` from the
repository root. The tests construct small fictional tables only. They cover
schema/key discovery, duplicate-invariant unhinted temporal discovery, composite keys, unique-key and unsupported-table
abstention, source duplication/conflicts, complete query-source exclusion,
malformed formats, raw/exact-format ties, source reliability leakage,
preservation of exact strings, rejection of malformed offset fields,
unzoned/overnight cautions, defensible explicit
offset checks, chronology conflicts, and bounded scores/config validation.
