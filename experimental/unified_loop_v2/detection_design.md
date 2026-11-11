# Intrinsic detection gate, version 2

This protocol is fixed before benchmark inference or evaluation. The same code and
defaults apply to every table; the table's dataset name is never consulted.
Inputs are immutable dirty observations, source probability scores and source
classifier metadata, completed `empty_carry` probability scores, and a Boolean
matrix supplied by the caller. That matrix must contain only violations of
retained, validated, nonempty, target-specific syntactic, domain, or cross-column
criteria. It must not include a classifier vote, a generic rarity predicate, an
empty-predicate fallback, or a repair/verifier judgment. The caller owns that rule
validation. The detector has no file loaders, repair input, reference input,
external catalog, or evaluation interface.

An optional caller-supplied `contextual_low_support` Boolean matrix attests an
additional intrinsic support deficit: an actionable unique alternative supported
by independent source/entity peers contradicts the original dirty observation.
The caller must construct it from immutable dirty groups before gating, excluding
the query observation and without repair state or correctness judgments. The
detector records this as `low_dirty_entity_peer_support`, separately from rule
violations and univariate rarity. It does not infer a calibrated correctness
label from it.

## Interface and score policy

`gate_detection(table, source_probabilities, empty_probabilities,
rule_violations, source_metadata, config=None, *, contextual_low_support=None)`
returns a dictionary with:

- `probabilities`: a copied NumPy float matrix beginning at completed
  `empty_carry` scores, with unsupported gated positives capped at 0.49.
- `mask`: a NumPy Boolean matrix exactly equal to `probabilities > 0.5`.
- `evidence`: JSON-serializable records for every changed score, including the
  original and final scores, source score, gate triggers, rule flag, and dirty
  leave-one-out support.
- `diagnostics`: JSON-serializable defaults, input identity, aggregate and column
  records, and evidence for gated positives retained by a rule or an intrinsic
  support deficit. This retains detection evidence without requiring a repair.

Scores are policy scores, not calibrated error probabilities. This gate can only
demote inherited positives; it never promotes an `empty_carry` negative, copies a
source positive into the output, or invents a repair. Scores in ungated columns
and all inherited negatives remain numerically unchanged. This intentional scope
means a supplied rule violation on an inherited negative is recorded in column
counts but does not itself create a detection.

A column is gated when either source metadata establishes an empty-predicate
fallback, or the source fraction of scores strictly greater than 0.5 is strictly
greater than 0.40. In such a column, an inherited positive survives only if its
validated rule flag is true or at least one comparable intrinsic support measure
is low, including the optional dirty entity-peer support deficit. Positive source
scores alone provide no independent support. Contextual support does not promote
inherited negatives.

The metadata parser reads only `classifier_columns[column]`, and only these
upstream diagnostic fields: `clean_predicate_count_after_filter`,
`empty_predicate_fallback`, `source_empty_predicate_fallback`, `source_edge_case`,
`score_kind`, and `classes`. A zero predicate count, an explicit true fallback
flag, or a specific empty/zero-predicate edge-case description establishes the
fallback. A positive predicate count or explicit false flag establishes its
absence unless contradicted by positive fallback evidence; contradictions are
reported and positive evidence wins. A constant class-1 classifier alone does
not establish an empty-predicate fallback. Missing, incomplete, or malformed
column diagnostics are explicitly reported as unknown, and the source-rate gate
still applies. No upstream binary decision matrix is used.

## Dirty support and abstention

Every count and format is built once from the original dirty strings. Neither
source scores, `empty_carry` scores, rule flags, accepted repairs, nor later
iterations alter those observations. Value counts use exact strings; format
signatures strip outside whitespace for comparison only. No value is rewritten.
Empty and whitespace-only cells have an empty signature. Nonempty formats map
letter runs to `A`, digit runs to `9`, whitespace runs to one space, and preserve
punctuation. Plain signed, decimal, and exponent numeric syntax shares the
`numeric` signature, allowing identifiers and numbers to vary in digit length.
Unicode letters use the same letter class. Case, numeric range, and lexical
meaning are not inferred.

All cell counts subtract the query cell. Format comparison excludes blank peers
and requires at least eight nonblank peers. A format signal is comparable only
when at least 90% of these peers have structured syntax (a digit, `@`, `://`, or
a single-token identifier/code field). It is low only when a dominant nonblank
peer format covers at least 90% of nonblank peers, while the query format has at
most one peer and at most 2% support. A blank query can therefore have low format
support in an otherwise strongly structured column. Blanks do not establish that
any semantic field is mandatory.

Value rarity is comparable only with at least 30 nonblank peers, at most 20%
distinct nonblank values, and at least 90% of nonblank peers in a repeated core.
Core membership requires at least three peer occurrences of that value. A query
then has low value support only with at most one exact peer and at most 2%
support. Missing values may have low value support in such a concentrated domain.
This is a statistical support deficit, not a claim that the value is impossible.

Headers are tokenized across underscores, spaces, punctuation, and camel case.
Fields with `name`, `title`, `description`, `address`, `author`, `authors`,
`actor`, `actors`, `cast`, `director`, `creator`, `text`, `comment`, `comments`,
`note`, `notes`, `abstract`, `summary`, `caption`, `biography`, `content`, or
`message` tokens abstain from both rarity measures. This protects legitimate
open text even if the current dirty sample happens to be repetitive. Otherwise,
unstructured lexical columns abstain from format comparison; domain value
comparison remains possible if its repeated-core conditions hold. Fields with
`id`, `identifier`, `uuid`, `guid`, `key`, or `code` tokens abstain from value
rarity but can be compared by format. Columns with at least 60% distinct nonblank
values also abstain from value rarity. Thus uniqueness does not make entity
identifiers erroneous; a supported identifier format supplies no low-support
signal. Header protections are generic and contain no dataset-name branch.

Small samples, heterogeneous formats, and free text produce `abstain`, not a low
support vote. In a gated column, abstention without a validated rule does not
justify retaining an inherited positive. A rule violation overrides even strong
dirty support, because repeated corruption can be well supported in dirty data.

## Fixed defaults and checks

`DetectionConfig` is frozen, rejects invalid values and unknown mapping keys, and
defaults to: source positive rate 0.40; demoted score 0.49; comparison peers 8;
domain peers 30; maximum domain distinct fraction 0.20; core peer count 3; core
mass 0.90; rare peer count 1; rare peer fraction 0.02; structured peer fraction
0.90; dominant format fraction 0.90; identifier-like distinct fraction 0.60.
The decision threshold is always strictly greater than 0.5 and is not configurable.
Returned diagnostics serialize every active configuration value. Config changes
must be decided and frozen before any benchmark evaluation.

Synthetic tests cover unsupported high-rate positives, rule retention without a
repair, contextual disagreement with a common valid format, unique identifiers,
rare free text, leave-one-out behavior, supported and
unsupported formats, concentrated-domain rarity, absent metadata, metadata
fallback parsing, unchanged ungated scores, input validation, and exact mask/score
agreement. No real-table inference or benchmark evaluation is part of these
tests. Arrays and metadata are never mutated. Work is linear in the number of
cells plus the number of distinct dirty values and format signatures.

## Limits

Dirty majority patterns can themselves be wrong; this gate does not prove
correctness. It intentionally loses detections when only unsupported broad
classifier positives are available. It can retain a rare legitimate category
as low support in a concentrated domain. It cannot detect semantic errors that
share a supported format without validated rules, and broad header protections
may abstain on fields that could support a useful domain rule. Missing metadata
cannot be reconstructed from a constant classifier alone. The rule matrix is a
caller attestation; this function cannot inspect the predicates behind it.
