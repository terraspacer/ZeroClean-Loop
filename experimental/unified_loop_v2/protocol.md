# Unified Loop v2 — prespecified intrinsic local study

This is one new method, `unified_loop_v2`, evaluated on Hospital, Flights,
Beers, Rayyan, Billionaire and Movies. Its evaluator filename is fixed as
`zeroclean_loop.json` before any new score is computed. All completed source,
conservative-carry and cloud studies remain intact. No per-dataset best arm,
round, threshold or output is selected after evaluation.

## Authorization and information boundary

The user explicitly authorized this new version and the following architecture
after seeing aggregate results of earlier studies. This is therefore a
user-directed follow-up experiment, not an assertion that its design was chosen
without prior aggregate benchmark feedback. No benchmark cell labels, clean
references, evaluation failure rows, or reference-valued diagnostics enter
implementation authors, inference, criteria, statistical estimates or repair
selection. Fictional fixtures are permitted for software verification. The
evaluator is a separate, permanently evaluator-only role and returns aggregate
counts and metrics, never labeled row examples to inference.

Only exact raw dirty CSVs, existing source ZeroED scores, qualified naive
ZeroED-to-ZeroEC repair proposals, the completed `empty_carry` probabilities,
and the fixed pass2 criteria and CTS-v2 proposals are scientific inputs.
`empty_carry` repairs and the final `seeded_carry` repair output are not used.
No external metadata, catalogs, acquired movie/bibliographic records, or their
review judgments are used. The model remains `gpt-6.1-sol` with `xhigh`
reasoning; authors and independent reviewers have fresh isolated contexts.
Existing model outputs are reused as inputs rather than rerunning completed
experiments. New module authorship is honestly distinguished from independent
per-cell model calls.

## Single fixed dataflow

1. Preserve all input strings, empty values, row order and column order. Read
   the cached empty-state probability matrix and validate strict `p > 0.5`
   semantics. No tentative repair table or seed value can influence this
   detector input.
2. Validate pass2 Python criteria in the unchanged resource-limited syscall
   sandbox, accepting only rules with fewer than 5% syntax/runtime failures.
   Evaluate retained rules on raw dirty tuples. A constant-true criterion is
   not independent detection evidence; constant-false criteria still protect
   counterfactuals from newly introduced violations. Other target-specific
   validated violations remain available. This is operational validation, not
   a guarantee of semantic truth.
3. Discover source/entity structure and exact temporal peer alternatives from
   the dirty table. Source agreement evidence can support detection without
   reading any tentative repair state. The full policy and limits are in
   `consensus_design.md`.
4. Gate inherited empty-state positives in explicit source fallback columns or
   columns whose source-positive fraction exceeds 0.40. Retain a positive only
   for a validated specific rule violation, comparable low dirty format/domain
   support, or actionable dirty entity/source disagreement. Otherwise set its
   policy score to 0.49. Never promote a negative in this version. Unique IDs
   and open-text rarity alone do not constitute errors. `detection_design.md`
   fixes the generic schema protections and leave-one-out thresholds.
5. Freeze the resulting `M_det` in memory before repair. On its positive cells
   only, build up to three distinct nonoriginal candidates: actionable
   key/source consensus first, then the noisy naive seed, then the original
   fixed CTS order. Preserve exact strings and merged provenance. Candidate
   selection cannot revise `M_det`.
6. For each candidate calculate the four counterfactual changes using raw dirty
   statistics only: target-rule satisfaction, functional-dependency support,
   NMI vicinity likelihood and domain/pattern support. Repaired values never
   enter the reference counts. Runtime failures, invalid formats and new rule
   violations block a candidate.
7. Apply the prespecified repair selector in `repair_design.md`. Ordinary seeds
   can be carried as explicitly noisy hypotheses under consistency guards.
   The product/organization/locality/measure catalog schema triggers a stricter
   gate: unique deterministic FD evidence, supported literal unit stripping,
   high-agreement distinct CTS peers, or strictly improving rule satisfaction.
   An unresolved repair leaves its cell detected but unchanged.
8. Check all proposed edits to each row together in the sandbox. Any introduced
   violation or runtime error rolls back every edit in that row, retaining the
   exact fixed detection score and mask. Every emitted repair must lie inside
   `M_det` and change the original string.

## Source consensus and exact representation

The entity/source discovery and catalog policy are schema-conditioned parts
of this single method; neither reads the dataset name to choose thresholds.
Each peer source contributes at most one exact-string vote. Exclude the query
source from its evidence, and the entire query entity from source reliability
estimation. Dirty format validity, defensible chronology and other-source
agreement supply fixed reliability components. Clock-only or unzoned times
do not establish arrival/departure ordering.

Actionability is shared by the proposal and repair modules: at least two
nonquery sources and two winning sources, weighted agreement at least 2/3,
normalized margin at least 0.20, unique raw-count and weighted winners, and no
defensible explicit chronology conflict. Scores are uncalibrated support
rankings. Distinct source labels do not establish independent real-world
provenance. Repeated dirty errors and wrong entity keys remain possible.

No date, timestamp, number, case or punctuation normalization manufactures a
canonical replacement. The separately specified catalog unit rule strips a
literal recognized suffix while preserving the exact numeric token, subject
to dirty peer grammar evidence. JSON null is abstention; an empty CSV string
is a distinct possible value and must not be manufactured from null.

## Freeze, execution, and audit

The immutable lock captures all six datasets' input files, module code,
design documents, exact default configurations and the protocol before the
first production inference run. All six outputs use those same defaults.
Code fixes found before freeze are documented by independent review. After
freeze, changed sources/inputs are rejected; a substantive scientific change
requires a separately identified later study, never overwrite of this one.

New fictional tests cover detection/repair isolation, query/source exclusion,
leave-entity-out reliability, exact strings, ties and duplicates, catalog
constraints, full-row rollback and serialization/provenance. An independent
code reviewer checks integration before the run. No old benchmark or completed
test suite is rerun merely for this study.

The foreground inference process holds the shared exclusive flock for its
entire lifetime, queues when occupied, serializes datasets, limits numerical
libraries to one thread and preserves a 12-GiB address-space guard:

`${ZEROCLEAN_HEAVY_LOCK:-/tmp/zeroclean-heavy-experiment.lock}`

Execution cohort is `local-unified-loop-v2-20261028`. Cloud and older source
timings remain separate. Original inputs, old predictions, original sandbox
and historical receipts are never rewritten. Output belongs only in
`results/unified_loop_v2/`, with frozen per-dataset predictions and a six-table
completion record. Baseline caches are copied byte-for-byte with their
reconstruction caveats.

Only after every output is frozen may a separate evaluator open the six clean
references. The existing exact-string harness records precision, recall and
F1 for detection and repair. An independent recount checks all method/dataset
counts, repair-mask containment and the shared protocol provenance. Wrong
dirty repairs count as both FP and FN; unchanged clean cells are never repair
TP. Gains are unweighted six-dataset absolute F1 percentage points.

Both measured source-baseline comparisons and official-reproduction limits
must be reported honestly. Historical official ZeroEC prompt identity remains
unproven. The new six-dataset results are reported even if any target fails;
no post-score adjustment or outcome-based mixture is permitted in v2.
