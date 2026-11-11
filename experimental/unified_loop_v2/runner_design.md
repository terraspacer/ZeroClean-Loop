# Unified Loop v2 orchestration contract

`run_study.py` is a new, isolated orchestrator. It never imports an evaluator.
It consumes only the six fixed raw dirty CSVs and, for each dataset, the five
fixed JSON inputs named below. It does not open earlier metrics, reports,
evaluation failures, clean/reference tables, reference masks, external metadata
or catalogs, or previous reviewer judgments. Its prior-design disclosure says
that this is a user-directed follow-up informed by prior aggregate observations,
not a newly blind study.

The one method is `unified_loop_v2`. Its evaluator filename is predetermined as
`zeroclean_loop.json`. Both the plan and protocol lock bind that mapping before
inference. Prediction metadata repeats `system="unified_loop_v2"`.

## Freeze and immutable input boundary

The CLI has exactly two phases:

```sh
python experimental/unified_loop_v2/run_study.py --phase freeze
python experimental/unified_loop_v2/run_study.py --phase run
```

`--datasets` defaults to Hospital, Flights, Beers, Rayyan, Billionaire and Movies
in that order. A freeze requires all six names, regardless of whether a later
run selects a subset. Duplicates and partial coverage fail. The runner serializes
the requested datasets. A subset run cannot create a global completion manifest
until all six dataset freezes exist and pass verification.

For every dataset `<dataset>`, the exact input allowlist is:

* `vendor/ZeroED/data/<dataset>_error-01.csv`
* `artifacts/llm/<dataset>/pass2.json`
* `artifacts/llm/<dataset>/proposals_cts_v2.json`
* `results/source_baselines_v1/predictions/<dataset>/zeroed.json`
* `results/source_baselines_v1/predictions/<dataset>/zeroed_zeroec_reconstruction.json`
* `results/conservative_seed_v1/predictions/<dataset>/empty_carry.json`

The freeze hashes all 36 files without running inference or sandbox rules. It
also hashes all Python and Markdown files immediately in this new study folder,
including all tests/design documents and the parent-owned `protocol.md`, and
the four reused `zeroclean/{data,types,statistics,sandbox}.py` files. The bound
sources also include `zeroclean/__init__.py`, `experimental/__init__.py` and this
study's `__init__.py`, recording an explicit absent sentinel for a missing
initializer. A later addition, removal or edit invalidates the lock. A static
import audit found no other mutable local transitive inference imports beyond
these package initializers and the already bound four helpers and three new
modules; remaining dependencies are Python standard-library or third-party
packages. Exact module
default configurations are serialized using `DetectionConfig`, the consensus
mapping proxy, and `RepairConfig`. The runner configuration, model
`gpt-6.1-sol`/`xhigh`, method mapping, cohort, allowlist and design disclosure are
part of `plan.json`; `protocol_lock.json` binds its exact bytes plus the code and
input hashes. Both files are under `experimental/unified_loop_v2`.

An identical freeze is idempotent. A changed lock, source, input, configuration,
or plan is rejected. Initial freezing is refused if study output already exists.
Every run validates the complete global lock before loading a table, calculating
statistics, or creating any file in `results/unified_loop_v2`. It rechecks the
lock before publishing outputs and before final completion. No `--force` path
exists.

## Detection and repair separation

Probability carriers must have rectangular finite numeric matrices in `[0,1]`
matching the dirty table. They must declare threshold `0.5` and operator `>`.
If an upstream binary mask is present, it must equal `probabilities > 0.5`;
ties at `0.5` remain negative. Malformed JSON, duplicate JSON keys, nonfinite
constants, invalid shapes and inconsistent decision masks fail closed.

The source and empty carriers contribute probabilities and metadata only. The
runner never indexes, validates, or uses the `empty_carry` repair map. It loads
seed replacement strings separately from the source sequential reconstruction
file. Seeds are not passed into detection. No `seeded_carry` file is read or
emitted.

The dirty rows, source scores/metadata, empty scores, validated rule OR matrix,
and actionable original dirty-peer contradictions are the detector's inputs.
`contextual_low_support[i,j]` is true only when the consensus module supplies an
exact nonoriginal proposal with `origin="key_source_consensus"` and the fixed
module flag `actionable is True`. The runner does not recalculate actionability
from a retrieval score or a seed. It passes the full consensus record, including
its nested evidence and unchanged actionable flag, under `candidate["consensus"]`.

Immediately after detection, the runner copies and hashes the score matrix and
`M_det`, verifies their strict-threshold equivalence, and makes both NumPy arrays
read-only. Repair selection and row rollback cannot change either. A final hash
check verifies this invariant.

## Sandbox criteria and counterfactual evidence

Cached pass2 rules are validated on every original dirty row in batches of at
most 1,000. Per-rule error counts are aggregated over the full table before the
strict `<5%` acceptance decision; a difficult batch alone does not discard a
globally usable rule. Syntax-invalid rules and tables without validation rows
cannot pass. Accepted rules are then evaluated in bounded batches on unchanged
dirty tuples.

A target-specific rule whose successful raw evaluations are all true is
nondiscriminating for detection and is excluded only from the detection OR.
Every other accepted target-specific violation remains available. Accepted
constant-false rules and accepted all-true rules stay in the counterfactual and
combined-row guard set. Diagnostics record each rule's successful/failed and
violation counts, resolved targets, detection applicability, and guard status.

For each positive detection cell, the pool retains at most three distinct exact
nonoriginal strings in this order: actionable key/source proposals, the naive
sequential seed, then CTS proposals in their existing order. Null means
abstention, without conversion to an empty string. Exact duplicates merge their
origins, full proposal/context records, source labels and deduplicated nonquery
peer rows, including duplicate records encountered after the cap has filled.
No candidate is synthesized or normalized. Consensus peer rows are recovered
only by matching the original dirty entity/source key; CTS peer indices come
from its fixed cached cell. A separate `cts_peer_rows` field on every retained
candidate contains the complete CTS-only per-cell peer union, collected before
the cap, identity and null filtering. Opposing peers attached to a pruned/no-op
proposal still vote. Generic merged `peer_rows` can contain consensus-only rows
for audit, but those rows never become CTS witness votes. The selector never
receives a candidate outside `M_det`.

`DirtyIndex` is constructed exclusively from the original dirty table and
pass2 dependencies. Counterfactual rows replace one exact cell at a time and
are sandboxed in chunks of at most 1,000. Four features are calculated:

1. Mean original-minus-counterfactual violation over accepted rules targeting
   the proposed column. `rule_gain` is the corresponding count difference.
2. Candidate-minus-original leave-one-out conditional FD support.
3. `tanh((candidate_vicinity - original_vicinity) / 4)`.
4. `tanh((candidate_domain_pattern_support - original_support) / 4)`.

No applicable target rule supplies zero rule delta/gain with an explicit zero
applicability count. Any `0 -> 1` transition in any accepted rule sets
`new_rule_violation`; any `-1` in the original or candidate vector sets
`counterfactual_error` and invalidates the criterion. Nonfinite features fail.

Format validity requires either the original dirty pattern or at least two
dirty exemplars of the candidate pattern. Patterns use the unchanged
`zeroclean.statistics.pattern` function. Control/format characters and line
separators are rejected. An empty candidate, an absent observed nonempty
grammar, or a novel/singleton differing pattern records explicit uncertainty
and cannot authorize a repair. These checks do not normalize the string.

After `select_repairs`, all selected edits to the same row are checked together
against every accepted criterion. Any new violation or original/counterfactual
runtime abstention rolls back every selected edit in that row. The detection
matrix and scores remain fixed. Final repairs must be inside `M_det`, must
differ from the dirty original, and must exactly match a retained candidate.

## Outputs, resource guard and verification

Every production artifact is under `results/unified_loop_v2`: primary and
byte-copied baseline predictions, JSONL candidate/decision/rollback audit traces,
compact dirty-only diagnostics, per-dataset `frozen/<dataset>.json`, append-only
progress events and the eventual six-dataset `completion_manifest.json`.
Dataset freezes expose the existing evaluator interface (`dirty_sha256`,
`prediction_files`, `inference_inputs`, `frozen_before_evaluation`) and add the
shared lock hash, method mapping, cohort, explicit `ground_truth_access=false`,
and audit/diagnostic hashes. Source ZeroED and qualified source sequential
baselines are copied byte-for-byte, preserving their original caveats and
metadata. Empty/seeded carrier arms are not additional prediction outputs.

Completed dataset outputs are verified and preserved on resume. Any changed
completed file is rejected. An incomplete dataset with existing output files
is also refused rather than overwritten; its progress log remains available
for inspection. Exclusive atomic publication never replaces a destination.

The foreground `--phase run` acquires and holds the exact shared exclusive
`heavy-experiment.lock` through inference, all output publication, and cohort
completion. It blocks in the queue if occupied. Before importing NumPy or other
scientific modules, it sets BLAS/OpenMP-related thread counts to one and limits
address space to at most 12 GiB without loosening a tighter existing cap. There
is no detached subprocess launcher. The only recorded run timings belong to
`local-unified-loop-v2-20261028`; previous/cloud timings are not combined.

`execute_cohort.py` is a small foreground controller for the parent-authorized
production launch. It holds one `heavy_execution` context across the read-only
all-six schema check, `freeze_study`, and `_run_study` in that sequence. Schema
failure prevents both freeze and inference; its console receipt prints only
dataset names, dimensions, counts and fixed failure stages/types. It never
executes sandbox rules or computes features during preflight. The controller
adds no method/configuration behavior and is included by the source manifest's
Python-file inventory. The author did not launch it.

Operational queued-attempt receipt: the initial schema-only foreground attempt
used PID 3296862 / exec session 13326 and blocked before the heavy lock. At the
parent's explicit request it was terminated with SIGTERM (exit 143) to avoid a
separate queue/release/requeue cycle. It never acquired the lock, imported
scientific modules, loaded real artifacts, froze anything or created study
output. No other holder/waiter was interrupted. The parent will launch the
reviewed controller after its narrow code review.

`test_integration.py` contains 23 fictional tests, using temporary directories
and stub sandbox/index implementations. It additionally exercises the real
three new modules on invented entity/source clocks. Coverage includes detector
input independence from seed changes, fixed-mask row rollback, candidate
cap/order/provenance, rule applicability and the strict aggregated error gate,
1,000-row counterfactual batches, malformed/nonfinite carriers, immutable
all-six freeze coverage, exact baseline byte copies, old-input preservation,
completion/resume behavior, exclusive writes and the foreground flock lifetime.
No real-data freeze, real inference, evaluation, or old test suite was run by
the orchestration author. Parent-owned independent review and global freeze
remain required before production execution.
