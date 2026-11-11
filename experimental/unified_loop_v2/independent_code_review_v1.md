# Independent pre-inference code review v1

Status: **code-review gate approved before freeze; no remaining code blockers**.
The completed runner, protocol and component fixes have been reviewed together.
Approval applies to the exact code hashes below. The parent must still complete
the separate allowlisted artifact schema-only compatibility preflight before
freezing production inputs; that preflight is outside this review's data scope.
No benchmark inference, benchmark rows,
reference data, evaluation masks, previous judgments, outcome reports, or metrics
were accessed. No production implementation was edited by this reviewer.

The read scope is the new `experimental/unified_loop_v2` Python/design/protocol
files, `zeroclean/{data,types,statistics,sandbox}.py`, and the code/interface in
`evaluate_e2e.py`. Fictional regression tests are recorded separately in
`reviewer_test_invariants.py`. After integration stabilized, the parent-authorized
combined new fictional suite passed **110 tests in 1.015 seconds, OK**. No old
completed suite was rerun. The reviewer changed only this report and the new
independent fictional test file, leaving production implementation changes to
their authors.

## Pre-fix findings, preserved for the final review

1. **P2 — strict ISO offset validation can accept malformed minutes.**
   `consensus.py:62-69` uses `datetime.fromisoformat` without checking the parsed
   offset fields. That parser normalizes offsets such as `+01:60`, although the
   clock-only parser separately rejects minutes above 59 at lines 84-87. This can
   make a malformed peer timestamp an actionable exact-string proposal and
   affirmative absolute chronology evidence. Require explicit offset-hour/minute
   range validation before constructing a dated absolute timestamp. Fictional
   regression: `test_invalid_iso_offset_cannot_become_a_temporal_repair`.
   Resolution: inspected the author's fix at current `consensus.py:63-69`:
   literal offset hours/minutes are checked before `fromisoformat`. The design
   now states this grammar. The parent reports nine selected fictional checks,
   including both independent regressions, passed. The final independent combined
   suite also passed this regression.

2. **P2 — exact duplicate source rows can enable temporal-role discovery.**
   `consensus.py:193-195` calculates the 80% parseability criterion over all raw
   rows. With an unhinted target column, duplicating valid source observations
   can change the result from nonapplicable to actionable despite unchanged
   distinct observations. Subsequent source votes and reliability contributions
   are capped, but discovery happens before those caps. Deduplicate temporal
   discovery observations under a declared policy so copies cannot establish a
   new temporal role. Fictional regression:
   `test_exact_source_duplicates_cannot_enable_temporal_discovery`.
   Resolution: inspected the author's fix at current `consensus.py:198-205`:
   discovery counts distinct complete exact dirty rows. The updated design
   declares this observation grain while preserving identical values from
   different contexts. The final independent combined suite passed this regression.

3. **Resolved documentation transition during concurrent repair.** The inspected repair source
   now uses shared two-thirds agreement, two winning sources, and a 0.20 margin,
   while the inspected design still stated 0.8 agreement. The author was already
   aligning this boundary. The subsequently inspected `repair_design.md:46-47`
   and `protocol.md:87-90` now state the shared two-thirds/two-winning-source
   boundary. This is retained as historical context, not a current defect.

4. **P1 — runner merged source-only peers into the catalog CTS witness.**
   In the first completed runner, `build_candidate_pool` merged source-consensus
   and CTS rows into one diagnostic `peer_rows` list, then merged the `cts`
   origin. The original selector interpreted every peer row on that item as a
   CTS vote. Three source-only rows could therefore manufacture a strict catalog
   witness when the actual CTS retrieval contained zero rows. The runner also
   discarded candidate-specific CTS peer rows belonging to strings beyond its
   cap, so opposing peers could disappear from the witness denominator. Keep
   a separate, complete per-cell CTS-only union before null/no-op/cap filtering,
   and never infer it from merged generic peers. Independent regressions:
   `test_source_only_peers_cannot_manufacture_catalog_cts_witness` and
   `test_over_cap_cts_specific_peers_remain_in_full_cell_vote_union`.
   Resolution: inspected runner `run_study.py:487-494,529`, which collects the
   complete CTS-only union before candidate filtering and attaches it separately
   to every retained record. Inspected selector `repair.py:243-246`, which uses
   only that explicit field and collects it before null/no-op/cap filtering.
   Both independent regressions and the author's added boundary tests passed
   in the final combined suite. Source-only diagnostic peers supply no CTS votes.

5. **P2 — imported local package initializers were outside the source freeze.**
   The first runner source manifest bound four `zeroclean` modules but omitted
   `zeroclean/__init__.py`, which Python executes before importing them. A
   changed initializer could execute with no source-lock mismatch. Adding an
   `experimental/__init__.py` could similarly affect study imports while
   escaping the study-folder glob. Bind local package initializer bytes and
   namespace-package absence, rejecting later additions as well as changes.
   Independent regression:
   `test_package_initializer_changes_or_additions_invalidate_freeze`.
   Resolution: inspected `run_study.py:40-41,171-182`, which binds content or an
   explicit absence sentinel for all three relevant local package initializers.
   The independent changed/added-initializer regression passed in the final suite.

## Final dataflow and execution observations

- Detection starts from copied empty-state probabilities, gates only inherited
  positives, retains supplied independent rule/statistical support, and leaves
  inherited negatives unchanged. The runner constructs its rule OR from accepted
  target-specific raw-dirty rule evaluations, excluding nondiscriminating all-true
  rules, and its contextual matrix solely from actionable raw-dirty nonidentity
  consensus proposals. Seeds enter only after this mask is fixed.
- Consensus query voting excludes the entire query source, caps duplicate source
  observations, preserves exact raw spellings, requires matching unique raw and
  weighted winners, and subtracts the full query entity from reliability.
- The selector snapshots its mask, checks candidate guards, merges exact
  duplicates conservatively, caps candidate strings, checks catalog FD conflicts,
  and uses the separate CTS-only witness. The runner freezes read-only score/mask
  copies before candidate preparation and verifies their hashes after selection.
  Individual counterfactual and combined-row original/candidate runtime abstentions
  and new violations block edits while retaining detection flags.
- Rule validation aggregates full-table error counts across bounded batches before
  enforcing strictly fewer than 5% failures. Constant-false criteria remain
  counterfactual guards. Counterfactual statistics come only from the unchanged
  dirty table; exact strings and complete consensus nested evidence are preserved.
- The runner applies one shared configuration and predetermined method filename
  across all six datasets. Its global lock binds all 36 allowed input files,
  configuration/defaults, study Python/Markdown, helper sources and package
  initializers before loading a production table or writing result output.
  Verification runs again before publication and completion. No evaluator is
  imported. The emitted completion manifest requires all six verified freezes.
- Completed outputs are verified and reused; incomplete or changed outputs are
  refused. Source baselines are byte-copied and verified. Original inputs and
  historical studies are not rewritten. The inspected dataset-freeze fields
  satisfy the `evaluate_e2e.py` interface.
- The CLI holds the shared flock around the full foreground run, output writes
  and completion. It sets numerical thread counts to one and a 12 GiB or tighter
  address-space soft limit before importing scientific modules. These control
  paths passed fictional mocked resource/actual temporary-lock tests; this review
  did not start a heavy production process.

## Verification and practical limits

The single final invocation was:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 python -m unittest experimental.unified_loop_v2.test_detection experimental.unified_loop_v2.test_consensus experimental.unified_loop_v2.test_repair experimental.unified_loop_v2.test_integration experimental.unified_loop_v2.reviewer_test_invariants -v
```

Result: **110 tests, zero failures/errors, OK**. All inputs were invented fixtures;
freeze/resume/output tests used temporary directories. Integration sandbox/index
fixtures are deliberately stubbed, so this does not claim a production sandbox
or performance run. The unchanged sandbox implementation was reviewed as code;
its old completed suite was not rerun. No outcome/accuracy claims were made.

No production `freeze`, inference or evaluation command was executed. Final
artifact/schema compatibility and subsequent evaluator-only aggregate recount
remain assigned to the parent and separate roles. All four substantive pre-fix
findings above are resolved in inspected code and covered by passing new tests.

## Approved source identities

SHA-256 immediately after the passing final suite:

```text
fbdadecc411edc8e0ab9386c6c9b96f79a712a614dff899f0628c151c0418067  detection.py
476aa6b782d9e50fb3f05c242d8204bd1f7a7ecdfaf57788b868e1ed2e4836b4  consensus.py
90db8f3ee864397ca440b9d8c8dfd354ca629821ad80574aee0ae25649a1c4a8  repair.py
1a59def8ad674aba21df3259d7731df3983779275c946b6d434f848c047f4ef7  run_study.py
46b63cf108759be3ed08a3ba750a83a3e98da317f7be6250048e60f71b6c6e99  protocol.md
3ecd30d83f05f6ba023fadc5d88e6db08b479e2c8d052efc5a43a95ed7415c28  reviewer_test_invariants.py
```

These paths are all relative to `experimental/unified_loop_v2`. This report is
finalized before global freeze so it and the independent tests can themselves
be bound by the source manifest. Subsequent result audits belong elsewhere.

## Pre-freeze queue-controller addendum

**Narrow controller gate approved; no new blockers.** At the parent's request,
reviewed `execute_cohort.py` and its design addition so a waiting run can keep
one shared lock across schema compatibility checking, global freeze, and all-six
inference. `run_study.py`, `detection.py`, `consensus.py` and `repair.py` retain
exactly the previously approved hashes above. This adds no method, candidate,
threshold or inference configuration behavior.

The controller imports only standard-library modules and the lightweight runner
before entering `heavy_execution`. Inside that single context it imports the
scientific runtime, performs only the allowlisted loader/schema checks, refuses
freeze/run on any schema failure, freezes all six, and invokes the unchanged
all-six runner. It does not release/reacquire between stages. The existing source
manifest already includes the new controller through its Python-file inventory.
Schema status output contains fixed stages/types and counts; exception messages
that could contain raw cell values are suppressed.

Ran **only four newly added fictional controller tests**, all passing in
**0.100 seconds**:

- `test_controller_holds_one_lock_through_preflight_freeze_and_run`
- `test_controller_schema_failure_never_freezes_or_runs_and_releases_lock`
- `test_controller_schema_failure_does_not_print_exception_values`
- `test_controller_import_loads_no_scientific_modules`

The first two use a real temporary flock and mocked preflight/freeze/inference;
the third injects fictional exceptions, and the fourth checks a fresh interpreter
import. The completed 110-test suite was not rerun. No real artifact preflight,
freeze, inference or evaluation was performed by this reviewer. The controller
may perform its authorized actual schema-only preflight under the shared lock;
the existing requirement that it succeed before freeze/inference remains enforced.

Updated approved SHA-256 identities for the additions/changed review resources:

```text
485d55dc9d697918629b3acc2f61752efa43c6b5f5458ecb0d6fd821bf0471f3  execute_cohort.py
4f0c32ed13ff0327d017868005f1bf3868700ecf6458ee00661b0d32baaaacc1  reviewer_test_invariants.py
51295304bcf9d75b455ce93aacf3367445c09816b2a9f820312f294e22583957  runner_design.md
```

The prior reviewer-test hash above records the earlier six-test snapshot; this
addendum supersedes it with the ten-test file containing the four new controller
checks. All other approved source identities remain unchanged. This addendum is
also finalized before the global source freeze.
