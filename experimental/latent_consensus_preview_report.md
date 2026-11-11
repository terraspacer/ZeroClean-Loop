# Flights: frozen, assumption-bound preview v1

All four raw time attributes were fitted separately under two disclosed hypotheses: distinct source names act independently, or conservative near-agreement groups act independently. The event key is raw `flight`; each attribute is hypothesized to describe one exact-string snapshot value. Observation dates, time zones, source lineage, aliases, and mixed-event confounding are unknown. No correctness labels or benchmark evaluation were used. No preview was applied to the live model.

The runner froze [parameters and assumptions](latent_consensus_flights_preview/pre_fit_manifest.json), model source, and [protocol](latent_consensus_flights_preview/protocol_frozen.md) before fitting. The fixed thresholds were not changed after seeing coverage. [The summary](latent_consensus_flights_preview/summary.json) contains every stable proposed string with its original string, row, source, and entity. Compressed per-assumption files retain complete row decisions, source fits, fold failures, and uncertainty diagnostics. [Artifact hashes](latent_consensus_flights_preview/artifact_manifest.json) also record validation that every original string and candidate's dirty-table support matched the raw input.

| Attribute | Independent names: proposed rows | Clustered sources: proposed rows | Same proposal passing both assumptions | Stable proposal entities |
| --- | ---: | ---: | ---: | ---: |
| Scheduled departure | 90 | 0 | 0 | 0 |
| Actual departure | 0 | 0 | 0 | 0 |
| Scheduled arrival | 173 | 138 | 116 | 46 |
| Actual arrival | 589 | 0 | 0 | 0 |

These are candidate coverage and stability counts, not detected-error counts or accuracy estimates. A row can abstain because it is blank, its source is unsupported, a fit failed, or uncertainty gates failed. Requiring agreement under both assumptions substantially changes the usable output.

Scheduled departure has 245 rows supported under both assumptions, with the same winning string, but none is a proposed change in both. Scheduled arrival has 963 rows supported under both assumptions, with no differing winning strings among them; 116 are proposed changes in both. A shared winner does not establish a true value.

The 116 stable scheduled-arrival proposals include:

- 26 with the same parsable 12-hour clock prefix in original and candidate, such as `9:40aDec 1` to `9:40 a.m.`. Removing a date annotation may discard meaningful context, even when the clock prefix agrees.
- 84 with different parsable clock prefixes, all involving `flightaware` or `wunderground` observations. Shared source behavior, differing event snapshots, or different field meanings remain possible explanations.
- 6 from the exact literal `Not Available` to a time string. Version 1 defines only blank/whitespace as missing; these rows illustrate unresolved sentinel semantics rather than verified corruption.

The clock-prefix diagnostic used `^\s*(\d{1,2}):(\d{2})\s*([ap])` case insensitively, converted valid 12-hour clocks to minutes, and ignored suffixes. It is descriptive and was not used to change candidates or thresholds. The source totals are `travelocity` 5, `orbitz` 16, `mytripandmore` 11, `wunderground` 42, and `flightaware` 42.

Actual-departure fits hit the lower reliability boundary in at least half the source parameters in every fold under both assumptions, so all rows abstain. The maximum observed boundary fraction is about 0.536 for independent names and 0.727 for clusters. Actual-arrival clusters likewise fail this gate in every fold, with a maximum fraction of 1.0; one fold also fails the fixed convergence criterion. Independent actual-arrival has one boundary-degenerate fold. These findings expose conflict with the positively oriented one-coin model and do not identify which sources are wrong.

The one-coin restriction `q > 0.5` for every source is stronger than categorical better-than-chance behavior when there are more than two possible categories. It may force weak sources to a boundary. A separately preregistered mathematical sensitivity experiment can investigate this restriction without changing v1 or its thresholds. Likewise, typed time equivalence, explicit missing sentinels, and observation-time semantics require a separate specification; none was introduced after examining v1 coverage.

The focused synthetic suite passes 10 tests. It covers independent-source estimation, entity-disjoint fitting, duplicate invariance, conflicting-source exclusion, clone grouping, low support, unknown-category mass, missingness, temporal context separation, invalid roles, and deterministic output. This validates implementation behavior under synthetic assumptions, not real-world correctness. The strongest supported conclusion is that source-assumption sensitivity is material and that the scheduled-arrival subset is a reviewable next experiment. No general ED/EC improvement across six benchmarks has been established.
