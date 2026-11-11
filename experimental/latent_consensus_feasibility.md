# Dirty-only structural feasibility

Inspected only the six tables returned by `zeroclean.data.load_dirty`. No reference tables, masks, existing predictions, historical judgments, or benchmark results were read. This structural inspection fitted no model and changed no cells; the subsequent explicitly hypothetical preview is documented [separately](latent_consensus_preview_report.md). The accompanying [JSON profile](latent_consensus_feasibility.json) records exact dirty fingerprints, column statistics, parameters, source groupings, and diagnostic counts. Counts refer to raw strings, not verified identities.

| Dataset | Dirty shape | Observed repeated structure | Source-consensus feasibility |
| --- | --- | --- | --- |
| Hospital | 1,000 by 17 | `ProviderNumber`: 72 nonempty raw keys; 45 repeated keys cover 973 rows | No source column. Repeated measures for a provider are not independent observers. Scores and samples vary by measurement. |
| Flights | 2,376 by 6 | `flight`: 100 keys, all repeated; `src`: 38 names across the table | Only explicit source/entity schema. Missing observation date, source lineage, exact-value semantics, and conditional independence remain unestablished. |
| Beers | 2,410 by 9 | `id` is unique; `brewery_id`: 559 nonempty keys, 431 repeated keys cover 2,281 rows | Brewery repetition concerns different beers, with no independent source identifiers. Brewery locations may support a separate consistency experiment only after semantic validation. |
| Rayyan | 1,000 by 11 | `id` is unique; `journal_issn`: 615 nonempty raw keys, 50 repeated keys cover 119 rows | Journal repetition does not create independent article observations. ISSNs and journal names may have legitimate alternate representations. |
| Billionaire | 2,614 by 22 | `Name` plus `Year`: 2,422 nonempty keys, 45 repeated keys cover 135 rows | No sources. Years cannot serve as observers of one fixed truth: age, wealth, location, rank, and other facts can change. Repeated raw names can also be invalid identifiers. |
| Movies | 7,390 by 17 | `Id` has 7,390 distinct keys and no repeated keys | No sources or repeated movie-ID observations. Shared titles and directors do not establish the same entity. Many attributes contain legitimate lists. |

None of the six supplies sufficient evidence, on its own, to assert the full fitting contract. The prototype therefore defaults to structural reporting and abstention on all six.

## Flights: source dependence is material before fitting

These are full-table structural stress-test counts. Training folds would independently estimate their source groups; these groups are not reused in a fit. An edge requires at least 12 shared nonempty observations and at least 0.98 exact agreement. Its connected components conservatively cap evidence. Agreement can reflect shared correctness, so none of these edges establishes copying.

| Attribute | Nonempty source/entity pairs | Source names with observations | Near-agreement edges | Conservative groups | Group/item conflicts excluded | Entities with different group claims |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sched_dep_time` | 1,592 | 36 | 208 | 11 | 3 | 67 |
| `act_dep_time` | 2,000 | 31 | 60 | 12 | 7 | 100 |
| `sched_arr_time` | 1,606 | 36 | 39 | 18 | 5 | 87 |
| `act_arr_time` | 1,994 | 32 | 30 | 16 | 22 | 100 |

There are no same-source/entity conflicting nonempty strings in these four raw attributes and no duplicate full rows in any of the six tables. Group conflicts above arise after conservative near-agreement components combine sources. Raw source-name counts substantially overstate the evidence available under this stress test.

For scheduled departure, 43 of 100 entities have fewer than four surviving groups; they already fail the default candidate support gate in this full-table illustration. The corresponding counts are 4, 2, and 2 for actual departure, scheduled arrival, and actual arrival. Passing this count cannot establish single-valuedness, independence, reliability, or a correct candidate. Fold-specific support, model stability, and sensitivity gates can require further abstention.

The dominant limitation is provenance, not a missing optimizer. A label-free source discovery experiment is plausible for Flights if its acquisition design and observation scope can be established. For the other five, inventing sources from row numbers, years, record IDs, or shared attributes would manufacture independent votes. This extension does not do that and cannot support a claim of unsupervised ED/EC improvement across all six.

The [protocol](latent_consensus_protocol.md) fixes eligibility checks, parameters, uncertainty language, synthetic validation, and the next experiment before any benchmark evaluation.
