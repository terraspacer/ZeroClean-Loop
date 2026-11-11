# Dirty-only latent consensus experiment, version 1

This experiment is isolated from ZeroClean-Loop. It produces structural diagnostics and, only under explicit assumptions, candidate previews. It never updates the input table or the live model. No benchmark reference, error mask, evaluation result, existing prediction, prior proposal, prior judgment, or benchmark answer catalog was used to design or test it.

## Research basis and deliberate restrictions

[Dawid and Skene (1979)](https://doi.org/10.2307/2346806) estimate observer error parameters without observed true labels using a latent categorical model and EM. This prototype uses a smaller one-coin model: one scalar agreement parameter per source group and attribute, with a symmetric distribution over wrong categories. It does not estimate a complete source confusion matrix.

[Allman, Matias, and Rhodes (2009)](https://arxiv.org/abs/0809.5032) establish generic identifiability for specified latent structures under independence and rank conditions. Merely counting three sources does not establish these conditions. The overlap and variation checks here are necessary practical screens, not a proof that source independence, model identifiability, or truth orientation holds. Distinct source names do not establish independent errors.

[Dong, Berti-Equille, and Srivastava (2009)](https://www.vldb.org/pvldb/vol2/vldb09-335.pdf) model source dependence and changing truth using update histories. These dirty tables do not supply such histories. Near-agreement grouping here is a conservative sensitivity test; it neither identifies copying direction nor distinguishes shared correctness from shared mistakes.

## Contracts and explicitly hypothetical previews

The caller must supply an entity key, an attribute, a source identifier, optional time/context columns, and written evidence or explicit hypotheses for all five contracts below. All flags default to false. A table profile cannot activate them. `Contract.hypothetical=True` records an authorized assumption-bound investigation when table contents cannot establish the contracts. Such a fit cannot be interpreted as identified truth or promoted automatically.

1. Each entity and context has exactly one appropriate categorical value. Entity keys themselves must identify the same object; typographic key errors are not silently linked.
2. Source groups give conditionally independent observations given that value, and different base entities supply independent training units. Source lineage or acquisition design can support this without human correctness labels. Schema repetition alone cannot.
3. The source-group reliability parameter is sufficiently stationary across the included entities and contexts. Training should be separated by attribute and by any known incompatible acquisition regime.
4. Sources are more likely correct than wrong, individually, on the modeled exact-string representation. The reliability constraint above one half provides truth orientation; the model cannot discover this assumption from agreement alone.
5. Missingness and the selection of multiply observed items are ignorable for the intended source comparison. Empty observations are omitted. Missing strings are never imputed by this version.

Single-valuedness and independent provenance are semantic inputs, not human labels for dirty cells. If they cannot be supported, the default output is an abstaining structural report. Explicitly hypothetical fits may explore sensitivity without making correctness claims. There is no source-reliability experiment for the five tables that lack any source provenance in their supplied schema. Flights has a suitable schema but still lacks a verified independence and observation-time contract.

## Fixed protocol before any benchmark evaluation

The defaults below are precommitted experimental choices. They were not selected using benchmark outcomes. Synthetic tests establish implementation behavior; they do not establish benchmark utility or calibrate real-data correctness.

| Choice | Version 1 default |
| --- | --- |
| Random seed / entity folds | 1701 / 5 deterministic SHA-256 folds |
| Compared source assumptions | All distinct names independent; conservative near-agreement clusters |
| Training support | At least 30 distinct base entities; 20 items per source group |
| Training views | At least 3 groups per item; each group participates in a triple sharing at least 12 items; overlap graph connected at threshold 12 |
| Disagreement support | At least 10 training items contain different nonempty claims after grouping |
| Candidate support | At least 4 groups, permitting a three-view leave-one-group-out check |
| Potential dependence screen | At least 12 shared training items and agreement at least 0.98 |
| Source parameter penalty / bounds | Beta(2, 2) penalty; constrained to [0.5001, 0.995] |
| EM starts / termination | 0.55, 0.75, 0.95; 200 iterations; maximum parameter change at most 1e-7 |
| Unobserved-category prior | 0.10; sensitivity fits at 0.05 and 0.20 |
| Source-fit sensitivity | All starts, both additional prior values, 12 entity-cluster bootstrap fits |
| Candidate threshold | Same observed-string winner in every sensitivity fit; conditional weight at least 0.98; margin at least 0.20 |
| Leave-one-group-out threshold | Same winner after removing each group, in every sensitivity fit; conditional weight at least 0.95 |
| Fold rejection | Insufficient support, any nonconvergence, or at least half the source parameters at a constraint boundary in any fit |

All contexts of one base entity share a fold. Source clustering, reliability fitting, source support pruning, and bootstrap sampling use only training entities for that fold. A held-out entity contributes no information to its source parameter fit. The full-table structural profile is separate and is not reused as the fitted grouping.

The authorized exploratory Flights preview treats each raw `flight` plus attribute as one snapshot event value. It hypothesizes one correct exact-string representation, independent base flight events, stationary positively oriented source reliability, ignorable missingness, and no mixed observation dates. None is established by raw table inspection. Source/entity pairs have at most one nonempty value in the observed data, which is compatible with this event model but cannot validate its snapshot or semantics. Run each of the four time attributes separately under both fixed source assumptions. Keep only candidate strings that independently pass all gates in both modes and agree; report remaining proposals and abstentions separately. Do not adjust thresholds for desired coverage. The runner freezes source, protocol, hashes, parameters, and hypotheses before fitting.

Within a source/entity/context pair, exact duplicate claims count once. If that source supplies distinct strings, all its claims for that item are excluded and the original row indices and strings remain in the diagnostics. Source groups are connected components of the near-agreement graph. Within a group/item, distinct values cause exclusion; agreement produces one observation. Transitive grouping is deliberately conservative and can exclude many otherwise useful observations. It is not a validated dependency model.

Training support is pruned until every remaining source group has the required number of items containing at least three groups. Disconnected components are rejected rather than comparing their unanchored reliability scales. Training includes unanimous and conflicting items; selecting only disagreements would distort the reliability estimate. Insufficient disagreement rejects a degenerate fit.

## Model and uncertainty semantics

For an item with K observed exact-string candidates, the model adds one latent unobserved category. Candidate priors share mass 1 minus rho uniformly; the additional category has mass rho. A source group with parameter q assigns q to a claim matching a latent category, and (1 minus q) divided by K to each alternative. EM updates q with a Beta penalty and explicit bounds.

This is a transductive model conditional on an observed, item-specific vocabulary. The unobserved category guards against forced selection among observed strings, but is only one approximation to open-world truth. It does not enumerate missing values or establish a coherent distribution over arbitrary unobserved strings. It cannot generate a repair string absent from the dirty observations for that item. Exact-string disagreements can reflect aliases, display formats, time zones, legitimate sets, or changing facts rather than errors.

Outputs report a minimum conditional model weight and margin over a finite ensemble, a maximum unobserved-state weight, leave-one-group-out sensitivity, and fitted reliability ranges. Bootstrap ranges are sensitivity ranges, not confidence intervals. Beta-penalized point estimates do not integrate parameter uncertainty. Stability across starts does not prove global optimization or identifiability. Neither a 0.98 conditional weight nor a q value estimates an empirical correctness rate or supplies a false-repair guarantee.

The unknown-value prior and symmetric error distribution remain strong assumptions. A shared systematic mistake can receive high weight even after these checks. Correlated sources that are not almost identical may remain undetected. Conversely, independently correct sources may be merged. A source-group parameter pools members whose reliability may differ. A boundary fit is exposed and conservatively rejected when widespread. Low-quality or adversarial sources, unobserved correct strings, and nonignorable missingness can invalidate the model.

## Reproduction and output provenance

Run only the focused synthetic suite:

```bash
.venv/bin/python -m pytest -q tests/test_latent_consensus.py
```

Generate a new schema-only profile (the destination must not already exist):

```bash
.venv/bin/python -m experimental.latent_consensus --output /tmp/latent-consensus-profile.json
```

The Python APIs are `profile_table`, `profile_claims`, and `run_experiment`. The latter takes a supplied `DirtyTable`, `Contract`, and optional `Parameters`; there is no fitting CLI switch that automatically asserts semantic contracts. The dirty table is frozen and remains unchanged.

Every output records the dirty-file fingerprint, module SHA-256, parameters, experiment version, and contract. Candidate records retain zero-based row indices, column names, exact original strings, exact proposed strings, entity/context keys, source identifiers, fold assignment, and active source groups. Group mappings, support failures, conflict rows, model fits, and sensitivity ranges remain inspectable. CLI output uses exclusive file creation. No output is automatically consumed by ZeroClean-Loop.

## Supported next experiment

First retain the structural feasibility report and run synthetic stress cases with independent versus copied sources, incorrect entity links, time-varying truth, missing true strings, multiclass confusions, and nonignorable missingness. Their truth is generated by construction, so they do not expose benchmark labels.

For real-data investigation, Flights is the only supplied schema with an explicit source/entity arrangement. Establish an observation-time and source-lineage contract using provenance that contains no cell correctness labels when available. The authorized assumption-based diagnostic preview marks source independence and event semantics as hypothetical and keeps its candidates separate from the live model. It cannot be described as identified truth or enable repairs automatically.

An entity-disjoint, held-source-out dirty prediction experiment can compare the model with equal-weight consensus using a vocabulary fixed independently of the held-out claim. This would measure predictive fit to dirty observations, not correctness. Do not use item vocabularies constructed from the withheld claim to claim predictive calibration. No such predictive comparison or benchmark evaluation was run here.

Any future benchmark evaluation must freeze this module, parameters, dirty fingerprints, contract, candidate files, and protocol before access. An evaluator may assess that frozen artifact separately. Do not return evaluation results to this design branch or select thresholds from them. No calibration claim or general improvement across all six datasets is supported by the present experiment.
