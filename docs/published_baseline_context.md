# Published baseline context and reproduction limits

Research-only audit, 2026-10-26. This document uses author manuscripts, official author repositories, publisher metadata, and static source inspection. It does not use experiment predictions, failure rows, supplied detection masks, or benchmark table contents. No inference was performed and no implementation was changed. Published aggregate scores here are comparison context and must not be supplied to inference workers.

Machine-readable companion: [published_baseline_reference.json](../results/published_baseline_reference.json), classified as `uncontrolled_reference_not_reproduced_measurement`.

## Precisely stated published ZeroED results

The source is the [ZeroED author manuscript, arXiv:2504.05345v1](https://arxiv.org/html/2504.05345v1), Tables II–III. Values are transcribed at the paper's printed precision; they are not graph estimates or full-precision measurements.

| Dataset | Published tuples | Published attributes | Published error % | ZeroED ED F1 |
| --- | ---: | ---: | ---: | ---: |
| Hospital | 1,000 | 20 | 4.82 | 0.811 |
| Flights | 2,376 | 7 | 34.51 | 0.722 |
| Beers | 2,410 | 11 | 12.98 | 0.774 |
| Rayyan | 1,000 | 11 | 29.19 | 0.732 |
| Billionaire | 2,615 | 22 | 9.84 | 0.767 |
| Movies | 7,390 | 17 | 4.97 | 0.765 |

Settings: Qwen2.5-72b; temperature 0; 4096 output tokens; 5% labeling; two related attributes; twenty-tuple batches; three-run means. Section III-C uses k-means. ZeroED needs no manual labels; label-dependent baselines receive two tuples. Precision/recall use correctly detected cells; F1 is their harmonic mean. [§IV-A](https://arxiv.org/html/2504.05345v1).

Five datasets contain real errors; Billionaire contains injected errors. Beers was manually cleaned; Movies uses Magellan duplicate labels; Rayyan uses owner cleaning. The inspected settings do not identify input hashes. [§IV-A](https://arxiv.org/html/2504.05345v1).

Algorithm 1 applies criterion-pass filtering without an explicit fallback for all-clean representatives or empty criteria. Section III-B serializes NULL as an empty string. [Methods](https://arxiv.org/html/2504.05345v1).

No approximate scores were extracted. These are ED scores; they are not ZeroEC correction scores or results for the present experiment.

## Which public source snapshot is being reproduced

The current adapter pins [WelkinNi/ZeroED `49678b8470180ab17e9c40bcedd2c00ac49de6ff`](https://github.com/WelkinNi/ZeroED/tree/49678b8470180ab17e9c40bcedd2c00ac49de6ff), committed 2025-07-11. The [repository URL printed in the manuscript](https://github.com/ZJU-Data-Governance-and-Services-Team/ZeroED) currently redirects to [ZJU-DataVIP-Lab/ZeroED](https://github.com/ZJU-DataVIP-Lab/ZeroED), a fork whose sole branch points to `2b84266e1d8b2014d36643d343429d9a20bd769e`. Its public metadata reports the last push on 2025-02-28. These are different source snapshots.

Static comparison found that their `utility.py` files are byte-identical, but their `main.py` files are not. The following behavior occurs in both inspected snapshots:

- The fresh clustering call requests `RANDOM`, including in the [paper-linked fork](https://github.com/ZJU-DataVIP-Lab/ZeroED/blob/2b84266e1d8b2014d36643d343429d9a20bd769e/main.py#L463). In [the pinned implementation](https://github.com/WelkinNi/ZeroED/blob/49678b8470180ab17e9c40bcedd2c00ac49de6ff/feature.py#L299), that branch selects random representatives with NumPy seed zero and assigns every row to its nearest representative. A `KMeans` branch exists but is not selected by the fresh entrypoint.
- The [local model call](https://github.com/WelkinNi/ZeroED/blob/49678b8470180ab17e9c40bcedd2c00ac49de6ff/utility.py#L107) literally names `./qwen2.5-72bs-instruct`; the API branch names `qwen2.5-7b-instruct`. Neither is a pinned model-weight hash. These literals must not be rewritten as proof of the exact model revision behind the published table.
- The [committed configuration](https://github.com/WelkinNi/ZeroED/blob/49678b8470180ab17e9c40bcedd2c00ac49de6ff/run_config.yaml) enables historical result rereading. A fresh run with this project's queue and declared seeds is a newly generated source execution, not a replay of the original responses.

The paper's tabulated dataset signatures and a shared dataset name do not certify that this project's input bytes are the published variants. The project loader selects `<dataset>_error-01.csv`; the pinned ZeroED feature code also hardcodes that suffix, while its main entrypoint has configurable error-rate suffixes. This audit inspected those path expressions, not table contents. See [project loader](../zeroclean/data.py), [feature loader](https://github.com/WelkinNi/ZeroED/blob/49678b8470180ab17e9c40bcedd2c00ac49de6ff/feature.py#L272), and [main loader](https://github.com/WelkinNi/ZeroED/blob/49678b8470180ab17e9c40bcedd2c00ac49de6ff/main.py#L844).

## Source evaluator and normalization facts

ZeroED's [final evaluator](https://github.com/WelkinNi/ZeroED/blob/49678b8470180ab17e9c40bcedd2c00ac49de6ff/measure.py#L3) reads both CSVs through Pandas with `dtype=str` and default missing-value recognition, then fills missing values with the literal `nan`. It defines truth by inequality of the resulting cells. Precision is correct detected cells divided by detected cells; recall uses the total unequal cells. Denominators include `1e-8`. No lowercase, whitespace, edit-distance, or semantic-equivalence normalization appears in that final metric path. The function named `normalize_string` in `main.py` instead normalizes serialized LLM response matching; it is not a final-reference normalization policy.

The pinned [intermediate label diagnostic](https://github.com/WelkinNi/ZeroED/blob/49678b8470180ab17e9c40bcedd2c00ac49de6ff/main.py#L451) separately treats a clean-side `nan` as erroneous when constructing `gt_wrong_dict`. This differs from final `measure_detect`, which only tests inequality. The source adapter omits this clean-aware intermediate diagnostic, as recorded in [baseline_audit.md](baseline_audit.md).

ZeroEC's [correction evaluator](https://github.com/YangChen32768/ZeroEC/blob/d76ef45509d7e0f9e7cc0fa98d445c06afbb157c/correction_zero_shot.py#L1200) uses exact equality after its CSV loaders apply Pandas missing-value handling and fill with `null`. A repair is a changed cell; a correct repair must equal the reference; recall's denominator is the original unequal cells. In the regular human-assisted entrypoint, [the evaluation call](https://github.com/YangChen32768/ZeroEC/blob/d76ef45509d7e0f9e7cc0fa98d445c06afbb157c/correction.py#L1272) uses the already human-repaired input as its dirty argument. Thus that code's denominator excludes errors already repaired in the seeded representative rows. This is a source-code fact; the inaccessible full paper's exact reporting protocol has not been verified.

The requested project evaluator instead preserves CSV strings, including empty fields and missing-value tokens, without Pandas NA conversion. It uses exact cell inequality and explicit zero-denominator rules. Its source is [data.py](../zeroclean/data.py) and [evaluation.py](../zeroclean/evaluation.py). A result under that contract is not automatically the same metric instance as a published or upstream Pandas-parsed result, even though the formulas have the same purpose.

## Empty-predicate source behavior

This section records static mechanics separately from the published claims.

In [pinned `gen_err_funcs`](https://github.com/WelkinNi/ZeroED/blob/49678b8470180ab17e9c40bcedd2c00ac49de6ff/main.py#L190), no error-labeled representative causes an immediate `False` return. [`task_func_gen`](https://github.com/WelkinNi/ZeroED/blob/49678b8470180ab17e9c40bcedd2c00ac49de6ff/main.py#L266) converts this to empty clean/dirty criterion lists. The later [clean-label filter](https://github.com/WelkinNi/ZeroED/blob/49678b8470180ab17e9c40bcedd2c00ac49de6ff/main.py#L967) removes a pseudo-clean cell when `pass_num / (len(criteria) + 1e-6) < 0.5`; an empty criterion list yields zero and removes it. The [paper-linked fork](https://github.com/ZJU-DataVIP-Lab/ZeroED/blob/2b84266e1d8b2014d36643d343429d9a20bd769e/main.py#L967) contains the same filter and earlier no-error return.

There is no guard in those inspected paths preserving pseudo-clean labels when the final criterion set is empty. If only synthetic error-labeled training examples remain, the [training function](https://github.com/WelkinNi/ZeroED/blob/49678b8470180ab17e9c40bcedd2c00ac49de6ff/main.py#L401) does not reject single-class labels. These are conditional code consequences, not a conclusion about how often they occur or why any experiment scored poorly. The existing adapter audit documents preservation of this path; no new experiment was run for this report.

## ZeroEC evidence and access limits

The official [ZeroEC README](https://github.com/YangChen32768/ZeroEC/blob/d76ef45509d7e0f9e7cc0fa98d445c06afbb157c/README.md) permits different LLMs and asks for dirty data plus error-detection results. The [publisher record](https://ieeexplore.ieee.org/document/11113193/) identifies the original paper, DOI [10.1109/ICDE65448.2025.00221](https://doi.org/10.1109/ICDE65448.2025.00221). Publisher full text remained robot-gated; direct PDF endpoints returned HTTP 418. No accessible primary full text was recovered in the bounded search. Therefore no precise published ZeroEC per-dataset model, score, normalization, or label-budget claim is made here.

Verified public-code assumptions are narrower:

- [`correction.py`](https://github.com/YangChen32768/ZeroEC/blob/d76ef45509d7e0f9e7cc0fa98d445c06afbb157c/correction.py#L1105) defaults to ten representative rows, fills selected rows from `clean_data`, and uses those repairs to create dataset-specific guidance. Its model strings are configuration placeholders. This path is human-assisted and is excluded from the project's dirty-only experiment.
- [`correction_zero_shot.py`](https://github.com/YangChen32768/ZeroEC/blob/d76ef45509d7e0f9e7cc0fa98d445c06afbb157c/correction_zero_shot.py#L1232) sets the human budget to zero and the correction model to `gpt-3.5-turbo-0125` at temperature zero. Its [prompt call](https://github.com/YangChen32768/ZeroEC/blob/d76ef45509d7e0f9e7cc0fa98d445c06afbb157c/correction_zero_shot.py#L1063) still includes fixed general correction examples, while leaving dataset-specific examples blank.
- [`perfect_detection.py`](https://github.com/YangChen32768/ZeroEC/blob/d76ef45509d7e0f9e7cc0fa98d445c06afbb157c/perfect_detection.py) can construct a supplied detection file from reference/dirty inequality. This audit did not open supplied mask files and does not assert that every provided mask has that origin. A ZeroED-predicted mask and a reference-derived mask are distinct experimental conditions.
- The zero-shot program's three named prompt files are missing from the pinned tree. Author-source candidates are available, but historical system-prompt identity remains unproven. See [zeroec_prompt_recovery.md](zeroec_prompt_recovery.md).

## Comparison interpretation for the requested protocol

The current requested protocol uses `gpt-6.1-sol` with `xhigh` reasoning, the raw-string evaluator, fresh dirty-only requests, no supplied human labels, no fixed correction examples, and source ZeroED decisions as the correction mask. The model designation here is the user's experiment setting, not an assertion about the upstream papers. The project preserves the pinned source's `RANDOM` representative path and recorded empty-predicate behavior. ZeroEC remains an explicitly labeled reconstruction because its original missing system prompt is unverified.

These are concrete protocol differences. They make the run suitable for a declared comparison under common present-day settings; they do not establish reproduction of published numeric performance. A low score under this protocol cannot, by itself, distinguish model effects, input-variant differences, parsing/evaluation differences, stochastic generation, or a source edge case. Conversely, reproducing a public source branch faithfully does not certify that the branch matches the configuration used for a paper's table.

Keep published Table III scores in a separately labeled research-reference column. Compare new methods against freshly executed baselines under the same requested protocol, retaining fidelity labels. Do not replace a missing completed baseline with the published numbers or describe an adapted/reconstructed pipeline as an exact published-performance reproduction.
