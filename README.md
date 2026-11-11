# ZeroClean-Loop

Supplemental code, sandboxed rule verifier, fallback-gated intrinsic detector, counterfactual candidate scorer, frozen cohort outputs, and reproduction scripts for **ZeroClean-Loop: Closed-Loop Counterfactual Error Detection and Repair for Autonomous Tabular Data Ingestion**.

---

## Repository Structure

| Path | Description |
| --- | --- |
| `zeroclean/` | Core library: exact-string CSV loader (`data.py`), syscall-confined rule sandbox (`sandbox.py`), leave-one-out support and NMI vicinity index (`statistics.py`), counterfactual feature engine (`loop.py`), verification queue (`verification.py`), closed-loop state driver (`closed_loop.py`), exact-string evaluator (`evaluation.py`), and upstream ZeroED/ZeroEC adapters (`upstream_zeroed.py`, `upstream_zeroec.py`). |
| `experimental/unified_loop_v6/` | ZeroClean-Loop v6 cohort implementation: fallback-gated intrinsic detection and single-determinant FD discovery (`detection.py`), key-grouped multi-source consensus (`consensus.py`), counterfactual candidate scoring and inverse-violation weighted whole-row guard (`repair.py`), cohort runner (`run_study.py`, `execute_cohort.py`), SHA-256 protocol lock (`protocol_lock.json`, `plan.json`), and v6 unit/integration tests (`test_*.py`). |
| `experimental/` | Cohort runners and matched study implementations for `unified_loop_v2`–`v5`, format-contract overlays (`format_overlay.py`), and Flights source-consensus specifications (`latent_consensus*.py`). |
| `scripts/` | Benchmark bootstrap (`bootstrap_benchmarks.py`), independent rational-arithmetic v6 auditor (`audit_unified_loop_v6_results.py`, `recount_v6_metrics.py`), cohort evaluators (`evaluate_unified_loop_v2.py`–`v6.py`), and paper table/figure generators (`generate_icde_tables.py`, `generate_icde_figures.py`, `f4p_style.py`). |
| `paper/` | Generated LaTeX macro/table artifacts (`generated_macros.tex`, `generated_table_*.tex`), single-source fact sheet (`paper_facts.json`), and rendered vector/raster figures (`figures/fig_example.*` through `figures/fig_diagnostics.*`, `figures/figure_data.json`). |
| `results/unified_loop_v6/` | Frozen ZeroClean-Loop v6 six-dataset predictions (`predictions/`), per-dataset SHA-256 freeze manifests (`frozen/`), stage diagnostics (`diagnostics/`), evaluator summary (`summary_metrics.json`), and independent audit receipt (`independent_audit.json`, `independent_audit.md`). |
| `results/` | Frozen `summary_metrics.json` and diagnostic artifacts for progressive cohorts (`unified_loop_v2`–`v5`, `source_v1`, `format_overlay_v1`, `conservative_seed_v1`) and ablation studies (`components_source_v2`, `flight_model_spec_v1`). |
| `docs/` | Technical notes on upstream ZeroED/ZeroEC baseline reconstruction, prompt recovery, dirty-only protocol boundaries, and closed-loop verification. |
| `tests/` | Unit and synthetic integration test suite covering rule sandboxing, leave-one-out statistics, CTS retrieval, counterfactual scoring, closed-loop exports, and evaluation gates. |
| `third_party.lock.json` | Exact Git repository URLs, commit SHAs, and SHA-256 digests for pinned `ZeroED` and `ZeroEC` benchmark checkouts plus the `v1.0.0` release bundle of large JSONL cell-level audit logs and frozen input packets. |

---

## Environment Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[test]"
```

Clone the pinned `ZeroED` (`49678b8470180ab17e9c40bcedd2c00ac49de6ff`) and `ZeroEC` (`d76ef45509d7e0f9e7cc0fa98d445c06afbb157c`) checkouts under `vendor/`, verify all 12 benchmark CSV SHA-256 digests, and download/unpack the SHA-256-verified `v1.0.0` release bundle (`zeroclean-loop-v6-audit-and-inputs.tar.gz`, containing the multi-megabyte cell-level `.jsonl` audit logs and 54 pinned input files):

```bash
.venv/bin/python scripts/bootstrap_benchmarks.py --download-release-bundle
```

---

## Mapping Paper Sections, Tables, and Figures to Artifacts

### 1. Benchmark Profile & Protocol Boundary (Section II-B, Section V-A, Table I)
- **54 SHA-256-pinned inference inputs across six datasets:** `experimental/unified_loop_v6/protocol_lock.json` and `results/unified_loop_v6/frozen/<dataset>.json`
- **Dataset dimensions, admitted rules $|\mathcal{Q}_{\mathrm{adm}}|$, candidate pools $|\mathcal{P}_{\mathrm{cand}}|$, flagged counts $N_{\mathrm{pos}}$, and local inference times $t_{\mathrm{inf}}$:** `results/unified_loop_v6/diagnostics/<dataset>.json` -> `paper/generated_table_datasets.tex`
- **Independent 146-binding, 18-output exact rational recount (219 integer fields, 128 ratio fields, 0 mismatches):** `results/unified_loop_v6/independent_audit.json`

### 2. Running Example & Execution Traces (Section I, Section III-A, Section IV-B, Figs. 1–3)
- **Flights entity `AA-404-MIA-MCO` source reports & predictions:** `vendor/ZeroED/data/flights_error-01.csv` and `results/unified_loop_v6/predictions/flights/`
- **Per-cell gate decisions, witnesses, counterfactual $\Delta\mathbf{f}_{i,j}(v')$ vectors, and whole-row guard logs:** `results/unified_loop_v6/audit/flights.jsonl` -> `paper/figures/figure_data.json`

### 3. Parallel Six-Dataset Comparison & Progressive Cohorts (Section V-B, Section V-C, Tables II–III, Fig. 4)
- **Source ZeroED, Sequential ZeroED+ZeroEC, Adapted baselines, and ZeroClean-Loop v6:** `results/unified_loop_v6/summary_metrics.json`
- **Seed-carry baselines (`empty_carry`, `seeded_carry`):** `results/conservative_seed_v1/summary_metrics.json`
- **Progressive cohorts (`source_v1`, `format_overlay_v1`, `unified_loop_v2`–`v6`):** `results/<study>/summary_metrics.json` -> `paper/generated_table_benchmark.tex` and `paper/generated_table_trajectory.tex`

### 4. Component, Seed-Carry, Grouping & Repair-Policy Ablations (Section V-D, Table IV, Fig. 5)
- **First-pass component ablation (`w/o MC`, `w/o CF`, `Full`):** `results/components_source_v2/summary_metrics.json` -> `paper/generated_table_ablation.tex`
- **Flights source-grouping & empirical Bayes specification (`attribute_near_*`, `profile_complete_*`):** `results/flight_model_spec_v1/summary_metrics.json`
- **Strict v4 vs. inverse-violation weighted v6 repair policy:** `results/unified_loop_v4/summary_metrics.json` and `results/unified_loop_v6/summary_metrics.json`

### 5. Cell-Level Diagnostics & Failure Accounting (Section V-E, Fig. 6)
- **Rayyan per-column non-empty coverage $\eta_j$ and gate behavior:** `results/unified_loop_v6/diagnostics/rayyan.json`
- **Exact cell-fate accounting across all 247,084 benchmark cells:** computed directly from `results/unified_loop_v6/predictions/` and recorded in `paper/figures/figure_data.json`

---

## Independent Rational-Arithmetic Recount

After running `scripts/bootstrap_benchmarks.py --download-release-bundle`, verify all 146 SHA-256 bindings and recount all 18 v6 system outputs (`zeroed`, `zeroed_zeroec_reconstruction`, `zeroclean_loop` across all six datasets) using Python's standard library and `fractions.Fraction`:

```bash
.venv/bin/python scripts/recount_v6_metrics.py
```

To run the standalone auditor's built-in synthetic gate and tamper-detection self-tests (requires no benchmark files):

```bash
.venv/bin/python scripts/audit_unified_loop_v6_results.py --self-test
```

---

## Regenerating All Paper Tables and Figures

```bash
.venv/bin/python scripts/generate_icde_tables.py
.venv/bin/python scripts/generate_icde_figures.py
```

`generate_icde_tables.py` writes `paper/generated_macros.tex`, `paper/generated_table_*.tex`, and `paper/paper_facts.json`. `generate_icde_figures.py` compiles the TikZ schematics (requires `tectonic` and `pdftoppm`) and renders the `figures4papers`-styled matplotlib plots (`paper/figures/fig_*.pdf` and 300 DPI `.png`), enforcing an automated bounding-box overlap check before saving.

---

## Running Unit Tests

```bash
.venv/bin/python -m pytest -q
```
