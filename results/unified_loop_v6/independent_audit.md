# Unified Loop v6 independent aggregate audit

Status: **passed**. Evaluated outputs: **18/18**.
Mismatches: **0**. Hash receipts: **146**.

Direct standard-library CSV/JSON recount; exact strings; stored strict `p > 0.5` masks; changed edits inside the mask.
Repair recall includes every original error. Wrong edits are FP; unrepaired or wrongly repaired original errors are FN.

## Independent qualified comparison

All five quantitative checks: **not all passed**.

- `ed_mean_gain`: pass
- `ed_each_dataset`: pass
- `flights`: pass
- `rayyan`: fail
- `repair_mean_gain`: pass

- `ed_macro_gain_exact`: 45068286908900375085854119620290072678795/160266344611463467018102974955787286170724 = 0.281208678080
- `repair_macro_gain_exact`: 8959843256951113069607644509847037/61077188519844281304579874439778120 = 0.146697048016

Exact macro F1 values and per-dataset aggregate gains are recorded as rational numbers in the JSON audit.

Official exact ZeroEC success remains **unverified**: the faithful exact baseline is absent and recovered prompt identity remains unproven.
The available sequential reconstruction is a qualified comparison, with zero fixed examples.
The study follows prior aggregate feedback and is not a previously untouched benchmark validation.

No row values, coordinates, labeled examples, or failure records are returned. Historical study artifacts are unchanged.

The full path/hash and package-initializer absence receipt is in `independent_audit.json`.
