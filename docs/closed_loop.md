# External verifier feedback

`scripts/run_closed_loop.py` creates a separate immutable experiment. It starts from a hash-bound candidate request, its complete verifier response, and the exact pass2 artifact. It does not load evaluation references or promote predictions into an existing experiment.

```sh
.venv/bin/python scripts/run_closed_loop.py --stage start \
  --payload PATH/request.json --verification PATH/verification.json \
  --artifact PATH/pass2.json --max-rounds 3 --queue
.venv/bin/python scripts/run_closed_loop.py --stage resume \
  --run-dir PATH/TO/PRINTED/RUN --queue
```

Without `--queue`, the driver writes the next request and reports the exact response path it needs. With `--queue`, it uses the shared content-addressed verifier queue. Missing responses produce `awaiting_verification`; they never count as convergence. Each fresh request requires complete cell and candidate judgments matching its own hash.

Each external round runs inference once, uses the external verifier scores, and checks the complete proposed tuples. The round prediction and exact inputs are frozen before the next request is created. The next request keeps `tuple` as the original dirty observation and adds `tentative_tuple`, explicitly marked as untrusted prior model proposals. For a cell's before/after comparison, the target starts at its original dirty value while the other accepted tentative changes are held fixed. Executable-rule, FD, vicinity, and domain deltas are recomputed in that context. Statistical counts and dependencies are built from the original dirty table only; proposed repairs never become observations or labels.

The protocol stops when two completed external rounds have identical repairs and probabilities, or after the configured maximum rounds (three by default). The aggregate prediction records the actual number of completed external rounds and their request, response, and prediction hashes. Every later request binds its parent prediction and input hashes. Resuming a completed run verifies its frozen artifacts and leaves them unchanged.

Candidate eligibility and candidate values remain fixed from the initial prepared request. This implementation does **not** rerun CTS retrieval or discover new candidate cells between rounds. It supplies fresh verifier feedback and recomputed deltas for the fixed candidate set. The final `prediction.json` remains in the versioned run directory for review and later promotion by explicit path.

To export a completed run for separate evaluation:

```sh
.venv/bin/python scripts/export_closed_loop.py --run-dir PATH/TO/COMPLETED/RUN \
  --baseline-root results/predictions --output results/source_v1
```

The exporter validates the completed round chain without running inference or querying models. It makes independent byte copies of the final method prediction, original-source `zeroed.json`, and qualified `zeroed_zeroec_reconstruction.json`. Both baselines must match the dirty fingerprint; the correction reconstruction must use zero fixed examples. The evaluator-compatible freeze records every source hash, exported prediction hash, external round history, and baseline fidelity. The qualified correction reconstruction remains separate from the exact ZeroEC baseline gate. Legacy ablations are excluded. Existing differing outputs are never overwritten, and pending runs cannot be exported.
