"""Only this module loads benchmark references. Never import it from inference."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from .data import read_csv, sha256


def prf(tp: int, predicted: int, actual: int):
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (predicted + actual) if predicted + actual else 0.0,
            "tp": int(tp), "fp": int(predicted - tp), "fn": int(actual - tp),
            "predicted": int(predicted), "actual": int(actual)}


def evaluate(table, prediction, root="vendor/ZeroED/data", threshold=.5, failure_limit=50):
    prediction.validate(table.shape)
    path = Path(root) / f"{table.name}_clean.csv"
    columns, rows = read_csv(path)
    if columns != table.columns or len(rows) != len(table.rows):
        raise ValueError(f"Reference alignment mismatch: {table.name}")
    dirty, clean = np.asarray(table.rows, dtype=object), np.asarray(rows, dtype=object)
    truth = dirty != clean
    detected = prediction.probabilities >= threshold
    if "upstream_binary_decisions" in prediction.metadata:
        decisions=np.asarray(prediction.metadata["upstream_binary_decisions"])
        if decisions.shape!=table.shape or decisions.dtype!=np.bool_:
            raise ValueError("Invalid original-source detection mask")
        if not np.array_equal(decisions,prediction.probabilities>.5):
            raise ValueError("Original-source decisions disagree with classifier probabilities")
        detected=decisions
    corrected = dirty.copy()
    for (i, j), value in prediction.repairs.items():
        if value != dirty[i,j] and not detected[i,j]:
            raise ValueError("A repair must be included in the reported detection mask")
        corrected[i,j] = value
    changes = corrected != dirty
    correct_repairs = changes & truth & (corrected == clean)
    # Wrong repairs on dirty cells count as FP and FN; unchanged clean cells never TP.
    metrics = {"detection": prf(int((truth & detected).sum()), int(detected.sum()), int(truth.sum())),
               "repair": prf(int(correct_repairs.sum()), int(changes.sum()), int(truth.sum())),
               "rows": len(rows), "columns": len(columns), "dirty_sha256": table.fingerprint,
               "reference_sha256": sha256(path), "reference_policy": "exact strings; no normalization"}
    failures = {}
    for label, mask in {"false_positives": detected & ~truth,
                        "false_negatives": truth & ~detected,
                        "wrong_repairs": changes & ~correct_repairs}.items():
        failures[label] = [{"row": int(i), "column": columns[j], "dirty": dirty[i,j],
                            "predicted": corrected[i,j], "reference": clean[i,j],
                            "probability": float(prediction.probabilities[i,j])}
                           for i,j in np.argwhere(mask)[:failure_limit]]
    failures["policy"] = "Evaluation-only diagnostics; forbidden input to zero-shot optimization."
    return metrics, failures


def assess_targets(metrics, baseline_ed="zeroed", baseline_repair="zeroed_zeroec", system="zeroclean_loop"):
    names = ("hospital", "flights", "beers", "rayyan", "billionaire", "movies")
    missing = [f"{n}/{v}" for n in names for v in (baseline_ed, baseline_repair, system)
               if v not in metrics.get(n, {})]
    targets = {"ed_mean_absolute_gain": .035, "repair_mean_absolute_gain": .08,
               "flights_ed_f1_strictly_above": .76, "rayyan_ed_f1_strictly_above": .77,
               "ed_improves_each_dataset": True}
    if missing:
        return {"met": False, "status": "unverified_missing_baselines_or_runs", "missing": missing, "targets": targets}
    ed = {n: metrics[n][system]["detection"]["f1"] - metrics[n][baseline_ed]["detection"]["f1"] for n in names}
    ec = {n: metrics[n][system]["repair"]["f1"] - metrics[n][baseline_repair]["repair"]["f1"] for n in names}
    checks = {"ed_mean_gain": sum(ed.values())/6 >= .035,
              "ed_each_dataset": all(v > 0 for v in ed.values()),
              "repair_mean_gain": sum(ec.values())/6 >= .08,
              "flights": metrics["flights"][system]["detection"]["f1"] > .76,
              "rayyan": metrics["rayyan"][system]["detection"]["f1"] > .77}
    return {"met": all(checks.values()), "status": "verified", "targets": targets, "checks": checks,
            "ed_gains": ed, "repair_gains": ec, "ed_macro_gain": sum(ed.values())/6,
            "repair_macro_gain": sum(ec.values())/6}
