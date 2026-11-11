"""Independent fictional regressions. Never load benchmark rows or artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout

import numpy as np

from experimental.unified_loop_v2.consensus import discover_consensus
from experimental.unified_loop_v2.repair import select_repairs
from experimental.unified_loop_v2 import run_study as runner
from experimental.unified_loop_v2 import test_integration as integration
from experimental.unified_loop_v2 import execute_cohort as controller


@dataclass(frozen=True)
class FictionalTable:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


class IndependentInvariantTests(unittest.TestCase):
    @staticmethod
    def catalog_rows(values):
        columns = ("product_id", "product_name", "manufacturer_id", "manufacturer_name",
                   "city", "state", "ounces", "entity_id", "source", "event_time")
        rows = tuple((f"P{i}", f"Product{i}", f"M{i}", f"Maker{i}", f"City{i}",
                      f"State{i}", "12", entity, source, value)
                     for i, (entity, source, value) in enumerate(values))
        return integration.fictional_table(rows, columns)

    def test_invalid_iso_offset_cannot_become_a_temporal_repair(self):
        rows = []
        for entity, day in (("E0", "01"), ("E1", "02")):
            for source, departure in (
                ("a", "malformed"),
                ("b", f"2030-01-{day}T08:10+01:60"),
                ("c", f"2030-01-{day}T08:10+01:60"),
            ):
                rows.append((entity, source, departure, f"2030-01-{day}T12:00+00:00"))
        table = FictionalTable(
            ("entity_id", "source", "departure_time", "arrival_time"), tuple(rows)
        )
        result = discover_consensus(table)
        self.assertFalse(
            any("+01:60" in proposal["value"]
                for proposals in result["candidates"].values() for proposal in proposals),
            "Out-of-range timezone minutes must fail validation, without normalization.",
        )

    def test_exact_source_duplicates_cannot_enable_temporal_discovery(self):
        rows = (
            ("E0", "a", "malformed"), ("E0", "b", "08:10"), ("E0", "c", "08:10"),
            ("E1", "a", "malformed"), ("E1", "b", "09:10"), ("E1", "c", "09:10"),
        )
        columns = ("entity_id", "source", "observation")
        base = discover_consensus(FictionalTable(columns, rows))
        duplicated = discover_consensus(FictionalTable(
            columns, rows + tuple(row for row in rows if row[1] != "a") * 20
        ))
        self.assertEqual(base["candidates"], duplicated["candidates"])
        self.assertEqual(base["diagnostics"]["status"], duplicated["diagnostics"]["status"])

    def test_two_thirds_consensus_remains_actionable_at_repair_boundary(self):
        rows = tuple(
            (entity, source, "08:10" if entity == "E1" and source == "d" else value)
            for entity in ("E0", "E1")
            for source, value in (("a", "malformed"), ("b", "08:10"),
                                  ("c", "08:10"), ("d", "09:10"))
        )
        table = FictionalTable(("entity_id", "source", "event_time"), rows)
        proposal = discover_consensus(table)["candidates"][(0, 2)][0]
        self.assertTrue(proposal["actionable"])
        self.assertAlmostEqual(proposal["agreement"], 2 / 3)
        candidate = {
            "value": proposal["value"], "origins": ["key_source_consensus"],
            "delta_features": [0.0] * 4, "rule_gain": 0.0,
            "new_rule_violation": False, "candidate_format_valid": True,
            "consensus": proposal,
        }
        mask = [[False, False, index == 0] for index in range(len(rows))]
        selected = select_repairs(table, mask, {(0, 2): [candidate]})
        self.assertEqual(selected["repairs"], {(0, 2): "08:10"})

    def test_source_only_peers_cannot_manufacture_catalog_cts_witness(self):
        values = tuple((entity, source, "08:11" if entity == "E0" and source == "a"
                        else "08:10" if entity == "E0" else "09:10")
                       for entity in ("E0", "E1") for source in ("a", "b", "c", "d"))
        table = self.catalog_rows(values)
        runtime = integration.fictional_runtime(real_detection=True, real_consensus=True)
        scores = np.full(table.shape, .1)
        scores[0, 9] = .9
        cts = {(0, 9): {"retrieved_row_indices": [], "candidates": [{"value": "08:10"}]}}
        result = runner.infer_table(table, scores, scores, {}, {}, {"rules": []}, cts, runtime)
        self.assertTrue(result["diagnostics"]["repair"]["catalog_safeguard"])
        self.assertEqual(result["pool"][(0, 9)][0]["peer_rows"], [1, 2, 3])
        self.assertEqual(result["repairs"], {}, "Source-only rows must not become raw CTS votes.")

    def test_over_cap_cts_specific_peers_remain_in_full_cell_vote_union(self):
        table = self.catalog_rows(tuple(("E0", str(i), value)
                                  for i, value in enumerate(("bad", "08:10", "08:10", "08:10", "09:10", "09:10"))))
        mask = np.zeros(table.shape, dtype=bool)
        mask[0, 9] = True
        cts = {(0, 9): {"retrieved_row_indices": [1, 2, 3], "candidates": [
            {"value": "08:10"}, {"value": "third"},
            {"value": "over cap", "peer_rows": [4, 5]},
        ]}}
        pool, _diagnostics = runner.build_candidate_pool(table, mask, {}, {(0, 9): "seed"}, cts)
        self.assertEqual([item["value"] for item in pool[(0, 9)]], ["seed", "08:10", "third"])
        for candidate in pool[(0, 9)]:
            candidate.update(delta_features=[0.] * 4, rule_gain=0.,
                             new_rule_violation=False, candidate_format_valid=True)
        selected = select_repairs(table, mask, pool)
        self.assertEqual(selected["repairs"], {}, "Opposing CTS rows from pruned strings still vote.")
        candidate = next(item for item in selected["decisions"][0]["candidate_reviews"] if item["value"] == "08:10")
        self.assertEqual(candidate["cts_evidence"]["peer_rows"], [1, 2, 3, 4, 5])

    def test_package_initializer_changes_or_additions_invalidate_freeze(self):
        for package, initial in (("zeroclean", True), ("experimental", False)):
            with self.subTest(package=package), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runtime = integration.FrozenStudyTests().fixture(root)
                initializer = root / package / "__init__.py"
                if initial:
                    initializer.write_text("# FICTIONAL INITIALIZER\n")
                runner.freeze_study(root, runner.DATASETS, runtime)
                initializer.write_text("# FICTIONAL CHANGED/ADDED INITIALIZER\n")
                with self.assertRaises(ValueError):
                    runner.verify_freeze(root)

    def test_controller_holds_one_lock_through_preflight_freeze_and_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "fictional-heavy.lock"
            runtime = object()
            events = []

            def locked_stage(stage):
                with lock_path.open("a+") as handle:
                    with self.assertRaises(BlockingIOError):
                        runner.fcntl.flock(handle.fileno(), runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
                events.append(stage)

            def load_runtime():
                locked_stage("runtime")
                return runtime

            def preflight(actual_root, actual_runtime):
                self.assertEqual(actual_root, root)
                self.assertIs(actual_runtime, runtime)
                locked_stage("preflight")
                return {"failures": 0}

            def freeze(actual_root, datasets, actual_runtime):
                self.assertEqual(actual_root, root)
                self.assertEqual(tuple(datasets), runner.DATASETS)
                self.assertIs(actual_runtime, runtime)
                locked_stage("freeze")

            def run(actual_root, datasets, actual_runtime):
                self.assertEqual(actual_root, root)
                self.assertEqual(tuple(datasets), runner.DATASETS)
                self.assertIs(actual_runtime, runtime)
                locked_stage("run")

            with mock.patch.object(runner, "HEAVY_LOCK", lock_path), \
                    mock.patch.object(runner, "bound_environment", side_effect=lambda: events.append("bounded")), \
                    mock.patch.object(runner, "_runtime", side_effect=load_runtime), \
                    mock.patch.object(controller, "schema_preflight", side_effect=preflight), \
                    mock.patch.object(runner, "freeze_study", side_effect=freeze), \
                    mock.patch.object(runner, "_run_study", side_effect=run), \
                    mock.patch.object(runner, "_sha256", return_value="fictional-lock-hash"), \
                    redirect_stdout(io.StringIO()):
                controller.run_cohort(root)
            self.assertEqual(events, ["bounded", "runtime", "preflight", "freeze", "run"])
            with lock_path.open("a+") as handle:
                runner.fcntl.flock(handle.fileno(), runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
                runner.fcntl.flock(handle.fileno(), runner.fcntl.LOCK_UN)

    def test_controller_schema_failure_never_freezes_or_runs_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "fictional-heavy.lock"
            with mock.patch.object(runner, "HEAVY_LOCK", lock_path), \
                    mock.patch.object(runner, "bound_environment"), \
                    mock.patch.object(runner, "_runtime", return_value=object()), \
                    mock.patch.object(controller, "schema_preflight", return_value={"failures": 1}), \
                    mock.patch.object(runner, "freeze_study") as freeze, \
                    mock.patch.object(runner, "_run_study") as run, \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Schema preflight failed"):
                    controller.run_cohort(root)
                freeze.assert_not_called()
                run.assert_not_called()
            with lock_path.open("a+") as handle:
                runner.fcntl.flock(handle.fileno(), runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
                runner.fcntl.flock(handle.fileno(), runner.fcntl.LOCK_UN)

    def test_controller_schema_failure_does_not_print_exception_values(self):
        runtime = mock.Mock()
        runtime.load_dirty.side_effect = ValueError("FICTIONAL_SENSITIVE_CELL_VALUE")
        output = io.StringIO()
        with redirect_stdout(output):
            summary = controller.schema_preflight(Path("/fictional-never-read"), runtime)
        self.assertEqual(summary["failures"], 6)
        self.assertFalse(summary["inference_performed"])
        self.assertFalse(summary["sandbox_executed"])
        self.assertNotIn("FICTIONAL_SENSITIVE_CELL_VALUE", output.getvalue())

    def test_controller_import_loads_no_scientific_modules(self):
        code = ("import sys; from experimental.unified_loop_v2 import execute_cohort; "
                "assert not ({'numpy', 'sklearn', 'rapidfuzz'} & set(sys.modules))")
        completed = subprocess.run([sys.executable, "-c", code], cwd=runner.ROOT,
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
