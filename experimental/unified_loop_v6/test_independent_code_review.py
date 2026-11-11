"""Independent boundary regressions; every table and artifact is fictional."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experimental.unified_loop_v6 import run_study as runner


ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndependentBoundaryTests(unittest.TestCase):
    def test_cached_and_fresh_union_is_uniform_and_excludes_repair_strings(self):
        executable = {"target": "value", "kind": "single", "description": "fictional criterion",
                      "source": "def check(row):\n    return row['value'] == 'bad'"}
        first = None
        for dataset in runner.DATASETS:
            table = SimpleNamespace(name=dataset, columns=("entity", "value"), fingerprint="f" * 64)
            identity = {"dataset": dataset, "dirty_sha256": table.fingerprint}
            old = {**identity, "rules": [{**executable, "id": "cached-rule"}], "dependencies": [],
                   "repairs": ["fictional repair field must not enter the union"]}
            fresh = {**identity, "rules": [{**executable, "id": "fresh-duplicate"},
                                          {**executable, "id": "fresh-other", "source": "def check(row):\n    return False"}],
                     "dependencies": []}
            result = runner.merge_rule_artifacts(table, old, fresh)
            self.assertEqual(len(result["rules"]), 2)
            self.assertEqual(result["rules"][0]["input_rule_origins"],
                             [{"origin": "cached", "id": "cached-rule"},
                              {"origin": "fresh", "id": "fresh-duplicate"}])
            self.assertNotIn("repairs", result)
            self.assertNotIn("fictional repair field", json.dumps(result))
            if first is None:
                first = result
            else:
                self.assertEqual(result, first)

    def test_all_three_independent_allowlists_bind_same_nine_roles(self):
        scorer = load_script("evaluate_unified_loop_v6")
        auditor = load_script("audit_unified_loop_v6_results")
        expected_roles = {"dirty", "pass2", "old_pass2", "fresh_packet", "fresh_response",
                          "cts", "source", "sequential", "empty"}
        for dataset in runner.DATASETS:
            paths = runner.input_paths(dataset)
            self.assertEqual(set(paths), expected_roles)
            self.assertEqual(scorer.input_paths(dataset), paths)
            self.assertEqual(auditor.input_paths(dataset), paths)

    def fixture(self, root):
        dataset = runner.DATASETS[0]  # A path key only; invented CSV content.
        paths = {role: root / path for role, path in runner.input_paths(dataset).items()}
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        paths["dirty"].write_text("entity,value\nfictional-a,one\nfictional-a,two\n", encoding="utf-8")
        dirty_hash = runner._sha256(paths["dirty"])
        packet = {"schema": "dirty_table_profile_v1", "table": {"id": dataset, "sha256": dirty_hash},
                  "columns": [{"name": "entity"}, {"name": "value"}], "row_samples": [],
                  "shape": {"rows": 2, "columns": 2}}
        paths["fresh_packet"].write_text(json.dumps(packet), encoding="utf-8")
        response = {"rules": [], "dependencies": [{"determinants": ["entity"], "dependent": "value",
                                                  "min_support": 3, "confidence": 1}]}
        paths["fresh_response"].write_text(json.dumps(response), encoding="utf-8")
        artifact = {"schema_version": runner.RULE_ARTIFACT_SCHEMA, "dataset": dataset,
                    "dirty_sha256": dirty_hash, "model": runner.MODEL,
                    "provenance": {"dirty_only": True, "clean_reference_access": False,
                                   "evaluation_feedback_access": False, "external_metadata_access": False,
                                   "input_packet_sha256": runner._sha256(paths["fresh_packet"]),
                                   "response_sha256": runner._sha256(paths["fresh_response"])}, **response}
        paths["pass2"].write_text(json.dumps(artifact), encoding="utf-8")
        paths["old_pass2"].write_text(json.dumps({"dataset": dataset, "dirty_sha256": dirty_hash,
                                                "rules": [], "dependencies": []}), encoding="utf-8")
        return dataset, paths, artifact, packet, response

    def test_accept_profile_generator_nested_packet_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, _paths, _artifact, _packet, _response = self.fixture(Path(directory))
            runner.validate_artifact_files(Path(directory), dataset)

    def test_exact_response_binding_distinguishes_boolean_from_number(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, paths, artifact, _packet, response = self.fixture(root)
            response = deepcopy(response)
            response["dependencies"][0]["confidence"] = True
            paths["fresh_response"].write_text(json.dumps(response), encoding="utf-8")
            artifact["provenance"]["response_sha256"] = runner._sha256(paths["fresh_response"])
            paths["pass2"].write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.validate_artifact_files(root, dataset)

    def test_packet_and_response_raw_bytes_remain_bound(self):
        for role in ("fresh_packet", "fresh_response"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset, paths, _artifact, _packet, _response = self.fixture(root)
                paths[role].write_bytes(paths[role].read_bytes() + b"\n")
                with self.assertRaises(ValueError):
                    runner.validate_artifact_files(root, dataset)

    def test_nested_packet_wrong_dirty_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, paths, artifact, packet, _response = self.fixture(root)
            packet["table"]["sha256"] = "a" * 64
            paths["fresh_packet"].write_text(json.dumps(packet), encoding="utf-8")
            artifact["provenance"]["input_packet_sha256"] = runner._sha256(paths["fresh_packet"])
            paths["pass2"].write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.validate_artifact_files(root, dataset)


if __name__ == "__main__":
    unittest.main()
