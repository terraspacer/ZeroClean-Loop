"""Fictional cohort controller checks; no benchmark inputs or labels are read."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experimental.unified_loop_v5 import execute_cohort as controller
from experimental.unified_loop_v5 import run_study as runner
from experimental.unified_loop_v5 import test_integration


class CohortControllerTests(unittest.TestCase):
    def fixture(self, root):
        runtime = test_integration.FrozenStudyTests().fixture(root)
        runtime.sandbox._validate_source = lambda _source: None
        return runtime

    def test_preflight_covers_every_role_without_inference_or_rule_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.fixture(root)
            statuses = []
            result = controller.preflight(root, runtime, lambda *args, **kwargs: statuses.append((args, kwargs)))
            self.assertEqual(result["counts"]["dataset_count"], 6)
            self.assertEqual(result["counts"]["input_file_count"], 54)
            self.assertEqual(result["counts"]["rows"], 18)
            self.assertEqual(len(statuses), 6)
            self.assertEqual(runtime.detection.calls, [])
            self.assertEqual(runtime.sandbox.validation_batches, [])
            self.assertEqual(runtime.sandbox.evaluation_batches, [])
            self.assertFalse((root / runner.LOCK).exists())
            self.assertFalse((root / runner.OUTPUT).exists())

    def test_controller_holds_one_lock_from_preflight_through_serialized_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.fixture(root)
            path = root / "heavy-experiment.lock"
            stages = []

            def assert_locked(stage, **_details):
                with path.open("a+") as handle:
                    with self.assertRaises(BlockingIOError):
                        runner.fcntl.flock(handle.fileno(), runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
                stages.append(stage)

            with mock.patch.object(runner, "HEAVY_LOCK", path), mock.patch.object(runner, "bound_environment"), \
                 mock.patch.object(runner, "_runtime", return_value=runtime), mock.patch.object(controller, "_status", side_effect=assert_locked), \
                 mock.patch("builtins.print"):
                completed = controller.execute(root)
            self.assertTrue(completed["six_dataset_inference_complete"])
            self.assertEqual(len(runtime.detection.calls), 6)
            self.assertEqual(stages, ["cohort_preflight", "cohort_preflight_complete", "global_freeze_complete", "cohort_complete"])

    def test_invalid_cached_identity_fails_before_freeze_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.fixture(root)
            path = root / runner.input_paths("movies")["old_pass2"]
            body = runner._read_json(path)
            body["dirty_sha256"] = "a" * 64
            path.write_bytes(runner._json_bytes(body))
            with self.assertRaises(ValueError):
                controller.preflight(root, runtime, lambda *_args, **_kwargs: None)
            self.assertFalse((root / runner.LOCK).exists())
            self.assertFalse((root / runner.OUTPUT).exists())

    def test_no_scientific_runtime_import_before_environment_bounds(self):
        events = []

        @contextmanager
        def bounds():
            events.append("bounded_and_locked")
            yield

        def runtime():
            self.assertEqual(events, ["bounded_and_locked"])
            raise RuntimeError("fictional stop before any input access")

        with mock.patch.object(runner, "heavy_execution", bounds), mock.patch.object(runner, "_runtime", side_effect=runtime), \
             mock.patch.object(controller, "_status"), self.assertRaises(RuntimeError):
            controller.execute(Path("/fictional-unused-root"))


if __name__ == "__main__":
    unittest.main()
