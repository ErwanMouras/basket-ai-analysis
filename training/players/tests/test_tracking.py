"""Opt-in real MLflow integration using a temporary database and artifact store."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(
    os.environ.get("PLAYERS_MLFLOW_TESTS") == "1",
    "Set PLAYERS_MLFLOW_TESTS=1 in an MLflow environment",
)
class TrackingTests(unittest.TestCase):
    def setUp(self):
        import mlflow
        from mlflow import MlflowClient

        self.mlflow = mlflow
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.uri = f"sqlite:///{self.root / 'mlflow.db'}"
        self.client = MlflowClient(tracking_uri=self.uri)
        self.experiment_id = self.client.create_experiment(
            "phase1",
            artifact_location=(self.root / "artifacts").as_uri(),
        )
        self.config = {
            "mlflow_uri": self.uri,
            "experiment": "phase1",
            "model": "contract-test",
            "dataset_id": "a" * 64,
            "export_id": "b" * 64,
            "output": str(self.root / "runs"),
            "purpose": "smoke",
        }
        self.environment = patch.dict(os.environ, {"MLFLOW_TRACKING_URI": self.uri})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_player_run_metrics_provenance_and_uri_override(self):
        from training.players.learning.tracking import log_metrics, tracked_run

        global_uri = self.mlflow.get_tracking_uri()
        self.config["mlflow_uri"] = "http://unused.invalid"
        with tracked_run(self.config, {"contract": 1}) as (client, run_id, output):
            log_metrics(client, run_id, {"val/recall": 0.75}, 2)
            self.assertIsNone(self.mlflow.active_run())
        self.assertEqual(self.mlflow.get_tracking_uri(), global_uri)
        runs = self.client.search_runs([self.experiment_id])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].info.status, "FINISHED")
        self.assertEqual(runs[0].data.metrics["val/recall"], 0.75)
        self.assertEqual(
            self.client.get_metric_history(run_id, "val/recall")[0].step, 2
        )
        self.assertEqual(self.config["mlflow_uri"], self.uri)
        environment = json.loads((output / "environment.json").read_text())
        files = environment["training_sha256"]
        self.assertIn("training/players/contracts.py", files)
        self.assertIn("training/common/tracking.py", files)
        self.assertFalse(any(path.startswith("training/ball/") for path in files))
        artifacts = {
            item.path for item in self.client.list_artifacts(run_id, "provenance")
        }
        self.assertIn("provenance/environment.json", artifacts)
        self.assertNotIn("provenance/references.json", artifacts)
        self.assertEqual(
            json.loads((output / "summary.json").read_text())["status"], "FINISHED"
        )

    def test_failed_and_interrupted_runs_are_terminated(self):
        from training.players.learning.tracking import tracked_run

        for error, status in ((RuntimeError, "FAILED"), (KeyboardInterrupt, "KILLED")):
            with self.subTest(status=status):
                with self.assertRaises(error):
                    with tracked_run(self.config, {}) as (_, run_id, output):
                        raise error("intentional test interruption")
                self.assertEqual(self.client.get_run(run_id).info.status, status)
                self.assertEqual(
                    json.loads((output / "summary.json").read_text())["status"], status
                )

    def test_ball_wrapper_keeps_reference_artifact_and_shared_provenance(self):
        from training.ball.learning.tracking import tracked_run

        with tracked_run(self.config, {}) as (_, run_id, output):
            pass
        files = json.loads((output / "environment.json").read_text())["training_sha256"]
        self.assertIn("training/ball/learning/torch_loop.py", files)
        self.assertIn("training/common/tracking.py", files)
        artifacts = {
            item.path for item in self.client.list_artifacts(run_id, "provenance")
        }
        self.assertIn("provenance/references.json", artifacts)
        self.assertEqual(self.client.get_run(run_id).info.status, "FINISHED")

    def test_active_framework_run_is_rejected_without_duplicate(self):
        from training.players.learning.tracking import tracked_run

        # Test the preflight guard without mutating MLflow's global URI or DB.
        with patch.object(self.mlflow, "active_run", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with tracked_run(self.config, {}):
                    self.fail("must not enter the body")
        self.assertEqual(self.client.search_runs([self.experiment_id]), [])


if __name__ == "__main__":
    unittest.main()
