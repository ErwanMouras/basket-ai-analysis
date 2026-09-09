"""Opt-in real training/validation/checkpoint/MLflow tests, one process per model."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from training.ball.tests.test_training import make_training_export

MODELS = set(os.environ.get("BALL_SMOKE_MODELS", "").split(","))


class TrainingSmoke(unittest.TestCase):
    def smoke(self, model, overrides=None):
        from mlflow import MlflowClient

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = make_training_export(root)
            uri = f"sqlite:///{root}/mlflow.db"
            recipe = {
                "dataset": str(
                    root
                    / "exports"
                    / ("yolo" if model == "yolo" else "tracknet-totnet")
                ),
                "output": str(root / "runs"),
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
                "precision": "fp32",
                "workers": 0,
                "cpu_threads": 2,
                "mlflow_uri": uri,
                "experiment": "smoke",
                "pin_memory": False,
            }
            if model == "yolo":
                recipe.update(
                    weights="yolo26n.yaml",
                    input_width=64,
                    input_height=64,
                    yolo={"pretrained": False, "mosaic": 0.0, "close_mosaic": 0},
                )
            else:
                recipe.update(
                    profile_file=str(profile), max_train_batches=1, max_val_batches=1
                )
            recipe.update(overrides or {})
            client = MlflowClient(tracking_uri=uri)
            client.create_experiment(
                "smoke", artifact_location=(root / "mlflow-artifacts").as_uri()
            )
            path = root / "recipe.yaml"
            path.write_text(yaml.safe_dump(recipe))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    f"training.ball.train_{model}",
                    "--config",
                    str(path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
                env={**os.environ, "MLFLOW_TRACKING_URI": uri},
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            client = MlflowClient(tracking_uri=uri)
            experiment = client.get_experiment_by_name("smoke")
            runs = client.search_runs([experiment.experiment_id])
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(run.info.status, "FINISHED")
            self.assertFalse(run.data.tags.get("mlflow.parentRunId"))
            self.assertTrue(any(key.startswith("train/") for key in run.data.metrics))
            self.assertTrue(any(key.startswith("val/") for key in run.data.metrics))
            self.assertIn("duration_seconds", run.data.metrics)
            self.assertIn("dataset_id", run.data.params)
            artifacts = client.list_artifacts(run.info.run_id, "checkpoints")
            expected = (
                {"best", "last"} if model == "tracknet_v4" else {"best.pt", "last.pt"}
            )
            self.assertEqual(
                {Path(artifact.path).name for artifact in artifacts}, expected
            )
            provenance = {
                Path(artifact.path).name
                for artifact in client.list_artifacts(run.info.run_id, "provenance")
            }
            self.assertTrue(
                {"manifest.json", "config.resolved.yaml", "environment.json"}
                <= provenance
            )
            # Training must never create a test reader or test metrics.
            self.assertFalse(any(key.startswith("test/") for key in run.data.metrics))

    @unittest.skipUnless(
        "yolo" in MODELS, "Set BALL_SMOKE_MODELS=yolo to execute real YOLO training"
    )
    def test_yolo(self):
        self.smoke("yolo")

    @unittest.skipUnless("tracknet_v3" in MODELS, "Opt-in V3 integration")
    def test_v3(self):
        self.smoke("tracknet_v3")

    @unittest.skipUnless("tracknet_v4" in MODELS, "Opt-in TensorFlow V4 integration")
    def test_v4(self):
        self.smoke("tracknet_v4")

    @unittest.skipUnless(
        "tracknet_v4" in MODELS, "Opt-in TensorFlow V4 TypeB integration"
    )
    def test_v4_type_b(self):
        self.smoke("tracknet_v4", {"fusion": "TypeB"})

    @unittest.skipUnless("tracknet_v5" in MODELS, "Opt-in V5 integration")
    def test_v5(self):
        self.smoke("tracknet_v5")

    @unittest.skipUnless(
        "tracknet_v5_totnet" in MODELS, "Opt-in V5 + TOTNet integration"
    )
    def test_v5_totnet(self):
        self.smoke("tracknet_v5_totnet")
