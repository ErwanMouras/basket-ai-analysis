"""Opt-in process restart tests for the framework-specific checkpoint adapters."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from training.ball.tests.test_training import make_training_export

MODEL = os.environ.get("BALL_RESUME_MODEL")
INTERRUPT_SCRIPT = """
import importlib
import sys
from unittest.mock import patch
from training.ball.learning.config import load_config
model, path = sys.argv[1:]
entry = importlib.import_module('training.ball.train_' + model)
logger_module = entry if model == 'tracknet_v4' else importlib.import_module('training.ball.learning.yolo_trainer')
original = logger_module.log_metrics

def interrupt(*args):
    original(*args)
    raise RuntimeError('Intentional checkpoint interruption')

with patch.object(logger_module, 'log_metrics', side_effect=interrupt):
    entry.train(load_config(model, path))
"""


@unittest.skipUnless(
    MODEL in ("yolo", "tracknet_v4"), "Set BALL_RESUME_MODEL=yolo or tracknet_v4"
)
class FrameworkResume(unittest.TestCase):
    def test_resume_continues_training_and_validation_in_a_new_run(self):
        from mlflow import MlflowClient

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = make_training_export(root)
            uri = f"sqlite:///{root}/mlflow.db"
            config = {
                "dataset": str(
                    root
                    / "exports"
                    / ("yolo" if MODEL == "yolo" else "tracknet-totnet")
                ),
                "output": str(root / "runs"),
                "epochs": 2,
                "batch_size": 2,
                "workers": 0,
                "cpu_threads": 2,
                "device": "cpu",
                "precision": "fp32",
                "mlflow_uri": uri,
                "experiment": "resume",
                "pin_memory": False,
            }
            if MODEL == "yolo":
                config.update(
                    weights="yolo26n.yaml",
                    input_width=64,
                    input_height=64,
                    yolo={"pretrained": False, "mosaic": 0.0, "close_mosaic": 0},
                )
            else:
                config.update(
                    profile_file=str(profile), max_train_batches=1, max_val_batches=1
                )
            client = MlflowClient(tracking_uri=uri)
            client.create_experiment(
                "resume", artifact_location=(root / "mlflow-artifacts").as_uri()
            )
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            env = {**os.environ, "MLFLOW_TRACKING_URI": uri}
            interrupted = subprocess.run(
                [sys.executable, "-c", INTERRUPT_SCRIPT, MODEL, str(path)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertNotEqual(interrupted.returncode, 0)
            self.assertIn("Intentional checkpoint interruption", interrupted.stdout)
            client = MlflowClient(tracking_uri=uri)
            experiment = client.get_experiment_by_name("resume")
            failed = client.search_runs([experiment.experiment_id])[0]
            self.assertEqual(failed.info.status, "FAILED")
            output = root / "runs" / failed.info.run_id
            checkpoint = output / (
                "yolo/weights/last.pt" if MODEL == "yolo" else "last"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    f"training.ball.train_{MODEL}",
                    "--config",
                    str(path),
                    "--resume",
                    str(checkpoint),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            runs = client.search_runs([experiment.experiment_id])
            self.assertEqual(len(runs), 2)
            finished = next(run for run in runs if run.info.status == "FINISHED")
            self.assertEqual(finished.data.tags["resumed_from_run"], failed.info.run_id)
            history = client.get_metric_history(
                finished.info.run_id, "val/box_loss" if MODEL == "yolo" else "val/loss"
            )
            self.assertEqual([metric.step for metric in history], [1])
            output = root / "runs" / finished.info.run_id
            if MODEL == "yolo":
                import torch

                before = torch.load(checkpoint, weights_only=False, map_location="cpu")
                after = torch.load(
                    output / "yolo/weights/last.pt",
                    weights_only=False,
                    map_location="cpu",
                )
                self.assertEqual(after["epoch"], 1)
                self.assertGreater(after["updates"], before["updates"])
                self.assertIsNotNone(after["optimizer"])
            else:
                import tensorflow as tf

                before = tf.train.load_checkpoint(str(checkpoint / "state"))
                after = tf.train.load_checkpoint(str(output / "last/state"))
                key = "optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE"
                self.assertGreater(after.get_tensor(key), before.get_tensor(key))
                state = json.loads((output / "last/training.json").read_text())
                self.assertEqual(state["epoch"], 1)
