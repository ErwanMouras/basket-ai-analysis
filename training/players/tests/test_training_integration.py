"""Opt-in real training, interrupted resume, weight loading, and local MLflow.

PLAYERS_TRAINING_TESTS=yolo|rfdetr, PLAYERS_TEST_WEIGHTS=/local/checkpoint
Optional: PLAYERS_TEST_DEVICE=cuda:0, PLAYERS_TEST_PRECISION=amp,
PLAYERS_TEST_OUTPUT=/persistent/local/path (otherwise temporary).
"""

import copy
import gc
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np

from training.common.provenance import ROOT
from training.players.export.publication import verify_bundle
from training.players.learning.config import load_config
from training.players.tests.training_fixtures import make_dataset

FAMILY = os.environ.get("PLAYERS_TRAINING_TESTS")


@unittest.skipUnless(
    FAMILY in ("yolo", "rfdetr"),
    "Set PLAYERS_TRAINING_TESTS and PLAYERS_TEST_WEIGHTS for a short real integration",
)
class TrainingIntegration(unittest.TestCase):
    def test_training_resume_matches_continuous_and_pretrained_loading(self):
        import torch
        from mlflow import MlflowClient

        from training.players.learning.run import train
        from training.players.learning.runtime import TechnicalInterruption
        from training.players.models import Detector

        if os.environ.get("PLAYERS_TEST_OUTPUT"):
            base = Path(os.environ["PLAYERS_TEST_OUTPUT"]).resolve() / (
                FAMILY + "-" + uuid.uuid4().hex[:8]
            )
            base.mkdir(parents=True)
        else:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            base = Path(temporary.name)
        print(f"Technical integration artifacts: {base}", flush=True)
        exports = make_dataset(base)
        before = verify_bundle(exports)
        config = load_config(
            FAMILY, ROOT / f"training/players/configs/train_{FAMILY}_smoke.yaml"
        )
        config.update(
            dataset=str(exports / ("yolo" if FAMILY == "yolo" else "coco")),
            output=str(base / "runs"),
            weights=str(Path(os.environ["PLAYERS_TEST_WEIGHTS"]).resolve()),
            device=os.environ.get("PLAYERS_TEST_DEVICE", "cpu"),
            precision=os.environ.get("PLAYERS_TEST_PRECISION", "fp32"),
            mlflow_uri="sqlite:///" + str(base / "mlflow.db"),
        )
        # No dataset and no optimizer are needed to load the original pretrained model.
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        detector = Detector(
            FAMILY,
            config["variant"],
            config["weights"],
            device="cpu",
            source_class=int(
                os.environ.get(
                    "PLAYERS_TEST_SOURCE_CLASS", "0" if FAMILY == "yolo" else "1"
                )
            ),
            resolution=config["resolution"],
        )
        detections = detector.predict(np.zeros((96, 128, 3), dtype=np.uint8))
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), visible_devices)
        self.assertIsInstance(detections, list)
        del detector
        gc.collect()
        continuous = train(config)
        del detections
        gc.collect()
        torch.cuda.empty_cache()
        with self.assertRaises((TechnicalInterruption, SystemExit)):
            train(config, stop_after_epoch=1)
        outputs = list((base / "runs").iterdir())
        interrupted = next(p for p in outputs if p != continuous)
        self.assertEqual(
            json.loads((interrupted / "summary.json").read_text())["status"], "KILLED"
        )
        relative = (
            "yolo/weights/last.pt" if FAMILY == "yolo" else "checkpoints/last.ckpt"
        )
        resume = {**config, "mode": "resume", "resume": str(interrupted / relative)}
        gc.collect()
        torch.cuda.empty_cache()
        resumed = train(resume)
        a = torch.load(continuous / relative, map_location="cpu", weights_only=False)
        b = torch.load(resumed / relative, map_location="cpu", weights_only=False)
        self.assertEqual(a["epoch"], 1)
        self.assertEqual(b["epoch"], 1)
        key = "optimizer" if FAMILY == "yolo" else "optimizer_states"
        self.assertTrue(b[key])
        optimizer = b["optimizer"] if FAMILY == "yolo" else b["optimizer_states"][0]
        self.assertTrue(
            optimizer["state"], "Smoke training must perform an optimizer update"
        )
        wa = a["players_training"]["model"] if FAMILY == "yolo" else a["state_dict"]
        wb = b["players_training"]["model"] if FAMILY == "yolo" else b["state_dict"]
        self.assertEqual(wa.keys(), wb.keys())
        max_difference = max(
            float((wa[k].float() - wb[k].float()).abs().max())
            for k in wa
            if wa[k].numel()
        )
        print(
            f"Continuous/resumed maximum weight difference: {max_difference}",
            flush=True,
        )
        # CUDA grid_sample is nondeterministic; exact equality is only asserted on CPU.
        tolerance = (
            0
            if config["device"] == "cpu"
            else (0.01 if config["precision"] == "amp" else 2e-4)
        )
        for name in wa:
            torch.testing.assert_close(
                wa[name], wb[name], rtol=tolerance, atol=tolerance, msg=name
            )

        def compare_state(left, right):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for name in left:
                    compare_state(left[name], right[name])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for lvalue, rvalue in zip(left, right):
                    compare_state(lvalue, rvalue)
            else:
                self.assertEqual(left, right)

        compare_state(a[key], b[key])
        if FAMILY == "yolo":
            compare_state(a["scaler"], b["scaler"])
            compare_state(
                a["players_training"]["scheduler"], b["players_training"]["scheduler"]
            )
            compare_state(a["ema"].state_dict(), b["ema"].state_dict())
        else:
            self.assertEqual(a["global_step"], b["global_step"])
            compare_state(a["lr_schedulers"], b["lr_schedulers"])
            for name, state in a["callbacks"].items():
                if "average_model_state_dict" in state:
                    compare_state(state, b["callbacks"][name])
        client = MlflowClient(tracking_uri=config["mlflow_uri"])
        experiment = client.get_experiment_by_name("players-training")
        runs = client.search_runs([experiment.experiment_id])
        self.assertEqual(len(runs), 3)
        self.assertEqual(
            sorted(r.info.status for r in runs), ["FINISHED", "FINISHED", "KILLED"]
        )
        current = client.get_run(resumed.name)
        self.assertEqual(current.data.tags["resumed_from_run"], interrupted.name)
        self.assertTrue(current.data.metrics)
        self.assertEqual(
            current.data.params["initial_weights_sha256"],
            a["players_training"]["contract"]["initial_weights_sha256"],
        )
        self.assertTrue(client.list_artifacts(resumed.name, "checkpoints"))
        self.assertEqual(before, verify_bundle(exports))
        gc.collect()
        trained = Detector(
            FAMILY,
            config["variant"],
            resumed / relative,
            source_class=0,
            resolution=config["resolution"],
        )
        self.assertIsInstance(
            trained.predict(np.zeros((96, 128, 3), dtype=np.uint8)), list
        )
        # A completed checkpoint must be rejected, and its weights remain usable for a new fine-tune.
        with self.assertRaisesRegex(ValueError, "epoch budget"):
            train({**config, "mode": "resume", "resume": str(resumed / relative)})
        from training.players.learning.config import validate_config

        fresh = copy.deepcopy(config)
        fresh.update(
            mode="finetune", weights=str(resumed / relative), resume=None, epochs=1
        )
        validate_config(fresh)
        del a, b, wa, wb, trained
        gc.collect()
        torch.cuda.empty_cache()
        fine_tuned = train(fresh)
        fine = torch.load(fine_tuned / relative, map_location="cpu", weights_only=False)
        self.assertEqual(fine["epoch"], 0)
        fine_run = client.get_run(fine_tuned.name)
        self.assertEqual(fine_run.data.tags["initialization"], "finetune")
        self.assertNotIn("resumed_from_run", fine_run.data.tags)
        self.assertEqual(len(client.search_runs([experiment.experiment_id])), 4)
        (base / "validation.json").write_text(
            json.dumps(
                {
                    "family": FAMILY,
                    "device": config["device"],
                    "precision": config["precision"],
                    "max_weight_difference": max_difference,
                    "continuous": continuous.name,
                    "interrupted": interrupted.name,
                    "resumed": resumed.name,
                    "fine_tuned": fine_tuned.name,
                    "dataset_id": before["yolo"]["dataset_id"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
