"""Opt-in numerical parity, geometry and resume tests for PyTorch training."""

import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from training.ball.learning.config import load_config, preflight
from training.ball.tests.test_training import make_training_export

TORCH_TESTS = os.environ.get("BALL_TORCH_TESTS") == "1"


@unittest.skipUnless(TORCH_TESTS, "Set BALL_TORCH_TESTS=1 in the PyTorch environment")
class TorchTrainingTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profile = make_training_export(self.root)

    def config(self, model="tracknet_v5_totnet"):
        path = self.root / "config.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "profile_file": str(self.profile),
                    "dataset": str(self.root / "exports/tracknet-totnet"),
                    "output": str(self.root / "runs"),
                    "mlflow_uri": f"sqlite:///{self.root}/mlflow.db",
                    "experiment": "test",
                    "epochs": 2,
                    "batch_size": 2,
                    "workers": 0,
                    "device": "cpu",
                    "precision": "fp32",
                    "max_train_batches": 1,
                    "max_val_batches": 1,
                    "cpu_threads": 2,
                    "pin_memory": False,
                }
            )
        )
        from mlflow import MlflowClient

        config = load_config(model, path)
        client = MlflowClient(tracking_uri=config["mlflow_uri"])
        client.create_experiment(
            "test", artifact_location=(self.root / "mlflow-artifacts").as_uri()
        )
        return config

    def test_weighted_loss_matches_sdk_at_weight_one_including_gradients(self):
        config = self.config()
        preflight(config)
        from losses_factory.losses.tracknetv2_loss import TrackNetV2Loss

        from training.ball.learning.totnet import VisibilityWeightedTrackNetLoss

        torch = self.torch
        predictions = (
            torch.linspace(0.001, 0.999, 48).reshape(2, 3, 2, 4).requires_grad_()
        )
        targets = torch.zeros_like(predictions)
        targets[:, 1, 0, 0] = 255
        reference = TrackNetV2Loss()(predictions, targets)
        actual = VisibilityWeightedTrackNetLoss(1)(predictions, targets)
        torch.testing.assert_close(reference, actual)
        torch.testing.assert_close(
            torch.autograd.grad(reference, predictions)[0],
            torch.autograd.grad(actual, predictions)[0],
        )
        weighted = VisibilityWeightedTrackNetLoss(2, reduction="none")(
            predictions, targets
        )
        base = VisibilityWeightedTrackNetLoss(1, reduction="none")(predictions, targets)
        torch.testing.assert_close(weighted[:, 1], 2 * base[:, 1])
        torch.testing.assert_close(weighted[:, 0], base[:, 0])

    def test_sdk_reader_matches_reference_images_and_targets(self):
        config = self.config("tracknet_v5")
        preflight(config)
        from datasets_factory.datasets.tennis_dataset import TennisDataset

        from training.ball.export import sdk_config
        from training.ball.learning.torch_data import SDKDataset

        original_file = sdk_config.__file__
        try:
            sdk_config.__file__ = str(
                self.root / "exports/tracknet-totnet/sdk_dataset.py"
            )
            reference_config = sdk_config.build_data(32, 32, workers=1)["data"]["train"]
        finally:
            sdk_config.__file__ = original_file
        reference_config.pop("type")
        reference = TennisDataset(**reference_config)[0]
        actual = SDKDataset(config, "train")[0]
        self.torch.testing.assert_close(reference["image"], actual["image"])
        self.torch.testing.assert_close(reference["target"], actual["target"])
        from training.ball.evaluation.inference import SDKFrames

        inference = SDKFrames(Path(config["dataset"]), config["geometry"])
        validation = SDKDataset(config, "val")
        self.torch.testing.assert_close(inference[0], validation[0]["image"])

    def test_occlusion_keeps_target_and_flip_crop_keep_coordinate_geometry(self):
        from training.ball.learning.totnet import (
            OcclusionAugment,
            TripletHorizontalFlip,
            TripletRandomZoomCrop,
        )

        torch = self.torch
        image = np.zeros((32, 32, 3), np.uint8)
        image[15:18, 15:18] = 255
        result = {"path": image, "coords": [(16, 16)] * 3, "visibility": [1] * 3}
        hidden = OcclusionAugment(prob=1, patch_scale=3, fill="black", seed=0)(result)
        self.assertFalse(hidden["path"].any())
        self.assertEqual(hidden["visibility"], [1] * 3)
        target = torch.zeros(3, 32, 32)
        target[:, 16, 16] = 255
        sample = {
            "image": np.zeros((32, 32, 9), np.uint8),
            "target": target,
            "coords": [(16, 16)] * 3,
            "visibility": [1] * 3,
        }
        flipped = TripletHorizontalFlip(1, coord_width=32, seed=0)(sample)
        self.assertEqual(flipped["coords"][1], (15, 16))
        self.assertEqual(float(flipped["target"][1, 16, 15]), 255)
        cropped = TripletRandomZoomCrop(1, (0.7, 0.7), 32, 32, seed=8)(flipped)
        x, y = cropped["coords"][1]
        yy, xx = np.where(cropped["target"][1].numpy() > 0)
        self.assertLess(abs(float(xx.mean()) - x), 1.5)
        self.assertLess(abs(float(yy.mean()) - y), 1.5)
        self.assertEqual(cropped["visibility"][1], 1)

    def test_excluded_occluded_position_stays_empty_in_training_reader(self):
        from training.ball.export.config import ExportConfig
        from training.ball.export.dataset import export_dataset
        from training.ball.learning.torch_data import SDKDataset

        config = self.config("tracknet_v5")
        export_dataset(
            self.root / "source",
            self.root / "exports",
            ExportConfig(occluded_position="exclude", tracknet_layouts=("sdk",)),
            ("tracknet-totnet",),
            progress=lambda message: None,
        )
        dataset = SDKDataset(config, "train")
        index = next(
            index
            for index, (_, frames) in enumerate(dataset.samples)
            if frames[1]["frame_index"] == 4
        )
        target = dataset[index]["target"]
        self.assertEqual(float(target[1].sum()), 0)
        self.assertGreater(float(target[0].sum()), 0)
        self.assertGreater(float(target[2].sum()), 0)

    def test_crop_removes_partial_heatmap_when_context_center_leaves_window(self):
        from training.ball.learning.totnet import TripletRandomZoomCrop

        target = self.torch.zeros(3, 32, 32)
        target[0, 12:21, :6] = 255
        target[1, 12:21, 12:21] = 255
        target[2, 12:21, 26:] = 255
        sample = {
            "image": np.zeros((32, 32, 9), np.uint8),
            "target": target,
            "coords": [(0, 16), (16, 16), (31, 16)],
            "visibility": [1, 1, 1],
        }
        result = TripletRandomZoomCrop(1, (0.7, 0.7), 32, 32, seed=8)(sample)
        self.assertIn(0, result["visibility"])
        self.assertEqual(result["visibility"][1], 1)
        for index, visible in enumerate(result["visibility"]):
            if not visible:
                self.assertEqual(float(result["target"][index].sum()), 0)

    def test_resume_restores_exact_cpu_training_and_records_failure(self):
        from mlflow import MlflowClient

        from training.ball.learning import torch_loop
        from training.ball.train_tracknet_v5_totnet import train

        config = self.config()
        # One uninterrupted run supplies the expected second-epoch model/optimizer state.
        complete = train(deepcopy(config))
        original_log = torch_loop.log_metrics

        def interrupt(client, run_id, metrics, epoch):
            original_log(client, run_id, metrics, epoch)
            raise RuntimeError("Intentional smoke interruption after checkpoint")

        with (
            patch.object(torch_loop, "log_metrics", side_effect=interrupt),
            self.assertRaisesRegex(RuntimeError, "Intentional"),
        ):
            train(deepcopy(config))
        client = MlflowClient(tracking_uri=config["mlflow_uri"])
        runs = client.search_runs([client.get_experiment_by_name("test").experiment_id])
        failed = next(run for run in runs if run.info.status == "FAILED")
        checkpoint = Path(config["output"]) / failed.info.run_id / "last.pt"
        resumed_config = deepcopy(config)
        resumed_config["resume"] = str(checkpoint)
        resumed = train(resumed_config)
        expected = self.torch.load(complete / "last.pt", weights_only=False)
        actual = self.torch.load(resumed / "last.pt", weights_only=False)
        self.assertEqual(actual["epoch"], 1)
        self.torch.testing.assert_close(
            expected["model"], actual["model"], rtol=0, atol=0
        )
        self.torch.testing.assert_close(
            expected["optimizer"], actual["optimizer"], rtol=0, atol=0
        )
        self.assertEqual(len(client.get_metric_history(resumed.name, "val/loss")), 1)
