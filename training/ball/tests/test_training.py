"""Critical training contracts without requiring an ML framework."""

import json
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from training.ball.annotator.model import Annotation
from training.ball.export.config import ExportConfig
from training.ball.export.dataset import export_dataset
from training.ball.learning.config import load_config, resume_contract, validate_config
from training.ball.learning.data import V4Arrays, consecutive, sdk_samples, v3_samples
from training.ball.learning.metrics import metric_values, update_counts
from training.ball.tests.test_export import create_clip


def make_training_export(root):
    source = root / "source"
    annotations = {index: Annotation(25, 20, 4, index == 4) for index in range(10)}
    create_clip(source, "train/train_game/video.avi", annotations=annotations)
    create_clip(source, "val/val_game/video.avi", seed=5, annotations=annotations)
    profiles = yaml.safe_load(
        Path("training/ball/configs/ball_training.yaml").read_text()
    )
    for profile in profiles["profiles"].values():
        profile.update(input_width=32, input_height=32)
    profile_path = root / "profiles.yaml"
    profile_path.write_text(yaml.safe_dump(profiles))
    export_dataset(
        source,
        root / "exports",
        ExportConfig(tracknet_layouts=("sdk", "v3", "v4")),
        ("yolo", "tracknet-totnet"),
        training_config=profile_path,
        progress=lambda message: None,
    )
    return profile_path


class TrainingContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profile = make_training_export(self.root)
        self.dataset = self.root / "exports/tracknet-totnet"

    def config(self, model):
        config = load_config(model)
        config["dataset"] = str(self.dataset)
        config["geometry"] = yaml.safe_load(self.profile.read_text())["profiles"][model]
        return config

    def test_model_profile_precision_and_unknown_keys_are_rejected(self):
        config = self.config("tracknet_v5")
        config["geometry"]["version"] = 4
        with self.assertRaisesRegex(ValueError, "version"):
            validate_config(config)
        path = self.root / "bad.yaml"
        path.write_text("learnng_rate: 0.01\n")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            load_config("tracknet_v5", path)
        path.write_text("input_width: -32\ninput_height: -32\n")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            load_config("yolo", path)
        config = self.config("tracknet_v4")
        config["precision"] = "bf16"
        with self.assertRaisesRegex(ValueError, "fp32"):
            validate_config(config)

    def test_sdk_rejects_cross_split_rows_and_frame_gaps(self):
        rows = sdk_samples(self.dataset, "train", 1)
        frames = deepcopy(rows[0][1])
        frames[1]["frame_index"] += 1
        with self.assertRaisesRegex(ValueError, "gap"):
            consecutive(frames)
        frames = deepcopy(rows[0][1])
        frames[1]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "split"):
            consecutive(frames)
        with self.assertRaisesRegex(ValueError, "only train and val"):
            sdk_samples(self.dataset, "test", 1)

    def test_v3_rejects_known_position_rounded_to_origin(self):
        profile = self.config("tracknet_v3")["geometry"]
        self.assertEqual(len(v3_samples(self.dataset, "train", profile)), 3)
        path = self.dataset / "frames.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        for frame in lines:
            if frame["split"] == "train":
                frame["position"] = [0.1, 0.1]
        path.write_text("".join(json.dumps(frame) + "\n" for frame in lines))
        import csv

        for path in (self.dataset / "v3/train").rglob("*_ball.csv"):
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row.update(X="0.1", Y="0.1")
            with path.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "sentinel"):
            v3_samples(self.dataset, "train", profile)

    def test_v4_is_bounded_and_rejects_preparation_changes(self):
        config = self.config("tracknet_v4")
        batches = V4Arrays(config, "train")
        for images, targets in batches.batches(0):
            self.assertLessEqual(len(images), config["batch_size"])
            self.assertEqual(images.shape[1:], (9, 32, 32))
            self.assertEqual(targets.shape[1:], (3, 32, 32))
        config["geometry"]["sequence_stride"] = 2
        with self.assertRaisesRegex(ValueError, "regenerate"):
            V4Arrays(config, "train")

    def test_resume_contract_preserves_data_and_optimizer_identity(self):
        config = self.config("tracknet_v5")
        moved = deepcopy(config)
        moved.update(output="/tmp/moved", resume="/tmp/last.pt", workers=0)
        self.assertEqual(resume_contract(config), resume_contract(moved))
        moved["learning_rate"] *= 2
        self.assertNotEqual(resume_contract(config), resume_contract(moved))

    def test_displaced_prediction_counts_as_false_positive_and_miss(self):
        predicted, targets = np.zeros((1, 3, 32, 32)), np.zeros((1, 3, 32, 32))
        targets[0, 0, 10, 10] = predicted[0, 0, 10, 10] = 1
        targets[0, 1, 10, 10] = predicted[0, 1, 20, 20] = 1
        counts = Counter()
        update_counts(counts, predicted, targets, 0.5, 4)
        values = metric_values(counts)
        self.assertEqual(
            (values["tp"], values["tn"], values["fp"], values["fn"]), (1, 1, 1, 1)
        )
        self.assertEqual(values["f1"], 0.5)
