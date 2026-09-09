"""PyTorch readers matching the SDK and V3 image/target conventions."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import checked_path, sdk_samples, v3_samples

FRAME_KEYS = ("path_prev", "path", "path_next")


class SDKDataset(Dataset):
    def __init__(self, config, split, augmentation=None):
        self.root = Path(config["dataset"])
        self.profile = config["geometry"]
        self.samples = sdk_samples(self.root, split, self.profile["sequence_stride"])
        self.augmentation = augmentation
        sizes = {
            (frame["width"], frame["height"])
            for _, frames in self.samples
            for frame in frames
        }
        if len(sizes) != 1:
            raise ValueError(
                "The SDK requires one exported image size; re-export with resize"
            )
        self.export_size = next(iter(sizes))
        for row, _ in self.samples:
            for key in FRAME_KEYS:
                checked_path(self.root, row[key])
                checked_path(self.root, row["gt_" + key])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row, frames = self.samples[index]
        w, h = self.profile["input_width"], self.profile["input_height"]
        result = {
            "coords": [
                frame["position"] or (float("nan"), float("nan")) for frame in frames
            ],
            "visibility": [int(frame["has_position"]) for frame in frames],
        }
        targets = []
        for key, frame in zip(FRAME_KEYS, frames):
            image = cv2.imread(str(self.root / row[key]))
            if image is None or image.shape[:2] != (frame["height"], frame["width"]):
                raise ValueError(f"Invalid SDK image: {row[key]}")
            result[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            target = cv2.imread(str(self.root / row["gt_" + key]), cv2.IMREAD_GRAYSCALE)
            if target is None or target.shape != image.shape[:2]:
                raise ValueError("SDK target/image shape mismatch")
            target = cv2.resize(target, (w, h), interpolation=cv2.INTER_NEAREST)
            if bool(target.any()) != frame["has_position"]:
                raise ValueError(
                    "SDK resize lost a known target; increase input or export heatmap support"
                )
            targets.append(target)
        if self.augmentation:
            result = self.augmentation.before_resize(result)
        result["image"] = np.concatenate(
            [cv2.resize(result[key], (w, h)) for key in FRAME_KEYS], axis=2
        )
        result["target"] = torch.from_numpy(np.stack(targets).astype(np.float32))
        if self.augmentation:
            result = self.augmentation.after_resize(result)
        return {
            "image": torch.from_numpy(result["image"].transpose(2, 0, 1).copy()).float()
            / 255,
            "target": result["target"],
        }


class V3Dataset(Dataset):
    def __init__(self, config, split):
        self.profile = config["geometry"]
        self.samples = v3_samples(Path(config["dataset"]), split, self.profile)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        w, h = self.profile["input_width"], self.profile["input_height"]
        yy, xx = np.ogrid[:h, :w]
        images, targets = [], []
        for path, frame in self.samples[index]:
            with Image.open(path) as image:
                # PIL's RGB default is bicubic, as in the reference reader.
                images.append(
                    np.asarray(image.convert("RGB").resize((w, h))).transpose(2, 0, 1)
                )
            target = np.zeros((h, w), np.float32)
            if frame["has_position"]:
                x = int(frame["position"][0] * w / frame["width"])
                y = int(frame["position"][1] * h / frame["height"])
                target = (
                    (xx - x) ** 2 + (yy - y) ** 2 <= self.profile["target_radius"] ** 2
                ).astype(np.float32)
            targets.append(target)
        return {
            "image": torch.from_numpy(np.concatenate(images).copy()).float() / 255,
            "target": torch.from_numpy(np.stack(targets)),
        }
