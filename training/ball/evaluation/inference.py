"""SDK V5 inference without heatmap targets or training augmentations."""

from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from training.ball.learning.data import checked_path, sdk_samples
from training.ball.learning.references import activate_reference
from training.ball.learning.torch_data import image_tensor, sdk_resize_triplet
from training.ball.learning.torch_loop import loader, seed_everything


class SDKFrames(Dataset):
    def __init__(self, root, geometry):
        self.root = Path(root)
        self.geometry = geometry
        # Dense inference covers all valid windows, independent of training stride.
        self.samples = sorted(
            sdk_samples(self.root, "val", 1),
            key=lambda sample: (sample[1][0]["clip_id"], sample[1][0]["frame_index"]),
        )
        for _, frames in self.samples:
            for frame in frames:
                checked_path(self.root, frame["image"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        images = []
        for frame in self.samples[index][1]:
            image = cv2.imread(str(self.root / frame["image"]))
            if image is None or image.shape[:2] != (frame["height"], frame["width"]):
                raise ValueError(f"Invalid SDK image: {frame['image']}")
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image_tensor(
            sdk_resize_triplet(
                images, self.geometry["input_width"], self.geometry["input_height"]
            )
        )


class V5Predictor:
    """Reuse the architecture and geometry saved with a local training checkpoint."""

    def __init__(self, config, manifest):
        checkpoint = torch.load(
            config["checkpoint"], map_location="cpu", weights_only=False
        )
        trained = checkpoint["config"]
        if trained["model"] not in ("tracknet_v5", "tracknet_v5_totnet"):
            raise ValueError(
                "This inference adapter supports V5 and V5 + TOTNet checkpoints"
            )
        for key in ("dataset_id", "export_id"):
            if trained[key] != manifest[key]:
                raise ValueError(f"Checkpoint {key} differs from evaluation export")
        reference_config = {**trained, "reference_root": config["reference_root"]}
        activate_reference(reference_config)
        if reference_config["reference"] != trained["reference"]:
            raise ValueError("Checkpoint reference differs from the installed model")
        from models_factory import build_model

        self.device = seed_everything(config)
        self.geometry = trained["geometry"]
        self.model = build_model(trained["architecture"])
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device).eval()
        self.config = config
        self.provenance = {
            "model": trained["model"],
            "geometry": self.geometry,
            "architecture": trained["architecture"],
            "reference": trained["reference"],
            "training_run_id": checkpoint["run_id"],
            "checkpoint_epoch": checkpoint["epoch"],
        }

    def predict(self, images):
        """Return B x 3 x H x W heatmaps for normalized SDK triplets."""
        precision = self.config["precision"]
        autocast = (
            nullcontext()
            if precision == "fp32"
            else torch.autocast(
                self.device.type,
                dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
            )
        )
        with torch.inference_mode(), autocast:
            output = self.model(
                images.to(self.device, non_blocking=self.config["pin_memory"])
            )
        result = output.float().cpu().numpy()
        expected = (
            len(images),
            3,
            self.geometry["input_height"],
            self.geometry["input_width"],
        )
        if result.shape != expected or not np.isfinite(result).all():
            raise ValueError("Invalid model heatmap shape or non-finite predictions")
        if result.min() < 0 or result.max() > 1:
            raise ValueError("Expected probability heatmaps in [0, 1]")
        return result

    def windows(self, dataset):
        offset = 0
        for images in loader(dataset, self.config, epoch=0, training=False):
            predictions = self.predict(images)
            for heatmaps in predictions:
                yield dataset.samples[offset][1], heatmaps
                offset += 1
                if offset % 250 == 0 or offset == len(dataset):
                    print(f"Inferred {offset}/{len(dataset)} windows", flush=True)
