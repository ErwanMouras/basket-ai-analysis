"""Load local pretrained or fine-tuned detectors without a dataset or training.

Input to predict is a uint8 BGR image. Outputs are source-pixel XYXY boxes,
class_id=0 (player), and confidence. The source class must be chosen explicitly.
"""

import argparse
import json
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from training.common.files import write_json
from training.common.provenance import file_hash
from training.players.learning.config import VARIANTS
from training.players.learning.runtime import check_versions


@contextmanager
def rfdetr_weights(path, *, variant=None):
    """Convert our full checkpoints to safe weight-only inputs, using their saved EMA."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if variant is not None:
        import rfdetr.config as rf_config

        config_class = getattr(
            rf_config, "RFDETR" + variant.split("_")[1].title() + "Config"
        )
        architecture = payload.get("model_config", payload.get("args", {}))
        if not isinstance(architecture, dict):
            architecture = vars(architecture)
        for key in ("encoder", "dec_layers", "hidden_dim"):
            if architecture.get(key) != config_class.model_fields[key].default:
                raise ValueError(
                    f"RF-DETR checkpoint architecture does not match {variant}: {key}"
                )
    if "players_training" not in payload:
        yield str(path)
        return
    if payload["players_training"]["family"] != "rfdetr":
        raise ValueError("Expected an RF-DETR checkpoint")
    weights = {
        k.removeprefix("model."): v
        for k, v in payload["state_dict"].items()
        if k.startswith("model.")
    }
    if payload["players_training"]["contract"]["rfdetr"]["use_ema"]:
        ema = next(
            (
                s["average_model_state_dict"]
                for s in payload["callbacks"].values()
                if isinstance(s, dict) and "average_model_state_dict" in s
            ),
            None,
        )
        if ema is None:
            raise ValueError("RF-DETR checkpoint is missing its EMA state")
        weights = {
            k.removeprefix("module.model."): v
            for k, v in ema.items()
            if k.startswith("module.model.")
        }
        if not weights:
            raise ValueError("Unsupported RF-DETR EMA layout")
    with tempfile.TemporaryDirectory(prefix="players-weights-") as temp:
        converted = Path(temp) / "weights.pth"
        torch.save(
            {
                "model": weights,
                "args": payload["model_config"],
                "model_config": payload["model_config"],
                "model_name": payload["model_config"]["model_name"],
                "class_names": ["player"],
            },
            converted,
        )
        yield str(converted)


def check_yolo_variant(model, variant):
    architecture = model.yaml
    name = str(architecture.get("yaml_file", ""))
    scale = architecture.get("scale", "")
    if (
        "yolo26" not in name
        or (scale and scale != variant[-1])
        or not getattr(model, "end2end", False)
    ):
        raise ValueError(f"Checkpoint architecture does not match {variant}")


class Detector:
    def __init__(
        self, family, variant, weights, *, device="cpu", resolution=None, source_class
    ):
        if family not in VARIANTS or variant not in VARIANTS[family]:
            raise ValueError("Unsupported detection variant")
        if type(source_class) is not int or source_class < 0:
            raise ValueError("Specify a nonnegative source class ID")
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:\d+", device):
            raise ValueError("Use one cpu or cuda:N device")
        if resolution is not None and (
            type(resolution) is not int or resolution < 64 or resolution % 32
        ):
            raise ValueError("resolution must be >=64 and divisible by 32")
        path = Path(weights).expanduser().resolve(strict=True)
        check_versions(family)
        self.family, self.source_class, self.device = family, source_class, device
        self.resolution = resolution or (
            640
            if family == "yolo"
            else {"nano": 384, "small": 512, "medium": 576, "large": 704}[
                variant.split("_")[1]
            ]
        )
        if family == "yolo":
            from ultralytics import YOLO

            self.model = YOLO(str(path), task="detect")
            check_yolo_variant(self.model.model, variant)
            if source_class not in self.model.names:
                raise ValueError("Selected class does not exist in the YOLO checkpoint")
        else:
            import rfdetr

            constructor = getattr(rfdetr, "RFDETR" + variant.split("_")[1].title())
            with rfdetr_weights(path, variant=variant) as prepared:
                self.model = constructor(
                    pretrain_weights=prepared, device=device, resolution=self.resolution
                )
            if source_class > self.model.model_config.num_classes:
                raise ValueError(
                    "Selected class does not exist in the RF-DETR checkpoint"
                )
        self.provenance = {
            "family": family,
            "variant": variant,
            "checkpoint_sha256": file_hash(path),
            "device": device,
            "resolution": self.resolution,
            "source_class": source_class,
        }

    def predict(self, image, *, confidence=0.25, max_detections=None, square=False):
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise ValueError("Expected a uint8 BGR image")
        if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if max_detections is not None and (
            type(max_detections) is not int or max_detections < 1
        ):
            raise ValueError("max_detections must be a positive integer")
        if self.family == "yolo":
            import torch

            result = self.model.predict(
                image,
                conf=confidence,
                imgsz=self.resolution,
                device=torch.device(self.device),
                classes=[self.source_class],
                verbose=False,
                **({"max_det": max_detections, "half": False, "rect": not square}
                   if max_detections is not None else {}),
            )[0]
            boxes, scores = (
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            )
        else:
            result = self.model.predict(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=confidence
            )
            mask = result.class_id == self.source_class
            boxes, scores = result.xyxy[mask], result.confidence[mask]
        height, width = image.shape[:2]
        detections = []
        for box, score in zip(boxes, scores):
            box = np.clip(box, [0, 0, 0, 0], [width, height, width, height]).tolist()
            if not all(np.isfinite(box)) or not np.isfinite(score):
                raise ValueError("Non-finite model prediction")
            if box[2] > box[0] and box[3] > box[1]:
                detections.append(
                    {"class_id": 0, "bbox": box, "confidence": float(score)}
                )
        if max_detections is not None:
            detections = sorted(detections, key=lambda d: -d["confidence"])[:max_detections]
        return detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=list(VARIANTS), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source-class", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--image", type=Path, help="Optional local image for inference")
    parser.add_argument("--output", type=Path, help="Optional report path")
    args = parser.parse_args()
    model = Detector(
        args.family,
        args.variant,
        args.weights,
        device=args.device,
        resolution=args.resolution,
        source_class=args.source_class,
    )
    result = {"model": model.provenance}
    if args.image:
        result.update(
            image_sha256=file_hash(args.image),
            predictions=model.predict(cv2.imread(str(args.image))),
        )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
