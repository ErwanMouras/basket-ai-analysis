"""Synthetic ground truth only, kept separate from all project datasets."""

import cv2
import numpy as np

from training.common.files import write_json
from training.players.annotator.media import MediaReader, source_record
from training.players.annotator.model import sidecar_path
from training.players.export.dataset import export_dataset


def make_weights(family, output):
    """Random initialization, never pretrained weights or a network download."""
    import torch
    from training.players.learning.runtime import seed_all

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    seed_all(42)
    if family == "yolo":
        from ultralytics import YOLO
        YOLO("yolo26n.yaml", task="detect").save(output)
    else:
        from rfdetr import RFDETRNano
        model = RFDETRNano(pretrain_weights=None, device="cpu", resolution=128, num_classes=1)
        config = model.model_config.model_dump(mode="json")
        torch.save({"model": model.model.model.state_dict(), "args": config,
                    "model_config": config, "model_name": config["model_name"],
                    "class_names": ["player"]}, output)


def make_dataset(base):
    source = base / "source"
    for split, count in (("train", 4), ("val", 2), ("test", 1)):
        for index in range(count):
            path = source / split / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            offset = {"train": 0, "val": 100, "test": 200}[split]
            image = np.random.default_rng(index + offset).integers(
                0, 100, (96, 128, 3), dtype=np.uint8
            )
            cv2.rectangle(image, (15, 10), (55, 88), (30, 200, 250), -1)
            if not cv2.imwrite(str(path), image):
                raise RuntimeError("Could not write fixture image")
            reader = MediaReader(path)
            try:
                record = source_record(
                    reader,
                    source,
                    match_id="fixture-" + split,
                    venue_id="fixture-" + split,
                    split=split,
                )
            finally:
                reader.close()
            boxes = (
                []
                if index == 1
                else [
                    {
                        "object_id": "fixture-player",
                        "class_id": 0,
                        "bbox": [15, 10, 55, 88],
                        "occluded": False,
                        "truncated": False,
                    }
                ]
            )
            write_json(
                sidecar_path(path),
                {
                    "schema_version": 1,
                    "artifact_type": "players_annotations",
                    "source": record,
                    "provenance": [
                        {"kind": "manual", "reference": "synthetic-training-fixture"}
                    ],
                    "frames": [
                        {"frame_index": 0, "review_status": "verified", "boxes": boxes}
                    ],
                },
            )
    export_dataset(source, base / "exports")
    return base / "exports"


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Generate random fixture weights offline")
    parser.add_argument("family", choices=("yolo", "rfdetr"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make_weights(args.family, args.output)
