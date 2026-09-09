"""Portable data configuration for the V4/V5 SDK. Copied into each SDK export."""

import json
from copy import deepcopy
from pathlib import Path


def build_data(input_width: int, input_height: int, version=5, batch_size=2, workers=4):
    """Call from a future SDK training config; resolution is chosen by that caller."""
    if version not in (4, 5):
        raise ValueError("This dataset adapter supports SDK V4 and V5")
    divisor = 16 if version == 5 else 8
    for value in (input_width, input_height):
        if type(value) is not int or value < divisor or value % divisor:
            raise ValueError(f"Network input dimensions must be divisible by {divisor}")
    if (
        type(batch_size) is not int
        or batch_size < 1
        or type(workers) is not int
        or workers < 1
    ):
        raise ValueError(
            "batch_size and workers must be positive integers for this SDK"
        )
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    geometry = next(
        source["geometry"]
        for source in manifest["sources"]
        if source["exported_frames"]
    )
    pipeline = [
        dict(type="LoadMultiImagesFromPaths", to_rgb=True),
        dict(
            type="Resize",
            keys=["path_prev", "path", "path_next"],
            size=(input_height, input_width),
        ),
        dict(
            type="ConcatChannels",
            keys=["path_prev", "path", "path_next"],
            output_key="image",
        ),
        dict(
            type="LoadAndFormatMultiTargets",
            keys=["gt_path_prev", "gt_path", "gt_path_next"],
            output_key="target",
        ),
        dict(
            type="Finalize",
            image_key="image",
            final_keys=["image", "target", "coords", "visibility", "original_info"],
        ),
    ]
    data = dict(samples_per_gpu=batch_size, workers_per_gpu=workers)
    for split in manifest["parameters"]["splits"]:
        data[split] = dict(
            type="TennisDataset",
            data_dir=str(root),
            csv_path=str(root / f"labels_context_{split}.csv"),
            input_width=input_width,
            input_height=input_height,
            pipeline=deepcopy(pipeline),
        )
    return {
        "data": data,
        "original_size": (geometry["height"], geometry["width"]),
        "input_size": (input_height, input_width),
    }
