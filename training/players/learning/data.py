"""Verified exports become private framework views; frameworks never modify exports."""

import json
import shutil
from pathlib import Path

import yaml

from training.common.files import write_json, write_jsonl
from training.common.provenance import object_hash
from training.players.export.verify import read_json, verify_export


def preflight(config):
    root = Path(config["dataset"]).resolve(strict=True)
    config["dataset"] = str(root)  # Pin the published generation once.
    output = Path(config["output"]).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Dataset and run output must be disjoint")
    manifest = verify_export(root)
    expected = "yolo" if config["model"] == "yolo" else "coco"
    if manifest["format"] != expected:
        raise ValueError(f"Expected a {expected} player export")
    rows = [
        json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()
    ]
    selected = []
    for split in ("train", "val"):
        limit = config[f"max_{split}_images"]
        items = [r for r in rows if r["split"] == split][:limit]
        if not items or not any(r["boxes"] for r in items):
            raise ValueError(
                f"{split} requires images and at least one annotated player"
            )
        selected.extend(items)
    config.update(
        dataset_id=manifest["dataset_id"],
        export_id=manifest["export_id"],
        selection_id=object_hash(selected),
    )
    return manifest, selected


def prepare(config, records, output):
    root = Path(config["dataset"])
    view = output / "data"
    view.mkdir()
    write_jsonl(output / "selection.jsonl", records)
    if config["model"] == "yolo":
        for row in records:
            target = view / row["image"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(root / row["image"])
            label = Path(row["image"].replace("images/", "labels/", 1)).with_suffix(
                ".txt"
            )
            (view / label).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / label, view / label)
        (view / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(view),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {0: "player"},
                }
            )
        )
    else:
        for split, name in (("train", "train"), ("val", "valid")):
            directory = view / name
            directory.mkdir()
            coco = read_json(root / "annotations" / f"{split}.json")
            names = {r["image"] for r in records if r["split"] == split}
            coco["images"] = [i for i in coco["images"] if i["file_name"] in names]
            ids = {i["id"] for i in coco["images"]}
            coco["annotations"] = [
                a for a in coco["annotations"] if a["image_id"] in ids
            ]
            for image in coco["images"]:
                original = root / image["file_name"]
                image["file_name"] = original.name
                (directory / original.name).symlink_to(original)
            write_json(directory / "_annotations.coco.json", coco)
    return view
