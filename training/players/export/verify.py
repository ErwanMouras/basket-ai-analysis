"""Offline integrity and semantic checks, without the original media or models."""

import json
from pathlib import Path

import cv2
import yaml

from training.common.provenance import file_hash, object_hash
from training.players.contracts import validate_annotations, validate_manifest
from training.players.export.config import ExportConfig
from training.players.export.formats import (
    coco_document,
    frame_record,
    label_path,
    statistics,
    yolo_config,
    yolo_label,
)
from training.players.export.geometry import Geometry


def dataset_identity(manifest):
    return object_hash(
        {
            k: v
            for k, v in manifest.items()
            if k not in ("dataset_id", "export_id", "format", "artifacts")
        }
    )


def export_identity(manifest):
    return object_hash({k: v for k, v in manifest.items() if k != "export_id"})


def artifact_inventory(root):
    inventory = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Export contains a symlink: {path}")
        if path.is_file() and path != root / "manifest.json":
            inventory[path.relative_to(root).as_posix()] = file_hash(path)
        elif not path.is_file() and not path.is_dir():
            raise ValueError(f"Unsupported export artifact: {path}")
    return inventory


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_export(path):
    """Verify a single resolved format directory; return its validated manifest."""
    root = Path(path).resolve(strict=True)
    if (root / "manifest.json").is_symlink():
        raise ValueError("Manifest must not be a symlink")
    manifest = validate_manifest(read_json(root / "manifest.json"))
    if (
        dataset_identity(manifest) != manifest["dataset_id"]
        or export_identity(manifest) != manifest["export_id"]
    ):
        raise ValueError("Invalid dataset_id or export_id")
    if artifact_inventory(root) != manifest["artifacts"]:
        raise ValueError(
            "Export artifact integrity mismatch (missing, extra or modified file)"
        )
    params = manifest["parameters"]
    config = ExportConfig(**params["config"])
    if list(config.splits) != manifest["splits"]:
        raise ValueError("Configuration/manifest splits mismatch")
    for split in config.splits:
        required = [root / "images" / split]
        if manifest["format"] == "yolo":
            required.append(root / "labels" / split)
        if any(not path.is_dir() for path in required):
            raise ValueError("Missing split directory")
    for name in ("frames.jsonl", "split_audit.json"):
        if params[name + "_sha256"] != manifest["artifacts"][name]:
            raise ValueError(f"Common artifact identity mismatch: {name}")
    if yaml.safe_load((root / "config.resolved.yaml").read_text()) != config.to_dict():
        raise ValueError("Resolved configuration mismatch")
    records = [
        json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()
    ]
    if not records:
        raise ValueError("An export requires verified frames")
    sources = {s["source"]["source_id"]: s["source"] for s in manifest["sources"]}
    seen, used_sources = set(), set()
    expected = {
        "frames.jsonl",
        "split_audit.json",
        "config.resolved.yaml",
        "stats.json",
    }
    for record in records:
        source = sources[record["source_id"]]
        key = (record["source_id"], record["frame_index"])
        if key in seen:
            raise ValueError("Duplicate exported frame")
        seen.add(key)
        used_sources.add(source["source_id"])
        frame = {
            "frame_index": record["frame_index"],
            "review_status": record["review_status"],
            "boxes": [
                {
                    **{k: v for k, v in b.items() if k != "source_bbox"},
                    "bbox": b["source_bbox"],
                }
                for b in record["boxes"]
            ],
        }
        validate_annotations(
            {
                "schema_version": 1,
                "artifact_type": "players_annotations",
                "source": source,
                "provenance": manifest["provenance"],
                "frames": [frame],
            }
        )
        geometry = Geometry.build(source["width"], source["height"], config)
        if record != frame_record(
            source, frame, geometry, config.image_format, record["image_sha256"]
        ):
            raise ValueError("Frame index or coordinate transformation mismatch")
        expected.add(record["image"])
        if manifest["artifacts"].get(record["image"]) != record["image_sha256"]:
            raise ValueError("Frame image hash mismatch")
        image = cv2.imread(str(root / record["image"]))
        if image is None or image.shape[:2] != (record["height"], record["width"]):
            raise ValueError("Invalid exported image dimensions")
        if manifest["format"] == "yolo":
            label = label_path(record)
            expected.add(label)
            if (root / label).read_text() != yolo_label(record):
                raise ValueError("YOLO labels disagree with the frame index")
    if used_sources != sources.keys():
        raise ValueError("Manifest includes a source without exported frames")
    if read_json(root / "stats.json") != statistics(records, config.splits):
        raise ValueError("Export statistics mismatch")
    if manifest["format"] == "yolo":
        expected.add("data.yaml")
        if yaml.safe_load((root / "data.yaml").read_text()) != yolo_config(
            config.splits
        ):
            raise ValueError("YOLO dataset configuration mismatch")
    else:
        for split in config.splits:
            name = f"annotations/{split}.json"
            expected.add(name)
            if read_json(root / name) != coco_document(records, split):
                raise ValueError("COCO annotations disagree with the frame index")
    if expected != manifest["artifacts"].keys():
        raise ValueError("Unexpected artifact layout")
    return manifest


def verify_pair(generation):
    generation = Path(generation).resolve(strict=True)
    if {p.name for p in generation.iterdir()} != {"yolo", "coco"}:
        raise ValueError("Expected exactly YOLO and COCO in a generation")
    if any((generation / fmt).is_symlink() for fmt in ("yolo", "coco")):
        raise ValueError("Format directories must not be symlinks inside a generation")
    manifests = {fmt: verify_export(generation / fmt) for fmt in ("yolo", "coco")}
    if any(m["format"] != fmt for fmt, m in manifests.items()):
        raise ValueError("Incorrect format directory")
    if manifests["yolo"]["dataset_id"] != manifests["coco"]["dataset_id"]:
        raise ValueError("YOLO and COCO refer to different datasets")
    for name in (
        "frames.jsonl",
        "split_audit.json",
        "stats.json",
        "config.resolved.yaml",
    ):
        if manifests["yolo"]["artifacts"][name] != manifests["coco"]["artifacts"][name]:
            raise ValueError("YOLO/COCO common artifacts differ")
    return manifests
