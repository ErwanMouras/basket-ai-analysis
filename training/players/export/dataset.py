"""Local-only, reproducible player dataset export. No DVC or model invocation."""

import hashlib
import platform
from pathlib import Path

import cv2
import numpy as np
import yaml

from training.common.files import write_bytes, write_json, write_jsonl
from training.common.provenance import ROOT, file_hash, object_hash, source_fingerprints
from training.players.annotator.media import MediaReader
from training.players.contracts import validate_manifest
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
from training.players.export.publication import publish, staging
from training.players.export.sources import check_unchanged, discover
from training.players.export.verify import (
    artifact_inventory,
    dataset_identity,
    export_identity,
    verify_pair,
)


def code_fingerprints():
    paths = list((ROOT / "training/players/export").glob("*.py"))
    paths += [
        ROOT / p
        for p in (
            "training/players/contracts.py",
            "training/players/annotator/media.py",
            "training/players/annotator/model.py",
            "training/common/config.py",
            "training/common/files.py",
            "training/common/provenance.py",
            "training/common/sources.py",
        )
    ]
    return source_fingerprints(ROOT, paths)


def write_yaml(path, value):
    write_bytes(
        path, yaml.safe_dump(value, sort_keys=True, allow_unicode=True).encode("utf-8")
    )


def build_pair(stage, clips, audit, config):
    roots = {fmt: stage / fmt for fmt in ("yolo", "coco")}
    for root in roots.values():
        for split in config.splits:
            (root / "images" / split).mkdir(parents=True)
    for split in config.splits:
        (roots["yolo"] / "labels" / split).mkdir(parents=True)
    records, provenance, pixels_by_split = [], {}, {}
    for clip in clips:
        document = clip["document"]
        source = document["source"]
        for entry in document["provenance"]:
            provenance[object_hash(entry)] = entry
        geometry = Geometry.build(source["width"], source["height"], config)
        reader = MediaReader(clip["media"])
        try:
            for frame in clip["frames"]:
                original = reader.read(frame["frame_index"])
                pixel_hash = hashlib.sha256(original.tobytes()).hexdigest()
                pixel_key = (original.shape, pixel_hash)
                if (
                    pixel_key in pixels_by_split
                    and pixels_by_split[pixel_key] != source["split"]
                ):
                    raise ValueError(
                        "Identical decoded frame occurs in multiple splits"
                    )
                pixels_by_split[pixel_key] = source["split"]
                options = (
                    [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
                    if config.image_format == "jpg"
                    else [cv2.IMWRITE_PNG_COMPRESSION, 3]
                )
                ok, encoded = cv2.imencode(
                    "." + config.image_format, geometry.image(original), options
                )
                if not ok:
                    raise ValueError("Could not encode an exported frame")
                content = encoded.tobytes()
                record = frame_record(
                    source,
                    frame,
                    geometry,
                    config.image_format,
                    hashlib.sha256(content).hexdigest(),
                )
                records.append(record)
                # Encode once, write the exact same bytes in both independent datasets.
                for root in roots.values():
                    write_bytes(root / record["image"], content)
        finally:
            reader.close()
    for root in roots.values():
        write_jsonl(root / "frames.jsonl", records)
        write_json(root / "split_audit.json", audit)
        write_json(root / "stats.json", statistics(records, config.splits))
        write_yaml(root / "config.resolved.yaml", config.to_dict())
    for record in records:
        write_bytes(
            roots["yolo"] / label_path(record), yolo_label(record).encode("utf-8")
        )
    write_yaml(roots["yolo"] / "data.yaml", yolo_config(config.splits))
    for split in config.splits:
        write_json(
            roots["coco"] / "annotations" / f"{split}.json",
            coco_document(records, split),
        )
    common = {
        "schema_version": 1,
        "artifact_type": "players_dataset_export",
        "classes": {"0": "player"},
        "splits": list(config.splits),
        "sources": [
            {
                "source": c["document"]["source"],
                "annotations_sha256": c["annotations_sha256"],
            }
            for c in clips
        ],
        "provenance": [provenance[key] for key in sorted(provenance)],
        "parameters": {
            "config": config.to_dict(),
            "code_sha256": code_fingerprints(),
            "software": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "pyyaml": yaml.__version__,
                "opencv_build_sha256": hashlib.sha256(
                    cv2.getBuildInformation().encode()
                ).hexdigest(),
            },
            **{
                name + "_sha256": file_hash(roots["yolo"] / name)
                for name in ("frames.jsonl", "split_audit.json")
            },
        },
    }
    for fmt, root in roots.items():
        manifest = {
            **common,
            "dataset_id": dataset_identity(common),
            "format": fmt,
            "artifacts": artifact_inventory(root),
        }
        manifest["export_id"] = export_identity(manifest)
        write_json(root / "manifest.json", validate_manifest(manifest))
    return verify_pair(stage)


def export_dataset(source, output, config=None):
    config = config if config is not None else ExportConfig()
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(source) or source.is_relative_to(resolved_output):
        raise ValueError("Source and output directories must be disjoint")
    clips, audit, checks, original_inventory = discover(source, config)
    with staging(output) as (root, stage):
        manifests = build_pair(stage, clips, audit, config)
        check_unchanged(source, checks, original_inventory)
        generation = publish(root, stage, manifests)
    return {
        "dataset_id": manifests["yolo"]["dataset_id"],
        "export_ids": {fmt: m["export_id"] for fmt, m in manifests.items()},
        "generation": str(generation),
        "output": str(output),
    }
