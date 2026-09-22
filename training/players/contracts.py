"""Strict, JSON-native v1 contracts in original source pixel coordinates.

Validation is pure: it neither reads media nor changes the supplied document.
Returned documents are independent copies. Unknown fields require an explicit
schema revision instead of silently discarding annotation information.
"""

import json
import math
import re
from copy import deepcopy
from pathlib import PurePosixPath

from training.common.config import SPLITS

SCHEMA_VERSION = 1
CLASSES = {0: "player"}
REVIEW_STATES = ("unannotated", "proposed", "in_progress", "verified")


def _fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("Expected an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError("Object keys must be strings")
    missing, unknown = (
        set(required) - value.keys(),
        value.keys() - set(required) - set(optional),
    )
    if missing or unknown:
        raise ValueError(
            f"Invalid fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _string(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty trimmed string")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 digest")


def _path(value):
    _string(value, "path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or path.as_posix() != value
        or value == "."
    ):
        raise ValueError(
            "Paths must be canonical relative POSIX paths without traversal"
        )


def _list(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")


def _provenance(entries):
    _list(entries, "provenance")
    if not entries:
        raise ValueError("At least one provenance entry is required")
    for entry in entries:
        _fields(entry, ("kind", "reference"), ("sha256",))
        if entry["kind"] not in ("manual", "model", "import"):
            raise ValueError("Unknown provenance kind")
        _string(entry["reference"], "reference")
        if "sha256" in entry:
            _hash(entry["sha256"])
        if entry["kind"] == "model" and "sha256" not in entry:
            raise ValueError("Model provenance requires the checkpoint SHA-256")


def _source(source):
    _fields(
        source,
        (
            "source_id",
            "kind",
            "path",
            "sha256",
            "width",
            "height",
            "frame_count",
            "fps",
            "match_id",
            "venue_id",
            "split",
        ),
    )
    _string(source["source_id"], "source_id")
    _path(source["path"])
    _hash(source["sha256"])
    _string(source["match_id"], "match_id")
    if source["venue_id"] is not None:
        _string(source["venue_id"], "venue_id")
    if source["split"] is not None and source["split"] not in SPLITS:
        raise ValueError("Unknown source split")
    for key in ("width", "height", "frame_count"):
        _integer(source[key], key, 1)
    if source["kind"] == "video":
        _number(source["fps"], "fps")
        if source["fps"] <= 0:
            raise ValueError("Video fps must be positive")
    elif source["kind"] == "image":
        if source["fps"] is not None or source["frame_count"] != 1:
            raise ValueError("Images require frame_count=1 and fps=null")
    else:
        raise ValueError("Source kind must be video or image")


def _header(document, artifact_type, required):
    _fields(document, ("schema_version", "artifact_type", *required))
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported players schema version")
    if document["artifact_type"] != artifact_type:
        raise ValueError(f"Expected artifact_type={artifact_type}")


def _frame_index(index, source, seen):
    _integer(index, "frame_index")
    if index >= source["frame_count"] or index in seen:
        raise ValueError("Frame index is duplicated or outside the source")
    seen.add(index)


def _bbox(box, source):
    _integer(box["class_id"], "class_id")
    if box["class_id"] != 0:
        raise ValueError("Only class_id=0 (player) is supported")
    coordinates = box["bbox"]
    _list(coordinates, "bbox")
    if len(coordinates) != 4:
        raise ValueError("bbox requires [x1, y1, x2, y2]")
    for value in coordinates:
        _number(value, "bbox coordinate")
    x1, y1, x2, y2 = coordinates
    if not (0 <= x1 < x2 <= source["width"] and 0 <= y1 < y2 <= source["height"]):
        raise ValueError("bbox must have positive area and lie within the source")


def _confidence(value):
    _number(value, "confidence")
    if not 0 <= value <= 1:
        raise ValueError("confidence must be between 0 and 1")


def validate_annotations(document):
    """Validate a complete sidecar; absent/unreviewed frames are never negatives."""
    _header(document, "players_annotations", ("source", "provenance", "frames"))
    source = document["source"]
    _source(source)
    _provenance(document["provenance"])
    _list(document["frames"], "frames")
    seen = set()
    for frame in document["frames"]:
        _fields(frame, ("frame_index", "review_status", "boxes"))
        _frame_index(frame["frame_index"], source, seen)
        if frame["review_status"] not in REVIEW_STATES:
            raise ValueError("Unknown review_status")
        _list(frame["boxes"], "boxes")
        if frame["review_status"] == "unannotated" and frame["boxes"]:
            raise ValueError("An unannotated frame cannot contain boxes")
        box_ids = set()
        for box in frame["boxes"]:
            _fields(
                box,
                ("object_id", "class_id", "bbox", "occluded", "truncated"),
                ("confidence",),
            )
            _string(box["object_id"], "object_id")
            if box["object_id"] in box_ids:
                raise ValueError("Duplicate object_id within a frame")
            box_ids.add(box["object_id"])
            _bbox(box, source)
            for key in ("occluded", "truncated"):
                if box[key] is not None and type(box[key]) is not bool:
                    raise ValueError(f"{key} must be boolean or null (unknown)")
            if "confidence" in box:
                _confidence(box["confidence"])
    return deepcopy(document)


def verified_frames(document):
    """Return only explicitly verified frames, including verified empty frames."""
    return [
        frame
        for frame in validate_annotations(document)["frames"]
        if frame["review_status"] == "verified"
    ]


def validate_predictions(document):
    """Validate detector outputs without requiring ground truth or a dataset split."""
    _header(document, "players_predictions", ("source", "model", "frames"))
    source = document["source"]
    _source(source)
    model = document["model"]
    _fields(
        model, ("family", "checkpoint_sha256", "training_run_id", "inference_config")
    )
    _string(model["family"], "family")
    _hash(model["checkpoint_sha256"])
    if model["training_run_id"] is not None:
        _string(model["training_run_id"], "training_run_id")
    _json_mapping(model["inference_config"])
    _list(document["frames"], "frames")
    seen = set()
    for frame in document["frames"]:
        _fields(frame, ("frame_index", "detections"))
        _frame_index(frame["frame_index"], source, seen)
        _list(frame["detections"], "detections")
        for detection in frame["detections"]:
            _fields(detection, ("class_id", "bbox", "confidence"))
            _bbox(detection, source)
            _confidence(detection["confidence"])
    return deepcopy(document)


def _json_mapping(value):
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")

    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError("Expected a JSON value")

    check(value)
    json.dumps(value, allow_nan=False)


def validate_manifest(document):
    """Validate export metadata, not the files or the derivation of its hashes."""
    _header(
        document,
        "players_dataset_export",
        (
            "dataset_id",
            "export_id",
            "format",
            "classes",
            "splits",
            "sources",
            "parameters",
            "artifacts",
            "provenance",
        ),
    )
    for key in ("dataset_id", "export_id"):
        _hash(document[key])
    if document["format"] not in ("yolo", "coco"):
        raise ValueError("Players export format must be yolo or coco")
    if document["classes"] != {"0": "player"}:
        raise ValueError('classes must be {"0": "player"}')
    splits = document["splits"]
    _list(splits, "splits")
    if (
        not splits
        or any(split not in SPLITS for split in splits)
        or len(set(splits)) != len(splits)
    ):
        raise ValueError("Expected distinct train/val/test splits")
    _json_mapping(document["parameters"])
    _provenance(document["provenance"])
    _list(document["sources"], "sources")
    if not document["sources"]:
        raise ValueError("An export needs at least one source")
    seen_ids, seen_paths, seen_matches, seen_hashes = set(), set(), {}, {}
    for entry in document["sources"]:
        _fields(entry, ("source", "annotations_sha256"))
        _hash(entry["annotations_sha256"])
        source = entry["source"]
        _source(source)
        if source["split"] not in splits:
            raise ValueError("Source split is not part of the export")
        if source["source_id"] in seen_ids or source["path"] in seen_paths:
            raise ValueError("Duplicate source identity or path")
        seen_ids.add(source["source_id"])
        seen_paths.add(source["path"])
        for seen, key in (
            (seen_matches, source["match_id"]),
            (seen_hashes, source["sha256"]),
        ):
            if key in seen and seen[key] != source["split"]:
                raise ValueError("Match or source content occurs in multiple splits")
            seen[key] = source["split"]
    inventory = document["artifacts"]
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("An export needs an artifact inventory")
    for path, digest in inventory.items():
        _path(path)
        if path == "manifest.json":
            raise ValueError("The manifest cannot include its own hash")
        _hash(digest)
    return deepcopy(document)
