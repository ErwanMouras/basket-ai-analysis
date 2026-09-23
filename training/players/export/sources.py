"""Audit every source/split, then select only explicitly verified player frames."""

import json
import math
from collections import Counter
from pathlib import Path

from training.common.config import SPLITS
from training.common.provenance import file_hash
from training.common.sources import VIDEO_EXTENSIONS
from training.players.annotator.media import (
    IMAGE_EXTENSIONS,
    MediaReader,
    resolve_identity,
)
from training.players.annotator.model import sidecar_path
from training.players.contracts import validate_annotations


def inventory(root):
    """Include missing media named by sidecars and reject hidden symlink subtrees."""
    media, sidecars = set(), set()
    for path in root.rglob("*"):
        if "nba_cache" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Source symlinks are not supported: {path}")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        ):
            media.add(path)
        if path.name.endswith(".playersann.json"):
            sidecars.add(path)
            media.add(path.with_name(path.name.removesuffix(".playersann.json")))
    return media, sidecars


def discover(root: Path, config):
    root = root.resolve(strict=True)
    media_paths, sidecars = inventory(root)
    checks, clips, audit = {}, [], []
    seen = {name: {} for name in ("match", "content", "source_id", "venue")}
    match_venues = {}

    def remember(path):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Input escapes source root: {path}")
        digest = file_hash(path) if path.exists() else None
        if path in checks and checks[path] != digest:
            raise ValueError(f"Input changed during audit: {path}")
        checks[path] = digest
        return checks[path]

    for media in sorted(media_paths):
        relative = media.relative_to(root).as_posix()
        split = media.relative_to(root).parts[0]
        if split not in SPLITS or len(media.relative_to(root).parts) < 2:
            raise ValueError(f"Sources must be under train/, val/ or test/: {relative}")
        digest = remember(media)
        if digest is None:
            raise ValueError(f"Missing media referenced by annotations: {relative}")
        sidecar = sidecar_path(media)
        annotation_hash = remember(sidecar)
        document = (
            validate_annotations(json.loads(sidecar.read_text(encoding="utf-8")))
            if annotation_hash
            else None
        )
        remember(media.with_name(media.name + ".meta.yaml"))
        for parent in media.parents:
            if not parent.is_relative_to(root):
                break
            remember(parent / "match.yaml")
        declared = (
            {key: document["source"][key] for key in ("match_id", "venue_id", "split")}
            if document
            else {}
        )
        identity = resolve_identity(media, root, **declared)
        if identity["split"] != split:
            raise ValueError(f"Folder/metadata split mismatch: {relative}")
        venue, match = identity["venue_id"], identity["match_id"]
        if venue is not None:
            if match in match_venues and match_venues[match] != venue:
                raise ValueError(f"Conflicting venue for match {match}")
            match_venues[match] = venue
        if config.split_by_venue and venue is None:
            raise ValueError(
                f"split_by_venue requires venue_id for every source: {relative}"
            )
        for name, key in (
            ("match", match),
            ("content", digest),
            ("venue", venue if config.split_by_venue else None),
        ):
            if key is None:
                continue
            if key in seen[name] and seen[name][key] != split:
                raise ValueError(f"{name} occurs in multiple splits: {relative}")
            seen[name][key] = split
        states = Counter()
        selected = []
        if document:
            source = document["source"]
            if source["split"] != split:
                raise ValueError(f"Annotation/folder split mismatch: {relative}")
            if source["venue_id"] != venue:
                raise ValueError(f"Annotation/metadata venue mismatch: {relative}")
            if source["path"] != relative or source["sha256"] != digest:
                raise ValueError(
                    f"Annotation source path or content mismatch: {relative}"
                )
            if source["source_id"] in seen["source_id"]:
                raise ValueError("Duplicate source_id")
            seen["source_id"][source["source_id"]] = relative
            # Validate geometry even on excluded splits; never mutate the sidecar.
            reader = MediaReader(media)
            try:
                for key in ("kind", "width", "height", "frame_count"):
                    if source[key] != getattr(reader, key):
                        raise ValueError(f"Annotation/media {key} mismatch: {relative}")
                if source["fps"] is not None and not math.isclose(
                    source["fps"], reader.fps, rel_tol=1e-6
                ):
                    raise ValueError(f"Annotation/media FPS mismatch: {relative}")
            finally:
                reader.close()
            states.update(frame["review_status"] for frame in document["frames"])
            states["absent"] = source["frame_count"] - len(document["frames"])
            if split in config.splits:
                selected = sorted(
                    (f for f in document["frames"] if f["review_status"] == "verified"),
                    key=lambda f: f["frame_index"],
                )
            if selected:
                clips.append(
                    {
                        "media": media,
                        "document": document,
                        "frames": selected,
                        "annotations_sha256": annotation_hash,
                    }
                )
        audit.append(
            {
                "path": relative,
                "sha256": digest,
                "annotations_sha256": annotation_hash,
                **identity,
                "status_counts": dict(sorted(states.items())),
                "selected_frames": len(selected),
                "reason": "selected"
                if selected
                else "split_not_selected"
                if split not in config.splits
                else "no_verified_frames"
                if document
                else "no_annotations",
            }
        )
    report = {
        "splits_checked": list(SPLITS),
        "split_by_venue": config.split_by_venue,
        "sources": audit,
        "inputs": {
            p.relative_to(root).as_posix(): d for p, d in sorted(checks.items())
        },
    }
    if not clips:
        raise ValueError(
            "No verified player frames in selected splits; annotate and explicitly verify frames first"
        )
    return clips, report, checks, (media_paths, sidecars)


def check_unchanged(root, checks, original_inventory):
    if inventory(root) != original_inventory:
        raise ValueError("Source inventory changed during export")
    for path, digest in checks.items():
        if path.is_symlink() or (file_hash(path) if path.exists() else None) != digest:
            raise ValueError(f"Source changed during export: {path}")
