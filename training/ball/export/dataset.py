"""Build requested datasets once, then publish complete outputs with rollback."""

import fcntl
import json
import os
import platform
import shutil
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import yaml

from training.ball.annotator.video import VideoReader

from .config import FORMATS, ExportConfig, training_profile
from .detection import write_coco, write_yolo
from .files import link_or_copy, write_image, write_json, write_jsonl
from .metadata import source_videos
from .sources import continuous_segments, discover, file_hash, frame_record, object_hash
from .temporal import triplets, write_sdk, write_v3, write_v4

ARTIFACT_TYPE = "ball_dataset_export"


@contextmanager
def output_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".export.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another export is already writing to {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_destination(root: Path, formats):
    for name in formats:
        destination = root / name
        if destination.is_symlink():
            raise ValueError(f"Refusing to replace a symlink: {destination}")
        if destination.exists():
            manifest = destination / "manifest.json"
            if not manifest.is_file():
                raise ValueError(
                    f"Destination is not an owned ball export: {destination}"
                )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("artifact_type") != ARTIFACT_TYPE
                or payload.get("format") != name
            ):
                raise ValueError(
                    f"Destination is not an owned {name} export: {destination}"
                )


def publish(staging: Path, root: Path, formats):
    """On ordinary failure, put every previous dataset back before propagating it."""
    previous = staging / ".previous"
    previous.mkdir()
    moved_old, moved_new = [], []
    try:
        for name in formats:
            if (root / name).exists():
                os.replace(root / name, previous / name)
                moved_old.append(name)
            os.replace(staging / name, root / name)
            moved_new.append(name)
    except BaseException:
        try:
            for name in reversed(moved_new):
                os.replace(root / name, staging / name)
            for name in reversed(moved_old):
                os.replace(previous / name, root / name)
        except OSError as exc:
            raise OSError(
                f"Rollback failed; previous exports are retained in {previous}"
            ) from exc
        raise


@contextmanager
def staging_directory(root: Path):
    staging = Path(tempfile.mkdtemp(prefix=".ball-export-", dir=root))
    try:
        yield staging
    except BaseException:
        previous = staging / ".previous"
        # Never let temporary-directory cleanup erase the last copy of an old
        # export when the filesystem also refuses its restoration.
        if not previous.exists() or not any(previous.iterdir()):
            shutil.rmtree(staging)
        raise
    else:
        shutil.rmtree(staging)


def export_dataset(
    source_root: Path,
    output_root: Path,
    config: ExportConfig,
    formats=FORMATS,
    training_config: Path | None = None,
    progress=print,
) -> dict:
    source_root = source_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    if output_root.is_relative_to(source_root) or source_root.is_relative_to(
        output_root
    ):
        raise ValueError("Export destination must not overlap the source dataset")
    if (
        not formats
        or any(name not in FORMATS for name in formats)
        or len(set(formats)) != len(formats)
    ):
        raise ValueError(f"Select distinct formats from {FORMATS}")
    profile = None
    if "tracknet-totnet" in formats and "v4" in config.tracknet_layouts:
        if training_config is None:
            raise ValueError("V4 NumPy preparation requires a training configuration")
        profile = training_profile(training_config, "tracknet_v4")
        if profile["version"] != 4:
            raise ValueError("The tracknet_v4 preparation profile must use version 4")

    with output_lock(output_root):
        validate_destination(output_root, formats)
        progress(
            "Validating source annotations, folder splits and video fingerprints..."
        )
        clips, excluded, source_checks = discover(source_root, config)
        audited_videos = {clip.video for clip in clips} | {
            source_root / source["video"] for source in excluded
        }
        clip_records = [
            [frame_record(clip, index, config) for index in clip.annotations]
            for clip in clips
        ]
        records = [record for group in clip_records for record in group]
        if "tracknet-totnet" in formats and "sdk" in config.tracknet_layouts:
            sizes = {(r["width"], r["height"]) for r in records}
            if len(sizes) != 1:
                raise ValueError(
                    "The SDK requires one exported image size; "
                    "set resize_width and resize_height"
                )
            if not any(triplets(clip_records)):
                raise ValueError(
                    "SDK export requires three consecutive selected frames"
                )
        if profile:
            # Float32 RGB triplets plus three target maps. No unbounded allocation.
            samples = sum(
                max(0, (len(segment) - 3) // profile["sequence_stride"] + 1)
                for group in clip_records
                for segment in continuous_segments(group)
            )
            size = samples * 12 * profile["input_width"] * profile["input_height"] * 4
            progress(f"V4 prepared arrays: approximately {size / 1024**3:.2f} GiB")
            if size > shutil.disk_usage(output_root).free * 0.9:
                raise ValueError("Insufficient disk space for the requested V4 tensors")

        software = {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        }
        code_paths = sorted(Path(__file__).parent.glob("*.py")) + [
            Path(__file__).parents[1] / "annotator" / name
            for name in ("model.py", "video.py")
        ]
        code_hash = object_hash(
            {
                path.relative_to(Path(__file__).parents[1]).as_posix(): file_hash(path)
                for path in code_paths
            }
        )
        source_settings = {
            key: value
            for key, value in config.to_dict().items()
            if key not in ("tracknet_layouts", "heatmap_radius", "heatmap_variance")
        }
        provenance = [clip.provenance for clip in clips]
        dataset_id = object_hash(
            {
                "sources": provenance,
                "settings": source_settings,
                "software": software,
                "exporter_sha256": code_hash,
            }
        )
        reports = {}
        with staging_directory(output_root) as staging:
            for name in formats:
                (staging / name).mkdir()
            shared = staging / ".frames"
            for clip, group in zip(clips, clip_records):
                progress(f"Extracting {clip.provenance['video']}: {len(group)} frames")
                reader = VideoReader(clip.video)
                try:
                    for record in group:
                        image = clip.geometry.transform_image(
                            reader.read(record["frame_index"])
                        )
                        frame_path = shared / record["image"]
                        write_image(frame_path, image, config.jpeg_quality)
                        for name in formats:
                            link_or_copy(frame_path, staging / name / record["image"])
                finally:
                    reader.close()
            for name in formats:
                root = staging / name
                progress(f"Writing {name} labels and metadata...")
                if name == "yolo":
                    report = write_yolo(root, records, config.splits)
                elif name == "coco":
                    report = write_coco(root, records, config.splits)
                else:
                    report = {}
                    if "sdk" in config.tracknet_layouts:
                        report["sdk_triplets"] = write_sdk(root, clip_records, config)
                    if "v3" in config.tracknet_layouts:
                        report["v3_segments"] = write_v3(root, clip_records)
                    if "v4" in config.tracknet_layouts:
                        report["v4_samples"] = write_v4(root, clip_records, profile)
                write_jsonl(root / "frames.jsonl", records)
                (root / "export_config.resolved.yaml").write_text(
                    yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
                )
                write_json(
                    root / "split_audit.json",
                    {
                        "splits_checked": ["train", "val", "test"],
                        "sources": provenance + excluded,
                    },
                )
                reports[name] = report

            # Catch source edits during extraction; never publish a mixed version.
            if set(source_videos(source_root)) != audited_videos:
                raise ValueError("Source videos changed during export")
            for path, digest in source_checks.items():
                current = file_hash(path) if path.exists() else None
                if current != digest:
                    raise ValueError(f"Source changed during export: {path}")

            # Reuse hashes of hard-linked images across formats.
            hash_cache = {}
            for name in formats:
                root = staging / name
                inventory = []
                for path in sorted(root.rglob("*")):
                    if not path.is_file():
                        continue
                    stat = path.stat()
                    key = (stat.st_dev, stat.st_ino)
                    if key not in hash_cache:
                        hash_cache[key] = file_hash(path)
                    inventory.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "bytes": stat.st_size,
                            "sha256": hash_cache[key],
                        }
                    )
                write_jsonl(root / "artifacts.jsonl", inventory)
                parameters = (
                    config.to_dict() if name == "tracknet-totnet" else source_settings
                )
                export_id = object_hash(
                    {
                        "dataset_id": dataset_id,
                        "format": name,
                        "parameters": parameters,
                        "artifacts": inventory,
                        "preparation": profile if name == "tracknet-totnet" else None,
                    }
                )
                manifest = {
                    "schema_version": 1,
                    "artifact_type": ARTIFACT_TYPE,
                    "format": name,
                    "dataset_id": dataset_id,
                    "export_id": export_id,
                    "exporter_sha256": code_hash,
                    "software": software,
                    "parameters": parameters,
                    "sources": provenance,
                    "excluded_sources": excluded,
                    "frames": "frames.jsonl",
                    "resolved_config": "export_config.resolved.yaml",
                    "split_audit": "split_audit.json",
                    "artifacts": "artifacts.jsonl",
                    "artifacts_sha256": file_hash(root / "artifacts.jsonl"),
                    "frame_counts": dict(Counter(r["split"] for r in records)),
                    "status_counts": dict(Counter(r["status"] for r in records)),
                    "report": reports[name],
                }
                if name == "tracknet-totnet" and profile:
                    manifest["training_preparation"] = profile
                write_json(root / "manifest.json", manifest)
            publish(staging, output_root, formats)
        progress(f"Export complete: {len(records)} frames; dataset {dataset_id[:12]}")
        return {"dataset_id": dataset_id, "frames": len(records), "formats": reports}


def verify_dataset(root: Path) -> dict:
    """Check artifacts and identity inputs; descriptive counters are not verified."""
    root = root.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_type") != ARTIFACT_TYPE
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("Not a supported ball dataset export")
    source_settings = {
        key: value
        for key, value in manifest["parameters"].items()
        if key not in ("tracknet_layouts", "heatmap_radius", "heatmap_variance")
    }
    dataset_id = object_hash(
        {
            "sources": manifest["sources"],
            "settings": source_settings,
            "software": manifest["software"],
            "exporter_sha256": manifest["exporter_sha256"],
        }
    )
    if dataset_id != manifest["dataset_id"]:
        raise ValueError("Dataset provenance fingerprint mismatch")
    inventory_path = root / "artifacts.jsonl"
    if file_hash(inventory_path) != manifest["artifacts_sha256"]:
        raise ValueError("Artifact inventory checksum mismatch")
    inventory = []
    with inventory_path.open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            path = root / entry["path"]
            if not path.resolve().is_relative_to(root):
                raise ValueError("Artifact path leaves the dataset")
            if (
                path.stat().st_size != entry["bytes"]
                or file_hash(path) != entry["sha256"]
            ):
                raise ValueError(f"Artifact checksum mismatch: {entry['path']}")
            inventory.append(entry)
    expected_paths = {entry["path"] for entry in inventory}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in (".cache", ".npz")
    } - {"manifest.json", "artifacts.jsonl"}
    # Ignore loader caches, but detect added images or labels that would silently
    # change a training dataset discovered by globbing.
    if actual_paths != expected_paths:
        raise ValueError("Export contains unexpected or missing files")
    expected_id = object_hash(
        {
            "dataset_id": manifest["dataset_id"],
            "format": manifest["format"],
            "parameters": manifest["parameters"],
            "artifacts": inventory,
            "preparation": manifest.get("training_preparation"),
        }
    )
    if expected_id != manifest["export_id"]:
        raise ValueError("Export fingerprint mismatch")
    return {"export_id": expected_id, "files": len(inventory)}
