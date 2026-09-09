"""Validate source clips and keep frame geometry explicit through resizing."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from training.ball.annotator.model import Annotation, Store, VideoMeta
from training.ball.annotator.video import VideoReader

from .config import ExportConfig
from .metadata import audit_sources


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def object_hash(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    content_width: int
    content_height: int
    scale_x: float
    scale_y: float
    left: int
    top: int

    @classmethod
    def for_video(cls, meta: VideoMeta, config: ExportConfig):
        width = config.resize_width or meta.width
        height = config.resize_height or meta.height
        content_width, content_height = width, height
        if config.resize_mode == "letterbox":
            scale = min(width / meta.width, height / meta.height)
            content_width = max(1, min(width, round(meta.width * scale)))
            content_height = max(1, min(height, round(meta.height * scale)))
        return cls(
            width,
            height,
            content_width,
            content_height,
            content_width / meta.width,
            content_height / meta.height,
            (width - content_width) // 2,
            (height - content_height) // 2,
        )

    def transform_image(self, image):
        if image.shape[:2] != (self.content_height, self.content_width):
            interpolation = cv2.INTER_AREA if self.scale_x < 1 else cv2.INTER_LINEAR
            image = cv2.resize(
                image,
                (self.content_width, self.content_height),
                interpolation=interpolation,
            )
        if image.shape[:2] == (self.height, self.width):
            return image
        padded = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        padded[
            self.top : self.top + self.content_height,
            self.left : self.left + self.content_width,
        ] = image
        return padded


@dataclass
class Clip:
    video: Path
    sidecar: Path
    split: str
    clip_id: str
    meta: VideoMeta
    annotations: dict[int, Annotation]
    geometry: Geometry
    provenance: dict


def annotation_status(ann: Annotation) -> str:
    if not ann.has_position:
        return "unknown_position"
    return "occluded_with_position" if ann.occluded else "visible"


def include_annotation(ann: Annotation, config: ExportConfig) -> bool:
    """Select frames independently of whether their known position is a target."""
    if not ann.has_position:
        return config.unknown_position == "empty"
    return True


def discover(root: Path, config: ExportConfig) -> tuple[list[Clip], list[dict], dict]:
    clips, excluded = [], []
    sources, checks = audit_sources(root, file_hash)
    for video, source in sources.items():
        sidecar = video.with_name(video.name + ".ballann.json")
        relative = sidecar.relative_to(root)
        split = source["split"]
        if not sidecar.resolve().is_relative_to(root):
            raise ValueError(f"Source symlink leaves the dataset root: {sidecar}")
        checks[sidecar] = file_hash(sidecar) if sidecar.exists() else None
        if split not in config.splits or not sidecar.exists():
            excluded.append(
                {
                    **source,
                    "sidecar": relative.as_posix(),
                    "reason": "split_not_selected"
                    if split not in config.splits
                    else "no_annotations",
                }
            )
            continue
        sidecar_hash = checks[sidecar]
        reader = VideoReader(video)
        try:
            store = Store(reader.meta)
            if (
                store.annotations
                and max(store.annotations) >= reader.playable_frame_count
            ):
                raise ValueError(
                    f"Annotation targets an undecodable trailing frame: {sidecar}"
                )
        finally:
            reader.close()
        if sidecar_hash != file_hash(sidecar):
            raise ValueError(f"Annotations changed while being read: {sidecar}")
        clip_id = "clip_" + object_hash(Path(*relative.parts[1:]).as_posix())[:16]
        counts = Counter(annotation_status(ann) for ann in store.annotations.values())
        counts["unannotated"] = store.meta.frame_count - len(store.annotations)
        selected = {
            index: ann
            for index, ann in sorted(store.annotations.items())
            if include_annotation(ann, config)
        }
        geometry = Geometry.for_video(store.meta, config)
        clips.append(
            Clip(
                video,
                sidecar,
                split,
                clip_id,
                store.meta,
                selected,
                geometry,
                {
                    **source,
                    "clip_id": clip_id,
                    "sidecar": relative.as_posix(),
                    "sidecar_sha256": sidecar_hash,
                    "video_metadata": {
                        key: value
                        for key, value in asdict(store.meta).items()
                        if key != "path"
                    },
                    "geometry": asdict(geometry),
                    "annotation_counts": dict(counts),
                    "exported_frames": len(selected),
                    "excluded_annotated_frames": len(store.annotations) - len(selected),
                },
            )
        )
    if not clips or not any(clip.annotations for clip in clips):
        raise ValueError("No frames match the selected splits and annotation policies")
    return clips, excluded, checks


def frame_record(clip: Clip, index: int, config: ExportConfig) -> dict:
    ann = clip.annotations[index]
    geo = clip.geometry
    # Keep the source annotation intact while omitting its position from training
    # targets. The frame still provides temporal context for its neighbours.
    position_excluded = (
        ann.has_position and ann.occluded and config.occluded_position == "exclude"
    )
    has_position = ann.has_position and not position_excluded
    position, box = None, None
    if has_position:
        cx, cy, radius = ann.cx, ann.cy, ann.radius
        position = [cx * geo.scale_x + geo.left, cy * geo.scale_y + geo.top]
        # Clip in source space first: letterbox padding is not part of the object.
        x1 = max(0, cx - radius) * geo.scale_x + geo.left
        y1 = max(0, cy - radius) * geo.scale_y + geo.top
        x2 = min(clip.meta.width, cx + radius) * geo.scale_x + geo.left
        y2 = min(clip.meta.height, cy + radius) * geo.scale_y + geo.top
        box = [x1, y1, x2 - x1, y2 - y1]
    return {
        "clip_id": clip.clip_id,
        "match_id": clip.provenance["match_id"],
        "venue_id": clip.provenance["venue_id"],
        "split": clip.split,
        "frame_index": index,
        "timestamp_seconds": index / clip.meta.fps,
        "image": (
            f"images/{clip.split}/{clip.clip_id}/"
            f"frame_{index:06d}.{config.image_format}"
        ),
        "width": geo.width,
        "height": geo.height,
        "source_annotation": asdict(ann),
        "status": annotation_status(ann),
        "has_position": has_position,
        "position_excluded": position_excluded,
        "occluded": ann.occluded,
        "position": position,
        "bbox": box,
    }


def continuous_segments(records: list[dict]) -> list[list[dict]]:
    segments = []
    for record in records:
        if not segments or record["frame_index"] != segments[-1][-1]["frame_index"] + 1:
            segments.append([])
        segments[-1].append(record)
    return segments
