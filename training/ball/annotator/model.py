"""NBA-compatible annotations in source pixels, with atomic persistence."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class Annotation:
    cx: float | None = None
    cy: float | None = None
    radius: float | None = None
    occluded: bool = False

    @property
    def has_position(self) -> bool:
        return self.cx is not None

    def validate(self, meta: VideoMeta) -> None:
        if type(self.occluded) is not bool:
            raise ValueError("occluded must be a boolean")
        coords = (self.cx, self.cy, self.radius)
        if all(value is None for value in coords):
            if not self.occluded:
                raise ValueError("An annotation without a position must be occluded")
            return
        if not all(is_number(value) for value in coords):
            raise ValueError(
                "cx, cy and radius must all be finite numbers, or all null"
            )
        if not (0 <= self.cx < meta.width and 0 <= self.cy < meta.height):
            raise ValueError("Ball position is outside the source image")
        if self.radius <= 0:
            raise ValueError("Ball radius must be positive")


def is_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def sidecar_path(video: Path) -> Path:
    return video.with_name(video.name + ".ballann.json")


class Store:
    def __init__(self, meta: VideoMeta) -> None:
        self.meta = meta
        self.path = sidecar_path(Path(meta.path))
        self.annotations: dict[int, Annotation] = {}
        self.default_radius = 14.0
        self._payload: dict = {}
        # Remember the bytes read, to detect another annotator editing this file.
        self._saved_bytes = self.path.read_bytes() if self.path.exists() else None
        if self._saved_bytes is not None:
            self._load(json.loads(self._saved_bytes))

    def _load(self, payload: object) -> None:
        if (
            not isinstance(payload, dict)
            or type(payload.get("schema_version")) is not int
        ):
            raise ValueError("Invalid annotation document: expected schema_version 1")
        if payload["schema_version"] != 1:
            raise ValueError("Unsupported annotation schema version")
        video = payload.get("video")
        if not isinstance(video, dict):
            raise ValueError("Missing video metadata")
        for field in ("width", "height", "frame_count"):
            if type(video.get(field)) is not int or video[field] != getattr(
                self.meta, field
            ):
                raise ValueError(f"Annotations do not match the video: {field}")
        if not is_number(video.get("fps")) or not math.isclose(
            video["fps"], self.meta.fps, rel_tol=0.001, abs_tol=0.001
        ):
            raise ValueError("Annotations do not match the video: fps")
        # Keeping the sidecar next to a video with the same basename permits moving
        # or copying the pair. Metadata is a compatibility check, not a content hash.
        stored_path = video.get("path")
        if (
            not isinstance(stored_path, str)
            or stored_path.replace("\\", "/").split("/")[-1]
            != Path(self.meta.path).name
        ):
            raise ValueError("Annotations do not match the video filename")
        radius = payload.get("default_radius", 14.0)
        if not is_number(radius) or radius <= 0:
            raise ValueError("default_radius must be a positive finite number")
        entries = payload.get("annotations")
        if not isinstance(entries, dict):
            raise ValueError("annotations must be an object indexed by frame number")
        for key, raw in entries.items():
            if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
                raise ValueError(f"Invalid frame index: {key!r}")
            frame_idx = int(key)
            self._validate_index(frame_idx)
            if not isinstance(raw, dict) or set(raw) - {
                "cx",
                "cy",
                "radius",
                "occluded",
            }:
                raise ValueError(f"Invalid annotation for frame {frame_idx}")
            ann = Annotation(**raw)
            ann.validate(self.meta)
            self.annotations[frame_idx] = ann
        self.default_radius = float(radius)
        self._payload = payload

    def _validate_index(self, frame_idx: int) -> None:
        if type(frame_idx) is not int or not 0 <= frame_idx < self.meta.frame_count:
            raise ValueError(f"Frame index outside the video: {frame_idx}")

    def update(self, frame_idx: int, annotation: Annotation | None) -> None:
        self._validate_index(frame_idx)
        updated = dict(self.annotations)
        if annotation is None:
            updated.pop(frame_idx, None)
        else:
            annotation.validate(self.meta)
            updated[frame_idx] = annotation
        self._save(updated, self.default_radius)

    def adjust_radius(self, frame_idx: int, delta: int) -> None:
        self._validate_index(frame_idx)
        ann = self.annotations.get(frame_idx)
        current = ann.radius if ann and ann.has_position else self.default_radius
        radius = max(
            1.0, min(float(max(self.meta.width, self.meta.height)), current + delta)
        )
        updated = dict(self.annotations)
        # Never assign a radius to an annotation whose position is unknown.
        if ann and ann.has_position:
            updated[frame_idx] = replace(ann, radius=radius)
        self._save(updated, radius)

    def _save(self, annotations: dict[int, Annotation], radius: float) -> None:
        existing = self.path.read_bytes() if self.path.exists() else None
        if existing != self._saved_bytes:
            raise ValueError(
                "The JSON was changed by another tool. Close and reopen this window."
            )
        payload = {
            **self._payload,
            "schema_version": 1,
            "video": {**self._payload.get("video", {}), **asdict(self.meta)},
            "default_radius": radius,
            "annotations": {
                str(i): asdict(ann) for i, ann in sorted(annotations.items())
            },
        }
        content = (
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        # Commit in-memory state only after a successful write.
        self.annotations = annotations
        self.default_radius = radius
        self._payload = payload
        self._saved_bytes = content


@dataclass(frozen=True)
class Viewport:
    """Aspect-preserving display geometry; annotations always use source pixels."""

    source_width: int
    source_height: int
    width: int
    height: int
    left: int
    top: int

    @classmethod
    def fit(
        cls, source_width: int, source_height: int, width: int, height: int
    ) -> Viewport:
        scale = min(width / source_width, height / source_height, 1.0)
        dw, dh = (
            max(1, round(source_width * scale)),
            max(1, round(source_height * scale)),
        )
        return cls(
            source_width, source_height, dw, dh, (width - dw) // 2, (height - dh) // 2
        )

    def to_source(self, x: float, y: float) -> tuple[float, float] | None:
        x, y = x - self.left, y - self.top
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        return x * self.source_width / self.width, y * self.source_height / self.height

    def to_display(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.left + x * self.width / self.source_width,
            self.top + y * self.height / self.source_height,
        )
