"""Local image/video access with a bounded cache and source identity checks."""

import math
from collections import OrderedDict
from pathlib import Path

import cv2

from training.common.config import SPLITS, read_yaml
from training.common.provenance import file_hash, object_hash
from training.common.sources import VIDEO_EXTENSIONS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class MediaReader:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(strict=True)
        stat = self.path.stat()
        self.signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        self.cap = None
        self.cache = OrderedDict()
        self.last = -1
        try:
            if self.path.suffix.lower() in IMAGE_EXTENSIONS:
                frame = cv2.imread(str(self.path))
                self.kind, self.frame_count, self.fps = "image", 1, None
            elif self.path.suffix.lower() in VIDEO_EXTENSIONS:
                self.cap = cv2.VideoCapture(str(self.path))
                if not self.cap.isOpened():
                    raise ValueError(f"Cannot open {self.path}")
                self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
                count = float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if (
                    not math.isfinite(count)
                    or count < 1
                    or not math.isfinite(self.fps)
                    or self.fps <= 0
                ):
                    raise ValueError("Invalid video metadata")
                self.kind, self.frame_count = "video", int(count)
                ok, frame = self.cap.read()
                if not ok:
                    frame = None
            else:
                raise ValueError(f"Unsupported media extension: {self.path.suffix}")
            if frame is None:
                raise ValueError(f"Cannot decode {self.path}")
            self.height, self.width = frame.shape[:2]
            self.cache[0], self.last = frame, 0
        except BaseException:
            self.close()
            raise

    def read(self, index):
        if type(index) is not int or not 0 <= index < self.frame_count:
            raise ValueError("Frame index outside media")
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        if index != self.last + 1 and not self.cap.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError(f"Cannot seek to frame {index}")
        ok, frame = self.cap.read()
        if not ok or frame is None or frame.shape[:2] != (self.height, self.width):
            self.last = -1
            raise ValueError(f"Cannot decode frame {index}; active frame is unchanged")
        self.last = index
        self.cache[index] = frame
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()


def resolve_identity(media, root, *, match_id=None, venue_id=None, split=None):
    """Use explicit arguments and existing metadata; conflicting identities fail."""
    media, root = Path(media).resolve(strict=True), Path(root).resolve(strict=True)
    relative = media.relative_to(root)
    documents = [{"match_id": match_id, "venue_id": venue_id, "split": split}]
    metadata = media.with_name(media.name + ".meta.yaml")
    if metadata.exists():
        if not metadata.resolve().is_relative_to(root):
            raise ValueError("Metadata escapes source root")
        documents.append(read_yaml(metadata))
    for parent in media.parents:
        if not parent.is_relative_to(root):
            break
        candidate = parent / "match.yaml"
        if candidate.exists():
            if not candidate.resolve().is_relative_to(root):
                raise ValueError("Match metadata escapes source root")
            documents.append(read_yaml(candidate))
            break
    if relative.parts[0] in SPLITS:
        documents.append({"split": relative.parts[0]})
    identity = {}
    for key in ("match_id", "venue_id", "split"):
        values = [d[key] for d in documents if d.get(key) is not None]
        if any(
            not isinstance(v, str) or not v.strip() or v != v.strip() for v in values
        ):
            raise ValueError(f"Invalid {key}")
        if len(set(values)) > 1:
            raise ValueError(f"Conflicting {key}: {values}")
        identity[key] = values[0] if values else None
    if identity["match_id"] is None:
        raise ValueError("Provide --match-id or match_id in .meta.yaml / match.yaml")
    return identity


def source_record(reader, root, **identity):
    path = reader.path.relative_to(Path(root).resolve(strict=True)).as_posix()
    digest = file_hash(reader.path)
    return {
        "source_id": object_hash({"path": path, "sha256": digest})[:24],
        "kind": reader.kind,
        "path": path,
        "sha256": digest,
        "width": reader.width,
        "height": reader.height,
        "frame_count": reader.frame_count,
        "fps": reader.fps,
        **identity,
    }


def selected_frames(count, *, start=0, stop=None, step=1):
    stop = count if stop is None else stop
    if (
        any(type(v) is not int for v in (start, stop, step))
        or not 0 <= start < stop <= count
        or step < 1
    ):
        raise ValueError("Require 0 <= start < stop <= frame_count and step >= 1")
    return range(start, stop, step)
