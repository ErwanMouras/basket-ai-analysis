"""Single-owner decoder; bounded prefetch is managed by the runner."""

from dataclasses import dataclass
from pathlib import Path
import math

import cv2

from training.common.provenance import file_hash


@dataclass
class Frame:
    index: int
    timestamp: float
    image: object


class Video:
    def __init__(self, path, timestamp_policy="source"):
        path = Path(path).resolve(strict=True)
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.close()
            raise ValueError("Video requires valid nominal FPS for tracking")
        self.policy = timestamp_policy
        self.source = {"path": str(path), "sha256": file_hash(path), "nominal_fps": self.fps,
                       "declared_frames": int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                       "timestamp_policy": timestamp_policy, "decoded_frames": 0}

    def frames(self, limit=None):
        previous = -1.0
        shape = None
        index = 0
        while limit is None or index < limit:
            ok, image = self.capture.read()
            if not ok:
                break
            timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 if self.policy == "source" else index / self.fps
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
                raise ValueError("Video timestamps must strictly increase; use timestamp_policy=fps only for known CFR input")
            if shape is not None and shape != image.shape:
                raise ValueError("Changing frame dimensions are unsupported")
            shape = image.shape
            self.source.update(width=shape[1], height=shape[0], decoded_frames=index + 1)
            yield Frame(index, timestamp, image)
            previous = timestamp
            index += 1
        if index == 0:
            raise ValueError("Video has no decoded frames")

    def close(self):
        self.capture.release()
