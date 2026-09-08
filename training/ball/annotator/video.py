"""Bounded frame cache. A failed decode must never return a different frame."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import cv2

from .model import VideoMeta


class VideoReader:
    def __init__(self, path: Path) -> None:
        path = path.expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Expected a video file: {path}")
        self.cap = cv2.VideoCapture(str(path))
        self._cache = OrderedDict()
        self._last_index = -1
        try:
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {path}")
            fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            count = float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if (
                not math.isfinite(fps)
                or fps <= 0
                or not math.isfinite(count)
                or count < 1
            ):
                raise ValueError("Video has invalid FPS or frame count metadata")
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise ValueError("Cannot decode the first video frame")
            height, width = frame.shape[:2]
            self.meta = VideoMeta(str(path), width, height, fps, int(count))
            self._last_index = 0
            self._cache[0] = frame
            self.playable_frame_count = self._find_playable_frame_count()
        except Exception:
            self.close()
            raise

    def _find_playable_frame_count(self) -> int:
        """Return the number of frames OpenCV can actually decode.

        Some containers report one or more trailing frames in their metadata even
        though OpenCV cannot decode them. Keep the original metadata for sidecar
        compatibility, but do not expose those indices to the interface.
        """
        last_index = self.meta.frame_count - 1
        while last_index > 0:
            try:
                self.read(last_index)
            except ValueError:
                last_index -= 1
            else:
                break
        return last_index + 1

    def read(self, frame_idx: int):
        if not 0 <= frame_idx < self.meta.frame_count:
            raise ValueError(f"Frame index outside video: {frame_idx}")
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            return self._cache[frame_idx]
        if frame_idx != self._last_index + 1:
            if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
                raise ValueError(f"Cannot seek to frame {frame_idx}")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._last_index = -1
            raise ValueError(
                f"Frame {frame_idx} is not decodable. No annotation was written for it."
            )
        if frame.shape[:2] != (self.meta.height, self.meta.width):
            self._last_index = -1
            raise ValueError("Video dimensions changed during decoding")
        self._last_index = frame_idx
        self._cache[frame_idx] = frame
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return frame

    def close(self) -> None:
        self.cap.release()
