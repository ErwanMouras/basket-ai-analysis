"""Detector-independent, online player tracking with explicitly pinned backends.

IDs belong to one video run, not to a roster or a persistent player identity.
Only confirmed tracks matched to a current detection receive an ID.
"""

import importlib.metadata
import math
from types import SimpleNamespace

import numpy as np

from training.common.config import merge_settings

DEFAULTS = {
    "enabled": False,
    "tracker": "botsort",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer_seconds": 1.0,
    "match_thresh": 0.8,
    "fuse_score": False,
    "gmc_method": "sparseOptFlow",
    "reset_frames": [],
}
ULTRALYTICS_VERSION = "8.4.39"
LAP_VERSION = "0.5.12"


def settings(overrides=None, *, score_floor=0.001):
    config = merge_settings(DEFAULTS, {} if overrides is None else overrides)
    for key in ("enabled", "fuse_score"):
        if type(config[key]) is not bool:
            raise ValueError(f"tracking.{key} must be boolean")
    if config["tracker"] not in ("botsort", "bytetrack"):
        raise ValueError("tracking.tracker must be botsort or bytetrack")
    for key in (
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "match_thresh",
    ):
        value = config[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value <= 1
        ):
            raise ValueError(f"tracking.{key} must be finite and in (0, 1]")
    if (
        not config["track_low_thresh"]
        < config["track_high_thresh"]
        <= config["new_track_thresh"]
    ):
        raise ValueError(
            "Require track_low_thresh < track_high_thresh <= new_track_thresh"
        )
    value = config["track_buffer_seconds"]
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("tracking.track_buffer_seconds must be finite and positive")
    if config["gmc_method"] not in ("sparseOptFlow", "none"):
        raise ValueError("tracking.gmc_method must be sparseOptFlow or none")
    frames = config["reset_frames"]
    if (
        not isinstance(frames, list)
        or any(type(f) is not int or f < 1 for f in frames)
        or frames != sorted(set(frames))
    ):
        raise ValueError(
            "tracking.reset_frames must be distinct increasing positive frame indices"
        )
    config["reset_frames"] = list(frames)
    if config["enabled"] and score_floor > config["track_low_thresh"]:
        raise ValueError(
            "score_floor must be <= tracking.track_low_thresh to retain low-score associations"
        )
    return config


class _Detections:
    """Carry original detection indices through the backend's confidence filters."""

    def __init__(self, data, indices=None):
        self.data = data
        self.indices = np.arange(len(data)) if indices is None else indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, selection):
        return _Detections(self.data[selection], self.indices[selection])

    @property
    def xyxy(self):
        return self.data[:, :4]

    @property
    def xywh(self):
        result = self.xyxy.copy()
        result[:, :2] = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        result[:, 2:] -= self.xyxy[:, :2]
        return result

    @property
    def conf(self):
        return self.data[:, 4]

    @property
    def cls(self):
        return self.data[:, 5]


class PlayerTracker:
    def __init__(self, config, *, fps):
        self.config = settings(config)
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Tracking requires a finite positive FPS")
        # Check before importing Ultralytics trackers: their matching module would
        # otherwise attempt to install lap automatically during inference.
        for package, expected in (
            ("ultralytics", ULTRALYTICS_VERSION),
            ("lap", LAP_VERSION),
        ):
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"Tracking needs {package}=={expected}; install the players lock"
                ) from exc
            if version != expected:
                raise RuntimeError(
                    f"Tracking requires {package}=={expected}, found {version}"
                )
        import lap

        if not getattr(lap, "__version__", None):
            raise RuntimeError(
                "lap installation is incomplete; reinstall the players lock"
            )
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._counter = 0
        self._next_frame = 0
        self._shape = None
        self.segment_id = 0
        self._resets = iter(self.config["reset_frames"])
        self._next_reset = next(self._resets, None)
        self.buffer_frames = max(
            1, math.ceil(fps * self.config["track_buffer_seconds"])
        )
        owner = self
        backend = BOTSORT if self.config["tracker"] == "botsort" else BYTETracker

        class LocalTracker(backend):
            @staticmethod
            def reset_id():
                # Upstream uses a process-global counter. This adapter allocates
                # per-run IDs, including when several videos are interleaved.
                pass

            def init_track(self, results, img=None):
                tracks = super().init_track(results, img)
                for track in tracks:
                    track.idx = int(results.indices[int(track.idx)])
                    track.next_id = owner._allocate_id
                return tracks

        args = SimpleNamespace(
            **self.config,
            track_buffer=self.buffer_frames,
            with_reid=False,
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            model="auto",
        )
        self._backend = LocalTracker(args, frame_rate=fps)
        self.provenance = {
            "backend": f"ultralytics.{self.config['tracker']}",
            "ultralytics_version": ULTRALYTICS_VERSION,
            "lap_version": LAP_VERSION,
            "with_reid": False,
            "gmc_method": self.config["gmc_method"]
            if self.config["tracker"] == "botsort"
            else "none",
            "fps": fps,
            "track_buffer_frames": self.buffer_frames,
            "id_scope": "tracking_run_id",
            "bbox_source": "current_detection",
            "unconfirmed_track_id": None,
            "config": self.config,
        }

    def _allocate_id(self):
        self._counter += 1
        return self._counter

    def update(self, detections, image, *, frame_index):
        if type(frame_index) is not int or frame_index != self._next_frame:
            raise ValueError(
                "Tracking requires consecutive frame indices starting at zero, including empty frames"
            )
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Tracking requires uint8 BGR images")
        if self._shape is not None and image.shape != self._shape:
            raise ValueError("Tracking image dimensions changed")
        height, width = image.shape[:2]
        rows = []
        for detection in detections:
            box, score = detection["bbox"], detection["confidence"]
            if (
                detection["class_id"] != 0
                or len(box) != 4
                or any(
                    type(v) not in (int, float) or not math.isfinite(v)
                    for v in [*box, score]
                )
                or not 0 <= score <= 1
                or not (
                    0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
                )
            ):
                raise ValueError("Invalid source-space player detection")
            rows.append([*box, score, 0])
        batch = _Detections(np.asarray(rows, dtype=np.float32).reshape(-1, 6))
        if frame_index == self._next_reset:
            self._backend.reset()
            self.segment_id += 1
            self._next_reset = next(self._resets, None)
        # Upstream ages lost tracks after matching. Expire them first to prevent
        # an already expired ID from being revived on the very next detection.
        next_backend_frame = self._backend.frame_id + 1
        self._backend.lost_stracks = [
            t
            for t in self._backend.lost_stracks
            if next_backend_frame - t.end_frame <= self.buffer_frames
        ]
        tracks = self._backend.update(batch, image)
        output = [{**d, "bbox": list(d["bbox"]), "track_id": None} for d in detections]
        seen_ids, seen_indices = set(), set()
        for row in tracks:
            if len(row) != 8 or not np.isfinite(row).all():
                raise RuntimeError(
                    "Unexpected tracker output; check the pinned backend"
                )
            index, track_id = int(row[7]), int(row[4])
            if (
                index != row[7]
                or track_id != row[4]
                or not 0 <= index < len(output)
                or track_id < 1
                or track_id in seen_ids
                or index in seen_indices
            ):
                raise RuntimeError("Invalid or duplicated tracking association")
            output[index]["track_id"] = track_id
            seen_ids.add(track_id)
            seen_indices.add(index)
        self._shape = image.shape
        self._next_frame += 1
        return output
