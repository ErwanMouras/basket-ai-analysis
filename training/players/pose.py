"""RTMPose-M COCO17 on existing player boxes, using a local ONNX model.

The affine crop, BGR normalization and SimCC decoding reproduce the NBA
RTMPose/rtmlib 0.0.15 path. No NBA or rtmlib package is required at runtime.
"""

import importlib.metadata
import math
import re
from pathlib import Path

import cv2
import numpy as np

from training.common.config import merge_settings
from training.common.provenance import ROOT, file_hash

MODEL_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
MODEL_SHA256 = "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c"
ORT_VERSION = "1.27.0"
KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
SKELETON = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)
DEFAULTS = {
    "enabled": False,
    "weights": "models/players/rtmpose-m.onnx",
    "device": "cpu",
    "cpu_threads": 4,
    "batch_size": 16,
    "min_detection_confidence": 0.25,
    "keypoint_min_confidence": 0.3,
    "min_bbox_area": 256.0,
    "max_players": 32,
}


def settings(overrides=None, *, score_floor=0.001):
    cfg = merge_settings(DEFAULTS, {} if overrides is None else overrides)
    if type(cfg["enabled"]) is not bool:
        raise ValueError("pose.enabled must be boolean")
    if not isinstance(cfg["device"], str) or not re.fullmatch(
        r"cpu|cuda:\d+", cfg["device"]
    ):
        raise ValueError("pose.device must be cpu or cuda:N")
    for name in ("cpu_threads", "batch_size", "max_players"):
        if type(cfg[name]) is not int or not 1 <= cfg[name] <= 256:
            raise ValueError(f"pose.{name} must be an integer in [1, 256]")
    for name in (
        "min_detection_confidence",
        "keypoint_min_confidence",
        "min_bbox_area",
    ):
        value = cfg[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"pose.{name} must be finite and nonnegative")
        if name != "min_bbox_area" and value > 1:
            raise ValueError(f"pose.{name} must be <= 1")
    if cfg["enabled"] and score_floor > cfg["min_detection_confidence"]:
        raise ValueError("score_floor must be <= pose.min_detection_confidence")
    if (
        not isinstance(cfg["weights"], str)
        or not cfg["weights"]
        or "://" in cfg["weights"]
    ):
        raise ValueError(
            "pose.weights must be a local ONNX path; run setup-players-pose first"
        )
    cfg["weights"] = str((ROOT / Path(cfg["weights"]).expanduser()).resolve())
    return cfg


def preprocess(image, bbox):
    """Return normalized CHW crop and source center/scale (width, height)."""
    box = np.asarray(bbox, dtype=np.float64)
    center = (box[:2] + box[2:]) / 2
    scale = (box[2:] - box[:2]) * 1.25
    scale = np.array([max(scale[0], scale[1] * 0.75), max(scale[1], scale[0] / 0.75)])
    half_width = scale[0] / 2
    source = np.float32(
        [center, center + [0, -half_width], center + [-half_width, -half_width]]
    )
    # Derive the third point after float32 rounding, as in the NBA affine path.
    direction = source[0] - source[1]
    source[2] = source[1] + np.array([-direction[1], direction[0]])
    target = np.float32([[96, 128], [96, 32], [0, 32]])
    matrix = cv2.getAffineTransform(source, target)
    crop = cv2.warpAffine(image, matrix, (192, 256), flags=cv2.INTER_LINEAR)
    crop = (crop - np.array([123.675, 116.28, 103.53])) / np.array(
        [58.395, 57.12, 57.375]
    )
    return (
        np.ascontiguousarray(crop.transpose(2, 0, 1), dtype=np.float32),
        center,
        scale,
    )


def decode(outputs, centers, scales):
    count = len(centers)
    if (
        len(outputs) != 2
        or outputs[0].shape != (count, 17, 384)
        or outputs[1].shape != (count, 17, 512)
    ):
        raise RuntimeError("RTMPose must return batch x 17 x 384/512 SimCC outputs")
    x, y = outputs
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("RTMPose returned non-finite outputs")
    scores = (x.max(axis=2) + y.max(axis=2)) / 2
    points = np.stack((x.argmax(axis=2), y.argmax(axis=2)), axis=2).astype(np.float32)
    points[scores <= 0] = -1
    points /= 2.0
    points = points / np.array([192, 256]) * np.asarray(scales)[:, None, :]
    points += np.asarray(centers)[:, None, :] - np.asarray(scales)[:, None, :] / 2
    return points, scores


def create_session(path, config):
    # CUDA 13 wheels link cudart at import time. Torch loads the matching
    # libraries from the players CUDA environment before ONNX Runtime imports.
    try:
        importlib.metadata.version("onnxruntime-gpu")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        import torch
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Install the updated players CPU or CUDA lock for RTMPose"
        ) from exc
    if ort.__version__ != ORT_VERSION:
        raise RuntimeError(
            f"Pose requires ONNX Runtime {ORT_VERSION}, found {ort.__version__}"
        )
    installed = []
    for name in ("onnxruntime", "onnxruntime-gpu"):
        try:
            importlib.metadata.version(name)
            installed.append(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if len(installed) != 1:
        raise RuntimeError("Install exactly one ONNX Runtime distribution: CPU or GPU")
    options = ort.SessionOptions()
    options.intra_op_num_threads = config["cpu_threads"]
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if config["device"].startswith("cuda:"):
        import torch

        device_id = int(config["device"].split(":")[1])
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            raise RuntimeError(
                "Requested pose CUDA device is unavailable; choose pose.device=cpu explicitly"
            )
        ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "Install the CUDA players lock to use CUDA pose inference"
            )
        providers = [
            ("CUDAExecutionProvider", {"device_id": device_id, "use_tf32": 0}),
            "CPUExecutionProvider",
        ]
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    if (
        config["device"].startswith("cuda:")
        and "CUDAExecutionProvider" not in session.get_providers()
    ):
        raise RuntimeError(
            "Pose CUDA initialization failed; refusing silent CPU fallback"
        )
    session.disable_fallback()
    return session


class PlayerPose:
    def __init__(self, config):
        self.config = settings(config)
        path = Path(self.config["weights"])
        if not path.is_file():
            raise FileNotFoundError(
                f"Pose model missing: {path}. Run make setup-players-pose"
            )
        digest = file_hash(path)
        if digest != MODEL_SHA256:
            raise ValueError(
                "Expected the pinned RTMPose-M COCO17 model; checkpoint SHA-256 mismatch"
            )
        self.session = create_session(path, self.config)
        self._initialize()
        self.provenance = {
            "model": "rtmpose-m",
            "format": "coco17",
            "checkpoint_sha256": digest,
            "model_url": MODEL_URL,
            "onnxruntime_version": ORT_VERSION,
            "device": self.config["device"],
            "providers": self.session.get_providers(),
            "precision": "fp32",
            "cuda_tf32": False,
            "input_size": [192, 256],
            "channel_order": "BGR",
            "padding": 1.25,
            "simcc_split_ratio": 2.0,
            "score_reduction": "mean_xy_maxima",
            "keypoint_names": list(KEYPOINT_NAMES),
            "skeleton": list(SKELETON),
            "batch_size_effective": self.batch_size,
            "config": self.config,
        }

    def _initialize(self):
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].type != "tensor(float)"
            or len(inputs[0].shape) != 4
            or inputs[0].shape[1:] != [3, 256, 192]
            or [o.name for o in outputs] != ["simcc_x", "simcc_y"]
        ):
            raise ValueError("Unsupported RTMPose ONNX input/output signature")
        batch = inputs[0].shape[0]
        if isinstance(batch, int) and batch != 1:
            raise ValueError(
                "Only dynamic batch or fixed batch=1 pose models are supported"
            )
        self.batch_size = 1 if batch == 1 else self.config["batch_size"]
        self.input_name = inputs[0].name

    def predict(self, image, detections):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Pose requires a uint8 BGR image")
        height, width = image.shape[:2]
        result, candidates = [], []
        for index, d in enumerate(detections):
            box, score = d["bbox"], d["confidence"]
            if (
                len(box) != 4
                or any(
                    type(v) not in (int, float) or not math.isfinite(v)
                    for v in [*box, score]
                )
                or d["class_id"] != 0
                or not 0 <= score <= 1
                or not (
                    0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
                )
            ):
                raise ValueError("Pose requires valid source-space player boxes")
            area = (box[2] - box[0]) * (box[3] - box[1])
            reason = (
                "low_detection_confidence"
                if score < self.config["min_detection_confidence"]
                else "small_bbox"
                if area < self.config["min_bbox_area"]
                else "player_limit"
            )
            result.append(
                {
                    **d,
                    "bbox": list(box),
                    "track_id": d.get("track_id"),
                    "detection_index": index,
                    "pose": None,
                    "pose_status": reason,
                }
            )
            if reason == "player_limit":
                candidates.append((index, area))
        selected = sorted(
            sorted(candidates, key=lambda row: (-row[1], row[0]))[
                : self.config["max_players"]
            ]
        )
        for start in range(0, len(selected), self.batch_size):
            chunk = selected[start : start + self.batch_size]
            prepared = [preprocess(image, detections[i]["bbox"]) for i, _ in chunk]
            batch = np.stack([p[0] for p in prepared])
            outputs = self.session.run(["simcc_x", "simcc_y"], {self.input_name: batch})
            points, scores = decode(
                outputs, [p[1] for p in prepared], [p[2] for p in prepared]
            )
            for row, (index, _) in enumerate(chunk):
                coords, confidence = points[row], scores[row]
                in_frame = (
                    (coords[:, 0] >= 0)
                    & (coords[:, 0] < width)
                    & (coords[:, 1] >= 0)
                    & (coords[:, 1] < height)
                )
                valid = (
                    in_frame
                    & (confidence > 0)
                    & (confidence >= self.config["keypoint_min_confidence"])
                )
                result[index]["pose"] = {
                    "format": "coco17",
                    "coordinate_space": "source",
                    "keypoints": [
                        point.tolist() if s > 0 else None
                        for point, s in zip(coords, confidence)
                    ],
                    "scores": confidence.astype(float).tolist(),
                    "valid": valid.tolist(),
                    "valid_keypoints": int(valid.sum()),
                    "score": float(confidence.mean()),
                }
                result[index]["pose_status"] = (
                    "estimated" if valid.any() else "low_keypoint_confidence"
                )
        return result


def draw_pose(image, pose, color):
    if pose is None:
        return
    points = [
        tuple(round(v) for v in p) if good and p is not None else None
        for p, good in zip(pose["keypoints"], pose["valid"])
    ]
    for a, b in SKELETON:
        if points[a] is not None and points[b] is not None:
            cv2.line(image, points[a], points[b], color, 2, cv2.LINE_AA)
    for point in points:
        if point is not None:
            cv2.circle(image, point, 3, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(image, point, 2, color, -1, cv2.LINE_AA)
