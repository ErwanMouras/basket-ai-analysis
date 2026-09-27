"""CPU torso crops and inexpensive quality gates; only tiny RGB crops persist."""

import cv2
import numpy as np
from PIL import Image


def candidate(image, detection, others, config):
    x1, y1, x2, y2 = detection["bbox"]
    w, h = x2 - x1, y2 - y1
    box = [x1 + 0.22 * w, y1 + 0.20 * h, x1 + 0.78 * w, y1 + 0.56 * h]
    region = "center_chest"
    pose = detection.get("pose")
    if pose and all(pose["valid"][i] for i in (5, 6, 11, 12)):
        pts = np.array([pose["keypoints"][i] for i in (5, 6, 11, 12)])
        left, top = pts.min(axis=0)
        right, bottom = pts.max(axis=0)
        # Reject profile views and inconsistent geometry; avoid borrowing a neighbour's torso.
        if right - left < 0.15 * w or bottom - top < 0.12 * h:
            return None
        if not (x1 <= left < right <= x2 and y1 <= top < bottom <= y2):
            return None
        box = [
            left + 0.10 * (right - left),
            top + 0.12 * (bottom - top),
            right - 0.10 * (right - left),
            bottom - 0.08 * (bottom - top),
        ]
        region = "pose_torso"
    ih, iw = image.shape[:2]
    a, b, c, d = [round(v) for v in box]
    a, b, c, d = max(0, a), max(0, b), min(iw, c), min(ih, d)
    if c - a < config["min_crop_width"] or d - b < config["min_crop_height"]:
        return None
    area = (c - a) * (d - b)
    overlap = 0.0
    for other in others:
        if (
            other is detection
            or other["confidence"] < config["min_detection_confidence"]
        ):
            continue
        ox1, oy1, ox2, oy2 = other["bbox"]
        overlap = max(
            overlap,
            max(0, min(c, ox2) - max(a, ox1))
            * max(0, min(d, oy2) - max(b, oy1))
            / area,
        )
    if overlap > config["max_overlap"]:
        return None
    crop = image[b:d, a:c]
    # Bound quality work even on 4K closeups.
    scale = min(1, 160 / max(crop.shape[:2]))
    gray = cv2.cvtColor(cv2.resize(crop, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    sharpness, contrast = (
        float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        float(gray.std()),
    )
    if sharpness < config["min_sharpness"] or contrast < config["min_contrast"]:
        return None
    quality = (
        0.4 * min(1, sharpness / 120)
        + 0.3 * min(1, contrast / 48)
        + 0.3 * min(1, area**0.5 / 96)
    ) * (1 - overlap)
    rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize(
        (128, 32), Image.Resampling.BICUBIC
    )
    return {
        "image": np.asarray(rgb).copy(),
        "bbox": [a, b, c, d],
        "region": region,
        "quality": quality,
        "sharpness": sharpness,
        "contrast": contrast,
        "overlap": overlap,
    }
