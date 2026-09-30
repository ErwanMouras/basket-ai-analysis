"""CPU torso crops and inexpensive quality gates; only tiny RGB crops persist."""

import cv2
import numpy as np
from PIL import Image


def candidate(image, detection, others, config, *, variant="lower"):
    if variant not in ("lower", "upper", "center"):
        raise ValueError("Unknown number crop variant")
    x1, y1, x2, y2 = detection["bbox"]
    w, h = x2 - x1, y2 - y1
    color_box = [x1 + 0.22 * w, y1 + 0.20 * h, x1 + 0.78 * w, y1 + 0.56 * h]
    box = ([x1 + 0.22 * w, y1 + 0.20 * h, x1 + 0.78 * w, y1 + 0.56 * h]
           if variant in ("upper", "center") else
           [x1 + 0.24 * w, y1 + 0.30 * h, x1 + 0.76 * w, y1 + 0.53 * h])
    region = "bbox_number_upper" if variant in ("upper", "center") else "bbox_number"
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
        color_box = [
            left + 0.10 * (right - left),
            top + 0.12 * (bottom - top),
            right - 0.10 * (right - left),
            bottom - 0.08 * (bottom - top),
        ]
        number_box = [
            left + 0.10 * (right - left),
            top + 0.48 * (bottom - top),
            right - 0.10 * (right - left),
            bottom + 0.12 * (bottom - top),
        ]
        if variant == "center":
            # Keep the printed digits; the full torso also contains sponsor text
            # and arms, which dilute confidence after resizing to 128 x 32.
            box = [left + 0.05 * (right - left), top + 0.37 * (bottom - top),
                   right - 0.05 * (right - left), top + 0.78 * (bottom - top)]
            region = "pose_number_center"
        elif variant == "upper":
            box, region = color_box, "pose_number_upper"
        else:
            box = number_box if (number_box[2] - number_box[0] >= config["min_crop_width"]
                                 and number_box[3] - number_box[1] >= config["min_crop_height"]) else color_box
            region = "pose_number" if box is number_box else "pose_number_upper"
    elif variant == "center":
        return None
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
    ca, cb, cc, cd = [round(v) for v in color_box]
    ca, cb, cc, cd = max(0, ca), max(0, cb), min(iw, cc), min(ih, cd)
    torso = image[cb:cd, ca:cc]
    color_rgb = Image.fromarray(cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)).resize(
        (128, 32), Image.Resampling.BICUBIC
    )
    return {
        "image": np.asarray(rgb).copy(),
        "color_image": np.asarray(color_rgb).copy(),
        "bbox": [a, b, c, d],
        "region": region,
        "quality": quality,
        "sharpness": sharpness,
        "contrast": contrast,
        "overlap": overlap,
    }
