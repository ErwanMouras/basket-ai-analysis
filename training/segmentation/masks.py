"""Lossless row-major RLE limited to a source-pixel ROI, never full-frame JSON."""
import math
import numpy as np


def encode(mask, bbox):
    x1, y1, x2, y2 = bbox
    h, w = mask.shape
    roi = [max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2))]
    a, b, c, d = roi
    if c <= a or d <= b:
        raise ValueError("Empty mask ROI")
    values = np.asarray(mask[b:d, a:c], dtype=bool).reshape(-1)
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    counts = np.diff(np.r_[0, changes, len(values)]).tolist()
    if values[0]:
        counts.insert(0, 0)
    return {"encoding": "roi_rle_c", "roi": roi, "counts": counts}


def validate(payload, width, height):
    if not isinstance(payload, dict) or payload.get("encoding") != "roi_rle_c":
        raise ValueError("Unsupported mask encoding")
    roi, counts = payload.get("roi"), payload.get("counts")
    if not isinstance(roi, list) or len(roi) != 4 or any(type(x) is not int for x in roi):
        raise ValueError("Invalid mask ROI")
    a, b, c, d = roi
    if not (0 <= a < c <= width and 0 <= b < d <= height):
        raise ValueError("Mask ROI outside source image")
    if not isinstance(counts, list) or not counts or any(type(x) is not int or x < 0 for x in counts) or sum(counts) != (c-a)*(d-b):
        raise ValueError("Invalid RLE run lengths")


def decode(payload):
    a, b, c, d = payload["roi"]
    validate(payload, c, d)
    counts = payload["counts"]
    return np.repeat(np.arange(len(counts)) % 2, counts).astype(bool).reshape(d-b, c-a)


def crop_mask(payload, bbox):
    """Decode only the compact ROI, return a crop in the requested source region."""
    a, b, c, d = bbox
    out = np.zeros((d-b, c-a), dtype=bool)
    x1, y1, x2, y2 = payload["roi"]
    l, t, r, bottom = max(a, x1), max(b, y1), min(c, x2), min(d, y2)
    if r > l and bottom > t:
        out[t-b:bottom-b, l-a:r-a] = decode(payload)[t-y1:bottom-y1, l-x1:r-x1]
    return out
