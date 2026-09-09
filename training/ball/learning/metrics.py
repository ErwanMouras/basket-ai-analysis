"""Heatmap localization metrics in network-input pixels, per window-frame."""

import cv2
import numpy as np


def center(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    x, y, w, h = max(
        (cv2.boundingRect(contour) for contour in contours),
        key=lambda box: box[2] * box[3],
    )
    return np.array([x + (w - 1) / 2, y + (h - 1) / 2])


def update_counts(counts, predictions, targets, threshold, tolerance):
    for predicted, target in zip(
        predictions.reshape(-1, *predictions.shape[-2:]),
        targets.reshape(-1, *targets.shape[-2:]),
    ):
        point, truth = center(predicted > threshold), center(target > 0)
        if truth is None:
            counts["tn" if point is None else "fp"] += 1
        elif point is None:
            counts["fn"] += 1
        else:
            distance = float(np.linalg.norm(point - truth))
            counts["distance_sum"] += distance
            counts["localized_pairs"] += 1
            if distance <= tolerance:
                counts["tp"] += 1
            else:
                # A displaced detection is both a false detection and a missed target.
                counts["fp"] += 1
                counts["fn"] += 1
                counts["displaced"] += 1
        counts["window_frames"] += 1


def metric_values(counts):
    precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    result = {
        key: counts[key]
        for key in (
            "tp",
            "tn",
            "fp",
            "fn",
            "displaced",
            "localized_pairs",
            "window_frames",
        )
    }
    result.update(
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(1e-12, precision + recall),
    )
    if counts["localized_pairs"]:
        result["center_error_px"] = counts["distance_sum"] / counts["localized_pairs"]
    return result
