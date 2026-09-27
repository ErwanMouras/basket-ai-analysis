"""Strict, bounded court inference configuration."""

import math
import re
from pathlib import Path

from training.common.config import merge_settings
from training.common.provenance import ROOT

DEFAULTS = {
    "enabled": False,
    "weights": "models/court/court_keypoints.pt",
    "device": "cpu",
    "resolution": 640,
    "detect_every": 5,
    "keypoint_confidence": 0.5,
    "min_inliers": 5,
    "min_inlier_ratio": 0.6,
    "ransac_pixels": 8.0,  # at 1080p, scaled with image height
    "max_reprojection_pixels": 6.0,
    "min_support_area_m2": 8.0,
    "max_propagation_seconds": 0.5,
    "max_temporal_jump_m": 1.5,
    "motion_width": 960,
    "motion_min_inliers": 12,
    "cut_pixel_difference": 0.30,
    "cut_histogram_distance": 0.55,
    "auto_cuts": True,
    "min_player_confidence": 0.25,
    "footpoint": "bbox",  # ankles are opt-in: they are not shoe-floor contacts
    "max_extrapolation_m": 4.0,
    "court_margin_m": 0.5,
    "overlay": True,
    "minimap": True,
    "excluded_regions": [[0.0, 0.87, 1.0, 1.0], [0.72, 0.0, 1.0, 0.18]],
}


def settings(overrides=None):
    cfg = merge_settings(DEFAULTS, overrides or {})
    for key in ("enabled", "auto_cuts", "overlay", "minimap"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"court.{key} must be boolean")
    if not isinstance(cfg["device"], str) or not re.fullmatch(r"cpu|cuda:\d+", cfg["device"]):
        raise ValueError("court.device must be cpu or cuda:N")
    for key in ("resolution", "detect_every", "min_inliers", "motion_width", "motion_min_inliers"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"court.{key} must be a positive integer")
    if not 5 <= cfg["min_inliers"] <= 18:
        raise ValueError("court.min_inliers must be in [5, 18]")
    if cfg["resolution"] % 32 or cfg["motion_width"] < 160:
        raise ValueError("court.resolution must be a multiple of 32; motion_width >= 160")
    for key in ("keypoint_confidence", "min_inlier_ratio", "ransac_pixels",
                "max_reprojection_pixels", "min_support_area_m2", "max_propagation_seconds",
                "max_temporal_jump_m", "cut_pixel_difference", "cut_histogram_distance",
                "min_player_confidence", "max_extrapolation_m", "court_margin_m"):
        value = cfg[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"court.{key} must be finite and nonnegative")
        if key in ("keypoint_confidence", "min_inlier_ratio", "cut_pixel_difference",
                   "cut_histogram_distance", "min_player_confidence") and not 0 < value <= 1:
            raise ValueError(f"court.{key} must be in (0, 1]")
    if cfg["footpoint"] not in ("bbox", "ankles"):
        raise ValueError("court.footpoint must be bbox or ankles")
    if not isinstance(cfg["weights"], str) or not cfg["weights"] or "://" in cfg["weights"]:
        raise ValueError("court.weights must be local; run make setup-court first")
    cfg["weights"] = str((ROOT / Path(cfg["weights"]).expanduser()).resolve())
    if not isinstance(cfg["excluded_regions"], list):
        raise ValueError("court.excluded_regions must be normalized rectangles")
    regions = []
    for region in cfg["excluded_regions"]:
        if (not isinstance(region, (list, tuple)) or len(region) != 4
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in region)
                or not (0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1)):
            raise ValueError("court.excluded_regions must be normalized xyxy rectangles")
        regions.append(list(region))
    cfg["excluded_regions"] = regions
    return cfg
