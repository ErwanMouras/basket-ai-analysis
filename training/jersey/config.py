"""Explicit budgets for causal, bounded jersey recognition."""

import math
import re
from pathlib import Path

from training.common.config import merge_settings
from training.common.provenance import ROOT

MODEL_URL = (
    "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt"
)
MODEL_SHA256 = "bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00"
DEFAULTS = {
    "enabled": False,
    "weights": "models/jersey/parseq.pt",
    "device": "cpu",
    "cpu_threads": 2,
    "batch_size": 4,
    "crops_per_video_second": 8.0,
    "max_duty_cycle": 0.15,
    "min_batch_interval_seconds": 0.1,
    "gpu_reserve_mb": 1024,
    "sample_interval_seconds": 0.1,
    "selection_window_seconds": 0.3,
    "read_interval_seconds": 0.75,
    "confirmed_interval_seconds": 3.0,
    "evidence_ttl_seconds": 12.0,
    "state_ttl_seconds": 10.0,
    "max_tracks": 64,
    "max_evidence": 8,
    "min_votes": 2,
    "min_agreement": 0.75,
    "min_ocr_confidence": 0.9,
    "min_detection_confidence": 0.4,
    "min_crop_width": 20,
    "min_crop_height": 24,
    "min_sharpness": 25.0,
    "min_contrast": 12.0,
    "max_overlap": 0.4,
}


def settings(overrides=None):
    cfg = merge_settings(DEFAULTS, {} if overrides is None else overrides)
    if type(cfg["enabled"]) is not bool:
        raise ValueError("jersey.enabled must be boolean")
    if not isinstance(cfg["device"], str) or not re.fullmatch(
        r"cpu|cuda:\d+", cfg["device"]
    ):
        raise ValueError("jersey.device must be cpu or cuda:N")
    for name in (
        "cpu_threads",
        "batch_size",
        "gpu_reserve_mb",
        "max_tracks",
        "max_evidence",
        "min_votes",
        "min_crop_width",
        "min_crop_height",
    ):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f"jersey.{name} must be a positive integer")
    if cfg["batch_size"] > 4 or cfg["max_tracks"] > 256 or cfg["max_evidence"] > 32:
        raise ValueError(
            "Jersey budgets: batch_size <= 4, max_tracks <= 256, max_evidence <= 32"
        )
    if cfg["min_votes"] < 2 or cfg["min_votes"] > cfg["max_evidence"]:
        raise ValueError("Require 2 <= min_votes <= max_evidence")
    for name, default in DEFAULTS.items():
        if type(default) is float:
            v = cfg[name]
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"jersey.{name} must be positive and finite")
    for name in (
        "max_duty_cycle",
        "min_agreement",
        "min_ocr_confidence",
        "min_detection_confidence",
        "max_overlap",
    ):
        if cfg[name] > 1:
            raise ValueError(f"jersey.{name} must be <= 1")
    if (
        cfg["selection_window_seconds"] > cfg["read_interval_seconds"]
        or cfg["sample_interval_seconds"] > cfg["read_interval_seconds"]
    ):
        raise ValueError(
            "Sampling and selection windows must not exceed the read interval"
        )
    if cfg["min_agreement"] <= 0.5:
        raise ValueError("jersey.min_agreement must exceed 0.5")
    if cfg["confirmed_interval_seconds"] < cfg["read_interval_seconds"]:
        raise ValueError("Confirmed tracks must be read less frequently")
    if cfg["evidence_ttl_seconds"] <= cfg["confirmed_interval_seconds"]:
        raise ValueError("Evidence TTL must exceed the recheck interval")
    if (
        not isinstance(cfg["weights"], str)
        or not cfg["weights"]
        or "://" in cfg["weights"]
    ):
        raise ValueError("jersey.weights must be a local checkpoint; run setup-jersey")
    cfg["weights"] = str((ROOT / Path(cfg["weights"]).expanduser()).resolve())
    return cfg


def number(text):
    """No stripping, digit concatenation, truncation or loss of leading zeros."""
    return text if isinstance(text, str) and re.fullmatch(r"[0-9]{1,2}", text) else None
