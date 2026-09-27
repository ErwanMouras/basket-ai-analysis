"""Strict configuration for the analysis entry point."""

import math
import re
from copy import deepcopy
from pathlib import Path

import yaml

from training.common.config import merge_settings
from training.court.config import DEFAULTS as COURT_DEFAULTS, settings as court_settings
from training.jersey.config import DEFAULTS as JERSEY_DEFAULTS, settings as jersey_settings
from training.players.pose import DEFAULTS as POSE_DEFAULTS, settings as pose_settings
from training.players.tracking import DEFAULTS as TRACKING_DEFAULTS, settings as tracking_settings

DEFAULTS = {
    "mode": "parallel", "workers": 4, "prefetch": 2, "cpu_threads": 4,
    "gpu_concurrency": 1, "max_frames": None, "timestamp_policy": "source",
    "include_intervals": [], "exclude_intervals": [],
    "required": ["players", "court"],
    "people": {"adapter": "yolo", "weights": "models/players/ebard_yolov8n.pt",
               "device": "cpu", "resolution": 640, "confidence": 0.1,
               "max_detections": 100, "class_roles": {"2": "player", "3": "referee"},
               "variant": "rfdetr_nano"},
    "ball": {"enabled": True, "adapter": "tracknet_v5", "weights": None,
             "reference_root": ".external/tracknet_v5", "device": "cpu",
             "threshold": 0.5},
    "court": {**COURT_DEFAULTS, "enabled": True, "overlay": False, "minimap": False},
    "pose": {**POSE_DEFAULTS, "enabled": True}, "jersey": {**JERSEY_DEFAULTS, "enabled": True},
    "tracking": TRACKING_DEFAULTS,
    "distance": {"smoothing_tau_s": 0.12, "deadband_m": 0.025,
                 "max_speed_m_s": 12.0, "max_gap_s": 0.2},
}


def positive(value, name, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def load_config(path=None, overrides=None):
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be an object")
    # Class maps are complete replacements, not recursive patches.
    cfg = merge_settings(deepcopy(DEFAULTS), {k: v for k, v in raw.items() if k != "people"})
    people = dict(raw.get("people", {}))
    roles = people.pop("class_roles", DEFAULTS["people"]["class_roles"])
    cfg["people"] = merge_settings(DEFAULTS["people"], people)
    cfg["people"]["class_roles"] = {str(k): v for k, v in roles.items()}
    for key, value in (overrides or {}).items():
        if value is not None:
            if key not in cfg:
                raise ValueError(f"Unknown override: {key}")
            cfg[key] = value
    if cfg["mode"] not in ("parallel", "sequential"):
        raise ValueError("mode must be parallel or sequential")
    for key in ("workers", "prefetch", "cpu_threads", "gpu_concurrency"):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 64:
            raise ValueError(f"{key} must be an integer in [1,64]")
    if cfg["max_frames"] is not None and (type(cfg["max_frames"]) is not int or cfg["max_frames"] < 1):
        raise ValueError("max_frames must be a positive integer or null")
    if cfg["timestamp_policy"] not in ("source", "fps"):
        raise ValueError("timestamp_policy must be source or fps (explicit CFR assumption)")
    for key in ("include_intervals", "exclude_intervals"):
        for span in cfg[key]:
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError(f"{key} requires [start_s, end_s] pairs")
            positive(span[0], key, zero=True)
            positive(span[1], key)
            if span[1] <= span[0]:
                raise ValueError("Intervals must have positive duration")
    if not isinstance(cfg["required"], list) or set(cfg["required"]) - {"players", "referees", "court", "ball", "pose", "jersey"}:
        raise ValueError("Unknown required capability")
    cfg["court"] = court_settings(cfg["court"])
    cfg["pose"] = pose_settings(cfg["pose"])
    cfg["jersey"] = jersey_settings(cfg["jersey"])
    cfg["tracking"] = tracking_settings(cfg["tracking"])
    cfg["tracking"]["enabled"] = True
    for key, value in cfg["distance"].items():
        positive(value, key, zero=key in ("smoothing_tau_s", "deadband_m"))
    if cfg["people"]["adapter"] not in ("yolo", "rfdetr"):
        raise ValueError("people.adapter must be yolo or rfdetr")
    roles = cfg["people"]["class_roles"]
    if not roles or any(not k.isdigit() or v not in ("player", "referee", "unknown") for k, v in roles.items()):
        raise ValueError("class_roles maps model class IDs to player/referee/unknown")
    if cfg["ball"]["adapter"] != "tracknet_v5":
        raise ValueError("Supported ball adapter: tracknet_v5 (including TOTNet checkpoints)")
    if type(cfg["ball"]["enabled"]) is not bool:
        raise ValueError("ball.enabled must be boolean")
    for name, key in (("people", "confidence"), ("ball", "threshold")):
        positive(cfg[name][key], f"{name}.{key}")
        if cfg[name][key] > 1:
            raise ValueError(f"{name}.{key} must be <= 1")
    for section in ("people", "ball"):
        if not isinstance(cfg[section]["device"], str) or not re.fullmatch(r"cpu|cuda:\d+", cfg[section]["device"]):
            raise ValueError(f"{section}.device must be cpu or cuda:N")
    for key in ("resolution", "max_detections"):
        if type(cfg["people"][key]) is not int or cfg["people"][key] < 1:
            raise ValueError(f"people.{key} must be a positive integer")
    if cfg["people"]["resolution"] % 32:
        raise ValueError("people.resolution must be a multiple of 32")
    return cfg


def included(config, timestamp):
    return (not config["include_intervals"] or any(a <= timestamp < b for a, b in config["include_intervals"])) and not any(
        a <= timestamp < b for a, b in config["exclude_intervals"])
