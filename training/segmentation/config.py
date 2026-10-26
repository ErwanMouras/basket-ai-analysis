"""Strict opt-in configuration shared by inference and evaluation."""
import math
from training.common.config import merge_settings

DEFAULTS = {
    "strategy": "disabled", "weights": None, "resolution": 1024,
    "class_roles": {"0": "unknown"}, "person_role_policy": "unknown",
    "referee_iou": 0.5, "segment_referees": True,
}


def settings(overrides=None):
    cfg = merge_settings(DEFAULTS, overrides or {})
    if cfg["strategy"] not in ("disabled", "box_prompt", "direct"):
        raise ValueError("segmentation.strategy must be disabled, box_prompt or direct")
    if cfg["strategy"] != "disabled" and (not isinstance(cfg["weights"], str) or not cfg["weights"] or "://" in cfg["weights"]):
        raise ValueError("Segmentation requires an explicit local checkpoint; install it before inference")
    if type(cfg["resolution"]) is not int or cfg["resolution"] < 64 or cfg["resolution"] % 32:
        raise ValueError("segmentation.resolution must be a multiple of 32 >= 64")
    if cfg["strategy"] == "box_prompt" and cfg["resolution"] != 1024:
        raise ValueError("Pinned SAM 2.1 requires resolution 1024")
    if type(cfg["segment_referees"]) is not bool:
        raise ValueError("segment_referees must be boolean")
    if cfg["person_role_policy"] not in ("unknown", "non_referee_person_proxy"):
        raise ValueError("Invalid experimental person_role_policy")
    if not isinstance(cfg["class_roles"], dict) or not cfg["class_roles"] or any(
        not isinstance(k, str) or not k.isdigit() or v not in ("player", "referee", "unknown")
        for k, v in cfg["class_roles"].items()
    ):
        raise ValueError("Segmentation class_roles must map string class IDs to explicit roles")
    v = cfg["referee_iou"]
    if type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1:
        raise ValueError("referee_iou must be in (0,1]")
    return cfg
