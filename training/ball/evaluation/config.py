"""Explicit validation-only evaluation recipes."""

from pathlib import Path

from training.ball.export.config import read_yaml
from training.ball.learning.config import merge_settings, positive
from training.ball.learning.references import ROOT


def load_config(path=None, checkpoint=None):
    config = read_yaml(ROOT / "training/ball/configs/evaluate_ball.yaml")
    if path:
        config = merge_settings(config, read_yaml(Path(path)))
    if checkpoint:
        config["checkpoint"] = str(checkpoint)
    if not config["checkpoint"]:
        raise ValueError("A checkpoint is required")
    if config["split"] != "val":
        raise ValueError(
            "Evaluation accepts only val; test is reserved for a frozen recipe"
        )
    for key in ("batch_size", "prefetch_factor", "cpu_threads", "video_max_width"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["video_max_width"] < 2:
        raise ValueError("video_max_width must be at least 2")
    for key in ("workers", "seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("pin_memory", "deterministic", "videos"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    positive(config, "threshold")
    if config["threshold"] >= 1:
        raise ValueError("threshold must be between zero and one")
    positive(config, "tolerance_px")
    if config["tolerance_space"] not in ("source", "export"):
        raise ValueError("tolerance_space must be source or export")
    if config["unknown_position"] not in ("ignore", "negative"):
        raise ValueError("unknown_position must be ignore or negative")
    if config["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    for key in ("checkpoint", "dataset", "output", "reference_root"):
        config[key] = str((ROOT / config[key]).resolve())
    return config
