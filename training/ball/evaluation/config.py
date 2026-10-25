"""Evaluation recipes, causal policy and content-bound final-test freezes."""

import json
from pathlib import Path

from training.ball.export.config import read_yaml
from training.ball.learning.config import merge_settings, positive
from training.ball.learning.references import ROOT
from training.common.provenance import file_hash, object_hash, source_fingerprints


def load_config(path=None, checkpoint=None):
    config = read_yaml(ROOT / "training/ball/configs/evaluate_ball.yaml")
    if path:
        config = merge_settings(config, read_yaml(Path(path)))
    if checkpoint:
        config["checkpoint"] = str(checkpoint)
    if not config["checkpoint"]:
        raise ValueError("A checkpoint is required")
    if config["split"] not in ("val", "test"):
        raise ValueError("Evaluation accepts val or frozen test")
    if config["split"] == "test" and not config["frozen_recipe"]:
        # A path is required even when preparing the freeze; it need not exist yet.
        raise ValueError("Test requires an explicit frozen_recipe path")
    if config["temporal_policy"] not in ("offline_mean", "causal_last"):
        raise ValueError("Unknown temporal_policy")
    if not isinstance(config["cut_frames"], dict) or any(
        not isinstance(clip, str) or not isinstance(indices, list)
        or any(type(i) is not int or i < 0 for i in indices)
        or len(indices) != len(set(indices))
        for clip, indices in config["cut_frames"].items()
    ):
        raise ValueError("cut_frames maps clip IDs to distinct nonnegative frame indices")
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
    for key in ("checkpoint", "dataset", "output", "reference_root", "frozen_recipe"):
        if config[key] is None:
            continue
        config[key] = str((ROOT / config[key]).resolve())
    return config


def frozen_payload(config, manifest, records):
    paths = list((ROOT / "training/ball/evaluation").glob("*.py")) + [
        ROOT / "training/ball/learning/data.py", ROOT / "training/ball/learning/metrics.py",
        ROOT / "training/ball/learning/torch_data.py",
    ]
    recipe = {k: v for k, v in config.items() if k not in {
        "checkpoint", "dataset", "output", "reference_root", "frozen_recipe", "mlflow_uri", "videos"}}
    payload = {"schema_version": 1, "recipe": recipe,
               "checkpoint_sha256": file_hash(Path(config["checkpoint"])),
               "dataset_id": manifest["dataset_id"], "export_id": manifest["export_id"],
               "selection_id": object_hash(records), "evaluator_id": object_hash(source_fingerprints(ROOT, paths))}
    return {**payload, "recipe_id": object_hash(payload)}


def require_frozen(config, payload):
    if config["split"] == "test":
        if not config["frozen_recipe"] or json.loads(Path(config["frozen_recipe"]).read_text()) != payload:
            raise ValueError("Frozen test no longer matches weights, data, code or protocol")


def validate_checkpoint_for_split(trained, split):
    if split == "test" and (trained.get("max_train_batches") or trained.get("max_val_batches") or trained.get("purpose") == "smoke"):
        raise ValueError("Smoke or batch-truncated checkpoints cannot enter final test")
