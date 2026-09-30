"""Strict evaluation recipes and fingerprints for comparable experiments."""

import re
from pathlib import Path

from training.common.config import merge_settings, read_yaml
from training.common.provenance import ROOT, object_hash
from training.players.learning.config import VARIANTS

EVALUATION_VARIANTS = {**VARIANTS, "yolo": [*VARIANTS["yolo"], "yolov8n"]}

DEFAULTS = {
    "model": "yolo", "variant": "yolo26n", "weights": None, "source_class": 0,
    "referee_weights": "models/players/ebard_yolov8n.pt", "referee_source_class": 3,
    "dataset": "exports/players/coco", "split": "val", "output": "runs/players/evaluation",
    "device": "cuda:0", "precision": "fp32", "resolution": 640, "cpu_threads": 4,
    "score_floor": 0.001, "score_threshold": 0.25, "iou_threshold": 0.5,
    "max_detections": 100, "warmup": 5, "repeats": 3, "max_images": None,
    "max_examples": 20, "purpose": "evaluation", "reference": None,
    "frozen_recipe": None, "mlflow_uri": "sqlite:///mlflow.db",
}


def load_config(path, checkpoint=None):
    config = merge_settings(DEFAULTS, read_yaml(Path(path)))
    if checkpoint:
        config["weights"] = str(checkpoint)
    validate_config(config)
    for key in ("dataset", "output", "weights", "referee_weights", "frozen_recipe"):
        if config[key] is not None:
            config[key] = str((ROOT / Path(config[key]).expanduser()).resolve())
    return config


def validate_config(config):
    if set(config) != set(DEFAULTS):
        raise ValueError("Unknown or missing evaluation settings")
    if config["model"] not in EVALUATION_VARIANTS or config["variant"] not in EVALUATION_VARIANTS[config["model"]]:
        raise ValueError("Invalid detector family/variant")
    if config["split"] not in ("val", "test"):
        raise ValueError("Evaluation split must be val or frozen test")
    if config["purpose"] not in ("smoke", "evaluation"):
        raise ValueError("Invalid evaluation purpose")
    if config["precision"] != "fp32":
        raise ValueError("The common evaluation protocol currently supports fp32 only")
    if not isinstance(config["device"], str) or not re.fullmatch(r"cpu|cuda:\d+", config["device"]):
        raise ValueError("Use cpu or cuda:N")
    for key in ("resolution", "cpu_threads", "max_detections", "warmup", "repeats"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["resolution"] < 64 or config["resolution"] % 32:
        raise ValueError("resolution must be >=64 and divisible by 32")
    if config["model"] == "rfdetr" and config["resolution"] % 64:
        raise ValueError("RF-DETR evaluation resolution must be divisible by 64")
    if not 10 <= config["max_detections"] <= 300:
        raise ValueError("Common max_detections must be in [10, 300]")
    for key in ("max_examples", "source_class", "referee_source_class"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["max_images"] is not None and (type(config["max_images"]) is not int or config["max_images"] < 1):
        raise ValueError("max_images must be null or positive")
    for key in ("score_floor", "score_threshold", "iou_threshold"):
        if type(config[key]) not in (int, float) or not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if not 0 < config["score_floor"] <= 0.01 or config["score_threshold"] < config["score_floor"]:
        raise ValueError("Use a low AP floor in (0, .01] and an operating score >= floor")
    if config["iou_threshold"] == 0:
        raise ValueError("IoU threshold must be positive")
    for key in ("dataset", "output", "weights", "mlflow_uri"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be an explicit nonempty string")
    if config["referee_weights"] is not None and (not isinstance(config["referee_weights"], str) or not config["referee_weights"]):
        raise ValueError("referee_weights must be null (same checkpoint) or a nonempty path")
    if config["model"] == "rfdetr" and config["referee_weights"] is None:
        raise ValueError("RF-DETR requires a separate referee YOLO checkpoint")
    if config["referee_weights"] is None and config["source_class"] == config["referee_source_class"]:
        raise ValueError("Player and referee source classes must differ")
    for key in ("reference", "frozen_recipe"):
        if config[key] is not None and (not isinstance(config[key], str) or not config[key]):
            raise ValueError(f"{key} must be null or a nonempty string")
    if config["split"] == "test" and (config["purpose"] == "smoke" or config["max_images"] is not None):
        raise ValueError("Final test requires a full selection and a frozen non-smoke recipe")


def frozen_payload(config, *, checkpoint_sha256, dataset_id, selection_id, evaluator_id,
                   referee_checkpoint_sha256=None):
    recipe = {k: v for k, v in config.items() if k not in {
        "weights", "dataset", "output", "mlflow_uri", "frozen_recipe", "max_examples"}}
    payload = {"schema_version": 1, "recipe": recipe, "checkpoint_sha256": checkpoint_sha256,
               "referee_checkpoint_sha256": referee_checkpoint_sha256,
               "dataset_id": dataset_id, "selection_id": selection_id, "evaluator_id": evaluator_id}
    return {**payload, "recipe_id": object_hash(payload)}


def require_frozen(config, expected):
    import json

    if config["split"] != "test":
        return
    if not config["frozen_recipe"]:
        raise ValueError("Test is reserved for a frozen recipe; use --freeze-recipe first")
    actual = json.loads(Path(config["frozen_recipe"]).read_text())
    if actual != expected:
        raise ValueError("Frozen recipe no longer matches weights, data, code or protocol")
