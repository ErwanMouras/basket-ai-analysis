"""Strict training recipes. Operational paths may change during a strict resume."""

import copy
import re
from pathlib import Path

from training.common.config import merge_settings, positive, read_yaml
from training.common.provenance import ROOT

DEFAULTS = {
    "model": "yolo",
    "variant": "yolo26n",
    "dataset": "exports/players/yolo",
    "output": "runs/players",
    "mode": "finetune",
    "weights": None,
    "resume": None,
    "epochs": 30,
    "batch_size": 4,
    "resolution": 640,
    "seed": 53,
    "device": "cuda:0",
    "precision": "fp32",
    "workers": 0,
    "cpu_threads": 4,
    "deterministic": True,
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "warmup_epochs": 0.0,
    "purpose": "training",
    "max_train_images": None,
    "max_val_images": None,
    "mlflow_uri": "sqlite:///mlflow.db",
    "experiment": "players-training",
    "yolo": {
        "mosaic": 0.0,
        "mixup": 0.0,
        "fliplr": 0.5,
        "scale": 0.5,
        "translate": 0.1,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "close_mosaic": 0,
        "lrf": 0.01,
        "cos_lr": True,
    },
    "rfdetr": {
        "lr_encoder": 0.00015,
        "grad_accum_steps": 1,
        "use_ema": True,
        "ema_decay": 0.993,
        "ema_tau": 100,
        "scale_jitter": True,
        "lr_min_factor": 0.01,
        "clip_max_norm": 0.1,
        "amp_init_scale": 128.0,
    },
}
VARIANTS = {
    "yolo": [f"yolo26{s}" for s in "nsmlx"],
    "rfdetr": [f"rfdetr_{s}" for s in ("nano", "small", "medium", "large")],
}


def load_config(model, path=None, *, resume=None):
    if model not in VARIANTS:
        raise ValueError("Expected yolo or rfdetr")
    defaults = copy.deepcopy(DEFAULTS)
    defaults.update(
        model=model,
        variant="yolo26n" if model == "yolo" else "rfdetr_nano",
        dataset=f"exports/players/{'yolo' if model == 'yolo' else 'coco'}",
        resolution=640 if model == "yolo" else 384,
    )
    config = merge_settings(defaults, read_yaml(Path(path)) if path else {})
    if config["model"] != model:
        raise ValueError("Recipe family does not match the command")
    if resume:
        config.update(mode="resume", resume=str(resume))
    validate_config(config)
    for key in ("dataset", "output", "weights", "resume"):
        if config[key] is not None:
            config[key] = str((ROOT / Path(config[key]).expanduser()).resolve())
    return config


def validate_config(config):
    if (
        config["model"] not in VARIANTS
        or config["variant"] not in VARIANTS[config["model"]]
    ):
        raise ValueError("Unsupported detection model variant")
    if config["mode"] not in ("scratch", "finetune", "resume"):
        raise ValueError("mode must be scratch, finetune or resume")
    if config["mode"] == "resume":
        if not config["resume"]:
            raise ValueError("Strict resume requires a complete player checkpoint")
    elif config["resume"]:
        raise ValueError("Use mode=resume for resume, weights for fine-tuning")
    if config["mode"] == "finetune" and not config["weights"]:
        raise ValueError("Fine-tuning requires explicit local weights")
    if config["mode"] == "scratch" and config["weights"]:
        raise ValueError("Scratch training must not specify weights")
    for key in ("epochs", "batch_size", "resolution", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "workers"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["workers"] != 0:
        raise ValueError(
            "Phase 4 supports workers=0 for reproducible epoch-boundary resume"
        )
    if config["resolution"] < 64 or config["resolution"] % 32:
        raise ValueError("resolution must be >=64 and divisible by 32")
    if not isinstance(config["device"], str) or not re.fullmatch(
        r"cpu|cuda:\d+", config["device"]
    ):
        raise ValueError("Use one cpu or cuda:N device")
    if config["precision"] not in ("fp32", "amp") or (
        config["device"] == "cpu" and config["precision"] != "fp32"
    ):
        raise ValueError("Use fp32 or CUDA amp")
    if type(config["deterministic"]) is not bool:
        raise ValueError("deterministic must be boolean")
    if config["optimizer"] != "AdamW":
        raise ValueError("Phase 4 recipes use explicit AdamW")
    if config["purpose"] not in ("smoke", "quick", "training"):
        raise ValueError("Unknown run purpose")
    for key in ("max_train_images", "max_val_images"):
        if config[key] is not None and (
            type(config[key]) is not int or config[key] < 1
        ):
            raise ValueError(f"{key} must be null or a positive integer")
    for key in ("learning_rate",):
        positive(config, key)
    for key in ("weight_decay", "warmup_epochs"):
        positive(config, key, allow_zero=True)
    for section, keys in (
        (
            "yolo",
            (
                "mosaic",
                "mixup",
                "fliplr",
                "scale",
                "translate",
                "hsv_h",
                "hsv_s",
                "hsv_v",
                "lrf",
            ),
        ),
        ("rfdetr", ("ema_decay", "lr_min_factor")),
    ):
        for key in keys:
            positive(config[section], key, allow_zero=True)
            if config[section][key] > 1:
                raise ValueError(f"{section}.{key} must be in [0, 1]")
    for section, key in (
        ("yolo", "cos_lr"),
        ("rfdetr", "use_ema"),
        ("rfdetr", "scale_jitter"),
    ):
        if type(config[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be boolean")
    for key in ("lr_encoder", "clip_max_norm", "ema_tau", "amp_init_scale"):
        positive(config["rfdetr"], key)
    if (
        type(config["rfdetr"]["grad_accum_steps"]) is not int
        or config["rfdetr"]["grad_accum_steps"] < 1
    ):
        raise ValueError("grad_accum_steps must be a positive integer")
    if (
        type(config["yolo"]["close_mosaic"]) is not int
        or config["yolo"]["close_mosaic"] < 0
    ):
        raise ValueError("close_mosaic must be a nonnegative integer")
    for key in ("dataset", "output", "mlflow_uri", "experiment"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")


def resume_contract(config):
    ignored = {
        "dataset",
        "output",
        "weights",
        "resume",
        "mode",
        "mlflow_uri",
        "experiment",
        "checkpoint_sha256",
        "parent_run_id",
    }
    return {k: v for k, v in config.items() if k not in ignored}
