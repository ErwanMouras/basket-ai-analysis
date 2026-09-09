"""Resolve recipes and reject unsupported combinations before training."""

import argparse
import json
import math
from pathlib import Path

from training.ball.export.config import read_yaml, training_profile
from training.ball.export.dataset import verify_dataset

from .references import ROOT, activate_reference


def positive(config, key, allow_zero=False):
    value = config[key]
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (not allow_zero and value == 0)
    ):
        raise ValueError(
            f"{key} must be {'nonnegative' if allow_zero else 'positive'} and finite"
        )


def merge_settings(defaults, overrides, prefix=""):
    """Merge recipe mappings while rejecting misspelled settings at every level."""
    result = defaults.copy()
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {prefix} settings: {sorted(unknown)}")
    for key, value in overrides.items():
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{key} must be a mapping")
            result[key] = merge_settings(defaults[key], value, f"{prefix}{key}.")
        else:
            result[key] = value
    return result


def load_config(model, path=None, resume=None):
    defaults = ROOT / f"training/ball/configs/train_{model}.yaml"
    config = read_yaml(defaults)
    if path:
        overrides = read_yaml(Path(path))
        config = merge_settings(config, overrides)
    if config["model"] != model:
        raise ValueError(f"This entry point requires model={model}")
    if resume:
        config["resume"] = resume
    for key in ("dataset", "output", "profile_file", "reference_root", "resume"):
        if config.get(key):
            config[key] = str((ROOT / config[key]).resolve())
    if model != "yolo":
        if config["profile"] != model:
            raise ValueError("The preparation profile must match the model")
        config["geometry"] = training_profile(Path(config["profile_file"]), model)
    else:
        if (
            any(
                type(config[key]) is not int or config[key] < 32
                for key in ("input_width", "input_height")
            )
            or config["input_width"] != config["input_height"]
            or config["input_width"] % 32
        ):
            raise ValueError(
                "YOLO training requires square input dimensions divisible by 32"
            )
        if (config["sequence_length"], config["sequence_stride"]) != (1, 1):
            raise ValueError("YOLO uses individual frames")
    validate_config(config)
    return config


def validate_config(config):
    for key in ("batch_size", "epochs", "prefetch_factor", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("workers", "seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("pin_memory", "deterministic"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("max_train_batches", "max_val_batches"):
        if config[key] is not None and (
            type(config[key]) is not int or config[key] < 1
        ):
            raise ValueError(f"{key} must be null or a positive integer")
    positive(config, "learning_rate")
    positive(config, "weight_decay", allow_zero=True)
    if config["optimizer"] not in ("Adam", "AdamW", "SGD", "Adadelta"):
        raise ValueError("Unsupported optimizer")
    model = config["model"]
    choices = ("fp32", "amp") if model == "yolo" else ("fp32", "fp16", "bf16")
    if config["precision"] not in choices:
        raise ValueError(f"precision must be one of {choices}")
    if model == "yolo":
        if config["yolo"]["augmentations"] != []:
            raise ValueError(
                "Use the explicit native YOLO augmentation settings; custom Albumentations objects are unsupported"
            )
        if config["optimizer"] == "Adadelta":
            raise ValueError("YOLO does not support Adadelta")
        if config["max_train_batches"] or config["max_val_batches"]:
            raise ValueError(
                "Use a small exported dataset for YOLO smoke tests; batch limits are temporal only"
            )
        return
    for key in ("tolerance_px", "grad_clip"):
        positive(config, key)
    positive(config, "min_learning_rate", allow_zero=True)
    if config["min_learning_rate"] > config["learning_rate"]:
        raise ValueError("min_learning_rate exceeds learning_rate")
    if not 0 < config["threshold"] < 1:
        raise ValueError("threshold must be between zero and one")
    if config["scheduler"] not in ("none", "cosine"):
        raise ValueError("scheduler must be none or cosine")
    profile = config["geometry"]
    common_keys = {
        "version",
        "input_width",
        "input_height",
        "sequence_length",
        "sequence_stride",
    }
    extra_keys = {
        "tracknet_v3": {"target_radius", "background_mode"},
        "tracknet_v4": {"target_radius"},
    }.get(model, {"occlusion_augmentation", "ball_present_weight"})
    if set(profile) - common_keys - extra_keys:
        raise ValueError("Unknown preparation profile settings")
    version = 3 if model == "tracknet_v3" else 4 if model == "tracknet_v4" else 5
    if profile["version"] != version:
        raise ValueError("Profile version does not match the architecture")
    if model == "tracknet_v3":
        if profile.get("background_mode", "") != "" or config["frame_mixup_alpha"] != 0:
            raise ValueError(
                'This V3 recipe supports background_mode="" and sample mixup only; no medians or frame interpolation'
            )
        positive(config, "mixup_alpha", allow_zero=True)
    if model == "tracknet_v4":
        if config["fusion"] not in ("TypeA", "TypeB"):
            raise ValueError("V4 fusion must be TypeA or TypeB")
        if (
            config["precision"] != "fp32"
            or config["workers"] != 0
            or config["pin_memory"]
            or config["prefetch_factor"] != 1
        ):
            raise ValueError(
                "Reference TensorFlow V4 requires fp32, workers=0, prefetch_factor=1 and pin_memory=false"
            )
    if model == "tracknet_v5_totnet":
        validate_augmentation(config["augmentation"])
    if model.startswith("tracknet_v5"):
        positive(profile, "ball_present_weight")
        if type(profile["occlusion_augmentation"]) is not bool:
            raise ValueError("occlusion_augmentation must be boolean")
        if model == "tracknet_v5" and (
            profile["occlusion_augmentation"] or profile["ball_present_weight"] != 1
        ):
            raise ValueError(
                "Use tracknet_v5_totnet for occlusion augmentation or weighted targets"
            )


def validate_augmentation(options):
    for name, values in options.items():
        probability = values["prob"]
        if type(probability) not in (float, int) or not 0 <= probability <= 1:
            raise ValueError(f"{name}.prob must be in [0, 1]")
        for key, value in values.items():
            if key.endswith("range") and (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(
                    type(item) not in (int, float) or not math.isfinite(item)
                    for item in value
                )
                or not 0 <= value[0] <= value[1]
            ):
                raise ValueError(f"Invalid augmentation range: {name}.{key}")
    for name, key, maximum in (
        ("brightness", "max_delta", 255),
        ("hue_saturation", "hue_delta_deg", 180),
    ):
        if not 0 <= options[name][key] <= maximum:
            raise ValueError(f"Invalid augmentation amplitude: {name}.{key}")
    low, high = options["jpeg"]["quality_range"]
    if not 1 <= low <= high <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    if options["motion_blur"]["kernel_range"][0] < 1:
        raise ValueError("Motion blur kernels must be positive")


def preflight(config):
    root = Path(config["dataset"])
    verify_dataset(root)
    # verify_dataset returns a report; keep the full dataset description as the run artifact.
    manifest = json.loads((root / "manifest.json").read_text())
    expected = "yolo" if config["model"] == "yolo" else "tracknet-totnet"
    if manifest["format"] != expected:
        raise ValueError(f"Expected an export with format={expected}")
    if not {"train", "val"} <= set(manifest["parameters"]["splits"]):
        raise ValueError("Both train and val exports are required")
    if config["model"] != "yolo":
        layout = {"tracknet_v3": "v3", "tracknet_v4": "v4"}.get(config["model"], "sdk")
        if layout not in manifest["parameters"]["tracknet_layouts"]:
            raise ValueError(
                f"Missing {layout} layout; re-export with TRACKNET_LAYOUTS=sdk,v3,v4"
            )
        activate_reference(config)
    config["dataset_id"] = manifest["dataset_id"]
    config["export_id"] = manifest["export_id"]
    config["annotation_policy"] = {
        key: manifest["parameters"][key]
        for key in ("unknown_position", "occluded_position")
    }
    return manifest


def parse_config(model):
    parser = argparse.ArgumentParser(
        description=f"Train {model} using train and val exports only."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    return load_config(model, args.config, str(args.resume) if args.resume else None)


def resume_contract(config):
    """Operational paths may change; data, architecture and optimization may not."""
    ignored = {
        "output",
        "resume",
        "mlflow_uri",
        "experiment",
        "reference_root",
        "profile_file",
        "dataset",
        "workers",
        "prefetch_factor",
        "pin_memory",
        "cpu_threads",
    }
    return {key: value for key, value in config.items() if key not in ignored}
