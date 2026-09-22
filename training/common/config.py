"""Shared YAML loading and strict recipe validation."""

import math
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test")


def read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


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
