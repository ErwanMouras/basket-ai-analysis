"""Small, strict configuration objects for export and training preparation."""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

FORMATS = ("yolo", "coco", "tracknet-totnet")
LAYOUTS = ("sdk", "v3", "v4")
SPLITS = ("train", "val", "test")


def read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


@dataclass(frozen=True)
class ExportConfig:
    image_format: str = "jpg"
    jpeg_quality: int = 95
    resize_width: int | None = None
    resize_height: int | None = None
    resize_mode: str = "letterbox"
    unknown_position: str = "exclude"
    occluded_position: str = "include"
    splits: tuple[str, ...] = ("train", "val")
    tracknet_layouts: tuple[str, ...] = ("sdk",)
    heatmap_radius: int = 40
    heatmap_variance: float = 10.0

    def __post_init__(self):
        for name, choices in (
            ("image_format", ("jpg", "png")),
            ("resize_mode", ("stretch", "letterbox")),
            ("unknown_position", ("exclude", "empty")),
            ("occluded_position", ("include", "exclude")),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer between 1 and 100")
        if (self.resize_width is None) != (self.resize_height is None):
            raise ValueError("resize_width and resize_height must be provided together")
        for value in (self.resize_width, self.resize_height):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("Resize dimensions must be positive integers")
        if type(self.heatmap_radius) is not int or self.heatmap_radius < 1:
            raise ValueError("heatmap_radius must be a positive integer")
        if (
            type(self.heatmap_variance) not in (int, float)
            or not math.isfinite(self.heatmap_variance)
            or self.heatmap_variance <= 0
        ):
            raise ValueError("heatmap_variance must be positive and finite")
        for name, choices in (("splits", SPLITS), ("tracknet_layouts", LAYOUTS)):
            values = getattr(self, name)
            if (
                not isinstance(values, (tuple, list))
                or not values
                or any(value not in choices for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"{name} must contain distinct values from {choices}")

    @classmethod
    def from_file(cls, path: Path):
        payload = read_yaml(path)
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown export settings: {sorted(unknown)}")
        return cls(**payload)

    def to_dict(self):
        return asdict(self)


def training_profile(path: Path, name: str) -> dict:
    """Validate model input geometry before materializing training tensors."""
    config = read_yaml(path)
    profiles = config.get("profiles", {})
    if name not in profiles or not isinstance(profiles[name], dict):
        raise ValueError(f"Missing training profile {name!r} in {path}")
    profile = profiles[name]
    version = profile.get("version")
    if type(version) is not int or version not in (3, 4, 5):
        raise ValueError("TrackNet version must be 3, 4 or 5")
    divisor = 16 if version == 5 else 8
    for key in ("input_width", "input_height"):
        value = profile.get(key)
        if type(value) is not int or value < divisor or value % divisor:
            raise ValueError(f"{key} must be positive and divisible by {divisor}")
    for key in ("sequence_length", "sequence_stride"):
        if type(profile.get(key)) is not int or profile[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if version in (4, 5) and profile["sequence_length"] != 3:
        raise ValueError("The supported V4/V5 implementations require three frames")
    radius = profile.get("target_radius", 2.5)
    if type(radius) not in (float, int) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("target_radius must be positive and finite")
    return profile
