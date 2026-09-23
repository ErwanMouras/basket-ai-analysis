"""Physical export settings, independent of network input dimensions."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

from training.common.config import SPLITS, read_yaml


@dataclass(frozen=True)
class ExportConfig:
    image_format: str = "png"
    jpeg_quality: int = 95
    resize_width: int | None = None
    resize_height: int | None = None
    resize_mode: str = "letterbox"
    splits: tuple[str, ...] = ("train", "val")
    split_by_venue: bool = False

    def __post_init__(self):
        if self.image_format not in ("png", "jpg") or self.resize_mode not in (
            "letterbox",
            "stretch",
        ):
            raise ValueError("Use png/jpg and letterbox/stretch")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer in [1, 100]")
        if (self.resize_width is None) != (self.resize_height is None):
            raise ValueError("Provide both resize dimensions or neither")
        for dimension in (self.resize_width, self.resize_height):
            if dimension is not None and (type(dimension) is not int or dimension < 1):
                raise ValueError("Resize dimensions must be positive integers")
        if type(self.split_by_venue) is not bool:
            raise ValueError("split_by_venue must be boolean")
        if (
            not isinstance(self.splits, (list, tuple))
            or not self.splits
            or any(s not in SPLITS for s in self.splits)
            or len(set(self.splits)) != len(self.splits)
        ):
            raise ValueError("Select distinct train/val/test splits")
        object.__setattr__(self, "splits", tuple(s for s in SPLITS if s in self.splits))

    @classmethod
    def from_file(cls, path: Path):
        raw = read_yaml(path)
        if raw.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown player export settings")
        return cls(**raw)

    def to_dict(self):
        return {**asdict(self), "splits": list(self.splits)}
