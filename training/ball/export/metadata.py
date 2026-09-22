"""Ball source identity API backed by the shared split audit."""

from pathlib import Path

from training.common.sources import VIDEO_EXTENSIONS as VIDEO_EXTENSIONS
from training.common.sources import audit_sources as _audit_sources
from training.common.sources import source_videos as _source_videos


def source_videos(root: Path) -> list[Path]:
    return _source_videos(root, sidecar_suffix=".ballann.json")


def audit_sources(root: Path, fingerprint) -> tuple[dict, dict]:
    return _audit_sources(root, fingerprint, sidecar_suffix=".ballann.json")
