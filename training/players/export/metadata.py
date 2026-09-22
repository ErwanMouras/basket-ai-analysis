"""Source identities shared with ball, selecting the players sidecar suffix."""

from pathlib import Path

from training.common.provenance import file_hash
from training.common.sources import audit_sources as _audit_sources
from training.common.sources import source_videos as _source_videos


def source_videos(root: Path) -> list[Path]:
    return _source_videos(root, sidecar_suffix=".playersann.json")


def audit_sources(root: Path, fingerprint=file_hash) -> tuple[dict, dict]:
    return _audit_sources(root, fingerprint, sidecar_suffix=".playersann.json")
