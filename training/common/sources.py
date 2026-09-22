"""Resolve explicit match identities and audit every split before frame selection."""

from pathlib import Path

from .config import SPLITS, read_yaml

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def source_videos(root: Path, *, sidecar_suffix: str) -> list[Path]:
    videos = {
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }
    # Include missing videos referenced by sidecars, so they fail explicitly.
    videos.update(
        path.with_name(path.name.removesuffix(sidecar_suffix))
        for path in root.rglob("*" + sidecar_suffix)
    )
    return sorted(videos)


def audit_sources(root: Path, fingerprint, *, sidecar_suffix: str) -> tuple[dict, dict]:
    """Check identities and byte-identical videos across train, val and test.

    `checks` also tracks absent metadata paths, so metadata appearing during the
    export cannot silently change a match identity or its inheritance.
    """
    sources, checks, documents = {}, {}, {}
    seen_matches, seen_videos, seen_groups, match_venues = {}, {}, {}, {}

    def read_metadata(path):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Metadata symlink leaves the dataset root: {path}")
        if path not in documents:
            checks[path] = fingerprint(path) if path.exists() else None
            documents[path] = read_yaml(path) if path.exists() else {}
        return documents[path]

    for video in source_videos(root, sidecar_suffix=sidecar_suffix):
        relative = video.relative_to(root)
        split = relative.parts[0]
        if split not in SPLITS or len(relative.parts) < 2:
            raise ValueError(
                f"Place annotations and videos under train/, val/ or test/: {relative}"
            )
        if not video.resolve().is_relative_to(root):
            raise ValueError(f"Source symlink leaves the dataset root: {video}")
        metadata_path = video.with_name(video.name + ".meta.yaml")
        metadata = read_metadata(metadata_path)
        if metadata.get("split") not in (None, split):
            raise ValueError(
                f"Split conflict in {metadata_path}: directory says {split}"
            )

        match_path, match = None, {}
        folder = video.parent
        while folder != root:
            candidate = folder / "match.yaml"
            document = read_metadata(candidate)
            if checks[candidate] is not None:
                match_path, match = candidate, document
                break
            folder = folder.parent

        identity = {}
        for key in ("match_id", "venue_id"):
            values = []
            for document in (metadata, match):
                value = document.get(key)
                if value is not None:
                    if (
                        not isinstance(value, str)
                        or not value.strip()
                        or value != value.strip()
                    ):
                        raise ValueError(
                            f"{key} must be a nonempty, trimmed string: {relative}"
                        )
                    values.append(value)
            if len(set(values)) > 1:
                raise ValueError(
                    f"Conflicting {key} between video and match metadata: {relative}"
                )
            identity[key] = values[0] if values else None
        if identity["match_id"] is None:
            raise ValueError(
                f"Missing match_id in {metadata_path} or a parent match.yaml"
            )
        match_id, venue_id = identity["match_id"], identity["venue_id"]
        if venue_id is not None:
            if match_id in match_venues and match_venues[match_id] != venue_id:
                raise ValueError(f"Conflicting venue_id for match {match_id}")
            match_venues[match_id] = venue_id

        checks[video] = fingerprint(video)
        group = Path(*relative.parts[1:-1]).as_posix()
        for seen, key, kind in (
            (seen_matches, match_id, "Match"),
            (seen_videos, checks[video], "Video content"),
            (seen_groups, group, "Source group"),
        ):
            if kind == "Source group" and key == ".":
                continue
            if key in seen and seen[key] != split:
                raise ValueError(
                    f"{kind} occurs in multiple splits ({seen[key]}, {split}): "
                    f"{relative}"
                )
            seen[key] = split
        sources[video] = {
            "video": relative.as_posix(),
            "split": split,
            "group": group,
            **identity,
            "video_sha256": checks[video],
            "metadata_sha256": checks[metadata_path],
            "identity_metadata": [
                {"path": path.relative_to(root).as_posix(), "sha256": checks[path]}
                for path in (metadata_path, match_path)
                if path is not None and checks[path] is not None
            ],
        }
    return sources, checks
