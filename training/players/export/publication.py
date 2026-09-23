"""Publish a complete pair by atomically replacing one Linux symlink.

Old generations remain readable. A consumer using both formats must resolve
``current`` once and keep that path for the duration of its operation.
"""

import fcntl
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from training.common.files import write_json
from training.common.provenance import object_hash
from training.players.export.verify import read_json, verify_pair

OWNER = {"schema_version": 1, "artifact_type": "players_export_bundle"}
MARKER = ".players-export.json"
ALLOWED = {MARKER, ".export.lock", ".generations", "current", "yolo", "coco", ".next"}


def inspect_root(root, *, check_pending=True):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Export output must be a real directory")
    entries = {p.name for p in root.iterdir()}
    marker = root / MARKER
    if marker.is_symlink() or (entries - {".export.lock"} and not marker.is_file()):
        raise ValueError("Refusing to overwrite a foreign output directory")
    if marker.exists() and read_json(marker) != OWNER:
        raise ValueError("Unknown export output owner")
    if entries - ALLOWED:
        raise ValueError("Foreign files in the export output")
    for name in (".generations", ".export.lock"):
        if (root / name).is_symlink():
            raise ValueError(f"Unsafe publication path: {name}")
    for fmt in ("yolo", "coco"):
        alias = root / fmt
        if os.path.lexists(alias) and (
            not alias.is_symlink() or os.readlink(alias) != f"current/{fmt}"
        ):
            raise ValueError(f"Foreign format output: {fmt}")
    for name in ("current", ".next") if check_pending else ("current",):
        pointer = root / name
        if os.path.lexists(pointer):
            if not pointer.is_symlink() or not re.fullmatch(
                r"\.generations/[0-9a-f]{64}", os.readlink(pointer)
            ):
                raise ValueError("Invalid generation pointer")
            target = root / os.readlink(pointer)
            if target.is_symlink() or not target.is_dir():
                raise ValueError("Invalid generation directory")


@contextmanager
def staging(output):
    root = Path(output).absolute()
    root.mkdir(parents=True, exist_ok=True)
    inspect_root(root, check_pending=False)
    fd = os.open(root / ".export.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    stage = None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "Another player export is publishing to this output"
            ) from exc
        inspect_root(root)
        if not (root / MARKER).exists():
            write_json(root / MARKER, OWNER)
        generations = root / ".generations"
        generations.mkdir(exist_ok=True)
        for fmt in ("yolo", "coco"):
            if not (root / fmt).is_symlink():
                (root / fmt).symlink_to(f"current/{fmt}", target_is_directory=True)
        stage = Path(tempfile.mkdtemp(prefix=".tmp-", dir=generations))
        yield root, stage
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        os.close(fd)


def publish(root, stage, manifests):
    bundle_id = object_hash({fmt: m["export_id"] for fmt, m in manifests.items()})
    target = root / ".generations" / bundle_id
    if os.path.lexists(target):
        if target.is_symlink() or verify_pair(target) != manifests:
            raise ValueError(
                "Existing generation is corrupt or belongs to a different export"
            )
    else:
        stage.rename(target)
    # Everything above may fail without changing the publicly selected pair.
    pointer = root / ".next"
    try:
        pointer.unlink(missing_ok=True)
        pointer.symlink_to(f".generations/{bundle_id}", target_is_directory=True)
        os.replace(pointer, root / "current")
    finally:
        pointer.unlink(missing_ok=True)
    return target


def verify_bundle(output):
    root = Path(output).absolute()
    inspect_root(root, check_pending=False)
    if not (root / MARKER).is_file() or any(
        not (root / fmt).is_symlink() for fmt in ("yolo", "coco")
    ):
        raise ValueError("Incomplete publication layout")
    generation = (root / "current").resolve(strict=True)
    manifests = verify_pair(generation)
    if generation.name != object_hash(
        {fmt: m["export_id"] for fmt, m in manifests.items()}
    ):
        raise ValueError("Invalid generation identity")
    return manifests
