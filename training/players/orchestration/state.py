"""Locks, immutable receipts and file inventories (including symlink targets)."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

from training.common.provenance import file_hash


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another process owns {path}") from exc
        yield
    finally:
        os.close(fd)


def inventory(roots):
    values = {}
    for root in map(Path, roots):
        if not root.exists():
            raise ValueError(f"Missing artifact: {root}")
        paths = root.rglob("*") if root.is_dir() else [root]
        for path in sorted(paths):
            if path.is_symlink() and path.is_dir():
                raise ValueError(f"Directory symlinks are not supported in receipts: {path}")
            if path.is_file():
                values[str(path.absolute())] = {"sha256": file_hash(path),
                    "symlink": os.readlink(path) if path.is_symlink() else None}
            elif not path.is_dir():
                raise ValueError(f"Broken or special artifact: {path}")
    return values


def input_files(paths):
    return {str(Path(p).absolute()): file_hash(Path(p)) if Path(p).is_file() else None for p in paths}


def inputs_match(inputs):
    return input_files(inputs) == inputs


def receipt_valid(receipt, fingerprint):
    try:
        return (receipt["fingerprint"] == fingerprint
                and receipt["artifacts"] == inventory(receipt["artifact_roots"]))
    except (OSError, ValueError, KeyError):
        return False
