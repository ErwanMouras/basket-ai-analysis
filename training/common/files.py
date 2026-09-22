"""Atomic UTF-8 artifacts; failed writes preserve the previous file."""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_writer(path: Path, *, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        options = {} if binary else {"encoding": "utf-8", "newline": ""}
        with tempfile.NamedTemporaryFile(
            mode="wb" if binary else "w",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
            **options,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_bytes(path: Path, content: bytes):
    with atomic_writer(path, binary=True) as handle:
        handle.write(content)


def write_json(path: Path, value):
    with atomic_writer(path) as handle:
        handle.write(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


def write_jsonl(path: Path, rows):
    with atomic_writer(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
