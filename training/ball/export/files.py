"""File helpers shared by the dataset writers."""

import csv
import json
import os
import shutil
from pathlib import Path

import cv2


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, columns: list[str], rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_image(path: Path, image, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    options = (
        [cv2.IMWRITE_JPEG_QUALITY, quality]
        if path.suffix == ".jpg"
        else [cv2.IMWRITE_PNG_COMPRESSION, 1]
    )
    if not cv2.imwrite(str(path), image, options):
        raise OSError(f"Cannot write image: {path}")


def link_or_copy(source: Path, target: Path):
    """Hard links keep each dataset portable without duplicating image bytes locally."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
