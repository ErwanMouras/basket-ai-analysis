"""Explicit, checksum-verified PARSeq installation (or local import)."""

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

from training.common.files import atomic_writer
from training.common.provenance import ROOT, file_hash
from training.jersey.config import DEFAULTS, MODEL_SHA256, MODEL_URL


def install(destination, source=None):
    destination = (ROOT / Path(destination).expanduser()).resolve()
    if destination.exists():
        if file_hash(destination) != MODEL_SHA256:
            raise ValueError("Existing checkpoint differs; choose another output")
        return destination
    with tempfile.TemporaryDirectory(prefix="jersey-") as temp:
        if source is None:
            source = Path(temp) / "parseq.pt"
            with (
                urllib.request.urlopen(MODEL_URL, timeout=30) as reader,
                source.open("wb") as out,
            ):
                shutil.copyfileobj(reader, out)
        with (
            Path(source).expanduser().open("rb") as reader,
            atomic_writer(destination, binary=True) as out,
        ):
            digest = hashlib.sha256()
            while block := reader.read(1024**2):
                digest.update(block)
                out.write(block)
            if digest.hexdigest() != MODEL_SHA256:
                raise ValueError("PARSeq checkpoint SHA-256 mismatch")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(DEFAULTS["weights"]))
    args = parser.parse_args()
    print(install(args.output, args.source))


if __name__ == "__main__":
    main()
