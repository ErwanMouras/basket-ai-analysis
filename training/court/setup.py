"""Install the reference court checkpoint with a pinned SHA-256."""

import argparse
import hashlib
import urllib.request
from contextlib import closing
from pathlib import Path

from training.common.files import atomic_writer
from training.common.provenance import ROOT, file_hash
from training.court.config import DEFAULTS
from training.court.model import MODEL_SHA256, MODEL_URL


def install(destination, *, source=None):
    destination = (ROOT / Path(destination).expanduser()).resolve()
    if destination.exists():
        if file_hash(destination) != MODEL_SHA256:
            raise ValueError("Existing court checkpoint has different bytes; choose another output")
        return destination
    reader = Path(source).expanduser().open("rb") if source else urllib.request.urlopen(MODEL_URL, timeout=60)
    with closing(reader), atomic_writer(destination, binary=True) as handle:
        digest = hashlib.sha256()
        while block := reader.read(1024 * 1024):
            digest.update(block)
            handle.write(block)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Court checkpoint SHA-256 mismatch")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(DEFAULTS["weights"]))
    args = parser.parse_args()
    print(install(args.output, source=args.source))


if __name__ == "__main__":
    main()
