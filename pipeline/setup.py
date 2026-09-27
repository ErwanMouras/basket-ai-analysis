"""Explicit installation of the provisional E-BARD player/referee checkpoint."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

REVISION = "3f4789c4431aa73269f60107a4ba0a5f86b7af8b"
URL = f"https://huggingface.co/GabrieleGiudici/E-BARD-detection-models/resolve/{REVISION}/BODD_yolov8n_0001.pt"
SHA256 = "dfe3534d51bb21024d1a400c37f0c1fbf0c8b96ea9a56a5f3cb5454813bfd641"


def install(destination):
    path = Path(destination)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
            raise FileExistsError(f"Refusing to replace a different checkpoint: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".download")
    os.close(fd)
    try:
        urllib.request.urlretrieve(URL, temporary)
        if hashlib.sha256(Path(temporary).read_bytes()).hexdigest() != SHA256:
            raise ValueError("E-BARD checkpoint SHA-256 mismatch")
        # Exclusive publication: never replace an existing destination.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/players/ebard_yolov8n.pt")
    print(install(parser.parse_args().output))


if __name__ == "__main__":
    main()
