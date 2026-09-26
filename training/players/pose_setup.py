"""Explicit, checksum-verified installation of the RTMPose-M reference model."""

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from training.common.files import atomic_writer
from training.common.provenance import ROOT, file_hash
from training.players.pose import DEFAULTS, MODEL_SHA256, MODEL_URL


def install(destination, *, source=None):
    destination = (ROOT / Path(destination).expanduser()).resolve()
    if destination.exists():
        if file_hash(destination) != MODEL_SHA256:
            raise ValueError(
                "Existing pose model has a different checksum; choose another output"
            )
        return destination
    with tempfile.TemporaryDirectory(prefix="players-pose-") as temp:
        if source is None:
            archive = Path(temp) / "model.zip"
            with (
                urllib.request.urlopen(MODEL_URL, timeout=30) as response,
                archive.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
            with zipfile.ZipFile(archive) as bundle:
                members = [n for n in bundle.namelist() if n.endswith(".onnx")]
                if len(members) != 1:
                    raise ValueError("Expected one ONNX model in the reference archive")
                model = Path(temp) / "model.onnx"
                with bundle.open(members[0]) as reader, model.open("wb") as handle:
                    shutil.copyfileobj(reader, handle)
        else:
            model = Path(source).expanduser().resolve(strict=True)
        # Verify the exact bytes being published, including when importing locally.
        with (
            model.open("rb") as reader,
            atomic_writer(destination, binary=True) as handle,
        ):
            digest = hashlib.sha256()
            while block := reader.read(1024 * 1024):
                digest.update(block)
                handle.write(block)
            if digest.hexdigest() != MODEL_SHA256:
                raise ValueError("RTMPose checkpoint SHA-256 mismatch")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="Import a local reference ONNX instead of downloading",
    )
    parser.add_argument("--output", type=Path, default=Path(DEFAULTS["weights"]))
    args = parser.parse_args()
    print(install(args.output, source=args.source))


if __name__ == "__main__":
    main()
