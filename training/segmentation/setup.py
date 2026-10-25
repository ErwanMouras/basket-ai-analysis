"""Explicit, checksum-pinned installation of the experiment checkpoints."""
import argparse
import os
from pathlib import Path
import tempfile
import urllib.request
from training.common.provenance import file_hash

# Official Ultralytics release assets; pin observed content, never silently update.
CHECKPOINTS = {
    "sam2.1_t.pt": "3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3",
    "yolo26l-seg.pt": "636024306410afa1732692322fba57d22ea2b1c2f07613fcee131a93d7dd380c",
    "yolo26x-seg.pt": "92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3",
}
BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"


def install(name, directory):
    expected = CHECKPOINTS[name]
    destination = Path(directory) / name
    if destination.exists():
        if file_hash(destination) != expected:
            raise FileExistsError(f"Checkpoint differs; refusing replacement: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".download")
    os.close(fd)
    try:
        urllib.request.urlretrieve(BASE_URL + name, temporary)
        if file_hash(Path(temporary)) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="models/segmentation")
    args = parser.parse_args()
    for name in CHECKPOINTS:
        print(install(name, args.directory))


if __name__ == "__main__":
    main()
