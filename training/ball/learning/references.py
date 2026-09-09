"""Install and verify immutable upstream implementations without patching them."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "training/ball/configs/references.json"


def reference_key(model):
    return "tracknet_v5" if model == "tracknet_v5_totnet" else model


def activate_reference(config):
    key = reference_key(config["model"])
    reference = json.loads(LOCK.read_text())[key]
    path = Path(config["reference_root"])
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != reference["revision"] or dirty:
        raise ValueError(
            f"Expected clean {key} revision {reference['revision']} at {path}"
        )
    config["reference"] = reference
    sys.path.insert(0, str(path / "src" if key == "tracknet_v4" else path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".external")
    args = parser.parse_args()
    for name, reference in json.loads(LOCK.read_text()).items():
        path = args.output / name
        if path.exists():
            activate_reference({"model": name, "reference_root": str(path)})
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", reference["url"], str(path)], check=True)
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", reference["revision"]],
            check=True,
        )


if __name__ == "__main__":
    main()
