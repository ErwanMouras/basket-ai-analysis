"""Copy the selected local models into a checksum-verified, portable bundle."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import yaml

from pipeline.config import load_config
from training.common.files import write_json
from training.common.provenance import ROOT, file_hash, object_hash


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("bundle_id") != object_hash({k: v for k, v in manifest.items() if k != "bundle_id"}):
        raise ValueError("Invalid bundle identity")
    for artifact in manifest["artifacts"]:
        path = (root / artifact["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or file_hash(path) != artifact["sha256"]:
            raise ValueError(f"Model bundle integrity mismatch: {artifact['path']}")
    reference = root / ".external" / "tracknet_v5"
    needs_reference = any(artifact.get("role") == "ball" for artifact in manifest["artifacts"])
    if needs_reference and not reference.is_dir():
        raise ValueError("Model bundle is missing its TrackNet reference")
    if reference.exists():
        actual = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if actual != manifest["tracknet_reference"]["revision"] or dirty:
            raise ValueError("Model reference revision or working tree differs")
    return manifest


def create(config_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new model bundle directory")
    config = load_config(config_path)
    if config["people"]["adapter"] != "yolo":
        raise ValueError("The portable CPU runtime requires a local YOLO people checkpoint; use the players environment for RF-DETR or registry inference")
    sources = []
    for section in ("people", "ball", "court", "pose", "jersey"):
        if section != "people" and not config[section]["enabled"]:
            continue
        source = Path(config[section]["weights"]).resolve(strict=True)
        destination = Path("models") / section / ("weights" + source.suffix)
        sources.append((section, source, destination, file_hash(source)))
    reference = json.loads((ROOT / "training/ball/configs/references.json").read_text())["tracknet_v5"]
    output.mkdir(parents=True)
    artifacts = []
    for section, source, destination, digest in sources:
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if file_hash(target) != digest or file_hash(source) != digest:
            raise ValueError("Model changed during bundle creation")
        config[section]["weights"] = destination.as_posix()
        config[section]["device"] = "cpu"
        artifacts.append({"role": section, "path": destination.as_posix(), "sha256": digest,
                          "bytes": target.stat().st_size, "origin": source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else source.name})
    if config["ball"]["enabled"]:
        source = Path(config["ball"]["reference_root"])
        shutil.copytree(source, output / ".external/tracknet_v5", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        config["ball"]["reference_root"] = ".external/tracknet_v5"
    config["gpu_concurrency"] = 1
    config["mode"] = "sequential"
    (output / "analysis.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    artifacts.append({"role": "configuration", "path": "analysis.yaml", "sha256": file_hash(output / "analysis.yaml")})
    manifest = {"schema_version": 1, "artifacts": artifacts, "tracknet_reference": reference,
                "scope": "Provisional local inference bundle. Redistribution rights are not granted by this manifest.",
                "source_config_sha256": file_hash(Path(config_path))}
    manifest["bundle_id"] = object_hash(manifest)
    write_json(output / "manifest.json", manifest)
    return verify(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("create")
    build.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    build.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    manifest = create(args.config, args.output) if args.command == "create" else verify(args.bundle)
    print(json.dumps({"bundle_id": manifest["bundle_id"], "artifacts": len(manifest["artifacts"])}, indent=2))


if __name__ == "__main__":
    main()
