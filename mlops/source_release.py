"""Package explicit project sources and the synthetic demo, excluding local data."""

import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path

from training.common.files import write_json
from training.common.provenance import ROOT, file_hash, git_state, object_hash

ROOT_FILES = {"README.md", "Makefile", "requirements.txt", ".gitignore", ".gitattributes", ".dockerignore", ".dvcignore", "datas.dvc", "compose.yaml"}
FOLDERS = {"pipeline", "training", "viewer", "mlops", "scripts", "configs", "deploy", "docs", "examples", "imagesReadMe", "reports", ".github"}
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".in", ".lock", ".txt", ".html", ".js", ".css", ".gif", ".png", ".svg", ".csv"}


def source_paths():
    result = []
    paths = {ROOT / name for name in ROOT_FILES}
    paths.update(ROOT / name for name in (".dvc/config", ".dvc/.gitignore"))
    for folder in FOLDERS:
        paths.update((ROOT / folder).rglob("*"))
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        if relative.parts[0] not in FOLDERS and relative.as_posix() not in ROOT_FILES and relative.as_posix() not in (".dvc/config", ".dvc/.gitignore"):
            continue
        if path.is_symlink() or not path.is_file() or "__pycache__" in relative.parts:
            continue
        if (relative.as_posix() in ROOT_FILES or relative.as_posix() in (".dvc/config", ".dvc/.gitignore")
                or path.suffix in SUFFIXES or path.name in ("Makefile", "LICENSE", "NOTICE") or path.name.startswith("Dockerfile")
                or relative.as_posix() == "examples/demo/videos/synthetic.mp4"):
            result.append(path)
    return result


def create(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new source archive")
    paths = source_paths()
    manifest = {"schema_version": 1, "git": git_state(ROOT),
                "files": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths},
                "scope": "Source and scripted demo only; no private datasets, credentials, learned weights or environments."}
    manifest["source_id"] = object_hash(manifest["files"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in paths:
                name = path.relative_to(ROOT).as_posix()
                content = path.read_bytes()
                if file_hash(path) != manifest["files"][name]:
                    raise ValueError("Source changed during packaging")
                info = tarfile.TarInfo("vyneros/" + name)
                info.size, info.mode = len(content), 0o644
                archive.addfile(info, io.BytesIO(content))
            content = (json.dumps(manifest, indent=2) + "\n").encode()
            info = tarfile.TarInfo("vyneros/SOURCE_MANIFEST.json")
            info.size, info.mode = len(content), 0o644
            archive.addfile(info, io.BytesIO(content))
    write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(create(args.output)["source_id"])


if __name__ == "__main__":
    main()
