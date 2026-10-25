"""Build and query the viewer from an extracted source release, not the workspace."""

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

from training.common.files import write_json


def verify_archive(archive, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new verification report")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "archive": str(archive)}
    container = None
    try:
        with tempfile.TemporaryDirectory(prefix="vyneros-source-check-") as temporary:
            root = Path(temporary)
            with tarfile.open(archive) as bundle:
                bundle.extractall(root, filter="data")
            source = root / "vyneros"
            manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text())
            for name, digest in manifest["files"].items():
                path = (source / name).resolve(strict=True)
                if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Archive manifest mismatch: {name}")
            if (source / "datas").exists() or (source / ".git").exists() or any(source.rglob("config.local")):
                raise ValueError("Source release contains private/local directories")
            if any(p.suffix in (".pt", ".pth", ".onnx", ".engine") for p in source.rglob("*")):
                raise ValueError("Source release contains learned model weights")
            image = "vyneros-viewer:source-check"
            with output.with_suffix(".build.log").open("w") as log:
                subprocess.run(["docker", "build", "-f", "deploy/Dockerfile.viewer", "-t", image, "."],
                               cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
            container = "vyneros-source-check-" + uuid4().hex[:12]
            subprocess.run(["docker", "run", "--rm", "-d", "--name", container, "-p", "127.0.0.1::8765",
                            "-v", f"{source}/examples/demo/runs:/data/runs:ro",
                            "-v", f"{source}/examples/demo/videos:/data/videos:ro", image], check=True, capture_output=True)
            port = subprocess.check_output(["docker", "port", container, "8765/tcp"], text=True).strip().rsplit(":", 1)[1]
            base = f"http://127.0.0.1:{port}"
            for _ in range(50):
                try:
                    with urlopen(base + "/api/runs", timeout=2) as response:
                        listed = json.load(response)
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError("Clean viewer failed to start")
            with urlopen(base + "/api/run?name=synthetic") as response:
                run = json.load(response)
            if run["video_available"] is not True:
                raise ValueError("Portable source video unavailable")
            with urlopen(Request(base + "/api/video?name=synthetic", headers={"Range": "bytes=0-15"})) as response:
                if response.status != 206 or len(response.read()) != 16:
                    raise ValueError("Portable range playback failed")
            report.update(status="passed", source_id=manifest["source_id"], files_verified=len(manifest["files"]),
                          no_private_dataset=True, no_learned_weights=True, no_local_credentials=True,
                          viewer_build="passed", portable_video_api="passed", sample_run_count=len(listed["runs"]))
    except BaseException as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if container:
            subprocess.run(["docker", "stop", container], capture_output=True)
        write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
