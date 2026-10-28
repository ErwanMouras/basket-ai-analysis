"""Normalize a trimmed MP4 for browser playback without changing inference data."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

from training.common.provenance import file_hash
from training.common.files import write_json


def prepare(run, *, encoder="h264_nvenc", fallback="libx264"):
    run = Path(run).resolve()
    manifest = json.loads((run / "run.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Playback preparation requires a completed analysis")
    source = Path(manifest["source"]["path"]).resolve(strict=True)
    if file_hash(source) != manifest["source"]["sha256"]:
        raise ValueError("Source video hash mismatch")
    destination = source.with_name(source.stem + ".viewer.mp4")
    if destination.exists() or (run / "playback.json").exists():
        raise FileExistsError("Playback already exists; preserve or remove it explicitly")
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.mp4")
    used = encoder
    failure = None
    try:
        for selected in dict.fromkeys([encoder, fallback]):
            if selected is None:
                continue
            options = (["-preset", "p4", "-cq", "20", "-b:v", "0"]
                       if selected == "h264_nvenc" else ["-preset", "fast", "-crf", "18"])
            command = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", str(source),
                       "-map", "0:v:0", "-map", "0:a:0?", "-c:v", selected, *options,
                       "-bf", "0", "-pix_fmt", "yuv420p", "-c:a", "copy",
                       "-fps_mode", "passthrough", "-movflags", "+faststart", str(temporary)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                used = selected
                break
            failure = result.stderr[-4000:]
            if selected != "h264_nvenc" or not any(
                word in failure.lower() for word in ("cuda", "nvenc", "nvidia", "device", "driver")
            ):
                raise RuntimeError(failure)
        else:
            raise RuntimeError(failure)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,start_time,width,height", "-of", "json", str(temporary)]))
        stream = probe["streams"][0]
        if int(stream["nb_read_frames"]) != manifest["frames"]:
            raise ValueError(f"Playback frame count differs: {stream['nb_read_frames']} vs {manifest['frames']}")
        if abs(float(stream["start_time"])) > 1e-6:
            raise ValueError("Playback does not start at source time zero")
        os.replace(temporary, destination)
        document = {"schema_version": 1, "source_sha256": manifest["source"]["sha256"],
                    "video_path": str(destination), "video_sha256": file_hash(destination),
                    "timestamp_policy": "source_pts", "frames": manifest["frames"],
                    "encoder": used, "gpu_encoder_failure": failure}
        write_json(run / "playback.json", document)
        return document
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.run), indent=2))


if __name__ == "__main__":
    main()
