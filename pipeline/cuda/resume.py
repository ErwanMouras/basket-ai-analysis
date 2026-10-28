"""Retain complete camera segments; replay the final segment to recover state.

Scene cuts reset court propagation, track filters and online identity votes.
The source directory is never mutated.
"""

import hashlib
import json
import os
from pathlib import Path

from pipeline.cuda.integrity import observation_metadata
from training.common.provenance import file_hash


def inspect_prefix(source, config, roster_sha256):
    source = Path(source).resolve(strict=True)
    manifest = json.loads((source / "run.json").read_text())
    if manifest.get("backend") != "native_cuda":
        raise ValueError("Resume requires a native CUDA run")
    if manifest.get("status") in {"running", "validating"} and manifest.get("pid"):
        try:
            os.kill(manifest["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("Cannot resume a source run that is still active")
    if manifest["roster_sha256"] != roster_sha256:
        raise ValueError("Resume roster differs from the original run")
    for key in manifest["config"]:
        if key not in {"mode", "workers", "gpu_concurrency", "max_frames"} and (
            manifest["config"][key] != config.get(key)
        ):
            raise ValueError(f"Resume config mismatch: {key}")
    observations = source / "observations.partial.jsonl"
    if not observations.is_file():
        observations = source / "observations.jsonl"
    offset = start_offset = start = segment = counter = prefix_counter = count = 0
    digest = hashlib.sha256()
    prefix_sha256 = digest.hexdigest()
    source_timestamps = []
    with observations.open("rb") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if stream.read(1):
                    raise ValueError("Corrupt observation prefix") from None
                break
            observation_metadata(row, manifest["run_id"], count)
            if config["max_frames"] and count >= config["max_frames"]:
                break
            if row["scene_cut"]:
                prefix_sha256 = digest.hexdigest()
                start, start_offset, segment, prefix_counter = (
                    count,
                    offset,
                    row["segment_id"],
                    counter,
                )
            for person in row["persons"]:
                if person["track_id"] is not None:
                    counter = max(counter, person["track_id"])
            offset += len(line)
            digest.update(line)
            source_timestamps.append(row.get("timestamp_s"))
            count += 1
    return {
        "source": source,
        "manifest": manifest,
        "observations": observations,
        "bytes": start_offset,
        "start": start,
        "segment": segment,
        "track_counter": prefix_counter,
        "available_frames": count,
        "manifest_sha256": file_hash(source / "run.json"),
        "prefix_sha256": prefix_sha256,
        "timestamps": source_timestamps[:start],
    }


def copy_prefix(checkpoint, output):
    output = Path(output)
    remaining = checkpoint["bytes"]
    digest = hashlib.sha256()
    with (
        checkpoint["observations"].open("rb") as source,
        (output / "observations.partial.jsonl").open("wb") as target,
    ):
        while remaining:
            chunk = source.read(min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Resume observation prefix was truncated")
            target.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    if digest.hexdigest() != checkpoint["prefix_sha256"]:
        raise ValueError("Copied observation prefix failed its integrity check")
    reads = checkpoint["source"] / "jersey_reads.partial.jsonl"
    if not reads.is_file():
        reads = checkpoint["source"] / "jersey_reads.jsonl"
    with (
        reads.open("rb") as source,
        (output / "jersey_reads.partial.jsonl").open("wb") as target,
    ):
        for line in source:
            row = json.loads(line)
            if row["frame_index"] >= checkpoint["start"]:
                break
            target.write(line)
