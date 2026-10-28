"""Compare CUDA implementations on the same frames and unchanged checkpoints.

The reference selects the existing unfused CUDA algorithms, not CPU variants.
It is a conservative baseline: common serialization and audit improvements
remain enabled. Both runs include identity resolution, statistics and health.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from pipeline.config import load_config
from pipeline.cuda import geometry, tracking, vision
from pipeline.cuda.run import analyze
from training.common.files import write_json


def _predict_many(kalman, tracks):
    for track in tracks:
        track.mean, track.covariance = kalman.predict(
            track.mean, track.covariance, lost=track.state == "lost"
        )


def _update_many(kalman, matches, rows):
    for track, index in matches:
        track.mean, track.covariance = kalman.update(
            track.mean, track.covariance, tracking.xyah(rows[index, :4])
        )


def reference(stack, device):
    stack.enter_context(torch.device(device))
    for target, value in (
        ("pipeline.cuda.geometry.hull", geometry._hull_reference),
        ("pipeline.cuda.geometry.optical_flow", geometry._optical_flow_reference),
        ("pipeline.cuda.geometry.Court._mask", geometry.Court._mask_reference),
        ("pipeline.cuda.geometry.Court.positions", geometry.Court._positions_reference),
        (
            "pipeline.cuda.tracking.linear_assignment",
            tracking._linear_assignment_reference,
        ),
        ("pipeline.cuda.tracking.Kalman.predict_many", _predict_many),
        ("pipeline.cuda.tracking.Kalman.update_many", _update_many),
        ("pipeline.cuda.kernels.four_point", geometry.dlt),
        ("pipeline.cuda.vision.dominant_color", vision._dominant_color_reference),
        (
            "pipeline.cuda.models.Models.prepare_graphs",
            lambda self, shape: __import__("kornia.color"),
        ),
    ):
        stack.enter_context(patch(target, value))


def measure(video, roster, config, output, frames, *, baseline=False):
    cfg = load_config(
        config, {"max_frames": frames, "mode": "sequential" if baseline else None}
    )
    with ExitStack() as stack:
        if baseline:
            reference(stack, cfg["people"]["device"])
        analyze(video, output, cfg, roster)
    manifest = json.loads((output / "run.json").read_text())
    health = json.loads((output / "health.json").read_text())
    return {
        "frames": manifest["frames"],
        "timings": manifest["timings"],
        "fps": manifest["frames"] / manifest["timings"]["perception_s"],
        "model_sha256": {
            key: value["checkpoint_sha256"]
            for key, value in manifest["models"].items()
            if "checkpoint_sha256" in value
        },
        "cuda_guard": manifest["cuda_guard"],
        "ratios": health["ratios"],
        "run": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "roster", "config", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--frames", type=int, default=150)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, baseline in (("reference", True), ("optimized", False)):
        results[name] = measure(
            args.video,
            args.roster,
            args.config,
            output / name,
            args.frames,
            baseline=baseline,
        )
        write_json(output / "benchmark.json", results)
        print(
            json.dumps({"variant": name, **results[name]}, ensure_ascii=False),
            flush=True,
        )
    if results["reference"]["model_sha256"] != results["optimized"]["model_sha256"]:
        raise ValueError("Benchmark changed a checkpoint")
    results["speedup_perception"] = (
        results["reference"]["timings"]["perception_s"]
        / results["optimized"]["timings"]["perception_s"]
    )
    results["speedup_total"] = (
        results["reference"]["timings"]["total_s"]
        / results["optimized"]["timings"]["total_s"]
    )
    write_json(output / "benchmark.json", results)


if __name__ == "__main__":
    main()
