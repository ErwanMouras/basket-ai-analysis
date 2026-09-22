"""Evaluate a V5 checkpoint on val and export predictions, metrics and review videos."""

import argparse
import json
from pathlib import Path

from training.ball.export.dataset import verify_dataset
from training.ball.export.files import write_json, write_jsonl
from training.ball.export.sources import file_hash
from training.ball.learning.data import frame_index
from training.ball.learning.tracking import log_metrics, tracked_run

from .config import load_config
from .frames import frame_result, merge_windows, report
from .video import write_review, write_videos


def evaluate(config):
    if config["split"] != "val":
        raise ValueError("Evaluation accepts only val")
    root = Path(config["dataset"])
    print("Verifying export integrity...", flush=True)
    verify_dataset(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["format"] != "tracknet-totnet"
        or "sdk" not in manifest["parameters"]["tracknet_layouts"]
    ):
        raise ValueError("V5 evaluation requires the SDK tracknet-totnet export")
    from .inference import SDKFrames, V5Predictor

    predictor = V5Predictor(config, manifest)
    dataset = SDKFrames(root, predictor.geometry)
    records = sorted(
        frame_index(root, "val").values(),
        key=lambda frame: (frame["clip_id"], frame["frame_index"]),
    )
    sources = {
        source["clip_id"]: source
        for source in manifest["sources"]
        if source["split"] == "val"
    }
    resolved = {
        **config,
        **predictor.provenance,
        "schema_version": 1,
        "purpose": "evaluation",
        "dataset_id": manifest["dataset_id"],
        "export_id": manifest["export_id"],
        "checkpoint_sha256": file_hash(Path(config["checkpoint"])),
        "fusion": "mean_heatmap",
        "decoder": "largest_component_bbox_center",
        "sequence_stride": 1,
        "temporal_mode": "offline",
        "trajectory_filter": "none",
        "annotation_policy": {
            key: manifest["parameters"][key]
            for key in ("unknown_position", "occluded_position")
        },
    }
    with tracked_run(resolved, manifest) as (client, run_id, output):
        client.set_tag(
            run_id, "training_run_id", predictor.provenance["training_run_id"]
        )
        print(
            f"Inferring {len(dataset)} windows over {len(records)} val frames...",
            flush=True,
        )
        predictions = dict(
            merge_windows(
                dataset.samples, predictor.windows(dataset), config["threshold"]
            )
        )
        rows = [
            frame_result(
                frame,
                sources[frame["clip_id"]],
                predictions.get(frame["image"]),
                predictor.geometry,
                config,
            )
            for frame in records
        ]
        metrics = report(rows)
        write_jsonl(output / "predictions.jsonl", rows)
        write_json(output / "metrics.json", metrics)
        write_review(output, rows)
        values = {
            f"val/{key}": value
            for key, value in metrics["overall"].items()
            if value is not None
        }
        for status, group in metrics["by_status"].items():
            values.update(
                {
                    f"val/status/{status}/{key}": value
                    for key, value in group.items()
                    if value is not None
                }
            )
        log_metrics(client, run_id, values, 0)
        for name in ("predictions.jsonl", "metrics.json", "review.csv"):
            client.log_artifact(run_id, str(output / name), "evaluation")
        if config["videos"]:
            print("Rendering review videos...", flush=True)
            write_videos(root, output, rows, sources, config["video_max_width"])
            client.log_artifact(run_id, str(output / "video_frames.csv"), "evaluation")
            client.log_artifacts(run_id, str(output / "videos"), "evaluation/videos")
        print(json.dumps(metrics["overall"], indent=2), flush=True)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    evaluate(load_config(args.config, args.checkpoint))


if __name__ == "__main__":
    main()
