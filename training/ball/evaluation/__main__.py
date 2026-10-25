"""Evaluate a V5 checkpoint on val and export predictions, metrics and review videos."""

import argparse
import json
from pathlib import Path

from training.ball.export.dataset import verify_dataset
from training.ball.export.files import write_json, write_jsonl
from training.ball.export.sources import file_hash
from training.ball.learning.data import frame_index
from training.ball.learning.tracking import log_metrics, tracked_run

from .config import load_config, frozen_payload, require_frozen
from .frames import causal_windows, frame_result, merge_windows, report
from .video import write_review, write_videos


def prepare(config):
    root = Path(config["dataset"]).resolve(strict=True)
    output = Path(config["output"]).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Dataset and output must be disjoint")
    print("Verifying export integrity...", flush=True)
    verify_dataset(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["format"] != "tracknet-totnet"
        or "sdk" not in manifest["parameters"]["tracknet_layouts"]
    ):
        raise ValueError("V5 evaluation requires the SDK tracknet-totnet export")
    records = sorted(
        frame_index(root, config["split"], allow_test=True).values(),
        key=lambda frame: (frame["clip_id"], frame["frame_index"]),
    )
    if not records:
        raise ValueError("Evaluation selection is empty")
    return root, manifest, records


def evaluate(config):
    root, manifest, records = prepare(config)
    frozen = frozen_payload(config, manifest, records)
    require_frozen(config, frozen)
    from .inference import SDKFrames, V5Predictor

    predictor = V5Predictor(config, manifest)
    dataset = SDKFrames(root, predictor.geometry, config["split"], config["cut_frames"])
    sources = {
        source["clip_id"]: source
        for source in manifest["sources"]
        if source["split"] == config["split"]
    }
    resolved = {
        **config,
        **predictor.provenance,
        "schema_version": 1,
        "purpose": "evaluation",
        "dataset_id": manifest["dataset_id"],
        "export_id": manifest["export_id"],
        "checkpoint_sha256": file_hash(Path(config["checkpoint"])),
        "fusion": "last_head" if config["temporal_policy"] == "causal_last" else "mean_heatmap",
        "decoder": "largest_component_bbox_center",
        "sequence_stride": 1,
        "temporal_mode": config["temporal_policy"],
        "coverage_scope": "exported frames only; missing source context restarts windows; cuts must be supplied",
        "evaluation_identity": frozen,
        "trajectory_filter": "none",
        "annotation_policy": {
            key: manifest["parameters"][key]
            for key in ("unknown_position", "occluded_position")
        },
    }
    with tracked_run(resolved, manifest) as (client, run_id, output):
        write_json(output / "protocol.json", frozen)
        client.set_tag(
            run_id, "training_run_id", predictor.provenance["training_run_id"]
        )
        print(
            f"Inferring {len(dataset)} windows over {len(records)} {config['split']} frames...",
            flush=True,
        )
        predictions = dict(
            (causal_windows if config["temporal_policy"] == "causal_last" else merge_windows)(
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
        _, current_manifest, current_records = prepare(config)
        if frozen_payload(config, current_manifest, current_records) != frozen:
            raise ValueError("Checkpoint, dataset or evaluator changed during evaluation")
        metrics = report(rows)
        write_jsonl(output / "predictions.jsonl", rows)
        write_json(output / "metrics.json", metrics)
        write_review(output, rows)
        values = {
            f"{config['split']}/{key}": value
            for key, value in metrics["overall"].items()
            if value is not None
        }
        for status, group in metrics["by_status"].items():
            values.update(
                {
                    f"{config['split']}/status/{status}/{key}": value
                    for key, value in group.items()
                    if value is not None
                }
            )
        log_metrics(client, run_id, values, 0)
        for name in ("predictions.jsonl", "metrics.json", "review.csv", "protocol.json"):
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
    parser.add_argument("--freeze-recipe", type=Path, help="Write the selection/weight/code freeze without inference")
    args = parser.parse_args()
    config = load_config(args.config, args.checkpoint)
    if args.freeze_recipe:
        if args.freeze_recipe.exists():
            parser.error("Use a new freeze path")
        _, manifest, records = prepare(config)
        write_json(args.freeze_recipe, frozen_payload(config, manifest, records))
    else:
        evaluate(config)


if __name__ == "__main__":
    main()
