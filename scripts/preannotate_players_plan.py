"""Preannotate the selected phase of a video inventory CSV without opening Tk."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from training.common.files import write_json
from training.common.provenance import file_hash
from training.players.annotator.media import MediaReader
from training.players.annotator.model import Store
from training.players.annotator.preannotate import class_mapping, preannotate_model


def plan_indices(plan, root, phases):
    groups = defaultdict(set)
    with plan.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"phase", "clip", "frame_index"} <= set(reader.fieldnames or []):
            raise ValueError("Plan requires phase, clip and frame_index columns")
        for row in reader:
            if int(row["phase"]) not in phases:
                continue
            video = (root / row["clip"] / "video.mp4").resolve(strict=True)
            if not video.is_relative_to(root.resolve()):
                raise ValueError("Plan video must be within the dataset root")
            index = int(row["frame_index"])
            if index < 0:
                raise ValueError("Frame index must be nonnegative")
            groups[video].add(index)
    if not groups:
        raise ValueError("No frames in the selected phase(s)")
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("datas"))
    parser.add_argument("--phase", action="append", type=int, choices=(1, 2, 3),
                        help="Repeat for multiple phases; defaults to train phase 1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--player-class", type=int, default=0)
    parser.add_argument("--referee-class", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a new report path")
    if args.report.resolve().is_relative_to(args.source.resolve()):
        parser.error("Write the report outside the source dataset")
    class_mapping(args.player_class, args.referee_class)
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    import torch

    torch.set_num_threads(args.cpu_threads)
    groups = plan_indices(args.plan, args.source, set(args.phase or [1]))
    checkpoint_hash = file_hash(args.checkpoint)
    report = {"status": "running", "plan": str(args.plan), "plan_sha256": file_hash(args.plan),
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
              "player_class": args.player_class, "referee_class": args.referee_class,
              "device": args.device, "phases": sorted(set(args.phase or [1])), "videos": []}
    try:
        for video, indices in sorted(groups.items()):
            print(f"Preannotation: {video} ({len(indices)} selected frames)", flush=True)
            reader = MediaReader(video)
            try:
                store = Store(reader, args.source)
                if any(index >= reader.frame_count for index in indices):
                    raise ValueError(f"Plan index outside source: {video}")
                if file_hash(args.checkpoint) != checkpoint_hash:
                    raise ValueError("Checkpoint changed during the batch")
                result = preannotate_model(
                    store, reader, args.checkpoint, sorted(indices), player_class=args.player_class,
                    referee_class=args.referee_class, device=args.device,
                )
                entry = {"video": str(video), "sidecar": str(store.path), "selected": len(indices),
                         "source_sha256": store.document["source"]["sha256"],
                         "sidecar_sha256": file_hash(store.path), **result}
                report["videos"].append(entry)
                write_json(args.report, report)
                print(json.dumps(entry), flush=True)
            finally:
                reader.close()
        report["status"] = "finished"
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(args.report, report)


if __name__ == "__main__":
    main()
