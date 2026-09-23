"""Local YOLO inference and explicit binding of legacy NBA detection caches."""

import argparse
import json
import math
from pathlib import Path

from training.common.provenance import file_hash
from training.players.contracts import validate_annotations

from .media import MediaReader, selected_frames
from .model import Store


def proposal(index, rows, player_class):
    boxes = []
    for number, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 6:
            raise ValueError("A detection requires x1,y1,x2,y2,confidence,class_id")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) for value in row
        ):
            raise ValueError("Detection values must be finite numbers")
        class_id = row[5]
        if int(class_id) != class_id or class_id < 0 or not 0 <= row[4] <= 1:
            raise ValueError("Invalid detection class or confidence")
        if class_id == player_class:
            boxes.append(
                {
                    "object_id": f"proposal-{index}-{number}",
                    "class_id": 0,
                    "bbox": list(row[:4]),
                    "confidence": row[4],
                    "occluded": None,
                    "truncated": None,
                }
            )
    return {"frame_index": index, "review_status": "proposed", "boxes": boxes}


def _validate_frames(store, frames, provenance):
    document = store.document
    document.update(frames=frames, provenance=provenance)
    validate_annotations(document)


def preannotate_cache(
    store, cache, indices, *, player_class=0, cache_video_sha256=None
):
    if type(player_class) is not int or player_class < 0:
        raise ValueError("player_class must be a nonnegative integer")
    cache = Path(cache).resolve(strict=True)
    digest = file_hash(cache)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict):
        raise ValueError("Invalid NBA cache")
    meta, source = payload["meta"], store.document["source"]
    if type(meta.get("version")) is not int or meta["version"] != 1:
        raise ValueError("Only NBA detection cache version 1 is supported")
    info = meta.get("source_info", {})
    if not isinstance(info, dict):
        raise ValueError("Missing cache video metadata")
    for key, source_key in (
        ("width", "width"),
        ("height", "height"),
        ("total_frames", "frame_count"),
    ):
        if type(info.get(key)) is not int or info[key] != source[source_key]:
            raise ValueError(f"Cache video mismatch: {key}")
    fps = info.get("fps")
    if (
        source["kind"] != "video"
        or type(fps) not in (int, float)
        or not math.isclose(fps, source["fps"], rel_tol=0.001)
    ):
        raise ValueError("Cache video FPS mismatch")
    if (
        not isinstance(meta.get("source"), str)
        or Path(meta["source"].replace("\\", "/")).name != store.media.name
    ):
        raise ValueError("Cache source filename mismatch")
    embedded = meta.get("source_sha256")
    binding = embedded if embedded is not None else cache_video_sha256
    if binding is None:
        raise ValueError(
            "Legacy cache has no video digest: explicitly bind it with --cache-video-sha256 after checking its origin"
        )
    if binding != source["sha256"] or (
        cache_video_sha256 is not None and cache_video_sha256 != source["sha256"]
    ):
        raise ValueError("Cache video SHA-256 mismatch")
    frames = payload.get("frames")
    if (
        not isinstance(frames, list)
        or type(meta.get("frames_cached")) is not int
        or meta["frames_cached"] != len(frames)
    ):
        raise ValueError("Cache frame count mismatch")
    selected = set(indices)
    if any(type(i) is not int or not 0 <= i < source["frame_count"] for i in selected):
        raise ValueError("Selected frame outside source")
    proposals = []
    for index, frame in enumerate(frames):
        if (
            not isinstance(frame, dict)
            or type(frame.get("frame_idx")) is not int
            or frame["frame_idx"] != index
            or not isinstance(frame.get("detections"), list)
        ):
            raise ValueError(
                "NBA cache must contain a contiguous prefix of indexed frames"
            )
        proposals.append(proposal(index, frame["detections"], player_class))
    reference = json.dumps(
        {
            "cache": str(cache),
            "source": meta["source"],
            "video_sha256": binding,
            "binding": "embedded" if embedded else "operator_asserted",
            "model": meta.get("players_model"),
            "player_class": player_class,
        },
        sort_keys=True,
    )
    provenance = [{"kind": "import", "reference": reference, "sha256": digest}]
    _validate_frames(store, proposals, provenance)
    if file_hash(cache) != digest:
        raise ValueError("Cache changed while reading")
    added = store.add_proposals(
        [f for f in proposals if f["frame_index"] in selected], provenance
    )
    return {
        "added": added,
        "selected": len(selected),
        "not_cached": len(selected - set(range(len(frames)))),
    }


def preannotate_model(
    store,
    reader,
    checkpoint,
    indices,
    *,
    player_class=0,
    device="cpu",
    confidence=0.25,
    iou=0.45,
    imgsz=640,
    model_factory=None,
):
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise ValueError("Provide a local Ultralytics detection .pt checkpoint")
    if (
        type(player_class) is not int
        or player_class < 0
        or type(imgsz) is not int
        or imgsz < 32
    ):
        raise ValueError("Invalid class or input size")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1
        for v in (confidence, iou)
    ):
        raise ValueError("confidence and iou must be in (0, 1]")
    digest = file_hash(checkpoint)
    if model_factory is None:
        from ultralytics import YOLO

        model_factory = YOLO
    model = model_factory(str(checkpoint))
    if file_hash(checkpoint) != digest:
        raise ValueError("Checkpoint changed while loading")
    if model.task != "detect" or player_class not in model.names:
        raise ValueError("Checkpoint must detect the explicitly selected class")
    settings = {
        "device": device,
        "conf": confidence,
        "iou": iou,
        "imgsz": imgsz,
        "classes": [player_class],
    }
    provenance = [
        {
            "kind": "model",
            "reference": json.dumps(
                {"checkpoint": str(checkpoint), **settings}, sort_keys=True
            ),
            "sha256": digest,
        }
    ]
    count = 0
    for index in indices:
        if store.frame(index)["review_status"] != "unannotated":
            continue
        frame = reader.read(index)
        result = model.predict(frame, **settings, verbose=False)[0]
        proposed = proposal(
            index, result.boxes.data.cpu().numpy().tolist(), player_class
        )
        # Each successful frame is persisted; an interruption loses no earlier work.
        count += store.add_proposals([proposed], provenance)
    return {"added": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--root", type=Path, default=Path("datas"))
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--checkpoint", type=Path)
    choice.add_argument("--cache", type=Path)
    parser.add_argument("--cache-video-sha256")
    parser.add_argument("--match-id")
    parser.add_argument("--venue-id")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--player-class", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    reader = None
    try:
        reader = MediaReader(args.video)
        store = Store(
            reader,
            args.root,
            match_id=args.match_id,
            venue_id=args.venue_id,
            split=args.split,
        )
        indices = selected_frames(
            reader.frame_count, start=args.start, stop=args.stop, step=args.step
        )
        if args.player_class < 0:
            raise ValueError("player-class must be nonnegative")
        if args.cache:
            report = preannotate_cache(
                store,
                args.cache,
                indices,
                player_class=args.player_class,
                cache_video_sha256=args.cache_video_sha256,
            )
        else:
            report = preannotate_model(
                store,
                reader,
                args.checkpoint,
                indices,
                player_class=args.player_class,
                device=args.device,
                confidence=args.confidence,
                iou=args.iou,
                imgsz=args.imgsz,
            )
        print(json.dumps({"sidecar": str(store.path), **report}, indent=2))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    finally:
        if reader:
            reader.close()


if __name__ == "__main__":
    main()
