"""Launch the player annotation editor."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "media", type=Path, help="Video, image, or directory of annotated images"
    )
    parser.add_argument("--root", type=Path, default=Path("datas"))
    parser.add_argument("--match-id")
    parser.add_argument("--venue-id")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--checkpoint", type=Path, help="Preannotate selected frames before opening the editor")
    parser.add_argument("--player-class", type=int, default=0)
    parser.add_argument("--referee-class", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    try:
        if args.checkpoint:
            import json

            from .media import MediaReader, selected_frames
            from .model import Store
            from .preannotate import preannotate_model

            reader = MediaReader(args.media)
            try:
                store = Store(reader, args.root, match_id=args.match_id,
                              venue_id=args.venue_id, split=args.split)
                report = preannotate_model(
                    store, reader, args.checkpoint,
                    selected_frames(reader.frame_count, start=args.start, stop=args.stop, step=args.step),
                    player_class=args.player_class, referee_class=args.referee_class,
                    device=args.device, confidence=args.confidence, iou=args.iou, imgsz=args.imgsz,
                )
                print(json.dumps({"sidecar": str(store.path), **report}), flush=True)
            finally:
                reader.close()

        from .app import run

        run(
            args.media,
            args.root,
            step=args.step,
            start=args.start,
            stop=args.stop,
            match_id=args.match_id,
            venue_id=args.venue_id,
            split=args.split,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
