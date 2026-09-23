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
    args = parser.parse_args()
    try:
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
