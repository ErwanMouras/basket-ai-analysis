"""Run with python -m training.ball.annotator /path/to/clip.mp4."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate the ball frame by frame with automatic JSON saving."
    )
    parser.add_argument(
        "video", type=Path, help="Local video; its .ballann.json is stored beside it."
    )
    args = parser.parse_args()
    try:
        from .app import run

        run(args.video)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
