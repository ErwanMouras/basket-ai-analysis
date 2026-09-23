"""Evaluate a detector, freeze a final-test recipe, or compare compatible runs."""

import argparse
from pathlib import Path

from training.common.files import write_json
from .compare import compare
from .config import frozen_payload, load_config
from .run import prepare, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--freeze-recipe", type=Path,
                          help="Write a locked recipe without inference; review/version it before final test")
    comparison = commands.add_parser("compare")
    comparison.add_argument("runs", nargs="+", type=Path)
    comparison.add_argument("--output", required=True, type=Path)
    comparison.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "compare":
        compare(args.runs, args.output, include_smoke=args.include_smoke)
        print(args.output / "comparison.html")
        return
    config = load_config(args.config, args.checkpoint)
    if args.freeze_recipe:
        if config["split"] != "test":
            parser.error("Only a final-test recipe can be frozen")
        if args.freeze_recipe.exists():
            parser.error("Refusing to overwrite a frozen recipe")
        _, _, _, identity, _ = prepare(config)
        write_json(args.freeze_recipe, frozen_payload(config, **identity))
        print(f"Frozen without inference: {args.freeze_recipe}")
    else:
        print(run(config) / "report.html")


if __name__ == "__main__":
    main()
