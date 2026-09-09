"""Command-line entry point; paths are interpreted from the caller's directory."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from .config import FORMATS, ExportConfig
from .dataset import export_dataset, verify_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Export ball annotations into reproducible datasets"
    )
    parser.add_argument("--source", type=Path, default=Path("datas"))
    parser.add_argument("--output", type=Path, default=Path("exports/ball"))
    parser.add_argument(
        "--config", type=Path, default=Path("training/ball/configs/ball_export.yaml")
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("training/ball/configs/ball_training.yaml"),
    )
    parser.add_argument("--format", choices=(*FORMATS, "all"), default="all")
    parser.add_argument(
        "--tracknet-layouts", help="Comma-separated sdk,v3,v4 (overrides YAML)"
    )
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="DATASET",
        help="Verify an existing export without exporting",
    )
    args = parser.parse_args()
    try:
        if args.verify:
            print(verify_dataset(args.verify))
            return 0
        config = ExportConfig.from_file(args.config)
        if args.tracknet_layouts:
            config = replace(
                config, tracknet_layouts=tuple(args.tracknet_layouts.split(","))
            )
        formats = FORMATS if args.format == "all" else (args.format,)
        export_dataset(
            args.source,
            args.output,
            config,
            formats,
            args.training_config,
            progress=lambda message: print(message, flush=True),
        )
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
