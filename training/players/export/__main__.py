"""Export both player datasets, or verify the published pair offline."""

import argparse
import json
from pathlib import Path

import yaml

from training.players.export.config import ExportConfig
from training.players.export.dataset import export_dataset
from training.players.export.publication import verify_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datas"))
    parser.add_argument("--output", type=Path, default=Path("exports/players"))
    parser.add_argument(
        "--config", type=Path, help="Export YAML (train/val by default)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify current pair without reading source media",
    )
    args = parser.parse_args(argv)
    try:
        if args.verify:
            manifests = verify_bundle(args.output)
            result = {
                "dataset_id": manifests["yolo"]["dataset_id"],
                "export_ids": {fmt: m["export_id"] for fmt, m in manifests.items()},
                "verified": True,
            }
        else:
            config = (
                ExportConfig.from_file(args.config) if args.config else ExportConfig()
            )
            result = export_dataset(args.source, args.output, config)
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(2, f"Player export failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
