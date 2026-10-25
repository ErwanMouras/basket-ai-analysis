"""Inventory MLflow, freeze a protocol, then compare ball checkpoints."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from training.common.files import write_json
from training.common.provenance import ROOT
from .protocol import freeze, validate


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    validate(protocol)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for candidate in protocol["candidates"]:
        directory = output / candidate["candidate_id"]
        subprocess.run([
            sys.executable, "-m", "training.ball.comparison.worker",
            "--protocol", str(protocol_path.resolve()), "--candidate", candidate["candidate_id"],
            "--output", str(directory.resolve()),
        ], cwd=ROOT, check=True)
        paths = list(directory.glob("*/metrics.json"))
        if len(paths) != 1:
            raise ValueError("Expected one completed comparison run")
        results.append(json.loads(paths[0].read_text()))
        write_json(output / "results.json", {"protocol_id": protocol["protocol_id"], "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("freeze")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    prepare.add_argument("--exports", nargs="+", default=[
        "exports/ball/tracknet-totnet", "exports/ball-v3-20261021/tracknet-totnet", "exports/ball/yolo"])
    execute = sub.add_parser("run")
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output, args.mlflow_uri, args.exports)
    else:
        run(args.protocol, args.output)


if __name__ == "__main__":
    main()
