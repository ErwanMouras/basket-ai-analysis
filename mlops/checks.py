"""Run offline regression suites with all real-training opt-ins disabled."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from training.common.files import write_json
from training.common.provenance import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PLAYERS_", "BALL_", "MLFLOW_"))}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled", MPLBACKEND="Agg",
               MLFLOW_TRACKING_URI=f"sqlite:///{args.output.resolve() / 'tests.db'}")
    suites = [("players", ["discover", "-s", "training/players/tests"]),
              ("ball", ["discover", "-s", "training/ball/tests"]),
              ("court", ["discover", "-s", "training/court/tests"]),
              ("jersey", ["discover", "-s", "training/jersey/tests"]),
              ("analysis", ["discover", "-s", "pipeline/tests"]),
              ("mlops", ["discover", "-s", "mlops/tests"]), ("viewer", ["viewer.test_server"])]
    report = {"training_executed": False, "annotation_sources_modified": False, "suites": [], "status": "running"}
    for name, options in suites:
        start = time.monotonic()
        result = subprocess.run([sys.executable, "-m", "unittest", *options, "-v"], cwd=ROOT,
                                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (args.output / f"{name}.log").write_text(result.stdout)
        found = re.search(r"Ran (\d+) tests?", result.stdout)
        skipped = re.search(r"skipped=(\d+)", result.stdout)
        entry = {"name": name, "returncode": result.returncode, "tests": int(found[1]) if found else 0,
                 "skipped": int(skipped[1]) if skipped else 0, "seconds": time.monotonic() - start}
        report["suites"].append(entry)
        write_json(args.output / "summary.json", report)
        print(json.dumps(entry), flush=True)
    result = subprocess.run(["node", "--check", "viewer/app.js"], cwd=ROOT, capture_output=True, text=True)
    (args.output / "javascript.log").write_text(result.stdout + result.stderr)
    report["javascript_returncode"] = result.returncode
    report["status"] = "passed" if all(s["returncode"] == 0 and s["tests"] > 0 for s in report["suites"]) and result.returncode == 0 else "failed"
    write_json(args.output / "summary.json", report)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
