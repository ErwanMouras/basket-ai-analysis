"""Reproducible, offline synthetic validation. No project dataset is consumed."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from training.common.files import write_json
from training.common.provenance import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite a validation")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--family", choices=("both", "yolo", "rfdetr"), default="both")
    parser.add_argument("--gpu-smoke", action="store_true", help="Explicit opt-in required with cuda:0")
    args = parser.parse_args()
    if (args.device != "cpu") != args.gpu_smoke:
        parser.error("Use --device cuda:0 --gpu-smoke together to enable GPU tests")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PLAYERS_", "BALL_", "MLFLOW_"))}
    env.update(PYTHONPATH=os.pathsep.join([str(Path(__file__).parent / "offline"), str(ROOT)]),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1", MPLBACKEND="Agg", WANDB_MODE="disabled", OMP_NUM_THREADS="2",
        MLFLOW_TRACKING_URI=f"sqlite:///{root / 'unit-mlflow.db'}",
        MLFLOW_ENABLE_ASYNC_LOGGING="false", PLAYERS_MLFLOW_TESTS="1")
    for name in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "YOLO_CONFIG_DIR"):
        env[name] = str(root / "cache" / name.lower())
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    summary = {"synthetic_only": True, "device": args.device, "family": args.family,
               "weights": "locally generated random initialization", "steps": [], "status": "RUNNING"}
    started = time.monotonic()

    def command(name, module, options, extra=None, report=False):
        argv = [sys.executable, "-m", module, *map(str, options)]
        if report:
            argv += ["--report", str(root / f"{name}.json")]
        print(f"{name}: running (log: {root / (name + '.log')})", flush=True)
        with (root / f"{name}.log").open("w") as log:
            done = subprocess.run(argv, env={**env, **(extra or {})}, cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT)
        entry = {"name": name, "returncode": done.returncode}
        if report and (root / f"{name}.json").exists():
            entry.update(json.loads((root / f"{name}.json").read_text()))
        summary["steps"].append(entry)
        write_json(root / "summary.json", summary)
        if done.returncode:
            raise RuntimeError(f"{name} failed; inspect {root / (name + '.log')}")
        if report and not entry.get("passed"):
            raise RuntimeError(f"{name} ran no successful test")

    try:
        command("players", "training.players.tests.runner", ["--discover", "training/players/tests"], report=True)
        command("ball", "training.players.tests.runner", ["--discover", "training/ball/tests"], report=True)
        for family in (("yolo", "rfdetr") if args.family == "both" else (args.family,)):
            weights = root / "weights" / (family + (".pt" if family == "yolo" else ".pth"))
            command(f"weights-{family}", "training.players.tests.training_fixtures", [family, weights])
            command(f"pipeline-{family}", "training.players.tests.runner",
                    ["--test", "training.players.tests.test_pipeline.RealPipelineTests"],
                    {"PLAYERS_PIPELINE_TESTS": family, "PLAYERS_TEST_WEIGHTS": str(weights),
                     "PLAYERS_TEST_DEVICE": args.device, "PLAYERS_TEST_OUTPUT": str(root / family)}, report=True)
        if args.family == "both":
            evaluations = [json.loads((root / family / "validation.json").read_text())["evaluation"]
                           for family in ("yolo", "rfdetr")]
            for name, flags in (("comparison-default", []), ("comparison-technical", ["--include-smoke"])):
                command(name, "training.players.evaluation",
                        ["compare", *evaluations, "--output", root / name, *flags])
            default = json.loads((root / "comparison-default/comparison.json").read_text())
            technical = json.loads((root / "comparison-technical/comparison.json").read_text())
            if default["rows"] or len(default["excluded"]) != 2:
                raise AssertionError("Both synthetic smoke runs must be excluded by default")
            if len(technical["rows"]) != 2 or len(technical["protocol_groups"]) != 1:
                raise AssertionError("The two fixture detectors must share exactly one comparison protocol")
        summary["status"] = "PASSED"
    except BaseException as exc:
        summary.update(status="KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                       error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        summary["seconds"] = time.monotonic() - started
        write_json(root / "summary.json", summary)
        print(f"Validation {summary['status']}: {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
