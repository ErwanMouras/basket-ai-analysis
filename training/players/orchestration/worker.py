"""One isolated process per stage; successful work gets a durable receipt."""

import argparse
import json
import os
import signal
from pathlib import Path

from training.common.files import write_json
from training.players.progress import emit
from .state import inputs_match, inventory, lock


def verify_inputs(job):
    if not inputs_match(job["inputs"]):
        raise ValueError("Stage inputs changed")
    if job["kind"] in ("validate", "export"):
        from training.players.export.config import ExportConfig
        from training.players.export.sources import discover
        source = Path(job["config"]["source"])
        _, audit, _, _ = discover(source, ExportConfig(**job["config"]["export"]))
        expected = {p: digest for p, digest in job["inputs"].items() if Path(p).is_relative_to(source)}
        actual = {str(source / p): digest for p, digest in audit["inputs"].items()}
        if actual != expected:
            raise ValueError("Source inventory changed since stage planning")


def execute(job, directory):
    config, kind = job["config"], job["kind"]
    if kind == "validate":
        from training.players.export.config import ExportConfig
        from training.players.export.sources import discover
        _, audit, _, _ = discover(Path(config["source"]), ExportConfig(**config["export"]))
        path = directory / "audit.json"
        write_json(path, audit)
        return {"audit": str(path)}, [str(path)]
    if kind == "export":
        from training.players.export.config import ExportConfig
        from training.players.export.dataset import export_dataset
        result = export_dataset(config["source"], config["output"], ExportConfig(**config["export"]))
        return result, [result["generation"]]
    if kind == "train":
        from training.players.learning.run import train
        path = train(config, stop_after_epoch=job.get("stop_after_epoch"))
        return json.loads((path / "result.json").read_text()) | {"output": str(path)}, [str(path)]
    if kind == "evaluate":
        from training.players.evaluation.run import run
        path = run(config)
        return {"output": str(path), "run_id": json.loads((path / "result.json").read_text())["run_id"]}, [str(path)]
    if kind == "compare":
        from training.players.evaluation.compare import compare
        compare(config["runs"], config["output"], include_smoke=config["include_smoke"])
        return {"output": config["output"]}, [config["output"]]
    raise ValueError(f"Unknown stage: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    directory = args.job.parent
    job = json.loads(args.job.read_text())
    os.environ["PLAYERS_PROGRESS_PATH"] = str(directory / "live.json")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    with lock(directory / "worker.lock"):
        try:
            emit(status="RUNNING", step=job["step"])
            verify_inputs(job)
            result, roots = execute(job, directory)
            verify_inputs(job)
            write_json(directory / "receipt.json", {"fingerprint": job["fingerprint"], "result": result,
                       "artifact_roots": roots, "artifacts": inventory(roots)})
            emit(status="FINISHED", outputs=result)
        except BaseException as exc:
            live = json.loads((directory / "live.json").read_text()) if (directory / "live.json").exists() else {}
            write_json(directory / "failure.json", {"fingerprint": job["fingerprint"], "live": live,
                       "error": {"type": type(exc).__name__, "message": str(exc)}})
            emit(status="KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                 error={"type": type(exc).__name__, "message": str(exc)})
            raise


if __name__ == "__main__":
    main()
