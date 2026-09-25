"""Sequential stages, isolated subprocesses and content-addressed reuse."""

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from training.common.files import write_json
from training.common.provenance import ROOT, file_hash, object_hash
from training.players.export.config import ExportConfig
from training.players.export.sources import discover
from .config import load_config
from .state import input_files, inventory, lock, receipt_valid


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


class Pipeline:
    def __init__(self, config, *, restart_incomplete=False, stop_after_epoch=None):
        self.config = config
        self.root = Path(config["output"])
        self.restart_incomplete = restart_incomplete
        self.stop_after_epoch = stop_after_epoch
        self.started = time.monotonic()
        self.progress = {}
        self.code = list((ROOT / "training/players").rglob("*.py")) + list((ROOT / "training/common").glob("*.py"))
        self.code = [p for p in self.code if "tests" not in p.parts]
        self.environment = {"python": platform.python_version(), "platform": platform.platform(),
            "host": platform.node(), "packages": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
            "tracking_uri_override": os.environ.get("MLFLOW_TRACKING_URI"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        try:
            self.environment["gpu"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            self.environment["gpu"] = None

    def save(self, **values):
        self.progress.update(values)
        self.progress["elapsed_seconds"] = time.monotonic() - self.started
        self.progress["updated_at"] = time.time()
        write_json(self.root / "progress.json", self.progress)

    def stage(self, name, kind, config, inputs):
        inputs = {**input_files(self.code), **inputs}
        fingerprint = object_hash({"config": config, "inputs": inputs, "environment": self.environment, "kind": kind})
        entry = self.progress["steps"].setdefault(name, {"attempts": []})
        self.save(step=name, run_id=None, epoch=None, epochs_total=None, epochs_completed=None, batch=None,
                  batches_total=None, eta_seconds=None, outputs={}, error=None)
        previous = None
        for attempt in reversed(entry["attempts"]):
            directory = Path(attempt["directory"])
            pid = attempt.get("pid")
            command = Path(f"/proc/{pid}/cmdline")
            if pid and command.exists() and str(directory / "job.json").encode() in command.read_bytes():
                raise ValueError(f"An existing stage process is still running: {pid}")
            # An orphan worker may still be doing useful work after its parent died.
            with lock(directory / "worker.lock"):
                pass
            receipt = read(directory / "receipt.json")
            if receipt and receipt_valid(receipt, fingerprint):
                entry.update(status="REUSED", fingerprint=fingerprint, result=receipt["result"],
                             artifacts=receipt["artifacts"], directory=str(directory))
                self.save(outputs=receipt["result"], run_id=receipt["result"].get("run_id"))
                return receipt["result"]
            live = read(directory / "live.json", {})
            if (previous is None and attempt["fingerprint"] == fingerprint and receipt is None
                    and (live.get("run_id") or live.get("checkpoints"))):
                previous = directory
        effective = dict(config)
        resume_inputs = {}
        if previous and kind == "train" and not self.restart_incomplete:
            live = read(previous / "live.json", {})
            candidates = live.get("checkpoints", {})
            last = next((Path(p) for p in candidates if Path(p).name in ("last.pt", "last.ckpt")), None)
            best = last.parent / ("best.pt" if config["model"] == "yolo" else "best.ckpt") if last else None
            if last and str(best) in candidates and all(Path(p).is_file() and file_hash(Path(p)) == digest for p, digest in candidates.items()):
                effective.update(mode="resume", resume=str(last))
                resume_inputs = input_files([last, best])
            elif live.get("run_id") and not self.restart_incomplete:
                raise ValueError("Interrupted training has no verified resumable checkpoint; inspect it, then explicitly use --restart-incomplete to restart the original recipe")
        directory = self.root / "attempts" / name / uuid.uuid4().hex
        directory.mkdir(parents=True)
        if kind in ("train", "evaluate"):
            effective["output"] = str(directory / "runs")
        elif kind == "compare":
            effective["output"] = str(directory / "report")
        # Export uses a stable publication root, independent of the attempt.
        job = {"step": name, "kind": kind, "config": effective,
               "inputs": {**inputs, **resume_inputs}, "fingerprint": fingerprint}
        if kind == "train" and self.stop_after_epoch is not None:
            job["stop_after_epoch"] = self.stop_after_epoch
        write_json(directory / "job.json", job)
        entry["attempts"].append({"directory": str(directory), "fingerprint": fingerprint})
        entry.update(status="RUNNING", fingerprint=fingerprint, directory=str(directory))
        self.save()
        process = None
        try:
            with (directory / "worker.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-m", "training.players.orchestration.worker", str(directory / "job.json")],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                entry["pid"] = process.pid
                entry["attempts"][-1]["pid"] = process.pid
                self.save()
                while process.poll() is None:
                    live = read(directory / "live.json", {})
                    fields = {k: live[k] for k in ("run_id", "epoch", "epochs_total", "epochs_completed", "batch", "batches_total", "outputs", "error") if k in live}
                    if live.get("epochs_completed") and live.get("epochs_total"):
                        fields["eta_seconds"] = live["elapsed_seconds"] / live["epochs_completed"] * max(0, live["epochs_total"] - live["epochs_completed"])
                    self.save(**fields)
                    time.sleep(.2)
                if process.returncode != 0:
                    failure = read(directory / "failure.json", {})
                    if failure.get("error", {}).get("type") in ("KeyboardInterrupt", "TechnicalInterruption"):
                        raise KeyboardInterrupt(f"Stage {name} interrupted; checkpoint state retained")
                    raise RuntimeError(f"Stage {name} failed; inspect {directory / 'worker.log'}: {failure.get('error')}")
            receipt = read(directory / "receipt.json")
            if not receipt or not receipt_valid(receipt, fingerprint):
                raise ValueError("Missing or invalid completed-stage receipt")
            entry.update(status="FINISHED", result=receipt["result"], artifacts=receipt["artifacts"])
            self.save(outputs=receipt["result"], run_id=receipt["result"].get("run_id"))
            return receipt["result"]
        except BaseException as exc:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                deadline = time.monotonic() + 20
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.2)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            entry.update(status="KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                         error={"type": type(exc).__name__, "message": str(exc)})
            self.save(error=entry["error"])
            raise

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        owner = self.root / ".players-pipeline.json"
        expected = {"schema_version": 1, "artifact_type": "players_pipeline"}
        if (owner.exists() and read(owner) != expected) or (not owner.exists() and any(self.root.iterdir())):
            raise ValueError("Refusing a foreign pipeline output directory")
        with lock(self.root / "pipeline.lock"):
            write_json(owner, expected)
            self.progress = read(self.root / "progress.json", {"steps": {}})
            self.save(status="RUNNING", step="validate", config=self.config, eta_seconds=None, error=None)
            try:
                _, audit, _, _ = discover(Path(self.config["source"]), ExportConfig(**self.config["export"]))
                source_inputs = {str(Path(self.config["source"]) / p): digest for p, digest in audit["inputs"].items()}
                source_inputs.update(input_files([self.config["export_config"]]))
                settings = {"source": self.config["source"], "export": self.config["export"]}
                self.stage("validate", "validate", settings, source_inputs)
                exported = self.stage("export", "export", {**settings, "output": str(self.root / "exports")}, source_inputs)
                generation = Path(exported["generation"])
                export_inputs = {p: v["sha256"] for p, v in inventory([generation]).items()}
                evaluations = []
                for trial in self.config["trials"]:
                    evaluation = dict(trial["evaluation"])
                    if trial["training"]:
                        training = {**trial["training"], "dataset": str(generation / ("yolo" if trial["training"]["model"] == "yolo" else "coco"))}
                        files = [trial["train"]]
                        files += [training[k] for k in ("weights", "resume") if training[k]]
                        if training["resume"]:
                            files.append(str(Path(training["resume"]).with_name("best.pt" if training["model"] == "yolo" else "best.ckpt")))
                        trained = self.stage("train-" + trial["name"], "train", training, {**export_inputs, **input_files(files)})
                        evaluation["weights"] = trained["best"]
                    evaluation["dataset"] = str(generation / "coco")
                    result = self.stage("evaluate-" + trial["name"], "evaluate", evaluation,
                                        {**export_inputs, **input_files([trial["evaluate"], evaluation["weights"]])})
                    evaluations.append(result["output"])
                result = self.stage("compare", "compare", {"runs": evaluations, "include_smoke": self.config["include_smoke"]},
                                    {p: v["sha256"] for p, v in inventory(evaluations).items()})
                self.save(status="FINISHED", step="complete", eta_seconds=0, outputs=result)
                return self.progress
            except BaseException as exc:
                self.save(status="KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                          error={"type": type(exc).__name__, "message": str(exc)})
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--restart-incomplete", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Short integration interruption; omit when resuming")
    args = parser.parse_args()
    result = Pipeline(load_config(args.config), restart_incomplete=args.restart_incomplete,
                      stop_after_epoch=args.stop_after_epoch).run()
    print(json.dumps(result["outputs"], indent=2))


if __name__ == "__main__":
    main()
