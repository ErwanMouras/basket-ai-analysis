"""One explicit MLflow run per execution, without framework autologging."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

from .references import ROOT


def git_state():
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True
        ).strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "status": git("status", "--porcelain"),
    }


def flatten(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = (
                json.dumps(value) if isinstance(value, (list, tuple)) else str(value)
            )
    return result


@contextmanager
def tracked_run(config, manifest):
    import mlflow
    from mlflow import MlflowClient

    if mlflow.active_run():
        raise RuntimeError(
            "An MLflow run is already active; launch training in a separate process"
        )
    uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow_uri"])
    config["mlflow_uri"] = uri
    mlflow.autolog(disable=True)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(config["experiment"])
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(config["experiment"])
    )
    git = git_state()
    run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": config["model"],
            "model": config["model"],
            "variant": config.get("fusion", config["model"]),
            "dataset_id": config["dataset_id"],
            "export_id": config["export_id"],
            "git.revision": git["revision"],
            "git.dirty": str(git["dirty"]),
            "purpose": config.get("purpose")
            or (
                "smoke"
                if config.get("max_train_batches") or config.get("max_val_batches")
                else "training"
            ),
        },
    )
    run_id = run.info.run_id
    output = Path(config["output"]) / run_id
    start = time.monotonic()
    status = "FAILED"
    try:
        output.mkdir(parents=True)
        (output / "config.resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        packages = {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        }
        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "git": git,
            "training_sha256": source_fingerprints(),
        }
        (output / "environment.json").write_text(json.dumps(environment, indent=2))
        for path in output.iterdir():
            client.log_artifact(run_id, str(path), "provenance")
        client.log_artifact(
            run_id, str(ROOT / "training/ball/configs/references.json"), "provenance"
        )
        for key, value in flatten(config).items():
            client.log_param(run_id, key, value[:6000])
        print(f"MLflow run {run_id}; output: {output}", flush=True)
        yield client, run_id, output
        status = "FINISHED"
    except KeyboardInterrupt:
        status = "KILLED"
        raise
    finally:
        duration = time.monotonic() - start
        if output.exists():
            summary = output / "summary.json"
            summary.write_text(
                json.dumps(
                    {"run_id": run_id, "status": status, "duration_seconds": duration},
                    indent=2,
                )
            )
            client.log_artifact(run_id, str(summary))
        client.log_metric(run_id, "duration_seconds", duration)
        client.set_terminated(run_id, status=status)


def log_metrics(client, run_id, values, epoch):
    from mlflow.entities import Metric

    timestamp = int(time.time() * 1000)
    client.log_batch(
        run_id,
        metrics=[
            Metric(key, float(value), timestamp, epoch) for key, value in values.items()
        ],
    )


def source_fingerprints():
    files = list((ROOT / "training/ball/learning").glob("*.py"))
    files += list((ROOT / "training/ball").glob("train_*.py"))
    files += list((ROOT / "training/ball/evaluation").glob("*.py"))
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
