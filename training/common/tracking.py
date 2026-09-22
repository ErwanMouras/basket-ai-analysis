"""One explicit MLflow run with task-specific provenance supplied by the caller."""

import importlib.metadata
import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

from .files import atomic_writer, write_json
from .provenance import git_state, source_fingerprints


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
def tracked_run(config, manifest, *, root: Path, source_paths, reference_paths=()):
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
    git = git_state(root)
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
        with atomic_writer(output / "config.resolved.yaml") as handle:
            handle.write(yaml.safe_dump(config, sort_keys=False))
        write_json(output / "manifest.json", manifest)
        packages = {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        }
        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "git": git,
            "training_sha256": source_fingerprints(root, source_paths),
        }
        write_json(output / "environment.json", environment)
        for path in output.iterdir():
            client.log_artifact(run_id, str(path), "provenance")
        for path in reference_paths:
            client.log_artifact(run_id, str(path), "provenance")
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
            write_json(
                summary,
                {
                    "run_id": run_id,
                    "status": status,
                    "duration_seconds": duration,
                },
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
