"""One explicitly-owned MLflow run per training attempt, including strict resumes."""

import copy
import importlib
import math
import re
import shutil
from contextlib import nullcontext
from pathlib import Path

from training.common.files import write_json, write_jsonl
from training.common.provenance import file_hash
from training.players.export.verify import verify_export
from training.players.learning.config import resume_contract, validate_config
from training.players.learning.data import preflight, prepare
from training.players.learning.runtime import code_hashes, configure
from training.players.learning.tracking import log_metrics, tracked_run


class Report:
    def __init__(self, tracking):
        self.client, self.run_id, self.output = tracking
        self.rows, self.logged = [], {}

    def progress(self, **values):
        from training.players.progress import emit
        emit(**values)

    def artifact(self, path, folder=None):
        self.client.log_artifact(self.run_id, str(path), folder)

    def metrics(self, values, epoch):
        metrics = {}
        for key, value in values.items():
            value = float(value)
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite training metric: {key}")
            metrics[re.sub(r"[^\w/ .-]", "_", key)] = value
        log_metrics(self.client, self.run_id, metrics, int(epoch))
        self.rows.append({"epoch": int(epoch), "metrics": metrics})
        write_jsonl(self.output / "metrics.jsonl", self.rows)
        self.artifact(self.output / "metrics.jsonl")
        self.progress(epochs_completed=int(epoch) + 1)

    def checkpoints(self, *paths):
        for value in paths:
            if not value:
                continue
            path = Path(value)
            if not path.is_file():
                continue
            digest = file_hash(path)
            if self.logged.get(path.name) == digest:
                continue
            self.artifact(path, "checkpoints")
            self.logged[path.name] = digest
        write_json(self.output / "checkpoints.json", self.logged)
        self.artifact(self.output / "checkpoints.json")
        self.progress(checkpoints={str(Path(p)): file_hash(Path(p))
                                   for p in paths if p and Path(p).is_file()})


def validate_resume_payload(config, checkpoint):
    state = checkpoint.get("players_training", {})
    if state.get("schema_version") != 1 or state.get("family") != config["model"]:
        raise ValueError(
            "Strict resume requires a complete player checkpoint from the same family"
        )
    config["initial_weights_sha256"] = state["contract"]["initial_weights_sha256"]
    if state["contract"] != resume_contract(config):
        changed = sorted(
            k
            for k in set(state["contract"]) | set(resume_contract(config))
            if state["contract"].get(k) != resume_contract(config).get(k)
        )
        raise ValueError(
            f"Strict resume contract mismatch: {changed}; use fine-tuning for a new experiment"
        )
    if checkpoint.get("epoch", -1) < 0 or checkpoint["epoch"] + 1 >= config["epochs"]:
        raise ValueError(
            "Checkpoint has no completed epoch or has reached the epoch budget"
        )
    required = (
        ("optimizer", "scaler", "ema")
        if config["model"] == "yolo"
        else ("optimizer_states", "lr_schedulers", "loops", "callbacks", "state_dict")
    )
    if any(checkpoint.get(k) is None for k in required) or not state.get("rng"):
        raise ValueError("Checkpoint is missing complete training state")
    if config["model"] == "yolo":
        if (
            any(not state.get(k) for k in ("model", "scheduler", "stopper"))
            or "loader_generator" not in state
        ):
            raise ValueError("YOLO checkpoint is missing resumable state")
    elif not checkpoint["optimizer_states"] or not checkpoint["lr_schedulers"]:
        raise ValueError("RF-DETR checkpoint is missing optimizer or scheduler state")
    elif config["rfdetr"]["use_ema"] and not any(
        isinstance(value, dict) and value.get("average_model_state_dict")
        for value in checkpoint["callbacks"].values()
    ):
        raise ValueError("RF-DETR checkpoint is missing EMA state")
    if (
        config["model"] == "rfdetr"
        and config["precision"] == "amp"
        and not checkpoint.get("MixedPrecision")
    ):
        raise ValueError("RF-DETR checkpoint is missing the AMP scaler")
    return state


def resume_preflight(config):
    import torch

    path = Path(config["resume"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = validate_resume_payload(config, checkpoint)
    best_path = path.parent / ("best.pt" if config["model"] == "yolo" else "best.ckpt")
    if not best_path.is_file():
        raise ValueError("Strict resume also requires the companion best checkpoint")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if (
        best.get("players_training", {}).get("contract") != state["contract"]
        or best.get("epoch", -1) > checkpoint["epoch"]
    ):
        raise ValueError(
            "Companion best checkpoint does not match the resumed training"
        )
    config.update(checkpoint_sha256=file_hash(path), parent_run_id=state["run_id"])
    return checkpoint, best_path


def train(config, *, stop_after_epoch=None):
    config = copy.deepcopy(config)
    validate_config(config)
    if stop_after_epoch is not None and (
        type(stop_after_epoch) is not int
        or not 1 <= stop_after_epoch < config["epochs"]
    ):
        raise ValueError(
            "Technical interruption must be after an epoch within the training budget"
        )
    manifest, records = preflight(config)
    config["runtime"] = configure(config)
    config["code_sha256"] = code_hashes()
    previous, best_path = None, None
    if config["mode"] == "resume":
        previous, best_path = resume_preflight(config)
    else:
        if config["weights"] is not None and not Path(config["weights"]).is_file():
            raise ValueError(
                "Initial weights must be an existing local file; no implicit downloads"
            )
        config["initial_weights_sha256"] = (
            file_hash(Path(config["weights"])) if config["weights"] else None
        )
    with tracked_run(config, manifest) as tracking:
        report = Report(tracking)
        report.client.set_tag(report.run_id, "initialization", config["mode"])
        report.client.set_tag(report.run_id, "hardware", config["runtime"]["hardware"])
        if previous:
            report.client.set_tag(
                report.run_id, "resumed_from_run", config["parent_run_id"]
            )
            directory = report.output / (
                "yolo/weights" if config["model"] == "yolo" else "checkpoints"
            )
            directory.mkdir(parents=True)
            shutil.copy2(best_path, directory / best_path.name)
        elif config["weights"]:
            initial = report.output / "initial_weights" / Path(config["weights"]).name
            initial.parent.mkdir()
            shutil.copy2(config["weights"], initial)
            if file_hash(initial) != config["initial_weights_sha256"]:
                raise ValueError("Initial weights changed during preparation")
            report.artifact(initial, "initial_weights")
            config["weights"] = str(initial)
        write_json(
            report.output / "initialization.json",
            {
                "mode": config["mode"],
                "weights_sha256": config["initial_weights_sha256"],
                "resume_sha256": config.get("checkpoint_sha256"),
                "parent_run_id": config.get("parent_run_id"),
                "best_sha256": file_hash(best_path) if best_path else None,
            },
        )
        report.artifact(report.output / "initialization.json", "provenance")
        view = prepare(config, records, report.output)
        report.artifact(report.output / "selection.jsonl", "provenance")
        adapter = importlib.import_module(
            f"training.players.learning.{config['model']}"
        )
        try:
            from training.players.models import rfdetr_weights

            weights_context = (
                rfdetr_weights(config["weights"], variant=config["variant"])
                if config["model"] == "rfdetr" and config["mode"] == "finetune"
                else nullcontext(config["weights"])
            )
            with weights_context as weights:
                config["weights"] = weights
                last, best = adapter.train(
                    config, view, report, previous, stop_after_epoch
                )
            verify_export(Path(config["dataset"]))
            write_json(
                report.output / "result.json",
                {
                    "run_id": report.run_id,
                    "dataset_id": config["dataset_id"],
                    "last": str(last),
                    "best": str(best),
                    "epochs_completed": config["epochs"],
                },
            )
            report.artifact(report.output / "result.json")
        except BaseException as exc:
            write_json(
                report.output / "error.json",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            report.artifact(report.output / "error.json")
            raise
        return report.output
