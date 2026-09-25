"""Pinned YOLO26 adapter with explicit logging and unstripped atomic checkpoints."""

import time
from copy import deepcopy
from unittest.mock import patch

import torch
import yaml
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from training.players.learning.config import resume_contract
from training.players.learning.runtime import (
    TechnicalInterruption,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_all,
)


class PlayersTrainer(DetectionTrainer):
    def __init__(self, config, report, overrides, stop_after_epoch=None):
        self.player_config, self.report = config, report
        self.stop_after_epoch = stop_after_epoch
        self.epoch_started = time.monotonic()
        # Device availability was checked by our runtime. Avoid Ultralytics mutating
        # CUDA_VISIBLE_DEVICES (especially CPU inference/training before a CUDA run).
        with patch(
            "ultralytics.engine.trainer.select_device",
            return_value=torch.device(config["device"]),
        ):
            super().__init__(overrides=overrides)
        self.args.device = config["device"].replace("cuda:", "")
        self.world_size, self.ddp = 1, False

    def _setup_train(self):
        # Native AMP probing downloads an unrelated checkpoint; precision is explicit here.
        with patch("ultralytics.engine.trainer.check_amp", return_value=True):
            super()._setup_train()

    def check_resume(self, overrides):
        super().check_resume(overrides)
        for key in ("project", "name", "data", "exist_ok"):
            setattr(self.args, key, overrides[key])
        self.args.save_dir = None

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode not in ("train", "val"):
            raise ValueError("Training may only access train and val")
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        return InfiniteDataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=mode == "train",
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            collate_fn=dataset.collate_fn,
            generator=torch.Generator().manual_seed(self.player_config["seed"]),
        )

    def run_callbacks(self, event):
        # Integrations get installed during BaseTrainer.__init__; remove them before every event.
        self.callbacks[event] = [
            cb
            for cb in self.callbacks.get(event, [])
            if not cb.__module__.startswith("ultralytics.utils.callbacks.")
        ]
        super().run_callbacks(event)
        if event == "on_pretrain_routine_end":
            from training.players.models import check_yolo_variant

            check_yolo_variant(unwrap_model(self.model), self.player_config["variant"])
            path = self.report.output / "framework.resolved.yaml"
            path.write_text(yaml.safe_dump(vars(self.args)))
            self.report.artifact(path, "provenance")
            self.report.client.log_param(
                self.report.run_id, "effective_amp", bool(self.amp)
            )
            if self.batch_size != self.player_config["batch_size"]:
                raise ValueError("Automatic batch-size changes are not supported")
        elif event == "on_train_epoch_start":
            self.progress_batch = 0
            self.report.progress(epoch=self.epoch + 1, batch=0, batches_total=len(self.train_loader))
            self.epoch_started = time.monotonic()
            self.train_loader.generator.manual_seed(
                self.player_config["seed"] + self.epoch
            )
            self.train_loader.reset()
            seed_all(self.player_config["seed"] + self.epoch)
        elif event == "on_train_batch_end":
            self.progress_batch += 1
            self.report.progress(batch=self.progress_batch)
        elif event == "on_fit_epoch_end":
            values = {
                **self.label_loss_items(self.tloss, prefix="train"),
                **self.metrics,
                **self.lr,
                "epoch_seconds": time.monotonic() - self.epoch_started,
            }
            self.report.metrics(values, self.epoch)
            self.report.checkpoints(self.last, self.best)
            if (
                self.stop_after_epoch is not None
                and self.epoch + 1 >= self.stop_after_epoch
            ):
                raise TechnicalInterruption(
                    "Requested technical interruption after a saved epoch"
                )

    def save_model(self):
        model = unwrap_model(self.model)
        if not all(
            torch.isfinite(v).all()
            for v in model.state_dict().values()
            if v.is_floating_point()
        ):
            raise FloatingPointError("Non-finite YOLO weights")
        # Do not call the native save: it first writes a non-atomic, half-precision optimizer.
        payload = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": None,
            "ema": deepcopy(unwrap_model(self.ema.ema)).float(),
            "updates": self.ema.updates,
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "scaler": self.scaler.state_dict(),
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, "fitness": self.fitness},
            "train_results": self.read_results_csv(),
            "version": "8.4.39",
            "players_training": {
                "schema_version": 1,
                "family": "yolo",
                "run_id": self.report.run_id,
                "contract": resume_contract(self.player_config),
                "model": deepcopy(model.state_dict()),
                "scheduler": self.scheduler.state_dict(),
                "rng": rng_state(),
                "stopper": vars(self.stopper),
                "loader_generator": self.train_loader.generator.get_state(),
            },
        }
        save_checkpoint(self.last, payload)
        if self.best_fitness == self.fitness:
            save_checkpoint(self.best, payload)
        return True

    def resume_training(self, checkpoint):
        super().resume_training(checkpoint)
        if self.resume:
            state = checkpoint["players_training"]
            unwrap_model(self.model).load_state_dict(state["model"], strict=True)
            self.scheduler.load_state_dict(state["scheduler"])
            vars(self.stopper).update(state["stopper"])
            self.train_loader.generator.set_state(state["loader_generator"].cpu())
            restore_rng(state["rng"])

    def final_eval(self):
        """Per-epoch validation is sufficient; preserve optimizer and scaler in both files."""


def train(config, view, report, previous=None, stop_after_epoch=None):
    weights = (
        config["resume"]
        if previous
        else config["weights"] or config["variant"] + ".yaml"
    )
    overrides = {
        **config["yolo"],
        "model": weights,
        "data": str(view / "data.yaml"),
        "epochs": config["epochs"],
        "batch": config["batch_size"],
        "nbs": config["batch_size"],
        "imgsz": config["resolution"],
        "optimizer": config["optimizer"],
        "lr0": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "warmup_epochs": config["warmup_epochs"],
        "warmup_bias_lr": 0.0,
        "seed": config["seed"],
        "device": config["device"].replace("cuda:", ""),
        "workers": 0,
        "amp": config["precision"] == "amp",
        "deterministic": config["deterministic"],
        "augmentations": [],
        "cache": False,
        "val": True,
        "split": "val",
        "save": True,
        "plots": False,
        "patience": 0,
        "project": str(report.output),
        "name": "yolo",
        "exist_ok": True,
        "resume": config["resume"] if previous else False,
    }
    trainer = PlayersTrainer(config, report, overrides, stop_after_epoch)
    try:
        trainer.train()
    finally:
        report.checkpoints(trainer.last, trainer.best)
        if trainer.csv.is_file():
            report.artifact(trainer.csv)
    if not trainer.last.is_file() or not trainer.best.is_file():
        raise RuntimeError("YOLO training produced no complete best/last checkpoints")
    return trainer.last, trainer.best
