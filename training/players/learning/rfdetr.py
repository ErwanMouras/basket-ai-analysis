"""RF-DETR 1.10.1 public Lightning primitives with explicit player checkpoints."""

import signal
import time

import torch
import yaml
from pytorch_lightning.callbacks import Checkpoint, ModelCheckpoint
from pytorch_lightning.plugins.io import TorchCheckpointIO
from rfdetr.config import (
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    TrainConfig,
)
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

from training.players.learning.config import resume_contract
from training.players.learning.runtime import (
    TechnicalInterruption,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_all,
)

MODEL_CONFIGS = {
    "rfdetr_nano": RFDETRNanoConfig,
    "rfdetr_small": RFDETRSmallConfig,
    "rfdetr_medium": RFDETRMediumConfig,
    "rfdetr_large": RFDETRLargeConfig,
}


class AtomicCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        save_checkpoint(path, checkpoint)

    def load_checkpoint(self, path, map_location=None, weights_only=None):
        # Resume has already validated this explicitly selected local training checkpoint.
        return torch.load(path, map_location=map_location or "cpu", weights_only=False)


class EpochSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, trainer, seed):
        self.dataset, self.trainer, self.seed = dataset, trainer, seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(
            self.seed + self.trainer.current_epoch
        )
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)


class PlayersDataModule(RFDETRDataModule):
    def train_dataloader(self):
        loader = super().train_dataloader()
        return torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=EpochSampler(loader.dataset, self.trainer, self.train_config.seed),
            num_workers=0,
            drop_last=True,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            generator=torch.Generator().manual_seed(self.train_config.seed),
        )

    def val_dataloader(self):
        loader = super().val_dataloader()
        loader.generator = torch.Generator().manual_seed(self.train_config.seed)
        return loader


class PlayersModule(RFDETRModelModule):
    def __init__(self, model_config, train_config, config, report):
        super().__init__(model_config, train_config)
        self.strict_loading = True
        self.player_config, self.report = config, report
        self.pending_rng = None
        self.save_hyperparameters(
            {
                "model_config": model_config.model_dump(mode="json"),
                "train_config": train_config.model_dump(mode="json"),
            }
        )

    def on_train_start(self):
        super().on_train_start()
        if self.pending_rng is not None:
            restore_rng(self.pending_rng)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        seed_all(self.player_config["seed"] + self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["players_training"] = {
            "schema_version": 1,
            "family": "rfdetr",
            "run_id": self.report.run_id,
            "contract": resume_contract(self.player_config),
            "rng": rng_state(),
        }
        checkpoint["model_config"] = self.model_config.model_dump(mode="json")
        checkpoint["class_names"] = ["player"]

    def on_load_checkpoint(self, checkpoint):
        # Strict resume accepts our format only. Do not silently interpolate or convert weights.
        self.pending_rng = checkpoint["players_training"]["rng"]


class PlayersCheckpoint(ModelCheckpoint):
    def __init__(self, report, stop_after_epoch):
        super().__init__(
            dirpath=report.output / "checkpoints",
            filename="best",
            monitor="val/mAP_50_95",
            mode="max",
            save_top_k=1,
            save_last=True,
            save_weights_only=False,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
            auto_insert_metric_name=False,
        )
        self.report, self.stop_after_epoch = report, stop_after_epoch
        self.started = time.monotonic()

    def load_state_dict(self, state_dict):
        # A resumed run owns its copy of best.ckpt; never remove files in the parent run.
        state = dict(state_dict)
        state["dirpath"] = self.dirpath
        state["best_model_path"] = str(self.report.output / "checkpoints/best.ckpt")
        state["last_model_path"] = str(self.report.output / "checkpoints/last.ckpt")
        state["kth_best_model_path"] = state["best_model_path"]
        state["best_k_models"] = (
            {state["best_model_path"]: state["best_model_score"]}
            if state["best_model_score"] is not None
            else {}
        )
        super().load_state_dict(state)

    def on_train_epoch_start(self, trainer, pl_module):
        self.started = time.monotonic()
        self.report.progress(epoch=trainer.current_epoch + 1, batch=0,
                             batches_total=int(trainer.num_training_batches))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.report.progress(batch=batch_idx + 1)

    def on_train_epoch_end(self, trainer, pl_module):
        self.report.metrics(
            {
                **trainer.callback_metrics,
                "epoch_seconds": time.monotonic() - self.started,
            },
            trainer.current_epoch,
        )
        super().on_train_epoch_end(trainer, pl_module)
        # Lightning 2.6.6 otherwise updates last only when the monitored best improves.
        if self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))
        self.report.checkpoints(self.last_model_path, self.best_model_path)
        if (
            self.stop_after_epoch is not None
            and trainer.current_epoch + 1 >= self.stop_after_epoch
        ):
            raise TechnicalInterruption(
                "Requested technical interruption after a saved epoch"
            )


def train(config, view, report, previous=None, stop_after_epoch=None):
    options = config["rfdetr"]
    mc = MODEL_CONFIGS[config["variant"]](
        num_classes=1,
        resolution=config["resolution"],
        pretrain_weights=None
        if previous or config["mode"] == "scratch"
        else config["weights"],
        device=config["device"],
        amp=config["precision"] == "amp",
        fused_optimizer=False,
        model_name="RFDETR" + config["variant"].split("_")[1].title(),
    )
    tc = TrainConfig(
        dataset_dir=str(view),
        dataset_file="roboflow",
        output_dir=str(report.output / "rfdetr"),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        eval_batch_size=config["batch_size"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_epochs=config["warmup_epochs"],
        seed=config["seed"],
        num_workers=0,
        optimizer="adamw",
        lr_scheduler="cosine",
        lr_scheduler_kwargs={"min_factor": options["lr_min_factor"]},
        lr_encoder=options["lr_encoder"],
        grad_accum_steps=options["grad_accum_steps"],
        use_ema=options["use_ema"],
        ema_decay=options["ema_decay"],
        ema_tau=options["ema_tau"],
        clip_max_norm=options["clip_max_norm"],
        multi_scale=False,
        expanded_scales=False,
        scale_jitter=options["scale_jitter"],
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
        early_stopping=False,
        run_test=False,
        class_names=["player"],
        num_sanity_val_steps=0,
        compute_val_loss=True,
        eval_interval=1,
        progress_bar=None,
        pin_memory=config["device"] != "cpu",
        persistent_workers=False,
    )
    module = PlayersModule(mc, tc, config, report)
    if module.model_config.num_classes != 1:
        raise ValueError("RF-DETR did not initialize a one-class player head")
    datamodule = PlayersDataModule(mc, tc)
    checkpoint = PlayersCheckpoint(report, stop_after_epoch)
    trainer = build_trainer(
        tc,
        mc,
        accelerator="cpu" if config["device"] == "cpu" else "gpu",
        devices=1
        if config["device"] == "cpu"
        else config["device"].split(":")[1] + ",",
        precision="32-true" if config["precision"] == "fp32" else "16-mixed",
        deterministic="warn" if config["deterministic"] else False,
        plugins=[AtomicCheckpointIO()],
        enable_model_summary=False,
    )
    trainer.callbacks = [
        cb for cb in trainer.callbacks if not isinstance(cb, Checkpoint)
    ] + [checkpoint]
    if config["precision"] == "amp":
        trainer.precision_plugin.scaler = torch.amp.GradScaler(
            "cuda", init_scale=options["amp_init_scale"]
        )
    torch.set_float32_matmul_precision("highest")
    path = report.output / "framework.resolved.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": mc.model_dump(mode="json"),
                "train": tc.model_dump(mode="json"),
                "precision": str(trainer.precision),
                "deterministic": config["deterministic"],
                "workers": 0,
                "amp_init_scale": options["amp_init_scale"],
            }
        )
    )
    report.artifact(path, "provenance")
    previous_signal = signal.getsignal(signal.SIGINT)
    try:
        try:
            trainer.fit(
                module,
                datamodule=datamodule,
                ckpt_path=config["resume"] if previous else None,
            )
        except SystemExit as exc:
            if trainer.interrupted:
                raise TechnicalInterruption("RF-DETR training interrupted") from exc
            raise
        if trainer.interrupted:
            raise TechnicalInterruption("RF-DETR training interrupted")
    finally:
        signal.signal(signal.SIGINT, previous_signal)
        report.checkpoints(
            report.output / "checkpoints/last.ckpt",
            report.output / "checkpoints/best.ckpt",
        )
        for name in ("metrics.csv", "hparams.yaml"):
            path = report.output / "rfdetr" / name
            if path.is_file():
                report.artifact(path, "framework")
    last, best = (
        report.output / "checkpoints/last.ckpt",
        report.output / "checkpoints/best.ckpt",
    )
    if not last.is_file() or not best.is_file():
        raise RuntimeError(
            "RF-DETR training produced no complete best/last checkpoints"
        )
    return last, best
