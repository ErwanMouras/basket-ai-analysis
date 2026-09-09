"""Small Ultralytics adapter for explicit logging and resumable checkpoints."""

import time
from copy import deepcopy

import torch
import yaml
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from .config import resume_contract
from .torch_loop import restore_rng, rng_state, save_checkpoint
from .tracking import log_metrics


class BallDetectionTrainer(DetectionTrainer):
    def __init__(self, config, tracking, overrides):
        self.ball_config = config
        self.tracking = tracking
        self.epoch_start = time.monotonic()
        super().__init__(overrides=overrides)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        # Ultralytics otherwise restores the old output/data paths from train_args.
        for key in ("project", "name", "data", "exist_ok"):
            setattr(self.args, key, overrides[key])
        self.args.save_dir = None

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode not in ("train", "val"):
            raise ValueError("Only train and val may be loaded during training")
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        workers = self.args.workers
        return InfiniteDataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=mode == "train",
            num_workers=workers,
            pin_memory=self.ball_config["pin_memory"] and self.device.type == "cuda",
            prefetch_factor=self.ball_config["prefetch_factor"] if workers else None,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.ball_config["seed"]),
        )

    def run_callbacks(self, event):
        # Framework integrations are added in BaseTrainer.__init__, before its first event.
        self.callbacks[event] = [
            callback
            for callback in self.callbacks.get(event, [])
            if not callback.__module__.startswith("ultralytics.utils.callbacks.")
        ]
        super().run_callbacks(event)
        client, run_id, output = self.tracking
        if event == "on_pretrain_routine_end":
            path = output / "ultralytics.resolved.yaml"
            path.write_text(yaml.safe_dump(vars(self.args), sort_keys=False))
            client.log_artifact(run_id, str(path), "provenance")
            client.log_param(run_id, "effective_amp", self.amp)
            client.log_param(run_id, "effective_workers", self.args.workers)
            client.log_param(
                run_id,
                "parameter_count",
                sum(parameter.numel() for parameter in self.model.parameters()),
            )
            client.set_tag(
                run_id,
                "hardware",
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else "cpu",
            )
        elif event == "on_train_epoch_start":
            self.epoch_start = time.monotonic()
        elif event == "on_fit_epoch_end":
            metrics = {
                **self.label_loss_items(self.tloss, prefix="train"),
                **self.metrics,
                **self.lr,
                "epoch_seconds": time.monotonic() - self.epoch_start,
            }
            metrics = {
                key.replace("(", "_").replace(")", ""): value
                for key, value in metrics.items()
            }
            log_metrics(client, run_id, metrics, self.epoch)

    def save_model(self):
        if super().save_model() is False:
            raise FloatingPointError("Ultralytics could not save a finite checkpoint")
        payload = torch.load(self.last, map_location="cpu", weights_only=False)
        payload["optimizer"] = deepcopy(self.optimizer.state_dict())
        payload["ema"] = deepcopy(self.ema.ema).float()
        payload["ball_training"] = {
            "model": deepcopy(unwrap_model(self.model).state_dict()),
            "scheduler": self.scheduler.state_dict(),
            "rng": rng_state(),
            "stopper": vars(self.stopper),
            "contract": resume_contract(self.ball_config),
            "run_id": self.tracking[1],
            "loader_generator": self.train_loader.generator.get_state(),
        }
        save_checkpoint(self.last, payload)
        if self.best_fitness == self.fitness:
            save_checkpoint(self.best, payload)

    def resume_training(self, checkpoint):
        super().resume_training(checkpoint)
        if not self.resume:
            return
        state = checkpoint["ball_training"]
        unwrap_model(self.model).load_state_dict(state["model"])
        self.scheduler.load_state_dict(state["scheduler"])
        vars(self.stopper).update(state["stopper"])
        restore_rng(state["rng"])
        self.train_loader.generator.set_state(state["loader_generator"].cpu())

    def final_eval(self):
        """Validation already ran this epoch; preserve optimizer states in best/last."""
