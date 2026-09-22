"""Epoch training, validation and complete epoch-boundary PyTorch checkpoints."""

import json
import os
import random
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import resume_contract
from .metrics import metric_values, update_counts
from .tracking import log_metrics


def seed_everything(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(config["deterministic"])
    torch.backends.cudnn.benchmark = not config["deterministic"]
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(config["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported PyTorch devices are cpu and cuda:N")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index or 0) >= torch.cuda.device_count()
    ):
        raise ValueError("Requested CUDA device is unavailable")
    if config["precision"] == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA")
    if (
        config["precision"] == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Requested CUDA device does not support bf16")
    return device


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2_seed = seed % (2**31)
    import cv2

    cv2.setNumThreads(1)
    cv2.setRNGSeed(cv2_seed)


def loader(dataset, config, epoch, training):
    generator = torch.Generator().manual_seed(
        config["seed"] + epoch * 2 + int(training)
    )
    options = {
        "batch_size": config["batch_size"],
        "shuffle": training,
        "num_workers": config["workers"],
        "pin_memory": config["pin_memory"],
        "generator": generator,
        "worker_init_fn": seed_worker,
    }
    if config["workers"]:
        options["prefetch_factor"] = config["prefetch_factor"]
        # Fresh processes avoid inheriting thread locks from CUDA/OpenCV after
        # validation when workers are recreated for the next training epoch.
        options["multiprocessing_context"] = "spawn"
    # Recreate workers per epoch so augmentation RNGs also resume at epoch boundaries.
    return DataLoader(dataset, **options)


def sample_mixup(images, targets, alpha):
    weights = np.random.beta(alpha, alpha, size=len(images))
    weights = np.maximum(weights, 1 - weights)
    weights = torch.as_tensor(
        weights[:, None, None, None], dtype=images.dtype, device=images.device
    )
    permutation = torch.randperm(len(images), device=images.device)
    return images * weights + images[permutation] * (
        1 - weights
    ), targets * weights + targets[permutation] * (1 - weights)


def run_epoch(
    model,
    dataset,
    loss_fn,
    optimizer,
    scaler,
    config,
    device,
    epoch,
    training,
    mixup_alpha=0,
):
    model.train(training)
    counts, loss_sum, sample_count = Counter(), 0.0, 0
    limit = config["max_train_batches" if training else "max_val_batches"]
    batches = loader(dataset, config, epoch, training)
    if getattr(dataset, "augmentation", None):
        dataset.augmentation.set_epoch(epoch)
    for index, batch in enumerate(batches):
        if limit is not None and index >= limit:
            break
        images = batch["image"].to(device, non_blocking=config["pin_memory"])
        targets = batch["target"].to(device, non_blocking=config["pin_memory"])
        if training and mixup_alpha:
            images, targets = sample_mixup(images, targets, mixup_alpha)
        if training:
            optimizer.zero_grad(set_to_none=True)
        dtype = torch.float16 if config["precision"] == "fp16" else torch.bfloat16
        autocast = (
            torch.autocast(device.type, dtype=dtype)
            if config["precision"] != "fp32"
            else nullcontext()
        )
        with torch.set_grad_enabled(training):
            with autocast:
                predictions = model(images)
            # WBCE logarithms and weights stay in fp32 under mixed precision.
            loss = loss_fn(predictions.float(), targets.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["grad_clip"], error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
        loss_sum += loss.item() * len(images)
        sample_count += len(images)
        if not training:
            update_counts(
                counts,
                predictions.detach().float().cpu().numpy(),
                targets.cpu().numpy(),
                config["threshold"],
                config["tolerance_px"],
            )
    if not sample_count:
        raise ValueError("No batches were produced")
    values = {"loss": loss_sum / sample_count, "samples": sample_count}
    if not training:
        values.update(metric_values(counts))
    prefix = "train" if training else "val"
    return {f"{prefix}/{key}": value for key, value in values.items()}


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def configure_optimizer(model, config, tracking):
    client, run_id, output = tracking
    optimizer_class = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
        "Adadelta": torch.optim.Adadelta,
    }[config["optimizer"]]
    optimizer = optimizer_class(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["min_learning_rate"]
        )
        if config["scheduler"] == "cosine"
        else None
    )
    runtime = output / "optimizer.resolved.json"
    runtime.write_text(
        json.dumps(
            {
                "optimizer": optimizer.defaults,
                "scheduler": scheduler.state_dict() if scheduler else None,
            },
            default=str,
            indent=2,
        )
    )
    client.log_artifact(run_id, str(runtime), "provenance")
    scaler = torch.amp.GradScaler("cuda", enabled=config["precision"] == "fp16")
    return optimizer, scheduler, scaler


def resume_training(model, optimizer, scheduler, scaler, config, tracking):
    client, run_id, output = tracking
    start_epoch, best = 0, float("inf")
    if config["resume"]:
        checkpoint = torch.load(
            config["resume"], map_location="cpu", weights_only=False
        )
        if checkpoint["contract"] != resume_contract(config):
            raise ValueError(
                "Resume configuration/data differs from the checkpoint contract"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best = checkpoint["epoch"] + 1, checkpoint["best_val_loss"]
        restore_rng(checkpoint["rng"])
        previous_best = Path(config["resume"]).parent / "best.pt"
        if not previous_best.is_file():
            raise ValueError("Resume requires best.pt beside the checkpoint")
        shutil.copy2(previous_best, output / "best.pt")
        client.set_tag(run_id, "resumed_from_run", checkpoint["run_id"])
    if start_epoch >= config["epochs"]:
        raise ValueError("Checkpoint already reached the configured epoch budget")
    return start_epoch, best


def fit(model, train, val, loss_fn, config, device, tracking, mixup_alpha=0):
    client, run_id, output = tracking
    model.to(device)
    optimizer, scheduler, scaler = configure_optimizer(model, config, tracking)
    start_epoch, best = resume_training(
        model, optimizer, scheduler, scaler, config, tracking
    )
    client.log_param(
        run_id,
        "parameter_count",
        sum(parameter.numel() for parameter in model.parameters()),
    )
    client.set_tag(
        run_id,
        "hardware",
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    )
    try:
        for epoch in range(start_epoch, config["epochs"]):
            start = time.monotonic()
            learning_rate = optimizer.param_groups[0]["lr"]
            metrics = run_epoch(
                model,
                train,
                loss_fn,
                optimizer,
                scaler,
                config,
                device,
                epoch,
                True,
                mixup_alpha,
            )
            metrics.update(
                run_epoch(
                    model, val, loss_fn, optimizer, scaler, config, device, epoch, False
                )
            )
            improved = metrics["val/loss"] < best
            best = min(best, metrics["val/loss"])
            if scheduler:
                scheduler.step()
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best,
                "rng": rng_state(),
                "contract": resume_contract(config),
                "config": config,
                "run_id": run_id,
            }
            save_checkpoint(output / "last.pt", payload)
            if improved:
                save_checkpoint(output / "best.pt", payload)
            metrics.update(
                learning_rate=learning_rate, epoch_seconds=time.monotonic() - start
            )
            log_metrics(client, run_id, metrics, epoch)
            print(
                f"Epoch {epoch + 1}: train={metrics['train/loss']:.6f}, val={metrics['val/loss']:.6f}",
                flush=True,
            )
    finally:
        for name in ("best.pt", "last.pt"):
            if (output / name).exists():
                client.log_artifact(run_id, str(output / name), "checkpoints")
    return output
