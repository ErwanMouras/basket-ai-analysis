"""Version guards, epoch seeding and atomic full-precision checkpoints."""

import importlib.metadata
import os
import platform
import random
from pathlib import Path

import numpy as np

from training.common.files import atomic_writer
from training.common.provenance import ROOT, source_fingerprints

PINS = {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "ultralytics": "8.4.39",
    "rfdetr": "1.10.1",
    "pytorch-lightning": "2.6.6",
    "transformers": "5.17.0",
    "torchmetrics": "1.8.2",
    "mlflow": "3.11.1",
}


def check_versions(family):
    required = ("torch", "torchvision", "mlflow") + (
        ("ultralytics",)
        if family == "yolo"
        else ("rfdetr", "pytorch-lightning", "transformers", "torchmetrics")
    )
    for package in required:
        actual = importlib.metadata.version(package).split("+")[0]
        if actual != PINS[package]:
            raise ValueError(
                f"Requires {package}=={PINS[package]}, found {actual}; use the players environment"
            )


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    check_versions(config["model"])
    if config["device"] != "cpu":
        index = int(config["device"].split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device is unavailable")
        torch.cuda.set_device(index)
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config["deterministic"]
    # RF-DETR grid_sample backward on CUDA is not bitwise deterministic.
    torch.use_deterministic_algorithms(config["deterministic"], warn_only=True)
    seed_all(config["seed"])
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "hardware": torch.cuda.get_device_name(config["device"])
        if config["device"] != "cpu"
        else platform.processor(),
        "packages": dict(
            sorted(
                (d.metadata["Name"].lower(), d.version)
                for d in importlib.metadata.distributions()
            )
        ),
    }


def code_hashes():
    paths = list((ROOT / "training/players/learning").glob("*.py"))
    paths += list((ROOT / "training/common").glob("*.py"))
    paths += list((ROOT / "training/players").glob("*.py"))
    return source_fingerprints(ROOT, paths)


def rng_state():
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def save_checkpoint(path, value):
    import torch

    with atomic_writer(Path(path), binary=True) as handle:
        torch.save(value, handle)


class TechnicalInterruption(KeyboardInterrupt):
    """Explicit short-test interruption after a complete epoch checkpoint."""
