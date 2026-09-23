"""Batch-one wall-clock measurements, with explicit CUDA synchronization."""

import os
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


def runtime(config):
    import torch
    from importlib.metadata import version

    torch.set_num_threads(config["cpu_threads"])
    # Ultralytics changes this process-global setting during import. Pin it for
    # both families before measuring instead of inheriting import-order effects.
    cv2.setNumThreads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if config["device"] != "cpu":
        torch.cuda.set_device(config["device"])
    cpu = platform.processor()
    if os.path.exists("/proc/cpuinfo"):
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    driver = None
    if config["device"] != "cpu":
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
                text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            driver = "unavailable"
    return {"host": platform.node(), "platform": platform.platform(), "cpu": cpu,
            "gpu": torch.cuda.get_device_name(config["device"]) if config["device"] != "cpu" else None,
            "device": config["device"], "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu_inventory_driver": driver, "opencv_threads": cv2.getNumThreads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cudnn": torch.backends.cudnn.version(), "opencv": cv2.__version__,
            "packages": {name: version(name) for name in ("pycocotools", "ultralytics", "rfdetr", "numpy")}}


def summarize(samples):
    return {"samples": len(samples), "median_ms": float(np.median(samples) * 1000),
            "p95_ms": float(np.percentile(samples, 95) * 1000),
            "images_per_second": len(samples) / sum(samples)}


def benchmark(detector, records, root, config):
    import torch
    import psutil

    def sync():
        if config["device"] != "cpu":
            torch.cuda.synchronize(config["device"])

    def predict(image):
        return detector.predict(image, confidence=config["score_floor"],
                                max_detections=config["max_detections"], square=True)

    first = cv2.imread(str(root / records[0]["image"]))
    for _ in range(config["warmup"]):
        predict(first)
        sync()
    if config["device"] != "cpu":
        torch.cuda.reset_peak_memory_stats(config["device"])
    rss = psutil.Process().memory_info().rss
    adapter_times, image_times, forward_times = [], [], []
    # Normal prediction path without hooks; decode outside the adapter interval.
    for _ in range(config["repeats"]):
        for record in records:
            sync()
            start = time.perf_counter()
            image = cv2.imread(str(root / record["image"]))
            if image is None:
                raise ValueError("Cannot decode benchmark image")
            adapter_start = time.perf_counter()
            predict(image)
            sync()
            end = time.perf_counter()
            adapter_times.append(end - adapter_start)
            image_times.append(end - start)
            rss = max(rss, psutil.Process().memory_info().rss)
    # Separate pass: measure the torch module only, without contaminating adapter timings.
    module = (detector.model.predictor.model if detector.family == "yolo"
              else detector.model.model.model)
    started = []

    def before(*args):
        sync()
        started.append(time.perf_counter())

    def after(*args):
        sync()
        forward_times.append(time.perf_counter() - started.pop())

    handles = [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
    try:
        for _ in range(config["repeats"]):
            for record in records:
                predict(cv2.imread(str(root / record["image"])))
    finally:
        for handle in handles:
            handle.remove()
    if len(forward_times) != len(adapter_times):
        raise RuntimeError("Expected exactly one instrumented model forward per image")
    gpu = config["device"] != "cpu"
    return {"forward": summarize(forward_times), "adapter": summarize(adapter_times),
            "image_pipeline": summarize(image_times), "video_pipeline": None,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(config["device"]) if gpu else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(config["device"]) if gpu else None,
            "cpu_rss_sampled_peak_bytes": rss,
            "samples_seconds": {"forward": forward_times, "adapter": adapter_times, "image_pipeline": image_times}}
