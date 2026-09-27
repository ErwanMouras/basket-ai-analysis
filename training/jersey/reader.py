"""Local PARSeq baseline. Lazy FP32 inference; no torch.hub or runtime downloads."""

import importlib.metadata
import math
import string
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from training.common.provenance import file_hash
from training.jersey.config import MODEL_SHA256, MODEL_URL, number

CHARSET = (
    string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation
)


class ResourceDeferred(RuntimeError):
    """The optional OCR budget cannot currently accommodate inference."""


class ParseqReader:
    def __init__(self, config, *, manage_threads=True):
        self.config = config
        self.manage_threads = manage_threads
        if file_hash(Path(config["weights"])) != MODEL_SHA256:
            raise ValueError("Expected the pinned PARSeq checkpoint (SHA-256 mismatch)")
        if importlib.metadata.version("timm") != "0.9.16":
            raise RuntimeError("Install timm==0.9.16 from the updated players lock")
        self.model = None
        self.failed_cuda = False
        self.provenance = {
            "model": "PARSeq generic",
            "checkpoint_sha256": MODEL_SHA256,
            "model_url": MODEL_URL,
            "device": config["device"],
            "precision": "fp32",
            "input_size": [128, 32],
            "alphabet": CHARSET,
            "max_decode_length": 3,
            "refine_iters": 1,
            "timm_version": "0.9.16",
            "weights_only": True,
        }

    def _gpu_guard(self, torch):
        if not self.config["device"].startswith("cuda:"):
            return
        device = int(self.config["device"].split(":")[1])
        if not torch.cuda.is_available() or device >= torch.cuda.device_count():
            raise RuntimeError(
                "Requested jersey CUDA device is unavailable; select CPU explicitly"
            )
        if self.failed_cuda:
            raise ResourceDeferred("cuda_oom_disabled")
        free, _ = torch.cuda.mem_get_info(device)
        # Reserve space for other stages plus a conservative transient OCR allowance.
        needed = self.config["gpu_reserve_mb"] + (
            512 if self.model is not None else 768
        )
        if free < needed * 1024**2:
            raise ResourceDeferred("gpu_memory_reserve")

    def read(self, crops):
        if not crops:
            return []
        if len(crops) > self.config["batch_size"]:
            raise ValueError("OCR batch exceeds configured budget")
        import torch

        self._gpu_guard(torch)
        previous_threads = torch.get_num_threads()
        cuda = self.config["device"].startswith("cuda:")
        try:
            if not cuda and self.manage_threads:
                torch.set_num_threads(self.config["cpu_threads"])
            if self.model is None:
                from training.jersey.vendor.parseq.model import PARSeq

                model = PARSeq(
                    97, 25, (32, 128), (4, 8), 384, 6, 4, 12, 12, 4, 1, True, 1, 0.1
                )
                model.load_state_dict(
                    torch.load(
                        self.config["weights"], map_location="cpu", weights_only=True
                    ),
                    strict=True,
                )
                self.model = (
                    model.eval().requires_grad_(False).to(self.config["device"])
                )
            if cuda:
                torch.cuda.synchronize(self.config["device"])
            batch = (
                torch.from_numpy(np.stack(crops))
                .permute(0, 3, 1, 2)
                .float()
                .div_(127.5)
                .sub_(1)
            )
            batch = batch.to(self.config["device"])
            with torch.inference_mode():
                logits = self.model(
                    SimpleNamespace(eos_id=0, bos_id=95, pad_id=96), batch, max_length=3
                )
                probabilities = logits.softmax(-1).cpu().numpy()
            if cuda:
                torch.cuda.synchronize(self.config["device"])
            return [decode(p) for p in probabilities]
        except torch.cuda.OutOfMemoryError as exc:
            self.model = None
            self.failed_cuda = True
            # Only on OOM, never in the per-frame loop.
            torch.cuda.empty_cache()
            raise ResourceDeferred("cuda_oom_disabled") from exc
        finally:
            if not cuda and self.manage_threads:
                torch.set_num_threads(previous_threads)


def decode(probabilities):
    """Require EOS and a full numeric string; include EOS in sequence confidence."""
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 95
        or not np.isfinite(probabilities).all()
    ):
        raise RuntimeError("Invalid PARSeq probabilities")
    ids = probabilities.argmax(-1)
    eos = np.flatnonzero(ids == 0)
    end = int(eos[0]) if len(eos) else len(ids)
    text = "".join(CHARSET[int(i) - 1] for i in ids[:end])
    confidence = math.prod(
        float(probabilities[i, ids[i]]) for i in range(min(end + 1, len(ids)))
    )
    return {
        "text": text,
        "number": number(text) if len(eos) else None,
        "confidence": confidence,
        "eos": bool(len(eos)),
    }
