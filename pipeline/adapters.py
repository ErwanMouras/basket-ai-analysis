"""Adapters for existing POC models, with an isolated TrackNet process."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import cv2
import numpy as np

from training.common.provenance import file_hash


class RoleTracker:
    """Separate association pools prevent an official inheriting a player's ID."""

    def __init__(self, config, fps, roles=("player", "referee", "unknown")):
        from training.players.tracking import PlayerTracker
        self.trackers = {role: PlayerTracker(config, fps=fps) for role in sorted(set(roles))}
        self.ids = {}
        self.counter = 0
        self.buffer = max(t.buffer_frames for t in self.trackers.values())
        self.provenance = {"association": "independent_per_role", "roles": list(self.trackers),
                           "backend": next(iter(self.trackers.values())).provenance}

    def update(self, detections, image, *, frame_index, scene_cut=False):
        if scene_cut:
            self.ids.clear()
        self.ids = {key: value for key, value in self.ids.items() if frame_index - value[1] <= self.buffer}
        output = [None] * len(detections)
        for role, tracker in self.trackers.items():
            indices = [i for i, d in enumerate(detections) if d["role"] == role]
            rows = tracker.update([detections[i] for i in indices], image,
                                  frame_index=frame_index, scene_cut=scene_cut)
            for i, row in zip(indices, rows, strict=True):
                local_id = row["track_id"]
                if local_id is not None:
                    key = (role, local_id)
                    if key not in self.ids:
                        self.counter += 1
                        self.ids[key] = (self.counter, frame_index)
                    row["track_id"] = self.ids[key][0]
                    self.ids[key] = (row["track_id"], frame_index)
                output[i] = row
        return output


class People:
    def __init__(self, config):
        self.config = config
        path = Path(config["weights"]).resolve(strict=True)
        self.roles = {int(k): v for k, v in config["class_roles"].items()}
        if config["adapter"] == "yolo":
            from ultralytics import YOLO
            self.model = YOLO(str(path), task="detect")
            names = self.model.names
            for cls, role in self.roles.items():
                if cls not in names:
                    raise ValueError(f"Class {cls} not present in checkpoint: {names}")
                # COCO person cannot substantiate a player/referee role.
                if names[cls].lower() == "person" and role != "unknown":
                    raise ValueError("A generic person class must map to unknown, not player/referee")
                canonical = names[cls].lower()
                if canonical in ("player", "referee", "basketball", "hoop") and role != "unknown" and role != canonical:
                    raise ValueError(f"Class {cls} is {names[cls]}, incompatible with role {role}")
        else:
            from training.players.models import Detector
            if len(self.roles) != 1:
                raise ValueError("RF-DETR adapter currently accepts one explicit source class")
            self.model = Detector("rfdetr", config["variant"], path,
                                  device=config["device"], resolution=config["resolution"],
                                  source_class=next(iter(self.roles)))
            names = {next(iter(self.roles)): "configured_source_class"}
        self.provenance = {"adapter": config["adapter"], "checkpoint_sha256": file_hash(path),
                           "weights": str(path), "class_names": names, "class_roles": config["class_roles"]}

    def predict(self, image):
        cfg = self.config
        if cfg["adapter"] == "rfdetr":
            role = next(iter(self.roles.values()))
            return [{**d, "role": role} for d in self.model.predict(
                image, confidence=cfg["confidence"], max_detections=cfg["max_detections"], square=True)]
        result = self.model.predict(image, imgsz=cfg["resolution"], conf=cfg["confidence"],
                                    classes=list(self.roles), device=cfg["device"],
                                    max_det=cfg["max_detections"], verbose=False)[0]
        return [{"bbox": box.tolist(), "confidence": float(score), "class_id": 0,
                 "source_class_id": int(cls), "role": self.roles[int(cls)]}
                for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                          result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy())]


_BALL = None


def _ball_init(config, cpu_threads):
    """Spawn context keeps upstream top-level imports out of the main process."""
    global _BALL
    import torch
    from training.ball.learning.references import activate_reference
    torch.set_num_threads(cpu_threads)
    checkpoint = torch.load(config["weights"], map_location="cpu", weights_only=False)
    trained = checkpoint["config"]
    if trained["model"] not in ("tracknet_v5", "tracknet_v5_totnet"):
        raise ValueError("Expected V5 or V5+TOTNet training checkpoint")
    ref = {**trained, "reference_root": str(Path(config["reference_root"]).resolve())}
    activate_reference(ref)
    if ref["reference"] != trained["reference"]:
        raise ValueError("TrackNet checkpoint/reference mismatch")
    from models_factory import build_model
    model = build_model(trained["architecture"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False).to(config["device"])
    _BALL = (model, trained["geometry"], config, {
        "adapter": trained["model"], "geometry": trained["geometry"],
        "checkpoint_sha256": file_hash(Path(config["weights"])),
        "training_run_id": checkpoint["run_id"], "reference": trained["reference"],
        "temporal_policy": "causal_triplet_last_head_no_overlap_fusion"})


def _ball_metadata():
    return _BALL[3]


def _ball_resources():
    import resource
    import torch
    device = _BALL[2]["device"]
    return {"peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "peak_torch_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.startswith("cuda") else None}


def _ball_predict(images):
    import torch
    from training.ball.learning.torch_data import image_tensor, sdk_resize_triplet
    from training.ball.learning.metrics import center
    model, geometry, config, _ = _BALL
    rgb = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in images]
    tensor = image_tensor(sdk_resize_triplet(rgb, geometry["input_width"], geometry["input_height"]))
    with torch.inference_mode():
        heatmap = model(tensor[None].to(config["device"]))[0, 2].float().cpu().numpy()
    if not np.isfinite(heatmap).all() or heatmap.min() < 0 or heatmap.max() > 1:
        raise ValueError("Invalid ball heatmap")
    point = center(heatmap > config["threshold"])
    return {"position_input": point.tolist() if point is not None else None,
            "confidence": float(heatmap.max()), "status": "ok" if point is not None else "not_detected",
            "context_frames": 3, "future_context_frames": 0}


class Ball:
    def __init__(self, config, cpu_threads):
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_ball_init, initargs=(config, cpu_threads))
        try:
            self.provenance = self.pool.submit(_ball_metadata).result()
        except BaseException:
            self.close()
            raise

    def predict(self, frames):
        geometry = self.provenance["geometry"]
        # Send only resized inputs to the worker; source coordinates are restored here.
        images = [cv2.resize(f.image, (geometry["input_width"], geometry["input_height"])) for f in frames]
        result = self.pool.submit(_ball_predict, images).result()
        point = result.pop("position_input")
        height, width = frames[-1].image.shape[:2]
        result["position_px"] = [point[0] * width / geometry["input_width"],
                                 point[1] * height / geometry["input_height"]] if point is not None else None
        result["source_frames"] = [f.index for f in frames]
        return result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def resources(self):
        return self.pool.submit(_ball_resources).result()
